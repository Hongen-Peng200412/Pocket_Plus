"""候选条件谱系网络（Candidate-Conditioned Lineage Network，CCLN）。

主要入口:
    - `CandidateConditionedLineageNetwork`: 先让每个候选分别读取 V/P/A 实体，再在
      候选树上交换信息，输出候选质量、结构化选择能量和 CLG 有效概率。
    - `CandidateTokenAttention`: 候选到单个实体模态的带几何偏置交叉注意力。
    - `TreeRelativeTransformerLayer`: 使用最低共同祖先、阈值和几何关系偏置的
      候选自注意力。

网络一次处理一个组件谱系组。实体数量和候选数量均可变；所有 `*_world` 坐标使用
连续世界 XYZ、单位 Å，离散 voxel 索引仍由 Dataset 以 ZYX 提供。
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from .density_munet_lite import DensityMUNetLite
from .input_fusion import ResidualSwiGLUFusion, VDensityFusion


def probability_features(probability: torch.Tensor, epsilon: float) -> torch.Tensor:
    """
    构造概率本身与数值稳定 logit 两维实体值特征。

    输入参数:
        - probability: torch.Tensor, `(N_entity,)`，已经 sigmoid 的实体概率
        - epsilon: float, 裁剪到 `[epsilon, 1-epsilon]` 的数值稳定常数

    输出:
        - features: torch.Tensor, `(N_entity, 2)`，末维依次为 `p` 与
          `log(p/(1-p))`
    """
    clipped = probability.to(dtype=torch.float32).clamp(float(epsilon), 1.0 - float(epsilon))
    return torch.stack([probability.to(dtype=torch.float32), torch.logit(clipped)], dim=-1)


class CandidateTokenAttention(nn.Module):
    """
    用候选 query 读取一个模态的 tokens，并加入固定 10D 几何/概率 bias。

    输入参数:
        - hidden_dim: int, query/token hidden dim
        - num_heads: int, attention head 数

    前向输入:
        - query: torch.Tensor, (hidden_dim,), 当前 candidate query
        - token: torch.Tensor, (N_token,hidden_dim)，本 candidate 可读的模态 token
        - relative_bias_input: torch.Tensor, (N_token,10)，三组相对坐标与概率
        - probability: torch.Tensor, (N_token,)，该模态 token 概率

    前向输出:
        - context: torch.Tensor, (hidden_dim,)，候选条件模态摘要；空 token 时返回 learned null
    """

    def __init__(self, hidden_dim: int, num_heads: int) -> None:
        """
        初始化候选到实体的多头注意力投影、几何偏置和空模态表示。

        输入参数:
            - hidden_dim: int, 候选查询、实体键值和输出的统一通道数
            - num_heads: int, 注意力头数；`hidden_dim` 必须能被其整除

        参数语义:
            - `probability_prior`: 跨注意力头共享的概率先验系数，从 0 开始学习。
            - `null_token`: 实体表为空时使用的可学习模态内容。
            - `modality_present`: 区分实体表为空或非空的二值可学习嵌入。
        """
        super().__init__()
        if int(hidden_dim) % int(num_heads) != 0:
            raise ValueError("hidden_dim 必须能被 num_heads 整除。")
        self.hidden_dim = int(hidden_dim)
        self.num_heads = int(num_heads)
        self.head_dim = self.hidden_dim // self.num_heads
        self.query_projection = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.key_projection = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.value_projection = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.output_projection = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.bias_mlp = nn.Sequential(
            nn.Linear(10, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.num_heads),
        )
        # 跨 head 共享的概率先验标量按契约从 0 开始，不在初始化时偏置任何模态。
        self.probability_prior = nn.Parameter(torch.zeros(()))
        self.null_token = nn.Parameter(torch.zeros(self.hidden_dim))
        self.modality_present = nn.Embedding(2, self.hidden_dim)

    def forward(
        self,
        query: torch.Tensor,
        token: torch.Tensor,
        relative_bias_input: torch.Tensor,
        probability: torch.Tensor,
    ) -> torch.Tensor:
        """
        计算带独立模态 bias 的 candidate-to-token attention。

        输入参数:
            - query: torch.Tensor, (hidden_dim,), 当前 candidate query
            - token: torch.Tensor, (N_token,hidden_dim)，当前可读 token
            - relative_bias_input: torch.Tensor, (N_token,10)，未做 Fourier 展开的 bias 输入
            - probability: torch.Tensor, (N_token,)，token 概率

        输出:
            - context: torch.Tensor, (hidden_dim,)，候选条件摘要
        """
        if token.shape[0] == 0:
            missing_index = torch.zeros((), dtype=torch.long, device=query.device)
            return self.null_token + self.modality_present(missing_index)
        if relative_bias_input.shape != (token.shape[0], 10) or probability.shape != (token.shape[0],):
            raise ValueError("relative_bias_input/probability 必须与 token 行严格对齐。")

        # `(H, d_h)` 查询和 `(H, N_token, d_h)` 键值，其中 H 为注意力头数。
        projected_query = self.query_projection(query).reshape(self.num_heads, self.head_dim)
        projected_key = self.key_projection(token).reshape(-1, self.num_heads, self.head_dim).transpose(0, 1)
        projected_value = self.value_projection(token).reshape(-1, self.num_heads, self.head_dim).transpose(0, 1)
        # `(H, N_token)`，内容相似度、10D 几何偏置和实体概率先验共同决定权重。
        attention_score = torch.einsum("hd,htd->ht", projected_query, projected_key) / math.sqrt(self.head_dim)
        learned_bias = self.bias_mlp(relative_bias_input).transpose(0, 1)
        attention_score = attention_score + learned_bias + self.probability_prior * probability.reshape(1, -1)
        attention_weight = torch.softmax(attention_score, dim=-1)
        context = torch.einsum("ht,htd->hd", attention_weight, projected_value).reshape(self.hidden_dim)
        present_index = torch.ones((), dtype=torch.long, device=query.device)
        return self.output_projection(context) + self.modality_present(present_index)


class TokenAttentionPool(nn.Module):
    """
    对同一 BOX 的实际模态 tokens 产生共享全局摘要。

    输入参数:
        - hidden_dim: int, token hidden dim
        - num_heads: int, MultiheadAttention head 数

    前向输入:
        - token: torch.Tensor, (N_token,hidden_dim)，当前模态全部实际 tokens

    前向输出:
        - pooled: torch.Tensor, (hidden_dim,)，共享 BOX 模态摘要；空表为 learned null
    """

    def __init__(self, hidden_dim: int, num_heads: int) -> None:
        """
        初始化单个可学习查询和多头注意力池化器。

        输入参数:
            - hidden_dim: int, 输入实体与输出摘要的统一通道数
            - num_heads: int, 多头注意力的头数

        空输入:
            - 当前模态没有实体时直接返回可学习 `null`，不构造伪实体。
        """
        super().__init__()
        self.query = nn.Parameter(torch.zeros(1, 1, int(hidden_dim)))
        self.null = nn.Parameter(torch.zeros(int(hidden_dim)))
        self.attention = nn.MultiheadAttention(int(hidden_dim), int(num_heads), batch_first=True)

    def forward(self, token: torch.Tensor) -> torch.Tensor:
        """
        汇聚当前模态 tokens。

        输入参数:
            - token: torch.Tensor, (N_token,hidden_dim)，当前 BOX 的实际 tokens

        输出:
            - pooled: torch.Tensor, (hidden_dim,)，全局模态摘要
        """
        if token.shape[0] == 0:
            return self.null
        pooled, _ = self.attention(self.query, token.unsqueeze(0), token.unsqueeze(0), need_weights=False)
        return pooled[0, 0]


class TreeRelativeTransformerLayer(nn.Module):
    """
    使用 LCA 相对特征 bias 的 pre-norm candidate self-attention 层。

    输入参数:
        - hidden_dim: int, candidate hidden dim
        - num_heads: int, attention head 数
        - ffn_hidden_dim: int, SwiGLU FFN hidden dim
        - relative_dim: int, tree pair 相对特征通道数
        - dropout: float, residual dropout

    前向输入:
        - candidate: torch.Tensor, (N_candidate,hidden_dim)，当前 candidate 表示
        - relative_feature: torch.Tensor, (N_candidate,N_candidate,relative_dim)，有序 candidate pair 特征

    前向输出:
        - updated: torch.Tensor, (N_candidate,hidden_dim)，tree-relative 更新结果
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        ffn_hidden_dim: int,
        relative_dim: int,
        dropout: float,
    ) -> None:
        """
        初始化一层带有候选对相对偏置的 pre-norm Transformer。

        输入参数:
            - hidden_dim: int, 候选表示通道数
            - num_heads: int, 自注意力头数；`hidden_dim` 必须能被其整除
            - ffn_hidden_dim: int, SwiGLU 前馈分支中每个门分支的通道数
            - relative_dim: int, 每个有序候选对的相对特征通道数
            - dropout: float, 注意力输出和前馈输出写入残差前的 dropout 概率
        """
        super().__init__()
        if int(hidden_dim) % int(num_heads) != 0:
            raise ValueError("hidden_dim 必须能被 num_heads 整除。")
        self.hidden_dim = int(hidden_dim)
        self.num_heads = int(num_heads)
        self.head_dim = self.hidden_dim // self.num_heads
        self.norm_attention = nn.LayerNorm(self.hidden_dim)
        self.query_key_value = nn.Linear(self.hidden_dim, 3 * self.hidden_dim)
        self.relative_bias = nn.Sequential(
            nn.Linear(int(relative_dim), self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.num_heads),
        )
        self.attention_output = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.norm_ffn = nn.LayerNorm(self.hidden_dim)
        self.ffn_gate = nn.Linear(self.hidden_dim, 2 * int(ffn_hidden_dim))
        self.ffn_output = nn.Linear(int(ffn_hidden_dim), self.hidden_dim)
        self.dropout = nn.Dropout(float(dropout))

    def forward(self, candidate: torch.Tensor, relative_feature: torch.Tensor) -> torch.Tensor:
        """
        执行一层 tree-relative Transformer。

        输入参数:
            - candidate: torch.Tensor, (N_candidate,hidden_dim)，candidate 表示
            - relative_feature: torch.Tensor, (N_candidate,N_candidate,relative_dim)，LCA/阈值/几何/体积 pair 特征

        输出:
            - updated: torch.Tensor, (N_candidate,hidden_dim)，更新后的 candidate 表示
        """
        # N_candidate，当前组件谱系组的候选数量。
        count = candidate.shape[0]
        normalized = self.norm_attention(candidate)
        query, key, value = self.query_key_value(normalized).chunk(3, dim=-1)
        query = query.reshape(count, self.num_heads, self.head_dim).transpose(0, 1)
        key = key.reshape(count, self.num_heads, self.head_dim).transpose(0, 1)
        value = value.reshape(count, self.num_heads, self.head_dim).transpose(0, 1)
        # `(H, N_candidate, N_candidate)`，内容分数与有序候选对偏置逐项相加。
        score = torch.einsum("hid,hjd->hij", query, key) / math.sqrt(self.head_dim)
        score = score + self.relative_bias(relative_feature).permute(2, 0, 1)
        weight = torch.softmax(score, dim=-1)
        attended = torch.einsum("hij,hjd->hid", weight, value).transpose(0, 1).reshape(count, self.hidden_dim)
        candidate = candidate + self.dropout(self.attention_output(attended))
        normalized_ffn = self.norm_ffn(candidate)
        gate_a, gate_b = self.ffn_gate(normalized_ffn).chunk(2, dim=-1)
        return candidate + self.dropout(self.ffn_output(F.silu(gate_a) * gate_b))


class CandidateConditionedLineageNetwork(nn.Module):
    """
    首版 CCLN：candidate-conditioned V/P/A 读取后再做 tree-relative Transformer。

    输入参数:
        - modalities: Sequence[str], producer 实际模态；只允许按顺序包含 V、P、A
        - a_source_order/a_source_dims: A 多层来源及通道；无 A 时均为空
        - p_source_order/p_source_dims: P 多层来源及通道；无 P 时均为空
        - hidden_dim: int, candidate/token hidden dim；正式为 128
        - num_heads: int, cross/tree attention head 数；正式为 4
        - tree_layers: int, tree-relative Transformer 层数；正式为 2
        - tree_ffn_hidden_dim: int, tree FFN hidden dim；正式为 256
        - candidate_attribute_dim: int, candidate 属性通道；当前固定为 16
        - modality_projected_dim: int, A/P 每个来源的投影通道数
        - modality_gate_hidden_dim: int, A/P residual_swiglu hidden dim
        - v_output_dim: int, V48+D 输出通道；正式为 48
        - v_meta_dim: int, V 基础 meta 通道；当前为概率二元组 2
        - v_correction_gate_hidden_dim: int, density correction gate hidden dim
        - use_density_context: bool, D 分支开关
        - density_input_shape_zyx/density_channels: DensityMUNetLite shape 与四层通道
        - density_bottleneck_heads/layers/ffn_dim: 仅最低分辨率 Transformer 参数
        - dropout: float, CCLN 与密度 bottleneck dropout
        - probability_epsilon: float, token probability logit clip 常数

    前向输入:
        - sample: dict[str, Any]，一个 CLG；字段由 SelectorDataset.__getitem__ 完整定义

    前向输出:
        - outputs: dict[str, torch.Tensor]，包含 E_content/E_tree、qhat、z、a_G 与 p_G
    """

    def __init__(
        self,
        modalities: Sequence[str],
        a_source_order: Sequence[str],
        a_source_dims: Mapping[str, int],
        p_source_order: Sequence[str],
        p_source_dims: Mapping[str, int],
        hidden_dim: int,
        num_heads: int,
        tree_layers: int,
        tree_ffn_hidden_dim: int,
        candidate_attribute_dim: int,
        modality_projected_dim: int,
        modality_gate_hidden_dim: int,
        v_output_dim: int,
        v_meta_dim: int,
        v_correction_gate_hidden_dim: int,
        use_density_context: bool,
        density_input_shape_zyx: Sequence[int],
        density_channels: Sequence[int],
        density_bottleneck_heads: int,
        density_bottleneck_layers: int,
        density_bottleneck_ffn_dim: int,
        dropout: float,
        probability_epsilon: float,
    ) -> None:
        """
        按冻结的模态、通道和结构参数构造完整 CCLN。

        输入参数:
            - modalities: Sequence[str], 实际模态顺序；必须以 V 开头且元素取自 V/P/A
            - a_source_order: Sequence[str], A 模态特征来源字段的固定拼接顺序
            - a_source_dims: Mapping[str, int], A 各来源的末维通道数
            - p_source_order: Sequence[str], P 模态特征来源字段的固定拼接顺序
            - p_source_dims: Mapping[str, int], P 各来源的末维通道数
            - hidden_dim: int, 候选与实体 token 的统一隐藏通道数
            - num_heads: int, 交叉注意力、树注意力和 CLG 汇聚的头数
            - tree_layers: int, 候选树相对 Transformer 层数
            - tree_ffn_hidden_dim: int, 候选树前馈分支的隐藏通道数
            - candidate_attribute_dim: int, 每个候选的输入属性通道数，当前为 16
            - modality_projected_dim: int, 每个 A/P 来源独立投影后的通道数
            - modality_gate_hidden_dim: int, A/P 多来源 SwiGLU 门分支通道数
            - v_output_dim: int, V48 与密度上下文融合后的通道数
            - v_meta_dim: int, 每个稀疏 V voxel 的基础附加特征通道数
            - v_correction_gate_hidden_dim: int, 密度修正门的隐藏通道数
            - use_density_context: bool, 是否运行任务专属密度上下文网络
            - density_input_shape_zyx: Sequence[int], 密度输入的离散 ZYX 网格尺寸
            - density_channels: Sequence[int], 密度编码器四个分辨率层的通道数
            - density_bottleneck_heads: int, 密度最低分辨率 Transformer 注意力头数
            - density_bottleneck_layers: int, 密度最低分辨率 Transformer 层数
            - density_bottleneck_ffn_dim: int, 密度 Transformer 前馈通道数
            - dropout: float, 树层和密度 Transformer 的 dropout 概率
            - probability_epsilon: float, 实体概率转 logit 前的裁剪常数

        约束:
            - 未启用的 A/P 模态不得配置对应来源；启用的模态仅消费清单中真实字段。
            - 密度分支关闭时不会补零伪造输入。
        """
        super().__init__()
        self.modalities = tuple(str(name) for name in modalities)
        invalid_modality = any(name not in {"V", "P", "A"} for name in self.modalities)
        if not self.modalities or self.modalities[0] != "V" or invalid_modality:
            raise ValueError("modalities 必须以 V 开头且只包含 V/P/A。")
        if len(set(self.modalities)) != len(self.modalities):
            raise ValueError("modalities 不得重复。")
        self.hidden_dim = int(hidden_dim)
        self.probability_epsilon = float(probability_epsilon)

        # 密度编码器仅在启用密度上下文时实例化，避免未使用参数进入 checkpoint。
        density_encoder = (
            DensityMUNetLite(
                input_shape_zyx=density_input_shape_zyx,
                channels=density_channels,
                bottleneck_heads=int(density_bottleneck_heads),
                bottleneck_layers=int(density_bottleneck_layers),
                bottleneck_ffn_dim=int(density_bottleneck_ffn_dim),
                dropout=float(dropout),
            )
            if use_density_context
            else None
        )
        self.v_fusion = VDensityFusion(
            meta_dim=int(v_meta_dim),
            output_dim=int(v_output_dim),
            correction_gate_hidden_dim=int(v_correction_gate_hidden_dim),
            use_density=bool(use_density_context),
            density_encoder=density_encoder,
        )
        self.v_token_projection = nn.Linear(int(v_output_dim) + 2, self.hidden_dim)

        # A/P 各自先在实体内部融合多层来源，再与概率二元组拼接成统一 token。
        self.a_fusion: ResidualSwiGLUFusion | None = None
        self.p_fusion: ResidualSwiGLUFusion | None = None
        if "A" in self.modalities:
            self.a_fusion = ResidualSwiGLUFusion(
                source_order=a_source_order,
                source_dims=a_source_dims,
                projected_dim=int(modality_projected_dim),
                meta_dim=0,
                output_dim=self.hidden_dim,
                gate_hidden_dim=int(modality_gate_hidden_dim),
            )
        elif a_source_order or a_source_dims:
            raise ValueError("无 A 模态的 producer 不得配置 A sources。")
        if "P" in self.modalities:
            self.p_fusion = ResidualSwiGLUFusion(
                source_order=p_source_order,
                source_dims=p_source_dims,
                projected_dim=int(modality_projected_dim),
                meta_dim=0,
                output_dim=self.hidden_dim,
                gate_hidden_dim=int(modality_gate_hidden_dim),
            )
        elif p_source_order or p_source_dims:
            raise ValueError("无 P 模态的 producer 不得配置 P sources。")
        self.a_token_projection = nn.Linear(self.hidden_dim + 2, self.hidden_dim) if "A" in self.modalities else None
        self.p_token_projection = nn.Linear(self.hidden_dim + 2, self.hidden_dim) if "P" in self.modalities else None

        # `(N_candidate, 16) -> (N_candidate, hidden_dim)` 的候选初始查询。
        self.candidate_query = nn.Sequential(
            nn.LayerNorm(int(candidate_attribute_dim)),
            nn.Linear(int(candidate_attribute_dim), self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.modality_attention = nn.ModuleDict(
            {name: CandidateTokenAttention(self.hidden_dim, int(num_heads)) for name in self.modalities}
        )
        self.modality_pool = nn.ModuleDict(
            {name: TokenAttentionPool(self.hidden_dim, int(num_heads)) for name in self.modalities}
        )
        # 每个候选拼接自身查询、逐模态候选条件摘要和逐模态 BOX 全局摘要。
        content_input_dim = self.hidden_dim * (1 + 2 * len(self.modalities))
        self.content_mlp = nn.Sequential(
            nn.LayerNorm(content_input_dim),
            nn.Linear(content_input_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.blob_head = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, 1),
        )
        # 树层只在所有实体读取完成后运行，避免实体表示被候选间消息传递污染。
        self.tree_layers = nn.ModuleList(
            [
                TreeRelativeTransformerLayer(
                    hidden_dim=self.hidden_dim,
                    num_heads=int(num_heads),
                    ffn_hidden_dim=int(tree_ffn_hidden_dim),
                    relative_dim=8,
                    dropout=float(dropout),
                )
                for _ in range(int(tree_layers))
            ]
        )
        self.selection_head = nn.Sequential(nn.LayerNorm(self.hidden_dim), nn.Linear(self.hidden_dim, 1))
        self.box_query_projection = nn.Linear(self.hidden_dim * len(self.modalities), self.hidden_dim)
        self.clg_attention = nn.MultiheadAttention(self.hidden_dim, int(num_heads), batch_first=True)
        self.clg_head = nn.Sequential(nn.LayerNorm(self.hidden_dim), nn.Linear(self.hidden_dim, 1))

    @staticmethod
    def _relative_bias_input(
        token_coord_world: torch.Tensor,
        candidate_center_world: torch.Tensor,
        oldest_center_world: torch.Tensor,
        box_center_world: torch.Tensor,
        box_half_size_world: torch.Tensor,
        probability: torch.Tensor,
    ) -> torch.Tensor:
        """
        构造固定 10D candidate-token bias 输入，不做 Fourier 展开。

            - token_coord_world: torch.Tensor, `(N_token, 3)`，连续世界坐标 XYZ，单位 Å
            - candidate_center_world: torch.Tensor, `(3,)`，当前候选中心的连续世界 XYZ
            - oldest_center_world: torch.Tensor, `(3,)`，CLG oldest 模态中心的连续
              世界 XYZ
            - box_center_world: torch.Tensor, `(3,)`，当前 BOX 中心的连续世界 XYZ
            - box_half_size_world: torch.Tensor, `(3,)`，连续世界 XYZ 各轴物理半边长，
              单位 Å
            - probability: torch.Tensor, (N_token,)，token 概率

        输出:
            - relative: torch.Tensor, `(N_token, 10)`，依次为实体相对候选中心、
              oldest 模态中心、BOX 中心的三组归一化 XYZ，以及实体概率
        """
        if token_coord_world.shape[0] == 0:
            return token_coord_world.new_empty((0, 10))
        return torch.cat(
            [
                (token_coord_world - candidate_center_world) / box_half_size_world,
                (token_coord_world - oldest_center_world) / box_half_size_world,
                (token_coord_world - box_center_world) / box_half_size_world,
                probability.reshape(-1, 1),
            ],
            dim=-1,
        )

    def _build_tokens(self, sample: Mapping[str, Any]) -> dict[str, torch.Tensor]:
        """
        在任何 candidate/tree 交互前融合 V/P/A 各自的真实来源。

        输入参数:
            - sample: Mapping[str, Any]，一个 SelectorDataset CLG 样本

        输出:
            - tokens: dict[str, torch.Tensor]，每个实际模态为 (N_entity,hidden_dim)
        """
        # `(N_V, 2)`，V 实体概率及其稳定 logit。
        v_probability_feature = probability_features(sample["V_probability"], self.probability_epsilon)
        v_value = self.v_fusion(
            voxel_final=sample["V_sources"]["voxel_final"],
            density_input=sample["density_input"] if self.v_fusion.use_density else None,
            voxel_index_local_zyx=sample["V_index_local_zyx"],
            voxel_batch_index=sample["V_batch_index"],
            meta=v_probability_feature,
        )
        # 每个值均为 `(N_entity, hidden_dim)`；键集合严格等于当前配置的实际模态。
        tokens: dict[str, torch.Tensor] = {
            "V": self.v_token_projection(torch.cat([v_value, v_probability_feature], dim=-1))
        }
        if "P" in self.modalities:
            assert self.p_fusion is not None and self.p_token_projection is not None
            p_probability_feature = probability_features(sample["P_probability"], self.probability_epsilon)
            p_value = self.p_fusion(sample["P_sources"], None)
            tokens["P"] = self.p_token_projection(torch.cat([p_value, p_probability_feature], dim=-1))
        if "A" in self.modalities:
            assert self.a_fusion is not None and self.a_token_projection is not None
            a_probability_feature = probability_features(sample["A_probability"], self.probability_epsilon)
            a_value = self.a_fusion(sample["A_sources"], None)
            tokens["A"] = self.a_token_projection(torch.cat([a_value, a_probability_feature], dim=-1))
        return tokens

    def forward(self, sample: Mapping[str, Any]) -> dict[str, torch.Tensor]:
        """
        对一个 CLG 运行 candidate-conditioned 多模态读取与 tree-relative 推理。

        输入参数:
            - sample: Mapping[str, Any]，至少包含:
                - candidate_attributes: torch.Tensor, (N_candidate,16)
                - candidate_centroid_world: torch.Tensor, (N_candidate,3), 连续世界坐标 XYZ，单位 Å
                - candidate_voxel_offsets/candidate_voxel_index: candidate→V membership; `candidate_voxel_index` 指向 V 表行，不是 voxel 坐标
                - tree_relative_feature: torch.Tensor, (N_candidate,N_candidate,8)
                - V_sources/V_probability/V_coord_world/V_index_local_zyx/V_batch_index；其中 world 为连续 XYZ，index 为 BOX-local 离散 ZYX voxel index
                - box_center_world/box_half_size_world: 连续世界 XYZ；box_shape_zyx: 离散 voxel 网格尺寸 ZYX
                - Find 另含 P_*、A_* 与 candidate_A_offsets/candidate_A_index

        输出:
            - outputs: dict[str, torch.Tensor]，包含:
                - "candidate_content": (N_candidate,hidden_dim)，tree Transformer 前 E_i^content
                - "candidate_tree": (N_candidate,hidden_dim)，tree Transformer 后 E_i^tree
                - "predicted_max_iou": (N_candidate,)，sigmoid blob readout qhat_i
                - "selection_logit": (N_candidate,)，只作反链能量的 z_i
                - "CLG_logit": scalar，独立 CLG readout 原始值 a_G
                - "CLG_valid_probability": scalar，sigmoid(a_G)=p_G
        """
        tokens = self._build_tokens(sample)
        # `(N_candidate, hidden_dim)`，仅由候选自身 16D 属性构成的查询。
        candidate_query = self.candidate_query(sample["candidate_attributes"])
        # 每个模态得到一个 `(hidden_dim,)` 的 BOX 全局摘要，供所有候选共享。
        global_summary = {name: self.modality_pool[name](tokens[name]) for name in self.modalities}
        oldest_v_center = sample["oldest_V_center_world"]
        oldest_a_center = sample.get("oldest_A_center_world")
        # 每个候选独立读取自身 V/A 成员和全量 P，再与共享模态摘要融合。
        candidate_context_rows: list[torch.Tensor] = []

        candidate_voxel_offsets = sample["candidate_voxel_offsets"].to(dtype=torch.long)
        candidate_voxel_index = sample["candidate_voxel_index"].to(dtype=torch.long)
        candidate_a_offsets = sample.get("candidate_A_offsets")
        candidate_a_index = sample.get("candidate_A_index")
        for candidate_index in range(candidate_query.shape[0]):
            modality_context: list[torch.Tensor] = []
            v_begin = int(candidate_voxel_offsets[candidate_index].item())
            v_end = int(candidate_voxel_offsets[candidate_index + 1].item())
            v_rows = candidate_voxel_index[v_begin:v_end]
            if v_rows.numel() == 0:
                raise ValueError("每个 CLG candidate 必须至少引用一个 V voxel。")
            v_probability = sample["V_probability"][v_rows]
            v_relative = self._relative_bias_input(
                token_coord_world=sample["V_coord_world"][v_rows],
                candidate_center_world=sample["candidate_centroid_world"][candidate_index],
                oldest_center_world=oldest_v_center,
                box_center_world=sample["box_center_world"],
                box_half_size_world=sample["box_half_size_world"],
                probability=v_probability,
            )
            modality_context.append(
                self.modality_attention["V"](
                    candidate_query[candidate_index], tokens["V"][v_rows], v_relative, v_probability
                )
            )

            if "P" in self.modalities:
                p_probability = sample["P_probability"]
                p_relative = self._relative_bias_input(
                    token_coord_world=sample["P_coord_world"],
                    candidate_center_world=sample["candidate_centroid_world"][candidate_index],
                    oldest_center_world=oldest_v_center,
                    box_center_world=sample["box_center_world"],
                    box_half_size_world=sample["box_half_size_world"],
                    probability=p_probability,
                )
                modality_context.append(
                    self.modality_attention["P"](
                        candidate_query[candidate_index], tokens["P"], p_relative, p_probability
                    )
                )

            if "A" in self.modalities:
                assert candidate_a_offsets is not None and candidate_a_index is not None
                a_begin = int(candidate_a_offsets[candidate_index].item())
                a_end = int(candidate_a_offsets[candidate_index + 1].item())
                a_rows = candidate_a_index[a_begin:a_end].to(dtype=torch.long)
                a_probability = sample["A_probability"][a_rows]
                if a_rows.numel() == 0:
                    a_relative = sample["A_coord_world"].new_empty((0, 10))
                else:
                    assert oldest_a_center is not None
                    candidate_a_center = sample["A_coord_world"][a_rows].mean(dim=0)
                    a_relative = self._relative_bias_input(
                        token_coord_world=sample["A_coord_world"][a_rows],
                        candidate_center_world=candidate_a_center,
                        oldest_center_world=oldest_a_center,
                        box_center_world=sample["box_center_world"],
                        box_half_size_world=sample["box_half_size_world"],
                        probability=a_probability,
                    )
                modality_context.append(
                    self.modality_attention["A"](
                        candidate_query[candidate_index], tokens["A"][a_rows], a_relative, a_probability
                    )
                )

            content_input = [candidate_query[candidate_index], *modality_context]
            content_input.extend(global_summary[name] for name in self.modalities)
            candidate_context_rows.append(self.content_mlp(torch.cat(content_input, dim=-1)))

        # `(N_candidate, hidden_dim)`，尚未进行候选树消息传递的内容表示。
        candidate_content = torch.stack(candidate_context_rows, dim=0)
        predicted_max_iou = torch.sigmoid(self.blob_head(candidate_content).squeeze(-1))
        candidate_tree = candidate_content
        for layer in self.tree_layers:
            candidate_tree = layer(candidate_tree, sample["tree_relative_feature"])
        selection_logit = self.selection_head(candidate_tree).squeeze(-1)

        # CLG readout 以全部模态的 BOX 摘要为查询，从树更新后的候选表读取一次。
        box_summary = torch.cat([global_summary[name] for name in self.modalities], dim=-1)
        clg_query = self.box_query_projection(box_summary).reshape(1, 1, self.hidden_dim)
        clg_context, _ = self.clg_attention(
            clg_query,
            candidate_tree.unsqueeze(0),
            candidate_tree.unsqueeze(0),
            need_weights=False,
        )
        clg_logit = self.clg_head(clg_context[0, 0]).reshape(())
        return {
            "candidate_content": candidate_content,
            "candidate_tree": candidate_tree,
            "predicted_max_iou": predicted_max_iou,
            "selection_logit": selection_logit,
            "CLG_logit": clg_logit,
            "CLG_valid_probability": torch.sigmoid(clg_logit),
        }
