"""
Stage1 最终分类头: 统一几何 cross-attn + real/pseudo 两个轻量分类尾部。

对齐契约（修改时必须全量同步）:

    - 本段、CLAUDE/plans/最后重构代码_v3.md 的 §2、src/model/stage1_model.py::_run_atom_head 和受影响的 tests/model 单测必须同步更新。
    - N_all 表示本头输入点数; N_real 表示真实原子数; N_pseudo 表示 P anchor 数; C_point=point_channels。
    - pseudo_mask is None 表示 real-only 路径, 此时 N_all=N_real, pseudo_* 输出全为 None。

前向输入:

    - point_feat: torch.Tensor, (N_all, C_point), floating, 最后一轮 point backbone 输出特征, mixed 路径下按 pseudo_atoms.py 的 mixed layout 排列; real / P 槽位由 pseudo_mask 区分。
    - point_state: dict[str, Any], 与 point_feat 同布局的点状态; 本头仅消费 point_state["batch"] (torch.Tensor, (N_all,), int64, 每个点所属 BOX 索引), 用于 cross-attn 的 radius 分组建图。
    - atom_coord_centered_world: torch.Tensor, (N_all, 3), floating, centered-world 坐标, 轴顺序 (x, y, z); 用于 cross-attn 的 radius 建图与 relative coord。
    - pseudo_mask: torch.Tensor | None, (N_all,), bool, True 表示 P anchor, False 表示 real atom; None 表示 real-only 路径; 非 None 时第一维必须与 point_feat 一致。

前向输出（dict, 键名固定, 6 项）:

    - "real_feat_before_interaction": torch.Tensor, (N_real, C_point), cross-attn 前的 real 特征。
    - "real_feat_after_interaction": torch.Tensor, (N_real, C_point), 经 pseudo_to_real 增量后的 real 特征; cross-attn 零初始化/冻结时与 before 逐元素相等。
    - "pseudo_feat_before_interaction": torch.Tensor | None, (N_pseudo, C_point), cross-attn 前的 P 特征; real-only 路径为 None。
    - "pseudo_feat_after_interaction": torch.Tensor | None, (N_pseudo, C_point), 经 real_to_pseudo 增量后的 P 特征; cross-attn 零初始化/冻结时与 before 逐元素相等; real-only 路径为 None。
    - "atom_logits": torch.Tensor, (N_real, atom_logit_dim), real atom 监督 logits。
    - "pseudo_logits": torch.Tensor | None, (N_pseudo, pseudo_ligand_logit_dim), P anchor ligand 区域归属 logits; real-only 路径为 None。

零初始化契约:
    两个 cross-attn 的 output_proj 权重和 bias 均零初始化, 且 after 一律以纯残差 feat_after = feat_before + crossattn(...) 加回, 中间不插 LayerNorm 等破坏逐元素恒等的算子。因此 cross-attn 处于零初始化或被冻结态时, *_after_interaction == *_before_interaction 逐元素精确成立。

执行顺序保证无循环依赖:
    pseudo_to_real 的增量只读 pseudo_before, real_to_pseudo 的增量只读 real_before 与 real_bind_prob(由 atom_logits 的第 0 通道 sigmoid 后 detach)。两条增量各自只依赖对方的 before 特征,不依赖对方的 after 特征, 故无环。
"""
from __future__ import annotations

import math
from typing import Any, Sequence

import torch
from torch import nn

from src.model.typed_point import validate_pseudo_mask

_TORCH_CLUSTER_IMPORT_ERROR: Exception | None = None
try:
    torch_cluster = __import__("torch_cluster")
except Exception as exc:  # pragma: no cover - 依赖当前本地环境
    torch_cluster = None
    _TORCH_CLUSTER_IMPORT_ERROR = exc

_TORCH_SCATTER_IMPORT_ERROR: Exception | None = None
try:
    torch_scatter = __import__("torch_scatter")
except Exception as exc:  # pragma: no cover - 依赖当前本地环境
    torch_scatter = None
    _TORCH_SCATTER_IMPORT_ERROR = exc


class GeometricCrossAttention(nn.Module):
    """
    统一的几何 cross-attn: query 点在 radius 邻域内聚合 source 点特征, 输出零初始化增量。

    两个方向复用本类:
        - real_to_pseudo: query=P, source=real, 吃 detached real bind prob。
        - pseudo_to_real: query=real, source=P, 不吃 bind prob。

    输入参数:
        - query_channels: int, query 点特征通道数, 也是本模块的输出通道数
        - source_channels: int, source 点特征通道数
        - num_heads: int, attention 头数; 必须整除 query_channels
        - radius: float, radius 建图的世界坐标半径
        - max_neighbors: int, 每个 query 点最多保留的 source 邻居数
        - detach_source_feat: bool, 取边特征前是否对 source 特征 detach
        - act_layer: type[nn.Module], 激活函数类
        - use_source_bind_prob: bool, 是否把 source 侧 bind probability 拼进边特征; True 时边特征维 +1

    前向输入:
        - query_feat: torch.Tensor, (N_query, query_channels), query 点特征
        - source_feat: torch.Tensor, (N_source, source_channels), source 点特征
        - query_coord: torch.Tensor, (N_query, 3), query 点 centered-world 坐标, 轴顺序 (x, y, z)
        - source_coord: torch.Tensor, (N_source, 3), source 点 centered-world 坐标, 轴顺序 (x, y, z)
        - query_batch: torch.Tensor, (N_query,), int64, query 点所属 BOX 索引
        - source_batch: torch.Tensor, (N_source,), int64, source 点所属 BOX 索引
        - source_bind_prob: torch.Tensor | None, (N_source, 1), use_source_bind_prob=True 时必填, class Stage1AtomHead已经外部detach

    前向输出:
        - delta: torch.Tensor, (N_query, query_channels), query 侧增量(不含残差); 空 query / 空 source / 空边时返回全零张量。
    """

    def __init__(
        self,
        query_channels: int,
        source_channels: int,
        num_heads: int,
        radius: float,
        max_neighbors: int,
        detach_source_feat: bool,
        act_layer: type[nn.Module],
        use_source_bind_prob: bool,
    ) -> None:
        super().__init__()
        if int(query_channels) <= 0:
            raise ValueError("query_channels 必须 > 0。")
        if int(source_channels) <= 0:
            raise ValueError("source_channels 必须 > 0。")
        if int(num_heads) <= 0 or int(query_channels) % int(num_heads) != 0:
            raise ValueError("num_heads 必须 > 0 且整除 query_channels。")
        if float(radius) <= 0.0:
            raise ValueError("radius 必须 > 0。")
        if int(max_neighbors) <= 0:
            raise ValueError("max_neighbors 必须 > 0。")

        self.query_channels = int(query_channels)
        self.source_channels = int(source_channels)
        self.num_heads = int(num_heads)
        self.head_dim = self.query_channels // self.num_heads
        self.radius = float(radius)
        self.max_neighbors = int(max_neighbors)
        self.detach_source_feat = bool(detach_source_feat)
        self.use_source_bind_prob = bool(use_source_bind_prob)

        # int, 边特征输入维: source 特征 + relative coord(3) + 可选 bind prob(1)
        edge_in_dim = self.source_channels + 3 + (1 if self.use_source_bind_prob else 0)
        # nn.Linear, (N_query, query_channels) -> (N_query, query_channels), query 投影
        self.query_proj = nn.Linear(self.query_channels, self.query_channels)
        # nn.Sequential, (E, edge_in_dim) -> (E, query_channels), 边特征到 key
        self.key_proj = nn.Sequential(
            nn.Linear(edge_in_dim, self.query_channels),
            act_layer(),
            nn.Linear(self.query_channels, self.query_channels),
        )
        # nn.Sequential, (E, edge_in_dim) -> (E, query_channels), 边特征到 value
        self.value_proj = nn.Sequential(
            nn.Linear(edge_in_dim, self.query_channels),
            act_layer(),
            nn.Linear(self.query_channels, self.query_channels),
        )
        # nn.Sequential, (E, 4) -> (E, num_heads), 由 relative coord 与归一化距离生成的 per-head 几何 bias
        self.geo_bias = nn.Sequential(nn.Linear(4, self.num_heads), act_layer(), nn.Linear(self.num_heads, self.num_heads))
        # nn.Linear, (N_query, query_channels) -> (N_query, query_channels), 输出投影; 权重与 bias 零初始化
        self.output_proj = nn.Linear(self.query_channels, self.query_channels)
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def forward(
        self,
        *,
        query_feat: torch.Tensor,
        source_feat: torch.Tensor,
        query_coord: torch.Tensor,
        source_coord: torch.Tensor,
        query_batch: torch.Tensor,
        source_batch: torch.Tensor,
        source_bind_prob: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if query_feat.shape[0] == 0:
            return query_feat.new_zeros((0, self.query_channels))
        if torch_cluster is None:
            raise ImportError("GeometricCrossAttention 需要 torch_cluster。") from _TORCH_CLUSTER_IMPORT_ERROR
        if torch_scatter is None:
            raise ImportError("GeometricCrossAttention 需要 torch_scatter。") from _TORCH_SCATTER_IMPORT_ERROR
        if source_feat.shape[0] == 0:
            return query_feat.new_zeros((int(query_feat.shape[0]), self.query_channels))

        # torch.Tensor, (2, E), radius 邻接; row=query 索引, col=source 索引
        edge_index = torch_cluster.radius(
            x=source_coord,
            y=query_coord,
            r=self.radius,
            batch_x=source_batch,
            batch_y=query_batch,
            max_num_neighbors=self.max_neighbors,
        )
        if int(edge_index.shape[1]) == 0:
            return query_feat.new_zeros((int(query_feat.shape[0]), self.query_channels))

        # torch.Tensor, (E,), 每条边的 query 端索引
        idx_query = edge_index[0]
        # torch.Tensor, (E,), 每条边的 source 端索引
        idx_source = edge_index[1]
        # torch.Tensor, (N_source, source_channels), 取边特征用的 source 特征(可选 detach)
        source_feat_used = source_feat.detach() if self.detach_source_feat else source_feat
        # torch.Tensor, (E, 3), source 相对 query 的 centered-world 位移
        rel_coord = source_coord[idx_source] - query_coord[idx_query]
        edge_parts = [source_feat_used[idx_source], rel_coord]
        if self.use_source_bind_prob:
            if source_bind_prob is None:
                raise RuntimeError("use_source_bind_prob=True 时必须传入 source_bind_prob。")
            edge_parts.append(source_bind_prob[idx_source])
        # torch.Tensor, (E, edge_in_dim), 拼接后的边特征
        edge_feat = torch.cat(edge_parts, dim=1)

        # torch.Tensor, (N_query, num_heads, head_dim), query 投影后按头切分
        query = self.query_proj(query_feat).reshape(-1, self.num_heads, self.head_dim)
        # torch.Tensor, (E, num_heads, head_dim), 边 key
        key = self.key_proj(edge_feat).reshape(-1, self.num_heads, self.head_dim)
        # torch.Tensor, (E, num_heads, head_dim), 边 value
        value = self.value_proj(edge_feat).reshape(-1, self.num_heads, self.head_dim)
        # torch.Tensor, (E, 4), 归一化 relative coord 与归一化距离, 几何 bias 输入
        geo_input = torch.cat([rel_coord / self.radius, rel_coord.norm(dim=1, keepdim=True) / self.radius], dim=1)
        # torch.Tensor, (E, num_heads), per-head 几何 bias
        geo_bias = self.geo_bias(geo_input)
        # torch.Tensor, (E, num_heads), 缩放点积 logits 加几何 bias
        attn_logits = (query[idx_query] * key).sum(dim=-1) / math.sqrt(float(self.head_dim)) + geo_bias
        # torch.Tensor, (E, num_heads), 以 query 为分组的 softmax 注意力权重
        attn = torch_scatter.scatter_softmax(attn_logits, idx_query, dim=0)
        # torch.Tensor, (E, query_channels = num_heads*head_dim), 加权 value
        weighted_value = (attn.unsqueeze(-1) * value).reshape(-1, self.query_channels)
        # torch.Tensor, (N_query, query_channels), 按 query 聚合的邻域特征
        aggregated = torch_scatter.scatter_sum(
            weighted_value, idx_query, dim=0, dim_size=int(query_feat.shape[0])
        )
        return self.output_proj(aggregated)


class Stage1AtomHead(nn.Module):
    """
    Stage1 最终分类头: real/pseudo 双向几何 cross-attn(纯残差, 零初始化) + 两个轻量分类尾部。

    分类尾部固定结构 LayerNorm -> Linear -> act -> Linear -> logits:
        - real_atom_head: real 特征 -> atom_logits。
        - pseudo_atom_head: P 特征 -> pseudo_logits。
    末层支持 prior bias 初始化(prior_prob / prior_probs / prior_prob_point_ligand)。

    输入参数:
        - point_channels: int, point backbone 输出通道数, 也是 real/P 特征与 cross-attn 的通道数
        - hidden_dim: int, 分类尾部隐藏通道数
        - atom_logit_dim: int, real_atom_head 输出通道数
        - pseudo_ligand_logit_dim: int, pseudo_atom_head 输出通道数; 1 表示二分类 sigmoid
        - act_layer: type[nn.Module], 激活函数类
        - interaction_radius: float, cross-attn radius 建图半径
        - interaction_max_neighbors: int, cross-attn 每个 query 点最大邻居数
        - interaction_num_heads: int, cross-attn 头数
        - interaction_detach_source_feat: bool, cross-attn 是否对 source 特征 detach
        - prior_prob: float | None, real_atom_head 末层单通道 sigmoid 正类先验; 与 prior_probs 互斥
        - prior_probs: Sequence[float] | None, real_atom_head 末层多通道 softmax 类别先验
        - prior_prob_point_ligand: float | None, pseudo_atom_head 末层单通道 sigmoid 正类先验; None 表示跳过 bias 先验初始化

    前向输入与输出: 见模块顶部 Docstring。
    """

    def __init__(
        self,
        point_channels: int,
        hidden_dim: int,
        atom_logit_dim: int,
        pseudo_ligand_logit_dim: int,
        act_layer: type[nn.Module],
        interaction_radius: float,
        interaction_max_neighbors: int,
        interaction_num_heads: int,
        interaction_detach_source_feat: bool,
        prior_prob: float | None,
        prior_probs: Sequence[float] | None,
        prior_prob_point_ligand: float | None,
    ) -> None:
        super().__init__()
        self.point_channels = int(point_channels)
        self.hidden_dim = int(hidden_dim)
        self.atom_logit_dim = int(atom_logit_dim)
        self.pseudo_ligand_logit_dim = int(pseudo_ligand_logit_dim)

        # GeometricCrossAttention, query=P, source=real, 吃 detached real bind prob
        self.real_to_pseudo = GeometricCrossAttention(
            query_channels=self.point_channels,
            source_channels=self.point_channels,
            num_heads=int(interaction_num_heads),
            radius=float(interaction_radius),
            max_neighbors=int(interaction_max_neighbors),
            detach_source_feat=bool(interaction_detach_source_feat),
            act_layer=act_layer,
            use_source_bind_prob=True,
        )
        # GeometricCrossAttention, query=real, source=P, 不吃 bind prob
        self.pseudo_to_real = GeometricCrossAttention(
            query_channels=self.point_channels,
            source_channels=self.point_channels,
            num_heads=int(interaction_num_heads),
            radius=float(interaction_radius),
            max_neighbors=int(interaction_max_neighbors),
            detach_source_feat=bool(interaction_detach_source_feat),
            act_layer=act_layer,
            use_source_bind_prob=False,
        )

        # nn.Sequential, (N_real, point_channels) -> (N_real, atom_logit_dim), real atom 分类尾部
        self.real_atom_head = nn.Sequential(
            nn.LayerNorm(self.point_channels),
            nn.Linear(self.point_channels, self.hidden_dim),
            act_layer(),
            nn.Linear(self.hidden_dim, self.atom_logit_dim),
        )
        # nn.Sequential, (N_pseudo, point_channels) -> (N_pseudo, pseudo_ligand_logit_dim), P 分类尾部
        self.pseudo_atom_head = nn.Sequential(
            nn.LayerNorm(self.point_channels),
            nn.Linear(self.point_channels, self.hidden_dim),
            act_layer(),
            nn.Linear(self.hidden_dim, self.pseudo_ligand_logit_dim),
        )

        if prior_prob is not None and prior_probs is not None:
            raise ValueError("prior_prob 和 prior_probs 不能同时配置。")
        if prior_probs is not None:
            self._init_linear_multiclass_prior_bias(self.real_atom_head[-1], self.atom_logit_dim, prior_probs)
        elif prior_prob is not None:
            if self.atom_logit_dim != 1:
                raise ValueError("多通道 atom head 请使用 prior_probs，不要使用单通道 prior_prob。")
            # float, sigmoid 正类先验对应的输出 bias = logit(prior)
            bias_val = -math.log((1.0 - float(prior_prob)) / float(prior_prob))
            nn.init.constant_(self.real_atom_head[-1].bias, bias_val)

        if prior_prob_point_ligand is not None:
            if self.pseudo_ligand_logit_dim != 1:
                raise ValueError("多通道 pseudo head 暂不支持单通道 prior_prob_point_ligand。")
            # float, P ligand sigmoid 正类先验对应的输出 bias = logit(prior)
            pseudo_bias_val = -math.log((1.0 - float(prior_prob_point_ligand)) / float(prior_prob_point_ligand))
            nn.init.constant_(self.pseudo_atom_head[-1].bias, pseudo_bias_val)

    @staticmethod
    def _init_linear_multiclass_prior_bias(
        layer: nn.Module,
        logit_dim: int,
        prior_probs: Sequence[float],
    ) -> None:
        """
        用 softmax 类别先验初始化 Linear 输出 bias。

        输入参数:
            - layer: nn.Module, 分类尾部最后一层, 必须是带 bias 的 nn.Linear
            - logit_dim: int, 输出类别通道数
            - prior_probs: Sequence[float], (logit_dim,), softmax 类别先验概率

        输出:
            - None, 原地更新 layer.bias
        """
        if int(logit_dim) <= 1:
            raise ValueError("prior_probs 只适用于多通道 softmax head。")
        # torch.Tensor, (logit_dim,), CPU float32 先验概率向量
        probs = torch.as_tensor(list(prior_probs), dtype=torch.float32)
        if probs.numel() != int(logit_dim):
            raise ValueError(f"prior_probs 长度 {probs.numel()} 与 logit_dim={logit_dim} 不一致。")
        if torch.any(probs <= 0):
            raise ValueError("prior_probs 中所有概率必须大于 0。")
        if not torch.isclose(probs.sum(), torch.tensor(1.0), rtol=1e-4, atol=1e-6):
            raise ValueError(f"prior_probs 总和必须为 1，实际为 {float(probs.sum())}。")
        if not isinstance(layer, nn.Linear) or layer.bias is None:
            raise TypeError("多分类先验初始化要求分类尾部最后一层是带 bias 的 nn.Linear。")
        with torch.no_grad():
            layer.bias.copy_(probs.log().to(device=layer.bias.device, dtype=layer.bias.dtype))

    def forward(
        self,
        point_feat: torch.Tensor,
        point_state: dict[str, Any],
        atom_coord_centered_world: torch.Tensor,
        pseudo_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | None]:
        pseudo_mask = validate_pseudo_mask(pseudo_mask, int(point_feat.shape[0]), name="Stage1AtomHead.forward")
        # torch.Tensor, (N_all,), int64, 每个点所属 BOX 索引
        point_batch = point_state["batch"]

        if pseudo_mask is None:
            # real-only 路径: 全部点都是真实原子, 无 P 槽位
            real_before = point_feat
            real_coord = atom_coord_centered_world
            real_batch = point_batch
            pseudo_before = None
        else:
            # torch.Tensor, (N_all,), bool, True 表示真实原子
            real_mask = ~pseudo_mask
            # torch.Tensor, (N_real, point_channels), cross-attn 前的 real 特征
            real_before = point_feat[real_mask]
            # torch.Tensor, (N_real, 3), real 点 centered-world 坐标
            real_coord = atom_coord_centered_world[real_mask]
            # torch.Tensor, (N_real,), real 点所属 BOX 索引
            real_batch = point_batch[real_mask]
            # torch.Tensor, (N_pseudo, point_channels), cross-attn 前的 P 特征
            pseudo_before = point_feat[pseudo_mask]
            # torch.Tensor, (N_pseudo, 3), P 点 centered-world 坐标
            pseudo_coord = atom_coord_centered_world[pseudo_mask]
            # torch.Tensor, (N_pseudo,), P 点所属 BOX 索引
            pseudo_batch = point_batch[pseudo_mask]

        # torch.Tensor, (N_real, point_channels), 经 pseudo_to_real 增量后的 real 特征(纯残差)
        real_after = real_before
        if pseudo_before is not None:
            real_after = real_before + self.pseudo_to_real(
                query_feat=real_before,
                source_feat=pseudo_before,
                query_coord=real_coord,
                source_coord=pseudo_coord,
                query_batch=real_batch,
                source_batch=pseudo_batch,
            )

        # torch.Tensor, (N_real, atom_logit_dim), real atom 监督 logits
        atom_logits = self.real_atom_head(real_after)
        # torch.Tensor, (N_real, 1), real bind probability(第 0 通道 sigmoid 后 detach), 喂给 real_to_pseudo
        real_bind_prob = torch.sigmoid(atom_logits[:, :1]).detach()

        if pseudo_before is None:
            pseudo_after = None
            pseudo_logits = None
        else:
            # torch.Tensor, (N_pseudo, point_channels), 经 real_to_pseudo 增量后的 P 特征(纯残差)
            pseudo_after = pseudo_before
            pseudo_after = pseudo_before + self.real_to_pseudo(
                query_feat=pseudo_before,
                source_feat=real_before,
                query_coord=pseudo_coord,
                source_coord=real_coord,
                query_batch=pseudo_batch,
                source_batch=real_batch,
                source_bind_prob=real_bind_prob,
            )
            # torch.Tensor, (N_pseudo, pseudo_ligand_logit_dim), P anchor ligand 区域归属 logits
            pseudo_logits = self.pseudo_atom_head(pseudo_after)

        return {
            "real_feat_before_interaction": real_before,
            "real_feat_after_interaction": real_after,
            "pseudo_feat_before_interaction": pseudo_before,
            "pseudo_feat_after_interaction": pseudo_after,
            "atom_logits": atom_logits,
            "pseudo_logits": pseudo_logits,
        }
