"""
Stage1 atom head 的 real/pseudo 双尾部实现。

对齐契约（修改时必须全量同步）:

    - 本段、CLAUDE/plans/implement/tri_ligand_sparse_refine/00-master.md、src/model/stage1_model.py::_run_atom_head 和 tests/model/test_stage1_atom_head.py 必须同步更新。
    - N_all 表示当前 atom head 输入点数; N_real 表示真实原子数; N_pseudo 表示 P anchor 数; C_point=point_channels; C_hidden=hidden_dim。
    - pseudo_mask is None 表示 real-only 路径, 此时 N_all=N_real 且不产生 pseudo_feature。

    - pseudo_mask: torch.Tensor, (N_all,), bool, True 表示 P anchor, False 表示 real atom; 若非 None 必须与 point_feat 第一维一致。
    - point_feat: torch.Tensor, (N_all, C_point), floating, 最后一轮 point backbone 输出特征, mixed 路径下按 pseudo_atoms.py 的 mixed layout 排列; 调用方(stage1_model)按 detach 路由开关传入已 detach 的 fused_point_feat。
    - pseudo_density_feat: torch.Tensor | None, (N_pseudo, C_density), floating, 可选 density cube 特征(anchor 顺序); 仅 pseudo_density_residual=True 时消费, 加到 pseudo 分支输出上。
    - point_state["coord"]: torch.Tensor, (N_all, 3), floating, Point/Block 使用的点坐标, 轴顺序 (x, y, z)。
    - point_state["batch"]: torch.Tensor, (N_all,), int64/long, 每个点所属 BOX 索引。
    - point_state["offset"]: torch.Tensor, (B,), int64/long, 每个 BOX 在展平点序列中的结束偏移。
    - point_state["grid_size"]: float, PTV3 序列化/网格化使用的点云 grid size。
    - point_state["grid_coord"]: torch.Tensor, (N_all, 3), int32/int64, 可选字段, 离散网格坐标。
    - atom_coord_centered_world: torch.Tensor, (N_all, 3), floating, token 可选拼接的 centered-world 坐标, 轴顺序 (x, y, z)。
    - atom_valid_mask: torch.Tensor, (N_all,), bool, real atom 监督掩码; P anchor 槽位必须为 False。
    - outputs["atom_tokens"]: torch.Tensor, (N_all, C_point) 或 (N_all, C_point+4), floating, token projection 前的输入 token; append_coord_mask=True 时追加 xyz 与 valid mask。
    - outputs["atom_hidden"]: torch.Tensor, (N_all, C_hidden), floating, shared attention stack 输出, 保留 mixed 全点顺序。
    - outputs["atom_logits"]: torch.Tensor | None, (N_real, atom_logit_dim), floating, 只对 real atom 输出的监督 logits; enable_atom_head_back=False 时为 None。
    - outputs["pseudo_feature"]: torch.Tensor | None, (N_pseudo, pseudo_feature_dim), floating, 只对 P anchor 输出; real-only 路径为 None; pseudo_density_residual=True 时含 density cube 残差。
"""
from __future__ import annotations

import math
from typing import Any, Sequence

import torch
from torch import nn

from src.model.typed_point import (
    TypedPointConfig,
    apply_type_aware_tensor_module,
    normalize_typed_point_cfg,
    validate_pseudo_mask,
)

_PTV3_HEAD_IMPORT_ERROR: Exception | None = None
try:
    from src.model.PTV3bakcbone.model import Point, Block
except Exception as exc:  # pragma: no cover - 依赖当前本地环境
    Point = None
    Block = None
    _PTV3_HEAD_IMPORT_ERROR = exc




class Stage1SerializedAttentionStack(nn.Module):
    def __init__(
        self,
        channels: int,
        num_heads: int,
        patch_size: int,
        num_layers: int,
        serialization_orders: Sequence[str],
        shuffle_orders: bool,
        qkv_bias: bool,
        qk_scale: float | None,
        attn_drop: float,
        proj_drop: float,
        enable_rpe: bool,
        enable_flash: bool,
        upcast_attention: bool,
        upcast_softmax: bool,
        atom_head_ffn_type: str,
        mlp_ratio: int,
        act_layer: type[nn.Module],
        cpe_impl: str,
        cpe_kernel_size: int,
        cpe_receptive_field: float,
        pointconv_block_max_neighbors: int,
        drop_path: float,
        pre_norm: bool,
        typed_point_cfg: dict[str, Any] | TypedPointConfig | None = None,
    ) -> None:
        """
        用 PTV3 Block 堆叠处理 Stage1 atom token。

        输入参数:
            - channels: int, forward 时 token 隐藏通道数
            - num_heads: int, attention 头数
            - patch_size: int, SerializedAttention patch 点数
            - num_layers: int, Block 层数
            - serialization_orders: Sequence[str], 序列化顺序列表
            - shuffle_orders: bool, forward 时是否打乱序列化顺序列表
            - qkv_bias: bool, QKV 线性层是否使用 bias
            - qk_scale: float | None, QK 缩放因子
            - attn_drop: float, attention dropout 概率
            - proj_drop: float, 输出投影 dropout 概率
            - enable_rpe: bool, 是否启用相对位置编码
            - enable_flash: bool, 是否启用 flash attention
            - upcast_attention: bool, attention 矩阵乘法前是否上转精度
            - upcast_softmax: bool, softmax 前是否上转精度
            - atom_head_ffn_type: str, atom head Block FFN 类型, 取值 "mlp" / "gated" / "none"
            - mlp_ratio: int, FFN 隐藏层膨胀倍率
            - act_layer: type[nn.Module], 激活函数类
            - cpe_impl: str, CPE 实现方式, 取值 "none" / "pointconv"
            - cpe_kernel_size: int, legacy 字段; pointconv CPE 不消费该值
            - cpe_receptive_field: float, pointconv CPE 世界坐标感受野半径
            - pointconv_block_max_neighbors: int, pointconv CPE 每个点最大邻居数
            - drop_path: float, Block 随机深度概率
            - pre_norm: bool, 是否使用 pre-norm Block

        forward 输入:
            - point_state: dict[str, Any], point backbone 输出的点状态, 至少包含 coord/batch/offset/grid_size
            - token_feat: torch.Tensor, (N, channels), atom token 投影后的隐藏特征
            - pseudo_mask: torch.Tensor | None, (N,), True 表示 P anchor; None 表示 real-only 路径

        forward 输出:
            - output_token_feat: torch.Tensor, (N, channels), attention stack 输出特征
        """
        super().__init__()
        if Block is None:
            raise ImportError("Stage1SerializedAttentionStack 需要 PTV3 Block。") from _PTV3_HEAD_IMPORT_ERROR
        self.channels = int(channels)
        self.serialization_orders = tuple(str(order_name) for order_name in serialization_orders)
        if len(self.serialization_orders) == 0:
            raise ValueError("serialization_orders 不能为空。")
        self.shuffle_orders = bool(shuffle_orders)
        # TypedPointConfig, atom head attention stack 使用的 typed point 配置
        self.typed_point_cfg = normalize_typed_point_cfg(typed_point_cfg)

        # nn.ModuleList, 长度 num_layers, atom token 的共享 attention Block 序列
        self.layers = nn.ModuleList(
            [
                Block(
                    channels=self.channels,
                    num_heads=int(num_heads),
                    patch_size=int(patch_size),
                    order_index=int(layer_idx % len(self.serialization_orders)),
                    cpe_impl=str(cpe_impl),
                    qkv_bias=bool(qkv_bias),
                    qk_scale=qk_scale,
                    attn_drop=float(attn_drop),
                    proj_drop=float(proj_drop),
                    enable_rpe=bool(enable_rpe),
                    enable_flash=bool(enable_flash),
                    upcast_attention=bool(upcast_attention),
                    upcast_softmax=bool(upcast_softmax),
                    ffn_type=str(atom_head_ffn_type),
                    mlp_ratio=int(mlp_ratio),
                    act_layer=act_layer,
                    pre_norm=bool(pre_norm),
                    drop_path=float(drop_path),
                    cpe_kernel_size=int(cpe_kernel_size),
                    cpe_receptive_field=float(cpe_receptive_field),
                    pointconv_block_max_neighbors=int(pointconv_block_max_neighbors),
                    separate_qkv=self.typed_point_cfg.use_separate_qkv,
                    separate_attn_proj=self.typed_point_cfg.use_separate_attn_proj,
                    separate_ffn=self.typed_point_cfg.use_separate_ffn,
                    separate_cpe=self.typed_point_cfg.use_separate_cpe,
                )
                for layer_idx in range(int(num_layers))
            ]
        )
        # nn.LayerNorm, (N, channels), stack 末尾输出归一化
        self.output_norm = nn.LayerNorm(self.channels)

    def forward(
        self,
        point_state: dict[str, Any],
        token_feat: torch.Tensor,
        pseudo_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if token_feat.shape[0] == 0:
            return token_feat
        if Point is None:
            raise ImportError("Stage1SerializedAttentionStack 需要 PTV3 Point。") from _PTV3_HEAD_IMPORT_ERROR
        pseudo_mask = validate_pseudo_mask(
            pseudo_mask,
            int(token_feat.shape[0]),
            name="Stage1SerializedAttentionStack.forward",
        )

        # dict[str, Any], (N, *), 用 point backbone 状态和 atom token 重建的 Point 输入
        point_dict = {
            "feat": token_feat,                # (N, C_atom)
            "coord": point_state["coord"],     # (N, 3)
            "batch": point_state["batch"],     # (N,), 每个点的batch索引
            "offset": point_state["offset"],   # (B,), 每个样本的结束索引
            "grid_size": point_state["grid_size"],  # float
        }
        if "grid_coord" in point_state:
            point_dict["grid_coord"] = point_state["grid_coord"]
        if pseudo_mask is not None:
            point_dict["pseudo_mask"] = pseudo_mask

        # Point, (N, channels), atom head attention 的点对象
        point = Point(point_dict)
        point.serialization(order=self.serialization_orders, shuffle_orders=self.shuffle_orders)

        for layer in self.layers:
            point = layer(point)

        # torch.Tensor, (N, channels), 归一化后的 atom hidden
        return self.output_norm(point.feat)




class Stage1AtomHead(nn.Module):
    def __init__(
        self,
        point_channels: int,
        hidden_dim: int,
        num_heads: int,
        patch_size: int,
        num_layers: int,
        serialization_orders: Sequence[str],
        shuffle_orders: bool,
        qkv_bias: bool,
        qk_scale: float | None,
        attn_drop: float,
        proj_drop: float,
        enable_rpe: bool,
        enable_flash: bool,
        upcast_attention: bool,
        upcast_softmax: bool,
        atom_logit_dim: int,
        pseudo_feature_dim: int | None,
        atom_head_ffn_type: str,
        mlp_ratio: int,
        act_layer: type[nn.Module],
        cpe_impl: str,
        cpe_kernel_size: int,
        cpe_receptive_field: float,
        pointconv_block_max_neighbors: int,
        drop_path: float,
        pre_norm: bool,
        append_coord_mask: bool,
        prior_prob: float | None = None,
        prior_probs: Sequence[float] | None = None,
        enable_atom_head_back: bool = True,
        pseudo_density_residual: bool = False,
        pseudo_density_in_dim: int | None = None,
        typed_point_cfg: dict[str, Any] | TypedPointConfig | None = None,
    ) -> None:
        """
        Stage1 atom head: mixed 点共享 attention, real 输出 atom logits, pseudo 输出 P anchor feature。

        输入参数:
            - point_channels: int, point backbone 输出通道数
            - hidden_dim: int, atom head 隐藏通道数
            - num_heads: int, attention 头数
            - patch_size: int, SerializedAttention patch 点数
            - num_layers: int, Block 层数
            - serialization_orders: Sequence[str], 序列化顺序列表
            - shuffle_orders: bool, forward 时是否打乱序列化顺序列表
            - qkv_bias: bool, QKV 线性层是否使用 bias
            - qk_scale: float | None, QK 缩放因子
            - attn_drop: float, attention dropout 概率
            - proj_drop: float, 输出投影 dropout 概率
            - enable_rpe: bool, 是否启用相对位置编码
            - enable_flash: bool, 是否启用 flash attention
            - upcast_attention: bool, attention 矩阵乘法前是否上转精度
            - upcast_softmax: bool, softmax 前是否上转精度
            - atom_logit_dim: int, real atom logits 输出通道数
            - pseudo_feature_dim: int | None, P anchor feature 输出通道数; None 表示等于 hidden_dim
            - atom_head_ffn_type: str, atom head Block FFN 类型, 取值 "mlp" / "gated" / "none"
            - mlp_ratio: int, FFN 隐藏层膨胀倍率
            - act_layer: type[nn.Module], 激活函数类
            - cpe_impl: str, CPE 实现方式, 取值 "none" / "pointconv"
            - cpe_kernel_size: int, legacy 字段; pointconv CPE 不消费该值
            - cpe_receptive_field: float, pointconv CPE 世界坐标感受野半径
            - pointconv_block_max_neighbors: int, pointconv CPE 每个点最大邻居数
            - drop_path: float, Block 随机深度概率
            - pre_norm: bool, 是否使用 pre-norm Block
            - append_coord_mask: bool, 是否把 centered-world 坐标和 atom_valid_mask(监督标志) 拼入 token
            - prior_prob: float | None, 单通道 sigmoid 正类先验概率
            - prior_probs: Sequence[float] | None, 多通道 softmax 类别先验概率
            - enable_atom_head_back: bool, 是否构造后置头 real_atom_logit_head; 关时 forward 的 atom_logits=None, prior bias 初始化随之跳过
            - pseudo_density_residual: bool, 是否在 pseudo_feature 上加 density cube 直通残差(LayerNorm->Linear, 末层零初始化)
            - pseudo_density_in_dim: int | None, density cube 特征通道数; 仅在 pseudo_density_residual=True 时必填, 由 stage1_model 注入为 point_backbone.atom_feature_dim

        前向输入:
            - point_feat: torch.Tensor, (N_all, point_channels), point backbone 输出点特征
            - point_state: dict[str, Any], 与 point_feat 同布局的点状态
            - atom_coord_centered_world: torch.Tensor, (N_all, 3), centered-world 坐标
            - atom_valid_mask: torch.Tensor, (N_all,), bool, real atom 监督掩码; P anchor 应为 False
            - pseudo_mask: torch.Tensor | None, (N_all,), True 表示 P anchor; None 表示 real-only 路径
            - pseudo_density_feat: torch.Tensor | None, (N_pseudo, pseudo_density_in_dim), P 来源 density cube 特征(anchor 顺序); 仅 pseudo_density_residual=True 时消费

        前向输出:
            - outputs: dict[str, torch.Tensor | None], atom head 输出字典
                - atom_tokens: torch.Tensor, (N_all, point_channels) 或 (N_all, point_channels + 4), token projection 前输入
                - atom_hidden: torch.Tensor, (N_all, hidden_dim), shared attention stack 输出
                - atom_logits: torch.Tensor | None, (N_real, atom_logit_dim), 真实原子的 logits; 后置头关时为 None
                - pseudo_feature: torch.Tensor | None, (N_pseudo, pseudo_feature_dim), P anchor refined feature; real-only 路径为 None
        """
        super().__init__()
        self.point_channels = int(point_channels)
        self.hidden_dim = int(hidden_dim)
        self.atom_logit_dim = int(atom_logit_dim)
        self.append_coord_mask = bool(append_coord_mask)
        self.pseudo_feature_dim = int(pseudo_feature_dim) if pseudo_feature_dim is not None else self.hidden_dim
        # TypedPointConfig, atom head 使用的 typed point 配置
        self.typed_point_cfg = normalize_typed_point_cfg(typed_point_cfg)

        # int, atom token projection 输入通道数; append_coord_mask=True 时追加 xyz 与 bool mask
        token_input_dim = self.point_channels + (4 if self.append_coord_mask else 0)
        if self.typed_point_cfg.use_separate_atom_token_proj:
            # nn.Sequential, real 点 atom token 输入投影
            self.atom_token_proj_real = self._build_atom_token_proj(token_input_dim, act_layer)
            # nn.Sequential, pseudo 点 atom token 输入投影
            self.atom_token_proj_pseudo = self._build_atom_token_proj(token_input_dim, act_layer)
            self.atom_token_proj = None
        else:
            # nn.Sequential, (N_all, token_input_dim) -> (N_all, hidden_dim), atom token 输入投影
            self.atom_token_proj = self._build_atom_token_proj(token_input_dim, act_layer)
            self.atom_token_proj_real = None
            self.atom_token_proj_pseudo = None
        # Stage1SerializedAttentionStack, (N_all, hidden_dim), mixed 或 real-only 共享 attention
        self.atom_attention_stack = Stage1SerializedAttentionStack(
            channels=self.hidden_dim,
            num_heads=int(num_heads),
            patch_size=int(patch_size),
            num_layers=int(num_layers),
            serialization_orders=serialization_orders,
            shuffle_orders=bool(shuffle_orders),
            qkv_bias=bool(qkv_bias),
            qk_scale=qk_scale,
            attn_drop=float(attn_drop),
            proj_drop=float(proj_drop),
            enable_rpe=bool(enable_rpe),
            enable_flash=bool(enable_flash),
            upcast_attention=bool(upcast_attention),
            upcast_softmax=bool(upcast_softmax),
            atom_head_ffn_type=str(atom_head_ffn_type),
            mlp_ratio=int(mlp_ratio),
            act_layer=act_layer,
            cpe_impl=str(cpe_impl),
            cpe_kernel_size=int(cpe_kernel_size),
            cpe_receptive_field=float(cpe_receptive_field),
            pointconv_block_max_neighbors=int(pointconv_block_max_neighbors),
            drop_path=float(drop_path),
            pre_norm=bool(pre_norm),
            typed_point_cfg=self.typed_point_cfg,
        )
        # bool, 是否构造后置头
        self.enable_atom_head_back = bool(enable_atom_head_back)
        # nn.Sequential | None, (N_real, hidden_dim) -> (N_real, atom_logit_dim), real atom 分类后置头; 关时为 None
        self.real_atom_logit_head = (
            nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim),
                act_layer(),
                nn.Linear(self.hidden_dim, self.atom_logit_dim),
            )
            if self.enable_atom_head_back
            else None
        )
        # nn.Sequential, (N_pseudo, hidden_dim) -> (N_pseudo, pseudo_feature_dim), P anchor feature 尾部
        self.pseudo_feature_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            act_layer(),
            nn.Linear(self.hidden_dim, self.pseudo_feature_dim),
        )

        # nn.Sequential | None, (N_pseudo, pseudo_density_in_dim) -> (N_pseudo, pseudo_feature_dim), density cube 直通残差; 关时为 None
        self.density_residual = None
        if bool(pseudo_density_residual):
            if pseudo_density_in_dim is None:
                raise ValueError("pseudo_density_residual=True 时必须提供 pseudo_density_in_dim。")
            self.density_residual = nn.Sequential(
                nn.LayerNorm(int(pseudo_density_in_dim)),
                nn.Linear(int(pseudo_density_in_dim), self.pseudo_feature_dim),
            )
            # 末层 Linear 零初始化, 使残差初值为 0、不改变 pseudo_feature
            nn.init.zeros_(self.density_residual[-1].weight)
            nn.init.zeros_(self.density_residual[-1].bias)

        if prior_prob is not None and prior_probs is not None:
            raise ValueError("prior_prob 和 prior_probs 不能同时配置。")
        if self.real_atom_logit_head is not None:
            # 后置头关时不构造分类尾部, prior bias 初始化随之跳过
            if prior_probs is not None:
                self._init_linear_multiclass_prior_bias(self.real_atom_logit_head[2], self.atom_logit_dim, prior_probs)
            elif prior_prob is not None:
                if self.atom_logit_dim != 1:
                    raise ValueError("多通道 atom head 请使用 prior_probs，不要使用单通道 prior_prob。")
                # float, sigmoid 正类先验对应的输出 bias
                bias_val = -math.log((1.0 - float(prior_prob)) / float(prior_prob))
                nn.init.constant_(self.real_atom_logit_head[2].bias, bias_val)

    def _build_atom_token_proj(
        self,
        token_input_dim: int,
        act_layer: type[nn.Module],
    ) -> nn.Module:
        """
        构造 atom token 输入投影模块。

        输入参数:
            - token_input_dim: int, atom token 输入通道数
            - act_layer: type[nn.Module], 激活函数类

        输出:
            - module: nn.Module, (N_all, token_input_dim) -> (N_all, hidden_dim) 的投影模块
        """
        return nn.Sequential(
            nn.Linear(token_input_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            act_layer(),
        )

    @staticmethod
    def _init_linear_multiclass_prior_bias(
        layer: nn.Module,
        logit_dim: int,
        prior_probs: Sequence[float],
    ) -> None:
        """
        用 softmax 类别先验初始化 Linear 输出 bias。

        输入参数:
            - layer: nn.Module, atom logits 最后一层, 必须是带 bias 的 nn.Linear
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
            raise TypeError("多分类先验初始化要求 atom logits 最后一层是带 bias 的 nn.Linear。")
        with torch.no_grad():
            layer.bias.copy_(probs.log().to(device=layer.bias.device, dtype=layer.bias.dtype))

    def forward(
        self,
        point_feat: torch.Tensor,
        point_state: dict[str, Any],
        atom_coord_centered_world: torch.Tensor,
        atom_valid_mask: torch.Tensor,
        pseudo_mask: torch.Tensor | None = None,
        pseudo_density_feat: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | None]:
        pseudo_mask = validate_pseudo_mask(pseudo_mask, int(point_feat.shape[0]), name="Stage1AtomHead.forward")

        if self.append_coord_mask:
            # torch.Tensor, (N_all, point_channels + 4), 点特征 + centered-world xyz + real 监督 mask
            atom_tokens = torch.cat(
                [
                    point_feat,
                    atom_coord_centered_world,
                    atom_valid_mask.to(dtype=point_feat.dtype).unsqueeze(-1),
                ],
                dim=-1,
            )
        else:
            # torch.Tensor, (N_all, point_channels), 纯 point backbone 特征 token
            atom_tokens = point_feat

        if self.typed_point_cfg.use_separate_atom_token_proj:
            # torch.Tensor, (N_all, hidden_dim), type-aware atom token 投影结果
            atom_hidden = apply_type_aware_tensor_module(
                atom_tokens,
                pseudo_mask,
                self.atom_token_proj_real,
                self.atom_token_proj_pseudo,
            )
        else:
            # torch.Tensor, (N_all, hidden_dim), shared atom token 投影结果
            atom_hidden = self.atom_token_proj(atom_tokens)
        # torch.Tensor, (N_all, hidden_dim), shared attention stack 输出; mixed 路径保留全点顺序
        atom_hidden = self.atom_attention_stack(
            point_state=point_state,
            token_feat=atom_hidden,
            pseudo_mask=pseudo_mask,
        )

        if pseudo_mask is None:
            # torch.Tensor, (N_all, hidden_dim), real-only 路径下全部点都是真实原子
            real_hidden = atom_hidden
            pseudo_feature = None
        else:
            # torch.Tensor, (N_all,), bool, True 表示真实原子
            real_mask = ~pseudo_mask
            # torch.Tensor, (N_real, hidden_dim), real atom hidden
            real_hidden = atom_hidden[real_mask]
            # torch.Tensor, (N_pseudo, pseudo_feature_dim), P anchor refined feature
            pseudo_feature = self.pseudo_feature_head(atom_hidden[pseudo_mask])
            if self.density_residual is not None and pseudo_density_feat is not None:
                # density cube 直通残差; 末层零初始化使初值不改变 pseudo_feature
                pseudo_feature = pseudo_feature + self.density_residual(pseudo_density_feat)

        # torch.Tensor | None, (N_real, atom_logit_dim), 真实原子的 logits; 后置头关时为 None
        atom_logits = self.real_atom_logit_head(real_hidden) if self.real_atom_logit_head is not None else None
        return {
            "atom_tokens": atom_tokens,
            "atom_hidden": atom_hidden,
            "atom_logits": atom_logits,
            "pseudo_feature": pseudo_feature,
        }
