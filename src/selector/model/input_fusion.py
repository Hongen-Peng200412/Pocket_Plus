"""融合 Stage1 多层实体特征和 experimental density 上下文。

主要入口:
    - `ResidualSwiGLUFusion`: 按冻结顺序投影多个同实体特征来源，再用残差 SwiGLU
      融合。
    - `VDensityFusion`: 以 sparse `voxel_final` 为基底，按配置加入密度修正。

关闭的密度分支不会接收零填充占位，也不会实例化未使用参数，因此 V48 与 V48+D
两种配置的输入契约与 checkpoint 参数集合明确可区分。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
from torch import nn
from torch.nn import functional as F

from .density_munet_lite import DensityMUNetLite


class ResidualSwiGLUFusion(nn.Module):
    """
    按冻结来源顺序融合实际存在的多层特征。

    输入参数:
        - source_order: Sequence[str], 来源字段的固定顺序；forward 必须提供完全相同的键集合
        - source_dims: Mapping[str, int], 每个来源末维通道数
        - projected_dim: int, 每个来源经 LayerNorm+Linear 后的统一通道数
        - meta_dim: int, 每行基础 meta 的末维通道数；没有 meta 时为 0
        - output_dim: int, 融合输出通道数
        - gate_hidden_dim: int, SwiGLU 中 a/b 的通道数

    前向输入:
        - sources: Mapping[str, torch.Tensor], 每个值均为 (..., C_s)，前导维必须一致
        - meta: torch.Tensor | None, (..., meta_dim)，meta_dim=0 时必须为 None

    前向输出:
        - fused: torch.Tensor, (..., output_dim)，按 residual_swiglu 公式融合的表示
    """

    def __init__(
        self,
        source_order: Sequence[str],
        source_dims: Mapping[str, int],
        projected_dim: int,
        meta_dim: int,
        output_dim: int,
        gate_hidden_dim: int,
    ) -> None:
        """
        初始化逐来源适配器、线性残差和 SwiGLU 门控分支。

        输入参数:
            - source_order: Sequence[str], 特征来源的唯一冻结顺序
            - source_dims: Mapping[str, int], 每个来源的输入通道数；键集合必须与
              `source_order` 完全相同
            - projected_dim: int, 每个来源独立投影后的通道数
            - meta_dim: int, 每个实体附加基础特征的通道数；0 表示不接受 meta
            - output_dim: int, 融合结果通道数
            - gate_hidden_dim: int, SwiGLU 两个门分支各自的通道数

        公式:
            - `joined = concat(project(source_s), meta)`
            - `output = LayerNorm(skip(joined) + out(SiLU(a) * b))`
        """
        super().__init__()
        self.source_order = tuple(str(name) for name in source_order)
        if not self.source_order or len(set(self.source_order)) != len(self.source_order):
            raise ValueError("source_order 必须是非空且无重复的来源序列。")
        if set(self.source_order) != set(source_dims):
            raise ValueError("source_dims 的键必须与 source_order 完全一致。")
        self.meta_dim = int(meta_dim)
        self.source_adapters = nn.ModuleDict(
            {
                name: nn.Sequential(
                    nn.LayerNorm(int(source_dims[name])),
                    nn.Linear(int(source_dims[name]), int(projected_dim)),
                )
                for name in self.source_order
            }
        )
        fused_input_dim = len(self.source_order) * int(projected_dim) + self.meta_dim
        self.skip = nn.Linear(fused_input_dim, int(output_dim))
        self.gate = nn.Linear(fused_input_dim, 2 * int(gate_hidden_dim))
        self.output = nn.Linear(int(gate_hidden_dim), int(output_dim))
        self.norm = nn.LayerNorm(int(output_dim))

    def forward(
        self,
        sources: Mapping[str, torch.Tensor],
        meta: torch.Tensor | None,
    ) -> torch.Tensor:
        """
        按配置顺序执行来源投影和 residual_swiglu。

        输入参数:
            - sources: Mapping[str, torch.Tensor], 每个值为 (..., C_s)，只包含真实存在的配置来源
            - meta: torch.Tensor | None, (..., meta_dim)，基础 meta；无 meta 时为 None

        输出:
            - fused: torch.Tensor, (..., output_dim)，融合结果
        """
        if set(sources) != set(self.source_order):
            missing = sorted(set(self.source_order) - set(sources))
            unexpected = sorted(set(sources) - set(self.source_order))
            raise KeyError(f"多源输入与冻结配置不一致: missing={missing}, unexpected={unexpected}")
        # 列表中每项 shape 为 `(..., projected_dim)`，顺序由冻结配置唯一决定。
        projected = [self.source_adapters[name](sources[name]) for name in self.source_order]
        leading_shape = projected[0].shape[:-1]
        if any(value.shape[:-1] != leading_shape for value in projected[1:]):
            raise ValueError("全部来源的前导实体维必须一致。")
        if self.meta_dim == 0:
            if meta is not None:
                raise ValueError("meta_dim=0 时不得传入 meta。")
            joined = torch.cat(projected, dim=-1)
        else:
            if meta is None or meta.shape[:-1] != leading_shape or meta.shape[-1] != self.meta_dim:
                raise ValueError("meta shape 必须与来源前导维一致且末维等于 meta_dim。")
            joined = torch.cat([*projected, meta], dim=-1)
        # 线性残差保留全部来源的直接路径，SwiGLU 分支学习非线性交互修正。
        residual = self.skip(joined)
        gate_a, gate_b = self.gate(joined).chunk(2, dim=-1)
        return self.norm(residual + self.output(F.silu(gate_a) * gate_b))


class VDensityFusion(nn.Module):
    """
    融合 sparse `voxel_final` 与任务专属密度上下文。

    输入参数:
        - meta_dim: int, 每个 V voxel 的基础 meta 通道数
        - output_dim: int, Selector 内部 V 输出通道数
        - correction_gate_hidden_dim: int, 密度修正逐通道门的隐层通道数
        - use_density: bool, 是否启用 DensityMUNetLite 分支 c
        - density_encoder: DensityMUNetLite | None, use_density=True 时的任务专属密度网络

    前向输入:
        - voxel_final: torch.Tensor, (N_v,C_voxel)，producer 的稀疏最终 V 特征；
          ``C_voxel`` 在首个批次物化投影层
        - density_input: torch.Tensor | None, (B,1,D,H,W)，exp_clipnorm_nopost
        - voxel_index_local_zyx: torch.Tensor, (N_v,3)，目标 V voxel 的 BOX-local 离散 voxel-index ZYX
        - voxel_batch_index: torch.Tensor, (N_v,)，目标 V voxel batch 行
        - meta: torch.Tensor | None, (N_v,meta_dim)，基础 V meta

    前向输出:
        - fused_v: torch.Tensor, (N_v,output_dim)，V48 或 V48+D 表示
    """

    def __init__(
        self,
        meta_dim: int,
        output_dim: int,
        correction_gate_hidden_dim: int,
        use_density: bool,
        density_encoder: DensityMUNetLite | None,
    ) -> None:
        """
        构造 V 基底投影、可选密度分支及逐通道修正门。

        输入参数:
            - meta_dim: int, 每个稀疏 V voxel 的基础附加特征通道数
            - output_dim: int, V 融合结果通道数
            - correction_gate_hidden_dim: int, 密度修正门的隐藏通道数
            - use_density: bool, 是否启用密度上下文分支 `c`
            - density_encoder: DensityMUNetLite | None, 密度分支编码器；必须与
              `use_density` 同时启用或同时关闭

        融合语义:
            - `base` 始终由 48D `voxel_final` 投影得到。
            - 启用的密度修正乘以逐通道 sigmoid 门，再加到 `base`。
            - 密度分支关闭时严格返回归一化后的 `base`。
        """
        super().__init__()
        self.meta_dim = int(meta_dim)
        self.output_dim = int(output_dim)
        self.use_density = bool(use_density)
        self.base_projection = nn.LazyLinear(self.output_dim)
        self.output_norm = nn.LayerNorm(self.output_dim)

        if self.use_density and density_encoder is None:
            raise ValueError("use_density=True 时必须显式提供 DensityMUNetLite。")
        if not self.use_density and density_encoder is not None:
            raise ValueError("use_density=False 时不应实例化未使用的 DensityMUNetLite。")
        self.density_encoder = density_encoder
        self.density_projection = (
            nn.Linear(int(density_encoder.output_channels), self.output_dim)
            if density_encoder is not None
            else None
        )

        # 门控输入只拼接当前配置真实启用的密度修正，不为关闭分支补零。
        gate_input_dim = self.output_dim * (1 + int(self.use_density)) + self.meta_dim
        self.density_gate = (
            nn.Sequential(
                nn.Linear(gate_input_dim, int(correction_gate_hidden_dim)),
                nn.SiLU(),
                nn.Linear(int(correction_gate_hidden_dim), self.output_dim),
            )
            if self.use_density
            else None
        )

    def forward(
        self,
        voxel_final: torch.Tensor,
        density_input: torch.Tensor | None,
        voxel_index_local_zyx: torch.Tensor,
        voxel_batch_index: torch.Tensor,
        meta: torch.Tensor | None,
    ) -> torch.Tensor:
        """
        计算当前配置的 V48 或 V48+D，并保持密度分支关闭时严格退化。

        输入参数:
            - voxel_final: torch.Tensor, (N_v,C_voxel)，sparse final V feature；
              首次调用固定 ``C_voxel``
            - density_input: torch.Tensor | None, (B,1,D,H,W)，启用 c 时的 exp_clipnorm_nopost
            - voxel_index_local_zyx: torch.Tensor, (N_v,3)，目标 V voxel 的 BOX-local 离散 voxel-index ZYX
            - voxel_batch_index: torch.Tensor, (N_v,)，目标 voxel batch 行
            - meta: torch.Tensor | None, (N_v,meta_dim)，基础 meta

        输出:
            - fused_v: torch.Tensor, (N_v,output_dim)，融合后的 V token value
        """
        # `(N_v, output_dim)`，由 sparse 48D Stage1 最终特征得到的始终存在的基底。
        base = self.base_projection(voxel_final)
        if not self.use_density:
            if density_input is not None:
                raise ValueError("未启用 density 分支时不得传入 density_input。")
            return self.output_norm(base)
        if self.meta_dim == 0:
            if meta is not None:
                raise ValueError("meta_dim=0 时不得传入 meta。")
        elif meta is None or meta.shape != (voxel_final.shape[0], self.meta_dim):
            raise ValueError("V meta shape 与配置不一致。")

        if density_input is None:
            raise ValueError("V48+D 配置必须提供 density_input。")
        assert self.density_encoder is not None and self.density_projection is not None
        # `(B, C_density, D, H, W)`，只在当前前向内存在的完整分辨率密度上下文。
        dense_context = self.density_encoder(density_input)
        local_index = voxel_index_local_zyx.to(dtype=torch.long)
        # `(N_v, C_density)`，按批次下标和 BOX-local ZYX 索引直接收集稀疏位置。
        density_rows = dense_context[
            voxel_batch_index.to(dtype=torch.long),
            :,
            local_index[:, 0],
            local_index[:, 1],
            local_index[:, 2],
        ]
        density_value = self.density_projection(density_rows)

        gate_parts = [base, density_value]
        if meta is not None:
            gate_parts.append(meta)
        gate_input = torch.cat(gate_parts, dim=-1)
        assert self.density_gate is not None
        fused = base + torch.sigmoid(self.density_gate(gate_input)) * density_value
        return self.output_norm(fused)
