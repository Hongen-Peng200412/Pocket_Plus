"""Stage1 多层实体特征的 residual_swiglu 与 V5+D 融合。"""

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
        residual = self.skip(joined)
        gate_a, gate_b = self.gate(joined).chunk(2, dim=-1)
        return self.norm(residual + self.output(F.silu(gate_a) * gate_b))


def sample_native_feature_grids(
    feature_grid: torch.Tensor,
    voxel_index_local_zyx: torch.Tensor,
    voxel_batch_index: torch.Tensor,
    box_shape_zyx: torch.Tensor,
) -> torch.Tensor:
    """
    在目标 80³ voxel center 处三线性采样一张原生低分辨率 feature grid。

    输入参数:
        - feature_grid: torch.Tensor, (B,C,D_s,H_s,W_s)，某一 producer 原生低分辨率网格
        - voxel_index_local_zyx: torch.Tensor, (N_v,3)，目标 voxel 的局部整数 ZYX index
        - voxel_batch_index: torch.Tensor, (N_v,)，每个目标 voxel 所属 batch 行
        - box_shape_zyx: torch.Tensor, (B,3) 或 (3,)，原始 BOX shape；正式配置为 80³

    输出:
        - sampled: torch.Tensor, (N_v,C)，与输入目标 voxel 行严格对齐的三线性采样特征
    """
    if box_shape_zyx.ndim == 1:
        box_shapes = box_shape_zyx.reshape(1, 3).expand(feature_grid.shape[0], -1)
    else:
        box_shapes = box_shape_zyx
    result = feature_grid.new_empty((voxel_index_local_zyx.shape[0], feature_grid.shape[1]))
    for batch_index in range(feature_grid.shape[0]):
        row_index = torch.nonzero(voxel_batch_index == batch_index, as_tuple=False).reshape(-1)
        if row_index.numel() == 0:
            continue
        indices_zyx = voxel_index_local_zyx[row_index].to(dtype=feature_grid.dtype)
        shape_zyx = box_shapes[batch_index].to(device=feature_grid.device, dtype=feature_grid.dtype)
        # align_corners=False 时，index i 的 voxel center 对应 2*(i+0.5)/L-1。
        normalized_zyx = 2.0 * (indices_zyx + 0.5) / shape_zyx - 1.0
        normalized_xyz = normalized_zyx[:, [2, 1, 0]]
        sample_grid = normalized_xyz.reshape(1, -1, 1, 1, 3)
        sampled = F.grid_sample(
            feature_grid[batch_index : batch_index + 1],
            sample_grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=False,
        )
        result[row_index] = sampled[0, :, :, 0, 0].transpose(0, 1)
    return result


class V5DensityFusion(nn.Module):
    """
    融合 sparse voxel_final、多尺度原生 V grid 与任务专属密度上下文。

    输入参数:
        - grid_source_order: Sequence[str], 多尺度来源顺序，正式为 voxel_ds_2/3/4/c4
        - grid_source_dims: Mapping[str, int], 每张原生 grid 的通道数
        - meta_dim: int, 每个 V voxel 的基础 meta 通道数
        - output_dim: int, V 输出通道数；正式为 48
        - fusion_projected_dim: int, 多尺度各来源投影通道数
        - fusion_gate_hidden_dim: int, residual_swiglu 隐层通道数
        - correction_gate_hidden_dim: int, m/c 逐通道 gate 的隐层通道数
        - use_multiscale: bool, 是否启用四张原生低分辨率网格分支 m
        - use_density: bool, 是否启用 DensityMUNetLite 分支 c
        - density_encoder: DensityMUNetLite | None, use_density=True 时的任务专属密度网络

    前向输入:
        - voxel_final: torch.Tensor, (N_v,48)，producer sparse final V feature
        - native_grids: Mapping[str, torch.Tensor], 每项为 (B,C,D_s,H_s,W_s)
        - density_input: torch.Tensor | None, (B,1,D,H,W)，exp_clipnorm_nopost
        - voxel_index_local_zyx: torch.Tensor, (N_v,3)，目标 V voxel 局部 ZYX index
        - voxel_batch_index: torch.Tensor, (N_v,)，目标 V voxel batch 行
        - box_shape_zyx: torch.Tensor, (B,3)，原始 BOX shape
        - meta: torch.Tensor | None, (N_v,meta_dim)，基础 V meta

    前向输出:
        - fused_v: torch.Tensor, (N_v,output_dim)，V48、V5 或 V5+D 表示
    """

    def __init__(
        self,
        grid_source_order: Sequence[str],
        grid_source_dims: Mapping[str, int],
        meta_dim: int,
        output_dim: int,
        fusion_projected_dim: int,
        fusion_gate_hidden_dim: int,
        correction_gate_hidden_dim: int,
        use_multiscale: bool,
        use_density: bool,
        density_encoder: DensityMUNetLite | None,
    ) -> None:
        super().__init__()
        self.grid_source_order = tuple(str(name) for name in grid_source_order)
        self.meta_dim = int(meta_dim)
        self.output_dim = int(output_dim)
        self.use_multiscale = bool(use_multiscale)
        self.use_density = bool(use_density)
        self.base_projection = nn.Linear(48, self.output_dim)
        self.output_norm = nn.LayerNorm(self.output_dim)

        if self.use_multiscale:
            self.multiscale_fusion: ResidualSwiGLUFusion | None = ResidualSwiGLUFusion(
                source_order=self.grid_source_order,
                source_dims=grid_source_dims,
                projected_dim=int(fusion_projected_dim),
                meta_dim=self.meta_dim,
                output_dim=self.output_dim,
                gate_hidden_dim=int(fusion_gate_hidden_dim),
            )
        else:
            self.multiscale_fusion = None
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

        # gate 输入只拼接当前配置真实启用的 correction，不为关闭分支补零。
        gate_input_dim = self.output_dim * (1 + int(self.use_multiscale) + int(self.use_density)) + self.meta_dim
        self.multiscale_gate = (
            nn.Sequential(
                nn.Linear(gate_input_dim, int(correction_gate_hidden_dim)),
                nn.SiLU(),
                nn.Linear(int(correction_gate_hidden_dim), self.output_dim),
            )
            if self.use_multiscale
            else None
        )
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
        native_grids: Mapping[str, torch.Tensor],
        density_input: torch.Tensor | None,
        voxel_index_local_zyx: torch.Tensor,
        voxel_batch_index: torch.Tensor,
        box_shape_zyx: torch.Tensor,
        meta: torch.Tensor | None,
    ) -> torch.Tensor:
        """
        计算当前配置的 V48、V5 或 V5+D，并保持双分支关闭时严格退化。

        输入参数:
            - voxel_final: torch.Tensor, (N_v,48)，sparse final V feature
            - native_grids: Mapping[str, torch.Tensor], 启用 m 时包含全部配置原生网格
            - density_input: torch.Tensor | None, (B,1,D,H,W)，启用 c 时的 exp_clipnorm_nopost
            - voxel_index_local_zyx: torch.Tensor, (N_v,3)，目标 voxel 局部 ZYX index
            - voxel_batch_index: torch.Tensor, (N_v,)，目标 voxel batch 行
            - box_shape_zyx: torch.Tensor, (B,3)，原始 BOX shape
            - meta: torch.Tensor | None, (N_v,meta_dim)，基础 meta

        输出:
            - fused_v: torch.Tensor, (N_v,output_dim)，融合后的 V token value
        """
        base = self.base_projection(voxel_final)
        if not self.use_multiscale and not self.use_density:
            return self.output_norm(base)
        if self.meta_dim == 0:
            if meta is not None:
                raise ValueError("meta_dim=0 时不得传入 meta。")
        elif meta is None or meta.shape != (voxel_final.shape[0], self.meta_dim):
            raise ValueError("V meta shape 与配置不一致。")

        corrections: list[torch.Tensor] = []
        multiscale_value: torch.Tensor | None = None
        density_value: torch.Tensor | None = None
        if self.use_multiscale:
            if set(native_grids) != set(self.grid_source_order):
                raise KeyError("native_grids 必须与冻结的多尺度来源清单完全一致。")
            sampled_sources = {
                name: sample_native_feature_grids(
                    feature_grid=native_grids[name],
                    voxel_index_local_zyx=voxel_index_local_zyx,
                    voxel_batch_index=voxel_batch_index,
                    box_shape_zyx=box_shape_zyx,
                )
                for name in self.grid_source_order
            }
            assert self.multiscale_fusion is not None
            multiscale_value = self.multiscale_fusion(sampled_sources, meta)
            corrections.append(multiscale_value)
        elif native_grids:
            raise ValueError("V48/V48+D 配置不得传入未消费的 native_grids。")

        if self.use_density:
            if density_input is None:
                raise ValueError("V5+D 配置必须提供 density_input。")
            assert self.density_encoder is not None and self.density_projection is not None
            dense_context = self.density_encoder(density_input)
            local_index = voxel_index_local_zyx.to(dtype=torch.long)
            density_rows = dense_context[
                voxel_batch_index.to(dtype=torch.long),
                :,
                local_index[:, 0],
                local_index[:, 1],
                local_index[:, 2],
            ]
            density_value = self.density_projection(density_rows)
            corrections.append(density_value)
        elif density_input is not None:
            raise ValueError("未启用 density 分支时不得传入 density_input。")

        gate_parts = [base, *corrections]
        if meta is not None:
            gate_parts.append(meta)
        gate_input = torch.cat(gate_parts, dim=-1)
        fused = base
        if multiscale_value is not None:
            assert self.multiscale_gate is not None
            fused = fused + torch.sigmoid(self.multiscale_gate(gate_input)) * multiscale_value
        if density_value is not None:
            assert self.density_gate is not None
            fused = fused + torch.sigmoid(self.density_gate(gate_input)) * density_value
        return self.output_norm(fused)
