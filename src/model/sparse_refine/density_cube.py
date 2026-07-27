from __future__ import annotations

import torch
from torch import nn

from src.model.utils import gather_voxel_cube


class DensityCubeEncoder(nn.Module):
    """
    抽取 P anchor 周围 density cube 并编码为 pseudo atom 初始特征. 

    输入参数:
        - in_channels: int | None, density voxel_grid 输入通道数; None 表示由训练入口 lazy 初始化
        - cube_size: int, 每个 P 周围抽取的 cube 边长, 必须为奇数, 推荐默认 11
        - hidden_channels: int, 3D 卷积隐藏通道数, 推荐值 64
        - num_conv: int, stride 下采样后的普通 Conv3d block 数, 推荐值 2
        - out_dim: int, 输出 pseudo_feat 维度, 必须对齐 point_backbone.atom_feature_dim
        - encoder_type: str, encoder 类型, 当前只支持 conv_gap
        - norm: str, 归一化类型, 取值 group/none
        - num_groups: int, GroupNorm group 数, hidden_channels 必须整除该值
        - act: str, 激活函数, 取值 silu/relu/gelu
        - chunk_size: int, 分块抽取 cube 的 P 数量, 推荐默认 512
        - num_downsample: int, stride=2 下采样卷积层数, 推荐默认 2

    前向输入:
        - voxel_grid: torch.Tensor, (B, C, D, H, W), density 输入体
        - anchor_voxel_zyx: torch.Tensor, (sumP, 3), P 来源 voxel 坐标, 轴顺序 z/y/x
        - anchor_batch_index: torch.Tensor, (sumP,), P 所属 BOX 索引

    前向输出:
        - pseudo_feat: torch.Tensor, (sumP, out_dim), P anchor 初始点特征
    """

    def __init__(
        self,
        in_channels: int | None,
        cube_size: int,
        hidden_channels: int,
        num_conv: int,
        out_dim: int,
        encoder_type: str,
        norm: str,
        num_groups: int,
        act: str,
        chunk_size: int,
        num_downsample: int,
    ) -> None:
        super().__init__()
        if in_channels is not None and int(in_channels) <= 0:
            raise ValueError("in_channels 必须 > 0。")
        if int(cube_size) < 1 or int(cube_size) % 2 == 0:
            raise ValueError("cube_size 必须为 >=1 的奇数。")
        if int(hidden_channels) <= 0:
            raise ValueError("hidden_channels 必须 > 0。")
        if int(num_conv) < 1:
            raise ValueError("num_conv 必须 >= 1。")
        if int(num_downsample) < 0:
            raise ValueError("num_downsample 必须 >= 0。")
        if int(out_dim) <= 0:
            raise ValueError("out_dim 必须 > 0。")
        if encoder_type != "conv_gap":
            raise ValueError("encoder_type 当前只支持 conv_gap。")
        if norm not in {"group", "none"}:
            raise ValueError("norm 只支持 group 或 none。")
        if act not in {"silu", "relu", "gelu"}:
            raise ValueError("act 只支持 silu/relu/gelu。")
        if int(chunk_size) <= 0:
            raise ValueError("chunk_size 必须 > 0。")
        if norm == "group" and int(num_groups) <= 0:
            raise ValueError("num_groups 必须 > 0。")
        if norm == "group" and int(hidden_channels) % int(num_groups) != 0:
            raise ValueError("hidden_channels 必须能被 num_groups 整除。")

        self.in_channels = None if in_channels is None else int(in_channels)
        self.cube_size = int(cube_size)
        self.hidden_channels = int(hidden_channels)
        self.num_conv = int(num_conv)
        self.out_dim = int(out_dim)
        self.encoder_type = str(encoder_type)
        self.norm = str(norm)
        self.num_groups = int(num_groups)
        self.act = str(act)
        self.chunk_size = int(chunk_size)
        self.num_downsample = int(num_downsample)
        self.encoder = nn.Sequential(*self._make_encoder_layers(self.in_channels))
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.proj = nn.Linear(self.hidden_channels, self.out_dim)

    def _make_activation(self) -> nn.Module:
        """
        构造配置指定的激活函数. 

        输出:
            - activation: nn.Module, 逐元素激活模块
        """
        if self.act == "silu":
            return nn.SiLU()
        if self.act == "relu":
            return nn.ReLU()
        return nn.GELU()

    def _make_conv_block(self, in_channels: int | None, out_channels: int, stride: int) -> list[nn.Module]:
        """
        构造 Conv3d + 可选 GroupNorm + activation block. 

        输入参数:
            - in_channels: int | None, 输入通道数; None 时使用 LazyConv3d
            - out_channels: int, 输出通道数
            - stride: int, Conv3d stride

        输出:
            - layers: list[nn.Module], 顺序执行的 block 层列表
        """
        if in_channels is None:
            layers: list[nn.Module] = [nn.LazyConv3d(out_channels, kernel_size=3, stride=stride, padding=1)]
        else:
            layers = [nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)]
        if self.norm == "group":
            layers.append(nn.GroupNorm(self.num_groups, out_channels))
        layers.append(self._make_activation())
        return layers

    def _make_encoder_layers(self, in_channels: int | None) -> list[nn.Module]:
        """
        构造 density cube encoder 主干层列表. 

        输入参数:
            - in_channels: int | None, 首层输入通道数; None 时首层使用 LazyConv3d

        输出:
            - layers: list[nn.Module], 顺序执行的 encoder 层
        """
        # list[nn.Module], density cube encoder 主干层
        layers: list[nn.Module] = []
        current_channels = in_channels
        for _ in range(self.num_downsample):
            layers.extend(self._make_conv_block(current_channels, self.hidden_channels, stride=2))
            current_channels = self.hidden_channels
        for _ in range(self.num_conv):
            layers.extend(self._make_conv_block(current_channels, self.hidden_channels, stride=1))
            current_channels = self.hidden_channels
        return layers

    def set_input_channels(self, in_channels: int) -> None:
        """
        设置 density voxel_grid 输入通道数并构造卷积主干. 

        输入参数:
            - in_channels: int, batch["voxel_grid"] 原始通道数, 不含 embed/online scatter 追加通道

        输出:
            - None, 原地重建 density cube encoder 主干
        """
        if int(in_channels) <= 0:
            raise ValueError("in_channels 必须 > 0。")
        if self.in_channels == int(in_channels):
            return
        self.in_channels = int(in_channels)
        device = self.proj.weight.device
        self.encoder = nn.Sequential(*self._make_encoder_layers(self.in_channels)).to(device=device)

    def forward(
        self,
        voxel_grid: torch.Tensor,
        anchor_voxel_zyx: torch.Tensor,
        anchor_batch_index: torch.Tensor,
        cube_size: int | None = None,
    ) -> torch.Tensor:
        """
        抽取 P anchor 周围 density cube 并编码为 pseudo_feat. 

        输入参数:
            - voxel_grid: torch.Tensor, (B, C, D, H, W), density 输入体
            - anchor_voxel_zyx: torch.Tensor, (sumP, 3), P 来源 voxel 坐标, 轴顺序 z/y/x
            - anchor_batch_index: torch.Tensor, (sumP,), P 所属 BOX 索引
            - cube_size: int | None, 本次抽取的 cube 边长; None 回退到实例默认 self.cube_size

        输出:
            - pseudo_feat: torch.Tensor, (sumP, out_dim), P anchor 初始点特征
        """
        if voxel_grid.ndim != 5:
            raise ValueError(f"voxel_grid 期望为 (B,C,D,H,W)，实际 {tuple(voxel_grid.shape)}。")
        if self.in_channels is None:
            self.in_channels = int(voxel_grid.shape[1])
        if int(voxel_grid.shape[1]) != int(self.in_channels):
            raise ValueError(f"voxel_grid 通道数必须为 {self.in_channels}，实际 {int(voxel_grid.shape[1])}。")
        if anchor_voxel_zyx.ndim != 2 or int(anchor_voxel_zyx.shape[1]) != 3:
            raise ValueError("anchor_voxel_zyx 必须为 (sumP,3)。")
        if anchor_batch_index.ndim != 1 or int(anchor_batch_index.shape[0]) != int(anchor_voxel_zyx.shape[0]):
            raise ValueError("anchor_batch_index 必须为 (sumP,) 且与 anchor_voxel_zyx 对齐。")

        # int, 实际使用的 cube 边长; None 回退到实例默认 self.cube_size
        cube_size = self.cube_size if cube_size is None else int(cube_size)
        if cube_size < 1 or cube_size % 2 == 0:
            raise ValueError("cube_size 必须为 >=1 的奇数。")

        # int, P anchor 总数
        num_anchors = int(anchor_voxel_zyx.shape[0])
        if num_anchors == 0:
            return voxel_grid.new_empty((0, self.out_dim))

        # torch.Tensor, (sumP, 3), P 来源 voxel 坐标, 已对齐到 voxel_grid device
        anchor_voxel_zyx = anchor_voxel_zyx.to(device=voxel_grid.device, dtype=torch.long)
        # torch.Tensor, (sumP,), P 所属 BOX 索引, 已对齐到 voxel_grid device
        anchor_batch_index = anchor_batch_index.to(device=voxel_grid.device, dtype=torch.long)
        # list[torch.Tensor], 每个 chunk 的 encoded pseudo_feat
        feature_parts: list[torch.Tensor] = []
        for start in range(0, num_anchors, self.chunk_size):
            end = min(start + self.chunk_size, num_anchors)
            # torch.Tensor, (P_chunk, C, cube_size, cube_size, cube_size), 当前 chunk 的 density cube(_valid_mask 此处无用)
            cube, _valid_mask = gather_voxel_cube(
                grid=voxel_grid,
                center_zyx=anchor_voxel_zyx[start:end],
                batch_index=anchor_batch_index[start:end],
                cube_size=cube_size,
                zero_fill=True,
            )
            # torch.Tensor, (P_chunk, hidden_channels, d, h, w), 编码后的 cube 特征
            encoded = self.encoder(cube)
            # torch.Tensor, (P_chunk, hidden_channels), GAP 后 cube 特征
            pooled = self.pool(encoded).flatten(1)
            # torch.Tensor, (P_chunk, out_dim), 当前 chunk 的 pseudo_feat
            feature_parts.append(self.proj(pooled))
        return torch.cat(feature_parts, dim=0)
