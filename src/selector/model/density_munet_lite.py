"""为 Selector、Stage2 或 Stage3 独立训练的轻量三维密度上下文网络。

`DensityMUNetLite` 只消费当前任务现场构造的单通道 experimental density。网络按
`80→40→20→10` 三次下采样，在最低分辨率执行一次或多次 Transformer，再用跳跃
连接恢复完整分辨率。输出只在真实稀疏 V voxel 位置被读取，不作为 dense 产物落盘。
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as F


def _group_count(channels: int) -> int:
    """
    选择能整除通道数且不超过 8 的 GroupNorm 分组数。

    输入参数:
        - channels: int, 当前 feature 通道数

    输出:
        - groups: int, GroupNorm group 数
    """
    for groups in (8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class ResidualConvBlock3d(nn.Module):
    """
    在一个分辨率层内执行两次 3×3 Conv3d 的 residual convolution block。

    输入参数:
        - in_channels: int, 输入通道数
        - out_channels: int, 输出通道数

    前向输入:
        - x: torch.Tensor, (B,in_channels,D,H,W)，当前层特征

    前向输出:
        - y: torch.Tensor, (B,out_channels,D,H,W)，残差卷积结果
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        """
        初始化两层三维卷积主分支和通道匹配残差分支。

        输入参数:
            - in_channels: int, 输入特征通道数
            - out_channels: int, 输出特征通道数；与输入不同时用 1×1×1 卷积
              投影残差
        """
        super().__init__()
        self.main = nn.Sequential(
            nn.GroupNorm(_group_count(in_channels), in_channels),
            nn.SiLU(),
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(_group_count(out_channels), out_channels),
            nn.SiLU(),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
        )
        self.skip = nn.Identity() if in_channels == out_channels else nn.Conv3d(in_channels, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        执行 residual convolution。

        输入参数:
            - x: torch.Tensor, (B,in_channels,D,H,W)，当前层特征

        输出:
            - y: torch.Tensor, (B,out_channels,D,H,W)，残差卷积结果
        """
        return self.skip(x) + self.main(x)


class DensityMUNetLite(nn.Module):
    """
    从 exp_clipnorm_nopost 生成任务专属 full-resolution 32D 密度上下文。

    输入参数:
        - input_shape_zyx: Sequence[int], 离散 voxel 网格尺寸 ZYX，正式为 (80,80,80)；其它配置也必须三轴能被 8 整除
        - channels: Sequence[int], 四层通道，正式为 (32,64,64,128)
        - bottleneck_heads: int, 10³ bottleneck Transformer attention head 数；正式为 4
        - bottleneck_layers: int, bottleneck Transformer 层数；正式为 1
        - bottleneck_ffn_dim: int, bottleneck Transformer FFN hidden dim
        - dropout: float, Transformer dropout

    前向输入:
        - density_input: torch.Tensor, (B,1,D,H,W)，BOX-local 离散 ZYX voxel 网格上的 exp_clipnorm_nopost

    前向输出:
        - density_feature: torch.Tensor, (B,32,D,H,W)，BOX-local 离散 ZYX voxel 网格上的密度上下文，只供实际 V voxel gather，不落盘 dense grid
    """

    def __init__(
        self,
        input_shape_zyx: Sequence[int],
        channels: Sequence[int],
        bottleneck_heads: int,
        bottleneck_layers: int,
        bottleneck_ffn_dim: int,
        dropout: float,
    ) -> None:
        """
        构造三层下采样、最低分辨率 Transformer 和对称卷积解码器。

        输入参数:
            - input_shape_zyx: Sequence[int], 输入密度的离散 ZYX 网格尺寸；三轴
              必须能被 8 整除
            - channels: Sequence[int], 从完整分辨率到最低分辨率的四个通道数
            - bottleneck_heads: int, 最低分辨率 Transformer 的注意力头数
            - bottleneck_layers: int, Transformer encoder 层数
            - bottleneck_ffn_dim: int, Transformer 前馈网络通道数
            - dropout: float, Transformer 内部 dropout 概率

        结构:
            - 编码分辨率依次为 `(D,H,W)`、`/2`、`/4`、`/8`。
            - 最低分辨率体素展平为序列，并加入由归一化 XYZ 坐标线性投影得到的
              位置编码。
            - 解码器逐级拼接同分辨率编码特征，最终输出 `channels[0]` 个通道。
        """
        super().__init__()
        self.input_shape_zyx = tuple(int(value) for value in input_shape_zyx)
        channel_values = tuple(int(value) for value in channels)
        if len(self.input_shape_zyx) != 3 or any(value % 8 != 0 for value in self.input_shape_zyx):
            raise ValueError("DensityMUNetLite 输入三轴必须能被 8 整除。")
        if len(channel_values) != 4:
            raise ValueError("channels 必须恰含四个分辨率层。")
        if channel_values[-1] % int(bottleneck_heads) != 0:
            raise ValueError("bottleneck 通道数必须能被 attention head 数整除。")
        self.output_channels = channel_values[0]

        # c0/c1/c2/c3 分别对应完整、1/2、1/4 和 1/8 线性分辨率。
        c0, c1, c2, c3 = channel_values
        self.input_projection = nn.Conv3d(1, c0, kernel_size=3, padding=1)
        self.encoder0 = ResidualConvBlock3d(c0, c0)
        self.down1 = nn.Conv3d(c0, c1, kernel_size=2, stride=2)
        self.encoder1 = ResidualConvBlock3d(c1, c1)
        self.down2 = nn.Conv3d(c1, c2, kernel_size=2, stride=2)
        self.encoder2 = ResidualConvBlock3d(c2, c2)
        self.down3 = nn.Conv3d(c2, c3, kernel_size=2, stride=2)
        self.encoder3 = ResidualConvBlock3d(c3, c3)

        self.position_projection = nn.Linear(3, c3)
        transformer_layer = nn.TransformerEncoderLayer(
            d_model=c3,
            nhead=int(bottleneck_heads),
            dim_feedforward=int(bottleneck_ffn_dim),
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.bottleneck_transformer = nn.TransformerEncoder(
            transformer_layer,
            num_layers=int(bottleneck_layers),
            enable_nested_tensor=False,
        )

        self.up2 = nn.ConvTranspose3d(c3, c2, kernel_size=2, stride=2)
        self.decoder2 = ResidualConvBlock3d(c2 + c2, c2)
        self.up1 = nn.ConvTranspose3d(c2, c1, kernel_size=2, stride=2)
        self.decoder1 = ResidualConvBlock3d(c1 + c1, c1)
        self.up0 = nn.ConvTranspose3d(c1, c0, kernel_size=2, stride=2)
        self.decoder0 = ResidualConvBlock3d(c0 + c0, c0)

    def _run_bottleneck_transformer(self, feature: torch.Tensor) -> torch.Tensor:
        """
        仅在最低分辨率把 3D feature 展平成 tokens 执行 Transformer。

        输入参数:
            - feature: torch.Tensor, (B,C,D_b,H_b,W_b)，BOX-local 离散 ZYX voxel 网格上的最低分辨率特征

        输出:
            - transformed: torch.Tensor, 与 feature 同 shape，含显式归一化 XYZ 位置编码
        """
        batch_size, channels, depth, height, width = feature.shape
        z = torch.linspace(-1.0, 1.0, depth, device=feature.device, dtype=feature.dtype)
        y = torch.linspace(-1.0, 1.0, height, device=feature.device, dtype=feature.dtype)
        x = torch.linspace(-1.0, 1.0, width, device=feature.device, dtype=feature.dtype)
        # `(D_b, H_b, W_b, 3)` 的归一化 ZYX 网格，换序后作为连续 XYZ 位置编码。
        grid_zyx = torch.stack(torch.meshgrid(z, y, x, indexing="ij"), dim=-1)
        position_xyz = grid_zyx[..., [2, 1, 0]].reshape(1, -1, 3)
        # `(B, D_b*H_b*W_b, C)`，空间 C-order 展平与恢复使用相同次序。
        tokens = feature.flatten(2).transpose(1, 2)
        tokens = tokens + self.position_projection(position_xyz).expand(batch_size, -1, -1)
        transformed = self.bottleneck_transformer(tokens)
        return transformed.transpose(1, 2).reshape(batch_size, channels, depth, height, width)

    def forward(self, density_input: torch.Tensor) -> torch.Tensor:
        """
        运行 80→40→20→10 编码、10³ Transformer 与卷积解码。

        输入参数:
            - density_input: torch.Tensor, (B,1,D,H,W)，BOX-local 离散 ZYX voxel 网格上的 exp_clipnorm_nopost

        输出:
            - density_feature: torch.Tensor, (B,32,D,H,W)，BOX-local 离散 ZYX full-resolution 密度上下文
        """
        if tuple(density_input.shape[2:]) != self.input_shape_zyx or density_input.shape[1] != 1:
            raise ValueError(
                f"density_input 必须为 (B,1,{self.input_shape_zyx}), 实际为 {tuple(density_input.shape)}"
            )
        # 编码张量分别位于 1、1/2、1/4 和 1/8 线性分辨率。
        enc0 = self.encoder0(self.input_projection(density_input))
        enc1 = self.encoder1(self.down1(enc0))
        enc2 = self.encoder2(self.down2(enc1))
        enc3 = self.encoder3(self.down3(enc2))
        bottleneck = self._run_bottleneck_transformer(enc3)
        # 每一级转置卷积结果与同分辨率编码特征按通道拼接后再做残差卷积。
        dec2 = self.decoder2(torch.cat([self.up2(bottleneck), enc2], dim=1))
        dec1 = self.decoder1(torch.cat([self.up1(dec2), enc1], dim=1))
        dec0 = self.decoder0(torch.cat([self.up0(dec1), enc0], dim=1))
        return F.silu(dec0)
