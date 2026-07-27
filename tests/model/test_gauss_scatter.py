"""
gauss_scatter_to_voxel_grid(3x3x3 各向同性高斯 scatter)的数值与边界单元测试. 

被测对象: src/model/stage1_embed_head.py::gauss_scatter_to_voxel_grid(纯 torch, 不依赖 PTV3/torch_cluster). 
设计契约见 CLAUDE/plans/implement/最后重构代码.md §1.9 与该函数 docstring. 
"""
from __future__ import annotations

import torch

from src.model.stage1_embed_head import gauss_scatter_to_voxel_grid


def test_mass_conservation_feature_channels() -> None:
    """
    逐原子高斯权重归一化(每原子总贡献=1), 故所有体素上的特征通道求和应等于每原子特征之和(质量守恒). 
    """
    torch.manual_seed(0)
    n_atom, channels = 40, 6
    feat = torch.randn(n_atom, channels)
    # torch.Tensor, (n_atom, 3), (x,y,z) 连续体素坐标, 全部落在 10^3 网格内部(留 0.5 边距避免越界)
    coord = torch.rand(n_atom, 3) * 8.0 + 1.0
    point_batch = torch.zeros(n_atom, dtype=torch.long)
    box_shape_zyx = torch.tensor([[10, 10, 10]])
    grid = gauss_scatter_to_voxel_grid(
        feat, coord, point_batch, box_shape_zyx, batch_size=1, sigma_voxel=0.7,
        add_occupancy_channels=False, add_centroid_channels=False,
    )
    assert grid.shape == (1, channels, 10, 10, 10), grid.shape
    # torch.Tensor, (channels,), 所有体素求和
    recon = grid[0].sum(dim=(1, 2, 3))
    assert torch.allclose(recon, feat.sum(dim=0), atol=1e-3), (recon[:3], feat.sum(0)[:3])


def test_output_channel_layout_with_occupancy_centroid() -> None:
    """
    occupancy(2) + centroid(3) 开启时输出通道数 = C + 5, 顺序为 [特征]++[occupancy]++[centroid]. 
    """
    n_atom, channels = 20, 4
    feat = torch.randn(n_atom, channels)
    coord = torch.rand(n_atom, 3) * 6.0 + 1.0
    point_batch = torch.zeros(n_atom, dtype=torch.long)
    box_shape_zyx = torch.tensor([[8, 8, 8]])
    grid = gauss_scatter_to_voxel_grid(
        feat, coord, point_batch, box_shape_zyx, batch_size=1, sigma_voxel=0.7,
        add_occupancy_channels=True, add_centroid_channels=True,
    )
    assert grid.shape == (1, channels + 5, 8, 8, 8), grid.shape


def test_empty_input_returns_zeros_with_right_shape() -> None:
    """
    空输入(N=0)返回正确 shape 的全零张量, 不报错. 
    """
    channels = 5
    box_shape_zyx = torch.tensor([[10, 10, 10]])
    grid = gauss_scatter_to_voxel_grid(
        torch.zeros(0, channels), torch.zeros(0, 3), torch.zeros(0, dtype=torch.long),
        box_shape_zyx, batch_size=1, sigma_voxel=0.7,
        add_occupancy_channels=True, add_centroid_channels=True,
    )
    assert grid.shape == (1, channels + 5, 10, 10, 10), grid.shape
    assert torch.count_nonzero(grid) == 0


def test_isotropy_single_atom_at_voxel_corner() -> None:
    """
    单原子放在某体素角点(整数坐标)时, 其 6 个轴向最近邻体素中心到原子的距离相等,
    故各向同性高斯赋予它们相等权重(各向同性核消除三线性的轴对齐偏置). 
    """
    channels = 1
    feat = torch.ones(1, channels)
    # 原子在 (5,5,5) 角点(整数), 邻域体素中心在 i+0.5; 沿 +x/+y/+z 与 -x/-y/-z 的中心距离对称
    coord = torch.tensor([[5.0, 5.0, 5.0]])
    point_batch = torch.zeros(1, dtype=torch.long)
    box_shape_zyx = torch.tensor([[10, 10, 10]])
    grid = gauss_scatter_to_voxel_grid(
        feat, coord, point_batch, box_shape_zyx, batch_size=1, sigma_voxel=0.7,
        add_occupancy_channels=False, add_centroid_channels=False,
    )[0, 0]  # (D,H,W) = (z,y,x)
    # 体素索引 i 对应中心 i+0.5. 原子在 5.0: 体素 4(中心 4.5, 距 0.5) 与体素 5(中心 5.5, 距 0.5) 沿每轴对称. 
    wx_lo = float(grid[5, 5, 4]); wx_hi = float(grid[5, 5, 5])
    wy_lo = float(grid[5, 4, 5]); wy_hi = float(grid[5, 5, 5])
    wz_lo = float(grid[4, 5, 5]); wz_hi = float(grid[5, 5, 5])
    assert abs(wx_lo - wx_hi) < 1e-5 and abs(wy_lo - wy_hi) < 1e-5 and abs(wz_lo - wz_hi) < 1e-5
    # 三个轴向的对称邻居权重也彼此相等(各向同性)
    assert abs(wx_lo - wy_lo) < 1e-5 and abs(wy_lo - wz_lo) < 1e-5


def test_out_of_bounds_atom_does_not_crash_and_conserves_in_bounds() -> None:
    """
    坐标落在网格外的原子: 越界邻域权重置 0, 逐原子归一化后该原子若全部越界则总贡献为 0; 不报错. 
    """
    channels = 3
    # 原子 0 在界内, 原子 1 远在界外(负坐标)
    feat = torch.randn(2, channels)
    coord = torch.tensor([[5.0, 5.0, 5.0], [-50.0, -50.0, -50.0]])
    point_batch = torch.zeros(2, dtype=torch.long)
    box_shape_zyx = torch.tensor([[10, 10, 10]])
    grid = gauss_scatter_to_voxel_grid(
        feat, coord, point_batch, box_shape_zyx, batch_size=1, sigma_voxel=0.7,
        add_occupancy_channels=False, add_centroid_channels=False,
    )
    # 仅界内原子 0 贡献质量, 总和应等于 feat[0]
    recon = grid[0].sum(dim=(1, 2, 3))
    assert torch.allclose(recon, feat[0], atol=1e-3), (recon, feat[0])
