from __future__ import annotations

import pytest
import torch
from torch import nn

from src.model.sparse_refine.density_cube import DensityCubeEncoder


def _make_encoder(**kwargs: object) -> DensityCubeEncoder:
    """
    构造测试用 DensityCubeEncoder. 

    输入参数:
        - kwargs: object, 覆盖默认构造参数

    输出:
        - encoder: DensityCubeEncoder, 测试用 encoder
    """
    params = {
        "in_channels": 2,
        "cube_size": 11,
        "hidden_channels": 8,
        "num_downsample": 2,
        "num_conv": 1,
        "out_dim": 4,
        "encoder_type": "conv_gap",
        "norm": "group",
        "num_groups": 4,
        "act": "silu",
        "chunk_size": 2,
    }
    params.update(kwargs)
    return DensityCubeEncoder(**params)


def test_density_cube_encoder_output_shape() -> None:
    """
    验证 density cube encoder 输出形状. 
    """
    encoder = _make_encoder()
    voxel_grid = torch.randn(1, 2, 6, 6, 6)
    anchor_voxel_zyx = torch.tensor([[1, 1, 1], [5, 5, 5], [0, 0, 0]], dtype=torch.long)
    anchor_batch_index = torch.tensor([0, 0, 0], dtype=torch.long)

    output = encoder(voxel_grid, anchor_voxel_zyx, anchor_batch_index)

    assert tuple(output.shape) == (3, 4)


def test_density_cube_encoder_empty_anchor_returns_empty_feature() -> None:
    """
    验证空 P anchor 返回空 pseudo feature. 
    """
    encoder = _make_encoder()
    voxel_grid = torch.randn(1, 2, 6, 6, 6)

    output = encoder(
        voxel_grid,
        torch.empty((0, 3), dtype=torch.long),
        torch.empty((0,), dtype=torch.long),
    )

    assert tuple(output.shape) == (0, 4)


def test_density_cube_encoder_rejects_even_cube_size() -> None:
    """
    验证偶数 cube_size 构造时报错. 
    """
    with pytest.raises(ValueError, match="cube_size"):
        _make_encoder(cube_size=10)


def test_density_cube_encoder_rejects_channel_mismatch() -> None:
    """
    验证 voxel_grid 通道数不匹配时报错. 
    """
    encoder = _make_encoder(in_channels=3)

    with pytest.raises(ValueError, match="通道"):
        encoder(torch.randn(1, 2, 6, 6, 6), torch.tensor([[0, 0, 0]]), torch.tensor([0]))


def test_density_cube_encoder_lazy_initializes_from_voxel_grid_channels() -> None:
    """
    验证 in_channels=None 时首轮 forward 按 voxel_grid 原始通道 lazy 初始化. 
    """
    encoder = _make_encoder(in_channels=None)
    voxel_grid = torch.randn(1, 3, 6, 6, 6)
    anchor_voxel_zyx = torch.tensor([[1, 1, 1]], dtype=torch.long)
    anchor_batch_index = torch.tensor([0], dtype=torch.long)

    output = encoder(voxel_grid, anchor_voxel_zyx, anchor_batch_index)

    assert encoder.in_channels == 3
    assert tuple(output.shape) == (1, 4)


def test_density_cube_encoder_uses_groupnorm_not_batchnorm_or_layernorm() -> None:
    """
    验证 density cube encoder 不使用 BatchNorm3d 或 LayerNorm. 
    """
    encoder = _make_encoder()

    assert not any(isinstance(module, nn.BatchNorm3d) for module in encoder.modules())
    assert not any(isinstance(module, nn.LayerNorm) for module in encoder.modules())
    assert any(isinstance(module, nn.GroupNorm) for module in encoder.modules())


def test_density_cube_encoder_rejects_bad_group_count() -> None:
    """
    验证 hidden_channels 不能被 num_groups 整除时 fail-fast. 
    """
    with pytest.raises(ValueError, match="num_groups"):
        _make_encoder(hidden_channels=10, num_groups=4)


def test_density_cube_encoder_rejects_non_conv_gap_encoder_type() -> None:
    """
    验证不支持非 conv_gap encoder_type. 
    """
    with pytest.raises(ValueError, match="conv_gap"):
        _make_encoder(encoder_type="residual")


def test_density_cube_encoder_forward_cube_size_override() -> None:
    """
    验证 forward 传入 cube_size 覆盖实例默认, 共享 encoder 可吃不同 cube 尺寸. 
    """
    encoder = _make_encoder(cube_size=11)
    voxel_grid = torch.randn(1, 2, 8, 8, 8)
    anchor_voxel_zyx = torch.tensor([[4, 4, 4]], dtype=torch.long)
    anchor_batch_index = torch.tensor([0], dtype=torch.long)

    out_default = encoder(voxel_grid, anchor_voxel_zyx, anchor_batch_index)
    out_override = encoder(voxel_grid, anchor_voxel_zyx, anchor_batch_index, cube_size=7)

    assert tuple(out_default.shape) == (1, 4)
    assert tuple(out_override.shape) == (1, 4)


def test_density_cube_encoder_forward_rejects_even_cube_size() -> None:
    """
    验证 forward 传入偶数 cube_size 时 fail-fast. 
    """
    encoder = _make_encoder()
    with pytest.raises(ValueError, match="cube_size"):
        encoder(
            torch.randn(1, 2, 8, 8, 8),
            torch.tensor([[4, 4, 4]], dtype=torch.long),
            torch.tensor([0], dtype=torch.long),
            cube_size=4,
        )


def test_density_cube_encoder_default_has_downsample_before_plain_conv() -> None:
    """
    验证默认结构先执行两层 stride=2 Conv3d, 再执行普通 Conv3d. 
    """
    encoder = _make_encoder(num_downsample=2, num_conv=1)
    convs = [module for module in encoder.encoder if isinstance(module, nn.Conv3d)]

    assert [conv.stride for conv in convs] == [(2, 2, 2), (2, 2, 2), (1, 1, 1)]
