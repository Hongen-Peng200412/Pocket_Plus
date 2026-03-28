# -*- coding: utf-8 -*-
"""
Stage1EmbedHead 及相关工具函数的单元测试。
"""
from __future__ import annotations

import torch

from src.model.stage1_embed_head import (
    trim_buffer_atoms,
    scatter_to_voxel_grid,
)


# ==============================================================
# trim_buffer_atoms 测试
# ==============================================================

def test_trim_buffer_atoms_all_core() -> None:
    """全部原子都在 core box 内时，裁剪不改变序列。"""
    n, c = 5, 8
    feat = torch.randn(n, c)
    coord = torch.randn(n, 3)
    batch = torch.tensor([0, 0, 0, 1, 1], dtype=torch.long)
    offset = torch.tensor([3, 5], dtype=torch.long)
    is_core = torch.ones(n, dtype=torch.bool)
    local_voxel = torch.rand(n, 3) * 4
    box_shape = torch.tensor([[4, 4, 4], [4, 4, 4]], dtype=torch.long)
    voxel_size = torch.tensor([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])

    result = trim_buffer_atoms(
        point_feat=feat,
        point_coord=coord,
        point_batch=batch,
        point_offset=offset,
        atom_is_in_core_box=is_core,
        atom_coord_local_voxel=local_voxel,
        box_shape_zyx=box_shape,
        voxel_size_world=voxel_size,
        allowed_buffer_radius_world=0.0,
    )
    assert result["keep_mask"].all()
    assert result["point_feat"].shape == (n, c)


def test_trim_buffer_atoms_removes_far_atoms() -> None:
    """buffer 原子距 core box 超过 allowed_radius 时被裁剪。"""
    # 3个原子: 2 core + 1 buffer(远离 box)
    feat = torch.randn(3, 4)
    coord = torch.tensor([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [100.0, 100.0, 100.0]])
    batch = torch.tensor([0, 0, 0], dtype=torch.long)
    offset = torch.tensor([3], dtype=torch.long)
    is_core = torch.tensor([True, True, False])
    # core box 尺寸: 4x4x4, voxel_size=1.0 => 世界范围 [0,4) x [0,4) x [0,4)
    local_voxel = torch.tensor([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [100.0, 100.0, 100.0]])
    box_shape = torch.tensor([[4, 4, 4]], dtype=torch.long)
    voxel_size = torch.tensor([[1.0, 1.0, 1.0]])

    result = trim_buffer_atoms(
        point_feat=feat,
        point_coord=coord,
        point_batch=batch,
        point_offset=offset,
        atom_is_in_core_box=is_core,
        atom_coord_local_voxel=local_voxel,
        box_shape_zyx=box_shape,
        voxel_size_world=voxel_size,
        allowed_buffer_radius_world=5.0,
    )
    # 应该只保留 2 个 core 原子, 第3个太远被裁掉
    assert result["keep_mask"].sum().item() == 2
    assert result["point_feat"].shape[0] == 2
    assert result["point_offset"].tolist() == [2]


def test_trim_buffer_atoms_inf_radius() -> None:
    """allowed_radius=inf 时不裁剪任何原子。"""
    n = 4
    feat = torch.randn(n, 3)
    result = trim_buffer_atoms(
        point_feat=feat,
        point_coord=torch.randn(n, 3),
        point_batch=torch.zeros(n, dtype=torch.long),
        point_offset=torch.tensor([n], dtype=torch.long),
        atom_is_in_core_box=torch.tensor([True, False, True, False]),
        atom_coord_local_voxel=torch.randn(n, 3),
        box_shape_zyx=torch.tensor([[4, 4, 4]], dtype=torch.long),
        voxel_size_world=torch.tensor([[1.0, 1.0, 1.0]]),
        allowed_buffer_radius_world=float("inf"),
    )
    assert result["keep_mask"].all()


def test_trim_buffer_atoms_empty() -> None:
    """空序列不报错。"""
    result = trim_buffer_atoms(
        point_feat=torch.empty(0, 4),
        point_coord=torch.empty(0, 3),
        point_batch=torch.empty(0, dtype=torch.long),
        point_offset=torch.tensor([0], dtype=torch.long),
        atom_is_in_core_box=torch.empty(0, dtype=torch.bool),
        atom_coord_local_voxel=torch.empty(0, 3),
        box_shape_zyx=torch.tensor([[4, 4, 4]], dtype=torch.long),
        voxel_size_world=torch.tensor([[1.0, 1.0, 1.0]]),
        allowed_buffer_radius_world=3.0,
    )
    assert result["point_feat"].shape == (0, 4)


# ==============================================================
# scatter_to_voxel_grid 测试
# ==============================================================

def test_scatter_to_voxel_grid_basic() -> None:
    """基本 scatter 验证: 单个原子散射到一个体素。"""
    # 1个原子在 batch=0, local_voxel=(0.5, 0.5, 0.5)—落在体素 (0,0,0)
    feat = torch.tensor([[1.0, 2.0]])
    local_voxel = torch.tensor([[0.5, 0.5, 0.5]])
    batch = torch.tensor([0], dtype=torch.long)
    box_shape = torch.tensor([[2, 2, 2]], dtype=torch.long)  # D=2, H=2, W=2

    grid = scatter_to_voxel_grid(
        point_feat=feat,
        atom_coord_local_voxel=local_voxel,
        point_batch=batch,
        box_shape_zyx=box_shape,
        batch_size=1,
        reduce="mean",
    )
    assert grid.shape == (1, 2, 2, 2, 2)
    # 体素 (d=0, h=0, w=0) 应该有值 [1.0, 2.0]
    assert grid[0, 0, 0, 0, 0].item() == 1.0
    assert grid[0, 1, 0, 0, 0].item() == 2.0


def test_scatter_to_voxel_grid_mean() -> None:
    """多个原子落在同一体素时，mean 聚合正确。"""
    feat = torch.tensor([[2.0], [4.0]])
    local_voxel = torch.tensor([[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]])
    batch = torch.tensor([0, 0], dtype=torch.long)
    box_shape = torch.tensor([[2, 2, 2]], dtype=torch.long)

    grid = scatter_to_voxel_grid(
        point_feat=feat,
        atom_coord_local_voxel=local_voxel,
        point_batch=batch,
        box_shape_zyx=box_shape,
        batch_size=1,
        reduce="mean",
    )
    # 两个原子都在 (0,0,0), mean = (2+4)/2 = 3.0
    assert abs(grid[0, 0, 0, 0, 0].item() - 3.0) < 1e-5


def test_scatter_to_voxel_grid_empty() -> None:
    """空原子时返回全零网格。"""
    grid = scatter_to_voxel_grid(
        point_feat=torch.empty(0, 3),
        atom_coord_local_voxel=torch.empty(0, 3),
        point_batch=torch.empty(0, dtype=torch.long),
        box_shape_zyx=torch.tensor([[4, 4, 4]], dtype=torch.long),
        batch_size=1,
        reduce="mean",
    )
    assert grid.shape == (1, 3, 4, 4, 4)
    assert grid.abs().sum().item() == 0.0


# ==============================================================
# Stage1EmbedHead 端到端形状测试 (使用 dummy Point)
# ==============================================================

def _make_dummy_batch(batch_size: int, atoms_per_box: int, channels: int, box_d: int, box_h: int, box_w: int):
    """构造 embed head forward 所需的 dummy 输入。"""
    total_n = batch_size * atoms_per_box
    atom_feat = torch.randn(total_n, channels)
    atom_coord = torch.randn(total_n, 3) * 2.0
    atom_batch_idx = torch.arange(batch_size).repeat_interleave(atoms_per_box).long()
    atom_offsets = torch.arange(1, batch_size + 1).long() * atoms_per_box

    # local_voxel 坐标在 box 内
    local_voxel = torch.rand(total_n, 3)
    local_voxel[:, 0] *= box_w
    local_voxel[:, 1] *= box_h
    local_voxel[:, 2] *= box_d

    box_shape = torch.tensor([[box_d, box_h, box_w]] * batch_size, dtype=torch.long)
    voxel_size = torch.tensor([[1.0, 1.0, 1.0]] * batch_size, dtype=torch.float32)
    is_core = torch.ones(total_n, dtype=torch.bool)

    return {
        "atom_feat": atom_feat,
        "atom_coord_centered_world": atom_coord,
        "atom_batch_index": atom_batch_idx,
        "atom_offsets": atom_offsets,
        "atom_coord_local_voxel": local_voxel,
        "box_shape_zyx": box_shape,
        "voxel_size_world": voxel_size,
        "atom_is_in_core_box": is_core,
    }


def test_embed_head_forward_shape() -> None:
    """验证 embed head 前向输出形状正确(需要 PTV3 依赖)。"""
    try:
        from src.model.stage1_embed_head import Stage1EmbedHead
    except ImportError:
        import pytest
        pytest.skip("PTV3 依赖不可用")

    B, N_per_box, F, D, H, W = 2, 10, 49, 4, 4, 4
    C_voxel, C_point = 8, 16

    embed = Stage1EmbedHead(
        atom_feature_dim=F,
        embed_hidden_dim=32,
        embed_voxel_out_channels=C_voxel,
        embed_point_out_channels=C_point,
        num_trunk_blocks=1,
        num_voxel_blocks=1,
        num_point_blocks=1,
        trunk_buffer_radii=[float("inf")],
        voxel_buffer_radii=[float("inf")],
        point_buffer_radii=[float("inf")],
        num_heads=2,
        patch_size=32,
        serialization_orders=["z"],
        shuffle_orders=False,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        enable_rpe=False,
        enable_flash=False,
        upcast_attention=False,
        upcast_softmax=False,
        scatter_reduce="mean",
        ffn_type="mlp",
        mlp_ratio=2,
        act_layer_name="gelu",
        point_grid_size=0.25,
        cpe_impl="none",
        cpe_kernel_size=5,
        cpe_receptive_field=2.0,
        pointconv_block_max_neighbors=16,
        drop_path=0.0,
        pre_norm=True,
    )
    embed.eval()

    dummy = _make_dummy_batch(B, N_per_box, F, D, H, W)
    with torch.no_grad():
        out = embed(**dummy)

    assert out["voxel_pdb_embed_grid"].shape == (B, C_voxel, D, H, W)
    assert out["embed_point_feat"] is not None
    assert out["embed_point_feat"].shape == (B * N_per_box, C_point)
    # 无裁剪时, global_keep_mask 全 True
    assert out["global_keep_mask"].all()
    assert out["atom_feat"].shape[0] == B * N_per_box


def test_embed_head_no_point_output() -> None:
    """embed_point_out_channels=0 时，embed_point_feat 应为 None。"""
    try:
        from src.model.stage1_embed_head import Stage1EmbedHead
    except ImportError:
        import pytest
        pytest.skip("PTV3 依赖不可用")

    B, N_per_box, F, D, H, W = 1, 5, 49, 4, 4, 4

    embed = Stage1EmbedHead(
        atom_feature_dim=F,
        embed_hidden_dim=32,
        embed_voxel_out_channels=8,
        embed_point_out_channels=0,
        num_trunk_blocks=1,
        num_voxel_blocks=1,
        num_point_blocks=0,
        trunk_buffer_radii=[float("inf")],
        voxel_buffer_radii=[float("inf")],
        point_buffer_radii=[],
        num_heads=2,
        patch_size=32,
        serialization_orders=["z"],
        shuffle_orders=False,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        enable_rpe=False,
        enable_flash=False,
        upcast_attention=False,
        upcast_softmax=False,
        scatter_reduce="mean",
        ffn_type="none",
        mlp_ratio=4,
        act_layer_name="gelu",
        point_grid_size=0.25,
        cpe_impl="none",
        cpe_kernel_size=5,
        cpe_receptive_field=2.0,
        pointconv_block_max_neighbors=16,
        drop_path=0.0,
        pre_norm=True,
    )
    embed.eval()

    dummy = _make_dummy_batch(B, N_per_box, F, D, H, W)
    with torch.no_grad():
        out = embed(**dummy)

    assert out["voxel_pdb_embed_grid"].shape == (B, 8, D, H, W)
    assert out["embed_point_feat"] is None
    assert out["global_keep_mask"].all()


def test_embed_head_empty_atoms() -> None:
    """sumN=0 时不报错。"""
    try:
        from src.model.stage1_embed_head import Stage1EmbedHead
    except ImportError:
        import pytest
        pytest.skip("PTV3 依赖不可用")

    embed = Stage1EmbedHead(
        atom_feature_dim=49,
        embed_hidden_dim=32,
        embed_voxel_out_channels=8,
        embed_point_out_channels=16,
        num_trunk_blocks=1,
        num_voxel_blocks=1,
        num_point_blocks=1,
        trunk_buffer_radii=[float("inf")],
        voxel_buffer_radii=[float("inf")],
        point_buffer_radii=[float("inf")],
        num_heads=2,
        patch_size=32,
        serialization_orders=["z"],
        shuffle_orders=False,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        enable_rpe=False,
        enable_flash=False,
        upcast_attention=False,
        upcast_softmax=False,
        scatter_reduce="mean",
        ffn_type="mlp",
        mlp_ratio=2,
        act_layer_name="gelu",
        point_grid_size=0.25,
        cpe_impl="none",
        cpe_kernel_size=5,
        cpe_receptive_field=2.0,
        pointconv_block_max_neighbors=16,
        drop_path=0.0,
        pre_norm=True,
    )
    embed.eval()

    dummy = {
        "atom_feat": torch.empty(0, 49),
        "atom_coord_centered_world": torch.empty(0, 3),
        "atom_batch_index": torch.empty(0, dtype=torch.long),
        "atom_offsets": torch.tensor([0], dtype=torch.long),
        "atom_coord_local_voxel": torch.empty(0, 3),
        "box_shape_zyx": torch.tensor([[4, 4, 4]], dtype=torch.long),
        "voxel_size_world": torch.tensor([[1.0, 1.0, 1.0]]),
        "atom_is_in_core_box": torch.empty(0, dtype=torch.bool),
    }
    with torch.no_grad():
        out = embed(**dummy)

    assert out["voxel_pdb_embed_grid"].shape == (1, 8, 4, 4, 4)
    assert out["embed_point_feat"] is not None
    assert out["embed_point_feat"].shape == (0, 16)
    assert out["global_keep_mask"].shape == (0,)

