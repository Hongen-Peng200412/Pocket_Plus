from __future__ import annotations

import pytest
import torch
from torch import nn

from src.model.utils import (
    CubeWeightingParams,
    FeatureCombine,
    FiLMCombine,
    build_zero_init_residual_mlp,
    gather_voxel_cube,
)


def _naive_cube(
    grid: torch.Tensor,
    center_zyx: torch.Tensor,
    batch_index: torch.Tensor,
    cube_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    朴素三重循环参照实现, 用于校验 gather_voxel_cube 与 zero-pad 抽取等价。

    输入参数:
        - grid: torch.Tensor, (B, C, D, H, W)
        - center_zyx: torch.Tensor, (N, 3), long
        - batch_index: torch.Tensor, (N,), long
        - cube_size: int

    输出:
        - cube: torch.Tensor, (N, C, k, k, k), 越界置 0
        - valid: torch.Tensor, (N, k, k, k), bool
    """
    _, channels, dim_z, dim_y, dim_x = grid.shape
    radius = cube_size // 2
    num = int(center_zyx.shape[0])
    cube = torch.zeros(num, channels, cube_size, cube_size, cube_size)
    valid = torch.zeros(num, cube_size, cube_size, cube_size, dtype=torch.bool)
    for n in range(num):
        b = int(batch_index[n])
        cz, cy, cx = (int(v) for v in center_zyx[n])
        for iz in range(cube_size):
            for iy in range(cube_size):
                for ix in range(cube_size):
                    z, y, x = cz + iz - radius, cy + iy - radius, cx + ix - radius
                    if 0 <= z < dim_z and 0 <= y < dim_y and 0 <= x < dim_x:
                        cube[n, :, iz, iy, ix] = grid[b, :, z, y, x]
                        valid[n, iz, iy, ix] = True
    return cube, valid


def test_gather_voxel_cube_matches_naive_zero_pad() -> None:
    """
    验证 gather_voxel_cube(zero_fill=True) 与朴素 zero-pad 抽取逐元素一致, 含边界越界。
    """
    torch.manual_seed(0)
    grid = torch.randn(2, 3, 5, 6, 7)
    center_zyx = torch.tensor([[0, 0, 0], [4, 5, 6], [2, 3, 3]], dtype=torch.long)
    batch_index = torch.tensor([0, 1, 1], dtype=torch.long)

    cube, valid = gather_voxel_cube(grid, center_zyx, batch_index, cube_size=3, zero_fill=True)
    ref_cube, ref_valid = _naive_cube(grid, center_zyx, batch_index, 3)

    assert torch.equal(valid, ref_valid)
    assert torch.allclose(cube, ref_cube)


def test_gather_voxel_cube_zero_fill_false_keeps_clamped_but_masks() -> None:
    """
    验证 zero_fill=False 时越界邻居 valid_mask=False(特征值由 clamp 决定, 由调用方屏蔽)。
    """
    grid = torch.randn(1, 1, 4, 4, 4)
    center = torch.tensor([[0, 0, 0]], dtype=torch.long)
    batch = torch.tensor([0], dtype=torch.long)

    _, valid = gather_voxel_cube(grid, center, batch, cube_size=3, zero_fill=False)
    # 角点: 仅 z/y/x 偏移 >=0 的邻居有效
    assert not bool(valid[0, 0, 0, 0])
    assert bool(valid[0, 1, 1, 1])


def test_gather_voxel_cube_rejects_even_cube_size() -> None:
    """
    验证偶数 cube_size 报错。
    """
    with pytest.raises(ValueError, match="cube_size"):
        gather_voxel_cube(torch.randn(1, 1, 4, 4, 4), torch.zeros(1, 3, dtype=torch.long), torch.zeros(1, dtype=torch.long), 2, True)


def test_cube_weighting_params_home_dominates() -> None:
    """
    验证 a>b>c 初值下, 距离相同的情况下 home 邻居 softmax 权重最大。
    """
    params = CubeWeightingParams(2.0, 1.5, 0.5, 1.0)
    # category: 全为"其他", 中心置为 home
    category = torch.full((1, 3, 3, 3), 2, dtype=torch.long)
    category[0, 1, 1, 1] = 0
    dist_sq = torch.zeros(1, 3, 3, 3)
    logit = params(category, dist_sq)
    weights = torch.softmax(logit.reshape(1, -1), dim=-1)
    # 中心(home)在展平后的下标 = 1*9 + 1*3 + 1 = 13
    assert int(weights.argmax()) == 13


def test_cube_weighting_params_atom_beats_other() -> None:
    """
    验证相同距离下"含原子"邻居权重高于"其他"邻居。
    """
    params = CubeWeightingParams(2.0, 1.5, 0.5, 1.0)
    category = torch.tensor([1, 2], dtype=torch.long).reshape(1, 2)
    dist_sq = torch.zeros(1, 2)
    logit = params(category, dist_sq)
    assert float(logit[0, 0]) > float(logit[0, 1])


@pytest.mark.parametrize("mode", ["concat_mlp", "film", "mini_residue"])
def test_real_density_combine_identity_at_init(mode: str) -> None:
    """
    验证三种 combine 模式开局(零初始化)恒等返回 embed。
    """
    combine = FeatureCombine(mode, main_dim=8, cond_dim=8, hidden_dim=16, act_layer=nn.SiLU)
    embed = torch.randn(5, 8)
    density = torch.randn(5, 8)
    out = combine(embed, density)
    assert torch.allclose(out, embed, atol=1e-6)


def test_build_zero_init_residual_mlp_outputs_zero_at_init() -> None:
    """
    验证零初始化残差 MLP 开局输出全 0。
    """
    mlp = build_zero_init_residual_mlp(6, 4, 8, nn.SiLU)
    out = mlp(torch.randn(3, 6))
    assert torch.allclose(out, torch.zeros(3, 4), atol=1e-7)


def test_film_combine_identity_at_init() -> None:
    """
    验证 FiLM 开局 gamma=beta=0, 输出恒等于主特征。
    """
    film = FiLMCombine(main_dim=6, cond_dim=4)
    feat = torch.randn(3, 6)
    out = film(feat, torch.randn(3, 4))
    assert torch.allclose(out, feat, atol=1e-7)


@pytest.mark.parametrize("mode", ["film", "film_plus"])
def test_voxel_point_fusion_film_modes_identity_at_init(mode: str) -> None:
    """
    验证 film / film_plus 开局 gamma=beta=0, voxel 条件不会立刻扰动 point 特征。
    """
    fusion = FeatureCombine(mode, main_dim=6, cond_dim=4, hidden_dim=10, act_layer=nn.SiLU)
    point_feat = torch.randn(5, 6)
    voxel_feat = torch.randn(5, 4)
    out = fusion(point_feat, voxel_feat)
    assert torch.allclose(out, point_feat, atol=1e-7)


def test_voxel_point_fusion_concat_linear_shape() -> None:
    """
    验证旧 concat_linear 模式仍输出 point_channels 维度。
    """
    fusion = FeatureCombine(
        "concat_mlp",
        main_dim=6,
        cond_dim=4,
        hidden_dim=12,
        act_layer=nn.SiLU,
        proj_drop=0.0,
    )
    out = fusion(torch.randn(5, 6), torch.randn(5, 4))
    assert tuple(out.shape) == (5, 6)


def test_vectorized_sampler_box_pos_scatter_gather_roundtrip() -> None:
    """
    验证向量化 trilinear 采样中 (box, pos_in_box) 的 padding 散射-回收保持原点顺序。

    这是 _sample_voxel_feature_trilinear 去 per-BOX 循环后正确性的关键: 每点经 (box,pos) 写入 padded
    网格、再按同一 (box,pos) 取回, 必须逐元素还原原顺序; 且组内 pos 恰为 0..count-1。
    """
    torch.manual_seed(1)
    point_batch = torch.tensor([0, 1, 0, 1, 1, 0], dtype=torch.long)
    point_count = int(point_batch.shape[0])
    batch_size = 2
    values = torch.randn(point_count, 3)

    counts = torch.bincount(point_batch, minlength=batch_size)
    n_max = int(counts.max())
    box_start = torch.zeros(batch_size, dtype=torch.long)
    box_start[1:] = torch.cumsum(counts, dim=0)[:-1]
    sort_idx = torch.argsort(point_batch, stable=True)
    pos_in_box = torch.empty(point_count, dtype=torch.long)
    pos_in_box[sort_idx] = torch.arange(point_count) - box_start[point_batch[sort_idx]]

    padded = torch.zeros(batch_size, n_max, 3)
    padded[point_batch, pos_in_box] = values
    recovered = padded[point_batch, pos_in_box]

    assert torch.equal(recovered, values)
    for box_idx in range(batch_size):
        mask = point_batch == box_idx
        assert sorted(pos_in_box[mask].tolist()) == list(range(int(counts[box_idx])))
