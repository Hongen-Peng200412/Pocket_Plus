"""residual_swiglu 与 V5+D 严格退化测试。"""

from __future__ import annotations

import pytest
import torch

from src.selector.model.input_fusion import ResidualSwiGLUFusion, V5DensityFusion


def test_residual_swiglu_requires_exact_ordered_source_set() -> None:
    """缺失来源不得补零，额外来源也不得被静默忽略。"""
    fusion = ResidualSwiGLUFusion(
        source_order=("L0", "L1"),
        source_dims={"L0": 3, "L1": 5},
        projected_dim=4,
        meta_dim=2,
        output_dim=6,
        gate_hidden_dim=8,
    )
    output = fusion(
        {"L0": torch.randn(7, 3), "L1": torch.randn(7, 5)},
        torch.randn(7, 2),
    )
    assert output.shape == (7, 6)
    with pytest.raises(KeyError):
        fusion({"L0": torch.randn(7, 3)}, torch.randn(7, 2))


def test_v5d_with_both_corrections_disabled_is_exact_v48() -> None:
    """关闭 m/c 时 forward 必须逐元素等于同一 base projection + norm。"""
    module = V5DensityFusion(
        grid_source_order=(),
        grid_source_dims={},
        meta_dim=2,
        output_dim=48,
        fusion_projected_dim=8,
        fusion_gate_hidden_dim=16,
        correction_gate_hidden_dim=16,
        use_multiscale=False,
        use_density=False,
        density_encoder=None,
    )
    voxel_final = torch.randn(5, 48)
    expected = module.output_norm(module.base_projection(voxel_final))
    actual = module(
        voxel_final=voxel_final,
        native_grids={},
        density_input=None,
        voxel_index_local_zyx=torch.zeros((5, 3), dtype=torch.long),
        voxel_batch_index=torch.zeros((5,), dtype=torch.long),
        box_shape_zyx=torch.tensor([[80, 80, 80]]),
        meta=torch.randn(5, 2),
    )
    assert torch.equal(actual, expected)
