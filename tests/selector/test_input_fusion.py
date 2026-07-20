"""residual_swiglu 与 V48+D 严格退化测试。"""

from __future__ import annotations

import pytest
import torch

from src.selector.model.input_fusion import ResidualSwiGLUFusion, VDensityFusion


def test_residual_swiglu_requires_exact_ordered_source_set() -> None:
    """缺失来源不得补零, 额外来源也不得被静默忽略. """
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


def test_v48d_with_density_disabled_is_exact_v48() -> None:
    """关闭密度分支时 forward 必须逐元素等于同一 base projection + norm。"""
    module = VDensityFusion(
        meta_dim=2,
        output_dim=48,
        correction_gate_hidden_dim=16,
        use_density=False,
        density_encoder=None,
    )
    voxel_final = torch.randn(5, 48)
    expected = module.output_norm(module.base_projection(voxel_final))
    actual = module(
        voxel_final=voxel_final,
        density_input=None,
        voxel_index_local_zyx=torch.zeros((5, 3), dtype=torch.long),
        voxel_batch_index=torch.zeros((5,), dtype=torch.long),
        meta=torch.randn(5, 2),
    )
    assert torch.equal(actual, expected)


@pytest.mark.parametrize("input_channels", (48, 64))
def test_v48d_materializes_base_projection_from_voxel_channels(
    input_channels: int,
) -> None:
    """首个批次可以把 48 或 64 通道的上游特征物化为同一 Selector 宽度. """
    module = VDensityFusion(
        meta_dim=0,
        output_dim=32,
        correction_gate_hidden_dim=16,
        use_density=False,
        density_encoder=None,
    )
    voxel_final = torch.randn(3, input_channels)

    output = module(
        voxel_final=voxel_final,
        density_input=None,
        voxel_index_local_zyx=torch.zeros((3, 3), dtype=torch.long),
        voxel_batch_index=torch.zeros((3,), dtype=torch.long),
        meta=None,
    )

    assert module.base_projection.in_features == input_channels
    assert output.shape == (3, 32)
