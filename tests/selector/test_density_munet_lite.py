"""DensityMUNetLite 的分辨率、通道与 bottleneck 边界测试。"""

from __future__ import annotations

import torch

from src.selector.model.density_munet_lite import DensityMUNetLite


def test_density_munet_lite_preserves_full_resolution_and_uses_one_transformer_layer() -> None:
    """轻量测试 shape 仍应走四层编码和仅最低分辨率 Transformer。"""
    model = DensityMUNetLite(
        input_shape_zyx=(16, 16, 16),
        channels=(4, 8, 8, 16),
        bottleneck_heads=4,
        bottleneck_layers=1,
        bottleneck_ffn_dim=32,
        dropout=0.0,
    )
    output = model(torch.randn(1, 1, 16, 16, 16))
    assert output.shape == (1, 4, 16, 16, 16)
    assert len(model.bottleneck_transformer.layers) == 1
    assert not any(isinstance(module, torch.nn.TransformerEncoder) for module in model.decoder0.modules())
