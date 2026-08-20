# -*- coding: utf-8 -*-
"""Stage1 V3 单 GPU owner 与异步 H2D/D2H smoke."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.inference.full_map import infer_full_map


@pytest.mark.skipif(not torch.cuda.is_available(), reason="需要本机 CUDA GPU")
@pytest.mark.parametrize(
    "device_index",
    tuple(range(min(2, max(1, torch.cuda.device_count())))),
)
def test_full_map_cuda_pipeline_keeps_scientific_shape(device_index: int) -> None:
    """真实 CUDA 前向, 异步 D2H 和有序融合必须生成有限完整图."""

    class Dataset:
        def full_map_context(self, pdb_id):
            return (
                (120, 120, 120),
                np.ones(3, dtype=np.float32),
                np.zeros(3, dtype=np.float32),
                None,
            )

        def materialize_request(self, request):
            start = np.asarray(request.box_start_zyx, dtype=np.float32)
            value = float(start.sum()) / 120.0
            return {
                "density_input": torch.full(
                    (1, 80, 80, 80), value, dtype=torch.float32
                )
            }

    class Wrapper(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = torch.nn.Sequential(
                torch.nn.Conv3d(1, 8, 3, padding=1),
                torch.nn.SiLU(),
                torch.nn.Conv3d(8, 1, 3, padding=1),
            )

        def forward_voxel_probability(self, batch):
            return self.layers(batch["density_input"])

    def collator(rows):
        return {
            "density_input": torch.stack(
                [row["density_input"] for row in rows], dim=0
            )
        }

    device = torch.device(f"cuda:{device_index}")
    wrapper = Wrapper().eval().to(device)
    result = infer_full_map(
        dataset=Dataset(),
        collator=collator,
        wrapper=wrapper,
        pdb_id="cuda",
        device=str(device),
        stride_zyx=(10, 10, 10),
        sigma=0.5,
        window_batch_size=2,
        window_workers=2,
        prefetch_batches=3,
        precision="float16",
        pending_fusion_batches=3,
    )
    assert result.probability_map.shape == (120, 120, 120)
    assert result.window_count == 125
    assert np.isfinite(result.probability_map).all()
