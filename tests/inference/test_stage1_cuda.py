# -*- coding: utf-8 -*-
"""验证 Stage1 V3 当前推理线程独占 GPU 时的异步 H2D/D2H."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest
import torch

from src.inference.centered import infer_centered_boxes
from src.inference.full_map import infer_full_map


# ================================================================================================


@pytest.mark.skipif(not torch.cuda.is_available(), reason="需要本机 CUDA GPU")
@pytest.mark.parametrize(
    "device_index",
    tuple(range(min(2, max(1, torch.cuda.device_count())))),
)
def test_full_map_cuda_pipeline_keeps_scientific_shape(device_index: int) -> None:
    """真实 CUDA 前向, 异步 D2H 和有序融合必须生成有限完整图."""

    class Dataset:
        """物化固定形状完整图滑窗的最小 CUDA Dataset."""

        def full_map_context(self, pdb_id):
            """返回 120³ 完整图的形状与世界几何."""

            return (
                (120, 120, 120),
                np.ones(3, dtype=np.float32),
                np.zeros(3, dtype=np.float32),
                None,
            )

        def materialize_request(self, request):
            """按滑窗起点构造可区分的单通道输入."""

            start = np.asarray(request.box_start_zyx, dtype=np.float32)
            value = float(start.sum()) / 120.0
            return {
                "density_input": torch.full((1, 80, 80, 80), value, dtype=torch.float32)
            }

    class Wrapper(torch.nn.Module):
        """用两层 3D 卷积提供真实 CUDA 概率前向."""

        def __init__(self):
            """建立 1->8->1 的轻量体素网络."""

            super().__init__()
            self.layers = torch.nn.Sequential(
                torch.nn.Conv3d(1, 8, 3, padding=1),
                torch.nn.SiLU(),
                torch.nn.Conv3d(8, 1, 3, padding=1),
            )

        def forward_voxel_probability(self, batch):
            """返回完整图流水要求的单通道 logits."""

            return self.layers(batch["density_input"])

    def collator(rows):
        """把相邻滑窗输入堆成模型 batch."""

        return {
            "density_input": torch.stack([row["density_input"] for row in rows], dim=0)
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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="需要本机 CUDA GPU")
@pytest.mark.parametrize(
    "device_index",
    tuple(range(min(2, max(1, torch.cuda.device_count())))),
)
def test_centered_cuda_pipeline_completes_d2h_and_cpu_arrange(
    tmp_path: Path,
    device_index: int,
) -> None:
    """真实 CUDA centered 前向, 异步 D2H 和 CPU 整理必须完成正式字段."""

    density_root = tmp_path / "density" / "cuda"
    density_root.mkdir(parents=True)
    density = np.zeros((1, 80, 80, 80), dtype=np.float32)
    np.save(density_root / "exp.npy", density)
    np.save(density_root / "sim.npy", density)

    class Dataset:
        """物化一个 80³ centered 请求的最小 CUDA Dataset."""

        root = tmp_path

        def materialize_request(self, request):
            """返回单通道密度与同形 hardmask."""

            return {
                "density_input": torch.ones((1, 80, 80, 80), dtype=torch.float32),
                "hardmask": torch.zeros((80, 80, 80), dtype=torch.bool),
            }

    class Wrapper(torch.nn.Module):
        """提供真实 CUDA ligand、auxiliary logits 与 voxel_final."""

        def __init__(self):
            """建立共享 V 特征与配体输出头."""

            super().__init__()
            self.features = torch.nn.Conv3d(1, 2, 1)
            self.ligand = torch.nn.Conv3d(2, 1, 1)

        def forward(self, batch):
            """返回 centered 共同字段所需的 ligand、auxiliary logits 与 voxel_final."""

            voxel_final = torch.nn.functional.silu(
                self.features(batch["density_input"])
            )
            return {
                "voxel_logits_ligand": self.ligand(voxel_final),
                "voxel_logits_aux": self.ligand(voxel_final),
                "voxel_features": {"voxel_final": voxel_final},
            }

    def collator(rows):
        """把 centered 密度和 hardmask 分别堆成模型 batch."""

        return {
            "density_input": torch.stack([row["density_input"] for row in rows], dim=0),
            "hardmask": torch.stack([row["hardmask"] for row in rows], dim=0),
        }

    blobs = {
        "voxel_count": np.asarray([1], dtype=np.int32),
        "fits_centered_box": np.asarray([True]),
        "centered_box_start_zyx": np.asarray([[0, 0, 0]], dtype=np.int32),
        "voxel_offsets": np.asarray([0, 1], dtype=np.int64),
        "voxel_index_global_zyx": np.asarray([[40, 40, 40]], dtype=np.int32),
        "source_probability": np.asarray([0.8], dtype=np.float32),
        "source_probability_mean": np.asarray([0.8], dtype=np.float32),
        "source_threshold_value": np.asarray([0.5], dtype=np.float32),
    }
    device = torch.device(f"cuda:{device_index}")
    wrapper = Wrapper().eval().to(device)
    with ThreadPoolExecutor(max_workers=1) as packer:
        packed, performance = infer_centered_boxes(
            dataset=Dataset(),
            collator=collator,
            wrapper=wrapper,
            pdb_id="cuda",
            producer="unet_c1",
            blobs=blobs,
            full_probability=np.zeros((80, 80, 80), dtype=np.float32),
            origin_xyz=np.zeros(3, dtype=np.float32),
            voxel_size_xyz=np.ones(3, dtype=np.float32),
            device=str(device),
            precision="float16",
            centered_batch_size=1,
            centered_workers=1,
            prefetch_batches=1,
            pending_cpu_batches=1,
            forward_min_voxels=1,
            packer=packer,
        )
        arrays = packed.result()
    assert arrays["centered_probability"].shape == (1,)
    assert arrays["voxel_final"].shape == (1, 2)
    assert arrays["voxel_final"].dtype == np.float16
    assert performance["entry_count"] == 1
