# -*- coding: utf-8 -*-
"""Stage1 V3 几何, 产物, 评分和评估契约测试."""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
import json
from pathlib import Path
import random
import sys
import threading
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import numpy as np
from omegaconf import OmegaConf
import pytest
import torch
from torchmetrics.classification import BinaryAveragePrecision

import src.inference.cli as cli_module
from src.inference.blobs import extract_probability_blobs
from src.inference.artifacts import (
    Stage1ArtifactPaths,
    f_alpha_tag,
    load_stage1_npz,
    publish_stage1_artifact,
)
from src.inference.calibration import (
    calibrate_semantic_thresholds,
    tune_centered_selection,
)
from src.inference.centered import infer_centered_boxes, pack_centered_entries
from src.inference.evaluation import (
    aggregate_semantic_prauc,
    aggregate_stage1_metrics,
    evaluate_centered_pdb,
    semantic_prauc_histogram,
)
from src.inference.full_map import (
    FullMapInferenceSession,
    FullMapResult,
    gaussian_window_weight,
    window_starts_zyx,
)
from src.inference.pipeline import (
    CENTERED_BLOB_LIMIT,
    run_blobs_stage,
    run_centered_stage,
    run_evaluate_stage,
    run_probability_stage,
    run_tune_stage,
)
from src.inference.scoring import (
    score_centered_candidates,
    sum_gaussian_atom_terms,
)


# ================================================================================================


def test_window_geometry_and_normalized_gaussian() -> None:
    """边界窗口必须覆盖末端, sigma 必须属于归一化坐标而不是体素单位."""

    starts = window_starts_zyx((101, 120, 130), (80, 80, 80), (50, 50, 50))
    assert starts[0] == (0, 0, 0)
    assert starts[-1] == (21, 40, 50)
    weight = gaussian_window_weight((80, 80, 80), 0.5)
    expected_corner = np.exp(-3.0 / (2.0 * 0.5**2))
    assert weight.dtype == np.float32
    assert np.isclose(weight[0, 0, 0], expected_corner, rtol=1e-5)
    assert weight[39, 39, 39] > weight[0, 0, 0]


def test_full_map_session_prefetches_next_pdb_without_duplicate_materialization() -> None:
    """同一 session 必须在首个 PDB 前向时预取下一 PDB 的首批窗口."""

    second_materialized = threading.Event()
    materialized_pdb_ids: list[str] = []

    class Dataset:
        """记录每个滑窗请求所属 PDB, 并通知第二个 PDB 已完成物化."""

        def full_map_context(
            self,
            _pdb_id: str,
        ) -> tuple[tuple[int, int, int], np.ndarray, np.ndarray, None]:
            """返回固定 80³ 完整图几何."""

            return (80, 80, 80), np.ones(3), np.zeros(3), None

        def materialize_request(self, request: Any) -> dict[str, torch.Tensor]:
            """记录请求所属 PDB 并返回固定密度张量."""

            materialized_pdb_ids.append(request.pdb_id)
            if request.pdb_id == "second":
                second_materialized.set()
            return {"density": torch.zeros((1, 80, 80, 80))}

    class Wrapper:
        """在首个 PDB 前向开始前确认第二个 PDB 的预取已经执行."""

        call_count = 0

        def forward_voxel_probability(
            self,
            batch: Mapping[str, torch.Tensor],
        ) -> torch.Tensor:
            """等待下一 PDB 预取证据后返回零 logits."""

            if self.call_count == 0:
                assert second_materialized.wait(timeout=5.0)
            self.call_count += 1
            batch_size = int(batch["density"].shape[0])
            return torch.zeros((batch_size, 1, 80, 80, 80))

    def collator(
        rows: Sequence[Mapping[str, torch.Tensor]],
    ) -> dict[str, torch.Tensor]:
        """按请求顺序拼接固定密度张量."""

        return {"density": torch.stack([row["density"] for row in rows])}

    wrapper = Wrapper()
    with FullMapInferenceSession(
        dataset=Dataset(),
        collator=collator,
        wrapper=wrapper,
        device="cpu",
        stride_zyx=(50, 50, 50),
        sigma=0.5,
        window_batch_size=1,
        window_workers=2,
        prefetch_batches=1,
        precision="float32",
        pending_fusion_batches=1,
    ) as session:
        first = session.infer("first", "second")
        second = session.infer("second", None)

    assert first.window_count == 1
    assert second.window_count == 1
    assert materialized_pdb_ids.count("first") == 1
    assert materialized_pdb_ids.count("second") == 1
    assert wrapper.call_count == 2


def test_inference_shell_pins_nested_numeric_threads_to_one() -> None:
    """外层推理与调参并发启用时, shell 必须覆盖继承环境并把数值库线程固定为 1."""

    project_root = Path(__file__).resolve().parents[2]
    shell_text = (
        project_root / "训练与运行" / "sh" / "infer" / "stage1_v3.sh"
    ).read_text(encoding="utf-8")
    for variable_name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        assert f"export {variable_name}=1" in shell_text
        assert f"{variable_name}:-1" not in shell_text


def test_sampling_comparison_shell_uses_official_entry_and_resource_profiles() -> None:
    """三模型正式入口应复用五阶段脚本, 并固定 A800/A100 完整图 batch."""

    project_root = Path(__file__).resolve().parents[2]
    shell_text = (
        project_root
        / "训练与运行"
        / "sh"
        / "infer"
        / "unet_c1_sampling_comparison.sh"
    ).read_text(encoding="utf-8")
    a800 = OmegaConf.load(
        project_root / "configs" / "inference" / "stage1_v3_a800_24cpu.yaml"
    )
    a100 = OmegaConf.load(
        project_root / "configs" / "inference" / "stage1_v3_a100_16cpu.yaml"
    )

    assert "训练与运行/sh/infer/stage1_v3.sh" in shell_text
    assert "stage1 centered" not in shell_text
    assert "unet_c1_pdb_centric_v1/artifacts" in shell_text
    assert "unet_c1_pdb_centric_v2/artifacts" in shell_text
    assert int(a800.window.batch_size) == 24
    assert int(a800.window.workers) == 18
    assert int(a800.window.prefetch_batches) == 18
    assert int(a800.calibration.workers) == 24
    assert int(a100.window.batch_size) == 12
    assert int(a100.window.workers) == 10
    assert int(a100.window.prefetch_batches) == 10
    assert int(a100.calibration.workers) == 16


def test_blobs_keep_all_components_and_sort_stably() -> None:
    """连通区域阶段不应用 min_voxels, 同均值按最小全图线性索引排序."""

    probability = np.zeros((80, 80, 80), dtype=np.float32)
    probability[1, 1, 1] = 0.8
    probability[70, 70, 70] = 0.8
    arrays = extract_probability_blobs(probability, 0.5)
    assert arrays["voxel_count"].tolist() == [1, 1]
    assert arrays["voxel_index_global_zyx"].tolist() == [[1, 1, 1], [70, 70, 70]]
    assert arrays["voxel_offsets"].tolist() == [0, 1, 2]
    assert arrays["fits_centered_box"].tolist() == [True, True]


def test_blob_sort_uses_archived_float32_mean_before_linear_tie_break() -> None:
    """正式 float32 均值相同后必须按最小全图线性索引排序."""

    base = np.float32(0.75)
    higher = np.nextafter(base, np.float32(1.0), dtype=np.float32)
    probability = np.zeros((80, 80, 80), dtype=np.float32)
    probability[1, 1, 1:3] = base
    probability[10, 10, 10] = base
    probability[10, 10, 11] = higher
    arrays = extract_probability_blobs(probability, 0.5)
    assert arrays["source_probability_mean"][0] == arrays["source_probability_mean"][1]
    assert arrays["voxel_index_global_zyx"][0].tolist() == [1, 1, 1]


def test_centered_start_uses_blob_bbox_feasible_interval() -> None:
    """偏斜 blob 只要包围盒可容纳, 就不能因质心起点偏移被误判."""

    probability = np.zeros((100, 100, 100), dtype=np.float32)
    probability[:80, 0, 0] = 0.8
    probability[70:80, :10, :10] = 0.8
    arrays = extract_probability_blobs(probability, 0.5)
    assert arrays["fits_centered_box"].tolist() == [True]
    assert arrays["centered_box_start_zyx"][0].tolist() == [0, 0, 0]


def test_centered_packing_preserves_offsets_and_feature_dtypes() -> None:
    """逐候选 V/A/P 值表必须共享各自 offsets, L0 保持 50 维 float32."""

    entries = []
    for source_index, voxel_count in ((3, 2), (7, 1)):
        entries.append(
            {
                "source_blob_index": np.asarray(source_index, dtype=np.int32),
                "source_blob_fits_centered_box": np.asarray(True),
                "source_voxel_count": np.asarray(voxel_count, dtype=np.int32),
                "box_start_zyx": np.asarray((0, 0, 0), dtype=np.int32),
                "box_shape_zyx": np.asarray((80, 80, 80), dtype=np.uint8),
                "box_origin_world": np.zeros(3, dtype=np.float32),
                "voxel_size_world": np.ones(3, dtype=np.float32),
                "source_probability_mean": np.asarray(0.8, dtype=np.float32),
                "source_threshold_value": np.asarray(0.5, dtype=np.float32),
                "voxel_index_local_zyx": np.zeros((voxel_count, 3), dtype=np.int16),
                "source_probability": np.full(voxel_count, 0.8, dtype=np.float32),
                "centered_probability": np.full(voxel_count, 0.7, dtype=np.float32),
                "voxel_aux_index_local_zyx": np.zeros((1, 3), dtype=np.int16),
                "voxel_aux_probability": np.ones(1, dtype=np.float32),
                "voxel_final": np.zeros((voxel_count, 4), dtype=np.float16),
                "A_global_index": np.asarray([source_index], dtype=np.int64),
                "A_coord_local_xyz": np.zeros((1, 3), dtype=np.float32),
                "A_coord_centered_world": np.zeros((1, 3), dtype=np.float32),
                "A_probability": np.asarray([0.9], dtype=np.float32),
                "A_feat_L0": np.zeros((1, 50), dtype=np.float32),
                "A_feat_L1": np.zeros((1, 2), dtype=np.float16),
                "A_feat_L2": np.zeros((1, 3), dtype=np.float16),
                "A_feat_L3": np.zeros((1, 4), dtype=np.float16),
                "P_coord_local_xyz": np.zeros((1, 3), dtype=np.float32),
                "P_probability": np.asarray([0.6], dtype=np.float32),
                "P_feat_L2": np.zeros((1, 2), dtype=np.float16),
                "P_feat_L3": np.zeros((1, 3), dtype=np.float16),
                "v_centroid_local_zyx": np.zeros(3, dtype=np.float32),
                "crop_start_local_zyx": np.zeros(3, dtype=np.int16),
                "crop_center_offset_zyx": np.zeros(3, dtype=np.float32),
                "crop_clipped_axis_mask": np.zeros(3, dtype=np.bool_),
                "experimental_density_48": np.zeros((48, 48, 48), dtype=np.float32),
                "simulated_density_48": np.zeros((48, 48, 48), dtype=np.float32),
                "source_probability_48": np.zeros((48, 48, 48), dtype=np.float32),
            }
        )
    arrays = pack_centered_entries(entries, "Find_1")
    assert arrays["voxel_offsets"].tolist() == [0, 2, 3]
    assert arrays["A_offsets"].tolist() == [0, 1, 2]
    assert arrays["A_feat_L0"].shape == (2, 50)
    assert arrays["A_feat_L0"].dtype == np.float32
    assert arrays["voxel_final"].shape == (3, 4)
    assert arrays["centered_box_index"].tolist() == [0, 1]
    assert arrays["source_blob_fits_centered_box"].tolist() == [True, True]
    assert arrays["source_voxel_count"].tolist() == [2, 1]


def test_centered_cpu_arranger_accepts_bfloat16_output(tmp_path: Path) -> None:
    """合成 CPU bf16 输出必须先转 float32 再进入 NumPy."""

    density_root = tmp_path / "density" / "demo"
    density_root.mkdir(parents=True)
    np.save(
        density_root / "exp.npy",
        np.zeros((1, 80, 80, 80), dtype=np.float32),
    )
    np.save(
        density_root / "sim.npy",
        np.ones((1, 80, 80, 80), dtype=np.float32),
    )

    class Dataset:
        """提供单个 centered 请求所需的最小 CPU Dataset."""

        root = tmp_path

        def materialize_request(self, request):
            """返回只含 hardmask 的单请求载荷."""

            hardmask = torch.zeros((80, 80, 80), dtype=torch.bool)
            hardmask[2, 3, 4] = True
            return {"hardmask": hardmask}

    class Wrapper:
        """返回 BF16 ligand, auxiliary logits 与 voxel_final 的最小 wrapper."""

        def __call__(self, batch):
            """按输入 batch 大小构造共同 centered 字段所需的 BF16 模型输出."""

            shape = (len(batch["hardmask"]), 1, 80, 80, 80)
            return {
                "voxel_logits_ligand": torch.zeros(shape, dtype=torch.bfloat16),
                "voxel_logits_aux": torch.zeros(shape, dtype=torch.bfloat16),
                "voxel_features": {
                    "voxel_final": torch.zeros(
                        (shape[0], 4, 80, 80, 80), dtype=torch.bfloat16
                    )
                },
            }

    def collator(rows):
        """把单请求 hardmask 堆成模型 batch."""

        return {"hardmask": torch.stack([row["hardmask"] for row in rows])}

    blobs = {
        "voxel_count": np.asarray([1], dtype=np.int32),
        "fits_centered_box": np.asarray([True]),
        "centered_box_start_zyx": np.asarray([[0, 0, 0]], dtype=np.int32),
        "voxel_offsets": np.asarray([0, 1], dtype=np.int64),
        "voxel_index_global_zyx": np.asarray([[1, 1, 1]], dtype=np.int32),
        "source_probability": np.asarray([0.8], dtype=np.float32),
        "source_probability_mean": np.asarray([0.8], dtype=np.float32),
        "source_threshold_value": np.asarray([0.5], dtype=np.float32),
    }
    with ThreadPoolExecutor(max_workers=1) as packer:
        packed, _ = infer_centered_boxes(
            dataset=Dataset(),
            collator=collator,
            wrapper=Wrapper(),
            pdb_id="demo",
            producer="unet_c1",
            blobs=blobs,
            full_probability=np.full((80, 80, 80), 2.0, dtype=np.float32),
            origin_xyz=np.zeros(3, dtype=np.float32),
            voxel_size_xyz=np.ones(3, dtype=np.float32),
            device="cpu",
            precision="bf16",
            centered_batch_size=1,
            centered_workers=1,
            prefetch_batches=1,
            pending_cpu_batches=1,
            forward_min_voxels=1,
            packer=packer,
        )
        arrays = packed.result()
    assert arrays["centered_probability"].dtype == np.float32
    assert arrays["voxel_final"].dtype == np.float16
    assert arrays["voxel_final"].shape == (1, 4)
    assert arrays["source_blob_fits_centered_box"].tolist() == [True]
    assert arrays["source_voxel_count"].tolist() == [1]
    assert {
        "voxel_aux_offsets",
        "voxel_aux_index_local_zyx",
        "voxel_aux_probability",
        "v_centroid_local_zyx",
        "crop_start_local_zyx",
        "crop_center_offset_zyx",
        "crop_clipped_axis_mask",
        "experimental_density_48",
        "simulated_density_48",
        "source_probability_48",
    } <= arrays.keys()
    assert arrays["voxel_aux_offsets"].tolist() == [0, 1]
    assert arrays["voxel_aux_index_local_zyx"].tolist() == [[2, 3, 4]]
    np.testing.assert_allclose(arrays["voxel_aux_probability"], [0.5])
    assert arrays["crop_start_local_zyx"].tolist() == [[0, 0, 0]]
    np.testing.assert_array_equal(arrays["experimental_density_48"], 0.0)
    np.testing.assert_array_equal(arrays["simulated_density_48"], 1.0)
    np.testing.assert_array_equal(arrays["source_probability_48"], 2.0)


def test_centered_materializes_requests_concurrently_and_preserves_order(
    tmp_path: Path,
) -> None:
    """同一 GPU batch 内的候选请求应独立并发物化, 归档顺序仍跟随 blobs."""

    density_root = tmp_path / "density" / "demo"
    density_root.mkdir(parents=True)
    np.save(density_root / "exp.npy", np.zeros((1, 80, 80, 80), dtype=np.float32))
    np.save(density_root / "sim.npy", np.zeros((1, 80, 80, 80), dtype=np.float32))
    materialize_barrier = threading.Barrier(2)

    class Dataset:
        """让同一 batch 的两个候选请求在物化屏障会合."""

        root = tmp_path

        def materialize_request(self, request: Any) -> dict[str, torch.Tensor]:
            """让两个请求在屏障会合后返回各自 hardmask."""

            materialize_barrier.wait(timeout=5.0)
            hardmask = torch.zeros((80, 80, 80), dtype=torch.bool)
            hardmask[request.box_start_zyx] = True
            return {"hardmask": hardmask}

    class Wrapper:
        """为并发物化测试返回固定形状的零模型输出."""

        def __call__(self, batch: Mapping[str, torch.Tensor]) -> dict[str, Any]:
            """返回与输入 batch 大小一致的零模型输出."""

            shape = (len(batch["hardmask"]), 1, 80, 80, 80)
            return {
                "voxel_logits_ligand": torch.zeros(shape),
                "voxel_logits_aux": torch.zeros(shape),
                "voxel_features": {
                    "voxel_final": torch.zeros((shape[0], 1, 80, 80, 80))
                },
            }

    def collator(
        rows: Sequence[Mapping[str, torch.Tensor]],
    ) -> dict[str, torch.Tensor]:
        """按请求顺序拼接 hardmask."""

        return {"hardmask": torch.stack([row["hardmask"] for row in rows])}

    blobs = {
        "voxel_count": np.asarray([1, 1], dtype=np.int32),
        "fits_centered_box": np.asarray([True, True]),
        "centered_box_start_zyx": np.asarray([[0, 0, 0], [0, 0, 0]], dtype=np.int32),
        "voxel_offsets": np.asarray([0, 1, 2], dtype=np.int64),
        "voxel_index_global_zyx": np.asarray([[1, 1, 1], [70, 70, 70]], dtype=np.int32),
        "source_probability": np.asarray([0.9, 0.8], dtype=np.float32),
        "source_probability_mean": np.asarray([0.9, 0.8], dtype=np.float32),
        "source_threshold_value": np.asarray([0.5], dtype=np.float32),
    }
    with ThreadPoolExecutor(max_workers=1) as packer:
        packed, _ = infer_centered_boxes(
            dataset=Dataset(),
            collator=collator,
            wrapper=Wrapper(),
            pdb_id="demo",
            producer="unet_c1",
            blobs=blobs,
            full_probability=np.zeros((80, 80, 80), dtype=np.float32),
            origin_xyz=np.zeros(3, dtype=np.float32),
            voxel_size_xyz=np.ones(3, dtype=np.float32),
            device="cpu",
            precision="float32",
            centered_batch_size=2,
            centered_workers=2,
            prefetch_batches=1,
            pending_cpu_batches=1,
            forward_min_voxels=1,
            packer=packer,
        )
        arrays = packed.result()

    assert arrays["source_blob_index"].tolist() == [0, 1]
    assert arrays["voxel_offsets"].tolist() == [0, 1, 2]


def test_oversized_blob_is_forwarded_and_archives_only_in_box_voxels(
    tmp_path: Path,
) -> None:
    """跨越 80 体素的 blob 仍需前向, 并单独保存完整体素数与可容纳标志."""

    density_root = tmp_path / "density" / "demo"
    density_root.mkdir(parents=True)
    np.save(density_root / "exp.npy", np.zeros((1, 100, 100, 100), dtype=np.float32))
    np.save(density_root / "sim.npy", np.zeros((1, 100, 100, 100), dtype=np.float32))

    class Dataset:
        """为超大 blob 测试返回固定 centered 输入."""

        root = tmp_path

        def materialize_request(self, _request: Any) -> dict[str, torch.Tensor]:
            """返回不含辅助受体体素的固定 hardmask."""

            return {"hardmask": torch.zeros((80, 80, 80), dtype=torch.bool)}

    class Wrapper:
        """为超大 blob 测试返回固定形状的零模型输出."""

        def __call__(self, batch: Mapping[str, torch.Tensor]) -> dict[str, Any]:
            """返回与输入 batch 大小一致的零模型输出."""

            shape = (len(batch["hardmask"]), 1, 80, 80, 80)
            return {
                "voxel_logits_ligand": torch.zeros(shape),
                "voxel_logits_aux": torch.zeros(shape),
                "voxel_features": {
                    "voxel_final": torch.zeros((shape[0], 1, 80, 80, 80))
                },
            }

    def collator(
        rows: Sequence[Mapping[str, torch.Tensor]],
    ) -> dict[str, torch.Tensor]:
        """按请求顺序拼接 hardmask."""

        return {"hardmask": torch.stack([row["hardmask"] for row in rows])}

    source_voxels = np.asarray([[z, 40, 40] for z in range(100)], dtype=np.int32)
    blobs = {
        "voxel_count": np.asarray([100], dtype=np.int32),
        "fits_centered_box": np.asarray([False]),
        "centered_box_start_zyx": np.asarray([[-1, -1, -1]], dtype=np.int32),
        "voxel_offsets": np.asarray([0, 100], dtype=np.int64),
        "voxel_index_global_zyx": source_voxels,
        "source_probability": np.full(100, 0.8, dtype=np.float32),
        "source_probability_mean": np.asarray([0.8], dtype=np.float32),
        "source_threshold_value": np.asarray([0.5], dtype=np.float32),
    }
    with ThreadPoolExecutor(max_workers=1) as packer:
        packed, performance = infer_centered_boxes(
            dataset=Dataset(),
            collator=collator,
            wrapper=Wrapper(),
            pdb_id="demo",
            producer="unet_c1",
            blobs=blobs,
            full_probability=np.zeros((100, 100, 100), dtype=np.float32),
            origin_xyz=np.zeros(3, dtype=np.float32),
            voxel_size_xyz=np.ones(3, dtype=np.float32),
            device="cpu",
            precision="float32",
            centered_batch_size=1,
            centered_workers=1,
            prefetch_batches=1,
            pending_cpu_batches=1,
            forward_min_voxels=1,
            packer=packer,
        )
        arrays = packed.result()

    assert performance["entry_count"] == 1
    assert arrays["source_blob_fits_centered_box"].tolist() == [False]
    assert arrays["source_voxel_count"].tolist() == [100]
    assert arrays["box_start_zyx"].tolist() == [[10, 0, 0]]
    assert arrays["voxel_offsets"].tolist() == [0, 80]
    assert arrays["voxel_index_local_zyx"][:, 0].tolist() == list(range(80))


def test_find_gaussian_score_uses_five_angstrom_cutoff() -> None:
    """恰好 5 Å 的 A 原子参与 Gaussian 分数, 超过 5 Å 的原子不参与."""

    centered = {
        "source_probability_mean": np.asarray([0.5], dtype=np.float32),
        "voxel_offsets": np.asarray([0, 1], dtype=np.int64),
        "voxel_index_local_zyx": np.asarray([[0, 0, 0]], dtype=np.int16),
        "voxel_size_world": np.asarray([[1.0, 1.0, 1.0]], dtype=np.float32),
        "A_offsets": np.asarray([0, 3], dtype=np.int64),
        "A_coord_local_xyz": np.asarray(
            [[0.5, 0.5, 0.5], [5.5, 0.5, 0.5], [5.6, 0.5, 0.5]],
            dtype=np.float32,
        ),
        "A_probability": np.ones(3, dtype=np.float32),
    }
    score = score_centered_candidates(
        centered,
        score_mode="gaussian",
        score_parameters={
            "tau_angstrom": 1.0,
            "lambda_positive": 0.2,
            "lambda_negative": 0.1,
        },
    )
    expected = 0.5 + 0.2 * (1.0 + np.exp(-12.5))
    assert np.allclose(score, [expected])


def test_find_centered_keeps_atom_at_ten_angstrom_boundary(tmp_path: Path) -> None:
    """Find centered 的 A 表保留距来源体素中心恰好 10 Å 的原子."""

    density_root = tmp_path / "density" / "demo"
    density_root.mkdir(parents=True)
    density = np.zeros((1, 80, 80, 80), dtype=np.float32)
    np.save(density_root / "exp.npy", density)
    np.save(density_root / "sim.npy", density)

    class Dataset:
        """提供一个来源体素与一个受体原子的 Find centered 输入."""

        root = tmp_path

        def materialize_request(self, request):
            """返回与单个受体原子逐项对齐的 Dataset 字段."""

            return {
                "hardmask": torch.zeros((80, 80, 80), dtype=torch.bool),
                "atom_counts": torch.tensor(1, dtype=torch.int64),
                "atom_global_indices": torch.tensor([7], dtype=torch.int64),
                "atom_feat": torch.zeros((1, 49), dtype=torch.float32),
                "atom_is_backbone": torch.tensor([False]),
            }

    class Wrapper:
        """返回位于来源体素中心 10 Å 处的单个 Find A 原子."""

        def __call__(self, batch):
            """构造一个候选所需的体素, A 原子与空 P 点输出."""

            batch_size = int(batch["hardmask"].shape[0])
            box_shape = (batch_size, 1, 80, 80, 80)
            return {
                "voxel_logits_ligand": torch.zeros(box_shape),
                "voxel_logits_aux": torch.zeros(box_shape),
                "voxel_features": {"voxel_final": torch.zeros(box_shape)},
                "atom_counts": torch.ones(batch_size, dtype=torch.int64),
                "atom_global_indices": torch.tensor([7], dtype=torch.int64),
                "atom_coord_local_voxel": torch.tensor(
                    [[10.5, 0.5, 0.5]], dtype=torch.float32
                ),
                "atom_logits": torch.zeros((1, 1)),
                "A_feat_L1": torch.zeros((1, 1)),
                "A_feat_L2": torch.zeros((1, 1)),
                "A_feat_L3": torch.zeros((1, 1)),
                "anchor_batch_index": torch.empty(0, dtype=torch.int64),
                "anchor_coord_local_voxel": torch.empty((0, 3)),
                "pseudo_logits": torch.empty((0, 1)),
                "P_feat_L2": torch.empty((0, 1)),
                "P_feat_L3": torch.empty((0, 1)),
            }

    def collator(rows):
        """把单个 Find Dataset 载荷拼成 batch 字段."""

        return {
            "hardmask": torch.stack([row["hardmask"] for row in rows]),
            "atom_counts": torch.stack([row["atom_counts"] for row in rows]),
            "atom_global_indices": torch.cat(
                [row["atom_global_indices"] for row in rows]
            ),
            "atom_feat": torch.cat([row["atom_feat"] for row in rows]),
            "atom_is_backbone": torch.cat([row["atom_is_backbone"] for row in rows]),
        }

    blobs = {
        "voxel_count": np.asarray([1], dtype=np.int32),
        "fits_centered_box": np.asarray([True]),
        "centered_box_start_zyx": np.zeros((1, 3), dtype=np.int32),
        "voxel_offsets": np.asarray([0, 1], dtype=np.int64),
        "voxel_index_global_zyx": np.zeros((1, 3), dtype=np.int32),
        "source_probability": np.asarray([0.8], dtype=np.float32),
        "source_probability_mean": np.asarray([0.8], dtype=np.float32),
        "source_threshold_value": np.asarray([0.5], dtype=np.float32),
    }
    with ThreadPoolExecutor(max_workers=1) as packer:
        packed, _ = infer_centered_boxes(
            dataset=Dataset(),
            collator=collator,
            wrapper=Wrapper(),
            pdb_id="demo",
            producer="Find_1",
            blobs=blobs,
            full_probability=np.zeros((80, 80, 80), dtype=np.float32),
            origin_xyz=np.zeros(3, dtype=np.float32),
            voxel_size_xyz=np.ones(3, dtype=np.float32),
            device="cpu",
            precision="float32",
            centered_batch_size=1,
            centered_workers=1,
            prefetch_batches=1,
            pending_cpu_batches=1,
            forward_min_voxels=1,
            packer=packer,
        )
        arrays = packed.result()
    assert arrays["A_global_index"].tolist() == [7]


def test_find_gaussian_score_reuses_calibration_numeric_terms_exactly() -> None:
    """校准与正式评分必须共享 float64 求和后规范为 float32 的数值顺序."""

    centered = {
        "source_probability_mean": np.asarray([0.25], dtype=np.float32),
        "voxel_offsets": np.asarray([0, 1], dtype=np.int64),
        "voxel_index_local_zyx": np.asarray([[0, 0, 0]], dtype=np.int16),
        "voxel_size_world": np.ones((1, 3), dtype=np.float32),
        "A_offsets": np.asarray([0, 2], dtype=np.int64),
        "A_coord_local_xyz": np.asarray(
            [[0.5, 0.5, 0.5], [1.5, 0.5, 0.5]], dtype=np.float32
        ),
        "A_probability": np.asarray([0.7, 0.2], dtype=np.float32),
    }
    positive, negative = sum_gaussian_atom_terms(
        centered["A_offsets"],
        np.asarray([0.0, 1.0], dtype=np.float32),
        centered["A_probability"],
        0.75,
    )
    expected = (
        centered["source_probability_mean"]
        + np.float32(0.064) * positive
        - np.float32(0.004) * negative
    )
    actual = score_centered_candidates(
        centered,
        score_mode="gaussian",
        score_parameters={
            "tau_angstrom": 0.75,
            "lambda_positive": 0.064,
            "lambda_negative": 0.004,
        },
    )
    np.testing.assert_array_equal(actual, expected)


def test_semantic_threshold_uses_pdb_equal_macro_curve() -> None:
    """PDB 等权语义目标不能被大体积 PDB 的 micro 计数主导."""

    # float32 与 bool, (1, 1, 1000), large 完整图的 ZYX 概率与逐体素语义真值.
    large_probability = np.concatenate(
        (
            np.full(100, 0.8, dtype=np.float32),
            np.full(900, 0.2, dtype=np.float32),
        )
    ).reshape(1, 1, 1000)
    large_target = np.concatenate(
        (
            np.ones(100, dtype=np.bool_),
            np.zeros(900, dtype=np.bool_),
        )
    ).reshape(1, 1, 1000)
    # float32 与 bool, (1, 1, 1), small 完整图的 ZYX 概率与逐体素语义真值.
    small_probability = np.asarray([[[0.2]]], dtype=np.float32)
    small_target = np.asarray([[[True]]], dtype=np.bool_)
    semantic = calibrate_semantic_thresholds(
        [
            (large_probability, large_target),
            (small_probability, small_target),
        ],
        denominator=10,
        alpha=1.0,
    )
    assert semantic["pdb_count"] == 2
    assert semantic["threshold_grid_index"] == 0
    assert semantic["scan"]["tp"].shape == (11,)
    assert semantic["scan"]["macro_f_beta_curve"].shape == (11,)
    assert "f_beta_curve" not in semantic["scan"]
    assert "micro_f_beta" not in semantic
    assert semantic["macro_f_beta"] == pytest.approx((2.0 / 11.0 + 1.0) / 2.0)

    # 同一汇总计数若按旧 micro 口径选择, 最优网格会落在 0.3 而不是 0.0.
    tp = semantic["scan"]["tp"].astype(np.float64)
    fp = semantic["scan"]["fp"].astype(np.float64)
    fn = semantic["scan"]["fn"].astype(np.float64)
    micro_curve = np.divide(
        2.0 * tp,
        2.0 * tp + fp + fn,
        out=np.zeros_like(tp),
        where=(2.0 * tp + fp + fn) > 0,
    )
    assert int(np.argmax(micro_curve)) == 3


def test_semantic_tuning_files_are_separate_from_pdb_split(
    tmp_path: Path,
) -> None:
    """生产者级语义文件必须进入 tuning, calibration 只保留逐 PDB 目录."""

    paths = Stage1ArtifactPaths(tmp_path, "unet_c1", "calibration", "demo")
    publish_stage1_artifact(
        paths.artifact("probability"),
        {"probability_map": np.asarray([[[0.8, 0.2]]], dtype=np.float32)},
        paths.complete("probability"),
    )
    density_root = tmp_path / "data" / "density" / "demo"
    density_root.mkdir(parents=True)
    np.save(
        density_root / "union_mask.npy",
        np.asarray([[[[True, False]]]], dtype=np.bool_),
    )
    result = run_blobs_stage(
        config=OmegaConf.create(
            {
                "blob_workers": 1,
                "calibration": {"semantic_denominator": 10},
            }
        ),
        data_root=tmp_path / "data",
        producer="unet_c1",
        split="calibration",
        pdb_ids=("demo",),
        output_root=tmp_path,
        alpha=1.0,
        semantic_threshold=None,
        fit_semantic=True,
        overwrite=False,
    )

    assert result is not None
    tuning_root = tmp_path / "unet_c1" / "tuning"
    assert (tuning_root / "F1_semantic.json").is_file()
    assert (tuning_root / "F1_semantic_scan.npz").is_file()
    assert {entry.name for entry in paths.pdb_root.parent.iterdir()} == {"demo"}


def test_evaluate_reports_distinct_micro_and_macro_metrics() -> None:
    """评估必须同时发布三类指标的 micro 与 PDB 等权 macro 结果."""

    # int16, (10, 3), large 的十个单体素候选分别命中十个真实 occurrence.
    large_coordinates = np.asarray(
        [[0, 0, index] for index in range(10)],
        dtype=np.int16,
    )
    # large 的三类 F1 均为 1, 并在 micro 计数中贡献十个命中.
    large = evaluate_centered_pdb(
        pdb_id="large",
        centered={
            "selected": np.ones(10, dtype=np.bool_),
            "score": np.linspace(1.0, 0.1, 10, dtype=np.float32),
            "source_blob_index": np.arange(10, dtype=np.int32),
            "voxel_offsets": np.arange(11, dtype=np.int64),
            "voxel_index_local_zyx": large_coordinates,
            "box_start_zyx": np.zeros((10, 3), dtype=np.int32),
        },
        occurrence_id=np.arange(10, dtype=np.int32),
        occurrence_voxel_zyx=tuple(
            large_coordinates[index : index + 1].astype(np.int32) for index in range(10)
        ),
        full_shape_zyx=(1, 1, 10),
        coverage_thresholds=(0.3, 0.5, 0.6),
        topk_values=(3, 4, 5),
    )
    # int16, (2, 3), small 两个候选框内的局部 ZYX 体素坐标; box_start_zyx 全为 0, 因而同一数值也是完整图 ZYX 坐标, 且只有第一项命中真实 occurrence.
    small_coordinates = np.asarray([[0, 0, 0], [0, 0, 1]], dtype=np.int16)
    # small 的 precision 为 1/2、recall 为 1, 用非对称分母分别约束 P 与 R.
    small = evaluate_centered_pdb(
        pdb_id="small",
        centered={
            "selected": np.ones(2, dtype=np.bool_),
            "score": np.asarray([1.0, 0.5], dtype=np.float32),
            "source_blob_index": np.arange(2, dtype=np.int32),
            "voxel_offsets": np.arange(3, dtype=np.int64),
            "voxel_index_local_zyx": small_coordinates,
            "box_start_zyx": np.zeros((2, 3), dtype=np.int32),
        },
        occurrence_id=np.asarray([0], dtype=np.int32),
        occurrence_voxel_zyx=(small_coordinates[:1].astype(np.int32),),
        full_shape_zyx=(1, 1, 2),
        coverage_thresholds=(0.3, 0.5, 0.6),
        topk_values=(3, 4, 5),
    )
    # 两个 PDB 共有 12 个预测候选、11 个真实 occurrence 和 11 个命中; small 使 macro precision 与 recall 也不同.
    metrics = aggregate_stage1_metrics(
        (large, small),
        coverage_thresholds=(0.3, 0.5, 0.6),
        topk_values=(3, 4, 5),
    )
    assert metrics["semantic_micro_precision"] == pytest.approx(11.0 / 12.0)
    assert metrics["semantic_micro_recall"] == pytest.approx(1.0)
    assert metrics["semantic_micro_f1"] == pytest.approx(22.0 / 23.0)
    assert metrics["semantic_macro_precision"] == pytest.approx(0.75)
    assert metrics["semantic_macro_recall"] == pytest.approx(1.0)
    assert metrics["semantic_macro_f1"] == pytest.approx(5.0 / 6.0)
    assert metrics["coverage_micro_precision_0p3"] == pytest.approx(11.0 / 12.0)
    assert metrics["coverage_micro_recall_0p3"] == pytest.approx(1.0)
    assert metrics["coverage_micro_f1_0p3"] == pytest.approx(22.0 / 23.0)
    assert metrics["coverage_macro_precision_0p3"] == pytest.approx(0.75)
    assert metrics["coverage_macro_recall_0p3"] == pytest.approx(1.0)
    assert metrics["coverage_macro_f1_0p3"] == pytest.approx(5.0 / 6.0)
    assert metrics["one_to_one_micro_precision_0p3"] == pytest.approx(11.0 / 12.0)
    assert metrics["one_to_one_micro_recall_0p3"] == pytest.approx(1.0)
    assert metrics["one_to_one_micro_f1_0p3"] == pytest.approx(22.0 / 23.0)
    assert metrics["one_to_one_macro_precision_0p3"] == pytest.approx(0.75)
    assert metrics["one_to_one_macro_recall_0p3"] == pytest.approx(1.0)
    assert metrics["one_to_one_macro_f1_0p3"] == pytest.approx(5.0 / 6.0)


def test_evaluate_reports_zero_precision_recall_for_empty_candidates() -> None:
    """含真实 occurrence 但没有候选时, semantic 与实例 P/R 都必须定义为 0.0."""

    evaluation = evaluate_centered_pdb(
        pdb_id="empty",
        centered={
            "selected": np.empty(0, dtype=np.bool_),
            "score": np.empty(0, dtype=np.float32),
            "source_blob_index": np.empty(0, dtype=np.int32),
            "voxel_offsets": np.zeros(1, dtype=np.int64),
            "voxel_index_local_zyx": np.empty((0, 3), dtype=np.int16),
            "box_start_zyx": np.empty((0, 3), dtype=np.int32),
        },
        occurrence_id=np.asarray([0], dtype=np.int32),
        occurrence_voxel_zyx=(np.asarray([[0, 0, 0]], dtype=np.int32),),
        full_shape_zyx=(1, 1, 1),
        coverage_thresholds=(0.3, 0.5, 0.6),
        topk_values=(3, 4, 5),
    )
    metrics = aggregate_stage1_metrics(
        (evaluation,),
        coverage_thresholds=(0.3, 0.5, 0.6),
        topk_values=(3, 4, 5),
    )

    assert metrics["semantic_micro_precision"] == 0.0
    assert metrics["semantic_micro_recall"] == 0.0
    assert metrics["semantic_macro_precision"] == 0.0
    assert metrics["semantic_macro_recall"] == 0.0
    assert metrics["coverage_micro_precision_0p3"] == 0.0
    assert metrics["coverage_micro_recall_0p3"] == 0.0
    assert metrics["coverage_macro_precision_0p3"] == 0.0
    assert metrics["coverage_macro_recall_0p3"] == 0.0
    assert metrics["one_to_one_micro_precision_0p3"] == 0.0
    assert metrics["one_to_one_micro_recall_0p3"] == 0.0
    assert metrics["one_to_one_macro_precision_0p3"] == 0.0
    assert metrics["one_to_one_macro_recall_0p3"] == 0.0


def test_candidate_prauc_scans_actual_scores_and_applies_only_size_gate() -> None:
    """候选 PRAUC 应忽略最终分数阈值, 并只按固定体素门槛掩码进入扫描."""

    first = evaluate_centered_pdb(
        pdb_id="first",
        centered={
            "selected": np.asarray([False, False]),
            "score": np.asarray([0.9, 0.8], dtype=np.float32),
            "source_blob_index": np.asarray([0, 1], dtype=np.int32),
            "voxel_offsets": np.asarray([0, 1, 2], dtype=np.int64),
            "voxel_index_local_zyx": np.asarray(
                [[0, 0, 2], [0, 0, 0]], dtype=np.int16
            ),
            "box_start_zyx": np.zeros((2, 3), dtype=np.int32),
            "prauc_eligible": np.asarray([True, True]),
        },
        occurrence_id=np.asarray([0], dtype=np.int32),
        occurrence_voxel_zyx=(np.asarray([[0, 0, 0]], dtype=np.int32),),
        full_shape_zyx=(1, 1, 3),
        coverage_thresholds=(0.3, 0.5, 0.6),
        topk_values=(3, 4, 5),
    )
    second = evaluate_centered_pdb(
        pdb_id="second",
        centered={
            "selected": np.asarray([False]),
            "score": np.asarray([0.9], dtype=np.float32),
            "source_blob_index": np.asarray([0], dtype=np.int32),
            "voxel_offsets": np.asarray([0, 1], dtype=np.int64),
            "voxel_index_local_zyx": np.asarray([[0, 0, 0]], dtype=np.int16),
            "box_start_zyx": np.zeros((1, 3), dtype=np.int32),
            "prauc_eligible": np.asarray([True]),
        },
        occurrence_id=np.asarray([0], dtype=np.int32),
        occurrence_voxel_zyx=(np.asarray([[0, 0, 0]], dtype=np.int32),),
        full_shape_zyx=(1, 1, 3),
        coverage_thresholds=(0.3, 0.5, 0.6),
        topk_values=(3, 4, 5),
    )
    metrics = aggregate_stage1_metrics(
        (first, second),
        coverage_thresholds=(0.3, 0.5, 0.6),
        topk_values=(3, 4, 5),
    )
    assert metrics["coverage_micro_prauc_0p3"] == pytest.approx(7.0 / 12.0)
    assert metrics["coverage_macro_prauc_0p3"] == pytest.approx(0.75)
    assert metrics["one_to_one_micro_prauc_0p3"] == pytest.approx(7.0 / 12.0)
    assert metrics["one_to_one_macro_prauc_0p3"] == pytest.approx(0.75)

    first.candidate_prauc_eligible[0] = False
    gated_metrics = aggregate_stage1_metrics(
        (first, second),
        coverage_thresholds=(0.3, 0.5, 0.6),
        topk_values=(3, 4, 5),
    )
    assert gated_metrics["coverage_micro_prauc_0p3"] == 1.0
    assert gated_metrics["coverage_macro_prauc_0p3"] == 1.0
    assert gated_metrics["one_to_one_micro_prauc_0p3"] == 1.0
    assert gated_metrics["one_to_one_macro_prauc_0p3"] == 1.0


@pytest.mark.parametrize(
    ("coverage_threshold", "voxel_count", "intersection_count"),
    (
        (0.3, 10, 3),
        (0.6, 5, 3),
    ),
)
def test_candidate_prauc_preserves_inclusive_decimal_boundary(
    coverage_threshold: float,
    voxel_count: int,
    intersection_count: int,
) -> None:
    """候选 PRAUC 必须与同名 F1 共用原始小数阈值的包含端点语义.

    构造中候选和 occurrence 均含 `voxel_count` 个体素, 两者交集恰为
    `intersection_count`. 两组参数分别形成 3/10=0.3 和 3/5=0.6;
    即使逐 PDB 事实把阈值归档为 float32, PRAUC 也不能提高比较边界.
    """

    # int16, (voxel_count, 3), 唯一候选在零基 BOX 局部 ZYX 坐标中占据 X 轴区间 [0, voxel_count), Z/Y 均为 0.
    candidate_zyx = np.stack(
        (
            np.zeros(voxel_count, dtype=np.int16),
            np.zeros(voxel_count, dtype=np.int16),
            np.arange(voxel_count, dtype=np.int16),
        ),
        axis=1,
    )
    # Python 整数, occurrence 的 X 轴起点; 使候选尾部恰有 intersection_count 个体素与其相交.
    occurrence_start = voxel_count - intersection_count
    # int32, (voxel_count, 3), 唯一 occurrence 在零基完整图 ZYX 坐标中占据 X 轴区间 [occurrence_start, occurrence_start + voxel_count); 本例 BOX 起点为 (0, 0, 0), 因而局部与完整图数值可直接比较.
    occurrence_zyx = np.stack(
        (
            np.zeros(voxel_count, dtype=np.int32),
            np.zeros(voxel_count, dtype=np.int32),
            np.arange(
                occurrence_start,
                occurrence_start + voxel_count,
                dtype=np.int32,
            ),
        ),
        axis=1,
    )
    # PdbEvaluation, 唯一候选已选且进入 PRAUC 扫描; 候选与 occurrence 的双向覆盖率都恰好等于 coverage_threshold.
    evaluation = evaluate_centered_pdb(
        pdb_id="boundary",
        centered={
            "selected": np.asarray([True]),
            "score": np.asarray([0.9], dtype=np.float32),
            "source_blob_index": np.asarray([0], dtype=np.int32),
            "voxel_offsets": np.asarray([0, voxel_count], dtype=np.int64),
            "voxel_index_local_zyx": candidate_zyx,
            "box_start_zyx": np.zeros((1, 3), dtype=np.int32),
            "prauc_eligible": np.asarray([True]),
        },
        occurrence_id=np.asarray([0], dtype=np.int32),
        occurrence_voxel_zyx=(occurrence_zyx,),
        full_shape_zyx=(1, 1, 2 * voxel_count - intersection_count),
        coverage_thresholds=(coverage_threshold,),
        topk_values=(1,),
    )
    # dict[str, object], 使用调用方原始 Python 浮点阈值汇总同一 PDB 的 F1 和 PRAUC.
    metrics = aggregate_stage1_metrics(
        (evaluation,),
        coverage_thresholds=(coverage_threshold,),
        topk_values=(1,),
    )
    # 字符串, 与正式 JSON 字段一致的小数阈值标签, 例如 0.3 对应 0p3.
    threshold_tag = str(coverage_threshold).replace(".", "p")

    assert metrics[f"coverage_micro_f1_{threshold_tag}"] == 1.0
    assert metrics[f"coverage_micro_prauc_{threshold_tag}"] == 1.0
    assert metrics[f"one_to_one_micro_f1_{threshold_tag}"] == 1.0
    assert metrics[f"one_to_one_micro_prauc_{threshold_tag}"] == 1.0


def test_semantic_prauc_keeps_voxel_micro_and_pdb_equal_macro_distinct() -> None:
    """语义 PRAUC 的 macro 必须让不同体素规模的 PDB 等权.

    large 含 100 个正体素和 100 个负体素, 排序完全正确.
    small 只含一对正负体素, 但排序完全相反. macro 因此是
    ``(1.0 + 0.5) / 2``, micro 则仍由 large 的 200 个体素主导.
    """

    # float32, (1, 1, 200), large 的正体素概率为 1, 负体素概率为 0.
    large_probability = np.concatenate(
        (
            np.ones(100, dtype=np.float32),
            np.zeros(100, dtype=np.float32),
        )
    ).reshape(1, 1, 200)
    # bool, (1, 1, 200), large 的前 100 个体素是 ligand 并集.
    large_target = np.concatenate(
        (
            np.ones(100, dtype=np.bool_),
            np.zeros(100, dtype=np.bool_),
        )
    ).reshape(1, 1, 200)
    # float32 与 bool, (1, 1, 2), small 的负体素概率高于正体素.
    small_probability = np.asarray([[[0.0, 1.0]]], dtype=np.float32)
    small_target = np.asarray([[[True, False]]], dtype=np.bool_)
    # 长度 2 的元组; 每项是一个 PDB 的 int64 (2, 1024) 正负体素计数.
    histograms = (
        semantic_prauc_histogram(large_probability, large_target),
        semantic_prauc_histogram(small_probability, small_target),
    )

    metrics = aggregate_semantic_prauc(histograms)

    assert metrics["semantic_macro_prauc"] == pytest.approx(0.75)
    assert metrics["semantic_micro_prauc"] == pytest.approx(
        0.5 / 101.0 + (100.0 / 101.0) ** 2
    )


def test_semantic_prauc_uses_training_float32_threshold_axis() -> None:
    """PRAUC 分箱必须复现训练指标在相邻 float32 上的端点语义."""

    # float32 标量, 下标 17 的内部训练阈值及其朝负无穷方向的相邻浮点数.
    training_threshold = torch.linspace(0.0, 1.0, 1024, dtype=torch.float32)[17]
    lower_probability = np.nextafter(
        np.float32(training_threshold.item()),
        np.float32(-np.inf),
    )
    # float32 与 bool, (1, 1, 2), 正体素恰好达到训练阈值, 负体素低 1 ULP.
    probability = np.asarray(
        [[[lower_probability, training_threshold.item()]]],
        dtype=np.float32,
    )
    target = np.asarray([[[False, True]]], dtype=np.bool_)
    metrics = aggregate_semantic_prauc((semantic_prauc_histogram(probability, target),))
    expected = BinaryAveragePrecision(thresholds=1024)(
        torch.from_numpy(probability.reshape(-1)),
        torch.from_numpy(target.reshape(-1)),
    )

    assert metrics["semantic_micro_prauc"] == float(expected)
    assert metrics["semantic_macro_prauc"] == float(expected)
    assert float(expected) == 1.0


def test_single_pdb_instance_metrics_keep_existing_contract() -> None:
    """单 PDB 的 coverage, Hungarian 与 top-K 事实保持原有定义."""

    centered = {
        "selected": np.asarray([True, True]),
        "score": np.asarray([0.9, 0.8], dtype=np.float32),
        "source_blob_index": np.asarray([4, 5], dtype=np.int32),
        "voxel_offsets": np.asarray([0, 2, 4], dtype=np.int64),
        "voxel_index_local_zyx": np.asarray(
            [[0, 0, 0], [0, 0, 1], [0, 0, 2], [0, 0, 3]],
            dtype=np.int16,
        ),
        "box_start_zyx": np.zeros((2, 3), dtype=np.int32),
    }
    evaluation = evaluate_centered_pdb(
        pdb_id="demo",
        centered=centered,
        occurrence_id=np.asarray([7], dtype=np.int32),
        occurrence_voxel_zyx=(
            np.asarray([[0, 0, 0], [0, 0, 1], [0, 0, 2], [0, 0, 3]], dtype=np.int32),
        ),
        full_shape_zyx=(1, 1, 4),
        coverage_thresholds=(0.3, 0.5, 0.6),
        topk_values=(3, 4, 5),
    )
    metrics = aggregate_stage1_metrics(
        (evaluation,),
        coverage_thresholds=(0.3, 0.5, 0.6),
        topk_values=(3, 4, 5),
    )
    assert evaluation.coverage_pred_hit.tolist() == [2, 2, 0]
    assert evaluation.coverage_gt_hit.tolist() == [1, 1, 0]
    assert evaluation.one_to_one_tp.tolist() == [1, 1, 0]
    assert metrics["semantic_micro_f1"] == 1.0
    assert metrics["semantic_macro_f1"] == 1.0
    assert metrics["top3_success_ratio_0p5"] == 1.0
    assert evaluation.one_to_one_match_offsets.tolist() == [0, 1, 2, 2]
    assert evaluation.topk_winning_candidate_rank[0, 1] == 0


def test_probability_science_archive_excludes_performance_fields(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """科学 NPZ 只保存概率与世界几何, 性能计时必须写到 status JSON."""

    result = FullMapResult(
        probability_map=np.full((80, 80, 80), 0.5, dtype=np.float32),
        origin_xyz=np.asarray([1.0, 2.0, 3.0], dtype=np.float32),
        voxel_size_xyz=np.ones(3, dtype=np.float32),
        window_count=1,
        wall_seconds=2.0,
        materialize_wait_seconds=0.2,
        fusion_wait_seconds=0.3,
    )
    paths = Stage1ArtifactPaths(tmp_path, "unet_c1", "validation", "demo")
    paths.complete("probability").parent.mkdir(parents=True, exist_ok=True)
    paths.complete("probability").write_text("stale", encoding="utf-8")

    class Session:
        """替代正式多 PDB session 并返回固定完整图结果."""

        def __init__(self, **_: object) -> None:
            """接受正式会话的全部关键字参数."""

        def __enter__(self) -> Session:
            """返回当前测试替身."""

            return self

        def __exit__(self, *_: object) -> None:
            """结束测试替身上下文."""

        def infer(
            self,
            pdb_id: str,
            next_pdb_id: str | None,
        ) -> FullMapResult:
            """确认旧完成标记已撤销, 且单 PDB 没有下一项."""

            assert pdb_id == "demo"
            assert next_pdb_id is None
            assert not paths.complete("probability").exists()
            return result

    monkeypatch.setattr("src.inference.pipeline.FullMapInferenceSession", Session)
    config = OmegaConf.create(
        {
            "publish_workers": 1,
            "pending_probability_pdbs": 1,
            "window": {
                "stride_zyx": [50, 50, 50],
                "gaussian_sigma": 0.5,
                "batch_size": 1,
                "workers": 1,
                "prefetch_batches": 1,
                "precision": "float32",
                "pending_fusion_batches": 1,
            },
        }
    )
    run_probability_stage(
        config=config,
        dataset=object(),
        collator=object(),
        wrapper=object(),
        device="cpu",
        producer="unet_c1",
        split="validation",
        pdb_ids=("demo",),
        output_root=tmp_path,
        overwrite=True,
    )
    archive = load_stage1_npz(paths.artifact("probability"), None)
    assert set(archive) == {"probability_map", "origin_xyz", "voxel_size_xyz"}
    selected_archive = load_stage1_npz(
        paths.artifact("probability"),
        ("origin_xyz",),
    )
    assert set(selected_archive) == {"origin_xyz"}
    assert (paths.pdb_root / "probability" / "geometry.json").is_file()
    assert (paths.pdb_root / "status" / "probability" / "performance.json").is_file()
    assert paths.complete("probability").is_file()
    marker = json.loads(paths.complete("probability").read_text(encoding="utf-8"))
    assert marker["output_role"] == "probability"
    assert marker["completed_at_utc"].endswith("+00:00")


def test_centered_selection_is_written_before_first_formal_completion(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """centered 阶段必须在首次压缩时同时写 score 与 selected."""

    paths = Stage1ArtifactPaths(tmp_path, "unet_c1", "validation", "demo")
    publish_stage1_artifact(
        paths.artifact("probability"),
        {
            "probability_map": np.zeros((80, 80, 80), dtype=np.float32),
            "origin_xyz": np.zeros(3, dtype=np.float32),
            "voxel_size_xyz": np.ones(3, dtype=np.float32),
        },
        paths.complete("probability"),
    )
    packed = Future()
    packed.set_result(
        {
            "source_probability_mean": np.asarray([0.8], dtype=np.float32),
            "source_voxel_count": np.asarray([9], dtype=np.int32),
            "voxel_offsets": np.asarray([0, 9], dtype=np.int64),
        }
    )
    captured = {}

    def infer_centered_boxes(**kwargs):
        """记录 centered 调用参数并返回固定打包 Future 与性能字段."""

        captured.update(kwargs)
        return packed, {
            "wall_seconds": 1.0,
            "materialize_wait_seconds": 0.1,
            "cpu_arrange_wait_seconds": 0.2,
            "batch_count": 1,
            "entry_count": 1,
        }

    monkeypatch.setattr(
        "src.inference.pipeline.infer_centered_boxes", infer_centered_boxes
    )
    publish_stage1_artifact(
        paths.artifact("F2_blobs"),
        {
            "blob_index": np.asarray([0], dtype=np.int32),
            "fits_centered_box": np.asarray([True]),
            "voxel_count": np.asarray([9], dtype=np.int32),
            "voxel_offsets": np.asarray([0, 9], dtype=np.int64),
            "voxel_index_global_zyx": np.zeros((9, 3), dtype=np.int32),
            "source_probability": np.full(9, 0.8, dtype=np.float32),
            "source_probability_mean": np.asarray([0.8], dtype=np.float32),
            "centered_box_start_zyx": np.zeros((1, 3), dtype=np.int32),
            "source_threshold_value": np.asarray([0.5], dtype=np.float32),
        },
        paths.complete("F2_blobs"),
    )
    config = OmegaConf.create(
        {
            "publish_workers": 2,
            "pending_centered_pdbs": 1,
            "centered": {
                "batch_size": 1,
                "workers": 1,
                "prefetch_batches": 1,
                "pending_cpu_batches": 1,
                "precision": "float32",
            },
        }
    )
    run_centered_stage(
        config=config,
        dataset=object(),
        collator=object(),
        wrapper=object(),
        device="cpu",
        producer="unet_c1",
        split="validation",
        pdb_ids=("demo",),
        output_root=tmp_path,
        alpha=2.0,
        forward_min_voxels=9,
        selection={
            "score_mode": "basic",
            "score_parameters": {},
            "score_threshold": 0.7,
            "prefiltered_min_voxel": 9,
            "min_voxels": 9,
        },
        overwrite=True,
        score_only=False,
        continue_on_blob_exceed=False,
    )
    arrays = load_stage1_npz(paths.artifact("F2_centered"), None)
    assert captured["forward_min_voxels"] == 9
    np.testing.assert_array_equal(captured["full_probability"], 0.0)
    assert arrays["selected"].tolist() == [True]
    assert paths.complete("F2_centered").is_file()


def test_cli_builds_current_dataset_without_hydra_dataclass_conversion(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """CLI 必须把真正的 ResolvedStage1Crop 序列交给当前 Dataset."""

    checkpoint = tmp_path / "model.ckpt"
    checkpoint.write_bytes(b"checkpoint")
    resolved = tmp_path / "resolved.yaml"
    resolved.write_text("model: test\n", encoding="utf-8")
    config = tmp_path / "inference.yaml"
    config.write_text("device: cpu\nalpha: 2.0\n", encoding="utf-8")
    pdb_list = tmp_path / "pdb.json"
    pdb_list.write_text('{"pdb_ids": ["demo"]}\n', encoding="utf-8")
    training_config = OmegaConf.create(
        {
            "dataset": {
                "_target_": "retired.Dataset",
                "all_data_path": str(tmp_path),
                "split_file": "retired.txt",
                "mode": "train",
                "stage1_model_name": "unet_c1",
                "box_pool_root": "retired",
                "density_channel_config": {
                    "clip_percentile": [1.0, 99.0],
                    "fit_mask_percentile": 70.0,
                    "enabled_channels": ["exp_clipnorm_nopost"],
                },
            }
        }
    )

    class Wrapper:
        """记录 CLI 传入的模型设备."""

        def to(self, device):
            """确认当前测试使用 CPU 设备."""

            assert str(device) == "cpu"

    captured = {}

    class Dataset:
        """记录 CLI 直接传给当前 Stage1Dataset 的构造参数."""

        root = tmp_path
        collate_fn = staticmethod(lambda rows: rows)

        def __init__(self, **kwargs):
            """保存 Dataset 构造参数供断言使用."""

            captured.update(kwargs)

    monkeypatch.setattr(
        cli_module, "load_stage1_wrapper", lambda **_: (Wrapper(), training_config)
    )
    monkeypatch.setattr("src.datasets.stage1_dataset.Stage1Dataset", Dataset)
    monkeypatch.setattr(
        cli_module,
        "run_probability_stage",
        lambda *args: captured.update(stage=args),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "stage1",
            "probability",
            "--config",
            str(config),
            "--checkpoint",
            str(checkpoint),
            "--resolved-config",
            str(resolved),
            "--pdb-json",
            str(pdb_list),
            "--output-root",
            str(tmp_path / "output"),
            "--producer",
            "unet_c1",
            "--model-code-source",
            "current_workspace",
            "--split",
            "calibration",
        ],
    )
    cli_module.main()
    assert captured["mode"] == "full_map"
    assert captured["split_file"][0].pdb_id == "demo"
    assert captured["split_file"][0].__class__.__name__ == "ResolvedStage1Crop"


def test_cli_rejects_duplicate_pdb_before_model_loading(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """同一 PDB 不能在校准事实与最终指标中采用两种重复口径."""

    config = tmp_path / "inference.yaml"
    config.write_text("device: cpu\nalpha: 2.0\n", encoding="utf-8")
    pdb_list = tmp_path / "pdb.json"
    pdb_list.write_text('["demo", "DEMO"]\n', encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "stage1",
            "probability",
            "--config",
            str(config),
            "--checkpoint",
            str(tmp_path / "unused.ckpt"),
            "--resolved-config",
            str(tmp_path / "unused.yaml"),
            "--pdb-json",
            str(pdb_list),
            "--output-root",
            str(tmp_path / "output"),
            "--producer",
            "unet_c1",
            "--model-code-source",
            "current_workspace",
            "--split",
            "calibration",
        ],
    )
    with pytest.raises(ValueError, match="唯一"):
        cli_module.main()


def test_cli_uses_fixed_random_sharding_for_production_stage(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """相同 JSON, seed, 分片数和编号必须得到相同 PDB 子序列."""

    config = tmp_path / "inference.yaml"
    config.write_text("alpha: 2.0\n", encoding="utf-8")
    pdb_json = tmp_path / "pdb.json"
    source = [f"pdb{index}" for index in range(10)]
    pdb_json.write_text(json.dumps(source), encoding="utf-8")
    captured = {}
    monkeypatch.setattr(
        cli_module,
        "run_blobs_stage",
        lambda *args: captured.update(pdb_ids=args[4]),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "stage1",
            "blobs",
            "--config",
            str(config),
            "--producer",
            "unet_base",
            "--pdb-json",
            str(pdb_json),
            "--split",
            "train",
            "--output-root",
            str(tmp_path / "output"),
            "--semantic-threshold",
            "0.5",
            "--shard-count",
            "3",
            "--shard-index",
            "1",
        ],
    )
    cli_module.main()
    expected = list(source)
    random.Random(3407).shuffle(expected)
    assert captured["pdb_ids"] == tuple(expected[1::3])


def test_cli_passes_explicit_all_candidate_evaluation_name(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """evaluate 必须显式接收结果名和全候选范围, 不隐式构造选择参数."""

    config = tmp_path / "inference.yaml"
    config.write_text("alpha: 2.0\n", encoding="utf-8")
    pdb_json = tmp_path / "pdb.json"
    pdb_json.write_text('["demo"]\n', encoding="utf-8")
    captured: dict[str, tuple[object, ...]] = {}

    def capture_evaluate_stage(*arguments: object) -> None:
        """记录 CLI 交给正式评估阶段的位置参数."""

        captured["arguments"] = arguments

    monkeypatch.setattr(
        cli_module,
        "run_evaluate_stage",
        capture_evaluate_stage,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "stage1",
            "evaluate",
            "--config",
            str(config),
            "--producer",
            "unet_c1",
            "--pdb-json",
            str(pdb_json),
            "--split",
            "validation",
            "--output-root",
            str(tmp_path / "output"),
            "--data-root",
            str(tmp_path / "data"),
            "--artifact",
            "blobs",
            "--evaluation-name",
            "raw_blobs",
            "--all-candidates",
        ],
    )
    cli_module.main()
    assert captured["arguments"][8] == "raw_blobs"
    assert captured["arguments"][9] is None


def test_cli_passes_blob_exceed_advisory_flag(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """centered 的显式提示模式开关必须原样传给正式阶段."""

    config = tmp_path / "inference.yaml"
    config.write_text("device: cpu\nalpha: 2.0\n", encoding="utf-8")
    pdb_json = tmp_path / "pdb.json"
    pdb_json.write_text('["demo"]\n', encoding="utf-8")
    selection = tmp_path / "selection.json"
    selection.write_text(
        json.dumps(
            {
                "score_mode": "basic",
                "score_parameters": {},
                "score_threshold": 0.5,
                "prefiltered_min_voxel": 1,
                "min_voxels": 1,
            }
        ),
        encoding="utf-8",
    )
    captured: dict[str, tuple[object, ...]] = {}
    monkeypatch.setattr(
        cli_module,
        "run_centered_stage",
        lambda *arguments: captured.update(arguments=arguments),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "stage1",
            "centered",
            "--config",
            str(config),
            "--producer",
            "unet_c1",
            "--pdb-json",
            str(pdb_json),
            "--split",
            "validation",
            "--output-root",
            str(tmp_path / "output"),
            "--selection-parameters",
            str(selection),
            "--score-only",
            "--continue-on-blob-exceed",
        ],
    )
    cli_module.main()
    assert captured["arguments"][-1] is True


def test_f_alpha_tag_uses_readable_decimal_path_names() -> None:
    """整数不保留小数点, 小数点改成 p, 相邻 Python float 不得碰撞."""

    assert f_alpha_tag(2.0) == "F2"
    assert f_alpha_tag(0.5) == "F0p5"
    assert f_alpha_tag(1.5) == "F1p5"
    assert f_alpha_tag(1.0000001) != f_alpha_tag(1.0000002)
    assert f_alpha_tag(0.33333331) != f_alpha_tag(0.33333332)


def test_evaluate_keeps_raw_and_filtered_results_side_by_side(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """全候选与过滤结果必须并存, 完整图 PRAUC 不随候选选择改变."""

    paths = Stage1ArtifactPaths(tmp_path, "unet_c1", "validation", "demo")
    publish_stage1_artifact(
        paths.artifact("F2_blobs"),
        {
            "blob_index": np.asarray([0, 1], dtype=np.int32),
            "source_probability_mean": np.asarray([0.9, 0.4], dtype=np.float32),
            "voxel_count": np.asarray([1, 1], dtype=np.int32),
            "voxel_offsets": np.asarray([0, 1, 2], dtype=np.int64),
            "voxel_index_global_zyx": np.asarray(
                [[0, 0, 0], [1, 1, 1]], dtype=np.int32
            ),
        },
        None,
    )
    publish_stage1_artifact(
        paths.artifact("probability"),
        {
            "probability_map": np.asarray(
                [
                    [[0.9, 0.1], [0.1, 0.1]],
                    [[0.1, 0.1], [0.1, 0.4]],
                ],
                dtype=np.float32,
            )
        },
        paths.complete("probability"),
    )
    ligand_area = tmp_path / "density" / "demo" / "ligand_area.npz"
    ligand_area.parent.mkdir(parents=True)
    np.savez_compressed(
        ligand_area,
        grid_shape_zyx=np.asarray([2, 2, 2], dtype=np.int32),
        mask_7=np.asarray([[0, 0, 0]], dtype=np.int32),
    )
    # bool, (1, 2, 2, 2), 首轴是数据文件保留的通道轴, 后三轴是完整图 ZYX.
    union_mask = np.zeros((1, 2, 2, 2), dtype=np.bool_)
    union_mask[0, 0, 0, 0] = True
    np.save(ligand_area.parent / "union_mask.npy", union_mask)
    config = OmegaConf.create(
        {"evaluation": {"coverage_thresholds": [0.5], "topk_values": [1]}}
    )
    real_scorer = score_centered_candidates

    def reject_scoring(*_args: object, **_kwargs: object) -> None:
        """证明全候选分支不会进入 basic 或 Gaussian 打分."""

        raise AssertionError("all-candidates must not call the scorer")

    monkeypatch.setattr(
        "src.inference.pipeline.score_centered_candidates", reject_scoring
    )
    # raw_metrics 保存同一完整图下全部 blobs 的候选指标与 PRAUC.
    raw_metrics = run_evaluate_stage(
        config,
        tmp_path,
        "unet_c1",
        "validation",
        ("demo",),
        tmp_path,
        2.0,
        "blobs",
        "raw_blobs",
        None,
    )
    monkeypatch.setattr("src.inference.pipeline.score_centered_candidates", real_scorer)
    # filtered_metrics 使用严格候选选择, 但完整图 PRAUC 应与 raw_metrics 相同.
    filtered_metrics = run_evaluate_stage(
        config,
        tmp_path,
        "unet_c1",
        "validation",
        ("demo",),
        tmp_path,
        2.0,
        "blobs",
        "basic_strict",
        {
            "score_mode": "basic",
            "score_parameters": {},
            "score_threshold": 0.8,
            "prefiltered_min_voxel": 1,
            "min_voxels": 1,
        },
    )

    raw = load_stage1_npz(paths.pdb_root / "evaluation" / "raw_blobs.npz", None)
    filtered = load_stage1_npz(paths.pdb_root / "evaluation" / "basic_strict.npz", None)
    assert raw["candidate_selected"].tolist() == [True, True]
    assert filtered["candidate_selected"].tolist() == [True, False]
    assert "candidate_prauc_eligible" not in raw
    assert "candidate_prauc_eligible" not in filtered
    assert raw_metrics["semantic_micro_prauc"] == 1.0
    assert raw_metrics["semantic_macro_prauc"] == 1.0
    assert filtered_metrics["semantic_micro_prauc"] == 1.0
    assert filtered_metrics["semantic_macro_prauc"] == 1.0
    evaluation_root = tmp_path / "unet_c1" / "validation" / "evaluation"
    assert (evaluation_root / "raw_blobs.metrics.json").is_file()
    assert (evaluation_root / "basic_strict.metrics.json").is_file()
    precision_recall_fields = {
        "semantic_micro_precision",
        "semantic_micro_recall",
        "semantic_macro_precision",
        "semantic_macro_recall",
        "coverage_micro_precision_0p5",
        "coverage_micro_recall_0p5",
        "coverage_macro_precision_0p5",
        "coverage_macro_recall_0p5",
        "one_to_one_micro_precision_0p5",
        "one_to_one_micro_recall_0p5",
        "one_to_one_macro_precision_0p5",
        "one_to_one_macro_recall_0p5",
    }
    for evaluation_name in ("raw_blobs", "basic_strict"):
        metrics_payload = json.loads(
            (evaluation_root / f"{evaluation_name}.metrics.json").read_text(
                encoding="utf-8"
            )
        )
        jsonl_payload = json.loads(
            (evaluation_root / f"{evaluation_name}.jsonl").read_text(
                encoding="utf-8"
            )
        )
        assert precision_recall_fields <= metrics_payload.keys()
        assert precision_recall_fields <= jsonl_payload.keys()


def _run_empty_blob_limit_case(
    tmp_path: Path,
    source_blob_count: int,
    continue_on_blob_exceed: bool,
) -> tuple[Stage1ArtifactPaths, object, Path]:
    """建立空候选输入并执行一次 Find centered 超量边界案例."""

    # paths 指向本案例唯一 PDB 的 probability, blobs, centered 和状态目录.
    paths = Stage1ArtifactPaths(tmp_path, "Find_0", "train", "demo")
    density_root = tmp_path / "density" / "demo"
    density_root.mkdir(parents=True)
    # float32, (1, 80, 80, 80), 空密度只为零候选 centered 物化提供正式输入文件.
    density = np.zeros((1, 80, 80, 80), dtype=np.float32)
    np.save(density_root / "exp.npy", density)
    np.save(density_root / "sim.npy", density)
    publish_stage1_artifact(
        paths.artifact("F2_blobs"),
        {
            "blob_index": np.arange(source_blob_count, dtype=np.int32),
            "voxel_offsets": np.zeros(source_blob_count + 1, dtype=np.int64),
            "voxel_index_global_zyx": np.empty((0, 3), dtype=np.int32),
            "source_probability": np.empty(0, dtype=np.float32),
            "source_probability_mean": np.zeros(source_blob_count, dtype=np.float32),
            "voxel_count": np.zeros(source_blob_count, dtype=np.int32),
            "fits_centered_box": np.zeros(source_blob_count, dtype=np.bool_),
            "centered_box_start_zyx": np.full(
                (source_blob_count, 3), -1, dtype=np.int32
            ),
            "source_threshold_value": np.asarray([0.5], dtype=np.float32),
        },
        paths.complete("F2_blobs"),
    )
    publish_stage1_artifact(
        paths.artifact("probability"),
        {
            "probability_map": np.zeros((80, 80, 80), dtype=np.float32),
            "origin_xyz": np.zeros(3, dtype=np.float32),
            "voxel_size_xyz": np.ones(3, dtype=np.float32),
        },
        paths.complete("probability"),
    )
    config = OmegaConf.create(
        {
            "publish_workers": 1,
            "pending_centered_pdbs": 1,
            "centered": {
                "precision": "float32",
                "batch_size": 1,
                "workers": 1,
                "prefetch_batches": 1,
                "pending_cpu_batches": 1,
            },
            "calibration": {
                "workers": 1,
                "min_voxel_values": [1],
                "gaussian_grid": {
                    "tau_angstrom": [1.0],
                    "lambda_positive": [0.0],
                    "lambda_negative": [0.0],
                },
                "gaussian_refinement": {
                    "lambda": [1.0],
                },
            },
            "evaluation": {
                "coverage_thresholds": [0.3, 0.5, 0.6],
                "topk_values": [3, 4, 5],
            },
        }
    )
    run_centered_stage(
        config,
        SimpleNamespace(root=tmp_path),
        None,
        None,
        "cpu",
        "Find_0",
        "train",
        ("demo",),
        tmp_path,
        2.0,
        1,
        None,
        False,
        False,
        continue_on_blob_exceed,
    )
    return paths, config, density_root


@pytest.mark.parametrize(
    (
        "source_blob_count",
        "continue_on_blob_exceed",
        "expect_exceed",
        "expect_centered",
    ),
    (
        (CENTERED_BLOB_LIMIT, False, False, True),
        (CENTERED_BLOB_LIMIT + 1, False, True, False),
        (CENTERED_BLOB_LIMIT + 1, True, True, True),
    ),
)
def test_centered_blob_limit_uses_strict_greater_than(
    tmp_path: Path,
    source_blob_count: int,
    continue_on_blob_exceed: bool,
    expect_exceed: bool,
    expect_centered: bool,
) -> None:
    """1000 为正常上限, 1001 总写标记并服从显式继续开关."""

    paths, _, _ = _run_empty_blob_limit_case(
        tmp_path,
        source_blob_count,
        continue_on_blob_exceed,
    )
    if expect_exceed:
        marker = json.loads(
            paths.blob_exceed("F2_centered").read_text(encoding="utf-8")
        )
        assert marker["source_blob_count"] == source_blob_count
        assert marker["limit"] == CENTERED_BLOB_LIMIT
    else:
        assert not paths.blob_exceed("F2_centered").exists()
    if expect_centered:
        assert paths.artifact("F2_centered").is_file()
        assert paths.complete("F2_centered").is_file()
    else:
        assert not paths.artifact("F2_centered").exists()


def test_blob_exceed_advisory_mode_completes_tune_and_evaluate(
    tmp_path: Path,
) -> None:
    """超量提示模式生成的 centered 必须被 tune 与 evaluate 正常消费."""

    paths, config, density_root = _run_empty_blob_limit_case(
        tmp_path,
        CENTERED_BLOB_LIMIT + 1,
        True,
    )
    np.savez(
        density_root / "ligand_area.npz",
        grid_shape_zyx=np.asarray([80, 80, 80], dtype=np.int32),
        mask_1=np.asarray([[0, 0, 0]], dtype=np.int32),
    )
    # bool, (1, 80, 80, 80), 唯一正体素与 ligand_area.npz 的 mask_1 对齐.
    union_mask = np.zeros((1, 80, 80, 80), dtype=np.bool_)
    union_mask[0, 0, 0, 0] = True
    np.save(density_root / "union_mask.npy", union_mask)
    selection = run_tune_stage(
        config=config,
        data_root=tmp_path,
        producer="Find_0",
        split="train",
        pdb_ids=("demo",),
        output_root=tmp_path,
        alpha=2.0,
        score_mode="gaussian",
        objective_beta=2.0,
        prefiltered_min_voxel=1,
    )
    metrics = run_evaluate_stage(
        config=config,
        data_root=tmp_path,
        producer="Find_0",
        split="train",
        pdb_ids=("demo",),
        output_root=tmp_path,
        alpha=2.0,
        artifact="centered",
        evaluation_name="blob_exceed_continue",
        selection=selection,
    )

    assert paths.blob_exceed("F2_centered").is_file()
    assert paths.complete("F2_centered").is_file()
    assert metrics["pdb_count"] == 1
    assert (tmp_path / "Find_0" / "tuning" / "F2_gaussian.json").is_file()
    assert (
        tmp_path
        / "Find_0"
        / "train"
        / "evaluation"
        / "blob_exceed_continue.metrics.json"
    ).is_file()


def test_centered_score_only_changes_two_fields(tmp_path: Path) -> None:
    """score-only 必须保留候选轴与任意其他正式数组的逐元素内容."""

    paths = Stage1ArtifactPaths(tmp_path, "unet_base", "validation", "demo")
    publish_stage1_artifact(
        paths.artifact("F2_centered"),
        {
            "source_probability_mean": np.asarray([0.8, 0.4], dtype=np.float32),
            "source_voxel_count": np.asarray([2, 1], dtype=np.int32),
            "voxel_offsets": np.asarray([0, 2, 3], dtype=np.int64),
            "source_blob_index": np.asarray([3, 7], dtype=np.int32),
            "voxel_final": np.arange(6, dtype=np.float16).reshape(3, 2),
        },
        paths.complete("F2_centered"),
    )
    config = OmegaConf.create({})
    run_centered_stage(
        config,
        None,
        None,
        None,
        "cpu",
        "unet_base",
        "validation",
        ("demo",),
        tmp_path,
        2.0,
        None,
        {
            "score_mode": "basic",
            "score_parameters": {},
            "score_threshold": 0.5,
            "prefiltered_min_voxel": 2,
            "min_voxels": 1,
        },
        False,
        True,
        False,
    )
    arrays = load_stage1_npz(paths.artifact("F2_centered"), None)
    assert arrays["score"].tolist() == pytest.approx([0.8, 0.4])
    assert arrays["selected"].tolist() == [True, False]
    np.testing.assert_array_equal(
        arrays["voxel_final"], np.arange(6, dtype=np.float16).reshape(3, 2)
    )


def test_tune_prefilter_is_fixed_before_basic_parameter_search() -> None:
    """预过滤必须先固定小候选为未入选, 且不限制后续 min_voxels 搜索值."""

    centered = {
        "source_blob_index": np.asarray([0, 1], dtype=np.int32),
        "source_probability_mean": np.asarray([0.9, 0.8], dtype=np.float32),
        "source_voxel_count": np.asarray([1, 2], dtype=np.int32),
        "voxel_offsets": np.asarray([0, 1, 3], dtype=np.int64),
        "voxel_index_local_zyx": np.asarray(
            [[2, 2, 2], [0, 0, 0], [0, 0, 1]], dtype=np.int16
        ),
        "box_start_zyx": np.zeros((2, 3), dtype=np.int32),
    }
    best = tune_centered_selection(
        centered_items=(("demo", centered),),
        ground_truth_by_pdb={
            "demo": (
                np.asarray([7], dtype=np.int32),
                (np.asarray([[0, 0, 0], [0, 0, 1]], dtype=np.int32),),
                (3, 3, 3),
            )
        },
        score_mode="basic",
        score_parameter_grid=None,
        refinement_multipliers=None,
        prefiltered_min_voxel=2,
        min_voxel_values=[1],
        objective_beta=2.0,
        coverage_thresholds=[0.3, 0.5, 0.6],
        topk_values=[3, 4, 5],
        workers=1,
    )
    assert best["prefiltered_min_voxel"] == 2
    assert best["min_voxels"] == 1
    assert best["score_threshold"] == pytest.approx(0.8)
    assert best["objective"] == pytest.approx(3.0)


def test_find_calibration_uses_coarse_refined_then_minimum_stages() -> None:
    """Find 必须先固定预过滤, 再搜索 Gaussian 参数与独立 min_voxels."""

    centered = {
        "source_blob_index": np.asarray([0], dtype=np.int32),
        "source_probability_mean": np.asarray([0.5], dtype=np.float32),
        "source_voxel_count": np.asarray([1], dtype=np.int32),
        "selected": np.asarray([False]),
        "score": np.asarray([0.0], dtype=np.float32),
        "voxel_offsets": np.asarray([0, 1], dtype=np.int64),
        "voxel_index_local_zyx": np.asarray([[0, 0, 0]], dtype=np.int16),
        "box_start_zyx": np.zeros((1, 3), dtype=np.int32),
        "voxel_size_world": np.ones((1, 3), dtype=np.float32),
        "A_offsets": np.asarray([0, 1], dtype=np.int64),
        "A_coord_local_xyz": np.asarray([[0.5, 0.5, 0.5]], dtype=np.float32),
        "A_probability": np.asarray([1.0], dtype=np.float32),
    }
    best = tune_centered_selection(
        centered_items=(("demo", centered),),
        ground_truth_by_pdb={
            "demo": (
                np.asarray([7], dtype=np.int32),
                (np.asarray([[0, 0, 0]], dtype=np.int32),),
                (1, 1, 1),
            )
        },
        score_mode="gaussian",
        score_parameter_grid={
            "tau_angstrom": [1.0],
            "lambda_positive": [0.2],
            "lambda_negative": [0.1],
        },
        refinement_multipliers={"lambda": [1.0]},
        prefiltered_min_voxel=2,
        min_voxel_values=[1, 2],
        objective_beta=2.0,
        coverage_thresholds=[0.3, 0.5, 0.6],
        topk_values=[3, 4, 5],
        workers=1,
    )
    assert tuple(best["stages"]) == ("coarse", "refined", "min_voxels")
    assert best["prefiltered_min_voxel"] == 2
    assert best["min_voxels"] == 1
    assert best["objective"] == pytest.approx(0.0)
    assert best["score_parameters"]["tau_angstrom"] == 1.0
