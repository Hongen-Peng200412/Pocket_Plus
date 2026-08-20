# -*- coding: utf-8 -*-
"""Stage1 V3 几何, 产物, 评分和评估契约测试."""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
import json
from pathlib import Path
import sys

import numpy as np
from omegaconf import OmegaConf
import pytest
import torch

import src.inference.cli as cli_module
from src.inference.blobs import extract_probability_blobs
from src.inference.artifacts import (
    Stage1ArtifactPaths,
    load_stage1_npz,
    publish_stage1_artifact,
)
from src.inference.calibration import (
    calibrate_semantic_thresholds,
    tune_centered_selection,
)
from src.inference.centered import infer_centered_boxes, pack_centered_entries
from src.inference.evaluation import (
    aggregate_stage1_metrics,
    evaluate_centered_pdb,
)
from src.inference.full_map import (
    FullMapResult,
    gaussian_window_weight,
    window_starts_zyx,
)
from src.inference.pipeline import produce_centered_role, produce_probability_map
from src.inference.scoring import (
    score_centered_candidates,
    sum_gaussian_atom_terms,
)
from src.inference.workflow import run_frozen_workflow


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
                "box_start_zyx": np.asarray((0, 0, 0), dtype=np.int32),
                "box_shape_zyx": np.asarray((80, 80, 80), dtype=np.uint8),
                "box_origin_world": np.zeros(3, dtype=np.float32),
                "voxel_size_world": np.ones(3, dtype=np.float32),
                "source_probability_mean": np.asarray(0.8, dtype=np.float32),
                "source_threshold_value": np.asarray(0.5, dtype=np.float32),
                "score": np.asarray(0.8, dtype=np.float32),
                "selected": np.asarray(False, dtype=np.bool_),
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
    arrays = pack_centered_entries(entries, True, True, True, True)
    assert arrays["voxel_offsets"].tolist() == [0, 2, 3]
    assert arrays["A_offsets"].tolist() == [0, 1, 2]
    assert arrays["A_feat_L0"].shape == (2, 50)
    assert arrays["A_feat_L0"].dtype == np.float32
    assert arrays["voxel_final"].shape == (3, 4)
    assert arrays["centered_box_index"].tolist() == [0, 1]


def test_centered_cpu_arranger_accepts_bfloat16_output(tmp_path: Path) -> None:
    """H100 bf16 输出复制到 CPU 后必须先转 float32 再进入 NumPy."""

    class Dataset:
        root = tmp_path

        def materialize_request(self, request):
            return {"hardmask": torch.zeros((80, 80, 80), dtype=torch.bool)}

    class Wrapper:
        def __call__(self, batch):
            shape = (len(batch["hardmask"]), 1, 80, 80, 80)
            return {
                "voxel_logits_ligand": torch.zeros(shape, dtype=torch.bfloat16),
                "voxel_logits_aux": torch.zeros(shape, dtype=torch.bfloat16),
            }

    def collator(rows):
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
            full_probability=np.zeros((80, 80, 80), dtype=np.float32),
            origin_xyz=np.zeros(3, dtype=np.float32),
            voxel_size_xyz=np.ones(3, dtype=np.float32),
            device="cpu",
            precision="bf16",
            centered_batch_size=1,
            centered_workers=1,
            prefetch_batches=1,
            pending_cpu_batches=1,
            min_voxels=1,
            centered_forward="full",
            save_voxel_final=False,
            save_dense48=False,
            packer=packer,
        )
        arrays = packed.result()
    assert arrays["centered_probability"].dtype == np.float32
    assert arrays["voxel_aux_probability"].dtype == np.float32


def test_find_gaussian_score_uses_five_angstrom_cutoff() -> None:
    """5 Å 内 A 原子贡献正项, 5 Å 外原子不参与 Gaussian 分数."""

    centered = {
        "source_probability_mean": np.asarray([0.5], dtype=np.float32),
        "voxel_offsets": np.asarray([0, 1], dtype=np.int64),
        "voxel_index_local_zyx": np.asarray([[0, 0, 0]], dtype=np.int16),
        "voxel_size_world": np.asarray([[1.0, 1.0, 1.0]], dtype=np.float32),
        "A_offsets": np.asarray([0, 2], dtype=np.int64),
        "A_coord_local_xyz": np.asarray(
            [[0.5, 0.5, 0.5], [8.0, 8.0, 8.0]], dtype=np.float32
        ),
        "A_probability": np.asarray([1.0, 1.0], dtype=np.float32),
    }
    score = score_centered_candidates(
        centered,
        score_mode="find_gaussian",
        score_parameters={
            "tau_angstrom": 1.0,
            "lambda_positive": 0.2,
            "lambda_negative": 0.1,
        },
    )
    assert np.allclose(score, [0.7])


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
        score_mode="find_gaussian",
        score_parameters={
            "tau_angstrom": 0.75,
            "lambda_positive": 0.064,
            "lambda_negative": 0.004,
        },
    )
    np.testing.assert_array_equal(actual, expected)


def test_semantic_and_instance_metrics_follow_micro_contract() -> None:
    """阈值扫描,coverage 和 Hungarian 均使用固定的全局计数定义."""

    semantic = calibrate_semantic_thresholds(
        [
            (
                np.asarray([0.9, 0.8, 0.2, 0.1], dtype=np.float32),
                np.asarray([True, True, False, False]),
            )
        ],
        denominator=10,
        betas=(1.0, 3.0),
    )
    assert semantic["thresholds"]["F1"]["grid_index"] == 3
    assert semantic["scan"]["tp"].shape == (11,)
    assert semantic["scan"]["f_beta_curve"].shape == (2, 11)

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

    def infer_full_map(**_):
        assert not paths.complete("probability").exists()
        return result

    monkeypatch.setattr("src.inference.pipeline.infer_full_map", infer_full_map)
    with ThreadPoolExecutor(max_workers=1) as publisher:
        arrays, future = produce_probability_map(
            paths=paths,
            dataset=object(),
            collator=object(),
            wrapper=object(),
            device="cpu",
            window_config={
                "stride_zyx": [50, 50, 50],
                "gaussian_sigma": 0.5,
                "batch_size": 1,
                "workers": 1,
                "prefetch_batches": 1,
                "precision": "float32",
                "pending_fusion_batches": 1,
            },
            publisher=publisher,
        )
        assert set(arrays) == {"probability_map", "origin_xyz", "voxel_size_xyz"}
        future.result()
    archive = load_stage1_npz(paths.artifact("probability"), tuple(arrays))
    assert set(archive) == set(arrays)
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
    """冻结 workflow 必须在 centered 首次压缩时同时写 score 与 selected."""

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
            "voxel_offsets": np.asarray([0, 9], dtype=np.int64),
            "score": np.zeros(1, dtype=np.float32),
            "selected": np.zeros(1, dtype=np.bool_),
        }
    )
    captured = {}

    def infer_centered_boxes(**kwargs):
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
    blobs = {
        "fits_centered_box": np.asarray([True]),
        "voxel_count": np.asarray([9], dtype=np.int32),
    }
    with ThreadPoolExecutor(max_workers=1) as publisher:
        future = produce_centered_role(
            paths=paths,
            dataset=object(),
            collator=object(),
            wrapper=object(),
            device="cpu",
            blobs=blobs,
            centered_role="F1_basic",
            min_voxels=9,
            centered_config={
                "precision": "float32",
                "batch_size": 1,
                "workers": 1,
                "prefetch_batches": 1,
                "pending_cpu_batches": 1,
                "forward": "voxel_only",
                "save_voxel_final": False,
                "save_dense48": False,
            },
            blob_limit=None,
            selection={
                "score_mode": "source_mean",
                "score_parameters": {},
                "score_threshold": 0.7,
                "min_voxels": 9,
            },
            publisher=publisher,
        )
        arrays = future.result()
    assert captured["min_voxels"] == 9
    assert captured["full_probability"] is None
    assert arrays["selected"].tolist() == [True]
    assert paths.complete("F1_basic").is_file()


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
    config.write_text(
        "device: cpu\ncalibration:\n  semantic_denominator: 32\n", encoding="utf-8"
    )
    pdb_list = tmp_path / "pdb.txt"
    pdb_list.write_text("demo\n", encoding="utf-8")
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
        def to(self, device):
            assert str(device) == "cpu"

    captured = {}

    class Dataset:
        root = tmp_path
        collate_fn = staticmethod(lambda rows: rows)

        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(
        cli_module, "load_stage1_wrapper", lambda **_: (Wrapper(), training_config)
    )
    monkeypatch.setattr("src.datasets.stage1_dataset.Stage1Dataset", Dataset)
    monkeypatch.setattr(
        cli_module,
        "run_calibration_workflow",
        lambda **kwargs: captured.update(workflow=kwargs),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "stage1",
            "calibrate",
            "--config",
            str(config),
            "--checkpoint",
            str(checkpoint),
            "--resolved-config",
            str(resolved),
            "--pdb-list",
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
    config.write_text("device: cpu\n", encoding="utf-8")
    pdb_list = tmp_path / "pdb.txt"
    pdb_list.write_text("demo\nDEMO\n", encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "stage1",
            "calibrate",
            "--config",
            str(config),
            "--checkpoint",
            str(tmp_path / "unused.ckpt"),
            "--resolved-config",
            str(tmp_path / "unused.yaml"),
            "--pdb-list",
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
    with pytest.raises(ValueError, match="重复"):
        cli_module.main()


def test_frozen_workflow_requires_fitted_calibration_marker() -> None:
    """冻结运行不能只凭 stage1_v3.json 绕过 calibration 完成标记."""

    checkpoint_path = "/models/stage1.ckpt"
    with pytest.raises(ValueError, match="calibration_fitted"):
        run_frozen_workflow(
            config=object(),
            calibration_payload={"checkpoint_path": checkpoint_path},
            calibration_complete={
                "checkpoint_path": checkpoint_path,
                "result_scope": "partial",
            },
            dataset=object(),
            collator=object(),
            wrapper=object(),
            device="cpu",
            producer="unet_c1",
            split="validation",
            pdb_ids=("demo",),
            output_root=Path("unused"),
            checkpoint_path=checkpoint_path,
        )


def test_find_calibration_uses_coarse_refined_then_minimum_stages() -> None:
    """Find 完整模式必须先冻结两轮 Gaussian 参数, 再单独选择 min_voxels."""

    centered = {
        "source_blob_index": np.asarray([0], dtype=np.int32),
        "source_probability_mean": np.asarray([0.5], dtype=np.float32),
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
        score_mode="find_gaussian",
        score_parameter_grid={
            "tau_angstrom": [1.0],
            "lambda_positive": [0.2],
            "lambda_negative": [0.1],
            "gauss_score_min": [0.5],
        },
        refinement_multipliers={"lambda": [1.0], "score_threshold": [1.0]},
        min_voxel_values=[1, 2],
        objective_beta=2.0,
        coverage_thresholds=[0.3, 0.5, 0.6],
        topk_values=[3, 4, 5],
    )
    assert tuple(best["stages"]) == ("coarse", "refined", "min_voxels")
    assert best["min_voxels"] == 1
    assert best["score_parameters"]["tau_angstrom"] == 1.0
