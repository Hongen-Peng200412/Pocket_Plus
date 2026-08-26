# -*- coding: utf-8 -*-
"""验证 Li 截断、精确候选比例搜索与隔离 calibration 实验入口."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf
import pytest

from ops.stage1_li_ratio_trial.run_trial import (
    run_li_calibration,
    run_li_validation,
)
from src.inference.artifacts import Stage1ArtifactPaths, publish_stage1_artifact
from src.inference.blobs import li_probability_threshold
from src.inference.calibration import tune_centered_selection
from src.inference.scoring import select_score_ratio_candidates


def _build_ratio_tuning_case() -> tuple[
    tuple[tuple[str, dict[str, np.ndarray]], ...],
    dict[str, tuple[np.ndarray, tuple[np.ndarray, ...], tuple[int, int, int]]],
]:
    """构造 4 候选 PDB 与 1 候选 PDB 共享比例轴的校准事实."""
    # large 的首个候选命中唯一 occurrence, 其后三个候选都是独立假阳性体素.
    large = {
        "source_blob_index": np.arange(4, dtype=np.int32),
        "source_probability_mean": np.asarray(
            [0.9, 0.8, 0.7, 0.6], dtype=np.float32
        ),
        "voxel_offsets": np.arange(5, dtype=np.int64),
        "voxel_index_local_zyx": np.asarray(
            [[0, 0, index] for index in range(4)], dtype=np.int32
        ),
        "box_start_zyx": np.zeros((4, 3), dtype=np.int32),
    }
    # small 的唯一候选命中唯一 occurrence; 它在比例 0.5 时首次进入选择集合.
    small = {
        "source_blob_index": np.asarray([0], dtype=np.int32),
        "source_probability_mean": np.asarray([0.85], dtype=np.float32),
        "voxel_offsets": np.asarray([0, 1], dtype=np.int64),
        "voxel_index_local_zyx": np.asarray([[0, 0, 0]], dtype=np.int32),
        "box_start_zyx": np.zeros((1, 3), dtype=np.int32),
    }
    ground_truth = {
        "large": (
            np.asarray([1], dtype=np.int32),
            (np.asarray([[0, 0, 0]], dtype=np.int32),),
            (1, 1, 4),
        ),
        "small": (
            np.asarray([1], dtype=np.int32),
            (np.asarray([[0, 0, 0]], dtype=np.int32),),
            (1, 1, 1),
        ),
    }
    return (("large", large), ("small", small)), ground_truth


def test_li_threshold_matches_historical_iteration_and_quantizes_upward() -> None:
    """Li 迭代必须使用均值初始化、端点背景和向上概率网格量化."""
    probability = np.asarray([0.1, 0.2, 0.8, 0.9], dtype=np.float32).reshape(
        1, 2, 2
    )
    raw_threshold, grid_index, applied_threshold = li_probability_threshold(
        probability,
        32768,
    )
    # expected 是背景均值 0.15 与前景均值 0.85 代入 Li 更新式的稳定值.
    expected = (0.15 - 0.85) / (np.log(0.15) - np.log(0.85))

    assert raw_threshold == pytest.approx(expected, abs=1e-7)
    assert grid_index == int(np.ceil(raw_threshold * 32768))
    assert applied_threshold == grid_index / 32768
    assert raw_threshold <= applied_threshold < raw_threshold + 1.0 / 32768


def test_score_ratio_uses_fixed_prefilter_population_and_half_up_count() -> None:
    """比例选择必须先固定预过滤总体, 再按 half-up 数量稳定截取高分候选."""
    scores = np.asarray([0.9, 0.9, 0.7, 1.0], dtype=np.float32)
    prefilter_eligible = np.asarray([True, True, True, False])
    selected = select_score_ratio_candidates(scores, prefilter_eligible, 0.5)

    # N=3 且 r=0.5 时保留 floor(2.0)=2 项; 并列 0.9 保持原候选顺序.
    assert selected.tolist() == [True, True, False, False]


def test_basic_ratio_scans_exact_states_and_parallel_result_is_identical() -> None:
    """不等候选数 PDB 的精确比例状态、两阶段目标和并发结果必须一致."""
    centered_items, ground_truth = _build_ratio_tuning_case()
    arguments = {
        "centered_items": centered_items,
        "ground_truth_by_pdb": ground_truth,
        "score_mode": "basic_ratio",
        "score_parameter_grid": None,
        "refinement_multipliers": None,
        "prefiltered_min_voxel": 1,
        "min_voxel_values": [1, 2],
        "objective_beta": 1.0,
        "coverage_thresholds": [0.3, 0.5, 0.6],
        "topk_values": [3, 4, 5],
    }
    serial = tune_centered_selection(**arguments, workers=1)
    parallel = tune_centered_selection(**arguments, workers=4)

    # r=0.5 时 large 保留 2/4, small 保留 1/1; 向 1 取相邻值稳定包含该边界.
    expected_ratio = float(np.nextafter(0.5, 1.0))
    assert serial == parallel
    assert serial["score_ratio_threshold"] == expected_ratio
    assert "score_threshold" not in serial
    assert serial["stages"]["score_ratio_threshold"]["objective"] == pytest.approx(
        2.5
    )
    assert serial["stages"]["min_voxels"]["objective"] == pytest.approx(2.5)


def test_basic_ratio_objective_tie_keeps_smaller_ratio() -> None:
    """两个候选数状态目标严格并列时必须保留先遇到的较小比例."""
    candidate = {
        "source_blob_index": np.asarray([0, 1], dtype=np.int32),
        "source_probability_mean": np.asarray([0.9, 0.8], dtype=np.float32),
        "voxel_offsets": np.asarray([0, 1, 2], dtype=np.int64),
        "voxel_index_local_zyx": np.asarray(
            [[0, 0, 0], [0, 0, 1]], dtype=np.int32
        ),
        "box_start_zyx": np.zeros((2, 3), dtype=np.int32),
    }
    # 一个 occurrence 含两个体素: 选择 1 项和 2 项时三项目标都精确等于 8/3.
    ground_truth = {
        "demo": (
            np.asarray([1], dtype=np.int32),
            (np.asarray([[0, 0, 0], [0, 0, 1]], dtype=np.int32),),
            (1, 1, 2),
        )
    }
    selection = tune_centered_selection(
        centered_items=(("demo", candidate),),
        ground_truth_by_pdb=ground_truth,
        score_mode="basic_ratio",
        score_parameter_grid=None,
        refinement_multipliers=None,
        prefiltered_min_voxel=1,
        min_voxel_values=[1],
        objective_beta=1.0,
        coverage_thresholds=[0.3, 0.5, 0.6],
        topk_values=[3, 4, 5],
        workers=1,
    )

    assert selection["score_ratio_threshold"] == float(np.nextafter(0.25, 1.0))
    assert selection["objective"] == pytest.approx(8.0 / 3.0)


def test_isolated_li_calibration_publishes_normal_artifact_layout(
    tmp_path: Path,
) -> None:
    """小型端到端实验必须发布 Li blobs、F1/F2 参数及 micro/macro 评估."""
    config = OmegaConf.create(
        {
            "calibration": {"workers": 2, "min_voxel_values": [1]},
            "evaluation": {
                "coverage_thresholds": [0.3, 0.5, 0.6],
                "topk_values": [3, 4, 5],
            },
        }
    )
    data_root = tmp_path / "data"
    output_root = tmp_path / "artifacts"
    pdb_ids = ("first", "second")
    for pdb_id in pdb_ids:
        # 三个高概率体素由低概率体素隔开, Li 截断后形成三个独立连通区域.
        probability = np.asarray(
            [[[0.9, 0.1, 0.8, 0.1, 0.7]]], dtype=np.float32
        )
        for split in ("calibration", "validation"):
            paths = Stage1ArtifactPaths(output_root, "unet_c1", split, pdb_id)
            publish_stage1_artifact(
                paths.artifact("probability"),
                {"probability_map": probability},
                paths.complete("probability"),
            )
        density_root = data_root / "density" / pdb_id
        density_root.mkdir(parents=True)
        np.savez_compressed(
            density_root / "ligand_area.npz",
            grid_shape_zyx=np.asarray([1, 1, 5], dtype=np.int32),
            mask_1=np.asarray([[0, 0, 0]], dtype=np.int32),
        )
        # union_mask 第一维是既有单通道轴, evaluate 读取索引 0 得到完整图 bool mask.
        union_mask = np.zeros((1, 1, 1, 5), dtype=np.bool_)
        union_mask[0, 0, 0, 0] = True
        np.save(density_root / "union_mask.npy", union_mask)

    run_li_calibration(
        config=config,
        pdb_ids=pdb_ids,
        data_root=data_root,
        output_root=output_root,
        producer="unet_c1",
        split="calibration",
        denominator=32768,
        prefiltered_min_voxel=1,
    )
    # tuning_before_validation 证明 validation 只读取而不替换 calibration 选择参数.
    tuning_root = output_root / "unet_c1" / "tuning"
    tuning_before_validation = {
        path.name: path.read_text(encoding="utf-8")
        for path in tuning_root.glob("Li_*_basic_ratio.json")
    }
    run_li_validation(
        config=config,
        pdb_ids=pdb_ids,
        data_root=data_root,
        output_root=output_root,
        producer="unet_c1",
        split="validation",
        denominator=32768,
    )
    assert tuning_before_validation == {
        path.name: path.read_text(encoding="utf-8")
        for path in tuning_root.glob("Li_*_basic_ratio.json")
    }

    first_paths = Stage1ArtifactPaths(
        output_root, "unet_c1", "calibration", "first"
    )
    with np.load(first_paths.artifact("Li_blobs"), allow_pickle=False) as blobs:
        assert {
            "li_threshold_raw",
            "li_threshold_grid_index",
            "li_threshold_applied",
        }.issubset(blobs.files)
    assert first_paths.complete("Li_blobs").is_file()
    for beta_tag in ("F1", "F2"):
        tuning_path = (
            output_root / "unet_c1" / "tuning" / f"Li_{beta_tag}_basic_ratio.json"
        )
        selection = json.loads(tuning_path.read_text(encoding="utf-8"))
        assert selection["candidate_role"] == "Li_blobs"
        assert "score_ratio_threshold" in selection
        assert "score_threshold" not in selection
        metrics_path = (
            output_root
            / "unet_c1"
            / "calibration"
            / "evaluation"
            / f"li_{beta_tag.lower()}_blobs_basic_ratio_selected.metrics.json"
        )
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        assert {
            "semantic_micro_f1",
            "semantic_macro_f1",
            "coverage_micro_f1_0p3",
            "coverage_macro_f1_0p3",
            "one_to_one_micro_f1_0p3",
            "one_to_one_macro_f1_0p3",
            "semantic_micro_prauc",
            "semantic_macro_prauc",
        }.issubset(metrics)
        validation_metrics_path = (
            output_root
            / "unet_c1"
            / "validation"
            / "evaluation"
            / f"li_{beta_tag.lower()}_blobs_basic_ratio_selected.metrics.json"
        )
        validation_metrics = json.loads(
            validation_metrics_path.read_text(encoding="utf-8")
        )
        assert validation_metrics["pdb_count"] == 2
