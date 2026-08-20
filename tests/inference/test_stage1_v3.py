# -*- coding: utf-8 -*-
"""Stage1 V3 几何、产物、评分和评估契约测试."""

from __future__ import annotations

import numpy as np

from src.inference.blobs import extract_probability_blobs
from src.inference.calibration import calibrate_semantic_thresholds
from src.inference.centered import pack_centered_entries
from src.inference.evaluation import (
    aggregate_stage1_metrics,
    evaluate_centered_pdb,
)
from src.inference.full_map import gaussian_window_weight, window_starts_zyx
from src.inference.scoring import score_centered_candidates, select_centered_candidates


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


def test_find_gaussian_score_uses_five_angstrom_cutoff() -> None:
    """5 Å 内 A 原子贡献正项, 5 Å 外原子不参与 Gaussian 分数."""

    centered = {
        "source_probability_mean": np.asarray([0.5], dtype=np.float32),
        "voxel_offsets": np.asarray([0, 1], dtype=np.int64),
        "voxel_index_local_zyx": np.asarray([[0, 0, 0]], dtype=np.int16),
        "voxel_size_world": np.asarray([[1.0, 1.0, 1.0]], dtype=np.float32),
        "A_offsets": np.asarray([0, 2], dtype=np.int64),
        "A_coord_local_xyz": np.asarray([[0.5, 0.5, 0.5], [8.0, 8.0, 8.0]], dtype=np.float32),
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


def test_semantic_and_instance_metrics_follow_micro_contract() -> None:
    """阈值扫描、coverage 和 Hungarian 均使用固定的全局计数定义."""

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
    assert metrics["top3_success_ratio_0p5"] == 1.0


def test_selection_uses_score_and_minimum_voxel_count() -> None:
    """最终 selected 必须同时满足冻结分数阈值和最小来源体素数."""

    centered = {
        "voxel_offsets": np.asarray([0, 7, 15], dtype=np.int64),
        "score": np.zeros(2, dtype=np.float32),
        "selected": np.zeros(2, dtype=np.bool_),
    }
    selected = select_centered_candidates(
        centered,
        score=np.asarray([0.9, 0.8], dtype=np.float32),
        score_threshold=0.75,
        min_voxels=8,
    )
    assert selected["selected"].tolist() == [False, True]
