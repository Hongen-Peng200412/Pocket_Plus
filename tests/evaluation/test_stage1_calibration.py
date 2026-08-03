"""整数直方图阈值扫描与完整区域 macro AP 测试. """

from __future__ import annotations

import numpy as np

from src.artifacts.paths import Stage1ArtifactPaths
from src.evaluation.calibration import (
    ThresholdHistogram,
    calibrate_published_full_maps_and_freeze_thresholds,
    calibrate_thresholds,
)
from src.evaluation.voxel_metrics import average_precision_full_grid, macro_average_precision
from src.inference.full_map import FullMapResult, publish_full_map


def test_histogram_counts_equal_bruteforce_thresholds() -> None:
    probability = np.asarray([0.0, 0.12, 0.25, 0.51, 0.99, 1.0], dtype=np.float32).reshape(1, 2, 3)
    target = np.asarray([0, 1, 0, 1, 1, 0], dtype=np.bool_).reshape(1, 2, 3)
    histogram = ThresholdHistogram(denominator=8)
    histogram.update(probability, target)
    tp, fp, fn = histogram.threshold_counts()
    for grid_index in range(9):
        prediction = probability >= grid_index / 8.0
        assert tp[grid_index] == np.logical_and(prediction, target).sum()
        assert fp[grid_index] == np.logical_and(prediction, ~target).sum()
        assert fn[grid_index] == np.logical_and(~prediction, target).sum()


def test_calibration_uses_first_maximum_and_actual_grid_indices() -> None:
    probability = np.full((1, 1, 2), 0.5, dtype=np.float32)
    target = np.asarray([[[True, False]]])
    result = calibrate_thresholds(
        probability_and_target=[(probability, target)],
        denominator=32768,
    )
    assert result.f_alpha_curve.shape == (7, 32769)
    assert np.all(result.alpha_threshold_grid_index == 0)
    assert np.all(result.t_alpha == 0.0)
    payload = result.thresholds_payload("Find_1", min_voxels=32, max_voxels=4096)
    assert payload["denominator"] == 32768
    assert len(payload["alpha_threshold_grid_index"]) == 7
    assert "physical_threshold_values" not in payload


def test_average_precision_includes_every_full_grid_voxel() -> None:
    probability = np.asarray([0.9, 0.8, 0.1], dtype=np.float32).reshape(1, 1, 3)
    target = np.asarray([1, 0, 1], dtype=np.bool_).reshape(1, 1, 3)
    # AP = 1/2 * 1 + 1/2 * 2/3 = 5/6; 中间负 voxel 不能被区域 mask 排除. 
    assert np.isclose(average_precision_full_grid(probability, target), 5.0 / 6.0)
    macro = macro_average_precision(
        [
            (probability, target),
            (np.zeros((1, 1, 1), dtype=np.float32), np.zeros((1, 1, 1), dtype=np.bool_)),
        ]
    )
    assert macro["n_valid_pdb"] == 1
    assert macro["n_total_pdb"] == 2
    assert np.isclose(macro["voxel_average_precision_macro"], 5.0 / 6.0)


def test_published_calibration_entry_freezes_thresholds_and_full_report(tmp_path) -> None:
    shape = (80, 80, 80)
    occurrence_indices: dict[str, dict[int, np.ndarray]] = {}
    for row, pdb_id in enumerate(("cal_a", "cal_b")):
        probability = np.zeros(shape, dtype=np.float32)
        zyx = (10 + row, 20, 30)
        probability[zyx] = 0.9
        if pdb_id == "cal_b":
            probability[zyx[0], zyx[1], 31:40] = 0.9
        linear = np.asarray([np.ravel_multi_index(zyx, shape)], dtype=np.int64)
        occurrence_indices[pdb_id] = {100 + row: linear}
        paths = Stage1ArtifactPaths(tmp_path, "unet_c1", "calibration", pdb_id)
        publish_full_map(
            paths=paths,
            result=FullMapResult(
                probability_map=probability,
                weight_sum=np.ones(shape, dtype=np.float32),
                window_starts_zyx=((0, 0, 0),),
            ),
            origin_xyz=(0.0, 0.0, 0.0),
            voxel_size_xyz=(1.0, 1.0, 1.0),
        )

    result, metrics = calibrate_published_full_maps_and_freeze_thresholds(
        output_root=tmp_path,
        stage1_model_name="unet_c1",
        calibration_pdb_ids=("cal_a", "cal_b"),
        occurrence_voxel_loader=lambda pdb_id, shape_zyx: occurrence_indices[pdb_id],
        min_voxels=1,
        max_voxels=10,
        denominator=8,
    )
    paths = Stage1ArtifactPaths(tmp_path, "unet_c1", "calibration", "cal_a")
    assert result.alpha_threshold_grid_index[3] == 1
    assert np.isclose(metrics["semantic_dice_micro_t_F1"], 4.0 / 13.0)
    assert np.isclose(metrics["semantic_dice_macro_t_F1"], (1.0 + 2.0 / 11.0) / 2.0)
    assert "semantic_dice_t_F1" not in metrics
    assert metrics["one_to_one_f1_0p5"] == 0.5
    assert metrics["top3_success_ratio_0p5"] == 0.5
    assert paths.thresholds_json.is_file()
    assert paths.threshold_scan_npz.is_file()
    assert paths.calibration_complete_path.is_file()


def test_fitted_metrics_exclude_blob_exceed_pdb_from_every_metric(tmp_path) -> None:
    """阈值扫描保留完整 calibration；冻结阈值后的指标完全排除组件超限 PDB。"""
    shape = (80, 80, 80)
    occurrence_indices: dict[str, dict[int, np.ndarray]] = {}

    normal_probability = np.zeros(shape, dtype=np.float32)
    normal_zyx = (40, 40, 40)
    normal_probability[normal_zyx] = 0.9
    occurrence_indices["normal"] = {
        1: np.asarray([np.ravel_multi_index(normal_zyx, shape)], dtype=np.int64)
    }

    exceed_probability = np.zeros(shape, dtype=np.float32)
    isolated_points = [
        (z, y, 40)
        for z in range(1, 79, 3)
        for y in range(1, 79, 3)
    ][:201]
    for zyx in isolated_points:
        exceed_probability[zyx] = 0.9
    occurrence_indices["exceed"] = {
        2: np.asarray(
            [np.ravel_multi_index(isolated_points[0], shape)], dtype=np.int64
        )
    }

    for pdb_id, probability in (
        ("normal", normal_probability),
        ("exceed", exceed_probability),
    ):
        publish_full_map(
            paths=Stage1ArtifactPaths(
                tmp_path, "Find_0", "calibration", pdb_id
            ),
            result=FullMapResult(
                probability_map=probability,
                weight_sum=np.ones(shape, dtype=np.float32),
                window_starts_zyx=((0, 0, 0),),
            ),
            origin_xyz=(0.0, 0.0, 0.0),
            voxel_size_xyz=(1.0, 1.0, 1.0),
        )

    _, metrics = calibrate_published_full_maps_and_freeze_thresholds(
        output_root=tmp_path,
        stage1_model_name="Find_0",
        calibration_pdb_ids=("normal", "exceed"),
        occurrence_voxel_loader=lambda pdb_id, shape_zyx: occurrence_indices[
            pdb_id
        ],
        min_voxels=1,
        max_voxels=1,
        denominator=8,
    )

    assert metrics["n_total_pdb"] == 2
    assert metrics["n_blob_exceed_pdb"] == 1
    assert metrics["n_evaluated_pdb"] == 1
    assert metrics["n_valid_voxel_ap_pdb"] == 1
    assert metrics["semantic_dice_micro_t_F1"] == 1.0
    assert metrics["semantic_dice_macro_t_F1"] == 1.0
    assert metrics["semantic_tp_t_F1"] == 1
    assert metrics["semantic_fp_t_F1"] == 0
    assert metrics["semantic_fn_t_F1"] == 0
    assert metrics["n_pred_instances"] == 1
    assert metrics["n_gt_instances"] == 1
    assert metrics["coverage_f1_0p3"] == 1.0
    assert metrics["one_to_one_f1_0p3"] == 1.0
    assert metrics["n_topk_eligible_pdb"] == 1
    assert metrics["top3_success_ratio_0p3"] == 1.0
