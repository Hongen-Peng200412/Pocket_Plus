"""Selector tau_G 实际概率扫描、first maximum 与曲线发布测试。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.selector.calibration import calibrate_tau_g


def _save_npz(path: Path, **arrays: np.ndarray) -> None:
    """保存 calibration 测试所需的数值归档。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def _build_two_clg_case(
    root: Path,
    occurrence_voxel_count: np.ndarray,
    candidate_occurrence_offsets: np.ndarray,
    overlap_occurrence_index: np.ndarray,
    intersection_voxel_count: np.ndarray,
) -> tuple[Path, Path, Path]:
    """构造两个单节点 CLG，p_G 分别为 0.2/0.8 的完整校正输入。"""
    stage1_root = root / "stage1_outputs"
    run_dir = root / "selector_run"
    pdb_id = "demo"
    components = stage1_root / "unet_c1" / "calibration" / pdb_id / "components"
    _save_npz(
        components / "forest.npz",
        tree_id=np.asarray([0, 1], dtype=np.int32),
        node_id=np.asarray([10, 20], dtype=np.int32),
        parent_node_id=np.asarray([-1, -1], dtype=np.int32),
        voxel_count=np.asarray([2, 3], dtype=np.int32),
    )
    _save_npz(
        components / "clg.npz",
        CLG_id=np.asarray([0, 1], dtype=np.int32),
        tree_id=np.asarray([0, 1], dtype=np.int32),
        candidate_offsets=np.asarray([0, 1, 2], dtype=np.int64),
        candidate_node_id=np.asarray([10, 20], dtype=np.int32),
    )
    _save_npz(
        components / "overlap.npz",
        candidate_occurrence_offsets=candidate_occurrence_offsets,
        overlap_occurrence_index=overlap_occurrence_index,
        intersection_voxel_count=intersection_voxel_count,
        occurrence_voxel_count=occurrence_voxel_count,
    )
    _save_npz(
        run_dir / "calibration" / pdb_id / "scores.npz",
        CLG_id=np.asarray([0, 1], dtype=np.int32),
        CLG_logit=np.asarray([-1.3862944, 1.3862944], dtype=np.float32),
        CLG_valid_probability=np.asarray([0.2, 0.8], dtype=np.float32),
        candidate_offsets=np.asarray([0, 1, 2], dtype=np.int64),
        predicted_max_iou=np.asarray([0.0, 1.0], dtype=np.float32),
        selection_logit=np.asarray([0.0, 0.0], dtype=np.float32),
    )
    frozen_path = run_dir / "input_CLG_list.json"
    frozen_path.parent.mkdir(parents=True, exist_ok=True)
    frozen_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "stage1_model_name": "unet_c1",
                "split_order": ["calibration"],
                "split_counts": {"calibration": 2},
                "items": [
                    {"split": "calibration", "pdb_id": pdb_id, "CLG_id": 0},
                    {"split": "calibration", "pdb_id": pdb_id, "CLG_id": 1},
                ],
            }
        ),
        encoding="utf-8",
    )
    return stage1_root, run_dir, frozen_path


def test_tau_g_uses_actual_probabilities_and_publishes_full_curve(tmp_path: Path) -> None:
    """低概率坏预测应被门控，最优 tau_G 为实际高概率值 0.8。"""
    stage1_root, run_dir, frozen_path = _build_two_clg_case(
        tmp_path,
        occurrence_voxel_count=np.asarray([3], dtype=np.int32),
        candidate_occurrence_offsets=np.asarray([0, 0, 1], dtype=np.int64),
        overlap_occurrence_index=np.asarray([0], dtype=np.int32),
        intersection_voxel_count=np.asarray([3], dtype=np.int32),
    )
    payload = calibrate_tau_g(
        selector_run_dir=run_dir,
        stage1_outputs_root=stage1_root,
        input_clg_list_path=frozen_path,
        stage1_model_name="unet_c1",
        lambda_count=0.05,
    )
    assert np.isclose(payload["tau_G"], 0.8)
    assert len(payload["curve"]) == 2
    assert payload["metrics"]["M_instance"] == 1.0
    saved = json.loads((run_dir / "calibration.json").read_text(encoding="utf-8"))
    assert saved == payload


def test_tau_g_tie_uses_first_threshold_in_ascending_scan(tmp_path: Path) -> None:
    """全部 M_instance 同为零时，first maximum 固定选择升序首个实际 p_G。"""
    stage1_root, run_dir, frozen_path = _build_two_clg_case(
        tmp_path,
        occurrence_voxel_count=np.empty((0,), dtype=np.int32),
        candidate_occurrence_offsets=np.asarray([0, 0, 0], dtype=np.int64),
        overlap_occurrence_index=np.empty((0,), dtype=np.int32),
        intersection_voxel_count=np.empty((0,), dtype=np.int32),
    )
    payload = calibrate_tau_g(
        selector_run_dir=run_dir,
        stage1_outputs_root=stage1_root,
        input_clg_list_path=frozen_path,
        stage1_model_name="unet_c1",
        lambda_count=0.05,
    )
    assert payload["best_curve_index"] == 0
    assert np.isclose(payload["tau_G"], 0.2)
    assert payload["curve"][0]["tau_G"] < payload["curve"][1]["tau_G"]
