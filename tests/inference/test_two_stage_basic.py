from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.inference.main.two_stage_basic import build_stage1_cfg, build_stage2_cfg, read_best_threshold, read_stage1_best_thresholds


def _base_cfg(tmp_path: Path) -> dict:
    return {
        "mode": "voxel_param_search",
        "output_root": str(tmp_path / "two_stage_basic"),
        "cache_root": str(tmp_path / "cache"),
        "ckpt_path": str(tmp_path / "model.ckpt"),
        "raw_pairs_json": str(tmp_path / "pairs.json"),
        "filter_strength": "advanced",
        "threshold": 0.05,
        "min_component_voxels": 6,
        "connectivity_policy": "19_none",
        "sigma_nearby": 1.0,
        "kernel_nearby": 3,
        "sigma_response": 1.0,
        "kernel_response": 3,
        "score_add": 1.0,
        "score_minus": 1.0,
        "voxel_score_min": 0.0,
        "instance_score_min": 0.0,
        "search_strategy": "grid",
        "stage1_objective_expr": "avg_voxel_f1",
        "stage2_objective_expr": "avg_instance_f1 + 0.5*avg_voxel_f1",
        "search_space": {
            "threshold": {"type": "float", "min": 0.05, "max": 0.95, "step": 0.1},
            "min_component_voxels": {"type": "int", "min": 5, "max": 100, "step": 5},
            "connectivity_policy": {"values": ["7_none", "19_none"]},
            "score_add": {"type": "float", "min": 0.0, "max": 3.0, "step": 0.5},
        },
        "fixed_search_params": [],
    }


def test_build_stage1_cfg_only_searches_threshold(tmp_path: Path) -> None:
    base_cfg = _base_cfg(tmp_path)
    stage1_cfg = build_stage1_cfg(base_cfg, base_cfg["output_root"])

    assert stage1_cfg["output_root"].endswith("stage1_threshold_only")
    assert stage1_cfg["filter_strength"] == "basic"
    assert stage1_cfg["connectivity_policy"] == "7_none"
    assert stage1_cfg["min_component_voxels"] == 10
    assert stage1_cfg["search_space"] == {
        "threshold": {"type": "float", "min": 0.0, "max": 1.0, "step": 0.01},
    }
    assert stage1_cfg["objective_expr"] == "avg_voxel_f1"
    assert "min_component_voxels" in stage1_cfg["fixed_search_params"]
    assert "connectivity_policy" in stage1_cfg["fixed_search_params"]


def test_build_stage2_cfg_uses_best_threshold_window_and_basic_grid(tmp_path: Path) -> None:
    base_cfg = _base_cfg(tmp_path)
    stage2_cfg = build_stage2_cfg(base_cfg, base_cfg["output_root"], 0.42)

    assert stage2_cfg["output_root"].endswith("stage2_threshold_component_policy")
    assert stage2_cfg["filter_strength"] == "basic"
    assert stage2_cfg["search_space"]["threshold"] == {"type": "float", "min": 0.32, "max": 0.52, "step": 0.01}
    assert "min_component_voxels" not in stage2_cfg["search_space"]
    assert "connectivity_policy" not in stage2_cfg["search_space"]
    assert stage2_cfg["objective_expr"] == "avg_instance_f1 + 0.5*avg_voxel_f1"
    assert "min_component_voxels" not in stage2_cfg["fixed_search_params"]
    assert "connectivity_policy" not in stage2_cfg["fixed_search_params"]


@pytest.mark.parametrize(
    ("best_threshold", "expected_min", "expected_max"),
    [
        (0.03, 0.0, 0.13),
        (0.98, 0.88, 1.0),
    ],
)
def test_build_stage2_cfg_clamps_threshold_window(
    tmp_path: Path,
    best_threshold: float,
    expected_min: float,
    expected_max: float,
) -> None:
    base_cfg = _base_cfg(tmp_path)
    stage2_cfg = build_stage2_cfg(base_cfg, base_cfg["output_root"], best_threshold)

    assert stage2_cfg["search_space"]["threshold"]["min"] == expected_min
    assert stage2_cfg["search_space"]["threshold"]["max"] == expected_max


def test_read_best_threshold_reads_stage1_best_params(tmp_path: Path) -> None:
    best_params_path = tmp_path / "best_params.json"
    best_params_path.write_text(json.dumps({"threshold": 0.37}), encoding="utf-8")

    assert read_best_threshold(str(tmp_path)) == 0.37


def test_read_best_threshold_rejects_out_of_range_value(tmp_path: Path) -> None:
    best_params_path = tmp_path / "best_params.json"
    best_params_path.write_text(json.dumps({"threshold": 1.01}), encoding="utf-8")

    with pytest.raises(ValueError, match="best threshold"):
        read_best_threshold(str(tmp_path))


def test_read_stage1_best_thresholds_reads_by_class_best_params(tmp_path: Path) -> None:
    best_params_path = tmp_path / "best_params.json"
    best_params_path.write_text(
        json.dumps(
            {
                "by_class": {
                    "metal_ion": {"threshold": 0.11},
                    "small_molecule": {"threshold": 0.42},
                }
            }
        ),
        encoding="utf-8",
    )

    assert read_stage1_best_thresholds(str(tmp_path)) == {"metal_ion": 0.11, "small_molecule": 0.42}


def test_build_stage2_cfg_uses_by_class_threshold_windows(tmp_path: Path) -> None:
    base_cfg = _base_cfg(tmp_path)
    stage2_cfg = build_stage2_cfg(base_cfg, base_cfg["output_root"], {"metal_ion": 0.11, "small_molecule": 0.42})

    assert stage2_cfg["search_space_by_class"]["metal_ion"]["threshold"] == {"type": "float", "min": 0.01, "max": 0.21, "step": 0.01}
    assert stage2_cfg["search_space_by_class"]["small_molecule"]["threshold"] == {"type": "float", "min": 0.32, "max": 0.52, "step": 0.01}
    assert "min_component_voxels" not in stage2_cfg["search_space"]
    assert "connectivity_policy" not in stage2_cfg["search_space"]
