"""原子 NPZ、完整图发布与 calibration 完成顺序测试。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.artifacts.io import (
    atomic_savez_compressed,
    load_npz_strict,
    pack_centered_entries,
    validate_centered_archive,
)
from src.artifacts.paths import Stage1ArtifactPaths
from src.evaluation.calibration import calibrate_thresholds, publish_threshold_calibration
from src.inference.full_map import FullMapResult, publish_full_map


def test_atomic_npz_forbids_pickle_object_arrays(tmp_path: Path) -> None:
    path = tmp_path / "bad.npz"
    with pytest.raises(TypeError, match="object dtype"):
        atomic_savez_compressed(path, {"bad": np.asarray([{"x": 1}], dtype=object)})
    assert not path.exists()


def test_probability_payload_precedes_role_complete(tmp_path: Path) -> None:
    paths = Stage1ArtifactPaths(tmp_path, "unet_c1", "validation", "1abc")
    result = FullMapResult(
        probability_map=np.full((2, 3, 4), 0.25, dtype=np.float32),
        weight_sum=np.ones((2, 3, 4), dtype=np.float32),
        window_starts_zyx=((0, 0, 0),),
    )
    publish_full_map(
        paths=paths,
        result=result,
        origin_xyz=(1.0, 2.0, 3.0),
        voxel_size_xyz=(1.0, 1.0, 1.0),
    )
    assert paths.role_complete_path("probability").is_file()
    assert load_npz_strict(paths.probability_npz)["probability_map"].dtype == np.float32
    geometry = json.loads(paths.probability_geometry_json.read_text(encoding="utf-8"))
    assert geometry["full_shape_zyx"] == [2, 3, 4]


def test_calibration_complete_is_written_after_all_three_payloads(tmp_path: Path) -> None:
    paths = Stage1ArtifactPaths(tmp_path, "Find_0", "calibration", "unused")
    result = calibrate_thresholds(
        [(np.asarray([[[0.8, 0.2]]], dtype=np.float32), np.asarray([[[1, 0]]], dtype=np.bool_))],
        denominator=8,
    )
    publish_threshold_calibration(
        paths=paths,
        result=result,
        min_voxels=32,
        max_voxels=4096,
        fitted_metrics={
            "voxel_average_precision_macro": np.float64(1.0),
            "coverage_f1": np.asarray([0.5, 0.75], dtype=np.float64),
        },
    )
    assert paths.thresholds_json.is_file()
    assert paths.threshold_scan_npz.is_file()
    assert paths.calibration_metrics_json.is_file()
    assert paths.calibration_complete_path.is_file()
    metrics = json.loads(paths.calibration_metrics_json.read_text(encoding="utf-8"))
    assert metrics["coverage_f1"] == [0.5, 0.75]


@pytest.mark.parametrize(
    "centered_role",
    ("F1_centered", "CLG_centered", "Selected_Refined_Centered"),
)
def test_empty_centered_role_remains_publishable(centered_role: str) -> None:
    arrays = pack_centered_entries((), centered_role)
    validate_centered_archive(arrays, centered_role)
    assert arrays["centered_box_index"].shape == (0,)
    assert arrays["voxel_offsets"].tolist() == [0]
    assert arrays["voxel_ds_2"].shape == (0, 256, 20, 20, 20)
