import json
from pathlib import Path

import numpy as np
import pytest

from src.artifacts import load_npz_strict
from src.component_lineage import ComponentForest, ComponentNode, ComponentTree
from src.inference.Gauss_Scorer import (
    GaussScorerParameters,
    add_gauss_fields,
    publish_li_gauss_fields,
    publish_gauss_fields,
    score_f1_centered_nodes,
)
from src.inference.Gauss_Scorer import cli as gauss_cli


def _forest_arrays() -> dict[str, np.ndarray]:
    eligible = ComponentNode(
        tree_id=0,
        node_id=0,
        threshold_grid_index=100,
        threshold_value=100 / 32768,
        voxel_global_linear_index=np.asarray([0, 1], dtype=np.int64),
        bbox_min_zyx=np.asarray([0, 0, 0], dtype=np.int32),
        bbox_max_zyx=np.asarray([0, 0, 1], dtype=np.int32),
        centroid_zyx=np.asarray([0.0, 0.0, 0.5], dtype=np.float32),
        probability_mean=0.4,
        probability_max=0.8,
        candidate_eligible=True,
        ineligible_reason_code=0,
    )
    ineligible = ComponentNode(
        tree_id=1,
        node_id=0,
        threshold_grid_index=90,
        threshold_value=90 / 32768,
        voxel_global_linear_index=np.asarray([7], dtype=np.int64),
        bbox_min_zyx=np.asarray([1, 1, 1], dtype=np.int32),
        bbox_max_zyx=np.asarray([1, 1, 1], dtype=np.int32),
        centroid_zyx=np.asarray([1.0, 1.0, 1.0], dtype=np.float32),
        probability_mean=0.2,
        probability_max=0.2,
        candidate_eligible=False,
        ineligible_reason_code=1,
    )
    return ComponentForest((ComponentTree(0, (eligible,)), ComponentTree(1, (ineligible,)))).to_arrays()


def _centered_arrays() -> dict[str, np.ndarray]:
    return {
        "centered_box_index": np.asarray([0], dtype=np.int32),
        "box_start_zyx": np.asarray([[0, 0, 0]], dtype=np.int32),
        "box_shape_zyx": np.asarray([[80, 80, 80]], dtype=np.uint8),
        "box_origin_world": np.asarray([[0.0, 0.0, 0.0]], dtype=np.float32),
        "voxel_size_world": np.asarray([[1.0, 1.0, 1.0]], dtype=np.float32),
        "source_tree_id": np.asarray([0], dtype=np.int32),
        "source_node_id": np.asarray([0], dtype=np.int32),
        "source_threshold_grid_index": np.asarray([100], dtype=np.int32),
        "source_threshold_value": np.asarray([100 / 32768], dtype=np.float32),
        "voxel_offsets": np.asarray([0, 2], dtype=np.int64),
        "voxel_index_local_zyx": np.asarray([[0, 0, 0], [0, 0, 1]], dtype=np.int16),
        "centered_probability": np.asarray([0.8, 0.7], dtype=np.float32),
        "voxel_final": np.ones((2, 1), dtype=np.float16),
        "voxel_aux_offsets": np.asarray([0, 0], dtype=np.int64),
        "voxel_aux_index_local_zyx": np.empty((0, 3), dtype=np.int16),
        "voxel_aux_probability": np.empty(0, dtype=np.float32),
        "P_offsets": np.asarray([0, 0], dtype=np.int64),
        "P_coord_local_xyz": np.empty((0, 3), dtype=np.float32),
        "P_probability": np.empty(0, dtype=np.float32),
        "P_feat_L2": np.empty((0, 1), dtype=np.float16),
        "P_feat_L3": np.empty((0, 1), dtype=np.float16),
        "A_offsets": np.asarray([0, 2], dtype=np.int64),
        "A_global_index": np.asarray([10, 11], dtype=np.int64),
        "A_coord_local_xyz": np.asarray([[0.5, 0.5, 0.5], [20.0, 20.0, 20.0]], dtype=np.float32),
        "A_coord_centered_world": np.asarray([[-39.5, -39.5, -39.5], [-20.0, -20.0, -20.0]], dtype=np.float32),
        "A_probability": np.asarray([0.9, 0.1], dtype=np.float32),
        "A_feat_L0": np.zeros((2, 49), dtype=np.float32),
        "A_feat_L1": np.zeros((2, 1), dtype=np.float16),
        "A_feat_L2": np.zeros((2, 1), dtype=np.float16),
        "A_feat_L3": np.zeros((2, 1), dtype=np.float16),
    }


def test_score_uses_sum_weighting_and_fixed_distance_cutoff() -> None:
    parameters = GaussScorerParameters(
        lambda_positive=2.0,
        lambda_negative=1.0,
        tau_angstrom=1.0,
        gauss_score_min=2.0,
    )
    scores, selected = score_f1_centered_nodes(
        forest_arrays=_forest_arrays(),
        centered_arrays=_centered_arrays(),
        full_shape_zyx=(2, 2, 2),
        parameters=parameters,
    )

    assert scores[0] == pytest.approx(0.4 + 2.0 * 0.9 - 0.1)
    assert bool(selected[0])
    assert np.isnan(scores[1])
    assert not bool(selected[1])


def test_add_and_publish_preserve_core_fields(tmp_path: Path) -> None:
    forest = _forest_arrays()
    scores = np.asarray([2.1, np.nan], dtype=np.float32)
    selected = np.asarray([True, False], dtype=np.bool_)
    updated = add_gauss_fields(forest, scores, selected)
    ComponentForest.from_arrays(updated)
    for field, value in forest.items():
        assert np.array_equal(updated[field], value)

    path = tmp_path / "forest.npz"
    np.savez_compressed(path, **forest)
    publish_gauss_fields(path, scores, selected)
    restored = load_npz_strict(path)
    assert np.array_equal(restored["gauss_score"], scores, equal_nan=True)
    assert np.array_equal(restored["gauss_selected"], selected)
    publish_gauss_fields(path, scores, selected)


def test_force_overwrite_replaces_only_gauss_fields(tmp_path: Path) -> None:
    """显式强制覆盖只替换两个 Gauss 字段，不改 forest 核心字段。"""

    forest = _forest_arrays()
    first_score = np.asarray([1.0, np.nan], dtype=np.float32)
    first_selected = np.asarray([True, False], dtype=np.bool_)
    second_score = np.asarray([2.0, np.nan], dtype=np.float32)
    path = tmp_path / "forest.npz"
    np.savez_compressed(path, **forest)
    publish_gauss_fields(path, first_score, first_selected)
    publish_gauss_fields(
        path,
        second_score,
        first_selected,
        force_overwrite=True,
    )
    restored = load_npz_strict(path)
    for field, value in forest.items():
        assert np.array_equal(restored[field], value)
    assert np.array_equal(restored["gauss_score"], second_score, equal_nan=True)


def test_forest_rejects_partial_or_inconsistent_gauss_fields() -> None:
    forest = _forest_arrays()
    with pytest.raises(KeyError, match="必须同时存在"):
        ComponentForest.from_arrays(
            {**forest, "gauss_score": np.asarray([1.0, np.nan], dtype=np.float32)}
        )
    with pytest.raises(ValueError, match="candidate_eligible"):
        ComponentForest.from_arrays(
            {
                **forest,
                "gauss_score": np.asarray([1.0, 2.0], dtype=np.float32),
                "gauss_selected": np.asarray([False, True], dtype=np.bool_),
            }
        )


def test_li_force_overwrite_rejects_partial_gauss_pair(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """强制刷新不能掩盖 Li 归档中只存在一个 Gauss 字段的损坏状态。"""

    path = tmp_path / "Li_centered.npz"
    np.savez_compressed(
        path,
        gauss_score=np.asarray([1.0], dtype=np.float32),
    )
    monkeypatch.setattr(
        "src.inference.Gauss_Scorer.scorer.validate_centered_archive",
        lambda *args, **kwargs: None,
    )
    with pytest.raises(KeyError, match="只存在一项"):
        publish_li_gauss_fields(
            path,
            np.asarray([2.0], dtype=np.float32),
            np.asarray([True], dtype=np.bool_),
            force_overwrite=True,
        )


def test_cli_incrementally_skips_pending_and_running_pdb(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """GPU 仍在生产时，CPU 回填应处理静止项并报告其余项，而不是整批失败。"""

    pdb_list = tmp_path / "pdb_ids.json"
    pdb_list.write_text(
        json.dumps(["complete", "pending", "running", "blob_exceed"]),
        encoding="utf-8",
    )
    calibration = tmp_path / "calibration.json"
    calibration.write_text(
        json.dumps(
            {
                "selected_parameters": {
                    "lambda_positive": 0.1,
                    "lambda_negative": 0.001,
                    "tau_angstrom": 1.0,
                    "gauss_score_min": 1.0,
                    "distance_cutoff_angstrom": 5.0,
                }
            }
        ),
        encoding="utf-8",
    )

    class _BlobExceedPath:
        def __init__(self, exists: bool) -> None:
            self._exists = exists

        def is_file(self) -> bool:
            return self._exists

    class _Paths:
        def __init__(
            self,
            *,
            output_root: str,
            stage1_model_name: str,
            split: str,
            pdb_id: str,
        ) -> None:
            del output_root, stage1_model_name, split
            self.pdb_id = pdb_id
            self.blob_exceed_path = _BlobExceedPath(pdb_id == "blob_exceed")
            self.forest_npz = Path(f"{pdb_id}.forest.npz")
            self.probability_npz = Path(f"{pdb_id}.probability.npz")

        def centered_npz(self, role: str) -> Path:
            return Path(f"{self.pdb_id}.{role}.npz")

    class _Lease:
        def __enter__(self) -> "_Lease":
            return self

        def __exit__(self, *args: object) -> None:
            del args

    class _LeaseFactory:
        @staticmethod
        def acquire(paths: _Paths, owner_token: str) -> _Lease | None:
            del owner_token
            return None if paths.pdb_id == "running" else _Lease()

    published: list[Path] = []
    monkeypatch.setattr(gauss_cli, "Stage1ArtifactPaths", _Paths)
    monkeypatch.setattr(
        gauss_cli,
        "is_role_complete",
        lambda paths, role: not (paths.pdb_id == "pending" and role == "F1_centered"),
    )
    monkeypatch.setattr(gauss_cli, "PdbRunningLease", _LeaseFactory)
    monkeypatch.setattr(
        gauss_cli,
        "load_npz_strict",
        lambda path: (
            {"probability_map": np.zeros((1, 1, 1), dtype=np.float32)}
            if "probability" in path.name
            else {}
        ),
    )
    monkeypatch.setattr(
        gauss_cli,
        "score_f1_centered_nodes",
        lambda **kwargs: (
            np.empty(0, dtype=np.float32),
            np.empty(0, dtype=np.bool_),
        ),
    )
    monkeypatch.setattr(
        gauss_cli,
        "publish_gauss_fields",
        lambda path, score, selected, **kwargs: published.append(path),
    )

    result = gauss_cli.main(
        [
            "--pdb-list",
            str(pdb_list),
            "--output-root",
            str(tmp_path),
            "--producer",
            "Find_0",
            "--split",
            "validation",
            "--calibration-json",
            str(calibration),
        ]
    )

    assert result == 0
    assert published == [Path("complete.forest.npz")]
    summary = json.loads(capsys.readouterr().out)
    assert summary["n_completed"] == 1
    assert summary["n_pending"] == 1
    assert summary["n_skipped_running"] == 1
    assert summary["n_blob_exceed"] == 1
    assert summary["pending"] == [
        {"pdb_id": "pending", "missing_roles": ["F1_centered"]}
    ]
    assert summary["skipped_running"] == ["running"]


def test_gauss_cli_defaults_to_force_overwrite() -> None:
    """正式回填入口默认使用当前冻结参数覆盖旧 Gauss 结果。"""

    arguments = gauss_cli.build_parser().parse_args(
        [
            "--pdb-list",
            "ids.json",
            "--output-root",
            "out",
            "--producer",
            "Find_0",
            "--split",
            "calibration",
            "--calibration-json",
            "calibration.json",
        ]
    )
    assert arguments.force_overwrite is True
    assert arguments.evaluate_on_blob_exceed is False
