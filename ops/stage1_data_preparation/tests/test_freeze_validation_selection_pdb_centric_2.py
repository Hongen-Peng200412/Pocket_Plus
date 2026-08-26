from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from ops.stage1_data_preparation import freeze_validation_selection_pdb_centric_2


def _write_validation_pool(root: Path, pdb_id: str, occurrence_count: int) -> None:
    """写入一个具有 30 个 bias 与 context 候选的最小 validation PDB pool."""

    # Path, 测试 BOX pool 根目录下的 validation 分区。
    pool_directory = root / "validation"
    pool_directory.mkdir(parents=True, exist_ok=True)
    # int32, (O,), 当前 PDB 的连续 occurrence 身份。
    occurrence_id = np.arange(occurrence_count, dtype=np.int32)
    # int32, (O, 3), 由 occurrence 身份构造的最小合法 ZYX 起点。
    start_zyx = np.stack([occurrence_id] * 3, axis=1)
    # int32, (30, 3), 当前 PDB 的 30 个 context 候选起点。
    context_start_zyx = np.stack([np.arange(30)] * 3, axis=1).astype(np.int32)
    np.savez(
        pool_directory / f"{pdb_id}.npz",
        pdb_id=np.asarray(pdb_id),
        occurrence_id=occurrence_id,
        center_start_zyx=start_zyx,
        bias_start_zyx=np.repeat(start_zyx[:, None, :], 30, axis=1),
        context_start_zyx=context_start_zyx,
    )


def test_hardcoded_v2_freezer_uses_all_manifest_pdbs_and_is_deterministic(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """V2 入口冻结全部 PDB、cap 后的 bias 和每个 PDB 16 个 context."""

    # tuple[tuple[str, int], ...], 测试 manifest 中按顺序保存的 PDB 身份与 occurrence 数量。
    pdb_occurrence_counts = (
        ("1abc", 4),
        ("2def", 6),
        ("3ghi", 1),
        ("4jkl", 2),
    )
    assert freeze_validation_selection_pdb_centric_2.OUTPUT_PATH.name == (
        "validation_selection_pdb_centric_v2.npz"
    )
    for pdb_id, occurrence_count in pdb_occurrence_counts:
        _write_validation_pool(tmp_path, pdb_id, occurrence_count)
    # dict[str, object], Stage1TrainingRequestSet 读取的最小 validation manifest。
    manifest = {
        "schema_version": 1,
        "splits": {
            "validation": [
                {"pdb_id": pdb_id, "path": f"validation/{pdb_id}.npz"}
                for pdb_id, _ in pdb_occurrence_counts
            ]
        },
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (tmp_path / "_COMPLETE").write_text("", encoding="utf-8")
    # Path, 测试中代替服务器正式位置的 V2 输出文件。
    output_path = tmp_path / "validation_selection_pdb_centric_v2.npz"
    monkeypatch.setattr(
        freeze_validation_selection_pdb_centric_2,
        "VALIDATION_POOL_DIRECTORY",
        tmp_path / "validation",
    )
    monkeypatch.setattr(
        freeze_validation_selection_pdb_centric_2,
        "OUTPUT_PATH",
        output_path,
    )

    freeze_validation_selection_pdb_centric_2.main()

    with np.load(output_path, allow_pickle=False) as selection:
        # dict[str, np.ndarray], 首次冻结的全部字段，用于逐项复现核验。
        first_arrays = {
            field_name: selection[field_name].copy()
            for field_name in selection.files
        }
        assert selection.files == [
            "validation_pdb_id",
            "center_pdb_index",
            "center_occurrence_id",
            "bias_pdb_index",
            "bias_occurrence_id",
            "bias_candidate_index",
            "context_pdb_index",
            "context_candidate_index",
            "pdb_foreground_box_num",
            "pdb_foreground_fraction_target",
            "pdb_occurrence_foreground_box_cap",
        ]
        assert selection["validation_pdb_id"].dtype.kind == "S"
        assert selection["validation_pdb_id"].tolist() == [
            pdb_id.encode("utf-8") for pdb_id, _ in pdb_occurrence_counts
        ]
        assert selection["center_pdb_index"].dtype == np.dtype(np.int32)
        assert selection["center_pdb_index"].shape == (0,)
        assert selection["center_occurrence_id"].dtype == np.dtype(np.int32)
        assert selection["center_occurrence_id"].shape == (0,)
        assert selection["bias_pdb_index"].dtype == np.dtype(np.int32)
        assert selection["bias_pdb_index"].shape == (13,)
        assert selection["bias_occurrence_id"].dtype == np.dtype(np.int32)
        assert selection["bias_occurrence_id"].shape == (13,)
        assert selection["bias_candidate_index"].dtype == np.dtype(np.int16)
        assert selection["bias_candidate_index"].shape == (13,)
        assert np.bincount(selection["bias_pdb_index"], minlength=4).tolist() == [
            min(occurrence_count, 50) for _, occurrence_count in pdb_occurrence_counts
        ]
        assert selection["bias_candidate_index"].min() >= 0
        assert selection["bias_candidate_index"].max() < 30
        assert selection["context_pdb_index"].dtype == np.dtype(np.int32)
        assert selection["context_pdb_index"].shape == (64,)
        assert selection["context_candidate_index"].dtype == np.dtype(np.int32)
        assert selection["context_candidate_index"].shape == (64,)
        assert np.bincount(selection["context_pdb_index"], minlength=4).tolist() == [
            16,
            16,
            16,
            16,
        ]
        assert selection["context_candidate_index"].min() >= 0
        assert selection["context_candidate_index"].max() < 30
        assert selection["pdb_foreground_box_num"].dtype == np.dtype(np.int32)
        assert selection["pdb_foreground_box_num"].shape == ()
        assert selection["pdb_foreground_box_num"].item() == 50
        assert selection["pdb_foreground_fraction_target"].dtype == np.dtype(np.float64)
        assert selection["pdb_foreground_fraction_target"].shape == ()
        assert selection["pdb_foreground_fraction_target"].item() == pytest.approx(25 / 33)
        assert selection["pdb_occurrence_foreground_box_cap"].dtype == np.dtype(np.int32)
        assert selection["pdb_occurrence_foreground_box_cap"].shape == ()
        assert selection["pdb_occurrence_foreground_box_cap"].item() == 1

    freeze_validation_selection_pdb_centric_2.main()
    with np.load(output_path, allow_pickle=False) as selection:
        for field_name, expected_array in first_arrays.items():
            np.testing.assert_array_equal(selection[field_name], expected_array)
