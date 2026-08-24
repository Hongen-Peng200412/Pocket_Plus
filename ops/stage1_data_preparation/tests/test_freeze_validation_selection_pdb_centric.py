from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from ops.stage1_data_preparation import freeze_validation_selection_pdb_centric


def _write_validation_pool(root: Path, pdb_id: str, occurrence_count: int) -> None:
    """写入一个具有正式候选维度的最小 validation PDB pool。"""

    pool_directory = root / "validation"
    pool_directory.mkdir(parents=True, exist_ok=True)
    occurrence_id = np.arange(occurrence_count, dtype=np.int32)
    start_zyx = np.stack([occurrence_id] * 3, axis=1)
    np.savez(
        pool_directory / f"{pdb_id}.npz",
        pdb_id=np.asarray(pdb_id),
        occurrence_id=occurrence_id,
        center_start_zyx=start_zyx,
        bias_start_zyx=np.repeat(start_zyx[:, None, :], 30, axis=1),
        context_start_zyx=np.stack([np.arange(30)] * 3, axis=1).astype(np.int32),
    )


def test_hardcoded_freezer_writes_the_pdb_centric_validation_selection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """硬编码冻结入口只发布 PDB 中心索引及三个采样参数。"""

    _write_validation_pool(tmp_path, "1abc", occurrence_count=4)
    _write_validation_pool(tmp_path, "2def", occurrence_count=6)
    manifest = {
        "schema_version": 1,
        "splits": {
            "validation": [
                {"pdb_id": "1abc", "path": "validation/1abc.npz"},
                {"pdb_id": "2def", "path": "validation/2def.npz"},
            ]
        },
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (tmp_path / "_COMPLETE").write_text("", encoding="utf-8")
    output_path = tmp_path / "validation_selection_pdb_centric.npz"
    monkeypatch.setattr(
        freeze_validation_selection_pdb_centric,
        "VALIDATION_POOL_DIRECTORY",
        tmp_path / "validation",
    )
    monkeypatch.setattr(
        freeze_validation_selection_pdb_centric,
        "OUTPUT_PATH",
        output_path,
    )

    freeze_validation_selection_pdb_centric.main()

    with np.load(output_path, allow_pickle=False) as selection:
        assert set(selection.files) == {
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
        }
        assert selection["validation_pdb_id"].tolist() == [b"1abc", b"2def"]
        assert selection["center_pdb_index"].shape == (0,)
        assert np.bincount(
            selection["bias_pdb_index"], minlength=2
        ).tolist() == [20, 25]
        assert np.bincount(
            selection["context_pdb_index"], minlength=2
        ).tolist() == [25, 25]
        assert selection["pdb_foreground_box_num"].item() == 25
        assert selection["pdb_foreground_fraction_target"].item() == 0.5
        assert selection["pdb_occurrence_foreground_box_cap"].item() == 5
