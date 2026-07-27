from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from src.datasets.ops.stage1_split import freeze_stage1_splits


def _write_exp_shape(data_root: Path, pdb_id: str, shape_zyx: tuple[int, int, int]) -> None:
    """写入 split freezer 所需的最小 E1 冻结几何元数据。"""

    density_directory = data_root / "density" / pdb_id
    density_directory.mkdir(parents=True)
    np.savez(
        density_directory / "exp.npz",
        canonical_shape_zyx=np.asarray(shape_zyx, dtype=np.int64),
    )


def _read_split_rows(path: Path) -> list[dict[str, object]]:
    """读取 freezer 发布的 JSON 行表。"""

    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, list)
    return value


def test_split_freezer_preserves_pdb_groups_counts_small_map_rule_and_determinism(
    tmp_path: Path,
) -> None:
    """同时验证分组、精确计数、80³ 候选约束及稳定 seed。"""

    data_root = tmp_path / "data"
    keep_list = tmp_path / "keep_list.jsonl"
    source_rows: list[dict[str, object]] = []
    short_pdb_ids = {"p002", "p009"}
    for pdb_index in range(12):
        pdb_id = f"p{pdb_index:03d}"
        shape = (79, 100, 100) if pdb_id in short_pdb_ids else (80, 90, 100)
        _write_exp_shape(data_root, pdb_id, shape)
        source_rows.extend(
            [
                {"pdb_id": pdb_id, "candidate_id": 0, "source_tag": f"{pdb_id}-a"},
                {"pdb_id": pdb_id, "candidate_id": 3, "source_tag": f"{pdb_id}-b"},
            ]
        )
    keep_list.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in source_rows),
        encoding="utf-8",
    )

    first_root = tmp_path / "split_first"
    second_root = tmp_path / "split_second"
    first_summary = freeze_stage1_splits(
        keep_list,
        data_root,
        first_root,
        seed=3407,
        validation_pdb_count=1,
        calibration_pdb_count=1,
    )
    second_summary = freeze_stage1_splits(
        keep_list,
        data_root,
        second_root,
        seed=3407,
        validation_pdb_count=1,
        calibration_pdb_count=1,
    )

    split_names = ("train", "validation", "calibration", "held_out_pool")
    emitted_rows: list[dict[str, object]] = []
    pdb_sets: dict[str, set[str]] = {}
    for split_name in split_names:
        first_path = first_root / f"{split_name}.json"
        second_path = second_root / f"{split_name}.json"
        assert first_path.read_bytes() == second_path.read_bytes()
        rows = _read_split_rows(first_path)
        emitted_rows.extend(rows)
        pdb_sets[split_name] = {str(row["pdb_id"]).lower() for row in rows}

    assert first_summary == second_summary
    assert len(pdb_sets["train"]) == math.floor(0.75 * 12)
    assert len(pdb_sets["validation"]) == 1
    assert len(pdb_sets["calibration"]) == 1
    assert len(pdb_sets["held_out_pool"]) == 1
    assert pdb_sets["validation"].isdisjoint(short_pdb_ids)
    assert pdb_sets["calibration"].isdisjoint(short_pdb_ids)
    assert all(
        pdb_sets[left].isdisjoint(pdb_sets[right])
        for index, left in enumerate(split_names)
        for right in split_names[index + 1 :]
    )
    assert sorted(emitted_rows, key=lambda row: (str(row["pdb_id"]), int(row["candidate_id"]))) == source_rows
    assert (first_root / "config.json").is_file()
    assert (first_root / "summary.json").is_file()
    assert (first_root / "_COMPLETE").is_file()


def test_split_freezer_rejects_insufficient_large_map_groups(tmp_path: Path) -> None:
    """validation/calibration 候选不足时不得用短轴 PDB 补数。"""

    data_root = tmp_path / "data"
    keep_list = tmp_path / "keep_list.jsonl"
    rows = []
    for pdb_index in range(8):
        pdb_id = f"q{pdb_index:03d}"
        _write_exp_shape(data_root, pdb_id, (80, 80, 80) if pdb_index == 0 else (79, 80, 80))
        rows.append({"pdb_id": pdb_id, "candidate_id": 0})
    keep_list.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    with pytest.raises(ValueError, match="不足以冻结"):
        freeze_stage1_splits(
            keep_list,
            data_root,
            tmp_path / "split",
            validation_pdb_count=1,
            calibration_pdb_count=1,
        )
