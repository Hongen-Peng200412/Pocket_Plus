"""Stage1 v3 日期、质量阈值与确定性集合划分测试。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ops.stage1_data_preparation import freeze_split as split_module


def write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    """写出测试使用的 UTF-8 JSONL。"""

    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )


def test_freeze_split_uses_strict_boundaries_and_200_100_rest(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """日期等于界线进入 held-out，质量等于界线被拒绝，合格 PDB 按 200/100/剩余划分。"""

    eligible_ids = [f"a{index:03d}" for index in range(302)]
    candidates = [
        {
            "candidate_id": index,
            "pdb_id": pdb_id,
            "map_resolution": 3.99,
            "cc_contour": 0.651,
        }
        for index, pdb_id in enumerate(eligible_ids)
    ]
    candidates.extend(
        [
            {"candidate_id": 1000, "pdb_id": "held", "map_resolution": 3.0, "cc_contour": 0.9},
            {"candidate_id": 1001, "pdb_id": "res4", "map_resolution": 4.0, "cc_contour": 0.9},
            {"candidate_id": 1002, "pdb_id": "cc065", "map_resolution": 3.0, "cc_contour": 0.65},
            {"candidate_id": 1003, "pdb_id": "missing", "map_resolution": 3.0, "cc_contour": 0.9},
            {"candidate_id": 1004, "pdb_id": "short", "map_resolution": 3.0, "cc_contour": 0.9},
        ]
    )
    candidates_path = tmp_path / "candidates.jsonl"
    pair_list_path = tmp_path / "pairs.jsonl"
    release_cache_path = tmp_path / "release.jsonl"
    write_jsonl(candidates_path, candidates)
    pair_records = [
        {"pdb_id": record["pdb_id"], "emdb_id": f"EMD-{index + 1}"}
        for index, record in enumerate(candidates)
    ]
    write_jsonl(pair_list_path, pair_records)
    release_records = []
    for index, record in enumerate(candidates):
        pdb_id = str(record["pdb_id"])
        release = None if pdb_id == "missing" else "2026-01-01" if pdb_id == "held" else "2025-12-31"
        release_records.append({"emdb_id": f"EMD-{index + 1}", "map_release": release})
    write_jsonl(release_cache_path, release_records)

    def fake_asset_audit(_data_root: Path, pdb_id: str) -> dict[str, object]:
        status = "short_map" if pdb_id == "short" else "eligible"
        return {"status": status, "shape_zyx": [79, 80, 80] if status == "short_map" else [80, 80, 80], "detail": ""}

    monkeypatch.setattr(split_module, "inspect_training_assets", fake_asset_audit)
    output_root = tmp_path / "split"
    split_module.freeze_split(
        argparse.Namespace(
            candidates=str(candidates_path),
            pair_list=str(pair_list_path),
            release_cache=str(release_cache_path),
            data_root=str(tmp_path / "data"),
            output_root=str(output_root),
            seed=3407,
        )
    )
    summary = json.loads((output_root / "summary.json").read_text(encoding="utf-8"))
    assert summary["splits"]["validation"]["pdb_count"] == 200
    assert summary["splits"]["calibration"]["pdb_count"] == 100
    assert summary["splits"]["train"]["pdb_count"] == 2
    assert summary["splits"]["held_out"]["pdb_count"] == 1
    assert summary["splits"]["quarantine_missing_release"]["pdb_count"] == 1
    assert summary["audit_status_counts"]["quality_rejected"] == 2
    assert summary["audit_status_counts"]["short_map"] == 1
    assert (output_root / "_COMPLETE").is_file()
    all_trainable = set()
    for name in ("train", "validation", "calibration"):
        rows = json.loads((output_root / f"{name}.json").read_text(encoding="utf-8"))
        ids = {row["pdb_id"] for row in rows}
        assert all_trainable.isdisjoint(ids)
        all_trainable.update(ids)
    assert all_trainable == set(eligible_ids)


def test_publish_pdb_splits_writes_five_unique_identity_lists(tmp_path: Path) -> None:
    """显式补建命令只按 PDB 去重来源记录, 并保留五个互斥集合。"""

    split_root = tmp_path / "split"
    split_root.mkdir()
    source_records = {
        "train": [
            {"pdb_id": "1ABC", "candidate_id": 0},
            {"pdb_id": "1abc", "candidate_id": 1},
            {"pdb_id": "2DEF", "candidate_id": 0},
        ],
        "validation": [{"pdb_id": "3ghi", "candidate_id": 0}],
        "calibration": [{"pdb_id": "4JKL", "candidate_id": 0}],
        "held_out": [
            {"pdb_id": "5mno", "candidate_id": 0},
            {"pdb_id": "5MNO", "candidate_id": 1},
        ],
        "quarantine_missing_release": [{"pdb_id": "6pqr", "candidate_id": 0}],
    }
    for split_name, records in source_records.items():
        (split_root / f"{split_name}.json").write_text(
            json.dumps(records),
            encoding="utf-8",
        )

    output_root = split_root / "pdb_split"
    split_module.publish_pdb_splits(
        argparse.Namespace(split_root=str(split_root), output_root=str(output_root))
    )

    assert json.loads((output_root / "train.json").read_text(encoding="utf-8")) == [
        "1abc",
        "2def",
    ]
    assert json.loads((output_root / "validation.json").read_text(encoding="utf-8")) == ["3ghi"]
    assert json.loads((output_root / "calibration.json").read_text(encoding="utf-8")) == ["4jkl"]
    assert json.loads((output_root / "held_out.json").read_text(encoding="utf-8")) == ["5mno"]
    assert json.loads(
        (output_root / "quarantine_missing_release.json").read_text(encoding="utf-8")
    ) == ["6pqr"]
    assert sorted(path.name for path in output_root.iterdir()) == [
        "calibration.json",
        "held_out.json",
        "quarantine_missing_release.json",
        "train.json",
        "validation.json",
    ]
