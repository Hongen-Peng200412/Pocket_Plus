"""验证正式 Stage1 训练准备产物的筛除与复制工具。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import ops.materialize_filtered_stage1_preparation as materialize_module


def _write_box_file(path: Path, pdb_id: str, occurrence_id: int) -> None:
    """写出一个字段完整、数组规模很小的单 PDB BOX 文件。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        pdb_id=np.asarray(pdb_id),
        occurrence_id=np.asarray([occurrence_id], dtype=np.int32),
        center_start_zyx=np.asarray([[1, 2, 3]], dtype=np.int32),
        bias_start_zyx=np.arange(90, dtype=np.int32).reshape(1, 30, 3),
        context_start_zyx=np.asarray([[4, 5, 6]], dtype=np.int32),
    )


def _write_source_preparation(source_root: Path) -> None:
    """写出同时包含保留 PDB 和排除 PDB 的最小源 preparation。"""

    inventory_root = source_root / "inventory"
    split_root = source_root / "split"
    box_pool_root = source_root / "box_pool"
    inventory_root.mkdir(parents=True)
    split_root.mkdir()
    box_pool_root.mkdir()

    # final_keep_list 保留原始 JSON 文本的字段顺序和附加字段。
    keep_list_records = [
        {"pdb_id": "1aaa", "candidate_id": 11, "note": "train-retained"},
        {"pdb_id": "1bad", "candidate_id": 12, "note": "train-excluded"},
        {"pdb_id": "2bbb", "candidate_id": 21, "note": "validation-retained"},
        {"pdb_id": "2bad", "candidate_id": 22, "note": "validation-excluded"},
    ]
    (inventory_root / "final_keep_list.jsonl").write_text(
        "".join(
            json.dumps(record, ensure_ascii=False) + "\n"
            for record in keep_list_records
        ),
        encoding="utf-8",
    )

    split_records = {
        "train.json": keep_list_records[:2],
        "validation.json": keep_list_records[2:],
        "calibration.json": [
            {"pdb_id": "3ccc", "candidate_id": 31, "note": "calibration"}
        ],
        "held_out_pool.json": [
            {"pdb_id": "4ddd", "candidate_id": 41, "note": "held-out"}
        ],
    }
    for filename, records in split_records.items():
        (split_root / filename).write_text(
            json.dumps(records, ensure_ascii=False),
            encoding="utf-8",
        )
    (split_root / "_COMPLETE").write_text("", encoding="utf-8")

    manifest = {
        "schema_version": 1,
        "splits": {
            "train": [
                {"pdb_id": "1aaa", "path": "train/1aaa.npz"},
                {"pdb_id": "1bad", "path": "train/1bad.npz"},
            ],
            "validation": [
                {"pdb_id": "2bbb", "path": "validation/2bbb.npz"},
                {"pdb_id": "2bad", "path": "validation/2bad.npz"},
            ],
        },
    }
    (box_pool_root / "manifest.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )
    _write_box_file(box_pool_root / "train" / "1aaa.npz", "1aaa", 11)
    _write_box_file(box_pool_root / "train" / "1bad.npz", "1bad", 12)
    _write_box_file(box_pool_root / "validation" / "2bbb.npz", "2bbb", 21)
    _write_box_file(box_pool_root / "validation" / "2bad.npz", "2bad", 22)

    # validation_pdb_id 的旧编号 0 指向 2bbb，旧编号 1 指向待排除的 2bad。
    np.savez_compressed(
        box_pool_root / "validation_selection.npz",
        validation_pdb_id=np.asarray(["2bbb", "2bad"], dtype="U4"),
        center_pdb_index=np.asarray([0, 1], dtype=np.int32),
        center_occurrence_id=np.asarray([21, 22], dtype=np.int32),
        bias_pdb_index=np.asarray([1, 0, 1], dtype=np.int32),
        bias_occurrence_id=np.asarray([22, 21, 22], dtype=np.int32),
        bias_candidate_index=np.asarray([2, 3, 4], dtype=np.int16),
        context_pdb_index=np.asarray([0, 1], dtype=np.int32),
        context_candidate_index=np.asarray([5, 6], dtype=np.int32),
    )
    (box_pool_root / "config.json").write_text(
        json.dumps({"box_shape_zyx": [80, 80, 80], "seed": 20260721}),
        encoding="utf-8",
    )
    (box_pool_root / "_COMPLETE").write_text("", encoding="utf-8")


def _redirect_fixed_paths(
    monkeypatch: pytest.MonkeyPatch,
    source_root: Path,
    target_root: Path,
    excluded_pdb_ids: frozenset[str],
) -> None:
    """把脚本顶部的固定路径和排除集合临时指向 pytest 临时目录。"""

    monkeypatch.setattr(materialize_module, "SOURCE_PREPARATION_ROOT", source_root)
    monkeypatch.setattr(
        materialize_module,
        "SOURCE_FINAL_KEEP_LIST",
        source_root / "inventory" / "final_keep_list.jsonl",
    )
    monkeypatch.setattr(
        materialize_module,
        "SOURCE_SPLIT_ROOT",
        source_root / "split",
    )
    monkeypatch.setattr(
        materialize_module,
        "SOURCE_BOX_POOL_ROOT",
        source_root / "box_pool",
    )
    monkeypatch.setattr(materialize_module, "TARGET_PREPARATION_ROOT", target_root)
    monkeypatch.setattr(
        materialize_module,
        "TARGET_FINAL_KEEP_LIST",
        target_root / "final_keep_list.jsonl",
    )
    monkeypatch.setattr(
        materialize_module,
        "TARGET_SPLIT_ROOT",
        target_root / "split",
    )
    monkeypatch.setattr(
        materialize_module,
        "TARGET_BOX_POOL_ROOT",
        target_root / "box_pool",
    )
    monkeypatch.setattr(
        materialize_module,
        "EXCLUDED_PDB_IDS",
        excluded_pdb_ids,
    )


def test_materialize_filters_every_reference_and_preserves_retained_box_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """筛除必须同步覆盖所有清单、BOX 文件引用和三类冻结验证请求。"""

    source_root = tmp_path / "source"
    target_root = tmp_path / "target"
    _write_source_preparation(source_root)
    target_root.mkdir()
    _redirect_fixed_paths(
        monkeypatch,
        source_root,
        target_root,
        frozenset({"1bad", "2bad"}),
    )

    retained_train_bytes = (source_root / "box_pool/train/1aaa.npz").read_bytes()
    retained_validation_bytes = (
        source_root / "box_pool/validation/2bbb.npz"
    ).read_bytes()
    source_config_bytes = (source_root / "box_pool/config.json").read_bytes()

    materialize_module.materialize_filtered_preparation()

    keep_list = [
        json.loads(text)
        for text in (target_root / "final_keep_list.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert [record["pdb_id"] for record in keep_list] == ["1aaa", "2bbb"]
    assert [record["note"] for record in keep_list] == [
        "train-retained",
        "validation-retained",
    ]

    expected_split_pdb_ids = {
        "train.json": ["1aaa"],
        "validation.json": ["2bbb"],
        "calibration.json": ["3ccc"],
        "held_out_pool.json": ["4ddd"],
    }
    for filename, expected_pdb_ids in expected_split_pdb_ids.items():
        records = json.loads((target_root / "split" / filename).read_text())
        assert [record["pdb_id"] for record in records] == expected_pdb_ids

    manifest = json.loads(
        (target_root / "box_pool/manifest.json").read_text(encoding="utf-8")
    )
    assert manifest == {
        "schema_version": 1,
        "splits": {
            "train": [{"pdb_id": "1aaa", "path": "train/1aaa.npz"}],
            "validation": [
                {"pdb_id": "2bbb", "path": "validation/2bbb.npz"}
            ],
        },
    }
    assert (target_root / "box_pool/train/1aaa.npz").read_bytes() == (
        retained_train_bytes
    )
    assert (target_root / "box_pool/validation/2bbb.npz").read_bytes() == (
        retained_validation_bytes
    )
    assert not (target_root / "box_pool/train/1bad.npz").exists()
    assert not (target_root / "box_pool/validation/2bad.npz").exists()
    assert (target_root / "box_pool/config.json").read_bytes() == (
        source_config_bytes
    )

    with np.load(
        target_root / "box_pool/validation_selection.npz",
        allow_pickle=False,
    ) as selection:
        assert set(selection.files) == materialize_module.VALIDATION_SELECTION_FIELDS
        np.testing.assert_array_equal(
            selection["validation_pdb_id"],
            np.asarray(["2bbb"], dtype="U4"),
        )
        np.testing.assert_array_equal(
            selection["center_pdb_index"],
            np.asarray([0], dtype=np.int32),
        )
        np.testing.assert_array_equal(
            selection["center_occurrence_id"],
            np.asarray([21], dtype=np.int32),
        )
        np.testing.assert_array_equal(
            selection["bias_pdb_index"],
            np.asarray([0], dtype=np.int32),
        )
        np.testing.assert_array_equal(
            selection["bias_occurrence_id"],
            np.asarray([21], dtype=np.int32),
        )
        np.testing.assert_array_equal(
            selection["bias_candidate_index"],
            np.asarray([3], dtype=np.int16),
        )
        np.testing.assert_array_equal(
            selection["context_pdb_index"],
            np.asarray([0], dtype=np.int32),
        )
        np.testing.assert_array_equal(
            selection["context_candidate_index"],
            np.asarray([5], dtype=np.int32),
        )

    assert (target_root / "split/_COMPLETE").is_file()
    assert (target_root / "box_pool/_COMPLETE").is_file()


def test_missing_excluded_pdb_only_prints_notice(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """未出现在源产物中的排除编号只提示，不阻止正式产物生成。"""

    source_root = tmp_path / "source"
    target_root = tmp_path / "target"
    _write_source_preparation(source_root)
    target_root.mkdir()
    _redirect_fixed_paths(
        monkeypatch,
        source_root,
        target_root,
        frozenset({"9zzz"}),
    )

    materialize_module.materialize_filtered_preparation()

    output = capsys.readouterr().out
    assert "9zzz" in output
    assert (target_root / "split/_COMPLETE").is_file()
    assert (target_root / "box_pool/_COMPLETE").is_file()


def test_existing_target_output_is_not_overwritten(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """任一正式输出入口已经存在时，脚本必须在读取和复制前停止。"""

    source_root = tmp_path / "source"
    target_root = tmp_path / "target"
    _write_source_preparation(source_root)
    target_root.mkdir()
    existing_keep_list = target_root / "final_keep_list.jsonl"
    existing_keep_list.write_text("keep me", encoding="utf-8")
    _redirect_fixed_paths(
        monkeypatch,
        source_root,
        target_root,
        frozenset({"1bad"}),
    )

    with pytest.raises(FileExistsError, match="拒绝覆盖"):
        materialize_module.materialize_filtered_preparation()

    assert existing_keep_list.read_text(encoding="utf-8") == "keep me"
    assert not (target_root / "split").exists()
    assert not (target_root / "box_pool").exists()


def test_invalid_validation_reference_stops_before_target_creation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """冻结验证请求引用不存在的 PDB 表编号时不得留下部分输出。"""

    source_root = tmp_path / "source"
    target_root = tmp_path / "target"
    _write_source_preparation(source_root)
    target_root.mkdir()
    _redirect_fixed_paths(
        monkeypatch,
        source_root,
        target_root,
        frozenset({"1bad"}),
    )
    selection_path = source_root / "box_pool/validation_selection.npz"
    with np.load(selection_path, allow_pickle=False) as selection:
        arrays = {name: np.asarray(selection[name]) for name in selection.files}
    arrays["center_pdb_index"] = np.asarray([0, 2], dtype=np.int32)
    np.savez_compressed(selection_path, **arrays)

    with pytest.raises(IndexError, match="超出 validation_pdb_id"):
        materialize_module.materialize_filtered_preparation()

    assert not (target_root / "final_keep_list.jsonl").exists()
    assert not (target_root / "split").exists()
    assert not (target_root / "box_pool").exists()
