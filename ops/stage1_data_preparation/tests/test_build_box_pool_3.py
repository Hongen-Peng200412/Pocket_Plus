"""迁移后元数据驱动的 Stage1 v3 BOX pool 测试。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ops.stage1_data_preparation.build_box_pool_3 import (
    build_migrated_pdb_box_pool,
    finalize_box_pool,
)


def test_box_pool_reads_shape_without_exp_grid(tmp_path: Path) -> None:
    """exp.npz 不含 grid 时仍生成训练与验证所需的确定性候选字段。"""

    data_root = tmp_path / "data"
    density_directory = data_root / "density" / "1abc"
    parse_directory = data_root / "parse" / "1abc"
    density_directory.mkdir(parents=True)
    parse_directory.mkdir(parents=True)
    np.savez_compressed(
        density_directory / "exp.npz",
        canonical_shape_zyx=np.asarray([80, 80, 80], dtype=np.int64),
        voxel_size=np.asarray([1.0, 1.0, 1.0], dtype=np.float32),
        origin=np.asarray([0.0, 0.0, 0.0], dtype=np.float32),
    )
    np.savez_compressed(
        density_directory / "ligand_area.npz",
        schema_version=np.asarray(3, dtype=np.int32),
        mask_0=np.asarray([[39, 39, 39], [40, 40, 40]], dtype=np.int32),
        grid_shape_zyx=np.asarray([80, 80, 80], dtype=np.int64),
    )
    np.savez_compressed(
        parse_directory / "receptor_tokens.npz",
        coords=np.asarray([[10.0, 10.0, 10.0], [70.0, 70.0, 70.0]], dtype=np.float32),
    )
    first = build_migrated_pdb_box_pool(data_root, "1abc", "train", 3407)
    second = build_migrated_pdb_box_pool(data_root, "1abc", "train", 3407)
    assert first["occurrence_id"].tolist() == [0]
    assert first["center_start_zyx"].shape == (1, 3)
    assert first["bias_start_zyx"].shape == (1, 30, 3)
    assert first["context_start_zyx"].shape == (500, 3)
    for field_name in ("occurrence_id", "center_start_zyx", "bias_start_zyx", "context_start_zyx"):
        np.testing.assert_array_equal(first[field_name], second[field_name])


def test_finalize_keeps_train_zero_five_five_and_freezes_validation_zero_one_one(
    tmp_path: Path,
) -> None:
    """finalize 分别发布训练 0:5:5 配置和冻结验证 0:1:1 索引。"""

    output_root = tmp_path / "box_pool"
    state_root = tmp_path / "state"
    state_root.mkdir()
    train_split = tmp_path / "train.json"
    validation_split = tmp_path / "validation.json"
    train_split.write_text(json.dumps([{"pdb_id": "1abc"}]), encoding="utf-8")
    validation_split.write_text(json.dumps([{"pdb_id": "2def"}]), encoding="utf-8")

    for split_name, pdb_id in (("train", "1abc"), ("validation", "2def")):
        pool_directory = output_root / split_name
        pool_directory.mkdir(parents=True)
        np.savez(
            pool_directory / f"{pdb_id}.npz",
            pdb_id=np.asarray(pdb_id),
            occurrence_id=np.asarray([7, 11], dtype=np.int32),
            center_start_zyx=np.zeros((2, 3), dtype=np.int32),
            bias_start_zyx=np.arange(2 * 30 * 3, dtype=np.int32).reshape(2, 30, 3),
            context_start_zyx=np.arange(4 * 3, dtype=np.int32).reshape(4, 3),
        )
    (state_root / "shard_000_of_001.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "shard_index": 0,
                "shard_count": 1,
                "assigned_count": 2,
                "pdb_reports": [
                    {"split": "train", "pdb_id": "1abc", "status": "published"},
                    {"split": "validation", "pdb_id": "2def", "status": "published"},
                ],
            }
        ),
        encoding="utf-8",
    )

    finalize_box_pool(
        argparse.Namespace(
            data_root=str(tmp_path / "data"),
            train_split=str(train_split),
            validation_split=str(validation_split),
            output_root=str(output_root),
            state_root=str(state_root),
            shard_count=1,
            seed=3407,
        )
    )

    config = json.loads((output_root / "config.json").read_text(encoding="utf-8"))
    summary = json.loads((output_root / "summary.json").read_text(encoding="utf-8"))
    assert config["entry_ratio"] == {"center": 0, "bias": 5, "context": 5}
    assert config["validation_entry_ratio"] == {"center": 0, "bias": 1, "context": 1}
    assert summary["validation_selection"] == {
        "pdb_count": 1,
        "center_count": 0,
        "bias_count": 2,
        "context_count": 2,
    }
    with np.load(output_root / "validation_selection.npz", allow_pickle=False) as selection:
        assert selection["validation_pdb_id"].tolist() == [b"2def"]
        assert selection["center_pdb_index"].shape == (0,)
        assert selection["bias_pdb_index"].shape == (2,)
        assert selection["context_pdb_index"].shape == (2,)
