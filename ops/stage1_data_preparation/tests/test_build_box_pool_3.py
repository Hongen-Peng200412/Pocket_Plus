"""迁移后元数据驱动的 Stage1 v3 BOX pool 测试。"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ops.stage1_data_preparation.build_box_pool_3 import build_migrated_pdb_box_pool


def test_box_pool_reads_shape_without_exp_grid(tmp_path: Path) -> None:
    """exp.npz 不含 grid 时仍生成确定性 0:5:5 所需候选字段。"""

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
