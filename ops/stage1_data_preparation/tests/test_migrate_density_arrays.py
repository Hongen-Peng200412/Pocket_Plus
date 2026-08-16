"""Stage1 完整体数组原子迁移的字段、幂等与拒绝覆盖测试。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ops.stage1_data_preparation.migrate_density_arrays import (
    MIGRATION_SPECS,
    migrate_density_directory,
)


def write_source_npz_files(density_directory: Path) -> dict[str, np.ndarray]:
    """写出四类小型来源 NPZ，并返回迁移目标数组。"""

    density_directory.mkdir(parents=True)
    shape = (1, 3, 4, 5)
    arrays = {
        "exp.npy": np.arange(np.prod(shape), dtype=np.float32).reshape(shape),
        "sim.npy": (np.arange(np.prod(shape), dtype=np.float32) + 10).reshape(shape),
        "ligand_dist.npy": np.arange(np.prod(shape), dtype=np.float16).reshape(shape),
        "union_mask.npy": (np.arange(np.prod(shape)).reshape(shape) % 3 == 0),
    }
    np.savez_compressed(
        density_directory / "exp.npz",
        grid=arrays["exp.npy"],
        voxel_size=np.asarray([1.0, 1.0, 1.0], dtype=np.float32),
        origin=np.asarray([0.0, 0.0, 0.0], dtype=np.float32),
        canonical_shape_zyx=np.asarray(shape[-3:], dtype=np.int64),
        source_tag=np.asarray("fixture"),
    )
    np.savez_compressed(
        density_directory / "sim.npz",
        grid=arrays["sim.npy"],
        voxel_size=np.asarray([1.0, 1.0, 1.0], dtype=np.float32),
        origin=np.asarray([0.0, 0.0, 0.0], dtype=np.float32),
        resolution=np.asarray(3.0, dtype=np.float32),
    )
    np.savez_compressed(
        density_directory / "ligand_dist.npz",
        distance=arrays["ligand_dist.npy"],
        grid_shape_zyx=np.asarray(shape[-3:], dtype=np.int64),
    )
    np.savez_compressed(
        density_directory / "ligand_area.npz",
        union_mask=arrays["union_mask.npy"],
        mask_0=np.asarray([[1, 1, 1]], dtype=np.int32),
        grid_shape_zyx=np.asarray(shape[-3:], dtype=np.int64),
    )
    return arrays


def test_migration_preserves_values_and_is_idempotent(tmp_path: Path) -> None:
    """首次迁移逐值保持数组和剩余字段，第二次执行只报告已完成。"""

    data_root = tmp_path / "data"
    density_directory = data_root / "density" / "1abc"
    expected_arrays = write_source_npz_files(density_directory)
    first_report = migrate_density_directory((str(data_root), "1abc"))
    assert {report["status"] for report in first_report["files"]} == {"migrated"}
    for spec in MIGRATION_SPECS:
        migrated = np.load(density_directory / spec.npy_name, mmap_mode="r", allow_pickle=False)
        np.testing.assert_array_equal(migrated, expected_arrays[spec.npy_name])
        with np.load(density_directory / spec.npz_name, allow_pickle=False) as rewritten:
            assert spec.npz_key not in rewritten.files
    with np.load(density_directory / "exp.npz", allow_pickle=False) as exp_meta:
        np.testing.assert_array_equal(exp_meta["canonical_shape_zyx"], [3, 4, 5])
        assert str(exp_meta["source_tag"].item()) == "fixture"
    with np.load(density_directory / "ligand_area.npz", allow_pickle=False) as area_meta:
        np.testing.assert_array_equal(area_meta["mask_0"], [[1, 1, 1]])

    second_report = migrate_density_directory((str(data_root), "1abc"))
    assert {report["status"] for report in second_report["files"]} == {"already_migrated"}


def test_migration_rejects_conflicting_existing_npy(tmp_path: Path) -> None:
    """来源字段仍存在时，已有 NPY 数值不一致必须停止且不得删除来源字段。"""

    data_root = tmp_path / "data"
    density_directory = data_root / "density" / "1abc"
    write_source_npz_files(density_directory)
    np.save(density_directory / "exp.npy", np.zeros((1, 3, 4, 5), dtype=np.float32))
    with pytest.raises(ValueError, match="数值不一致"):
        migrate_density_directory((str(data_root), "1abc"))
    with np.load(density_directory / "exp.npz", allow_pickle=False) as source:
        assert "grid" in source.files


def test_migration_rejects_missing_npy_after_key_removal(tmp_path: Path) -> None:
    """NPZ 已无大字段而 NPY 缺失属于不可自动修复的硬错误。"""

    data_root = tmp_path / "data"
    density_directory = data_root / "density" / "1abc"
    write_source_npz_files(density_directory)
    with np.load(density_directory / "exp.npz", allow_pickle=False) as source:
        remaining = {key: np.array(source[key], copy=True) for key in source.files if key != "grid"}
    np.savez_compressed(density_directory / "exp.npz", **remaining)
    with pytest.raises(FileNotFoundError, match="目标 .*exp.npy 不存在"):
        migrate_density_directory((str(data_root), "1abc"))
