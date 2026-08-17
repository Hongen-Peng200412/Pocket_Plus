"""把 Stage1 四类完整体数组从 NPZ 迁移到同目录 NPY。

命令入口是 ``migrate-shard`` 和 ``finalize``；函数入口分别为 :func:`run_shard`、:func:`finalize_migration` 和 :func:`migrate_npz_array`。每个数组先写入同目录临时 NPY 并逐块核对，再原子发布 NPY，最后原子替换移除大数组字段的原 NPZ。若进程在两次替换之间退出，下一次执行会识别已有 NPY 和仍含来源字段的 NPZ，继续完成剩余步骤。

本模块会修改 ``<data_root>/density/<pdb_id>`` 中的正式文件；不删除 PDB 目录，不改写 NPZ 的其他字段，不负责修改 Dataset 读取代码。
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ops.stage1_data_preparation.atomic_io import (
    atomic_write_json,
    atomic_write_text,
    fsync_directory,
)


@dataclass(frozen=True)
class ArrayMigrationSpec:
    """描述一类从 NPZ 迁移到 NPY 的稳定文件契约。

    字段:
        - npz_name: str；来源 NPZ 文件名。
        - npz_key: str；来源 NPZ 中要迁出的完整体数组字段名。
        - npy_name: str；同目录目标 NPY 文件名。
        - dtype: np.dtype；来源和目标数组必须使用的 dtype。
    """

    npz_name: str
    npz_key: str
    npy_name: str
    dtype: np.dtype[Any]


MIGRATION_SPECS = (
    ArrayMigrationSpec("exp.npz", "grid", "exp.npy", np.dtype(np.float32)),
    ArrayMigrationSpec("sim.npz", "grid", "sim.npy", np.dtype(np.float32)),
    ArrayMigrationSpec("ligand_dist.npz", "distance", "ligand_dist.npy", np.dtype(np.float16)),
    ArrayMigrationSpec("ligand_area.npz", "union_mask", "union_mask.npy", np.dtype(np.bool_)),
)
COMPARE_CHUNK_ELEMENTS = 16 * 1024 * 1024


def arrays_equal(left: np.ndarray, right: np.ndarray) -> bool:
    """以有限临时内存逐块比较两个 NumPy 数组。

    输入参数:
        - left: np.ndarray；来源数组。
        - right: np.ndarray；目标数组。

    返回值:
        - equal: bool；形状和 dtype 不同直接为假；相同后按 C 顺序分块比较，浮点/复数允许对应 NaN 相等。
    """

    if left.shape != right.shape or left.dtype != right.dtype:
        return False
    # np.ndarray (N_value,)；C 顺序展平视图；每次比较只为当前块分配有限大小的临时数组。
    left_flat = np.asarray(left).reshape(-1)
    right_flat = np.asarray(right).reshape(-1)
    for start in range(0, left_flat.size, COMPARE_CHUNK_ELEMENTS):
        stop = min(start + COMPARE_CHUNK_ELEMENTS, left_flat.size)
        left_chunk = left_flat[start:stop]
        right_chunk = right_flat[start:stop]
        if np.issubdtype(left.dtype, np.inexact):
            equal = np.array_equal(left_chunk, right_chunk, equal_nan=True)
        else:
            equal = np.array_equal(left_chunk, right_chunk)
        if not equal:
            return False
    return True


def validate_migrated_array(
    npy_path: Path,
    spec: ArrayMigrationSpec,
    expected_shape: tuple[int, ...] | None,
) -> np.memmap:
    """以内存映射打开目标 NPY，并核对 dtype、单通道四维形状和可选完整形状。

    输入参数:
        - npy_path: Path；目标 NPY 文件。
        - spec: ArrayMigrationSpec；该数组的目标 dtype 和文件契约。
        - expected_shape: tuple[int, ...] | None；来源数组形状；为 ``None`` 时只检查 ``(1, D, H, W)`` 结构。

    返回值:
        - migrated: np.memmap；只读目标数组；调用方只用于验证，不原地改写。
    """

    migrated = np.load(npy_path, mmap_mode="r", allow_pickle=False)
    if migrated.dtype != spec.dtype:
        raise TypeError(f"{npy_path}: dtype 应为 {spec.dtype}，实际 {migrated.dtype}。")
    if migrated.ndim != 4 or migrated.shape[0] != 1:
        raise ValueError(f"{npy_path}: 完整体数组必须为 (1,D,H,W)，实际 {migrated.shape}。")
    if expected_shape is not None and migrated.shape != expected_shape:
        raise ValueError(f"{npy_path}: shape 应为 {expected_shape}，实际 {migrated.shape}。")
    return migrated


def replace_npz_without_migrated_key(
    npz_path: Path,
    source_keys: tuple[str, ...],
    migrated_key: str,
    remaining: dict[str, np.ndarray],
) -> None:
    """用移除大数组字段后的 NPZ 原子替换来源文件。

    输入参数:
        - npz_path: Path；要替换的正式来源 NPZ。
        - source_keys: tuple[str, ...]；来源 NPZ 原字段顺序。
        - migrated_key: str；已经迁移到 NPY、必须从 NPZ 移除的字段。
        - remaining: dict[str, np.ndarray]；除迁出字段外的字段副本，必须逐值保持不变。

    状态变化:
        - 在来源同目录写压缩临时 NPZ，核对字段集合、顺序和值后用 ``os.replace`` 替换原文件并同步目录；替换前失败会保留旧来源文件并清理临时文件，替换后的同步或复核异常不保证回滚。
    """

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{npz_path.name}.", suffix=".tmp", dir=npz_path.parent
    )
    temporary_path = Path(temporary_name)
    expected_keys = tuple(key for key in source_keys if key != migrated_key)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.savez_compressed(handle, **remaining)
            handle.flush()
            os.fsync(handle.fileno())
        with np.load(temporary_path, allow_pickle=False) as rewritten:
            if tuple(rewritten.files) != expected_keys:
                raise ValueError(f"{temporary_path} 的字段顺序或集合不正确。")
            for key, expected in remaining.items():
                if not arrays_equal(expected, np.asarray(rewritten[key])):
                    raise ValueError(f"{temporary_path}:{key} 与来源字段不一致。")
        os.replace(temporary_path, npz_path)
        fsync_directory(npz_path.parent)
    finally:
        temporary_path.unlink(missing_ok=True)


def migrate_npz_array(density_directory: Path, spec: ArrayMigrationSpec) -> dict[str, Any]:
    """迁移一个 NPZ 大数组并返回文件级审计报告。

    输入参数:
        - density_directory: Path；单个 PDB 的 ``density/<pdb_id>`` 目录。
        - spec: ArrayMigrationSpec；来源 NPZ 字段、目标 NPY 文件和 dtype 契约。

    返回字段:
        - ``npz_name``：str；来源 NPZ 文件名。
        - ``npy_name``：str；目标 NPY 文件名。
        - ``status``：str；``source_absent``、``already_migrated`` 或 ``migrated``。
        - ``shape``：list[int]；目标存在时的 ``(1, D, H, W)`` 形状。
        - ``dtype``：str；目标存在时的 dtype 名称。
        - ``npy_size_bytes``：int；目标存在时的文件字节数。

    状态变化:
        - 只修改当前 density 目录：原子发布 NPY，并原子替换移除 ``spec.npz_key`` 的压缩 NPZ；其他 NPZ 字段逐值保持不变。
    """

    npz_path = density_directory / spec.npz_name
    npy_path = density_directory / spec.npy_name
    if not npz_path.is_file():
        if npy_path.exists():
            raise FileNotFoundError(f"{npy_path} 存在，但来源 {npz_path} 不存在。")
        return {"npz_name": spec.npz_name, "npy_name": spec.npy_name, "status": "source_absent"}

    with np.load(npz_path, allow_pickle=False) as source:
        source_keys = tuple(source.files)
        if spec.npz_key not in source_keys:
            if not npy_path.is_file():
                raise FileNotFoundError(
                    f"{npz_path} 已缺少 {spec.npz_key}，但目标 {npy_path} 不存在。"
                )
            migrated = validate_migrated_array(npy_path, spec, expected_shape=None)
            return {
                "npz_name": spec.npz_name,
                "npy_name": spec.npy_name,
                "status": "already_migrated",
                "shape": list(migrated.shape),
                "dtype": migrated.dtype.name,
                "npy_size_bytes": npy_path.stat().st_size,
            }
        # np.ndarray (1, D, H, W)；从压缩 NPZ 解出的权威完整体数组，后续逐值写入 NPY。
        source_array = np.array(source[spec.npz_key], copy=True)
        if source_array.dtype != spec.dtype:
            raise TypeError(
                f"{npz_path}:{spec.npz_key} dtype 应为 {spec.dtype}，实际 {source_array.dtype}。"
            )
        if source_array.ndim != 4 or source_array.shape[0] != 1:
            raise ValueError(
                f"{npz_path}:{spec.npz_key} 必须为 (1,D,H,W)，实际 {source_array.shape}。"
            )
        # dict[str, np.ndarray]；除迁出字段外的 NPZ 字段副本，在关闭 source 前独立复制。
        remaining = {
            key: np.array(source[key], copy=True) for key in source_keys if key != spec.npz_key
        }

    if npy_path.exists():
        migrated = validate_migrated_array(npy_path, spec, source_array.shape)
        if not arrays_equal(source_array, migrated):
            raise ValueError(f"{npy_path} 与 {npz_path}:{spec.npz_key} 数值不一致，拒绝覆盖。")
    else:
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{npy_path.name}.", suffix=".tmp", dir=density_directory
        )
        temporary_path = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                np.save(handle, source_array, allow_pickle=False)
                handle.flush()
                os.fsync(handle.fileno())
            migrated = validate_migrated_array(temporary_path, spec, source_array.shape)
            if not arrays_equal(source_array, migrated):
                raise ValueError(f"临时 NPY 与 {npz_path}:{spec.npz_key} 数值不一致。")
            del migrated
            os.replace(temporary_path, npy_path)
            fsync_directory(density_directory)
        finally:
            temporary_path.unlink(missing_ok=True)

    replace_npz_without_migrated_key(npz_path, source_keys, spec.npz_key, remaining)
    with np.load(npz_path, allow_pickle=False) as rewritten:
        if tuple(rewritten.files) != tuple(key for key in source_keys if key != spec.npz_key):
            raise ValueError(f"{npz_path} 重写后的字段顺序或集合不正确。")
        for key, expected in remaining.items():
            actual = np.asarray(rewritten[key])
            if not arrays_equal(expected, actual):
                raise ValueError(f"{npz_path}:{key} 在重写后发生变化。")
    migrated = validate_migrated_array(npy_path, spec, source_array.shape)
    return {
        "npz_name": spec.npz_name,
        "npy_name": spec.npy_name,
        "status": "migrated",
        "shape": list(migrated.shape),
        "dtype": migrated.dtype.name,
        "npy_size_bytes": npy_path.stat().st_size,
    }


def migrate_density_directory(argument: tuple[str, str]) -> dict[str, Any]:
    """迁移一个 PDB density 目录中的四类数组并返回 PDB 级审计报告。

    输入参数:
        - argument: tuple[str, str]；依次为 A-G 数据根目录和小写 PDB identity。

    返回值:
        - report: dict[str, Any]；包含 ``pdb_id`` 和四个 ``MIGRATION_SPECS`` 文件报告。
    """

    data_root, pdb_id = argument
    density_directory = Path(data_root) / "density" / pdb_id
    file_reports = [migrate_npz_array(density_directory, spec) for spec in MIGRATION_SPECS]
    return {"pdb_id": pdb_id, "files": file_reports}


def run_shard(arguments: argparse.Namespace) -> None:
    """让一个 Slurm 数组元素以多进程迁移其稳定分配的 PDB 目录。

    输入参数:
        - arguments: argparse.Namespace；包含 data-root、run-root、shard-count、shard-index 和 workers。

    状态变化:
        - 按 density 子目录名字典序和分片切片确定 PDB 集合；最多启动 ``workers`` 个进程；在 ``run_root/shards`` 原子写入分片状态 JSON。

    失败语义:
        - 参数无效或 run root 已有 ``_COMPLETE`` 时拒绝修改共享资产；空分片只写空报告。
    """

    data_root = Path(arguments.data_root).resolve()
    run_root = Path(arguments.run_root).resolve()
    shard_count = int(arguments.shard_count)
    shard_index = int(arguments.shard_index)
    workers = int(arguments.workers)
    if shard_count <= 0 or not 0 <= shard_index < shard_count or workers <= 0:
        raise ValueError("shard_count、shard_index 或 workers 无效。")
    if (run_root / "_COMPLETE").exists():
        raise FileExistsError(f"迁移已完成，不允许再次修改共享资产：{run_root}")
    density_root = data_root / "density"
    # list[str]；按 PDB identity 字典序冻结的全部 density 子目录，切片后每个 PDB 只属于一个分片。
    all_pdb_ids = sorted(path.name.lower() for path in density_root.iterdir() if path.is_dir())
    assigned_pdb_ids = all_pdb_ids[shard_index::shard_count]
    process_count = min(workers, len(assigned_pdb_ids))
    process_arguments = [(str(data_root), pdb_id) for pdb_id in assigned_pdb_ids]
    if process_count == 0:
        reports: list[dict[str, Any]] = []
    elif process_count == 1:
        reports = [migrate_density_directory(process_arguments[0])]
    else:
        with ProcessPoolExecutor(max_workers=process_count) as executor:
            reports = list(executor.map(migrate_density_directory, process_arguments))
    state_path = run_root / "shards" / f"shard_{shard_index:03d}_of_{shard_count:03d}.json"
    atomic_write_json(
        state_path,
        {
            "schema_version": 1,
            "shard_index": shard_index,
            "shard_count": shard_count,
            "data_root": str(data_root),
            "assigned_pdb_count": len(assigned_pdb_ids),
            "pdb_reports": reports,
        },
    )
    print(json.dumps({"shard_index": shard_index, "pdb_count": len(reports)}, ensure_ascii=False))


def finalize_migration(arguments: argparse.Namespace) -> None:
    """合并全部分片状态，复查正式目录并发布迁移完成标记。

    输入参数:
        - arguments: argparse.Namespace；包含 data-root、run-root 和 shard-count。

    状态变化:
        - 核对全部 PDB 是否恰由一个分片报告；逐文件确认来源 NPZ 已移除迁出字段且目标 NPY 可 mmap；最后写入 summary 和 ``_COMPLETE``。

    失败语义:
        - 分片缺失、重复、PDB 集合不一致或任一来源字段仍存在时不发布完成标记。
    """

    data_root = Path(arguments.data_root).resolve()
    run_root = Path(arguments.run_root).resolve()
    shard_count = int(arguments.shard_count)
    density_root = data_root / "density"
    expected_pdb_ids = sorted(path.name.lower() for path in density_root.iterdir() if path.is_dir())
    reports_by_pdb: dict[str, dict[str, Any]] = {}
    for shard_index in range(shard_count):
        state_path = run_root / "shards" / f"shard_{shard_index:03d}_of_{shard_count:03d}.json"
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state["shard_index"] != shard_index or state["shard_count"] != shard_count:
            raise ValueError(f"分片身份不匹配：{state_path}")
        for report in state["pdb_reports"]:
            pdb_id = str(report["pdb_id"])
            if pdb_id in reports_by_pdb:
                raise ValueError(f"PDB 被多个分片重复处理：{pdb_id}")
            reports_by_pdb[pdb_id] = report
    if sorted(reports_by_pdb) != expected_pdb_ids:
        missing = sorted(set(expected_pdb_ids).difference(reports_by_pdb))[:20]
        extra = sorted(set(reports_by_pdb).difference(expected_pdb_ids))[:20]
        raise ValueError(f"分片集合不完整，missing={missing}, extra={extra}")

    status_counts = {"migrated": 0, "already_migrated": 0, "source_absent": 0}
    total_npy_bytes = 0
    for pdb_id in expected_pdb_ids:
        density_directory = density_root / pdb_id
        for spec in MIGRATION_SPECS:
            npz_path = density_directory / spec.npz_name
            npy_path = density_directory / spec.npy_name
            if not npz_path.exists():
                if npy_path.exists():
                    raise FileNotFoundError(f"来源缺失但 NPY 存在：{npy_path}")
                continue
            with np.load(npz_path, allow_pickle=False) as source:
                if spec.npz_key in source.files:
                    raise ValueError(f"{npz_path} 仍含已迁出字段 {spec.npz_key}。")
            migrated = validate_migrated_array(npy_path, spec, expected_shape=None)
            total_npy_bytes += npy_path.stat().st_size
            del migrated
        for file_report in reports_by_pdb[pdb_id]["files"]:
            status_counts[str(file_report["status"])] += 1

    summary = {
        "schema_version": 1,
        "data_root": str(data_root),
        "pdb_count": len(expected_pdb_ids),
        "target_file_count": len(expected_pdb_ids) * len(MIGRATION_SPECS),
        "status_counts": status_counts,
        "published_npy_bytes": total_npy_bytes,
        "shard_count": shard_count,
    }
    atomic_write_json(run_root / "summary.json", summary)
    atomic_write_text(run_root / "_COMPLETE", "")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main() -> None:
    """解析 ``migrate-shard`` 或 ``finalize`` 子命令并调用正式入口。

    返回值:
        - None；成功时写入迁移报告，失败时保留异常供任务系统报告。
    """

    parser = argparse.ArgumentParser(description="原子迁移 Stage1 完整体数组。")
    subparsers = parser.add_subparsers(dest="command", required=True)
    shard_parser = subparsers.add_parser("migrate-shard")
    shard_parser.add_argument("--data-root", required=True)
    shard_parser.add_argument("--run-root", required=True)
    shard_parser.add_argument("--shard-count", type=int, required=True)
    shard_parser.add_argument("--shard-index", type=int, required=True)
    shard_parser.add_argument("--workers", type=int, required=True)
    finalize_parser = subparsers.add_parser("finalize")
    finalize_parser.add_argument("--data-root", required=True)
    finalize_parser.add_argument("--run-root", required=True)
    finalize_parser.add_argument("--shard-count", type=int, required=True)
    arguments = parser.parse_args()
    if arguments.command == "migrate-shard":
        run_shard(arguments)
    else:
        finalize_migration(arguments)


if __name__ == "__main__":
    main()
