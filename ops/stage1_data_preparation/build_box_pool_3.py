"""从迁移后的 Stage1 V3 正式资产并行生成并发布 0:5:5 BOX pool。

命令入口是 ``build-shard`` 和 ``finalize``；函数入口分别是 :func:`run_shard`、:func:`finalize_box_pool` 和 :func:`build_migrated_pdb_box_pool`。本模块只读取迁移后的 ``exp.npz`` 几何元数据、occurrence 稀疏 mask 和 ``receptor_tokens.npz`` 坐标，生成完整图内真实 80³ BOX 的 ZYX 起点。完整图形状来自 ``exp.npz:canonical_shape_zyx``，不依赖已经迁到 ``exp.npy`` 的 ``grid`` 字段；第二版 ``stage1_preparation_box_pool_2`` 不会被读取、覆盖或删除。

产物边界:
    - 每个 PDB 一个非压缩 NPZ：``pdb_id`` 为字符串标量；``occurrence_id`` 为 int32 ``(O,)``；``center_start_zyx`` 为 int32 ``(O,3)``；``bias_start_zyx`` 为 int32 ``(O,30,3)``；``context_start_zyx`` 为 int32 ``(C,3)``，最后一维均为完整图零基 ZYX 起点。
    - 分片状态 JSON 的顶层字段为 ``schema_version``、``shard_index``、``shard_count``、``assigned_count`` 和 ``pdb_reports``；每条报告含 ``split``、``pdb_id``、``status``、``occurrence_count``、``context_count``。
    - ``finalize`` 只在分片集合完整且每个 PDB NPZ 具备上述字段时发布 ``manifest.json``、``validation_selection.npz``、``config.json``、``summary.json`` 和 ``_COMPLETE``；manifest 的 ``splits`` 为 train/validation 路径清单，summary 记录请求数、发布数、零 context 数和冻结选择计数。
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from ops.stage1_data_preparation.atomic_io import atomic_save_npz, atomic_write_json, atomic_write_text
from ops.stage1_data_preparation.utils import (
    build_occurrence_pool_rows,
    derive_pdb_seed,
    freeze_validation_selection,
    generate_context_starts,
    load_occurrence_masks,
)
from src.datasets.stage1_requests import (
    BOX_POOL_MANIFEST_FILENAME,
    STAGE1_BOX_SHAPE_ZYX,
    resolve_stage1_start,
)


CENTER_PER_OCCURRENCE = 0
BIAS_PER_OCCURRENCE = 5
CONTEXT_PER_OCCURRENCE = 5
CONTEXT_MIN_CORE_ATOMS = 0
EXTRA_BIAS_DRIFT_MAX_ANGSTROM = 3.0
BIAS_CANDIDATES_PER_OCCURRENCE = 30
CONTEXT_TARGET = 500
CONTEXT_MAX_ATTEMPTS = 3000


def load_split_pdb_ids(path: Path) -> tuple[str, ...]:
    """从 split JSON 候选记录提取唯一 PDB identity 并排序。

    输入参数:
        - path: Path；顶层必须是候选记录 JSON list；字典记录从 ``pdb_id`` 字段读取 identity。

    返回值:
        - pdb_ids: tuple[str, ...]；去重、去空白、转小写并按字典序排列的 PDB identity。

    失败语义:
        - 顶层不是 list、没有有效 identity 或出现空 identity 时抛出异常；函数不保留候选重复项。
    """

    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list):
        raise TypeError(f"{path} 顶层必须是 JSON list。")
    pdb_ids = {
        str(record.get("pdb_id", "")).strip().lower()
        for record in value
        if isinstance(record, dict)
    }
    if "" in pdb_ids or not pdb_ids:
        raise ValueError(f"{path} 缺少有效 PDB identity。")
    return tuple(sorted(pdb_ids))


def build_migrated_pdb_box_pool(
    data_root: Path,
    pdb_id: str,
    split_name: str,
    seed: int,
) -> dict[str, np.ndarray]:
    """从迁移后正式资产构造一个 PDB 的 center、bias 和 context 起点池。

    输入参数:
        - data_root: Path；A-G 正式资产根目录，包含 ``density`` 和 ``parse``。
        - pdb_id: str；当前 PDB identity，目录名和输出字段使用同一小写值。
        - split_name: str；``train`` 或 ``validation``，参与 PDB 独立 seed 派生。
        - seed: int；box pool 的全局基准 seed。

    返回字段:
        - ``pdb_id``：字符串标量；当前 PDB identity。
        - ``occurrence_id``：int32 ``(O,)``；按升序排列的 occurrence identity。
        - ``center_start_zyx``：int32 ``(O, 3)``；逐 occurrence 的完整图 ZYX 居中 BOX 起点。
        - ``bias_start_zyx``：int32 ``(O, 30, 3)``；逐 occurrence 的 30 个 bias BOX 起点，最后一维为 ZYX。
        - ``context_start_zyx``：int32 ``(C, 3)``；当前 PDB 的均匀合法 context 起点，最多 500 个。

    读取契约:
        - ``exp.npz`` 提供 schema 2 的 ``canonical_shape_zyx``、``voxel_size`` 和 ``origin``；``ligand_area.npz`` 提供 schema 3 occurrence mask；``receptor_tokens.npz:coords`` 提供世界 XYZ 受体坐标。
    """

    density_directory = data_root / "density" / pdb_id
    with np.load(density_directory / "exp.npz", allow_pickle=False) as exp_meta:
        grid_shape_zyx = np.asarray(exp_meta["canonical_shape_zyx"], dtype=np.int64)
        voxel_size_xyz = np.asarray(exp_meta["voxel_size"], dtype=np.float32)
        origin_xyz = np.asarray(exp_meta["origin"], dtype=np.float32)
    resolve_stage1_start((0, 0, 0), grid_shape_zyx)
    # dict[int, np.ndarray int32]；occurrence identity 到完整图非空 ZYX voxel index 的映射。
    occurrence_masks = load_occurrence_masks(
        density_directory / "ligand_area.npz", grid_shape_zyx
    )
    if not occurrence_masks:
        raise ValueError(f"{pdb_id}: ligand_area.npz 不含 occurrence mask。")
    with np.load(data_root / "parse" / pdb_id / "receptor_tokens.npz", allow_pickle=False) as receptor:
        receptor_coords_xyz = np.asarray(receptor["coords"], dtype=np.float32)
    random_generator = np.random.default_rng(derive_pdb_seed(seed, split_name, pdb_id))
    occurrence_rows = build_occurrence_pool_rows(
        occurrence_masks,
        grid_shape_zyx,
        random_generator,
        voxel_size_world=voxel_size_xyz,
        extra_bias_drift_max_angstrom=EXTRA_BIAS_DRIFT_MAX_ANGSTROM,
        bias_candidates_per_occurrence=BIAS_CANDIDATES_PER_OCCURRENCE,
    )
    context_start_zyx = generate_context_starts(
        receptor_coords_world=receptor_coords_xyz,
        full_origin_world=origin_xyz,
        voxel_size_world=voxel_size_xyz,
        full_shape_zyx=grid_shape_zyx,
        rng=random_generator,
        target_count=CONTEXT_TARGET,
        max_attempts=CONTEXT_MAX_ATTEMPTS,
        min_core_atoms=CONTEXT_MIN_CORE_ATOMS,
    )
    return {
        "pdb_id": np.asarray(pdb_id),
        **occurrence_rows,
        "context_start_zyx": context_start_zyx,
    }


def build_one_pool(argument: tuple[str, str, str, str, int]) -> dict[str, Any]:
    """生成并原子发布一个 PDB 起点池，返回分片汇总字段。

    输入参数:
        - argument: tuple[str, str, str, str, int]；依次为 ``data_root``、``output_root``、``split_name``、``pdb_id`` 和全局 seed。

    返回值:
        - report: dict[str, Any]；包含 split、PDB identity、``published`` 状态、occurrence 数和 context 数；由分片状态 JSON 收集。

    文件副作用:
        - 在 ``output_root/<split_name>/<pdb_id>.npz`` 原子写入非压缩起点池；已有 ``_COMPLETE`` 的 pool 不应进入该入口。
    """

    data_root, output_root, split_name, pdb_id, seed = argument
    pool = build_migrated_pdb_box_pool(Path(data_root), pdb_id, split_name, seed)
    destination = Path(output_root) / split_name / f"{pdb_id}.npz"
    atomic_save_npz(destination, pool, compressed=False)
    return {
        "split": split_name,
        "pdb_id": pdb_id,
        "status": "published",
        "occurrence_count": int(pool["occurrence_id"].shape[0]),
        "context_count": int(pool["context_start_zyx"].shape[0]),
    }


def run_shard(arguments: argparse.Namespace) -> None:
    """让一个 Slurm 数组元素生成并记录其分配到的 train、validation PDB。

    输入参数:
        - arguments: argparse.Namespace；必须包含 data/output/state root、两个 split JSON、``shard_count``、``shard_index``、``workers`` 和 ``seed``。

    状态变化:
        - 按 ``tagged_ids[shard_index::shard_count]`` 分配 PDB；使用最多 ``workers`` 个进程生成 NPZ；在 state root 原子写入一个分片 JSON。

    失败语义:
        - 分片参数无效、train/validation 有 PDB 重叠或输出已经发布时直接失败；空分片只写零报告，不伪造 PDB 产物。
    """

    shard_count = int(arguments.shard_count)
    shard_index = int(arguments.shard_index)
    workers = int(arguments.workers)
    if shard_count <= 0 or not 0 <= shard_index < shard_count or workers <= 0:
        raise ValueError("shard_count、shard_index 或 workers 无效。")
    output_root = Path(arguments.output_root).resolve()
    state_root = Path(arguments.state_root).resolve()
    if (output_root / "_COMPLETE").exists():
        raise FileExistsError(f"BOX pool 已完成，不允许覆盖：{output_root}")
    train_ids = load_split_pdb_ids(Path(arguments.train_split))
    validation_ids = load_split_pdb_ids(Path(arguments.validation_split))
    overlap = sorted(set(train_ids).intersection(validation_ids))
    if overlap:
        raise ValueError(f"train 与 validation 共享 PDB：{overlap[:20]}")
    tagged_ids = [("train", pdb_id) for pdb_id in train_ids]
    tagged_ids.extend(("validation", pdb_id) for pdb_id in validation_ids)
    assigned = tagged_ids[shard_index::shard_count]
    process_arguments = [
        (arguments.data_root, str(output_root), split_name, pdb_id, int(arguments.seed))
        for split_name, pdb_id in assigned
    ]
    process_count = min(workers, len(process_arguments))
    if process_count == 0:
        reports: list[dict[str, Any]] = []
    elif process_count == 1:
        reports = [build_one_pool(process_arguments[0])]
    else:
        with ProcessPoolExecutor(max_workers=process_count) as executor:
            reports = list(executor.map(build_one_pool, process_arguments))
    state_path = state_root / f"shard_{shard_index:03d}_of_{shard_count:03d}.json"
    atomic_write_json(
        state_path,
        {
            "schema_version": 1,
            "shard_index": shard_index,
            "shard_count": shard_count,
            "assigned_count": len(assigned),
            "pdb_reports": reports,
        },
    )
    print(json.dumps({"shard_index": shard_index, "pdb_count": len(reports)}, ensure_ascii=False))


def finalize_box_pool(arguments: argparse.Namespace) -> None:
    """核对全部分片和 PDB NPZ，并原子发布 pool 的最终索引与完成标记。

    输入参数:
        - arguments: argparse.Namespace；必须包含 data/split/output/state root、``shard_count`` 和 ``seed``。

    发布顺序:
        - 读取全部分片状态并核对分片身份、PDB 集合和每个 NPZ 的必需字段；随后写入 ``manifest.json``、``validation_selection.npz``、``config.json``、``summary.json`` 和空的 ``_COMPLETE``。

    失败语义:
        - 缺少分片、重复 PDB、集合不一致、单 PDB 字段不全或已存在 ``_COMPLETE`` 时不发布完成标记；已存在 ``_COMPLETE`` 的正式 pool 不会被覆盖，但未完成目录中的同名 NPZ 可能在分片阶段被重写。
    """

    output_root = Path(arguments.output_root).resolve()
    state_root = Path(arguments.state_root).resolve()
    shard_count = int(arguments.shard_count)
    if (output_root / "_COMPLETE").exists():
        raise FileExistsError(f"BOX pool 已完成，不允许重复发布：{output_root}")
    train_ids = load_split_pdb_ids(Path(arguments.train_split))
    validation_ids = load_split_pdb_ids(Path(arguments.validation_split))
    expected = {("train", pdb_id) for pdb_id in train_ids}
    expected.update(("validation", pdb_id) for pdb_id in validation_ids)
    reports_by_identity: dict[tuple[str, str], dict[str, Any]] = {}
    for shard_index in range(shard_count):
        state_path = state_root / f"shard_{shard_index:03d}_of_{shard_count:03d}.json"
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state["shard_index"] != shard_index or state["shard_count"] != shard_count:
            raise ValueError(f"分片身份不匹配：{state_path}")
        for report in state["pdb_reports"]:
            identity = (str(report["split"]), str(report["pdb_id"]))
            if identity in reports_by_identity:
                raise ValueError(f"PDB 被多个分片重复处理：{identity}")
            reports_by_identity[identity] = report
    if set(reports_by_identity) != expected:
        missing = sorted(expected.difference(reports_by_identity))[:20]
        extra = sorted(set(reports_by_identity).difference(expected))[:20]
        raise ValueError(f"分片集合不完整，missing={missing}, extra={extra}")

    manifest_entries: dict[str, list[dict[str, str]]] = {"train": [], "validation": []}
    summary: dict[str, Any] = {
        "seed": int(arguments.seed),
        "train": {"requested_pdb": len(train_ids), "published_pdb": 0, "zero_context_pdb_count": 0},
        "validation": {
            "requested_pdb": len(validation_ids),
            "published_pdb": 0,
            "zero_context_pdb_count": 0,
        },
    }
    for split_name, pdb_ids in (("train", train_ids), ("validation", validation_ids)):
        for pdb_id in pdb_ids:
            pool_path = output_root / split_name / f"{pdb_id}.npz"
            with np.load(pool_path, allow_pickle=False) as pool:
                required_fields = {
                    "pdb_id",
                    "occurrence_id",
                    "center_start_zyx",
                    "bias_start_zyx",
                    "context_start_zyx",
                }
                if not required_fields.issubset(pool.files):
                    raise KeyError(f"{pool_path} 缺少字段：{sorted(required_fields.difference(pool.files))}")
                context_count = int(np.asarray(pool["context_start_zyx"]).shape[0])
            if context_count == 0:
                summary[split_name]["zero_context_pdb_count"] += 1
            manifest_entries[split_name].append(
                {"pdb_id": pdb_id, "path": f"{split_name}/{pdb_id}.npz"}
            )
            summary[split_name]["published_pdb"] += 1

    manifest = {"schema_version": 1, "splits": manifest_entries}
    atomic_write_json(output_root / BOX_POOL_MANIFEST_FILENAME, manifest)
    selection_summary = freeze_validation_selection(
        validation_pool_directory=output_root / "validation",
        output_path=output_root / "validation_selection.npz",
        seed=int(arguments.seed),
        center_per_occurrence=CENTER_PER_OCCURRENCE,
        bias_per_occurrence=BIAS_PER_OCCURRENCE,
        context_per_occurrence=CONTEXT_PER_OCCURRENCE,
    )
    summary["validation_selection"] = selection_summary
    summary["manifest"] = {name: len(records) for name, records in manifest_entries.items()}
    config = {
        "schema_version": 1,
        "box_shape_zyx": list(STAGE1_BOX_SHAPE_ZYX),
        "bias_candidates_per_occurrence": BIAS_CANDIDATES_PER_OCCURRENCE,
        "bias_radius_formula": "R=(3*K_occ/(4*pi))**(1/3)",
        "extra_bias_drift_max_angstrom": EXTRA_BIAS_DRIFT_MAX_ANGSTROM,
        "extra_bias_drift_length_sampling": "uniform_0_to_max_angstrom",
        "context_generator": {
            "sampling": "uniform_integer_legal_start_per_axis",
            "target_count": CONTEXT_TARGET,
            "max_attempts": CONTEXT_MAX_ATTEMPTS,
            "min_core_receptor_heavy_atoms": CONTEXT_MIN_CORE_ATOMS,
            "ligand_filter": False,
        },
        "occurrence_cap_per_pdb_per_epoch": 50,
        "entry_ratio": {
            "center": CENTER_PER_OCCURRENCE,
            "bias": BIAS_PER_OCCURRENCE,
            "context": CONTEXT_PER_OCCURRENCE,
        },
        "seed": int(arguments.seed),
        "seed_rule": "sha256(base_seed|split_name|pdb_id) first_uint64",
        "exp_shape_source": "exp.npz:canonical_shape_zyx",
    }
    atomic_write_json(output_root / "config.json", config)
    atomic_write_json(output_root / "summary.json", summary)
    atomic_write_text(output_root / "_COMPLETE", "")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main() -> None:
    """解析 ``build-shard`` 或 ``finalize`` 子命令并调用对应入口。

    命令参数:
        - ``build-shard``：额外需要 ``shard-index`` 和 ``workers``，生成当前分片的 PDB pool。
        - ``finalize``：读取全部分片状态，发布 manifest、validation selection 和 ``_COMPLETE``。

    返回值:
        - None；成功时由子命令写入产物并打印 JSON 摘要，失败时保留异常供 Slurm 任务报告。
    """

    parser = argparse.ArgumentParser(description="生成 Stage1 v3 0:5:5 BOX pool。")
    subparsers = parser.add_subparsers(dest="command", required=True)
    shard_parser = subparsers.add_parser("build-shard")
    finalize_parser = subparsers.add_parser("finalize")
    for command_parser in (shard_parser, finalize_parser):
        command_parser.add_argument("--data-root", required=True)
        command_parser.add_argument("--train-split", required=True)
        command_parser.add_argument("--validation-split", required=True)
        command_parser.add_argument("--output-root", required=True)
        command_parser.add_argument("--state-root", required=True)
        command_parser.add_argument("--shard-count", type=int, required=True)
        command_parser.add_argument("--seed", type=int, required=True)
    shard_parser.add_argument("--shard-index", type=int, required=True)
    shard_parser.add_argument("--workers", type=int, required=True)
    arguments = parser.parse_args()
    if arguments.command == "build-shard":
        run_shard(arguments)
    else:
        finalize_box_pool(arguments)


if __name__ == "__main__":
    main()
