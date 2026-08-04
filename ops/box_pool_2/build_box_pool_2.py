# -*- coding: utf-8 -*-
"""并行生成并原子发布 Stage1 第二版 BOX pool。"""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from src.datasets.ops.stage1_box_pool import (
    _POOL_SEED,
    _atomic_save_npz,
    _atomic_write_text,
    _load_split_pdb_ids,
    build_pdb_box_pool,
    freeze_validation_selection,
)
from src.datasets.stage1_requests import BOX_POOL_MANIFEST_FILENAME, STAGE1_BOX_SHAPE_ZYX


CENTER_PER_OCCURRENCE = 0
BIAS_PER_OCCURRENCE = 5
CONTEXT_PER_OCCURRENCE = 5
CONTEXT_MIN_CORE_ATOMS = 0
EXTRA_BIAS_DRIFT_MAX_ANGSTROM = 3.0
CONTEXT_TARGET = 500
CONTEXT_MAX_ATTEMPTS = 3000


def _build_one(argument: tuple[str, str, str, str, int]) -> dict[str, object]:
    """生成一个 PDB 的第二版起点池，并返回分片统计。"""

    data_root, output_root, split_name, pdb_id, seed = argument
    try:
        pool = build_pdb_box_pool(
            data_root=data_root,
            pdb_id=pdb_id,
            split_name=split_name,
            seed=seed,
            context_min_core_atoms=CONTEXT_MIN_CORE_ATOMS,
            extra_bias_drift_max_angstrom=EXTRA_BIAS_DRIFT_MAX_ANGSTROM,
        )
    except ValueError as error:
        if split_name == "train" and "完整图三轴必须不小于" in str(error):
            return {"split": split_name, "pdb_id": pdb_id, "status": "short_map"}
        raise
    destination = Path(output_root) / split_name / f"{pdb_id}.npz"
    _atomic_save_npz(destination, **pool)
    return {
        "split": split_name,
        "pdb_id": pdb_id,
        "status": "published",
        "context_count": int(pool["context_start_zyx"].shape[0]),
    }


def build_shard(arguments: argparse.Namespace) -> None:
    """让一个 Slurm 数组元素并行生成自己负责的 PDB。"""

    shard_count = int(arguments.shard_count)
    shard_index = int(arguments.shard_index)
    if shard_count <= 0 or not 0 <= shard_index < shard_count:
        raise ValueError("shard_count 必须为正，shard_index 必须位于 [0, shard_count)。")

    output_root = Path(arguments.output_root)
    if (output_root / "_COMPLETE").exists():
        raise FileExistsError(f"正式 BOX pool 已完成，不允许覆盖：{output_root}")
    train_ids = _load_split_pdb_ids(arguments.train_split)
    validation_ids = _load_split_pdb_ids(arguments.validation_split)
    overlap = sorted(set(train_ids).intersection(validation_ids))
    if overlap:
        raise ValueError(f"train/validation 不得共享 PDB：{overlap[:10]}")

    tagged_ids = [("train", pdb_id) for pdb_id in train_ids]
    tagged_ids.extend(("validation", pdb_id) for pdb_id in validation_ids)
    assigned = tagged_ids[shard_index::shard_count]
    worker_count = max(1, min(int(arguments.workers), len(assigned)))
    worker_arguments = [
        (arguments.data_root, str(output_root), split_name, pdb_id, int(arguments.seed))
        for split_name, pdb_id in assigned
    ]
    if worker_count == 1:
        rows = [_build_one(worker_arguments[0])] if worker_arguments else []
    elif worker_arguments:
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            rows = list(executor.map(_build_one, worker_arguments))
    else:
        rows = []

    state_path = Path(arguments.state_root) / f"shard_{shard_index:03d}_of_{shard_count:03d}.json"
    _atomic_write_text(
        state_path,
        json.dumps(
            {
                "shard_index": shard_index,
                "shard_count": shard_count,
                "assigned_count": len(assigned),
                "rows": rows,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
    )
    print(json.dumps({"shard_index": shard_index, "completed": len(rows)}, ensure_ascii=False))


def finalize(arguments: argparse.Namespace) -> None:
    """核对全部分片并最后发布清单、冻结验证请求和完成标记。"""

    output_root = Path(arguments.output_root)
    if (output_root / "_COMPLETE").exists():
        raise FileExistsError(f"正式 BOX pool 已完成，不允许重复发布：{output_root}")
    train_ids = _load_split_pdb_ids(arguments.train_split)
    validation_ids = _load_split_pdb_ids(arguments.validation_split)
    expected = {("train", value) for value in train_ids}
    expected.update(("validation", value) for value in validation_ids)

    rows_by_identity: dict[tuple[str, str], dict[str, object]] = {}
    for shard_index in range(int(arguments.shard_count)):
        state_path = Path(arguments.state_root) / (
            f"shard_{shard_index:03d}_of_{int(arguments.shard_count):03d}.json"
        )
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if int(state["shard_index"]) != shard_index or int(state["shard_count"]) != int(arguments.shard_count):
            raise ValueError(f"分片状态身份不符：{state_path}")
        for row in state["rows"]:
            identity = (str(row["split"]), str(row["pdb_id"]))
            if identity in rows_by_identity:
                raise ValueError(f"PDB 被多个分片重复处理：{identity}")
            rows_by_identity[identity] = row
    if set(rows_by_identity) != expected:
        missing = sorted(expected.difference(rows_by_identity))[:20]
        extra = sorted(set(rows_by_identity).difference(expected))[:20]
        raise ValueError(f"分片集合不完整，missing={missing}, extra={extra}")

    manifest_entries: dict[str, list[dict[str, str]]] = {"train": [], "validation": []}
    summary: dict[str, object] = {
        "seed": int(arguments.seed),
        "train": {
            "requested_pdb": len(train_ids),
            "published_pdb": 0,
            "short_map_count": 0,
            "zero_context_pdb_count": 0,
            "underfilled_context_pdb_count": 0,
        },
        "validation": {
            "requested_pdb": len(validation_ids),
            "published_pdb": 0,
            "zero_context_pdb_count": 0,
            "underfilled_context_pdb_count": 0,
        },
    }
    for split_name, pdb_ids in (("train", train_ids), ("validation", validation_ids)):
        for pdb_id in pdb_ids:
            row = rows_by_identity[(split_name, pdb_id)]
            if row["status"] == "short_map":
                if split_name != "train":
                    raise ValueError(f"validation 不允许短图：{pdb_id}")
                summary["train"]["short_map_count"] += 1  # type: ignore[index]
                continue
            pool_path = output_root / split_name / f"{pdb_id}.npz"
            if not pool_path.is_file():
                raise FileNotFoundError(f"分片声称完成但 NPZ 不存在：{pool_path}")
            with np.load(pool_path, allow_pickle=False) as pool:
                context_count = int(np.asarray(pool["context_start_zyx"]).shape[0])
            if context_count == 0:
                summary[split_name]["zero_context_pdb_count"] += 1  # type: ignore[index]
            elif context_count < CONTEXT_PER_OCCURRENCE:
                summary[split_name]["underfilled_context_pdb_count"] += 1  # type: ignore[index]
            manifest_entries[split_name].append(
                {"pdb_id": pdb_id, "path": f"{split_name}/{pdb_id}.npz"}
            )
            summary[split_name]["published_pdb"] += 1  # type: ignore[index]

    manifest = {"schema_version": 1, "splits": manifest_entries}
    _atomic_write_text(
        output_root / BOX_POOL_MANIFEST_FILENAME,
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
    )
    selection_summary = freeze_validation_selection(
        validation_pool_directory=output_root / "validation",
        output_path=output_root / "validation_selection.npz",
        seed=int(arguments.seed),
        center_per_occurrence=CENTER_PER_OCCURRENCE,
        bias_per_occurrence=BIAS_PER_OCCURRENCE,
        context_per_occurrence=CONTEXT_PER_OCCURRENCE,
    )
    summary["validation_selection"] = selection_summary
    summary["manifest"] = {
        split_name: len(entries) for split_name, entries in manifest_entries.items()
    }
    config = {
        "box_shape_zyx": list(STAGE1_BOX_SHAPE_ZYX),
        "bias_candidates_per_occurrence": 30,
        "bias_radius_formula": "R=(3*K_occ/(4*pi))**(1/3)",
        "extra_bias_drift_max_angstrom": EXTRA_BIAS_DRIFT_MAX_ANGSTROM,
        "extra_bias_drift_length_sampling": "uniform_0_to_max_angstrom",
        "bias_selected_per_epoch": BIAS_PER_OCCURRENCE,
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
        "train_random_rotation_90_degree": True,
        "seed": int(arguments.seed),
        "seed_rule": "base_seed、split_name 与 pdb_id 的既有稳定派生规则",
    }
    _atomic_write_text(
        output_root / "config.json",
        json.dumps(config, ensure_ascii=False, indent=2) + "\n",
    )
    _atomic_write_text(
        output_root / "summary.json",
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
    )
    _atomic_write_text(output_root / "_COMPLETE", "")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def _main() -> None:
    parser = argparse.ArgumentParser(description="生成 Stage1 第二版 0:5:5 BOX pool。")
    parser.add_argument("command", choices=("build-shard", "finalize"))
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--train-split", required=True)
    parser.add_argument("--validation-split", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--state-root", required=True)
    parser.add_argument("--shard-count", type=int, required=True)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--workers", type=int, default=max(1, os.cpu_count() or 1))
    parser.add_argument("--seed", type=int, default=_POOL_SEED)
    arguments = parser.parse_args()
    if arguments.command == "build-shard":
        build_shard(arguments)
    else:
        finalize(arguments)


if __name__ == "__main__":
    _main()
