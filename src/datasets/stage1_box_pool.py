# -*- coding: utf-8 -*-
"""AdaLigand Stage1 训练 BOX pool 的确定性几何原语与一键生成入口。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Sequence

import numpy as np

from src.datasets.stage1_requests import (
    BOX_POOL_MANIFEST_FILENAME,
    STAGE1_BOX_SHAPE_ZYX,
    _load_manifest_pool_paths,
    centered_start_from_sparse_mask,
    resolve_stage1_start,
)


_CONTEXT_TARGET = 500
_CONTEXT_MAX_ATTEMPTS = 3000
_CONTEXT_MIN_CORE_ATOMS = 1000
_POOL_SEED = 3407


def _pdb_seed(base_seed: int, split_name: str, pdb_id: str) -> int:
    """从稳定 PDB identity 派生与 worker/遍历顺序无关的随机 seed。"""

    payload = f"{int(base_seed)}|{str(split_name).lower()}|{str(pdb_id).lower()}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], byteorder="little", signed=False)


def _atomic_save_npz(path: Path, **arrays: np.ndarray) -> None:
    """在目标同目录写临时 NPZ，完成后原子发布。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(descriptor)
    temp_path = Path(temp_name)
    try:
        with temp_path.open("wb") as handle:
            np.savez(handle, **arrays)
        temp_path.replace(path)
    finally:
        temp_path.unlink(missing_ok=True)


def _atomic_write_text(path: Path, content: str) -> None:
    """原子发布 UTF-8 文本或完成标记。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(descriptor)
    temp_path = Path(temp_name)
    try:
        temp_path.write_text(content, encoding="utf-8")
        temp_path.replace(path)
    finally:
        temp_path.unlink(missing_ok=True)


def _load_split_pdb_ids(path: str | Path) -> tuple[str, ...]:
    """从冻结 split JSON 读取并排序唯一 PDB identity。"""

    source = Path(path)
    value = json.loads(source.read_text(encoding="utf-8"))
    if isinstance(value, dict):
        for field_name in ("entries", "pairs", "items", "pdb_ids"):
            if field_name in value:
                value = value[field_name]
                break
    if not isinstance(value, list):
        raise TypeError(f"{source}: split 必须是列表，或包含 entries/pairs/items/pdb_ids 列表。")
    pdb_ids: list[str] = []
    for row in value:
        pdb_id = row.get("pdb_id") if isinstance(row, dict) else row
        pdb_id = str(pdb_id).strip().lower()
        if not pdb_id or pdb_id == "none":
            raise ValueError(f"{source}: split 条目缺少有效 pdb_id。")
        pdb_ids.append(pdb_id)
    return tuple(sorted(set(pdb_ids)))


def _load_occurrence_masks(path: Path, expected_shape_zyx: Sequence[int]) -> dict[int, np.ndarray]:
    """读取 E3 schema-v3 的全部非空 ``mask_{candidate_id}``。"""

    expected_shape = np.asarray(expected_shape_zyx, dtype=np.int64)
    with np.load(path, allow_pickle=False) as data:
        if "schema_version" not in data or int(np.asarray(data["schema_version"]).item()) != 3:
            raise ValueError(f"{path}: BOX pool 只接受 ligand_area schema_version=3。")
        if "grid_shape_zyx" in data and not np.array_equal(
            np.asarray(data["grid_shape_zyx"], dtype=np.int64), expected_shape
        ):
            raise ValueError(f"{path}: grid_shape_zyx 与 exp grid 不一致。")
        mask_keys = sorted(
            (key for key in data.files if key.startswith("mask_") and key[5:].isdigit()),
            key=lambda key: int(key[5:]),
        )
        masks = {
            int(key[5:]): np.asarray(data[key], dtype=np.int32)
            for key in mask_keys
        }
    for occurrence_id, sparse in masks.items():
        if sparse.ndim != 2 or sparse.shape[1] != 3 or sparse.shape[0] == 0:
            raise ValueError(f"{path}: mask_{occurrence_id} 必须为非空 [K,3] ZYX。")
        if np.any(sparse < 0) or np.any(sparse >= expected_shape[None, :]):
            raise ValueError(f"{path}: mask_{occurrence_id} 含越界体素。")
    return masks


def generate_context_starts(
    receptor_coords_world: np.ndarray,
    full_origin_world: Sequence[float],
    voxel_size_world: Sequence[float],
    full_shape_zyx: Sequence[int],
    rng: np.random.Generator,
    target_count: int = _CONTEXT_TARGET,
    max_attempts: int = _CONTEXT_MAX_ATTEMPTS,
    min_core_atoms: int = _CONTEXT_MIN_CORE_ATOMS,
) -> np.ndarray:
    """按现有随机 context 规则生成合法 80³ 起点。

    每个轴在完整图合法起点闭区间内均匀采样；BOX core 中 receptor 重原子数
    不少于 ``min_core_atoms`` 才保留。候选不绑定 occurrence，也不按 ligand
    内容过滤。达到目标数或耗尽尝试次数即停止，重复起点保持原样。
    """

    coords = np.asarray(receptor_coords_world, dtype=np.float64)
    origin = np.asarray(full_origin_world, dtype=np.float64)
    voxel_size = np.asarray(voxel_size_world, dtype=np.float64)
    full_shape = np.asarray(full_shape_zyx, dtype=np.int64)
    if coords.ndim != 2 or coords.shape[1] != 3 or not np.isfinite(coords).all():
        raise ValueError("receptor_coords_world 必须为有限 [N,3] XYZ 数组。")
    if origin.shape != (3,) or voxel_size.shape != (3,) or np.any(voxel_size <= 0):
        raise ValueError("full_origin_world/voxel_size_world 必须为合法 (3,) XYZ。")
    resolve_stage1_start((0, 0, 0), full_shape)
    if int(target_count) <= 0 or int(max_attempts) <= 0 or int(min_core_atoms) <= 0:
        raise ValueError("context target/max_attempts/min_core_atoms 必须为正整数。")

    # float64[N,3]，转成 corner 语义的连续 ZYX voxel 坐标；receptor_tokens 已仅含重原子。
    local_xyz = (coords - origin[None, :]) / voxel_size[None, :]
    local_zyx = local_xyz[:, [2, 1, 0]]
    max_start = full_shape - np.asarray(STAGE1_BOX_SHAPE_ZYX, dtype=np.int64)
    starts: list[np.ndarray] = []
    for _ in range(int(max_attempts)):
        requested = np.asarray(
            [rng.integers(0, int(axis_max) + 1) for axis_max in max_start],
            dtype=np.int64,
        )
        upper = requested + np.asarray(STAGE1_BOX_SHAPE_ZYX, dtype=np.int64)
        in_core = np.all((local_zyx >= requested[None, :]) & (local_zyx < upper[None, :]), axis=1)
        if int(np.count_nonzero(in_core)) < int(min_core_atoms):
            continue
        starts.append(requested.astype(np.int32, copy=False))
        if len(starts) >= int(target_count):
            break
    return np.asarray(starts, dtype=np.int32).reshape(-1, 3)


def sample_bias_starts(
    sparse_voxel_zyx: np.ndarray,
    full_shape_zyx: Sequence[int],
    rng: np.random.Generator,
    num_candidates: int = 30,
    box_shape_zyx: Sequence[int] = STAGE1_BOX_SHAPE_ZYX,
) -> np.ndarray:
    """按体积均匀球偏移生成 occurrence 的冻结 bias 起点。

    球半径固定为 ``R=(3*K_occ/(4*pi))**(1/3)`` voxel；方向均匀、半径使用
    ``U**(1/3)``，不重试、不去重。每个偏移后的请求都调用统一起点解析器。

    返回:
        ``int32[num_candidates,3]``，数组顺序 ZYX。
    """

    sparse = np.asarray(sparse_voxel_zyx, dtype=np.int64)
    if sparse.ndim != 2 or sparse.shape[1] != 3 or sparse.shape[0] == 0:
        raise ValueError("sparse_voxel_zyx 必须为非空 (K,3) ZYX 数组。")
    if int(num_candidates) <= 0:
        raise ValueError("num_candidates 必须为正整数。")
    box_shape = np.asarray(box_shape_zyx, dtype=np.float64)
    centroid_corner_zyx = sparse.astype(np.float64).mean(axis=0) + 0.5
    radius = float((3.0 * sparse.shape[0] / (4.0 * np.pi)) ** (1.0 / 3.0))
    directions = rng.normal(size=(int(num_candidates), 3))
    direction_norm = np.linalg.norm(directions, axis=1, keepdims=True)
    directions = directions / np.maximum(direction_norm, np.finfo(np.float64).tiny)
    radii = radius * np.cbrt(rng.random(int(num_candidates)))
    biased_centers = centroid_corner_zyx[None, :] + directions * radii[:, None]
    requested_starts = np.rint(biased_centers - box_shape[None, :] / 2.0)
    return np.asarray(
        [resolve_stage1_start(start, full_shape_zyx, box_shape_zyx) for start in requested_starts],
        dtype=np.int32,
    )


def build_occurrence_pool_rows(
    occurrence_masks_zyx: dict[int, np.ndarray],
    full_shape_zyx: Sequence[int],
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    """为一个 PDB 的全部非空 occurrence 构造 center 与 30 个 bias 起点。"""

    occurrence_ids = np.asarray(sorted(int(value) for value in occurrence_masks_zyx), dtype=np.int32)
    centers: list[tuple[int, int, int]] = []
    biases: list[np.ndarray] = []
    for occurrence_id in occurrence_ids.tolist():
        sparse = np.asarray(occurrence_masks_zyx[occurrence_id], dtype=np.int32)
        centers.append(centered_start_from_sparse_mask(sparse, full_shape_zyx))
        biases.append(sample_bias_starts(sparse, full_shape_zyx, rng, num_candidates=30))
    return {
        "occurrence_id": occurrence_ids,
        "center_start_zyx": np.asarray(centers, dtype=np.int32).reshape(-1, 3),
        "bias_start_zyx": np.asarray(biases, dtype=np.int32).reshape(-1, 30, 3),
    }


def build_pdb_box_pool(
    data_root: str | Path,
    pdb_id: str,
    split_name: str,
    seed: int = _POOL_SEED,
) -> dict[str, np.ndarray]:
    """从一份 A—G PDB 三件套构造 center/bias/context 索引池。"""

    root = Path(data_root)
    pdb_id = str(pdb_id).strip().lower()
    density_dir = root / "density" / pdb_id
    with np.load(density_dir / "exp.npz", allow_pickle=False) as data:
        if not {"grid", "voxel_size", "origin"}.issubset(data.files):
            raise KeyError(f"{density_dir / 'exp.npz'} 缺少 grid/voxel_size/origin。")
        grid_shape_zyx = np.asarray(data["grid"].shape[-3:], dtype=np.int64)
        voxel_size = np.asarray(data["voxel_size"], dtype=np.float32)
        origin = np.asarray(data["origin"], dtype=np.float32)
    resolve_stage1_start((0, 0, 0), grid_shape_zyx)
    occurrence_masks = _load_occurrence_masks(density_dir / "ligand_area.npz", grid_shape_zyx)
    if not occurrence_masks:
        raise ValueError(f"{pdb_id}: ligand_area.npz 不含任何 occurrence mask。")

    with np.load(root / "parse" / pdb_id / "receptor_tokens.npz", allow_pickle=False) as data:
        if "coords" not in data:
            raise KeyError(f"{pdb_id}: receptor_tokens.npz 缺少 coords。")
        receptor_coords = np.asarray(data["coords"], dtype=np.float32)
    rng = np.random.default_rng(_pdb_seed(seed, split_name, pdb_id))
    occurrence_rows = build_occurrence_pool_rows(occurrence_masks, grid_shape_zyx, rng)
    contexts = generate_context_starts(
        receptor_coords_world=receptor_coords,
        full_origin_world=origin,
        voxel_size_world=voxel_size,
        full_shape_zyx=grid_shape_zyx,
        rng=rng,
    )
    return {
        "pdb_id": np.asarray(pdb_id),
        **occurrence_rows,
        "context_start_zyx": contexts,
    }


def freeze_validation_selection(
    validation_pool_directory: str | Path,
    output_path: str | Path,
    seed: int = _POOL_SEED,
) -> dict[str, int]:
    """一次冻结 validation 的 ``1:5:3`` 真实读取项。"""

    from src.datasets.stage1_requests import _load_pdb_pool

    pool_directory = Path(validation_pool_directory)
    manifest_entries = _load_manifest_pool_paths(
        pool_directory.parent,
        pool_directory.name,
        require_complete=False,
    )
    pools = [
        _load_pdb_pool(path, expected_pdb_id=pdb_id)
        for pdb_id, path in manifest_entries
    ]
    pdb_ids = [pool.pdb_id for pool in pools]
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), 1]))
    center_pdb_index: list[int] = []
    center_occurrence_id: list[int] = []
    bias_pdb_index: list[int] = []
    bias_occurrence_id: list[int] = []
    bias_candidate_index: list[int] = []
    context_pdb_index: list[int] = []
    context_candidate_index: list[int] = []

    for pdb_index, pool in enumerate(pools):
        occurrence_count = int(pool.occurrence_id.shape[0])
        selected_rows = rng.choice(occurrence_count, size=min(50, occurrence_count), replace=False)
        context_count = int(pool.context_start_zyx.shape[0])
        for occurrence_row in selected_rows.tolist():
            occurrence_id = int(pool.occurrence_id[occurrence_row])
            center_pdb_index.append(pdb_index)
            center_occurrence_id.append(occurrence_id)
            for candidate_index in rng.choice(30, size=5, replace=False).tolist():
                bias_pdb_index.append(pdb_index)
                bias_occurrence_id.append(occurrence_id)
                bias_candidate_index.append(int(candidate_index))
            if context_count > 0:
                for candidate_index in rng.choice(
                    context_count, size=3, replace=context_count < 3
                ).tolist():
                    context_pdb_index.append(pdb_index)
                    context_candidate_index.append(int(candidate_index))

    max_pdb_width = max(1, max(len(value.encode("utf-8")) for value in pdb_ids))
    arrays = {
        "validation_pdb_id": np.asarray(pdb_ids, dtype=f"S{max_pdb_width}"),
        "center_pdb_index": np.asarray(center_pdb_index, dtype=np.int32),
        "center_occurrence_id": np.asarray(center_occurrence_id, dtype=np.int32),
        "bias_pdb_index": np.asarray(bias_pdb_index, dtype=np.int32),
        "bias_occurrence_id": np.asarray(bias_occurrence_id, dtype=np.int32),
        "bias_candidate_index": np.asarray(bias_candidate_index, dtype=np.int16),
        "context_pdb_index": np.asarray(context_pdb_index, dtype=np.int32),
        "context_candidate_index": np.asarray(context_candidate_index, dtype=np.int32),
    }
    _atomic_save_npz(Path(output_path), **arrays)
    return {
        "pdb_count": len(pdb_ids),
        "center_count": len(center_pdb_index),
        "bias_count": len(bias_pdb_index),
        "context_count": len(context_pdb_index),
    }


def build_stage1_box_pools(
    data_root: str | Path,
    train_split: str | Path,
    validation_split: str | Path,
    output_root: str | Path,
    seed: int = _POOL_SEED,
) -> dict[str, object]:
    """端到端生成 train/validation pool、冻结 selection 与完成标记。

    已存在的单 PDB pool 会重新按同一稳定 seed 原子覆盖；只有全部输入成功后才发布
    根目录 ``_COMPLETE``。train 中三轴短于 80 的 PDB 只计入普通 summary，validation
    若出现同类错误则 fail-fast，因为 split 选择本应已经排除它。
    """

    root = Path(output_root)
    train_ids = _load_split_pdb_ids(train_split)
    validation_ids = _load_split_pdb_ids(validation_split)
    overlap = sorted(set(train_ids).intersection(validation_ids))
    if overlap:
        raise ValueError(f"train/validation 不得共享 PDB，发现: {overlap[:10]}。")
    root.mkdir(parents=True, exist_ok=True)
    (root / "_COMPLETE").unlink(missing_ok=True)
    (root / BOX_POOL_MANIFEST_FILENAME).unlink(missing_ok=True)
    (root / "validation_selection.npz").unlink(missing_ok=True)
    summary: dict[str, object] = {
        "seed": int(seed),
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
    manifest_entries: dict[str, list[dict[str, str]]] = {"train": [], "validation": []}
    for split_name, pdb_ids in (("train", train_ids), ("validation", validation_ids)):
        output_directory = root / split_name
        output_directory.mkdir(parents=True, exist_ok=True)
        for pdb_id in pdb_ids:
            try:
                pool = build_pdb_box_pool(data_root, pdb_id, split_name, seed=seed)
            except ValueError as error:
                if split_name == "train" and "完整图三轴必须不小于" in str(error):
                    summary["train"]["short_map_count"] += 1  # type: ignore[index]
                    continue
                raise
            _atomic_save_npz(output_directory / f"{pdb_id}.npz", **pool)
            context_count = int(pool["context_start_zyx"].shape[0])
            if context_count == 0:
                summary[split_name]["zero_context_pdb_count"] += 1  # type: ignore[index]
            elif context_count < 3:
                summary[split_name]["underfilled_context_pdb_count"] += 1  # type: ignore[index]
            manifest_entries[split_name].append(
                {"pdb_id": pdb_id, "path": (Path(split_name) / f"{pdb_id}.npz").as_posix()}
            )
            summary[split_name]["published_pdb"] += 1  # type: ignore[index]

    manifest = {
        "schema_version": 1,
        "splits": manifest_entries,
    }
    _atomic_write_text(
        root / BOX_POOL_MANIFEST_FILENAME,
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
    )
    selection_summary = freeze_validation_selection(
        validation_pool_directory=root / "validation",
        output_path=root / "validation_selection.npz",
        seed=seed,
    )
    summary["validation_selection"] = selection_summary
    summary["manifest"] = {
        split_name: len(entries) for split_name, entries in manifest_entries.items()
    }
    config = {
        "box_shape_zyx": list(STAGE1_BOX_SHAPE_ZYX),
        "bias_candidates_per_occurrence": 30,
        "bias_radius_formula": "R=(3*K_occ/(4*pi))**(1/3)",
        "bias_selected_per_epoch": 5,
        "context_generator": {
            "sampling": "uniform_integer_legal_start_per_axis",
            "target_count": _CONTEXT_TARGET,
            "max_attempts": _CONTEXT_MAX_ATTEMPTS,
            "min_core_receptor_heavy_atoms": _CONTEXT_MIN_CORE_ATOMS,
            "ligand_filter": False,
        },
        "occurrence_cap_per_pdb_per_epoch": 50,
        "entry_ratio": {"center": 1, "bias": 5, "context": 3},
        "train_random_rotation_90_degree": True,
        "seed": int(seed),
        "seed_rule": "sha256(base_seed|split_name|pdb_id) first_uint64",
    }
    _atomic_write_text(root / "config.json", json.dumps(config, ensure_ascii=False, indent=2) + "\n")
    _atomic_write_text(root / "summary.json", json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    _atomic_write_text(root / "_COMPLETE", "")
    return summary


def _main() -> None:
    """解析 CLI 并执行 Stage1 BOX pool 一键生成。"""

    parser = argparse.ArgumentParser(description="生成 AdaLigand Stage1 train/validation BOX pool。")
    parser.add_argument("--data-root", required=True, help="A—G 正式数据根目录。")
    parser.add_argument("--train-split", required=True, help="冻结 train.json。")
    parser.add_argument("--validation-split", required=True, help="冻结 validation.json。")
    parser.add_argument("--output-root", required=True, help="stage1_preparation/box_pool 输出目录。")
    parser.add_argument("--seed", type=int, default=_POOL_SEED, help="BOX pool 基准随机 seed。")
    arguments = parser.parse_args()
    summary = build_stage1_box_pools(
        data_root=arguments.data_root,
        train_split=arguments.train_split,
        validation_split=arguments.validation_split,
        output_root=arguments.output_root,
        seed=arguments.seed,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    _main()
