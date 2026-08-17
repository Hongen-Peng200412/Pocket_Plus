# -*- coding: utf-8 -*-
"""为 Stage1 V3 数据准备提供确定性 BOX 起点生成与验证集冻结。

这些函数只服务于一次性数据准备脚本，不属于训练时 Dataset。训练代码读取已经
发布的 ``box_pool``，不会反向依赖 ``ops``。
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from ops.stage1_data_preparation.atomic_io import atomic_save_npz
from src.datasets.stage1_requests import (
    BOX_POOL_MANIFEST_FILENAME,
    STAGE1_BOX_SHAPE_ZYX,
    centered_start_from_sparse_mask,
    resolve_stage1_start,
)


def derive_pdb_seed(base_seed: int, split_name: str, pdb_id: str) -> int:
    """派生不受 worker 数和遍历顺序影响的单 PDB 随机种子。"""

    payload = f"{int(base_seed)}|{split_name.lower()}|{pdb_id.lower()}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little", signed=False)


def load_occurrence_masks(
    path: Path,
    expected_shape_zyx: Sequence[int],
) -> dict[int, np.ndarray]:
    """读取 ``ligand_area.npz`` 中按 occurrence 保存的稀疏体素坐标。

    ``expected_shape_zyx`` 是完整图的三轴体素数。返回字典以整数
    ``occurrence_id`` 为键；每个值是 int32 ``(K_occ,3)`` 数组，列顺序为
    ZYX，坐标是完整图内的零基 voxel index。函数拒绝空 mask、越界坐标、
    非 ``(K_occ,3)`` 数组以及不匹配的 schema 或完整图形状。
    """

    expected_shape = np.asarray(expected_shape_zyx, dtype=np.int64)
    with np.load(path, allow_pickle=False) as data:
        if int(np.asarray(data["schema_version"]).item()) != 3:
            raise ValueError(f"{path}: 只接受 ligand_area schema_version=3。")
        if not np.array_equal(
            np.asarray(data["grid_shape_zyx"], dtype=np.int64),
            expected_shape,
        ):
            raise ValueError(f"{path}: grid_shape_zyx 与完整密度图不一致。")
        mask_keys = sorted(
            (key for key in data.files if key.startswith("mask_") and key[5:].isdigit()),
            key=lambda key: int(key[5:]),
        )
        masks = {
            int(key[5:]): np.asarray(data[key], dtype=np.int32)
            for key in mask_keys
        }
    for occurrence_id, sparse_zyx in masks.items():
        if sparse_zyx.ndim != 2 or sparse_zyx.shape[1] != 3 or sparse_zyx.shape[0] == 0:
            raise ValueError(f"{path}: mask_{occurrence_id} 必须为非空 (K,3) ZYX 数组。")
        if np.any(sparse_zyx < 0) or np.any(sparse_zyx >= expected_shape[None, :]):
            raise ValueError(f"{path}: mask_{occurrence_id} 含越界体素。")
    return masks


def generate_context_starts(
    receptor_coords_world: np.ndarray,
    full_origin_world: Sequence[float],
    voxel_size_world: Sequence[float],
    full_shape_zyx: Sequence[int],
    rng: np.random.Generator,
    target_count: int,
    max_attempts: int,
    min_core_atoms: int,
) -> np.ndarray:
    """在合法范围内生成满足受体原子数条件的 context BOX 起点。

    ``receptor_coords_world`` 是 float ``(N,3)`` 世界 XYZ 坐标，origin 与
    voxel size 也是世界 XYZ 量；``full_shape_zyx`` 与输出均采用完整图 ZYX
    体素顺序。每次尝试在各轴合法整数起点上均匀采样，80³ BOX 内至少包含
    ``min_core_atoms`` 个受体原子时才接收。返回 int32 ``(K,3)``，其中
    ``K <= target_count``；达到 ``max_attempts`` 时保留已经找到的起点。
    """

    coords_xyz = np.asarray(receptor_coords_world, dtype=np.float64)
    origin_xyz = np.asarray(full_origin_world, dtype=np.float64)
    voxel_size_xyz = np.asarray(voxel_size_world, dtype=np.float64)
    full_shape = np.asarray(full_shape_zyx, dtype=np.int64)
    resolve_stage1_start((0, 0, 0), full_shape)

    local_xyz = (coords_xyz - origin_xyz[None, :]) / voxel_size_xyz[None, :]
    local_zyx = local_xyz[:, [2, 1, 0]]
    box_shape = np.asarray(STAGE1_BOX_SHAPE_ZYX, dtype=np.int64)
    max_start = full_shape - box_shape
    starts: list[np.ndarray] = []
    for _ in range(int(max_attempts)):
        requested = np.asarray(
            [rng.integers(0, int(axis_max) + 1) for axis_max in max_start],
            dtype=np.int64,
        )
        if int(min_core_atoms) > 0:
            in_core = np.all(
                (local_zyx >= requested[None, :])
                & (local_zyx < (requested + box_shape)[None, :]),
                axis=1,
            )
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
    *,
    num_candidates: int,
    voxel_size_world: Sequence[float],
    extra_drift_max_angstrom: float,
) -> np.ndarray:
    """按 occurrence 体积和额外世界距离漂移生成 bias BOX 起点。

    ``sparse_voxel_zyx`` 是完整图内整数 ``(K_occ,3)`` ZYX voxel index；
    ``voxel_size_world`` 是世界 XYZ 的 Å/voxel。函数以体素中心计算质心，
    在等体积球内均匀采样偏移，再叠加不超过
    ``extra_drift_max_angstrom`` Å 的世界 XYZ 漂移。BOX 起点使用
    ``numpy.rint`` 取整并夹入合法范围，返回 int32 ``(num_candidates,3)``
    完整图 ZYX corner index。
    """

    sparse = np.asarray(sparse_voxel_zyx, dtype=np.int64)
    centroid_corner_zyx = sparse.astype(np.float64).mean(axis=0) + 0.5
    radius = float((3.0 * sparse.shape[0] / (4.0 * np.pi)) ** (1.0 / 3.0))

    directions = rng.normal(size=(int(num_candidates), 3))
    direction_norm = np.linalg.norm(directions, axis=1, keepdims=True)
    directions /= np.maximum(direction_norm, np.finfo(np.float64).tiny)
    radii = radius * np.cbrt(rng.random(int(num_candidates)))
    biased_centers = centroid_corner_zyx[None, :] + directions * radii[:, None]

    if float(extra_drift_max_angstrom) > 0:
        drift_directions_xyz = rng.normal(size=(int(num_candidates), 3))
        drift_norm = np.linalg.norm(drift_directions_xyz, axis=1, keepdims=True)
        drift_directions_xyz /= np.maximum(drift_norm, np.finfo(np.float64).tiny)
        drift_lengths = rng.uniform(
            0.0,
            float(extra_drift_max_angstrom),
            size=int(num_candidates),
        )
        voxel_size_xyz = np.asarray(voxel_size_world, dtype=np.float64)
        drift_voxel_xyz = (
            drift_directions_xyz * drift_lengths[:, None] / voxel_size_xyz[None, :]
        )
        biased_centers += drift_voxel_xyz[:, [2, 1, 0]]

    box_shape = np.asarray(STAGE1_BOX_SHAPE_ZYX, dtype=np.float64)
    requested_starts = np.rint(biased_centers - box_shape[None, :] / 2.0)
    return np.asarray(
        [resolve_stage1_start(start, full_shape_zyx) for start in requested_starts],
        dtype=np.int32,
    )


def build_occurrence_pool_rows(
    occurrence_masks_zyx: dict[int, np.ndarray],
    full_shape_zyx: Sequence[int],
    rng: np.random.Generator,
    *,
    voxel_size_world: Sequence[float],
    extra_bias_drift_max_angstrom: float,
    bias_candidates_per_occurrence: int,
) -> dict[str, np.ndarray]:
    """为一个 PDB 的全部 occurrence 生成中心与偏移候选数组。

    输入字典以整数 occurrence_id 为键，值为完整图内 int ``(K_occ,3)``
    ZYX voxel index。返回三个相互按第一维对齐的数组：
    ``occurrence_id`` 为 int32 ``(O,)``；``center_start_zyx`` 为 int32
    ``(O,3)``；``bias_start_zyx`` 为 int32
    ``(O,bias_candidates_per_occurrence,3)``。所有起点都是完整图内 80³ BOX
    的 ZYX corner index。
    """

    occurrence_ids = np.asarray(sorted(occurrence_masks_zyx), dtype=np.int32)
    centers: list[tuple[int, int, int]] = []
    biases: list[np.ndarray] = []
    for occurrence_id in occurrence_ids.tolist():
        sparse_zyx = occurrence_masks_zyx[occurrence_id]
        centers.append(centered_start_from_sparse_mask(sparse_zyx, full_shape_zyx))
        biases.append(
            sample_bias_starts(
                sparse_zyx,
                full_shape_zyx,
                rng,
                num_candidates=bias_candidates_per_occurrence,
                voxel_size_world=voxel_size_world,
                extra_drift_max_angstrom=extra_bias_drift_max_angstrom,
            )
        )
    return {
        "occurrence_id": occurrence_ids,
        "center_start_zyx": np.asarray(centers, dtype=np.int32).reshape(-1, 3),
        "bias_start_zyx": np.asarray(biases, dtype=np.int32).reshape(
            -1,
            int(bias_candidates_per_occurrence),
            3,
        ),
    }


def _load_validation_pool(pool_path: Path, expected_pdb_id: str) -> dict[str, np.ndarray]:
    """读取一个验证 PDB 的冻结起点数组。

    返回 ``occurrence_id: int32 (O,)``、``center_start_zyx: int32 (O,3)``、
    ``bias_start_zyx: int32 (O,30,3)`` 与
    ``context_start_zyx: int32 (C,3)``。后三个起点数组均使用完整图内 80³
    BOX 的零基 ZYX corner index；前三个数组按 occurrence 位置对齐。文件内
    ``pdb_id`` 与 ``expected_pdb_id`` 不一致时失败。
    """

    with np.load(pool_path, allow_pickle=False) as data:
        pdb_value = np.asarray(data["pdb_id"]).item()
        if isinstance(pdb_value, bytes):
            pdb_value = pdb_value.decode("utf-8")
        if str(pdb_value).strip().lower() != expected_pdb_id:
            raise ValueError(f"{pool_path}: PDB 身份与 manifest 不一致。")
        return {
            "occurrence_id": np.asarray(data["occurrence_id"], dtype=np.int32),
            "center_start_zyx": np.asarray(data["center_start_zyx"], dtype=np.int32),
            "bias_start_zyx": np.asarray(data["bias_start_zyx"], dtype=np.int32),
            "context_start_zyx": np.asarray(data["context_start_zyx"], dtype=np.int32),
        }


def freeze_validation_selection(
    validation_pool_directory: Path,
    output_path: Path,
    *,
    seed: int,
    center_per_occurrence: int,
    bias_per_occurrence: int,
    context_per_occurrence: int,
) -> dict[str, int]:
    """冻结 validation 请求索引，不复制 BOX 起点数组。

    函数按 manifest 顺序读取 validation PDB，每个 PDB 无放回选择至多 50 个
    occurrence，再按参数选择 center、bias 与 context。context 候选少于请求数
    时允许有放回采样。输出 NPZ 字段如下：

    - ``validation_pdb_id``：定宽 bytes ``(P,)``；其他 PDB index 均索引该数组。
    - ``center_pdb_index``、``center_occurrence_id``：int32 ``(N_center,)``。
    - ``bias_pdb_index``、``bias_occurrence_id``：int32 ``(N_bias,)``。
    - ``bias_candidate_index``：int16 ``(N_bias,)``；索引对应 PDB pool 的候选轴。
    - ``context_pdb_index``：int32 ``(N_context,)``；索引 PDB 身份数组。
    - ``context_candidate_index``：int32 ``(N_context,)``；索引对应 context 数组。

    文件以非压缩 NPZ 原子发布。返回字典的 ``pdb_count`` 是验证 PDB 数，
    ``center_count``、``bias_count`` 与 ``context_count`` 分别是三类冻结请求数。
    manifest、pool 字段缺失或请求数超过可无放回选择的 bias 候选时直接失败。
    """

    pool_directory = Path(validation_pool_directory)
    manifest = json.loads(
        (pool_directory.parent / BOX_POOL_MANIFEST_FILENAME).read_text(encoding="utf-8")
    )
    entries = [
        (str(record["pdb_id"]).strip().lower(), pool_directory.parent / record["path"])
        for record in manifest["splits"][pool_directory.name]
    ]
    pools = [
        _load_validation_pool(pool_path, expected_pdb_id=pdb_id)
        for pdb_id, pool_path in entries
    ]
    pdb_ids = [pdb_id for pdb_id, _ in entries]
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), 1]))

    center_pdb_index: list[int] = []
    center_occurrence_id: list[int] = []
    bias_pdb_index: list[int] = []
    bias_occurrence_id: list[int] = []
    bias_candidate_index: list[int] = []
    context_pdb_index: list[int] = []
    context_candidate_index: list[int] = []

    for pdb_index, pool in enumerate(pools):
        occurrence_count = int(pool["occurrence_id"].shape[0])
        selected_rows = rng.choice(
            occurrence_count,
            size=min(50, occurrence_count),
            replace=False,
        )
        context_count = int(pool["context_start_zyx"].shape[0])
        for occurrence_row in selected_rows.tolist():
            occurrence_id = int(pool["occurrence_id"][occurrence_row])
            if int(center_per_occurrence) == 1:
                center_pdb_index.append(pdb_index)
                center_occurrence_id.append(occurrence_id)
            for candidate_index in rng.choice(
                int(pool["bias_start_zyx"].shape[1]),
                size=int(bias_per_occurrence),
                replace=False,
            ).tolist():
                bias_pdb_index.append(pdb_index)
                bias_occurrence_id.append(occurrence_id)
                bias_candidate_index.append(int(candidate_index))
            if context_count > 0:
                for candidate_index in rng.choice(
                    context_count,
                    size=int(context_per_occurrence),
                    replace=context_count < int(context_per_occurrence),
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
    atomic_save_npz(Path(output_path), arrays, compressed=False)
    return {
        "pdb_count": len(pdb_ids),
        "center_count": len(center_pdb_index),
        "bias_count": len(bias_pdb_index),
        "context_count": len(context_pdb_index),
    }
