# -*- coding: utf-8 -*-
"""为 Stage1 V3 一次性数据准备提供确定性 BOX 起点和验证选择工具。

本模块只被 ``ops/stage1_data_preparation`` 下的构建脚本调用，不属于训练时 Dataset。输入是迁移后正式资产中的 occurrence 稀疏 mask、受体世界坐标和 V3 pool；输出是完整图内的 80³ ZYX corner 起点或冻结 validation 索引。训练代码只读取已发布的 pool，不反向导入本模块。
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
    """从全局 seed、split 和 PDB identity 派生稳定的单 PDB 随机种子。

    输入参数:
        - base_seed: int；数据准备任务的全局基准 seed。
        - split_name: str；``train`` 或 ``validation`` 等 split 名称，规范化为小写后参与哈希。
        - pdb_id: str；PDB identity，规范化为小写后参与哈希。

    返回值:
        - seed: int；SHA-256 前 8 字节按 little-endian 无符号解释的非负整数；不依赖 worker 数、遍历顺序或 Python hash 随机化。
    """

    payload = f"{int(base_seed)}|{split_name.lower()}|{pdb_id.lower()}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little", signed=False)


def load_occurrence_masks(
    path: Path,
    expected_shape_zyx: Sequence[int],
) -> dict[int, np.ndarray]:
    """读取并验证 ``ligand_area.npz`` 中每个 occurrence 的稀疏 ZYX 坐标。

    输入参数:
        - path: Path；一个 PDB 的 ``ligand_area.npz``，必须是 schema 3 并包含 ``mask_<occurrence_id>`` 字段。
        - expected_shape_zyx: Sequence[int] ``(3,)``；对应完整密度图的 ``(D_full, H_full, W_full)`` 体素形状。

    返回值:
        - masks: dict[int, np.ndarray]；键是 occurrence 的整数 identity，值是读取后转换为 int32 的 ``(K_occ,3)`` 完整图零基 ZYX voxel index；``K_occ`` 必须大于零。

    字段语义:
        - ``grid_shape_zyx``：NPZ 中声明的完整图 ZYX 形状，必须逐项等于 ``expected_shape_zyx``。
        - ``mask_<id>``：每行一个 occurrence 非空体素，列顺序固定为 Z、Y、X；函数不转换为密集 mask。

    失败语义:
        - schema、完整图形状、数组维度、空 mask 或体素边界不符合契约时抛出 ``ValueError``；本函数不额外验证稀疏坐标的原始 dtype、排序或去重状态。
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
    """随机生成满足核心受体原子数条件的 context BOX 起点。

    输入参数:
        - receptor_coords_world: np.ndarray float ``(N_receptor, 3)``；完整受体表的世界 XYZ 坐标，单位为 Å。
        - full_origin_world: Sequence[float] ``(3,)``；完整图 voxel-grid corner 的世界 XYZ 原点，单位为 Å。
        - voxel_size_world: Sequence[float] ``(3,)``；世界 XYZ 体素尺寸，单位为 Å。
        - full_shape_zyx: Sequence[int] ``(3,)``；完整图 ZYX 体素形状。
        - rng: np.random.Generator；已由调用方按 PDB 派生 seed 初始化的随机源。
        - target_count: int；最多接受的 context 起点数。
        - max_attempts: int；最多尝试的随机起点数。
        - min_core_atoms: int；一个 BOX 至少包含的核心受体原子数；零表示不筛选原子数。

    返回值:
        - starts_zyx: np.ndarray int32 ``(K, 3)``；完整图内 80³ BOX 的 ZYX corner index，``K <= target_count``；尝试耗尽时保留已接受起点。

    坐标约定:
        - 受体输入是世界 XYZ，函数先换为完整图连续 ZYX voxel 坐标；随机起点始终是完整图离散 ZYX index，不产生补零 BOX。
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
    """围绕 occurrence 质心生成带体积偏移和世界距离漂移的 bias 起点。

    输入参数:
        - sparse_voxel_zyx: np.ndarray int ``(K_occ, 3)``；完整图内非空体素的零基 ZYX index。
        - full_shape_zyx: Sequence[int] ``(3,)``；完整图 ZYX 体素形状。
        - rng: np.random.Generator；按 PDB 和 split 派生的确定性随机源。
        - num_candidates: int；每个 occurrence 生成的候选数。
        - voxel_size_world: Sequence[float] ``(3,)``；世界 XYZ 体素尺寸，单位为 Å。
        - extra_drift_max_angstrom: float；在 occurrence 体积偏移之外叠加的最大世界距离，单位为 Å。

    返回值:
        - starts_zyx: np.ndarray int32 ``(num_candidates, 3)``；经 ``numpy.rint`` 取整并裁剪到合法完整图范围的 ZYX BOX corner index。

    采样语义:
        - occurrence 质心使用体素中心坐标；体积半径在等体积球内按体积均匀采样；额外漂移先在世界 XYZ 中采样，再换为 ZYX voxel 单位。
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
    """为一个 PDB 的全部 occurrence 构造中心起点和 bias 候选数组。

    输入参数:
        - occurrence_masks_zyx: dict[int, np.ndarray]；occurrence identity 到 int ``(K_occ, 3)`` 完整图 ZYX voxel index 的映射。
        - full_shape_zyx: Sequence[int] ``(3,)``；完整图 ZYX 体素形状。
        - rng: np.random.Generator；该 PDB 的确定性随机源。
        - voxel_size_world: Sequence[float] ``(3,)``；世界 XYZ 体素尺寸，单位为 Å。
        - extra_bias_drift_max_angstrom: float；bias 候选的最大额外世界距离漂移，单位为 Å。
        - bias_candidates_per_occurrence: int；每个 occurrence 的 bias 候选数量。

    返回值:
        - occurrence_id: np.ndarray int32 ``(O,)``；按 occurrence identity 升序排列。
        - center_start_zyx: np.ndarray int32 ``(O, 3)``；每个 occurrence 的质心居中 BOX 起点，与 ``occurrence_id`` 第 0 维对齐。
        - bias_start_zyx: np.ndarray int32 ``(O, B, 3)``；每个 occurrence 的 ``B`` 个 bias 起点，与 ``occurrence_id`` 第 0 维对齐。

    坐标约定:
        - 所有起点都是完整图内真实 80³ BOX 的零基 ZYX corner index；不产生补零坐标。
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
    """读取一个 validation PDB pool 的冻结起点数组。

    输入参数:
        - pool_path: Path；manifest 列出的 validation PDB NPZ。
        - expected_pdb_id: str；manifest 期望的 PDB identity，比较前规范化为小写。

    返回字段:
        - occurrence_id: np.ndarray int32 ``(O,)``；occurrence identity，作为中心和 bias 数组的第 0 维索引。
        - center_start_zyx: np.ndarray int32 ``(O, 3)``；逐 occurrence 的完整图 ZYX BOX corner index。
        - bias_start_zyx: np.ndarray int32 ``(O, B, 3)``；逐 occurrence 的 ``B`` 个 bias BOX corner index。
        - context_start_zyx: np.ndarray int32 ``(C, 3)``；完整图内的 context BOX corner index。

    文件契约:
        - ``pdb_id`` 标量必须与 ``expected_pdb_id`` 一致；起点数组仍采用完整图内 80³ BOX 的零基 ZYX corner 约定。
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
    """按 validation manifest 顺序冻结请求索引，不复制 BOX 起点数组。

    输入参数:
        - validation_pool_directory: Path；已发布 pool 的 validation split 目录；其父目录必须有 manifest。
        - output_path: Path；要原子发布的非压缩 NPZ 路径。
        - seed: int；validation 选择的确定性基准 seed。
        - center_per_occurrence: int；只支持 0 或 1；当前实现仅在值等于 1 时为每个抽中 occurrence 写入一个 center 请求，值为 0 或其他值都不会写入 center。
        - bias_per_occurrence: int；每个抽中 occurrence 无放回选择的 bias 候选数。
        - context_per_occurrence: int；每个抽中 occurrence 选择的 context 候选数；候选不足时允许有放回。

    输出 NPZ 字段:
        - ``validation_pdb_id``：定宽 bytes ``(P,)``；PDB identity 表，所有 ``*_pdb_index`` 字段索引它。
        - ``center_pdb_index``：int32 ``(N_center,)``；center 请求的 PDB 表索引。
        - ``center_occurrence_id``：int32 ``(N_center,)``；center 请求引用的 occurrence identity。
        - ``bias_pdb_index``：int32 ``(N_bias,)``；bias 请求的 PDB 表索引。
        - ``bias_occurrence_id``：int32 ``(N_bias,)``；bias 请求引用的 occurrence identity。
        - ``bias_candidate_index``：int16 ``(N_bias,)``；对应 PDB pool 的 bias 候选轴下标。
        - ``context_pdb_index``：int32 ``(N_context,)``；context 请求的 PDB 表索引。
        - ``context_candidate_index``：int32 ``(N_context,)``；对应 PDB pool 的 context 候选轴下标。

    返回值:
        - counts: dict[str, int]；``pdb_count`` 是 manifest validation PDB 数，``center_count``、``bias_count`` 和 ``context_count`` 分别是三类冻结请求数；这些计数反映实际写入的 NPZ 数组长度。

    选择语义:
        - 每个 PDB 最多无放回抽取 50 个 occurrence；center 和 bias 沿抽中 occurrence 写入，context 候选不足时才允许有放回；NPZ 通过 ``atomic_save_npz`` 原子发布。
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
