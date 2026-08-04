# -*- coding: utf-8 -*-
"""从 A-G 正式资产生成 Stage1 的确定性 80³ BOX 起点池. 

阅读入口:
    1. :func:`build_pdb_box_pool` 为一个 PDB 生成 occurrence、center、bias 和 context 起点字段. 
    2. :func:`freeze_validation_selection` 把 validation 的配置指定请求索引冻结为一个 NPZ.
    3. :func:`build_stage1_box_pools` 批量发布 train/validation 的 PDB NPZ、根 manifest、配置、摘要和 ``_COMPLETE``. 

单 PDB NPZ 字段:
    - pdb_id: 字符串标量; 当前 PDB 身份. 
    - occurrence_id: int32 ``(N_occ,)``; occurrence 编号. 
    - center_start_zyx: int32 ``(N_occ, 3)``; 与 occurrence_id 第 0 维对齐的 center 起点, 轴序为 ZYX. 
    - bias_start_zyx: int32 ``(N_occ, 30, 3)``; 与 occurrence_id 第 0 维对齐的 30 个 bias 起点, 轴序为 ZYX. 
    - context_start_zyx: int32 ``(N_context, 3)``; 当前 PDB 共享的 context 起点, 轴序为 ZYX. 

validation selection NPZ 字段:
    - validation_pdb_id: bytes ``(N_pdb,)``; PDB 身份数组. 
    - center_pdb_index: int32 ``(N_center,)``; 每个 center 请求引用 validation_pdb_id 的下标. 
    - center_occurrence_id: int32 ``(N_center,)``; 与 center_pdb_index 同下标定位 occurrence. 
    - bias_pdb_index: int32 ``(N_bias,)``; 每个 bias 请求引用 validation_pdb_id 的下标. 
    - bias_occurrence_id: int32 ``(N_bias,)``; 与 bias_pdb_index 同下标定位 occurrence. 
    - bias_candidate_index: int16 ``(N_bias,)``; 与 bias_pdb_index 同下标定位 occurrence 的 bias 候选下标. 
    - context_pdb_index: int32 ``(N_context,)``; 每个 context 请求引用 validation_pdb_id 的下标. 
    - context_candidate_index: int32 ``(N_context,)``; 与 context_pdb_index 同下标定位 context 候选下标. 

文件副作用:
    - <output_root>/train/{pdb_id}.npz: train PDB 起点字段. 
    - <output_root>/validation/{pdb_id}.npz: validation PDB 起点字段. 
    - <output_root>/manifest.json: schema 版本和 train/validation PDB 相对路径清单. 
    - <output_root>/validation_selection.npz: 固定 validation 请求索引. 
    - <output_root>/config.json: BOX 形状、候选数量、抽样比例和随机规则. 
    - <output_root>/summary.json: 请求、发布、短图、context 和 selection 计数. 
    - <output_root>/_COMPLETE: 上述正式产物全部写出后最后创建的空完成标记. 

这里保存的是整数起点和身份索引, 不保存 80³ 密度或标签数组; 请求层读取起点, Dataset 再从权威整图资产物化真实 BOX. 
"""

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


# ===================================================================== 辅助函数 =====================================================================
def _pdb_seed(base_seed: int, split_name: str, pdb_id: str) -> int:
    """
    根据 base_seed 等3个传入变量派生与 worker/遍历顺序无关的随机 seed. 

    输入参数:
        - base_seed: int, BOX pool 的基准 seed
        - split_name: str, 当前 `train` 或 `validation` split 名称
        - pdb_id: str, 当前 PDB identity

    输出:
        - pdb_seed: int, 由 SHA-256 前 8 字节解释得到的无符号 64-bit seed
    """
    payload = f"{int(base_seed)}|{str(split_name).lower()}|{str(pdb_id).lower()}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], byteorder="little", signed=False)


def _atomic_save_npz(path: Path, **arrays: np.ndarray) -> None:
    """
    在目标同目录写临时 NPZ, 完成后原子发布. 

    输入参数:
        - path: Path, 正式 NPZ 输出路径
        - arrays: 键值对, 值为 np.ndarray , 每个键成为一个 NPZ 字段

    输出:
        - None: 完整临时文件原子替换到 `path`

    文件副作用:
        - 创建 `path` 的父目录（若尚不存在）; 
        - 用临时 NPZ 原子替换同名正式文件, 临时文件不会保留. 
    """
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
    """
    原子发布 UTF-8 文本或完成标记. 

    输入参数:
        - path: Path, 正式文本输出路径
        - content: str, 要写入的完整 UTF-8 文本; 空字符串用于完成标记

    输出:
        - None: 完整临时文件原子替换到 `path`

    文件副作用:
        - 创建 `path` 的父目录（若尚不存在）; 
        - 用 UTF-8 临时文件原子替换同名正式文件, 临时文件不会保留. 
    """
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
    """
    加载 path 对应的文件, 查找并排序唯一 PDB identity(pdb_id). 

    输入参数:
        - path: str | Path, split JSON 路径; 内容可为列表或包含 `entries/pairs/items/pdb_ids` 字段的 JSON 对象

    输出:
        - pdb_ids: tuple[str,...], 去重并按字典序排列的规范化 PDB identity
    """
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
# ===================================================================== 辅助函数 =====================================================================







# -----------------------------------------------------------------------------------------------------------------------------------------------------
def _load_occurrence_masks(path: Path, expected_shape_zyx: Sequence[int]) -> dict[int, np.ndarray]:
    """
    读取 path（来自 E3 schema-v3）, 返回 occurrence 的占据 mask. 

    输入参数:
        - path: Path, `ligand_area.npz` 路径
        - expected_shape_zyx: Sequence[int], (3,), 对应 exp 完整图的 ZYX shape, 仅用于检验

    输出:
        - masks: dict[int,np.ndarray], occurrence_id 到 (K_occ,3) [也就是: int32 ZYX voxel indices] 的映射
    """
    # np.ndarray[int64], (3,), exp 完整图的 ZYX shape, 用于验证所有稀疏坐标边界. 
    expected_shape = np.asarray(expected_shape_zyx, dtype=np.int64)
    with np.load(path, allow_pickle=False) as data:
        if "schema_version" not in data or int(np.asarray(data["schema_version"]).item()) != 3:
            raise ValueError(f"{path}: BOX pool 只接受 ligand_area schema_version=3。")
        if "grid_shape_zyx" in data and not np.array_equal(np.asarray(data["grid_shape_zyx"], dtype=np.int64), expected_shape):
            raise ValueError(f"{path}: grid_shape_zyx 与 exp grid 不一致。")
        mask_keys = sorted(
            (key for key in data.files if key.startswith("mask_") and key[5:].isdigit()),
            key=lambda key: int(key[5:]),
        )
        # dict[int,np.ndarray[int32]], occurrence_id -> (K_occ,3) 非空 ZYX 体素坐标. 
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
    """
    按现有随机 context 规则生成合法 80³ 起点. 

    输入参数:
        - receptor_coords_world: np.ndarray, (N_atom,3), receptor 重原子的世界 XYZ 坐标
        - full_origin_world: Sequence[float], (3,), 完整图 corner 的世界 XYZ 原点
        - voxel_size_world: Sequence[float], (3,), 世界 XYZ 每体素尺寸
        - full_shape_zyx: Sequence[int], (3,), 完整图 ZYX shape
        - rng: np.random.Generator, 随机数生成对象（当前实现下已保证稳定性）
        - target_count: int, 最多保留的 context 起点数
        - max_attempts: int, 最多采样尝试次数
        - min_core_atoms: int, 一个 context BOX 至少包含的 core receptor 原子数

    输出:
        - context_starts_zyx: np.ndarray, (N_context,3), int32, 合法的完整图离散 ZYX voxel-index context BOX corner （左下）起点; N_context 不超过 `target_count`

    每个轴在完整图合法起点闭区间内均匀采样; BOX core 中 receptor 重原子数不少于 ``min_core_atoms`` 才保留, 候选不绑定 occurrence, 达到目标数或耗尽尝试次数即停止, 重复起点保持原样. 
    """
    coords = np.asarray(receptor_coords_world, dtype=np.float64)
    origin = np.asarray(full_origin_world, dtype=np.float64)
    voxel_size = np.asarray(voxel_size_world, dtype=np.float64)
    full_shape = np.asarray(full_shape_zyx, dtype=np.int64)
    if coords.ndim != 2 or coords.shape[1] != 3 or not np.isfinite(coords).all():
        raise ValueError("receptor_coords_world 必须为有限 [N,3] XYZ 数组。")
    if origin.shape != (3,) or voxel_size.shape != (3,) or np.any(voxel_size <= 0):
        raise ValueError("full_origin_world/voxel_size_world 必须为合法 (3,) XYZ。")
    resolve_stage1_start((0, 0, 0), full_shape)   # 仅用于检验(该函数内置报错)
    if int(target_count) <= 0 or int(max_attempts) <= 0 or int(min_core_atoms) < 0:
        raise ValueError("context target/max_attempts 必须为正整数，min_core_atoms 必须为非负整数。")

    # np.ndarray[float64], (N_atom,3), 世界 XYZ 坐标转换成完整图连续 voxel XYZ 坐标. 
    local_xyz = (coords - origin[None, :]) / voxel_size[None, :]
    # np.ndarray[float64], (N_atom,3), 换轴后的连续 ZYX voxel 坐标. 
    local_zyx = local_xyz[:, [2, 1, 0]]
    # np.ndarray[int64], (3,), 各轴合法起点的闭区间上界 full_shape-box_shape. 
    max_start = full_shape - np.asarray(STAGE1_BOX_SHAPE_ZYX, dtype=np.int64)
    starts: list[np.ndarray] = []
    for _ in range(int(max_attempts)):
        # np.ndarray[int64], (3,), 当前随机 context 的合法 ZYX 起点. 
        requested = np.asarray(
            [rng.integers(0, int(axis_max) + 1) for axis_max in max_start],
            dtype=np.int64,
        )
        if int(min_core_atoms) > 0:
            upper = requested + np.asarray(STAGE1_BOX_SHAPE_ZYX, dtype=np.int64)
            # np.ndarray[bool], (N_atom,), receptor 原子是否落在当前 80³ core 的半开区间内.
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
    voxel_size_world: Sequence[float] = (1.0, 1.0, 1.0),
    extra_drift_max_angstrom: float = 0.0,
) -> np.ndarray:
    """
    对于一列 occurrence, 按"体积均匀球偏移法"生成相应的冻结 bias-boxs 起点. 

    输入参数:
        - sparse_voxel_zyx: np.ndarray, (K_occ,3), occurrence 的非空完整图 ZYX voxel indices
        - full_shape_zyx: Sequence[int], (3,), 完整图 ZYX shape
        - rng: np.random.Generator, 当前 PDB 独立的稳定随机流
        - num_candidates: int, 要生成的 bias 候选数; 正式值为 30
        - box_shape_zyx: Sequence[int], (3,), 输出 BOX 的 ZYX shape

    球半径固定为 ``R=(3*K_occ/(4*pi))**(1/3)`` voxel; 方向均匀、半径使用 ``U**(1/3)``, 不重试、不去重. 每个偏移后的请求都调用统一起点解析器. 

    输出:
        - bias_starts_zyx: np.ndarray, (num_candidates,3), int32, 按生成顺序保存的完整图离散 ZYX voxel-index BOX corner 起点
    """
    sparse = np.asarray(sparse_voxel_zyx, dtype=np.int64)
    if sparse.ndim != 2 or sparse.shape[1] != 3 or sparse.shape[0] == 0:
        raise ValueError("sparse_voxel_zyx 必须为非空 (K,3) ZYX 数组。")
    if int(num_candidates) <= 0:
        raise ValueError("num_candidates 必须为正整数。")
    box_shape = np.asarray(box_shape_zyx, dtype=np.float64)
    voxel_size_xyz = np.asarray(voxel_size_world, dtype=np.float64)
    if voxel_size_xyz.shape != (3,) or not np.isfinite(voxel_size_xyz).all() or np.any(voxel_size_xyz <= 0):
        raise ValueError("voxel_size_world 必须为逐轴有限且为正的 (3,) XYZ 数组。")
    if not np.isfinite(float(extra_drift_max_angstrom)) or float(extra_drift_max_angstrom) < 0:
        raise ValueError("extra_drift_max_angstrom 必须为有限非负数。")
    # np.ndarray[float64], (3,), occurrence 体素中心的连续 corner-语义 ZYX 质心. 
    centroid_corner_zyx = sparse.astype(np.float64).mean(axis=0) + 0.5
    # float, 与 K_occ 等体积球的 voxel 半径, 用于限定 bias center 的采样范围. 
    radius = float((3.0 * sparse.shape[0] / (4.0 * np.pi)) ** (1.0 / 3.0))
    # np.ndarray[float64], (N_bias,3), 归一化后的均匀球面方向. 
    directions = rng.normal(size=(int(num_candidates), 3))  # 生成形状为 (K, 3) 的数组, 每个元素服从均值0方差1的正态分布
    direction_norm = np.linalg.norm(directions, axis=1, keepdims=True)
    directions = directions / np.maximum(direction_norm, np.finfo(np.float64).tiny)
    # np.ndarray[float64], (N_bias,), 立方根变换使偏移在球体体积内均匀. 
    radii = radius * np.cbrt(rng.random(int(num_candidates)))                  # 生成形状为 (K,) 的一维数组, 每个元素在[0,1) 均匀分布
    # np.ndarray[float64], (N_bias,3), occurrence 质心加球内偏移后的候选 BOX 中心. 
    biased_centers = centroid_corner_zyx[None, :] + directions * radii[:, None]
    if float(extra_drift_max_angstrom) > 0:
        # 额外漂移在真实 XYZ 空间采样，再按实际体素尺寸换算为 ZYX 体素位移。
        drift_directions_xyz = rng.normal(size=(int(num_candidates), 3))
        drift_norm = np.linalg.norm(drift_directions_xyz, axis=1, keepdims=True)
        drift_directions_xyz = drift_directions_xyz / np.maximum(
            drift_norm,
            np.finfo(np.float64).tiny,
        )
        drift_lengths = rng.uniform(0.0, float(extra_drift_max_angstrom), size=int(num_candidates))
        drift_voxel_xyz = drift_directions_xyz * drift_lengths[:, None] / voxel_size_xyz[None, :]
        biased_centers = biased_centers + drift_voxel_xyz[:, [2, 1, 0]]
    requested_starts = np.rint(biased_centers - box_shape[None, :] / 2.0)
    return np.asarray(
        [resolve_stage1_start(start, full_shape_zyx, box_shape_zyx) for start in requested_starts],
        dtype=np.int32,
    )


def build_occurrence_pool_rows(
    occurrence_masks_zyx: dict[int, np.ndarray],
    full_shape_zyx: Sequence[int],
    rng: np.random.Generator,
    voxel_size_world: Sequence[float] = (1.0, 1.0, 1.0),
    extra_bias_drift_max_angstrom: float = 0.0,
) -> dict[str, np.ndarray]:
    """
    为一个 PDB 的全部非空 occurrence 构造 center 与 30 个 bias 起点. 

    输入参数:
        - occurrence_masks_zyx: dict[int,np.ndarray], occurrence_id 到 (K_occ,3) ZYX voxel indices 的映射
        - full_shape_zyx: Sequence[int], (3,), 完整图 ZYX shape
        - rng: np.random.Generator, 当前 PDB 的稳定随机流

    输出:
        - rows: dict[str,np.ndarray], 包含:
            - occurrence_id: (N_occ,), int32, 排序后的 occurrence identity
            - center_start_zyx: (N_occ,3), int32, 每个 occurrence 的唯一完整图离散 ZYX voxel-index BOX corner 起点
            - bias_start_zyx: (N_occ,30,3), int32, 每个 occurrence 的冻结完整图离散 ZYX voxel-index BOX corner 起点
    """
    # np.ndarray[int32], (N_occ,), 将 occurrence_id 升序排列. 
    occurrence_ids = np.asarray(sorted(int(value) for value in occurrence_masks_zyx), dtype=np.int32)  # 遍历字典 occurrence_masks_zyx 的所有键
    # list[(z,y,x)], 长度 N_occ, 每个 occurrence 唯一的居中 BOX 起点. 
    centers: list[tuple[int, int, int]] = []
    # list[np.ndarray], 长度 N_occ, 每项形状 (30,3), 保存该 occurrence 的 bias 起点. 
    biases: list[np.ndarray] = []
    for occurrence_id in occurrence_ids.tolist():
        sparse = np.asarray(occurrence_masks_zyx[occurrence_id], dtype=np.int32)
        centers.append(centered_start_from_sparse_mask(sparse, full_shape_zyx))
        biases.append(
            sample_bias_starts(
                sparse,
                full_shape_zyx,
                rng,
                num_candidates=30,
                voxel_size_world=voxel_size_world,
                extra_drift_max_angstrom=extra_bias_drift_max_angstrom,
            )
        )
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
    context_min_core_atoms: int = _CONTEXT_MIN_CORE_ATOMS,
    extra_bias_drift_max_angstrom: float = 0.0,
) -> dict[str, np.ndarray]:
    """
    从一份 A-G PDB 三件套（"exp.npz"、"ligand_area.npz"、"receptor_tokens.npz"）构造 center、bias 和 context 索引池. 

    输入参数:
        - data_root: str | Path, A-G 正式数据根目录
        - pdb_id: str, 当前 PDB identity
        - split_name: str, 当前 split 名称(train/val); 参与 PDB seed 派生
        - seed: int, BOX pool 基准 seed

    输出:
        - pool: dict[str,np.ndarray], 包含:
            - pdb_id: 标量字符串数组, 当前 PDB identity
            - occurrence_id: (N_occ,), int32, 排序后的 occurrence identity
            - center_start_zyx: (N_occ,3), int32, 完整图离散 ZYX voxel-index 居中 BOX corner 起点
            - bias_start_zyx: (N_occ,30,3), int32, 完整图离散 ZYX voxel-index bias BOX corner 起点
            - context_start_zyx: (N_context,3), int32, 整 PDB 共享的完整图离散 ZYX voxel-index context BOX corner 起点
    """
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
    # dict[int,np.ndarray[int32]], occurrence_id -> (K_occ,3) 完整图 ZYX 稀疏占据坐标. 
    occurrence_masks = _load_occurrence_masks(density_dir / "ligand_area.npz", grid_shape_zyx)
    if not occurrence_masks:
        raise ValueError(f"{pdb_id}: ligand_area.npz 不含任何 occurrence mask。")

    with np.load(root / "parse" / pdb_id / "receptor_tokens.npz", allow_pickle=False) as data:
        if "coords" not in data:
            raise KeyError(f"{pdb_id}: receptor_tokens.npz 缺少 coords。")
        receptor_coords = np.asarray(data["coords"], dtype=np.float32)
    rng = np.random.default_rng(_pdb_seed(seed, split_name, pdb_id))
    # dict[str,np.ndarray], 对齐的 occurrence_id/center(N,3)/bias(N,30,3) 冻结表. 
    occurrence_rows = build_occurrence_pool_rows(
        occurrence_masks,
        grid_shape_zyx,
        rng,
        voxel_size_world=voxel_size,
        extra_bias_drift_max_angstrom=extra_bias_drift_max_angstrom,
    )
    # np.ndarray[int32], (N_context,3), 与 occurrence 无关且满足 core 原子数门槛的起点. 
    contexts = generate_context_starts(
        receptor_coords_world=receptor_coords,
        full_origin_world=origin,
        voxel_size_world=voxel_size,
        full_shape_zyx=grid_shape_zyx,
        rng=rng,
        min_core_atoms=context_min_core_atoms,
    )
    return {
        "pdb_id": np.asarray(pdb_id),
        **occurrence_rows,
        "context_start_zyx": contexts,
    }






# -----------------------------------------------------------------------------------------------------------------------------------------------------
def build_stage1_box_pools(
    data_root: str | Path,
    train_split: str | Path,
    validation_split: str | Path,
    output_root: str | Path,
    seed: int = _POOL_SEED,
    center_per_occurrence: int = 1,
    bias_per_occurrence: int = 5,
    context_per_occurrence: int = 3,
    context_min_core_atoms: int = _CONTEXT_MIN_CORE_ATOMS,
    extra_bias_drift_max_angstrom: float = 0.0,
) -> dict[str, object]:
    """
    端到端生成 train/validation pool、冻结 selection 与完成标记. 

    输入参数:
        - data_root: ``str | Path``; A-G 正式数据根目录. 
        - train_split: ``str | Path``; 冻结 train split JSON 路径. 
        - validation_split: ``str | Path``; 冻结 validation split JSON 路径. 
        - output_root: ``str | Path``; 通常为 ``<stage1_preparation>/box_pool`` 的输出根目录: .../stage1_preparation/box_pool
        - seed: int; BOX pool 基准 seed. 

    输出字段:
        - seed: int; 本次 BOX pool 使用的 seed. 
        - train: dict[str, int]; train 请求数、发布数、短图数和 context 数量统计. 
        - validation: dict[str, int]; validation 请求数、发布数和 context 数量统计. 
        - validation_selection: dict[str, int]; 冻结 validation 的 PDB、center、bias 和 context 数量. 
        - manifest: dict[str, int]; manifest 中 train 与 validation PDB NPZ 数量. 

    文件副作用:
        目录 `<output_root>`、`<output_root>/train` 和 `<output_root>/validation` 不存在时创建; 写入新产物前删除根目录已有的 `_COMPLETE`、`manifest.json` 和 `validation_selection.npz`. 
        - `<output_root>/train/{pdb_id}.npz`: 每个成功处理的 train PDB 创建或原子替换一个 NPZ; 起点最后一维按完整图 ZYX 排列, 三轴短于 80 的 PDB 不发布 NPZ. 
            - pdb_id: 字符串标量数组, 当前 PDB identity. 
            - occurrence_id: int32, (N_occ,), occurrence 编号. 
            - center_start_zyx: int32, (N_occ, 3), 与 occurrence_id 第一维对齐的居中 BOX 起点. 
            - bias_start_zyx: int32, (N_occ, 30, 3), 与 occurrence_id 第一维对齐的 30 个 bias BOX 起点. 
            - context_start_zyx: int32, (N_context, 3), 当前 PDB 共享的 context BOX 起点. 
        - <output_root>/validation/{pdb_id}.npz: 每个成功处理的 validation PDB 创建或原子替换一个 NPZ. 
            - pdb_id: 字符串标量数组; 当前 PDB 身份. 
            - occurrence_id: int32 ``(N_occ,)``; occurrence 编号. 
            - center_start_zyx: int32 ``(N_occ, 3)``; 与 occurrence_id 第 0 维对齐的 center 起点, 轴序为 ZYX. 
            - bias_start_zyx: int32 ``(N_occ, 30, 3)``; 与 occurrence_id 第 0 维对齐的 bias 起点, 轴序为 ZYX. 
            - context_start_zyx: int32 ``(N_context, 3)``; 当前 PDB 共享的 context 起点, 轴序为 ZYX. 
        - `<output_root>/manifest.json`: JSON object, 创建或原子替换. 
            - schema_version: int, 清单格式版本. 
            - splits: dict[str, list[dict[str, str]]], 包含 train 和 validation 的 PDB 清单. 
                - train: list[dict[str, str]], 每个元素包含 `pdb_id: str` 与 `path: str`, `path` 是相对于 box-pool 根目录的 NPZ 路径. 
                - validation: list[dict[str, str]], 每个元素包含 `pdb_id: str` 与 `path: str`, `path` 是相对于 box-pool 根目录的 NPZ 路径. 
        - `<output_root>/validation_selection.npz`: 创建或原子替换; 每个索引数组分别与对应的 PDB、occurrence 或候选起点集合对齐. 
            - validation_pdb_id: bytes, (N_pdb,), validation PDB identity. 
            - center_pdb_index: int32, (N_center,), 索引 validation_pdb_id. 
            - center_occurrence_id: int32, (N_center,), 与 center_pdb_index 共同定位居中 BOX 的 occurrence. 
            - bias_pdb_index: int32, (N_bias,), 索引 validation_pdb_id. 
            - bias_occurrence_id: int32, (N_bias,), 与 bias_pdb_index 共同定位 bias BOX 的 occurrence. 
            - bias_candidate_index: int16, (N_bias,), 定位 occurrence 的第几个 bias 起点. 
            - context_pdb_index: int32, (N_context,), 索引 validation_pdb_id. 
            - context_candidate_index: int32, (N_context,), 定位 PDB 的第几个 context 起点. 
        - `<output_root>/config.json`: JSON object, 创建或原子替换. 
            - box_shape_zyx: list[int], (3,), BOX 的完整图 ZYX 形状. 
            - bias_candidates_per_occurrence: int, 每个 occurrence 冻结的 bias 候选数. 
            - bias_radius_formula: str, bias 半径计算公式. 
            - bias_selected_per_epoch: int, 每个 epoch 从 bias 候选中选择的数量. 
            - context_generator: dict, context 起点生成规则. 
                - sampling: str, context 起点采样方式. 
                - target_count: int, 每个 PDB 目标 context 起点数. 
                - max_attempts: int, context 起点采样最大尝试次数. 
                - min_core_receptor_heavy_atoms: int, context 起点要求覆盖的最少受体重原子数. 
                - ligand_filter: bool, context 起点是否使用配体过滤. 
            - occurrence_cap_per_pdb_per_epoch: int, 每个 PDB 每个 epoch 的 occurrence 上限. 
            - entry_ratio: dict[str, int], center、bias、context 的读取比例. 
                - center: int, center 读取比例. 
                - bias: int, bias 读取比例. 
                - context: int, context 读取比例. 
            - train_random_rotation_90_degree: bool, train 是否使用 90 度整数旋转增强. 
            - seed: int, BOX pool 基准 seed. 
            - seed_rule: str, 从 seed、split_name 和 pdb_id 派生 PDB seed 的规则. 
        - `<output_root>/summary.json`: JSON object, 创建或原子替换. 
            - seed: int, 本次 BOX pool 使用的 seed. 
            - train: dict[str, int], train 统计. 
                - requested_pdb: int, 请求处理的 train PDB 数量. 
                - published_pdb: int, 成功发布 NPZ 的 train PDB 数量. 
                - short_map_count: int, 完整图三轴短于 80 的 train PDB 数量. 
                - zero_context_pdb_count: int, 没有 context 起点的 train PDB 数量. 
                - underfilled_context_pdb_count: int, context 起点少于 3 个的 train PDB 数量. 
            - validation: dict[str, int], validation 统计. 
                - requested_pdb: int, 请求处理的 validation PDB 数量. 
                - published_pdb: int, 成功发布 NPZ 的 validation PDB 数量. 
                - zero_context_pdb_count: int, 没有 context 起点的 validation PDB 数量. 
                - underfilled_context_pdb_count: int, context 起点少于 3 个的 validation PDB 数量. 
            - validation_selection: dict[str, int], validation_selection 统计. 
                - pdb_count: int, validation PDB 数量. 
                - center_count: int, center 读取项数量. 
                - bias_count: int, bias 读取项数量. 
                - context_count: int, context 读取项数量. 
            - manifest: dict[str, int], train 与 validation 清单中的 NPZ 数量. 
                - train: int, manifest.splits.train 的 NPZ 数量. 
                - validation: int, manifest.splits.validation 的 NPZ 数量. 
        - `<output_root>/_COMPLETE`: 上述产物全部成功写出后最后创建的空完成标记; 根目录中未列出的其他文件不会被本函数删除, 单 PDB NPZ 只会按相同路径原子替换. 
        示例: 若 `output_root` 为 `C:\\data\\stage1_preparation\\box_pool`, 根目录产物包括 `manifest.json`、`validation_selection.npz`、`config.json`、`summary.json` 和 `_COMPLETE`, 单 PDB NPZ 位于其 `train\\` 或 `validation\\` 子目录. 

    已存在的单 PDB pool 会重新按同一稳定 seed 原子覆盖; 只有全部输入成功后才发布
    根目录 ``_COMPLETE``. train 中三轴短于 80 的 PDB 只计入普通 summary, validation
    若出现同类错误则 fail-fast, 因为 split 选择本应已经排除它. 
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
                pool = build_pdb_box_pool(
                    data_root,
                    pdb_id,
                    split_name,
                    seed=seed,
                    context_min_core_atoms=context_min_core_atoms,
                    extra_bias_drift_max_angstrom=extra_bias_drift_max_angstrom,
                )
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
        center_per_occurrence=center_per_occurrence,
        bias_per_occurrence=bias_per_occurrence,
        context_per_occurrence=context_per_occurrence,
    )
    summary["validation_selection"] = selection_summary
    summary["manifest"] = {split_name: len(entries) for split_name, entries in manifest_entries.items()}
    config = {
        "box_shape_zyx": list(STAGE1_BOX_SHAPE_ZYX),
        "bias_candidates_per_occurrence": 30,
        "bias_radius_formula": "R=(3*K_occ/(4*pi))**(1/3)",
        "extra_bias_drift_max_angstrom": float(extra_bias_drift_max_angstrom),
        "bias_selected_per_epoch": int(bias_per_occurrence),
        "context_generator": {
            "sampling": "uniform_integer_legal_start_per_axis",
            "target_count": _CONTEXT_TARGET,
            "max_attempts": _CONTEXT_MAX_ATTEMPTS,
            "min_core_receptor_heavy_atoms": int(context_min_core_atoms),
            "ligand_filter": False,
        },
        "occurrence_cap_per_pdb_per_epoch": 50,
        "entry_ratio": {
            "center": int(center_per_occurrence),
            "bias": int(bias_per_occurrence),
            "context": int(context_per_occurrence),
        },
        "train_random_rotation_90_degree": True,
        "seed": int(seed),
        "seed_rule": "sha256(base_seed|split_name|pdb_id) first_uint64",
    }
    _atomic_write_text(root / "config.json", json.dumps(config, ensure_ascii=False, indent=2) + "\n")
    _atomic_write_text(root / "summary.json", json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    _atomic_write_text(root / "_COMPLETE", "")
    return summary

def freeze_validation_selection(                 # 最后才用到的函数
    validation_pool_directory: str | Path,
    output_path: str | Path,
    seed: int = _POOL_SEED,
    center_per_occurrence: int = 1,
    bias_per_occurrence: int = 5,
    context_per_occurrence: int = 3,
) -> dict[str, int]:
    """
    一次冻结 validation 的配置指定比例真实读取项.

    输入参数:
        - validation_pool_directory: ``str | Path``, 是.../(stage1_preparation)/box_pool/validation(这是个文件夹), 仅用于定位 boox_pool 这层地址, 从而找到 manifest.json. 
        - output_path: ``str | Path``; ``validation_selection.npz`` 正式路径. 
        - seed: int; validation 冻结抽样 seed. 

    输出字段:
        - pdb_count: int; validation PDB 数量. 
        - center_count: int; 冻结 center 请求数量. 
        - bias_count: int; 冻结 bias 请求数量. 
        - context_count: int; 冻结 context 请求数量. 

    文件副作用:
        - output_path: ``validation_selection.npz``; 父目录由写入函数创建, 文件被创建或原子替换, 其他 validation pool 文件不会被删除. 
            - validation_pdb_id: bytes ``(N_pdb,)``; validation PDB 身份. 
            - center_pdb_index: int32 ``(N_center,)``; 索引 validation_pdb_id. 
            - center_occurrence_id: int32 ``(N_center,)``; 与 center_pdb_index 共同定位 center occurrence. 
            - bias_pdb_index: int32 ``(N_bias,)``; 索引 validation_pdb_id. 
            - bias_occurrence_id: int32 ``(N_bias,)``; 与 bias_pdb_index 共同定位 bias occurrence. 
            - bias_candidate_index: int16 ``(N_bias,)``; 定位 occurrence 的 bias 候选下标. 
            - context_pdb_index: int32 ``(N_context,)``; 索引 validation_pdb_id. 
            - context_candidate_index: int32 ``(N_context,)``; 定位 PDB 的 context 候选下标. 
        示例: 若 ``output_path`` 为 ``C:\\data\\stage1_preparation\\box_pool\\validation_selection.npz``, 则文件直接写入此路径. 
    """
    from src.datasets.stage1_requests import _load_pdb_pool

    if int(center_per_occurrence) not in (0, 1):
        raise ValueError("center_per_occurrence 只允许 0 或 1。")
    if not 0 <= int(bias_per_occurrence) <= 30:
        raise ValueError("bias_per_occurrence 必须位于 [0,30]。")
    if int(context_per_occurrence) < 0:
        raise ValueError("context_per_occurrence 必须为非负整数。")
    if int(center_per_occurrence) + int(bias_per_occurrence) + int(context_per_occurrence) <= 0:
        raise ValueError("validation 每个 occurrence 至少启用一种请求。")

    pool_directory = Path(validation_pool_directory)
    manifest_entries = _load_manifest_pool_paths(
        pool_directory.parent,
        pool_directory.name,
        require_complete=False,
    )
    # list[_PdbPool], 与 manifest 顺序一致的 validation PDB pool. 
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
        # 对每个 pdb 循环, pool 是这个pdb的pool
        occurrence_count = int(pool.occurrence_id.shape[0])
        selected_rows = rng.choice(occurrence_count, size=min(50, occurrence_count), replace=False)
        context_count = int(pool.context_start_zyx.shape[0])
        for occurrence_row in selected_rows.tolist():
            occurrence_id = int(pool.occurrence_id[occurrence_row])
            if int(center_per_occurrence) == 1:
                center_pdb_index.append(pdb_index)
                center_occurrence_id.append(occurrence_id)
            for candidate_index in rng.choice(
                30,
                size=int(bias_per_occurrence),
                replace=False,
            ).tolist():
                bias_pdb_index.append(pdb_index)
                bias_occurrence_id.append(occurrence_id)
                bias_candidate_index.append(int(candidate_index))
            if context_count > 0 and int(context_per_occurrence) > 0:
                for candidate_index in rng.choice(
                    context_count,
                    size=int(context_per_occurrence),
                    replace=context_count < int(context_per_occurrence),
                ).tolist():
                    context_pdb_index.append(pdb_index)
                    context_candidate_index.append(int(candidate_index))

    max_pdb_width = max(1, max(len(value.encode("utf-8")) for value in pdb_ids))
    # dict[str,np.ndarray], validation_selection.npz 的 ragged 索引表, 不复制 BOX 起点本身. 
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


def _main() -> None:
    """
    解析 CLI 并执行 Stage1 BOX pool 一键生成. 

    输出:
        - None: 写出 BOX pool 产物, 并把 summary 以 JSON 打印到标准输出

    文件副作用:
        - 将命令行参数转交给 `build_stage1_box_pools`; 实际写入的目录、NPZ、清单、配置和 `_COMPLETE` 完成标记由 `--output-root` 与该函数的文件副作用契约决定. 
    """

    parser = argparse.ArgumentParser(description="生成 AdaLigand Stage1 train/validation BOX pool。")
    parser.add_argument("--data-root", required=True, help="A—G 正式数据根目录。")
    parser.add_argument("--train-split", required=True, help="冻结 train.json。")
    parser.add_argument("--validation-split", required=True, help="冻结 validation.json。")
    parser.add_argument("--output-root", required=True, help="stage1_preparation/box_pool 输出目录。")
    parser.add_argument("--seed", type=int, default=_POOL_SEED, help="BOX pool 基准随机 seed。")
    parser.add_argument("--center-per-occurrence", type=int, default=1, help="每个 occurrence 的 center 请求数。")
    parser.add_argument("--bias-per-occurrence", type=int, default=5, help="每个 occurrence 的 bias 请求数。")
    parser.add_argument("--context-per-occurrence", type=int, default=3, help="每个 occurrence 的 context 请求数。")
    parser.add_argument(
        "--context-min-core-atoms",
        type=int,
        default=_CONTEXT_MIN_CORE_ATOMS,
        help="context BOX 的最少受体重原子数；0 表示不设门槛。",
    )
    parser.add_argument(
        "--extra-bias-drift-max-angstrom",
        type=float,
        default=0.0,
        help="叠加在原 bias 上的额外物理漂移长度上界（Å）。",
    )
    arguments = parser.parse_args()
    summary = build_stage1_box_pools(
        data_root=arguments.data_root,
        train_split=arguments.train_split,
        validation_split=arguments.validation_split,
        output_root=arguments.output_root,
        seed=arguments.seed,
        center_per_occurrence=arguments.center_per_occurrence,
        bias_per_occurrence=arguments.bias_per_occurrence,
        context_per_occurrence=arguments.context_per_occurrence,
        context_min_core_atoms=arguments.context_min_core_atoms,
        extra_bias_drift_max_angstrom=arguments.extra_bias_drift_max_angstrom,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    _main()
