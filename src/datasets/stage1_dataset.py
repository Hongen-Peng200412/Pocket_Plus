# -*- coding: utf-8 -*-
"""从 Stage1 请求和 A-G 正式资产物化一个可直接训练或推理的 80³ 样本. 

阅读入口:
    1. :class:`Stage1Dataset` 接收 ``ResolvedStage1Crop``, 根据请求身份读取整图、受体原子和标签. 
    2. :meth:`Stage1Dataset._materialize` 裁剪密度与标签, 生成体素监督, 并为 Find 生成受体原子字段. 
    3. :func:`_to_tensor_sample` 按字段契约转换 dtype, ``stage1_collate.py`` 再把多个单样本组成批次. 

单样本顶层字段:
    - pdb_id: Python str; 当前 PDB 身份. 
    - request_role: Python str; 请求来源角色. 
    - occurrence_id: Python int 或 None; 请求引用的 occurrence 编号. 
    - candidate_index: Python int 或 None; 请求引用的 bias/context 候选下标. 
    - box_start_zyx: int32 ``(3,)``; 完整图离散 ZYX voxel-index BOX corner 起点. 
    - box_shape_zyx: int64 ``(3,)``; 固定为 ``(80, 80, 80)`` 的 ZYX BOX 形状. 
    - box_origin_world: float32 ``(3,)``; BOX voxel-grid corner 的世界 XYZ 坐标, 单位 Å. 
    - voxel_size_world: float32 ``(3,)``; 世界 XYZ 每体素尺寸, 单位 Å. 
    - density_input: float32 ``(C_density, 80, 80, 80)``; producer 专属的 ZYX 密度通道. 
    - hardmask: bool ``(80, 80, 80)``; 核心 BOX 中受体原子占据体素. 
    - voxel_label: bool ``(80, 80, 80)``; 核心 BOX 中 binding 受体原子占据体素. 
    - ligand_area_target: bool ``(80, 80, 80)``; 完整配体区域并集的 BOX 裁剪. 
    - protein_mainchain_target: int64 ``(80, 80, 80)``; 蛋白主链背景/N/CA/C/O 类别编号. 
    - nucleic_mainchain_target: int64 ``(80, 80, 80)``; 核酸主链背景/P/O5'/C5'/C4'/C3'/O3' 类别编号. 
    - ligand_inverse_distance_target: float32 ``(80, 80, 80)``; 由最近配体原子距离按 ``1/(1+distance_Å)`` 转换的回归目标. 
    - atom_global_indices: int64 ``(N_A,)``; Find 选择的受体原子在完整受体数组中的编号. 
    - atom_feat: float32 ``(N_A, 49)``; 与 atom_global_indices 第 0 维逐原子对齐的基础特征. 
    - atom_coord_world: float32 ``(N_A, 3)``; 逐原子世界 XYZ 坐标, 单位 Å. 
    - atom_coord_local_voxel: float32 ``(N_A, 3)``; 逐原子 BOX-local 连续 voxel XYZ 坐标. 
    - atom_coord_centered_world: float32 ``(N_A, 3)``; 逐原子相对 BOX 中心的世界 XYZ 坐标, 单位 Å. 
    - atom_is_in_core_box: bool ``(N_A,)``; 逐原子是否位于核心 80³ BOX. 
    - atom_label: bool ``(N_A,)``; 与 atom_global_indices 第 0 维逐原子对齐的 binding 标签. 

文件读取:
    - density/<pdb_id>/exp.npz: ``grid``、``voxel_size``、``origin`` 三个实验密度字段. 
    - density/<pdb_id>/sim.npz: Find 额外读取的模拟密度字段, 几何必须与 exp 完全一致. 
    - density/<pdb_id>/ligand_area.npz: ``union_mask`` 完整图配体区域并集. 
    - density/<pdb_id>/ligand_dist.npz: ``distance`` float16 最近配体原子距离图及其空间契约字段. 
    - parse/<pdb_id>/receptor_tokens.npz: ``coords``、``feat`` 以及辅助监督需要的 ``res_type``、``atom_name``. 
    - labels/<pdb_id>/atom_labels.npz: ``binding_atom`` 逐受体原子结合区域标签. 

缓存只减少同一 DataLoader worker 的重复读取, 不改变请求顺序、裁剪内容或标签数值; Dataset 不补零、不现场删除失败样本. 
"""

from __future__ import annotations

import random
from collections import OrderedDict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from src.auxiliary_supervision import (
    NUCLEIC_MAINCHAIN_CLASS_BY_ATOM_NAME,
    PROTEIN_MAINCHAIN_CLASS_BY_ATOM_NAME,
)
from src.datasets.box_geometry import (
    build_atom_coordinates,
    build_atom_features,
    build_hardmask_from_atom_coordinates,
    select_atoms_for_box,
)
from src.datasets.density_channel_builder import (
    ALL_CHANNEL_NAMES,
    DensityChannelConfig,
    build_density_channels,
)
from src.datasets.stage1_collate import Stage1BatchCollator
from src.datasets.stage1_requests import (
    STAGE1_BOX_SHAPE_ZYX,
    ResolvedStage1Crop,
    build_request_source,
    resolve_stage1_start,
)
from src.stage1_producers import FIND_MODEL_NAMES, STAGE1_MODEL_NAMES


_AUXILIARY_SUPERVISION_MODEL_NAMES = {"Find_1", "unet_c1"}


class _ByteLruCache:
    """
    按数组真实字节数限制 worker-local LRU 缓存. 

    输入参数:
        - max_bytes: int, 当前 DataLoader worker 可用于缓存 CPU NumPy 资产的最大总字节数; 0 表示禁用

    状态:
        - values: OrderedDict[str,tuple[dict[str,np.ndarray],int]], 按最近使用顺序保存资产及其字节数
        - current_bytes: int, 当前缓存总字节数

    缓存 key 由资产类型和 ``pdb_id`` 组成, value 始终是 CPU NumPy 数组字典. 它不缓存已经裁好的 BOX, 因此不同请求仍会从同一权威整图独立裁剪. 
    """

    def __init__(self, max_bytes: int) -> None:
        """
        初始化空的按字节受限 LRU. 

        输入参数:
            - max_bytes: int, 缓存总字节上限; 负值规范化为 0

        输出:
            - None: 原地初始化空缓存状态
        """
        self.max_bytes = max(0, int(max_bytes))
        self.current_bytes = 0
        self.values: OrderedDict[str, tuple[dict[str, np.ndarray], int]] = OrderedDict()

    @staticmethod
    def _size_bytes(value: Mapping[str, np.ndarray]) -> int:
        """
        统计一个缓存 value 中 NumPy 数组的真实字节数. 

        输入参数:
            - value: Mapping[str,np.ndarray], 同一资产的字段映射

        输出:
            - size_bytes: int, 所有 NumPy 数组 `nbytes` 之和
        """
        return sum(int(array.nbytes) for array in value.values() if isinstance(array, np.ndarray))

    def get(self, key: str) -> dict[str, np.ndarray] | None:
        """
        读取一个缓存资产并把它移动到最近使用端. 

        输入参数:
            - key: str, 由 PDB identity 与资产类型组成的缓存键

        输出:
            - value: dict[str,np.ndarray] | None, 命中时返回共享 CPU 数组字典, 未命中时返回 None
        """
        item = self.values.pop(key, None)
        if item is None:
            return None
        self.values[key] = item
        return item[0]

    def put(self, key: str, value: dict[str, np.ndarray]) -> None:
        """
        写入一个缓存资产, 并按 LRU 顺序淘汰直到满足字节上限. 

        输入参数:
            - key: str, 由 PDB identity 与资产类型组成的缓存键
            - value: dict[str,np.ndarray], 要缓存的共享 CPU 数组字典

        输出:
            - None: value 超限或缓存禁用时不写入, 否则原地更新 LRU 状态
        """
        if self.max_bytes == 0:
            return
        size = self._size_bytes(value)
        if size > self.max_bytes:
            return
        old = self.values.pop(key, None)
        if old is not None:
            self.current_bytes -= old[1]
        while self.values and self.current_bytes + size > self.max_bytes:
            _, (_, removed_size) = self.values.popitem(last=False)
            self.current_bytes -= removed_size
        self.values[key] = (value, size)
        self.current_bytes += size


def _grid_from_npz(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    读取 Stage E `grid/voxel_size/origin` 并返回独立 CPU 数组. 

    输入参数:
        - path: Path; ``density/<pdb_id>/exp.npz`` 或 ``sim.npz`` 路径. 

    输出字段:
        - grid_zyx: float32 ``(D, H, W)``; 完整图密度数组, 轴序为 ZYX, 输入文件的单通道 ``(1, D, H, W)`` 已去除通道维. 
        - voxel_size_xyz: float32 ``(3,)``; 世界 XYZ 每体素尺寸, 单位 Å. 
        - origin_xyz: float32 ``(3,)``; 完整图 voxel-grid corner 的世界 XYZ 坐标, 单位 Å. 
    """
    with np.load(path, allow_pickle=False) as data:
        missing = sorted({"grid", "voxel_size", "origin"}.difference(data.files))
        if missing:
            raise KeyError(f"{path} 缺少密度字段: {missing}。")
        grid = np.asarray(data["grid"], dtype=np.float32)
        voxel_size = np.asarray(data["voxel_size"], dtype=np.float32)
        origin = np.asarray(data["origin"], dtype=np.float32)
    if grid.ndim != 4 or grid.shape[0] != 1:
        raise ValueError(f"{path}: grid 必须为 (1,Z,Y,X)，实际 {grid.shape}。")
    if voxel_size.shape != (3,) or origin.shape != (3,):
        raise ValueError(f"{path}: voxel_size 与 origin 必须为 (3,) XYZ。")
    if not np.isfinite(grid).all() or not np.isfinite(voxel_size).all() or not np.isfinite(origin).all():
        raise ValueError(f"{path}: 密度或几何字段包含 NaN/Inf。")
    if np.any(voxel_size <= 0):
        raise ValueError(f"{path}: voxel_size 必须逐轴为正。")
    return grid[0], voxel_size, origin


def _crop_80(array: np.ndarray, start_zyx: Sequence[int]) -> np.ndarray:
    """
    从已验证边界的完整 voxel grid 裁出真实 80³ 数组. 

    输入参数:
        - array: np.ndarray ``(D_full, H_full, W_full)``; 完整图 ZYX voxel grid. 
        - start_zyx: ``Sequence[int]`` ``(3,)``; 完整图离散 ZYX voxel-index BOX corner 起点. 

    输出字段:
        - crop: np.ndarray ``(80, 80, 80)``; 与输入 dtype 相同的连续内存裁剪. 
    """
    z0, y0, x0 = (int(value) for value in start_zyx)
    crop = array[z0 : z0 + 80, y0 : y0 + 80, x0 : x0 + 80]
    if crop.shape != STAGE1_BOX_SHAPE_ZYX:
        raise RuntimeError(f"Stage1 crop 必须恰为 80³，实际 {crop.shape}。")
    return np.ascontiguousarray(crop)


def _mainchain_class_targets(
    atom_coord_local_voxel: np.ndarray,
    atom_is_in_core_box: np.ndarray,
    res_type: np.ndarray,
    atom_name: np.ndarray,
    box_shape_zyx: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """把核心 BOX 内的受体原子散射成蛋白与核酸主链类别图. 
    输入数组的第 0 维逐受体原子对齐. ``res_type`` 的 0..19 表示蛋白残基, 20..27 表示核酸残基, 28 表示未知残基. 
    每个原子用 BOX 内连续 XYZ 体素坐标的向下取整结果确定 home voxel, 再换成 ZYX 数组索引. 未命中指定主链原子名、位于 8 Å 缓冲区或残基类型未知的原子不会写入类别图; 相应体素保持背景类别 0. 

    输入字段:
        - atom_coord_local_voxel: float32 ``(N_A, 3)``; 逐受体原子对齐的 BOX-local 连续 voxel XYZ 坐标. 
        - atom_is_in_core_box: bool ``(N_A,)``; 逐受体原子是否位于核心 80³ BOX. 
        - res_type: uint8 ``(N_A,)``; 0..19 为蛋白残基, 20..27 为核酸残基, 28 为未知残基. 
        - atom_name: 字符串数组 ``(N_A,)``; 逐受体原子对齐的原子名. 
        - box_shape_zyx: int 数组 ``(3,)``; 输出体素图的 ZYX 形状. 

    输出字段:
        - protein_target: uint8 ``(D, H, W)``; 0 为背景, 非零编号由蛋白主链原子名映射到 N、CA、C、O 类别. 
        - nucleic_target: uint8 ``(D, H, W)``; 0 为背景, 非零编号由核酸主链原子名映射到 P、O5'、C5'、C4'、C3'、O3' 类别. 
    """
    depth, height, width = (int(value) for value in box_shape_zyx)
    protein_target = np.zeros((depth, height, width), dtype=np.uint8)
    nucleic_target = np.zeros((depth, height, width), dtype=np.uint8)
    if atom_coord_local_voxel.shape[0] == 0:
        return protein_target, nucleic_target

    core_mask = np.asarray(atom_is_in_core_box, dtype=bool)
    voxel_xyz = np.floor(np.asarray(atom_coord_local_voxel, dtype=np.float32)).astype(np.int64)
    names = [
        value.decode("ascii").strip() if isinstance(value, bytes) else str(value).strip()
        for value in np.asarray(atom_name).reshape(-1).tolist()
    ]
    residue_types = np.asarray(res_type, dtype=np.uint8).reshape(-1)
    if not (len(names) == residue_types.size == voxel_xyz.shape[0] == core_mask.size):
        raise ValueError("主链类别字段必须与局部受体原子表逐项对应。")

    box_shape_xyz = np.asarray([width, height, depth], dtype=np.int64)
    valid = core_mask & np.all(voxel_xyz >= 0, axis=1) & np.all(voxel_xyz < box_shape_xyz, axis=1)
    for index in np.flatnonzero(valid).tolist():
        residue_type = int(residue_types[index])
        class_id = 0
        target = None
        if 0 <= residue_type <= 19:
            class_id = PROTEIN_MAINCHAIN_CLASS_BY_ATOM_NAME.get(names[index], 0)
            target = protein_target
        elif 20 <= residue_type <= 27:
            class_id = NUCLEIC_MAINCHAIN_CLASS_BY_ATOM_NAME.get(names[index], 0)
            target = nucleic_target
        if class_id == 0 or target is None:
            continue
        x, y, z = (int(value) for value in voxel_xyz[index])
        target[z, y, x] = class_id
    return protein_target, nucleic_target


def _rotate_zyx_coordinates(
    coord_zyx: np.ndarray,
    box_shape_zyx: np.ndarray,
    axis1: int,
    axis2: int,
    k: int,
) -> np.ndarray:
    """
    按 `np.rot90` 语义旋转 BOX-local 连续 ZYX voxel 坐标. 

    输入参数:
        - coord_zyx: np.ndarray, (N,3), BOX-local 连续 ZYX voxel 坐标, 采用 voxel-grid corner 语义
        - box_shape_zyx: np.ndarray, (3,), BOX 的离散 ZYX voxel shape
        - axis1: int, `np.rot90` 的第一个 ZYX 空间轴
        - axis2: int, `np.rot90` 的第二个 ZYX 空间轴
        - k: int, 逆时针 90 度旋转次数; 按 `k % 4` 生效

    输出:
        - rotated_coord_zyx: np.ndarray, (N,3), float32, 旋转后的 BOX-local 连续 ZYX voxel 坐标
    """
    rotated = np.asarray(coord_zyx, dtype=np.float32).copy()
    working_shape = np.asarray(box_shape_zyx, dtype=np.float32).copy()
    for _ in range(k % 4):
        old_coord = rotated.copy()
        old_shape = working_shape.copy()
        rotated[:, axis1] = old_shape[axis2] - old_coord[:, axis2]
        rotated[:, axis2] = old_coord[:, axis1]
        working_shape[axis1] = old_shape[axis2]
        working_shape[axis2] = old_shape[axis1]
    return rotated


def _apply_synced_rotation(sample: dict[str, Any]) -> dict[str, Any]:
    """
    对 density、监督与 Find 原子坐标执行同一次随机 90 度旋转. 

    输入参数:
        - sample: ``dict[str, Any]``; 单 BOX NumPy 样本, ``density_input`` 的空间轴为 ZYX, 原子坐标最后一维为 XYZ. 

    状态变化:
        - density_input: 对空间轴执行同一 90 度旋转, 通道维不变. 
        - ``hardmask``、``voxel_label``、``ligand_area_target``、``protein_mainchain_target``、``nucleic_mainchain_target``、``ligand_inverse_distance_target``: 对 ZYX 空间轴执行同一旋转. 
        - ``atom_coord_local_voxel``、``atom_coord_centered_world``、``atom_coord_world``: 按相同几何变换更新 XYZ 坐标. 
        - ``voxel_size_world``、``box_origin_world``: 更新旋转后的世界 XYZ 几何字段. 

    输出字段:
        - dict[str, Any]: 与输入相同的字典; 当随机次数为 0 时保持原字典不变. 
    """
    axis1, axis2 = np.random.choice([0, 1, 2], size=2, replace=False).tolist()
    k = random.randint(0, 3)
    if k == 0:
        return sample
    box_shape = sample["box_shape_zyx"]
    voxel_size = sample["voxel_size_world"]
    if not (int(box_shape[0]) == int(box_shape[1]) == int(box_shape[2])):
        raise ValueError("Stage1 90° 旋转只支持立方体 BOX。")
    voxel_size_zyx = np.asarray(voxel_size, dtype=np.float32)[[2, 1, 0]].copy()
    if k % 2 == 1:
        voxel_size_zyx[[axis1, axis2]] = voxel_size_zyx[[axis2, axis1]]
    rotated_voxel_size = voxel_size_zyx[[2, 1, 0]].astype(np.float32, copy=False)
    sample["voxel_size_world"] = rotated_voxel_size

    sample["density_input"] = np.rot90(sample["density_input"], k=k, axes=(axis1 + 1, axis2 + 1)).copy()
    sample["hardmask"] = np.rot90(sample["hardmask"], k=k, axes=(axis1, axis2)).copy()
    for field_name in (
        "voxel_label",
        "ligand_area_target",
        "protein_mainchain_target",
        "nucleic_mainchain_target",
        "ligand_inverse_distance_target",
    ):
        if field_name in sample:
            sample[field_name] = np.rot90(sample[field_name], k=k, axes=(axis1, axis2)).copy()

    if "atom_coord_local_voxel" not in sample:
        return sample
    local_zyx = sample["atom_coord_local_voxel"][:, [2, 1, 0]]
    sample["atom_coord_local_voxel"] = _rotate_zyx_coordinates(
        local_zyx, box_shape, int(axis1), int(axis2), k
    )[:, [2, 1, 0]].astype(np.float32, copy=False)
    centered_zyx = sample["atom_coord_centered_world"][:, [2, 1, 0]].copy()
    for _ in range(k):
        old = centered_zyx.copy()
        centered_zyx[:, axis1] = -old[:, axis2]
        centered_zyx[:, axis2] = old[:, axis1]
    sample["atom_coord_centered_world"] = centered_zyx[:, [2, 1, 0]].astype(np.float32, copy=False)
    box_shape_xyz = box_shape[[2, 1, 0]].astype(np.float32)
    box_center_world = (sample["box_origin_world"] + 0.5 * box_shape_xyz * rotated_voxel_size)
    sample["atom_coord_world"] = (sample["atom_coord_centered_world"] + box_center_world[None, :]).astype(np.float32, copy=False)
    return sample


def _to_tensor_sample(sample: dict[str, Any]) -> dict[str, Any]:
    """
    按 BOX-level 契约把 NumPy 数组转换为显式 dtype 的 torch tensor. 

    输入参数:
        - sample: dict[str,Any], 单 BOX NumPy 样本; 身份字段为 Python 标量, 数值字段为 NumPy 数组

    输出字段:
        - tensor_sample: dict[str, Any]; 保留输入键集合, 身份字段保持 Python 值, 数值字段转换为连续 CPU tensor. 
        - box_start_zyx: int32 索引 tensor. 
        - box_shape_zyx: int64 形状 tensor. 
        - atom_global_indices: int64 逐原子编号 tensor. 
        - box_origin_world: float32 世界 XYZ 原点 tensor. 
        - voxel_size_world: float32 世界 XYZ 体素尺寸 tensor. 
        - density_input: float32 密度通道 tensor. 
        - ligand_inverse_distance_target: float32 距离回归目标 tensor. 
        - hardmask: bool 受体占据掩码 tensor. 
        - ligand_area_target: bool 配体区域目标 tensor. 
        - voxel_label: bool binding 体素目标 tensor. 
        - atom_is_in_core_box: bool 逐原子核心 BOX 掩码 tensor. 
        - atom_label: bool 逐原子 binding 标签 tensor. 
        - protein_mainchain_target: int64 蛋白主链类别编号 tensor. 
        - nucleic_mainchain_target: int64 核酸主链类别编号 tensor. 
        - atom_feat: float32 逐原子特征 tensor. 
        - atom_coord_world: float32 逐原子世界 XYZ 坐标 tensor. 
        - atom_coord_local_voxel: float32 逐原子 BOX-local voxel XYZ 坐标 tensor. 
        - atom_coord_centered_world: float32 逐原子相对 BOX 中心的世界 XYZ 坐标 tensor. 
    """
    dtype_by_field = {
        "box_start_zyx": torch.int32,
        "box_shape_zyx": torch.int64,
        "box_origin_world": torch.float32,
        "voxel_size_world": torch.float32,
        "density_input": torch.float32,
        "hardmask": torch.bool,
        "ligand_area_target": torch.bool,
        "voxel_label": torch.bool,
        "protein_mainchain_target": torch.int64,
        "nucleic_mainchain_target": torch.int64,
        "ligand_inverse_distance_target": torch.float32,
        "atom_global_indices": torch.int64,
        "atom_feat": torch.float32,
        "atom_coord_world": torch.float32,
        "atom_coord_local_voxel": torch.float32,
        "atom_coord_centered_world": torch.float32,
        "atom_is_in_core_box": torch.bool,
        "atom_label": torch.bool,
    }
    result = dict(sample)
    for field_name, dtype in dtype_by_field.items():
        if field_name in result:
            array = np.ascontiguousarray(result[field_name])
            result[field_name] = torch.as_tensor(array, dtype=dtype)
    return result


class Stage1Dataset(Dataset):
    """
    统一物化 train、val、full_map 和 centered 四类 80³ 请求. 

    构造参数:
        - all_data_path: str; A-G 正式根目录, 包含 ``density``、``parse`` 和 ``labels``. 
        - split_file: ``str | Path | Sequence[ResolvedStage1Crop]``; 训练 BOX pool 目录、固定验证请求文件或推理层传入的内存请求序列. 
        - mode: str; 取值为 ``train``、``val``、``validation``、``full_map`` 或 ``centered``. 
        - stage1_model_name: str; ``STAGE1_MODEL_NAMES`` 中的 producer 身份, 决定密度通道和是否返回 Find 原子字段. 
        - box_pool_root: ``str | None``; 包含 ``manifest.json`` 与 validation selection 的 ``stage1_preparation/box_pool`` 根目录. 
        - density_channel_config: ``Mapping[str, Any]``; 密度裁剪、拟合和启用通道的配置. 
        - atom_buffer_radius: float; 核心 BOX 外选择受体原子的世界坐标缓冲半径, 当前固定为 8.0 Å. 
        - request_seed: int; 训练请求层的基准 seed. 
        - occurrence_cap_per_pdb: int; 每个 PDB 的一级 occurrence 候选上限。
        - occurrence_ratio: float; 每个 PDB 从一级候选中保留的 occurrence 比例。
        - entry_ratio: Mapping[str, int] | None; 每个 occurrence 展开的 center、bias 和 context 数量。
        - cache_max_bytes: int; 每个 DataLoader worker 的受体表、标签和完整图 LRU 缓存字节上限. 
        - enable_random_rotation: bool; 训练模式是否对密度、标签和原子坐标同步执行随机 90 度旋转. 
        - name: str; Dataset 的显示名称. 
        - split_train: ``str | Sequence[str] | None``; 兼容参数, 当前实现不读取. 
        - split_val: ``str | Sequence[str] | None``; 兼容参数, 当前实现不读取. 
        - class_names: ``Sequence[str]``; 固定为 ``("background", "foreground")``, 长度必须为 2. 

    单样本输出字段由模块 Docstring 的同名字段清单定义; Find 与 ``unet_c1`` 共用体素物化路径, ``unet_c1`` 仍构造辅助监督但不返回逐原子输入表. 

    边界:
        - 请求起点必须已经由 ``resolve_stage1_start`` 合法化, Dataset 不补零. 
        - Dataset 不在 ``__getitem__`` 中删除失败样本或修改请求源. 
    """
    collate_fn = Stage1BatchCollator()

    def __init__(
        self,
        all_data_path: str,
        split_file: str | Path | Sequence[ResolvedStage1Crop],
        mode: str,
        stage1_model_name: str,
        box_pool_root: str | None,
        density_channel_config: Mapping[str, Any],
        atom_buffer_radius: float = 8.0,
        request_seed: int = 3407,
        occurrence_cap_per_pdb: int = 50,
        occurrence_ratio: float = 1.0,
        entry_ratio: Mapping[str, int] | None = None,
        cache_max_bytes: int = 536_870_912,
        enable_random_rotation: bool = True,
        name: str = "stage1",
        split_train: str | Sequence[str] | None = None,
        split_val: str | Sequence[str] | None = None,
        class_names: Sequence[str] = ("background", "foreground"),
    ) -> None:
        """解析请求源、密度通道契约和每个 DataLoader worker 的完整图缓存. 
        构造参数:
            - all_data_path: str; A-G 正式根目录, 包含 ``density``、``parse`` 和 ``labels``. 
            - split_file: ``str | Path | Sequence[ResolvedStage1Crop]``; 训练 BOX pool 目录、固定验证请求文件或推理层传入的内存请求序列. 
            - mode: str; 取值为 ``train``、``val``、``validation``、``full_map`` 或 ``centered``. 
            - stage1_model_name: str; Find_0等, ``STAGE1_MODEL_NAMES`` 中的 producer 身份, 决定密度通道和是否返回 Find 原子字段. 
            - box_pool_root: ``str | None``; 包含 ``manifest.json`` 与 validation selection 的 ``stage1_preparation/box_pool`` 根目录. 
            - density_channel_config: ``Mapping[str, Any]``; 密度裁剪、拟合和启用通道的配置. 
            - atom_buffer_radius: float; 核心 BOX 外选择受体原子的世界坐标缓冲半径, 当前固定为 8.0 Å. 
            - request_seed: int; 训练请求层的基准 seed. 
            - occurrence_cap_per_pdb: int; 每个 PDB 的一级 occurrence 候选上限。
            - occurrence_ratio: float; 每个 PDB 从一级候选中保留的 occurrence 比例。
            - entry_ratio: Mapping[str, int] | None; 每个 occurrence 展开的三类 BOX 数量。
            - cache_max_bytes: int; 每个 DataLoader worker 的受体表、标签和完整图 LRU 缓存字节上限. 
            - enable_random_rotation: bool; 训练模式是否对密度、标签和原子坐标同步执行随机 90 度旋转. 
            - name: str; Dataset 的显示名称. 
            - split_train: ``str | Sequence[str] | None``; 兼容参数, 当前实现不读取. 
            - split_val: ``str | Sequence[str] | None``; 兼容参数, 当前实现不读取. 
            - class_names: ``Sequence[str]``; 固定为 ``("background", "foreground")``, 长度必须为 2. 

        单样本输出字段由模块 Docstring 的同名字段清单定义; Find 与 ``unet_c1`` 共用体素物化路径, ``unet_c1`` 仍构造辅助监督但不返回逐原子输入表. 

        ``request_seed``、``occurrence_cap_per_pdb``、``occurrence_ratio`` 和 ``entry_ratio`` 只交给请求层决定读取哪些训练 BOX。
        """
        super().__init__()
        del split_train, split_val
        self.name = str(name)
        self.root = Path(all_data_path)
        self.mode = str(mode).lower()
        self.stage1_model_name = str(stage1_model_name)
        self.class_names = tuple(str(value) for value in class_names)
        if self.mode not in {"train", "val", "validation", "full_map", "centered"}:
            raise ValueError(f"未知 Stage1 Dataset mode: {mode!r}。")
        if self.stage1_model_name not in STAGE1_MODEL_NAMES:
            raise ValueError(f"stage1_model_name 必须属于 {STAGE1_MODEL_NAMES}。")
        if self.class_names != ("background", "foreground"):
            raise ValueError("AdaLigand Stage1 当前只接受 [background, foreground] 二分类顺序。")
        if float(atom_buffer_radius) != 8.0:
            raise ValueError("AdaLigand Find Dataset 的 atom_buffer_radius 固定为 8.0 Å。")
        self.atom_buffer_radius = 8.0
        self.enable_random_rotation = bool(enable_random_rotation and self.mode == "train")
        if isinstance(split_file, (list, tuple)) and all(isinstance(request, ResolvedStage1Crop) for request in split_file):
            self.request_source = tuple(split_file)
            if not self.request_source:
                raise ValueError("内存 Stage1 请求序列不能为空。")
        else:
            self.request_source = build_request_source(
                split_file=split_file,
                mode=self.mode,
                box_pool_root=box_pool_root,
                seed=int(request_seed),
                occurrence_cap_per_pdb=int(occurrence_cap_per_pdb),
                occurrence_ratio=float(occurrence_ratio),
                entry_ratio=entry_ratio,
            )
        # dict[str,Any], 从 Hydra dataset 配置解析出的密度通道构造契约. 
        channel_cfg = dict(density_channel_config)
        self.density_config = DensityChannelConfig(
            clip_percentile=tuple(float(value) for value in channel_cfg["clip_percentile"]),
            fit_mask_percentile=float(channel_cfg["fit_mask_percentile"]),
            enabled_channels=[str(value) for value in channel_cfg["enabled_channels"]],
        )
        # list[str], 长度 C_density, 展开 `all` 后的模型输入通道顺序. 
        resolved_channels = (
            list(ALL_CHANNEL_NAMES)
            if "all" in [value.lower() for value in self.density_config.enabled_channels]
            else list(self.density_config.enabled_channels)
        )
        if self.stage1_model_name in FIND_MODEL_NAMES and resolved_channels != list(ALL_CHANNEL_NAMES):
            raise ValueError("Find producer 必须按权威顺序启用完整 56D ALL density channels。")
        if self.stage1_model_name == "unet_c1" and resolved_channels != ["exp_clipnorm_nopost"]:
            raise ValueError("unet_c1 density 输入必须恰为 exp_clipnorm_nopost。")
        self.resolved_density_channels = tuple(resolved_channels)
        # 同一个按字节受限的 worker-local cache 同时保存轻量 receptor 表与最近使用的原始整图. 
        # 完整图推理会连续消费同一 PDB 的数百个窗口, 因此必须避免每个窗口重新解压 exp/sim; 训练的随机 PDB 访问仍由同一上限自然淘汰大数组. 
        self._source_cache = _ByteLruCache(cache_max_bytes)

    def set_epoch(self, epoch: int) -> None:
        """
        通知动态训练请求源切换 epoch; 固定请求源保持不变. 

        输入参数:
            - epoch: int; 当前训练 epoch 编号. 

        状态变化:
            - 动态 ``Stage1TrainingRequestSet`` 重建目标 epoch 的请求. 
            - 固定请求列表保持不变. 
        """
        set_epoch = getattr(self.request_source, "set_epoch", None)
        if callable(set_epoch):
            set_epoch(int(epoch))

    def __len__(self) -> int:
        """
        返回当前请求源的样本数. 

        输出字段:
            - int: 当前 epoch 或冻结请求表的请求数量. 
        """
        return len(self.request_source)

    def describe_index(self, index: int) -> str:
        """
        生成人类可读的请求身份摘要. 

        输入参数:
            - index: int; 请求源中的 0-based 请求位置. 

        输出字段:
            - str: 包含 PDB、role、完整图离散 ZYX BOX corner 起点和 occurrence 身份的可读摘要. 
        """
        request = self.request_source[index]
        return (
            f"pdb_id={request.pdb_id}, role={request.role}, "
            f"start_zyx={request.box_start_zyx}, occurrence_id={request.occurrence_id}"
        )

    def _load_structure(self, pdb_id: str, require_targets: bool) -> dict[str, np.ndarray]:
        """
        读取并缓存 receptor 49D 基础表及可选 binding label. 

        输入参数:
            - pdb_id: str; 当前 PDB 身份. 
            - require_targets: bool; 是否同时读取逐原子 binding 监督. 

        输出字段:
            - coords: float32 ``(N_receptor, 3)``; 受体原子的世界 XYZ 坐标, 单位 Å. 
            - feat: float32 ``(N_receptor, 49)``; 与 coords 第 0 维逐原子对齐的基础原子特征. 
            - binding_atom: bool ``(N_receptor,)``; 与 coords 第 0 维逐原子对齐的结合区域标签, 仅 ``require_targets=True`` 时读取. 
            - res_type: uint8 ``(N_receptor,)``; 辅助监督使用的残基类别编号, 仅辅助监督模型且 ``require_targets=True`` 时读取. 
            - atom_name: 字符串数组 ``(N_receptor,)``; 辅助监督使用的原子名, 仅辅助监督模型且 ``require_targets=True`` 时读取. 

        文件读取:
            - parse/<pdb_id>/receptor_tokens.npz: 提供 coords、feat、res_type 和 atom_name. 
            - labels/<pdb_id>/atom_labels.npz: 提供 binding_atom. 
        """
        cache_key = f"{pdb_id}|targets={int(require_targets)}"
        cached = self._source_cache.get(cache_key)
        if cached is not None:
            return cached
        receptor_path = self.root / "parse" / pdb_id / "receptor_tokens.npz"
        with np.load(receptor_path, allow_pickle=False) as data:
            # np.ndarray[float32], (N_receptor,3), receptor 原子的世界 XYZ 坐标. 
            coords = np.asarray(data["coords"], dtype=np.float32)
            # float32, (N_receptor, 49), 与 coords 第 0 维逐受体原子对齐的基础特征. 
            feat = np.asarray(data["feat"], dtype=np.float32)
        if coords.ndim != 2 or coords.shape[1] != 3 or feat.shape != (coords.shape[0], 49):
            raise ValueError(f"{receptor_path}: coords/feat 必须为 [N,3]/[N,49]。")
        if not np.isfinite(coords).all() or not np.isfinite(feat).all():
            raise ValueError(f"{receptor_path}: coords/feat 包含 NaN/Inf。")
        structure = {"coords": coords, "feat": feat}
        if require_targets:
            label_path = self.root / "labels" / pdb_id / "atom_labels.npz"
            with np.load(label_path, allow_pickle=False) as data:
                binding_atom = np.asarray(data["binding_atom"], dtype=bool)
            if binding_atom.shape != (coords.shape[0],):
                raise ValueError(f"{label_path}: binding_atom 必须与 receptor 行数一致。")
            structure["binding_atom"] = binding_atom
            if self.stage1_model_name in _AUXILIARY_SUPERVISION_MODEL_NAMES:
                with np.load(receptor_path, allow_pickle=False) as data:
                    res_type = np.asarray(data["res_type"], dtype=np.uint8)
                    atom_name = np.asarray(data["atom_name"])
                if res_type.shape != (coords.shape[0],) or atom_name.shape != (coords.shape[0],):
                    raise ValueError(f"{receptor_path}: res_type/atom_name 必须与 receptor 原子数一致。")
                structure["res_type"] = res_type
                structure["atom_name"] = atom_name
        self._source_cache.put(cache_key, structure)
        return structure

    def _load_density_grid(
        self,
        pdb_id: str,
        grid_name: str,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        读取并按真实字节数缓存一个 PDB 的 exp/sim 原始完整图. 

        输入参数:
            - pdb_id: str; 当前 PDB 身份. 
            - grid_name: str; ``exp`` 或 ``sim``; 缓存只保存原始 float32 grid 与几何, 不保存派生密度通道. 

        输出字段:
            - grid_zyx: float32 ``(D_full, H_full, W_full)``; 完整图 ZYX voxel grid. 
            - voxel_size_xyz: float32 ``(3,)``; 世界 XYZ 每体素尺寸, 单位 Å. 
            - origin_xyz: float32 ``(3,)``; 完整图 voxel-grid corner 的世界 XYZ 坐标, 单位 Å. 

        调用方只能裁剪读取, 不得原地修改这些 worker-local 共享数组. 
        """
        if grid_name not in {"exp", "sim"}:
            raise ValueError(f"未知 density grid_name={grid_name!r}。")
        cache_key = f"{pdb_id}|density={grid_name}"
        cached = self._source_cache.get(cache_key)
        if cached is None:
            grid, voxel_size, origin = _grid_from_npz(
                self.root / "density" / pdb_id / f"{grid_name}.npz"
            )
            cached = {
                "grid": grid,
                "voxel_size": voxel_size,
                "origin": origin,
            }
            self._source_cache.put(cache_key, cached)
        return cached["grid"], cached["voxel_size"], cached["origin"]

    def _load_ligand_union(
        self,
        pdb_id: str,
        expected_shape_zyx: Sequence[int],
    ) -> np.ndarray:
        """
        读取并缓存 schema-v3 occurrence union mask, 保持完整图 bool 语义. 

        输入参数:
            - pdb_id: str; 当前 PDB 身份. 
            - expected_shape_zyx: ``Sequence[int]`` ``(3,)``; exp 完整图的 ZYX voxel-grid 形状. 

        输出字段:
            - union_mask: bool ``(1, D_full, H_full, W_full)``; 完整图 ZYX voxel grid 上所有 occurrence 的配体区域并集. 
        """
        expected_shape = tuple(int(value) for value in expected_shape_zyx)
        cache_key = f"{pdb_id}|ligand_union"
        cached = self._source_cache.get(cache_key)
        if cached is None:
            ligand_path = self.root / "density" / pdb_id / "ligand_area.npz"
            with np.load(ligand_path, allow_pickle=False) as data:
                union_mask = np.asarray(data["union_mask"], dtype=bool)
            if union_mask.shape != (1, *expected_shape):
                raise ValueError(f"{ligand_path}: union_mask 与 exp grid 形状不一致。")
            cached = {"union_mask": union_mask}
            self._source_cache.put(cache_key, cached)
        union_mask = cached["union_mask"]
        if union_mask.shape != (1, *expected_shape):
            raise ValueError(f"{pdb_id}: 缓存的 union_mask 与当前 exp grid 形状不一致。")
        return union_mask

    def _load_ligand_distance(
        self,
        pdb_id: str,
        expected_shape_zyx: Sequence[int],
        expected_voxel_size_xyz: np.ndarray,
        expected_origin_xyz: np.ndarray,
    ) -> np.ndarray:
        """核对空间契约后读取与实验密度图逐体素对齐的最近配体原子距离图. 

        文件字段:
            - distance: float16 ``(1, D, H, W)``; 每个完整图体素到最近配体原子的距离, 单位 Å; 有配体时为有限非负值, 无配体时全部为正无穷. 
            - schema_version: uint16 标量; 当前必须为 ``1``. 
            - grid_shape_zyx: int64 ``(3,)``; 完整图 ZYX 形状, 必须与 ``exp.npz:grid`` 一致. 
            - voxel_size_xyz: float32 ``(3,)``; 世界 XYZ 每体素尺寸, 必须与 ``exp.npz:voxel_size`` 完全一致. 
            - origin_xyz: float32 ``(3,)``; 完整图 voxel-grid corner 的世界 XYZ 坐标, 必须与 ``exp.npz:origin`` 完全一致. 
            - distance_unit: 字符串标量 ``"angstrom"``; 距离单位. 

        输出:
            - np.ndarray: 缓存中的完整图距离数组 ``(1, D, H, W)``; 调用方只裁剪, 不改写缓存数组. 
        """
        expected_shape = tuple(int(value) for value in expected_shape_zyx)
        expected_voxel_size = np.asarray(expected_voxel_size_xyz, dtype=np.float32)
        expected_origin = np.asarray(expected_origin_xyz, dtype=np.float32)
        cache_key = f"{pdb_id}|ligand_distance"
        cached = self._source_cache.get(cache_key)
        if cached is None:
            distance_path = self.root / "density" / pdb_id / "ligand_dist.npz"
            with np.load(distance_path, allow_pickle=False) as data:
                required = {
                    "distance",
                    "schema_version",
                    "grid_shape_zyx",
                    "voxel_size_xyz",
                    "origin_xyz",
                    "distance_unit",
                }
                missing = sorted(required.difference(data.files))
                if missing:
                    raise KeyError(f"{distance_path} 缺少字段 {missing}。")
                distance = data["distance"].copy()
                schema_version = np.asarray(data["schema_version"])
                grid_shape = np.asarray(data["grid_shape_zyx"])
                voxel_size = np.asarray(data["voxel_size_xyz"])
                origin = np.asarray(data["origin_xyz"])
                distance_unit = np.asarray(data["distance_unit"])
            if distance.dtype != np.float16:
                raise ValueError(f"{distance_path}: distance 必须为 float16。")
            if distance.shape != (1, *expected_shape):
                raise ValueError(f"{distance_path}: distance 与 exp grid 形状不一致。")
            finite = np.isfinite(distance)
            positive_infinity = np.isposinf(distance)
            if (
                np.isnan(distance).any()
                or np.isneginf(distance).any()
                or np.any(distance[finite] < 0)
                or (np.any(finite) and np.any(positive_infinity))
            ):
                raise ValueError(f"{distance_path}: distance 必须全部有限且非负，或全部为正无穷。")
            if (
                schema_version.dtype != np.uint16
                or schema_version.shape != ()
                or int(schema_version) != 1
                or grid_shape.dtype != np.int64
                or not np.array_equal(grid_shape, np.asarray(expected_shape, dtype=np.int64))
                or voxel_size.dtype != np.float32
                or not np.array_equal(voxel_size, expected_voxel_size)
                or origin.dtype != np.float32
                or not np.array_equal(origin, expected_origin)
                or distance_unit.shape != ()
                or str(distance_unit.item()) != "angstrom"
            ):
                raise ValueError(f"{distance_path}: 空间字段或 schema 与 exp grid 不一致。")
            cached = {"distance": distance}
            self._source_cache.put(cache_key, cached)
        return cached["distance"]

    def _materialize(self, request: ResolvedStage1Crop) -> dict[str, Any]:
        """
        把一个已解析请求物化为 producer 专属单样本字段. 

        输入参数:
            - request: ResolvedStage1Crop, PDB、完整图离散 ZYX voxel-index BOX corner 起点、role 与监督开关已经冻结的请求

        输出字段:
            - pdb_id: Python str; 当前 PDB 身份. 
            - request_role: Python str; 当前请求角色. 
            - occurrence_id: Python int 或 None; 当前请求引用的 occurrence 编号. 
            - candidate_index: Python int 或 None; 当前请求引用的 bias/context 候选下标. 
            - box_start_zyx: int32 ``(3,)`` tensor; 完整图离散 ZYX voxel-index BOX corner 起点. 
            - box_shape_zyx: int64 ``(3,)`` tensor; 固定为 ``(80, 80, 80)`` 的 ZYX 形状. 
            - box_origin_world: float32 ``(3,)`` tensor; BOX voxel-grid corner 的世界 XYZ 坐标, 单位 Å. 
            - voxel_size_world: float32 ``(3,)`` tensor; 世界 XYZ 每体素尺寸, 单位 Å. 
            - density_input: float32 ``(C_density, 80, 80, 80)`` tensor; ZYX voxel grid 上的 producer 密度通道. 
            - hardmask: bool ``(80, 80, 80)`` tensor; 核心 BOX 内受体原子占据体素. 
            - voxel_label: bool ``(80, 80, 80)`` tensor; 核心 BOX 内 binding 受体原子占据体素. 
            - ligand_area_target: bool ``(80, 80, 80)`` tensor; 完整配体区域并集的 BOX 裁剪. 
            - protein_mainchain_target: int64 ``(80, 80, 80)`` tensor; 蛋白背景/N/CA/C/O 类别编号. 
            - nucleic_mainchain_target: int64 ``(80, 80, 80)`` tensor; 核酸背景/P/O5'/C5'/C4'/C3'/O3' 类别编号. 
            - ligand_inverse_distance_target: float32 ``(80, 80, 80)`` tensor; 有限距离按 ``1/(1+distance_Å)`` 转换, 无配体的正无穷距离转换为 0. 
            - atom_global_indices: int64 ``(N_A,)`` tensor; 被选择受体原子在完整受体数组中的编号. 
            - atom_feat: float32 ``(N_A, 49)`` tensor; 与 atom_global_indices 第 0 维逐原子对齐的特征. 
            - atom_coord_world: float32 ``(N_A, 3)`` tensor; 被选择受体原子的世界 XYZ 坐标, 单位 Å. 
            - atom_coord_local_voxel: float32 ``(N_A, 3)`` tensor; 被选择受体原子的 BOX-local 连续 voxel XYZ 坐标. 
            - atom_coord_centered_world: float32 ``(N_A, 3)`` tensor; 相对 BOX 几何中心的世界 XYZ 坐标, 单位 Å. 
            - atom_is_in_core_box: bool ``(N_A,)`` tensor; 第 i 个值表示第 i 个被选择原子是否位于核心 80³ BOX 内. 
            - atom_label: bool ``(N_A,)`` tensor; 与 atom_global_indices 第 0 维逐原子对齐的 binding 标签. 
        """
        # exp_grid: np.ndarray[float32], (D_full,H_full,W_full), 完整实验密度图. 
        # voxel_size/full_origin: np.ndarray[float32], (3,), 世界 XYZ 几何. 
        exp_grid, voxel_size, full_origin = self._load_density_grid(request.pdb_id, "exp")
        full_shape = np.asarray(exp_grid.shape, dtype=np.int64)
        resolved_start = resolve_stage1_start(request.box_start_zyx, full_shape)
        if resolved_start != request.box_start_zyx:
            raise ValueError(
                "Stage1 Dataset 只消费预先解析的起点："
                f"request={request.box_start_zyx}, resolved={resolved_start}, pdb={request.pdb_id}。"
            )
        # np.ndarray[int32], (3,), 当前 BOX 的完整图离散 ZYX voxel-index corner 起点. 
        start_zyx = np.asarray(resolved_start, dtype=np.int32)
        # np.ndarray[int64], (3,), 固定 (80,80,80) 的 ZYX shape. 
        box_shape_zyx = np.asarray(STAGE1_BOX_SHAPE_ZYX, dtype=np.int64)
        # np.ndarray[float32], (3,), 完整图离散 voxel-index 起点由 ZYX 换轴为 XYZ, 供世界坐标原点计算. 
        start_xyz = start_zyx[[2, 1, 0]].astype(np.float32)
        # np.ndarray[float32], (3,), 当前 BOX 在世界 XYZ 中的 corner 原点. 
        box_origin = (full_origin + start_xyz * voxel_size).astype(np.float32, copy=False)

        structure = self._load_structure(request.pdb_id, request.require_targets)
        # dict[str, np.ndarray], 从完整受体数组选出核心 BOX 外加 8 Å 范围内的原子, 
        # 并标记其中真正位于核心 BOX 的原子. 
        selection = select_atoms_for_box(
            atom_coords_world=structure["coords"],
            box_origin_world=box_origin,
            voxel_size_world=voxel_size,
            box_shape_zyx=box_shape_zyx,
            buffer_radius=self.atom_buffer_radius,
        )
        selected_idx = selection["selected_idx"]
        # dict[str,np.ndarray], 三套 (N_A,3) XYZ 坐标: 绝对世界坐标、BOX-local 连续 voxel 坐标、BOX-center-relative 世界坐标. 
        atom_coordinates = build_atom_coordinates(
            atom_coords_world=structure["coords"],
            selected_idx=selected_idx,
            box_origin_world=box_origin,
            voxel_size_world=voxel_size,
            box_shape_zyx=box_shape_zyx,
        )
        core_mask = selection["atom_is_in_core_box"]
        # np.ndarray[bool], (80,80,80), 只由 core receptor 原子占据生成的硬掩码. 
        hardmask = build_hardmask_from_atom_coordinates(
            atom_coord_local_voxel=atom_coordinates["atom_coord_local_voxel"],
            atom_is_in_core_box=core_mask,
            box_shape_zyx=box_shape_zyx,
        ).astype(bool, copy=False)

        # np.ndarray[float32], (80,80,80), 当前起点的实验密度裁剪. 
        exp_crop = _crop_80(exp_grid, start_zyx)
        if self.stage1_model_name.startswith("Find"):
            sim_grid, sim_voxel_size, sim_origin = self._load_density_grid(request.pdb_id, "sim")
            if sim_grid.shape != exp_grid.shape or not np.array_equal(sim_voxel_size, voxel_size) or not np.array_equal(sim_origin, full_origin):
                raise ValueError(f"{request.pdb_id}: exp/sim 的 shape、voxel_size、origin 必须完全一致。")
            sim_crop = _crop_80(sim_grid, start_zyx)
        else:
            sim_crop = None
        # np.ndarray[float32], (C_density,80,80,80), producer 专属的最终体素输入. 
        density_input = build_density_channels(
            exp_raw=exp_crop,
            sim_raw=sim_crop,
            config=self.density_config,
            receptor_mask=hardmask,
        )

        # dict[str,Any], 单 BOX 模型输入；固定形状密度网格稍后转 tensor，身份字段保留 Python 值。
        sample: dict[str, Any] = {
            "pdb_id": request.pdb_id,
            "request_role": request.role,
            "occurrence_id": request.occurrence_id,
            "candidate_index": request.candidate_index,
            "box_start_zyx": start_zyx,
            "box_shape_zyx": box_shape_zyx,
            "box_origin_world": box_origin,
            "voxel_size_world": voxel_size.astype(np.float32, copy=False),
            "density_input": density_input,
            "hardmask": hardmask,
        }
        if request.require_targets:
            # np.ndarray[bool], (N_A,), core+buffer 局部原子表对应的 binding 标签. 
            binding_selected = structure["binding_atom"][selected_idx]
            sample["voxel_label"] = build_hardmask_from_atom_coordinates(
                atom_coord_local_voxel=atom_coordinates["atom_coord_local_voxel"],
                atom_is_in_core_box=core_mask & binding_selected,
                box_shape_zyx=box_shape_zyx,
            ).astype(bool, copy=False)
            union_mask = self._load_ligand_union(request.pdb_id, full_shape)
            sample["ligand_area_target"] = _crop_80(union_mask[0], start_zyx).astype(bool, copy=False)

            if self.stage1_model_name in _AUXILIARY_SUPERVISION_MODEL_NAMES:
                protein_target, nucleic_target = _mainchain_class_targets(
                    atom_coord_local_voxel=atom_coordinates["atom_coord_local_voxel"],
                    atom_is_in_core_box=core_mask,
                    res_type=structure["res_type"][selected_idx],
                    atom_name=structure["atom_name"][selected_idx],
                    box_shape_zyx=box_shape_zyx,
                )
                sample["protein_mainchain_target"] = protein_target
                sample["nucleic_mainchain_target"] = nucleic_target
                ligand_distance = _crop_80(
                    self._load_ligand_distance(
                        request.pdb_id,
                        full_shape,
                        voxel_size,
                        full_origin,
                    )[0],
                    start_zyx,
                ).astype(np.float32, copy=False)
                sample["ligand_inverse_distance_target"] = np.where(
                    np.isfinite(ligand_distance), 1.0 / (1.0 + ligand_distance), 0.0
                ).astype(np.float32, copy=False)

        if self.stage1_model_name.startswith("Find"):
            sample.update(atom_coordinates)
            sample["atom_global_indices"] = selected_idx.astype(np.int64, copy=False)
            sample["atom_feat"] = build_atom_features(structure["feat"], selected_idx)
            sample["atom_is_in_core_box"] = core_mask.astype(bool, copy=False)
            if request.require_targets:
                sample["atom_label"] = structure["binding_atom"][selected_idx].astype(bool, copy=False)
        if self.enable_random_rotation:
            sample = _apply_synced_rotation(sample)
        return _to_tensor_sample(sample)

    def __getitem__(self, index: int) -> dict[str, Any]:
        """
        按请求对象在请求源中的位置物化一个 Stage1 样本. 

        输入参数:
            - index: int; 请求对象在请求源中的位置编号, 从 0 开始. 

        输出字段:
            - dict[str, Any]: 与 ``materialize_request`` 相同的单 BOX tensor 字典. 
        """
        return self.materialize_request(self.request_source[index])

    def materialize_request(
        self,
        request: ResolvedStage1Crop,
    ) -> dict[str, Any]:
        """物化一个调用方已经解析完成的 80³ 请求: return self._materialize(request) .

        输入参数:
            - request: ``ResolvedStage1Crop``; PDB 身份、完整图离散 ZYX BOX 起点、监督开关和 role 已由共享 resolver 冻结. 

        输出字段:
            - dict[str, Any]: 与 ``dataset[index]`` 完全相同的单 BOX tensor 字典; ``require_targets`` 只决定是否附加监督字段, 不改变模型输入字段. 
        """
        if not isinstance(request, ResolvedStage1Crop):
            raise TypeError("request 必须是 ResolvedStage1Crop。")
        return self._materialize(request)

    def full_map_context(
        self,
        pdb_id: str,
    ) -> tuple[tuple[int, int, int], np.ndarray, np.ndarray, np.ndarray]:
        """返回完整图滑窗所需的 shape、几何与 receptor 世界坐标. 

        输入参数:
            - pdb_id: str; 当前小写或可规范化为小写的 PDB 身份. 

        输出字段:
            - shape_zyx: tuple[int, int, int] ``(3,)``; exp 完整图的 ZYX voxel-grid 形状. 
            - voxel_size_xyz: float32 ``(3,)``; 世界 XYZ 每体素尺寸, 单位 Å. 
            - origin_xyz: float32 ``(3,)``; 完整图 voxel-grid corner 的世界 XYZ 坐标, 单位 Å. 
            - receptor_coord_xyz: float32 ``(N_receptor, 3)``; 受体原子的世界 XYZ 坐标, 单位 Å. 

        数据来自与 `materialize_request` 相同的 worker-local 有界缓存, 因而装配层不会为读取 shape/hardmask 额外解压一次完整 exp 或 receptor. 
        """
        identity = str(pdb_id).strip().lower()
        exp_grid, voxel_size, origin = self._load_density_grid(identity, "exp")
        receptor = self._load_structure(identity, require_targets=False)
        return (
            tuple(int(value) for value in exp_grid.shape),
            voxel_size,
            origin,
            receptor["coords"],
        )
