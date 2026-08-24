# -*- coding: utf-8 -*-
"""把 Stage1 V3 请求和 A-G 正式资产物化为一个 80³ 单样本。

本模块的主入口是 :class:`Stage1Dataset`；它接收共享请求解析器产生的 :class:`ResolvedStage1Crop`，从正式资产读取完整图、受体原子和监督数组，裁出真实的 80³ BOX，再按 producer 契约生成 NumPy 字段并转为 CPU tensor。:func:`_to_tensor_sample` 只负责 dtype 和连续内存转换，批处理由 ``stage1_collate.py`` 完成。

单样本字段契约:
    - pdb_id: str；当前 PDB 的小写 identity。
    - request_role: str；``train_occurrence``、``validation``、``center``、``bias``、``context``、``sliding`` 或 ``centered``，与 ``ResolvedStage1Crop.role`` 一致。
    - occurrence_id: int | None；请求引用的 occurrence 编号；没有 occurrence 语义时为 ``None``。
    - candidate_index: int | None；请求引用的 bias/context 候选下标；没有候选下标时为 ``None``。
    - box_start_zyx: int32 ``(3,)``；完整图离散体素的 ZYX corner index，三个分量分别对应 depth、height、width。
    - box_shape_zyx: int64 ``(3,)``；固定为 ``(80, 80, 80)`` 的 ZYX BOX 形状。
    - box_origin_world: float32 ``(3,)``；BOX voxel-grid corner 的世界 XYZ 坐标，单位为 Å。
    - voxel_size_world: float32 ``(3,)``；世界 XYZ 轴的体素尺寸，单位为 Å。
    - density_input: float32 ``(C_density, 80, 80, 80)``；producer 所需的 ZYX 密度通道，通道顺序由 ``resolved_density_channels`` 决定。
    - hardmask: bool ``(80, 80, 80)``；核心 80³ BOX 内受体原子占据的 ZYX 体素掩码。
    - voxel_label: bool ``(80, 80, 80)``；核心 BOX 内 binding 受体原子占据的 ZYX 体素标签，仅在 ``require_targets`` 为真时存在。
    - ligand_area_target: bool ``(80, 80, 80)``；完整图中全部 occurrence 的并集 mask 的 ZYX BOX 裁剪，仅在 ``require_targets`` 为真时存在。
    - protein_mainchain_target: int64 ``(80, 80, 80)``；蛋白主链背景、N、CA、C、O 的类别编号，仅辅助监督 producer 存在。
    - nucleic_mainchain_target: int64 ``(80, 80, 80)``；核酸主链背景、P、O5'、C5'、C4'、C3'、O3' 的类别编号，仅辅助监督 producer 存在。
    - ligand_inverse_distance_target: float32 ``(80, 80, 80)``；最近配体原子距离（Å）经 ``1/(1+distance)`` 转换后的 ZYX 回归目标，仅辅助监督 producer 且仅 ``require_targets`` 为真时存在。
    - Find producer 原子字段：以下 ``atom_*`` 字段只在 Find 数据请求中构造，不属于 ``unet_c1`` 的输入契约。
    - atom_global_indices: int64 ``(N_A,)``；Find producer 所选局部受体原子在完整受体表中的索引。
    - atom_feat: float32 ``(N_A, 49)``；与局部受体原子逐项对齐的基础特征；第 50 个主链特征由模型输入边界拼接。
    - atom_is_backbone: bool ``(N_A,)``；与 ``atom_feat`` 第 0 维逐项对齐，表示蛋白质或核酸主链原子。
    - atom_coord_world: float32 ``(N_A, 3)``；局部受体原子的世界 XYZ 坐标，单位为 Å。
    - atom_coord_local_voxel: float32 ``(N_A, 3)``；局部受体的 BOX-local 连续 voxel XYZ 坐标。
    - atom_coord_centered_world: float32 ``(N_A, 3)``；局部受体原子相对 BOX 几何中心的世界 XYZ 坐标，单位为 Å。
    - atom_is_in_core_box: bool ``(N_A,)``；逐局部原子是否落在核心 80³ BOX 内；缓冲区原子为假。
    - atom_label: bool ``(N_A,)``；与 ``atom_global_indices`` 第 0 维逐项对齐的 binding 标签，仅 Find 的目标请求存在。

正式资产读取边界:
    - ``density/<pdb_id>/exp.npy`` 和 ``sim.npy``：只读 mmap 的 ``(1, D, H, W)`` float32 NPY；同名 NPZ 保存 schema、voxel size、origin 和形状元数据。
    - ``density/<pdb_id>/union_mask.npy``：只读 mmap 的 ``(1, D, H, W)`` bool occurrence 并集；``ligand_area.npz`` 保存 schema、形状和几何元数据。
    - ``density/<pdb_id>/ligand_dist.npy``：只读 mmap 的 ``(1, D, H, W)`` float16 最近配体原子距离图；``ligand_dist.npz`` 保存距离单位和几何元数据。
    - ``parse/<pdb_id>/receptor_tokens.npz``：提供 ``coords``、``feat``、``is_backbone``；辅助监督请求另外读取 ``res_type`` 和 ``atom_name``。
    - ``labels/<pdb_id>/atom_labels.npz``：提供与完整受体表逐原子对齐的 ``binding_atom`` bool 标签。

缓存消除同一 DataLoader worker 或同一推理进程中物化线程的重复读取, 不改变请求顺序, 裁剪范围或标签值; Dataset 不补零, 也不在 ``__getitem__`` 中静默删除失败请求.
"""

from __future__ import annotations

import random
import threading
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
from src.stage1_producers import FIND_MODEL_NAMES


_AUXILIARY_SUPERVISION_MODEL_NAMES = {"Find_1", "unet_c1", "unet_base", "unet_diff"}


class _ByteLruCache:
    """维护可由同一进程物化线程共享的按字节 LRU 资产缓存.

    输入参数:
        - max_bytes: int; 当前进程内该缓存的总字节上限, 负值按零处理, 零表示禁用写入.

    状态字段:
        - max_bytes: int; 规范化后的缓存上限.
        - current_bytes: int; 当前 ``values`` 中所有数组 ``nbytes`` 之和.
        - values: OrderedDict[str, tuple[dict[str, np.ndarray], int]]; 按最近使用顺序保存资产字典及其字节数, 键由调用方用 PDB identity 和资产类型组成.

    缓存只保存完整图或轻量原子表, 不保存裁好的 BOX. ``get()`` 与 ``put()``
    只在 OrderedDict 和字节计数更新期间持有可重入锁; mmap 裁块与密度通道计算
    不在锁内. 不同请求始终从权威整图重新裁剪, 淘汰不会改变请求内容.
    """

    def __init__(self, max_bytes: int) -> None:
        """初始化一个没有条目的按字节受限 LRU 缓存.

        输入参数:
            - max_bytes: int; 缓存总字节上限, 负值规范化为 ``0``.

        状态变化:
            - ``max_bytes``, ``current_bytes`` 和 ``values`` 被初始化为空缓存状态.
        """
        self.max_bytes = max(0, int(max_bytes))
        self.current_bytes = 0
        self.values: OrderedDict[str, tuple[dict[str, np.ndarray], int]] = OrderedDict()
        self._lock = threading.RLock()

    @staticmethod
    def _size_bytes(value: Mapping[str, np.ndarray]) -> int:
        """计算一个资产字段映射中 NumPy 数组的实际内存占用。

        输入参数:
            - value: Mapping[str, np.ndarray]；同一资产的字段映射，非 NumPy 值不计入大小。

        返回值:
            - size_bytes: int；所有数组 ``nbytes`` 的非负整数和。
        """
        return sum(int(array.nbytes) for array in value.values() if isinstance(array, np.ndarray))

    def get(self, key: str) -> dict[str, np.ndarray] | None:
        """读取一个缓存资产并将命中条目移动到 LRU 队尾。

        输入参数:
            - key: str；调用方构造的资产键，通常包含 PDB identity 和资产类型。

        返回值:
            - value: dict[str, np.ndarray] | None；命中时返回缓存中的 CPU NumPy 字段映射，未命中时返回 ``None``。

        状态变化:
            - 命中条目从原顺序位置移动到最近使用位置；未命中不修改缓存。
        """
        with self._lock:
            item = self.values.pop(key, None)
            if item is None:
                return None
            self.values[key] = item
            return item[0]

    def put(self, key: str, value: dict[str, np.ndarray]) -> None:
        """写入一个资产并按最近使用顺序淘汰旧条目。

        输入参数:
            - key: str；调用方构造的资产键。
            - value: dict[str, np.ndarray]；要缓存的 CPU NumPy 字段映射。

        状态变化:
            - 已有同键条目先移除；若新条目不超过上限，则从最久未使用条目开始淘汰，直到总字节数可容纳新条目。
            - 缓存禁用或单条目超过上限时不写入，且不会抛出容量异常。
        """
        with self._lock:
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


def _load_mmap_array(path: Path, expected_dtype: np.dtype) -> np.memmap:
    """以只读 mmap 打开一个 V3 体素 NPY，并检查文件级形状与 dtype。

    输入参数:
        - path: Path；要读取的 NPY 文件，预期内容为 ``(1, D, H, W)``，第 0 维是单通道包装维。
        - expected_dtype: np.dtype；文件必须使用的 NumPy dtype，例如 ``float32``、``float16`` 或 ``bool``。

    返回值:
        - array: np.memmap；只读的 ``(1, D, H, W)`` mmap，调用方负责在实际 80³ BOX 上裁剪，不在这里扫描完整体数组。

    失败语义:
        - NPY 不是可 mmap 的四维单通道数组，或 dtype 与 ``expected_dtype`` 不一致时抛出 ``ValueError``。
    """

    array = np.load(path, mmap_mode="r", allow_pickle=False)
    if not isinstance(array, np.memmap) or array.ndim != 4 or array.shape[0] != 1:
        raise ValueError(f"{path}: NPY 必须是可内存映射的 (1,D,H,W) 数组。")
    if array.dtype != np.dtype(expected_dtype):
        raise ValueError(f"{path}: dtype 应为 {np.dtype(expected_dtype)}，实际为 {array.dtype}。")
    return array


def _crop_80(array: np.ndarray, start_zyx: Sequence[int]) -> np.ndarray:
    """从完整图裁出一个真实的 80³ ZYX BOX，并在裁块上检查数值。

    输入参数:
        - array: np.ndarray ``(D_full, H_full, W_full)``；完整图的 ZYX 体素网格，允许是 mmap 视图。
        - start_zyx: Sequence[int] ``(3,)``；完整图离散体素的 ZYX BOX corner index，边界由请求解析阶段保证。

    返回值:
        - crop: np.ndarray ``(80, 80, 80)``；与输入 dtype 相同的连续 ZYX 裁块；浮点数组在返回前拒绝 NaN 和 Inf。

    失败语义:
        - 裁块形状不是 ``(80, 80, 80)``，或浮点裁块包含非有限值时抛出异常；不会用零填充越界区域。
    """
    z0, y0, x0 = (int(value) for value in start_zyx)
    crop = array[z0 : z0 + 80, y0 : y0 + 80, x0 : x0 + 80]
    if crop.shape != STAGE1_BOX_SHAPE_ZYX:
        raise RuntimeError(f"Stage1 crop 必须恰为 80³，实际 {crop.shape}。")
    crop = np.ascontiguousarray(crop)
    if np.issubdtype(crop.dtype, np.floating) and not np.isfinite(crop).all():
        raise ValueError("Stage1 浮点裁块包含 NaN 或 Inf。")
    return crop


def _mainchain_class_targets(
    atom_coord_local_voxel: np.ndarray,
    atom_is_in_core_box: np.ndarray,
    res_type: np.ndarray,
    atom_name: np.ndarray,
    box_shape_zyx: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """把核心 BOX 内的受体原子散射为蛋白和核酸主链类别图。

    函数只处理 ``atom_is_in_core_box`` 为真的原子；连续坐标按 XYZ 解释，向下取整后换为 ZYX 数组索引。未命中主链原子名、残基类型未知或位于 8 Å 缓冲区的原子不写入，背景保持类别 ``0``。所有输入第 0 维必须逐局部受体原子对齐。

    输入参数:
        - atom_coord_local_voxel: np.ndarray float32 ``(N_A, 3)``；BOX-local 连续 voxel XYZ 坐标。
        - atom_is_in_core_box: np.ndarray bool ``(N_A,)``；逐原子核心 BOX 成员标志。
        - res_type: np.ndarray uint8 ``(N_A,)``；``0..19`` 为蛋白残基，``20..27`` 为核酸残基，``28`` 为未知残基。
        - atom_name: np.ndarray 字符串 ``(N_A,)``；与坐标第 0 维对齐的原子名。
        - box_shape_zyx: np.ndarray int ``(3,)``；输出体素图的 ZYX 形状。

    返回值:
        - protein_target: np.ndarray uint8 ``(D, H, W)``；背景为 0，非零值是 N、CA、C、O 的蛋白主链类别编号。
        - nucleic_target: np.ndarray uint8 ``(D, H, W)``；背景为 0，非零值是 P、O5'、C5'、C4'、C3'、O3' 的核酸主链类别编号。
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
    """按 ``np.rot90`` 的离散网格语义旋转 BOX-local 连续 ZYX 坐标。

    坐标分量按 ZYX 顺序传入，采用 voxel-grid corner 坐标；每次 90° 旋转都使用旋转前两轴的 ``shape``，因此连续坐标与密度、掩码的索引变换保持一致。

    输入参数:
        - coord_zyx: np.ndarray float32 ``(N_A, 3)``；局部原子的连续 ZYX voxel 坐标。
        - box_shape_zyx: np.ndarray int ``(3,)``；旋转前 BOX 的 ZYX 体素形状。
        - axis1: int；``np.rot90`` 的第一个 ZYX 空间轴，取 ``0``、``1`` 或 ``2``。
        - axis2: int；``np.rot90`` 的第二个 ZYX 空间轴，与 ``axis1`` 不同。
        - k: int；逆时针 90° 旋转次数，实际使用 ``k % 4``。

    返回值:
        - rotated_coord_zyx: np.ndarray float32 ``(N_A, 3)``；旋转后的局部连续 ZYX 坐标，不修改输入数组。
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
    """对单 BOX 的空间字段和 Find 原子坐标执行同一次随机 90° 旋转。

    输入参数:
        - sample: dict[str, Any]；单 BOX NumPy 字段映射；``density_input`` 的空间轴是 ``(D, H, W)`` ZYX，体素图目标是 ``(D, H, W)``，原子坐标最后一维是 XYZ。Find 字段不存在时只旋转体素字段。

    状态变化:
        - ``density_input``：保留通道轴，沿选定的两个 ZYX 空间轴调用 ``np.rot90``。
        - ``hardmask``、``voxel_label``、``ligand_area_target``、``protein_mainchain_target``、``nucleic_mainchain_target`` 和 ``ligand_inverse_distance_target``：使用同一轴和旋转次数变换 ZYX 网格。
        - ``atom_coord_local_voxel``、``atom_coord_centered_world`` 和 ``atom_coord_world``：按同一几何变换更新逐原子 XYZ 坐标；``voxel_size_world`` 按交换的空间轴同步更新。

    返回值:
        - sample: dict[str, Any]；原字典本身；旋转次数为零时不修改任何字段，否则在原映射内替换旋转后的连续数组。

    失败语义:
        - ``box_shape_zyx`` 不是立方体时抛出 ``ValueError``；本轮增强不支持非立方体 BOX。
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
    """按 BOX-level 字段契约把 NumPy 数组转换为连续 CPU tensor。

    输入参数:
        - sample: dict[str, Any]；单 BOX NumPy 字段映射，identity、role 和候选编号保持 Python 标量，数值字段由 ``dtype_by_field`` 指定转换。

    返回值:
        - tensor_sample: dict[str, Any]；保留输入键集合；``box_start_zyx`` 为 int32 ``(3,)``，``box_shape_zyx`` 为 int64 ``(3,)``，几何和密度字段为 float32，体素掩码及原子标志为 bool。
        - tensor_sample["protein_mainchain_target"]: torch.Tensor int64 ``(D, H, W)``；蛋白主链类别图。
        - tensor_sample["nucleic_mainchain_target"]: torch.Tensor int64 ``(D, H, W)``；核酸主链类别图。
        - tensor_sample["ligand_inverse_distance_target"]: torch.Tensor float32 ``(D, H, W)``；距离回归目标。
        - tensor_sample["atom_feat"]: torch.Tensor float32 ``(N_A, 49)``；Find 原子基础特征；``atom_is_backbone`` 是同一原子轴上的独立 bool 字段。
        - tensor_sample["atom_coord_world"]: torch.Tensor float32 ``(N_A, 3)``；世界 XYZ 坐标；局部和中心坐标保持相同的逐原子第 0 维。

    状态变化:
        - 输入 NumPy 数组不会被原地改 dtype；非连续或只读数组先复制为连续可转 tensor 的内存。
        - 未列入 ``dtype_by_field`` 的身份字段和扩展字段保持原对象语义。
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
        "atom_is_backbone": torch.bool,
        "atom_is_in_core_box": torch.bool,
        "atom_label": torch.bool,
    }
    result = dict(sample)
    for field_name, dtype in dtype_by_field.items():
        if field_name in result:
            array = np.ascontiguousarray(result[field_name])
            if not array.flags.writeable:
                array = array.copy()
            result[field_name] = torch.as_tensor(array, dtype=dtype)
    return result


# ================================================================================================


class Stage1Dataset(Dataset):
    """统一物化 train、validation、full_map 和 centered 的真实 80³ 请求。

    构造参数:
        - all_data_path: str；A-G 正式根目录，下面必须有 ``density``、``parse`` 和 ``labels``。
        - split_file: str | Path | Sequence[ResolvedStage1Crop]；训练 pool、冻结 validation 请求文件或推理层传入的内存请求序列。
        - mode: str；请求模式，支持 ``train``、``val``、``validation``、``full_map`` 和 ``centered``。
        - stage1_model_name: str；producer 身份名称；Find 名称决定是否附加受体原子字段，密度输入由 ``density_channel_config.enabled_channels`` 决定。
        - box_pool_root: str | None；包含 V3 ``manifest.json`` 和 validation selection 的 pool 根目录；内存请求序列可不提供。
        - density_channel_config: Mapping[str, Any]；密度裁剪、拟合和启用通道的配置映射。
        - pdb_foreground_box_num: int; 每个 PDB 的目标 bias BOX 数量; 默认值为 25, 当前 Hydra 配置显式传入; 内存请求序列分支忽略该参数.
        - pdb_foreground_fraction_target: float; bias BOX 占目标 bias 与 context BOX 总数的比例; 默认值为 0.5, 当前 Hydra 配置显式传入; 内存请求序列分支忽略该参数.
        - pdb_occurrence_foreground_box_cap: int; 单个 occurrence 每个 epoch 的 bias BOX 数量上限; 默认值为 25, 当前 Hydra 配置显式传入; 内存请求序列分支忽略该参数.
        - atom_buffer_radius: float；核心 BOX 外选择受体原子的世界坐标缓冲半径，本 Dataset 固定为 ``8.0 Å``。
        - request_seed: int；训练请求层用于确定性展开的基准 seed。
        - cache_max_bytes: int；每个 DataLoader worker 的受体表、监督数组和完整图 LRU 缓存字节上限。
        - enable_random_rotation: bool；训练模式是否同步旋转密度、所有体素目标、原子坐标和体素几何。
        - name: str；Dataset 的显示名称，不参与请求解析。
        - split_train: str | Sequence[str] | None；历史兼容参数，当前实现显式丢弃，不参与 V3 请求构造。
        - split_val: str | Sequence[str] | None；历史兼容参数，当前实现显式丢弃，不参与 V3 请求构造。
        - class_names: Sequence[str]；二分类名称，必须严格为 ``("background", "foreground")``。

    输出契约:
        - Find producer：体素字段加上 ``atom_*`` 局部受体字段，输入特征是 49D ``atom_feat`` 与独立的 bool ``atom_is_backbone``。
        - ``unet_c1``：使用同一体素物化路径并可生成主链、配体区域和距离辅助监督，不返回逐原子输入表。
        - 所有模式：输出字段和 dtype 由本模块顶部清单定义；不对越界 BOX 补零。

    生命周期边界:
        - 请求起点必须已经由 ``resolve_stage1_start`` 合法化；Dataset 只验证请求与完整图一致，不重新修正起点。
        - ``__getitem__`` 不删除失败样本、不修改请求源；读取失败直接抛出，让调用方看见资产契约问题。
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
        pdb_foreground_box_num: int = 25,
        pdb_foreground_fraction_target: float = 0.5,
        pdb_occurrence_foreground_box_cap: int = 25,
        atom_buffer_radius: float = 8.0,
        request_seed: int = 3407,
        cache_max_bytes: int = 536_870_912,
        enable_random_rotation: bool = True,
        name: str = "stage1",
        split_train: str | Sequence[str] | None = None,
        split_val: str | Sequence[str] | None = None,
        class_names: Sequence[str] = ("background", "foreground"),
    ) -> None:
        """解析请求源, 密度通道契约和当前进程的完整图缓存.

        输入参数:
            - all_data_path: str；A-G 正式资产根目录。
            - split_file: str | Path | Sequence[ResolvedStage1Crop]；V3 pool、冻结请求文件或已解析请求序列。
            - mode: str；决定 ``build_request_source`` 选择训练、validation 或推理请求展开方式。
            - stage1_model_name: str；producer 身份名称；不限制 density-only producer 的密度通道组合。
            - box_pool_root: str | None；V3 pool 根目录；内存请求序列不需要该路径。
            - density_channel_config: Mapping[str, Any]；传给 ``DensityChannelConfig`` 的通道字段。
            - pdb_foreground_box_num: int; 每个 PDB 的目标 bias BOX 数量; 训练目录分支在调用方省略时使用默认值 25, 当前 Hydra 配置显式传入; 内存请求序列分支忽略该参数.
            - pdb_foreground_fraction_target: float; bias BOX 占目标 bias 与 context BOX 总数的比例; 训练目录分支在调用方省略时使用默认值 0.5, 当前 Hydra 配置显式传入; 内存请求序列分支忽略该参数.
            - pdb_occurrence_foreground_box_cap: int; 单个 occurrence 每个 epoch 的 bias BOX 数量上限; 训练目录分支在调用方省略时使用默认值 25, 当前 Hydra 配置显式传入; 内存请求序列分支忽略该参数.
            - atom_buffer_radius: float；必须为 ``8.0``，用于局部受体原子选择。
            - request_seed: int；仅传给 ``build_request_source`` 生成训练周期请求。
            - cache_max_bytes: int; 当前 Dataset 实例的完整图和受体资产缓存上限.
            - enable_random_rotation: bool；仅在 train 模式开启同步空间增强。
            - name: str；Dataset 显示名称。
            - split_train: str | Sequence[str] | None；历史兼容参数，构造时丢弃。
            - split_val: str | Sequence[str] | None；历史兼容参数，构造时丢弃。
            - class_names: Sequence[str]；必须是背景在前、前景在后的二分类名称。

        状态变化:
            - ``request_source`` 保存固定请求序列或动态训练请求集。
            - ``resolved_density_channels`` 保存展开 ``all`` 后的通道顺序。
            - ``_source_cache`` 初始化为当前 Dataset 实例共享的线程安全按字节 LRU.
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
        if self.class_names != ("background", "foreground"):
            raise ValueError("AdaLigand Stage1 当前只接受 [background, foreground] 二分类顺序。")
        if float(atom_buffer_radius) != 8.0:
            raise ValueError("AdaLigand Find Dataset 的 atom_buffer_radius 固定为 8.0 Å。")
        self.atom_buffer_radius = 8.0
        self.enable_random_rotation = bool(enable_random_rotation and self.mode == "train")
        if isinstance(split_file, (list, tuple)) and all(
            isinstance(request, ResolvedStage1Crop) for request in split_file
        ):
            self.request_source = tuple(split_file)
            if not self.request_source:
                raise ValueError("内存 Stage1 请求序列不能为空。")
        else:
            self.request_source = build_request_source(
                split_file=split_file,
                mode=self.mode,
                box_pool_root=box_pool_root,
                seed=int(request_seed),
                pdb_foreground_box_num=int(pdb_foreground_box_num),
                pdb_foreground_fraction_target=float(pdb_foreground_fraction_target),
                pdb_occurrence_foreground_box_cap=int(
                    pdb_occurrence_foreground_box_cap
                ),
            )
        # dict[str, Any]；从 Hydra dataset 配置读取的密度通道构造字段。
        channel_cfg = dict(density_channel_config)
        self.density_config = DensityChannelConfig(
            clip_percentile=tuple(float(value) for value in channel_cfg["clip_percentile"]),
            fit_mask_percentile=float(channel_cfg["fit_mask_percentile"]),
            enabled_channels=[str(value) for value in channel_cfg["enabled_channels"]],
        )
        # list[str]；长度为 C_density，展开 ``all`` 后的 producer 输入通道顺序。
        resolved_channels = (
            list(ALL_CHANNEL_NAMES)
            if "all" in [value.lower() for value in self.density_config.enabled_channels]
            else list(self.density_config.enabled_channels)
        )
        if self.stage1_model_name in FIND_MODEL_NAMES and resolved_channels != list(ALL_CHANNEL_NAMES):
            raise ValueError("Find producer 必须按权威顺序启用完整 56D ALL density channels。")
        self.resolved_density_channels = tuple(resolved_channels)
        # bool; 任一 sim、diff 或 posdiff 通道都要求同时读取模拟密度整图。
        self.requires_sim_density = any(
            channel_name.startswith(("sim_", "diff_", "posdiff_"))
            for channel_name in self.resolved_density_channels
        )
        # _ByteLruCache; 同一 Dataset 实例共享轻量 receptor 表, 监督数组和最近使用的原始整图.
        # full_map 连续消费同一 PDB 的多个窗口时复用缓存命中; 首次并发 miss 允许重复打开 mmap.
        self._source_cache = _ByteLruCache(cache_max_bytes)

    def set_epoch(self, epoch: int) -> None:
        """通知动态训练请求源切换到指定 epoch。

        输入参数:
            - epoch: int；训练周期编号；传给动态 ``Stage1TrainingRequestSet`` 作为请求展开的确定性输入。

        状态变化:
            - 动态请求源重建该 epoch 的请求集合；固定 tuple 请求源保持不变。
        """
        set_epoch = getattr(self.request_source, "set_epoch", None)
        if callable(set_epoch):
            set_epoch(int(epoch))

    def __len__(self) -> int:
        """返回当前请求源可索引的 BOX 请求数量。

        返回值:
            - count: int；动态训练请求源当前 epoch 的请求数，或冻结请求序列的固定长度。
        """
        return len(self.request_source)

    def describe_index(self, index: int) -> str:
        """生成一个请求位置的人类可读身份摘要。

        输入参数:
            - index: int；请求源中的零基位置；越界行为由请求源的索引实现决定。

        返回值:
            - description: str；包含 PDB identity、请求 role、完整图 ZYX 起点和 occurrence 编号的单行摘要。
        """
        request = self.request_source[index]
        return (
            f"pdb_id={request.pdb_id}, role={request.role}, "
            f"start_zyx={request.box_start_zyx}, occurrence_id={request.occurrence_id}"
        )

    def _load_structure(self, pdb_id: str, require_targets: bool) -> dict[str, np.ndarray]:
        """读取并缓存一个 PDB 的完整受体表及请求所需监督字段。

        输入参数:
            - pdb_id: str；当前 PDB identity；目录名必须与正式资产目录一致。
            - require_targets: bool；为真时额外读取逐原子 binding 标签，并在辅助监督 producer 中读取残基类型和原子名。

        返回值:
            - coords: np.ndarray float32 ``(N_receptor, 3)``；受体原子的世界 XYZ 坐标，单位为 Å。
            - feat: np.ndarray float32 ``(N_receptor, 49)``；与 ``coords`` 第 0 维逐项对齐的基础原子特征。
            - is_backbone: np.ndarray bool ``(N_receptor,)``；与 ``coords`` 第 0 维逐项对齐的主链标志；来源是 ``receptor_tokens.npz`` 的既有字段。
            - binding_atom: np.ndarray bool ``(N_receptor,)``；与完整受体表逐项对齐的 binding 标签，仅 ``require_targets`` 为真时存在。
            - res_type: np.ndarray uint8 ``(N_receptor,)``；辅助监督的残基类型编号，仅辅助监督 producer 的目标请求存在。
            - atom_name: np.ndarray 字符串 ``(N_receptor,)``；辅助监督的原子名，仅辅助监督 producer 的目标请求存在。

        文件读取:
            - ``parse/<pdb_id>/receptor_tokens.npz``：提供 ``coords``、``feat``、``is_backbone``，以及按需读取的 ``res_type``、``atom_name``。
            - ``labels/<pdb_id>/atom_labels.npz``：提供 ``binding_atom``；其第 0 维必须与受体表完全一致。

        缓存语义:
            - cache key 同时编码 PDB identity 和 ``require_targets``，避免无监督读取错误复用带标签或辅助字段的结构表。
        """
        cache_key = f"{pdb_id}|targets={int(require_targets)}"
        cached = self._source_cache.get(cache_key)
        if cached is not None:
            return cached
        receptor_path = self.root / "parse" / pdb_id / "receptor_tokens.npz"
        with np.load(receptor_path, allow_pickle=False) as data:
            # np.ndarray float32 (N_receptor, 3)；完整受体表的世界 XYZ 坐标，单位为 Å。
            coords = np.asarray(data["coords"], dtype=np.float32)
            # np.ndarray float32 (N_receptor, 49)；与 coords 第 0 维逐受体原子对齐的基础特征。
            feat_base = np.asarray(data["feat"], dtype=np.float32)
            # np.ndarray bool (N_receptor,)；True 表示蛋白质或核酸主链原子，来自已有 receptor_tokens 字段。
            is_backbone = np.asarray(data["is_backbone"], dtype=bool)
        if (
            coords.ndim != 2
            or coords.shape[1] != 3
            or feat_base.shape != (coords.shape[0], 49)
            or is_backbone.shape != (coords.shape[0],)
        ):
            raise ValueError(
                f"{receptor_path}: coords/feat/is_backbone 必须为 [N,3]/[N,49]/[N]。"
            )
        if not np.isfinite(coords).all() or not np.isfinite(feat_base).all():
            raise ValueError(f"{receptor_path}: coords/feat 包含 NaN/Inf。")
        structure = {"coords": coords, "feat": feat_base, "is_backbone": is_backbone}
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
        """mmap 并缓存一个 PDB 的 exp 或 sim 完整图及其几何元数据。

        输入参数:
            - pdb_id: str；当前 PDB identity；读取 ``density/<pdb_id>`` 子目录。
            - grid_name: str；只能是 ``exp`` 或 ``sim``；缓存的是原始 float32 图，不缓存派生通道。

        返回值:
            - grid_zyx: np.ndarray float32 mmap ``(D_full, H_full, W_full)``；去除 NPY 单通道包装维后的完整 ZYX 网格。
            - voxel_size_xyz: np.ndarray float32 ``(3,)``；世界 XYZ 轴每体素尺寸，单位为 Å。
            - origin_xyz: np.ndarray float32 ``(3,)``；完整图 voxel-grid corner 的世界 XYZ 坐标，单位为 Å。

        文件契约:
            - ``<grid_name>.npy`` 必须是 ``(1, D_full, H_full, W_full)`` float32 NPY；``<grid_name>.npz`` 的 schema 必须为 2，且声明形状与 NPY 一致。
            - exp NPZ 的 ``canonical_shape_zyx`` 被核对；sim 的完整图形状取自 NPY，并在物化阶段与 exp 的实际 shape 比较；本入口不额外读取 sim NPZ 的 shape 字段。

        缓存语义:
            - 返回的完整图可能是只读 mmap; 调用方只能裁剪读取, 不得原地改写 Dataset 实例共享数组.
            - 这里不扫描完整图的 NaN/Inf；数值检查发生在实际 80³ 裁块上。
        """
        cache_key = f"{pdb_id}|density={grid_name}"
        cached = self._source_cache.get(cache_key)
        if cached is None:
            density_directory = self.root / "density" / pdb_id
            grid = _load_mmap_array(density_directory / f"{grid_name}.npy", np.float32)
            with np.load(density_directory / f"{grid_name}.npz", allow_pickle=False) as metadata:
                schema_version = int(np.asarray(metadata["schema_version"]).item())
                voxel_size = np.asarray(metadata["voxel_size"], dtype=np.float32)
                origin = np.asarray(metadata["origin"], dtype=np.float32)
                declared_shape = (
                    np.asarray(metadata["canonical_shape_zyx"], dtype=np.int64)
                    if grid_name == "exp"
                    else np.asarray(grid.shape[1:], dtype=np.int64)
                )
            if schema_version != 2:
                raise ValueError(f"{density_directory}: {grid_name}.npz schema_version 必须为 2。")
            if not np.array_equal(declared_shape, np.asarray(grid.shape[1:], dtype=np.int64)):
                raise ValueError(f"{density_directory}: {grid_name} NPY 形状与 NPZ 元数据不一致。")
            cached = {
                "grid": grid[0],
                "voxel_size": voxel_size,
                "origin": origin,
            }
            self._source_cache.put(cache_key, cached)
        return cached["grid"], cached["voxel_size"], cached["origin"]

    def _load_ligand_union(
        self,
        pdb_id: str,
        expected_shape_zyx: Sequence[int],
        expected_voxel_size_xyz: np.ndarray,
        expected_origin_xyz: np.ndarray,
    ) -> np.ndarray:
        """读取并缓存 schema-v3 occurrence union mask，保持完整图 bool 语义。

        输入参数:
            - pdb_id: str；当前 PDB identity。
            - expected_shape_zyx: Sequence[int] ``(3,)``；exp 完整图的 ZYX 形状。
            - expected_voxel_size_xyz: np.ndarray float32 ``(3,)``；exp 的世界 XYZ 体素尺寸，单位为 Å。
            - expected_origin_xyz: np.ndarray float32 ``(3,)``；exp voxel-grid corner 的世界 XYZ 坐标，单位为 Å。

        返回值:
            - union_mask: np.ndarray bool mmap ``(1, D_full, H_full, W_full)``；所有 occurrence 配体区域的完整图 ZYX 并集，保留 NPY 的单通道包装维。

        文件契约:
            - ``union_mask.npy`` 必须与 exp 形状一致；``ligand_area.npz`` 的 schema 必须为 3，``grid_shape_zyx``、``voxel_size_xyz`` 和 ``origin_xyz`` 必须与 exp 几何完全一致。
        """
        expected_shape = tuple(int(value) for value in expected_shape_zyx)
        cache_key = f"{pdb_id}|ligand_union"
        cached = self._source_cache.get(cache_key)
        if cached is None:
            density_directory = self.root / "density" / pdb_id
            ligand_path = density_directory / "ligand_area.npz"
            union_mask = _load_mmap_array(density_directory / "union_mask.npy", np.bool_)
            with np.load(ligand_path, allow_pickle=False) as data:
                schema_version = int(np.asarray(data["schema_version"]).item())
                declared_shape = np.asarray(data["grid_shape_zyx"], dtype=np.int64)
                voxel_size = np.asarray(data["voxel_size_xyz"], dtype=np.float32)
                origin = np.asarray(data["origin_xyz"], dtype=np.float32)
            if schema_version != 3:
                raise ValueError(f"{ligand_path}: schema_version 必须为 3。")
            if not np.array_equal(declared_shape, np.asarray(expected_shape, dtype=np.int64)):
                raise ValueError(f"{ligand_path}: grid_shape_zyx 与实验密度不一致。")
            if not np.array_equal(voxel_size, expected_voxel_size_xyz) or not np.array_equal(
                origin,
                expected_origin_xyz,
            ):
                raise ValueError(f"{ligand_path}: voxel_size_xyz/origin_xyz 与实验密度不一致。")
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
        """核对空间契约后读取与 exp 逐体素对齐的最近配体原子距离图。

        输入参数:
            - pdb_id: str；当前 PDB identity。
            - expected_shape_zyx: Sequence[int] ``(3,)``；exp 完整图的 ZYX 形状。
            - expected_voxel_size_xyz: np.ndarray float32 ``(3,)``；exp 世界 XYZ 体素尺寸。
            - expected_origin_xyz: np.ndarray float32 ``(3,)``；exp voxel-grid corner 的世界 XYZ 原点。

        文件字段:
            - ``ligand_dist.npy``：float16 ``(1, D_full, H_full, W_full)``；完整图到最近配体原子的距离，单位为 Å；实际 80³ 裁块必须有限且非负。
            - ``ligand_dist.npz:schema_version``：uint16 标量，必须为 ``1``。
            - ``ligand_dist.npz:grid_shape_zyx``：int64 ``(3,)``，必须与 exp 形状一致。
            - ``ligand_dist.npz:voxel_size_xyz``：float32 ``(3,)``，必须与 exp 的世界 XYZ 体素尺寸逐项一致。
            - ``ligand_dist.npz:origin_xyz``：float32 ``(3,)``，必须与 exp 的世界 XYZ 原点逐项一致。
            - ``ligand_dist.npz:distance_unit``：字符串标量 ``"angstrom"``，声明距离单位。

        返回值:
            - distance: np.ndarray float16 mmap ``(1, D_full, H_full, W_full)``；缓存的完整图距离数组，调用方只裁剪读取。

        数值检查边界:
            - 为避免 mmap miss 时扫描完整文件，本函数只核对文件级 dtype、形状和几何；NaN、Inf 与负数由实际 80³ 裁块路径检查。
        """
        expected_shape = tuple(int(value) for value in expected_shape_zyx)
        expected_voxel_size = np.asarray(expected_voxel_size_xyz, dtype=np.float32)
        expected_origin = np.asarray(expected_origin_xyz, dtype=np.float32)
        cache_key = f"{pdb_id}|ligand_distance"
        cached = self._source_cache.get(cache_key)
        if cached is None:
            density_directory = self.root / "density" / pdb_id
            distance_path = density_directory / "ligand_dist.npz"
            distance = _load_mmap_array(density_directory / "ligand_dist.npy", np.float16)
            with np.load(distance_path, allow_pickle=False) as data:
                required = {
                    "schema_version",
                    "grid_shape_zyx",
                    "voxel_size_xyz",
                    "origin_xyz",
                    "distance_unit",
                }
                missing = sorted(required.difference(data.files))
                if missing:
                    raise KeyError(f"{distance_path} 缺少字段 {missing}。")
                schema_version = np.asarray(data["schema_version"])
                grid_shape = np.asarray(data["grid_shape_zyx"])
                voxel_size = np.asarray(data["voxel_size_xyz"])
                origin = np.asarray(data["origin_xyz"])
                distance_unit = np.asarray(data["distance_unit"])
            if distance.dtype != np.float16:
                raise ValueError(f"{distance_path}: distance 必须为 float16。")
            if distance.shape != (1, *expected_shape):
                raise ValueError(f"{distance_path}: distance 与 exp grid 形状不一致。")
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
        """把一个已经解析的请求物化为 producer 专属的单 BOX tensor 字典。

        输入参数:
            - request: ResolvedStage1Crop；PDB identity、完整图离散 ZYX BOX 起点、请求 role 和监督开关已经由共享请求层冻结。

        返回值:
            - pdb_id、request_role、occurrence_id、candidate_index：请求身份和来源字段，保持 Python 标量。
            - box_start_zyx: torch.Tensor int32 ``(3,)``；完整图离散 ZYX BOX corner index。
            - box_shape_zyx: torch.Tensor int64 ``(3,)``；固定为 ``(80, 80, 80)`` 的 ZYX 形状。
            - box_origin_world: torch.Tensor float32 ``(3,)``；BOX corner 的世界 XYZ 坐标，单位为 Å。
            - voxel_size_world: torch.Tensor float32 ``(3,)``；世界 XYZ 轴体素尺寸，单位为 Å。
            - density_input: torch.Tensor float32 ``(C_density, 80, 80, 80)``；producer 的 ZYX 密度通道。
            - hardmask: torch.Tensor bool ``(80, 80, 80)``；核心 BOX 的受体占据掩码。
            - voxel_label: torch.Tensor bool ``(80, 80, 80)``；核心 BOX 的 binding 原子体素标签，仅目标请求存在。
            - ligand_area_target: torch.Tensor bool ``(80, 80, 80)``；occurrence union mask 裁剪，仅目标请求存在。
            - protein_mainchain_target: torch.Tensor int64 ``(80, 80, 80)``；蛋白主链类别图，仅辅助监督 producer 存在。
            - nucleic_mainchain_target: torch.Tensor int64 ``(80, 80, 80)``；核酸主链类别图，仅辅助监督 producer 存在。
            - ligand_inverse_distance_target: torch.Tensor float32 ``(80, 80, 80)``；有限非负距离经 ``1/(1+distance_Å)`` 转换后的目标。
            - atom_global_indices: torch.Tensor int64 ``(N_A,)``；局部受体原子在完整受体表中的索引，仅 Find producer 存在。
            - atom_feat: torch.Tensor float32 ``(N_A, 49)``；局部受体基础特征，仅 Find producer 存在。
            - atom_is_backbone: torch.Tensor bool ``(N_A,)``；与 ``atom_feat`` 第 0 维对齐的主链标志，仅 Find producer 存在。
            - atom_coord_world: torch.Tensor float32 ``(N_A, 3)``；局部原子的世界 XYZ 坐标，单位为 Å。
            - atom_coord_local_voxel: torch.Tensor float32 ``(N_A, 3)``；局部连续 voxel XYZ 坐标。
            - atom_coord_centered_world: torch.Tensor float32 ``(N_A, 3)``；相对 BOX 中心的世界 XYZ 坐标，单位为 Å。
            - atom_is_in_core_box: torch.Tensor bool ``(N_A,)``；逐局部原子的核心 BOX 成员标志。
            - atom_label: torch.Tensor bool ``(N_A,)``；逐局部原子的 binding 标签，仅 Find 目标请求存在。

        处理边界:
            - Find 读取 exp 和 sim 并要求两者的完整图形状、体素尺寸和原点逐项一致；``unet_c1`` 只读取 exp。
            - 体素数值只在实际 80³ 裁块上检查；不对 mmap 完整图执行全量有限性扫描。
            - 返回前由 :func:`_to_tensor_sample` 统一转换 dtype；不补零，不在本函数中删除请求。
        """
        # np.ndarray float32 (D_full, H_full, W_full)；实验完整图，去除 NPY 的单通道包装维。
        # np.ndarray float32 (3,)；世界 XYZ 体素尺寸和完整图 voxel-grid corner 原点，单位为 Å。
        exp_grid, voxel_size, full_origin = self._load_density_grid(request.pdb_id, "exp")
        full_shape = np.asarray(exp_grid.shape, dtype=np.int64)
        resolved_start = resolve_stage1_start(request.box_start_zyx, full_shape)
        if resolved_start != request.box_start_zyx:
            raise ValueError(
                "Stage1 Dataset 只消费预先解析的起点："
                f"request={request.box_start_zyx}, resolved={resolved_start}, pdb={request.pdb_id}。"
            )
        # np.ndarray int32 (3,)；当前 BOX 的完整图离散 ZYX corner index。
        start_zyx = np.asarray(resolved_start, dtype=np.int32)
        # np.ndarray int64 (3,)；固定 80³ BOX 的 ZYX 形状。
        box_shape_zyx = np.asarray(STAGE1_BOX_SHAPE_ZYX, dtype=np.int64)
        # np.ndarray float32 (3,)；将 ZYX 起点换成世界 XYZ 乘法所需的轴顺序。
        start_xyz = start_zyx[[2, 1, 0]].astype(np.float32)
        # np.ndarray float32 (3,)；当前 BOX voxel-grid corner 的世界 XYZ 原点，单位为 Å。
        box_origin = (full_origin + start_xyz * voxel_size).astype(np.float32, copy=False)

        structure = self._load_structure(request.pdb_id, request.require_targets)
        # dict[str, np.ndarray]；从完整受体表选择核心 BOX 外 8 Å 缓冲内的原子，并给出逐局部原子的核心成员标志。
        selection = select_atoms_for_box(
            atom_coords_world=structure["coords"],
            box_origin_world=box_origin,
            voxel_size_world=voxel_size,
            box_shape_zyx=box_shape_zyx,
            buffer_radius=self.atom_buffer_radius,
        )
        selected_idx = selection["selected_idx"]
        # dict[str, np.ndarray]；三套逐原子 (N_A, 3) XYZ 坐标：世界坐标、BOX-local 连续 voxel 坐标和相对 BOX 中心的世界坐标。
        atom_coordinates = build_atom_coordinates(
            atom_coords_world=structure["coords"],
            selected_idx=selected_idx,
            box_origin_world=box_origin,
            voxel_size_world=voxel_size,
            box_shape_zyx=box_shape_zyx,
        )
        core_mask = selection["atom_is_in_core_box"]
        # np.ndarray bool (80, 80, 80)；只由核心受体原子占据生成的 ZYX 硬掩码。
        hardmask = build_hardmask_from_atom_coordinates(
            atom_coord_local_voxel=atom_coordinates["atom_coord_local_voxel"],
            atom_is_in_core_box=core_mask,
            box_shape_zyx=box_shape_zyx,
        ).astype(bool, copy=False)

        # np.ndarray float32 (80, 80, 80)；当前 ZYX 起点的实验密度裁块。
        exp_crop = _crop_80(exp_grid, start_zyx)
        if self.requires_sim_density:
            sim_grid, sim_voxel_size, sim_origin = self._load_density_grid(request.pdb_id, "sim")
            if sim_grid.shape != exp_grid.shape or not np.array_equal(sim_voxel_size, voxel_size) or not np.array_equal(sim_origin, full_origin):
                raise ValueError(f"{request.pdb_id}: exp/sim 的 shape、voxel_size、origin 必须完全一致。")
            sim_crop = _crop_80(sim_grid, start_zyx)
        else:
            sim_crop = None
        # np.ndarray float32 (C_density, 80, 80, 80)；按 producer 通道顺序构造的最终 ZYX 输入。
        density_input = build_density_channels(
            exp_raw=exp_crop,
            sim_raw=sim_crop,
            config=self.density_config,
            receptor_mask=hardmask,
        )

        # dict[str, Any]；单 BOX NumPy 字段映射；固定形状数组稍后转 tensor，身份字段保留 Python 值。
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
            # np.ndarray bool (N_A,)；与局部原子表逐项对齐的 binding 标签，随后仅核心原子写入 voxel_label。
            binding_selected = structure["binding_atom"][selected_idx]
            sample["voxel_label"] = build_hardmask_from_atom_coordinates(
                atom_coord_local_voxel=atom_coordinates["atom_coord_local_voxel"],
                atom_is_in_core_box=core_mask & binding_selected,
                box_shape_zyx=box_shape_zyx,
            ).astype(bool, copy=False)
            union_mask = self._load_ligand_union(
                request.pdb_id,
                full_shape,
                voxel_size,
                full_origin,
            )
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
                if np.any(ligand_distance < 0):
                    raise ValueError(f"{request.pdb_id}: ligand_dist.npy 的 80³ 裁块包含负距离。")
                sample["ligand_inverse_distance_target"] = np.where(
                    np.isfinite(ligand_distance), 1.0 / (1.0 + ligand_distance), 0.0
                ).astype(np.float32, copy=False)

        if self.stage1_model_name.startswith("Find"):
            sample.update(atom_coordinates)
            sample["atom_global_indices"] = selected_idx.astype(np.int64, copy=False)
            sample["atom_feat"] = build_atom_features(structure["feat"], selected_idx)
            sample["atom_is_backbone"] = structure["is_backbone"][selected_idx].astype(bool, copy=False)
            sample["atom_is_in_core_box"] = core_mask.astype(bool, copy=False)
            if request.require_targets:
                sample["atom_label"] = structure["binding_atom"][selected_idx].astype(bool, copy=False)
        if self.enable_random_rotation:
            sample = _apply_synced_rotation(sample)
        return _to_tensor_sample(sample)

    def __getitem__(self, index: int) -> dict[str, Any]:
        """按请求源位置物化一个 Stage1 单 BOX tensor 字典。

        输入参数:
            - index: int；请求源中的零基位置。

        返回值:
            - sample: dict[str, Any]；与 ``materialize_request`` 相同的 producer 专属字段；失败的资产或契约错误直接向调用方抛出。
        """
        return self.materialize_request(self.request_source[index])

    def materialize_request(
        self,
        request: ResolvedStage1Crop,
    ) -> dict[str, Any]:
        """物化一个调用方已经解析完成的 80³ 请求，不重新随机生成起点。

        输入参数:
            - request: ResolvedStage1Crop；PDB identity、完整图 ZYX BOX 起点、监督开关和 role 已由共享 resolver 冻结。

        返回值:
            - sample: dict[str, Any]；与 ``dataset[index]`` 相同的单 BOX tensor 字典；``require_targets`` 只控制监督字段是否附加，不改变密度输入和局部受体字段的 producer 选择。

        失败语义:
            - ``request`` 不是 ResolvedStage1Crop 时抛出 ``TypeError``；完整图、几何或裁块契约失败时由下游读取函数抛出异常。
        """
        if not isinstance(request, ResolvedStage1Crop):
            raise TypeError("request 必须是 ResolvedStage1Crop。")
        return self._materialize(request)

    def full_map_context(
        self,
        pdb_id: str,
    ) -> tuple[tuple[int, int, int], np.ndarray, np.ndarray, np.ndarray]:
        """返回 full_map 滑窗所需的完整 exp 几何和受体坐标。

        输入参数:
            - pdb_id: str；当前 PDB identity；首尾空白会去除并转为小写，以匹配正式资产目录。

        返回值:
            - shape_zyx: tuple[int, int, int]；exp 完整图的 ``(D_full, H_full, W_full)`` ZYX 形状。
            - voxel_size_xyz: np.ndarray float32 ``(3,)``；世界 XYZ 体素尺寸，单位为 Å。
            - origin_xyz: np.ndarray float32 ``(3,)``；完整图 voxel-grid corner 的世界 XYZ 原点，单位为 Å。
            - receptor_coord_xyz: np.ndarray float32 ``(N_receptor, 3)``；完整受体表的世界 XYZ 坐标，单位为 Å。

        缓存语义:
            - 数据来自与 ``materialize_request`` 相同的 Dataset 实例级 LRU; 命中后可由物化线程共享. 首次并发 miss 可能重复打开同一 mmap, 但缓存状态保持线程安全.
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
