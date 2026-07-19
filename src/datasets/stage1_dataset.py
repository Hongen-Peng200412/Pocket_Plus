# -*- coding: utf-8 -*-
"""AdaLigand Stage1 的统一 80³ Dataset 与 materializer。"""

from __future__ import annotations

import random
from collections import OrderedDict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

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


_STAGE1_MODEL_NAMES = {"Find_0", "Find_1", "unet_c1"}


class _ByteLruCache:
    """按数组真实字节数限制的 worker-local LRU 缓存。"""

    def __init__(self, max_bytes: int) -> None:
        self.max_bytes = max(0, int(max_bytes))
        self.current_bytes = 0
        self.values: OrderedDict[str, tuple[dict[str, np.ndarray], int]] = OrderedDict()

    @staticmethod
    def _size_bytes(value: Mapping[str, np.ndarray]) -> int:
        return sum(int(array.nbytes) for array in value.values() if isinstance(array, np.ndarray))

    def get(self, key: str) -> dict[str, np.ndarray] | None:
        item = self.values.pop(key, None)
        if item is None:
            return None
        self.values[key] = item
        return item[0]

    def put(self, key: str, value: dict[str, np.ndarray]) -> None:
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
    """读取 Stage E ``grid/voxel_size/origin`` 并返回独立 CPU 数组。"""

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
    """从已验证边界的整图裁出真实 80³ 数组。"""

    z0, y0, x0 = (int(value) for value in start_zyx)
    crop = array[z0 : z0 + 80, y0 : y0 + 80, x0 : x0 + 80]
    if crop.shape != STAGE1_BOX_SHAPE_ZYX:
        raise RuntimeError(f"Stage1 crop 必须恰为 80³，实际 {crop.shape}。")
    return np.ascontiguousarray(crop)


def _rotate_zyx_coordinates(
    coord_zyx: np.ndarray,
    box_shape_zyx: np.ndarray,
    axis1: int,
    axis2: int,
    k: int,
) -> np.ndarray:
    """按 ``np.rot90`` 语义旋转 corner 连续 ZYX 坐标。"""

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
    """对 density、监督与 Find 原子坐标执行同一次随机 90° 旋转。"""

    axis1, axis2 = np.random.choice([0, 1, 2], size=2, replace=False).tolist()
    k = random.randint(0, 3)
    if k == 0:
        return sample
    box_shape = sample["box_shape_zyx"]
    voxel_size = sample["voxel_size_world"]
    if not (int(box_shape[0]) == int(box_shape[1]) == int(box_shape[2])):
        raise ValueError("Stage1 90° 旋转只支持立方体 BOX。")
    if not np.allclose(voxel_size, float(voxel_size[0]), rtol=3.0e-2, atol=3.0e-4):
        raise ValueError(f"Stage1 90° 旋转要求近似各向同性 voxel，实际 {voxel_size.tolist()}。")

    sample["density_input"] = np.rot90(
        sample["density_input"], k=k, axes=(axis1 + 1, axis2 + 1)
    ).copy()
    sample["hardmask"] = np.rot90(sample["hardmask"], k=k, axes=(axis1, axis2)).copy()
    for field_name in ("voxel_label", "ligand_area_target"):
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
    box_center_world = sample["box_origin_world"] + 0.5 * box_shape_xyz * voxel_size
    sample["atom_coord_world"] = (
        sample["atom_coord_centered_world"] + box_center_world[None, :]
    ).astype(np.float32, copy=False)
    return sample


def _to_tensor_sample(sample: dict[str, Any]) -> dict[str, Any]:
    """按 BOX-level 契约把 NumPy 数组转换为显式 dtype 的 torch tensor。"""

    dtype_by_field = {
        "box_start_zyx": torch.int32,
        "box_shape_zyx": torch.int64,
        "box_origin_world": torch.float32,
        "voxel_size_world": torch.float32,
        "density_input": torch.float32,
        "hardmask": torch.bool,
        "ligand_area_target": torch.bool,
        "voxel_label": torch.bool,
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
    """统一 materialize train/val/full_map/centered 四类 80³ 请求。

    参数:
        all_data_path: AdaLigand A—G 正式根目录；读取 ``density/parse/labels``。
        split_file: 训练池目录、``validation_selection.npz``、普通冻结请求文件，或
            推理装配层传入的非空 ``ResolvedStage1Crop`` 内存序列。
        mode: ``train``、``val``、``full_map`` 或 ``centered``。
        stage1_model_name: ``Find_0``、``Find_1`` 或 ``unet_c1``。
        box_pool_root: ``stage1_preparation/box_pool``；验证 selection 需要。
        density_channel_config: clip/fit 参数与显式通道名。
        atom_buffer_radius: Find 直接加载 core+8 Å，正式值必须为 8.0。
        request_seed: train 每 epoch 请求选择的固定基准 seed。
        cache_max_bytes: 每个 DataLoader worker 的轻量 receptor/label LRU 字节上限。
        enable_random_rotation: 仅 train 生效的同步 90° 旋转。

    注意:
        Dataset 不补零、不跳样本、不现场过滤；请求起点必须已经由同一个 resolver
        合法化。``unet_c1`` 仍读取 receptor 构造 auxiliary target，但不返回 atom 输入表。
    """

    collate_fn = Stage1BatchCollator()

    def __init__(
        self,
        all_data_path: str,
        split_file: str | Sequence[str] | Sequence[ResolvedStage1Crop],
        mode: str,
        stage1_model_name: str,
        box_pool_root: str | None,
        density_channel_config: Mapping[str, Any],
        atom_buffer_radius: float = 8.0,
        request_seed: int = 3407,
        cache_max_bytes: int = 536_870_912,
        enable_random_rotation: bool = True,
        name: str = "stage1",
        split_train: str | Sequence[str] | None = None,
        split_val: str | Sequence[str] | None = None,
        class_names: Sequence[str] = ("background", "foreground"),
    ) -> None:
        super().__init__()
        del split_train, split_val
        self.name = str(name)
        self.root = Path(all_data_path)
        self.mode = str(mode).lower()
        self.stage1_model_name = str(stage1_model_name)
        self.class_names = tuple(str(value) for value in class_names)
        if self.mode not in {"train", "val", "validation", "full_map", "centered"}:
            raise ValueError(f"未知 Stage1 Dataset mode: {mode!r}。")
        if self.stage1_model_name not in _STAGE1_MODEL_NAMES:
            raise ValueError(f"stage1_model_name 必须属于 {_STAGE1_MODEL_NAMES}。")
        if self.class_names != ("background", "foreground"):
            raise ValueError("AdaLigand Stage1 当前只接受 [background, foreground] 二分类顺序。")
        if float(atom_buffer_radius) != 8.0:
            raise ValueError("AdaLigand Find Dataset 的 atom_buffer_radius 固定为 8.0 Å。")
        self.atom_buffer_radius = 8.0
        self.enable_random_rotation = bool(enable_random_rotation and self.mode == "train")
        if isinstance(split_file, (list, tuple)) and all(
            isinstance(request, ResolvedStage1Crop) for request in split_file
        ):
            if not split_file:
                raise ValueError("内存 Stage1 请求序列不能为空。")
            self.request_source = tuple(split_file)
        else:
            self.request_source = build_request_source(
                split_file=split_file,
                mode=self.mode,
                box_pool_root=box_pool_root,
                seed=int(request_seed),
            )
        channel_cfg = dict(density_channel_config)
        self.density_config = DensityChannelConfig(
            clip_percentile=tuple(float(value) for value in channel_cfg["clip_percentile"]),
            fit_mask_percentile=float(channel_cfg["fit_mask_percentile"]),
            enabled_channels=[str(value) for value in channel_cfg["enabled_channels"]],
        )
        resolved_channels = (
            list(ALL_CHANNEL_NAMES)
            if "all" in [value.lower() for value in self.density_config.enabled_channels]
            else list(self.density_config.enabled_channels)
        )
        if self.stage1_model_name.startswith("Find") and resolved_channels != list(ALL_CHANNEL_NAMES):
            raise ValueError("Find_0/Find_1 必须按权威顺序启用完整 56D ALL density channels。")
        if self.stage1_model_name == "unet_c1" and resolved_channels != ["exp_clipnorm_nopost"]:
            raise ValueError("unet_c1 density 输入必须恰为 exp_clipnorm_nopost。")
        self.resolved_density_channels = tuple(resolved_channels)
        # 同一个按字节受限的 worker-local cache 同时保存轻量 receptor 表与最近使用的
        # 原始整图。完整图推理会连续消费同一 PDB 的数百个窗口，因此必须避免每个
        # 窗口重新解压 exp/sim；训练的随机 PDB 访问仍由同一上限自然淘汰大数组。
        self._source_cache = _ByteLruCache(cache_max_bytes)

    def set_epoch(self, epoch: int) -> None:
        """通知动态训练请求源切换 epoch；固定请求源保持不变。"""

        set_epoch = getattr(self.request_source, "set_epoch", None)
        if callable(set_epoch):
            set_epoch(int(epoch))

    def __len__(self) -> int:
        return len(self.request_source)

    def describe_index(self, index: int) -> str:
        request = self.request_source[index]
        return (
            f"pdb_id={request.pdb_id}, role={request.role}, "
            f"start_zyx={request.box_start_zyx}, occurrence_id={request.occurrence_id}"
        )

    def _load_structure(self, pdb_id: str, require_targets: bool) -> dict[str, np.ndarray]:
        """读取并缓存 receptor 49D 基础表及可选 binding label。"""

        cache_key = f"{pdb_id}|targets={int(require_targets)}"
        cached = self._source_cache.get(cache_key)
        if cached is not None:
            return cached
        receptor_path = self.root / "parse" / pdb_id / "receptor_tokens.npz"
        with np.load(receptor_path, allow_pickle=False) as data:
            if "coords" not in data or "feat" not in data:
                raise KeyError(f"{receptor_path} 缺少 coords 或 feat。")
            coords = np.asarray(data["coords"], dtype=np.float32)
            feat = np.asarray(data["feat"], dtype=np.float32)
        if coords.ndim != 2 or coords.shape[1] != 3 or feat.shape != (coords.shape[0], 49):
            raise ValueError(f"{receptor_path}: coords/feat 必须为 [N,3]/[N,49]。")
        if not np.isfinite(coords).all() or not np.isfinite(feat).all():
            raise ValueError(f"{receptor_path}: coords/feat 包含 NaN/Inf。")
        structure = {"coords": coords, "feat": feat}
        if require_targets:
            label_path = self.root / "labels" / pdb_id / "atom_labels.npz"
            with np.load(label_path, allow_pickle=False) as data:
                if "binding_atom" not in data:
                    raise KeyError(f"{label_path} 缺少 binding_atom。")
                binding_atom = np.asarray(data["binding_atom"], dtype=bool)
            if binding_atom.shape != (coords.shape[0],):
                raise ValueError(f"{label_path}: binding_atom 必须与 receptor 行数一致。")
            structure["binding_atom"] = binding_atom
        self._source_cache.put(cache_key, structure)
        return structure

    def _load_density_grid(
        self,
        pdb_id: str,
        grid_name: str,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """读取并按真实字节数缓存一个 PDB 的 exp/sim 原始整图。

        参数:
            pdb_id: 当前 PDB identity。
            grid_name: ``exp`` 或 ``sim``；缓存只保存原始 float32 网格与几何，
                不保存 56D 派生通道。

        返回:
            ``(grid_zyx, voxel_size_xyz, origin_xyz)``。调用方只能裁剪读取，不得
            原地修改这些 worker-local 共享数组。
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
        """读取并缓存 schema-v3 occurrence union mask，保持完整图 bool 语义。"""

        expected_shape = tuple(int(value) for value in expected_shape_zyx)
        cache_key = f"{pdb_id}|ligand_union"
        cached = self._source_cache.get(cache_key)
        if cached is None:
            ligand_path = self.root / "density" / pdb_id / "ligand_area.npz"
            with np.load(ligand_path, allow_pickle=False) as data:
                if "union_mask" not in data:
                    raise KeyError(f"{ligand_path} 缺少 union_mask。")
                union_mask = np.asarray(data["union_mask"], dtype=bool)
            if union_mask.shape != (1, *expected_shape):
                raise ValueError(f"{ligand_path}: union_mask 与 exp grid 形状不一致。")
            cached = {"union_mask": union_mask}
            self._source_cache.put(cache_key, cached)
        union_mask = cached["union_mask"]
        if union_mask.shape != (1, *expected_shape):
            raise ValueError(f"{pdb_id}: 缓存的 union_mask 与当前 exp grid 形状不一致。")
        return union_mask

    def _materialize(self, request: ResolvedStage1Crop) -> dict[str, Any]:
        """把一个已解析请求物化为模型专属单样本字段。"""

        exp_grid, voxel_size, full_origin = self._load_density_grid(
            request.pdb_id, "exp"
        )
        full_shape = np.asarray(exp_grid.shape, dtype=np.int64)
        resolved_start = resolve_stage1_start(request.box_start_zyx, full_shape)
        if resolved_start != request.box_start_zyx:
            raise ValueError(
                "Stage1 Dataset 只消费预先解析的起点："
                f"request={request.box_start_zyx}, resolved={resolved_start}, pdb={request.pdb_id}。"
            )
        start_zyx = np.asarray(resolved_start, dtype=np.int32)
        box_shape_zyx = np.asarray(STAGE1_BOX_SHAPE_ZYX, dtype=np.int64)
        start_xyz = start_zyx[[2, 1, 0]].astype(np.float32)
        box_origin = (full_origin + start_xyz * voxel_size).astype(np.float32, copy=False)

        structure = self._load_structure(request.pdb_id, request.require_targets)
        selection = select_atoms_for_box(
            atom_coords_world=structure["coords"],
            box_origin_world=box_origin,
            voxel_size_world=voxel_size,
            box_shape_zyx=box_shape_zyx,
            buffer_radius=self.atom_buffer_radius,
        )
        selected_idx = selection["selected_idx"]
        atom_coordinates = build_atom_coordinates(
            atom_coords_world=structure["coords"],
            selected_idx=selected_idx,
            box_origin_world=box_origin,
            voxel_size_world=voxel_size,
            box_shape_zyx=box_shape_zyx,
        )
        core_mask = selection["atom_is_in_core_box"]
        hardmask = build_hardmask_from_atom_coordinates(
            atom_coord_local_voxel=atom_coordinates["atom_coord_local_voxel"],
            atom_is_in_core_box=core_mask,
            box_shape_zyx=box_shape_zyx,
        ).astype(bool, copy=False)

        exp_crop = _crop_80(exp_grid, start_zyx)
        if self.stage1_model_name.startswith("Find"):
            sim_grid, sim_voxel_size, sim_origin = self._load_density_grid(
                request.pdb_id, "sim"
            )
            if sim_grid.shape != exp_grid.shape or not np.array_equal(sim_voxel_size, voxel_size) or not np.array_equal(sim_origin, full_origin):
                raise ValueError(f"{request.pdb_id}: exp/sim 的 shape、voxel_size、origin 必须完全一致。")
            sim_crop = _crop_80(sim_grid, start_zyx)
        else:
            sim_crop = None
        density_input = build_density_channels(
            exp_raw=exp_crop,
            sim_raw=sim_crop,
            config=self.density_config,
            receptor_mask=hardmask,
        )

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
            binding_selected = structure["binding_atom"][selected_idx]
            sample["voxel_label"] = build_hardmask_from_atom_coordinates(
                atom_coord_local_voxel=atom_coordinates["atom_coord_local_voxel"],
                atom_is_in_core_box=core_mask & binding_selected,
                box_shape_zyx=box_shape_zyx,
            ).astype(bool, copy=False)
            union_mask = self._load_ligand_union(request.pdb_id, full_shape)
            sample["ligand_area_target"] = _crop_80(union_mask[0], start_zyx).astype(
                bool, copy=False
            )

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
        return self.materialize_request(self.request_source[index])

    def materialize_request(
        self,
        request: ResolvedStage1Crop,
    ) -> dict[str, Any]:
        """物化一个调用方已经解析完成的 80³ 请求。

        参数:
            request: :class:`ResolvedStage1Crop`，其起点、PDB 身份、监督开关和
                请求角色已经由共享 resolver 冻结。完整图和居中推理用本入口复用
                与训练完全相同的密度通道、core+8 Å 原子选择、hardmask 与坐标逻辑。

        返回:
            与 ``dataset[index]`` 完全相同的单样本字典。``require_targets`` 只决定
            是否附加监督字段，不改变任何模型输入字段。
        """
        if not isinstance(request, ResolvedStage1Crop):
            raise TypeError("request 必须是 ResolvedStage1Crop。")
        return self._materialize(request)

    def full_map_context(
        self,
        pdb_id: str,
    ) -> tuple[tuple[int, int, int], np.ndarray, np.ndarray, np.ndarray]:
        """返回完整图滑窗所需的 shape、几何与 receptor 世界坐标。

        参数:
            pdb_id: 当前小写或可规范化为小写的 PDB 身份。

        返回:
            ``(shape_zyx, voxel_size_xyz, origin_xyz, receptor_coord_xyz)``。
            数据来自与 :meth:`materialize_request` 相同的 worker-local 有界缓存，
            因而装配层不会为读取 shape/hardmask 额外解压一次完整 exp 或 receptor。
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
