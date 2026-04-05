# -*- coding: utf-8 -*-
"""
=============================================================================
BoxPointDataset
=============================================================================
本文件实现 stage1 使用的“体素 + 点云”联合数据集。

核心职责:
1. 从一个 BOX 样本中读取 voxel 特征、voxel 标签和几何元信息。
2. 从对应 pdb 的 `atoms.npz + labels.npz` 中读取 atom 坐标、atom 特征与标签。
3. 按 BOX 范围把 atom 拆成 core 原子和 buffer 原子，并构造三套坐标表示。
4. 生成 voxel_valid_mask 与 atom_valid_mask（atom 监督需同时满足 core 内且避开裁边区域）。
5. 在训练模式下，对 voxel 与 atom 同步施加 90 度旋转增强。

坐标与轴顺序约定:
    - 世界坐标统一使用 `(x, y, z)`。
    - 体素数组的空间轴统一使用 `(Z, Y, X)`，也就是 numpy / torch 中常见的 `(D, H, W)`。
    - `atom_coord_local_voxel` 的数值语义是“连续 voxel 坐标(角点语义)”，字段顺序仍然保持 `(x, y, z)`。
    - 若 atom 恰好落在某个 voxel 中心，则该字段的数值会是 `(index_x + 0.5, index_y + 0.5, index_z + 0.5)`。
    - 因此后续若要送入 `grid_sample(align_corners=True)`，必须先减去 `0.5`，把角点语义转成以 voxel 中心索引为基准的采样语义。

输出样本约定:
    - voxel 侧返回 `voxel_grid / voxel_label / hardmask / voxel_valid_mask` 等字段。
    - atom 侧返回 `atom_coord_world / atom_coord_local_voxel / atom_coord_centered_world / atom_feat / atom_label / atom_is_in_core_box / atom_valid_mask` 等字段。
    - 元信息保留 `sample_name / pdb_id / class_name / instance_id / is_center_box`。
=============================================================================
"""
from __future__ import annotations

import json
import random
import re
from collections import OrderedDict
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from .box_geometry import (
    build_atom_coordinates,
    build_atom_features,
    build_atom_valid_mask,
    build_hardmask_from_atom_coordinates,
    build_voxel_valid_mask,
    select_atoms_for_box,
)
from .box_point_collate import box_point_collate
from .box_sample_builder import build_box_point_numpy_sample, to_torch_sample


_SAMPLE_NAME_PATTERN = re.compile(
    r"^(?P<pdb_id>.+)_(?P<instance_id>-?\d+)_(?P<rxx>-?\d+)_(?P<ryy>-?\d+)_(?P<rzz>-?\d+)(?P<center>_C)?$"
)
 

class BoxPointDataset(Dataset):
    collate_fn = staticmethod(box_point_collate)

    def __init__(
        self,
        all_data_path: Optional[str] = None,
        split_file: list[str] | tuple[str, ...] | None = None,
        mode: Optional[str] = None,
        sample_root_path: Optional[str] = None,
        data_folder_names: list[str] | None = None,
        class_folder_names: list[str] | None = None,
        class_mapping: list[int] | None = None,
        atom_buffer_radius: float = 4.0,
        cache_size: int = 128,
        valid_crop_margin: int = 2,
        enable_random_rotation: bool = True,
        **kwargs: Any,
    ) -> None:
        """
            功能概述:
                - 输入一个 BOX 样本名，读取该 BOX 的体素特征、体素标签和几何元信息。
                - 再根据样本名解析出的 `pdb_id`，读取整条结构上的 atom 坐标、atom 特征、atom 标签。
                - 最终把一个 BOX 视角下需要的 voxel 信息和 atom 信息打包为同一个 sample dict。

            输入参数 (Input Parameters):
                - sample_root_path: str, 原始结构缓存目录。通常要求 `sample_root_path / pdb_id /` 下存在 `atoms.npz` 与 `labels.npz`。


                - all_data_path: str, BOX 数据所在根目录。目录结构为 all_data_path / data_folder_names[0] / class_folder_names[0] / 9dic_2_0_17_20(pdbid_instanceid_rxx_ryy_rzz_centerornot)
                - data_folder_names: list[str], BOX 文件夹名称列表。含 `"label"` 的目录会被视为标签，含 `"emdb"` 的目录会执行 z-score 归一化，其余目录都作为 voxel 特征通道拼接; 当前为 ["emdb_BOX", "pdb_feature_BOX", "pdb_label_BOX", "emdb_npz"]
                - class_folder_names: list[str], 类别目录名称列表，与 `split_file` 的顺序严格一一对应, 当前为 ["metal_ion", "peptide", "nucleic", "small_molecule"]
                - class_mapping: list[int] 或 None, 可选的标签类别重映射表，同时作用于 voxel 标签和 atom 标签。



                - split_file: list[str], 样本划分的.json文件. split_file[k] 代表 class_folder_name[k] 对应的划分.json路径, 因而它的长度必须和 class_folder_name 相同, 并且顺序和含义一致！！ split_file[k]读取后形如 ["9dic_2_0_17_20", ...]
                - mode: str, 训练 / 验证 / 测试模式，用于控制 `is_train` 与数据增强开关。


                - atom_buffer_radius: float, 在 core box 之外额外保留的原子缓冲半径，单位与世界坐标一致。
                - valid_crop_margin: int, voxel_valid_mask 默认向内裁掉的边带宽度，单位是 voxel 个数。
                - cache_size: int, 每个 dataloader worker 内部 LRU cache 的最大容量。
                - enable_random_rotation: bool, 是否在训练模式下启用 voxel / atom 同步 90 度旋转。


            输出样本 (Return Sample):
                - 返回一个 `dict[str, Any]`，其中既包含 voxel 侧字段，也包含 atom 侧字段。
                - 所有 numpy 字段会在 `__getitem__` 末尾被统一转成 torch tensor，元信息字段保持 Python 原生类型。

            坐标约定:
                - 世界坐标始终使用 `(x, y, z)`。
                - voxel grid 的空间轴始终使用 `(Z, Y, X)`。
                - `atom_coord_local_voxel` 的数值语义是连续 voxel 坐标(角点语义)，但字段顺序仍然保持 `(x, y, z)`。
                - 若后续要做 `grid_sample(align_corners=True)`，需要先把该坐标减去 `0.5` 再做归一化。
        """
        super().__init__()
        del kwargs
        if all_data_path is None:
            raise ValueError("all_data_path is empty")
        if sample_root_path is None:
            raise ValueError("sample_root_path is empty")
        if split_file is None:
            raise ValueError("split_file is empty")
        if mode is None:
            raise ValueError("mode is empty")

        self.all_data_path = Path(all_data_path)                    # BOX 根目录
        self.sample_root_path = Path(sample_root_path)              # atoms.npz / labels.npz 根目录
        self.class_mapping = class_mapping                          # voxel 标签类别重映射表
        self.atom_buffer_radius = float(atom_buffer_radius)         # atom buffer 半径, 单位=世界坐标
        self.cache_size = int(cache_size)                           # 每个 worker 的 cache 容量
        self.valid_crop_margin = int(valid_crop_margin)             # 边界裁边宽度, 单位=voxel

        self.data_folder_names = list(data_folder_names)            # ["emdb_BOX", "pdb_feature_BOX", "pdb_label_BOX"]
        self.class_folder_names = list(class_folder_names)          # ["metal_ion", "peptide", "nucleic", "small_molecule"]
        self.collate_fn = box_point_collate                         # 让外部 DataLoader 直接复用项目内 collate

        if self.valid_crop_margin < 0:
            raise ValueError("valid_crop_margin must be >= 0")

        self._validate_folder_layout()
        self.is_train = self._parse_mode(mode)
        self.enable_random_rotation = bool(enable_random_rotation and self.is_train)

        sample_name_lists = self._load_split_lists(split_file)
        self.total_sample = self._build_sample_index(sample_name_lists)

        # str 为pdb_id
        self.structure_cache: OrderedDict[str, dict[str, np.ndarray]] = OrderedDict()  # 每个 worker 各自持有一个轻量 LRU cache，避免重复反复读取 atoms.npz / labels.npz。

        # 首次读取 atoms.npz 时记录原始 atom feature 维度，并对后续样本保持一致性校验。
        self.atom_raw_feature_dim: Optional[int] = None

    def _validate_folder_layout(self) -> None:
        """
        see me 校验最关键的目录约束：
            - 所有 `emdb` 文件夹必须排在最前面。
            - `label` 文件夹必须且只允许出现一个。

        这样做的原因是:
            - 当前数据契约默认把密度类通道放在前部，保持与既有训练配置的输入通道顺序一致；
            - `label` 文件夹仍然要求唯一，避免 voxel 标签来源不明确。
        """
        non_emdb_seen = False
        label_folder_count = 0
        for folder_name in self.data_folder_names:
            if "label" in folder_name:
                label_folder_count += 1
            if "emdb" in folder_name:
                if non_emdb_seen:
                    raise ValueError("含有 'emdb' 的数据文件夹必须排在所有其他特征文件夹之前，" f"当前配置为: {self.data_folder_names}")
            else:
                non_emdb_seen = True

        if label_folder_count != 1:
            raise ValueError("第一版 BoxPointDataset 约定必须且只存在一个 label 文件夹，" f"当前统计到的数量为: {label_folder_count}")

    def _parse_mode(self, mode: str) -> bool:
        """
        将配置里的 mode 归一成 `is_train`。

        返回:
            - `True`: 训练模式，允许启用随机旋转增强。
            - `False`: 验证 / 测试模式，不做随机增强。
        """
        mode_lower = str(mode).lower()
        if mode_lower in {"train", "fit"}:
            return True
        if mode_lower in {"val", "valid", "validation", "test", "evaluate"}:
            return False
        raise ValueError(f"Unknown mode: {mode}")

    def __len__(self) -> int:
        """
        返回拍平后的样本总数。
        """
        return len(self.total_sample)







    # ------------------------------------------------- 处理样本名 -----------------------------------------------
    def _load_split_lists(self, split_file: Any) -> list[list[str]]:
        """
            读取每个类别对应的 json 划分文件。

            输入:
                - see me split_file: 应是一个可迭代对象，长度必须与 `class_folder_names` 完全一致且意义对齐: split_file[k] 对应 class_folder_names[k] 的划分文件

            输出:
                - sample_name_lists: list[list[str]], 其中外层维度对应类别，内层是该类别下的样本名列表 (比如第一个元素的sample_name_lists[i]是 list[str], 是关于"metal_ion"的BOX文件名)
        """
        if isinstance(split_file, (str, bytes)):
            raise TypeError("split_file must be an iterable of json paths, not a single string")
        try:
            split_paths = list(split_file)
        except TypeError as exc:
            raise TypeError("split_file must be an iterable of json paths") from exc

        if len(split_paths) != len(self.class_folder_names):
            raise ValueError("split_file 的长度必须与 class_folder_names 一一对应，" f"当前分别为 {len(split_paths)} 和 {len(self.class_folder_names)}")

        sample_name_lists: list[list[str]] = []
        for split_path in split_paths:
            with open(split_path, "r", encoding="utf-8") as file_obj:
                sample_names = json.load(file_obj)
            if not isinstance(sample_names, list):
                raise TypeError(f"split json must contain list[str], but got: {type(sample_names)}")
            sample_name_lists.append(sample_names)
        return sample_name_lists

    def _build_sample_index(self, sample_name_lists: list[list[str]]) -> list[dict[str, Any]]:
        """
        将 `[class][sample]` 拍平成线性样本索引, sample_name_lists 是前一个函数的返回值。

        返回结果 `self.total_sample` 是 list[dict[str, Any]], 其每一项都显式保存:
            - `class_name`: 类别目录名(如"metal_ion")。
            - `sample_name`: 具体样本文件名（不含 `.npz` 后缀）。
        """
        total_sample: list[dict[str, Any]] = []
        for class_idx, sample_names in enumerate(sample_name_lists):
            class_name = self.class_folder_names[class_idx]
            for sample_name in sample_names:
                total_sample.append(
                    {
                        "class_name": class_name,
                        "sample_name": sample_name,
                    }
                )
        return total_sample








    # ------------------------------------------------- 加载体素(BOX)数据 -------------------------------------------------
    def _parse_sample_name(self, sample_name: str) -> dict[str, Any]:
        """
        从BOX的样本名中解析 `pdb_id / instance_id / is_center_box`
        """
        matched = _SAMPLE_NAME_PATTERN.match(sample_name)
        if matched is None:
            raise ValueError(f"sample_name format is invalid: {sample_name}")

        return {
            "pdb_id": matched.group("pdb_id").lower(),
            "instance_id": int(matched.group("instance_id")),
            "rxx": int(matched.group("rxx")),
            "ryy": int(matched.group("ryy")),
            "rzz": int(matched.group("rzz")),
            "is_center_box": matched.group("center") is not None,
        }

    def _extract_box_meta(
        self, npz_file: Any, grid_shape_zyx: tuple[int, ...]
    ) -> dict[str, np.ndarray]:
        """
        从单个 BOX 的 npz 中读取几何元信息。

        输入:
            - `origin`: 世界坐标系下 BOX 左下近角点，顺序 `(x, y, z)`。
            - `voxel_size`: 每个 voxel 在世界坐标中的尺寸，顺序 `(x, y, z)`。
            - `grid_shape_zyx`: 当前网格数组的空间形状，顺序 `(Z, Y, X)`。

        输出:
            - `box_origin_world`: (3,), 世界坐标系下 BOX 左下近角点，顺序 `(x, y, z)`。
            - `voxel_size_world`: (3,), 每个 voxel 在世界坐标中的尺寸，顺序 `(x, y, z)`。
            - `x_range`: (2,), x 轴范围，顺序 `(min, max)`。
            - `y_range`: (2,), y 轴范围，顺序 `(min, max)`。
            - `z_range`: (2,), z 轴范围，顺序 `(min, max)`。

        说明:
            - 若 npz 已显式提供 `x_range / y_range / z_range`，则直接使用, 否则根据 `origin + shape * voxel_size` 反推出各轴范围。
            - 注意 `box_shape_zyx` 与 `box_shape_xyz` 只是轴顺序不同.
        """
        box_origin_world = np.asarray(npz_file["origin"], dtype=np.float32).reshape(3)
        voxel_size_world = np.asarray(npz_file["voxel_size"], dtype=np.float32).reshape(3)
        box_shape_zyx = np.asarray(grid_shape_zyx[-3:], dtype=np.int64)

        if "x_range" in npz_file and "y_range" in npz_file and "z_range" in npz_file:
            x_range = np.asarray(npz_file["x_range"], dtype=np.float32).reshape(2)
            y_range = np.asarray(npz_file["y_range"], dtype=np.float32).reshape(2)
            z_range = np.asarray(npz_file["z_range"], dtype=np.float32).reshape(2)
        else:
            # `grid` 的空间轴是 Z/Y/X，而世界坐标下推导边界时要切回 X/Y/Z 顺序。
            box_shape_xyz = box_shape_zyx[[2, 1, 0]].astype(np.float32)
            box_max_world = box_origin_world + box_shape_xyz * voxel_size_world
            x_range = np.asarray([box_origin_world[0], box_max_world[0]], dtype=np.float32)
            y_range = np.asarray([box_origin_world[1], box_max_world[1]], dtype=np.float32)
            z_range = np.asarray([box_origin_world[2], box_max_world[2]], dtype=np.float32)

        return {
            "box_origin_world": box_origin_world,
            "voxel_size_world": voxel_size_world,
            "x_range": x_range,
            "y_range": y_range,
            "z_range": z_range,
        }

    def _parse_voxel_label(self, grid: np.ndarray) -> np.ndarray:
        """
        将 label grid 统一整理成 `(D, H, W)` 的 int64 语义，兼容两种常见情况:
            - `(1, D, H, W)`: 旧逻辑里常见的单通道标签。
            - `(D, H, W)`: 已经 squeeze 过的标签。
        """
        if grid.ndim == 4:
            if grid.shape[0] != 1:
                raise ValueError(f"label grid should be (1, D, H, W), but got: {grid.shape}")
            grid = grid[0]
        if grid.ndim != 3:
            raise ValueError(f"label grid should be 3D after squeeze, but got: {grid.shape}")
        return np.rint(grid).astype(np.int64, copy=False)

    def _zscore_emdb_grid(self, grid: np.ndarray) -> np.ndarray:
        """
        对 EMDB 通道做 z-score 归一化: 默认把 EMDB 通道看成密度值或差图，用整体均值和整体标准差做标准化
        """
        grid_mean = float(np.mean(grid))
        grid_std = float(np.std(grid))
        return ((grid - grid_mean) / (grid_std + 1e-8)).astype(np.float32, copy=False)

    def _apply_class_mapping(self, label: np.ndarray, mapping: list[int]) -> np.ndarray:
        """
        将原始类别 ID 映射成新的类别 ID。

        例如 `mapping=[0, 1, 2, 2]` 时，表示把原始类别 3 合并到新类别 2。
        """
        mapped_label = np.zeros_like(label, dtype=np.int64)
        for old_class_id, new_class_id in enumerate(mapping):
            mapped_label[label == old_class_id] = int(new_class_id)
        return mapped_label

    def _load_box_npz_triplet(self, class_name: str, sample_name: str) -> dict[str, np.ndarray]:
        """
        读取一个 BOX 的 voxel 数据与几何元信息. class_name 形如"metal_ion", sample_name 形如"9dic_2_0_17_20"(pdbid_instanceid_rxx_ryy_rzz_centerornot)

        对于 class_name 下的样本名为 sample_name 的这个特定样本，遍历 `self.data_folder_names` 指定的多个目录，并按以下规则组织:
            - 含 `"label"` 的目录读成 voxel 标签，最终统一为 `(D, H, W)` 的 int64。
            - 其余目录都视为 voxel 特征，沿 channel 维拼接成 `(C, D, H, W)`。
            - 含 `"emdb"` 的特征目录会额外执行 z-score 归一化。

        返回字段:
            - `voxel_grid`: np.ndarray, `(C, D, H, W)`, float32。
            - `voxel_label`: np.ndarray, `(D, H, W)`, int64。
            - `box_origin_world`: np.ndarray, `(3,)`, 世界坐标系下 BOX 左下近角点，顺序 (x, y, z)。
            - `voxel_size_world`: np.ndarray, `(3,)`, 每个 voxel 在世界坐标中的尺寸，顺序 (x, y, z)。
            - `box_shape_zyx`: np.ndarray, `(3,)`, 按 `(Z, Y, X)` 排列的 BOX 尺寸。
            - `x_range / y_range / z_range`: np.ndarray, `(2,)`, 当前 BOX 在世界坐标中的范围。
        """
        voxel_parts: list[np.ndarray] = []
        voxel_label: Optional[np.ndarray] = None
        box_origin_world: Optional[np.ndarray] = None
        voxel_size_world: Optional[np.ndarray] = None
        x_range: Optional[np.ndarray] = None
        y_range: Optional[np.ndarray] = None
        z_range: Optional[np.ndarray] = None

        for folder_name in self.data_folder_names:
            npz_path = self.all_data_path / folder_name / class_name / f"{sample_name}.npz"
            if not npz_path.exists():
                raise FileNotFoundError(f"BOX npz not found: {npz_path}")

            with np.load(npz_path) as npz_file:
                grid = np.asarray(npz_file["grid"])   # label 时通常为 (1,D,H,W) 或 (D,H,W), feature 时通常为 (C,D,H,W)
                meta = self._extract_box_meta(npz_file=npz_file, grid_shape_zyx=grid.shape[-3:])

            if box_origin_world is None:
                box_origin_world = meta["box_origin_world"]
                voxel_size_world = meta["voxel_size_world"]
                x_range = meta["x_range"]
                y_range = meta["y_range"]
                z_range = meta["z_range"]

            if "label" in folder_name:
                voxel_label = self._parse_voxel_label(grid)
                if self.class_mapping is not None:
                    voxel_label = self._apply_class_mapping(voxel_label, self.class_mapping)
            else:
                grid = grid.astype(np.float32, copy=False)   # feature 一律转 float32, 便于后续直接拼接
                if "emdb" in folder_name:   # TODO: 当前逻辑下 emdb 被强制归一化, 未来需要修改
                    grid = self._zscore_emdb_grid(grid)
                voxel_parts.append(grid)

        if voxel_label is None:
            raise RuntimeError(f"label grid is missing for sample: {sample_name}")
        if box_origin_world is None or voxel_size_world is None:
            raise RuntimeError(f"box metadata is missing for sample: {sample_name}")

        voxel_grid = np.concatenate(voxel_parts, axis=0).astype(np.float32, copy=False)
        box_shape_zyx = np.asarray(voxel_grid.shape[-3:], dtype=np.int64)

        return {
            "voxel_grid": voxel_grid,
            "voxel_label": voxel_label,
            "box_origin_world": box_origin_world,
            "voxel_size_world": voxel_size_world,
            "box_shape_zyx": box_shape_zyx,
            "x_range": x_range,
            "y_range": y_range,
            "z_range": z_range,
        }









    # -------------------------------------------------加载原子数据(原始解析数据) -------------------------------------------------
    def _cache_get(
        self, cache_obj: OrderedDict[str, dict[str, np.ndarray]], key: str
    ) -> Optional[dict[str, np.ndarray]]:
        """
        返回缓存中 pdb_id 对应的数据(并顺道把该键移到末尾以优化内存)
        """
        if key not in cache_obj:
            return None
        cache_obj.move_to_end(key)
        return cache_obj[key]

    def _cache_put(
        self,
        cache_obj: OrderedDict[str, dict[str, np.ndarray]],
        key: str,
        value: dict[str, np.ndarray],
    ) -> None:
        """
        超过容量时弹出最旧项目
        """
        if self.cache_size <= 0:
            return
        cache_obj[key] = value
        cache_obj.move_to_end(key)
        while len(cache_obj) > self.cache_size:
            cache_obj.popitem(last=False)

    def _load_structure_npz_cached(self, pdb_id: str) -> dict[str, np.ndarray]:
        """
        读取并缓存 `atoms.npz + labels.npz`。

        读取内容:
            - `atoms.npz`: `coords / features`
            - `labels.npz`: `binding_mask / pocket_class_ids / instance_ids`

        输入参数:
        - pdb_id: str, 字符串类型, 表示需要读取的样本的 pdb 编号或 ID 名称。

        输出结果注释:
        - 返回一个 dict[str, numpy.ndarray] 字典，包含以下键值对:
          - "atom_coord_world": numpy.ndarray, 形如 (N_atom, 3), 原子在世界坐标系下的三维坐标 (x, y, z)。
          - "atom_feature_raw": numpy.ndarray, 形如 (N_atom, F_raw), 原子的原始特征 (通常为 49 维)。
          - "binding_mask": numpy.ndarray, 形如 (N_atom,), bool 类型, 原子的二分类结合掩码标签 (属于 binding 区域为 True)。
          - "pocket_class_ids": numpy.ndarray, 形如 (N_atom,), int64 类型, 原子的多分类口袋类别 ID(目前0~4)。
          - "instance_ids": numpy.ndarray, 形如 (N_atom,), int64 类型, 原子的实例 ID。
        """
        # dict[str, numpy.ndarray] 或是 None, 尝试从缓存结构中获取对应 pdb_id 的全部原子和标注数据
        cached = self._cache_get(self.structure_cache, pdb_id)
        if cached is not None:
            return cached

        sample_dir = self.sample_root_path / pdb_id
        atoms_path = sample_dir / "atoms.npz"
        labels_path = sample_dir / "labels.npz"
        if not atoms_path.exists():
            raise FileNotFoundError(f"atoms.npz not found: {atoms_path}")
        if not labels_path.exists():
            raise FileNotFoundError(f"labels.npz not found: {labels_path}")

        with np.load(atoms_path) as atoms_npz:
            # numpy.ndarray, (N_atom, 3), float32 类型，全部原子的世界坐标
            atom_coord_world = np.asarray(atoms_npz["coords"], dtype=np.float32)
            # numpy.ndarray, (N_atom, F_raw), float32 类型，全部原子的特征数据
            atom_feature_raw = np.asarray(atoms_npz["features"], dtype=np.float32)

        with np.load(labels_path) as labels_npz:
            # numpy.ndarray, (N_atom,), bool 类型，每个原子是否属于口袋的0/1标签
            binding_mask = np.asarray(labels_npz["binding_mask"], dtype=bool)
            # numpy.ndarray, (N_atom,), int64 类型，每个原子对应的口袋具体类型分类的标签(目前0~4)
            pocket_class_ids = np.asarray(labels_npz["pocket_class_ids"], dtype=np.int64)
            # numpy.ndarray, (N_atom,), int64 类型，每个原子归属的实例级 ID
            instance_ids = np.asarray(labels_npz["instance_ids"], dtype=np.int64)

        current_raw_dim = int(atom_feature_raw.shape[1])
        if self.atom_raw_feature_dim is None:
            self.atom_raw_feature_dim = current_raw_dim
        elif self.atom_raw_feature_dim != current_raw_dim:
            raise ValueError(
                "atom raw feature dim is inconsistent across samples: "
                f"{self.atom_raw_feature_dim} vs {current_raw_dim}"
            )

        structure_data = {
            "atom_coord_world": atom_coord_world,
            "atom_feature_raw": atom_feature_raw,
            "binding_mask": binding_mask,
            "pocket_class_ids": pocket_class_ids,
            "instance_ids": instance_ids,
        }
        
        # 将上面构造好的全部数据对象结构放入缓存字典结构内，以便下一次请求此 pdb 时能快速读取
        self._cache_put(self.structure_cache, pdb_id, structure_data)
        return structure_data







    # ------------------------------------------------- 旋转辅助函数 -------------------------------------------------
    def _sample_rotation_params(self) -> tuple[int, int, int]:
        """
        采样一组 90 度旋转参数。返回的 `(axis1, axis2)` 以 `(Z, Y, X)` 为空间轴编号, 其中 `k` 表示在该平面内执行多少次 90 度逆时针旋转。

        - return: tuple[int, int, int], (axis1, axis2, k)
        """
        axis1, axis2 = np.random.choice([0, 1, 2], size=2, replace=False)
        k = random.randint(0, 3)
        return int(axis1), int(axis2), int(k)

    def _rotate_voxel_arrays(
        self,
        voxel_grid: np.ndarray,
        voxel_label: np.ndarray,
        hardmask: np.ndarray,
        voxel_valid_mask: np.ndarray,
        axis1: int,
        axis2: int,
        k: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        对 4D/3D 体素数组做完全同步的 90 度旋转。
        """
        rotated_voxel_grid = np.rot90(voxel_grid, k=k, axes=(axis1 + 1, axis2 + 1)).copy()
        rotated_voxel_label = np.rot90(voxel_label, k=k, axes=(axis1, axis2)).copy()
        rotated_hardmask = np.rot90(hardmask, k=k, axes=(axis1, axis2)).copy()
        rotated_voxel_valid_mask = np.rot90(voxel_valid_mask, k=k, axes=(axis1, axis2)).copy()
        return rotated_voxel_grid, rotated_voxel_label, rotated_hardmask, rotated_voxel_valid_mask

    def _rotate_shape_zyx(
        self, box_shape_zyx: np.ndarray, axis1: int, axis2: int, k: int
    ) -> np.ndarray:
        """
        只根据 rot90 的参数更新 box shape(当 `k` 为奇数时，旋转平面对应的两个轴长度会交换；当 `k` 为偶数时，shape 数值保持不变)。

        输入:
            - box_shape_zyx: numpy.ndarray, 形如 (3,), 表示原始的 BOX 空间大小信息，顺序为 (depth, height, width)。
            - axis1: int, 第一个旋转所在的轴索引 (0=z, 1=y, 2=x)。
            - axis2: int, 第二个旋转所在的轴索引 (0=z, 1=y, 2=x)。
            - k: int, 旋转的次数(每次90度)。

        输出:
            - numpy.ndarray, 形如 (3,), int64 类型，旋转后的 BOX 空间大小信息，顺序为 (depth, height, width)。
        """
        rotated_shape = np.asarray(box_shape_zyx, dtype=np.int64).copy()
        for _ in range(k % 4):
            old_shape = rotated_shape.copy()
            rotated_shape[axis1] = old_shape[axis2]
            rotated_shape[axis2] = old_shape[axis1]
        return rotated_shape

    def _rotate_zyx_coords(
        self,
        coord_zyx: np.ndarray,
        box_shape_zyx: np.ndarray,
        axis1: int,
        axis2: int,
        k: int,
    ) -> np.ndarray:
        """
        将连续体素坐标 `(z, y, x)` 和体素同步旋转。

        输入:
            - coord_zyx: numpy.ndarray, 形如 (N_selected, 3), 连续体素网格空间中的坐标，顺序为 (Z, Y, X)。
            - box_shape_zyx: numpy.ndarray, 形如 (3,), 原始未旋转的 BOX 空间大小，顺序为 (depth, height, width)。
            - axis1: int, 第一个旋转所在的轴索引 (0=z, 1=y, 2=x)。
            - axis2: int, 第二个旋转所在的轴索引 (0=z, 1=y, 2=x)。
            - k: int, 旋转的次数(每次90度)。

        输出:
            - numpy.ndarray, 形如 (N_selected, 3), float32 类型，旋转后的连续体素坐标，顺序为 (Z, Y, X)。
        """
        rotated_coord = np.asarray(coord_zyx, dtype=np.float32).copy()
        working_shape = np.asarray(box_shape_zyx, dtype=np.float32).copy()

        for _ in range(k % 4):
            old_coord = rotated_coord.copy()
            old_shape = working_shape.copy()
            # 对应 np.rot90 在二维平面中的坐标变换。
            # 注意这里的 coord_zyx 使用的是“角点语义”连续 voxel 坐标:
            # 若某个 voxel 中心位于 old_axis2 = j + 0.5，
            # 则旋转后应位于 new_axis1 = old_size_axis2 - (j + 0.5)。
            # 因此公式是 `new_axis1 = old_size_axis2 - old_axis2`，
            # 而不是离散 index 语义下常见的 `old_size_axis2 - 1 - old_axis2`。
            # new_axis2 = old_axis1
            rotated_coord[:, axis1] = old_shape[axis2] - old_coord[:, axis2]
            rotated_coord[:, axis2] = old_coord[:, axis1]
            working_shape[axis1] = old_shape[axis2]
            working_shape[axis2] = old_shape[axis1]

        return rotated_coord.astype(np.float32, copy=False)

    def _rotate_point_coords_local_voxel(
        self,
        atom_coord_local_voxel: np.ndarray,
        box_shape_zyx: np.ndarray,
        axis1: int,
        axis2: int,
        k: int,
    ) -> np.ndarray:
        """
        旋转 atom 的连续 voxel 坐标。内部由(x, y,z)排成 (z, y, x)，旋转完成后再换回 `(x, y, z)` 返回。

        输入:
            - atom_coord_local_voxel: numpy.ndarray, 形如 (N_selected, 3), 旋转前的原子的连续体素坐标，顺序为 (x, y, z)。
              该坐标保留角点语义，因此若某原子落在 voxel 中心，其坐标数值会带 `0.5` 偏移。
            - box_shape_zyx: numpy.ndarray, 形如 (3,), 原始的 BOX 空间大小信息，顺序为 (depth, height, width)。
            - axis1: int, 第一个旋转所在的轴索引 (0=z, 1=y, 2=x)。
            - axis2: int, 第二个旋转所在的轴索引 (0=z, 1=y, 2=x)。
            - k: int, 旋转的次数(每次90度)。

        输出:
            - numpy.ndarray, 形如 (N_selected, 3), float32 类型，旋转后的原子的连续体素坐标，顺序仍为 (x, y, z)。
        """
        if atom_coord_local_voxel.shape[0] == 0 or (k % 4) == 0:
            return atom_coord_local_voxel.astype(np.float32, copy=True)

        coord_zyx = atom_coord_local_voxel[:, [2, 1, 0]]
        rotated_zyx = self._rotate_zyx_coords(
            coord_zyx=coord_zyx,
            box_shape_zyx=box_shape_zyx,
            axis1=axis1,
            axis2=axis2,
            k=k,
        )
        return rotated_zyx[:, [2, 1, 0]].astype(np.float32, copy=False)

    def _rotate_point_coords_centered_world(
        self,
        atom_coord_centered_world: np.ndarray,
        axis1: int,
        axis2: int,
        k: int,
    ) -> np.ndarray:
        """
        旋转以 BOX 中心为原点的世界坐标(按xyz)。

        输入:
            - atom_coord_centered_world: numpy.ndarray, 形如 (N_selected, 3), 以 BOX 中心为原点的世界坐标，顺序为 (x, y, z)。
            - axis1: int, 第一个旋转所在的轴索引 (0=z, 1=y, 2=x)。
            - axis2: int, 第二个旋转所在的轴索引 (0=z, 1=y, 2=x)。
            - k: int, 旋转的次数(每次90度)。

        输出:
            - numpy.ndarray, 形如 (N_selected, 3), float32 类型，旋转后的以 BOX 中心为原点的世界坐标，顺序为 (x, y, z)。
        """
        if atom_coord_centered_world.shape[0] == 0 or (k % 4) == 0:
            return atom_coord_centered_world.astype(np.float32, copy=True)

        coord_zyx = atom_coord_centered_world[:, [2, 1, 0]].astype(np.float32, copy=True)
        for _ in range(k % 4):
            old_coord = coord_zyx.copy()
            coord_zyx[:, axis1] = -old_coord[:, axis2]
            coord_zyx[:, axis2] = old_coord[:, axis1]

        return coord_zyx[:, [2, 1, 0]].astype(np.float32, copy=False)

    def _check_rotation_is_supported(
        self,
        voxel_size_world: np.ndarray,
        box_shape_zyx: np.ndarray,
        axis1: int,
        axis2: int,
        k: int,
    ) -> None:
        """
        验证是否支持该旋转操作。只支持"近似各向同性体素 + 旋转平面两轴边长一致"的 90 度增强。

        输入:
            - voxel_size_world: numpy.ndarray, 形如 (3,), 原子的世界坐标物理分辨率尺寸，顺序为 (x, y, z)。
            - box_shape_zyx: numpy.ndarray, 形如 (3,), BOX 空间体素大小，顺序为 (depth, height, width)。
            - axis1: int, 第一个旋转所在的轴索引 (0=z, 1=y, 2=x)。
            - axis2: int, 第二个旋转所在的轴索引 (0=z, 1=y, 2=x)。
            - k: int, 旋转的次数(每次90度)。
        """
        if (k % 4) == 0:
            return

        if not np.allclose(
            voxel_size_world,
            float(voxel_size_world[0]),
            rtol=3e-2,
            atol=3e-4,
        ):
            # raise ValueError(
            #     "当前样本的 voxel_size_world 不是近似各向同性，"
            #     f"第一版不同步 90 度旋转不支持该情况: {voxel_size_world}"
            # )
            print("当前样本的 voxel_size_world 不是近似各向同性，"
                f"第一版不同步 90 度旋转不支持该情况: {voxel_size_world}")

        if (k % 2) == 1 and int(box_shape_zyx[axis1]) != int(box_shape_zyx[axis2]):
            # raise ValueError(
            #     "当前样本的旋转平面两轴长度不同，"
            #     "第一版为了避免世界坐标语义混乱，不支持这种 90 度轴交换旋转。"
            #     f" box_shape_zyx={box_shape_zyx}, axis1={axis1}, axis2={axis2}, k={k}"
            # )
            print("当前样本的旋转平面两轴长度不同，"
                "第一版为了避免世界坐标语义混乱，不支持这种 90 度轴交换旋转。"
                f" box_shape_zyx={box_shape_zyx}, axis1={axis1}, axis2={axis2}, k={k}")

    def _apply_synced_rotation(self, sample_dict: dict[str, Any]) -> dict[str, Any]:
        """
        对 sample 内 voxel 与 atom 做完全同步的 90 度旋转增强。
        
        输入:
            - sample_dict: dict[str, typing.Any], 包含未应用旋转增强前的原始样本数据字典(条目很多)

        输出:
            - dict[str, typing.Any], 返回经过旋转后的包含修改好的 voxel/atom 等信息的字典。
        """
        axis1, axis2, k = self._sample_rotation_params()
        self._check_rotation_is_supported(
            voxel_size_world=sample_dict["voxel_size_world"],
            box_shape_zyx=sample_dict["box_shape_zyx"],
            axis1=axis1,
            axis2=axis2,
            k=k,
        )
        if (k % 4) == 0:
            return sample_dict

        # 先旋转 voxel 侧数组
        (
            sample_dict["voxel_grid"],
            sample_dict["voxel_label"],
            sample_dict["hardmask"],
            sample_dict["voxel_valid_mask"],
        ) = self._rotate_voxel_arrays(
            voxel_grid=sample_dict["voxel_grid"],
            voxel_label=sample_dict["voxel_label"],
            hardmask=sample_dict["hardmask"],
            voxel_valid_mask=sample_dict["voxel_valid_mask"],
            axis1=axis1,
            axis2=axis2,
            k=k,
        )

        # 再旋转 point 侧坐标
        original_shape_zyx = sample_dict["box_shape_zyx"]
        sample_dict["atom_coord_local_voxel"] = self._rotate_point_coords_local_voxel(
            atom_coord_local_voxel=sample_dict["atom_coord_local_voxel"],
            box_shape_zyx=original_shape_zyx,
            axis1=axis1,
            axis2=axis2,
            k=k,
        )
        sample_dict["atom_coord_centered_world"] = self._rotate_point_coords_centered_world(
            atom_coord_centered_world=sample_dict["atom_coord_centered_world"],
            axis1=axis1,
            axis2=axis2,
            k=k,
        )
        # 最后更新 BOX 的 shape
        sample_dict["box_shape_zyx"] = self._rotate_shape_zyx(
            box_shape_zyx=original_shape_zyx,
            axis1=axis1,
            axis2=axis2,
            k=k,
        )

        # 旋转后，根据 `center = origin + 0.5 * shape_xyz * voxel_size_world` 重新计算当前 BOX 的世界坐标中心。
        box_shape_xyz = sample_dict["box_shape_zyx"][[2, 1, 0]].astype(np.float32)
        box_center_world = (sample_dict["box_origin_world"] + 0.5 * box_shape_xyz * sample_dict["voxel_size_world"]).astype(
            np.float32, copy=False
        )

        # 旋转后，`atom_coord_world = atom_coord_centered_world + 新的 box_center_world`。
        sample_dict["atom_coord_world"] = (
            sample_dict["atom_coord_centered_world"] + box_center_world[None, :]
        ).astype(np.float32, copy=False)
        sample_dict["atom_valid_mask"] = build_atom_valid_mask(
            atom_coord_local_voxel=sample_dict["atom_coord_local_voxel"],
            atom_is_in_core_box=sample_dict["atom_is_in_core_box"],
            box_shape_zyx=sample_dict["box_shape_zyx"],
            valid_crop_margin=float(self.valid_crop_margin),
        )

        return sample_dict









    # # ------------------------------------------------- 返回 -------------------------------------------------
    def __getitem__(self, index: int) -> dict[str, Any]:
        """
        读取并组装第 `index` 个样本的数据字典。

        输入:
            - index: int, 数据集中的样本索引。

        输出:
            - dict[str, typing.Any], 包含一个训练或者推理所需的完整数据的字典，所有 numpy 数组均已转换为 torch.Tensor 类型, 包含: 
                - `voxel_grid`: torch.Tensor, 形如 (C, D, H, W)，float32 类型的体素特征网格。
                - `voxel_label`: torch.Tensor, 形如 (D, H, W)，int64 类型的体素分类真值标签。
                - `hardmask`: torch.Tensor, 形如 (D, H, W)，int64 类型，标记该体素是否真实存在有效的结构。
                - `voxel_valid_mask`: torch.Tensor, 形如 (D, H, W)，bool 类型，是否在去除边缘 margin 之后的核心监督区域。
                - `atom_coord_world`: torch.Tensor, 形如 (N_selected, 3)，float32 类型的点云世界坐标。
                - `atom_feat`: torch.Tensor, 形如 (N_selected, F_raw)，float32 类型的点特征。
                ...

        主流程:
            1. 先从 `self.total_sample` 中拿到 `sample_name`(BOX名) 和 `class_name`(类别名)。
            2. 读取 BOX 的 voxel 数据、标签和几何元信息。
            3. 读取该样本 pdb 对应的结构级 atom 数据与全图元信息。
            4. 按 BOX 范围筛选原子，构造坐标、标签、特征和监督 mask。
            5. 如有需要，对 voxel 与 atom 一起做同步旋转增强。
            6. 最后把 numpy 的 sample 转换成 torch tensor 字典返回。
        """
        try:
            # dict, 记录该样本属于哪个类别以及样本名
            sample_meta = self.total_sample[index]          
            # str, 例如 "9dic_2_0_17_20"
            sample_name = sample_meta["sample_name"]        
            # str, 例如 "small_molecule"
            class_name = sample_meta["class_name"]          

            # 第一步: 解析样本名，得到 pdb_id / instance_id / 中心框标记等元信息。
            parsed_name = self._parse_sample_name(sample_name)
            # 第二步: 读取 BOX 级别的信息: voxel 特征、标签和局部几何信息。
            box_data = self._load_box_npz_triplet(class_name, sample_name)
            # 第三步: 读取该 pdb 对应的结构级原子信息。
            structure_data = self._load_structure_npz_cached(parsed_name["pdb_id"])


            # 第四步 + sample dict 组装: 调用共享 builder
            sample_dict = build_box_point_numpy_sample(
                voxel_grid=box_data["voxel_grid"],
                voxel_label=box_data["voxel_label"],
                atom_coords_world_full=structure_data["atom_coord_world"],
                atom_features_raw_full=structure_data["atom_feature_raw"],
                atom_labels_full=structure_data["pocket_class_ids"],
                box_origin_world=box_data["box_origin_world"],
                voxel_size_world=box_data["voxel_size_world"],
                box_shape_zyx=box_data["box_shape_zyx"],
                atom_buffer_radius=self.atom_buffer_radius,
                valid_crop_margin=self.valid_crop_margin,
                class_mapping=self.class_mapping,
            )
            # 追加元信息
            sample_dict["sample_name"] = sample_name
            sample_dict["pdb_id"] = parsed_name["pdb_id"]
            sample_dict["class_name"] = class_name
            sample_dict["instance_id"] = parsed_name["instance_id"]
            sample_dict["is_center_box"] = parsed_name["is_center_box"]

            # 第五步: 训练模式下，若开启随机旋转，需对该样本数据做统一的 90 度旋转。
            if self.enable_random_rotation:
                sample_dict = self._apply_synced_rotation(sample_dict)

            # 第六步: 统一把 sample 中的 numpy 数组整理为 torch tensor 返回。
            return self._to_torch_sample(sample_dict)

        except Exception as exc:
            import traceback

            error_msg = (
                f"\n[BoxPointDataset Error] index={index}, "
                f"sample={self.total_sample[index] if index < len(self.total_sample) else 'Unknown'}\n"
                f"{traceback.format_exc()}\n"
            )
            print(error_msg, flush=True)
            raise RuntimeError(error_msg) from exc

    def _to_torch_sample(self, sample_dict: dict[str, Any]) -> dict[str, Any]:
        """
        整理类型, 将 numpy 数组转换为 torch 数组并整理dtype。
        委托给共享 to_torch_sample, 保留方法壳以兼容可能的外部子类引用。
        """
        return to_torch_sample(sample_dict)
