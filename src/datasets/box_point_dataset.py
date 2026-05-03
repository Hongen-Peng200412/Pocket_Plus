# -*- coding: utf-8 -*-
"""
=============================================================================
BoxPointDataset
=============================================================================
本文件实现 stage1 使用的“体素 + 点云”联合数据集。这是主脚本


src\datasets 内的调用逻辑:
1. 前继: 
    - src\datasets\density_channel_builder.py 负责生成由 emdb_exp_BOX、emdb_sim_BOX 生成密度体素特征; 
    - src\datasets\box_geometry.py 负责处理坐标与hardmask的解析
之后, src\datasets\box_sample_builder.py 负责调用 box_geometry.py 做简单的装配. 它们都在 wrapper 前就使用完毕了.

2. 后继: 
    - src\datasets\box_point_collate.py 负责把单样本结果组装为 batch 
    - src\datasets\balanced_foreground_sampler.py 负责 batch 内具体样本的采样
它们将在训练(wrapper)时应用


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

典型数据目录结构:
    all_data_path/
    ├── emdb_BOX/            ← 密度图特征 (CDHW)
    │   ├── small_molecule/
    │   │   ├── 9f3f_0_0_0_0_C.npz(具体样本)
    │   │   └── ...
    │   └── metal_ion/ ...
    ├── pdb_feature_BOX/     ← 已不再离线生成, 改为模型在线 scatter (DEPRECATED)
    │   └── ...
    └── pdb_label_BOX/       ← 标签 (CDHW -> DHW)  [可选]
        └── ... 
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
from omegaconf import DictConfig, OmegaConf
import torch
from torch.utils.data import Dataset

from .box_geometry import (
    build_atom_coordinates,
    build_atom_features,
    build_atom_valid_mask,
    build_hardmask_from_atom_coordinates,
    build_hardmask_from_world_coordinates,
    build_voxel_valid_mask,
    select_atoms_for_box,
)
from .box_point_collate import box_point_collate
from .box_sample_builder import build_box_point_numpy_sample, to_torch_sample
from .density_channel_builder import DensityChannelConfig, build_density_channels

# set[str], density_channel_builder 管辖的目录名集合
_DENSITY_BUILDER_FOLDERS: set[str] = {"emdb_exp_BOX", "emdb_sim_BOX"}


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
        density_channel_config: Optional[dict | DensityChannelConfig] = None,
        **kwargs: Any,
    ) -> None:
        """
            把一个 BOX 视角下需要的 voxel 信息和 atom 信息打包为同一个 sample dict。

            输入参数 (Input Parameters):
                - sample_root_path: str, 原始结构缓存目录。要求 `sample_root_path / pdb_id /` 下存在 `atoms.npz` 与 `labels.npz`。


                - all_data_path: str, BOX 数据所在根目录。目录结构形如 all_data_path / data_folder_names[0] / class_folder_names[0] / 9dic_2_0_17_20(pdbid_instanceid_rxx_ryy_rzz_centerornot)
                - data_folder_names: list[str], BOX 文件夹名称列表, 默认为 ["emdb_exp_BOX", "emdb_sim_BOX","pdb_label_BOX", "ligand_dist_BOX"]。
                    - 含 `"ligand_dist"` 的目录会作为辅助监督信号单独加载(不拼入 voxel_grid)
                    - 含 `"label"` 的目录会被视为体素标签
                    - 其余目录, 也就是 "emdb_exp_BOX", "emdb_sim_BOX" 都默认视为密度特征, 由 src\datasets\box_sample_builder.py 统一运算并装配; 
                    - pdb_feature_BOX 已不再离线生成
                - class_folder_names: list[str], 类别目录名称列表，与 `split_file` 的顺序严格一一对应, 典型配置为 ["metal_ion", "peptide", "nucleic", "small_molecule", "random_BOX"]("random_BOX"是随机切分的类不作为类别); 
                # see me: 为了正负类均匀取样, 要使"random_BOX"放到最后
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
                - `atom_coord_local_voxel` 的数值语义是连续 voxel 坐标(角点语义)，但字段顺序仍然保持 `(x, y, z)`。若后续要做 `grid_sample(align_corners=True)`，需要先把该坐标减去 `0.5` 再做归一化。
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

        self.data_folder_names = list(data_folder_names)            # ["emdb_BOX", "pdb_label_BOX"] (pdb_feature_BOX 已不再离线生成)
        self.class_folder_names = list(class_folder_names)          # ["metal_ion", "peptide", "nucleic", "small_molecule"]
        self.collate_fn = box_point_collate                         # 让外部 DataLoader 直接复用项目内 collate

        if self.valid_crop_margin < 0:
            raise ValueError("valid_crop_margin must be >= 0")

        self._validate_folder_layout()
        self.is_train = self._parse_mode(mode)
        self.enable_random_rotation = bool(enable_random_rotation and self.is_train)

        # ---------- 密度通道 builder ----------
        if density_channel_config is None:
            self.density_config = DensityChannelConfig()
        elif isinstance(density_channel_config, DictConfig):
            self.density_config = DensityChannelConfig(
                **OmegaConf.to_container(density_channel_config, resolve=True)
            )
        elif isinstance(density_channel_config, dict):
            self.density_config = DensityChannelConfig(**density_channel_config)
        elif isinstance(density_channel_config, DensityChannelConfig):
            self.density_config = density_channel_config
        else:
            raise TypeError( f"density_channel_config 类型不支持: {type(density_channel_config)}")

        sample_name_lists = self._load_split_lists(split_file)
        self.total_sample = self._build_sample_index(sample_name_lists)

        # str 为pdb_id
        self.structure_cache: OrderedDict[str, dict[str, np.ndarray]] = OrderedDict()  # 每个 worker 各自持有一个轻量 LRU cache，避免重复反复读取 atoms.npz / labels.npz。

        # 首次读取 atoms.npz 时记录原始 atom feature 维度，并对后续样本保持一致性校验。
        self.atom_raw_feature_dim: Optional[int] = None


    def _validate_folder_layout(self) -> None:
        """
        校验目录约束：
        """
        expected_data_folder_names = {"emdb_exp_BOX", "emdb_sim_BOX", "pdb_label_BOX", "ligand_dist_BOX"}
        if set(self.data_folder_names) != expected_data_folder_names:
            print(f"waring: data_folder_names 预期为 {expected_data_folder_names}, " f"当前配置为: {self.data_folder_names}")
            # raise ValueError(f"data_folder_names 必须为 {expected_data_folder_names}, " f"当前配置为: {self.data_folder_names}")
        if "random_BOX" not in self.class_folder_names:
            raise ValueError("random_BOX is not in data_folder_names")
        if "random_BOX" in self.class_folder_names and self.class_folder_names[-1] != "random_BOX":
            raise ValueError("random_BOX must be the last element in class_folder_names")

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
                - sample_name_lists: list[list[str]], 其中外层维度对应类别，内层是该类别下的样本名列表 (比如第一个元素的sample_name_lists[0]是 list[str], 是关于"metal_ion"的BOX文件名)
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

        返回结果:
            - `self.total_sample`: list[dict[str, Any]], 其每一项是:
                - `class_name`: 类别目录名(如"metal_ion")。
                - `sample_name`: 具体样本文件名（不含 `.npz` 后缀）。
            - `self.foreground_indices`: list[int], 保证含正类的样本索引 (class_name != "random_BOX" 的所有样本)
            - `self.background_indices`: list[int], 多数为纯背景的样本索引 (class_name == "random_BOX" 的所有样本)
            - 若数据集中无 "random_BOX" 类别，background_indices 为空列表
        """
        total_sample: list[dict[str, Any]] = []
        # list[int], 保证含正类样本的索引，对应 class_name != "random_BOX" 的条目
        self.foreground_indices: list[int] = []
        # list[int], 多数为纯背景样本的索引，对应 class_name == "random_BOX" 的条目
        self.background_indices: list[int] = []
        for class_idx, sample_names in enumerate(sample_name_lists):
            class_name = self.class_folder_names[class_idx]
            # bool, 当前类别是否为背景类别: 最后一个默认为 "random_BOX""
            is_background_class = (class_idx == len(self.class_folder_names) - 1)
            for sample_name in sample_names:
                # int, 当前样本在扁平索引中的位置
                idx = len(total_sample)
                total_sample.append(
                    {
                        "class_name": class_name,
                        "sample_name": sample_name,
                    }
                )
                if is_background_class:
                    self.background_indices.append(idx)
                else:
                    self.foreground_indices.append(idx)
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
            - `npz_file`: 切分为BOX的 npz 文件
            - `grid_shape_zyx`: 当前网格数组的空间形状，顺序 `(Z, Y, X)`。

        输出:
            - `box_origin_world`: (3,), 世界坐标系下 BOX 左下近角点，顺序 `(x, y, z)`。
            - `voxel_size_world`: (3,), 每个 voxel 在世界坐标中的尺寸，顺序 `(x, y, z)`。
            - `x_range`: (2,), x 轴范围，顺序 `(min, max)`。
            - `y_range`: (2,), y 轴范围，顺序 `(min, max)`。
            - `z_range`: (2,), z 轴范围，顺序 `(min, max)`。
        """
        box_origin_world = np.asarray(npz_file["origin"], dtype=np.float32).reshape(3)
        voxel_size_world = np.asarray(npz_file["voxel_size"], dtype=np.float32).reshape(3)
        x_range = np.asarray(npz_file["x_range"], dtype=np.float32).reshape(2)
        y_range = np.asarray(npz_file["y_range"], dtype=np.float32).reshape(2)
        z_range = np.asarray(npz_file["z_range"], dtype=np.float32).reshape(2)

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



    def _apply_class_mapping(self, label: np.ndarray, mapping: list[int]) -> np.ndarray:
        """
        将原始类别 ID 映射成新的类别 ID。

        例如 `mapping=[0, 1, 2, 2]` 时，表示把原始类别 3 合并到新类别 2。
        """
        mapped_label = np.zeros_like(label, dtype=np.int64)
        for old_class_id, new_class_id in enumerate(mapping):
            mapped_label[label == old_class_id] = int(new_class_id)
        return mapped_label

    def _load_box_npz_raw(self, class_name: str, sample_name: str) -> dict[str, Any]:
        """
        读取一个 BOX 的原始 voxel 数据与几何元信息, 但暂时不调用 density_channel_builder。
        class_name 形如"metal_ion", sample_name 形如"9dic_2_0_17_20"(pdbid_instanceid_rxx_ryy_rzz_centerornot)

        对于 class_name 下的样本名为 sample_name 的这个特定样本，遍历 `self.data_folder_names` 指定的多个目录，并按以下规则组织:
            - 含 `"label"` 的目录读成 voxel 标签(结合原子的硬标签)，最终统一为 `(D, H, W)` 的 int64。
            - 含 `"emdb"` 的目录暂存到 density_raws, 由调用侧统一处理。
            - 含 `"ligand_dist"` 的目录作为辅助监督信号单独加载, 并立即做 class_mapping 通道选择 + min 归约。

        返回字段:
            - `density_raws`: dict[str, np.ndarray], 键为目录名(如 "emdb_exp_BOX", "emdb_sim_BOX"), 值为 (D, H, W) float32 原始密度
            - `voxel_label`: np.ndarray, `(D, H, W)`, int64。
            - `ligand_dist_map`: np.ndarray | None, `(D, H, W)`, float32, 归约后的单通道 ligand 距离图。

            - `box_origin_world`: np.ndarray, `(3,)`, 世界坐标系下 BOX 左下近角点，顺序 (x, y, z)。
            - `voxel_size_world`: np.ndarray, `(3,)`, 每个 voxel 在世界坐标中的尺寸，顺序 (x, y, z)。
            - `box_shape_zyx`: np.ndarray, `(3,)`, 按 `(Z, Y, X)` 排列的 BOX 尺寸。
            - `x_range / y_range / z_range`: np.ndarray, `(2,)`, 当前 BOX 在世界坐标中的范围。
        """
        voxel_label: Optional[np.ndarray] = None
        box_origin_world: Optional[np.ndarray] = None
        voxel_size_world: Optional[np.ndarray] = None
        x_range: Optional[np.ndarray] = None
        y_range: Optional[np.ndarray] = None
        z_range: Optional[np.ndarray] = None

        # dict[str, np.ndarray], 暂存由 density_channel_builder 管辖的原始密度 grid
        density_raws: dict[str, np.ndarray] = {}
        # np.ndarray | None, (D, H, W), float32, 归约后的单通道 ligand 距离图
        ligand_dist_map: np.ndarray | None = None
        # np.ndarray | None, (3,), int64, BOX 空间尺寸
        box_shape_zyx: np.ndarray | None = None


        for folder_name in self.data_folder_names:
            npz_path = self.all_data_path / folder_name / class_name / f"{sample_name}.npz"

            with np.load(npz_path) as npz_file:
                grid = np.asarray(npz_file["grid"])   # label 时通常为 (1,D,H,W) 或 (D,H,W), feature 时通常为 (C,D,H,W)
                meta = self._extract_box_meta(npz_file=npz_file, grid_shape_zyx=grid.shape[-3:])
            if box_origin_world is None:
                box_origin_world = meta["box_origin_world"]
                voxel_size_world = meta["voxel_size_world"]
                x_range = meta["x_range"]
                y_range = meta["y_range"]
                z_range = meta["z_range"]
                box_shape_zyx = np.asarray(grid.shape[-3:], dtype=np.int64)

            if "ligand_dist" in folder_name:
                # ligand_dist 目录: 辅助监督信号, 加载后立即做通道选择 + min 归约
                # np.ndarray, (K, D, H, W), float32, 原始多通道距离图, 通道 i 对应 class_id = i+1 (见 bind.py bind_LigandMinDist_to_EMDB)
                ligand_dist_raw = grid.astype(np.float32, copy=False)
                if self.class_mapping is not None:
                    # 只选 class_mapping 映射到非零(前景)的通道, class_mapping 索引就是 class_id, 值 >0 表示该类保留为前景
                    selected_channels = [
                        cls_id - 1 for cls_id in range(1, len(self.class_mapping))
                        if self.class_mapping[cls_id] > 0
                    ]
                    if len(selected_channels) > 0:
                        # np.ndarray, (D, H, W), float32, 选中通道取 min
                        ligand_dist_map = np.min(ligand_dist_raw[selected_channels], axis=0)
                    else:
                        # 无可选通道, 用全通道 min
                        ligand_dist_map = np.min(ligand_dist_raw, axis=0)
                else:
                    # 无 class_mapping, 全通道 min
                    ligand_dist_map = np.min(ligand_dist_raw, axis=0)
            elif "label" in folder_name:
                voxel_label = self._parse_voxel_label(grid)
                if self.class_mapping is not None:
                    voxel_label = self._apply_class_mapping(voxel_label, self.class_mapping)
            elif folder_name in _DENSITY_BUILDER_FOLDERS:
                # 暂存到 density_raws，由 builder 统一处理, 确保是 (D, H, W) 而非 (1, D, H, W)
                grid = grid.astype(np.float32, copy=False)
                if grid.ndim == 4 and grid.shape[0] == 1:
                    grid = grid[0]
                density_raws[folder_name] = grid
            else:
                print(f"Error: Unknown folder_name: {folder_name}, 它既不是 label 或 ligand_dist, 也不在 _DENSITY_BUILDER_FOLDER")

        return {
            "density_raws": density_raws,
            "voxel_label": voxel_label,
            "ligand_dist_map": ligand_dist_map,
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

        输入:
            - pdb_id: str, 字符串类型, 表示需要读取的样本的 pdb 编号或 ID 名称。

        输出:
            - 返回一个 dict[str, numpy.ndarray] 字典，包含以下键值对:
                - `"atom_coord_world"`: numpy.ndarray, 形如 (N_atom, 3), 原子在世界坐标系下的三维坐标 (x, y, z)。
                - `"atom_feature_raw"`: numpy.ndarray, 形如 (N_atom, F_raw), 原子的原始特征 (通常为 49 维)。
                - `"binding_mask"`: numpy.ndarray, 形如 (N_atom,), bool 类型, 原子的二分类结合掩码标签 (属于 binding 区域为 True)。
                - `"pocket_class_ids"`: numpy.ndarray, 形如 (N_atom,), int64 类型, 原子的多分类口袋类别 ID(目前0~4)。
                - `"instance_ids"`: numpy.ndarray, 形如 (N_atom,), int64 类型, 原子的实例 ID。
        """
        # dict[str, numpy.ndarray] 或是 None, 尝试从缓存结构中获取对应 pdb_id 的全部原子和标注数据
        cached = self._cache_get(self.structure_cache, pdb_id)
        if cached is not None:
            return cached

        sample_dir = self.sample_root_path / pdb_id
        atoms_path = sample_dir / "atoms.npz"
        labels_path = sample_dir / "labels.npz"

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
            raise ValueError(f"atom raw feature dim is inconsistent across samples: {self.atom_raw_feature_dim} vs {current_raw_dim}")

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
        k: int,
    ) -> None:
        """
        验证是否支持该旋转操作。只支持"各向同性体素 + 立方体 BOX"的 90 度增强。

        输入:
            - voxel_size_world: numpy.ndarray, 形如 (3,), 每个体素在世界坐标中的尺寸，顺序为 (x, y, z)。
            - box_shape_zyx: numpy.ndarray, 形如 (3,), BOX 空间体素大小，顺序为 (depth, height, width)。
            - k: int, 旋转的次数(每次90度)。
        """
        if (k % 4) == 0:
            return

        # 立方体 BOX 检查: 三轴边长必须完全相同
        sz = box_shape_zyx.astype(np.int64)
        if not (sz[0] == sz[1] == sz[2]):
            raise ValueError(
                f"90 度旋转增强仅支持立方体 BOX, 当前 box_shape_zyx={box_shape_zyx.tolist()}"
            )

        # 各向同性体素检查
        if not np.allclose(
            voxel_size_world,
            float(voxel_size_world[0]),
            rtol=3e-2,
            atol=3e-4,
        ):
            print("当前样本的 voxel_size_world 不是近似各向同性，"
                f"90 度旋转不支持该情况: {voxel_size_world}")

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
            k=k,
        )
        if (k % 4) == 0:
            return sample_dict

        # ---- 1. 旋转 voxel 侧数组 ----
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
        # ligand_dist_map 同步旋转: (D, H, W) → 3D, axes=(axis1, axis2)
        if "ligand_dist_map" in sample_dict:
            sample_dict["ligand_dist_map"] = np.rot90(
                sample_dict["ligand_dist_map"], k=k, axes=(axis1, axis2)
            ).copy()

        # ---- 2. 旋转 point 侧坐标 ----
        # 立方体 BOX: box_shape_zyx 旋转后不变, 无需更新
        box_shape_zyx = sample_dict["box_shape_zyx"]
        sample_dict["atom_coord_local_voxel"] = self._rotate_point_coords_local_voxel(
            atom_coord_local_voxel=sample_dict["atom_coord_local_voxel"],
            box_shape_zyx=box_shape_zyx,
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

        # ---- 3. 重算 atom_coord_world 与 atom_valid_mask ----
        # 立方体 + 各向同性体素: box_center_world 旋转前后不变
        # np.ndarray, (3,), float32, BOX 中心世界坐标
        box_shape_xyz = box_shape_zyx[[2, 1, 0]].astype(np.float32)
        box_center_world = (
            sample_dict["box_origin_world"] + 0.5 * box_shape_xyz * sample_dict["voxel_size_world"]
        ).astype(np.float32, copy=False)
        # atom_coord_world = atom_coord_centered_world + box_center_world
        sample_dict["atom_coord_world"] = (
            sample_dict["atom_coord_centered_world"] + box_center_world[None, :]
        ).astype(np.float32, copy=False)
        sample_dict["atom_valid_mask"] = build_atom_valid_mask(
            atom_coord_local_voxel=sample_dict["atom_coord_local_voxel"],
            atom_is_in_core_box=sample_dict["atom_is_in_core_box"],
            box_shape_zyx=box_shape_zyx,
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

            # 第二步: 读取 BOX 级别的原始数据(不调用 density builder)。
            box_raw = self._load_box_npz_raw(class_name, sample_name)

            # 第三步: 读取该 pdb 对应的结构级原子信息。
            structure_data = self._load_structure_npz_cached(parsed_name["pdb_id"])

            # 第四步: 用受体原子坐标构造 receptor_mask, 供 density builder 的 exp-sim 拟合使用。
            # np.ndarray, (D, H, W), bool, 受体原子对应的体素掩码
            receptor_mask = build_hardmask_from_world_coordinates(
                atom_coords_world=structure_data["atom_coord_world"],
                box_origin_world=box_raw["box_origin_world"],
                voxel_size_world=box_raw["voxel_size_world"],
                box_shape_zyx=box_raw["box_shape_zyx"],
            ).astype(bool)

            # 第五步: 调用 density builder, 传入 receptor_mask。
            # np.ndarray, (C_density, D, H, W), float32, builder 输出的多通道密度特征
            if len(box_raw["density_raws"]) == 0:
                density_grid = np.empty(
                    (0, *tuple(int(v) for v in box_raw["box_shape_zyx"])),
                    dtype=np.float32,
                )
            else:
                density_grid = build_density_channels(
                    exp_raw=box_raw["density_raws"].get("emdb_exp_BOX"),
                    sim_raw=box_raw["density_raws"].get("emdb_sim_BOX"),
                    config=self.density_config,
                    receptor_mask=receptor_mask,
                )

            # 第六步: 拼接 voxel_grid。
            # np.ndarray, (C, D, H, W), float32, 密度通道即为完整体素特征
            voxel_grid = density_grid.astype(np.float32, copy=False)
            # np.ndarray, (3,), int64, BOX 空间尺寸
            box_shape_zyx = np.asarray(voxel_grid.shape[-3:], dtype=np.int64)

            # 第七步: 取出已归约的 ligand 距离图。
            # np.ndarray | None, (D, H, W), float32, 归约后的距离图
            ligand_dist_map = box_raw["ligand_dist_map"]

            # 第八步: 调用共享 builder 组装 sample dict。
            sample_dict = build_box_point_numpy_sample(
                voxel_grid=voxel_grid,
                voxel_label=box_raw["voxel_label"],
                atom_coords_world_full=structure_data["atom_coord_world"],
                atom_features_raw_full=structure_data["atom_feature_raw"],
                atom_labels_full=structure_data["pocket_class_ids"],
                box_origin_world=box_raw["box_origin_world"],
                voxel_size_world=box_raw["voxel_size_world"],
                box_shape_zyx=box_shape_zyx,
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
            # 追加可选的 ligand 距离图
            if ligand_dist_map is not None:
                sample_dict["ligand_dist_map"] = ligand_dist_map

            # 第九步: 训练模式下，若开启随机旋转，需对该样本数据做统一的 90 度旋转。
            if self.enable_random_rotation:
                sample_dict = self._apply_synced_rotation(sample_dict)

            # 第十步: 统一把 sample 中的 numpy 数组整理为 torch tensor 返回。
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
