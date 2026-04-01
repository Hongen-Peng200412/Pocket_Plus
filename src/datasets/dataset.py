import json
import os
import random
import re
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from src.datasets.box_geometry import build_hardmask_from_world_coordinates
from utils.network_tools import random_rotation90_plus


torch.manual_seed(114514)
random.seed(114514)
np.random.seed(114514)


_BOX_SAMPLE_NAME_PATTERN = re.compile(
    r"^(?P<pdb_id>.+)_(?P<instance_id>-?\d+)_(?P<rxx>-?\d+)_(?P<ryy>-?\d+)_(?P<rzz>-?\d+)(?P<center>_C)?$"
)


class MyDatasets(Dataset):
    """
    一个文件路径的具体格式为 : all_data_path / data_folder_names / class_folder_name / sample_name, 其中sample_name通过 split_file[class_folder_name] 读取
    输入参数:
        - all_data_path: str, 数据集根目录
        - split_file: list[str], 样本划分的.json文件. split_file[k] 代表 class_folder_name[k] 对应的划分.json路径, 因而它的长度必须和 class_folder_name 相同, 并且顺序和含义一致！！ split_file[k]读取后形如 ["9dic_2_0_17_20", ...], 注意只能是 list[str] 不能是 str.
        - mode: "train"/"val"/"test"

        - data_folder_names: list[str], 目前形如 ["emdb_BOX", "pdb_feature_BOX", "pdb_label_BOX"]
        - class_folder_names: list[str], 目前形如 ["metal_ion", "peptide", "nucleic", "small_molecule"]
        - class_mapping: list[int], 可选, 类别映射表. 例如 [0, 1, 2, 2] 将原始的第2类口袋+第3类口袋, 映射为第2类口袋————此时需保证原本有3类口袋 + 0做背景. 索引对应原始类别, 值对应新的类别

    Note：
        - 对于 data_folder_names: 
            - 如果data_folder_names[k] 含有字段"label", 那么就对它看成标签; 别的都当做特征的一部分进行拼接
            - 如果 data_folder_names[k] 含有字段"emdb", 那么就对它做归一化(看成密度图或差图)
        - 默认所有的从.npz读取的原始数据, 都有形状 CDHW(目前确实如此) , 但是标签会转化为 DHW 以适配 Pocket\src\modules\losses.py 
    """

    def __init__(
        self,
        all_data_path: Optional[str] = None,
        split_file: list[str] = None,
        mode: Optional[str] = None,

        sample_root_path: Optional[str] = None,
        data_folder_names: list[str] = ["emdb_BOX", "pdb_feature_BOX", "pdb_label_BOX"],    #NOTE: 之后固定下来: "pdb_label_BOX" 当作标签, 别的都当做特征的一部分进行拼接
        class_folder_names: list[str] = ["metal_ion", "peptide", "nucleic", "small_molecule"], 
        class_mapping: list[int] = None, 
        **kwargs,
    ):
        self.all_data_path = all_data_path
        self.sample_root_path = sample_root_path
        self.data_folder_names = data_folder_names
        self.class_folder_names = class_folder_names
        self.class_mapping = class_mapping

        # 检查 emdb 文件夹是否在最前面
        non_emdb_seen = False
        for folder_name in self.data_folder_names:
            if "emdb" in folder_name:
                if non_emdb_seen:
                    raise ValueError(f"含有 'emdb' 的特征文件夹必须排在最前面，但当前配置为: {self.data_folder_names}")
            else:
                non_emdb_seen = True

        if mode is None:
            raise ValueError("mode is empty!")
        mode_lower = str(mode).lower()
        if mode_lower in {"train", "fit"}:
            istrain = True
        elif mode_lower in {"val", "valid", "validation"}:
            istrain = False
        elif mode_lower in {"test", "evaluate"}:
            istrain = False
        else:
            raise ValueError(f"Unknown mode: {mode}")

        self.istrain = bool(istrain)


        sample_name_lists = []                   # 将会是 list[list], 长度= class_folder_name 的长度
        for i in split_file:
            with open(i, "r", encoding="utf-8") as f:
                sample_name_lists.append(json.load(f))
        
        self.total_sample = []                   # List[dict], 每个元素形为{k:样本文件名}, 此样本对应文件位置= all_data_path / data_folder_names / class_folder_name[k] / sample_name
        for i, this_list in enumerate(sample_name_lists):
            for sample_name in this_list:
                item = {i: sample_name}
                self.total_sample.append(item)


    def __len__(self):
        return len(self.total_sample)


    def __getitem__(self, index):
        try:
            grid = []
            item = self.total_sample[index]          # 形为 {k: sample_name}
            k, sample_name = next( iter(item.items()) )    # 键, 值; k 代表本样本属于第几个 class 

            # np.ndarray | None, (3,), 当前 BOX 的世界坐标原点
            box_origin_world = None
            # np.ndarray | None, (3,), 当前 BOX 的体素尺寸
            voxel_size_world = None
            # np.ndarray | None, (3,), 当前 BOX 的空间大小, 顺序为 (D, H, W)
            box_shape_zyx = None

            for data_folder_name in self.data_folder_names:
                sample_path = os.path.join(self.all_data_path, data_folder_name, self.class_folder_names[k], sample_name+".npz")
                # NOTE: 如果data_folder_names[k] 含有字段"label", 那么就对它看成标签
                if "label" in data_folder_name:
                    with np.load(sample_path) as data:
                        label = data["grid"]
                        if box_origin_world is None:
                            box_origin_world = np.asarray(data["origin"], dtype=np.float32).reshape(3)
                            voxel_size_world = np.asarray(data["voxel_size"], dtype=np.float32).reshape(3)
                            box_shape_zyx = np.asarray(label.shape[-3:], dtype=np.int64)
                    if self.class_mapping is not None:
                        label = class_mapping(label, self.class_mapping)
                else:
                    # 注意就目前来说, 所有数据都是 CDHW 的形状
                    with np.load(sample_path) as data:
                        _grid = data["grid"]
                        if box_origin_world is None:
                            box_origin_world = np.asarray(data["origin"], dtype=np.float32).reshape(3)
                            voxel_size_world = np.asarray(data["voxel_size"], dtype=np.float32).reshape(3)
                            box_shape_zyx = np.asarray(_grid.shape[-3:], dtype=np.int64)
                    # NOTE: 如果data_folder_names[k] 含有字段"emdb", 那么就对它做归一化(看成密度图或差图)
                    if "emdb" in data_folder_name:
                        _grid = (_grid - np.mean(_grid)) / (np.std(_grid) + 1e-8)
                    grid.append(_grid)
            grid = np.concatenate(grid, axis=0)       # np.ndarray, (C, D, H, W), float, 拼接后的完整特征网格

            if self.sample_root_path is None:
                raise ValueError("MyDatasets 现在要求显式传入 sample_root_path，用于几何 hardmask 生成")
            if box_origin_world is None or voxel_size_world is None or box_shape_zyx is None:
                raise RuntimeError(f"[MyDatasets] 样本缺少 BOX 几何元信息: {sample_name}")

            # str, 标量, 从 BOX 样本名解析出的结构 ID
            pdb_id = _parse_box_sample_name(sample_name)
            atoms_path = Path(self.sample_root_path) / pdb_id / "atoms.npz"
            if not atoms_path.exists():
                raise FileNotFoundError(f"[MyDatasets] 未找到 atoms.npz: {atoms_path}")
            with np.load(atoms_path) as data:
                atom_coords_world = np.asarray(data["coords"], dtype=np.float32)

            # 生成 hardmask: 基于原子几何位置写入 home voxel occupancy
            hardmask = build_hardmask_from_world_coordinates(
                atom_coords_world=atom_coords_world,
                box_origin_world=box_origin_world,
                voxel_size_world=voxel_size_world,
                box_shape_zyx=box_shape_zyx,
            )

            # 数据增强与后处理 (random_rotation90_plus 接受 *x 变参, 对所有输入施加相同旋转)
            if self.istrain:
                grid, label, hardmask = random_rotation90_plus(grid, label, hardmask)
            grid = torch.tensor(grid, dtype=torch.float32)
            label = torch.tensor(np.round(label).astype(np.int64), dtype=torch.int64).squeeze(0)               # 1DHW --> DHW
            hardmask = torch.tensor(hardmask, dtype=torch.int64)           # (D, H, W), int64, 取值0或1
            return grid, label, hardmask
            
        except Exception as e:
            # 捕获异常并打印详细信息，防止 Dataloader 子进程死掉而没有任何提示
            import traceback
            error_msg = f"\n[Dataset Error] Error loading sample index={index}, file={self.total_sample[index] if hasattr(self, 'total_sample') else 'Unknown'}\n{traceback.format_exc()}\n"
            print(error_msg, flush=True)
            raise RuntimeError(error_msg)


if __name__ == "__main__":
    pass  # 将要补充简单测试



def class_mapping(label: np.ndarray, mapping: list[int]) -> np.ndarray:
    """
    将标签中的类别 ID 进行映射
    
    Args:
        - label: np.ndarray, (D, H, W), int64, 原始标签
        - mapping: list[int], 映射表, 索引为原始类别 ID, 值对应新的类别 ID
    
    Returns:
        - mapped_label: np.ndarray, (D, H, W), int64, 映射后的标签
    """
    mapped_label = np.zeros_like(label)
    for old_class_id, new_class_id in enumerate(mapping):
        mapped_label[label == old_class_id] = new_class_id
    return mapped_label


def _parse_box_sample_name(sample_name: str) -> str:
    """
    从 BOX 样本名中解析 `pdb_id`。

    Args:
        - sample_name: str, 标量, BOX 样本名

    Returns:
        - pdb_id: str, 小写结构 ID
    """
    matched = _BOX_SAMPLE_NAME_PATTERN.match(sample_name)
    if matched is None:
        raise ValueError(f"[MyDatasets] 非法 sample_name: {sample_name}")
    return matched.group("pdb_id").lower()
