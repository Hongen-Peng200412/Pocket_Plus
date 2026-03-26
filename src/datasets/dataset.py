import json
import os
import random
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from utils.network_tools import random_rotation90_plus


torch.manual_seed(114514)
random.seed(114514)
np.random.seed(114514)


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

        data_folder_names: list[str] = ["emdb_BOX", "pdb_feature_BOX", "pdb_label_BOX"],    #NOTE: 之后固定下来: "pdb_label_BOX" 当作标签, 别的都当做特征的一部分进行拼接
        class_folder_names: list[str] = ["metal_ion", "peptide", "nucleic", "small_molecule"], 
        class_mapping: list[int] = None, 
        **kwargs,
    ):
        self.all_data_path = all_data_path
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

            # int, 记录 EMDB 密度图占据的通道总数, 用于后续 hardmask 计算
            emdb_channels = 0

            for data_folder_name in self.data_folder_names:
                sample_path = os.path.join(self.all_data_path, data_folder_name, self.class_folder_names[k], sample_name+".npz")
                # NOTE: 如果data_folder_names[k] 含有字段"label", 那么就对它看成标签
                if "label" in data_folder_name:
                    with np.load(sample_path) as data:
                        label = data["grid"]
                    if self.class_mapping is not None:
                        label = class_mapping(label, self.class_mapping)
                else:
                    # 注意就目前来说, 所有数据都是 CDHW 的形状
                    with np.load(sample_path) as data:
                        _grid = data["grid"]
                    # NOTE: 如果data_folder_names[k] 含有字段"emdb", 那么就对它做归一化(看成密度图或差图)
                    if "emdb" in data_folder_name:
                        _grid = (_grid - np.mean(_grid)) / (np.std(_grid) + 1e-8)
                        emdb_channels += _grid.shape[0]   # int, 累加 EMDB 通道数(通常为1)
                    grid.append(_grid)
            grid = np.concatenate(grid, axis=0)       # np.ndarray, (C, D, H, W), float, 拼接后的完整特征网格

            # 生成 hardmask: 判断 pdb_feature 部分是否全零 (全零 <--> 该体素无原子)
            hardmask = return_hardmask(grid, emdb_channels=emdb_channels)   # np.ndarray, (D, H, W), int64, 取值0或1

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


def return_hardmask(grid: np.ndarray, emdb_channels: int = 1) -> np.ndarray:
    """
    NOTE: 约定, 密度图类的特征占据在特征列表(data_folder_name)的位置, 属于最前部分 .
    接受 Mydatasets 里面加载完成的 grid, 按照针对 grid 特征向量的堆叠逻辑
    (Pocket\Make_Data\PDB_processor\features 和 Pocket\processedPDB_EMDB_binder\bind.py),
    返回 hardmask, 供 loss 使用.

    原理:
        grid 的前 emdb_channels 个通道是 EMDB 密度图(经过 z-score 归一化, 全零不代表无原子);
        剩余通道是 pdb_feature(来自 bind_AtomsFeature_to_EMDB, 未归一化, 体素全零 ⟺ 无原子).
        因此只需检查 grid[emdb_channels:] 在 channel 轴上是否存在非零值.

    输入参数 / Args:
        - grid: np.ndarray, (C, D, H, W), float32, 拼接后的完整特征网格.
                C = emdb_channels + pdb_feature_channels.
        - emdb_channels: int, 默认1. EMDB 密度图占据的前若干个通道数.(根据 data_folder_names 中有 "emdb" 字段的文件夹决定占据的通道数)

    输出 / Return:
        - hardmask: np.ndarray, (D, H, W), int64, 取值为0或1.
                    某个体素处取值为 0 ⟺ 这个体素处没有原子.

    意义: 关注先验信息"这个体素处没有原子那么一定不是口袋",
          所以让计算损失时忽略没有原子的体素.
    """
    # 增加简易校验：如果网格通道数甚至都不到 emdb_channels，肯定有问题
    if grid.shape[0] <= emdb_channels:
        raise ValueError(f"[return_hardmask] grid 的通道数({grid.shape[0]})不够或未包含其余特征，emdb_channels={emdb_channels}")

    # np.ndarray, (pdb_feature_channels, D, H, W), float32
    # 取 pdb_feature 部分 (跳过前 emdb_channels 个 EMDB 密度图通道)
    pdb_features = grid[emdb_channels:]
    # np.ndarray, (D, H, W), bool
    # 对 channel 轴(axis=0)做 any: 只要有一个通道非零, 说明该体素有原子存在
    has_atom = np.any(pdb_features != 0, axis=0)
    # np.ndarray, (D, H, W), int64, 取值0或1
    hardmask = has_atom.astype(np.int64)
    return hardmask