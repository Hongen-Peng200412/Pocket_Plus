"""把 Stage1 ligand logits 转为概率，并应用 producer 专属 receptor hardmask。

主要入口:
    - `logits_to_probability`: 接受 PyTorch 或 NumPy 的单通道 voxel logits，使用数值稳定 sigmoid 返回 CPU float32 概率。
    - `postprocess_ligand_probability`: 对 Find 模型来源清零 receptor home voxel，对 `unet_c1` 保留原概率。

该后处理同时服务完整图融合与居中推理。模型始终输出 logits；hardmask 只作用于 sigmoid 概率，不改变模型特征或输入批次。
"""

from __future__ import annotations

from typing import Any
import torch
import numpy as np


# 需要清零 receptor home voxels 的两个正式 Find producer。
FIND_MODEL_NAMES: tuple[str, ...] = ("Find_0", "Find_1")
# 三个正式 Stage1 producer；`unet_c1` 不应用 receptor hardmask。
STAGE1_MODEL_NAMES: tuple[str, ...] = (*FIND_MODEL_NAMES, "unet_c1")


def logits_to_probability(voxel_logits_ligand: Any) -> np.ndarray:
    """
    把 voxel-only 入口返回的单通道 logits 转成 CPU float32 概率。

    输入参数:
        - voxel_logits_ligand: torch.Tensor | np.ndarray, (B, 1, D, H, W) 或 (B, D, H, W), 单通道 ligand logits；五维输入的 channel 维必须为 1。

    输出:
        - probability: float32, (B, D, H, W), 搬到 CPU 后的逐 voxel sigmoid 概率。
    """
    try:
        if torch.is_tensor(voxel_logits_ligand):
            # torch.Tensor, (B, 1, D, H, W) 或 (B, D, H, W), 从计算图分离并提升到 float32 的 logits。
            logits = voxel_logits_ligand.detach().to(dtype=torch.float32)
            if logits.ndim == 5:
                if int(logits.shape[1]) != 1:
                    raise ValueError("voxel ligand logits 的 channel 维必须为 1")
                logits = logits[:, 0]
            if logits.ndim != 4:
                raise ValueError("voxel ligand logits 必须为 (B,1,D,H,W) 或 (B,D,H,W)")
            return torch.sigmoid(logits).cpu().numpy().astype(np.float32, copy=False)
    except ImportError:
        pass

    # float32, (B, 1, D, H, W) 或 (B, D, H, W), NumPy 调用方的单通道 logits。
    logits_array = np.asarray(voxel_logits_ligand, dtype=np.float32)
    if logits_array.ndim == 5:
        if int(logits_array.shape[1]) != 1:
            raise ValueError("voxel ligand logits 的 channel 维必须为 1")
        logits_array = logits_array[:, 0]
    if logits_array.ndim != 4:
        raise ValueError("voxel ligand logits 必须为 (B,1,D,H,W) 或 (B,D,H,W)")
    # float32, (B, D, H, W), 使用正负 logits 分段公式计算的数值稳定 sigmoid 概率。
    probability = np.empty_like(logits_array, dtype=np.float32)
    # bool, (B, D, H, W), 选择直接计算 `1/(1+exp(-x))` 的非负 logits，避免负值分支中的指数溢出。
    positive = logits_array >= 0
    probability[positive] = 1.0 / (1.0 + np.exp(-logits_array[positive]))
    exp_logits = np.exp(logits_array[~positive])
    probability[~positive] = exp_logits / (1.0 + exp_logits)
    return probability


def postprocess_ligand_probability(
    probability: np.ndarray,
    stage1_model_name: str,
    receptor_hardmask: np.ndarray | None,
) -> np.ndarray:
    """
    若 receptor_hardmask 不为None: processed *= np.logical_not(hardmask).astype(np.float32)

    输入参数:
        - probability: numeric, `(..., D, H, W)`，完整图或 BOX-local ZYX voxel 网格上的连续概率。
        - stage1_model_name: str, `STAGE1_MODEL_NAMES` 中的模型来源身份。
        - receptor_hardmask: np.ndarray | None, `(D, H, W)`，与概率最后三维对齐的 receptor home-voxel 布尔 mask；Find 必须提供，True 表示最终概率清零，unet_c1 不读取。

    输出:
        - processed: float32, (..., D, H, W), 与输入 shape 相同且与输入内存解耦的概率；Find hardmask 位置严格为 0。
    """
    if stage1_model_name not in STAGE1_MODEL_NAMES:
        raise ValueError(f"未知 stage1_model_name={stage1_model_name!r}")
    # float32, (..., D, H, W), 与输入数组解耦的 producer 专属后处理结果。
    processed = np.asarray(probability, dtype=np.float32).copy()
    if not bool(np.all(np.isfinite(processed))):
        raise ValueError("probability 含非有限值")
    if stage1_model_name in FIND_MODEL_NAMES:
        if receptor_hardmask is None:
            raise ValueError("两个 Find 的 probability 后处理必须显式提供 receptor_hardmask")
        # bool, (D, H, W), 完整图或当前 centered BOX 的 receptor home voxels；广播到全部前导维。
        hardmask = np.asarray(receptor_hardmask, dtype=np.bool_)
        if hardmask.shape != processed.shape[-3:]:
            raise ValueError("receptor_hardmask 必须与 probability 最后三维同 shape")
        processed *= np.logical_not(hardmask).astype(np.float32)
    return processed
