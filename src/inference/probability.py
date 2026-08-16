"""把 Stage1 ligand logits 转为连续概率图。

主要入口:
    - `logits_to_probability`: 接受 PyTorch 或 NumPy 的单通道 voxel logits，使用数值稳定 sigmoid 返回 CPU float32 概率。
    - `postprocess_ligand_probability`: 校验模型来源并返回独立的连续概率数组。

该后处理同时服务完整图融合与居中推理。`receptor_hardmask` 仅为兼容既有调用
签名而保留，不再乘入概率图。
"""

from __future__ import annotations

from typing import Any
import torch
import numpy as np

from src.stage1_producers import FIND_MODEL_NAMES, STAGE1_MODEL_NAMES


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
    校验并复制连续概率；不使用 receptor hardmask 清零受体原子所在体素。

    输入参数:
        - probability: numeric, `(..., D, H, W)`，完整图或 BOX-local ZYX voxel 网格上的连续概率。
        - stage1_model_name: str, `STAGE1_MODEL_NAMES` 中的模型来源身份。
        - receptor_hardmask: np.ndarray | None, 兼容既有调用方的保留参数；当前不参与计算。

    输出:
        - processed: float32, (..., D, H, W), 与输入 shape 相同且与输入内存解耦的连续概率。
    """
    if stage1_model_name not in STAGE1_MODEL_NAMES:
        raise ValueError(f"未知 stage1_model_name={stage1_model_name!r}")
    # float32, (..., D, H, W), 与输入数组解耦的 producer 专属后处理结果。
    processed = np.asarray(probability, dtype=np.float32).copy()
    if not bool(np.all(np.isfinite(processed))):
        raise ValueError("probability 含非有限值")
    return processed
