"""Stage1 ligand logits 的 sigmoid 与 producer-specific hardmask 后处理。"""

from __future__ import annotations

from typing import Any

import numpy as np


FIND_MODEL_NAMES: tuple[str, ...] = ("Find_0", "Find_1")
STAGE1_MODEL_NAMES: tuple[str, ...] = (*FIND_MODEL_NAMES, "unet_c1")


def logits_to_probability(voxel_logits_ligand: Any) -> np.ndarray:
    """
    把 voxel-only 入口返回的单通道 logits 转成 CPU float32 概率。

    输入参数:
        - voxel_logits_ligand: torch.Tensor | np.ndarray, (B,1,D,H,W) 或 (B,D,H,W)

    输出:
        - probability: np.ndarray, (B,D,H,W), float32，逐元素 sigmoid 后概率
    """
    try:
        import torch

        if torch.is_tensor(voxel_logits_ligand):
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

    logits_array = np.asarray(voxel_logits_ligand, dtype=np.float32)
    if logits_array.ndim == 5:
        if int(logits_array.shape[1]) != 1:
            raise ValueError("voxel ligand logits 的 channel 维必须为 1")
        logits_array = logits_array[:, 0]
    if logits_array.ndim != 4:
        raise ValueError("voxel ligand logits 必须为 (B,1,D,H,W) 或 (B,D,H,W)")
    probability = np.empty_like(logits_array, dtype=np.float32)
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
    对融合后或 centered sigmoid 概率应用冻结的 producer-specific 规则。

    输入参数:
        - probability: np.ndarray, (...,D,H,W), 连续概率
        - stage1_model_name: str, `Find_0`、`Find_1` 或 `unet_c1`
        - receptor_hardmask: np.ndarray | None, 与最后三维相同的 receptor home-voxel
          bool mask；两个 Find 必须提供，unet_c1 不读取

    输出:
        - processed: np.ndarray, 与输入同 shape 的 float32 概率；Find hardmask 位置为 0
    """
    if stage1_model_name not in STAGE1_MODEL_NAMES:
        raise ValueError(f"未知 stage1_model_name={stage1_model_name!r}")
    processed = np.asarray(probability, dtype=np.float32).copy()
    if not bool(np.all(np.isfinite(processed))):
        raise ValueError("probability 含非有限值")
    if stage1_model_name in FIND_MODEL_NAMES:
        if receptor_hardmask is None:
            raise ValueError("两个 Find 的 probability 后处理必须显式提供 receptor_hardmask")
        hardmask = np.asarray(receptor_hardmask, dtype=np.bool_)
        if hardmask.shape != processed.shape[-3:]:
            raise ValueError("receptor_hardmask 必须与 probability 最后三维同 shape")
        processed *= np.logical_not(hardmask).astype(np.float32)
    return processed
