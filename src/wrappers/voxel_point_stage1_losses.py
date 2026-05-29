from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from src.modules.losses import AdaptiveClassificationCompositeLoss, UnifiedCompositeLoss


@dataclass(frozen=True)
class LossTerm:
    """
    单个损失分支的数值、权重与日志值。

    输入参数:
        - name: str, 损失分支名; 对外日志使用 atom/receptor/voxel_ligand/ligand_sparse_refine
        - value: torch.Tensor, 标量, 原始损失值
        - weight: float, Python 标量, 总损失中的静态权重
        - logged_value: torch.Tensor, 标量, detach 后用于日志记录的损失值
    """

    name: str
    value: torch.Tensor
    weight: float
    logged_value: torch.Tensor


def loss_output_to_tensor(loss_out: torch.Tensor | Mapping[str, Any]) -> torch.Tensor:
    """
    将损失模块输出规范化为标量 tensor。

    输入参数:
        - loss_out: torch.Tensor 或 Mapping[str, Any], 损失模块返回值; Mapping 必须包含 loss 或 total_loss

    输出:
        - loss: torch.Tensor, 标量, 用于反向传播的损失值
    """
    if isinstance(loss_out, torch.Tensor):
        return loss_out
    if "loss" in loss_out:
        return loss_out["loss"]
    if "total_loss" in loss_out:
        return loss_out["total_loss"]
    raise KeyError("loss module output mapping must contain 'loss' or 'total_loss'.")


def compute_atom_loss_term(
    *,
    outputs: Mapping[str, Any],
    batch: Mapping[str, Any],
    loss_module: nn.Module,
    weight: float,
) -> LossTerm:
    """
    计算 atom 分支损失项。

    输入参数:
        - outputs: Mapping[str, Any], backbone 输出; 包含 atom_logits, 可包含 atom_target/atom_valid_mask
        - batch: Mapping[str, Any], 当前 batch; 包含 atom_label, 可包含 atom_valid_mask
        - loss_module: nn.Module, atom 损失模块
        - weight: float, atom 损失权重

    输出:
        - loss_term: LossTerm, atom 分支损失项
    """
    # torch.Tensor, (sumN, C_atom), 原子级预测 logits
    atom_logits = outputs["atom_logits"]
    # torch.Tensor, (sumN,), 原子级真值标签
    atom_target = outputs.get("atom_target", batch["atom_label"])
    # torch.Tensor | None, (sumN,), 原子有效掩码
    atom_valid_mask = outputs.get("atom_valid_mask", batch.get("atom_valid_mask"))
    if atom_logits.shape[0] != atom_target.shape[0]:
        raise RuntimeError(
            "Atom supervision shape mismatch before loss: "
            f"atom_logits.shape={tuple(atom_logits.shape)}, "
            f"atom_target.shape={tuple(atom_target.shape)}"
        )
    if atom_valid_mask is not None and atom_valid_mask.shape[0] != atom_target.shape[0]:
        raise RuntimeError(
            "Atom valid-mask shape mismatch before loss: "
            f"atom_valid_mask.shape={tuple(atom_valid_mask.shape)}, "
            f"atom_target.shape={tuple(atom_target.shape)}"
        )

    if isinstance(loss_module, (UnifiedCompositeLoss, AdaptiveClassificationCompositeLoss)):
        loss_out = loss_module(logits=atom_logits, target=atom_target, hardmask=atom_valid_mask)
    else:
        raise RuntimeError("loss_module must be an instance of UnifiedCompositeLoss or AdaptiveClassificationCompositeLoss.")
    # torch.Tensor, 标量, atom 原始损失
    value = loss_output_to_tensor(loss_out)
    return LossTerm(name="atom", value=value, weight=float(weight), logged_value=value.detach())


def compute_receptor_loss_term(
    *,
    outputs: Mapping[str, Any],
    batch: Mapping[str, Any],
    loss_module: nn.Module,
    weight: float,
) -> LossTerm | None:
    """
    计算 receptor 对外语义的体素辅助监督损失项。

    输入参数:
        - outputs: Mapping[str, Any], backbone 输出; 可包含 voxel_logits_aux
        - batch: Mapping[str, Any], 当前 batch; 包含 voxel_label/hardmask/voxel_valid_mask
        - loss_module: nn.Module, 体素辅助损失模块
        - weight: float, 体素辅助损失权重

    输出:
        - loss_term: LossTerm | None, receptor 分支损失项; 未产出 voxel_logits_aux 时为 None
    """
    # torch.Tensor | None, (B, C_aux, D, H, W), 体素辅助预测 logits
    voxel_logits_aux = outputs.get("voxel_logits_aux")
    if voxel_logits_aux is None:
        return None

    # torch.Tensor, (B, D, H, W), 体素级真值标签
    voxel_target = batch["voxel_label"]
    # torch.Tensor, (B, 1, D, H, W), 几何 hardmask
    hardmask = batch["hardmask"]
    # torch.Tensor, (B, 1, D, H, W), 边界有效掩码
    voxel_valid_mask = batch["voxel_valid_mask"]
    if isinstance(loss_module, (UnifiedCompositeLoss, AdaptiveClassificationCompositeLoss)):
        loss_out = loss_module(
            logits=voxel_logits_aux,
            target=voxel_target,
            hardmask=hardmask,
            valid_mask=voxel_valid_mask,
        )
    else:
        raise RuntimeError("loss_module must be an instance of UnifiedCompositeLoss or AdaptiveClassificationCompositeLoss.")
    # torch.Tensor, 标量, receptor 原始损失
    value = loss_output_to_tensor(loss_out)
    return LossTerm(name="receptor", value=value, weight=float(weight), logged_value=value.detach())


def compute_voxel_ligand_loss_term(
    *,
    outputs: Mapping[str, Any],
    batch: Mapping[str, Any],
    loss_module: nn.Module,
    weight: float,
) -> LossTerm | None:
    """
    计算 dense voxel ligand 占据损失项。

    输入参数:
        - outputs: Mapping[str, Any], backbone 输出; 可包含 voxel_logits_ligand
        - batch: Mapping[str, Any], 当前 batch; 可包含 ligand_dist_map, 必须包含 voxel_valid_mask
        - loss_module: nn.Module, voxel ligand 损失模块
        - weight: float, voxel ligand 损失权重

    输出:
        - loss_term: LossTerm | None, voxel_ligand 分支损失项; 缺少 logits 或 ligand_dist_map 时为 None
    """
    # torch.Tensor | None, (B, C_ligand, D, H, W), ligand 预测 logits
    voxel_logits_ligand = outputs.get("voxel_logits_ligand")
    if voxel_logits_ligand is None:
        return None
    # torch.Tensor | None, (B, D, H, W), ligand 距离图
    ligand_dist_map = batch.get("ligand_dist_map")
    if ligand_dist_map is None:
        return None

    # torch.Tensor, (B, 1, D, H, W), 边界有效掩码
    voxel_valid_mask = batch["voxel_valid_mask"]
    if isinstance(loss_module, (UnifiedCompositeLoss, AdaptiveClassificationCompositeLoss)):
        loss_out = loss_module(
            logits=voxel_logits_ligand,
            target=None,
            hardmask=None,
            valid_mask=voxel_valid_mask,
            ligand_dist_map=ligand_dist_map,
        )
    else:
        raise RuntimeError("loss_module must be an instance of UnifiedCompositeLoss or AdaptiveClassificationCompositeLoss.")
    # torch.Tensor, 标量, voxel ligand 原始损失
    value = loss_output_to_tensor(loss_out)
    return LossTerm(name="voxel_ligand", value=value, weight=float(weight), logged_value=value.detach())


def compute_sparse_refine_loss_term(
    *,
    logits_C: torch.Tensor,
    target_C: torch.Tensor,
    valid_C: torch.Tensor,
    loss_module: AdaptiveClassificationCompositeLoss,
    weight: float,
    effective_weight: torch.Tensor,
) -> tuple[LossTerm, torch.Tensor]:
    """
    计算 C 级 sparse refine 损失项。

    输入参数:
        - logits_C: torch.Tensor, (sumC, C_ligand), C 级 refined logits
        - target_C: torch.Tensor, (sumC,), C 级 hard-label target
        - valid_C: torch.Tensor, (sumC,), C 级有效监督掩码
        - loss_module: AdaptiveClassificationCompositeLoss, C 级分类复合损失
        - weight: float, 配置中的最终损失权重
        - effective_weight: torch.Tensor, 标量, schedule 后当前 step 实际使用权重

    输出:
        - loss_term: LossTerm, ligand_sparse_refine 分支损失项
        - effective_weight: torch.Tensor, 标量, schedule 后当前 step 实际使用权重
    """
    loss_out = loss_module(logits=logits_C, target=target_C, hardmask=valid_C)
    # torch.Tensor, 标量, sparse refine 原始损失
    value = loss_output_to_tensor(loss_out)
    return (
        LossTerm(
            name="ligand_sparse_refine",
            value=value,
            weight=float(weight),
            logged_value=value.detach(),
        ),
        effective_weight.detach(),
    )
