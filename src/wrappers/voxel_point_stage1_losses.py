from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from src.modules.losses import AdaptiveClassificationCompositeLoss, LigandSparseRefineDeltaLoss, UnifiedCompositeLoss

# -------------------------------------------- 工具 --------------------------------------------
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




# -------------------------------------------- 四个实际损失的计算 --------------------------------------------
# 点————结合位点
def compute_atom_loss_term(
    *,
    outputs: Mapping[str, Any],
    batch: Mapping[str, Any],
    loss_module: nn.Module,
    weight: float,
) -> LossTerm:
    """
    计算最终 atom 分支损失项。

    输入参数:
        - outputs: Mapping[str, Any], backbone 输出; 包含 atom_logits、atom_target 与 atom_is_in_core_box
        - batch: Mapping[str, Any], 当前 batch; 包含 atom_label 与 atom_is_in_core_box
        - loss_module: nn.Module, atom 损失模块
        - weight: float, atom 损失权重

    输出:
        - loss_term: LossTerm, atom 分支损失项
    """
    # torch.Tensor, (sumN, C_atom), 原子级预测 logits
    atom_logits = outputs["atom_logits"]
    # torch.Tensor, (sumN,), 原子级真值标签
    atom_target = outputs.get("atom_target", batch["atom_label"])
    # torch.Tensor, (sumN,), bool, 原子是否落在 core box 内; 唯一 atom 监督掩码
    atom_core_mask = outputs.get("atom_is_in_core_box", batch["atom_is_in_core_box"])
    if atom_logits.shape[0] != atom_target.shape[0]:
        raise RuntimeError(
            "Atom supervision shape mismatch before loss: "
            f"atom_logits.shape={tuple(atom_logits.shape)}, "
            f"atom_target.shape={tuple(atom_target.shape)}"
        )
    if atom_core_mask is not None and atom_core_mask.shape[0] != atom_target.shape[0]:
        raise RuntimeError(
            "Atom valid-mask shape mismatch before loss: "
            f"atom_core_mask.shape={tuple(atom_core_mask.shape)}, "
            f"atom_target.shape={tuple(atom_target.shape)}"
        )

    if isinstance(loss_module, (UnifiedCompositeLoss, AdaptiveClassificationCompositeLoss)):
        loss_out = loss_module(logits=atom_logits, target=atom_target, hardmask=atom_core_mask)
    else:
        raise RuntimeError("loss_module must be an instance of UnifiedCompositeLoss or AdaptiveClassificationCompositeLoss.")
    # torch.Tensor, 标量, atom 原始损失
    value = loss_output_to_tensor(loss_out)
    return LossTerm(name="atom", value=value, weight=float(weight), logged_value=value.detach())

# 点————P(虚拟原子) ligand 区域归属
def compute_pseudo_loss_term(
    *,
    outputs: Mapping[str, Any],
    loss_module: nn.Module,
    weight: float,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
) -> LossTerm:
    """
    计算最终 P(虚拟原子) ligand 区域归属损失项。

    输入参数:
        - outputs: Mapping[str, Any], backbone 输出; 包含 pseudo_logits
        - loss_module: nn.Module, pseudo ligand 损失模块(单通道 sigmoid composite)
        - weight: float, pseudo 损失权重
        - target: torch.Tensor, (N_pseudo,), P 级 ligand 区域硬标签
        - valid_mask: torch.Tensor, (N_pseudo,), bool, P 级有效监督掩码

    输出:
        - loss_term: LossTerm, pseudo 分支损失项
    """
    # torch.Tensor, (N_pseudo, C_pseudo), P 级预测 logits
    pseudo_logits = outputs["pseudo_logits"]
    if pseudo_logits.shape[0] != target.shape[0]:
        raise RuntimeError(
            "Pseudo supervision shape mismatch before loss: "
            f"pseudo_logits.shape={tuple(pseudo_logits.shape)}, target.shape={tuple(target.shape)}"
        )
    if isinstance(loss_module, (UnifiedCompositeLoss, AdaptiveClassificationCompositeLoss)):
        loss_out = loss_module(logits=pseudo_logits, target=target, hardmask=valid_mask)
    else:
        raise RuntimeError("loss_module must be an instance of UnifiedCompositeLoss or AdaptiveClassificationCompositeLoss.")
    # torch.Tensor, 标量, pseudo 原始损失
    value = loss_output_to_tensor(loss_out)
    return LossTerm(name="pseudo", value=value, weight=float(weight), logged_value=value.detach())

# 体素————受体区域
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
        - batch: Mapping[str, Any], 当前 batch; 包含 voxel_label/hardmask
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
    if isinstance(loss_module, (UnifiedCompositeLoss, AdaptiveClassificationCompositeLoss)):
        loss_out = loss_module(
            logits=voxel_logits_aux,
            target=voxel_target,
            hardmask=hardmask,
        )
    else:
        raise RuntimeError("loss_module must be an instance of UnifiedCompositeLoss or AdaptiveClassificationCompositeLoss.")
    # torch.Tensor, 标量, receptor 原始损失
    value = loss_output_to_tensor(loss_out)
    return LossTerm(name="receptor", value=value, weight=float(weight), logged_value=value.detach())

# 体素————ligand区域
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
        - batch: Mapping[str, Any], 当前 batch; 可包含 ligand_dist_map
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

    if isinstance(loss_module, (UnifiedCompositeLoss, AdaptiveClassificationCompositeLoss)):
        loss_out = loss_module(
            logits=voxel_logits_ligand,
            target=None,
            hardmask=None,
            ligand_dist_map=ligand_dist_map,
        )
    else:
        raise RuntimeError("loss_module must be an instance of UnifiedCompositeLoss or AdaptiveClassificationCompositeLoss.")
    # torch.Tensor, 标量, voxel ligand 原始损失
    value = loss_output_to_tensor(loss_out)
    return LossTerm(name="voxel_ligand", value=value, weight=float(weight), logged_value=value.detach())

# refine loss
def compute_sparse_refine_loss_term(
    *,
    logits_C: torch.Tensor,
    target_C: torch.Tensor,
    valid_C: torch.Tensor,
    loss_module: AdaptiveClassificationCompositeLoss,
    weight: float,
    effective_weight: torch.Tensor,
    delta_loss_module: LigandSparseRefineDeltaLoss | None = None,
    base_prob_C: torch.Tensor | None = None,
    batch_index_C: torch.Tensor | None = None,
    w_rank: float = 0.0,
) -> tuple[LossTerm, torch.Tensor, dict[str, torch.Tensor]]:
    """
    计算 C 级 sparse refine 组合损失项: L_cls + w_rank·L_rank。

    两项 refine 损失共用同一 schedule(effective_weight)与同一静态 weight, 合成单个 ligand_sparse_refine term;
    delta_loss_module 为 None 或 w_rank=0 时退化为纯分类 L_cls(就是分类损失, AdaptiveClassificationCompositeLoss, focal+dice)。

    输入参数:
        - logits_C: torch.Tensor, (sumC, C_ligand), C 级 refined logits
        - target_C: torch.Tensor, (sumC,), C 级 hard-label target
        - valid_C: torch.Tensor, (sumC,), C 级有效监督掩码
        - loss_module: AdaptiveClassificationCompositeLoss, C 级分类复合损失(L_cls)
        - weight: float, 配置中的最终损失权重
        - effective_weight: torch.Tensor, 标量, schedule 后当前 step 实际使用权重
        - delta_loss_module: LigandSparseRefineDeltaLoss | None, ranking 损失模块; None 表示只算 L_cls
        - base_prob_C: torch.Tensor | None, (sumC,), candidate_set 已 detach 的 base 概率; delta 启用时必填
        - batch_index_C: torch.Tensor | None, (sumC,), 每个 C 行所属 BOX 索引; delta 启用时必填
        - w_rank: float, ranking 权重

    输出:
        - loss_term: LossTerm, ligand_sparse_refine 组合分支损失项
        - effective_weight: torch.Tensor, 标量, schedule 后当前 step 实际使用权重
        - component_logs: dict[str, torch.Tensor], 各分量 detach 日志值, 含 ligand_sparse_refine_cls; delta 启用时另含 *_rank
    """
    loss_out = loss_module(logits=logits_C, target=target_C, hardmask=valid_C)
    # torch.Tensor, 标量, L_cls 分类损失
    cls_value = loss_output_to_tensor(loss_out)
    # torch.Tensor, 标量, 组合后的 refine 总损失
    combined_value = cls_value
    # dict[str, torch.Tensor], 各分量 detach 日志值
    component_logs = {"ligand_sparse_refine_cls": cls_value.detach()}
    if delta_loss_module is not None and float(w_rank) > 0.0:
        # NOTE: torch.Tensor, (sumC,), 单通道 refined logit 与 base 概率(二分类取第 0 通道)
        delta_out = delta_loss_module(
            refined_logit=logits_C[:, 0],
            base_prob=base_prob_C,
            target=target_C,
            valid=valid_C.bool(),
            batch_index=batch_index_C,
        )
        combined_value = combined_value + float(w_rank) * delta_out["rank"]
        component_logs["ligand_sparse_refine_rank"] = delta_out["rank"].detach()
    return (
        LossTerm(
            name="ligand_sparse_refine",
            value=combined_value,
            weight=float(weight),
            logged_value=combined_value.detach(),
        ),
        effective_weight.detach(),
        component_logs,
    )
