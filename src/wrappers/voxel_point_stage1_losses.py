"""把 Stage1 模型输出和 batch 监督字段转换为可加权的标量损失项。

``voxel_point_stage1.py`` 调用本模块的 ``compute_*_loss_term`` 函数，再按 :class:`LossTerm.weight` 汇总总损失。分类分支把 Focal、Dice 等具体计算交给配置实例化的复合损失模块；本模块只选择预测、监督和有效掩码。配体距离分支对完整 80³ BOX 的反距离值计算逐体素平均 MSE。
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from src.modules.losses import AdaptiveClassificationCompositeLoss, LigandSparseRefineDeltaLoss, UnifiedCompositeLoss

# 工具：把不同损失模块的返回值统一成标量 tensor。
@dataclass(frozen=True)
class LossTerm:
    """保存一个损失分支的未加权值、静态权重和日志值。

    字段:
        - name: str；损失分支名称，例如 ``atom``、``receptor`` 或 ``voxel_ligand``。
        - value: torch.Tensor；参与反向传播的标量未加权损失。
        - weight: float；总损失组合时乘用的静态系数。
        - logged_value: torch.Tensor；与 ``value`` 数值相同但从计算图分离的标量日志值。
    """

    name: str
    value: torch.Tensor
    weight: float
    logged_value: torch.Tensor

def loss_output_to_tensor(loss_out: torch.Tensor | Mapping[str, Any]) -> torch.Tensor:
    """从损失模块返回值提取参与反向传播的标量 tensor。

    输入参数:
        - loss_out: torch.Tensor | Mapping[str, Any]；直接 tensor 原样返回；mapping 必须包含 ``loss`` 或 ``total_loss`` 键。

    返回值:
        - loss: torch.Tensor；损失模块约定的标量反向传播值；本函数不 detach、不加权、不改变设备。

    失败语义:
        - mapping 同时缺少 ``loss`` 和 ``total_loss`` 时抛出 ``KeyError``。
    """
    if isinstance(loss_out, torch.Tensor):
        return loss_out
    if "loss" in loss_out:
        return loss_out["loss"]
    if "total_loss" in loss_out:
        return loss_out["total_loss"]
    raise KeyError("loss module output mapping must contain 'loss' or 'total_loss'.")




# 监督分支：受体原子、P anchor、体素和 sparse refine。
def compute_atom_loss_term(
    *,
    outputs: Mapping[str, Any],
    batch: Mapping[str, Any],
    loss_module: nn.Module,
    weight: float,
) -> LossTerm:
    """计算真实受体原子 binding 分类损失项。

    输入参数:
        - outputs: Mapping[str, Any]；读取 ``atom_logits`` ``(N_A, C_atom)``，可选读取 ``atom_target`` 和 ``atom_is_in_core_box``。
        - batch: Mapping[str, Any]；回退读取 ``atom_label`` ``(N_A,)`` 和 ``atom_is_in_core_box`` ``(N_A,)``。
        - loss_module: nn.Module；必须是 ``UnifiedCompositeLoss`` 或 ``AdaptiveClassificationCompositeLoss``。
        - weight: float；该分支的静态总损失权重。

    返回值:
        - loss_term: LossTerm；保存未加权原子损失和日志值。

    失败语义:
        - logits、target 或有效掩码第 0 维不对齐，或损失模块类型不支持时抛出异常。
    """
    # torch.Tensor (N_A, C_atom)；拼接后局部受体原子的分类 logits。
    atom_logits = outputs["atom_logits"]
    # torch.Tensor (N_A,)；与 atom_logits 第 0 维逐局部原子对齐的类别标签。
    atom_target = outputs.get("atom_target", batch["atom_label"])
    # torch.Tensor bool (N_A,)；True 表示原子位于核心 BOX 并参加原子分类损失。
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
    # torch.Tensor 标量；未乘静态 weight 的 atom 原始损失。
    value = loss_output_to_tensor(loss_out)
    return LossTerm(name="atom", value=value, weight=float(weight), logged_value=value.detach())

# 虚拟 P anchor 是否属于配体区域。
def compute_pseudo_loss_term(
    *,
    outputs: Mapping[str, Any],
    loss_module: nn.Module,
    weight: float,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
) -> LossTerm:
    """计算 P anchor 的配体区域归属损失项。

    输入参数:
        - outputs: Mapping[str, Any]；读取 ``pseudo_logits`` ``(N_P, C_pseudo)``。
        - loss_module: nn.Module；必须是支持 ``hardmask`` 的统一分类损失模块。
        - weight: float；P anchor 分支静态权重。
        - target: torch.Tensor ``(N_P,)``；P anchor 配体区域硬标签。
        - valid_mask: torch.Tensor bool ``(N_P,)``；P anchor 有效监督掩码。

    返回值:
        - loss_term: LossTerm；P anchor 未加权分类损失。

    失败语义:
        - logits 与 target 第 0 维不一致，或损失模块类型不支持时抛出异常。
    """
    # torch.Tensor (N_P, C_pseudo)；每个 P anchor 的配体区域分类 logits。
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
    # torch.Tensor 标量；未乘静态 weight 的 pseudo 原始损失。
    value = loss_output_to_tensor(loss_out)
    return LossTerm(name="pseudo", value=value, weight=float(weight), logged_value=value.detach())

# 体素是否属于受体结合区域。
def compute_receptor_loss_term(
    *,
    outputs: Mapping[str, Any],
    batch: Mapping[str, Any],
    loss_module: nn.Module,
    weight: float,
) -> LossTerm | None:
    """计算受体结合区域的全体素辅助监督损失。

    输入参数:
        - outputs: Mapping[str, Any]；可包含 ``voxel_logits_aux`` ``(B, C_aux, D, H, W)``。
        - batch: Mapping[str, Any]；提供 ``voxel_label`` ``(B, D, H, W)`` 和 ``hardmask`` ``(B, D, H, W)``。
        - loss_module: nn.Module；统一分类复合损失模块。
        - weight: float；receptor 分支静态权重。

    返回值:
        - loss_term: LossTerm | None；没有受体体素 logits 时为 ``None``，否则返回未加权损失。
    """
    # torch.Tensor (B, C_aux, D, H, W) 或 None；受体结合区域体素预测 logits。
    voxel_logits_aux = outputs.get("voxel_logits_aux")
    if voxel_logits_aux is None:
        return None

    # torch.Tensor bool (B, D, H, W)；每个 ZYX 体素的受体 binding 标签。
    voxel_target = batch["voxel_label"]
    # torch.Tensor bool (B, D, H, W)；受体原子占据掩码；损失模块只在掩码覆盖体素上计算。
    hardmask = batch["hardmask"]
    if isinstance(loss_module, (UnifiedCompositeLoss, AdaptiveClassificationCompositeLoss)):
        loss_out = loss_module(
            logits=voxel_logits_aux,
            target=voxel_target,
            hardmask=hardmask,
        )
    else:
        raise RuntimeError("loss_module must be an instance of UnifiedCompositeLoss or AdaptiveClassificationCompositeLoss.")
    # torch.Tensor 标量；未乘静态 weight 的 receptor 原始损失。
    value = loss_output_to_tensor(loss_out)
    return LossTerm(name="receptor", value=value, weight=float(weight), logged_value=value.detach())

# 体素是否属于配体区域。
def compute_voxel_ligand_loss_term(
    *,
    outputs: Mapping[str, Any],
    batch: Mapping[str, Any],
    loss_module: nn.Module,
    weight: float,
) -> LossTerm | None:
    """计算 dense 配体区域体素损失。

    输入参数:
        - outputs: Mapping[str, Any]；可包含 ``voxel_logits_ligand`` ``(B, C_ligand, D, H, W)``。
        - batch: Mapping[str, Any]；AdaLigand V3 提供 ``ligand_area_target`` ``(B, D, H, W)``；旧配置可提供 ``ligand_dist_map``。
        - loss_module: nn.Module；统一分类复合损失模块。
        - weight: float；voxel ligand 分支静态权重。

    返回值:
        - loss_term: LossTerm | None；缺少 logits 或任何可用 target 时为 ``None``。

    兼容边界:
        - ``ligand_area_target`` 优先；只有缺少该字段时才沿用旧 ``ligand_dist_map`` 派生路径。
    """
    # torch.Tensor (B, C_ligand, D, H, W) 或 None；配体区域体素预测 logits。
    voxel_logits_ligand = outputs.get("voxel_logits_ligand")
    if voxel_logits_ligand is None:
        return None
    # torch.Tensor bool (B, D, H, W) 或 None；所有 occurrence 并集在当前 80³ BOX 中的 ZYX 裁剪。
    ligand_area_target = batch.get("ligand_area_target")
    if ligand_area_target is not None:
        if isinstance(loss_module, (UnifiedCompositeLoss, AdaptiveClassificationCompositeLoss)):
            loss_out = loss_module(
                logits=voxel_logits_ligand,
                target=ligand_area_target,
                hardmask=None,
                ligand_dist_map=None,
            )
        else:
            raise RuntimeError("loss_module must be an instance of UnifiedCompositeLoss or AdaptiveClassificationCompositeLoss.")
        value = loss_output_to_tensor(loss_out)
        return LossTerm(name="voxel_ligand", value=value, weight=float(weight), logged_value=value.detach())

    # 兼容旧 Pocket_Plus 配置的 ligand_dist_map 派生 target；AdaLigand V3 使用上面的 ligand_area_target。
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
    # torch.Tensor 标量；未乘静态 weight 的 voxel ligand 原始损失。
    value = loss_output_to_tensor(loss_out)
    return LossTerm(name="voxel_ligand", value=value, weight=float(weight), logged_value=value.detach())


def compute_mainchain_class_loss_term(
    *,
    outputs: Mapping[str, Any],
    batch: Mapping[str, Any],
    loss_module: nn.Module,
    weight: float,
    polymer_name: str,
) -> LossTerm | None:
    """计算蛋白或核酸主链原子类别的全体素复合分类损失. 

    输入参数:
        - outputs: 模型输出字典; 读取 ``voxel_logits_{polymer_name}``. 
        - batch: Stage1 批次字典; 读取 ``{polymer_name}_mainchain_target``. 
        - loss_module: 多分类 Focal 与 Dice 复合损失. 
        - weight: 该分支进入总损失的静态系数. 
        - polymer_name: ``protein`` 或 ``nucleic``, 同时决定预测和监督字段名. 

    输出:
        - loss_term: 未启用对应预测头时为 None; 否则保存全体素平均分类损失. 

    类别 0 是背景. 每个正类分别作为唯一前景计算 Dice, 再在正类之间平均; 
    Focal 与 Dice 的具体系数由 ``loss_module`` 保存. 
    """

    if polymer_name not in {"protein", "nucleic"}:
        raise ValueError("polymer_name 只允许 protein 或 nucleic。")
    # torch.Tensor (B, C_polymer, D, H, W) 或 None；类别轴顺序由 auxiliary_supervision.py 的固定映射定义。
    logits = outputs.get(f"voxel_logits_{polymer_name}")
    if logits is None:
        return None
    # torch.Tensor int64 (B, D, H, W)；0 是背景，正数类别编号与 logits 的第 1 维逐项对应。
    target = batch[f"{polymer_name}_mainchain_target"]
    if not isinstance(loss_module, AdaptiveClassificationCompositeLoss):
        raise TypeError("主链类别损失必须使用 AdaptiveClassificationCompositeLoss。")
    value = loss_output_to_tensor(loss_module(logits=logits, target=target, hardmask=None))
    return LossTerm(
        name=f"{polymer_name}_mainchain",
        value=value,
        weight=float(weight),
        logged_value=value.detach(),
    )


def compute_ligand_distance_loss_term(
    *,
    outputs: Mapping[str, Any],
    batch: Mapping[str, Any],
    weight: float,
) -> LossTerm | None:
    """计算最近配体反距离变换的全体素平均 MSE。

    输入参数:
        - outputs: Mapping[str, Any]；必须在 ``voxel_logits_distance`` 提供 torch.Tensor ``(B, 1, D, H, W)``，没有该键或值为 ``None`` 时返回 ``None``。
        - batch: Mapping[str, Any]；必须在 ``ligand_inverse_distance_target`` 提供 torch.Tensor ``(B, D, H, W)``；V3 Dataset 由有限、非负的 80³ 距离裁块计算 ``1 / (1 + distance_Å)``。
        - weight: float；wrapper 汇总该分支时使用的静态权重。

    返回值:
        - loss_term: LossTerm | None；sigmoid 后 logits 与目标的逐体素平均 MSE，或在没有距离输出时为 ``None``。

    失败语义:
        - logits 不是单通道五维张量，或目标形状不等于 ``(B, D, H, W)`` 时抛出 ``ValueError``。
    """

    logits = outputs.get("voxel_logits_distance")
    if logits is None:
        return None
    # torch.Tensor float (B, D, H, W)；每个体素中心到最近配体原子的反距离监督值。
    target = batch["ligand_inverse_distance_target"].to(device=logits.device, dtype=logits.dtype)
    if logits.ndim != 5 or logits.shape[1] != 1 or logits[:, 0].shape != target.shape:
        raise ValueError(
            "配体距离预测与监督形状不一致: "
            f"logits={tuple(logits.shape)}, target={tuple(target.shape)}。"
        )
    value = F.mse_loss(torch.sigmoid(logits[:, 0]), target, reduction="mean")
    return LossTerm(
        name="ligand_distance",
        value=value,
        weight=float(weight),
        logged_value=value.detach(),
    )

# sparse refine：在 C 候选轴上组合分类与可选排序损失。
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
    """计算 C 候选级 sparse refine 组合损失 ``L_cls + w_rank * L_rank``。

    输入参数:
        - logits_C: torch.Tensor ``(N_C, C_ligand)``；C 候选的 refined logits。
        - target_C: torch.Tensor ``(N_C,)``；C 候选的 hard label。
        - valid_C: torch.Tensor bool ``(N_C,)``；C 候选有效监督掩码。
        - loss_module: AdaptiveClassificationCompositeLoss；C 候选分类损失 ``L_cls``。
        - weight: float；配置中的最终静态损失权重。
        - effective_weight: torch.Tensor 标量；当前 schedule step 的有效权重，原样返回其 detach 值用于日志/状态。
        - delta_loss_module: LigandSparseRefineDeltaLoss | None；可选排序损失模块；为空时只计算 ``L_cls``。
        - base_prob_C: torch.Tensor | None ``(N_C,)``；candidate set 已 detach 的 base 概率；启用排序损失时由调用方提供。
        - batch_index_C: torch.Tensor | None ``(N_C,)``；每个 C 候选所属 BOX index；启用排序损失时由调用方提供。
        - w_rank: float；排序损失的静态组合系数。

    返回值:
        - loss_term: LossTerm；组合后的 ``ligand_sparse_refine`` 分支。
        - effective_weight: torch.Tensor 标量；schedule 后有效权重的 detach 值。
        - component_logs: dict[str, torch.Tensor]；detach 的分类和可选排序分量日志。

    组合语义:
        - ``delta_loss_module`` 为空或 ``w_rank <= 0`` 时只保留分类损失；否则在同一分支中追加 ``w_rank * L_rank``。
    """
    loss_out = loss_module(logits=logits_C, target=target_C, hardmask=valid_C)
    # torch.Tensor 标量；C 候选分类损失 L_cls。
    cls_value = loss_output_to_tensor(loss_out)
    # torch.Tensor 标量；当前 refine 分支组合后的损失。
    combined_value = cls_value
    # dict[str, torch.Tensor]；各分量从计算图分离后的日志值。
    component_logs = {"ligand_sparse_refine_cls": cls_value.detach()}
    if delta_loss_module is not None and float(w_rank) > 0.0:
        # torch.Tensor (N_C,)；单通道 refined logit 与 candidate set 的 base 概率逐候选对齐。
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
