"""把 Stage1 模型输出与批次监督字段转换为可加权的标量损失项. 

``voxel_point_stage1.py`` 调用本模块的 ``compute_*_loss_term`` 函数, 
再按 :class:`LossTerm.weight` 汇总总损失. 分类分支把具体 Focal 与 Dice
计算交给配置实例化的复合损失模块; 本模块负责选择预测、监督和有效掩码. 
配体距离分支单独对完整 BOX 的反距离值计算逐体素平均 MSE. 
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from src.modules.losses import AdaptiveClassificationCompositeLoss, LigandSparseRefineDeltaLoss, UnifiedCompositeLoss

# -------------------------------------------- 工具 --------------------------------------------
@dataclass(frozen=True)
class LossTerm:
    """
    单个损失分支的数值、权重与日志值. 

    输入参数:
        - name: str, 损失分支名称; 日志使用 atom、receptor、voxel_ligand 等固定名称. 
        - value: 标量张量, 参与反向传播的未加权损失. 
        - weight: float, 总损失使用的静态系数. 
        - logged_value: 标量张量, 从计算图分离后写入日志的未加权损失. 
    """

    name: str
    value: torch.Tensor
    weight: float
    logged_value: torch.Tensor

def loss_output_to_tensor(loss_out: torch.Tensor | Mapping[str, Any]) -> torch.Tensor:
    """
    将损失模块输出规范化为标量 tensor. 

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




# -------------------------------------------- 各监督分支 --------------------------------------------
# 受体原子是否属于结合区域. 
def compute_atom_loss_term(
    *,
    outputs: Mapping[str, Any],
    batch: Mapping[str, Any],
    loss_module: nn.Module,
    weight: float,
) -> LossTerm:
    """
    计算最终 atom 分支损失项. 

    输入参数:
        - outputs: Mapping[str, Any], backbone 输出; 包含 atom_logits、atom_target 与 atom_is_in_core_box
        - batch: Mapping[str, Any], 当前 batch; 包含 atom_label 与 atom_is_in_core_box
        - loss_module: nn.Module, atom 损失模块
        - weight: float, atom 损失权重

    输出:
        - loss_term: LossTerm, atom 分支损失项
    """
    # (N_atom, C_atom), 拼接受体原子的分类 logits. 
    atom_logits = outputs["atom_logits"]
    # (N_atom,), 与 atom_logits 第 0 维逐原子对齐的类别标签. 
    atom_target = outputs.get("atom_target", batch["atom_label"])
    # bool, (N_atom,), True 表示该受体原子位于核心 BOX 并参加原子分类损失. 
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

# 虚拟 P 锚点是否属于配体区域. 
def compute_pseudo_loss_term(
    *,
    outputs: Mapping[str, Any],
    loss_module: nn.Module,
    weight: float,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
) -> LossTerm:
    """
    计算最终 P(虚拟原子) ligand 区域归属损失项. 

    输入参数:
        - outputs: Mapping[str, Any], backbone 输出; 包含 pseudo_logits
        - loss_module: nn.Module, pseudo ligand 损失模块(单通道 sigmoid composite)
        - weight: float, pseudo 损失权重
        - target: torch.Tensor, (N_pseudo,), P 级 ligand 区域硬标签
        - valid_mask: torch.Tensor, (N_pseudo,), bool, P 级有效监督掩码

    输出:
        - loss_term: LossTerm, pseudo 分支损失项
    """
    # (N_pseudo, C_pseudo), 每个虚拟 P 锚点的配体区域分类 logits. 
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

# 体素是否属于受体结合区域. 
def compute_receptor_loss_term(
    *,
    outputs: Mapping[str, Any],
    batch: Mapping[str, Any],
    loss_module: nn.Module,
    weight: float,
) -> LossTerm | None:
    """
    计算 receptor 对外语义的体素辅助监督损失项. 

    输入参数:
        - outputs: Mapping[str, Any], backbone 输出; 可包含 voxel_logits_aux
        - batch: Mapping[str, Any], 当前 batch; 包含 voxel_label/hardmask
        - loss_module: nn.Module, 体素辅助损失模块
        - weight: float, 体素辅助损失权重

    输出:
        - loss_term: LossTerm | None, receptor 分支损失项; 未产出 voxel_logits_aux 时为 None
    """
    # (B, C_aux, D, H, W) 或 None, 受体结合区域体素预测 logits. 
    voxel_logits_aux = outputs.get("voxel_logits_aux")
    if voxel_logits_aux is None:
        return None

    # (B, D, H, W), 每个体素的受体结合区域标签. 
    voxel_target = batch["voxel_label"]
    # bool, (B, 1, D, H, W), 受体原子占据掩码; 仅掩码覆盖的体素参加该损失. 
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

# 体素是否属于配体区域. 
def compute_voxel_ligand_loss_term(
    *,
    outputs: Mapping[str, Any],
    batch: Mapping[str, Any],
    loss_module: nn.Module,
    weight: float,
) -> LossTerm | None:
    """
    计算 dense voxel ligand 占据损失项. 

    输入参数:
        - outputs: Mapping[str, Any], backbone 输出; 可包含 voxel_logits_ligand
        - batch: Mapping[str, Any], 当前 batch; AdaLigand 使用 ligand_area_target, 旧配置可含 ligand_dist_map
        - loss_module: nn.Module, voxel ligand 损失模块
        - weight: float, voxel ligand 损失权重

    输出:
        - loss_term: LossTerm | None, voxel_ligand 分支损失项; 缺少 logits 或任何 target 时为 None
    """
    # (B, C_ligand, D, H, W) 或 None, 配体区域体素预测 logits. 
    voxel_logits_ligand = outputs.get("voxel_logits_ligand")
    if voxel_logits_ligand is None:
        return None
    # bool, (B, D, H, W) 或 None, 所有配体 occurrence 并集在当前 80³ BOX 中的裁剪. 
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

    # 旧 Pocket_Plus 配置继续允许以 ligand_dist_map 派生 target; AdaLigand 配置不会进入此分支. 
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
    # (B, C_polymer, D, H, W) 或 None, 类别通道顺序由 auxiliary_supervision.py 固定. 
    logits = outputs.get(f"voxel_logits_{polymer_name}")
    if logits is None:
        return None
    # int64, (B, D, H, W), 0 为背景, 其余编号与 logits 的类别维一一对应. 
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
    """计算最近配体距离变换值的全体素平均 MSE. 

    模型输出先经 sigmoid 映射到 ``[0, 1]``, 再与
    ``ligand_inverse_distance_target = 1 / (1 + distance_Å)`` 比较。V3 Dataset
    要求实际 80³ 裁块中的距离有限且非负。
    """

    logits = outputs.get("voxel_logits_distance")
    if logits is None:
        return None
    # (B, D, H, W), 每个体素中心到最近配体原子的反距离监督值. 
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
    计算 C 级 sparse refine 组合损失项: L_cls + w_rank·L_rank. 

    两项 refine 损失共用同一 schedule(effective_weight)与同一静态 weight, 合成单个 ligand_sparse_refine term;
    delta_loss_module 为 None 或 w_rank=0 时退化为纯分类 L_cls(就是分类损失, AdaptiveClassificationCompositeLoss, focal+dice). 

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
