from __future__ import annotations

import functools
from typing import Any, Dict, Optional

import lightning as pl
from src.modules.losses import AdaptiveClassificationCompositeLoss, UnifiedCompositeLoss
from src.modules.lr_schedulers import WarmupThenReduceLROnPlateau
import torch
from hydra.utils import instantiate
from torch import nn


class VoxelPointStage1Wrapper(pl.LightningModule):
    """
    Stage1 体素+点融合模型的 Lightning 训练封装。

    负责将 backbone（VolumePointStage1Model）、损失函数、优化器、调度器以及
    验证指标统一在一个 LightningModule 中管理，完成训练/验证/优化器配置的全流程。

    输入参数:
        - backbone: nn.Module, VolumePointStage1Model 实例或 Hydra 配置
        - atom_loss: nn.Module, 原子级二分类损失（如 BinaryFocalLossWithAlpha）
        - voxel_aux_loss: nn.Module | None, 体素辅助监督损失；为 None 时不启用
        - optimizer: dict | None, 优化器的 Hydra 配置字典
        - scheduler: dict | None, 学习率调度器的 Hydra 配置字典
        - atom_loss_weight: float, 标量, 原子损失权重, 建议值 1.0
        - voxel_aux_loss_weight: float, 标量, 体素辅助损失权重, 建议值 0.2
        - monitor_metric: str, 验证时监控的指标名, 建议值 "val/atom_pr_auc"
        - interval: str, 调度器更新间隔, 建议值 "epoch"
        - frequency: int, 标量, 调度器更新频率, 建议值 1
        - compile: bool, 标量, 是否使用 torch.compile 编译 backbone
    """

    def __init__(
        self,
        backbone: nn.Module,
        atom_loss: nn.Module | None = None,
        voxel_aux_loss: nn.Module | None = None,
        voxel_ligand_loss: nn.Module | None = None,
        optimizer: Optional[Dict[str, Any]] = None,
        scheduler: Optional[Dict[str, Any]] = None,
        atom_loss_weight: float = 1.0,
        voxel_aux_loss_weight: float = 0.0,
        voxel_ligand_loss_weight: float = 0.0,
        monitor_metric: str = "val/atom_pr_auc",
        voxel_ligand_pr_auc_thresholds: Optional[int] = 4096,
        interval: str = "epoch",
        frequency: int = 1,
        compile: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        # 将除 nn.Module 以外的超参数保存到 self.hparams，方便日志和检查点恢复
        self.save_hyperparameters(ignore=["backbone", "atom_loss", "voxel_aux_loss", "voxel_ligand_loss"])

        # nn.Module, 体素+点融合主干网络（VolumePointStage1Model）
        self.backbone = backbone if isinstance(backbone, nn.Module) else instantiate(backbone)
        # nn.Module | None, 原子级二分类损失函数; 为 None 时不计算 atom 损失(UNet-only 消融模式)
        self.atom_loss = (
            atom_loss
            if (atom_loss is None or isinstance(atom_loss, nn.Module))
            else instantiate(atom_loss)
        )
        # nn.Module | None, 体素辅助监督损失函数；为 None 时不参与梯度
        self.voxel_aux_loss = (
            voxel_aux_loss
            if (voxel_aux_loss is None or isinstance(voxel_aux_loss, nn.Module))
            else instantiate(voxel_aux_loss)
        )
        # nn.Module | None, 体素 ligand 占据损失函数; 为 None 时不参与梯度
        self.voxel_ligand_loss = (
            voxel_ligand_loss
            if (voxel_ligand_loss is None or isinstance(voxel_ligand_loss, nn.Module))
            else instantiate(voxel_ligand_loss)
        )

        if compile:
            self.backbone = torch.compile(self.backbone)

        # BinaryAveragePrecision, 验证阶段的 PR-AUC 指标（在 CPU 上计算以节省显存）
        # 各指标仅在对应损失启用时构建; 一个 epoch 可能多次验证，每次都会 reset/update/compute
        from torchmetrics.classification import BinaryAveragePrecision
        self._val_metric_update_counts: dict[str, int] = {}
        self._voxel_ligand_pr_auc_is_binned = voxel_ligand_pr_auc_thresholds is not None
        self._multiclass_metric_names: dict[str, list[str]] = {"atom": [], "voxel_aux": [], "voxel_ligand": []}
        self._class_names = self._resolve_class_names(kwargs)
        if self.atom_loss is not None:
            self.val_atom_pr_auc = BinaryAveragePrecision(compute_on_cpu=True)
            self._val_metric_update_counts["val/atom_pr_auc"] = 0
            self._init_multiclass_ap_metrics("atom", self.atom_loss, BinaryAveragePrecision, None)
        if self.voxel_aux_loss is not None:
            self.val_voxel_aux_pr_auc = BinaryAveragePrecision(compute_on_cpu=True)
            self._val_metric_update_counts["val/voxel_aux_pr_auc"] = 0
            self._init_multiclass_ap_metrics("voxel_aux", self.voxel_aux_loss, BinaryAveragePrecision, None)
        if self.voxel_ligand_loss is not None:
            self.val_voxel_ligand_pr_auc = BinaryAveragePrecision(
                compute_on_cpu=True,
                thresholds=voxel_ligand_pr_auc_thresholds,
            )
            self._val_metric_update_counts["val/voxel_ligand_pr_auc"] = 0
            self._init_multiclass_ap_metrics("voxel_ligand", self.voxel_ligand_loss, BinaryAveragePrecision, voxel_ligand_pr_auc_thresholds)

        # WarmupThenReduceLROnPlateau | None, 手动管理的 validation 级 plateau 调度器
        self._warmup_plateau_scheduler: WarmupThenReduceLROnPlateau | None = None
        # dict[str, Any] | None, checkpoint 恢复时暂存的 plateau 调度器状态
        self._pending_warmup_plateau_state: dict[str, Any] | None = None

    # ------------------------------------------------------------------
    # 工具方法
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_class_names(kwargs: dict[str, Any]) -> list[str]:
        """
        从 wrapper 额外配置中解析类别名列表。

        输入参数:
            - kwargs: dict[str, Any], Hydra 传入 wrapper 但未显式声明的额外配置项

        输出:
            - class_names: list[str], (C,), 类别名列表; 未配置时返回二分类默认类别名
        """
        # list[str] | None, (C,), 配置中的类别名列表
        class_names = kwargs.get("class_names", None)
        if class_names is None:
            return ["background", "foreground"]
        return [str(name) for name in class_names]

    def _init_multiclass_ap_metrics(self, prefix: str, loss_module: nn.Module, metric_cls: Any, thresholds: Optional[int]) -> None:
        """
        为多分类前景类别创建逐类 AP 指标对象。

        输入参数:
            - prefix: str, 指标名前缀, 例如 atom / voxel_aux / voxel_ligand
            - loss_module: nn.Module, 当前监督分支的 loss 模块, 通过 num_classes 判断是否为多分类
            - metric_cls: Any, torchmetrics AP 指标类
            - thresholds: int | None, binned AP 阈值数量; None 表示使用非 binned 指标

        输出:
            - None, 原地注册 self.val_{prefix}_ap_{class_name} 指标
        """
        # int, 当前分支类别数; <=2 时只保留二分类 PR-AUC
        num_classes = int(getattr(loss_module, "num_classes", 1))
        if num_classes <= 2:
            return
        if len(self._class_names) != num_classes:
            raise ValueError(f"len(self._class_names) != num_classes")
        # list[str], 可变长度, 当前分支逐前景类别 AP 指标名
        metric_names: list[str] = []
        for class_id in range(1, num_classes):
            # str, 当前前景类别名
            class_name = self._class_names[class_id]
            # str, Lightning 日志中使用的逐类 AP 指标名
            metric_name = f"val/{prefix}_ap_{class_name}"
            # dict[str, Any], torchmetrics 指标构造参数
            metric_kwargs = {"compute_on_cpu": True}
            if thresholds is not None:
                metric_kwargs["thresholds"] = thresholds
            setattr(self, f"val_{prefix}_ap_{class_name}", metric_cls(**metric_kwargs))
            self._val_metric_update_counts[metric_name] = 0
            metric_names.append(metric_name)
        # str, 当前分支前景类别 macro AP 指标名
        macro_name = f"val/{prefix}_macro_ap"
        self._val_metric_update_counts[macro_name] = 0
        self._multiclass_metric_names[prefix] = metric_names

    @staticmethod
    def _extract_batch(batch: Any) -> dict[str, Any]:
        """
        校验并透传 batch 字典
        """
        if not isinstance(batch, dict):
            raise TypeError(f"VoxelPointStage1Wrapper expects dict batch, got {type(batch)!r}")
        return batch

    @staticmethod
    def _loss_output_to_tensor(loss_out: Any) -> torch.Tensor:
        """
        统一损失函数返回格式：若返回 tuple，取第 0 项作为标量损失
        """
        return loss_out[0] if isinstance(loss_out, tuple) else loss_out

    def _mark_val_metric_updated(self, metric_name: str) -> None:
        """
        记录当前验证轮次内某个 metric 收到过至少一次有效 update。
        """
        self._val_metric_update_counts[metric_name] = self._val_metric_update_counts.get(metric_name, 0) + 1

    @staticmethod
    def _update_metric_on_tensor_device(metric_obj: Any, preds: torch.Tensor, targets: torch.Tensor) -> None:
        """
        在 preds 所在设备上更新 torchmetrics 指标。

        输入参数:
            - metric_obj: Any, torchmetrics 指标对象, 内部可能包含 thresholds 等状态张量
            - preds: torch.Tensor, (M,), 当前 batch 的预测分数
            - targets: torch.Tensor, (M,), 当前 batch 的硬标签

        输出:
            - None, 原地更新 metric_obj 状态
        """
        metric_obj.to(preds.device)
        metric_obj.update(preds, targets)

    @staticmethod
    def _ligand_target_from_dist(ligand_dist_map: torch.Tensor, hard_label_threshold: float) -> torch.Tensor:
        """
        从 ligand 距离图生成验证指标使用的硬标签。

        输入参数:
            - ligand_dist_map: torch.Tensor, (B, D, H, W) 或 (B, C, D, H, W), ligand 距离监督图
            - hard_label_threshold: float, ligand 距离阈值

        输出:
            - target: torch.Tensor, (B, D, H, W), 二分类 0/1 标签或多分类类别 ID 标签
        """
        if ligand_dist_map.ndim == 4:
            # torch.Tensor, (B, D, H, W), 二分类距离阈值标签, 取值 0/1
            return (ligand_dist_map < float(hard_label_threshold)).long()
        if ligand_dist_map.ndim != 5:
            raise ValueError(f"ligand_dist_map 期望为 (B,D,H,W) 或 (B,C,D,H,W)，实际 {tuple(ligand_dist_map.shape)}")
        # torch.Tensor, (B, C-1, D, H, W), 前景类别距离图, 排除背景通道
        foreground_dist = ligand_dist_map[:, 1:]
        # min_dist: torch.Tensor, (B, D, H, W), 最近前景类别距离
        # min_index: torch.Tensor, (B, D, H, W), 最近前景类别在 foreground_dist 中的 0 基索引
        min_dist, min_index = foreground_dist.min(dim=1)
        # torch.Tensor, (B, D, H, W), 最近前景类别 ID, 取值范围 1..C-1
        target = min_index.long() + 1
        return torch.where(min_dist < float(hard_label_threshold), target, torch.zeros_like(target))

    def _update_binary_or_multiclass_ap(
        self,
        prefix: str,
        logits: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
        binary_metric: Any,
        binary_metric_name: str,
    ) -> None:
        """
        根据 logits 通道数更新二分类 PR-AUC 或多分类逐前景类 AP。

        输入参数:
            - prefix: str, 指标名前缀, 例如 atom / voxel_aux / voxel_ligand
            - logits: torch.Tensor, (N, C) 或 (B, C, D, H, W), 当前分支 logits
            - target: torch.Tensor, (N,) 或 (B, D, H, W), 当前分支硬标签
            - mask: torch.Tensor, (N,) 或 (B, D, H, W), 当前分支参与指标统计的位置
            - binary_metric: Any, 二分类 BinaryAveragePrecision 指标对象
            - binary_metric_name: str, 二分类指标日志名

        输出:
            - None, 原地更新 torchmetrics 指标状态
        """
        if mask.sum() <= 0:
            return
        if logits.shape[1] == 1:
            # torch.Tensor, (M,), 有效位置 sigmoid 前景概率
            preds = torch.sigmoid(logits[:, 0]).detach().float().reshape(-1)[mask.reshape(-1)].cpu()
            # torch.Tensor, (M,), 有效位置二分类标签, 取值 0/1
            targets = target.reshape(-1).long()[mask.reshape(-1)].cpu()
            self._update_metric_on_tensor_device(binary_metric, preds, targets)
            self._mark_val_metric_updated(binary_metric_name)
            return
        # torch.Tensor, 与 logits 同形, 多分类 softmax 概率
        prob = torch.softmax(logits, dim=1).detach().float()
        # torch.Tensor, (N_all,), 展平后的类别 ID 标签
        target_flat = target.reshape(-1).long()
        # torch.Tensor[bool], (N_all,), 展平后的有效统计掩码
        mask_flat = mask.reshape(-1)
        for class_id in range(1, logits.shape[1]):
            class_name = self._class_names[class_id] if class_id < len(self._class_names) else f"class_{class_id}"
            metric_name = f"val/{prefix}_ap_{class_name}"
            metric_obj = getattr(self, f"val_{prefix}_ap_{class_name}", None)
            if metric_obj is None:
                continue
            preds = prob[:, class_id].reshape(-1)[mask_flat].cpu()
            targets = (target_flat[mask_flat] == class_id).long().cpu()
            self._update_metric_on_tensor_device(metric_obj, preds, targets)
            self._mark_val_metric_updated(metric_name)
            self._mark_val_metric_updated(f"val/{prefix}_macro_ap")

    # ------------------------------------------------------------------
    # 前向 & 损失计算
    # ------------------------------------------------------------------

    def forward(self, batch: dict[str, Any]) -> dict[str, Any]:
        """
        前向推理，直接委托给 backbone。

        输入参数:
            - batch: dict[str, Any], 包含体素网格、原子特征等的 batch 字典

        输出:
            - outputs: dict[str, Any], backbone 输出字典，至少包含:
                - "atom_logits": torch.Tensor, (sumN, 1), 原子级预测 logits
                - "recycle_passes_used": int, 实际使用的 recycle 轮数
                - "voxel_logits_aux": torch.Tensor | None, (B, 1, D, H, W), 体素辅助预测(可选)
        """
        return self.backbone(batch)

    def _compute_atom_loss(self, outputs: dict[str, Any], batch: dict[str, Any]) -> torch.Tensor | None:
        """
        计算原子级二分类损失。若 atom_loss 模块为 None 则返回 None。

        输入参数:
            - outputs: dict[str, Any], backbone 前向输出
            - batch: dict[str, Any], 当前 batch 字典

        输出:
            - atom_loss: torch.Tensor | None, 标量, 原子级损失值; None 表示不计算
        """
        if self.atom_loss is None:
            return None
        # torch.Tensor, (sumN, 1), 原子级预测 logits
        atom_logits = outputs["atom_logits"]
        # torch.Tensor, (sumN,), 原子级真值标签(0/1)
        atom_target = outputs.get("atom_target", batch["atom_label"])
        # torch.Tensor | None, (sumN,), 原子有效掩码(1=有效, 0=padding)
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
        # 根据损失类型分发: UnifiedCompositeLoss 使用新接口, 旧类使用原有接口
        if isinstance(self.atom_loss, (UnifiedCompositeLoss, AdaptiveClassificationCompositeLoss)):
            loss_out = self.atom_loss(
                logits=atom_logits,
                target=atom_target,
                hardmask=atom_valid_mask,
            )
        else:
            loss_out = self.atom_loss(
                atom_logits,
                atom_target,
                reduction="mean",
                hardmask=atom_valid_mask,
            )
        return self._loss_output_to_tensor(loss_out)

    def _compute_voxel_aux_loss(
        self,
        outputs: dict[str, Any],
        batch: dict[str, Any],
    ) -> torch.Tensor | None:
        """
        计算体素辅助监督损失。若未配置辅助损失或 backbone 未产出辅助 logits，则返回 None。

        输入参数:
            - outputs: dict[str, Any], backbone 前向输出
            - batch: dict[str, Any], 当前 batch 字典

        输出:
            - voxel_aux_loss: torch.Tensor | None, 标量, 体素辅助损失值；为 None 表示不参与总损失
        """
        if self.voxel_aux_loss is None:
            return None

        # torch.Tensor | None, (B, 1, D, H, W), 体素辅助预测 logits
        voxel_logits_aux = outputs.get("voxel_logits_aux")
        if voxel_logits_aux is None:
            return None

        # torch.Tensor, (B, D, H, W), 体素级真值标签
        voxel_target = batch["voxel_label"]
        # torch.Tensor, (B, 1, D, H, W), 几何 hardmask
        hardmask = batch["hardmask"]
        # torch.Tensor, (B, 1, D, H, W), 边界有效掩码
        voxel_valid_mask = batch["voxel_valid_mask"]

        # 根据损失类型分发
        if isinstance(self.voxel_aux_loss, (UnifiedCompositeLoss, AdaptiveClassificationCompositeLoss)):
            loss_out = self.voxel_aux_loss(
                logits=voxel_logits_aux,
                target=voxel_target,
                hardmask=hardmask,
                valid_mask=voxel_valid_mask,
            )
        else:
            # 旧类 (FocalTverskyCombinedLoss): 合并 hardmask 和 valid_mask 后传入
            voxel_loss_hardmask = hardmask.bool() & voxel_valid_mask.bool()
            loss_out = self.voxel_aux_loss(
                voxel_logits_aux,
                voxel_target,
                reduction="mean",
                hardmask=voxel_loss_hardmask,
            )
        return self._loss_output_to_tensor(loss_out)

    def _compute_voxel_ligand_loss(
        self,
        outputs: dict[str, Any],
        batch: dict[str, Any],
    ) -> torch.Tensor | None:
        """
        计算体素 ligand 占据损失。若未配置或数据中无 ligand_dist_map，返回 None。

        输入参数:
            - outputs: dict[str, Any], backbone 前向输出
            - batch: dict[str, Any], 当前 batch 字典

        输出:
            - voxel_ligand_loss: torch.Tensor | None, 标量
        """
        if self.voxel_ligand_loss is None:
            return None
        # torch.Tensor | None, (B, 1, D, H, W), ligand 预测 logits
        voxel_logits_ligand = outputs.get("voxel_logits_ligand")
        if voxel_logits_ligand is None:
            return None
        # torch.Tensor | None, (B, D, H, W), ligand 距离图
        ligand_dist_map = batch.get("ligand_dist_map")
        if ligand_dist_map is None:
            return None

        # # torch.Tensor, (B, 1, D, H, W), 几何 hardmask
        # hardmask = batch["hardmask"]
        # torch.Tensor, (B, 1, D, H, W), 边界有效掩码
        voxel_valid_mask = batch["voxel_valid_mask"]

        if isinstance(self.voxel_ligand_loss, (UnifiedCompositeLoss, AdaptiveClassificationCompositeLoss)):
            loss = self.voxel_ligand_loss(
                logits=voxel_logits_ligand,
                target=None,
                hardmask=None,
                valid_mask=voxel_valid_mask,
                ligand_dist_map=ligand_dist_map,
            )
        else:
            loss = self.voxel_ligand_loss(
                logits=voxel_logits_ligand,
                target=None,
                hardmask=None,
                valid_mask=voxel_valid_mask,
                ligand_dist_map=ligand_dist_map,
                reduction="mean",
            )
        return loss

    def _compute_total_loss(
        self,
        outputs: dict[str, Any],
        batch: dict[str, Any],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        汇总原子损失、体素辅助损失和体素 ligand 损失，得到加权总损失。

        输入参数:
            - outputs: dict[str, Any], backbone 前向输出
            - batch: dict[str, Any], 当前 batch 字典

        输出:
            - total_loss: torch.Tensor, 标量, 加权总损失
            - loss_dict: dict[str, torch.Tensor], 各分项损失字典
        """
        loss_dict: dict[str, torch.Tensor] = {}
        # torch.Tensor, 标量, 累积总损失; 初始化为 0
        total_loss = torch.tensor(0.0, device=self.device, dtype=torch.float32)

        # torch.Tensor | None, 标量, 原子级损失
        atom_loss = self._compute_atom_loss(outputs=outputs, batch=batch)
        if atom_loss is not None:
            total_loss = total_loss + float(self.hparams.atom_loss_weight) * atom_loss
            loss_dict["atom_loss"] = atom_loss

        # torch.Tensor | None, 标量, 体素辅助损失
        voxel_aux_loss = self._compute_voxel_aux_loss(outputs=outputs, batch=batch)
        if voxel_aux_loss is not None:
            total_loss = total_loss + float(self.hparams.voxel_aux_loss_weight) * voxel_aux_loss
            loss_dict["voxel_aux_loss"] = voxel_aux_loss

        # torch.Tensor | None, 标量, 体素 ligand 占据损失
        voxel_ligand_loss = self._compute_voxel_ligand_loss(outputs=outputs, batch=batch)
        if voxel_ligand_loss is not None:
            total_loss = total_loss + float(self.hparams.voxel_ligand_loss_weight) * voxel_ligand_loss
            loss_dict["voxel_ligand_loss"] = voxel_ligand_loss

        loss_dict["total_loss"] = total_loss
        return total_loss, loss_dict

    # ------------------------------------------------------------------
    # 验证指标
    # ------------------------------------------------------------------

    def _update_val_atom_metric(self, outputs: dict[str, Any], batch: dict[str, Any]) -> None:
        """
        用当前 batch 的预测与真值更新验证阶段的 atom PR-AUC 指标。
        当 atom_loss 为 None (UNet-only 模式) 时直接跳过。

        仅对 atom_valid_mask 为 True 且标签不等于 ignore_index 的原子进行统计。

        输入参数:
            - outputs: dict[str, Any], backbone 前向输出
            - batch: dict[str, Any], 当前 batch 字典
        """
        if self.atom_loss is None:
            return
        # torch.Tensor, (sumN, 1), 原子级预测 logits
        atom_logits = outputs["atom_logits"]
        # torch.Tensor, (sumN,), 原子级真值标签
        atom_target = outputs.get("atom_target", batch["atom_label"])
        # torch.Tensor, (sumN,), bool, 原子有效掩码
        atom_valid_mask = outputs.get("atom_valid_mask", batch["atom_valid_mask"]).bool()

        if atom_logits.shape[0] != atom_target.shape[0]:
            raise RuntimeError(
                "Atom supervision shape mismatch before metric update: "
                f"atom_logits.shape={tuple(atom_logits.shape)}, "
                f"atom_target.shape={tuple(atom_target.shape)}"
            )

        # int | None, 损失函数中指定的忽略标签索引
        ignore_index = getattr(self.atom_loss, "ignore_index", None)
        # torch.Tensor, (sumN,), bool, 最终有效掩码（排除 padding 和 ignore_index）
        valid = atom_valid_mask
        if ignore_index is not None:
            valid = valid & (atom_target != ignore_index)

        if valid.sum() <= 0:
            return

        self._update_binary_or_multiclass_ap(
            prefix="atom",
            logits=atom_logits,
            target=atom_target,
            mask=valid,
            binary_metric=self.val_atom_pr_auc,
            binary_metric_name="val/atom_pr_auc",
        )

    def _update_val_voxel_aux_metric(self, outputs: dict[str, Any], batch: dict[str, Any]) -> None:
        """
        用当前 batch 的体素辅助预测更新验证指标。
        掩码逻辑与 voxel_aux_loss 完全一致: hardmask AND valid_mask。

        输入参数:
            - outputs: dict[str, Any], backbone 前向输出
            - batch: dict[str, Any], 当前 batch 字典
        """
        if self.voxel_aux_loss is None:
            return
        # torch.Tensor | None, (B, 1, D, H, W), 体素辅助预测 logits
        voxel_logits_aux = outputs.get("voxel_logits_aux")
        if voxel_logits_aux is None:
            return

        target = batch["voxel_label"].long()
        effective_mask = batch["hardmask"].squeeze(1).bool() & batch["voxel_valid_mask"].squeeze(1).bool()
        self._update_binary_or_multiclass_ap(
            prefix="voxel_aux",
            logits=voxel_logits_aux,
            target=target,
            mask=effective_mask,
            binary_metric=self.val_voxel_aux_pr_auc,
            binary_metric_name="val/voxel_aux_pr_auc",
        )

    def _update_val_voxel_ligand_metric(self, outputs: dict[str, Any], batch: dict[str, Any]) -> None:
        """
        用当前 batch 的体素 ligand 预测更新验证指标。
        掩码逻辑与 voxel_ligand_loss 完全一致: 仅 valid_mask, 不使用 hardmask。

        输入参数:
            - outputs: dict[str, Any], backbone 前向输出
            - batch: dict[str, Any], 当前 batch 字典
        """
        if self.voxel_ligand_loss is None:
            return
        # torch.Tensor | None, (B, 1, D, H, W), ligand 预测 logits
        voxel_logits_ligand = outputs.get("voxel_logits_ligand")
        if voxel_logits_ligand is None:
            return
        # torch.Tensor | None, (B, D, H, W), ligand 距离图
        ligand_dist_map = batch.get("ligand_dist_map")
        if ligand_dist_map is None:
            return

        hard_label_threshold = getattr(self.voxel_ligand_loss, "hard_label_threshold", None)
        if hard_label_threshold is not None:
            voxel_target = self._ligand_target_from_dist(ligand_dist_map, float(hard_label_threshold))
        else:
            voxel_target = batch["voxel_label"]
        mask = batch["voxel_valid_mask"].squeeze(1).bool()
        self._update_binary_or_multiclass_ap(
            prefix="voxel_ligand",
            logits=voxel_logits_ligand,
            target=voxel_target,
            mask=mask,
            binary_metric=self.val_voxel_ligand_pr_auc,
            binary_metric_name="val/voxel_ligand_pr_auc",
        )

    # ------------------------------------------------------------------
    # 训练 / 验证步骤
    # ------------------------------------------------------------------

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        """
        单个训练步：前向 → 计算总损失 → 记录日志。

        输入参数:
            - batch: Any, DataLoader 产出的 batch
            - batch_idx: int, 标量, 当前 batch 在 epoch 内的索引

        输出:
            - total_loss: torch.Tensor, 标量, 加权总损失（用于反向传播）
        """
        batch_dict = self._extract_batch(batch)
        outputs = self(batch_dict)
        total_loss, loss_dict = self._compute_total_loss(outputs=outputs, batch=batch_dict)

        # 记录总损失
        self.log("train/loss", total_loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
        # 记录原子损失（仅当 atom_loss 启用时）
        if "atom_loss" in loss_dict:
            self.log(
                "train/atom_loss",
                loss_dict["atom_loss"],
                prog_bar=False,
                on_step=True,
                on_epoch=True,
                sync_dist=True,
            )
        # 记录体素辅助损失（仅当启用时）
        if "voxel_aux_loss" in loss_dict:
            self.log(
                "train/voxel_aux_loss",
                loss_dict["voxel_aux_loss"],
                prog_bar=False,
                on_step=True,
                on_epoch=True,
                sync_dist=True,
            )
        # 记录体素 ligand 损失（仅当启用时）
        if "voxel_ligand_loss" in loss_dict:
            self.log(
                "train/voxel_ligand_loss",
                loss_dict["voxel_ligand_loss"],
                prog_bar=False,
                on_step=True,
                on_epoch=True,
                sync_dist=True,
            )
        # 记录当前 step 实际使用的 recycle 轮数
        self.log(
            "train/recycle_passes",
            float(outputs["recycle_passes_used"]),
            prog_bar=False,
            on_step=True,
            on_epoch=False,
            sync_dist=True,
        )
        return total_loss

    def validation_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        """
        单个验证步：前向 → 计算总损失 → 更新 PR-AUC 指标 → 记录日志。

        输入参数:
            - batch: Any, DataLoader 产出的 batch
            - batch_idx: int, 标量, 当前 batch 在 epoch 内的索引

        输出:
            - total_loss: torch.Tensor, 标量, 加权总损失
        """
        batch_dict = self._extract_batch(batch)
        outputs = self(batch_dict)
        total_loss, loss_dict = self._compute_total_loss(outputs=outputs, batch=batch_dict)
        # 用当前 batch 更新 PR-AUC 指标
        self._update_val_atom_metric(outputs=outputs, batch=batch_dict)
        self._update_val_voxel_aux_metric(outputs=outputs, batch=batch_dict)
        self._update_val_voxel_ligand_metric(outputs=outputs, batch=batch_dict)

        # 记录总损失
        self.log("val/loss", total_loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        # 记录原子损失（仅当 atom_loss 启用时）
        if "atom_loss" in loss_dict:
            self.log(
                "val/atom_loss",
                loss_dict["atom_loss"],
                prog_bar=False,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )
        # 记录体素辅助损失（仅当启用时）
        if "voxel_aux_loss" in loss_dict:
            self.log(
                "val/voxel_aux_loss",
                loss_dict["voxel_aux_loss"],
                prog_bar=False,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )
        # 记录体素 ligand 损失（仅当启用时）
        if "voxel_ligand_loss" in loss_dict:
            self.log(
                "val/voxel_ligand_loss",
                loss_dict["voxel_ligand_loss"],
                prog_bar=False,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )
        # 记录 recycle 轮数
        self.log(
            "val/recycle_passes",
            float(outputs["recycle_passes_used"]),
            prog_bar=False,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        return total_loss

    def _compute_log_reset_metric(self, metric_obj, metric_name: str) -> None:
        """
        通用的验证指标计算-日志-重置流程。
        在本地 GPU 上 compute, 通过 sync_dist 跨卡汇聚, 然后 reset。

        输入参数:
            - metric_obj: BinaryAveragePrecision, torchmetrics 指标对象
            - metric_name: str, 日志中使用的指标名 (如 "val/atom_pr_auc")
        """
        preds_state = getattr(metric_obj, "preds", None)
        target_state = getattr(metric_obj, "target", None)
        if isinstance(preds_state, list) and isinstance(target_state, list):
            if len(preds_state) == 0 or len(target_state) == 0:
                metric_obj.reset()
                return

        # 临时禁用 torchmetrics 内部同步，手动在 GPU 上做 sync_dist
        prev_to_sync = getattr(metric_obj, "_to_sync", True)
        metric_obj._to_sync = False
        # float, 标量, 当前 GPU 本地 PR-AUC 值
        try:
            score_local = metric_obj.compute()
        except ValueError as exc:
            metric_obj._to_sync = prev_to_sync
            if "No samples to concatenate" in str(exc):
                metric_obj.reset()
                return
            raise
        metric_obj._to_sync = prev_to_sync

        # torch.Tensor, 标量, 移到当前 GPU 以便 sync_dist 正常工作
        score_gpu = score_local.to(self.device)
        metric_obj.reset()
        self.log(
            metric_name,
            score_gpu,
            prog_bar=(metric_name == self.hparams.monitor_metric),  # 仅主指标显示进度条
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )

    def _compute_log_reset_metric_safe(self, metric_obj, metric_name: str) -> torch.Tensor:
        """
        对空样本验证轮次安全地计算、日志记录并重置单个 metric。

        输入参数:
            - metric_obj: BinaryAveragePrecision, torchmetrics 指标对象
            - metric_name: str, 日志中使用的指标名, 如 "val/atom_pr_auc"

        输出:
            - score_gpu: torch.Tensor, (), 当前 rank 上计算得到的指标值
        """
        # int, 当前验证轮次内该 metric 收到的有效 update 次数
        local_updates = int(self._val_metric_update_counts.get(metric_name, 0))
        if local_updates <= 0:
            # torch.Tensor, (), 空样本验证轮次的安全指标值
            score_gpu = torch.tensor(0.0, device=self.device, dtype=torch.float32)
        else:
            prev_to_sync = getattr(metric_obj, "_to_sync", True)
            metric_obj._to_sync = False
            try:
                # torch.Tensor, (), 当前 rank 本地 metric 计算值
                score_local = metric_obj.compute()
            finally:
                metric_obj._to_sync = prev_to_sync
            score_gpu = score_local.to(self.device)

        metric_obj.reset()
        self._val_metric_update_counts[metric_name] = 0
        self.log(
            metric_name,
            score_gpu,
            prog_bar=(metric_name == self.hparams.monitor_metric),
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        return score_gpu

    def _compute_log_reset_multiclass_metrics(self) -> dict[str, torch.Tensor]:
        """
        计算、日志记录并重置所有多分类逐类 AP 与 macro AP 指标。

        输出:
            - computed_metrics: dict[str, torch.Tensor], 当前 validation end 中已计算出的指标名到标量张量的映射
        """
        # dict[str, torch.Tensor], validation end 现场计算出的多分类指标
        computed_metrics: dict[str, torch.Tensor] = {}
        for prefix, metric_names in self._multiclass_metric_names.items():
            if len(metric_names) == 0:
                continue
            # list[torch.Tensor], 当前 prefix 下有有效 update 的逐类 AP
            class_scores: list[torch.Tensor] = []
            for metric_name in metric_names:
                class_name = metric_name.rsplit("_ap_", 1)[1]
                metric_obj = getattr(self, f"val_{prefix}_ap_{class_name}")
                # int, 当前验证轮次内该逐类 AP 收到的有效 update 次数
                local_updates = int(self._val_metric_update_counts.get(metric_name, 0))
                if local_updates <= 0:
                    # torch.Tensor, (), 当前类别无有效样本时的安全 AP
                    score_gpu = torch.tensor(0.0, device=self.device, dtype=torch.float32)
                else:
                    prev_to_sync = getattr(metric_obj, "_to_sync", True)
                    metric_obj._to_sync = False
                    try:
                        score_gpu = metric_obj.compute().to(self.device)
                    finally:
                        metric_obj._to_sync = prev_to_sync
                    class_scores.append(score_gpu)
                metric_obj.reset()
                self._val_metric_update_counts[metric_name] = 0
                computed_metrics[metric_name] = score_gpu
                self.log(
                    metric_name,
                    score_gpu,
                    prog_bar=(metric_name == self.hparams.monitor_metric),
                    on_step=False,
                    on_epoch=True,
                    sync_dist=True,
                )
            macro_name = f"val/{prefix}_macro_ap"
            macro_score = torch.stack(class_scores).mean() if len(class_scores) > 0 else torch.tensor(0.0, device=self.device)
            self._val_metric_update_counts[macro_name] = 0
            computed_metrics[macro_name] = macro_score
            self.log(
                macro_name,
                macro_score,
                prog_bar=(macro_name == self.hparams.monitor_metric),
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )
        return computed_metrics

    def _sync_metric_for_scheduler(self, metric_value: torch.Tensor) -> torch.Tensor:
        """
        将 plateau scheduler 使用的主指标同步为各 rank 一致的标量。

        输入参数:
            - metric_value: torch.Tensor, (), 当前 rank 上的主指标值

        输出:
            - synced_metric: torch.Tensor, (), 各 rank 一致的主指标均值
        """
        # torch.Tensor, (), 当前 rank 上用于调度器的主指标
        metric_tensor = metric_value.detach().to(self.device).float().reshape(())
        if int(self.trainer.world_size) > 1:
            # torch.Tensor, (world_size,), all_gather 后的各 rank 主指标
            gathered_metric = self.all_gather(metric_tensor).float()
            metric_tensor = gathered_metric.mean()
        return metric_tensor

    def _step_warmup_plateau_scheduler(self, computed_metrics: dict[str, torch.Tensor]) -> None:
        """
        在每次 validation end 后按主指标推进 warmup_plateau 的 plateau 部分。

        输入参数:
            - computed_metrics: dict[str, torch.Tensor], on_validation_epoch_end 现场计算出的指标名到标量张量的映射

        输出:
            - None, 原地更新 optimizer 中各 param group 的学习率
        """
        if self._warmup_plateau_scheduler is None:
            return
        if self.trainer.sanity_checking:
            return

        # str, plateau scheduler 监控的主指标名
        monitor_name = str(self.hparams.monitor_metric)
        if monitor_name in computed_metrics:
            # torch.Tensor, (), 当前 validation end 现场计算出的主指标
            metric_value = computed_metrics[monitor_name]
        elif monitor_name in self.trainer.callback_metrics:
            callback_metric = self.trainer.callback_metrics[monitor_name]
            metric_value = callback_metric if isinstance(callback_metric, torch.Tensor) else torch.tensor(float(callback_metric), device=self.device)
        else:
            raise RuntimeError(
                f"warmup_plateau scheduler monitor metric {monitor_name!r} is not available after validation."
            )

        # torch.Tensor, (), DDP 同步后的主指标值
        synced_metric = self._sync_metric_for_scheduler(metric_value)
        self._warmup_plateau_scheduler.step_plateau(synced_metric, global_step=int(self.global_step))

    def on_validation_epoch_end(self) -> None:
        """
        验证 epoch 结束时，计算并日志所有已启用的 PR-AUC 指标，然后推进 plateau 调度器。

        流程:
            1. 临时关闭 torchmetrics 自动同步，在各 GPU 本地 compute
            2. 将结果通过 self.log(sync_dist=True) 写入日志
            3. reset 指标状态，为下一次验证做准备
            4. 若启用 warmup_plateau, 用主指标推进 ReduceLROnPlateau
        """
        # dict[str, torch.Tensor], 当前 validation end 已计算出的指标集合
        computed_metrics: dict[str, torch.Tensor] = {}
        if hasattr(self, "val_atom_pr_auc"):
            computed_metrics["val/atom_pr_auc"] = self._compute_log_reset_metric_safe(
                self.val_atom_pr_auc,
                "val/atom_pr_auc",
            )
        if hasattr(self, "val_voxel_aux_pr_auc"):
            computed_metrics["val/voxel_aux_pr_auc"] = self._compute_log_reset_metric_safe(
                self.val_voxel_aux_pr_auc,
                "val/voxel_aux_pr_auc",
            )
        if hasattr(self, "val_voxel_ligand_pr_auc"):
            computed_metrics["val/voxel_ligand_pr_auc"] = self._compute_log_reset_metric_safe(
                self.val_voxel_ligand_pr_auc,
                "val/voxel_ligand_pr_auc",
            )
        computed_metrics.update(self._compute_log_reset_multiclass_metrics())
        self._step_warmup_plateau_scheduler(computed_metrics)

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        """
        保存手动管理的 warmup_plateau plateau 状态。

        输入参数:
            - checkpoint: dict[str, Any], Lightning 即将写入磁盘的 checkpoint 字典

        输出:
            - None, 原地向 checkpoint 写入 plateau scheduler 状态
        """
        if self._warmup_plateau_scheduler is not None:
            checkpoint["warmup_plateau_reduce_on_plateau_state"] = self._warmup_plateau_scheduler.state_dict()

    def on_load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        """
        恢复手动管理的 warmup_plateau plateau 状态。

        输入参数:
            - checkpoint: dict[str, Any], Lightning 从磁盘读取的 checkpoint 字典

        输出:
            - None, 立即恢复或暂存 plateau scheduler 状态
        """
        if "warmup_plateau_reduce_on_plateau_state" not in checkpoint:
            return
        # dict[str, Any], checkpoint 中保存的 plateau scheduler 状态
        plateau_state = checkpoint["warmup_plateau_reduce_on_plateau_state"]
        if self._warmup_plateau_scheduler is None:
            self._pending_warmup_plateau_state = plateau_state
        else:
            self._warmup_plateau_scheduler.load_state_dict(plateau_state)

    # ------------------------------------------------------------------
    # 优化器 & 调度器
    # ------------------------------------------------------------------

    def _resolve_warmup_steps(self, sched_cfg: Any) -> int:
        """
        从 scheduler 配置中解析 warmup step 数。

        输入参数:
            - sched_cfg: Any, Hydra scheduler 配置, 需要包含 total_steps/warmup_steps/warmup_ratio

        输出:
            - warmup_steps: int, 线性 warmup 覆盖的 optimizer step 数
        """
        total_steps = sched_cfg["total_steps"]
        if total_steps is None:
            total_steps = getattr(self.trainer, "estimated_stepping_batches", None)
        if total_steps is None or int(total_steps) <= 0:
            raise RuntimeError("warmup scheduler requires a positive total_steps value.")
        total_steps = int(total_steps)

        warmup_steps = sched_cfg["warmup_steps"]
        if warmup_steps is None:
            warmup_ratio = float(sched_cfg["warmup_ratio"])
            if not (0.0 <= warmup_ratio < 1.0):
                raise ValueError(f"warmup_ratio must be in [0, 1), got {warmup_ratio}.")
            warmup_steps = int(round(total_steps * warmup_ratio))
        warmup_steps = int(warmup_steps)
        if warmup_steps < 0 or warmup_steps > total_steps:
            raise ValueError(
                f"warmup_steps must be in [0, total_steps], got warmup_steps={warmup_steps}, total_steps={total_steps}."
            )
        return warmup_steps

    def _build_warmup_only_scheduler(
        self,
        optimizer: torch.optim.Optimizer,
        sched_cfg: Any,
    ) -> torch.optim.lr_scheduler.LRScheduler:
        """
        构建仅包含 step 级线性 warmup 的 scheduler。

        输入参数:
            - optimizer: torch.optim.Optimizer, 被调度的优化器
            - sched_cfg: Any, Hydra scheduler 配置, 需要包含 total_steps/warmup_steps/warmup_ratio/warmup_start_factor

        输出:
            - scheduler: torch.optim.lr_scheduler.LRScheduler, Lightning step 级调度器
        """
        warmup_steps = self._resolve_warmup_steps(sched_cfg)
        start_factor = float(sched_cfg["warmup_start_factor"])
        if not (0.0 < start_factor <= 1.0):
            raise ValueError(f"warmup_start_factor must be in (0, 1], got {start_factor}.")

        if warmup_steps == 0 or start_factor == 1.0:
            return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _: 1.0)

        return torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=start_factor,
            end_factor=1.0,
            total_iters=warmup_steps,
        )

    def _build_warmup_plateau_scheduler(
        self,
        optimizer: torch.optim.Optimizer,
        sched_cfg: Any,
    ) -> WarmupThenReduceLROnPlateau:
        """
        构建 step 级 warmup + validation 级 plateau 的组合 scheduler。

        输入参数:
            - optimizer: torch.optim.Optimizer, 被调度的优化器
            - sched_cfg: Any, Hydra scheduler 配置, 需要显式包含 warmup 与 plateau 全部字段

        输出:
            - scheduler: WarmupThenReduceLROnPlateau, 组合调度器对象
        """
        scheduler = WarmupThenReduceLROnPlateau(
            optimizer,
            warmup_steps=self._resolve_warmup_steps(sched_cfg),
            warmup_start_factor=float(sched_cfg["warmup_start_factor"]),
            mode=str(sched_cfg["mode"]),
            factor=float(sched_cfg["factor"]),
            patience=int(sched_cfg["patience"]),
            threshold=float(sched_cfg["threshold"]),
            threshold_mode=str(sched_cfg["threshold_mode"]),
            cooldown=int(sched_cfg["cooldown"]),
            min_lr=sched_cfg["min_lr"],
            eps=float(sched_cfg["eps"]),
        )
        if self._pending_warmup_plateau_state is not None:
            scheduler.load_state_dict(self._pending_warmup_plateau_state)
            self._pending_warmup_plateau_state = None
        return scheduler

    def configure_optimizers(self) -> dict[str, Any]:
        """
        配置优化器与学习率调度器。

        支持三种优化器配置方式:
            1. opt_cfg 为 None → 使用默认 AdamW(lr=1e-4, weight_decay=1e-2)
            2. opt_cfg 为 functools.partial 或 callable → 直接调用
            3. opt_cfg 为 Hydra 配置字典 → 通过 instantiate 实例化

        调度器配置同理：functools.partial / callable / Hydra 字典，
        返回的 lr_scheduler 字典会自动绑定 monitor_metric 用于 ReduceLROnPlateau 等。

        输出:
            - config: dict[str, Any], 包含 "optimizer" 以及可选的 "lr_scheduler" 子字典
        """
        opt_cfg = self.hparams.optimizer
        if opt_cfg is None:
            # 回退默认优化器（仅在未通过 YAML 指定时触发）
            optimizer = torch.optim.AdamW(
                params=filter(lambda p: p.requires_grad, self.parameters()),
                lr=1e-4,
                weight_decay=1e-2,
            )
        else:
            # filter, 仅保留需要梯度的参数
            trainable_params = filter(lambda p: p.requires_grad, self.parameters())
            if isinstance(opt_cfg, functools.partial):
                optimizer = opt_cfg(params=trainable_params)
            elif callable(opt_cfg) and not hasattr(opt_cfg, "keys"):
                optimizer = opt_cfg(params=trainable_params)
            else:
                optimizer = instantiate(opt_cfg, params=trainable_params)
                if not isinstance(optimizer, torch.optim.Optimizer):
                    raise TypeError("Failed to instantiate optimizer.")

        sched_cfg = self.hparams.scheduler
        if sched_cfg is None:
            return {"optimizer": optimizer}

        if isinstance(sched_cfg, functools.partial):
            scheduler = sched_cfg(optimizer=optimizer)
        elif callable(sched_cfg) and not hasattr(sched_cfg, "keys"):
            scheduler = sched_cfg(optimizer=optimizer)
        elif hasattr(sched_cfg, "get") and sched_cfg.get("name", None) == "warmup_plateau":
            self._warmup_plateau_scheduler = self._build_warmup_plateau_scheduler(
                optimizer=optimizer,
                sched_cfg=sched_cfg,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": self._warmup_plateau_scheduler.lightning_warmup_config(),
            }
        elif hasattr(sched_cfg, "get") and sched_cfg.get("name", None) == "warmup_only":
            scheduler = self._build_warmup_only_scheduler(optimizer=optimizer, sched_cfg=sched_cfg)
        else:
            scheduler = instantiate(sched_cfg, optimizer=optimizer)
            # 某些调度器工厂返回的是 callable 而非真正的 scheduler 实例，需要额外调用一次
            if hasattr(scheduler, "__call__") and not hasattr(scheduler, "step"):
                scheduler = scheduler()

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": self.hparams.monitor_metric,
                "interval": self.hparams.interval,
                "frequency": self.hparams.frequency,
            },
        }
