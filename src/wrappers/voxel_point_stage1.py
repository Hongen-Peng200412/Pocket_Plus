from __future__ import annotations

import functools
from typing import Any, Dict, Optional

import lightning as pl
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
        atom_loss: nn.Module,
        voxel_aux_loss: nn.Module | None = None,
        optimizer: Optional[Dict[str, Any]] = None,
        scheduler: Optional[Dict[str, Any]] = None,
        atom_loss_weight: float = 1.0,
        voxel_aux_loss_weight: float = 0.0,
        monitor_metric: str = "val/atom_pr_auc",
        interval: str = "epoch",
        frequency: int = 1,
        compile: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        # 将除 nn.Module 以外的超参数保存到 self.hparams，方便日志和检查点恢复
        self.save_hyperparameters(ignore=["backbone", "atom_loss", "voxel_aux_loss"])

        # nn.Module, 体素+点融合主干网络（VolumePointStage1Model）
        self.backbone = backbone if isinstance(backbone, nn.Module) else instantiate(backbone)
        # nn.Module, 原子级二分类损失函数
        self.atom_loss = atom_loss if isinstance(atom_loss, nn.Module) else instantiate(atom_loss)
        # nn.Module | None, 体素辅助监督损失函数；为 None 时不参与梯度
        self.voxel_aux_loss = (
            voxel_aux_loss
            if (voxel_aux_loss is None or isinstance(voxel_aux_loss, nn.Module))
            else instantiate(voxel_aux_loss)
        )

        if compile:
            self.backbone = torch.compile(self.backbone)

        from torchmetrics.classification import BinaryAveragePrecision

        # BinaryAveragePrecision, 验证阶段的原子级 PR-AUC 指标（在 CPU 上计算以节省显存）
        self.val_atom_pr_auc = BinaryAveragePrecision(compute_on_cpu=True)

    # ------------------------------------------------------------------
    # 工具方法
    # ------------------------------------------------------------------

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

    def _compute_atom_loss(self, outputs: dict[str, Any], batch: dict[str, Any]) -> torch.Tensor:   # TODO: 原子级别的二分类focal_loss还没写
        """
        计算原子级二分类损失。

        输入参数:
            - outputs: dict[str, Any], backbone 前向输出
            - batch: dict[str, Any], 当前 batch 字典

        输出:
            - atom_loss: torch.Tensor, 标量, 原子级损失值(reduction=sum)
        """
        # torch.Tensor, (sumN, 1), 原子级预测 logits
        atom_logits = outputs["atom_logits"]
        # torch.Tensor, (sumN,), 原子级真值标签(0/1)
        atom_target = batch["atom_label"]
        # torch.Tensor | None, (sumN,), 原子有效掩码(1=有效, 0=padding)
        atom_valid_mask = batch.get("atom_valid_mask")
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

        # torch.Tensor, (B, 1, D, H, W), 体素级真值标签
        voxel_target = batch["voxel_label"]
        # torch.Tensor | None, (B, 1, D, H, W), 体素级有效掩码
        hardmask = batch["hardmask"]
        voxel_valid_mask = batch["voxel_valid_mask"]
        voxel_loss_hardmask = hardmask.bool() & voxel_valid_mask.bool()
        loss_out = self.voxel_aux_loss(
            voxel_logits_aux,
            voxel_target,
            reduction="mean",
            hardmask=voxel_loss_hardmask,
        )
        return self._loss_output_to_tensor(loss_out)

    def _compute_total_loss(
        self,
        outputs: dict[str, Any],
        batch: dict[str, Any],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        汇总原子损失与可选的体素辅助损失，得到加权总损失。

        输入参数:
            - outputs: dict[str, Any], backbone 前向输出
            - batch: dict[str, Any], 当前 batch 字典

        输出:
            - total_loss: torch.Tensor, 标量, 加权总损失
            - loss_dict: dict[str, torch.Tensor], 各分项损失字典, 键包括:
                - "atom_loss": 原子级损失
                - "voxel_aux_loss": 体素辅助损失(仅当启用时)
                - "total_loss": 最终加权总损失
        """
        # torch.Tensor, 标量, 原子级损失
        atom_loss = self._compute_atom_loss(outputs=outputs, batch=batch)
        # torch.Tensor, 标量, 按权重加权后的总损失（先加原子部分）
        total_loss = float(self.hparams.atom_loss_weight) * atom_loss
        loss_dict: dict[str, torch.Tensor] = {"atom_loss": atom_loss}

        # torch.Tensor | None, 标量, 体素辅助损失
        voxel_aux_loss = self._compute_voxel_aux_loss(outputs=outputs, batch=batch)
        if voxel_aux_loss is not None:
            total_loss = total_loss + float(self.hparams.voxel_aux_loss_weight) * voxel_aux_loss
            loss_dict["voxel_aux_loss"] = voxel_aux_loss

        loss_dict["total_loss"] = total_loss
        return total_loss, loss_dict

    # ------------------------------------------------------------------
    # 验证指标
    # ------------------------------------------------------------------

    def _update_val_atom_metric(self, outputs: dict[str, Any], batch: dict[str, Any]) -> None:
        """
        用当前 batch 的预测与真值更新验证阶段的 atom PR-AUC 指标。

        仅对 atom_valid_mask 为 True 且标签不等于 ignore_index 的原子进行统计。

        输入参数:
            - outputs: dict[str, Any], backbone 前向输出
            - batch: dict[str, Any], 当前 batch 字典
        """
        # torch.Tensor, (sumN, 1), 原子级预测 logits
        atom_logits = outputs["atom_logits"]
        # torch.Tensor, (sumN,), 原子级真值标签
        atom_target = batch["atom_label"]
        # torch.Tensor, (sumN,), bool, 原子有效掩码
        atom_valid_mask = batch["atom_valid_mask"].bool()

        # int | None, 损失函数中指定的忽略标签索引
        ignore_index = getattr(self.atom_loss, "ignore_index", None)
        # torch.Tensor, (sumN,), bool, 最终有效掩码（排除 padding 和 ignore_index）
        valid = atom_valid_mask
        if ignore_index is not None:
            valid = valid & (atom_target != ignore_index)

        if valid.sum() <= 0:
            return

        # torch.Tensor, (N_valid,), float, sigmoid 后的预测概率（仅有效原子, 在 CPU 上）
        preds = torch.sigmoid(atom_logits[:, 0]).detach().float()[valid].cpu()
        # torch.Tensor, (N_valid,), long, 有效原子的真值标签（在 CPU 上）
        targets = atom_target[valid].long().cpu()
        self.val_atom_pr_auc.update(preds, targets)

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
        # 记录原子损失
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

        # 记录总损失
        self.log("val/loss", total_loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        # 记录原子损失
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

    def on_validation_epoch_end(self) -> None:
        """
        验证 epoch 结束时，在本地 GPU 上计算 PR-AUC 并通过 sync_dist 汇聚到所有进程。

        流程:
            1. 临时关闭自动同步，在各 GPU 本地 compute PR-AUC
            2. 将结果移到当前 GPU，通过 self.log(sync_dist=True) 做跨卡平均
            3. reset 指标状态，为下一 epoch 做准备
        """
        # 临时禁用 torchmetrics 内置同步，手动在 GPU 上做 sync_dist
        prev_to_sync = getattr(self.val_atom_pr_auc, "_to_sync", True)
        self.val_atom_pr_auc._to_sync = False
        # float, 标量, 当前 GPU 本地 PR-AUC 值
        score_local = self.val_atom_pr_auc.compute()
        self.val_atom_pr_auc._to_sync = prev_to_sync

        # torch.Tensor, 标量, 移到当前 GPU 以便 sync_dist 正常工作
        score_gpu = score_local.to(self.device)
        self.val_atom_pr_auc.reset()
        self.log(
            self.hparams.monitor_metric,
            score_gpu,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )

    # ------------------------------------------------------------------
    # 优化器 & 调度器
    # ------------------------------------------------------------------

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
