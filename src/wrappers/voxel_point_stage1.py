from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import lightning as pl
import torch
from hydra.utils import instantiate
from torch import nn

from src.modules.losses import AdaptiveClassificationCompositeLoss
from src.wrappers.voxel_point_stage1_diagnostics import (
    CpcDiagnosticsConfig,
    CpcValidationDiagnostics,
    SourceFolderRegistry,
)
from src.wrappers.voxel_point_stage1_logging import log_scalar_payload, log_wandb_curves, write_validation_artifacts
from src.wrappers.voxel_point_stage1_losses import (
    LossTerm,
    compute_atom_loss_term,
    compute_receptor_loss_term,
    compute_sparse_refine_loss_term,
    compute_voxel_ligand_loss_term,
)
from src.wrappers.voxel_point_stage1_metrics import MetricBranchSpec, ValidationMetricManager
from src.wrappers.voxel_point_stage1_scheduler import configure_stage1_optimizers


class VoxelPointStage1Wrapper(pl.LightningModule):
    """
    Stage1 体素+点融合模型的 Lightning thin coordinator。

    输入参数:
        - backbone: nn.Module 或 Hydra 配置, Stage1 主干网络
        - atom_loss: nn.Module | None, 原子级监督损失
        - voxel_aux_loss: nn.Module | None, receptor 对外语义的体素辅助监督损失
        - voxel_ligand_loss: nn.Module | None, dense ligand 体素监督损失
        - ligand_sparse_refine_loss: nn.Module | None, C 级 sparse refine 监督损失
        - optimizer: Any, Hydra optimizer 配置、callable 或 None
        - scheduler: Any, Hydra scheduler 配置、callable 或 None
        - atom_loss_weight: float, atom loss 静态权重
        - voxel_aux_loss_weight: float, receptor loss 静态权重
        - voxel_ligand_loss_weight: float, voxel ligand loss 静态权重
        - ligand_sparse_refine_loss_weight: float, sparse refine loss 最终权重
        - ligand_sparse_refine_loss_schedule: Mapping[str, Any] | None, sparse refine loss 独立 warmup 配置
        - monitor_metric: str, scheduler/checkpoint 监控指标 key
        - voxel_ligand_pr_auc_thresholds: int | None, dense ligand AP 阈值配置
        - val_metric_device_policy: str, validation metric 设备策略, 允许 auto/cpu/gpu
        - initial_p_best_by_class: Sequence[float] | None, (K,), candidate best-F1 初始阈值
        - initial_p_sampling_by_class: Sequence[float] | None, (K,), candidate sampling 初始阈值
        - class_names: Sequence[str], (C,), task class 名, 必须显式传入
        - validation_diagnostics: Mapping[str, Any], CPC diagnostics 配置
        - interval: str, Lightning scheduler interval
        - frequency: int, Lightning scheduler frequency
        - compile: bool, 是否 torch.compile backbone
    """

    def __init__(
        self,
        backbone: nn.Module,
        atom_loss: nn.Module | None = None,
        voxel_aux_loss: nn.Module | None = None,
        voxel_ligand_loss: nn.Module | None = None,
        ligand_sparse_refine_loss: nn.Module | None = None,
        optimizer: Any = None,
        scheduler: Any = None,
        atom_loss_weight: float = 1.0,
        voxel_aux_loss_weight: float = 0.0,
        voxel_ligand_loss_weight: float = 0.0,
        ligand_sparse_refine_loss_weight: float = 0.0,
        ligand_sparse_refine_loss_schedule: Mapping[str, Any] | None = None,
        monitor_metric: str = "val_score/global/atom_PRAUC",
        voxel_ligand_pr_auc_thresholds: int | None = 1024,
        val_metric_device_policy: str = "auto",
        initial_p_best_by_class: Sequence[float] | None = None,
        initial_p_sampling_by_class: Sequence[float] | None = None,
        class_names: Sequence[str] | None = None,
        validation_diagnostics: Mapping[str, Any] | None = None,
        interval: str = "epoch",
        frequency: int = 1,
        compile: bool = False,
    ) -> None:
        super().__init__()
        if class_names is None:
            raise ValueError("VoxelPointStage1Wrapper 必须显式传入 class_names。")
        self.save_hyperparameters(ignore=["backbone", "atom_loss", "voxel_aux_loss", "voxel_ligand_loss", "ligand_sparse_refine_loss"])
        self.backbone = backbone if isinstance(backbone, nn.Module) else instantiate(backbone)
        self.atom_loss = atom_loss if (atom_loss is None or isinstance(atom_loss, nn.Module)) else instantiate(atom_loss)
        self.voxel_aux_loss = voxel_aux_loss if (voxel_aux_loss is None or isinstance(voxel_aux_loss, nn.Module)) else instantiate(voxel_aux_loss)
        self.voxel_ligand_loss = voxel_ligand_loss if (voxel_ligand_loss is None or isinstance(voxel_ligand_loss, nn.Module)) else instantiate(voxel_ligand_loss)
        self.ligand_sparse_refine_loss = (
            ligand_sparse_refine_loss
            if (ligand_sparse_refine_loss is None or isinstance(ligand_sparse_refine_loss, nn.Module))
            else instantiate(ligand_sparse_refine_loss)
        )
        if self.ligand_sparse_refine_loss is not None and not isinstance(self.ligand_sparse_refine_loss, AdaptiveClassificationCompositeLoss):
            raise TypeError("ligand_sparse_refine_loss 必须为 AdaptiveClassificationCompositeLoss。")
        if compile:
            self.backbone = torch.compile(self.backbone)

        # tuple[str, ...], (C,), task class 名
        self.class_names = tuple(str(name) for name in class_names)
        # tuple[int, ...] | None, (K,), candidate builder 前景类别 ID
        self._sparse_candidate_class_ids = self._resolve_sparse_candidate_class_ids()
        self._validate_sparse_refine_config()
        # torch.Tensor | None, (K,), runtime best-F1 threshold cache
        self._cached_voxel_ligand_p_best_by_class = self._init_candidate_threshold_cache(initial_p_best_by_class, "initial_p_best_by_class")
        # torch.Tensor | None, (K,), runtime sampling threshold cache
        self._cached_voxel_ligand_p_sampling_by_class = self._init_candidate_threshold_cache(initial_p_sampling_by_class, "initial_p_sampling_by_class")
        # torch.Tensor | None, (K,), runtime best-F1 score cache
        self._cached_voxel_ligand_best_f1_before_refine_by_class = None if self._sparse_candidate_class_ids is None else torch.full((len(self._sparse_candidate_class_ids),), float("nan"), dtype=torch.float32)
        # int, candidate fixed-topk warmup step 数
        self._candidate_warmup_steps = 0
        # int, validation epoch end 序号
        self._validation_index = 0
        self._warmup_plateau_scheduler = None
        self._pending_warmup_plateau_state: Mapping[str, Any] | None = None

        self.val_metrics = ValidationMetricManager(
            branches=self._build_metric_branch_specs(voxel_ligand_pr_auc_thresholds),
            metric_device_policy=str(val_metric_device_policy),
        )
        diagnostics_cfg = dict(validation_diagnostics or {})
        self._diagnostics_meta_key = str(diagnostics_cfg.get("source_folder_meta_key", "class_name"))
        self.source_folders = SourceFolderRegistry.from_names(diagnostics_cfg.get("source_folder_names", ("unknown",)))
        self.cpc_diagnostics = self._build_cpc_diagnostics(diagnostics_cfg, voxel_ligand_pr_auc_thresholds)
        self._sync_sparse_candidate_runtime_to_backbone()

    def _unwrap_backbone(self) -> nn.Module:
        """
        返回未被 torch.compile 包装的 backbone。

        输出:
            - backbone: nn.Module, 原始 Stage1 backbone
        """
        return getattr(self.backbone, "_orig_mod", self.backbone)

    def _build_metric_branch_specs(self, voxel_ligand_pr_auc_thresholds: int | None) -> tuple[MetricBranchSpec, ...]:
        """
        构造常规 validation AP/PRAUC 分支配置。

        输入参数:
            - voxel_ligand_pr_auc_thresholds: int | None, dense ligand AP 阈值配置

        输出:
            - specs: tuple[MetricBranchSpec, ...], 可启用 metric 分支配置
        """
        return (
            MetricBranchSpec("atom", self.atom_loss is not None, int(getattr(self.atom_loss, "num_classes", 2)), self.class_names, None),
            MetricBranchSpec("receptor", self.voxel_aux_loss is not None, int(getattr(self.voxel_aux_loss, "num_classes", 2)), self.class_names, None),
            MetricBranchSpec("voxel_ligand", self.voxel_ligand_loss is not None, int(getattr(self.voxel_ligand_loss, "num_classes", 2)), self.class_names, voxel_ligand_pr_auc_thresholds),
        )

    def _build_cpc_diagnostics(self, diagnostics_cfg: Mapping[str, Any], voxel_ligand_pr_auc_thresholds: int | None) -> CpcValidationDiagnostics:
        """
        构造 CPC diagnostics manager。

        输入参数:
            - diagnostics_cfg: Mapping[str, Any], CPC diagnostics 配置
            - voxel_ligand_pr_auc_thresholds: int | None, fallback 到配置 num_bins 的阈值数量

        输出:
            - diagnostics: CpcValidationDiagnostics, validation diagnostics manager
        """
        candidate_builder = getattr(self._unwrap_backbone(), "candidate_set_builder", None)
        candidate_class_ids = self._sparse_candidate_class_ids or (1,)
        adaptive_expand_factor = tuple(getattr(candidate_builder, "adaptive_expand_factor", tuple(1.0 for _ in candidate_class_ids)))
        max_candidate_voxels_per_class = tuple(getattr(candidate_builder, "max_candidate_voxels_per_class", tuple(0 for _ in candidate_class_ids)))
        return CpcValidationDiagnostics(
            config=CpcDiagnosticsConfig(
                enabled=bool(diagnostics_cfg.get("enabled", True)),
                source_folder_breakdown=bool(diagnostics_cfg.get("source_folder_breakdown", True)),
                num_bins=int(diagnostics_cfg.get("num_bins", voxel_ligand_pr_auc_thresholds or 1024)),
                write_local_artifacts=bool(diagnostics_cfg.get("write_local_artifacts", True)),
                log_wandb_curves=bool(diagnostics_cfg.get("log_wandb_curves", True)),
                wandb_curve_every_n_validation=int(diagnostics_cfg.get("wandb_curve_every_n_validation", 1)),
                output_subdir=str(diagnostics_cfg.get("output_subdir", "validation_diagnostics")),
            ),
            class_names=self.class_names,
            candidate_class_ids=candidate_class_ids,
            source_folders=self.source_folders,
            adaptive_expand_factor=adaptive_expand_factor,
            max_candidate_voxels_per_class=max_candidate_voxels_per_class,
        )

    def _resolve_sparse_candidate_class_ids(self) -> tuple[int, ...] | None:
        """
        从 backbone 读取 sparse candidate class ids。

        输出:
            - class_ids: tuple[int, ...] | None, (K,), builder 未启用时为 None
        """
        backbone = self._unwrap_backbone()
        if not hasattr(backbone, "get_sparse_candidate_class_ids"):
            return None
        class_ids = backbone.get_sparse_candidate_class_ids()
        return None if class_ids is None else tuple(int(class_id) for class_id in class_ids)

    def _validate_sparse_refine_config(self) -> None:
        """
        校验 sparse refine loss 与 candidate builder 配置关系。

        输出:
            - None, 配置不一致时 fail-fast
        """
        if self.ligand_sparse_refine_loss is None:
            return
        if self._sparse_candidate_class_ids is None:
            raise ValueError("ligand_sparse_refine_loss 启用时必须同时启用 sparse candidate builder。")
        num_classes = int(self.ligand_sparse_refine_loss.num_classes)
        invalid_ids = [class_id for class_id in self._sparse_candidate_class_ids if class_id <= 0 or class_id >= num_classes]
        if invalid_ids:
            raise ValueError(f"candidate_class_ids={self._sparse_candidate_class_ids} 与 ligand_sparse_refine_loss.num_classes={num_classes} 不匹配。")

    def _init_candidate_threshold_cache(self, initial_values: Sequence[float] | None, value_name: str) -> torch.Tensor | None:
        """
        从显式 initial 参数初始化 candidate threshold cache。

        输入参数:
            - initial_values: Sequence[float] | None, (K,), 用户显式给定的初始阈值
            - value_name: str, 错误信息中的参数名

        输出:
            - cache: torch.Tensor | None, (K,), builder 未启用或未提供 initial 时为 None
        """
        if self._sparse_candidate_class_ids is None:
            if initial_values is not None:
                raise ValueError(f"{value_name} 只能在 candidate builder 启用时配置。")
            return None
        if initial_values is None:
            return None
        cache = torch.as_tensor(list(initial_values), dtype=torch.float32).reshape(-1)
        if int(cache.numel()) != len(self._sparse_candidate_class_ids):
            raise ValueError(f"{value_name} 长度必须等于 candidate_class_ids 数量。")
        if not bool(torch.isfinite(cache).all()):
            raise ValueError(f"{value_name} 不能包含 NaN/Inf。")
        return cache

    @staticmethod
    def _extract_batch(batch: Any) -> dict[str, Any]:
        """
        校验并透传 batch 字典。

        输入参数:
            - batch: Any, DataLoader 产出的 batch

        输出:
            - batch_dict: dict[str, Any], 当前 batch 字典
        """
        if not isinstance(batch, dict):
            raise TypeError(f"VoxelPointStage1Wrapper expects dict batch, got {type(batch)!r}")
        return batch

    def forward(self, batch: dict[str, Any]) -> dict[str, Any]:
        """
        前向推理, 直接委托给 backbone。

        输入参数:
            - batch: dict[str, Any], 当前 batch 字典

        输出:
            - outputs: dict[str, Any], backbone 输出字典
        """
        return self.backbone(batch)

    @staticmethod
    def _normalize_voxel_valid_mask(voxel_valid_mask: torch.Tensor, spatial_shape_zyx: tuple[int, int, int]) -> torch.Tensor:
        """
        将 voxel_valid_mask 规范化为 `(B,D,H,W)` bool 掩码。

        输入参数:
            - voxel_valid_mask: torch.Tensor, (B,D,H,W) 或 (B,1,D,H,W), 有效体素掩码
            - spatial_shape_zyx: tuple[int, int, int], 期望空间形状 `(D,H,W)`

        输出:
            - mask: torch.Tensor, (B,D,H,W), bool 有效体素掩码
        """
        mask = voxel_valid_mask.squeeze(1).bool() if voxel_valid_mask.ndim == 5 and voxel_valid_mask.shape[1] == 1 else voxel_valid_mask.bool()
        if tuple(mask.shape[-3:]) != spatial_shape_zyx:
            raise ValueError(f"voxel_valid_mask 空间形状 {tuple(mask.shape[-3:])} 与 dense target {spatial_shape_zyx} 不一致。")
        return mask

    def _ligand_target_from_dist(self, ligand_dist_map: torch.Tensor, logit_dim: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """
        从 voxel ligand loss 配置生成 dense hard target。

        输入参数:
            - ligand_dist_map: torch.Tensor, (B,D,H,W)/(B,1,D,H,W)/(B,C,D,H,W), ligand 距离监督图
            - logit_dim: int, 当前 logits 通道数
            - device: torch.device, 输出 target 所在设备
            - dtype: torch.dtype, 距离图计算 dtype

        输出:
            - target: torch.Tensor, (B,D,H,W), hard-label target
        """
        if self.voxel_ligand_loss is None or not hasattr(self.voxel_ligand_loss, "target_from_ligand_dist_map"):
            raise TypeError("voxel_ligand_loss 必须支持 target_from_ligand_dist_map()。")
        return self.voxel_ligand_loss.target_from_ligand_dist_map(ligand_dist_map=ligand_dist_map, logit_dim=logit_dim, device=device, dtype=dtype)

    def _sample_ligand_refine_supervision(self, outputs: dict[str, Any], batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        """
        从 dense ligand 距离图采样 C 级 sparse refine 监督。

        输入参数:
            - outputs: dict[str, Any], backbone 输出, 包含 ligand_refine_logits_C/candidate_batch_index/candidate_voxel_zyx
            - batch: dict[str, Any], 当前 batch, 包含 ligand_dist_map/voxel_valid_mask

        输出:
            - supervision: dict[str, torch.Tensor], dense target/mask 与 C 级 target/mask
        """
        logits_C = outputs["ligand_refine_logits_C"]
        dense_target = self.ligand_sparse_refine_loss.target_from_ligand_dist_map(
            ligand_dist_map=batch["ligand_dist_map"],
            logit_dim=int(logits_C.shape[1]),
            device=logits_C.device,
            dtype=logits_C.dtype,
        )
        dense_valid_mask = self._normalize_voxel_valid_mask(
            batch["voxel_valid_mask"],
            spatial_shape_zyx=tuple(int(value) for value in dense_target.shape[-3:]),
        ).to(device=logits_C.device)
        idx_b = outputs["candidate_batch_index"].to(device=logits_C.device, dtype=torch.long)
        idx_zyx = outputs["candidate_voxel_zyx"].to(device=logits_C.device, dtype=torch.long)
        target_C = dense_target[idx_b, idx_zyx[:, 0], idx_zyx[:, 1], idx_zyx[:, 2]]
        valid_C = dense_valid_mask[idx_b, idx_zyx[:, 0], idx_zyx[:, 1], idx_zyx[:, 2]]
        supervision = {
            "ligand_dense_target": dense_target,
            "ligand_dense_valid_mask": dense_valid_mask,
            "ligand_refine_target_C": target_C,
            "ligand_refine_valid_mask_C": valid_C,
        }
        outputs.update(supervision)
        return supervision

    def _compute_sparse_refine_loss_effective_weight(self) -> torch.Tensor:
        """
        计算当前 global_step 下 sparse refine loss 的有效权重。

        输出:
            - weight: torch.Tensor, (), 当前 step 使用的 loss 权重
        """
        sched_cfg = self.hparams.ligand_sparse_refine_loss_schedule
        if sched_cfg is None:
            return torch.tensor(float(self.hparams.ligand_sparse_refine_loss_weight), device=self.device, dtype=torch.float32)
        start_weight = float(sched_cfg["start_weight"])
        final_weight = float(sched_cfg["final_weight"])
        warmup_steps = int(sched_cfg["warmup_steps"])
        if warmup_steps == 0:
            return torch.tensor(final_weight, device=self.device, dtype=torch.float32)
        try:
            trainer = self.trainer
        except RuntimeError:
            trainer = None
        global_step = int(getattr(trainer, "global_step", self.global_step)) if trainer is not None else int(self.global_step)
        progress = min(max(float(global_step) / float(warmup_steps), 0.0), 1.0)
        return torch.tensor(start_weight + progress * (final_weight - start_weight), device=self.device, dtype=torch.float32)

    def _compute_total_loss(self, outputs: dict[str, Any], batch: dict[str, Any]) -> tuple[torch.Tensor, list[LossTerm], dict[str, torch.Tensor]]:
        """
        汇总各监督分支 loss。

        输入参数:
            - outputs: dict[str, Any], backbone 输出
            - batch: dict[str, Any], 当前 batch 字典

        输出:
            - total_loss: torch.Tensor, (), 加权总损失
            - loss_terms: list[LossTerm], 各分支 loss term
            - extra_logs: dict[str, torch.Tensor], 额外 loss 日志项
        """
        total_loss = torch.tensor(0.0, device=self.device, dtype=torch.float32)
        loss_terms: list[LossTerm] = []
        extra_logs: dict[str, torch.Tensor] = {}
        if self.atom_loss is not None:
            loss_terms.append(compute_atom_loss_term(outputs=outputs, batch=batch, loss_module=self.atom_loss, weight=float(self.hparams.atom_loss_weight)))
        if self.voxel_aux_loss is not None:
            term = compute_receptor_loss_term(outputs=outputs, batch=batch, loss_module=self.voxel_aux_loss, weight=float(self.hparams.voxel_aux_loss_weight))
            if term is not None:
                loss_terms.append(term)
        if self.voxel_ligand_loss is not None:
            term = compute_voxel_ligand_loss_term(outputs=outputs, batch=batch, loss_module=self.voxel_ligand_loss, weight=float(self.hparams.voxel_ligand_loss_weight))
            if term is not None:
                loss_terms.append(term)
        if self.ligand_sparse_refine_loss is not None and outputs.get("ligand_refine_logits_C") is not None:
            supervision = self._sample_ligand_refine_supervision(outputs=outputs, batch=batch)
            effective_weight = self._compute_sparse_refine_loss_effective_weight()
            term, logged_weight = compute_sparse_refine_loss_term(
                logits_C=outputs["ligand_refine_logits_C"],
                target_C=supervision["ligand_refine_target_C"],
                valid_C=supervision["ligand_refine_valid_mask_C"],
                loss_module=self.ligand_sparse_refine_loss,
                weight=float(self.hparams.ligand_sparse_refine_loss_weight),
                effective_weight=effective_weight,
            )
            loss_terms.append(term)
            extra_logs["ligand_sparse_refine_weight_effective"] = logged_weight
        for term in loss_terms:
            weight = extra_logs["ligand_sparse_refine_weight_effective"] if term.name == "ligand_sparse_refine" else term.value.new_tensor(term.weight)
            total_loss = total_loss + weight * term.value
        return total_loss, loss_terms, extra_logs

    def _log_loss_terms(self, prefix: str, total_loss: torch.Tensor, loss_terms: Sequence[LossTerm], extra_logs: Mapping[str, torch.Tensor]) -> None:
        """
        记录 train/val loss 日志。

        输入参数:
            - prefix: str, 日志前缀, train_loss 或 val_loss
            - total_loss: torch.Tensor, (), 加权总损失
            - loss_terms: Sequence[LossTerm], 各分支 loss term
            - extra_logs: Mapping[str, torch.Tensor], 额外 loss 日志项

        输出:
            - None, 原地调用 self.log
        """
        self.log(f"{prefix}/global/total", total_loss, prog_bar=self.hparams.monitor_metric == f"{prefix}/global/total", on_step=prefix == "train_loss", on_epoch=True, sync_dist=True)
        for term in loss_terms:
            self.log(f"{prefix}/global/{term.name}", term.logged_value, prog_bar=False, on_step=prefix == "train_loss", on_epoch=True, sync_dist=True)
        for name, value in extra_logs.items():
            self.log(f"{prefix}/global/{name}", value, prog_bar=False, on_step=prefix == "train_loss", on_epoch=True, sync_dist=True)

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        """
        训练步: 同步 candidate runtime -> forward -> loss -> 日志。

        输入参数:
            - batch: Any, DataLoader 产出的 batch
            - batch_idx: int, 当前 batch index

        输出:
            - total_loss: torch.Tensor, (), 加权总损失
        """
        del batch_idx
        batch_dict = self._extract_batch(batch)
        self._sync_sparse_candidate_runtime_to_backbone()
        outputs = self(batch_dict)
        total_loss, loss_terms, extra_logs = self._compute_total_loss(outputs=outputs, batch=batch_dict)
        self._log_loss_terms("train_loss", total_loss, loss_terms, extra_logs)
        if "recycle_passes_used" in outputs:
            self.log("train/runtime/recycle_passes", float(outputs["recycle_passes_used"]), prog_bar=False, on_step=True, on_epoch=False, sync_dist=True)
        return total_loss

    def validation_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        """
        验证步: 按 dense -> C -> P -> refined 时间顺序编排 helper 更新。

        输入参数:
            - batch: Any, DataLoader 产出的 batch
            - batch_idx: int, 当前 batch index

        输出:
            - total_loss: torch.Tensor, (), 加权总损失
        """
        del batch_idx
        batch_dict = self._extract_batch(batch)
        source_folder_idx = self.source_folders.encode_batch(batch_dict[self._diagnostics_meta_key], device=self.device)
        self._sync_sparse_candidate_runtime_to_backbone()
        outputs = self(batch_dict)
        total_loss, loss_terms, extra_logs = self._compute_total_loss(outputs=outputs, batch=batch_dict)
        if self.atom_loss is not None and "atom_logits" in outputs:
            atom_mask = outputs.get("atom_valid_mask", batch_dict.get("atom_valid_mask", torch.ones_like(batch_dict["atom_label"], dtype=torch.bool)))
            self.val_metrics.update_branch(branch_name="atom", logits=outputs["atom_logits"], target=outputs.get("atom_target", batch_dict["atom_label"]), mask=atom_mask)
        if self.voxel_aux_loss is not None and "voxel_logits_aux" in outputs:
            receptor_mask = (batch_dict["hardmask"].bool() & batch_dict["voxel_valid_mask"].bool()).squeeze(1)
            self.val_metrics.update_branch(branch_name="receptor", logits=outputs["voxel_logits_aux"], target=batch_dict["voxel_label"], mask=receptor_mask)
        if self.voxel_ligand_loss is not None and "voxel_logits_ligand" in outputs and "ligand_dist_map" in batch_dict:
            ligand_target = self._ligand_target_from_dist(batch_dict["ligand_dist_map"], int(outputs["voxel_logits_ligand"].shape[1]), outputs["voxel_logits_ligand"].device, outputs["voxel_logits_ligand"].dtype)
            ligand_valid = self._normalize_voxel_valid_mask(batch_dict["voxel_valid_mask"], tuple(int(value) for value in ligand_target.shape[-3:]))
            self.val_metrics.update_branch(branch_name="voxel_ligand", logits=outputs["voxel_logits_ligand"], target=ligand_target, mask=ligand_valid, source_folder_idx=source_folder_idx)
            allow_cache_update = self._allow_validation_cache_update()
            self.cpc_diagnostics.update_uncapped_best(logits=outputs["voxel_logits_ligand"], target=ligand_target, valid_mask=ligand_valid, source_folder_idx=source_folder_idx, allow_cache_update=allow_cache_update)
            candidate_outputs: Mapping[str, torch.Tensor] = outputs
            self.cpc_diagnostics.update_uncapped_sampling(logits=outputs["voxel_logits_ligand"], target=ligand_target, valid_mask=ligand_valid, candidate_outputs=candidate_outputs, source_folder_idx=source_folder_idx, selection_mode=self._candidate_selection_mode(), use_fixed_warmup=self._using_candidate_warmup())
            self.cpc_diagnostics.update_capped(target=ligand_target, valid_mask=ligand_valid, candidate_outputs=candidate_outputs, source_folder_idx=source_folder_idx)
            if "ligand_refine_target_C" in outputs:
                dense_num_gt = self._dense_num_gt_by_box(ligand_target, ligand_valid)
                self.cpc_diagnostics.update_unrefined(candidate_outputs=candidate_outputs, target_C=outputs["ligand_refine_target_C"], valid_C=outputs["ligand_refine_valid_mask_C"], dense_num_gt=dense_num_gt, source_folder_idx=source_folder_idx)
                if "ligand_refine_logits_C" in outputs:
                    self.cpc_diagnostics.update_refined(refined_logits_C=outputs["ligand_refine_logits_C"], candidate_outputs=candidate_outputs, target_C=outputs["ligand_refine_target_C"], valid_C=outputs["ligand_refine_valid_mask_C"], dense_num_gt=dense_num_gt, source_folder_idx=source_folder_idx)
        self._log_loss_terms("val_loss", total_loss, loss_terms, extra_logs)
        return total_loss

    def _dense_num_gt_by_box(self, target: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        """
        统计每个 BOX 中 candidate class 的 dense GT 正例数。

        输入参数:
            - target: torch.Tensor, (B,D,H,W), hard-label target
            - valid_mask: torch.Tensor, (B,D,H,W), 有效体素掩码

        输出:
            - dense_num_gt: torch.Tensor, (B,), 每个 BOX 的 GT 正例数
        """
        class_ids = self._sparse_candidate_class_ids or (1,)
        positive = torch.zeros_like(target, dtype=torch.bool)
        for class_id in class_ids:
            positive |= target.long() == int(class_id)
        return (positive & valid_mask.bool()).reshape(target.shape[0], -1).sum(dim=1)

    def _candidate_selection_mode(self) -> str:
        """
        返回当前 candidate builder selection mode。

        输出:
            - selection_mode: str, candidate selection mode
        """
        candidate_builder = getattr(self._unwrap_backbone(), "candidate_set_builder", None)
        return str(getattr(candidate_builder, "selection_mode", "adaptive_threshold"))

    def _using_candidate_warmup(self) -> bool:
        """
        判断当前 validation 是否处于 candidate warmup fixed-topk 阶段。

        输出:
            - using_warmup: bool, True 表示当前 global_step 小于 candidate warmup steps
        """
        return int(self.global_step) < int(self._candidate_warmup_steps)

    def _allow_validation_cache_update(self) -> bool:
        """
        判断当前 validation 是否允许写回 candidate threshold cache。

        输出:
            - allow_update: bool, True 表示普通 fit validation 可写 cache
        """
        try:
            trainer = self.trainer
        except RuntimeError:
            return False
        return trainer is not None and not bool(getattr(trainer, "sanity_checking", False)) and not self._is_tuning_trainer(trainer)

    @staticmethod
    def _is_tuning_trainer(trainer: Any) -> bool:
        """
        判断当前 trainer 是否处于 Lightning tuner 生命周期。

        输入参数:
            - trainer: Any, Lightning Trainer 或测试 stub

        输出:
            - is_tuning: bool, True 表示 tuner 探测阶段
        """
        state_fn = getattr(getattr(trainer, "state", None), "fn", None)
        state_fn_name = str(getattr(state_fn, "value", state_fn)).lower()
        return "tun" in state_fn_name

    def _sync_sparse_candidate_runtime_to_backbone(self) -> None:
        """
        将 wrapper 中 candidate runtime/cache 同步给 backbone。

        输出:
            - None, builder 未启用时 no-op
        """
        if self._sparse_candidate_class_ids is None:
            return
        backbone = self._unwrap_backbone()
        try:
            trainer = self.trainer
        except RuntimeError:
            trainer = None
        global_step = int(getattr(trainer, "global_step", self.global_step)) if trainer is not None else int(self.global_step)
        allow_warmup = False if trainer is None else bool(getattr(trainer, "sanity_checking", False)) or "fit" in str(getattr(getattr(trainer, "state", None), "fn", "")).lower() or self._is_tuning_trainer(trainer)
        if hasattr(backbone, "set_sparse_candidate_runtime"):
            backbone.set_sparse_candidate_runtime(global_step=global_step, candidate_warmup_steps=int(self._candidate_warmup_steps), allow_warmup_fixed_topk=allow_warmup)
        if hasattr(backbone, "set_sparse_candidate_thresholds"):
            backbone.set_sparse_candidate_thresholds(p_best_by_class=self._cached_voxel_ligand_p_best_by_class, p_sampling_by_class=self._cached_voxel_ligand_p_sampling_by_class)

    def on_validation_epoch_start(self) -> None:
        """
        validation epoch 开始时重置 helper manager 状态。

        输出:
            - None, 原地 reset validation managers
        """
        self.val_metrics.reset()
        self.cpc_diagnostics.reset()

    def on_validation_epoch_end(self) -> None:
        """
        validation epoch 结束时计算 payload、写日志与 artifact。

        输出:
            - None, 原地记录 validation scalar/curve/artifact
        """
        metric_payload = self.val_metrics.compute_payload()
        cpc_payload = self.cpc_diagnostics.compute_payload(sync_fn=self._all_reduce_sum)
        payload = {**metric_payload, **cpc_payload.scalars}
        self._update_candidate_threshold_cache_from_payload(payload)
        self._sync_sparse_candidate_runtime_to_backbone()
        log_scalar_payload(module=self, payload=payload, monitor_metric=str(self.hparams.monitor_metric), sync_dist=True)
        trainer = self.trainer
        if bool(getattr(trainer, "is_global_zero", True)):
            if self.cpc_diagnostics.config.write_local_artifacts:
                write_validation_artifacts(run_dir=self._run_dir(), output_subdir=self.cpc_diagnostics.config.output_subdir, epoch=int(self.current_epoch), global_step=int(self.global_step), payload=cpc_payload)
            if self.cpc_diagnostics.config.log_wandb_curves:
                log_wandb_curves(module=self, curves=cpc_payload.curves, validation_index=int(self._validation_index), every_n=int(self.cpc_diagnostics.config.wandb_curve_every_n_validation))
        self._validation_index += 1
        self.val_metrics.reset()
        self.cpc_diagnostics.reset()
        self._step_warmup_plateau_scheduler(payload)

    def _update_candidate_threshold_cache_from_payload(self, payload: Mapping[str, torch.Tensor]) -> None:
        """
        从 diagnostics payload 写回 candidate threshold cache。

        输入参数:
            - payload: Mapping[str, torch.Tensor], validation epoch scalar payload

        输出:
            - None, 原地更新 runtime cache
        """
        if self._sparse_candidate_class_ids is None or not self._allow_validation_cache_update():
            return
        p_best_values: list[torch.Tensor] = []
        best_f1_values: list[torch.Tensor] = []
        for class_id in self._sparse_candidate_class_ids:
            class_name = self.class_names[int(class_id)]
            suffix = "" if len(self.class_names) <= 2 else f"_{class_name}"
            p_best_values.append(payload[f"val_uncapped/best/global/p_best{suffix}"].detach().cpu().float())
            best_f1_values.append(payload[f"val_uncapped/best/global/best_F1{suffix}"].detach().cpu().float())
        p_best = torch.stack(p_best_values).reshape(-1)
        best_f1 = torch.stack(best_f1_values).reshape(-1)
        if bool(torch.isfinite(p_best).all()):
            self._cached_voxel_ligand_p_best_by_class = p_best
            self._cached_voxel_ligand_p_sampling_by_class = p_best.clone()
        if bool(torch.isfinite(best_f1).all()):
            self._cached_voxel_ligand_best_f1_before_refine_by_class = best_f1

    def _all_reduce_sum(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        对固定形状 tensor 执行 DDP sum 同步。

        输入参数:
            - tensor: torch.Tensor, 任意固定形状, 当前 rank 统计值

        输出:
            - reduced: torch.Tensor, 与输入同形状, all-reduce sum 后统计值
        """
        if not torch.distributed.is_available() or not torch.distributed.is_initialized():
            return tensor
        reduced = tensor.clone()
        torch.distributed.all_reduce(reduced, op=torch.distributed.ReduceOp.SUM)
        return reduced

    def _run_dir(self) -> Path:
        """
        解析当前 logger/run 输出目录。

        输出:
            - run_dir: Path, validation artifact 根目录
        """
        trainer = self.trainer
        logger = getattr(self, "logger", None)
        log_dir = getattr(logger, "save_dir", None) or getattr(trainer, "default_root_dir", ".")
        return Path(log_dir)

    def _step_warmup_plateau_scheduler(self, computed_metrics: Mapping[str, torch.Tensor]) -> None:
        """
        validation end 后推进手动 warmup_plateau scheduler。

        输入参数:
            - computed_metrics: Mapping[str, torch.Tensor], 当前 validation scalar payload

        输出:
            - None, 未启用 warmup_plateau 时 no-op
        """
        if self._warmup_plateau_scheduler is None or self.trainer.sanity_checking:
            return
        monitor_name = str(self.hparams.monitor_metric)
        if monitor_name not in computed_metrics:
            raise RuntimeError(f"warmup_plateau scheduler monitor metric {monitor_name!r} is not available after validation.")
        metric_value = computed_metrics[monitor_name].detach().to(self.device).float().reshape(())
        self._warmup_plateau_scheduler.step_plateau(metric_value, global_step=int(self.global_step))

    def _normalize_candidate_checkpoint_tensor(self, value: Any, value_name: str) -> torch.Tensor:
        """
        将 checkpoint 中的 candidate cache 规范化为 CPU float 向量。

        输入参数:
            - value: Any, checkpoint 中读取的张量或可转张量对象
            - value_name: str, 错误信息中的字段名

        输出:
            - tensor: torch.Tensor, (K,), CPU float candidate cache
        """
        tensor = torch.as_tensor(value).detach().cpu().float().reshape(-1)
        if int(tensor.numel()) != len(self._sparse_candidate_class_ids):
            raise ValueError(f"checkpoint 中的 {value_name} 长度必须等于 candidate_class_ids 数量。")
        return tensor

    def _load_candidate_checkpoint_finite_tensor(self, value: Any, value_name: str) -> torch.Tensor:
        """
        从 checkpoint 读取并校验 finite candidate cache。

        输入参数:
            - value: Any, checkpoint 中读取的张量或可转张量对象
            - value_name: str, 错误信息中的字段名

        输出:
            - tensor: torch.Tensor, (K,), finite CPU float candidate cache
        """
        tensor = self._normalize_candidate_checkpoint_tensor(value, value_name)
        if not bool(torch.isfinite(tensor).all()):
            raise ValueError(f"checkpoint 中的 {value_name} 不能包含 NaN/Inf。")
        return tensor

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        """
        保存训练继续所需 runtime state。

        输入参数:
            - checkpoint: dict[str, Any], Lightning checkpoint 字典

        输出:
            - None, 原地写入 scheduler/candidate cache
        """
        if self._warmup_plateau_scheduler is not None:
            checkpoint["warmup_plateau_reduce_on_plateau_state"] = self._warmup_plateau_scheduler.state_dict()
        if self._sparse_candidate_class_ids is None:
            return
        checkpoint["voxel_ligand_candidate_class_ids"] = self._sparse_candidate_class_ids
        if self._cached_voxel_ligand_p_best_by_class is None or self._cached_voxel_ligand_p_sampling_by_class is None:
            return
        p_best = self._normalize_candidate_checkpoint_tensor(self._cached_voxel_ligand_p_best_by_class, "voxel_ligand_p_best_by_class")
        p_sampling = self._normalize_candidate_checkpoint_tensor(self._cached_voxel_ligand_p_sampling_by_class, "voxel_ligand_p_sampling_by_class")
        if not (bool(torch.isfinite(p_best).all()) and bool(torch.isfinite(p_sampling).all())):
            return
        checkpoint["voxel_ligand_p_best_by_class"] = p_best
        checkpoint["voxel_ligand_p_sampling_by_class"] = p_sampling
        if self._cached_voxel_ligand_best_f1_before_refine_by_class is not None:
            best_f1 = self._normalize_candidate_checkpoint_tensor(self._cached_voxel_ligand_best_f1_before_refine_by_class, "voxel_ligand_best_f1_before_refine_by_class")
            if bool(torch.isfinite(best_f1).all()):
                checkpoint["voxel_ligand_best_f1_before_refine_by_class"] = best_f1

    def on_load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        """
        恢复训练继续所需 runtime state。

        输入参数:
            - checkpoint: dict[str, Any], Lightning checkpoint 字典

        输出:
            - None, 原地恢复 scheduler/candidate cache
        """
        if self._sparse_candidate_class_ids is not None:
            checkpoint_class_ids = checkpoint.get("voxel_ligand_candidate_class_ids", None)
            if checkpoint_class_ids is not None and tuple(int(x) for x in checkpoint_class_ids) != self._sparse_candidate_class_ids:
                raise ValueError("checkpoint 中的 voxel_ligand_candidate_class_ids 与当前 candidate_class_ids 不一致。")
            has_p_best = "voxel_ligand_p_best_by_class" in checkpoint
            has_p_sampling = "voxel_ligand_p_sampling_by_class" in checkpoint
            if has_p_best != has_p_sampling:
                raise ValueError("checkpoint 中的 voxel_ligand_p_best_by_class 与 voxel_ligand_p_sampling_by_class 必须成对出现。")
            if has_p_best:
                self._cached_voxel_ligand_p_best_by_class = self._load_candidate_checkpoint_finite_tensor(checkpoint["voxel_ligand_p_best_by_class"], "voxel_ligand_p_best_by_class")
                self._cached_voxel_ligand_p_sampling_by_class = self._load_candidate_checkpoint_finite_tensor(checkpoint["voxel_ligand_p_sampling_by_class"], "voxel_ligand_p_sampling_by_class")
            if "voxel_ligand_best_f1_before_refine_by_class" in checkpoint:
                self._cached_voxel_ligand_best_f1_before_refine_by_class = self._load_candidate_checkpoint_finite_tensor(checkpoint["voxel_ligand_best_f1_before_refine_by_class"], "voxel_ligand_best_f1_before_refine_by_class")
            self._sync_sparse_candidate_runtime_to_backbone()
        if "warmup_plateau_reduce_on_plateau_state" in checkpoint:
            self._pending_warmup_plateau_state = checkpoint["warmup_plateau_reduce_on_plateau_state"]

    def configure_optimizers(self) -> Any:
        """
        配置 optimizer 与 scheduler。

        输出:
            - config: Any, Lightning configure_optimizers 返回值
        """
        config, warmup_steps, plateau_scheduler, pending_state = configure_stage1_optimizers(
            module=self,
            optimizer_config=self.hparams.optimizer,
            scheduler_config=self.hparams.scheduler,
            interval=str(self.hparams.interval),
            frequency=int(self.hparams.frequency),
            monitor_metric=str(self.hparams.monitor_metric),
            pending_warmup_plateau_state=self._pending_warmup_plateau_state,
        )
        self._candidate_warmup_steps = int(warmup_steps)
        self._warmup_plateau_scheduler = plateau_scheduler
        self._pending_warmup_plateau_state = pending_state
        self._sync_sparse_candidate_runtime_to_backbone()
        return config
