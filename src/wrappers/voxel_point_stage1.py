"""连接 Stage1 模型、正式损失、验证指标和 Lightning 生命周期。

主要入口 :class:`VoxelPointStage1Wrapper` 接收
``Stage1BatchCollator`` 生成的批次字典，将前向输出解释为受体原子、受体结合区域、
配体区域、蛋白主链、核酸主链和配体距离监督。它还负责优化器、学习率调度、
验证指标、候选阈值与 checkpoint 状态，但不重复实现模型网络或 Dataset。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import lightning as pl
import torch
from hydra.utils import instantiate
from torch import nn

from src.auxiliary_supervision import (
    NUCLEIC_MAINCHAIN_CLASS_NAMES,
    PROTEIN_MAINCHAIN_CLASS_NAMES,
)
from src.modules.losses import AdaptiveClassificationCompositeLoss, LigandSparseRefineDeltaLoss
from src.wrappers.voxel_point_stage1_diagnostics import (
    CpcDiagnosticsConfig,
    CpcValidationDiagnostics,
)
from src.wrappers.voxel_point_stage1_logging import log_scalar_payload, log_wandb_curves, write_validation_artifacts
from src.wrappers.voxel_point_stage1_losses import (
    LossTerm,
    compute_atom_loss_term,
    compute_ligand_distance_loss_term,
    compute_mainchain_class_loss_term,
    compute_pseudo_loss_term,
    compute_receptor_loss_term,
    compute_sparse_refine_loss_term,
    compute_voxel_ligand_loss_term,
)
from src.wrappers.voxel_point_stage1_metrics import MetricBranchSpec, ValidationMetricManager
from src.wrappers.voxel_point_stage1_scheduler import configure_stage1_optimizers
from src.utils.module_freeze import set_fully_frozen_submodules_eval


class VoxelPointStage1Wrapper(pl.LightningModule):
    """
    协调 Stage1 模型、监督、指标、调度和 checkpoint 生命周期。

    本类不重新实现模型网络；普通 ``forward`` 和只计算体素概率的调用都转发给
    ``backbone``。它把模型输出按当前 CPC 配置组合成加权总损失，并保存下一阶段
    只恢复模型参数时仍需继承的候选阈值状态。
    输入参数:
        - 初始化参数: 见 `__init__` 的完整参数契约

    前向输入:
        - batch: dict[str,Any]，Stage1Dataset/Collator 生成的批次字典；
          稠密体素字段为 ``(B, ..., D, H, W)``，受体原子字段按第 0 维拼接。

    前向输出:
        - outputs: dict[str,Any]，模型输出字典，包含体素、点、受体原子预测及必要的对齐字段。
    """

    def __init__(
        self,
        backbone: nn.Module,
        name: str = "default",
        atom_loss: nn.Module | None = None,
        voxel_aux_loss: nn.Module | None = None,
        voxel_ligand_loss: nn.Module | None = None,
        protein_mainchain_loss: nn.Module | None = None,
        nucleic_mainchain_loss: nn.Module | None = None,
        ligand_sparse_refine_loss: nn.Module | None = None,
        ligand_sparse_refine_delta_loss: nn.Module | None = None,
        ligand_pseudo_loss: nn.Module | None = None,
        optimizer: Any = None,
        scheduler: Any = None,
        ligand_sparse_refine_w_rank: float = 0.0,
        atom_loss_weight: float = 1.0,
        voxel_aux_loss_weight: float = 0.0,
        voxel_ligand_loss_weight: float = 0.0,
        protein_mainchain_loss_weight: float = 0.0,
        nucleic_mainchain_loss_weight: float = 0.0,
        ligand_distance_loss_weight: float = 0.0,
        ligand_sparse_refine_loss_weight: float = 0.0,
        ligand_sparse_refine_loss_schedule: Mapping[str, Any] | None = None,
        pseudo_loss_weight: float = 1.0,
        monitor_metric: str = "val_score/global/atom_PRAUC",
        monitor_mode: str = "max",
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
        """
        初始化 Stage1 体素与点融合模型的 Lightning 协调器。

        输入参数:
            - 基本
                - backbone: nn.Module 或 Hydra 配置, Stage1 主干网络
                - name: str, 模型配置名, 作为 Hydra 元信息保存
                - class_names: Sequence[str], (C,), task class 名, 必须显式传入
                - validation_diagnostics: Mapping[str, Any], CPC diagnostics 配置

            - 损失
                - atom_loss: nn.Module | None, 原子级监督损失
                - voxel_aux_loss: nn.Module | None, receptor 对外语义的体素辅助监督损失
                - voxel_ligand_loss: nn.Module | None, dense ligand 体素监督损失
                - protein_mainchain_loss: nn.Module | None，蛋白背景/N/CA/C/O 多分类复合损失。
                - nucleic_mainchain_loss: nn.Module | None，核酸背景/P/O5'/C5'/C4'/C3'/O3' 多分类复合损失。
                - ligand_sparse_refine_loss: nn.Module | None, C 级 sparse refine 分类监督损失(L_cls)
                - ligand_sparse_refine_delta_loss: nn.Module | None, LigandSparseRefineDeltaLoss(ranking-only); None 或 w_rank=0 时退化为纯分类
                - ligand_sparse_refine_w_rank: float, ranking(L_rank)权重
                - ligand_pseudo_loss: nn.Module | None, P(虚拟原子) ligand 区域归属监督损失(单通道 sigmoid composite); None 表示不计算 P 监督
                - atom_loss_weight: float, 最终 atom loss 静态权重
                - voxel_aux_loss_weight: float, receptor loss 静态权重
                - voxel_ligand_loss_weight: float, voxel ligand loss 静态权重
                - protein_mainchain_loss_weight: float，蛋白主链分类损失的静态权重。
                - nucleic_mainchain_loss_weight: float，核酸主链分类损失的静态权重。
                - ligand_distance_loss_weight: float，配体反距离全体素平均 MSE 的静态权重。
                - ligand_sparse_refine_loss_weight: float, sparse refine loss 最终权重
                - pseudo_loss_weight: float, 最终 pseudo loss 静态权重

            - 优化器和调度器
                - optimizer: Any, Hydra optimizer 配置、callable 或 None
                - scheduler: Any, Hydra scheduler 配置、callable 或 None
                - ligand_sparse_refine_loss_schedule: Mapping[str, Any] | None, sparse refine loss 独立调度配置; 含 start_on(硬 0 延迟)与 warmup(线性升)两段, 要求 start_on <= warmup
                - interval: str, Lightning scheduler interval
                - frequency: int, Lightning scheduler frequency

            - 性能度量
                - monitor_metric: str, scheduler/checkpoint 监控指标 key
                - monitor_mode: str, scheduler/checkpoint 监控方向, 由 train.py 同步消费
                - voxel_ligand_pr_auc_thresholds: int | None, dense ligand AP 阈值配置(默认1024)

            - other
                - val_metric_device_policy: str, validation metric 设备策略, 允许 auto/cpu/gpu
                - initial_p_best_by_class: Sequence[float] | None, (K,), candidate best-F1 初始阈值
                - initial_p_sampling_by_class: Sequence[float] | None, (K,), candidate sampling 初始阈值
                - compile: bool, 是否 torch.compile backbone
        """
        super().__init__()
        if class_names is None:
            raise ValueError("VoxelPointStage1Wrapper 必须显式传入 class_names。")
        self.save_hyperparameters(ignore=["backbone", "atom_loss", "voxel_aux_loss", "voxel_ligand_loss", "protein_mainchain_loss", "nucleic_mainchain_loss", "ligand_sparse_refine_loss", "ligand_sparse_refine_delta_loss", "ligand_pseudo_loss"])
        self.model_name = str(name)
        self.monitor_mode = str(monitor_mode)
        self.backbone = backbone if isinstance(backbone, nn.Module) else instantiate(backbone)
        self.atom_loss = atom_loss if (atom_loss is None or isinstance(atom_loss, nn.Module)) else instantiate(atom_loss)
        self.voxel_aux_loss = voxel_aux_loss if (voxel_aux_loss is None or isinstance(voxel_aux_loss, nn.Module)) else instantiate(voxel_aux_loss)
        self.voxel_ligand_loss = voxel_ligand_loss if (voxel_ligand_loss is None or isinstance(voxel_ligand_loss, nn.Module)) else instantiate(voxel_ligand_loss)
        self.protein_mainchain_loss = protein_mainchain_loss if (protein_mainchain_loss is None or isinstance(protein_mainchain_loss, nn.Module)) else instantiate(protein_mainchain_loss)
        self.nucleic_mainchain_loss = nucleic_mainchain_loss if (nucleic_mainchain_loss is None or isinstance(nucleic_mainchain_loss, nn.Module)) else instantiate(nucleic_mainchain_loss)
        self.ligand_sparse_refine_loss = (
            ligand_sparse_refine_loss
            if (ligand_sparse_refine_loss is None or isinstance(ligand_sparse_refine_loss, nn.Module))
            else instantiate(ligand_sparse_refine_loss)
        )
        if self.ligand_sparse_refine_loss is not None and not isinstance(self.ligand_sparse_refine_loss, AdaptiveClassificationCompositeLoss):
            raise TypeError("ligand_sparse_refine_loss 必须为 AdaptiveClassificationCompositeLoss。")
        self.ligand_pseudo_loss = (
            ligand_pseudo_loss
            if (ligand_pseudo_loss is None or isinstance(ligand_pseudo_loss, nn.Module))
            else instantiate(ligand_pseudo_loss)
        )
        if self.ligand_pseudo_loss is not None and not isinstance(self.ligand_pseudo_loss, AdaptiveClassificationCompositeLoss):
            raise TypeError("ligand_pseudo_loss 必须为 AdaptiveClassificationCompositeLoss。")
        self.ligand_sparse_refine_delta_loss = (
            ligand_sparse_refine_delta_loss
            if (ligand_sparse_refine_delta_loss is None or isinstance(ligand_sparse_refine_delta_loss, nn.Module))
            else instantiate(ligand_sparse_refine_delta_loss)
        )
        if self.ligand_sparse_refine_delta_loss is not None and not isinstance(self.ligand_sparse_refine_delta_loss, LigandSparseRefineDeltaLoss):
            raise TypeError("ligand_sparse_refine_delta_loss 必须为 LigandSparseRefineDeltaLoss。")
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
        self._cached_voxel_ligand_best_f1_before_refine_by_class = None if self._sparse_candidate_class_ids is None   else torch.full((len(self._sparse_candidate_class_ids),), float("nan"), dtype=torch.float32)

        # int, candidate fixed-topk warmup step 数
        self._candidate_warmup_steps = 0
        # int, validation epoch end 序号
        self._validation_index = 0
        # dict[str, torch.Tensor], 最近一次 validation end 聚合出的标量; train.py 的通用 scheduler callback 消费它
        self._last_validation_payload: dict[str, torch.Tensor] = {}

        # ValidationMetricManager, 管理 atom/receptor/voxel_ligand 常规 AP/PRAUC
        self.val_metrics = ValidationMetricManager(
            branches=self._build_metric_branch_specs(voxel_ligand_pr_auc_thresholds),
            metric_device_policy=str(val_metric_device_policy),
        )
        # dict[str, Any], validation_diagnostics 配置副本
        diagnostics_cfg = dict(validation_diagnostics or {})
        # CpcValidationDiagnostics, dense -> C -> refined 验证诊断统计器
        self.cpc_diagnostics = self._build_cpc_diagnostics(diagnostics_cfg, voxel_ligand_pr_auc_thresholds)
        self._sync_sparse_candidate_runtime_to_backbone()




    # ------------------------------------------------ 启动函数 -----------------------------------------------
    def _unwrap_backbone(self) -> nn.Module:
        """
        返回未被 torch.compile 包装的 backbone。

        输出:
            - backbone: nn.Module, 原始 Stage1 backbone
        """
        return getattr(self.backbone, "_orig_mod", self.backbone)

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
        # torch.Tensor, (K,), CPU float, 显式初始化的 candidate threshold cache
        cache = torch.as_tensor(list(initial_values), dtype=torch.float32).reshape(-1)
        if int(cache.numel()) != len(self._sparse_candidate_class_ids):
            raise ValueError(f"{value_name} 长度必须等于 candidate_class_ids 数量。")
        if not bool(torch.isfinite(cache).all()):
            raise ValueError(f"{value_name} 不能包含 NaN/Inf。")
        return cache

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
            MetricBranchSpec("pseudo", self.ligand_pseudo_loss is not None, int(getattr(self.ligand_pseudo_loss, "num_classes", 2)), self.class_names, voxel_ligand_pr_auc_thresholds),
            MetricBranchSpec("receptor", self.voxel_aux_loss is not None, int(getattr(self.voxel_aux_loss, "num_classes", 2)), self.class_names, None),
            MetricBranchSpec("voxel_ligand", self.voxel_ligand_loss is not None, int(getattr(self.voxel_ligand_loss, "num_classes", 2)), self.class_names, voxel_ligand_pr_auc_thresholds),
            MetricBranchSpec(
                "protein_mainchain",
                self.protein_mainchain_loss is not None and float(self.hparams.protein_mainchain_loss_weight) > 0.0,
                len(PROTEIN_MAINCHAIN_CLASS_NAMES),
                PROTEIN_MAINCHAIN_CLASS_NAMES,
                voxel_ligand_pr_auc_thresholds,
                False,
                True,
            ),
            MetricBranchSpec(
                "nucleic_mainchain",
                self.nucleic_mainchain_loss is not None and float(self.hparams.nucleic_mainchain_loss_weight) > 0.0,
                len(NUCLEIC_MAINCHAIN_CLASS_NAMES),
                NUCLEIC_MAINCHAIN_CLASS_NAMES,
                voxel_ligand_pr_auc_thresholds,
                False,
                True,
            ),
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
        # nn.Module | None, backbone 内部 sparse candidate builder
        candidate_builder = getattr(self._unwrap_backbone(), "candidate_set_builder", None)
        # tuple[int, ...], (K,), diagnostics 使用的候选前景类 id
        candidate_class_ids = self._sparse_candidate_class_ids or (1,)
        # tuple[float, ...], (K,), adaptive threshold 扩张倍数
        adaptive_expand_factor = tuple(getattr(candidate_builder, "adaptive_expand_factor", tuple(1.0 for _ in candidate_class_ids)))
        # tuple[int, ...], (K,), 每 BOX/类候选 C 上限
        max_candidate_voxels_per_class = tuple(getattr(candidate_builder, "max_candidate_voxels_per_class", tuple(0 for _ in candidate_class_ids)))
        return CpcValidationDiagnostics(
            config=CpcDiagnosticsConfig(
                enabled=bool(diagnostics_cfg.get("enabled", True)),
                num_bins=int(diagnostics_cfg.get("num_bins", voxel_ligand_pr_auc_thresholds or 1024)),
                write_local_artifacts=bool(diagnostics_cfg.get("write_local_artifacts", True)),
                log_wandb_curves=bool(diagnostics_cfg.get("log_wandb_curves", True)),
                wandb_curve_every_n_validation=int(diagnostics_cfg.get("wandb_curve_every_n_validation", 1)),
                output_subdir=str(diagnostics_cfg.get("output_subdir", "validation_diagnostics")),
            ),
            class_names=self.class_names,
            candidate_class_ids=candidate_class_ids,
            adaptive_expand_factor=adaptive_expand_factor,
            max_candidate_voxels_per_class=max_candidate_voxels_per_class,
        )

    def _sync_sparse_candidate_runtime_to_backbone(self) -> None:
        """
        将 wrapper 中 candidate runtime/cache 同步给 backbone:
        - 调用 backbone.set_sparse_candidate_runtime():
            - 传入当前的 global_step; 
            - 传入固定的 self._candidate_warmup_steps(通过钩子 def configure_optimizers 自动解析); 
            - 传入 allow_warmup(它基本总是正的, 但会由backbone里面的 def _should_use_candidate_fixed_topk 进一步判别)
        - 调用 backbone.set_sparse_candidate_thresholds():
            - 用(这里的) self._cached_voxel_ligand_p_best_by_class 更新 backbone 的 self._candidate_p_best_by_class
            - 用(这里的) self._cached_voxel_ligand_p_sampling_by_class 更新 backbone 的 self._candidate_p_sampling_by_class
        """
        if self._sparse_candidate_class_ids is None:
            return
        # nn.Module, 原始 Stage1 backbone
        backbone = self._unwrap_backbone()
        try:
            trainer = self.trainer
        except RuntimeError:
            trainer = None
        # int, 当前 Lightning global step
        global_step = int(getattr(trainer, "global_step", self.global_step)) if trainer is not None else int(self.global_step)
        # bool, 当前生命周期是否允许 candidate fixed-topk warmup
        allow_warmup = False if trainer is None else (
            bool(getattr(trainer, "sanity_checking", False))   # fit 开始前的 sanity check 阶段
            or "fit" in str(getattr(getattr(trainer, "state", None), "fn", "")).lower()  # 正式训练阶段
            or self._is_tuning_trainer(trainer)  # LR/batch-size tuner 探测阶段
        )
        if hasattr(backbone, "set_sparse_candidate_runtime"):
            backbone.set_sparse_candidate_runtime(global_step=global_step, candidate_warmup_steps=int(self._candidate_warmup_steps), allow_warmup_fixed_topk=allow_warmup)
        if hasattr(backbone, "set_sparse_candidate_thresholds"):
            backbone.set_sparse_candidate_thresholds(p_best_by_class=self._cached_voxel_ligand_p_best_by_class, p_sampling_by_class=self._cached_voxel_ligand_p_sampling_by_class)

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









    # ------------------------------------------ 第一类工具函数(浅显;非钩子) ------------------------------------------
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

    def forward_voxel_probability(self, batch: dict[str, Any]) -> torch.Tensor:
        """
        把正式 Stage1 wrapper 的 voxel-only 调用原样转发给 backbone。

        输入参数:
            - batch: dict[str,Any], Stage1 batch; voxel 网格为 BOX-local 离散 ZYX 体素，atom 坐标字段仍按 backbone 契约提供

        输出:
            - voxel_logits_ligand: torch.Tensor, (B,1,D,H,W), BOX-local 离散 ZYX voxel 网格上的 sigmoid 前 ligand logits

        推理层负责 sigmoid、Find hardmask 与整图融合；wrapper 不重复 checkpoint 或阈值逻辑。
        """

        backbone = self._unwrap_backbone()
        if not hasattr(backbone, "forward_voxel_probability"):
            raise AttributeError("当前 Stage1 backbone 未实现 forward_voxel_probability。")
        return backbone.forward_voxel_probability(batch)

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

    def _ligand_target_from_batch(
        self,
        batch: Mapping[str, Any],
        logit_dim: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """
        读取 AdaLigand 直接 union target，并保留旧距离图配置兼容。

        ``ligand_area_target`` 是 schema v3 ``union_mask`` 的 80³ crop，不乘
        ``hardmask``。旧 ``ligand_dist_map`` 只服务未迁移的通用 Pocket_Plus 配置。

        输入参数:
            - batch: Mapping[str,Any], 含 `(B,D,H,W)` 或 `(B,1,D,H,W)` BOX-local 离散 ZYX ligand target
            - logit_dim: int, 当前 ligand logits 通道数
            - device: torch.device, target 输出设备
            - dtype: torch.dtype, 旧距离图转换时使用的计算 dtype

        输出:
            - target: torch.Tensor, (B,D,H,W), BOX-local 离散 ZYX voxel 的 hard-label target
        """

        direct_target = batch.get("ligand_area_target")
        if direct_target is not None:
            target = direct_target.to(device=device)
            if int(logit_dim) == 1:
                if target.ndim == 5 and int(target.shape[1]) == 1:
                    target = target[:, 0]
                if target.ndim != 4:
                    raise ValueError("二分类 ligand_area_target 必须为 (B,D,H,W)。")
                return target.to(dtype=torch.long)
            if target.ndim != 4:
                raise ValueError("多分类 ligand_area_target 必须为 (B,D,H,W) 类别索引。")
            return target.to(dtype=torch.long)
        ligand_dist_map = batch.get("ligand_dist_map")
        if ligand_dist_map is None:
            raise KeyError("batch 缺少 ligand_area_target。")
        return self._ligand_target_from_dist(ligand_dist_map, logit_dim, device, dtype)

    def _sample_ligand_refine_supervision(self, outputs: dict[str, Any], batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        """
        从 dense ligand 距离图采样 C 级 sparse refine 监督。

        输入参数:
            - outputs: dict[str, Any], backbone 输出, 包含 ligand_refine_logits_C/candidate_batch_index/candidate_voxel_zyx
            - batch: dict[str, Any], 当前 batch, 包含 ligand_dist_map

        输出:
            - supervision: dict[str, torch.Tensor], dense 级(全局) target/mask 与 C 级 target/mask
        """
        # torch.Tensor, (sumC,C_logits), C 级 refined logits
        logits_C = outputs["ligand_refine_logits_C"]
        # torch.Tensor, (B,D,H,W), dense ligand hard-label target
        dense_target = self._ligand_target_from_batch(
            batch=batch,
            logit_dim=int(logits_C.shape[1]),
            device=logits_C.device,
            dtype=logits_C.dtype,
        )
        # torch.Tensor, (B,D,H,W), bool, 体素网格即 BOX 本体, dense ligand 全体素有效
        dense_valid_mask = torch.ones_like(dense_target, dtype=torch.bool, device=logits_C.device)
        # torch.Tensor, (sumC,), long, C 中每个候选 voxel 所属 BOX index
        idx_b = outputs["candidate_batch_index"].to(device=logits_C.device, dtype=torch.long)
        # torch.Tensor[int64], (sumC,3), C 候选的 BOX-local 离散 voxel-index ZYX
        idx_zyx = outputs["candidate_voxel_zyx"].to(device=logits_C.device, dtype=torch.long)
        # torch.Tensor, (sumC,), C 级 hard-label target
        target_C = dense_target[idx_b, idx_zyx[:, 0], idx_zyx[:, 1], idx_zyx[:, 2]]
        # torch.Tensor, (sumC,), bool, C 级有效监督掩码
        valid_C = dense_valid_mask[idx_b, idx_zyx[:, 0], idx_zyx[:, 1], idx_zyx[:, 2]]
        # dict[str, torch.Tensor], dense 与 C 级 sparse refine 监督字段
        supervision = {
            "ligand_dense_target": dense_target,
            "ligand_dense_valid_mask": dense_valid_mask,
            "ligand_refine_target_C": target_C,
            "ligand_refine_valid_mask_C": valid_C,
        }
        outputs.update(supervision)
        return supervision

    def _sample_ligand_pseudo_supervision(self, outputs: dict[str, Any], batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        """
        从 dense ligand 距离图采样 P(虚拟原子)级 ligand 区域归属监督。

        输入参数:
            - outputs: dict[str, Any], backbone 输出, 包含 pseudo_logits/pseudo_voxel_zyx/pseudo_batch_index
            - batch: dict[str, Any], 当前 batch, 包含 ligand_dist_map

        输出:
            - supervision: dict[str, torch.Tensor], 包含:
                - "pseudo_ligand_target": torch.Tensor, (N_pseudo,), P 级 ligand 区域硬标签
                - "pseudo_ligand_valid_mask": torch.Tensor, (N_pseudo,), bool, P 级有效监督掩码
        """
        # torch.Tensor, (N_pseudo, pseudo_ligand_logit_dim), P 后置 ligand logits
        pseudo_logits = outputs["pseudo_logits"]
        # torch.Tensor, (B,D,H,W), dense ligand hard-label target
        dense_target = self._ligand_target_from_batch(
            batch=batch,
            logit_dim=int(pseudo_logits.shape[1]),
            device=pseudo_logits.device,
            dtype=pseudo_logits.dtype,
        )
        # torch.Tensor, (B,D,H,W), bool, 体素网格即 BOX 本体, P home 体素监督全有效
        dense_valid_mask = torch.ones_like(dense_target, dtype=torch.bool, device=pseudo_logits.device)
        # torch.Tensor, (N_pseudo,), long, P home 体素所属 BOX index
        idx_b = outputs["pseudo_batch_index"].to(device=pseudo_logits.device, dtype=torch.long)
        # torch.Tensor[int64], (N_pseudo,3), P home 的 BOX-local 离散 voxel-index ZYX
        idx_zyx = outputs["pseudo_voxel_zyx"].to(device=pseudo_logits.device, dtype=torch.long)
        # torch.Tensor, (N_pseudo,), P 级 ligand 区域硬标签
        target_P = dense_target[idx_b, idx_zyx[:, 0], idx_zyx[:, 1], idx_zyx[:, 2]]
        # torch.Tensor, (N_pseudo,), bool, P 级有效监督掩码
        valid_P = dense_valid_mask[idx_b, idx_zyx[:, 0], idx_zyx[:, 1], idx_zyx[:, 2]]
        # dict[str, torch.Tensor], P 级 pseudo 监督字段(前后置头共用同一 target/valid)
        supervision = {
            "pseudo_ligand_target": target_P,
            "pseudo_ligand_valid_mask": valid_P,
        }
        outputs.update(supervision)
        return supervision

    def _resolve_sparse_refine_loss_warmup_steps(self, sched_cfg: Mapping[str, Any]) -> int:
        """
        解析 sparse refine loss 独立 warmup 步数。

        输入参数:
            - sched_cfg: Mapping[str, Any], ligand_sparse_refine_loss_schedule 配置, 包含 warmup_steps 或 warmup_ratio

        输出:
            - warmup_steps: int, 从 start_weight 线性过渡到 final_weight 的 optimizer step 数
        """
        warmup_steps = sched_cfg["warmup_steps"]
        if warmup_steps is not None:
            return int(warmup_steps)
        try:
            trainer = self.trainer
        except RuntimeError:
            trainer = None
        total_steps = getattr(trainer, "estimated_stepping_batches", None) if trainer is not None else None
        if total_steps is None or int(total_steps) <= 0:
            raise RuntimeError("ligand_sparse_refine_loss_schedule 需要 trainer.estimated_stepping_batches 为正数。")
        return int(round(int(total_steps) * float(sched_cfg["warmup_ratio"])))

    def _resolve_sparse_refine_loss_start_on_steps(self, sched_cfg: Mapping[str, Any]) -> int:
        """
        解析 sparse refine loss 硬 0 延迟步数(global_step < start_on 时权重硬等于 0)。

        输入参数:
            - sched_cfg: Mapping[str, Any], ligand_sparse_refine_loss_schedule 配置, 包含 start_on_steps 或 start_on_ratio

        输出:
            - start_on_steps: int, refine 权重保持硬 0 的 optimizer step 数; start_on_ratio 为 null 时返回 0(复现旧斜坡)
        """
        # 旧版 schedule dict 不含 start_on_* 键时按 None 处理, 等价于无延迟(复现旧斜坡)
        start_on_steps = sched_cfg.get("start_on_steps")
        if start_on_steps is not None:
            return int(start_on_steps)
        if sched_cfg.get("start_on_ratio") is None:
            return 0
        try:
            trainer = self.trainer
        except RuntimeError:
            trainer = None
        total_steps = getattr(trainer, "estimated_stepping_batches", None) if trainer is not None else None
        if total_steps is None or int(total_steps) <= 0:
            raise RuntimeError("ligand_sparse_refine_loss_schedule start_on_ratio 需要 trainer.estimated_stepping_batches 为正数。")
        return int(round(int(total_steps) * float(sched_cfg["start_on_ratio"])))

    def _compute_sparse_refine_loss_effective_weight(self) -> torch.Tensor:
        """
        计算当前 global_step 下 sparse refine loss 的有效权重。

        输出:
            - weight: torch.Tensor, (), 当前 step 使用的 loss 权重
        """
        # Mapping[str, Any] | None, sparse refine loss warmup 配置
        sched_cfg = self.hparams.ligand_sparse_refine_loss_schedule
        if sched_cfg is None:
            return torch.tensor(float(self.hparams.ligand_sparse_refine_loss_weight), device=self.device, dtype=torch.float32)
        # float, sparse refine loss warmup 起始权重
        start_weight = float(sched_cfg["start_weight"])
        # float, sparse refine loss warmup 最终权重
        final_weight = float(sched_cfg["final_weight"])
        # int, sparse refine loss warmup 步数
        warmup_steps = self._resolve_sparse_refine_loss_warmup_steps(sched_cfg)
        # int, sparse refine loss 硬 0 延迟步数; 必须 <= warmup
        start_on_steps = self._resolve_sparse_refine_loss_start_on_steps(sched_cfg)
        if start_on_steps > warmup_steps:
            raise ValueError("ligand_sparse_refine_loss_schedule: start_on 必须 <= warmup。")
        try:
            trainer = self.trainer
        except RuntimeError:
            trainer = None
        # int, 当前训练 global step
        global_step = int(getattr(trainer, "global_step", self.global_step)) if trainer is not None else int(self.global_step)
        if global_step < start_on_steps:
            return torch.tensor(0.0, device=self.device, dtype=torch.float32)
        if global_step >= warmup_steps:
            # 同时覆盖 warmup_steps==0 与 start_on==warmup 的阶跃
            return torch.tensor(final_weight, device=self.device, dtype=torch.float32)
        # float, [start_on, warmup] 区间内的线性进度, 取值范围 [0,1)
        progress = (float(global_step) - start_on_steps) / float(warmup_steps - start_on_steps)
        return torch.tensor(start_weight + progress * (final_weight - start_weight), device=self.device, dtype=torch.float32)

    def _compute_total_loss(self, outputs: dict[str, Any], batch: dict[str, Any]) -> tuple[torch.Tensor, list[LossTerm], dict[str, torch.Tensor]]:
        """
        计算当前批次中已启用的 Stage1 损失项并汇总加权总损失。

        输入参数:
            - outputs: dict[str, Any], backbone 输出
            - batch: dict[str, Any], 当前 batch 字典

        输出:
            - total_loss: 标量张量，所有启用分支的 ``weight * value`` 之和。
            - loss_terms: list[LossTerm]，每个已启用监督分支的未加权损失与静态权重。
            - extra_logs: dict[str, torch.Tensor]，稀疏细化的动态权重和分量日志。
        """
        # 标量张量，当前批次的加权总损失。
        total_loss = torch.tensor(0.0, device=self.device, dtype=torch.float32)
        # list[LossTerm]，当前批次实际产生的监督分支损失。
        loss_terms: list[LossTerm] = []
        # dict[str, torch.Tensor]，不直接作为独立损失分支的标量日志。
        extra_logs: dict[str, torch.Tensor] = {}
        if self.atom_loss is not None:
            if outputs.get("atom_logits") is not None:
                loss_terms.append(compute_atom_loss_term(
                    outputs=outputs, batch=batch, loss_module=self.atom_loss,
                    weight=float(self.hparams.atom_loss_weight)))
        if self.voxel_aux_loss is not None:
            term = compute_receptor_loss_term(outputs=outputs, batch=batch, loss_module=self.voxel_aux_loss, weight=float(self.hparams.voxel_aux_loss_weight))
            if term is not None:
                loss_terms.append(term)
        if self.voxel_ligand_loss is not None:
            term = compute_voxel_ligand_loss_term(outputs=outputs, batch=batch, loss_module=self.voxel_ligand_loss, weight=float(self.hparams.voxel_ligand_loss_weight))
            if term is not None:
                loss_terms.append(term)
        if self.protein_mainchain_loss is not None and float(self.hparams.protein_mainchain_loss_weight) > 0.0:
            term = compute_mainchain_class_loss_term(
                outputs=outputs,
                batch=batch,
                loss_module=self.protein_mainchain_loss,
                weight=float(self.hparams.protein_mainchain_loss_weight),
                polymer_name="protein",
            )
            if term is not None:
                loss_terms.append(term)
        if self.nucleic_mainchain_loss is not None and float(self.hparams.nucleic_mainchain_loss_weight) > 0.0:
            term = compute_mainchain_class_loss_term(
                outputs=outputs,
                batch=batch,
                loss_module=self.nucleic_mainchain_loss,
                weight=float(self.hparams.nucleic_mainchain_loss_weight),
                polymer_name="nucleic",
            )
            if term is not None:
                loss_terms.append(term)
        if float(self.hparams.ligand_distance_loss_weight) > 0.0:
            term = compute_ligand_distance_loss_term(
                outputs=outputs,
                batch=batch,
                weight=float(self.hparams.ligand_distance_loss_weight),
            )
            if term is not None:
                loss_terms.append(term)
        if self.ligand_pseudo_loss is not None and outputs.get("pseudo_logits") is not None:
            # P 锚点配体区域监督；前后预测头共用同一标签和有效掩码，不使用预热调度。
            pseudo_supervision = self._sample_ligand_pseudo_supervision(outputs=outputs, batch=batch)
            if float(self.hparams.pseudo_loss_weight) > 0.0:
                loss_terms.append(compute_pseudo_loss_term(
                    outputs=outputs, loss_module=self.ligand_pseudo_loss,
                    weight=float(self.hparams.pseudo_loss_weight),
                    target=pseudo_supervision["pseudo_ligand_target"],
                    valid_mask=pseudo_supervision["pseudo_ligand_valid_mask"]))
        if self.ligand_sparse_refine_loss is not None and outputs.get("ligand_refine_logits_C") is not None:
            # 完整体素网格和稀疏候选 C 的配体区域监督字段。
            supervision = self._sample_ligand_refine_supervision(outputs=outputs, batch=batch)
            # 标量张量，当前优化步骤的稀疏细化有效权重。
            effective_weight = self._compute_sparse_refine_loss_effective_weight()
            # 是否启用候选排序损失：模块存在、排序权重大于 0 且基础概率可用。
            delta_on = (
                self.ligand_sparse_refine_delta_loss is not None
                and float(self.hparams.ligand_sparse_refine_w_rank) > 0.0
                and outputs.get("candidate_prob") is not None
            )
            term, logged_weight, component_logs = compute_sparse_refine_loss_term(
                logits_C=outputs["ligand_refine_logits_C"],
                target_C=supervision["ligand_refine_target_C"],
                valid_C=supervision["ligand_refine_valid_mask_C"],
                loss_module=self.ligand_sparse_refine_loss,
                weight=float(self.hparams.ligand_sparse_refine_loss_weight),
                effective_weight=effective_weight,
                delta_loss_module=self.ligand_sparse_refine_delta_loss if delta_on else None,
                base_prob_C=outputs.get("candidate_prob") if delta_on else None,
                batch_index_C=outputs.get("candidate_batch_index") if delta_on else None,
                w_rank=float(self.hparams.ligand_sparse_refine_w_rank),
            )
            loss_terms.append(term)
            extra_logs["ligand_sparse_refine_weight_effective"] = logged_weight
            extra_logs.update(component_logs)
            # torch.Tensor, (), 当前 refine 距离 softmax 的正温度; 监控可学温度往尖锐/平缓哪个方向走
            extra_logs["sparse_refine_temperature"] = self._unwrap_backbone().sparse_refine_head.log_temperature.exp().detach()
        for term in loss_terms:
            if term.name == "ligand_sparse_refine":
                # torch.Tensor, (), schedule 后当前 step 的有效权重
                weight = extra_logs["ligand_sparse_refine_weight_effective"]
                # nan_to_num 仅作用于 refine term: schedule 硬 0 阶段的 0 * NaN 会污染总 loss 并把 NaN 灌进全模型梯度;
                # 其余监督分支的 NaN 视为真实 bug, 不在此处静默吞掉。term.logged_value 仍保留原始值用于告警。
                total_loss = total_loss + weight * torch.nan_to_num(term.value)
            else:
                # torch.Tensor, (), 当前 loss term 实际参与总损失的权重
                weight = term.value.new_tensor(term.weight)
                if float(term.weight) != 0.0:
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
            - None, 用 self.log() 记录 total_loss 和所有的 loss_terms, 以及 extra_logs
        """
        self.log(f"{prefix}/global/total", total_loss, prog_bar=self.hparams.monitor_metric == f"{prefix}/global/total", on_step=prefix == "train_loss", on_epoch=True, sync_dist=True)
        for term in loss_terms:
            self.log(f"{prefix}/global/{term.name}", term.logged_value, prog_bar=False, on_step=prefix == "train_loss", on_epoch=True, sync_dist=True)
        for name, value in extra_logs.items():
            self.log(f"{prefix}/global/{name}", value, prog_bar=False, on_step=prefix == "train_loss", on_epoch=True, sync_dist=True)

    ############################## 第一类钩子 ##############################
    """   
    这个文件里被框架自动调用的 hook, voxel_point_stage1.py 重写并被 fit() 自动驱动的有这几个：
            方法	                   触发时机（由 Trainer 自动调用）
     configure_optimizers	     fit 启动时调用一次，构造 optimizer/scheduler
     on_load_checkpoint	         断点续训恢复时（在训练循环开始前）
     training_step	             每个训练 batch 调用一次
     on_validation_epoch_start	 每次进入 validation loop 前
     validation_step	         每个 validation batch 调用一次
     on_validation_epoch_end	 每次 validation loop 结束
     on_save_checkpoint	         每次写 checkpoint 时
    """
    def train(self, mode: bool = True) -> VoxelPointStage1Wrapper:
        """
        覆写 train, 在进入 train 模式后把完全冻结的子树重新切回 eval。

        Lightning 每个 train epoch 会调用 model.train() 把全部子模块设回 training=True, 撤销之前的冻结
        eval 状态; 这里在 super().train() 之后重新固定冻结子树, 保证其 BN/Dropout 前向不随训练漂移。
        无冻结参数时 set_fully_frozen_submodules_eval 为 no-op。

        输入参数:
            - mode: bool, True 进入 train, False 进入 eval

        输出:
            - self: VoxelPointStage1Wrapper, 与 nn.Module.train 一致返回自身
        """
        super().train(mode)
        if mode:
            set_fully_frozen_submodules_eval(self)
        return self

    def configure_optimizers(self) -> Any:
        """
        配置 optimizer 与 scheduler。

        输出:
            - config: Any, Lightning configure_optimizers 返回值
        """
        # Any, Lightning optimizer/scheduler 配置返回值
        config, warmup_steps = configure_stage1_optimizers(
            module=self,
            optimizer_config=self.hparams.optimizer,
            scheduler_config=self.hparams.scheduler,
            interval=str(self.hparams.interval),
            frequency=int(self.hparams.frequency),
            monitor_metric=str(self.hparams.monitor_metric),
        )
        # int, candidate fixed-topk warmup step 数
        self._candidate_warmup_steps = int(warmup_steps)
        self._sync_sparse_candidate_runtime_to_backbone()
        return config

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
        # dict[str, Any], 当前训练 batch 字典
        batch_dict = self._extract_batch(batch)
        self._sync_sparse_candidate_runtime_to_backbone()
        # dict[str, Any], backbone 输出字典
        outputs = self(batch_dict)
        # 当前批次的总损失、分支损失和额外标量日志。
        total_loss, loss_terms, extra_logs = self._compute_total_loss(outputs=outputs, batch=batch_dict)
        self._log_loss_terms("train_loss", total_loss, loss_terms, extra_logs)
        if "recycle_passes_used" in outputs:
            self.log("train/runtime/recycle_passes", float(outputs["recycle_passes_used"]), prog_bar=False, on_step=True, on_epoch=False, sync_dist=True)
        return total_loss










    # ------------------------------------------ 第二类工具函数(浅显;非钩子) ------------------------------------------
    def _allow_validation_cache_update(self) -> bool:
        """
        判断当前 validation 是否允许写回 candidate threshold cache: 不在一开始的 sanity_checking 和 tuning 截断就可写

        输出:
            - allow_update: bool, True 表示普通 fit validation 可写 cache
        """
        try:
            trainer = self.trainer
        except RuntimeError:
            return False
        return trainer is not None and not bool(getattr(trainer, "sanity_checking", False)) and not self._is_tuning_trainer(trainer)

    def _candidate_selection_mode(self) -> str:
        """
        返回当前 candidate builder selection mode(adaptive_threshold 等)

        输出:
            - selection_mode: str, candidate selection mode
        """
        # nn.Module | None, backbone 内部 sparse candidate builder
        candidate_builder = getattr(self._unwrap_backbone(), "candidate_set_builder", None)
        return str(getattr(candidate_builder, "selection_mode", "adaptive_threshold"))

    def _using_candidate_warmup(self) -> bool:
        """
        判断当前 validation 是否处于 candidate warmup fixed-topk 阶段。

        输出:
            - using_warmup: bool, `global_step < _candidate_warmup_steps` 时为 True
        """
        return int(self.global_step) < int(self._candidate_warmup_steps)

    def _dense_num_gt_by_box(self, target: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        """
        统计每个 BOX 中 candidate class 的 dense GT 正例数。

        输入参数:
            - target: torch.Tensor, (B,D,H,W), hard-label target
            - valid_mask: torch.Tensor, (B,D,H,W), 有效体素掩码

        输出:
            - dense_num_gt: torch.Tensor, (B,K), 每个 BOX/候选类的 GT 正例数
        """
        # tuple[int, ...], (K,), sparse refine 候选前景类 id
        class_ids = self._sparse_candidate_class_ids or (1,)
        # torch.Tensor, (B,D,H,W), bool, dense GT 有效掩码
        valid = valid_mask.bool()
        counts = []
        for class_id in class_ids:
            positive = (target.long() == int(class_id)) & valid
            # append 内容: torch.Tensor, (B,), 当前候选类在每个 BOX 内的 dense GT 正例数
            counts.append(positive.reshape(target.shape[0], -1).sum(dim=1))
        return torch.stack(counts, dim=1)

    ############################## 第二类钩子 ##############################
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
        # dict[str, Any], 当前 validation batch 字典
        batch_dict = self._extract_batch(batch)
        self._sync_sparse_candidate_runtime_to_backbone()
        # dict[str, Any], backbone 输出字典
        outputs = self(batch_dict)
        # torch.Tensor/list[LossTerm]/dict[str, torch.Tensor], 当前 batch 总损失、分支损失和额外日志
        total_loss, loss_terms, extra_logs = self._compute_total_loss(outputs=outputs, batch=batch_dict)
        if self.atom_loss is not None and outputs.get("atom_logits") is not None:
            # bool, (N_atom,)，只有核心 BOX 内的受体原子参加原子分类 PRAUC。
            atom_mask = outputs.get("atom_is_in_core_box", batch_dict["atom_is_in_core_box"])
            self.val_metrics.update_branch(branch_name="atom", logits=outputs["atom_logits"], target=outputs.get("atom_target", batch_dict["atom_label"]), mask=atom_mask)
        if self.ligand_pseudo_loss is not None and outputs.get("pseudo_logits") is not None and "pseudo_ligand_target" in outputs:
            self.val_metrics.update_branch(
                branch_name="pseudo",
                logits=outputs["pseudo_logits"],
                target=outputs["pseudo_ligand_target"],
                mask=outputs["pseudo_ligand_valid_mask"],
            )
        if self.voxel_aux_loss is not None and "voxel_logits_aux" in outputs:
            # bool, (B, D, H, W)，只有受体原子占据体素参加受体结合区域 PRAUC。
            receptor_mask = batch_dict["hardmask"].bool().squeeze(1)
            self.val_metrics.update_branch(branch_name="receptor", logits=outputs["voxel_logits_aux"], target=batch_dict["voxel_label"], mask=receptor_mask)
        if (
            self.voxel_ligand_loss is not None
            and "voxel_logits_ligand" in outputs
            and ("ligand_area_target" in batch_dict or "ligand_dist_map" in batch_dict)
        ):
            # int64, (B, D, H, W)，每个体素的配体区域类别编号。
            ligand_target = self._ligand_target_from_batch(
                batch_dict,
                int(outputs["voxel_logits_ligand"].shape[1]),
                outputs["voxel_logits_ligand"].device,
                outputs["voxel_logits_ligand"].dtype,
            )
            # bool, (B, D, H, W)，配体区域 PRAUC 覆盖完整 80³ BOX。
            ligand_valid = torch.ones_like(ligand_target, dtype=torch.bool, device=ligand_target.device)
            self.val_metrics.update_branch(branch_name="voxel_ligand", logits=outputs["voxel_logits_ligand"], target=ligand_target, mask=ligand_valid)
            # bool, 当前 validation 是否启用 CPC diagnostics
            diagnostics_enabled = bool(self.cpc_diagnostics.config.enabled)
            if diagnostics_enabled:
                # bool, 当前 validation 是否允许将 diagnostics 阈值写回 candidate cache
                allow_cache_update = self._allow_validation_cache_update()
                self.cpc_diagnostics.update_uncapped_best(logits=outputs["voxel_logits_ligand"], target=ligand_target, valid_mask=ligand_valid, allow_cache_update=allow_cache_update)
            # Mapping[str, torch.Tensor], candidate builder 输出张量集合
            candidate_outputs: Mapping[str, torch.Tensor] = outputs
            # bool, 当前 backbone 输出是否包含 C 候选集核心字段
            has_candidate_outputs = all(name in candidate_outputs for name in ("candidate_batch_index", "candidate_voxel_zyx", "candidate_counts"))
            if diagnostics_enabled and has_candidate_outputs:
                self.cpc_diagnostics.update_uncapped_sampling(logits=outputs["voxel_logits_ligand"], target=ligand_target, valid_mask=ligand_valid, candidate_outputs=candidate_outputs, selection_mode=self._candidate_selection_mode(), use_fixed_warmup=self._using_candidate_warmup())
                self.cpc_diagnostics.update_capped(target=ligand_target, valid_mask=ligand_valid, candidate_outputs=candidate_outputs)
            if diagnostics_enabled and has_candidate_outputs and "ligand_refine_target_C" in outputs:
                # torch.Tensor, (B,K), 每个 BOX/候选类的 dense GT 正例数
                dense_num_gt = self._dense_num_gt_by_box(ligand_target, ligand_valid)
                self.cpc_diagnostics.update_unrefined(candidate_outputs=candidate_outputs, target_C=outputs["ligand_refine_target_C"], valid_C=outputs["ligand_refine_valid_mask_C"], dense_num_gt=dense_num_gt) # outputs["ligand_refine_valid_mask_C"] 就是由batch["voxel_valid_mask"] 导出的(见 def _sample_ligand_refine_supervision )
                if "ligand_refine_logits_C" in outputs:
                    self.cpc_diagnostics.update_refined(refined_logits_C=outputs["ligand_refine_logits_C"], candidate_outputs=candidate_outputs, target_C=outputs["ligand_refine_target_C"], valid_C=outputs["ligand_refine_valid_mask_C"], dense_num_gt=dense_num_gt)
        if self.protein_mainchain_loss is not None and float(self.hparams.protein_mainchain_loss_weight) > 0.0:
            target = batch_dict["protein_mainchain_target"]
            self.val_metrics.update_branch(
                branch_name="protein_mainchain",
                logits=outputs["voxel_logits_protein"],
                target=target,
                mask=torch.ones_like(target, dtype=torch.bool),
            )
        if self.nucleic_mainchain_loss is not None and float(self.hparams.nucleic_mainchain_loss_weight) > 0.0:
            target = batch_dict["nucleic_mainchain_target"]
            self.val_metrics.update_branch(
                branch_name="nucleic_mainchain",
                logits=outputs["voxel_logits_nucleic"],
                target=target,
                mask=torch.ones_like(target, dtype=torch.bool),
            )
        self._log_loss_terms("val_loss", total_loss, loss_terms, extra_logs)
        return total_loss











    ############################## 第三类钩子 ##############################
    def on_validation_epoch_start(self) -> None:
        """
        validation epoch 开始时重置 helper manager 状态: self.cpc_diagnostics.reset() 以及 self.val_metrics.reset().
        它在每一次 validation loop 开始前都会被调用，不是只在「epoch 末那一次验证」才用, 下同
        """
        self.val_metrics.reset()
        if self.cpc_diagnostics.config.enabled:
            self.cpc_diagnostics.reset()

    def on_validation_epoch_end(self) -> None:
        """
        validation epoch 结束时计算 payload、写日志与 artifact。

        输出:
            - None, 原地记录 validation scalar/curve/artifact
        """
        # dict[str, torch.Tensor], 常规 AP/PRAUC validation scalar payload
        metric_payload = self.val_metrics.compute_payload()
        # dict[str, torch.Tensor], 合并后的 validation scalar payload
        payload = dict(metric_payload)
        cpc_payload = None
        if self.cpc_diagnostics.config.enabled:
            # CpcDiagnosticsPayload, CPC diagnostics validation payload
            cpc_payload = self.cpc_diagnostics.compute_payload(sync_fn=self._all_reduce_sum)
            payload.update(cpc_payload.scalars)
            self._update_candidate_threshold_cache_from_payload(payload)
        self._last_validation_payload = {key: value.detach() for key, value in payload.items()}
        self._sync_sparse_candidate_runtime_to_backbone()
        # payload 已由 TorchMetrics 的通信组和 CPC 固定形状统计完成跨卡聚合。
        # 这里不再让 Lightning 用默认 NCCL 通信组二次同步 CPU PRAUC 标量。
        log_scalar_payload(module=self, payload=payload, monitor_metric=str(self.hparams.monitor_metric), sync_dist=False)
        # pl.Trainer, 当前 Lightning trainer
        trainer = self.trainer
        if bool(getattr(trainer, "is_global_zero", True)):
            if cpc_payload is not None and self.cpc_diagnostics.config.write_local_artifacts:
                write_validation_artifacts(run_dir=self._run_dir(), output_subdir=self.cpc_diagnostics.config.output_subdir, epoch=int(self.current_epoch), global_step=int(self.global_step), payload=cpc_payload)
            if cpc_payload is not None and self.cpc_diagnostics.config.log_wandb_curves:
                log_wandb_curves(module=self, curves=cpc_payload.curves, validation_index=int(self._validation_index), every_n=int(self.cpc_diagnostics.config.wandb_curve_every_n_validation))
        self._validation_index += 1
        self.val_metrics.reset()
        if self.cpc_diagnostics.config.enabled:
            self.cpc_diagnostics.reset()

    # ------------------------------------------ 第三类工具函数(浅显;非钩子) ------------------------------------------
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
        # torch.Tensor, 任意固定形状, 当前 rank 的本地统计副本
        reduced = tensor.clone()
        torch.distributed.all_reduce(reduced, op=torch.distributed.ReduceOp.SUM)
        return reduced

    def _update_candidate_threshold_cache_from_payload(self, payload: Mapping[str, torch.Tensor]) -> None:
        """
        从 diagnostics payload 写回 candidate threshold cache。

        输入参数:
            - payload: Mapping[str, torch.Tensor], validation epoch scalar payload

        原地更新 runtime cache(它们都是从 val_uncapped_best 里面直接拿的):
            - self._cached_voxel_ligand_p_best_by_class
            - self._cached_voxel_ligand_p_sampling_by_class
            - self._cached_voxel_ligand_best_f1_before_refine_by_class
        """
        if self._sparse_candidate_class_ids is None or not self._allow_validation_cache_update():
            return
        # list[torch.Tensor], 每个 candidate class 的 p_best scalar
        p_best_values: list[torch.Tensor] = []
        # list[torch.Tensor], 每个 candidate class 的 p_sampling scalar
        p_sampling_values: list[torch.Tensor] = []
        # list[torch.Tensor], 每个 candidate class 的 dense best-F1 scalar
        best_f1_values: list[torch.Tensor] = []
        for class_id in self._sparse_candidate_class_ids:
            # str, 当前 candidate class 对应的 task class 名
            class_name = self.class_names[int(class_id)]
            # str, 多分类 metric leaf 的 task class suffix
            suffix = "" if len(self.class_names) <= 2 else f"_{class_name}"
            # str, 当前 candidate class 的 p_best scalar 写回键
            p_best_key = f"val_uncapped_best/global/p_best{suffix}"
            # str, 当前 candidate class 的 p_sampling scalar 写回键
            p_sampling_key = f"val_uncapped_best/global/p_sampling{suffix}"
            # str, 当前 candidate class 的 best_F1 scalar 写回键
            best_f1_key = f"val_uncapped_best/global/best_F1{suffix}"
            if p_best_key not in payload or p_sampling_key not in payload or best_f1_key not in payload:
                # 已过滤 sanity/tuning(见 _allow_validation_cache_update), 到此处必为正常 fit validation: 缺键意味着该候选类整轮验证没有 GT 正例, 直接 fail-fast
                raise RuntimeError(
                    f"candidate class {class_name!r} 在本次 fit validation 缺少 best 阈值 "
                    f"({p_best_key}/{p_sampling_key}), 通常意味着该类整轮验证没有 GT 正例; "
                    f"与'验证集每类都含正类'的前提冲突。"
                )
            p_best_values.append(payload[p_best_key].detach().cpu().float())
            p_sampling_values.append(payload[p_sampling_key].detach().cpu().float())
            best_f1_values.append(payload[best_f1_key].detach().cpu().float())
        # torch.Tensor, (K,), CPU float, best-F1 threshold cache 候选值
        p_best = torch.stack(p_best_values).reshape(-1)
        # torch.Tensor, (K,), CPU float, sampling threshold cache 候选值
        p_sampling = torch.stack(p_sampling_values).reshape(-1)
        # torch.Tensor, (K,), CPU float, dense best-F1 score cache 候选值
        best_f1 = torch.stack(best_f1_values).reshape(-1)
        if bool(torch.isfinite(p_best).all()):
            self._cached_voxel_ligand_p_best_by_class = p_best
        if bool(torch.isfinite(p_sampling).all()):
            self._cached_voxel_ligand_p_sampling_by_class = p_sampling
        if bool(torch.isfinite(best_f1).all()):
            self._cached_voxel_ligand_best_f1_before_refine_by_class = best_f1

    def _run_dir(self) -> Path:
        """
        解析当前 logger/run 输出目录。

        输出:
            - run_dir: Path, validation artifact 根目录
        """
        # pl.Trainer, 当前 Lightning trainer
        trainer = self.trainer
        # Any, 当前 Lightning logger
        logger = getattr(self, "logger", None)
        # str | Path, logger save_dir 或 trainer default_root_dir
        log_dir = getattr(logger, "save_dir", None) or getattr(trainer, "default_root_dir", ".")
        return Path(log_dir)

    # ------------------------------------------ 第四类工具函数(浅显;非钩子) ------------------------------------------
    def _normalize_candidate_checkpoint_tensor(self, value: Any, value_name: str) -> torch.Tensor:
        """
        将 checkpoint 中的 value 规范化为 CPU float 向量。

        输入参数:
            - value: Any, checkpoint 中读取的张量或可转张量对象
            - value_name: str, 错误信息中的字段名, 仅用于提示

        输出:
            - tensor: torch.Tensor, (K,), CPU float candidate cache
        """
        # torch.Tensor, (K,), CPU float, checkpoint 中的 candidate cache
        tensor = torch.as_tensor(value).detach().cpu().float().reshape(-1)
        if int(tensor.numel()) != len(self._sparse_candidate_class_ids):
            raise ValueError(f"checkpoint 中的 {value_name} 长度必须等于 candidate_class_ids 数量。")
        return tensor

    def _load_candidate_checkpoint_finite_tensor(self, value: Any, value_name: str) -> torch.Tensor:
        """
        从 checkpoint 读取 value 并校验它是否有限(NaN/Inf则报错)。

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

    ############################## 第四类钩子 ##############################
    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        """
        保存训练继续所需 runtime state。

        输入参数:
            - checkpoint: dict[str, Any], Lightning checkpoint 字典

        原地写入:
            - voxel_ligand_candidate_class_ids
            - voxel_ligand_p_best_by_class; voxel_ligand_p_sampling_by_class
            - voxel_ligand_best_f1_before_refine_by_class
        """
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
            - None, 原地恢复 candidate cache
        """
        if self._sparse_candidate_class_ids is not None:
            # Any | None, checkpoint 中保存的 candidate class id 列表
            checkpoint_class_ids = checkpoint.get("voxel_ligand_candidate_class_ids", None)
            if checkpoint_class_ids is not None and tuple(int(x) for x in checkpoint_class_ids) != self._sparse_candidate_class_ids:
                raise ValueError("checkpoint 中的 voxel_ligand_candidate_class_ids 与当前 candidate_class_ids 不一致。")
            # bool, checkpoint 是否包含 p_best cache
            has_p_best = "voxel_ligand_p_best_by_class" in checkpoint
            # bool, checkpoint 是否包含 p_sampling cache
            has_p_sampling = "voxel_ligand_p_sampling_by_class" in checkpoint
            if has_p_best != has_p_sampling:
                raise ValueError("checkpoint 中的 voxel_ligand_p_best_by_class 与 voxel_ligand_p_sampling_by_class 必须成对出现。")
            if has_p_best:
                self._cached_voxel_ligand_p_best_by_class = self._load_candidate_checkpoint_finite_tensor(checkpoint["voxel_ligand_p_best_by_class"], "voxel_ligand_p_best_by_class")
                self._cached_voxel_ligand_p_sampling_by_class = self._load_candidate_checkpoint_finite_tensor(checkpoint["voxel_ligand_p_sampling_by_class"], "voxel_ligand_p_sampling_by_class")
            if "voxel_ligand_best_f1_before_refine_by_class" in checkpoint:
                self._cached_voxel_ligand_best_f1_before_refine_by_class = self._load_candidate_checkpoint_finite_tensor(checkpoint["voxel_ligand_best_f1_before_refine_by_class"], "voxel_ligand_best_f1_before_refine_by_class")
            self._sync_sparse_candidate_runtime_to_backbone()


