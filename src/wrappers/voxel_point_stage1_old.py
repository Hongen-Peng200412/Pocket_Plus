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
        ligand_sparse_refine_loss: nn.Module | None = None,
        optimizer: Optional[Dict[str, Any]] = None,
        scheduler: Optional[Dict[str, Any]] = None,
        atom_loss_weight: float = 1.0,
        voxel_aux_loss_weight: float = 0.0,
        voxel_ligand_loss_weight: float = 0.0,
        ligand_sparse_refine_loss_weight: float = 0.0,
        ligand_sparse_refine_loss_schedule: dict[str, Any] | None = None,
        monitor_metric: str = "val/atom_pr_auc",
        voxel_ligand_pr_auc_thresholds: Optional[int] = 1024,
        val_metric_device_policy: str = "auto",
        initial_p_best_by_class: list[float] | tuple[float, ...] | None = None,
        initial_p_sampling_by_class: list[float] | tuple[float, ...] | None = None,
        interval: str = "epoch",
        frequency: int = 1,
        compile: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        # 将除 nn.Module 以外的超参数保存到 self.hparams，方便日志和检查点恢复
        self.save_hyperparameters(ignore=["backbone", "atom_loss", "voxel_aux_loss", "voxel_ligand_loss", "ligand_sparse_refine_loss"])

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
        # AdaptiveClassificationCompositeLoss | None, C 级 sparse refine 损失函数; 为 None 时不参与梯度
        self.ligand_sparse_refine_loss = (
            ligand_sparse_refine_loss
            if (ligand_sparse_refine_loss is None or isinstance(ligand_sparse_refine_loss, nn.Module))
            else instantiate(ligand_sparse_refine_loss)
        )
        if self.ligand_sparse_refine_loss is not None and not isinstance(self.ligand_sparse_refine_loss, AdaptiveClassificationCompositeLoss):
            raise TypeError("ligand_sparse_refine_loss 必须为 AdaptiveClassificationCompositeLoss。")

        if compile:
            self.backbone = torch.compile(self.backbone)

        # tuple[int, ...] | None, sparse candidate builder 配置的候选类别 ID
        self._sparse_candidate_class_ids: tuple[int, ...] | None = self._resolve_sparse_candidate_class_ids()
        self._validate_ligand_sparse_refine_loss_config()
        # torch.Tensor | None, (K,), best-F1 阈值缓存; builder 未启用时为 None
        self._cached_voxel_ligand_p_best_by_class: torch.Tensor | None = self._init_candidate_threshold_cache(
            initial_p_best_by_class,
            "initial_p_best_by_class",
        )
        # torch.Tensor | None, (K,), sampling 阈值缓存; builder 未启用时为 None
        self._cached_voxel_ligand_p_sampling_by_class: torch.Tensor | None = self._init_candidate_threshold_cache(
            initial_p_sampling_by_class,
            "initial_p_sampling_by_class",
        )
        # torch.Tensor | None, (K,), 每类 best-F1 这个值本身的缓存; builder 未启用时为 None
        self._cached_voxel_ligand_best_f1_before_refine_by_class: torch.Tensor | None = (
            None if self._sparse_candidate_class_ids is None else torch.full((len(self._sparse_candidate_class_ids),), float("nan"), dtype=torch.float32)
        )
        self._candidate_warmup_steps = 0

        # BinaryAveragePrecision, 验证阶段的 PR-AUC 指标；binned 指标默认在 GPU 更新，非 binned 指标默认在 CPU 累积状态
        # 各指标仅在对应损失启用时构建; 一个 epoch 可能多次验证，每次都会 reset/update/compute
        from torchmetrics.classification import BinaryAveragePrecision
        self._val_metric_update_counts: dict[str, int] = {}
        self._val_metric_specs: dict[str, dict[str, Any]] = {}
        self._val_metric_devices: dict[str, torch.device] = {}
        self._multiclass_metric_names: dict[str, list[str]] = {"atom": [], "voxel_aux": [], "voxel_ligand": []}
        self._class_names = self._resolve_class_names(kwargs)
        if self.atom_loss is not None:
            self.val_atom_pr_auc = BinaryAveragePrecision(compute_on_cpu=True)
            self._register_val_metric("val/atom_pr_auc", thresholds=None, branch="atom")
            self._init_multiclass_ap_metrics("atom", self.atom_loss, BinaryAveragePrecision, None)
        if self.voxel_aux_loss is not None:
            self.val_voxel_aux_pr_auc = BinaryAveragePrecision(compute_on_cpu=True)
            self._register_val_metric("val/voxel_aux_pr_auc", thresholds=None, branch="voxel_aux")
            self._init_multiclass_ap_metrics("voxel_aux", self.voxel_aux_loss, BinaryAveragePrecision, None)
        if self.voxel_ligand_loss is not None:
            self.val_voxel_ligand_pr_auc = BinaryAveragePrecision(
                compute_on_cpu=voxel_ligand_pr_auc_thresholds is None,
                thresholds=voxel_ligand_pr_auc_thresholds,
            )
            self._register_val_metric("val/voxel_ligand_pr_auc", thresholds=voxel_ligand_pr_auc_thresholds, branch="voxel_ligand")
            self._init_multiclass_ap_metrics("voxel_ligand", self.voxel_ligand_loss, BinaryAveragePrecision, voxel_ligand_pr_auc_thresholds)

        if self._sparse_candidate_class_ids is not None:
            if voxel_ligand_pr_auc_thresholds is None or int(voxel_ligand_pr_auc_thresholds) <= 0:
                raise ValueError("candidate threshold stats 启用时 voxel_ligand_pr_auc_thresholds 必须为正整数。")
            # int, threshold histogram bin 数
            self._voxel_ligand_threshold_bin_count: int | None = int(voxel_ligand_pr_auc_thresholds)
            # torch.Tensor, (num_bins,), bin lower-edge 阈值网格
            self.register_buffer(
                "_voxel_ligand_threshold_grid",
                torch.arange(int(voxel_ligand_pr_auc_thresholds), dtype=torch.float32) / float(voxel_ligand_pr_auc_thresholds),
                persistent=False,
            )
            # torch.Tensor, (K,num_bins), 每类正例概率 histogram
            self.register_buffer(
                "_voxel_ligand_pos_hist_by_class",
                torch.zeros((len(self._sparse_candidate_class_ids), int(voxel_ligand_pr_auc_thresholds)), dtype=torch.long),
                persistent=False,
            )
            # torch.Tensor, (K,num_bins), 每类负例概率 histogram
            self.register_buffer(
                "_voxel_ligand_neg_hist_by_class",
                torch.zeros((len(self._sparse_candidate_class_ids), int(voxel_ligand_pr_auc_thresholds)), dtype=torch.long),
                persistent=False,
            )
            if self.ligand_sparse_refine_loss is not None:
                # torch.Tensor, (K,num_bins), C 内正例 score histogram
                self.register_buffer(
                    "_ligand_sparse_refine_pos_hist_by_class",
                    torch.zeros((len(self._sparse_candidate_class_ids), int(voxel_ligand_pr_auc_thresholds)), dtype=torch.long),
                    persistent=False,
                )
                # torch.Tensor, (K,num_bins), C 内负例 score histogram
                self.register_buffer(
                    "_ligand_sparse_refine_neg_hist_by_class",
                    torch.zeros((len(self._sparse_candidate_class_ids), int(voxel_ligand_pr_auc_thresholds)), dtype=torch.long),
                    persistent=False,
                )
                # torch.Tensor, (K,), dense valid GT 正例数
                self.register_buffer(
                    "_ligand_sparse_refine_total_gt_pos_by_class",
                    torch.zeros((len(self._sparse_candidate_class_ids),), dtype=torch.long),
                    persistent=False,
                )
                self.register_buffer("_ligand_sparse_refine_candidate_count_total", torch.zeros((), dtype=torch.long), persistent=False)
                self.register_buffer("_ligand_sparse_refine_anchor_count_total", torch.zeros((), dtype=torch.long), persistent=False)
                self.register_buffer("_ligand_sparse_refine_box_count_total", torch.zeros((), dtype=torch.long), persistent=False)
        else:
            self._voxel_ligand_threshold_bin_count = None
            self._voxel_ligand_threshold_grid = None
            self._voxel_ligand_pos_hist_by_class = None
            self._voxel_ligand_neg_hist_by_class = None
        self._sync_sparse_candidate_runtime_to_backbone()

        # WarmupThenReduceLROnPlateau | None, 手动管理的 validation 级 plateau 调度器
        self._warmup_plateau_scheduler: WarmupThenReduceLROnPlateau | None = None
        # dict[str, Any] | None, checkpoint 恢复时暂存的 plateau 调度器状态
        self._pending_warmup_plateau_state: dict[str, Any] | None = None

    # ------------------------------------------------------------------
    # 工具方法
    # ------------------------------------------------------------------

    def _unwrap_backbone(self) -> nn.Module:
        """
        返回未被 torch.compile 包装的 backbone。

        输出:
            - backbone: nn.Module, 原始 VolumePointStage1Model 或等价模块
        """
        return getattr(self.backbone, "_orig_mod", self.backbone)

    def _resolve_sparse_candidate_class_ids(self) -> tuple[int, ...] | None:
        """
        从 backbone 的 candidate builder 读取候选类别 ID。

        输出:
            - class_ids: tuple[int, ...] | None, builder 未启用时为 None; 启用时为前景类别 ID
        """
        backbone = self._unwrap_backbone()
        if not hasattr(backbone, "get_sparse_candidate_class_ids"):
            return None
        class_ids = backbone.get_sparse_candidate_class_ids()
        if class_ids is None:
            return None
        if self.voxel_ligand_loss is None:
            raise ValueError("candidate builder 启用时必须配置 voxel_ligand_loss。")
        num_classes = int(getattr(self.voxel_ligand_loss, "num_classes", 2))
        if num_classes <= 2 and tuple(class_ids) != (1,):
            raise ValueError("二分类 voxel_ligand_loss 只允许 candidate_class_ids=(1,)。")
        if num_classes > 2:
            invalid_ids = [class_id for class_id in class_ids if class_id <= 0 or class_id >= num_classes]
            if invalid_ids:
                raise ValueError(f"candidate_class_ids={tuple(class_ids)} 与 voxel_ligand_loss.num_classes={num_classes} 不匹配。")
        return tuple(int(class_id) for class_id in class_ids)

    def _validate_ligand_sparse_refine_loss_config(self) -> None:
        """
        校验 sparse refine loss 与 candidate class 配置是否一致。

        输出:
            - None, 配置不一致时 fail-fast
        """
        if self.ligand_sparse_refine_loss is None:
            return
        if self._sparse_candidate_class_ids is None:
            raise ValueError("ligand_sparse_refine_loss 启用时必须同时启用 sparse candidate builder。")
        num_classes = int(self.ligand_sparse_refine_loss.num_classes)
        if num_classes <= 2:
            if self._sparse_candidate_class_ids != (1,):
                raise ValueError("二分类 ligand_sparse_refine_loss 只允许 candidate_class_ids=(1,)。")
            return
        invalid_ids = [class_id for class_id in self._sparse_candidate_class_ids if class_id <= 0 or class_id >= num_classes]
        if invalid_ids:
            raise ValueError(f"candidate_class_ids={self._sparse_candidate_class_ids} 与 ligand_sparse_refine_loss.num_classes={num_classes} 不匹配。")

    def _init_candidate_threshold_cache(
        self,
        initial_values: list[float] | tuple[float, ...] | None,
        value_name: str,
    ) -> torch.Tensor | None:
        """
        从显式 initial 参数初始化 candidate threshold cache。

        输入参数:
            - initial_values: list[float] | tuple[float, ...] | None, (K,), 用户显式给定的初始阈值
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

    def _normalize_candidate_checkpoint_tensor(self, value: Any, value_name: str) -> torch.Tensor:
        """
        将 checkpoint 中的 candidate cache 规范化为 CPU float 向量。

        输入参数:
            - value: Any, checkpoint 中读取的张量或可转张量对象
            - value_name: str, 错误信息中的字段名

        输出:
            - tensor: torch.Tensor, (K,), CPU float candidate cache
        """
        # torch.Tensor, (K,), checkpoint cache 的 CPU float 视图
        tensor = torch.as_tensor(value).detach().cpu().float().reshape(-1)
        if int(tensor.numel()) != len(self._sparse_candidate_class_ids):
            raise ValueError(f"checkpoint 中的 {value_name} 长度必须等于 candidate_class_ids 数量。")
        return tensor

    def _load_candidate_checkpoint_finite_tensor(self, value: Any, value_name: str) -> torch.Tensor:
        """
        从 checkpoint 读取并校验 candidate finite cache。

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

    def _register_val_metric(self, metric_name: str, thresholds: Optional[int], branch: str, class_id: int | None = None) -> None:
        """
        注册验证指标的更新策略元信息。

        输入参数:
            - metric_name: str, Lightning 日志中的指标名
            - thresholds: int | None, binned AP 的阈值数量; None 表示非 binned AP
            - branch: str, 指标所属分支, 例如 atom / voxel_aux / voxel_ligand
            - class_id: int | None, 多分类前景类别 ID; 二分类主指标为 None

        输出:
            - None, 原地记录 metric 更新次数与设备策略元信息
        """
        self._val_metric_update_counts[metric_name] = 0
        self._val_metric_specs[metric_name] = {
            "thresholds": thresholds,
            "binned": thresholds is not None,
            "branch": branch,
            "class_id": class_id,
        }

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
            metric_kwargs = {"compute_on_cpu": thresholds is None}
            if thresholds is not None:
                metric_kwargs["thresholds"] = thresholds
            setattr(self, f"val_{prefix}_ap_{class_name}", metric_cls(**metric_kwargs))
            self._register_val_metric(metric_name, thresholds=thresholds, branch=prefix, class_id=class_id)
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

    def _reset_voxel_ligand_threshold_histograms(self) -> None:
        """
        清空 voxel ligand candidate threshold histogram。

        输出:
            - None, 原地清零 pos/neg histogram; builder 未启用时 no-op
        """
        if self._sparse_candidate_class_ids is None:
            return
        self._voxel_ligand_pos_hist_by_class.zero_()
        self._voxel_ligand_neg_hist_by_class.zero_()

    def _reset_ligand_sparse_refine_metric_buffers(self) -> None:
        """
        清空 sparse refine validation metric buffers。

        输出:
            - None, 原地清零 C 级 histogram、dense GT 计数与 C/P/BOX 数量累计
        """
        if self.ligand_sparse_refine_loss is None:
            return
        self._ligand_sparse_refine_pos_hist_by_class.zero_()
        self._ligand_sparse_refine_neg_hist_by_class.zero_()
        self._ligand_sparse_refine_total_gt_pos_by_class.zero_()
        self._ligand_sparse_refine_candidate_count_total.zero_()
        self._ligand_sparse_refine_anchor_count_total.zero_()
        self._ligand_sparse_refine_box_count_total.zero_()

    def _update_voxel_ligand_best_f1_stats(
        self,
        logits: torch.Tensor,
        ligand_dist_map: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> None:
        """
        累加 candidate threshold best-F1 使用的 voxel ligand 概率 histogram。

        输入参数:
            - logits: torch.Tensor, (B,1,D,H,W) 或 (B,C,D,H,W), ligand head logits
            - ligand_dist_map: torch.Tensor, (B,D,H,W) 或 (B,C,D,H,W), ligand 距离监督图
            - valid_mask: torch.Tensor, (B,D,H,W) 或 (B,1,D,H,W), 有效体素掩码

        输出:
            - None, 原地累加 pos/neg histogram
        """
        if self._sparse_candidate_class_ids is None:
            return
        if valid_mask.ndim == 5 and valid_mask.shape[1] == 1:
            mask = valid_mask.squeeze(1).bool()
        elif valid_mask.ndim == 4:
            mask = valid_mask.bool()
        else:
            raise ValueError(f"voxel_valid_mask 期望为 (B,D,H,W) 或 (B,1,D,H,W)，实际 {tuple(valid_mask.shape)}")
        # torch.Tensor, (B,D,H,W), 多分类类别 ID 或二分类 0/1 标签
        target = self._ligand_target_from_dist(
            ligand_dist_map=ligand_dist_map,
            logit_dim=int(logits.shape[1]),
            device=logits.device,
            dtype=logits.dtype,
        )
        mask = mask.to(device=logits.device)
        if logits.shape[1] == 1:
            if self._sparse_candidate_class_ids != (1,):
                raise ValueError("单通道 voxel_ligand logits 只允许 candidate_class_ids=(1,)。")
            # torch.Tensor, (B,1,D,H,W), 二分类前景概率
            prob_by_class = torch.sigmoid(logits[:, :1]).detach().float()
        else:
            if max(self._sparse_candidate_class_ids) >= int(logits.shape[1]):
                raise ValueError(f"candidate_class_ids={self._sparse_candidate_class_ids} 超出 logits channel 数 {int(logits.shape[1])}。")
            # torch.Tensor, (B,C,D,H,W), 多分类 softmax 概率
            prob = torch.softmax(logits, dim=1).detach().float()
            # torch.Tensor, (K,), 候选类别 channel 索引
            class_index = torch.as_tensor(self._sparse_candidate_class_ids, device=logits.device, dtype=torch.long)
            prob_by_class = prob.index_select(dim=1, index=class_index)
        num_bins = int(self._voxel_ligand_threshold_bin_count)
        for class_pos, class_id in enumerate(self._sparse_candidate_class_ids):
            # torch.Tensor, (M,), 当前类在有效体素上的概率
            prob_flat = prob_by_class[:, class_pos].reshape(-1)[mask.reshape(-1)]
            # torch.Tensor, (M,), 当前类 one-vs-rest 硬标签
            target_flat = (target.reshape(-1)[mask.reshape(-1)] == int(class_id))
            if prob_flat.numel() == 0:
                continue
            # torch.Tensor, (M,), 概率所在 histogram bin
            bin_idx = torch.floor(prob_flat.clamp(0.0, 1.0) * num_bins).long().clamp(max=num_bins - 1)
            # torch.Tensor, (num_bins,), 当前 batch 正例 histogram
            pos_hist = torch.bincount(bin_idx[target_flat], minlength=num_bins).to(device=self.device, dtype=torch.long)
            # torch.Tensor, (num_bins,), 当前 batch 负例 histogram
            neg_hist = torch.bincount(bin_idx[~target_flat], minlength=num_bins).to(device=self.device, dtype=torch.long)
            self._voxel_ligand_pos_hist_by_class[class_pos] += pos_hist
            self._voxel_ligand_neg_hist_by_class[class_pos] += neg_hist

    @staticmethod
    def _is_tuning_trainer(trainer: Any) -> bool:
        """
        判断当前 trainer 是否处于 Lightning tuner 生命周期。

        输入参数:
            - trainer: Any, Lightning Trainer 或测试 stub

        输出:
            - is_tuning: bool, True 表示 batch-size tuning 等 tuner 探测阶段
        """
        # str, trainer state.fn 的小写字符串表示; Lightning 版本间枚举名可能不同
        state_fn = getattr(getattr(trainer, "state", None), "fn", None)
        state_fn_name = str(getattr(state_fn, "value", state_fn)).lower()
        return "tun" in state_fn_name

    def _sync_sparse_candidate_runtime_to_backbone(self) -> None:
        """
        将 wrapper 中的 candidate runtime/cache 同步给 backbone。

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
        if trainer is None:
            allow_warmup = False
            global_step = 0
        else:
            state_fn = getattr(getattr(trainer, "state", None), "fn", None)
            state_fn_name = str(getattr(state_fn, "value", state_fn)).lower()
            allow_warmup = (
                bool(getattr(trainer, "sanity_checking", False))
                or "fit" in state_fn_name
                or self._is_tuning_trainer(trainer)
            )
            global_step = int(getattr(trainer, "global_step", self.global_step))
        if hasattr(backbone, "set_sparse_candidate_runtime"):
            backbone.set_sparse_candidate_runtime(
                global_step=global_step,
                candidate_warmup_steps=int(self._candidate_warmup_steps),
                allow_warmup_fixed_topk=allow_warmup,
            )
        if hasattr(backbone, "set_sparse_candidate_thresholds"):
            backbone.set_sparse_candidate_thresholds(
                p_best_by_class=self._cached_voxel_ligand_p_best_by_class,
                p_sampling_by_class=self._cached_voxel_ligand_p_sampling_by_class,
            )

    def _resolve_metric_update_device(self, metric_name: str, source_device: torch.device) -> torch.device:
        """
        根据指标类型选择 update 设备。

        输入参数:
            - metric_name: str, Lightning 日志中的指标名
            - source_device: torch.device, 当前 logits/preds 所在设备

        输出:
            - device: torch.device, 本次 metric.update 使用的设备
        """
        policy = str(self.hparams.val_metric_device_policy)
        if policy == "cpu":
            return torch.device("cpu")
        if policy == "gpu":
            return source_device if source_device.type != "cpu" else self.device
        if policy != "auto":
            raise ValueError(f"Unknown val_metric_device_policy: {policy!r}")

        spec = self._val_metric_specs[metric_name]
        if bool(spec["binned"]):
            return source_device if source_device.type != "cpu" else self.device
        return torch.device("cpu")

    def _update_val_metric(self, metric_name: str, metric_obj: Any, preds: torch.Tensor, targets: torch.Tensor) -> None:
        """
        按指标设备策略更新 torchmetrics 指标。

        输入参数:
            - metric_name: str, Lightning 日志中的指标名
            - metric_obj: Any, torchmetrics 指标对象, 内部可能包含 thresholds 等状态张量
            - preds: torch.Tensor, (M,), 当前 batch 的预测分数
            - targets: torch.Tensor, (M,), 当前 batch 的硬标签

        输出:
            - None, 原地更新 metric_obj 状态
        """
        device = self._resolve_metric_update_device(metric_name, preds.device)
        # GPU -> CPU 的异步拷贝不能立刻交给 CPU metric 读取，否则标签可能仍是未就绪内容。
        non_blocking = device.type != "cpu"
        preds = preds.to(device, non_blocking=non_blocking)
        targets = targets.to(device, non_blocking=non_blocking)
        if self._val_metric_devices.get(metric_name) != device:
            metric_obj.to(device)
            self._val_metric_devices[metric_name] = device
        metric_obj.update(preds, targets)

    def _ligand_target_from_dist(
        self,
        ligand_dist_map: torch.Tensor,
        logit_dim: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """
        从 voxel ligand loss 配置生成验证指标使用的硬标签。

        输入参数:
            - ligand_dist_map: torch.Tensor, 二分类 (B,D,H,W)/(B,1,D,H,W) 或多分类 (B,C,D,H,W), ligand 距离监督图
            - logit_dim: int, 当前 logits 通道数; 二分类为 1, 多分类为类别数
            - device: torch.device, 输出 target 所在设备
            - dtype: torch.dtype, 距离图计算 dtype

        输出:
            - target: torch.Tensor, (B,D,H,W), 二分类 0/1 标签或多分类类别 ID 标签
        """
        if self.voxel_ligand_loss is None or not hasattr(self.voxel_ligand_loss, "target_from_ligand_dist_map"):
            raise TypeError("voxel_ligand_loss 必须支持 target_from_ligand_dist_map() 才能更新 ligand 指标。")
        return self.voxel_ligand_loss.target_from_ligand_dist_map(
            ligand_dist_map=ligand_dist_map,
            logit_dim=logit_dim,
            device=device,
            dtype=dtype,
        )

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
            preds = torch.sigmoid(logits[:, 0]).detach().float().reshape(-1)[mask.reshape(-1)]
            # torch.Tensor, (M,), 有效位置二分类标签, 取值 0/1
            targets = target.reshape(-1).long()[mask.reshape(-1)]
            self._update_val_metric(binary_metric_name, binary_metric, preds, targets)
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
            preds = prob[:, class_id].reshape(-1)[mask_flat]
            targets = (target_flat[mask_flat] == class_id).long()
            self._update_val_metric(metric_name, metric_obj, preds, targets)
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

    @staticmethod
    def _normalize_voxel_valid_mask(
        voxel_valid_mask: torch.Tensor,
        spatial_shape_zyx: tuple[int, int, int],
    ) -> torch.Tensor:
        """
        将 voxel_valid_mask 规范化为 `(B,D,H,W)` bool 掩码。

        输入参数:
            - voxel_valid_mask: torch.Tensor, (B,D,H,W) 或 (B,1,D,H,W), 有效体素掩码
            - spatial_shape_zyx: tuple[int,int,int], 期望空间形状 `(D,H,W)`

        输出:
            - mask: torch.Tensor, (B,D,H,W), bool 有效体素掩码
        """
        if voxel_valid_mask.ndim == 5 and voxel_valid_mask.shape[1] == 1:
            mask = voxel_valid_mask.squeeze(1).bool()
        elif voxel_valid_mask.ndim == 4:
            mask = voxel_valid_mask.bool()
        else:
            raise ValueError(f"voxel_valid_mask 期望为 (B,D,H,W) 或 (B,1,D,H,W)，实际 {tuple(voxel_valid_mask.shape)}")
        if tuple(mask.shape[-3:]) != tuple(int(value) for value in spatial_shape_zyx):
            raise ValueError(f"voxel_valid_mask 空间形状 {tuple(mask.shape[-3:])} 与 dense target {spatial_shape_zyx} 不一致。")
        return mask

    def _sample_ligand_refine_supervision(
        self,
        outputs: dict[str, Any],
        batch: dict[str, Any],
    ) -> dict[str, torch.Tensor]:
        """
        从 dense ligand 距离图采样 C 级 sparse refine 监督。

        输入参数:
            - outputs: dict[str, Any], backbone 输出, 包含 ligand_refine_logits_C/candidate_batch_index/candidate_voxel_zyx
            - batch: dict[str, Any], 当前 batch, 包含 ligand_dist_map/voxel_valid_mask

        输出:
            - supervision: dict[str, torch.Tensor], dense target/mask 与 C 级 target/mask
        """
        logits_C = outputs["ligand_refine_logits_C"]
        logit_dim = int(logits_C.shape[1])
        # torch.Tensor, (B,D,H,W), dense hard-label target
        dense_target = self.ligand_sparse_refine_loss.target_from_ligand_dist_map(
            ligand_dist_map=batch["ligand_dist_map"],
            logit_dim=logit_dim,
            device=logits_C.device,
            dtype=logits_C.dtype,
        )
        # torch.Tensor, (B,D,H,W), dense valid mask
        dense_valid_mask = self._normalize_voxel_valid_mask(
            batch["voxel_valid_mask"],
            spatial_shape_zyx=tuple(int(value) for value in dense_target.shape[-3:]),
        ).to(device=logits_C.device)
        # torch.Tensor, (sumC,), 每个 C 所属 BOX 索引
        idx_b = outputs["candidate_batch_index"].to(device=logits_C.device, dtype=torch.long)
        # torch.Tensor, (sumC,3), 每个 C 的 z/y/x 离散坐标
        idx_zyx = outputs["candidate_voxel_zyx"].to(device=logits_C.device, dtype=torch.long)
        # torch.Tensor, (sumC,), C 上 hard-label target
        target_C = dense_target[idx_b, idx_zyx[:, 0], idx_zyx[:, 1], idx_zyx[:, 2]]
        # torch.Tensor, (sumC,), C 上 valid mask
        valid_C = dense_valid_mask[idx_b, idx_zyx[:, 0], idx_zyx[:, 1], idx_zyx[:, 2]]
        supervision = {
            "ligand_dense_target": dense_target,
            "ligand_dense_valid_mask": dense_valid_mask,
            "ligand_refine_target_C": target_C,
            "ligand_refine_valid_mask_C": valid_C,
        }
        outputs.update(supervision)
        return supervision

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

    def _resolve_ligand_sparse_refine_loss_warmup_steps(self) -> int:
        """
        解析 sparse refine loss 独立 linear warmup 步数。

        输出:
            - warmup_steps: int, sparse refine loss 从 start_weight 到 final_weight 的步数
        """
        sched_cfg = self.hparams.ligand_sparse_refine_loss_schedule
        if sched_cfg is None:
            return 0
        if str(sched_cfg["mode"]) != "linear_warmup":
            raise ValueError("ligand_sparse_refine_loss_schedule.mode 只支持 linear_warmup。")
        warmup_steps = sched_cfg["warmup_steps"]
        if warmup_steps is not None:
            warmup_steps = int(warmup_steps)
            if warmup_steps < 0:
                raise ValueError("ligand_sparse_refine_loss_schedule.warmup_steps 必须 >= 0。")
            return warmup_steps
        warmup_ratio = float(sched_cfg["warmup_ratio"])
        if not (0.0 <= warmup_ratio < 1.0):
            raise ValueError("ligand_sparse_refine_loss_schedule.warmup_ratio 必须在 [0,1) 内。")
        total_steps = getattr(self.trainer, "estimated_stepping_batches", None)
        if total_steps is None or int(total_steps) <= 0:
            raise RuntimeError("ligand_sparse_refine_loss_schedule 需要 trainer.estimated_stepping_batches 为正数。")
        return int(round(int(total_steps) * warmup_ratio))

    def _ligand_sparse_refine_loss_effective_weight(self) -> torch.Tensor:
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
        warmup_steps = self._resolve_ligand_sparse_refine_loss_warmup_steps()
        if warmup_steps == 0:
            return torch.tensor(final_weight, device=self.device, dtype=torch.float32)
        try:
            trainer = self.trainer
        except RuntimeError:
            trainer = None
        global_step = int(getattr(trainer, "global_step", self.global_step)) if trainer is not None else int(self.global_step)
        progress = min(max(float(global_step) / float(warmup_steps), 0.0), 1.0)
        weight = start_weight + progress * (final_weight - start_weight)
        return torch.tensor(weight, device=self.device, dtype=torch.float32)

    def _compute_ligand_sparse_refine_loss(
        self,
        outputs: dict[str, Any],
        batch: dict[str, Any],
    ) -> torch.Tensor | None:
        """
        计算 C 级 sparse refine loss。

        输入参数:
            - outputs: dict[str, Any], backbone 输出, 可包含 ligand_refine_logits_C
            - batch: dict[str, Any], 当前 batch, 包含 ligand_dist_map 与 voxel_valid_mask

        输出:
            - loss: torch.Tensor | None, (), C 级 sparse refine loss; 未启用或无 logits 时为 None
        """
        if self.ligand_sparse_refine_loss is None:
            return None
        if outputs.get("ligand_refine_logits_C") is None:
            return None
        supervision = self._sample_ligand_refine_supervision(outputs=outputs, batch=batch)
        loss_out = self.ligand_sparse_refine_loss(
            logits=outputs["ligand_refine_logits_C"],
            target=supervision["ligand_refine_target_C"],
            hardmask=supervision["ligand_refine_valid_mask_C"],
        )
        return self._loss_output_to_tensor(loss_out)

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

        # torch.Tensor | None, 标量, C 级 sparse refine 损失
        ligand_sparse_refine_loss = self._compute_ligand_sparse_refine_loss(outputs=outputs, batch=batch)
        if ligand_sparse_refine_loss is not None:
            effective_weight = self._ligand_sparse_refine_loss_effective_weight()
            total_loss = total_loss + effective_weight * ligand_sparse_refine_loss
            loss_dict["ligand_sparse_refine_loss"] = ligand_sparse_refine_loss
            loss_dict["ligand_sparse_refine_loss_weight_effective"] = effective_weight

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

    def _update_val_ligand_sparse_refine_metric(self, outputs: dict[str, Any], batch: dict[str, Any]) -> None:
        """
        用当前 batch 的 C 级 refined logits 更新 sparse refine 验证指标。

        输入参数:
            - outputs: dict[str, Any], backbone 输出, 包含 ligand_refine_logits_C 与 C/P 计数字段
            - batch: dict[str, Any], 当前 batch, 包含 ligand_dist_map 与 voxel_valid_mask

        输出:
            - None, 原地累加 sparse refine histogram 与数量计数
        """
        if self.ligand_sparse_refine_loss is None:
            return
        if outputs.get("ligand_refine_logits_C") is None:
            return
        supervision = self._sample_ligand_refine_supervision(outputs=outputs, batch=batch)
        logits_C = outputs["ligand_refine_logits_C"]
        valid_C = supervision["ligand_refine_valid_mask_C"].bool()
        target_C = supervision["ligand_refine_target_C"].long()
        dense_valid = supervision["ligand_dense_valid_mask"].bool()
        dense_target = supervision["ligand_dense_target"].long()
        num_bins = int(self._voxel_ligand_threshold_bin_count)
        for class_pos, class_id in enumerate(self._sparse_candidate_class_ids):
            if logits_C.shape[1] == 1:
                score_C = torch.sigmoid(logits_C[:, 0]).detach().float()
            else:
                score_C = torch.softmax(logits_C, dim=1)[:, int(class_id)].detach().float()
            # torch.Tensor, (M,), 有效 C 上当前类分数
            score_valid = score_C[valid_C]
            # torch.Tensor, (M,), 有效 C 上 hard-label target
            target_valid = target_C[valid_C]
            # torch.Tensor, (M,), True 表示有效 C 是当前前景类正例
            positive_C = target_valid == int(class_id)
            if score_valid.numel() > 0:
                bin_idx = torch.floor(score_valid.clamp(0.0, 1.0) * num_bins).long().clamp(max=num_bins - 1)
                pos_hist = torch.bincount(bin_idx[positive_C], minlength=num_bins).to(device=self.device, dtype=torch.long)
                neg_hist = torch.bincount(bin_idx[~positive_C], minlength=num_bins).to(device=self.device, dtype=torch.long)
                self._ligand_sparse_refine_pos_hist_by_class[class_pos] += pos_hist
                self._ligand_sparse_refine_neg_hist_by_class[class_pos] += neg_hist
            self._ligand_sparse_refine_total_gt_pos_by_class[class_pos] += ((dense_valid) & (dense_target == int(class_id))).sum().to(device=self.device, dtype=torch.long)
        self._ligand_sparse_refine_candidate_count_total += outputs["candidate_counts"].sum().to(device=self.device, dtype=torch.long)
        self._ligand_sparse_refine_anchor_count_total += outputs["anchor_counts"].sum().to(device=self.device, dtype=torch.long)
        self._ligand_sparse_refine_box_count_total += torch.tensor(int(dense_target.shape[0]), device=self.device, dtype=torch.long)

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

        if getattr(self.voxel_ligand_loss, "hard_label_threshold", None) is not None:
            voxel_target = self._ligand_target_from_dist(
                ligand_dist_map=ligand_dist_map,
                logit_dim=int(voxel_logits_ligand.shape[1]),
                device=voxel_logits_ligand.device,
                dtype=voxel_logits_ligand.dtype,
            )
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
        self._update_voxel_ligand_best_f1_stats(
            logits=voxel_logits_ligand,
            ligand_dist_map=ligand_dist_map,
            valid_mask=batch["voxel_valid_mask"],
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
        self._sync_sparse_candidate_runtime_to_backbone()
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
        if "ligand_sparse_refine_loss" in loss_dict:
            self.log(
                "train/ligand_sparse_refine_loss",
                loss_dict["ligand_sparse_refine_loss"],
                prog_bar=False,
                on_step=True,
                on_epoch=True,
                sync_dist=True,
            )
            self.log(
                "train/ligand_sparse_refine_loss_weight_effective",
                loss_dict["ligand_sparse_refine_loss_weight_effective"],
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
        self._sync_sparse_candidate_runtime_to_backbone()
        outputs = self(batch_dict)
        total_loss, loss_dict = self._compute_total_loss(outputs=outputs, batch=batch_dict)
        # 用当前 batch 更新 PR-AUC 指标；tuner 阶段也完整计算，以便尽早暴露真实验证链路问题
        self._update_val_atom_metric(outputs=outputs, batch=batch_dict)
        self._update_val_voxel_aux_metric(outputs=outputs, batch=batch_dict)
        self._update_val_voxel_ligand_metric(outputs=outputs, batch=batch_dict)
        self._update_val_ligand_sparse_refine_metric(outputs=outputs, batch=batch_dict)

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
        if "ligand_sparse_refine_loss" in loss_dict:
            self.log(
                "val/ligand_sparse_refine_loss",
                loss_dict["ligand_sparse_refine_loss"],
                prog_bar=False,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )
            self.log(
                "val/ligand_sparse_refine_loss_weight_effective",
                loss_dict["ligand_sparse_refine_loss_weight_effective"],
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

    def _compute_log_reset_metric_safe(self, metric_obj, metric_name: str) -> torch.Tensor | None:
        """
        有有效 update 时计算、日志记录并重置单个 metric。

        输入参数:
            - metric_obj: BinaryAveragePrecision, torchmetrics 指标对象
            - metric_name: str, 日志中使用的指标名, 如 "val/atom_pr_auc"

        输出:
            - score_gpu: torch.Tensor | None, (), 当前 rank 上计算得到的指标值; 无 update 时为 None
        """
        local_updates = int(self._val_metric_update_counts.get(metric_name, 0))
        update_count = torch.tensor(float(local_updates), device=self.device, dtype=torch.float32)
        try:
            trainer = self.trainer
        except RuntimeError:
            trainer = None
        if trainer is not None and int(getattr(trainer, "world_size", 1)) > 1:
            update_count = self.all_gather(update_count).sum()
        if update_count <= 0:
            metric_obj.reset()
            self._val_metric_update_counts[metric_name] = 0
            return None
        if local_updates <= 0:
            score_gpu = torch.tensor(0.0, device=self.device, dtype=torch.float32)
        else:
            prev_to_sync = getattr(metric_obj, "_to_sync", True)
            metric_obj._to_sync = False
            try:
                # torch.Tensor, (), 当前 rank 本地 metric 计算值
                score_local = metric_obj.compute()
            finally:
                metric_obj._to_sync = prev_to_sync
            # torch.Tensor, (), 当前 rank 指标值
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
                update_count = torch.tensor(float(local_updates), device=self.device, dtype=torch.float32)
                try:
                    trainer = self.trainer
                except RuntimeError:
                    trainer = None
                if trainer is not None and int(getattr(trainer, "world_size", 1)) > 1:
                    update_count = self.all_gather(update_count).sum()
                if update_count <= 0:
                    metric_obj.reset()
                    self._val_metric_update_counts[metric_name] = 0
                    continue
                if local_updates <= 0:
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
            self._val_metric_update_counts[macro_name] = 0
            if len(class_scores) == 0:
                continue
            macro_score = torch.stack(class_scores).mean()
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

    def _compute_log_update_voxel_ligand_best_f1_thresholds(self) -> dict[str, torch.Tensor]:
        """
        从本轮 validation histogram 计算并缓存 voxel ligand best-F1 与 sampling 阈值。

        输出:
            - metrics: dict[str, torch.Tensor], 当前 validation end 现场计算出的阈值指标
        """
        if self._sparse_candidate_class_ids is None:
            return {}
        try:
            trainer = self.trainer
        except RuntimeError:
            trainer = None
        if trainer is not None and (bool(getattr(trainer, "sanity_checking", False)) or self._is_tuning_trainer(trainer)):
            self._reset_voxel_ligand_threshold_histograms()
            return {}
        # torch.Tensor, (K,num_bins), 当前 rank 本地正例 histogram
        pos_hist = self._voxel_ligand_pos_hist_by_class.detach().to(device=self.device, dtype=torch.float32)
        # torch.Tensor, (K,num_bins), 当前 rank 本地负例 histogram
        neg_hist = self._voxel_ligand_neg_hist_by_class.detach().to(device=self.device, dtype=torch.float32)
        if trainer is not None and int(getattr(trainer, "world_size", 1)) > 1:
            pos_hist = self.all_gather(pos_hist).sum(dim=0)
            neg_hist = self.all_gather(neg_hist).sum(dim=0)
        # torch.Tensor, (K,num_bins), 每个 bin lower-edge 作为阈值时的 TP/FP/FN
        tp_at_threshold = torch.cumsum(pos_hist.flip(-1), dim=-1).flip(-1)
        fp_at_threshold = torch.cumsum(neg_hist.flip(-1), dim=-1).flip(-1)
        fn_at_threshold = pos_hist.sum(dim=-1, keepdim=True) - tp_at_threshold
        denominator = 2.0 * tp_at_threshold + fp_at_threshold + fn_at_threshold
        f1_by_threshold = torch.where(denominator > 0.0, 2.0 * tp_at_threshold / denominator, torch.zeros_like(denominator))
        # torch.Tensor, (K,), 本轮计算得到的 best-F1 阈值候选; 无旧缓存时先填 NaN
        new_p_best = (
            torch.full((len(self._sparse_candidate_class_ids),), float("nan"), device=self.device, dtype=torch.float32)
            if self._cached_voxel_ligand_p_best_by_class is None
            else self._cached_voxel_ligand_p_best_by_class.to(device=self.device).clone()
        )
        # torch.Tensor, (K,), 本轮计算得到的 sampling 阈值候选; 无旧缓存时先填 NaN
        new_p_sampling = (
            torch.full((len(self._sparse_candidate_class_ids),), float("nan"), device=self.device, dtype=torch.float32)
            if self._cached_voxel_ligand_p_sampling_by_class is None
            else self._cached_voxel_ligand_p_sampling_by_class.to(device=self.device).clone()
        )
        # torch.Tensor, (K,), 本轮计算得到的 best-F1 分数候选; 无旧缓存时先填 NaN
        new_best_f1 = self._cached_voxel_ligand_best_f1_before_refine_by_class.clone().to(device=self.device)
        all_hist = pos_hist + neg_hist
        threshold_grid = self._voxel_ligand_threshold_grid.to(device=self.device)
        if pos_hist.sum() <= 0:
            self._reset_voxel_ligand_threshold_histograms()
            raise RuntimeError("validation loop 内所有 candidate class 均无 voxel ligand GT 正例。")
        metrics: dict[str, torch.Tensor] = {}
        updated_class_positions: list[int] = []
        for class_pos, class_id in enumerate(self._sparse_candidate_class_ids):
            if all_hist[class_pos].sum() <= 0 or pos_hist[class_pos].sum() <= 0:
                continue
            best_bin = int(torch.argmax(f1_by_threshold[class_pos]).item())
            n_best_total = tp_at_threshold[class_pos, best_bin] + fp_at_threshold[class_pos, best_bin]
            candidate_builder = self._unwrap_backbone().candidate_set_builder
            n_sampling_total = int(torch.ceil(n_best_total * float(candidate_builder.adaptive_expand_factor[class_pos])).item())
            cumulative_all = torch.cumsum(all_hist[class_pos].flip(0), dim=0)
            if n_sampling_total <= 0:
                sampling_bin = best_bin
            elif cumulative_all[-1] < n_sampling_total:
                sampling_bin = 0
            else:
                reversed_pos = int((cumulative_all >= n_sampling_total).nonzero(as_tuple=False)[0].item())
                sampling_bin = int(all_hist.shape[1] - 1 - reversed_pos)
            new_p_best[class_pos] = threshold_grid[best_bin]
            new_p_sampling[class_pos] = threshold_grid[sampling_bin]
            new_best_f1[class_pos] = f1_by_threshold[class_pos, best_bin]
            updated_class_positions.append(class_pos)
            class_name = self._class_names[class_id] if class_id < len(self._class_names) else f"class_{class_id}"
            metrics[f"val/voxel_ligand_p_best_by_class_{class_name}"] = new_p_best[class_pos]
            metrics[f"val/voxel_ligand_p_sampling_by_class_{class_name}"] = new_p_sampling[class_pos]
            metrics[f"val/voxel_ligand_best_f1_before_refine_{class_name}"] = new_best_f1[class_pos]
        if len(updated_class_positions) > 0:
            updated_index = torch.as_tensor(updated_class_positions, device=self.device, dtype=torch.long)
            metrics["val/voxel_ligand_best_f1_before_refine_macro"] = new_best_f1.index_select(0, updated_index).mean()
        self._cached_voxel_ligand_p_best_by_class = new_p_best.detach().cpu()
        self._cached_voxel_ligand_p_sampling_by_class = new_p_sampling.detach().cpu()
        self._cached_voxel_ligand_best_f1_before_refine_by_class = new_best_f1.detach().cpu()
        self._sync_sparse_candidate_runtime_to_backbone()
        if trainer is not None:
            for metric_name, metric_value in metrics.items():
                self.log(
                    metric_name,
                    metric_value.to(device=self.device),
                    prog_bar=(metric_name == self.hparams.monitor_metric),
                    on_step=False,
                    on_epoch=True,
                    sync_dist=True,
                )
        if len(metrics) > 0 and (trainer is None or bool(getattr(trainer, "is_global_zero", True))):
            print(
                "[CandidateThreshold] "
                f"p_best_by_class={self._cached_voxel_ligand_p_best_by_class.tolist()}, "
                f"p_sampling_by_class={self._cached_voxel_ligand_p_sampling_by_class.tolist()}, "
                f"best_f1_by_class={self._cached_voxel_ligand_best_f1_before_refine_by_class.tolist()}"
            )
        self._reset_voxel_ligand_threshold_histograms()
        return metrics

    def _compute_log_reset_ligand_sparse_refine_metrics(self) -> dict[str, torch.Tensor]:
        """
        计算、记录并重置 sparse refine best-F1、candidate recall 与 C/P 数量指标。

        输出:
            - metrics: dict[str, torch.Tensor], 当前 validation end 现场计算出的 sparse refine 指标
        """
        if self.ligand_sparse_refine_loss is None:
            return {}
        try:
            trainer = self.trainer
        except RuntimeError:
            trainer = None
        if trainer is not None and (bool(getattr(trainer, "sanity_checking", False)) or self._is_tuning_trainer(trainer)):
            self._reset_ligand_sparse_refine_metric_buffers()
            return {}
        # torch.Tensor, (K,num_bins), 当前 rank 本地 C 内正例 histogram
        pos_hist = self._ligand_sparse_refine_pos_hist_by_class.detach().to(device=self.device, dtype=torch.float32)
        # torch.Tensor, (K,num_bins), 当前 rank 本地 C 内负例 histogram
        neg_hist = self._ligand_sparse_refine_neg_hist_by_class.detach().to(device=self.device, dtype=torch.float32)
        # torch.Tensor, (K,), 当前 rank 本地 dense GT 正例数
        total_gt_pos = self._ligand_sparse_refine_total_gt_pos_by_class.detach().to(device=self.device, dtype=torch.float32)
        # torch.Tensor, (3,), 当前 rank 本地 C/P/BOX 数量
        count_totals = torch.stack(
            (
                self._ligand_sparse_refine_candidate_count_total.detach(),
                self._ligand_sparse_refine_anchor_count_total.detach(),
                self._ligand_sparse_refine_box_count_total.detach(),
            )
        ).to(device=self.device, dtype=torch.float32)
        if trainer is not None and int(getattr(trainer, "world_size", 1)) > 1:
            pos_hist = self.all_gather(pos_hist).sum(dim=0)
            neg_hist = self.all_gather(neg_hist).sum(dim=0)
            total_gt_pos = self.all_gather(total_gt_pos).sum(dim=0)
            count_totals = self.all_gather(count_totals).sum(dim=0)
        if total_gt_pos.sum() <= 0:
            self._reset_ligand_sparse_refine_metric_buffers()
            raise RuntimeError("validation loop 内所有 candidate class 均无 sparse refine GT 正例。")
        tp_at_threshold = torch.cumsum(pos_hist.flip(-1), dim=-1).flip(-1)
        fp_at_threshold = torch.cumsum(neg_hist.flip(-1), dim=-1).flip(-1)
        fn_at_threshold = total_gt_pos[:, None] - tp_at_threshold
        denominator = 2.0 * tp_at_threshold + fp_at_threshold + fn_at_threshold
        f1_by_threshold = torch.where(
            denominator > 0.0,
            2.0 * tp_at_threshold / denominator,
            torch.zeros_like(denominator),
        )
        metrics: dict[str, torch.Tensor] = {}
        best_f1_values: list[torch.Tensor] = []
        recall_values: list[torch.Tensor] = []
        for class_pos, class_id in enumerate(self._sparse_candidate_class_ids):
            if total_gt_pos[class_pos] <= 0:
                continue
            best_f1 = torch.max(f1_by_threshold[class_pos])
            candidate_recall = pos_hist[class_pos].sum() / total_gt_pos[class_pos]
            class_name = self._class_names[class_id] if class_id < len(self._class_names) else f"class_{class_id}"
            metrics[f"val/ligand_sparse_refine_best_f1_{class_name}"] = best_f1
            metrics[f"val/candidate_recall_{class_name}"] = candidate_recall
            best_f1_values.append(best_f1)
            recall_values.append(candidate_recall)
        metrics["val/ligand_sparse_refine_best_f1_macro"] = torch.stack(best_f1_values).mean()
        metrics["val/candidate_recall_macro"] = torch.stack(recall_values).mean()
        if count_totals[2] > 0:
            metrics["val/num_candidate_voxels"] = count_totals[0] / count_totals[2]
            metrics["val/num_anchor_points"] = count_totals[1] / count_totals[2]
        if trainer is not None:
            for metric_name, metric_value in metrics.items():
                self.log(
                    metric_name,
                    metric_value.to(device=self.device),
                    prog_bar=(metric_name == self.hparams.monitor_metric),
                    on_step=False,
                    on_epoch=True,
                    sync_dist=True,
                )
        self._reset_ligand_sparse_refine_metric_buffers()
        return metrics

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

    def on_validation_epoch_start(self) -> None:
        """
        验证 epoch 开始时清空 candidate threshold histogram 与 sparse refine metric buffers。

        输出:
            - None, 原地清空本轮 validation metric 状态
        """
        self._reset_voxel_ligand_threshold_histograms()
        self._reset_ligand_sparse_refine_metric_buffers()

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
            metric_value = self._compute_log_reset_metric_safe(
                self.val_atom_pr_auc,
                "val/atom_pr_auc",
            )
            if metric_value is not None:
                computed_metrics["val/atom_pr_auc"] = metric_value
        if hasattr(self, "val_voxel_aux_pr_auc"):
            metric_value = self._compute_log_reset_metric_safe(
                self.val_voxel_aux_pr_auc,
                "val/voxel_aux_pr_auc",
            )
            if metric_value is not None:
                computed_metrics["val/voxel_aux_pr_auc"] = metric_value
        if hasattr(self, "val_voxel_ligand_pr_auc"):
            metric_value = self._compute_log_reset_metric_safe(
                self.val_voxel_ligand_pr_auc,
                "val/voxel_ligand_pr_auc",
            )
            if metric_value is not None:
                computed_metrics["val/voxel_ligand_pr_auc"] = metric_value
        computed_metrics.update(self._compute_log_reset_multiclass_metrics())
        computed_metrics.update(self._compute_log_update_voxel_ligand_best_f1_thresholds())
        computed_metrics.update(self._compute_log_reset_ligand_sparse_refine_metrics())
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
        try:
            trainer = self.trainer
        except RuntimeError:
            trainer = None
        if self._sparse_candidate_class_ids is not None:
            checkpoint["voxel_ligand_candidate_class_ids"] = self._sparse_candidate_class_ids
            if trainer is not None and self._is_tuning_trainer(trainer):
                return
            if self._cached_voxel_ligand_p_best_by_class is None or self._cached_voxel_ligand_p_sampling_by_class is None:
                return
            # torch.Tensor, (K,), 待写入 checkpoint 的 best-F1 threshold cache
            p_best_tensor = self._normalize_candidate_checkpoint_tensor(
                self._cached_voxel_ligand_p_best_by_class,
                "voxel_ligand_p_best_by_class",
            )
            # torch.Tensor, (K,), 待写入 checkpoint 的 sampling threshold cache
            p_sampling_tensor = self._normalize_candidate_checkpoint_tensor(
                self._cached_voxel_ligand_p_sampling_by_class,
                "voxel_ligand_p_sampling_by_class",
            )
            if not (bool(torch.isfinite(p_best_tensor).all()) and bool(torch.isfinite(p_sampling_tensor).all())):
                return
            checkpoint["voxel_ligand_p_best_by_class"] = p_best_tensor
            checkpoint["voxel_ligand_p_sampling_by_class"] = p_sampling_tensor
            if self._cached_voxel_ligand_best_f1_before_refine_by_class is not None:
                # torch.Tensor, (K,), 待写入 checkpoint 的 best-F1 分数 cache
                best_f1_tensor = self._normalize_candidate_checkpoint_tensor(
                    self._cached_voxel_ligand_best_f1_before_refine_by_class,
                    "voxel_ligand_best_f1_before_refine_by_class",
                )
                if bool(torch.isfinite(best_f1_tensor).all()):
                    checkpoint["voxel_ligand_best_f1_before_refine_by_class"] = best_f1_tensor

    def on_load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        """
        恢复手动管理的 warmup_plateau plateau 状态。

        输入参数:
            - checkpoint: dict[str, Any], Lightning 从磁盘读取的 checkpoint 字典

        输出:
            - None, 立即恢复或暂存 plateau scheduler 状态
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
                self._cached_voxel_ligand_p_best_by_class = self._load_candidate_checkpoint_finite_tensor(
                    checkpoint["voxel_ligand_p_best_by_class"],
                    "voxel_ligand_p_best_by_class",
                )
                self._cached_voxel_ligand_p_sampling_by_class = self._load_candidate_checkpoint_finite_tensor(
                    checkpoint["voxel_ligand_p_sampling_by_class"],
                    "voxel_ligand_p_sampling_by_class",
                )
            if "voxel_ligand_best_f1_before_refine_by_class" in checkpoint:
                self._cached_voxel_ligand_best_f1_before_refine_by_class = self._load_candidate_checkpoint_finite_tensor(
                    checkpoint["voxel_ligand_best_f1_before_refine_by_class"],
                    "voxel_ligand_best_f1_before_refine_by_class",
                )
            self._sync_sparse_candidate_runtime_to_backbone()
        if "warmup_plateau_reduce_on_plateau_state" in checkpoint:
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
        warmup_steps: int,
    ) -> torch.optim.lr_scheduler.LRScheduler:
        """
        构建仅包含 step 级线性 warmup 的 scheduler。

        输入参数:
            - optimizer: torch.optim.Optimizer, 被调度的优化器
            - sched_cfg: Any, Hydra scheduler 配置, 需要包含 total_steps/warmup_steps/warmup_ratio/warmup_start_factor
            - warmup_steps: int, 已解析出的 warmup step 数

        输出:
            - scheduler: torch.optim.lr_scheduler.LRScheduler, Lightning step 级调度器
        """
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
        warmup_steps: int,
    ) -> WarmupThenReduceLROnPlateau:
        """
        构建 step 级 warmup + validation 级 plateau 的组合 scheduler。

        输入参数:
            - optimizer: torch.optim.Optimizer, 被调度的优化器
            - sched_cfg: Any, Hydra scheduler 配置, 需要显式包含 warmup 与 plateau 全部字段
            - warmup_steps: int, 已解析出的 warmup step 数

        输出:
            - scheduler: WarmupThenReduceLROnPlateau, 组合调度器对象
        """
        scheduler = WarmupThenReduceLROnPlateau(
            optimizer,
            warmup_steps=warmup_steps,
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
            self._candidate_warmup_steps = 0
            self._sync_sparse_candidate_runtime_to_backbone()
            return {"optimizer": optimizer}

        if isinstance(sched_cfg, functools.partial):
            self._candidate_warmup_steps = 0
            self._sync_sparse_candidate_runtime_to_backbone()
            scheduler = sched_cfg(optimizer=optimizer)
        elif callable(sched_cfg) and not hasattr(sched_cfg, "keys"):
            self._candidate_warmup_steps = 0
            self._sync_sparse_candidate_runtime_to_backbone()
            scheduler = sched_cfg(optimizer=optimizer)
        elif hasattr(sched_cfg, "get") and sched_cfg.get("name", None) == "warmup_plateau":
            self._candidate_warmup_steps = self._resolve_warmup_steps(sched_cfg)
            self._sync_sparse_candidate_runtime_to_backbone()
            self._warmup_plateau_scheduler = self._build_warmup_plateau_scheduler(
                optimizer=optimizer,
                sched_cfg=sched_cfg,
                warmup_steps=self._candidate_warmup_steps,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": self._warmup_plateau_scheduler.lightning_warmup_config(),
            }
        elif hasattr(sched_cfg, "get") and sched_cfg.get("name", None) == "warmup_only":
            self._candidate_warmup_steps = self._resolve_warmup_steps(sched_cfg)
            self._sync_sparse_candidate_runtime_to_backbone()
            scheduler = self._build_warmup_only_scheduler(
                optimizer=optimizer,
                sched_cfg=sched_cfg,
                warmup_steps=self._candidate_warmup_steps,
            )
        else:
            self._candidate_warmup_steps = 0
            self._sync_sparse_candidate_runtime_to_backbone()
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
