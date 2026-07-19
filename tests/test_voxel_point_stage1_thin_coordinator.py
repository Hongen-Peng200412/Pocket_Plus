from __future__ import annotations

import inspect
from types import SimpleNamespace

import torch
from torch import nn

from src.modules.losses import AdaptiveClassificationCompositeLoss
from src.wrappers.voxel_point_stage1 import VoxelPointStage1Wrapper
from src.wrappers.voxel_point_stage1_diagnostics import CpcValidationDiagnostics
from src.wrappers.voxel_point_stage1_metrics import ValidationMetricManager


class _ThinBackbone(nn.Module):
    """
    测试用 Stage1 backbone stub。

    输入参数:
        - candidate_class_ids: tuple[int, ...], (K,), sparse refine 候选前景类别 ID

    前向输入:
        - batch: dict[str, torch.Tensor], 测试直接传入的 batch 字典

    前向输出:
        - outputs: dict[str, torch.Tensor], 测试直接传入的 outputs 字典
    """

    def __init__(self, candidate_class_ids: tuple[int, ...]) -> None:
        super().__init__()
        self.candidate_set_builder = SimpleNamespace(
            candidate_class_ids=candidate_class_ids,
            adaptive_expand_factor=tuple(1.0 for _ in candidate_class_ids),
            max_candidate_voxels_per_class=tuple(4 for _ in candidate_class_ids),
            selection_mode="adaptive_threshold",
        )
        # tuple[torch.Tensor | None, torch.Tensor | None] | None, wrapper 同步的阈值缓存
        self.synced_thresholds = None
        # tuple[int, int, bool] | None, wrapper 同步的 candidate runtime
        self.synced_runtime = None

    def get_sparse_candidate_class_ids(self) -> tuple[int, ...]:
        """
        返回 sparse candidate builder 配置的候选类别 ID。

        输出:
            - class_ids: tuple[int, ...], (K,), 候选前景类别 ID
        """
        return tuple(self.candidate_set_builder.candidate_class_ids)

    def set_sparse_candidate_thresholds(
        self,
        p_best_by_class: torch.Tensor | None,
        p_sampling_by_class: torch.Tensor | None,
    ) -> None:
        """
        记录 wrapper 同步的阈值缓存。

        输入参数:
            - p_best_by_class: torch.Tensor | None, (K,), best-F1 阈值缓存
            - p_sampling_by_class: torch.Tensor | None, (K,), sampling 阈值缓存

        输出:
            - None, 原地记录 synced_thresholds
        """
        self.synced_thresholds = (p_best_by_class, p_sampling_by_class)

    def set_sparse_candidate_runtime(
        self,
        global_step: int,
        candidate_warmup_steps: int,
        allow_warmup_fixed_topk: bool,
    ) -> None:
        """
        记录 wrapper 同步的 candidate runtime。

        输入参数:
            - global_step: int, 当前 optimizer step
            - candidate_warmup_steps: int, scheduler warmup step 数
            - allow_warmup_fixed_topk: bool, 是否允许 warmup fixed topk

        输出:
            - None, 原地记录 synced_runtime
        """
        self.synced_runtime = (int(global_step), int(candidate_warmup_steps), bool(allow_warmup_fixed_topk))

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        透传测试预构造输出。

        输入参数:
            - batch: dict[str, torch.Tensor], 包含 outputs 的测试 batch

        输出:
            - outputs: dict[str, torch.Tensor], backbone 输出字典
        """
        return batch["outputs"]

    def forward_voxel_probability(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """返回测试预置的 voxel-only logits。"""

        return batch["voxel_logits"]


def _loss(num_classes: int) -> AdaptiveClassificationCompositeLoss:
    """
    构造测试用 hard-label 复合损失。

    输入参数:
        - num_classes: int, 分类类别数

    输出:
        - loss: AdaptiveClassificationCompositeLoss, 测试用损失模块
    """
    return AdaptiveClassificationCompositeLoss(
        num_classes=num_classes,
        hard_label_threshold=1.7,
        focal_gamma=2.0,
        focal_alpha=[0.5 for _ in range(num_classes)],
        focal_eps=1.0e-6,
        tversky_alpha=0.5,
        tversky_beta=0.5,
        tversky_smooth=1.0,
        w_focal=0.7,
        w_tversky=0.3,
        w_mse=0.0,
    )


def _wrapper() -> VoxelPointStage1Wrapper:
    """
    构造启用 diagnostics 的 thin coordinator wrapper。

    输出:
        - wrapper: VoxelPointStage1Wrapper, 测试用 wrapper 实例
    """
    return VoxelPointStage1Wrapper(
        backbone=_ThinBackbone(candidate_class_ids=(1,)),
        voxel_ligand_loss=_loss(2),
        ligand_sparse_refine_loss=_loss(2),
        ligand_sparse_refine_loss_weight=1.0,
        voxel_ligand_pr_auc_thresholds=4,
        class_names=["background", "foreground"],
        validation_diagnostics={
            "enabled": True,
            "num_bins": 4,
            "write_local_artifacts": False,
            "log_wandb_curves": False,
            "wandb_curve_every_n_validation": 1,
            "output_subdir": "validation_diagnostics",
        },
    )


def test_wrapper_instantiates_metric_and_diagnostics_managers() -> None:
    """
    验证 wrapper 构造期只接入 helper manager, 不在本体注册旧 TorchMetrics 私有结构。
    """
    wrapper = _wrapper()

    assert isinstance(wrapper.val_metrics, ValidationMetricManager)
    assert isinstance(wrapper.cpc_diagnostics, CpcValidationDiagnostics)
    assert "voxel_ligand" in wrapper.val_metrics.branch_specs
    assert not hasattr(wrapper, "_val_metric_specs")
    assert not hasattr(wrapper, "val_voxel_ligand_pr_auc")


def test_wrapper_does_not_restore_old_metric_private_api() -> None:
    """
    验证 thin coordinator 不保留旧 wrapper 的大块 metric 私有 API。
    """
    forbidden_names = {
        "_update_voxel_ligand_best_f1_stats",
        "_compute_log_update_voxel_ligand_best_f1_thresholds",
        "_update_val_ligand_sparse_refine_metric",
        "_compute_log_reset_ligand_sparse_refine_metrics",
        "_compute_log_reset_multiclass_metrics",
        "_register_val_metric",
    }

    for name in forbidden_names:
        assert not hasattr(VoxelPointStage1Wrapper, name)


def test_validation_step_keeps_readable_time_order() -> None:
    """
    验证 validation_step 保持 coordinator 时间顺序, 而不是内联旧 metric 计算细节。
    """
    source = inspect.getsource(VoxelPointStage1Wrapper.validation_step)

    expected_order = [
        "_extract_batch",
        "_sync_sparse_candidate_runtime_to_backbone",
        "outputs = self(batch_dict)",
        "_compute_total_loss",
        "val_metrics.update_branch",
        "cpc_diagnostics.update_uncapped_best",
        "cpc_diagnostics.update_uncapped_sampling",
        "cpc_diagnostics.update_capped",
        "cpc_diagnostics.update_unrefined",
        "cpc_diagnostics.update_refined",
    ]
    cursor = -1
    for token in expected_order:
        next_pos = source.find(token)
        assert next_pos > cursor, token
        cursor = next_pos


def test_threshold_checkpoint_cache_remains_wrapper_runtime_state() -> None:
    """
    验证 candidate threshold cache 仍由 wrapper 保存和恢复, 不被 validation metric state 污染。
    """
    wrapper = _wrapper()
    wrapper._cached_voxel_ligand_p_best_by_class = torch.tensor([0.25])
    wrapper._cached_voxel_ligand_p_sampling_by_class = torch.tensor([0.125])
    wrapper._cached_voxel_ligand_best_f1_before_refine_by_class = torch.tensor([0.75])
    checkpoint: dict[str, object] = {}

    wrapper.on_save_checkpoint(checkpoint)
    restored = _wrapper()
    restored.on_load_checkpoint(checkpoint)

    torch.testing.assert_close(restored._cached_voxel_ligand_p_best_by_class, torch.tensor([0.25]))
    torch.testing.assert_close(restored._cached_voxel_ligand_p_sampling_by_class, torch.tensor([0.125]))
    torch.testing.assert_close(restored._cached_voxel_ligand_best_f1_before_refine_by_class, torch.tensor([0.75]))
    assert not any("val_metrics" in key for key in checkpoint)
    assert not any("cpc_diagnostics" in key for key in checkpoint)


def test_adaligand_direct_union_target_has_priority_over_legacy_distance_map() -> None:
    """验证 schema-v3 union target 直接进入 loss，不被 hardmask 或旧距离图替换。"""

    wrapper = _wrapper()
    direct = torch.tensor([[[[False, True], [True, False]]]])
    legacy = torch.full((1, 2, 2, 2), 99.0)

    target = wrapper._ligand_target_from_batch(
        {"ligand_area_target": direct, "ligand_dist_map": legacy},
        logit_dim=1,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    assert target.dtype == torch.long
    assert target.shape == (1, 2, 2, 2)
    assert torch.equal(target, direct[:, 0].long())


def test_wrapper_voxel_only_entry_is_a_thin_logit_delegation() -> None:
    """验证 wrapper 不重复模型、checkpoint、sigmoid 或阈值逻辑。"""

    wrapper = _wrapper()
    logits = torch.randn(1, 1, 2, 2, 2)

    returned = wrapper.forward_voxel_probability({"voxel_logits": logits})

    assert returned is logits
