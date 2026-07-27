from __future__ import annotations

import pytest
import torch
from torch import nn

from src.model.sparse_refine.candidate_set import SparseCandidateSetBuilder
from src.modules.losses import AdaptiveClassificationCompositeLoss
from src.wrappers.voxel_point_stage1 import VoxelPointStage1Wrapper


class _BackboneForThresholds(nn.Module):
    """
    测试用 candidate threshold backbone stub. 
    """

    def __init__(self, candidate_class_ids: tuple[int, ...] = (1, 2)) -> None:
        super().__init__()
        self.candidate_set_builder = SparseCandidateSetBuilder(
            candidate_class_ids=candidate_class_ids,
            warmup_topc_per_class=[1 for _ in candidate_class_ids],
            adaptive_expand_factor=[2.0 for _ in candidate_class_ids],
            max_candidate_voxels_per_class=[10 for _ in candidate_class_ids],
            min_candidate_voxels_per_class=[0 for _ in candidate_class_ids],
            selection_mode="adaptive_threshold",
        )
        self.synced_thresholds: tuple[torch.Tensor | None, torch.Tensor | None] | None = None
        self.synced_runtime: tuple[int, int, bool] | None = None

    def get_sparse_candidate_class_ids(self) -> tuple[int, ...]:
        """
        返回 candidate builder 类别 ID. 

        输出:
            - class_ids: tuple[int, ...], candidate 前景类别 ID
        """
        return tuple(self.candidate_set_builder.candidate_class_ids)

    def set_sparse_candidate_thresholds(
        self,
        p_best_by_class: torch.Tensor | None,
        p_sampling_by_class: torch.Tensor | None,
    ) -> None:
        """
        记录 wrapper 同步的阈值缓存. 
        """
        self.synced_thresholds = (p_best_by_class, p_sampling_by_class)

    def set_sparse_candidate_runtime(
        self,
        global_step: int,
        candidate_warmup_steps: int,
        allow_warmup_fixed_topk: bool,
    ) -> None:
        """
        记录 wrapper 同步的 runtime 状态. 
        """
        self.synced_runtime = (int(global_step), int(candidate_warmup_steps), bool(allow_warmup_fixed_topk))

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        返回测试直接提供的 logits. 
        """
        return {
            "voxel_logits_ligand": batch["voxel_logits_ligand"],
            "voxel_logits_aux": batch["voxel_logits_ligand"],
            "recycle_passes_used": 1,
        }


def _make_loss(num_classes: int, hard_label_threshold: float | None) -> AdaptiveClassificationCompositeLoss:
    """
    构造测试用 voxel ligand loss. 

    输入参数:
        - num_classes: int, 分类类别数
        - hard_label_threshold: float | None, ligand 距离硬标签阈值

    输出:
        - loss: AdaptiveClassificationCompositeLoss, hard-label 复合损失
    """
    return AdaptiveClassificationCompositeLoss(
        num_classes=num_classes,
        hard_label_threshold=hard_label_threshold,
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


def _make_wrapper(
    hard_label_threshold: float | None = 1.7,
    candidate_class_ids: tuple[int, ...] = (1, 2),
    initial_p_best_by_class: list[float] | tuple[float, ...] | None = None,
    initial_p_sampling_by_class: list[float] | tuple[float, ...] | None = None,
) -> VoxelPointStage1Wrapper:
    """
    构造带 candidate builder 的 wrapper. 

    输入参数:
        - hard_label_threshold: float | None, ligand 距离硬标签阈值
        - candidate_class_ids: tuple[int, ...], candidate 前景类别 ID
        - initial_p_best_by_class: list[float] | tuple[float, ...] | None, (K,), 显式 best-F1 初始阈值
        - initial_p_sampling_by_class: list[float] | tuple[float, ...] | None, (K,), 显式 sampling 初始阈值

    输出:
        - wrapper: VoxelPointStage1Wrapper, 测试用 wrapper
    """
    wrapper = VoxelPointStage1Wrapper(
        backbone=_BackboneForThresholds(candidate_class_ids=candidate_class_ids),
        voxel_ligand_loss=_make_loss(num_classes=3, hard_label_threshold=hard_label_threshold),
        voxel_ligand_pr_auc_thresholds=4,
        initial_p_best_by_class=initial_p_best_by_class,
        initial_p_sampling_by_class=initial_p_sampling_by_class,
        class_names=["background", "metal", "small"],
        validation_diagnostics={"enabled": True},
    )
    return wrapper


def test_threshold_cache_starts_none_without_initial() -> None:
    """
    验证无显式 initial 时 wrapper 不制造 NaN threshold cache. 
    """
    wrapper = _make_wrapper()

    assert wrapper._cached_voxel_ligand_p_best_by_class is None
    assert wrapper._cached_voxel_ligand_p_sampling_by_class is None
    assert wrapper._unwrap_backbone().synced_thresholds == (None, None)


def test_initial_threshold_cache_syncs_to_backbone() -> None:
    """
    验证显式 initial threshold cache 会同步到 backbone. 
    """
    wrapper = _make_wrapper(
        initial_p_best_by_class=[0.25, 0.5],
        initial_p_sampling_by_class=[0.1, 0.2],
    )

    p_best, p_sampling = wrapper._unwrap_backbone().synced_thresholds
    torch.testing.assert_close(p_best, torch.tensor([0.25, 0.5]))
    torch.testing.assert_close(p_sampling, torch.tensor([0.1, 0.2]))


def test_voxel_ligand_best_f1_updates_p_best_by_class() -> None:
    """
    验证 validation histogram 能更新 p_best_by_class 与 p_sampling_by_class. 
    """
    wrapper = _make_wrapper()
    wrapper._allow_validation_cache_update = lambda: True
    logits = torch.zeros(1, 3, 1, 1, 4)
    logits[:, 1, 0, 0, 0] = 5.0
    logits[:, 1, 0, 0, 1] = 4.0
    logits[:, 2, 0, 0, 2] = 5.0
    logits[:, 2, 0, 0, 3] = 4.0
    ligand_dist_map = torch.full((1, 3, 1, 1, 4), 5.0)
    ligand_dist_map[:, 1, 0, 0, 0] = 0.5
    ligand_dist_map[:, 2, 0, 0, 2] = 0.5
    valid_mask = torch.ones(1, 1, 1, 4, dtype=torch.bool)

    target = wrapper.voxel_ligand_loss.target_from_ligand_dist_map(
        ligand_dist_map=ligand_dist_map,
        logit_dim=3,
        device=logits.device,
        dtype=logits.dtype,
    )
    wrapper.cpc_diagnostics.update_uncapped_best(
        logits=logits,
        target=target,
        valid_mask=valid_mask,
        allow_cache_update=True,
    )
    wrapper.cpc_diagnostics.update_uncapped_sampling(
        logits=logits,
        target=target,
        valid_mask=valid_mask,
        candidate_outputs={
            "candidate_p_sampling_by_class": torch.tensor([[0.25, 0.25]]),
            "candidate_target_counts_by_class": torch.tensor([[2, 2]]),
        },
        selection_mode="adaptive_threshold",
        use_fixed_warmup=False,
    )
    metrics = wrapper.cpc_diagnostics.compute_payload(sync_fn=lambda tensor: tensor).scalars
    wrapper._update_candidate_threshold_cache_from_payload(metrics)

    assert "val_uncapped_best/global/best_F1_metal" in metrics
    assert "val_uncapped_best/global/best_F1_small" in metrics
    assert torch.isfinite(wrapper._cached_voxel_ligand_p_best_by_class).all()
    torch.testing.assert_close(wrapper._cached_voxel_ligand_p_sampling_by_class, torch.tensor([0.0, 0.0]))
    wrapper._sync_sparse_candidate_runtime_to_backbone()
    p_best, p_sampling = wrapper._unwrap_backbone().synced_thresholds
    torch.testing.assert_close(p_best, wrapper._cached_voxel_ligand_p_best_by_class)
    torch.testing.assert_close(p_sampling, torch.tensor([0.0, 0.0]))


def test_threshold_cache_uses_best_panel_supervised_sampling_threshold() -> None:
    """
    验证 wrapper cache 写回时 p_sampling 来自 dense best 面板的 supervised calibration 阈值. 
    """
    wrapper = _make_wrapper(candidate_class_ids=(1,))
    wrapper._allow_validation_cache_update = lambda: True
    payload = {
        "val_uncapped_best/global/p_best_metal": torch.tensor(0.75),
        "val_uncapped_best/global/p_sampling_metal": torch.tensor(0.5),
        "val_uncapped_best/global/best_F1_metal": torch.tensor(0.8),
        "val_uncapped_sampling/global/p_sampling_p50_metal": torch.tensor(0.25),
    }

    wrapper._update_candidate_threshold_cache_from_payload(payload)

    torch.testing.assert_close(wrapper._cached_voxel_ligand_p_best_by_class, torch.tensor([0.75]))
    torch.testing.assert_close(wrapper._cached_voxel_ligand_p_sampling_by_class, torch.tensor([0.5]))


def test_threshold_cache_fails_fast_when_best_threshold_keys_missing() -> None:
    """
    验证无正例 best 面板缺少阈值键时 wrapper 保持旧 cache. 
    """
    wrapper = _make_wrapper(
        candidate_class_ids=(1,),
        initial_p_best_by_class=[0.75],
        initial_p_sampling_by_class=[0.5],
    )
    wrapper._allow_validation_cache_update = lambda: True
    payload = {
        "val_uncapped_best/global/best_F1_metal": torch.tensor(float("nan")),
    }

    with pytest.raises(RuntimeError, match="candidate class"):
        wrapper._update_candidate_threshold_cache_from_payload(payload)


def test_threshold_stats_reject_hard_label_threshold_none() -> None:
    """
    验证 loss helper 在 hard_label_threshold=None 时 fail-fast. 
    """
    wrapper = _make_wrapper(hard_label_threshold=None)
    logits = torch.zeros(1, 3, 1, 1, 1)
    ligand_dist_map = torch.zeros(1, 3, 1, 1, 1)

    with pytest.raises(ValueError, match="hard_label_threshold"):
        wrapper.voxel_ligand_loss.target_from_ligand_dist_map(
            ligand_dist_map=ligand_dist_map,
            logit_dim=3,
            device=logits.device,
            dtype=logits.dtype,
        )


def test_threshold_class_ids_reject_sparse_refine_loss_mismatch() -> None:
    """
    验证 candidate_class_ids 与 sparse refine loss num_classes 不一致时 fail-fast. 
    """
    with pytest.raises(ValueError, match="candidate_class_ids"):
        VoxelPointStage1Wrapper(
            backbone=_BackboneForThresholds(candidate_class_ids=(1, 2)),
            voxel_ligand_loss=_make_loss(num_classes=3, hard_label_threshold=1.7),
            ligand_sparse_refine_loss=_make_loss(num_classes=2, hard_label_threshold=1.7),
            class_names=["background", "metal", "small"],
        )


def test_threshold_histograms_reset_on_validation_epoch_start() -> None:
    """
    验证 validation start 会清空 diagnostics histogram. 
    """
    wrapper = _make_wrapper()
    wrapper.cpc_diagnostics.uncapped_best_pos_hist += 1
    wrapper.cpc_diagnostics.uncapped_best_neg_hist += 1

    wrapper.on_validation_epoch_start()

    assert int(wrapper.cpc_diagnostics.uncapped_best_pos_hist.sum().item()) == 0
    assert int(wrapper.cpc_diagnostics.uncapped_best_neg_hist.sum().item()) == 0


def test_threshold_cache_skips_checkpoint_fields_until_thresholds_are_finite() -> None:
    """
    验证 threshold cache 非成对 finite 时不写入 checkpoint threshold 字段. 
    """
    wrapper = _make_wrapper()
    checkpoint: dict[str, object] = {}

    wrapper.on_save_checkpoint(checkpoint)

    assert checkpoint["voxel_ligand_candidate_class_ids"] == (1, 2)
    assert "voxel_ligand_p_best_by_class" not in checkpoint
    assert "voxel_ligand_p_sampling_by_class" not in checkpoint


def test_threshold_cache_saved_and_loaded_from_checkpoint() -> None:
    """
    验证 checkpoint metadata 保存并恢复 candidate threshold cache. 
    """
    wrapper = _make_wrapper()
    wrapper._cached_voxel_ligand_p_best_by_class = torch.tensor([0.25, 0.5])
    wrapper._cached_voxel_ligand_p_sampling_by_class = torch.tensor([0.0, 0.25])
    wrapper._cached_voxel_ligand_best_f1_before_refine_by_class = torch.tensor([0.8, 0.6])
    checkpoint: dict[str, object] = {}

    wrapper.on_save_checkpoint(checkpoint)
    restored = _make_wrapper()
    restored.on_load_checkpoint(checkpoint)

    torch.testing.assert_close(restored._cached_voxel_ligand_p_best_by_class, torch.tensor([0.25, 0.5]))
    torch.testing.assert_close(restored._cached_voxel_ligand_p_sampling_by_class, torch.tensor([0.0, 0.25]))
    torch.testing.assert_close(restored._cached_voxel_ligand_best_f1_before_refine_by_class, torch.tensor([0.8, 0.6]))


def test_threshold_cache_load_rejects_nan() -> None:
    """
    验证 checkpoint threshold cache 含 NaN 时 fail-fast. 
    """
    wrapper = _make_wrapper()
    checkpoint = {
        "voxel_ligand_candidate_class_ids": (1, 2),
        "voxel_ligand_p_best_by_class": torch.tensor([0.25, float("nan")]),
        "voxel_ligand_p_sampling_by_class": torch.tensor([0.1, 0.2]),
    }

    with pytest.raises(ValueError, match="NaN/Inf"):
        wrapper.on_load_checkpoint(checkpoint)


def test_threshold_cache_load_rejects_best_f1_nan_or_inf() -> None:
    """
    验证 checkpoint best_f1_by_class 含 NaN/Inf 时 fail-fast. 
    """
    wrapper = _make_wrapper()
    checkpoint = {
        "voxel_ligand_candidate_class_ids": (1, 2),
        "voxel_ligand_best_f1_before_refine_by_class": torch.tensor([0.8, float("inf")]),
    }

    with pytest.raises(ValueError, match="NaN/Inf"):
        wrapper.on_load_checkpoint(checkpoint)


def test_threshold_cache_load_rejects_length_mismatch() -> None:
    """
    验证 checkpoint threshold cache 长度不匹配时 fail-fast. 
    """
    wrapper = _make_wrapper()
    checkpoint = {
        "voxel_ligand_candidate_class_ids": (1, 2),
        "voxel_ligand_p_best_by_class": torch.tensor([0.25, 0.5]),
        "voxel_ligand_p_sampling_by_class": torch.tensor([0.25]),
    }

    with pytest.raises(ValueError, match="长度"):
        wrapper.on_load_checkpoint(checkpoint)


def test_threshold_cache_load_rejects_unpaired_thresholds() -> None:
    """
    验证 checkpoint threshold cache 不是成对出现时 fail-fast. 
    """
    wrapper = _make_wrapper()
    checkpoint = {
        "voxel_ligand_candidate_class_ids": (1, 2),
        "voxel_ligand_p_best_by_class": torch.tensor([0.25, 0.5]),
    }

    with pytest.raises(ValueError, match="成对"):
        wrapper.on_load_checkpoint(checkpoint)


def test_threshold_cache_load_rejects_candidate_class_id_mismatch() -> None:
    """
    验证 checkpoint class ids 与当前配置不一致时 fail-fast. 
    """
    wrapper = _make_wrapper(candidate_class_ids=(1,))
    checkpoint = {"voxel_ligand_candidate_class_ids": (2,)}

    with pytest.raises(ValueError, match="candidate_class_ids"):
        wrapper.on_load_checkpoint(checkpoint)
