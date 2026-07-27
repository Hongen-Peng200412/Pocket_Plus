from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from src.modules.losses import AdaptiveClassificationCompositeLoss
from src.wrappers.voxel_point_stage1 import VoxelPointStage1Wrapper
from src.wrappers.voxel_point_stage1_losses import compute_sparse_refine_loss_term


class _BackboneForSparseLoss(nn.Module):
    """
    测试用 sparse refine loss backbone stub. 
    """

    def __init__(self, candidate_class_ids: tuple[int, ...]) -> None:
        super().__init__()
        self.candidate_set_builder = SimpleNamespace(candidate_class_ids=candidate_class_ids)

    def get_sparse_candidate_class_ids(self) -> tuple[int, ...]:
        """
        返回 candidate builder 类别 ID. 

        输出:
            - class_ids: tuple[int, ...], candidate 前景类别 ID
        """
        return tuple(self.candidate_set_builder.candidate_class_ids)

    def set_sparse_candidate_thresholds(self, p_best_by_class, p_sampling_by_class) -> None:
        """
        接收 wrapper 同步的 threshold cache. 
        """

    def set_sparse_candidate_runtime(self, global_step: int, candidate_warmup_steps: int, allow_warmup_fixed_topk: bool) -> None:
        """
        接收 wrapper 同步的 runtime 状态. 
        """

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        返回测试 batch 中预构造的输出. 
        """
        return batch["outputs"]


def _loss(num_classes: int) -> AdaptiveClassificationCompositeLoss:
    """
    构造测试用 AdaptiveClassificationCompositeLoss. 

    输入参数:
        - num_classes: int, 分类类别数

    输出:
        - loss: AdaptiveClassificationCompositeLoss, hard-label 复合损失
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


def _wrapper(num_classes: int, candidate_class_ids: tuple[int, ...]) -> VoxelPointStage1Wrapper:
    """
    构造启用 sparse refine loss 的 wrapper. 
    """
    return VoxelPointStage1Wrapper(
        backbone=_BackboneForSparseLoss(candidate_class_ids=candidate_class_ids),
        voxel_ligand_loss=_loss(num_classes),
        ligand_sparse_refine_loss=_loss(num_classes),
        ligand_sparse_refine_loss_weight=1.0,
        voxel_ligand_pr_auc_thresholds=4,
        class_names=["background", "foreground"] if num_classes == 2 else ["background", "metal", "small"],
    )


def test_target_from_ligand_dist_map_binary_accepts_4d_dist() -> None:
    """
    验证二分类 target helper 接受 (B,D,H,W) 距离图. 
    """
    loss = _loss(2)
    dist = torch.tensor([[[[0.5, 2.0]]]])

    target = loss.target_from_ligand_dist_map(dist, logit_dim=1, device=dist.device, dtype=dist.dtype)

    torch.testing.assert_close(target, torch.tensor([[[[1, 0]]]], dtype=torch.long))


def test_target_from_ligand_dist_map_binary_accepts_single_channel_5d_dist() -> None:
    """
    验证二分类 target helper 接受 (B,1,D,H,W) 距离图. 
    """
    loss = _loss(2)
    dist = torch.tensor([[[[[0.5, 2.0]]]]])

    target = loss.target_from_ligand_dist_map(dist, logit_dim=1, device=dist.device, dtype=dist.dtype)

    torch.testing.assert_close(target, torch.tensor([[[[1, 0]]]], dtype=torch.long))


def test_target_from_ligand_dist_map_binary_rejects_multiclass_dist() -> None:
    """
    验证二分类 target helper 拒绝多通道距离图. 
    """
    loss = _loss(2)
    dist = torch.zeros(1, 3, 1, 1, 1)

    with pytest.raises(ValueError, match="单通道"):
        loss.target_from_ligand_dist_map(dist, logit_dim=1, device=dist.device, dtype=dist.dtype)


def test_target_from_ligand_dist_map_multiclass_uses_nearest_foreground() -> None:
    """
    验证多分类 target helper 使用最近且过阈值的前景类别. 
    """
    loss = _loss(3)
    dist = torch.full((1, 3, 1, 1, 3), 5.0)
    dist[:, 1, 0, 0, 0] = 0.5
    dist[:, 1, 0, 0, 1] = 0.8
    dist[:, 2, 0, 0, 1] = 0.4

    target = loss.target_from_ligand_dist_map(dist, logit_dim=3, device=dist.device, dtype=dist.dtype)

    torch.testing.assert_close(target, torch.tensor([[[[1, 2, 0]]]], dtype=torch.long))


def test_sparse_refine_loss_uses_all_box_voxels_not_message_mask() -> None:
    """
    验证 sparse refine loss 使用全 BOX 体素监督, 不被 message mask 裁掉. 
    """
    wrapper = _wrapper(num_classes=2, candidate_class_ids=(1,))
    outputs = {
        "ligand_refine_logits_C": torch.tensor([[0.0], [0.0]], requires_grad=True),
        "candidate_batch_index": torch.tensor([0, 0]),
        "candidate_voxel_zyx": torch.tensor([[0, 0, 0], [0, 0, 1]]),
        "candidate_message_valid_mask": torch.tensor([False, False]),
    }
    batch = {
        "ligand_dist_map": torch.tensor([[[[0.5, 2.0]]]]),
    }

    supervision = wrapper._sample_ligand_refine_supervision(outputs, batch)
    term, _, _ = compute_sparse_refine_loss_term(
        logits_C=outputs["ligand_refine_logits_C"],
        target_C=supervision["ligand_refine_target_C"],
        valid_C=supervision["ligand_refine_valid_mask_C"],
        loss_module=wrapper.ligand_sparse_refine_loss,
        weight=1.0,
        effective_weight=torch.tensor(1.0),
    )
    loss = term.value

    assert loss is not None
    assert torch.isfinite(loss)
    assert int(outputs["ligand_refine_valid_mask_C"].sum().item()) == 2


def test_sparse_refine_loss_accepts_binary_C_logits() -> None:
    """
    验证 sparse refine loss 接受二分类 (sumC,1) logits. 
    """
    wrapper = _wrapper(num_classes=2, candidate_class_ids=(1,))
    outputs = {
        "ligand_refine_logits_C": torch.tensor([[0.0], [1.0]], requires_grad=True),
        "candidate_batch_index": torch.tensor([0, 0]),
        "candidate_voxel_zyx": torch.tensor([[0, 0, 0], [0, 0, 1]]),
    }
    batch = {
        "ligand_dist_map": torch.tensor([[[[0.5, 2.0]]]]),
    }

    supervision = wrapper._sample_ligand_refine_supervision(outputs, batch)
    term, _, _ = compute_sparse_refine_loss_term(
        logits_C=outputs["ligand_refine_logits_C"],
        target_C=supervision["ligand_refine_target_C"],
        valid_C=supervision["ligand_refine_valid_mask_C"],
        loss_module=wrapper.ligand_sparse_refine_loss,
        weight=1.0,
        effective_weight=torch.tensor(1.0),
    )
    loss = term.value

    assert loss is not None
    assert torch.isfinite(loss)


def test_sparse_refine_loss_accepts_empty_multiclass_C_logits() -> None:
    """
    验证 sparse refine loss 接受空三分类 C logits. 
    """
    wrapper = _wrapper(num_classes=3, candidate_class_ids=(1, 2))
    outputs = {
        "ligand_refine_logits_C": torch.zeros(0, 3, requires_grad=True),
        "candidate_batch_index": torch.empty(0, dtype=torch.long),
        "candidate_voxel_zyx": torch.empty(0, 3, dtype=torch.long),
    }
    dist = torch.full((1, 3, 1, 1, 1), 5.0)
    batch = {
        "ligand_dist_map": dist,
    }

    supervision = wrapper._sample_ligand_refine_supervision(outputs, batch)
    term, _, _ = compute_sparse_refine_loss_term(
        logits_C=outputs["ligand_refine_logits_C"],
        target_C=supervision["ligand_refine_target_C"],
        valid_C=supervision["ligand_refine_valid_mask_C"],
        loss_module=wrapper.ligand_sparse_refine_loss,
        weight=1.0,
        effective_weight=torch.tensor(1.0),
    )
    loss = term.value

    assert loss is not None
    torch.testing.assert_close(loss, torch.tensor(0.0))


def test_sparse_refine_loss_accepts_multiclass_C_logits() -> None:
    """
    验证 sparse refine loss 接受三分类 (sumC,3) logits. 
    """
    wrapper = _wrapper(num_classes=3, candidate_class_ids=(1, 2))
    outputs = {
        "ligand_refine_logits_C": torch.zeros(3, 3, requires_grad=True),
        "candidate_batch_index": torch.tensor([0, 0, 0]),
        "candidate_voxel_zyx": torch.tensor([[0, 0, 0], [0, 0, 1], [0, 0, 2]]),
    }
    dist = torch.full((1, 3, 1, 1, 3), 5.0)
    dist[:, 1, 0, 0, 0] = 0.5
    dist[:, 2, 0, 0, 1] = 0.5
    batch = {
        "ligand_dist_map": dist,
    }

    supervision = wrapper._sample_ligand_refine_supervision(outputs, batch)
    term, _, _ = compute_sparse_refine_loss_term(
        logits_C=outputs["ligand_refine_logits_C"],
        target_C=supervision["ligand_refine_target_C"],
        valid_C=supervision["ligand_refine_valid_mask_C"],
        loss_module=wrapper.ligand_sparse_refine_loss,
        weight=1.0,
        effective_weight=torch.tensor(1.0),
    )
    loss = term.value

    assert loss is not None
    assert torch.isfinite(loss)


def test_ligand_sparse_refine_loss_schedule_linear_warmup() -> None:
    """
    验证 sparse refine loss 独立 linear warmup 权重. 
    """
    wrapper = VoxelPointStage1Wrapper(
        backbone=_BackboneForSparseLoss(candidate_class_ids=(1,)),
        voxel_ligand_loss=_loss(2),
        ligand_sparse_refine_loss=_loss(2),
        ligand_sparse_refine_loss_weight=1.0,
        ligand_sparse_refine_loss_schedule={
            "mode": "linear_warmup",
            "start_weight": 0.0,
            "final_weight": 1.0,
            "warmup_steps": 10,
            "warmup_ratio": 0.15,
        },
        voxel_ligand_pr_auc_thresholds=4,
        class_names=["background", "foreground"],
    )

    wrapper.trainer = SimpleNamespace(global_step=0)
    torch.testing.assert_close(wrapper._compute_sparse_refine_loss_effective_weight(), torch.tensor(0.0))
    wrapper.trainer = SimpleNamespace(global_step=5)
    torch.testing.assert_close(wrapper._compute_sparse_refine_loss_effective_weight(), torch.tensor(0.5))
    wrapper.trainer = SimpleNamespace(global_step=10)
    torch.testing.assert_close(wrapper._compute_sparse_refine_loss_effective_weight(), torch.tensor(1.0))


def test_ligand_sparse_refine_loss_schedule_uses_warmup_ratio() -> None:
    """
    验证 sparse refine loss warmup_steps 为 null 时按 trainer 总 step 数和 warmup_ratio 解析. 
    """
    wrapper = VoxelPointStage1Wrapper(
        backbone=_BackboneForSparseLoss(candidate_class_ids=(1,)),
        voxel_ligand_loss=_loss(2),
        ligand_sparse_refine_loss=_loss(2),
        ligand_sparse_refine_loss_weight=1.0,
        ligand_sparse_refine_loss_schedule={
            "mode": "linear_warmup",
            "start_weight": 0.0,
            "final_weight": 1.0,
            "warmup_steps": None,
            "warmup_ratio": 0.25,
        },
        voxel_ligand_pr_auc_thresholds=4,
        class_names=["background", "foreground"],
    )

    wrapper.trainer = SimpleNamespace(global_step=5, estimated_stepping_batches=20)

    torch.testing.assert_close(wrapper._compute_sparse_refine_loss_effective_weight(), torch.tensor(1.0))
