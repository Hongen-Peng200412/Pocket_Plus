from __future__ import annotations

import torch

from src.wrappers.voxel_point_stage1_diagnostics import (
    CpcDiagnosticsConfig,
    CpcValidationDiagnostics,
)


def _config() -> CpcDiagnosticsConfig:
    """
    构造测试用 diagnostics 配置。
    """
    return CpcDiagnosticsConfig(
        enabled=True,
        num_bins=4,
        write_local_artifacts=False,
        log_wandb_curves=False,
        wandb_curve_every_n_validation=1,
        output_subdir="validation_diagnostics",
    )


def _diagnostics(num_classes: int, candidate_class_ids: tuple[int, ...]) -> CpcValidationDiagnostics:
    """
    构造 sparse refine diagnostics 测试对象。
    """
    class_names = ["background", "foreground"] if num_classes == 2 else ["background", "metal_ion", "small_molecule"]
    return CpcValidationDiagnostics(
        config=_config(),
        class_names=class_names,
        candidate_class_ids=candidate_class_ids,
        adaptive_expand_factor=[1.0 for _ in candidate_class_ids],
        max_candidate_voxels_per_class=[4 for _ in candidate_class_ids],
    )


def test_sparse_refine_score_f1_counts_missing_candidates_as_fn() -> None:
    """
    验证 val_score/refined_F1 的 FN 包含未进入 C 的 GT 正体素。
    """
    diagnostics = _diagnostics(num_classes=2, candidate_class_ids=(1,))
    candidate_outputs = {"candidate_logits": torch.tensor([[6.0]])}

    diagnostics.update_refined(
        refined_logits_C=torch.tensor([[6.0]]),
        candidate_outputs=candidate_outputs,
        target_C=torch.tensor([1]),
        valid_C=torch.tensor([True]),
        dense_num_gt=torch.tensor([[2]]),
    )
    payload = diagnostics.compute_payload(sync_fn=lambda tensor: tensor)

    torch.testing.assert_close(payload.scalars["val_refined/global/F1"], torch.tensor(1.0))
    torch.testing.assert_close(payload.scalars["val_score/global/refined_F1"], torch.tensor(2.0 / 3.0))


def test_candidate_recall_is_covered_positive_over_total_positive() -> None:
    """
    验证 val_capped recall 等于 C 覆盖正体素数除以 dense GT 正体素数。
    """
    diagnostics = _diagnostics(num_classes=2, candidate_class_ids=(1,))
    target = torch.tensor([[[[1, 1, 1, 1]]]])
    valid_mask = torch.ones(1, 1, 1, 4, dtype=torch.bool)
    candidate_outputs = {
        "candidate_batch_index": torch.tensor([0, 0, 0]),
        "candidate_voxel_zyx": torch.tensor([[0, 0, 0], [0, 0, 1], [0, 0, 2]]),
        "candidate_counts": torch.tensor([3]),
        "anchor_counts": torch.tensor([2]),
    }

    diagnostics.update_capped(
        target=target,
        valid_mask=valid_mask,
        candidate_outputs=candidate_outputs,
    )
    payload = diagnostics.compute_payload(sync_fn=lambda tensor: tensor)

    torch.testing.assert_close(payload.scalars["val_capped/global/recall"], torch.tensor(0.75))


def test_candidate_recall_zero_when_gt_exists_but_C_misses_all() -> None:
    """
    验证有 GT 但 C 无正例覆盖时 val_capped recall 为 0。
    """
    diagnostics = _diagnostics(num_classes=2, candidate_class_ids=(1,))
    target = torch.tensor([[[[1, 0]]]])
    valid_mask = torch.ones(1, 1, 1, 2, dtype=torch.bool)
    candidate_outputs = {
        "candidate_batch_index": torch.tensor([0]),
        "candidate_voxel_zyx": torch.tensor([[0, 0, 1]]),
        "candidate_counts": torch.tensor([1]),
        "anchor_counts": torch.tensor([1]),
    }

    diagnostics.update_capped(
        target=target,
        valid_mask=valid_mask,
        candidate_outputs=candidate_outputs,
    )
    payload = diagnostics.compute_payload(sync_fn=lambda tensor: tensor)

    torch.testing.assert_close(payload.scalars["val_capped/global/recall"], torch.tensor(0.0))


def test_sparse_refine_metric_names_follow_tri_class_names() -> None:
    """
    验证三分类 val_refined 与 val_score 指标名使用配置类别名后缀。
    """
    diagnostics = _diagnostics(num_classes=3, candidate_class_ids=(1, 2))
    candidate_outputs = {"candidate_logits": torch.tensor([[0.0, 6.0, -6.0], [0.0, -6.0, 6.0]])}

    diagnostics.update_refined(
        refined_logits_C=torch.tensor([[0.0, 6.0, -6.0], [0.0, -6.0, 6.0]]),
        candidate_outputs=candidate_outputs,
        target_C=torch.tensor([1, 2]),
        valid_C=torch.tensor([True, True]),
        dense_num_gt=torch.tensor([[1, 1]]),
    )
    payload = diagnostics.compute_payload(sync_fn=lambda tensor: tensor)

    assert "val_refined/global/F1_metal_ion" in payload.scalars
    assert "val_refined/global/F1_small_molecule" in payload.scalars
    assert "val_score/global/refined_F1_metal_ion" in payload.scalars
    assert "val_score/global/refined_F1_small_molecule" in payload.scalars


def test_num_candidate_and_anchor_points_are_mean_per_box() -> None:
    """
    验证 C/P 数量日志为 validation loop 内平均每 BOX 数量。
    """
    diagnostics = _diagnostics(num_classes=2, candidate_class_ids=(1,))
    target = torch.tensor([[[[1, 1]]], [[[1, 0]]]], dtype=torch.long)
    valid_mask = torch.ones(2, 1, 1, 2, dtype=torch.bool)
    candidate_outputs = {
        "candidate_batch_index": torch.tensor([0, 0, 1]),
        "candidate_voxel_zyx": torch.tensor([[0, 0, 0], [0, 0, 1], [0, 0, 0]]),
        "candidate_counts": torch.tensor([2, 1]),
        "anchor_counts": torch.tensor([1, 3]),
    }
    diagnostics.update_capped(
        target=target,
        valid_mask=valid_mask,
        candidate_outputs=candidate_outputs,
    )
    payload = diagnostics.compute_payload(sync_fn=lambda tensor: tensor)

    torch.testing.assert_close(payload.scalars["val_capped/global/num_C"], torch.tensor(1.5))
    torch.testing.assert_close(payload.scalars["val_capped/global/num_P"], torch.tensor(2.0))
