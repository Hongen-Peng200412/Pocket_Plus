from __future__ import annotations

import torch

from src.wrappers.voxel_point_stage1_diagnostics import CpcDiagnosticsConfig, CpcValidationDiagnostics


def _config() -> CpcDiagnosticsConfig:
    """
    构造测试用 diagnostics 配置. 

    输出:
        - config: CpcDiagnosticsConfig, 4-bin global-only diagnostics 配置
    """
    return CpcDiagnosticsConfig(
        enabled=True,
        num_bins=4,
        write_local_artifacts=False,
        log_wandb_curves=False,
        wandb_curve_every_n_validation=1,
        output_subdir="validation_diagnostics",
    )


def _diagnostics() -> CpcValidationDiagnostics:
    """
    构造二分类 CPC diagnostics 测试对象. 

    输出:
        - diagnostics: CpcValidationDiagnostics, 候选类为 foreground 的测试对象
    """
    return CpcValidationDiagnostics(
        config=_config(),
        class_names=["background", "foreground"],
        candidate_class_ids=[1],
        adaptive_expand_factor=[7.0],
        max_candidate_voxels_per_class=[4],
    )


def _hist_count(payload, hist_name: str, bin_index: int) -> float:
    """
    从 histogram payload 中取 foreground 指定 bin 的计数. 

    输入参数:
        - payload: CpcDiagnosticsPayload, diagnostics epoch 输出
        - hist_name: str, histogram 名
        - bin_index: int, 概率 bin 下标

    输出:
        - count: float, 对应 bin 的计数
    """
    table = payload.histograms[hist_name]
    for row in table.rows:
        if row[0] == "foreground" and int(row[3]) == int(bin_index):
            return float(row[6])
    raise AssertionError(f"{hist_name} missing foreground bin {bin_index}")


def test_uncapped_best_outputs_global_f1_and_sampling_threshold() -> None:
    """
    验证 dense best-F1 diagnostics 输出 global 标量和 supervised p_sampling. 
    """
    diagnostics = _diagnostics()
    logits = torch.logit(torch.tensor([[[[[0.9, 0.1]]]]]), eps=1e-6)
    target = torch.tensor([[[[1, 0]]]])
    valid_mask = torch.ones(1, 1, 1, 2, dtype=torch.bool)

    diagnostics.update_uncapped_best(
        logits=logits,
        target=target,
        valid_mask=valid_mask,
        allow_cache_update=True,
    )
    payload = diagnostics.compute_payload(sync_fn=lambda tensor: tensor)

    torch.testing.assert_close(payload.scalars["val_uncapped_best/global/best_F1"], torch.tensor(1.0))
    torch.testing.assert_close(payload.scalars["val_uncapped_best/global/p_sampling"], torch.tensor(0.0))


def test_reset_preserves_threshold_grid_and_clears_histograms() -> None:
    """
    验证 reset 保留阈值网格并清空统计 histogram. 
    """
    diagnostics = _diagnostics()
    grid_before = diagnostics.threshold_grid.clone()
    diagnostics.uncapped_best_pos_hist += 1

    diagnostics.reset()

    torch.testing.assert_close(diagnostics.threshold_grid, grid_before)
    assert int(diagnostics.uncapped_best_pos_hist.sum().item()) == 0


def test_uncapped_sampling_uses_real_per_box_boundary() -> None:
    """
    验证 val_uncapped_sampling 使用 candidate builder 记录的 per-BOX 边界统计. 
    """
    diagnostics = _diagnostics()
    logits = torch.logit(torch.tensor([[[[[0.9, 0.6, 0.4, 0.1]]]]]), eps=1e-6)
    target = torch.tensor([[[[1, 0, 1, 0]]]])
    valid_mask = torch.ones(1, 1, 1, 4, dtype=torch.bool)
    candidate_outputs = {
        "candidate_batch_index": torch.tensor([0, 0]),
        "candidate_voxel_zyx": torch.tensor([[0, 0, 0], [0, 0, 1]]),
        "candidate_p_sampling_by_class": torch.tensor([[0.5]]),
        "candidate_target_counts_by_class": torch.tensor([[2]]),
    }

    diagnostics.update_uncapped_sampling(
        logits=logits,
        target=target,
        valid_mask=valid_mask,
        candidate_outputs=candidate_outputs,
        selection_mode="topk",
        use_fixed_warmup=False,
    )
    payload = diagnostics.compute_payload(sync_fn=lambda tensor: tensor)

    torch.testing.assert_close(payload.scalars["val_uncapped_sampling/global/sampling_precision"], torch.tensor(0.5))
    torch.testing.assert_close(payload.scalars["val_uncapped_sampling/global/sampling_recall"], torch.tensor(0.5))
    torch.testing.assert_close(payload.scalars["val_uncapped_sampling/global/sampling_F1"], torch.tensor(0.5))
    torch.testing.assert_close(payload.scalars["val_uncapped_sampling/global/numC_sampling_target"], torch.tensor(2.0))
    assert _hist_count(payload, "uncapped_sampling_boundary_hist", 2) == 1.0


def test_uncapped_sampling_nan_boundary_counts_false_negatives() -> None:
    """
    验证缺失 sampling boundary 时不会跳过 dense GT 正例统计. 
    """
    diagnostics = _diagnostics()
    logits = torch.logit(torch.tensor([[[[[0.9, 0.1, 0.8, 0.2]]]]]), eps=1e-6)
    target = torch.tensor([[[[1, 0, 1, 0]]]])
    valid_mask = torch.ones(1, 1, 1, 4, dtype=torch.bool)
    candidate_outputs = {
        "candidate_batch_index": torch.empty((0,), dtype=torch.long),
        "candidate_voxel_zyx": torch.empty((0, 3), dtype=torch.long),
        "candidate_p_sampling_by_class": torch.tensor([[float("nan")]]),
        "candidate_target_counts_by_class": torch.tensor([[0]]),
    }

    diagnostics.update_uncapped_sampling(
        logits=logits,
        target=target,
        valid_mask=valid_mask,
        candidate_outputs=candidate_outputs,
        selection_mode="topk",
        use_fixed_warmup=False,
    )
    payload = diagnostics.compute_payload(sync_fn=lambda tensor: tensor)

    torch.testing.assert_close(payload.scalars["val_uncapped_sampling/global/sampling_fn"], torch.tensor(2.0))
    torch.testing.assert_close(payload.scalars["val_uncapped_sampling/global/sampling_recall"], torch.tensor(0.0))
    assert _hist_count(payload, "uncapped_sampling_boundary_hist", 0) == 0.0


def test_uncapped_sampling_uses_builder_candidate_rows() -> None:
    """
    验证 sampling 统计使用 builder 已输出的候选行, 不在 diagnostics 内按 cutoff 重选. 
    """
    diagnostics = _diagnostics()
    logits = torch.zeros(1, 1, 1, 1, 4)
    target = torch.tensor([[[[1, 1, 1, 1]]]])
    valid_mask = torch.ones(1, 1, 1, 4, dtype=torch.bool)
    candidate_outputs = {
        "candidate_batch_index": torch.tensor([0, 0]),
        "candidate_voxel_zyx": torch.tensor([[0, 0, 0], [0, 0, 1]]),
        "candidate_p_sampling_by_class": torch.tensor([[0.5]]),
        "candidate_target_counts_by_class": torch.tensor([[2]]),
    }

    diagnostics.update_uncapped_sampling(
        logits=logits,
        target=target,
        valid_mask=valid_mask,
        candidate_outputs=candidate_outputs,
        selection_mode="topk",
        use_fixed_warmup=False,
    )
    payload = diagnostics.compute_payload(sync_fn=lambda tensor: tensor)

    torch.testing.assert_close(payload.scalars["val_uncapped_sampling/global/sampling_recall"], torch.tensor(0.5))


def test_capped_recall_and_routed_prob_hist_use_final_C() -> None:
    """
    验证 val_capped recall 和 capped_routed_prob_hist 基于最终唯一 C 统计. 
    """
    diagnostics = _diagnostics()
    target = torch.tensor([[[[1, 1, 1, 1]]]])
    valid_mask = torch.ones(1, 1, 1, 4, dtype=torch.bool)
    candidate_outputs = {
        "candidate_batch_index": torch.tensor([0, 0, 0]),
        "candidate_voxel_zyx": torch.tensor([[0, 0, 0], [0, 0, 1], [0, 0, 2]]),
        "candidate_counts": torch.tensor([3]),
        "candidate_class": torch.tensor([1, 1, 1]),
        "candidate_prob": torch.tensor([0.9, 0.6, 0.2]),
        "anchor_counts": torch.tensor([2]),
    }

    diagnostics.update_capped(target=target, valid_mask=valid_mask, candidate_outputs=candidate_outputs)
    payload = diagnostics.compute_payload(sync_fn=lambda tensor: tensor)

    torch.testing.assert_close(payload.scalars["val_capped/global/recall"], torch.tensor(0.75))
    torch.testing.assert_close(payload.scalars["val_capped/global/num_C"], torch.tensor(3.0))
    torch.testing.assert_close(payload.scalars["val_capped/global/num_P"], torch.tensor(2.0))
    assert _hist_count(payload, "capped_routed_prob_hist", 0) == 1.0
    assert _hist_count(payload, "capped_routed_prob_hist", 2) == 1.0
    assert _hist_count(payload, "capped_routed_prob_hist", 3) == 1.0


def test_refined_local_f1_and_score_f1_use_different_fn_spaces() -> None:
    """
    验证 val_refined/F1 使用 C 内 FN, val_score/refined_F1 使用 dense 全空间 FN. 
    """
    diagnostics = _diagnostics()
    refined_logits_C = torch.tensor([[6.0]])
    candidate_outputs = {"candidate_logits": torch.tensor([[6.0]])}
    target_C = torch.tensor([1])
    valid_C = torch.tensor([True])
    dense_num_gt = torch.tensor([[2]])

    diagnostics.update_unrefined(
        candidate_outputs=candidate_outputs,
        target_C=target_C,
        valid_C=valid_C,
        dense_num_gt=dense_num_gt,
    )
    diagnostics.update_refined(
        refined_logits_C=refined_logits_C,
        candidate_outputs=candidate_outputs,
        target_C=target_C,
        valid_C=valid_C,
        dense_num_gt=dense_num_gt,
    )
    payload = diagnostics.compute_payload(sync_fn=lambda tensor: tensor)

    torch.testing.assert_close(payload.scalars["val_refined/global/F1"], torch.tensor(1.0))
    torch.testing.assert_close(payload.scalars["val_score/global/refined_F1"], torch.tensor(2.0 / 3.0))


def test_refined_score_f1_is_zero_when_C_misses_dense_positive() -> None:
    """
    验证 C 内无正例但 dense 空间有正例时端到端 refined_F1 记为 0. 
    """
    diagnostics = _diagnostics()

    diagnostics.update_refined(
        refined_logits_C=torch.empty((0, 1)),
        candidate_outputs={},
        target_C=torch.empty((0,), dtype=torch.long),
        valid_C=torch.empty((0,), dtype=torch.bool),
        dense_num_gt=torch.tensor([[2]]),
    )
    payload = diagnostics.compute_payload(sync_fn=lambda tensor: tensor)

    assert torch.isnan(payload.scalars["val_refined/global/F1"])
    torch.testing.assert_close(payload.scalars["val_score/global/refined_F1"], torch.tensor(0.0))
