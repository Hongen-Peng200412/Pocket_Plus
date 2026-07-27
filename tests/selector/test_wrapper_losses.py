"""Selector 三项 loss 与条件权重作用域测试. """

from __future__ import annotations

import torch
from torch import nn

from src.selector.wrapper import SelectorWrapper


class _UnusedModel(nn.Module):
    """测试只传预计算 outputs 时使用的占位模型. """

    def forward(self, sample: dict) -> dict:
        """该测试路径不应调用占位模型. """
        raise AssertionError(sample)


def test_negative_clg_blob_loss_trains_but_antichain_does_not() -> None:
    """oracle 模式的负 CLG 仍应产生 qhat 梯度, 但不得产生 z 反链梯度. """
    wrapper = SelectorWrapper(
        model=_UnusedModel(),
        lambda_count=0.05,
        condition_weighting="oracle",
        gamma_focal=0.0,
        w_clg=1.0,
        w_blob=1.0,
        w_antichain=1.0,
        smooth_l1_beta=1.0,
    )
    qhat = torch.tensor([0.7, 0.2], requires_grad=True)
    selection_logit = torch.tensor([0.3, -0.1], requires_grad=True)
    clg_logit = torch.tensor(-0.4, requires_grad=True)
    probability = torch.sigmoid(clg_logit)
    sample = {
        "CLG_is_valid": torch.tensor(0.0),
        "candidate_max_iou": torch.tensor([0.0, 0.0]),
        "oracle_selected_candidate_index": torch.empty((0,), dtype=torch.long),
        "closure_parent_index": (-1, 0),
        "closure_candidate_index_by_node": (0, 1),
    }
    outputs = [{
        "CLG_logit": clg_logit,
        "CLG_valid_probability": probability,
        "predicted_max_iou": qhat,
        "selection_logit": selection_logit,
    }]
    losses = wrapper.compute_loss([sample], outputs)
    losses["total_loss"].backward()
    assert qhat.grad is not None and float(qhat.grad.abs().sum()) > 0.0
    assert selection_logit.grad is not None and float(selection_logit.grad.abs().sum()) == 0.0
    assert clg_logit.grad is not None and float(clg_logit.grad.abs()) > 0.0
