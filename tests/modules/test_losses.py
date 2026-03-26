from __future__ import annotations

import torch
import torch.nn.functional as F

from src.modules.losses import BinaryFocalLossWithAlpha


def _build_loss() -> BinaryFocalLossWithAlpha:
    return BinaryFocalLossWithAlpha(
        from_logits=True,
        gamma=0.0,
        use_adaptive_alpha=False,
        alpha_tune=None,
        scale=1.0,
    )


def test_binary_focal_mean_without_hardmask_matches_plain_mean() -> None:
    loss_fn = _build_loss()
    logits = torch.zeros((3, 1), dtype=torch.float32)
    target = torch.tensor([0, 1, 0], dtype=torch.long)

    loss = loss_fn(logits, target, reduction="mean")

    base_loss = 0.5 * F.binary_cross_entropy_with_logits(
        input=logits,
        target=target.unsqueeze(1).float(),
        reduction="none",
    ).squeeze(1)
    assert torch.isclose(loss, base_loss.mean())


def test_binary_focal_mean_counts_only_nonzero_hardmask_positions() -> None:
    loss_fn = _build_loss()
    logits = torch.zeros((3, 1), dtype=torch.float32)
    target = torch.tensor([0, 1, 0], dtype=torch.long)
    hardmask = torch.tensor([[0.0], [0.5], [1.0]], dtype=torch.float32)

    loss = loss_fn(logits, target, reduction="mean", hardmask=hardmask)

    base_loss = 0.5 * F.binary_cross_entropy_with_logits(
        input=logits,
        target=target.unsqueeze(1).float(),
        reduction="none",
    ).squeeze(1)
    expected = (base_loss * hardmask.squeeze(1)).sum() / torch.tensor(2.0)
    assert torch.isclose(loss, expected)


def test_binary_focal_mean_returns_zero_when_hardmask_is_all_zero() -> None:
    loss_fn = _build_loss()
    logits = torch.zeros((3, 1), dtype=torch.float32)
    target = torch.tensor([0, 1, 0], dtype=torch.long)
    hardmask = torch.zeros((3, 1), dtype=torch.float32)

    loss = loss_fn(logits, target, reduction="mean", hardmask=hardmask)

    assert loss.item() == 0.0
