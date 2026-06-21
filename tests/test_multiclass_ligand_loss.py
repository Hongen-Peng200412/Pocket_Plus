from __future__ import annotations

import pytest
import torch

from src.modules.losses import AdaptiveClassificationCompositeLoss


def _make_multiclass_ligand_loss() -> AdaptiveClassificationCompositeLoss:
    return AdaptiveClassificationCompositeLoss(
        num_classes=3,
        hard_label_threshold=1.7,
        focal_gamma=2.0,
        focal_alpha=[0.5, 0.25, 0.25],
        focal_eps=1.0e-6,
        tversky_alpha=0.5,
        tversky_beta=0.5,
        tversky_smooth=1.0,
        w_focal=0.7,
        w_tversky=0.3,
        w_mse=0.0,
    )


def test_multiclass_dist_target_uses_foreground_channels_after_background_placeholder() -> None:
    loss_fn = _make_multiclass_ligand_loss()
    logits = torch.zeros(1, 3, 1, 2, 3)
    dist = torch.full((1, 3, 1, 2, 3), float("inf"))
    dist[:, 1] = torch.tensor([[[2.0, 0.5, 4.0], [3.0, 1.2, 2.1]]])
    dist[:, 2] = torch.tensor([[[0.4, 3.0, 1.0], [0.2, 2.5, 2.0]]])

    target = loss_fn._target_from_multiclass_dist(dist, logits)

    expected = torch.tensor([[[[2, 1, 2], [2, 1, 0]]]])
    torch.testing.assert_close(target, expected)


def test_multiclass_ligand_loss_accepts_distance_map_with_background_placeholder() -> None:
    loss_fn = _make_multiclass_ligand_loss()
    logits = torch.randn(2, 3, 2, 3, 4)
    dist = torch.full((2, 3, 2, 3, 4), float("inf"))
    dist[:, 1] = 2.0
    dist[:, 2] = 3.0
    dist[0, 1, 0, 0, 0] = 0.2
    dist[1, 2, 1, 2, 3] = 0.4
    valid_mask = torch.ones(2, 2, 3, 4, dtype=torch.bool)

    loss = loss_fn(
        logits=logits,
        target=None,
        valid_mask=valid_mask,
        ligand_dist_map=dist,
    )

    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_multiclass_ligand_loss_rejects_foreground_only_distance_map() -> None:
    loss_fn = _make_multiclass_ligand_loss()
    logits = torch.zeros(1, 3, 2, 2, 2)
    foreground_only_dist = torch.zeros(1, 2, 2, 2, 2)

    with pytest.raises(ValueError, match="多分类 ligand_dist_map"):
        loss_fn(
            logits=logits,
            target=None,
            ligand_dist_map=foreground_only_dist,
        )
