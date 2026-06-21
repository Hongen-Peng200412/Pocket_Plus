from __future__ import annotations

import torch

from src.modules.losses import AdaptiveClassificationCompositeLoss


def _loss(num_classes: int) -> AdaptiveClassificationCompositeLoss:
    return AdaptiveClassificationCompositeLoss(
        num_classes=num_classes,
        hard_label_threshold=1.7,
        focal_gamma=2.0,
        focal_alpha=None,
        focal_eps=1.0e-6,
        tversky_alpha=0.3,
        tversky_beta=0.7,
        tversky_smooth=1.0,
        w_focal=1.0,
        w_tversky=1.0,
    )


def test_ligand_metric_target_uses_background_placeholder_channel() -> None:
    dist = torch.full((1, 3, 1, 2, 3), float("inf"))
    dist[:, 1] = torch.tensor([[[2.0, 0.5, 4.0], [3.0, 1.2, 2.1]]])
    dist[:, 2] = torch.tensor([[[0.4, 3.0, 1.0], [0.2, 2.5, 2.0]]])

    loss = _loss(num_classes=3)

    target = loss.target_from_ligand_dist_map(dist, logit_dim=3, device=dist.device, dtype=dist.dtype)

    expected = torch.tensor([[[[2, 1, 2], [2, 1, 0]]]])
    torch.testing.assert_close(target, expected)


def test_ligand_metric_target_keeps_binary_distance_map_path() -> None:
    dist = torch.tensor([[[[2.0, 0.5], [1.6, 1.8]]]])

    loss = _loss(num_classes=2)

    target = loss.target_from_ligand_dist_map(dist, logit_dim=1, device=dist.device, dtype=dist.dtype)

    expected = torch.tensor([[[[0, 1], [1, 0]]]])
    torch.testing.assert_close(target, expected)
