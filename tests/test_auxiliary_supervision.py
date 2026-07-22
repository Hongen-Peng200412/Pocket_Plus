from __future__ import annotations

import pytest
import torch

from src.modules.losses import AdaptiveClassificationCompositeLoss
from src.wrappers.voxel_point_stage1_losses import (
    compute_ligand_distance_loss_term,
    compute_mainchain_class_loss_term,
)


def _classification_loss(num_classes: int) -> AdaptiveClassificationCompositeLoss:
    return AdaptiveClassificationCompositeLoss(
        num_classes=num_classes,
        sigma=2.0,
        hard_label_threshold=None,
        focal_gamma=2.0,
        focal_alpha=None,
        focal_alpha_neg=0.5,
        focal_alpha_pos=0.5,
        focal_eps=1.0e-6,
        tversky_alpha=0.5,
        tversky_beta=0.5,
        tversky_smooth=1.0,
        w_focal=0.7,
        w_tversky=0.3,
        w_mse=0.0,
    )


@pytest.mark.parametrize(("polymer_name", "num_classes"), (("protein", 5), ("nucleic", 7)))
def test_mainchain_class_loss_uses_complete_voxel_grid(
    polymer_name: str,
    num_classes: int,
) -> None:
    logits = torch.zeros((1, num_classes, 2, 2, 2), requires_grad=True)
    target = torch.zeros((1, 2, 2, 2), dtype=torch.long)
    target[0, 0, 0, 0] = num_classes - 1
    term = compute_mainchain_class_loss_term(
        outputs={f"voxel_logits_{polymer_name}": logits},
        batch={f"{polymer_name}_mainchain_target": target},
        loss_module=_classification_loss(num_classes),
        weight=0.1,
        polymer_name=polymer_name,
    )

    assert term is not None
    assert torch.isfinite(term.value)
    term.value.backward()
    assert logits.grad is not None
    assert torch.count_nonzero(logits.grad).item() > 0


def test_ligand_distance_loss_is_full_grid_mean_mse_after_sigmoid() -> None:
    logits = torch.zeros((1, 1, 2, 2, 2), requires_grad=True)
    target = torch.zeros((1, 2, 2, 2))
    term = compute_ligand_distance_loss_term(
        outputs={"voxel_logits_distance": logits},
        batch={"ligand_inverse_distance_target": target},
        weight=0.2,
    )

    assert term is not None
    assert term.value.item() == pytest.approx(0.25)
    term.value.backward()
    assert logits.grad is not None
