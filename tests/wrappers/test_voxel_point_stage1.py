from __future__ import annotations

import torch
from torch import nn

from src.wrappers.voxel_point_stage1 import VoxelPointStage1Wrapper


class _RecordingLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.last_hardmask: torch.Tensor | None = None

    def forward(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        reduction: str = "mean",
        hardmask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        self.last_hardmask = None if hardmask is None else hardmask.detach().clone()
        return logits.sum() * 0.0


def test_voxel_aux_loss_uses_hardmask_and_voxel_valid_mask() -> None:
    atom_loss = _RecordingLoss()
    voxel_aux_loss = _RecordingLoss()
    wrapper = VoxelPointStage1Wrapper(
        backbone=nn.Identity(),
        atom_loss=atom_loss,
        voxel_aux_loss=voxel_aux_loss,
    )
    outputs = {"voxel_logits_aux": torch.zeros((2, 1, 2, 2, 2), dtype=torch.float32)}
    batch = {
        "voxel_label": torch.zeros((2, 2, 2, 2), dtype=torch.long),
        "hardmask": torch.tensor(
            [
                [[[1, 0], [1, 1]], [[0, 1], [1, 1]]],
                [[[1, 1], [0, 1]], [[1, 0], [1, 0]]],
            ],
            dtype=torch.bool,
        ),
        "voxel_valid_mask": torch.tensor(
            [
                [[[1, 1], [0, 1]], [[1, 1], [1, 0]]],
                [[[1, 0], [1, 1]], [[1, 1], [0, 0]]],
            ],
            dtype=torch.bool,
        ),
    }

    loss = wrapper._compute_voxel_aux_loss(outputs=outputs, batch=batch)

    expected = batch["hardmask"] & batch["voxel_valid_mask"]
    assert loss.item() == 0.0
    assert voxel_aux_loss.last_hardmask is not None
    assert torch.equal(voxel_aux_loss.last_hardmask, expected)
