from __future__ import annotations

import fnmatch
import hashlib
from pathlib import Path
from unittest.mock import Mock

import pytest
import torch
from hydra.utils import instantiate
from lightning.pytorch.utilities.enums import GradClipAlgorithmType
from torch import nn

from src.wrappers.voxel_point_stage1 import (
    FIND_VOXEL_PARAMETER_PREFIXES,
    VoxelPointStage1Wrapper,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
# AUTO 提交 7aae1f... 的 B_trunk 可训练参数名称排序后以换行连接所得 SHA-256.
AUTO_B_TRUNK_PARAMETER_NAMES_SHA256 = (
    "57cb249c0410369a60c17b09950432a23069c541ac4ddf56e9ef4cf9e9467e2e"
)


class _Find1Backbone(nn.Module):
    """提供与生产参数前缀一致的最小可训练骨架."""

    def __init__(self) -> None:
        super().__init__()
        self.embed_head = nn.Module()
        self.embed_head.voxel_input_proj = nn.Linear(1, 1, bias=False)
        self.embed_head.voxel_out_proj_with_offset = nn.Linear(1, 1, bias=False)
        self.embed_head.point_input_proj = nn.Linear(1, 1, bias=False)
        self.voxel_backbone = nn.Linear(1, 1, bias=False)
        self.point_backbone = nn.Linear(1, 1, bias=False)


def _make_wrapper(*, gradient_clip_mode: str) -> VoxelPointStage1Wrapper:
    return VoxelPointStage1Wrapper(
        backbone=_Find1Backbone(),
        class_names=("background", "foreground"),
        validation_diagnostics={"enabled": False},
        gradient_clip_mode=gradient_clip_mode,
    )


def test_find1_parameter_groups_are_disjoint_and_exhaustive() -> None:
    """体素组与其余组必须互斥并覆盖全部可训练参数."""

    wrapper = _make_wrapper(gradient_clip_mode="find_voxel_point")
    voxel_parameters = tuple(
        parameter
        for name, parameter in wrapper.named_parameters()
        if parameter.requires_grad and name.startswith(FIND_VOXEL_PARAMETER_PREFIXES)
    )
    other_parameters = tuple(
        parameter
        for name, parameter in wrapper.named_parameters()
        if parameter.requires_grad and not name.startswith(FIND_VOXEL_PARAMETER_PREFIXES)
    )
    voxel_ids = {id(parameter) for parameter in voxel_parameters}
    other_ids = {id(parameter) for parameter in other_parameters}
    trainable_ids = {
        id(parameter) for parameter in wrapper.parameters() if parameter.requires_grad
    }

    assert voxel_ids
    assert other_ids
    assert voxel_ids.isdisjoint(other_ids)
    assert voxel_ids | other_ids == trainable_ids


def test_find1_voxel_group_matches_auto_b_trunk_parameter_intersection() -> None:
    """生产体素组应等于 AUTO B_trunk 与完整 Find_1 的参数交集."""

    hydra = pytest.importorskip("hydra")
    with hydra.initialize_config_dir(
        config_dir=str(PROJECT_ROOT / "configs"),
        version_base=None,
    ):
        cfg = hydra.compose(
            config_name="base",
            overrides=[
                "+experiment=CPC1/Find_1",
                "model.backbone.point_backbone.enable_flash=false",
                "model.backbone.embed_head.enable_flash=false",
            ],
        )
    wrapper = instantiate(
        cfg.model,
        optimizer=cfg.train.optimizer,
        scheduler=cfg.train.scheduler,
        gradient_clip_mode=cfg.train.gradient_clip_mode,
        compile=False,
    )
    frozen_patterns = tuple(str(pattern) for pattern in cfg.frozen_module.patterns)
    for name, parameter in wrapper.named_parameters():
        if any(fnmatch.fnmatch(name, pattern) for pattern in frozen_patterns):
            parameter.requires_grad = False

    voxel_parameters = tuple(
        parameter
        for name, parameter in wrapper.named_parameters()
        if parameter.requires_grad and name.startswith(FIND_VOXEL_PARAMETER_PREFIXES)
    )
    other_parameters = tuple(
        parameter
        for name, parameter in wrapper.named_parameters()
        if parameter.requires_grad and not name.startswith(FIND_VOXEL_PARAMETER_PREFIXES)
    )
    voxel_ids = {id(parameter) for parameter in voxel_parameters}
    voxel_names = sorted(
        name
        for name, parameter in wrapper.named_parameters()
        if id(parameter) in voxel_ids
    )
    all_trainable = {
        id(parameter) for parameter in wrapper.parameters() if parameter.requires_grad
    }
    other_ids = {id(parameter) for parameter in other_parameters}

    assert len(all_trainable) == 1594
    assert len(voxel_names) == 353
    assert all(name.startswith(FIND_VOXEL_PARAMETER_PREFIXES) for name in voxel_names)
    assert sum(
        name.startswith("backbone.embed_head.voxel_input_proj.")
        for name in voxel_names
    ) == 6
    assert sum(
        name.startswith("backbone.embed_head.voxel_out_proj_with_offset.")
        for name in voxel_names
    ) == 4
    assert sum(name.startswith("backbone.voxel_backbone.") for name in voxel_names) == 343
    assert voxel_ids.isdisjoint(other_ids)
    assert voxel_ids | other_ids == all_trainable
    assert hashlib.sha256("\n".join(voxel_names).encode("utf-8")).hexdigest() == (
        AUTO_B_TRUNK_PARAMETER_NAMES_SHA256
    )


def test_find1_grouped_clipping_clips_each_group_independently() -> None:
    """两个组分别裁剪到 0.5, 合并后的范数可以大于 0.5."""

    wrapper = _make_wrapper(gradient_clip_mode="find_voxel_point")
    voxel_parameters = tuple(
        parameter
        for name, parameter in wrapper.named_parameters()
        if name.startswith(FIND_VOXEL_PARAMETER_PREFIXES)
    )
    other_parameters = tuple(
        parameter
        for name, parameter in wrapper.named_parameters()
        if not name.startswith(FIND_VOXEL_PARAMETER_PREFIXES)
    )
    for parameter in voxel_parameters:
        parameter.grad = torch.full_like(parameter, 3.0)
    for parameter in other_parameters:
        parameter.grad = torch.full_like(parameter, 4.0)
    optimizer = torch.optim.AdamW(wrapper.parameters(), lr=1e-4)

    wrapper.configure_gradient_clipping(
        optimizer,
        gradient_clip_val=0.5,
        gradient_clip_algorithm="norm",
    )

    voxel_norm = torch.linalg.vector_norm(
        torch.cat([parameter.grad.flatten() for parameter in voxel_parameters])
    )
    other_norm = torch.linalg.vector_norm(
        torch.cat([parameter.grad.flatten() for parameter in other_parameters])
    )
    combined_norm = torch.linalg.vector_norm(torch.stack([voxel_norm, other_norm]))
    assert voxel_norm.item() == pytest.approx(0.5, abs=1e-6)
    assert other_norm.item() == pytest.approx(0.5, abs=1e-6)
    assert combined_norm.item() == pytest.approx(2**0.5 * 0.5, abs=1e-6)


def test_global_clipping_keeps_lightning_clip_entry() -> None:
    """global 模式继续调用 Lightning 原来的统一裁剪入口."""

    wrapper = _make_wrapper(gradient_clip_mode="global")
    wrapper.clip_gradients = Mock()
    optimizer = torch.optim.AdamW(wrapper.parameters(), lr=1e-4)

    wrapper.configure_gradient_clipping(
        optimizer,
        gradient_clip_val=0.5,
        gradient_clip_algorithm="norm",
    )

    wrapper.clip_gradients.assert_called_once_with(
        optimizer,
        gradient_clip_val=0.5,
        gradient_clip_algorithm="norm",
    )


def test_find1_grouped_clipping_rejects_value_algorithm() -> None:
    """双组模式不得把 value 请求静默解释成 norm 裁剪."""

    wrapper = _make_wrapper(gradient_clip_mode="find_voxel_point")
    optimizer = torch.optim.AdamW(wrapper.parameters(), lr=1e-4)

    with pytest.raises(ValueError, match="只支持 norm"):
        wrapper.configure_gradient_clipping(
            optimizer,
            gradient_clip_val=0.5,
            gradient_clip_algorithm="value",
        )


def test_find1_grouped_clipping_accepts_lightning_norm_enum() -> None:
    """双组模式接受 Lightning 实际传入的 norm 枚举值."""

    wrapper = _make_wrapper(gradient_clip_mode="find_voxel_point")
    for parameter in wrapper.parameters():
        parameter.grad = torch.ones_like(parameter)
    optimizer = torch.optim.AdamW(wrapper.parameters(), lr=1e-4)

    wrapper.configure_gradient_clipping(
        optimizer,
        gradient_clip_val=0.5,
        gradient_clip_algorithm=GradClipAlgorithmType.NORM,
    )
