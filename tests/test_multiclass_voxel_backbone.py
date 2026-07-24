from __future__ import annotations

import math

import pytest
import torch
from torch import nn

from src.auxiliary_supervision import (
    NUCLEIC_MAINCHAIN_CLASS_NAMES,
    PROTEIN_MAINCHAIN_CLASS_NAMES,
)
from src.model.stage1_voxel_backbone import Stage1VoxelBackbone


def _make_small_voxel_backbone(**kwargs) -> Stage1VoxelBackbone:
    return Stage1VoxelBackbone(
        in_channels=2,
        feature_channels=4,
        planes=(2, 8, 8, 8, 8, 8, 4, 2, 2),
        gradient_checkpoint=False,
        return_feature_keys=("voxel_final",),
        aux_head_hidden_channels=4,
        num_conv3d_aux=1,
        ligand_head_hidden_channels=4,
        num_conv3d_ligand=1,
        **kwargs,
    )


def test_multiclass_voxel_heads_use_configured_logit_dims_and_prior_probs() -> None:
    backbone = _make_small_voxel_backbone(
        voxel_aux_logit_dim=3,
        voxel_ligand_logit_dim=3,
        prior_probs=[0.98, 0.01, 0.01],
    )

    assert isinstance(backbone.voxel_aux_head[-1], nn.Conv3d)
    assert isinstance(backbone.voxel_ligand_head[-1], nn.Conv3d)
    assert backbone.voxel_aux_head[-1].out_channels == 3
    assert backbone.voxel_ligand_head[-1].out_channels == 3
    assert backbone.voxel_aux_head[0].in_channels == 4
    assert backbone.voxel_aux_head[0].out_channels == 4
    expected_bias = torch.tensor([math.log(0.98), math.log(0.01), math.log(0.01)])
    torch.testing.assert_close(backbone.voxel_aux_head[-1].bias.detach().cpu(), expected_bias)
    torch.testing.assert_close(backbone.voxel_ligand_head[-1].bias.detach().cpu(), expected_bias)


def test_multiclass_voxel_backbone_rejects_single_channel_prior_prob() -> None:
    with pytest.raises(ValueError, match="多通道 softmax head"):
        _make_small_voxel_backbone(
            voxel_aux_logit_dim=3,
            voxel_ligand_logit_dim=3,
            prior_prob=0.01,
        )


def test_named_sigmoid_priors_init_aux_and_ligand_independently() -> None:
    """
    单通道 sigmoid 路径: prior_prob_voxel_receptor 与 prior_prob_voxel_ligand 各自独立初始化 aux/ligand 头末层 bias。
    """
    backbone = _make_small_voxel_backbone(
        voxel_aux_logit_dim=1,
        voxel_ligand_logit_dim=1,
        prior_prob_voxel_receptor=0.1,
        prior_prob_voxel_ligand=0.01,
    )
    # float, sigmoid 正类先验对应的输出 bias = logit(prior)
    aux_expected = -math.log((1.0 - 0.1) / 0.1)
    ligand_expected = -math.log((1.0 - 0.01) / 0.01)
    torch.testing.assert_close(backbone.voxel_aux_head[-1].bias.detach().cpu(),
                               torch.full((1,), aux_expected))
    torch.testing.assert_close(backbone.voxel_ligand_head[-1].bias.detach().cpu(),
                               torch.full((1,), ligand_expected))


def test_named_sigmoid_priors_none_skips_bias_init() -> None:
    """
    主开关关(stage1_model 传 None)时, aux/ligand 命名先验为 None 且无 legacy prior_prob, 则跳过先验 bias 初始化,
    末层 bias 保持 Conv3d 构造默认(均匀初始化, 非 logit(prior))。
    """
    backbone = _make_small_voxel_backbone(
        voxel_aux_logit_dim=1,
        voxel_ligand_logit_dim=1,
        prior_prob_voxel_receptor=None,
        prior_prob_voxel_ligand=None,
    )
    # float, 若被先验初始化则 bias 会等于该值; 跳过时不应等于它
    aux_prior_logit = -math.log((1.0 - 0.1) / 0.1)
    ligand_prior_logit = -math.log((1.0 - 0.01) / 0.01)
    assert not torch.allclose(backbone.voxel_aux_head[-1].bias.detach().cpu(),
                              torch.full((1,), aux_prior_logit), atol=1e-4)
    assert not torch.allclose(backbone.voxel_ligand_head[-1].bias.detach().cpu(),
                              torch.full((1,), ligand_prior_logit), atol=1e-4)


def test_multiclass_prior_probs_take_precedence_over_named_sigmoid() -> None:
    """
    多通道 prior_probs 与命名 sigmoid 先验同时给定时, 多通道路径优先, 命名 sigmoid 被忽略(不冲突、不报错)。
    """
    backbone = _make_small_voxel_backbone(
        voxel_aux_logit_dim=3,
        voxel_ligand_logit_dim=3,
        prior_probs=[0.98, 0.01, 0.01],
        prior_prob_voxel_receptor=0.1,
        prior_prob_voxel_ligand=0.01,
    )
    expected_bias = torch.tensor([math.log(0.98), math.log(0.01), math.log(0.01)])
    torch.testing.assert_close(backbone.voxel_aux_head[-1].bias.detach().cpu(), expected_bias)
    torch.testing.assert_close(backbone.voxel_ligand_head[-1].bias.detach().cpu(), expected_bias)


def test_auxiliary_heads_and_disabled_multiscale_output_contract() -> None:
    backbone = Stage1VoxelBackbone(
        in_channels=2,
        feature_channels=2,
        planes=(2, 8, 8, 8, 8, 8, 4, 2, 2),
        gradient_checkpoint=False,
        return_feature_keys=("voxel_final",),
        aux_head_hidden_channels=4,
        num_conv3d_aux=0,
        ligand_head_hidden_channels=4,
        num_conv3d_ligand=0,
        enable_multiscale_output=False,
        enable_structure_heads=True,
    )

    assert backbone.conv_end_3 is None
    assert backbone.conv_end_5 is None
    assert backbone.conv_end_7 is None
    assert backbone.conv_end is None
    final = torch.zeros((1, 2, 4, 4, 4))
    assert backbone.voxel_protein_head(final).shape == (
        1,
        len(PROTEIN_MAINCHAIN_CLASS_NAMES),
        4,
        4,
        4,
    )
    assert backbone.voxel_nucleic_head(final).shape == (
        1,
        len(NUCLEIC_MAINCHAIN_CLASS_NAMES),
        4,
        4,
        4,
    )
    assert backbone.voxel_distance_head(final).shape == (1, 1, 4, 4, 4)
    assert torch.sigmoid(backbone.voxel_distance_head[-1].bias).item() == pytest.approx(1.0 / 11.0)
