# -*- coding: utf-8 -*-
"""
Stage1PointBackbone 的 recycle 输入融合单元测试。
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from src.model.stage1_point_backbone import Stage1PointBackbone


def _make_backbone(**overrides) -> Stage1PointBackbone:
    """构造一个 backend='zeros' 的最小可测点分支实例。"""
    defaults = dict(
        backend="zeros",
        atom_feature_dim=2,
        point_grid_size=0.25,
        input_embed_dim=2,
        input_embed_hidden_dim=2,
        out_channels=4,
        recycle_feature_dim=2,
        recycle_in_norm_mode="none",
        recycle_use_gate=False,
        recycle_gate_init=0.0,
        serialization_orders=("z",),
        shuffle_orders=False,
        stride=(4, 2, 2, 2),
        embedding_kernel_size=5,
        embedding_impl="pointconv",
        cpe_impl="pointconv",
        embedding_receptive_field=5.0,
        pointconv_embed_max_neighbors=16,
        pointconv_block_max_neighbors=16,
        enc_cpe_kernel_size=(5, 5, 5, 5, 5),
        dec_cpe_kernel_size=(5, 5, 5, 5, 5),
        enc_cpe_receptive_field=(2.0, 2.0, 4.0, 4.0, 4.0),
        dec_cpe_receptive_field=(2.0, 2.0, 4.0, 4.0, 4.0),
        enc_depths=(1, 1, 1, 1, 1),
        enc_channels=(4, 4, 8, 8, 8),
        enc_num_head=(1, 1, 1, 1, 1),
        enc_patch_size=(8, 8, 8, 8, 8),
        dec_depths=(1, 1, 1, 1, 1),
        dec_channels=(4, 4, 8, 8, 8),
        dec_num_head=(1, 1, 1, 1, 1),
        dec_patch_size=(8, 8, 8, 8, 8),
        mlp_ratio=2,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.0,
        pre_norm=True,
        enable_rpe=False,
        enable_flash=False,
        upcast_attention=False,
        upcast_softmax=False,
        cls_mode=False,
        pdnorm_bn=False,
        pdnorm_ln=False,
        pdnorm_decouple=True,
        pdnorm_adaptive=False,
        pdnorm_affine=True,
        pdnorm_conditions=("ScanNet",),
        act_layer_name="gelu",
        ffn_type="mlp",
    )
    defaults.update(overrides)
    return Stage1PointBackbone(**defaults)


def test_build_point_input_feat_adds_projected_recycle_residual() -> None:
    """关闭 norm / gate 时，应退化为 `atom_feat + Linear(recycle_in)`。"""
    backbone = _make_backbone(recycle_in_norm_mode="none", recycle_use_gate=False)
    backbone.atom_input_proj = nn.Identity()
    with torch.no_grad():
        backbone.recycle_input_proj.weight.copy_(torch.eye(2))
        backbone.recycle_input_proj.bias.zero_()

    atom_feat = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    recycle_in = torch.tensor([[0.5, -1.0], [1.5, 2.5]])

    point_input_feat = backbone._build_point_input_feat(atom_feat=atom_feat, recycle_in=recycle_in)

    expected = atom_feat + recycle_in
    assert torch.allclose(point_input_feat, expected)


def test_build_point_input_feat_gate_zero_blocks_recycle_residual() -> None:
    """门控初值为 0 时，recycle 残差应被完全关闭。"""
    backbone = _make_backbone(recycle_in_norm_mode="none", recycle_use_gate=True, recycle_gate_init=0.0)
    backbone.atom_input_proj = nn.Identity()
    with torch.no_grad():
        backbone.recycle_input_proj.weight.copy_(torch.eye(2))
        backbone.recycle_input_proj.bias.zero_()

    atom_feat = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    recycle_in = torch.tensor([[0.5, -1.0], [1.5, 2.5]])

    point_input_feat = backbone._build_point_input_feat(atom_feat=atom_feat, recycle_in=recycle_in)

    assert torch.allclose(point_input_feat, atom_feat)


def test_build_point_input_feat_layernorm_normalizes_recycle_input() -> None:
    """开启 layernorm 时，应先归一化 recycle 输入再做残差相加。"""
    backbone = _make_backbone(recycle_in_norm_mode="layernorm", recycle_use_gate=False)
    backbone.atom_input_proj = nn.Identity()
    with torch.no_grad():
        backbone.recycle_input_proj.weight.copy_(torch.eye(2))
        backbone.recycle_input_proj.bias.zero_()

    atom_feat = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    recycle_in = torch.tensor([[2.0, 0.0], [6.0, 2.0]])

    point_input_feat = backbone._build_point_input_feat(atom_feat=atom_feat, recycle_in=recycle_in)

    expected = atom_feat + F.layer_norm(recycle_in, normalized_shape=(2,))
    assert torch.allclose(point_input_feat, expected, atol=1e-6, rtol=1e-6)
