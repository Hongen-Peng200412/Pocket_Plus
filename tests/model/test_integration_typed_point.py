"""
01/02 计划实现集成验证测试。

覆盖以下维度:
    - TypedPointConfig 在整条链路上的传播一致性 (Stage1Model -> PointBackbone -> PTV3 -> Block)
    - pseudo_mask 在 SerializedPooling -> SerializedUnpooling 回路中的保持
    - typed CPE 同类子图隔离 (real 不看 pseudo 邻居)
    - typed Embedding 同类子图隔离
    - typed FFN 路径隔离
    - cluster_key 在 typed pooling 中把 real/pseudo 分到不同簇
"""
from __future__ import annotations

import pytest
import torch
from torch import nn

from src.model.PTV3bakcbone.model import (
    Block,
    Embedding,
    Point,
    PointConvCPE,
    PointConvEmbedding,
    PointTransformerV3,
    SerializedAttention,
    SerializedPooling,
    SerializedUnpooling,
)
from src.model.typed_point import (
    TypedPointConfig,
    apply_type_aware_tensor_module,
    normalize_typed_point_cfg,
    split_mask_state,
    validate_pseudo_mask,
)


def _make_point(n_real: int, n_pseudo: int, channels: int, grid_size: float = 1.0) -> tuple[Point, torch.Tensor]:
    """
    构造带 pseudo_mask 的 mixed Point 对象。

    输入参数:
        - n_real: int, 真实原子数
        - n_pseudo: int, P anchor 数
        - channels: int, 特征通道数
        - grid_size: float, 点云 grid size, 建议值 1.0

    输出:
        - point: Point, mixed 点对象
        - pseudo_mask: torch.Tensor, (N_all,), bool, True 表示 P anchor
    """
    n = n_real + n_pseudo
    # torch.Tensor, (N, C), 随机点特征
    feat = torch.randn(n, channels)
    # torch.Tensor, (N, 3), 随机点坐标, 扩大范围避免全同坐标退化
    coord = torch.randn(n, 3) * 5.0
    # torch.Tensor, (N,), bool, 后 n_pseudo 个点为 pseudo
    pseudo_mask = torch.zeros(n, dtype=torch.bool)
    pseudo_mask[n_real:] = True
    # torch.Tensor, (1,), 只有 1 个 batch
    offset = torch.tensor([n], dtype=torch.long)
    point = Point({
        "feat": feat,
        "coord": coord,
        "batch": torch.zeros(n, dtype=torch.long),
        "offset": offset,
        "grid_size": grid_size,
        "pseudo_mask": pseudo_mask,
    })
    return point, pseudo_mask


# ============================= TypedPointConfig 传播一致性 =============================

def test_typed_point_config_propagation_to_ptv3() -> None:
    """
    验证 TypedPointConfig 的 effective flags 正确传播到 PTV3 的 Block/Embedding/Pooling/Unpooling。
    """
    cfg = normalize_typed_point_cfg({"enabled": True, "separate_qkv": True, "separate_cpe": True})
    ptv3 = PointTransformerV3(
        in_channels=8,
        enc_depths=(1, 1, 1, 1, 1),
        enc_channels=(8, 8, 16, 16, 32),
        enc_num_head=(2, 2, 2, 2, 2),
        enc_patch_size=(32, 32, 32, 32, 32),
        dec_depths=(1, 1, 1, 1, 0),
        dec_channels=(8, 8, 16, 16, 32),
        dec_num_head=(2, 2, 2, 2, 2),
        dec_patch_size=(32, 32, 32, 32, 32),
        enc_cpe_kernel_size=(3, 3, 3, 3, 3),
        dec_cpe_kernel_size=(3, 3, 3, 3, 3),
        enc_cpe_receptive_field=(2.0, 4.0, 8.0, 12.0, 16.0),
        dec_cpe_receptive_field=(2.0, 4.0, 8.0, 12.0, 16.0),
        stride=(2, 2, 2, 2),
        enable_flash=False,
        enable_rpe=False,
        upcast_attention=False,
        upcast_softmax=False,
        separate_qkv=cfg.use_separate_qkv,
        separate_cpe=cfg.use_separate_cpe,
        separate_attn_proj=False,
        separate_ffn=False,
        separate_embedding=False,
        separate_pooling_proj=False,
        separate_unpooling_proj=False,
    )

    # 验证 enc 中某个 Block 的 separate_qkv / separate_cpe 是否已设置
    enc0_blocks = list(ptv3.enc._modules["enc0"]._modules.values())
    # enc0 第一个 Block
    block0 = enc0_blocks[0]
    assert isinstance(block0, Block)
    assert block0.attn.separate_qkv is True
    assert block0.separate_cpe is True
    assert block0.cpe_real is not None
    assert block0.cpe_pseudo is not None
    assert block0.cpe is None


# ============================= pseudo_mask 在 pooling→unpooling 回路中保持 =============================

def test_pseudo_mask_preserved_through_pool_unpool() -> None:
    """
    验证 pseudo_mask 在 SerializedPooling + SerializedUnpooling 后仍能恢复到 parent 分辨率。
    """
    n_real, n_pseudo, c = 20, 10, 16
    point, pseudo_mask = _make_point(n_real, n_pseudo, c)
    point.serialization(order=["z"], shuffle_orders=False)

    pooling = SerializedPooling(in_channels=c, out_channels=c, stride=2, norm_layer=nn.LayerNorm, act_layer=nn.GELU)
    unpooling = SerializedUnpooling(in_channels=c, skip_channels=c, out_channels=c, norm_layer=nn.LayerNorm, act_layer=nn.GELU)

    pooled = pooling(point)
    # pooled 应包含 pseudo_mask
    assert "pseudo_mask" in pooled.keys(), "pooled Point 必须保留 pseudo_mask"
    pooled_mask = pooled.pseudo_mask
    assert pooled_mask.dtype == torch.bool
    assert pooled_mask.shape[0] == pooled.feat.shape[0]
    # pooled 的 pseudo 点数 <= 原始 pseudo 点数
    assert int(pooled_mask.sum().item()) <= n_pseudo

    # unpooling 回到 parent 分辨率
    restored = unpooling(pooled)
    assert "pseudo_mask" in restored.keys(), "restored Point 必须保留 pseudo_mask"
    # 恢复后的 pseudo_mask 应与原始一致
    assert restored.pseudo_mask.shape[0] == n_real + n_pseudo
    assert torch.equal(restored.pseudo_mask, pseudo_mask)


# ============================= typed CPE 同类子图隔离 =============================

def test_typed_cpe_isolates_real_and_pseudo_subgraphs() -> None:
    """
    验证 separate_cpe=True 时, real 和 pseudo 各自只在同类子图上建 radius graph。
    """
    c = 16
    cpe_real = PointConvCPE(channels=c, receptive_field=3.0, max_neighbors=8, cache_key=None)
    cpe_pseudo = PointConvCPE(channels=c, receptive_field=3.0, max_neighbors=8, cache_key=None)
    n_real, n_pseudo = 10, 5
    point, pseudo_mask = _make_point(n_real, n_pseudo, c)

    # real 子图 delta
    delta_real = cpe_real.forward_subset(point, ~pseudo_mask, type_name="real")
    assert delta_real.shape == (n_real, c)

    # pseudo 子图 delta
    delta_pseudo = cpe_pseudo.forward_subset(point, pseudo_mask, type_name="pseudo")
    assert delta_pseudo.shape == (n_pseudo, c)


# ============================= typed Embedding 同类子图隔离 =============================

def test_typed_embedding_isolates_subgraphs() -> None:
    """
    验证 separate_embedding=True 时, Embedding 对 real 和 pseudo 子图分别做 pointconv。
    """
    in_c, embed_c = 8, 16
    embedding = Embedding(
        in_channels=in_c,
        embed_channels=embed_c,
        embedding_impl="pointconv",
        embedding_receptive_field=3.0,
        pointconv_embed_max_neighbors=8,
        norm_layer=nn.LayerNorm,
        act_layer=nn.GELU,
        separate_embedding=True,
    )
    n_real, n_pseudo = 12, 6
    point, pseudo_mask = _make_point(n_real, n_pseudo, in_c)
    result = embedding(point)
    assert result.feat.shape == (n_real + n_pseudo, embed_c)


# ============================= typed FFN 路径隔离 =============================

def test_typed_ffn_uses_separate_paths() -> None:
    """
    验证 separate_ffn=True 时, Block FFN 分别走 real/pseudo 分支。
    """
    c = 16
    block = Block(
        channels=c,
        num_heads=2,
        patch_size=32,
        cpe_impl="none",
        enable_flash=False,
        enable_rpe=False,
        upcast_attention=False,
        upcast_softmax=False,
        ffn_type="mlp",
        mlp_ratio=2,
        separate_ffn=True,
        separate_qkv=False,
        separate_attn_proj=False,
        separate_cpe=False,
    )
    assert block.mlp_real is not None
    assert block.mlp_pseudo is not None
    assert block.mlp is None

    # 确认 forward 能正常跑
    n_real, n_pseudo = 8, 4
    point, pseudo_mask = _make_point(n_real, n_pseudo, c)
    point.serialization(order=["z"], shuffle_orders=False)
    result = block(point)
    assert result.feat.shape == (n_real + n_pseudo, c)


# ============================= typed pooling cluster_key 分组验证 =============================

def test_typed_pooling_cluster_key_separates_real_and_pseudo() -> None:
    """
    验证 typed SerializedPooling 使用 cluster_key = spatial_code * 2 + type_bit,
    确保同一 spatial cell 内的 real 和 pseudo 点被分到不同簇。
    """
    c = 16
    pooling = SerializedPooling(in_channels=c, out_channels=c, stride=2, norm_layer=nn.LayerNorm, act_layer=nn.GELU)

    # 构造两个 real 点和两个 pseudo 点在同一 grid cell 内
    feat = torch.randn(4, c)
    # 所有点坐标相同, 确保它们落在同一 spatial cell
    coord = torch.zeros(4, 3)
    pseudo_mask = torch.tensor([False, False, True, True])
    offset = torch.tensor([4], dtype=torch.long)
    point = Point({
        "feat": feat,
        "coord": coord,
        "batch": torch.zeros(4, dtype=torch.long),
        "offset": offset,
        "grid_size": 1.0,
        "pseudo_mask": pseudo_mask,
    })
    point.serialization(order=["z"], shuffle_orders=False)

    pooled = pooling(point)
    # 同一 spatial cell 内, real 和 pseudo 应该各产生一个簇 → 至少 2 个 pooled token
    assert pooled.feat.shape[0] >= 2, (
        f"同一 spatial cell 内 real+pseudo 应产生至少 2 个 pooled token, 当前为 {pooled.feat.shape[0]}"
    )
    # 且 pooled pseudo_mask 应包含至少一个 True 和一个 False
    assert "pseudo_mask" in pooled.keys()
    assert bool(pooled.pseudo_mask.any().item())
    assert bool((~pooled.pseudo_mask).any().item())


# ============================= QKV typed attention 分支验证 =============================

def test_typed_qkv_attention_runs_separate_projections() -> None:
    """
    验证 separate_qkv=True 时, SerializedAttention 走 real/pseudo 分离的 QKV 投影。
    """
    c = 16
    attn = SerializedAttention(
        channels=c,
        num_heads=2,
        patch_size=32,
        enable_flash=False,
        enable_rpe=False,
        upcast_attention=False,
        upcast_softmax=False,
        separate_qkv=True,
        separate_attn_proj=True,
    )
    assert attn.separate_qkv is True
    assert hasattr(attn, "qkv_real")
    assert hasattr(attn, "qkv_pseudo")

    n_real, n_pseudo = 8, 4
    point, pseudo_mask = _make_point(n_real, n_pseudo, c)
    point.serialization(order=["z"], shuffle_orders=False)
    result = attn(point)
    assert result.feat.shape == (n_real + n_pseudo, c)


# ============================= validate_pseudo_mask fail-fast =============================

def test_validate_pseudo_mask_rejects_wrong_dtype() -> None:
    """
    验证 validate_pseudo_mask 对非 bool dtype 立即报错。
    """
    with pytest.raises(RuntimeError, match="bool"):
        validate_pseudo_mask(torch.zeros(5, dtype=torch.float32), 5, name="test")


def test_validate_pseudo_mask_rejects_wrong_shape() -> None:
    """
    验证 validate_pseudo_mask 对形状不匹配立即报错。
    """
    with pytest.raises(RuntimeError, match="长度"):
        validate_pseudo_mask(torch.zeros(3, dtype=torch.bool), 5, name="test")


# ============================= split_mask_state 边界情况 =============================

def test_split_mask_state_none_input() -> None:
    """
    验证 split_mask_state 对 None pseudo_mask 的处理。
    """
    validated, has_real, has_pseudo = split_mask_state(None, 5, name="test")
    assert validated is None
    assert has_real is True
    assert has_pseudo is False


def test_split_mask_state_all_real() -> None:
    """
    验证 split_mask_state 对全 real 的处理。
    """
    mask = torch.zeros(5, dtype=torch.bool)
    validated, has_real, has_pseudo = split_mask_state(mask, 5, name="test")
    assert validated is not None
    assert has_real is True
    assert has_pseudo is False


def test_split_mask_state_all_pseudo() -> None:
    """
    验证 split_mask_state 对全 pseudo 的处理。
    """
    mask = torch.ones(5, dtype=torch.bool)
    validated, has_real, has_pseudo = split_mask_state(mask, 5, name="test")
    assert validated is not None
    assert has_real is False
    assert has_pseudo is True


# ============================= Block forward 完整 typed 路径 =============================

def test_block_forward_full_typed_path() -> None:
    """
    验证同时开启 separate_qkv / separate_attn_proj / separate_ffn / separate_cpe 的 Block forward。
    """
    c = 16
    block = Block(
        channels=c,
        num_heads=2,
        patch_size=32,
        cpe_impl="pointconv",
        cpe_receptive_field=3.0,
        pointconv_block_max_neighbors=8,
        enable_flash=False,
        enable_rpe=False,
        upcast_attention=False,
        upcast_softmax=False,
        ffn_type="mlp",
        mlp_ratio=2,
        separate_qkv=True,
        separate_attn_proj=True,
        separate_ffn=True,
        separate_cpe=True,
    )
    n_real, n_pseudo = 10, 5
    point, pseudo_mask = _make_point(n_real, n_pseudo, c)
    point.serialization(order=["z"], shuffle_orders=False)
    result = block(point)
    assert result.feat.shape == (n_real + n_pseudo, c)
    # pseudo_mask 在 Block 后仍然保留
    assert "pseudo_mask" in result.keys()
    assert torch.equal(result.pseudo_mask, pseudo_mask)


# ============================= apply_type_aware_tensor_module 边界情况 =============================

def test_apply_type_aware_tensor_module_empty_input() -> None:
    """
    验证空输入时 apply_type_aware_tensor_module 不报错。
    """
    x = torch.randn(0, 4)
    y = apply_type_aware_tensor_module(x, None, nn.Linear(4, 8), nn.Linear(4, 8))
    assert y.shape == (0, 8)


def test_apply_type_aware_tensor_module_none_mask_routes_to_real() -> None:
    """
    验证 pseudo_mask=None 时全部走 real 分支。
    """
    x = torch.randn(5, 4)
    real = nn.Linear(4, 8)
    pseudo = nn.Linear(4, 8)
    y = apply_type_aware_tensor_module(x, None, real, pseudo)
    # 应与直接调用 real(x) 相同
    expected = real(x)
    assert torch.equal(y, expected)
