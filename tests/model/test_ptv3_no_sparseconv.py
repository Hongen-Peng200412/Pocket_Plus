from __future__ import annotations

import pytest
import torch

from src.model.PTV3bakcbone.model import Block, Embedding, Point, PointConvCPE, PointConvEmbedding, PointTransformerV3


def test_block_rejects_sparseconv_cpe() -> None:
    """
    验证 Block 不再接受 sparseconv CPE。
    """
    with pytest.raises(ValueError, match="pointconv"):
        Block(
            channels=8,
            num_heads=2,
            patch_size=4,
            cpe_impl="sparseconv",
            enable_flash=False,
            upcast_attention=False,
            upcast_softmax=False,
        )


def test_embedding_rejects_sparseconv_impl() -> None:
    """
    验证 Embedding 不再接受 sparseconv stem。
    """
    with pytest.raises(ValueError, match="pointconv"):
        Embedding(
            in_channels=4,
            embed_channels=8,
            embedding_impl="sparseconv",
        )


def test_point_transformer_defaults_pointconv() -> None:
    """
    验证 PointTransformerV3 默认 embedding_impl 与 cpe_impl 均为 pointconv。
    """
    model = PointTransformerV3(
        in_channels=4,
        order=("z",),
        stride=(2,),
        enc_depths=(1, 1),
        enc_channels=(8, 16),
        enc_num_head=(2, 4),
        enc_patch_size=(8, 8),
        dec_depths=(1, 0),
        dec_channels=(8, 16),
        dec_num_head=(2, 4),
        dec_patch_size=(8, 8),
        enc_cpe_kernel_size=(5, 5),
        dec_cpe_kernel_size=(5, 5),
        enc_cpe_receptive_field=(2.0, 4.0),
        dec_cpe_receptive_field=(2.0, 4.0),
        enable_flash=False,
        upcast_attention=False,
        upcast_softmax=False,
    )
    assert model.embedding_impl == "pointconv"
    assert model.cpe_impl == "pointconv"


def test_ptv3_module_has_no_spconv_dependency() -> None:
    """
    验证新 PTV3 模块不再暴露 spconv 相关成员。
    """
    import src.model.PTV3bakcbone.model as ptv3_model

    assert not hasattr(ptv3_model, "spconv")
    model = PointTransformerV3(
        in_channels=4,
        order=("z",),
        stride=(2,),
        enc_depths=(1, 1),
        enc_channels=(8, 16),
        enc_num_head=(2, 4),
        enc_patch_size=(8, 8),
        dec_depths=(1, 0),
        dec_channels=(8, 16),
        dec_num_head=(2, 4),
        dec_patch_size=(8, 8),
        enc_cpe_kernel_size=(5, 5),
        dec_cpe_kernel_size=(5, 5),
        enc_cpe_receptive_field=(2.0, 4.0),
        dec_cpe_receptive_field=(2.0, 4.0),
        enable_flash=False,
        upcast_attention=False,
        upcast_softmax=False,
    )
    module_names = [module.__class__.__name__ for module in model.modules()]
    assert "SubMConv3d" not in module_names
    assert all("SparseConv" not in name for name in module_names)


def _make_point_input(num_points: int, channels: int) -> dict[str, torch.Tensor | float]:
    """
    构造单 BOX 的 PTV3 Point 输入字典。

    输入参数:
        - num_points: int, 点数 N
        - channels: int, 点特征通道数 C

    输出:
        - data_dict: dict[str, torch.Tensor | float], 包含 feat/coord/batch/grid_size
    """
    # torch.Tensor, (N, C), 线性排列的测试点特征
    feat = torch.arange(num_points * channels, dtype=torch.float32).view(num_points, channels) / 10.0
    # torch.Tensor, (N, 3), 让 radius graph 存在多个邻居的测试坐标
    coord = torch.stack(
        [
            torch.arange(num_points, dtype=torch.float32) * 0.2,
            torch.zeros(num_points, dtype=torch.float32),
            torch.zeros(num_points, dtype=torch.float32),
        ],
        dim=1,
    )
    return {
        "feat": feat,
        "coord": coord,
        "batch": torch.zeros(num_points, dtype=torch.long),
        "grid_size": 0.25,
    }


def _make_serialized_point(
    data_dict: dict[str, torch.Tensor | float],
    call_sparsify: bool,
) -> Point:
    """
    构造已 serialization 的 Point, 可选调用 no-op sparsify。

    输入参数:
        - data_dict: dict[str, torch.Tensor | float], Point 输入字典
        - call_sparsify: bool, 是否在 serialization 后显式调用 Point.sparsify()

    输出:
        - point: Point, 已包含 serialized_* 与 grid_coord 的点对象
    """
    point = Point({key: value.clone() if torch.is_tensor(value) else value for key, value in data_dict.items()})
    point.serialization(order=("z",), shuffle_orders=False)
    if call_sparsify:
        point.sparsify()
    return point


def _make_tiny_point_transformer() -> PointTransformerV3:
    """
    构造最小 pointconv PTV3 测试网络。

    输出:
        - model: PointTransformerV3, 两层 encoder + 一层 decoder 的小网络
    """
    return PointTransformerV3(
        in_channels=4,
        order=("z",),
        stride=(2,),
        enc_depths=(1, 1),
        enc_channels=(8, 16),
        enc_num_head=(2, 4),
        enc_patch_size=(8, 8),
        dec_depths=(1, 0),
        dec_channels=(8, 16),
        dec_num_head=(2, 4),
        dec_patch_size=(8, 8),
        enc_cpe_kernel_size=(5, 5),
        dec_cpe_kernel_size=(5, 5),
        enc_cpe_receptive_field=(1.0, 2.0),
        dec_cpe_receptive_field=(1.0, 2.0),
        embedding_receptive_field=1.0,
        pointconv_embed_max_neighbors=8,
        pointconv_block_max_neighbors=8,
        enable_flash=False,
        upcast_attention=False,
        upcast_softmax=False,
        drop_path=0.0,
        shuffle_orders=False,
    )


def test_point_sparsify_no_sparse_conv_feat() -> None:
    """
    验证 Point.sparsify 只补充 grid_coord, 不产生 sparse_conv_feat。
    """
    point = Point(_make_point_input(num_points=3, channels=4))
    point.sparsify()
    assert "grid_coord" in point
    assert "sparse_conv_feat" not in point


def test_pointconv_embedding_same_with_or_without_sparsify() -> None:
    """
    验证 pointconv embedding 不依赖 Point.sparsify 的副作用。
    """
    data_dict = _make_point_input(num_points=6, channels=4)
    point_without_sparsify = _make_serialized_point(data_dict, call_sparsify=False)
    point_with_sparsify = _make_serialized_point(data_dict, call_sparsify=True)
    embedding = PointConvEmbedding(
        in_channels=4,
        embed_channels=8,
        receptive_field=1.0,
        max_neighbors=8,
        norm_layer=None,
        act_layer=None,
    )
    output_without_sparsify = embedding(point_without_sparsify).feat
    output_with_sparsify = embedding(point_with_sparsify).feat
    assert torch.allclose(output_without_sparsify, output_with_sparsify, atol=1e-6)
    assert "sparse_conv_feat" not in point_without_sparsify
    assert "sparse_conv_feat" not in point_with_sparsify


def test_pointconv_cpe_same_with_or_without_sparsify() -> None:
    """
    验证 pointconv CPE 不依赖 Point.sparsify 的副作用。
    """
    data_dict = _make_point_input(num_points=6, channels=8)
    point_without_sparsify = _make_serialized_point(data_dict, call_sparsify=False)
    point_with_sparsify = _make_serialized_point(data_dict, call_sparsify=True)
    cpe = PointConvCPE(
        channels=8,
        receptive_field=1.0,
        max_neighbors=8,
        cache_key="test",
        norm_layer=None,
    )
    output_without_sparsify = cpe(point_without_sparsify).feat
    output_with_sparsify = cpe(point_with_sparsify).feat
    assert torch.allclose(output_without_sparsify, output_with_sparsify, atol=1e-6)
    assert "sparse_conv_feat" not in point_without_sparsify
    assert "sparse_conv_feat" not in point_with_sparsify


def test_block_pointconv_same_with_or_without_sparsify() -> None:
    """
    验证 pointconv CPE + attention Block 不依赖 Point.sparsify 的副作用。
    """
    torch.manual_seed(7)
    data_dict = _make_point_input(num_points=6, channels=8)
    point_without_sparsify = _make_serialized_point(data_dict, call_sparsify=False)
    point_with_sparsify = _make_serialized_point(data_dict, call_sparsify=True)
    block = Block(
        channels=8,
        num_heads=2,
        patch_size=4,
        cpe_impl="pointconv",
        cpe_receptive_field=1.0,
        pointconv_block_max_neighbors=8,
        enable_flash=False,
        upcast_attention=False,
        upcast_softmax=False,
        drop_path=0.0,
        order_index=0,
    )
    block.eval()
    output_without_sparsify = block(point_without_sparsify).feat
    output_with_sparsify = block(point_with_sparsify).feat
    assert torch.allclose(output_without_sparsify, output_with_sparsify, atol=1e-6)
    assert "sparse_conv_feat" not in point_without_sparsify
    assert "sparse_conv_feat" not in point_with_sparsify


def test_point_transformer_same_with_or_without_sparsify_after_serialization() -> None:
    """
    验证整网从已 serialization Point 开始时, 旧 no-op sparsify 不改变 pointconv PTV3 输出。
    """
    torch.manual_seed(11)
    data_dict = _make_point_input(num_points=8, channels=4)
    model = _make_tiny_point_transformer()
    model.eval()
    point_without_sparsify = _make_serialized_point(data_dict, call_sparsify=False)
    point_with_sparsify = _make_serialized_point(data_dict, call_sparsify=True)
    with torch.no_grad():
        output_without_sparsify = model.dec(model.enc(model.embedding(point_without_sparsify))).feat
        output_with_sparsify = model.dec(model.enc(model.embedding(point_with_sparsify))).feat
    assert torch.allclose(output_without_sparsify, output_with_sparsify, atol=1e-6)
    assert "sparse_conv_feat" not in point_without_sparsify
    assert "sparse_conv_feat" not in point_with_sparsify


def test_point_transformer_forward_keeps_grid_coord_without_sparsify() -> None:
    """
    验证 PointTransformerV3.forward 删除 point.sparsify 调用后仍保留 grid_coord 与输出点数。
    """
    torch.manual_seed(13)
    model = _make_tiny_point_transformer()
    model.eval()
    with torch.no_grad():
        output = model(_make_point_input(num_points=8, channels=4))
    assert output.feat.shape == (8, 8)
    assert "grid_coord" in output
    assert "sparse_conv_feat" not in output
