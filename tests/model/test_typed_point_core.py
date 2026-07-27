from __future__ import annotations

import pytest
import torch
from torch import nn

from src.model.PTV3bakcbone.model import Block, Embedding, Point, PointTransformerV3, SerializedPooling, SerializedUnpooling
from src.model.typed_point import apply_type_aware_tensor_module, normalize_typed_point_cfg


class _FailModule(nn.Module):
    """
    一调用就失败的测试模块. 
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        抛出测试异常. 

        输入参数:
            - x: torch.Tensor, 任意形状, 测试输入

        输出:
            - y: torch.Tensor, 不会返回
        """
        del x
        raise AssertionError("不应调用该分支")


class _AddConstant(nn.Module):
    """
    为输入张量加上固定常数的测试模块. 

    输入参数:
        - value: float, 加到输入张量上的常数

    前向输入:
        - x: torch.Tensor, 任意形状, 输入张量

    前向输出:
        - y: torch.Tensor, 与 x 相同, 加常数后的张量
    """

    def __init__(self, value: float) -> None:
        super().__init__()
        self.value = float(value)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.value


class _FloatAddConstant(_AddConstant):
    """
    模拟 AMP 下返回 float32 输出的 typed tensor 分支. 
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.float() + self.value


class _FloatSubsetModule(nn.Module):
    """
    模拟 AMP 下返回 float32 输出的 typed point 子图分支. 

    输入参数:
        - value: float, 加到子图特征上的常数
    """

    def __init__(self, value: float) -> None:
        super().__init__()
        self.value = float(value)

    def forward_subset(
        self,
        point: Point,
        keep_mask: torch.Tensor,
        type_name: str | None = None,
    ) -> torch.Tensor:
        """
        返回选中点的 float32 特征. 

        输入参数:
            - point: Point, mixed 点对象
            - keep_mask: torch.Tensor, (N_all,), bool, 当前子集掩码
            - type_name: str | None, CPE 接口携带的点类型名

        输出:
            - feat: torch.Tensor, (N_keep, C), float32, 加常数后的子集特征
        """
        del type_name
        return point.feat[keep_mask].float() + self.value


def _make_serialized_point(
    feat: torch.Tensor,
    coord: torch.Tensor,
    pseudo_mask: torch.Tensor | None,
) -> Point:
    """
    构造已序列化的单 BOX Point. 

    输入参数:
        - feat: torch.Tensor, (N_all, C), 点特征
        - coord: torch.Tensor, (N_all, 3), 点坐标
        - pseudo_mask: torch.Tensor | None, (N_all,), P anchor 掩码

    输出:
        - point: Point, 已包含 serialized_* 字段的点对象
    """
    # dict[str, Any], Point 构造输入字段
    point_dict = {
        "feat": feat.clone(),
        "coord": coord.clone(),
        "batch": torch.zeros(int(feat.shape[0]), dtype=torch.long),
        "offset": torch.tensor([int(feat.shape[0])], dtype=torch.long),
        "grid_size": 1.0,
    }
    if pseudo_mask is not None:
        # torch.Tensor, (N_all,), True 表示 P anchor
        point_dict["pseudo_mask"] = pseudo_mask.clone()
    point = Point(point_dict)
    point.serialization(order=("z",), shuffle_orders=False)
    return point


def test_typed_point_config_defaults_enabled() -> None:
    """
    默认 typed point 配置应启用所有 effective 分参开关. 
    """
    cfg = normalize_typed_point_cfg(None)

    assert cfg.enabled is True
    assert cfg.use_separate_qkv is True
    assert cfg.use_separate_attn_proj is True
    assert cfg.use_separate_ffn is True
    assert cfg.use_separate_cpe is True
    assert cfg.use_separate_embedding is True
    assert cfg.use_separate_pooling_proj is True
    assert cfg.use_separate_unpooling_proj is True
    assert cfg.use_separate_fusion is True
    assert cfg.use_separate_point_input_proj is True
    assert cfg.use_separate_atom_token_proj is True


def test_typed_point_config_enabled_false_gates_all_flags() -> None:
    """
    enabled=false 时所有 use_separate_* 都应关闭. 
    """
    cfg = normalize_typed_point_cfg({"enabled": False})

    assert cfg.use_separate_qkv is False
    assert cfg.use_separate_attn_proj is False
    assert cfg.use_separate_ffn is False
    assert cfg.use_separate_cpe is False
    assert cfg.use_separate_embedding is False
    assert cfg.use_separate_pooling_proj is False
    assert cfg.use_separate_unpooling_proj is False
    assert cfg.use_separate_fusion is False
    assert cfg.use_separate_point_input_proj is False
    assert cfg.use_separate_atom_token_proj is False


@pytest.mark.parametrize(
    "bad_cfg",
    [
        {"bad": True},
        {"separate_token_proj": True},
        {"enabled": "false"},
    ],
)
def test_typed_point_config_rejects_bad_configs(bad_cfg: dict[str, object]) -> None:
    """
    typed point 配置遇到未知字段或非 bool 值时应 fail-fast. 
    """
    with pytest.raises(ValueError):
        normalize_typed_point_cfg(bad_cfg)


def test_apply_type_aware_tensor_module_scatters_to_original_order() -> None:
    """
    type-aware tensor helper 应保持 mixed 输入顺序. 
    """
    # torch.Tensor, (4, 1), mixed 输入特征
    x = torch.arange(4, dtype=torch.float32).view(4, 1)
    # torch.Tensor, (4,), bool, 第 1/3 行走 pseudo 分支
    pseudo_mask = torch.tensor([False, True, False, True])

    y = apply_type_aware_tensor_module(
        x,
        pseudo_mask,
        _AddConstant(1.0),
        _AddConstant(-1.0),
    )

    assert torch.equal(y.squeeze(-1), torch.tensor([1.0, 0.0, 3.0, 2.0]))


def test_apply_type_aware_tensor_module_uses_branch_output_dtype() -> None:
    """
    mixed 恢复张量应采用 typed 分支输出精度, 而不是 bfloat16 输入精度. 
    """
    # torch.Tensor, (4, 1), 模拟 bf16 AMP 下的 mixed 输入特征
    x = torch.arange(4, dtype=torch.bfloat16).view(4, 1)
    # torch.Tensor, (4,), bool, 第 1/3 行走 pseudo 分支
    pseudo_mask = torch.tensor([False, True, False, True])

    y = apply_type_aware_tensor_module(
        x,
        pseudo_mask,
        _FloatAddConstant(1.0),
        _FloatAddConstant(-1.0),
    )

    assert y.dtype == torch.float32
    assert torch.equal(y.squeeze(-1), torch.tensor([1.0, 0.0, 3.0, 2.0]))


def test_apply_type_aware_tensor_module_skips_empty_branch() -> None:
    """
    all-real / all-pseudo 时不应调用空分支 module. 
    """
    # torch.Tensor, (3, 2), 测试输入特征
    x = torch.randn(3, 2)
    real = nn.Identity()
    pseudo = nn.Identity()

    y_real = apply_type_aware_tensor_module(x, torch.zeros(3, dtype=torch.bool), real, _FailModule())
    y_pseudo = apply_type_aware_tensor_module(x, torch.ones(3, dtype=torch.bool), _FailModule(), pseudo)

    assert torch.equal(y_real, x)
    assert torch.equal(y_pseudo, x)


def test_block_shared_cpe_keeps_input_residual() -> None:
    """
    shared CPE 路径应保留原始输入残差, 不应把 CPE delta 加两次. 
    """
    # torch.Tensor, (4, 8), 输入点特征
    feat = torch.randn(4, 8)
    # torch.Tensor, (4, 3), 足够接近以产生邻居边的点坐标
    coord = torch.arange(4, dtype=torch.float32).view(4, 1).repeat(1, 3) * 0.1
    point = _make_serialized_point(feat=feat, coord=coord, pseudo_mask=None)
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
        ffn_type="none",
        separate_qkv=False,
        separate_attn_proj=False,
        separate_ffn=False,
        separate_cpe=False,
    )

    cpe_delta = block._run_cpe(point, None)

    assert torch.allclose(point.feat, feat)
    assert cpe_delta.shape == feat.shape


def test_embedding_typed_all_real_and_all_pseudo_paths() -> None:
    """
    typed embedding 应支持 all-real 与 all-pseudo 两种显式 mask. 
    """
    pytest.importorskip("torch_cluster")
    # torch.Tensor, (4, 3), 同 BOX 点坐标
    coord = torch.arange(4, dtype=torch.float32).view(4, 1).repeat(1, 3) * 0.1
    # torch.Tensor, (4, 3), 输入点特征
    feat = torch.randn(4, 3)
    embedding = Embedding(
        in_channels=3,
        embed_channels=6,
        embedding_impl="pointconv",
        embedding_receptive_field=1.0,
        pointconv_embed_max_neighbors=8,
        norm_layer=nn.BatchNorm1d,
        act_layer=nn.GELU,
        separate_embedding=True,
    )

    real_point = _make_serialized_point(feat=feat, coord=coord, pseudo_mask=torch.zeros(4, dtype=torch.bool))
    pseudo_point = _make_serialized_point(feat=feat, coord=coord, pseudo_mask=torch.ones(4, dtype=torch.bool))

    assert embedding(real_point).feat.shape == (4, 6)
    assert embedding(pseudo_point).feat.shape == (4, 6)
    assert isinstance(embedding._pointconv_embed_real.norm, nn.LayerNorm)
    assert isinstance(embedding._pointconv_embed_pseudo.norm, nn.LayerNorm)


def test_embedding_typed_mixed_uses_subset_output_dtype() -> None:
    """
    typed embedding mixed 恢复应兼容 bf16 输入与 float32 子图输出. 
    """
    # torch.Tensor, (4, 3), bf16 mixed 输入特征
    feat = torch.arange(12, dtype=torch.bfloat16).view(4, 3)
    # torch.Tensor, (4, 3), 当前测试不参与替代子图模块计算的坐标占位
    coord = torch.zeros(4, 3)
    # torch.Tensor, (4,), bool, mixed P anchor 掩码
    pseudo_mask = torch.tensor([False, True, False, True])
    point = _make_serialized_point(feat=feat, coord=coord, pseudo_mask=pseudo_mask)
    embedding = Embedding(in_channels=3, embed_channels=3, separate_embedding=True)
    embedding._pointconv_embed_real = _FloatSubsetModule(1.0)
    embedding._pointconv_embed_pseudo = _FloatSubsetModule(-1.0)

    output = embedding(point).feat

    assert output.dtype == torch.float32
    assert torch.equal(output[~pseudo_mask], feat[~pseudo_mask].float() + 1.0)
    assert torch.equal(output[pseudo_mask], feat[pseudo_mask].float() - 1.0)


def test_block_typed_cpe_mixed_uses_subset_output_dtype() -> None:
    """
    typed CPE mixed 恢复应兼容 bf16 输入与 float32 子图输出. 
    """
    # torch.Tensor, (4, 8), bf16 mixed 输入特征
    feat = torch.arange(32, dtype=torch.bfloat16).view(4, 8)
    # torch.Tensor, (4, 3), 当前测试不参与替代子图模块计算的坐标占位
    coord = torch.zeros(4, 3)
    # torch.Tensor, (4,), bool, mixed P anchor 掩码
    pseudo_mask = torch.tensor([False, True, False, True])
    point = _make_serialized_point(feat=feat, coord=coord, pseudo_mask=pseudo_mask)
    block = Block(
        channels=8,
        num_heads=2,
        patch_size=4,
        cpe_impl="none",
        enable_flash=False,
        upcast_attention=False,
        upcast_softmax=False,
    )
    block.cpe_real = _FloatSubsetModule(1.0)
    block.cpe_pseudo = _FloatSubsetModule(-1.0)

    delta = block._run_cpe(point, pseudo_mask)

    assert delta is not None
    assert delta.dtype == torch.float32
    assert torch.equal(delta[~pseudo_mask], feat[~pseudo_mask].float() + 1.0)
    assert torch.equal(delta[pseudo_mask], feat[pseudo_mask].float() - 1.0)


def test_serialized_pooling_splits_same_cell_by_type_and_keeps_spatial_code() -> None:
    """
    typed pooling 应按 type 拆分同一空间 cell, 且 serialized_code 保持纯空间编码. 
    """
    # torch.Tensor, (2, 4), 输入点特征
    feat = torch.randn(2, 4)
    # torch.Tensor, (2, 3), 两个点处于同一 stride=2 pooled cell
    coord = torch.tensor([[0.0, 0.0, 0.0], [0.2, 0.0, 0.0]], dtype=torch.float32)
    # torch.Tensor, (2,), bool, real 与 pseudo 各一个
    pseudo_mask = torch.tensor([False, True])
    point = _make_serialized_point(feat=feat, coord=coord, pseudo_mask=pseudo_mask)
    pooling = SerializedPooling(
        in_channels=4,
        out_channels=5,
        stride=2,
        norm_layer=nn.BatchNorm1d,
        act_layer=nn.GELU,
        reduce="mean",
        shuffle_orders=False,
        traceable=True,
        separate_pooling_proj=True,
    )

    pooled = pooling(point)

    assert pooled.feat.shape == (2, 5)
    assert torch.equal(pooled.pseudo_mask, pseudo_mask)
    assert torch.equal(pooled.serialized_code[0], torch.zeros(2, dtype=pooled.serialized_code.dtype))
    assert torch.equal(pooled.pooling_inverse, torch.tensor([0, 1], dtype=pooled.pooling_inverse.dtype))


def test_serialized_unpooling_preserves_parent_pseudo_mask() -> None:
    """
    typed unpooling 输出应保留 parent 分辨率的 pseudo_mask. 
    """
    # torch.Tensor, (4, 4), 输入点特征
    feat = torch.randn(4, 4)
    # torch.Tensor, (4, 3), 两两进入 pooled cell 的点坐标
    coord = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.2, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [2.2, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    # torch.Tensor, (4,), bool, parent 分辨率 P anchor 掩码
    pseudo_mask = torch.tensor([False, False, True, True])
    point = _make_serialized_point(feat=feat, coord=coord, pseudo_mask=pseudo_mask)
    pooling = SerializedPooling(
        in_channels=4,
        out_channels=6,
        stride=2,
        norm_layer=None,
        act_layer=None,
        reduce="mean",
        shuffle_orders=False,
        traceable=True,
        separate_pooling_proj=True,
    )
    pooled = pooling(point)
    unpooling = SerializedUnpooling(
        in_channels=6,
        skip_channels=4,
        out_channels=4,
        norm_layer=nn.BatchNorm1d,
        act_layer=nn.GELU,
        traceable=False,
        separate_unpooling_proj=True,
    )

    unpooled = unpooling(pooled)

    assert unpooled.feat.shape == (4, 4)
    assert torch.equal(unpooled.pseudo_mask, pseudo_mask)


def test_point_transformer_preserves_pseudo_mask_through_encoder_decoder() -> None:
    """
    PointTransformerV3 mixed 前向应在 encoder/decoder 后保留原分辨率 pseudo_mask. 
    """
    pytest.importorskip("torch_cluster")
    torch.manual_seed(17)
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
        separate_qkv=True,
        separate_attn_proj=True,
        separate_ffn=True,
        separate_cpe=True,
        separate_embedding=True,
        separate_pooling_proj=True,
        separate_unpooling_proj=True,
    )
    model.eval()
    # torch.Tensor, (8, 4), mixed 输入点特征
    feat = torch.randn(8, 4)
    # torch.Tensor, (8, 3), 单 BOX 点坐标
    coord = torch.arange(8, dtype=torch.float32).view(8, 1).repeat(1, 3) * 0.1
    # torch.Tensor, (8,), bool, mixed P anchor 掩码
    pseudo_mask = torch.tensor([False, False, True, False, True, False, True, True])

    with torch.no_grad():
        output = model(
            {
                "feat": feat,
                "coord": coord,
                "batch": torch.zeros(8, dtype=torch.long),
                "offset": torch.tensor([8], dtype=torch.long),
                "grid_size": 1.0,
                "pseudo_mask": pseudo_mask,
            }
        )

    assert output.feat.shape == (8, 8)
    assert torch.equal(output.pseudo_mask, pseudo_mask)


def test_point_transformer_typed_flags_reach_core_modules() -> None:
    """
    PointTransformerV3 typed flags 应传递到 embedding、pooling、unpooling 与 Block. 
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
        enc_cpe_receptive_field=(1.0, 2.0),
        dec_cpe_receptive_field=(1.0, 2.0),
        enable_flash=False,
        upcast_attention=False,
        upcast_softmax=False,
        separate_qkv=True,
        separate_attn_proj=True,
        separate_ffn=True,
        separate_cpe=True,
        separate_embedding=True,
        separate_pooling_proj=True,
        separate_unpooling_proj=True,
    )

    blocks = [module for module in model.modules() if isinstance(module, Block)]
    poolings = [module for module in model.modules() if isinstance(module, SerializedPooling)]
    unpoolings = [module for module in model.modules() if isinstance(module, SerializedUnpooling)]

    assert model.embedding.separate_embedding is True
    assert all(block.attn.separate_qkv for block in blocks)
    assert all(block.attn.separate_attn_proj for block in blocks)
    assert all(block.separate_ffn for block in blocks)
    assert all(block.separate_cpe for block in blocks)
    assert all(pooling.separate_pooling_proj for pooling in poolings)
    assert all(unpooling.separate_unpooling_proj for unpooling in unpoolings)
