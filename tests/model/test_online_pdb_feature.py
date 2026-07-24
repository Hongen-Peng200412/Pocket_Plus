# -*- coding: utf-8 -*-
"""
test_online_pdb_feature.py — 在线 pdb_feature scatter 机制的单元测试
==================================================================
测试覆盖:
  1. set_input_channels: online_pdb_feature=True 时是否按当前 scatter 核正确加通道
  2. scatter_to_voxel_grid: 给定简单原子特征, 输出体素网格形状是否正确
  3. raw_pdb_grid 梯度隔离: 验证在线 scatter 不参与梯度回传
  4. point-only embed_head: online scatter 使用裁剪后的 raw atom_feat, 不使用 embed 后点特征
"""
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
import numpy as np

# ---- 被测模块 ----
from src.model.stage1_embed_head import Stage1EmbedHead, scatter_to_voxel_grid, soft_scatter_to_voxel_grid
from src.model.stage1_model import VolumePointStage1Model
import src.model.stage1_model as stage1_model_mod


# ==================================================================
# 辅助工厂函数
# ==================================================================
def _make_mock_voxel_backbone(expected_in_channels: int | None = None):
    """
    构建一个带 set_input_channels 的 mock voxel_backbone。

    输入参数:
        - expected_in_channels: int|None, 若非 None, 则在调用 set_input_channels 时校验实际值

    输出:
        - backbone: MagicMock, 模拟的体素分支模块
    """
    backbone = MagicMock(spec=torch.nn.Module)
    backbone.set_input_channels = MagicMock()
    if expected_in_channels is not None:
        def _check(actual):
            assert actual == expected_in_channels, (
                f"set_input_channels 期望 {expected_in_channels}, 实际 {actual}"
            )
        backbone.set_input_channels.side_effect = _check
    return backbone


def _make_minimal_stage1_model(
    online_pdb_feature: bool = False,
    online_pdb_feature_dim: int = 49,
    embed_head=None,
    online_pdb_feature_scatter_kernel: str = "legacy",
    online_pdb_feature_add_occupancy: bool = False,
    online_pdb_feature_add_centroid: bool = False,
):
    """
    构建只含 set_input_channels 相关逻辑的最小 Stage1Model 模拟对象。
    直接访问 Stage1Model 构造函数需要大量真实模块依赖, 此处用 SimpleNamespace 模拟关键属性。

    输入参数:
        - online_pdb_feature: bool, 是否启用在线 scatter
        - online_pdb_feature_dim: int, 原子特征维度, 建议值 49
        - embed_head: nn.Module|None, embed head 模块

    输出:
        - model: SimpleNamespace, 模拟对象, 含 set_input_channels 方法
    """
    from types import SimpleNamespace

    model = SimpleNamespace()
    model.embed_head = embed_head
    model.online_pdb_feature = bool(online_pdb_feature)
    model.online_pdb_feature_reduce = "sum"
    model.online_pdb_feature_scatter_kernel = str(online_pdb_feature_scatter_kernel)
    model.online_pdb_feature_add_occupancy = bool(online_pdb_feature_add_occupancy)
    model.online_pdb_feature_add_centroid = bool(online_pdb_feature_add_centroid)
    model.online_pdb_feature_dim = int(online_pdb_feature_dim)
    model.voxel_backbone = _make_mock_voxel_backbone()
    model.density_cube_encoder = None
    model.real_density_cube_encoder = None
    model._online_pdb_voxel_channels = VolumePointStage1Model._online_pdb_voxel_channels.__get__(model)
    model.set_input_channels = VolumePointStage1Model.set_input_channels.__get__(model)
    return model


# ==================================================================
# Test 1: set_input_channels 通道计算
# ==================================================================
class TestSetInputChannels:
    """验证 set_input_channels 在不同配置下的通道累加逻辑。"""

    def test_no_online_no_embed(self):
        """
        既不启用 online_pdb_feature 也不启用 embed_head 时,
        voxel_backbone 收到的通道数应等于数据集原始通道数。
        """
        model = _make_minimal_stage1_model(online_pdb_feature=False, embed_head=None)
        model.set_input_channels(1)
        model.voxel_backbone.set_input_channels.assert_called_once_with(1)

    def test_online_adds_49(self):
        """
        online_pdb_feature=True 且 embed_head=None 时,
        voxel_backbone 收到的通道数应为 data_ch + 49。
        """
        model = _make_minimal_stage1_model(online_pdb_feature=True, online_pdb_feature_dim=49)
        model.set_input_channels(1)
        model.voxel_backbone.set_input_channels.assert_called_once_with(1 + 49)

    def test_online_gauss27_adds_auxiliary_channels(self):
        """
        gauss27 且开启 occupancy/centroid 时,
        voxel_backbone 收到的通道数应为 data_ch + raw_dim + 2 + 3。
        """
        model = _make_minimal_stage1_model(
            online_pdb_feature=True,
            online_pdb_feature_dim=49,
            online_pdb_feature_scatter_kernel="gauss27",
            online_pdb_feature_add_occupancy=True,
            online_pdb_feature_add_centroid=True,
        )
        model.set_input_channels(56)
        model.voxel_backbone.set_input_channels.assert_called_once_with(56 + 49 + 2 + 3)

    def test_online_custom_dim(self):
        """
        当 online_pdb_feature_dim 为自定义值时, 通道计算也应正确。
        """
        model = _make_minimal_stage1_model(online_pdb_feature=True, online_pdb_feature_dim=32)
        model.set_input_channels(3)
        model.voxel_backbone.set_input_channels.assert_called_once_with(3 + 32)

    def test_embed_head_takes_precedence(self):
        """
        embed_head 启用时, 即使 online_pdb_feature=True,
        通道增量应来自 embed_head 而非 online scatter。
        """
        mock_eh = MagicMock()
        mock_eh.has_voxel_output = True
        mock_eh.embed_voxel_out_channels = 64
        mock_eh.add_occupancy_channels = True  # +2
        mock_eh.voxel_embed_as_tune = False
        model = _make_minimal_stage1_model(
            online_pdb_feature=True, online_pdb_feature_dim=49, embed_head=mock_eh
        )
        model.set_input_channels(1)
        # 应为 1 + 64 + 2 = 67, 不是 1 + 49 = 50
        model.voxel_backbone.set_input_channels.assert_called_once_with(67)

    def test_point_only_embed_head_keeps_online_scatter(self):
        """
        point-only embed_head 没有 voxel_pdb_embed_grid,
        因此 online scatter 仍应作为 voxel 分支的原子注入路径。
        """
        mock_eh = MagicMock()
        mock_eh.has_voxel_output = False
        mock_eh.embed_voxel_out_channels = 0
        mock_eh.add_occupancy_channels = False
        model = _make_minimal_stage1_model(
            online_pdb_feature=True,
            online_pdb_feature_dim=49,
            embed_head=mock_eh,
            online_pdb_feature_scatter_kernel="gauss27",
            online_pdb_feature_add_occupancy=True,
            online_pdb_feature_add_centroid=True,
        )
        model.set_input_channels(56)
        model.voxel_backbone.set_input_channels.assert_called_once_with(56 + 49 + 2 + 3)


# ==================================================================
# Test 2: scatter_to_voxel_grid 输出形状
# ==================================================================
class TestScatterToVoxelGrid:
    """验证 scatter_to_voxel_grid 在简单输入下的输出形状。"""

    def test_basic_shape(self):
        """
        3 个原子, 特征维度 49, 网格 (4, 4, 4), batch_size=1
        输出应为 (1, 49, 4, 4, 4)。
        """
        # torch.Tensor, (3, 49), 随机原子特征
        point_feat = torch.randn(3, 49)
        # torch.Tensor, (3, 3), 合法的体素坐标 (x, y, z)
        coord = torch.tensor([[1.5, 2.0, 0.5], [0.0, 0.0, 0.0], [3.0, 3.0, 3.0]])
        # torch.Tensor, (3,), batch 索引, 全在 batch 0
        batch_idx = torch.zeros(3, dtype=torch.long)
        # torch.Tensor, (1, 3), 体素网格尺寸 (Z, Y, X)
        box_shape = torch.tensor([[4, 4, 4]])

        out = scatter_to_voxel_grid(
            point_feat=point_feat,
            atom_coord_local_voxel=coord,
            point_batch=batch_idx,
            box_shape_zyx=box_shape,
            batch_size=1,
            reduce="sum",
            add_occupancy_channels=False,
        )
        assert out.shape == (1, 49, 4, 4, 4), f"输出形状错误: {out.shape}"

    def test_empty_atoms(self):
        """
        0 个原子时, 输出应为全零张量且形状正确。
        """
        point_feat = torch.zeros(0, 49)
        coord = torch.zeros(0, 3)
        batch_idx = torch.zeros(0, dtype=torch.long)
        box_shape = torch.tensor([[8, 8, 8]])

        out = scatter_to_voxel_grid(
            point_feat=point_feat,
            atom_coord_local_voxel=coord,
            point_batch=batch_idx,
            box_shape_zyx=box_shape,
            batch_size=1,
            reduce="sum",
        )
        assert out.shape == (1, 49, 8, 8, 8)
        assert (out == 0).all(), "0 个原子时输出应全零"

    def test_empty_atoms_with_occupancy_channels(self):
        """
        0 个原子且开启 occupancy 通道时, 输出仍应保留额外 2 个通道。
        """
        point_feat = torch.zeros(0, 49)
        coord = torch.zeros(0, 3)
        batch_idx = torch.zeros(0, dtype=torch.long)
        box_shape = torch.tensor([[8, 8, 8]])

        out = scatter_to_voxel_grid(
            point_feat=point_feat,
            atom_coord_local_voxel=coord,
            point_batch=batch_idx,
            box_shape_zyx=box_shape,
            batch_size=1,
            reduce="sum",
            add_occupancy_channels=True,
        )
        assert out.shape == (1, 51, 8, 8, 8)
        assert (out == 0).all(), "0 个原子时 occupancy 通道也应全零"

    def test_empty_atoms_with_soft_scatter_occupancy_channels(self):
        """
        soft scatter 的空输入路径应与 hard scatter 保持相同通道约定。
        """
        point_feat = torch.zeros(0, 16)
        coord = torch.zeros(0, 3)
        batch_idx = torch.zeros(0, dtype=torch.long)
        box_shape = torch.tensor([[6, 6, 6]])

        out = soft_scatter_to_voxel_grid(
            point_feat=point_feat,
            atom_coord_local_voxel=coord,
            point_batch=batch_idx,
            box_shape_zyx=box_shape,
            batch_size=1,
            reduce="sum",
            add_occupancy_channels=True,
        )
        assert out.shape == (1, 18, 6, 6, 6)
        assert (out == 0).all(), "soft scatter 空输入应返回全零张量"

    def test_out_of_bound_atoms_discarded(self):
        """
        超出体素网格范围的原子应被丢弃, 不影响输出形状。
        """
        point_feat = torch.ones(2, 10)
        # 第一个在范围内 (x=1, y=1, z=1), 第二个越界 (x=100)
        coord = torch.tensor([[1.0, 1.0, 1.0], [100.0, 0.0, 0.0]])
        batch_idx = torch.zeros(2, dtype=torch.long)
        box_shape = torch.tensor([[4, 4, 4]])

        out = scatter_to_voxel_grid(
            point_feat=point_feat,
            atom_coord_local_voxel=coord,
            point_batch=batch_idx,
            box_shape_zyx=box_shape,
            batch_size=1,
            reduce="sum",
        )
        assert out.shape == (1, 10, 4, 4, 4)
        # 只有 (x=1, y=1, z=1) 处的体素有值, 越界的被丢弃
        assert out[0, :, 1, 1, 1].sum() == 10.0, "范围内原子特征应被聚合"

    def test_multi_batch(self):
        """
        多 batch 情况下的 scatter。
        """
        # batch 0: 2 个原子; batch 1: 1 个原子
        point_feat = torch.randn(3, 5)
        coord = torch.tensor([[0.5, 0.5, 0.5], [1.5, 1.5, 1.5], [2.0, 2.0, 2.0]])
        batch_idx = torch.tensor([0, 0, 1])
        box_shape = torch.tensor([[4, 4, 4], [4, 4, 4]])

        out = scatter_to_voxel_grid(
            point_feat=point_feat,
            atom_coord_local_voxel=coord,
            point_batch=batch_idx,
            box_shape_zyx=box_shape,
            batch_size=2,
            reduce="sum",
        )
        assert out.shape == (2, 5, 4, 4, 4)

    def test_occupancy_channels(self):
        """
        add_occupancy_channels=True 时, 输出通道数应增加 2。
        """
        point_feat = torch.randn(2, 49)
        coord = torch.tensor([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]])
        batch_idx = torch.zeros(2, dtype=torch.long)
        box_shape = torch.tensor([[4, 4, 4]])

        out = scatter_to_voxel_grid(
            point_feat=point_feat,
            atom_coord_local_voxel=coord,
            point_batch=batch_idx,
            box_shape_zyx=box_shape,
            batch_size=1,
            reduce="sum",
            add_occupancy_channels=True,
        )
        assert out.shape == (1, 49 + 2, 4, 4, 4), f"occupancy 通道错误: {out.shape}"


# ==================================================================
# Test 2.5: embed head 裁剪到空点集
# ==================================================================
class TestStage1EmbedHeadEmptyTrim:
    """验证 embed head block 后裁剪为空时不会继续序列化空 Point。"""

    def test_run_blocks_returns_empty_state_after_trim_all_atoms(self):
        """
        当 buffer 裁剪移除全部原子时, _run_blocks_with_trim 应返回 None point 和空字段。
        """
        class IdentityBlock(torch.nn.Module):
            def forward(self, point):
                return point

        head = object.__new__(Stage1EmbedHead)
        point = SimpleNamespace(feat=torch.ones(2, 4))
        coord = torch.tensor([[10.0, 0.0, 0.0], [12.0, 0.0, 0.0]])
        batch = torch.zeros(2, dtype=torch.long)
        offset = torch.tensor([2], dtype=torch.long)
        core = torch.zeros(2, dtype=torch.bool)
        local_voxel = torch.tensor([[10.0, 0.0, 0.0], [12.0, 0.0, 0.0]])
        global_keep = torch.ones(2, dtype=torch.bool)

        (
            next_point,
            next_coord,
            next_batch,
            next_offset,
            next_core,
            next_local_voxel,
            next_global_keep,
        ) = head._run_blocks_with_trim(
            point=point,
            blocks=torch.nn.ModuleList([IdentityBlock()]),
            buffer_radii=(0.0,),
            cur_coord=coord,
            cur_batch=batch,
            cur_offset=offset,
            cur_core=core,
            cur_local_voxel=local_voxel,
            box_shape_zyx=torch.tensor([[4, 4, 4]]),
            voxel_size_world=torch.tensor([[1.0, 1.0, 1.0]]),
            global_keep_mask=global_keep,
        )

        assert next_point is None
        assert next_coord.shape == (0, 3)
        assert next_batch.shape == (0,)
        assert next_offset.tolist() == [0]
        assert next_core.shape == (0,)
        assert next_local_voxel.shape == (0, 3)
        assert next_global_keep.tolist() == [False, False]


def test_find1_voxel_only_matches_full_embed_voxel_branch() -> None:
    """验证 Find_1 非块式 MLP/centroid/residual/Gaussian 最短路径逐元素等价。"""

    torch.manual_seed(17)
    head = Stage1EmbedHead(
        atom_feature_dim=49,
        embed_hidden_dim=128,
        embed_voxel_out_channels=49,
        embed_point_out_channels=64,
        num_trunk_blocks=0,
        num_voxel_blocks=0,
        num_point_blocks=3,
        trunk_buffer_radii=(),
        voxel_buffer_radii=(),
        point_buffer_radii=(8.0, 4.0, 0.0),
        num_heads=4,
        patch_size=16,
        serialization_orders=("z",),
        shuffle_orders=False,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        enable_rpe=False,
        enable_flash=False,
        upcast_attention=False,
        upcast_softmax=False,
        scatter_reduce="sum",
        ffn_type="gated",
        mlp_ratio=3,
        act_layer_name="gelu",
        point_grid_size=0.25,
        cpe_impl="none",
        cpe_kernel_size=5,
        cpe_receptive_field=2.0,
        pointconv_block_max_neighbors=16,
        drop_path=0.0,
        pre_norm=True,
        embed_residual_enabled=True,
        embed_point_gate_enabled=False,
        embed_voxel_gate_enabled=False,
        add_occupancy_channels=True,
        use_soft_splatting=True,
        use_centroid_encoding=True,
        use_gaussian_splatting=True,
        voxel_embed_as_tune=False,
    )
    head.eval()
    atom_feat = torch.randn(4, 49)
    atom_coord_local = torch.tensor(
        [[1.25, 1.50, 1.75], [1.80, 1.20, 1.40], [15.50, 8.0, 8.0], [18.0, 8.0, 8.0]],
        dtype=torch.float32,
    )
    core = torch.tensor([True, True, True, False])
    batch_index = torch.zeros(4, dtype=torch.long)
    box_shape = torch.tensor([[16, 16, 16]], dtype=torch.long)
    centered_world = atom_coord_local - 8.0

    full = head(
        atom_feat=atom_feat,
        atom_coord_centered_world=centered_world,
        atom_batch_index=batch_index,
        atom_offsets=torch.tensor([4], dtype=torch.long),
        atom_coord_local_voxel=atom_coord_local,
        box_shape_zyx=box_shape,
        voxel_size_world=torch.ones(1, 3),
        atom_is_in_core_box=core,
    )["voxel_pdb_embed_grid"]
    short = head.forward_voxel_only(
        atom_feat=atom_feat,
        atom_coord_local_voxel=atom_coord_local,
        atom_batch_index=batch_index,
        box_shape_zyx=box_shape,
        atom_is_in_core_box=core,
    )

    assert full.shape == short.shape == (1, 51, 16, 16, 16)
    torch.testing.assert_close(short, full, rtol=0.0, atol=0.0)


def test_find2_voxel_tune_is_56d_without_voxel_raw_residual() -> None:
    """验证 Find_2 固定输出 56D Gaussian tune，并保留 point residual 路径。"""

    torch.manual_seed(23)
    head = Stage1EmbedHead(
        atom_feature_dim=49,
        embed_hidden_dim=128,
        embed_voxel_out_channels=49,
        embed_point_out_channels=64,
        num_trunk_blocks=0,
        num_voxel_blocks=0,
        num_point_blocks=3,
        trunk_buffer_radii=(),
        voxel_buffer_radii=(),
        point_buffer_radii=(8.0, 4.0, 0.0),
        num_heads=4,
        patch_size=16,
        serialization_orders=("z",),
        shuffle_orders=False,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        enable_rpe=False,
        enable_flash=False,
        upcast_attention=False,
        upcast_softmax=False,
        scatter_reduce="sum",
        ffn_type="gated",
        mlp_ratio=3,
        act_layer_name="gelu",
        point_grid_size=0.25,
        cpe_impl="none",
        cpe_kernel_size=5,
        cpe_receptive_field=2.0,
        pointconv_block_max_neighbors=16,
        drop_path=0.0,
        pre_norm=True,
        embed_residual_enabled=True,
        embed_point_gate_enabled=False,
        embed_voxel_gate_enabled=False,
        add_occupancy_channels=True,
        use_soft_splatting=True,
        use_centroid_encoding=True,
        use_gaussian_splatting=True,
        voxel_embed_as_tune=True,
    )
    head.eval()
    atom_feat = torch.randn(4, 49)
    atom_coord_local = torch.tensor(
        [[1.25, 1.50, 1.75], [1.80, 1.20, 1.40], [15.50, 8.0, 8.0], [18.0, 8.0, 8.0]],
        dtype=torch.float32,
    )
    core = torch.tensor([True, True, True, False])
    batch_index = torch.zeros(4, dtype=torch.long)
    box_shape = torch.tensor([[16, 16, 16]], dtype=torch.long)
    outputs = head(
        atom_feat=atom_feat,
        atom_coord_centered_world=atom_coord_local - 8.0,
        atom_batch_index=batch_index,
        atom_offsets=torch.tensor([4], dtype=torch.long),
        atom_coord_local_voxel=atom_coord_local,
        box_shape_zyx=box_shape,
        voxel_size_world=torch.ones(1, 3),
        atom_is_in_core_box=core,
    )
    short = head.forward_voxel_only(
        atom_feat=atom_feat,
        atom_coord_local_voxel=atom_coord_local,
        atom_batch_index=batch_index,
        box_shape_zyx=box_shape,
        atom_is_in_core_box=core,
    )

    assert head.embed_voxel_out_channels == 56
    assert head.add_occupancy_channels is False
    assert head.embed_voxel_add_proj is None
    assert head.embed_point_add_proj is not None
    assert outputs["embed_point_feat"].shape[1] == 64
    assert outputs["voxel_pdb_embed_grid"].shape == short.shape == (1, 56, 16, 16, 16)
    torch.testing.assert_close(short, outputs["voxel_pdb_embed_grid"], rtol=0.0, atol=0.0)


def test_find2_voxel_input_adds_tune_instead_of_concatenating() -> None:
    """验证 Find_2 的 56D voxel embed 与 density56 逐元素直接相加。"""

    model = SimpleNamespace()
    model.embed_head = SimpleNamespace(voxel_embed_as_tune=True)
    model.online_pdb_feature = False
    model._build_voxel_input = VolumePointStage1Model._build_voxel_input.__get__(model)
    density = torch.full((1, 56, 2, 2, 2), 2.0)
    tune = torch.full((1, 56, 2, 2, 2), 3.0)

    voxel_input = model._build_voxel_input(
        {"voxel_grid": density},
        {"voxel_pdb_embed_grid": tune},
    )

    assert voxel_input.shape == density.shape
    assert torch.equal(voxel_input, torch.full_like(density, 5.0))


# ==================================================================
# Test 3: 梯度隔离
# ==================================================================
class TestGradientIsolation:
    """验证在线 scatter 的输出不参与梯度回传。"""

    def test_no_grad_scatter(self):
        """
        在 torch.no_grad + detach 下执行 scatter,
        输出应 requires_grad=False。
        """
        point_feat = torch.randn(3, 49, requires_grad=True)
        coord = torch.tensor([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [0.5, 0.5, 0.5]])
        batch_idx = torch.zeros(3, dtype=torch.long)
        box_shape = torch.tensor([[4, 4, 4]])

        with torch.no_grad():
            out = scatter_to_voxel_grid(
                point_feat=point_feat.detach(),
                atom_coord_local_voxel=coord,
                point_batch=batch_idx,
                box_shape_zyx=box_shape,
                batch_size=1,
                reduce="sum",
            )
        assert not out.requires_grad, "在线 scatter 输出不应有梯度"


# ==================================================================
# Test 4: point-only embed head 下的 raw scatter 语义
# ==================================================================
class TestPointOnlyEmbedHeadOnlineScatter:
    """验证 point-only embed head 不会污染 online raw scatter 的输入。"""

    def test_build_voxel_input_uses_trimmed_raw_atom_feat(self, monkeypatch):
        """
        _run_embed_head_once 会把 batch["atom_feat"] 替换成 embed 后特征。
        但 online voxel scatter 的语义是 raw atom_feat, 因此 _build_voxel_input
        必须优先使用同步裁剪后的 _online_pdb_raw_atom_feat。
        """
        model = SimpleNamespace()
        model.online_pdb_feature = True
        model.online_pdb_feature_scatter_kernel = "gauss27"
        model.online_pdb_feature_sigma_voxel = 0.7
        model.online_pdb_feature_add_occupancy = True
        model.online_pdb_feature_add_centroid = True
        model.online_pdb_feature_reduce = "sum"
        model.online_pdb_feature_use_soft_splatting = False
        model._build_voxel_input = VolumePointStage1Model._build_voxel_input.__get__(model)

        raw_atom_feat = torch.full((2, 49), 3.0)
        embed_atom_feat = torch.full((2, 64), 7.0)
        batch = {
            "voxel_grid": torch.zeros(1, 56, 4, 4, 4),
            "atom_feat": embed_atom_feat,
            "_online_pdb_raw_atom_feat": raw_atom_feat,
            "atom_coord_local_voxel": torch.tensor([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]]),
            "atom_batch_index": torch.zeros(2, dtype=torch.long),
            "box_shape_zyx": torch.tensor([[4, 4, 4]], dtype=torch.long),
        }
        seen: dict[str, torch.Tensor] = {}

        def fake_gauss_scatter_to_voxel_grid(**kwargs):
            seen["point_feat"] = kwargs["point_feat"]
            return torch.ones(1, 54, 4, 4, 4)

        monkeypatch.setattr(stage1_model_mod, "gauss_scatter_to_voxel_grid", fake_gauss_scatter_to_voxel_grid)

        voxel_input = model._build_voxel_input(batch, embed_output=None)

        assert seen["point_feat"].shape == (2, 49)
        assert torch.equal(seen["point_feat"], raw_atom_feat)
        assert voxel_input.shape == (1, 56 + 54, 4, 4, 4)


# ==================================================================
# Test 5: split_and_select_box hardmask 逻辑
# ==================================================================
class TestHardmaskFromAtoms:
    """验证从 atom_coords 计算 hardmask 的逻辑正确性 (对应 split_and_select_box.py 修改)。"""

    def test_basic_hardmask(self):
        """
        给定若干原子坐标和体素网格参数,
        验证计算出的 hardmask 在对应体素位置为 True。
        """
        # 模拟参数
        origin_arr = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        voxel_size_arr = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        Z, Y, X = 4, 4, 4

        # np.ndarray, (3, 3), 3 个原子的世界坐标
        atom_coords = np.array([
            [0.5, 0.5, 0.5],   # -> voxel (0, 0, 0) -> hardmask[0, 0, 0] = True
            [2.3, 1.7, 3.1],   # -> voxel (2, 1, 3) -> hardmask[3, 1, 2] = True
            [10.0, 10.0, 10.0], # -> 越界, 应被丢弃
        ], dtype=np.float32)

        # 复现 split_and_select_box.py 中的 hardmask 计算逻辑
        hardmask_full = np.zeros((Z, Y, X), dtype=bool)
        voxel_ijk = np.floor((atom_coords - origin_arr) / voxel_size_arr).astype(int)
        valid = (
            (voxel_ijk[:, 0] >= 0) & (voxel_ijk[:, 0] < X) &
            (voxel_ijk[:, 1] >= 0) & (voxel_ijk[:, 1] < Y) &
            (voxel_ijk[:, 2] >= 0) & (voxel_ijk[:, 2] < Z)
        )
        v = voxel_ijk[valid]
        hardmask_full[v[:, 2], v[:, 1], v[:, 0]] = True

        # 验证: 两个合法原子对应的体素为 True
        assert hardmask_full[0, 0, 0] == True, "原子 (0.5,0.5,0.5) 应落在 [0,0,0]"
        assert hardmask_full[3, 1, 2] == True, "原子 (2.3,1.7,3.1) 应落在 [3,1,2]"
        # 验证: 总共只有 2 个体素被标记
        assert np.sum(hardmask_full) == 2, f"应有 2 个体素被标记, 实际 {np.sum(hardmask_full)}"
        # 验证: 越界原子没有被标记
        assert valid[2] == False, "第 3 个原子应被过滤"

    def test_empty_atoms(self):
        """
        当 atom_coords 为空时, hardmask 应全 False。
        """
        origin_arr = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        voxel_size_arr = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        Z, Y, X = 4, 4, 4

        atom_coords = np.zeros((0, 3), dtype=np.float32)

        hardmask_full = np.zeros((Z, Y, X), dtype=bool)
        if len(atom_coords) > 0:
            voxel_ijk = np.floor((atom_coords - origin_arr) / voxel_size_arr).astype(int)
            valid = (
                (voxel_ijk[:, 0] >= 0) & (voxel_ijk[:, 0] < X) &
                (voxel_ijk[:, 1] >= 0) & (voxel_ijk[:, 1] < Y) &
                (voxel_ijk[:, 2] >= 0) & (voxel_ijk[:, 2] < Z)
            )
            v = voxel_ijk[valid]
            hardmask_full[v[:, 2], v[:, 1], v[:, 0]] = True

        assert np.sum(hardmask_full) == 0, "空原子坐标应产生全 False hardmask"

    def test_non_unit_voxel_size(self):
        """
        当 voxel_size 非 1.0 时, 体素索引计算应正确缩放。
        """
        origin_arr = np.array([10.0, 20.0, 30.0], dtype=np.float32)
        voxel_size_arr = np.array([0.7, 0.7, 0.7], dtype=np.float32)
        Z, Y, X = 72, 72, 72

        # 原子在 origin + (1.05, 1.05, 1.05) -> voxel (1, 1, 1)
        atom_coords = np.array([[11.05, 21.05, 31.05]], dtype=np.float32)

        hardmask_full = np.zeros((Z, Y, X), dtype=bool)
        voxel_ijk = np.floor((atom_coords - origin_arr) / voxel_size_arr).astype(int)
        valid = (
            (voxel_ijk[:, 0] >= 0) & (voxel_ijk[:, 0] < X) &
            (voxel_ijk[:, 1] >= 0) & (voxel_ijk[:, 1] < Y) &
            (voxel_ijk[:, 2] >= 0) & (voxel_ijk[:, 2] < Z)
        )
        v = voxel_ijk[valid]
        hardmask_full[v[:, 2], v[:, 1], v[:, 0]] = True

        # (11.05 - 10.0) / 0.7 = 1.5 -> floor -> 1
        assert hardmask_full[1, 1, 1] == True
        assert np.sum(hardmask_full) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
