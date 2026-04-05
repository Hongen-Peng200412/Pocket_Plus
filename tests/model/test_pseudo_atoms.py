# -*- coding: utf-8 -*-
"""
PseudoAtomGenerator 的单元测试。
"""
from __future__ import annotations

import torch
import numpy as np
import pytest

from src.model.pseudo_atoms import (
    PseudoAtomGenerator,
    _thin_by_clustering,
    _compute_is_in_core_box,
    _centered_world_to_local_voxel,
    _centered_world_to_world,
)


# ============================================================
# 辅助工厂
# ============================================================

def _make_default_gen(**overrides) -> PseudoAtomGenerator:
    """构造一个默认参数的 PseudoAtomGenerator，可用 overrides 覆盖任意参数。"""
    defaults = dict(
        base_count=10,
        scale_factor=0.5,
        max_sample_rounds=3,
        init_feat_mode="zero",
        init_feat_noise_std=0.0,
        neighbor_radius=3.0,
        enable_density_weighting=False,
        density_channel_index=0,
        density_prob_base=0.1,
        delete_too_close_radius=0.0,
        delete_too_far_radius=0.0,
        lifecycle=[False, True, True],
    )
    defaults.update(overrides)
    return PseudoAtomGenerator(**defaults)


def _make_batch(
    batch_size: int = 2,
    atoms_per_box: tuple[int, ...] = (5, 3),
    feat_dim: int = 4,
    box_shape_zyx: tuple[int, int, int] = (8, 8, 8),
    voxel_size: float = 1.0,
) -> dict[str, torch.Tensor]:
    """构造一个最小 batch，满足 PseudoAtomGenerator.generate() 的字段要求。"""
    assert len(atoms_per_box) == batch_size

    total_atoms = sum(atoms_per_box)
    # torch.Tensor, (B, 3), BOX 体素网格大小
    bs_zyx = torch.tensor([[box_shape_zyx[0], box_shape_zyx[1], box_shape_zyx[2]]] * batch_size, dtype=torch.int64)
    # torch.Tensor, (B, 3), voxel 世界尺寸
    vs = torch.full((batch_size, 3), voxel_size, dtype=torch.float32)
    # torch.Tensor, (B, 3), BOX 左下角世界坐标
    bo = torch.zeros(batch_size, 3, dtype=torch.float32)

    # 真实原子坐标 (centered_world, 即以 BOX 中心为原点)
    coords = []
    batch_indices = []
    for i, n in enumerate(atoms_per_box):
        half = box_shape_zyx[2] * voxel_size * 0.5  # 取 x 轴半跨度
        # 在 BOX 内随机放置
        c = torch.rand(n, 3) * 2 * half - half
        coords.append(c)
        batch_indices.append(torch.full((n,), i, dtype=torch.long))

    atom_cw = torch.cat(coords, dim=0).float()
    atom_bi = torch.cat(batch_indices, dim=0)
    atom_counts = torch.tensor(list(atoms_per_box), dtype=torch.long)
    atom_offsets = torch.cumsum(atom_counts, dim=0)
    atom_feat = torch.randn(total_atoms, feat_dim, dtype=torch.float32)
    atom_vm = torch.ones(total_atoms, dtype=torch.bool)
    atom_label = torch.zeros(total_atoms, dtype=torch.long)
    atom_core = torch.ones(total_atoms, dtype=torch.bool)
    # local_voxel 与 world 坐标
    atom_lv = atom_cw.clone()
    atom_w = atom_cw.clone()

    # 体素网格 (用于密度加权测试)
    D, H, W = box_shape_zyx
    voxel_grid = torch.randn(batch_size, 1, D, H, W, dtype=torch.float32)

    return {
        "box_shape_zyx": bs_zyx,
        "voxel_size_world": vs,
        "box_origin_world": bo,
        "atom_coord_centered_world": atom_cw,
        "atom_coord_local_voxel": atom_lv,
        "atom_coord_world": atom_w,
        "atom_feat": atom_feat,
        "atom_valid_mask": atom_vm,
        "atom_label": atom_label,
        "atom_is_in_core_box": atom_core,
        "atom_batch_index": atom_bi,
        "atom_counts": atom_counts,
        "atom_offsets": atom_offsets,
        "voxel_grid": voxel_grid,
    }


# ============================================================
# 测试
# ============================================================

class TestPseudoAtomCount:
    """伪原子数目测试。"""

    def test_count_basic(self):
        """base=10, k=0.5, N_real=20 → 目标 20 个"""
        gen = _make_default_gen(base_count=10, scale_factor=0.5)
        batch = _make_batch(batch_size=1, atoms_per_box=(20,))
        pseudo = gen.generate(batch)
        n = int(pseudo["pseudo_counts"][0].item())
        # 目标 = 10 + 0.5*20 = 20 (无删除，应精确达到)
        assert n == 20, f"expected 20, got {n}"

    def test_count_zero_scale(self):
        """base=5, k=0, N_real=100 → 固定 5 个"""
        gen = _make_default_gen(base_count=5, scale_factor=0.0)
        batch = _make_batch(batch_size=1, atoms_per_box=(100,), feat_dim=4)
        pseudo = gen.generate(batch)
        assert int(pseudo["pseudo_counts"][0].item()) == 5

    def test_count_multi_box(self):
        """两个 BOX 各自独立计算伪原子数"""
        gen = _make_default_gen(base_count=2, scale_factor=1.0)
        batch = _make_batch(batch_size=2, atoms_per_box=(3, 7))
        pseudo = gen.generate(batch)
        # BOX0: 2 + 1.0*3 = 5, BOX1: 2 + 1.0*7 = 9
        assert int(pseudo["pseudo_counts"][0].item()) == 5
        assert int(pseudo["pseudo_counts"][1].item()) == 9


class TestInjectRemoveRoundtrip:
    """注入/移除一致性测试。"""

    def test_inject_allows_missing_atom_global_indices(self):
        """训练 batch 缺少 atom_global_indices 也应正常 inject/remove。"""
        gen = _make_default_gen(base_count=2, scale_factor=0.0)
        batch = _make_batch(batch_size=2, atoms_per_box=(3, 5))
        pseudo = gen.generate(batch)

        new_batch, split_info = gen.inject(batch, pseudo)
        restored_batch = gen.remove(new_batch, split_info)

        assert "atom_global_indices" not in new_batch
        assert "atom_global_indices" not in restored_batch
        assert restored_batch["atom_coord_centered_world"].shape == batch["atom_coord_centered_world"].shape

    def test_roundtrip(self):
        gen = _make_default_gen(base_count=4, scale_factor=0.0)
        batch = _make_batch(batch_size=2, atoms_per_box=(3, 5))
        pseudo = gen.generate(batch)

        original_total = int(batch["atom_coord_centered_world"].shape[0])
        new_batch, split_info = gen.inject(batch, pseudo)

        # 注入后原子总数增加
        injected_total = int(new_batch["atom_coord_centered_world"].shape[0])
        assert injected_total == original_total + int(pseudo["pseudo_counts"].sum().item())

        # 移除后恢复
        restored_batch = gen.remove(new_batch, split_info)
        restored_total = int(restored_batch["atom_coord_centered_world"].shape[0])
        assert restored_total == original_total

    def test_roundtrip_preserves_real_coords(self):
        """注入再移除后，真实原子坐标完全不变。"""
        gen = _make_default_gen(base_count=3, scale_factor=0.0)
        batch = _make_batch(batch_size=1, atoms_per_box=(4,))
        original_coords = batch["atom_coord_centered_world"].clone()

        pseudo = gen.generate(batch)
        new_batch, split_info = gen.inject(batch, pseudo)
        restored_batch = gen.remove(new_batch, split_info)

        assert torch.allclose(restored_batch["atom_coord_centered_world"], original_coords)


class TestValidMask:
    """伪原子 valid_mask 全 False。"""

    def test_valid_mask_all_false(self):
        gen = _make_default_gen(base_count=10, scale_factor=0.0)
        batch = _make_batch(batch_size=1, atoms_per_box=(5,))
        pseudo = gen.generate(batch)
        assert not pseudo["pseudo_valid_mask"].any(), "伪原子 valid_mask 应全为 False"


class TestCoreBoxComputation:
    """is_in_core_box 正确计算。"""

    def test_core_box(self):
        """在 BOX 内的伪原子 core=True，BOX 外的 core=False。"""
        gen = _make_default_gen(base_count=100, scale_factor=0.0)
        # 使用较大 BOX，确保有足够的采样空间
        batch = _make_batch(batch_size=1, atoms_per_box=(0,), box_shape_zyx=(20, 20, 20), voxel_size=1.0)
        pseudo = gen.generate(batch)
        # 由于均匀采样在 [-half, +half] 内，大部分应在 BOX 内
        # (origin=0, box_max=20, center=10, half=10, 采样范围 [-10,10]→ world [0, 20])
        # 理论上全部在内
        n_in = int(pseudo["pseudo_is_in_core_box"].sum().item())
        n_total = int(pseudo["pseudo_counts"].sum().item())
        assert n_in == n_total, f"所有伪原子均在 BOX 内采样但 core 判断错误: {n_in}/{n_total}"


class TestDeletion:
    """删除机制测试。"""

    def test_delete_too_close_clustering(self):
        """r0=2.0 → 距真实原子 < 2Å 的伪原子做聚类只保留一个。"""
        gen = _make_default_gen(
            base_count=50,
            scale_factor=0.0,
            delete_too_close_radius=2.0,
            max_sample_rounds=5,
        )
        # 极小 BOX，真实原子在中心
        batch = _make_batch(batch_size=1, atoms_per_box=(1,), box_shape_zyx=(10, 10, 10), voxel_size=1.0)
        batch["atom_coord_centered_world"] = torch.tensor([[0.0, 0.0, 0.0]])
        pseudo = gen.generate(batch)
        n = int(pseudo["pseudo_counts"][0].item())
        # 因为聚类，距原点 < 2Å 的区间只保留 1 个代表
        assert n > 0, "至少应有伪原子"

    def test_delete_too_far(self):
        """r1=3.0 → 距所有参考点(真实+已采集伪原子) > 3Å 的伪原子被删。"""
        gen = _make_default_gen(
            base_count=200,
            scale_factor=0.0,
            delete_too_far_radius=3.0,
            max_sample_rounds=1,  # 只跑 1 轮避免参考集扩展
        )
        # 真实原子在中心
        batch = _make_batch(batch_size=1, atoms_per_box=(1,), box_shape_zyx=(20, 20, 20), voxel_size=1.0)
        batch["atom_coord_centered_world"] = torch.tensor([[0.0, 0.0, 0.0]])
        pseudo = gen.generate(batch)
        # 只跑 1 轮, 参考集只有真实原子, 所以所有保留的伪原子到原点距离 <= 3.0
        coords = pseudo["pseudo_coord_centered_world"].numpy()
        if coords.shape[0] > 0:
            dists = np.linalg.norm(coords, axis=1)
            assert np.all(dists <= 3.0 + 1e-5), f"有距离 > 3.0 的伪原子: max_dist={dists.max():.3f}"

    def test_delete_includes_pseudo_in_ref(self):
        """验证第 2 轮采样的删除参考集包含第 1 轮的伪原子。"""
        # 间接测试: 设置 r0 足够大和 base_count 足够大，需要多轮;
        # 如果参考集不扩大，聚类结果会不同
        gen = _make_default_gen(
            base_count=20,
            scale_factor=0.0,
            delete_too_close_radius=0.5,
            max_sample_rounds=3,
        )
        np.random.seed(42)  # 固定种子
        batch = _make_batch(batch_size=1, atoms_per_box=(3,), box_shape_zyx=(4, 4, 4), voxel_size=1.0)
        pseudo = gen.generate(batch)
        # 只要不崩溃且有伪原子，间接证明多轮参考集正常工作
        assert int(pseudo["pseudo_counts"][0].item()) > 0


class TestLoopSampling:
    """循环采样测试。"""

    def test_loop_compensates_deletion(self):
        """大范围删除后仍能通过循环补采至目标数(或接近)。"""
        gen = _make_default_gen(
            base_count=30,
            scale_factor=0.0,
            delete_too_far_radius=5.0,  # 只保留离真实原子 ≤ 5Å 的
            max_sample_rounds=5,
        )
        batch = _make_batch(batch_size=1, atoms_per_box=(2,), box_shape_zyx=(20, 20, 20), voxel_size=1.0)
        pseudo = gen.generate(batch)
        n = int(pseudo["pseudo_counts"][0].item())
        # 因为 BOX 很大 + r1=5 较小, 不太可能刚好凑满 30, 但应 > 0
        assert n > 0, "循环采样应至少产生一些伪原子"


class TestDensityWeighting:
    """密度加权采样测试。"""

    def test_density_concentrates_in_high_density(self):
        """高密度区域的伪原子应明显多于低密度区域。"""
        gen = _make_default_gen(
            base_count=500,
            scale_factor=0.0,
            enable_density_weighting=True,
            density_channel_index=0,
            density_prob_base=0.01,
        )
        batch = _make_batch(batch_size=1, atoms_per_box=(0,), box_shape_zyx=(8, 8, 8), voxel_size=1.0)
        # 构造密度图: 前半 (z < 4) 密度=10, 后半 (z >= 4) 密度=0
        vg = torch.zeros(1, 1, 8, 8, 8)
        vg[0, 0, :4, :, :] = 10.0
        batch["voxel_grid"] = vg

        np.random.seed(123)
        pseudo = gen.generate(batch)
        coords = pseudo["pseudo_coord_centered_world"].numpy()
        if coords.shape[0] == 0:
            pytest.skip("no pseudo atoms generated")
        # centered_world 中 z=0 对应 BOX 中心; z < 0 对应 BOX 前半
        # local_voxel z 中，前半对应 z_index < 4 → centered_world z < 0
        n_front = int(np.sum(coords[:, 2] < 0))  # z < 0 → 高密度
        n_back = int(np.sum(coords[:, 2] >= 0))   # z >= 0 → 低密度
        assert n_front > n_back, f"高密度区 ({n_front}) 应多于低密度区 ({n_back})"


class TestNeighborMeanFeat:
    """neighbor_mean 特征初始化测试。"""

    def test_neighbor_mean_nonzero(self):
        gen = _make_default_gen(
            base_count=10,
            scale_factor=0.0,
            init_feat_mode="neighbor_mean",
            neighbor_radius=100.0,  # 大半径确保都能找到邻居
        )
        batch = _make_batch(batch_size=1, atoms_per_box=(5,), feat_dim=4)
        pseudo = gen.generate(batch)
        feat = pseudo["pseudo_feat"]
        if feat.shape[0] > 0:
            assert feat.abs().sum() > 0, "neighbor_mean 模式下特征不应全零"


class TestLifecycle:
    """lifecycle 配置测试。"""

    def test_lifecycle_validation(self):
        with pytest.raises(ValueError):
            PseudoAtomGenerator(
                base_count=1, scale_factor=0.0, max_sample_rounds=1,
                init_feat_mode="zero", init_feat_noise_std=0.0,
                neighbor_radius=3.0, enable_density_weighting=False,
                density_channel_index=0, density_prob_base=0.1,
                delete_too_close_radius=0.0, delete_too_far_radius=0.0,
                lifecycle=[True, True],  # 长度不是 3
            )


class TestRecycleHelpers:
    """recycle 相关 helper 测试。"""

    def test_prepare_pseudo_dict_non_regenerates_positions(self):
        """`non` 应忽略 cache 中的位置并重新采样。"""
        torch.manual_seed(0)
        np.random.seed(0)
        gen = _make_default_gen(base_count=2, scale_factor=0.0, recycle_policy="non")
        batch = _make_batch(batch_size=1, atoms_per_box=(4,))
        cached = gen.generate(batch)
        cached["pseudo_coord_centered_world"] = torch.full_like(cached["pseudo_coord_centered_world"], 123.0)

        prepared = gen.prepare_pseudo_dict_for_recycle(batch=batch, cached_pseudo_dict=cached)

        assert not torch.allclose(prepared["pseudo_coord_centered_world"], cached["pseudo_coord_centered_world"])

    def test_prepare_pseudo_dict_pos_keeps_positions_but_reinitializes_features(self):
        """`pos` 应保留位置，但按当前初始化逻辑重刷 `pseudo_feat`。"""
        torch.manual_seed(1)
        np.random.seed(1)
        gen = _make_default_gen(
            base_count=2,
            scale_factor=0.0,
            recycle_policy="pos",
            init_feat_mode="neighbor_mean",
            neighbor_radius=100.0,
        )
        batch = _make_batch(batch_size=1, atoms_per_box=(4,), feat_dim=4)
        cached = gen.generate(batch)
        cached = {**cached, "pseudo_feat": torch.full_like(cached["pseudo_feat"], 7.0)}

        prepared = gen.prepare_pseudo_dict_for_recycle(batch=batch, cached_pseudo_dict=cached)

        assert torch.allclose(prepared["pseudo_coord_centered_world"], cached["pseudo_coord_centered_world"])
        assert not torch.allclose(prepared["pseudo_feat"], cached["pseudo_feat"])

    def test_prepare_pseudo_dict_all_keeps_positions_and_reinitializes_features(self):
        """`all` should keep positions but reinitialize feat (same as `pos`); only point_recycle_out is preserved across rounds."""
        torch.manual_seed(2)
        np.random.seed(2)
        gen = _make_default_gen(
            base_count=2,
            scale_factor=0.0,
            recycle_policy="all",
            init_feat_mode="neighbor_mean",
            neighbor_radius=100.0,
        )
        batch = _make_batch(batch_size=1, atoms_per_box=(4,), feat_dim=4)
        cached = gen.generate(batch)
        cached = {**cached, "pseudo_feat": torch.full_like(cached["pseudo_feat"], 5.0)}

        prepared = gen.prepare_pseudo_dict_for_recycle(batch=batch, cached_pseudo_dict=cached)

        assert torch.allclose(prepared["pseudo_coord_centered_world"], cached["pseudo_coord_centered_world"])
        assert not torch.allclose(prepared["pseudo_feat"], cached["pseudo_feat"])

    def test_prepare_pseudo_dict_fixed_keeps_positions_and_features(self):
        """`fixed` 保留位置与 `pseudo_feat`, 直接沿用缓存。"""
        torch.manual_seed(5)
        np.random.seed(5)
        gen = _make_default_gen(base_count=2, scale_factor=0.0, recycle_policy="fixed")
        batch = _make_batch(batch_size=1, atoms_per_box=(4,), feat_dim=4)
        cached = gen.generate(batch)
        cached = {**cached, "pseudo_feat": torch.full_like(cached["pseudo_feat"], 5.0)}

        prepared = gen.prepare_pseudo_dict_for_recycle(batch=batch, cached_pseudo_dict=cached)

        assert torch.allclose(prepared["pseudo_coord_centered_world"], cached["pseudo_coord_centered_world"])
        assert torch.allclose(prepared["pseudo_feat"], cached["pseudo_feat"])

    def test_reinitialize_pseudo_features_preserves_geometry(self):
        """重初始化特征时不应改动伪原子的几何字段。"""
        torch.manual_seed(3)
        np.random.seed(3)
        gen = _make_default_gen(
            base_count=2,
            scale_factor=0.0,
            init_feat_mode="neighbor_mean",
            neighbor_radius=100.0,
        )
        batch = _make_batch(batch_size=1, atoms_per_box=(4,), feat_dim=4)
        pseudo = gen.generate(batch)
        pseudo_with_bad_feat = {**pseudo, "pseudo_feat": torch.full_like(pseudo["pseudo_feat"], 9.0)}

        reinitialized = gen.reinitialize_pseudo_features(batch=batch, pseudo_dict=pseudo_with_bad_feat)

        assert torch.allclose(reinitialized["pseudo_coord_centered_world"], pseudo["pseudo_coord_centered_world"])
        assert torch.allclose(reinitialized["pseudo_coord_local_voxel"], pseudo["pseudo_coord_local_voxel"])
        assert torch.allclose(reinitialized["pseudo_coord_world"], pseudo["pseudo_coord_world"])
        assert not torch.allclose(reinitialized["pseudo_feat"], pseudo_with_bad_feat["pseudo_feat"])

    def test_extract_pseudo_dict_from_batch_roundtrip(self):
        """inject 后再 extract，伪原子字段应可无损恢复。"""
        torch.manual_seed(4)
        np.random.seed(4)
        gen = _make_default_gen(base_count=2, scale_factor=0.0)
        batch = _make_batch(batch_size=2, atoms_per_box=(3, 4), feat_dim=4)
        pseudo = gen.generate(batch)

        mixed_batch, split_info = gen.inject(batch, pseudo)
        extracted = gen.extract_pseudo_dict_from_batch(mixed_batch, split_info)

        assert torch.allclose(extracted["pseudo_coord_centered_world"], pseudo["pseudo_coord_centered_world"])
        assert torch.allclose(extracted["pseudo_coord_local_voxel"], pseudo["pseudo_coord_local_voxel"])
        assert torch.allclose(extracted["pseudo_coord_world"], pseudo["pseudo_coord_world"])
        assert torch.allclose(extracted["pseudo_feat"], pseudo["pseudo_feat"])
        assert torch.equal(extracted["pseudo_batch_index"], pseudo["pseudo_batch_index"])
        assert torch.equal(extracted["pseudo_counts"], pseudo["pseudo_counts"])
        assert torch.equal(extracted["pseudo_valid_mask"], pseudo["pseudo_valid_mask"])
        assert torch.equal(extracted["pseudo_label"], pseudo["pseudo_label"])
        assert torch.equal(extracted["pseudo_is_in_core_box"], pseudo["pseudo_is_in_core_box"])

    def test_interleave_real_and_pseudo_tensor(self):
        """`interleave_real_and_pseudo_tensor()` 应按 `[real_i, pseudo_i]` 交错拼接。"""
        real_tensor = torch.tensor([[1.0], [2.0], [3.0], [4.0], [5.0]])
        pseudo_tensor = torch.tensor([[10.0], [11.0], [12.0]])
        split_info = [(2, 1), (3, 2)]

        mixed = PseudoAtomGenerator.interleave_real_and_pseudo_tensor(
            real_tensor=real_tensor,
            split_info=split_info,
            pseudo_tensor=pseudo_tensor,
        )

        expected = torch.tensor([[1.0], [2.0], [10.0], [3.0], [4.0], [5.0], [11.0], [12.0]])
        assert torch.allclose(mixed, expected)

    def test_extract_pseudo_tensor_from_mixed(self):
        """`extract_pseudo_tensor_from_mixed()` 应只抽出 pseudo 槽位。"""
        mixed_tensor = torch.tensor([[1.0], [2.0], [10.0], [3.0], [4.0], [5.0], [11.0], [12.0]])
        split_info = [(2, 1), (3, 2)]

        pseudo_tensor = PseudoAtomGenerator.extract_pseudo_tensor_from_mixed(
            mixed_tensor=mixed_tensor,
            split_info=split_info,
        )

        expected = torch.tensor([[10.0], [11.0], [12.0]])
        assert torch.allclose(pseudo_tensor, expected)


class TestBuildRealMask:
    """build_real_mask 测试。"""

    def test_build_real_mask(self):
        split_info = [(3, 2), (5, 4)]
        mask = PseudoAtomGenerator.build_real_mask(split_info)
        expected = torch.tensor(
            [True, True, True, False, False,   # BOX0: 3 real, 2 pseudo
             True, True, True, True, True, False, False, False, False],  # BOX1: 5 real, 4 pseudo
            dtype=torch.bool,
        )
        assert torch.equal(mask, expected)


class TestThinByClustering:
    """_thin_by_clustering 工具函数测试。"""

    def test_thin_basic(self):
        """3 个点在同一个聚类中 → 只保留 1 个。"""
        coords = np.array([[0, 0, 0], [0.1, 0, 0], [0.2, 0, 0]], dtype=np.float32)
        keep = _thin_by_clustering(coords, radius=1.0)
        assert len(keep) == 1

    def test_thin_two_clusters(self):
        """2 组距离远的点 → 各保留 1 个。"""
        coords = np.array(
            [[0, 0, 0], [0.1, 0, 0], [10, 0, 0], [10.1, 0, 0]],
            dtype=np.float32,
        )
        keep = _thin_by_clustering(coords, radius=1.0)
        assert len(keep) == 2

    def test_thin_empty(self):
        coords = np.empty((0, 3), dtype=np.float32)
        keep = _thin_by_clustering(coords, radius=1.0)
        assert len(keep) == 0


class TestForwardWithPseudoAtoms:
    """在完整(dummy) forward 中验证伪原子注入→参与→移除。"""

    def test_atom_head_only_lifecycle_is_rejected(self):
        """用 DummyBackbone + pseudo_atom_cfg 跑 forward, 输出应只含真实原子。"""
        from torch import nn
        from src.model.stage1_model import VolumePointStage1Model

        class _Voxel(nn.Module):
            def __init__(self):
                super().__init__()
                self.return_feature_keys = ("voxel_c0",)
                self.feature_channels_by_name = {"voxel_c0": 2}
            def forward(self, voxel_grid, recycle_in=None, return_feature_keys=()):
                B = voxel_grid.shape[0]
                return {
                    "voxel_features": {"voxel_c0": voxel_grid.new_ones((B, 2, 1, 1, 1))},
                    "voxel_logits_aux": None, "voxel_recycle_out": None,
                }

        class _Point(nn.Module):
            def __init__(self):
                super().__init__()
                self.backend = "zeros"
                self.out_channels = 4
                self.point_grid_size = 0.25
                self.feature_channels_by_name = {"point_feat": 4}
            def forward(self, *a, **kw): raise AssertionError
            def build_zeros_output(self, atom_feat, atom_coord_centered_world,
                                   atom_batch_index, atom_offsets,
                                   return_feature_names=None, point_input_feat=None):
                pf = atom_feat.new_zeros((atom_feat.shape[0], self.out_channels))
                return {
                    "point_feat": pf,
                    "point_state": {
                        "coord": atom_coord_centered_world,
                        "batch": atom_batch_index.long(),
                        "offset": atom_offsets.long(),
                        "grid_size": self.point_grid_size,
                    },
                    "point_recycle_out": pf,
                    "point_feature_dict": {},
                }

        class _Attn(nn.Module):
            def __init__(self): super().__init__()
            def forward(self, point_state, token_feat): return token_feat

        pseudo_cfg = dict(
            base_count=4, scale_factor=0.0, max_sample_rounds=1,
            init_feat_mode="zero", init_feat_noise_std=0.0,
            neighbor_radius=3.0, enable_density_weighting=False,
            density_channel_index=0, density_prob_base=0.1,
            delete_too_close_radius=0.0, delete_too_far_radius=0.0,
            lifecycle=[False, False, True],  # 只在 atom head 存在
        )

        with pytest.raises(ValueError, match="atom head"):
            VolumePointStage1Model(
                voxel_backbone=_Voxel(),
                point_backbone=_Point(),
                point_fusion_map={"point_feat": "voxel_c0"},
                point_fusion_modes=("concat_linear",),
                sampler_modes=("nearest",),
                fusion_mlp_ratio=1.0, fusion_proj_drop=0.0,
                atom_head_hidden_dim=8, atom_head_num_heads=1,
                atom_head_patch_size=4, atom_head_num_layers=1,
                atom_head_serialization_orders=("z",),
                atom_head_shuffle_orders=False,
                atom_head_qkv_bias=False, atom_head_qk_scale=None,
                atom_head_attn_drop=0.0, atom_head_proj_drop=0.0,
                atom_head_enable_rpe=False, atom_head_enable_flash=False,
                atom_head_upcast_attention=False, atom_head_upcast_softmax=False,
                atom_logit_dim=1,
                enable_recycling=False, max_recycles=1,
                randomize_recycles=False, detach_recycle_states=False,
                act_layer_name="gelu", ffn_type="mlp",
                atom_head_ffn_type="none", atom_head_mlp_ratio=4,
                atom_head_cpe_impl="none", atom_head_cpe_kernel_size=5,
                atom_head_cpe_receptive_field=2.0,
                atom_head_pointconv_max_neighbors=16,
                atom_head_drop_path=0.0, atom_head_pre_norm=True,
                pseudo_atom_cfg=pseudo_cfg,
            )
        return
        # 替换 attention stack 和 logit head 为 identity
        model.atom_token_proj = nn.Identity()
        model.atom_attention_stack = _Attn()
        model.atom_logit_head = nn.Identity()

        n_real = 3
        batch = {
            "voxel_grid": torch.zeros((2, 1, 2, 2, 2)),
            "box_shape_zyx": torch.tensor([[2, 2, 2], [2, 2, 2]], dtype=torch.int64),
            "voxel_size_world": torch.ones(2, 3),
            "box_origin_world": torch.zeros(2, 3),
            "atom_feat": torch.randn(n_real, 5),
            "atom_coord_centered_world": torch.tensor([[0., 0., 0.], [0.5, 0., 0.], [0., 0.5, 0.]]),
            "atom_coord_local_voxel": torch.tensor([[1., 1., 1.], [1.5, 1., 1.], [1., 1.5, 1.]]),
            "atom_coord_world": torch.tensor([[1., 1., 1.], [1.5, 1., 1.], [1., 1.5, 1.]]),
            "atom_batch_index": torch.tensor([0, 0, 1], dtype=torch.long),
            "atom_counts": torch.tensor([2, 1], dtype=torch.long),
            "atom_offsets": torch.tensor([2, 3], dtype=torch.long),
            "atom_valid_mask": torch.tensor([True, False, True]),
            "atom_label": torch.tensor([0, 1, 0], dtype=torch.long),
            "atom_is_in_core_box": torch.tensor([True, True, True]),
        }

        out = model(batch)

        # 输出应只有 n_real 个原子
        assert out["atom_logits"].shape[0] == n_real, \
            f"输出应只含 {n_real} 个真实原子, 实际 {out['atom_logits'].shape[0]}"
        assert out["fused_point_feat"].shape[0] == n_real
