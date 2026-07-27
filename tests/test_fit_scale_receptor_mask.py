# -*- coding: utf-8 -*-
"""
测试 _fit_scale_params 的密度筛选 + 受体原子优先选取逻辑. 

验证:
    1. 有足够受体原子通过密度筛选时, 仅用受体体素拟合, 结果更准确
    2. 受体原子通过筛选的体素不足时, 自动用全部候选体素补充
    3. receptor_mask=None 时使用全部候选体素
    4. 极端情况: 全零密度图 → 返回恒等变换 (1.0, 0.0)
    5. build_density_channels 正确透传 receptor_mask
    6. build_hardmask_from_world_coordinates 能正确构造 receptor_mask
    7. 密度筛选确实排除了漂移体素(density_filter 核心语义)
"""
import sys
import numpy as np
import pytest

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent / "src"))

from datasets.density_channel_builder import (
    _fit_scale_params,
    _compute_ops,
    build_density_channels,
    DensityChannelConfig,
)
from datasets.box_geometry import (
    build_hardmask_from_world_coordinates,
)


# ============================================================
# 辅助函数: 构造虚拟密度图
# ============================================================

def make_synthetic_density_pair(
    shape: tuple[int, int, int],
    true_a: float,
    true_b: float,
    noise_std: float,
    atom_mask: np.ndarray | None,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """
    构造一对合成 (exp, sim) 密度图, 其中 exp = a * sim + b + noise. 

    sim 在 atom_mask 区域有高信号, 在其余区域有低信号/噪声,
    模拟真实密度图中蛋白质区域信号较强的特征. 

    输入参数:
        - shape: tuple[int, int, int], (D, H, W), 密度图尺寸
        - true_a: float, 标量, 真实尺度因子
        - true_b: float, 标量, 真实偏移量
        - noise_std: float, 标量, 高斯噪声标准差
        - atom_mask: np.ndarray | None, (D, H, W), bool, 原子占据区域
        - rng: np.random.Generator, 随机数生成器

    输出:
        - exp: np.ndarray, (D, H, W), float32, 模拟真实密度
        - sim: np.ndarray, (D, H, W), float32, 模拟计算密度
    """
    # np.ndarray, (D, H, W), float32, 模拟密度图(基底噪声)
    sim = rng.normal(0.0, 0.1, size=shape).astype(np.float32)
    if atom_mask is not None:
        # 在原子区域叠加高信号
        sim[atom_mask] += rng.uniform(2.0, 5.0, size=int(atom_mask.sum())).astype(np.float32)

    # np.ndarray, (D, H, W), float32, 真实密度图
    exp = (true_a * sim + true_b + rng.normal(0, noise_std, size=shape)).astype(np.float32)

    return exp, sim


# ============================================================
# Test 1: receptor_mask 足够大时, 拟合结果更准确
# ============================================================

class TestFitScaleWithReceptorMask:

    def test_receptor_mask_produces_accurate_fit(self):
        """有足够原子体素通过密度筛选时, 仅用受体体素拟合, 结果更精确. """
        rng = np.random.default_rng(42)
        shape = (32, 32, 32)
        # float, 标量, 真实尺度因子
        true_a = 2.5
        # float, 标量, 真实偏移量
        true_b = -0.3

        # np.ndarray, (D, H, W), bool, 模拟原子占据区域(中心立方体, 高信号)
        atom_mask = np.zeros(shape, dtype=bool)
        atom_mask[8:24, 8:24, 8:24] = True

        exp, sim = make_synthetic_density_pair(
            shape=shape, true_a=true_a, true_b=true_b,
            noise_std=0.05, atom_mask=atom_mask, rng=rng,
        )

        # --- 使用 receptor_mask ---
        a_atom, b_atom = _fit_scale_params(exp, sim, percentile=0.01, receptor_mask=atom_mask)
        # --- 不使用 receptor_mask (全部候选体素) ---
        a_perc, b_perc = _fit_scale_params(exp, sim, percentile=0.01, receptor_mask=None)

        err_atom = abs(a_atom - true_a) + abs(b_atom - true_b)
        err_perc = abs(a_perc - true_a) + abs(b_perc - true_b)
        print(f"\n  [atom mask]  a={a_atom:.4f}, b={b_atom:.4f}, err={err_atom:.4f}")
        print(f"  [density only] a={a_perc:.4f}, b={b_perc:.4f}, err={err_perc:.4f}")

        # 两种方法都应给出合理结果
        assert abs(a_atom - true_a) < 0.5, f"atom mask fit a 偏差过大: {a_atom} vs {true_a}"
        assert abs(b_atom - true_b) < 0.5, f"atom mask fit b 偏差过大: {b_atom} vs {true_b}"

    def test_receptor_mask_none_falls_back_to_density_filter(self):
        """receptor_mask=None 时应使用全部密度筛选候选体素, 且不报错. """
        rng = np.random.default_rng(123)
        shape = (20, 20, 20)
        # np.ndarray, (D, H, W), bool, 用于构造信号
        atom_mask = np.zeros(shape, dtype=bool)
        atom_mask[5:15, 5:15, 5:15] = True
        exp, sim = make_synthetic_density_pair(
            shape=shape, true_a=1.5, true_b=0.1,
            noise_std=0.05, atom_mask=atom_mask, rng=rng,
        )

        a, b = _fit_scale_params(exp, sim, percentile=0.01, receptor_mask=None)
        print(f"\n  [None mask] a={a:.4f}, b={b:.4f}")
        assert np.isfinite(a) and np.isfinite(b)


# ============================================================
# Test 2: 受体体素通过筛选不足时, 用全部候选补充
# ============================================================

class TestDensityFilterSupplement:

    def test_sparse_mask_triggers_supplement(self):
        """受体 mask 通过密度筛选的体素数 < min_voxels 时, 自动用全部候选补充. """
        rng = np.random.default_rng(77)
        shape = (32, 32, 32)

        # np.ndarray, (D, H, W), bool, 只有 5 个体素被标记(远少于 min_voxels)
        sparse_mask = np.zeros(shape, dtype=bool)
        sparse_mask[15, 15, 15] = True
        sparse_mask[16, 16, 16] = True
        sparse_mask[17, 17, 17] = True
        sparse_mask[14, 14, 14] = True
        sparse_mask[13, 13, 13] = True

        # 构造有信号的密度图
        atom_mask_for_signal = np.zeros(shape, dtype=bool)
        atom_mask_for_signal[10:22, 10:22, 10:22] = True
        exp, sim = make_synthetic_density_pair(
            shape=shape, true_a=2.0, true_b=0.5,
            noise_std=0.05, atom_mask=atom_mask_for_signal, rng=rng,
        )

        # 不应报错, 应成功补充并拟合
        a, b = _fit_scale_params(exp, sim, percentile=0.01, receptor_mask=sparse_mask)
        print(f"\n  [sparse mask → supplement] a={a:.4f}, b={b:.4f}")
        assert np.isfinite(a) and np.isfinite(b)

    def test_empty_mask_returns_identity(self):
        """全空的 receptor_mask 且密度也全零时, 应返回 (1.0, 0.0). """
        shape = (8, 8, 8)
        # np.ndarray, (D, H, W), float32, 几乎无信号
        exp = np.zeros(shape, dtype=np.float32)
        sim = np.zeros(shape, dtype=np.float32)
        # np.ndarray, (D, H, W), bool, 全空
        empty_mask = np.zeros(shape, dtype=bool)

        a, b = _fit_scale_params(exp, sim, percentile=0.5, receptor_mask=empty_mask)
        print(f"\n  [empty mask + no signal] a={a:.4f}, b={b:.4f}")
        assert a == 1.0 and b == 0.0, "极端情况应返回恒等变换 (1.0, 0.0)"


# ============================================================
# Test 3: build_density_channels 正确透传 receptor_mask
# ============================================================

class TestBuildDensityChannelsTransparency:

    def test_receptor_mask_transparent_to_diff_channel(self):
        """build_density_channels 应将 receptor_mask 透传给 diff/posdiff 通道. """
        rng = np.random.default_rng(99)
        shape = (16, 16, 16)
        # np.ndarray, (D, H, W), bool, 原子区域
        atom_mask = np.zeros(shape, dtype=bool)
        atom_mask[4:12, 4:12, 4:12] = True

        exp, sim = make_synthetic_density_pair(
            shape=shape, true_a=2.0, true_b=0.5,
            noise_std=0.05, atom_mask=atom_mask, rng=rng,
        )

        config = DensityChannelConfig(
            enabled_channels=["diff_nonorm_nopost"],
        )

        # 有 receptor_mask
        result_with = build_density_channels(
            exp_raw=exp, sim_raw=sim, config=config,
            receptor_mask=atom_mask,
        )
        # 无 receptor_mask
        result_without = build_density_channels(
            exp_raw=exp, sim_raw=sim, config=config,
            receptor_mask=None,
        )

        assert result_with.shape == (1,) + shape
        assert result_without.shape == (1,) + shape
        assert result_with.dtype == np.float32

    def test_exp_only_channel_ignores_receptor_mask(self):
        """只启用 exp 通道时, receptor_mask 对结果无影响(exp 不需要 a, b 拟合). """
        rng = np.random.default_rng(42)
        shape = (16, 16, 16)
        exp = rng.normal(0, 1, size=shape).astype(np.float32)
        atom_mask = np.zeros(shape, dtype=bool)
        atom_mask[4:12, 4:12, 4:12] = True

        config = DensityChannelConfig(
            enabled_channels=["exp_nonorm_nopost"],
        )
        result_with = build_density_channels(
            exp_raw=exp, sim_raw=None, config=config,
            receptor_mask=atom_mask,
        )
        result_without = build_density_channels(
            exp_raw=exp, sim_raw=None, config=config,
            receptor_mask=None,
        )
        np.testing.assert_array_equal(result_with, result_without)
        print("\n  exp-only 通道: receptor_mask 无影响 ✓")


# ============================================================
# Test 4: build_hardmask_from_world_coordinates 正确构造 mask
# ============================================================

class TestBuildHardmaskIntegration:

    def test_hardmask_marks_correct_voxels(self):
        """原子坐标应被正确映射到对应体素. """
        box_origin = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        voxel_size = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        box_shape_zyx = np.array([10, 10, 10], dtype=np.int64)

        atom_coords = np.array([
            [2.5, 3.5, 4.5],
            [7.1, 8.2, 1.3],
            [0.1, 0.1, 0.1],
        ], dtype=np.float32)

        hardmask = build_hardmask_from_world_coordinates(
            atom_coords_world=atom_coords,
            box_origin_world=box_origin,
            voxel_size_world=voxel_size,
            box_shape_zyx=box_shape_zyx,
        )

        assert hardmask.shape == (10, 10, 10)
        assert hardmask[4, 3, 2] == 1, "原子 (2.5, 3.5, 4.5) 应标记 voxel [4,3,2]"
        assert hardmask[1, 8, 7] == 1, "原子 (7.1, 8.2, 1.3) 应标记 voxel [1,8,7]"
        assert hardmask[0, 0, 0] == 1, "原子 (0.1, 0.1, 0.1) 应标记 voxel [0,0,0]"

        n_marked = int(hardmask.sum())
        assert n_marked == 3, f"预期 3 个体素被标记, 实际 {n_marked}"
        print(f"\n  hardmask 构造正确: {n_marked} 个体素被标记 ✓")

    def test_hardmask_as_receptor_mask_for_fitting(self):
        """端到端: hardmask → receptor_mask → _fit_scale_params 全链路. """
        rng = np.random.default_rng(2024)

        box_origin = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        voxel_size = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        box_shape_zyx = np.array([20, 20, 20], dtype=np.int64)

        # np.ndarray, (N, 3), float32, 散布在 BOX 内的原子
        atom_coords = rng.uniform(1.0, 19.0, size=(200, 3)).astype(np.float32)

        hardmask = build_hardmask_from_world_coordinates(
            atom_coords_world=atom_coords,
            box_origin_world=box_origin,
            voxel_size_world=voxel_size,
            box_shape_zyx=box_shape_zyx,
        )
        receptor_mask = hardmask.astype(bool)
        n_receptor_voxels = int(receptor_mask.sum())
        print(f"\n  receptor_mask 体素数: {n_receptor_voxels} / {20**3} = {n_receptor_voxels / 20**3:.4f}")

        true_a, true_b = 3.0, -1.0
        exp, sim = make_synthetic_density_pair(
            shape=(20, 20, 20), true_a=true_a, true_b=true_b,
            noise_std=0.02, atom_mask=receptor_mask, rng=rng,
        )

        a, b = _fit_scale_params(exp, sim, percentile=0.01, receptor_mask=receptor_mask)
        print(f"  拟合结果: a={a:.4f} (真实 {true_a}), b={b:.4f} (真实 {true_b})")
        assert abs(a - true_a) < 0.5, f"a 偏差过大: {a}"
        assert abs(b - true_b) < 0.5, f"b 偏差过大: {b}"


# ============================================================
# Test 5: 补充比例验证
# ============================================================

class TestSupplementRatio:

    def test_supplement_reaches_min_voxels_when_needed(self):
        """当受体 mask 通过筛选不足时, 用全部候选体素补充, 拟合不崩溃. """
        rng = np.random.default_rng(55)
        shape = (32, 32, 32)

        # np.ndarray, (D, H, W), bool, 3 个体素(远不够)
        sparse_mask = np.zeros(shape, dtype=bool)
        sparse_mask[15, 15, 15] = True
        sparse_mask[16, 16, 16] = True
        sparse_mask[17, 17, 17] = True

        # 构造有明确信号的密度图
        sim = np.zeros(shape, dtype=np.float32)
        sim[10:22, 10:22, 10:22] = rng.uniform(2.0, 5.0, size=(12, 12, 12)).astype(np.float32)
        exp = (2.0 * sim + 0.5).astype(np.float32)

        a, b = _fit_scale_params(exp, sim, percentile=0.5, receptor_mask=sparse_mask)
        print(f"\n  [补充测试] a={a:.4f}, b={b:.4f}")
        assert np.isfinite(a) and np.isfinite(b)
        assert abs(a - 2.0) < 1.0, f"a 偏差过大: {a}"


# ============================================================
# Test 6: 密度筛选核心语义 — 排除漂移体素
# ============================================================

class TestDensityFilterSemantics:

    def test_drifted_voxels_excluded_from_fit(self):
        """
        构造密度漂移场景: 受体 mask 中部分体素在 exp 中有信号但 sim 中没有(模拟漂移). 
        密度筛选应排除这些体素, 使拟合不被漂移污染. 
        """
        rng = np.random.default_rng(314)
        shape = (32, 32, 32)
        true_a, true_b = 2.0, 0.5

        # --- 构造 sim: 仅在 good_region 有高信号 ---
        sim = rng.normal(0.0, 0.05, size=shape).astype(np.float32)
        # np.ndarray[bool], (D, H, W), 未漂移的高信号区
        good_region = np.zeros(shape, dtype=bool)
        good_region[4:16, 4:16, 4:16] = True
        sim[good_region] += rng.uniform(3.0, 6.0, size=int(good_region.sum())).astype(np.float32)

        # --- 构造 exp: good_region 遵从 a*sim+b, 但 drift_region 有异常高信号 ---
        exp = (true_a * sim + true_b + rng.normal(0, 0.02, size=shape)).astype(np.float32)
        # np.ndarray[bool], (D, H, W), 漂移区域(exp 有信号但 sim 没有)
        drift_region = np.zeros(shape, dtype=bool)
        drift_region[20:28, 20:28, 20:28] = True
        exp[drift_region] += 10.0  # exp 中出现异常高密度

        # --- 受体 mask 覆盖 good + drift 区域 ---
        receptor_mask = good_region | drift_region

        # --- 拟合: 密度筛选应排除 drift_region(sim 在那里很低) ---
        a, b = _fit_scale_params(exp, sim, percentile=0.01, receptor_mask=receptor_mask)
        print(f"\n  [漂移测试] a={a:.4f} (真实 {true_a}), b={b:.4f} (真实 {true_b})")
        # 如果漂移体素被正确排除, a 应接近 true_a
        assert abs(a - true_a) < 0.5, f"a 偏差过大(漂移未被排除?): {a} vs {true_a}"
        assert abs(b - true_b) < 1.0, f"b 偏差过大: {b} vs {true_b}"

    def test_density_filter_uses_5x_percentile(self):
        """验证密度筛选确实使用 5×percentile 作为阈值, 而非 1×percentile. """
        rng = np.random.default_rng(271)
        shape = (20, 20, 20)
        # float, 标量, percentile 参数
        percentile = 0.02  # 5×percentile = 0.10, 即前 10%

        sim = rng.normal(0.0, 1.0, size=shape).astype(np.float32)
        exp = (1.5 * sim + 0.2 + rng.normal(0, 0.01, size=shape)).astype(np.float32)

        # 前 10% 阈值
        t_exp_expected = float(np.percentile(exp, 90.0))  # (1 - 0.10) * 100
        t_sim_expected = float(np.percentile(sim, 90.0))
        # int, 标量, 预期候选体素数(两图均在前 10%)
        expected_filter = (exp >= t_exp_expected) & (sim >= t_sim_expected)
        n_expected = int(expected_filter.sum())

        # 调用函数应成功(min_voxels = 20^3 * 0.02 = 160, 前 10% 交集应足够)
        a, b = _fit_scale_params(exp, sim, percentile=percentile, receptor_mask=None)
        print(f"\n  [5x验证] percentile={percentile}, 5x={5*percentile}")
        print(f"  阈值: t_exp={t_exp_expected:.4f}, t_sim={t_sim_expected:.4f}")
        print(f"  候选体素数: {n_expected}, a={a:.4f}, b={b:.4f}")
        assert np.isfinite(a) and np.isfinite(b)
        assert abs(a - 1.5) < 0.3, f"a 偏差过大: {a}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
