# -*- coding: utf-8 -*-
"""
=============================================================================
density_channel_builder — 密度通道在线计算模块
=============================================================================
给定原始密度图 (exp/sim)，按 (op, norm, post) 配置生成候选通道。
所有计算在 CPU 上按顺序依次执行，返回 CPU numpy array。

通道命名约定:
    {op}_{norm}_{post}
    示例:
        exp_clipnorm_gauss1
        diff_clipnorm_DoG2

阶段 A — 基本运算 (op):
    exp      : ρ_exp
    sim      : ρ_sim
    diff     : ρ_exp - (a·ρ_sim + b)
    posdiff  : max(0, ρ_exp - (a·ρ_sim + b))

阶段 B — 归一化 (norm):
    nonorm   : 无归一化
    clipnorm : 按各通道自身数据分布去除极端值后 z-score

阶段 C — 后处理 (post):
    nopost   : 无后处理
    gauss1   : Gaussian σ=1
    gauss2   : Gaussian σ=2
    DoG1     : DoG σ=1, m=1.6
    DoG2     : DoG σ=2, m=1.6
    smooth1  : receptor_mask 附近 Gaussian 扣除, σ=1, radius=2
    smooth2  : receptor_mask 附近 Gaussian 扣除, σ=2, radius=2
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage
from dataclasses import dataclass, field
from typing import Optional



@dataclass
class DensityChannelConfig:
    """
    密度通道配置。

    输入参数:
        - clip_percentile: tuple[float, float], clip 范围百分位, 建议值 (0.001, 0.999)
        - fit_mask_percentile: float, 尺度匹配 mask 阈值百分位(同时控制最低体素比例阈值), 密度筛选使用 5×percentile, 建议值 0.003
        - enabled_channels: list[str], 启用的通道列表; ["all"] 代表全部 56 通道
    """
    clip_percentile: tuple[float, float] = (0.001, 0.999)
    fit_mask_percentile: float = 0.003
    enabled_channels: list[str] = field(default_factory=lambda: ["all"])


# ============================================================
# 基础运算 (op)
# ============================================================
def _fit_scale_params(
    exp: np.ndarray,
    sim: np.ndarray,
    percentile: float,
    receptor_mask: np.ndarray | None,
) -> tuple[float, float]:
    """
    Masked least squares 拟合尺度参数 a, b。备选拟合对象的优先级: 含有原子的体素、通过密度筛选的体素.
    所有参与拟合的体素(包括原子mask)必须同时通过密度筛选: 在模拟密度图和真实密度图中, 密度值均位于前 5×percentile 的高密度区间(用于应对密度漂移)。

    输入参数:
        - exp: np.ndarray, (D, H, W), float32, 真实密度
        - sim: np.ndarray, (D, H, W), float32, 模拟密度
        - percentile: float, 标量, 最低体素比例阈值(取值 0~1), 密度筛选使用 5×percentile
        - receptor_mask: np.ndarray | None, (D, H, W), bool, 受体原子对应的体素掩码

    输出:
        - a: float, 标量, 尺度因子
        - b: float, 标量, 偏移量
    """
    # int, 标量, 总体素数
    n_total = int(exp.size)
    # int, 标量, 最低体素数阈值(总体素 × percentile)
    min_voxels = max(int(n_total * percentile), 10)

    # --- 密度筛选: 两张图中密度值均位于前 5×percentile 的体素 ---
    # float, 标量, 密度筛选百分位(上限截断到 1.0)
    density_pct = min(5.0 * percentile, 1.0)
    # float, 标量, exp 前 density_pct 的密度下界
    t_exp = float(np.percentile(exp, (1.0 - density_pct) * 100))
    # float, 标量, sim 前 density_pct 的密度下界
    t_sim = float(np.percentile(sim, (1.0 - density_pct) * 100))
    # np.ndarray[bool], (D, H, W), 密度筛选通过的体素(两图均在前 5×percentile)
    density_filter = (exp >= t_exp) & (sim >= t_sim)

    if receptor_mask is not None:
        # np.ndarray[bool], (D, H, W), 受体原子体素中通过密度筛选的部分(最优先)
        priority = np.asarray(receptor_mask, dtype=bool) & density_filter
        # int, 标量, 优先区域体素数
        n_priority = int(priority.sum())

        if n_priority >= min_voxels:
            # 优先区域已足够, 仅使用受体原子体素
            mask = priority
        else:
            # 优先区域不足, 用全部候选体素(含优先部分)补充
            mask = density_filter
    else:
        # 无受体信息, 直接使用全部候选体素
        mask = density_filter

    # int, 标量, 最终 mask 内体素数
    n_masked = int(mask.sum())

    if n_masked < 10:
        # 极端情况：几乎无信号，返回恒等变换
        return 1.0, 0.0

    # np.ndarray, (n_masked,), float32, mask 内的 sim 体素值
    sim_masked = sim[mask]
    # np.ndarray, (n_masked,), float32, mask 内的 exp 体素值
    exp_masked = exp[mask]

    # float, 标量, sim 方差; 若为零则无法拟合
    if float(sim_masked.var()) < 1e-12:
        return 1.0, 0.0

    # np.ndarray, (n_masked, 2), float32, 设计矩阵 [sim, 1]
    X = np.column_stack([sim_masked, np.ones_like(sim_masked)])
    # np.ndarray, (2,), float32, 最小二乘解 [a, b]
    params = np.linalg.lstsq(X, exp_masked, rcond=None)[0]
    return float(params[0]), float(params[1])

def _compute_ops(
    exp: np.ndarray,
    sim: np.ndarray,
    percentile: float,
    receptor_mask: np.ndarray | None,
) -> dict[str, np.ndarray]:
    """
    计算所有基本运算通道: exp, sim, diff, posdiff。

    输入参数:
        - exp: np.ndarray, (D, H, W), float32, 原始真实密度
        - sim: np.ndarray, (D, H, W), float32, 原始模拟密度
        - percentile: float, 标量, mask 阈值百分位
        - receptor_mask: np.ndarray | None, (D, H, W), bool, 受体原子对应的体素掩码

    输出:
        - dict[str, np.ndarray], 键为 op 名称, 值为 (D, H, W) float32 array
    """
    # float, float, 尺度匹配参数
    a, b = _fit_scale_params(exp, sim, percentile, receptor_mask)
    # np.ndarray, (D, H, W), float32, 差值图
    diff = exp - (a * sim + b)

    return {
        "exp": exp,
        "sim": sim,
        "diff": diff,
        "posdiff": np.maximum(diff, 0),
    }





# ============================================================
# 归一化 (norm)
# ============================================================
def _clip_and_norm(x: np.ndarray, clip_percentile: tuple[float, float]) -> np.ndarray:
    """
    按自身数据分布 clip 极端值后 z-score 归一化，每个 op 通道(exp/sim/diff/posdiff)按各自值域 clip。

    输入参数:
        - x: np.ndarray, (D, H, W), float32, 输入体素
        - clip_percentile: tuple[float, float], clip 范围百分位, 如 (0.0003, 0.9997)

    输出:
        - np.ndarray, (D, H, W), float32, z-score 归一化后的体素
    """
    # float, 标量, 当前数据的 clip 下界
    lo = float(np.percentile(x, clip_percentile[0] * 100))
    # float, 标量, 当前数据的 clip 上界
    hi = float(np.percentile(x, clip_percentile[1] * 100))
    # np.ndarray, (D, H, W), float32, clip 后的数据
    x_clipped = np.clip(x, lo, hi)
    # float, 标量, 均值
    mu = float(x_clipped.mean())
    # float, 标量, 标准差(加 eps 防除零)
    sigma = float(x_clipped.std()) + 1e-8
    return (x_clipped - mu) / sigma

def _apply_norm(
    op_result: np.ndarray,
    norm: str,
    clip_percentile: tuple[float, float],
) -> np.ndarray:
    """
    对单个 op 结果应用归一化。

    输入参数:
        - op_result: np.ndarray, (D, H, W), float32, 基本运算结果
        - norm: str, 标量, "nonorm" 或 "clipnorm"
        - clip_percentile: tuple[float, float], clip 范围百分位

    输出:
        - np.ndarray, (D, H, W), float32, 归一化后结果
    """
    if norm == "nonorm":
        return op_result
    elif norm == "clipnorm":
        return _clip_and_norm(op_result, clip_percentile)
    else:
        raise ValueError(f"Unknown norm: {norm}")





# ============================================================
# 后处理 (post)
# ============================================================
def _gaussian_filter_3d(x: np.ndarray, sigma: float) -> np.ndarray:
    """
    各向同性 3D Gaussian 滤波。

    输入参数:
        - x: np.ndarray, (D, H, W), float32, 输入体素
        - sigma: float, 标量, Gaussian 标准差(单位: voxel)

    输出:
        - np.ndarray, (D, H, W), float32, 滤波结果
    """
    return ndimage.gaussian_filter(x, sigma=sigma, mode="reflect")

def _dog_3d(x: np.ndarray, sigma: float) -> np.ndarray:
    """
    Difference of Gaussian (DoG) 算子。

    DoG(σ, m=1.6) = G_{σ·m} * x - G_σ * x。
    近似 LoG，用于 blob 检测，局部极值点约等于口袋候选位置。

    输入参数:
        - x: np.ndarray, (D, H, W), float32, 输入体素
        - sigma: float, 标量, 基础 Gaussian 标准差(单位: voxel)

    输出:
        - np.ndarray, (D, H, W), float32, DoG 结果
    """
    # float, 标量, DoG 倍率因子
    m = 1.6
    return ndimage.gaussian_filter(x, sigma=sigma * m, mode="reflect") \
           - ndimage.gaussian_filter(x, sigma=sigma, mode="reflect")


def _build_smooth_strength(
    receptor_mask: np.ndarray,
    sigma: float,
    radius: float,
) -> np.ndarray:
    """
    根据 receptor_mask 构建局部 Gaussian 扣除强度图。

    输入参数:
        - receptor_mask: np.ndarray, (D, H, W), bool, receptor 体素掩码
        - sigma: float, 标量, Gaussian 标准差(单位: voxel)
        - radius: float, 标量, 截断半径(单位: voxel)

    输出:
        - smooth_strength: np.ndarray, (D, H, W), float32, 局部扣除强度图, receptor 体素为 1
    """
    # np.ndarray[bool], (D, H, W), receptor 体素掩码
    mask = np.asarray(receptor_mask, dtype=bool)
    # np.ndarray, (D, H, W), 每个体素到最近 receptor 体素的欧氏距离
    distance = ndimage.distance_transform_edt(~mask)
    # np.ndarray, (D, H, W), 半径截断前的 Gaussian 扣除强度
    smooth_strength = np.exp(-(distance ** 2) / (2.0 * sigma * sigma))
    smooth_strength[distance > radius] = 0.0
    return smooth_strength.astype(np.float32)


# dict[str, Optional[float]], post 名称 → sigma 映射
_POST_SIGMA_MAP: dict[str, Optional[float]] = {
    "nopost": None,
    "gauss1": 1.0,
    "gauss2": 2.0,
    "DoG1": 1.0,
    "DoG2": 2.0,
    "smooth1": 1.0,
    "smooth2": 2.0,
}

def _apply_post(x: np.ndarray, post: str) -> np.ndarray:
    """
    对单个通道应用后处理算子。

    输入参数:
        - x: np.ndarray, (D, H, W), float32, 输入体素
        - post: str, 标量, "nopost" / "gauss1" / "gauss2" / "DoG1" / "DoG2" / "smooth1" / "smooth2"

    输出:
        - np.ndarray, (D, H, W), float32, 后处理结果
    """
    if post == "nopost":
        return x
    if post not in _POST_SIGMA_MAP:
        raise ValueError(f"Unknown post: {post}")
    sigma = _POST_SIGMA_MAP[post]
    if sigma is None:
        return x
    if post.startswith("gauss"):
        return _gaussian_filter_3d(x, sigma)
    elif post.startswith("DoG"):
        return _dog_3d(x, sigma)
    elif post.startswith("smooth"):
        raise ValueError("smooth post 需要在 build_density_channels 中结合 receptor_mask 计算")
    else:
        raise ValueError(f"Unknown post: {post}")

















# ============================================================
# 全量通道枚举
# ============================================================
# list[str], 4 种基本运算
OPS: list[str] = ["exp", "sim", "diff", "posdiff"]
# list[str], 2 种归一化方式
NORMS: list[str] = ["nonorm", "clipnorm"]
# list[str], 7 种后处理方式
POSTS: list[str] = ["nopost", "gauss1", "gauss2", "DoG1", "DoG2", "smooth1", "smooth2"]

# list[str], 全部 4×2×7 = 56 个常规通道名
ALL_CHANNEL_NAMES: list[str] = [
    f"{op}_{norm}_{post}"
    for op in OPS
    for norm in NORMS
    for post in POSTS
]

# set[str], 需要 exp_raw 和 sim_raw 同时存在的 op 集合
_OPS_NEEDING_BOTH: set[str] = {"diff", "posdiff"}


def _parse_channel_name(name: str) -> tuple[str, str, str]:
    """
    解析通道名, 提取 (op, norm, post)。

    输入参数:
        - name: str, 标量, 通道名, 如 "diff_clipnorm_DoG2"

    输出:
        - tuple[str, str, str], (op, norm, post)
    """
    parts = name.split("_")
    if len(parts) < 3:
        raise ValueError(f"Invalid channel name: {name}")

    # str, 运算名
    op = parts[0]
    if op not in OPS:
        raise ValueError(f"Unknown op in channel name '{name}': {op}")

    # str, 归一化名; 可能是 "nonorm" 或 "clipnorm"
    norm = parts[1]
    if norm not in NORMS:
        raise ValueError(f"Unknown norm in channel name '{name}': {norm}")

    # str, 后处理名; 可能含下划线(如 DoG1)
    post = "_".join(parts[2:])
    if post not in POSTS:
        raise ValueError(f"Unknown post in channel name '{name}': {post}")

    return op, norm, post













# ============================================================
# 主入口
# ============================================================
def build_density_channels(
    exp_raw: Optional[np.ndarray],
    sim_raw: Optional[np.ndarray],
    config: DensityChannelConfig,
    receptor_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    密度通道在线计算主入口。
    给定原始密度图和配置，按 (op, norm, post) 生成启用的通道子集，所有计算在 CPU 上执行，返回 CPU numpy array。

    输入参数:
        - exp_raw: np.ndarray 或 None, (D, H, W), float32, 原始真实密度
        - sim_raw: np.ndarray 或 None, (D, H, W), float32, 原始模拟密度
        - config: DensityChannelConfig, 通道配置
        - receptor_mask: np.ndarray 或 None, (D, H, W), bool, 受体原子对应的体素掩码; 用于 diff/posdiff 通道的 exp-sim 尺度拟合

    输出:
        - np.ndarray, (C_enabled, D, H, W), float32, 按 enabled_channels 顺序拼接的密度通道
    """
    # ---------- 展开 enabled_channels ----------
    # list[str], 实际启用的通道名列表
    if "all" in config.enabled_channels:
        enabled = list(ALL_CHANNEL_NAMES)
    else:
        enabled = list(config.enabled_channels)
    if not enabled:
        raise ValueError("enabled_channels 为空，无法生成任何密度通道")

    # ---------- 确定需要哪些 op ----------
    # set[str], 所有需要计算的 op 集合
    needed_ops: set[str] = set()
    for name in enabled:
        op, _, _ = _parse_channel_name(name)
        needed_ops.add(op)

    # ---------- 检查输入可用性 ----------
    has_exp = exp_raw is not None
    has_sim = sim_raw is not None
    needs_both = bool(needed_ops & _OPS_NEEDING_BOTH)
    if "exp" in needed_ops and not has_exp:
        raise ValueError("启用了 exp 通道但未提供 exp_raw")
    if "sim" in needed_ops and not has_sim:
        raise ValueError("启用了 sim 通道但未提供 sim_raw")
    if needs_both and (not has_exp or not has_sim):
        raise ValueError("启用了 diff/posdiff 通道但未同时提供 exp_raw 和 sim_raw")

    # ---------- 计算 op 结果 ----------
    # dict[str, np.ndarray], op 名称 → (D, H, W) 结果
    all_ops: dict[str, np.ndarray] = {}
    if has_exp and has_sim and needs_both:
        all_ops = _compute_ops(exp_raw, sim_raw, config.fit_mask_percentile, receptor_mask)
    else:
        if has_exp:
            all_ops["exp"] = exp_raw
        if has_sim:
            all_ops["sim"] = sim_raw



    # ---------- 逐通道生成 ----------
    # dict, 复用同一 (op, norm) 的归一化结果, 避免 5 个 post 重复 percentile/clip/std
    norm_cache: dict[tuple[str, str], np.ndarray] = {}
    # dict, 复用同一 (op, norm, sigma) 的 Gaussian 结果, 避免 gauss/DoG 重复滤波
    gaussian_cache: dict[tuple[str, str, float], np.ndarray] = {}
    # dict, 复用同一 sigma 的 smooth 强度图, 避免不同 op/norm 重复计算距离变换
    smooth_cache: dict[float, np.ndarray] = {}

    def get_normed(op_name: str, norm_name: str) -> np.ndarray:
        cache_key = (op_name, norm_name)
        if cache_key not in norm_cache:
            norm_cache[cache_key] = _apply_norm(all_ops[op_name], norm_name, config.clip_percentile)
        return norm_cache[cache_key]

    def get_gaussian(op_name: str, norm_name: str, sigma: float) -> np.ndarray:
        cache_key = (op_name, norm_name, float(sigma))
        if cache_key not in gaussian_cache:
            gaussian_cache[cache_key] = _gaussian_filter_3d(get_normed(op_name, norm_name), sigma)
        return gaussian_cache[cache_key]

    def get_smooth_strength(sigma: float) -> np.ndarray:
        cache_key = float(sigma)
        if cache_key not in smooth_cache:
            if receptor_mask is None:
                raise ValueError("启用了 smooth post 但未提供 receptor_mask")
            smooth_cache[cache_key] = _build_smooth_strength(receptor_mask, cache_key, radius=2.0)
        return smooth_cache[cache_key]

    # list[np.ndarray], 各启用通道的 (D, H, W) 结果
    channels: list[np.ndarray] = []
    for name in enabled:
        op_name, norm_name, post_name = _parse_channel_name(name)
        if post_name == "nopost":
            final = get_normed(op_name, norm_name)
        elif post_name.startswith("gauss"):
            sigma = float(_POST_SIGMA_MAP[post_name])
            final = get_gaussian(op_name, norm_name, sigma)
        elif post_name.startswith("DoG"):
            sigma = float(_POST_SIGMA_MAP[post_name])
            final = get_gaussian(op_name, norm_name, sigma * 1.6) - get_gaussian(op_name, norm_name, sigma)
        elif post_name.startswith("smooth"):
            sigma = float(_POST_SIGMA_MAP[post_name])
            final = get_normed(op_name, norm_name) * (1.0 - get_smooth_strength(sigma))
        else:
            final = _apply_post(get_normed(op_name, norm_name), post_name)
        channels.append(final)

    if not channels:
        raise ValueError(
            "未生成任何密度通道。请检查 enabled_channels 配置和输入数据。"
            f" enabled_channels={config.enabled_channels},"
            f" has_exp={has_exp}, has_sim={has_sim}"
        )

    # np.ndarray, (C_enabled, D, H, W), float32, 拼接后的密度通道
    return np.stack(channels, axis=0).astype(np.float32)


# ============================================================
# 工具函数: 自动检测 diff/posdiff 通道索引
# ============================================================
def detect_diff_posdiff_indices(enabled_channels: list[str]) -> list[int]:
    """
    从 enabled_channels 列表中检测基本运算为 diff 或 posdiff 的通道索引。

    输入参数:
        - enabled_channels: list[str], 可变长度, 启用的通道名列表; ["all"] 代表全部 56 通道

    输出:
        - indices: list[int], 可变长度, 基本运算为 diff 或 posdiff 的通道在 enabled_channels 中的索引
    """
    # list[str], 解析后的实际通道名列表
    if "all" in enabled_channels:
        resolved = list(ALL_CHANNEL_NAMES)
    else:
        resolved = list(enabled_channels)
    # list[int], 基本运算为 diff 或 posdiff 的通道索引
    indices: list[int] = []
    for i, name in enumerate(resolved):
        op, _, _ = _parse_channel_name(name)
        if op in ("diff", "posdiff"):
            indices.append(i)
    return indices
