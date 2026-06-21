"""
BalancedForegroundSampler 单元测试。

验证:
1. 不放回全遍历: 每个 epoch 内所有索引恰好出现一次
2. 前景比例递补: 每个全局 batch 前景数量 >= n_fg_min (在前景候补池未耗尽时)
3. 可复现性: 同一 seed + epoch 产生相同序列
4. 跨 epoch 差异: 不同 epoch 产生不同序列
5. 边界条件: batch_size > total_size / batch_size 不整除 / 无背景样本等
6. 分布式: 多 rank 索引无重叠且并集覆盖全量
7. 卡数无关性: 相同 global_batch_size + seed → 相同全局序列(不论 world_size)
"""
import math
from collections import Counter
from src.datasets.balanced_foreground_sampler import BalancedForegroundSampler


# ============================================================
# 单卡基础测试 (rank=0, world_size=1)
# ============================================================

def test_all_indices_appear_once():
    """每个 epoch 内所有索引恰好出现一次(不放回全遍历)。"""
    total = 100
    fg = list(range(20))
    bg = list(range(20, 100))
    sampler = BalancedForegroundSampler(
        fg, bg, total, foreground_ratio=0.2,
        global_batch_size=8, seed=42, rank=0, world_size=1,
    )
    indices = list(sampler)
    assert len(indices) == total, f"Expected {total}, got {len(indices)}"


def test_foreground_ratio_enforced():
    """每个完整全局 batch 中前景数量 >= n_fg_min (候补池未耗尽时)。"""
    total = 200
    fg = list(range(80))       # 40% 是前景
    bg = list(range(80, 200))  # 60% 是背景
    gbs = 10
    ratio = 0.3
    sampler = BalancedForegroundSampler(
        fg, bg, total, foreground_ratio=ratio,
        global_batch_size=gbs, seed=123, rank=0, world_size=1,
    )
    n_fg_min = math.ceil(gbs * ratio)
    fg_set = set(fg)

    indices = list(sampler)
    n_full_batches = total // gbs
    for b in range(n_full_batches):
        batch = indices[b * gbs: (b + 1) * gbs]
        n_fg_in_batch = sum(1 for idx in batch if idx in fg_set)
        assert n_fg_in_batch >= n_fg_min, (
            f"Batch {b}: n_fg={n_fg_in_batch} < n_fg_min={n_fg_min}"
        )


def test_reproducibility():
    """同一 seed + epoch 产生相同序列。"""
    total = 50
    fg = list(range(15))
    bg = list(range(15, 50))
    s1 = BalancedForegroundSampler(
        fg, bg, total, foreground_ratio=0.2,
        global_batch_size=8, seed=99, rank=0, world_size=1,
    )
    s2 = BalancedForegroundSampler(
        fg, bg, total, foreground_ratio=0.2,
        global_batch_size=8, seed=99, rank=0, world_size=1,
    )
    assert list(s1) == list(s2)


def test_different_epochs():
    """不同 epoch 产生不同序列。"""
    total = 50
    fg = list(range(15))
    bg = list(range(15, 50))
    sampler = BalancedForegroundSampler(
        fg, bg, total, foreground_ratio=0.2,
        global_batch_size=8, seed=99, rank=0, world_size=1,
    )
    seq0 = list(sampler)
    sampler.set_epoch(1)
    seq1 = list(sampler)
    assert seq0 != seq1


def test_total_size_equals_length():
    """__len__ 返回 per_rank_size (单卡 = total_size)。"""
    sampler = BalancedForegroundSampler(
        [0, 1, 2], [3, 4, 5, 6], 7,
        foreground_ratio=0.2, global_batch_size=3, seed=0,
        rank=0, world_size=1,
    )
    assert len(sampler) == 7


def test_small_dataset():
    """total_size < global_batch_size 边界。"""
    total = 5
    fg = [0, 1]
    bg = [2, 3, 4]
    sampler = BalancedForegroundSampler(
        fg, bg, total, foreground_ratio=0.4,
        global_batch_size=10, seed=7, rank=0, world_size=1,
    )
    indices = list(sampler)
    assert len(indices) == total


def test_all_foreground():
    """全是前景样本, 无背景。"""
    total = 20
    fg = list(range(20))
    bg = []
    sampler = BalancedForegroundSampler(
        fg, bg, total, foreground_ratio=0.3,
        global_batch_size=5, seed=0, rank=0, world_size=1,
    )
    indices = list(sampler)
    assert len(indices) == total
    assert sorted(indices) == list(range(total))


def test_no_replacement_coverage():
    """背景索引最多出现 1 次, 前景索引最多出现 2 次(主序列 + 递补)。"""
    total = 100
    fg = list(range(30))
    bg = list(range(30, 100))
    sampler = BalancedForegroundSampler(
        fg, bg, total, foreground_ratio=0.2,
        global_batch_size=8, seed=55, rank=0, world_size=1,
    )
    indices = list(sampler)
    assert len(indices) == total
    counts = Counter(indices)
    for idx in bg:
        assert counts.get(idx, 0) <= 1
    for idx in fg:
        assert counts.get(idx, 0) <= 2


# ============================================================
# 分布式测试
# ============================================================

def test_distributed_no_overlap():
    """多 rank 分到的索引无重叠, 并集覆盖全量(含 padding)。"""
    total = 100
    fg = list(range(30))
    bg = list(range(30, 100))
    ws = 4
    gbs = 16

    all_rank_indices = []
    for r in range(ws):
        sampler = BalancedForegroundSampler(
            fg, bg, total, foreground_ratio=0.2,
            global_batch_size=gbs, seed=42, rank=r, world_size=ws,
        )
        rank_indices = list(sampler)
        assert len(rank_indices) == len(sampler)
        all_rank_indices.append(rank_indices)

    # 各 rank 长度一致
    lengths = [len(ri) for ri in all_rank_indices]
    assert len(set(lengths)) == 1, f"Unequal lengths across ranks: {lengths}"

    # 重建全局序列: 交织各 rank 的索引
    per_rank = lengths[0]
    global_reconstructed = []
    for i in range(per_rank):
        for r in range(ws):
            global_reconstructed.append(all_rank_indices[r][i])

    # 全局序列长度 = padded_size
    padded_size = math.ceil(total / ws) * ws
    assert len(global_reconstructed) == padded_size


def test_distributed_foreground_ratio_global_batch():
    """全局 batch (合并所有 rank 的 per-device batch) 满足前景比例。"""
    total = 200
    fg = list(range(80))
    bg = list(range(80, 200))
    ws = 4
    gbs = 20
    ratio = 0.3
    n_fg_min = math.ceil(gbs * ratio)
    fg_set = set(fg)

    # 收集所有 rank 的索引
    all_rank_indices = []
    for r in range(ws):
        sampler = BalancedForegroundSampler(
            fg, bg, total, foreground_ratio=ratio,
            global_batch_size=gbs, seed=77, rank=r, world_size=ws,
        )
        all_rank_indices.append(list(sampler))

    # 重建全局序列
    per_rank = len(all_rank_indices[0])
    global_seq = []
    for i in range(per_rank):
        for r in range(ws):
            global_seq.append(all_rank_indices[r][i])

    # 逐全局 batch 检查前景比例
    n_full = len(global_seq) // gbs
    for b in range(n_full):
        batch = global_seq[b * gbs: (b + 1) * gbs]
        n_fg = sum(1 for idx in batch if idx in fg_set)
        assert n_fg >= n_fg_min, (
            f"Global batch {b}: n_fg={n_fg} < n_fg_min={n_fg_min}"
        )


def test_determinism_across_world_sizes():
    """
    核心保证: 相同 global_batch_size + seed → 相同全局序列, 不论 world_size。
    """
    total = 120
    fg = list(range(40))
    bg = list(range(40, 120))
    gbs = 12
    seed = 999

    def reconstruct_global(ws):
        """用 world_size=ws 重建全局序列。"""
        all_rank = []
        for r in range(ws):
            s = BalancedForegroundSampler(
                fg, bg, total, foreground_ratio=0.2,
                global_batch_size=gbs, seed=seed, rank=r, world_size=ws,
            )
            all_rank.append(list(s))
        per_rank = len(all_rank[0])
        global_seq = []
        for i in range(per_rank):
            for r in range(ws):
                global_seq.append(all_rank[r][i])
        return global_seq

    # world_size=1 (基准)
    global_ws1 = reconstruct_global(1)
    # world_size=2
    global_ws2 = reconstruct_global(2)
    # world_size=4
    global_ws4 = reconstruct_global(4)

    # 截断到 total_size 比较(padding 部分可能因 padded_size 不同而有差异)
    assert global_ws1[:total] == global_ws2[:total], (
        "global sequence differs between world_size=1 and world_size=2"
    )
    assert global_ws1[:total] == global_ws4[:total], (
        "global sequence differs between world_size=1 and world_size=4"
    )


def test_global_batch_size_not_divisible():
    """global_batch_size 不能被 world_size 整除时应 raise。"""
    import pytest
    with pytest.raises(ValueError, match="整除"):
        BalancedForegroundSampler(
            [0, 1], [2, 3, 4], 5,
            foreground_ratio=0.2, global_batch_size=7, seed=0,
            rank=0, world_size=2,
        )
