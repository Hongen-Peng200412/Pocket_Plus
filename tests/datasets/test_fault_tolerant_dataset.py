# -*- coding: utf-8 -*-
"""
FaultTolerantDataset 单元测试。

使用合成 fake dataset 验证容错机制, 不依赖真实数据文件。
"""
from __future__ import annotations

import pytest
import torch
from torch.utils.data import Dataset

from src.utils.fault_tolerant_dataset import (
    FaultTolerantDataset,
    maybe_wrap_dataset,
    _describe_dataset_index,
)


# ============================================================
# 辅助 Fake Dataset
# ============================================================
class _FakeDataset(Dataset):
    """
    生成简单的 {index: int, value: float} 样本, 可通过 bad_indices
    控制哪些索引会抛异常。

    输入参数:
        - size: int, 标量, 数据集大小
        - bad_indices: set[int], 会抛 FileNotFoundError 的索引集合
    """

    def __init__(self, size: int, bad_indices: set[int] | None = None):
        self.size = size
        self.bad_indices = bad_indices or set()
        # list[dict], 模拟 total_sample 属性
        self.total_sample = [{"sample_name": f"sample_{i}"} for i in range(size)]

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> dict:
        if index in self.bad_indices:
            raise FileNotFoundError(f"Fake missing file for index={index}")
        return {"index": index, "value": float(index) * 0.1}


class _FakeDatasetWithCollate(_FakeDataset):
    """带 collate_fn 的 Fake Dataset。"""

    @staticmethod
    def collate_fn(batch):
        return batch


# ============================================================
# 测试用例
# ============================================================
class TestFaultTolerantDataset:
    """FaultTolerantDataset 核心功能测试。"""

    def test_all_normal_samples(self):
        """所有样本正常时, FaultTolerantDataset 应透明地返回每个样本。"""
        ds = _FakeDataset(size=10)
        ft_ds = FaultTolerantDataset(
            base_dataset=ds, stage="train",
            max_resample_attempts=8, sample_timeout_sec=None, log_first_n=50,
        )
        for i in range(10):
            sample = ft_ds[i]
            assert sample["index"] == i
            assert abs(sample["value"] - i * 0.1) < 1e-6

    def test_single_bad_sample_replaced(self):
        """单个坏样本 (index=5) 被替换为 index=6。"""
        ds = _FakeDataset(size=20, bad_indices={5})
        ft_ds = FaultTolerantDataset(
            base_dataset=ds, stage="train",
            max_resample_attempts=8, sample_timeout_sec=None, log_first_n=50,
        )
        sample = ft_ds[5]
        assert sample["index"] == 6, f"Expected replacement index=6, got {sample['index']}"

    def test_consecutive_bad_samples_skip(self):
        """连续坏样本 (5,6,7) 跳到 index=8。"""
        ds = _FakeDataset(size=20, bad_indices={5, 6, 7})
        ft_ds = FaultTolerantDataset(
            base_dataset=ds, stage="train",
            max_resample_attempts=8, sample_timeout_sec=None, log_first_n=50,
        )
        sample = ft_ds[5]
        assert sample["index"] == 8

    def test_max_resample_exceeded_raises(self):
        """连续坏样本超过 max_resample_attempts 时应硬失败。"""
        # 让 index 0~9 全部为坏
        ds = _FakeDataset(size=20, bad_indices=set(range(10)))
        ft_ds = FaultTolerantDataset(
            base_dataset=ds, stage="train",
            max_resample_attempts=4, sample_timeout_sec=None, log_first_n=50,
        )
        with pytest.raises(RuntimeError, match="连续"):
            ft_ds[0]

    def test_disabled_config_passes_through(self):
        """enabled=false 时, maybe_wrap_dataset 应原样返回。"""
        ds = _FakeDataset(size=10, bad_indices={5})
        cfg = {"enabled": False, "stages": ["train", "val"],
               "max_resample_attempts": 8, "sample_timeout_sec": None, "log_first_n": 50}
        wrapped = maybe_wrap_dataset(ds, "train", cfg)
        assert wrapped is ds  # 未包装

    def test_enabled_config_wraps(self):
        """enabled=true 时, maybe_wrap_dataset 应包装。"""
        ds = _FakeDataset(size=10)
        cfg = {"enabled": True, "stages": ["train", "val"],
               "max_resample_attempts": 8, "sample_timeout_sec": None, "log_first_n": 50}
        wrapped = maybe_wrap_dataset(ds, "train", cfg)
        assert isinstance(wrapped, FaultTolerantDataset)
        assert len(wrapped) == 10

    def test_stage_not_in_stages_skips(self):
        """stage 不在 stages 列表中时, 不包装。"""
        ds = _FakeDataset(size=10)
        cfg = {"enabled": True, "stages": ["train"],
               "max_resample_attempts": 8, "sample_timeout_sec": None, "log_first_n": 50}
        wrapped = maybe_wrap_dataset(ds, "val", cfg)
        assert wrapped is ds

    def test_collate_fn_passthrough(self):
        """collate_fn 应透传到 FaultTolerantDataset。"""
        ds = _FakeDatasetWithCollate(size=10)
        ft_ds = FaultTolerantDataset(
            base_dataset=ds, stage="train",
            max_resample_attempts=8, sample_timeout_sec=None, log_first_n=50,
        )
        assert ft_ds.collate_fn is _FakeDatasetWithCollate.collate_fn

    def test_log_first_n_limit(self):
        """log_first_n=2 时, 最多打印 2 条日志。"""
        ds = _FakeDataset(size=20, bad_indices={0, 1, 2, 3})
        ft_ds = FaultTolerantDataset(
            base_dataset=ds, stage="train",
            max_resample_attempts=8, sample_timeout_sec=None, log_first_n=2,
        )
        # 触发 4 个坏样本
        for i in range(4):
            ft_ds[i]
        # logged_count 应为 2 (log_first_n 限制)
        assert ft_ds.logged_count == 2

    def test_known_bad_indices_cached(self):
        """已知坏索引在第二次访问时应直接跳过, 不再尝试。"""
        ds = _FakeDataset(size=20, bad_indices={5})
        ft_ds = FaultTolerantDataset(
            base_dataset=ds, stage="train",
            max_resample_attempts=8, sample_timeout_sec=None, log_first_n=50,
        )
        # 第一次: 触发异常并记录
        sample1 = ft_ds[5]
        assert 5 in ft_ds.known_bad_indices
        # 第二次: 直接走 known_bad_indices 快速路径
        sample2 = ft_ds[5]
        assert sample2["index"] == sample1["index"]

    def test_wrap_around_at_boundary(self):
        """末尾坏样本应环绕到开头寻找替代。"""
        ds = _FakeDataset(size=5, bad_indices={4})
        ft_ds = FaultTolerantDataset(
            base_dataset=ds, stage="train",
            max_resample_attempts=8, sample_timeout_sec=None, log_first_n=50,
        )
        sample = ft_ds[4]
        # (4+1)%5 = 0, 应返回 index=0
        assert sample["index"] == 0

    def test_describe_dataset_index_with_total_sample(self):
        """_describe_dataset_index 应返回 total_sample 的内容。"""
        ds = _FakeDataset(size=5)
        desc = _describe_dataset_index(ds, 3)
        assert "sample_3" in desc

    def test_none_config_returns_original(self):
        """ft_cfg=None 时, maybe_wrap_dataset 应原样返回。"""
        ds = _FakeDataset(size=10)
        wrapped = maybe_wrap_dataset(ds, "train", None)
        assert wrapped is ds
