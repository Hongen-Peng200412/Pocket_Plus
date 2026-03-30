# -*- coding: utf-8 -*-
"""
=============================================================================
FaultTolerantDataset — 通用数据加载容错包装器
=============================================================================

在 DDP 多卡训练中, 如果某个 rank 的 dataset.__getitem__ 抛出异常或因读取损坏
文件而长时间挂起, 其他 rank 会在 sync_dist=True 的标量 ALLREDUCE 处无限等待,
最终触发 NCCL watchdog 超时 (默认 30 分钟) 导致整个作业被杀掉。

本模块提供的 FaultTolerantDataset 在 __getitem__ 层面捕获异常并替换坏样本:
    - 每次 sampler 请求仍返回"一个有效样本", 不会在 DDP 下出现 batch 步数失配。
    - 不修改底层 dataset 的实现, 也不在 collate 阶段做过滤。

性能影响:
    - 在所有样本均正常的情况下, 唯一额外开销是一次 try/except 和一个 set 查找,
      对训练吞吐几乎为零影响。
    - 仅在坏样本命中时才触发线性探测和日志打印。
"""

import os
import signal
import sys
import time
import traceback
from contextlib import contextmanager
from typing import Any, Optional

import torch
from torch.utils.data import Dataset


# ============================================================
# 1. 自定义异常: 单样本读取超时
# ============================================================
class SampleLoadTimeoutError(RuntimeError):
    """
    单样本读取超时时抛出的异常。

    输入参数:
        - index: int, 标量, 超时的样本索引
        - timeout_sec: int, 标量, 超时阈值(秒)
    """
    pass


# ============================================================
# 2. 超时上下文管理器
# ============================================================
@contextmanager
def _dataset_timeout_ctx(timeout_sec: Optional[int], index: int):
    """
    为一次 dataset[index] 调用提供超时保护。

    输入参数:
        - timeout_sec: int|None, 标量, 超时秒数; None 或 ≤0 表示不启用超时
        - index: int, 标量, 当前请求的样本索引 (仅用于错误消息)

    行为:
        - Linux/Posix: 使用 signal.SIGALRM 实现硬超时, 仅在主线程可用。
        - Windows / 子 worker 进程: 自动降级为 no-op。
    """
    if timeout_sec is None or timeout_sec <= 0:
        yield
        return

    # signal.alarm 仅限 Unix + 主线程
    _can_use_signal = (
        hasattr(signal, "SIGALRM")
        and hasattr(signal, "alarm")
        and _is_main_thread()
    )

    if not _can_use_signal:
        yield
        return

    def _handler(signum, frame):
        raise SampleLoadTimeoutError(
            f"样本 index={index} 读取超时 (>{timeout_sec}s)"
        )

    old_handler = signal.signal(signal.SIGALRM, _handler)
    signal.alarm(timeout_sec)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


def _is_main_thread() -> bool:
    """
    判断当前是否为主线程。
    """
    import threading
    return threading.current_thread() is threading.main_thread()


# ============================================================
# 3. 通用日志描述
# ============================================================
def _describe_dataset_index(dataset: Dataset, index: int) -> str:
    """
    为坏样本生成一段可读的日志描述。

    输入参数:
        - dataset: Dataset, 被包装的原始数据集
        - index: int, 标量, 样本索引

    输出:
        - str, 样本的可读描述字符串

    优先级:
        1. dataset.describe_index(index) (若数据集自行提供)
        2. dataset.total_sample[index] (若存在该属性)
        3. 仅打印 index
    """
    if hasattr(dataset, "describe_index") and callable(dataset.describe_index):
        try:
            return str(dataset.describe_index(index))
        except Exception:
            pass
    if hasattr(dataset, "total_sample"):
        try:
            return str(dataset.total_sample[index])
        except Exception:
            pass
    return f"index={index}"


# ============================================================
# 4. FaultTolerantDataset 包装器
# ============================================================
class FaultTolerantDataset(Dataset):
    """
    通用数据加载容错包装器: 在 __getitem__ 层面捕获异常, 自动替换坏样本。

    输入参数:
        - base_dataset: Dataset, 被包装的原始数据集
        - stage: str, 标量, 当前阶段 ("train" / "val")
        - max_resample_attempts: int, 标量, 单次请求失败后最多补偿次数
        - sample_timeout_sec: int|None, 标量, 单样本读取超时秒数
        - log_first_n: int, 标量, 每个 worker 最多打印的坏样本日志数

    行为:
        - 收到 sampler 请求的原始 index 后, 先尝试原 index;
          失败则按 (index+1)%len, (index+2)%len, ... 线性探测替代。
        - 每次尝试均在 _dataset_timeout_ctx 内执行 base_dataset[candidate]。
        - 成功返回有效样本; 连续失败超过 max_resample_attempts 则硬失败。
    """

    def __init__(
        self,
        base_dataset: Dataset,
        stage: str,
        max_resample_attempts: int,
        sample_timeout_sec: Optional[int],
        log_first_n: int,
    ):
        super().__init__()
        self.base_dataset = base_dataset
        self.stage = stage
        self.max_resample_attempts = max_resample_attempts
        self.sample_timeout_sec = sample_timeout_sec
        self.log_first_n = log_first_n

        # set[int], 本 worker 已知的坏索引
        self.known_bad_indices: set[int] = set()
        # int, 已打印的坏样本日志计数
        self.logged_count: int = 0

    def __len__(self) -> int:
        return len(self.base_dataset)

    @property
    def collate_fn(self):
        """
        透传 base_dataset 的 collate_fn (若存在), 供 DataLoader 使用。
        """
        return getattr(self.base_dataset, "collate_fn", None)

    def __getattr__(self, name: str) -> Any:
        """
        透传未在本类中定义的属性访问到 base_dataset, 例如 total_sample 等。
        """
        if name in ("base_dataset", "stage", "max_resample_attempts",
                     "sample_timeout_sec", "log_first_n",
                     "known_bad_indices", "logged_count"):
            raise AttributeError(name)
        return getattr(self.base_dataset, name)

    def __getitem__(self, index: int) -> Any:
        """
        容错版 __getitem__: 先尝试原始 index, 失败后线性探测替代样本。

        输入参数:
            - index: int, 标量, sampler 请求的原始样本索引

        输出:
            - 与 base_dataset[index] 结构完全一致的样本数据
        """
        total_len = len(self.base_dataset)

        # 快速路径: 原始索引已知为坏样本, 直接跳过
        if index in self.known_bad_indices:
            return self._probe_replacement(index, total_len, start_offset=1)

        # 正常路径: 尝试原始索引
        try:
            with _dataset_timeout_ctx(self.sample_timeout_sec, index):
                return self.base_dataset[index]
        except Exception as exc:
            self._record_bad_index(index, exc)
            return self._probe_replacement(index, total_len, start_offset=1)

    def _probe_replacement(self, original_index: int, total_len: int, start_offset: int) -> Any:
        """
        线性探测替代样本。

        输入参数:
            - original_index: int, 标量, 失败的原始样本索引
            - total_len: int, 标量, 数据集总长度
            - start_offset: int, 标量, 起始偏移量

        输出:
            - 与 base_dataset[candidate] 结构一致的有效样本数据
        """
        for attempt in range(start_offset, self.max_resample_attempts + 1):
            candidate = (original_index + attempt) % total_len

            if candidate in self.known_bad_indices:
                continue

            try:
                with _dataset_timeout_ctx(self.sample_timeout_sec, candidate):
                    sample = self.base_dataset[candidate]

                # 成功: 打印补偿日志
                self._log_replacement(original_index, candidate, attempt)
                return sample

            except Exception as exc:
                self._record_bad_index(candidate, exc)

        # 全部失败: 硬报错
        raise RuntimeError(
            f"[FaultTolerant] stage={self.stage} 连续 {self.max_resample_attempts} 个"
            f"候选样本均失败 (原始 index={original_index})。"
            f" 已知坏索引数: {len(self.known_bad_indices)}/{total_len}。"
            f" 请检查数据完整性。"
        )

    def _record_bad_index(self, index: int, exc: Exception) -> None:
        """
        记录坏索引并打印结构化错误日志。
        """
        self.known_bad_indices.add(index)

        if self.logged_count >= self.log_first_n:
            return

        self.logged_count += 1

        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info else "main"

        desc = _describe_dataset_index(self.base_dataset, index)
        exc_type = type(exc).__name__
        exc_msg = str(exc).split("\n")[0][:200]  # 只取第一行, 截断过长信息

        print(
            f"[FaultTolerant] stage={self.stage} bad_index={index} "
            f"worker={worker_id} pid={os.getpid()} "
            f"err={exc_type}: {exc_msg} "
            f"desc={desc}",
            flush=True,
        )

        if self.logged_count == self.log_first_n:
            print(
                f"[FaultTolerant] stage={self.stage} 已达到日志上限 "
                f"(log_first_n={self.log_first_n}), 后续坏样本将静默跳过。",
                flush=True,
            )

    def _log_replacement(self, original_index: int, replacement_index: int, attempt: int) -> None:
        """
        打印替代样本的补偿日志。
        """
        if self.logged_count >= self.log_first_n:
            return

        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info else "main"

        print(
            f"[FaultTolerant] stage={self.stage} "
            f"index={original_index}->{replacement_index} "
            f"attempt={attempt}/{self.max_resample_attempts} "
            f"worker={worker_id} pid={os.getpid()}",
            flush=True,
        )


# ============================================================
# 5. 便捷包装函数
# ============================================================
def maybe_wrap_dataset(
    dataset: Dataset,
    stage: str,
    ft_cfg,
) -> Dataset:
    """
    根据配置决定是否用 FaultTolerantDataset 包装原始数据集。

    输入参数:
        - dataset: Dataset, 原始数据集实例
        - stage: str, 标量, 当前阶段 ("train" / "val" / "test")
        - ft_cfg: DictConfig 或 dict 或 None, data_fault_tolerance 配置子树

    输出:
        - Dataset, 包装后的数据集 (若未启用则原样返回)
    """
    if ft_cfg is None:
        return dataset

    enabled = ft_cfg.get("enabled", False) if hasattr(ft_cfg, "get") else False
    if not enabled:
        return dataset

    stages = ft_cfg.get("stages", ["train", "val"]) if hasattr(ft_cfg, "get") else ["train", "val"]
    if stage not in stages:
        return dataset

    max_resample_attempts = int(ft_cfg.get("max_resample_attempts", 8))
    sample_timeout_sec_raw = ft_cfg.get("sample_timeout_sec", None)
    sample_timeout_sec = int(sample_timeout_sec_raw) if sample_timeout_sec_raw is not None else None
    log_first_n = int(ft_cfg.get("log_first_n", 50))

    wrapped = FaultTolerantDataset(
        base_dataset=dataset,
        stage=stage,
        max_resample_attempts=max_resample_attempts,
        sample_timeout_sec=sample_timeout_sec,
        log_first_n=log_first_n,
    )

    # 仅 rank 0 打印初始化信息 (非分布式环境下也打印)
    rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))
    if rank == 0:
        print(
            f"[FaultTolerant] 已启用 stage={stage}, "
            f"max_attempts={max_resample_attempts}, "
            f"timeout={sample_timeout_sec}s, "
            f"log_first_n={log_first_n}, "
            f"dataset_len={len(dataset)}",
            flush=True,
        )

    return wrapped
