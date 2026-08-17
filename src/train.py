"""
可通用代码——————训练代码
"""

import sys
import os
import random
import fnmatch
import shutil
from collections.abc import Mapping
from typing import Any
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:256,expandable_segments:True")

import rootutils
from pathlib import Path
import numpy as np

# Setup Root, 设置根目录; rootutils, (module), 用于自动查找项目根目录的工具库
ROOT = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
Pocket_Plus_ROOT = Path(__file__).resolve().parent.parent  # Pocket_Plus/
if str(Pocket_Plus_ROOT) not in sys.path:
    sys.path.insert(0, str(Pocket_Plus_ROOT))

from src.utils.slurm_utils import (
    fix_gloo_socket_ifname as _fix_gloo_socket_ifname,
    log_distributed_launch_state as _log_distributed_launch_state,
)
from src.utils.module_freeze import set_fully_frozen_submodules_eval

# 统一使用 slurm_utils 中的网卡推导逻辑, 避免本地旧实现与 sbatch helper 出现分叉. 
_fix_gloo_socket_ifname()

import torch
import torch.multiprocessing
from torch.utils.data import Sampler
from torch.utils.data.distributed import DistributedSampler
torch.set_float32_matmul_precision("high")
torch.multiprocessing.set_sharing_strategy('file_system')
import hydra
from omegaconf import DictConfig, OmegaConf
import lightning as pl
from lightning.pytorch.callbacks import Callback, ModelCheckpoint, LearningRateMonitor, RichProgressBar


from src.utils.wandb_utils import _setup_wandb_mode




from lightning.pytorch.loggers import WandbLogger
import wandb
# Setup Root and PYTHONPATH already moved to the top of the file
FEEDBACK_ROOT = Path(                # NOTE: 返回结果的存放目录由这里更改
    os.environ.get("EXPERIMENT_FEEDBACK_ROOT", str(ROOT / "feedback_plus"))
)
from src.utils.experiment_manager import ExperimentManager
from src.utils.fault_tolerant_dataset import maybe_wrap_dataset
# 模型、包装器和回调类现在将根据配置动态导入, 无需手动导入


def _resolve_init_checkpoint(init_from: str, feedback_root: Path, current_run_dir: Path) -> Path:
    """
    解析 model-only 初始化 ckpt 路径. 

    输入参数:
        - init_from: str, `.ckpt` 文件、run 目录, 或 `"***"` 作业内上一阶段哨兵
        - feedback_root: Path, feedback_plus 根目录
        - current_run_dir: Path, 当前 run 目录; 解析 `"***"` 时用于排除自身

    输出:
        - ckpt_path: Path, 真实存在的 ckpt 文件路径
    """
    init_text = str(init_from).strip()
    if init_text == "***":
        from src.utils.ckpt_resolve import resolve_previous_best_checkpoint_by_job

        return resolve_previous_best_checkpoint_by_job(
            feedback_root=feedback_root,
            current_run_dir=current_run_dir,
        )

    init_path = Path(init_text).expanduser()
    if init_path.is_file():
        return init_path
    if init_path.is_dir():
        ckpt_path = init_path / "checkpoints" / "BEST.ckpt"
        if ckpt_path.is_file():
            return ckpt_path
        raise FileNotFoundError(f"init_from 目录必须包含 checkpoints/BEST.ckpt: {init_path}")
    raise FileNotFoundError(f"init_from 指向的 ckpt 文件或 run 目录不存在: {init_path}")


def _load_model_only_checkpoint(model: torch.nn.Module, ckpt_path: Path, verbose: bool) -> None:
    """
    只加载 checkpoint 中的模型权重, 不恢复 optimizer、scheduler 与 global_step. 

    AdaLigand 用它把同名 Find 的 CPC1 BEST 交给 CPC2. ``state_dict`` 加载后仍调用
    wrapper 的 ``on_load_checkpoint``, 使 P 候选阈值等模型运行状态与 CPC1 对齐; 
    epoch 和优化器时间线则从 CPC2 自己的第 0 步重新开始. 

    输入参数:
        - model: torch.nn.Module, 已实例化并完成 lazy 初始化的 LightningModule
        - ckpt_path: Path, ckpt 文件路径
        - verbose: bool, 是否打印加载摘要

    输出:
        - None, 原地写入模型参数; missing/unexpected 非零时 fail-fast
    """
    checkpoint = torch.load(str(ckpt_path), map_location="cpu")
    state_dict = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict, strict=True)
    if isinstance(checkpoint, dict) and hasattr(model, "on_load_checkpoint"):
        model.on_load_checkpoint(checkpoint)
    if verbose:
        print(
            "[Train] model-only init_from: "
            f"path={ckpt_path}, loaded_keys={len(state_dict)}, strict=True, lifecycle=restored"
        )


def _apply_frozen_module(model: torch.nn.Module, frozen_cfg: DictConfig, verbose: bool) -> None:
    """
    按显式 name-pattern 冻结参数. 

    输入参数:
        - model: torch.nn.Module, 当前 LightningModule
        - frozen_cfg: DictConfig, 含 `patterns: list[str]`(必选, 命中 0 即报错)与可选
          `optional_patterns: list[str]`(命中 0 仅打印, 用于消融里可能缺席的模块)
        - verbose: bool, 是否打印冻结摘要

    输出:
        - None, 原地设置 requires_grad
    """
    # tuple[str, ...], 必须命中的冻结 pattern; 任一命中 0 个参数视为配置/typo 错误
    required_patterns = tuple(str(pattern) for pattern in frozen_cfg["patterns"])
    if len(required_patterns) == 0:
        raise ValueError("frozen_module.patterns 不能为空。")
    # tuple[str, ...], 允许命中 0 的冻结 pattern; 仅在对应模块本次实例化时才存在(如 real density 调制)
    optional_patterns = tuple(str(pattern) for pattern in (frozen_cfg.get("optional_patterns", None) or ()))
    # tuple[str, ...], 冻结判定时统一参与匹配的全部 pattern
    all_patterns = required_patterns + optional_patterns

    matched_by_pattern = {pattern: 0 for pattern in all_patterns}
    trainable_count = 0
    frozen_count = 0
    trainable_modules: set[str] = set()
    for name, parameter in model.named_parameters():
        # bool, 该参数是否命中任一 required/optional pattern
        matched_any = False
        for pattern in all_patterns:
            if fnmatch.fnmatch(name, pattern):
                matched_by_pattern[pattern] += 1
                matched_any = True
        if matched_any:
            parameter.requires_grad = False
            frozen_count += int(parameter.numel())
        else:
            parameter.requires_grad = True
            trainable_count += int(parameter.numel())
            trainable_modules.add(name.rsplit(".", 1)[0] if "." in name else name)

    # list[str], required pattern 中命中 0 的项; 非空即 fail-fast(防 typo / 误冻)
    unmatched_required = [pattern for pattern in required_patterns if matched_by_pattern[pattern] == 0]
    if unmatched_required:
        raise RuntimeError(f"frozen_module.patterns 中存在未匹配任何参数的 pattern: {unmatched_required}")
    if trainable_count == 0:
        raise RuntimeError("frozen_module 应用后没有任何可训练参数。")

    # 冻结仅设 requires_grad=False 不会停住 BN 的 running stats 与 Dropout; 让完全冻结子树进入 eval, 固定其前向. 
    # wrapper.train() 覆写会在每个 epoch 后重复维持; 这里做一次初始 eval 并取计数用于日志/自检. 
    # int, int: 被切到 eval 的极大冻结子树数; 其中带 running 统计的 BN 数(本会漂移、现已固定)
    num_frozen_subtrees, num_frozen_bn = set_fully_frozen_submodules_eval(model)

    if verbose:
        preview = ", ".join(sorted(trainable_modules)[:30])
        suffix = "" if len(trainable_modules) <= 30 else f", ... (+{len(trainable_modules) - 30})"
        print(
            "[Train] frozen_module applied: "
            f"patterns={len(required_patterns)}, optional_patterns={len(optional_patterns)}, "
            f"trainable_params={trainable_count:,}, frozen_params={frozen_count:,}"
        )
        print(f"[Train] trainable module preview: {preview}{suffix}")
        print(f"[Train] frozen→eval: {num_frozen_subtrees} 个完全冻结子树切到 eval(含 {num_frozen_bn} 个带 running 统计的 BN)")
        # 逐条提示命中 0 的 optional pattern; 正式实验(模块应在位)里出现即说明 typo 或配置错
        for pattern in optional_patterns:
            if matched_by_pattern[pattern] == 0:
                print(f"[！！Train！！] frozen_module optional pattern matched 0 params (skipped): {pattern}")


class LearningRateReductionStopper(Callback):
    """
    通用 LR 衰减计数停训回调. 

    输入参数:
        - stop_after_lr_reductions: int, 任一 optimizer param group 的学习率实际下降达到该次数后停止训练

    行为:
        - 只观察 optimizer 当前 lr, 不关心 scheduler 类型或触发位置. 
        - 在 validation end 和下一次 train batch 前都检查一次, 避免依赖 LightningModule 与 Callback 的 hook 顺序. 
        - 计数写入 checkpoint, 可用于断点续训; model-only 初始化不会恢复该 callback state. 
    """

    def __init__(self, stop_after_lr_reductions: int) -> None:
        super().__init__()
        if int(stop_after_lr_reductions) <= 0:
            raise ValueError("stop_after_lr_reductions 必须 > 0。")
        self.stop_after_lr_reductions = int(stop_after_lr_reductions)
        self.lr_reduction_count = 0
        self._last_lrs: tuple[float, ...] | None = None

    @staticmethod
    def _current_lrs(trainer: pl.Trainer) -> tuple[float, ...]:
        """
        读取 trainer 当前所有 optimizer param group 的学习率. 

        输入参数:
            - trainer: pl.Trainer, 当前训练器

        输出:
            - lrs: tuple[float, ...], 展平后的学习率序列
        """
        lrs: list[float] = []
        for optimizer in getattr(trainer, "optimizers", []):
            for group in optimizer.param_groups:
                lrs.append(float(group["lr"]))
        return tuple(lrs)

    def _observe(self, trainer: pl.Trainer) -> None:
        """
        比较当前 LR 与上一次记录值, 若发生实际下降则累加并按阈值停训. 

        输入参数:
            - trainer: pl.Trainer, 当前训练器

        输出:
            - None, 原地更新计数或 trainer.should_stop
        """
        if bool(getattr(trainer, "sanity_checking", False)):
            return
        current_lrs = self._current_lrs(trainer)
        if not current_lrs:
            return
        if self._last_lrs is None or len(self._last_lrs) != len(current_lrs):
            self._last_lrs = current_lrs
            return
        lr_reduced = any(curr < prev for prev, curr in zip(self._last_lrs, current_lrs))
        self._last_lrs = current_lrs
        if not lr_reduced:
            return

        self.lr_reduction_count += 1
        if self.lr_reduction_count >= self.stop_after_lr_reductions:
            if bool(getattr(trainer, "is_global_zero", True)):
                print(
                    "[Train] 检测到实际 LR 衰减 "
                    f"{self.lr_reduction_count} 次，达到 "
                    f"stop_after_lr_reductions={self.stop_after_lr_reductions}，将在当前流程结束后停止训练。"
                )
            trainer.should_stop = True

    def on_train_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        del pl_module
        self._last_lrs = self._current_lrs(trainer)

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        del pl_module
        self._observe(trainer)

    def on_validation_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        del pl_module
        self._observe(trainer)

    def on_train_batch_start(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        batch: object,
        batch_idx: int,
    ) -> None:
        del pl_module, batch, batch_idx
        self._observe(trainer)

    def state_dict(self) -> dict[str, object]:
        return {
            "lr_reduction_count": int(self.lr_reduction_count),
            "last_lrs": None if self._last_lrs is None else list(self._last_lrs),
        }

    def load_state_dict(self, state_dict: dict[str, object]) -> None:
        self.lr_reduction_count = int(state_dict.get("lr_reduction_count", 0))
        last_lrs = state_dict.get("last_lrs", None)
        self._last_lrs = None if last_lrs is None else tuple(float(value) for value in last_lrs)


def _resolve_scheduler_warmup_steps(*, trainer: pl.Trainer, sched_cfg: Mapping[str, Any]) -> int:
    """
    从 cfg.train.scheduler 解析 warmup step 数. 

    输入参数:
        - trainer: pl.Trainer, 当前训练器; 提供 estimated_stepping_batches
        - sched_cfg: Mapping[str, Any], scheduler 配置; 必须含 total_steps/warmup_steps/warmup_ratio

    输出:
        - warmup_steps: int, warmup 覆盖的 optimizer step 数
    """
    total_steps = sched_cfg["total_steps"]
    if total_steps is None:
        total_steps = getattr(trainer, "estimated_stepping_batches", None)
    if total_steps is None or int(total_steps) <= 0:
        raise RuntimeError("warmup_plateau scheduler requires a positive total_steps value.")
    total_steps = int(total_steps)

    warmup_steps = sched_cfg["warmup_steps"]
    if warmup_steps is None:
        warmup_ratio = float(sched_cfg["warmup_ratio"])
        if not (0.0 <= warmup_ratio < 1.0):
            raise ValueError(f"warmup_ratio must be in [0, 1), got {warmup_ratio}.")
        warmup_steps = int(round(total_steps * warmup_ratio))
    warmup_steps = int(warmup_steps)
    if warmup_steps < 0 or warmup_steps > total_steps:
        raise ValueError(
            f"warmup_steps must be in [0, total_steps], got warmup_steps={warmup_steps}, total_steps={total_steps}."
        )
    return warmup_steps


class WarmupPlateauController(Callback):
    """
    通用 validation 级 ReduceLROnPlateau 控制器. 

    输入参数:
        - sched_cfg: Mapping[str, Any], cfg.train.scheduler 中 warmup_plateau 的完整配置
        - monitor_metric: str, validation 后已写入 trainer.callback_metrics 的监控指标名

    行为:
        - step 级 warmup 仍交给 Lightning 原生 lr_scheduler;
        - plateau 部分在每次 validation 完成后按 monitor_metric 手动推进;
        - plateau 状态由本 callback 写入 Lightning checkpoint, 不再下沉到 wrapper. 
    """

    def __init__(self, *, sched_cfg: Mapping[str, Any], monitor_metric: str) -> None:
        super().__init__()
        self.sched_cfg = dict(sched_cfg)
        self.monitor_metric = str(monitor_metric)
        self.warmup_steps: int | None = None
        self._plateau_schedulers: list[torch.optim.lr_scheduler.ReduceLROnPlateau] = []
        self._pending_plateau_states: list[dict[str, Any]] | None = None
        self._last_stepped_validation: tuple[int, int, int] | None = None
        self._validation_index = 0

    def _build_schedulers(self, trainer: pl.Trainer) -> None:
        """
        在 optimizer 已由 LightningModule 构建后创建 plateau scheduler. 

        输入参数:
            - trainer: pl.Trainer, 当前训练器; trainer.optimizers 必须已初始化

        输出:
            - None, 原地写入 self._plateau_schedulers
        """
        optimizers = list(getattr(trainer, "optimizers", []))
        if not optimizers:
            return
        self.warmup_steps = _resolve_scheduler_warmup_steps(trainer=trainer, sched_cfg=self.sched_cfg)
        self._plateau_schedulers = [
            torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode=str(self.sched_cfg["mode"]),
                factor=float(self.sched_cfg["factor"]),
                patience=int(self.sched_cfg["patience"]),
                threshold=float(self.sched_cfg["threshold"]),
                threshold_mode=str(self.sched_cfg["threshold_mode"]),
                cooldown=int(self.sched_cfg["cooldown"]),
                min_lr=self.sched_cfg["min_lr"],
                eps=float(self.sched_cfg["eps"]),
            )
            for optimizer in optimizers
        ]
        if self._pending_plateau_states is not None:
            if len(self._pending_plateau_states) != len(self._plateau_schedulers):
                raise RuntimeError(
                    "warmup_plateau checkpoint 中的 optimizer 数量与当前训练器不一致: "
                    f"checkpoint={len(self._pending_plateau_states)}, current={len(self._plateau_schedulers)}。"
                )
            for scheduler, state in zip(self._plateau_schedulers, self._pending_plateau_states):
                scheduler.load_state_dict(state)
            self._pending_plateau_states = None

    def _monitor_value(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> float | None:
        """
        读取当前 validation 的 plateau 监控指标. 

        输入参数:
            - trainer: pl.Trainer, 当前训练器
            - pl_module: pl.LightningModule, 当前模型; 可提供最近一次 validation payload

        输出:
            - value: float | None, sanity checking 或指标尚未写入时返回 None
        """
        if bool(getattr(trainer, "sanity_checking", False)):
            return None
        payload = getattr(pl_module, "_last_validation_payload", {})
        if self.monitor_metric in payload:
            metric = payload[self.monitor_metric]
            if isinstance(metric, torch.Tensor):
                metric = metric.detach().float().reshape(()).item()
            return float(metric)

        metrics = getattr(trainer, "callback_metrics", {})
        if self.monitor_metric not in metrics:
            raise RuntimeError(f"warmup_plateau scheduler monitor metric {self.monitor_metric!r} is not available after validation.")
        metric = metrics[self.monitor_metric]
        if isinstance(metric, torch.Tensor):
            metric = metric.detach().float().reshape(()).item()
        return float(metric)

    def _step_if_ready(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """
        若当前 validation 指标可用, 则推进 plateau scheduler. 

        输入参数:
            - trainer: pl.Trainer, 当前训练器
            - pl_module: pl.LightningModule, 当前模型

        输出:
            - None, 原地更新 plateau 状态与 optimizer lr
        """
        if not self._plateau_schedulers:
            self._build_schedulers(trainer)
        if not self._plateau_schedulers:
            raise RuntimeError("warmup_plateau scheduler 未找到 optimizer, 无法构建 plateau controller。")
        if int(trainer.global_step) < int(self.warmup_steps or 0):
            return
        validation_key = (
            int(trainer.global_step),
            int(getattr(trainer, "current_epoch", 0)),
            int(self._validation_index),
        )
        if self._last_stepped_validation == validation_key:
            return
        metric_value = self._monitor_value(trainer, pl_module)
        if metric_value is None:
            return
        for scheduler in self._plateau_schedulers:
            scheduler.step(metric_value)
        self._last_stepped_validation = validation_key
        self._validation_index += 1

    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        del pl_module
        self._build_schedulers(trainer)

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        del trainer, pl_module

    def on_validation_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self._step_if_ready(trainer, pl_module)

    def state_dict(self) -> dict[str, Any]:
        states = (
            self._pending_plateau_states
            if self._pending_plateau_states is not None
            else [scheduler.state_dict() for scheduler in self._plateau_schedulers]
        )
        return {
            "warmup_steps": self.warmup_steps,
            "plateau_schedulers": states,
            "last_stepped_validation": self._last_stepped_validation,
            "validation_index": int(self._validation_index),
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.warmup_steps = None if state_dict.get("warmup_steps", None) is None else int(state_dict["warmup_steps"])
        plateau_states = state_dict.get("plateau_schedulers", [])
        self._pending_plateau_states = [dict(state) for state in plateau_states]
        last_validation = state_dict.get("last_stepped_validation", None)
        if last_validation is None:
            self._last_stepped_validation = None
        else:
            last_values = tuple(int(value) for value in last_validation)
            if len(last_values) != 3:
                raise RuntimeError("WarmupPlateauController checkpoint 中 last_stepped_validation 长度必须为 3。")
            self._last_stepped_validation = last_values
        self._validation_index = int(state_dict.get("validation_index", 0))


class BestCheckpointAlias(Callback):
    """
    维护 `checkpoints/BEST.ckpt` 稳定别名. 

    输入参数:
        - checkpoint_callback: ModelCheckpoint, 主 TOP checkpoint 回调
        - alias_path: Path, 固定 BEST.ckpt 输出路径
    """

    def __init__(self, checkpoint_callback: ModelCheckpoint, alias_path: Path) -> None:
        super().__init__()
        self.checkpoint_callback = checkpoint_callback
        self.alias_path = alias_path

    def _refresh_alias(self, trainer: pl.Trainer) -> None:
        """
        若主 checkpoint 已有 best_model_path, 则复制为固定 BEST.ckpt. 

        输入参数:
            - trainer: pl.Trainer, 当前训练器

        输出:
            - None, 仅 rank0 写文件
        """
        if not bool(getattr(trainer, "is_global_zero", True)):
            return
        best_model_path = str(getattr(self.checkpoint_callback, "best_model_path", "") or "")
        if not best_model_path:
            return
        source_path = Path(best_model_path)
        if not source_path.is_file():
            return
        self.alias_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self.alias_path.with_suffix(".tmp")
        shutil.copy2(source_path, tmp_path)
        tmp_path.replace(self.alias_path)

    def on_validation_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        del pl_module
        self._refresh_alias(trainer)

    def on_train_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        del pl_module
        self._refresh_alias(trainer)


class PeriodicCheckpointSaver(Callback):
    """
    按固定 epoch 间隔保存独立的 PERIODIC checkpoint. 

    输入参数:
        - dirpath: Path, checkpoint 输出目录
        - every_n_epochs: int, 每隔多少个 epoch 保存一次
        - filename_template: str, 文件名模板, 可使用 `{epoch:02d}` 占位符
    """

    def __init__(self, dirpath: Path, every_n_epochs: int, filename_template: str) -> None:
        super().__init__()
        if int(every_n_epochs) <= 0:
            raise ValueError("every_n_epochs 必须 > 0。")
        self.dirpath = dirpath
        self.every_n_epochs = int(every_n_epochs)
        self.filename_template = filename_template

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """
        在 epoch 结束时按间隔保存完整训练断点. 

        输入参数:
            - trainer: pl.Trainer, 当前训练器
            - pl_module: pl.LightningModule, 当前模型, 此处不直接读取

        输出:
            - None, 仅 rank0 写 checkpoint 文件
        """
        del pl_module
        if not bool(getattr(trainer, "is_global_zero", True)):
            return
        epoch_index = int(trainer.current_epoch)
        if (epoch_index + 1) % self.every_n_epochs != 0:
            return
        self.dirpath.mkdir(parents=True, exist_ok=True)
        filename = self.filename_template.format(epoch=epoch_index)
        checkpoint_path = self.dirpath / f"{filename}.ckpt"
        trainer.save_checkpoint(str(checkpoint_path))


def _get_config_name() -> str:
    """
    从命令行参数中解析 --config 参数, 用于指定 Hydra 配置文件名. 
    用法: python src/train.py --config baseline  或  python src/train.py --config=baseline
    默认值: "default"
    """
    import argparse
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", type=str, default="base",
                        help="Hydra config file name (without .yaml)")
    args, _ = parser.parse_known_args()
    # 从 sys.argv 中移除 --config 及其值, 防止 Hydra 解析时报错
    cleaned = []
    skip_next = False
    for i, arg in enumerate(sys.argv):
        if skip_next:
            skip_next = False
            continue
        if arg == "--config":
            skip_next = True
            continue
        if arg.startswith("--config="):
            continue
        cleaned.append(arg)
    sys.argv = cleaned
    return args.config

_CONFIG_NAME = _get_config_name()


def _has_uninitialized_parameters(module: torch.nn.Module) -> bool:
    """Return True when any parameter is still lazy/uninitialized."""
    from torch.nn.parameter import UninitializedParameter

    for p in module.parameters():
        if isinstance(p, UninitializedParameter):
            return True
    return False


def _needs_input_channel_initialization(model: torch.nn.Module) -> bool:
    """Return True when a backbone submodule is waiting for dataset input channels."""
    backbone = getattr(model, "backbone", None)
    maybe_compiled_backbone = getattr(backbone, "_orig_mod", backbone)
    # 共享/独立 density encoder 都可能 lazy in_channels, DDP 前都要 materialize
    for encoder_name in ("density_cube_encoder", "real_density_cube_encoder"):
        encoder = getattr(maybe_compiled_backbone, encoder_name, None)
        if encoder is not None and getattr(encoder, "in_channels", None) is None:
            return True
    return False


def _extract_input_tensor(sample):
    """Extract input tensor from dataset sample."""
    if torch.is_tensor(sample):
        return sample
    if isinstance(sample, dict):
        if "density_input" in sample and torch.is_tensor(sample["density_input"]):
            return sample["density_input"]
        if "voxel_grid" in sample and torch.is_tensor(sample["voxel_grid"]):
            return sample["voxel_grid"]
        raise TypeError("Unsupported dict sample format; expected density_input or voxel_grid Tensor.")
    if isinstance(sample, (list, tuple)) and len(sample) > 0 and torch.is_tensor(sample[0]):
        return sample[0]
    raise TypeError(
        "Unsupported sample format; expected Tensor, dict['voxel_grid'], or tuple/list with Tensor at index 0."
    )


def _initialize_lazy_modules_before_ddp(model: torch.nn.Module, datamodule: pl.LightningDataModule, verbose: bool = True) -> None:
    """
    Materialize lazy parameters before Lightning wraps model with DDP.
    """
    if not _has_uninitialized_parameters(model) and not _needs_input_channel_initialization(model):
        return

    datamodule.setup(stage="fit")
    if not hasattr(datamodule, "train_ds"):
        raise RuntimeError("Datamodule setup did not create train_ds; cannot initialize lazy modules.")

    sample = datamodule.train_ds[0]
    x = _extract_input_tensor(sample)
    if x.dim() == 4:
        in_channels = int(x.shape[0])  # C,D,H,W from dataset sample
    elif x.dim() == 5:
        in_channels = int(x.shape[1])  # B,C,D,H,W from pre-batched sample
    else:
        raise ValueError(f"Unexpected input tensor shape for lazy init: {tuple(x.shape)}")

    backbone = getattr(model, "backbone", None)
    maybe_compiled_backbone = getattr(backbone, "_orig_mod", backbone)
    if not hasattr(maybe_compiled_backbone, "set_input_channels"):
        raise RuntimeError(
            "Model has uninitialized parameters, but no set_input_channels() hook is available."
        )

    maybe_compiled_backbone.set_input_channels(in_channels)
    if verbose:
        voxel_backbone = getattr(maybe_compiled_backbone, "voxel_backbone", None)
        density_cube_encoder = getattr(maybe_compiled_backbone, "density_cube_encoder", None)
        voxel_backbone_in_channels = getattr(voxel_backbone, "in_channels", None)
        density_cube_in_channels = getattr(density_cube_encoder, "in_channels", None)
        print(
            "[Train] Lazy modules initialized before DDP: "
            f"raw [voxel_grid] channels={in_channels}, "
            f"[voxel_backbone] in_channels={voxel_backbone_in_channels}, "
            f"[density_cube] _in_channels={density_cube_in_channels}"
        )

    if _has_uninitialized_parameters(model):
        raise RuntimeError(
            "Model still has uninitialized parameters after pre-DDP initialization."
        )




# Hydra 将所有子配置合并成一个大的 cfg 对象传入 main 函数: main(cfg)
def _get_eager_backbone(model: torch.nn.Module) -> torch.nn.Module | None:
    """Return the underlying backbone, unwrapping torch.compile if needed."""
    backbone = getattr(model, "backbone", None)
    if backbone is None:
        return None
    return getattr(backbone, "_orig_mod", backbone)


def _prepare_model_for_batch_size_tuning(model: torch.nn.Module, verbose: bool = True) -> dict[str, object]:
    """Make batch-size probing follow a worst-case recycle path."""
    state: dict[str, object] = {}
    backbone = _get_eager_backbone(model)
    if backbone is None:
        return state

    if hasattr(backbone, "randomize_recycles"):
        state["randomize_recycles"] = getattr(backbone, "randomize_recycles")
        if getattr(backbone, "randomize_recycles"):
            setattr(backbone, "randomize_recycles", False)
            if verbose:
                print("[Train] Batch size tuning: force deterministic max recycle passes for worst-case memory probing")
    return state


def _restore_model_after_batch_size_tuning(model: torch.nn.Module, state: dict[str, object]) -> None:
    """Restore model attributes mutated for batch-size tuning."""
    backbone = _get_eager_backbone(model)
    if backbone is None:
        return
    for attr_name, attr_value in state.items():
        setattr(backbone, attr_name, attr_value)


def _find_strict_batch_size(safe_bs: int, num_devices: int, global_batch_size: int) -> int | None:
    """
    从 safe_bs 向下搜索满足 global_batch_size % (bs * num_devices) == 0 的最大 bs. 

    输入参数:
        - safe_bs: int, 标量, 安全系数缩放后的单卡 batch size 上界
        - num_devices: int, 标量, 当前训练的总设备数 (world_size)
        - global_batch_size: int, 标量, 目标全局 batch size

    输出:
        - strict_bs: int | None, 标量, 满足整除约束的最大 bs; 若不存在返回 None
    """
    for bs in range(safe_bs, 0, -1):
        if global_batch_size % (bs * num_devices) == 0:
            return bs
    return None


def _align_global_batch_size(
    per_device_bs: int,
    num_devices: int,
    global_batch_size: int,
    strict: bool,
    verbose: bool,
) -> tuple[int, int]:
    """
    根据 per_device_bs 与 num_devices 反算 accumulate_steps, 可选严格对齐. 

    输入参数:
        - per_device_bs: int, 标量, 当前单卡 batch size
        - num_devices: int, 标量, 总设备数
        - global_batch_size: int, 标量, 目标全局 batch size
        - strict: bool, 标量, 是否严格对齐 (向下微调 per_device_bs 使得整除)
        - verbose: bool, 标量, 是否打印调试信息

    输出:
        - (aligned_bs, accumulate_steps): tuple[int, int]
            - aligned_bs: int, 标量, 对齐后的单卡 batch size (strict=False 时等于输入)
            - accumulate_steps: int, 标量, 梯度累积步数
    """
    if strict:
        strict_bs = _find_strict_batch_size(per_device_bs, num_devices, global_batch_size)
        if strict_bs is not None:
            accumulate_steps = global_batch_size // (strict_bs * num_devices)
            return strict_bs, accumulate_steps
        else:
            if verbose:
                print(
                    f"[Train] [WARN] strict_global_batch_size 无法找到满足整除的 bs "
                    f"(safe_bs={per_device_bs}, devices={num_devices}, target={global_batch_size}), "
                    f"退回 floor-div 行为"
                )
    # 非严格模式或严格模式退回: 使用 floor-div
    accumulate_steps = max(1, global_batch_size // (per_device_bs * num_devices))
    return per_device_bs, accumulate_steps


def _resolve_val_check_interval(val_per_epoch: int) -> float:
    """
    Map the user-facing `val_per_epoch` setting onto Lightning's native
    validation scheduling.
    """
    val_per_epoch = max(1, int(val_per_epoch))
    return 1.0 / float(val_per_epoch)


class SeededEpochRandomSampler(Sampler[int]):
    """Deterministic train sampler whose index order depends only on seed and epoch."""

    def __init__(self, data_source, seed: int) -> None:
        self.data_source = data_source
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)
        set_dataset_epoch = getattr(self.data_source, "set_epoch", None)
        if callable(set_dataset_epoch):
            set_dataset_epoch(self.epoch)

    def __iter__(self):
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        indices = torch.randperm(len(self.data_source), generator=generator).tolist()
        return iter(indices)

    def __len__(self) -> int:
        return len(self.data_source)


class EpochAwareDistributedSampler(DistributedSampler):
    """在 DDP sampler epoch 切换时同步刷新动态 Stage1 请求源. """

    def set_epoch(self, epoch: int) -> None:
        super().set_epoch(epoch)
        set_dataset_epoch = getattr(self.dataset, "set_epoch", None)
        if callable(set_dataset_epoch):
            set_dataset_epoch(int(epoch))


class DatasetEpochController(Callback):
    """在每个训练 epoch 开始前同步动态请求源与 sampler 的 epoch. """

    @staticmethod
    def _set_epoch(trainer: pl.Trainer, epoch: int) -> None:
        """同步 DataModule 持有的 Dataset 与当前 train DataLoader sampler. """

        datamodule = getattr(trainer, "datamodule", None)
        train_dataset = getattr(datamodule, "train_ds", None)
        set_dataset_epoch = getattr(train_dataset, "set_epoch", None)
        if callable(set_dataset_epoch):
            set_dataset_epoch(int(epoch))
        train_loader = getattr(trainer, "train_dataloader", None)
        sampler = getattr(train_loader, "sampler", None)
        set_sampler_epoch = getattr(sampler, "set_epoch", None)
        if callable(set_sampler_epoch):
            set_sampler_epoch(int(epoch))

    def on_train_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """在当前 epoch 消费前刷新; 普通 Dataset 无副作用. """

        del pl_module
        self._set_epoch(trainer, int(trainer.current_epoch))

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """预先发布下一 epoch 请求, 覆盖 DataLoader 可能提前创建 iterator 的实现差异. """

        del pl_module
        self._set_epoch(trainer, int(trainer.current_epoch) + 1)


def _seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % (2 ** 32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)


@hydra.main(version_base="1.3", config_path="../configs", config_name=_CONFIG_NAME)


def main(cfg: DictConfig):
    """
    主训练入口点. 
    Args:
        - cfg: DictConfig, (Dict-like), Hydra 解析后的完整配置对象, 包含 model, dataset, train 等所有参数
    Returns:
        - None
    """
    
    if os.environ.get("RANK") is not None:
        print(
            f"[Train] Dist Env: RANK={os.environ.get('RANK')}, "
            f"LOCAL_RANK={os.environ.get('LOCAL_RANK')}, "
            f"WORLD_SIZE={os.environ.get('WORLD_SIZE')}"
        )

    train_seed = cfg.train.get("seed", None)
    if train_seed is not None:
        pl.seed_everything(int(train_seed), workers=bool(cfg.train.get("seed_workers", False)))
        print(f"[Train] Global seed set to {int(train_seed)}")

    # 1. -------------- 初始化实验管理器, 创建目录并立即保存配置 --------------
    # ExperimentManager, (Object), 自定义的实验管理器实例, 负责目录创建、配置备份和清理
    exp_manager = ExperimentManager(
        config=cfg,
        project_root=str(ROOT),
        feedback_root=str(FEEDBACK_ROOT),
        experiment_group=cfg.experiment_group,
    )
    # 更新 Pl Lightning 默认根目录为我们的新运行目录
    # str, (Path String), 实验反馈的保存路径, 格式: {feedback_root}/logs/{experiment_group}/{tag}____{timestamp}
    run_dir = str(exp_manager.run_dir)
    if exp_manager.is_rank_zero:
        os.makedirs(os.path.join(run_dir, "checkpoints"), exist_ok=True)
    
    

    # 2. -------------- 动态确定 Num Workers --------------
    req_workers = cfg.train.num_workers
    # 获取可用的CPU核心数
    try:
        avail_cores = len(os.sched_getaffinity(0))
    except AttributeError:
        avail_cores = os.cpu_count() or 1
    # 最终使用的Num Workers = min(requested, available)
    final_workers = min(req_workers, avail_cores)
    if exp_manager.is_rank_zero:
        print(f"[Train] Worker Config: Requested={req_workers}, Available={avail_cores}, Using={final_workers}")
    try:
        cfg.train.num_workers = final_workers
    except Exception as e:
        print(f"[Train] Could not update cfg.train.num_workers directly: {e}. Passing explicitly if possible.")



    # 3. -------------- 实例化 DataModule --------------
    if exp_manager.is_rank_zero:
        print(f"[Train] Instantiating DataModule with: {cfg.dataset.name}")
    class UnifiedDataModule(pl.LightningDataModule):
        """
        训练的统一数据模块. 
        """
        def __init__(self, dataset_cfg, train_cfg):
            """
            Args:
                - dataset_cfg: DictConfig, (Dict-like), 数据集相关的配置 (configs/dataset/...)
                - train_cfg: DictConfig, (Dict-like), 训练相关的配置 (configs/train/...)
            """
            super().__init__() 
            # DictConfig, (Dict-like), 保存数据集配置
            self.dataset_cfg = dataset_cfg
            # DictConfig, (Dict-like), 保存训练配置
            self.train_cfg = train_cfg
            
        @property
        def batch_size(self):
            return self.train_cfg.batch_size
            
        @batch_size.setter
        def batch_size(self, value):
            self.train_cfg.batch_size = value
            
        def setup(self, stage=None):  
            """
            初始化训练dataset (self.train_ds) 和验证dataset (self.val_ds)
            Args:
                - stage: str, (Optional), 当前阶段 ('fit', 'validate', 'test', 'predict')
            #NOTE: 训练时初始化Dataset的逻辑见这里(半通用), 它部分地依赖于特定dataset的参数, 依赖关系如下:
                - 表示"所有数据的根目录"的变量名只能是 all_data_path (若训练/验证相统一) ;  all_data_path_train, all_data_path_val (若训练/验证相分离).
                - 表示"训练/验证集划分文件"的变量名只能是 split_file, 且在配置文件中必须分别用变量名 split_train, split_val 来指定训练集和验证集的划分文件.
                - 在Dataset的初始化时, 必须含有变量名"mode", 它必须可以接受"train"与"val"(虽然可以通过内部转义扩大范围), 表示当前数据集用于训练或验证.
            """
            if stage == "fit" or stage is None:
                # 以下的两行代码允许了训练数据集和验证数据集的根目录可以不同, 它们通过"all_data_path_train"和"all_data_path_val"指定
                _train_data_path = self.dataset_cfg.get(   # 先找 "all_data_path_train" 再找 "all_data_path", 最后None
                    "all_data_path_train",
                    self.dataset_cfg.get("all_data_path", None)
                )
                _val_data_path = self.dataset_cfg.get(   # 先找 "all_data_path_val" 再找 "all_data_path", 最后None
                    "all_data_path_val",
                    self.dataset_cfg.get("all_data_path", None)
                )

                # 通过 Hydra 实例化训练数据集
                self.train_ds = hydra.utils.instantiate(
                    self.dataset_cfg,
                    split_file=self.dataset_cfg.split_train,
                    all_data_path=_train_data_path,
                    mode="train"
                )
                # 通过 Hydra 实例化验证数据集
                self.val_ds = hydra.utils.instantiate(
                    self.dataset_cfg,
                    split_file=self.dataset_cfg.split_val,
                    all_data_path=_val_data_path,
                    mode="val"
                )

                # --- 容错包装: 坏样本自动替换, 防止 DDP 卡死 ---
                _ft_cfg = self.train_cfg.get("data_fault_tolerance", None)
                if _ft_cfg is not None:
                    self.train_ds = maybe_wrap_dataset(self.train_ds, "train", _ft_cfg)
                    self.val_ds = maybe_wrap_dataset(self.val_ds, "val", _ft_cfg)

        def _get_stage_seed(self, stage: str) -> int:
            base_seed = int(self.train_cfg.get("seed", 0) or 0)
            offsets = self.train_cfg.get("dataloader_seed_offsets", {})
            return base_seed + int(offsets.get(stage, 0))

        def _build_sampler(self, ds, stage: str, shuffle: bool):
            stage_seed = self._get_stage_seed(stage)
            world_size = int(getattr(self.trainer, "world_size", 1) or 1)

            if world_size > 1:
                return EpochAwareDistributedSampler(ds, shuffle=(stage == "train" and shuffle), seed=stage_seed)
            if stage == "train" and shuffle:
                return SeededEpochRandomSampler(ds, seed=stage_seed)
            return None

        def _get_dataloader(self, ds, stage: str, shuffle: bool = False):
            """
            统一 DataLoader 的创建逻辑. 
            支持根据样本类型自动选择 torch_geometric 或 torch.utils.data.DataLoader
            """
            backend = self.train_cfg.dataloader_backend
            
            # 探测第一个样本
            use_pyg = False
            if backend.lower() == "pyg":
                use_pyg = True
            elif backend.lower() == "torch":
                use_pyg = False
            else: # "auto"
                try:
                    import torch_geometric
                    sample = ds[0]
                    from torch_geometric.data import Data, Batch
                    if isinstance(sample, (Data, Batch)):
                        use_pyg = True
                except ImportError:
                    use_pyg = False
            
            if use_pyg:
                import torch_geometric.loader
                loader_class = torch_geometric.loader.DataLoader
            else:
                loader_class = torch.utils.data.DataLoader

            sampler = self._build_sampler(ds, stage=stage, shuffle=shuffle)
            generator = torch.Generator()
            generator.manual_seed(self._get_stage_seed(stage))
            collate_fn = None if use_pyg else getattr(ds, "collate_fn", None)
                
            loader_arguments = {
                "dataset": ds,
                "batch_size": self.train_cfg.batch_size,
                "shuffle": bool(shuffle and sampler is None),
                "sampler": sampler,
                "num_workers": self.train_cfg.num_workers,
                "pin_memory": self.train_cfg.get("pin_memory", True) if sys.platform != "win32" else False,
                "collate_fn": collate_fn,
                "worker_init_fn": _seed_worker,
                "generator": generator,
            }
            if int(self.train_cfg.num_workers) > 0:
                loader_arguments["prefetch_factor"] = int(self.train_cfg.get("prefetch_factor", 4))
                # Dataset 请求会随训练周期切换；常驻 worker 无法接收主进程中的新请求表。
                loader_arguments["persistent_workers"] = False
            return loader_class(**loader_arguments)

        def train_dataloader(self):
            """
            Returns:
                - DataLoader, (torch.utils.data.DataLoader), 训练数据加载器
            """
            return self._get_dataloader(self.train_ds, stage="train", shuffle=True)
            
        def val_dataloader(self):
            """
            Returns:
                - DataLoader, (torch.utils.data.DataLoader), 验证数据加载器
            """
            return self._get_dataloader(self.val_ds, stage="val", shuffle=False)
    
    dm = UnifiedDataModule(cfg.dataset, cfg.train)



    # 4. -------------- 实例化模型 --------------
    if exp_manager.is_rank_zero:
        print(f"[Train] 实例化(Instantiate)模型的名字: {cfg.model.name}")
        print(f"[Train] 实例化(Instantiate)模型的路径: {cfg.model._target_}")
    global_batch_size = cfg.train.get("global_batch_size", None)
    enable_batch_size_tuning = bool(cfg.train.get("enable_batch_size_tuning", False))
    batch_size_tuning_enabled = (
        enable_batch_size_tuning
        and global_batch_size is not None
        and global_batch_size > 0
    )
    compile_requested = bool(cfg.model.get("compile", False))
    compile_deferred = bool(compile_requested and batch_size_tuning_enabled)
    if compile_requested and exp_manager.is_rank_zero:
        print("[Train] torch.compile 将在 lazy 初始化、model-only 加载与冻结之后执行")

    model = hydra.utils.instantiate(
        cfg.model,
        optimizer=cfg.train.optimizer,
        scheduler=cfg.train.scheduler,
        compile=False,
    )
    _initialize_lazy_modules_before_ddp(model, dm, verbose=exp_manager.is_rank_zero)
    init_from = cfg.get("init_from", None)
    if init_from is not None:
        if str(init_from).strip() == "***" and exp_manager.is_rank_zero:
            print("[Train] [WARN] init_from='***'，将按当前 SLURM_JOB_ID 自动解析上一阶段 checkpoints/BEST.ckpt。")
        ckpt_path = _resolve_init_checkpoint(
            init_from=str(init_from),
            feedback_root=FEEDBACK_ROOT,
            current_run_dir=Path(run_dir),
        )
        _load_model_only_checkpoint(model=model, ckpt_path=ckpt_path, verbose=exp_manager.is_rank_zero)
    frozen_cfg = cfg.get("frozen_module", None)
    if frozen_cfg is not None:
        _apply_frozen_module(model=model, frozen_cfg=frozen_cfg, verbose=exp_manager.is_rank_zero)
    if compile_requested and not compile_deferred:
        if exp_manager.is_rank_zero:
            print("[Train] 开始执行 torch.compile(backbone)...")
        model.backbone = torch.compile(model.backbone)
    _fix_gloo_socket_ifname()
    _log_distributed_launch_state("LazyInit完成")



    # 5. -------------- 日志记录器 --------------
    prefer_online = not cfg.offline
    wandb_mode = _setup_wandb_mode(prefer_online=prefer_online, verbose=exp_manager.is_rank_zero)
    is_offline = (wandb_mode == "offline")

    # bool, wandb 日志名称格式: True = "{model}-{dataset}-{tag}", False = "{tag}"
    _wandb_long_name = bool(cfg.get("wandb_log_LongName", False))
    if _wandb_long_name:
        _wandb_run_name = f"{cfg.model.name}-{cfg.dataset.name}-{cfg.tag}"
    else:
        _wandb_run_name = f"{cfg.tag}"

    logger = WandbLogger(
        project=cfg.project_name,
        name=_wandb_run_name,
        save_dir=run_dir,
        offline=is_offline,
        log_model=False
    )

    # ---- WandB 在线模式真实连通性兜底 ----
    # WandbLogger 构造时不会调用 wandb.init(), 而是延迟到 trainer.fit() 内部
    # 首次访问 logger.experiment 时才触发. 如果此时超时, 会直接崩溃且无法回退. 
    # 因此在此处主动提前触发 wandb.init(), 捕获任何异常后自动回退为离线模式. 
    # 注意: 由于此时 Lightning 尚未初始化分布式环境, 只有真实 Rank 0 才能访问 logger.experiment, 否则会导致多进程并发写 WandB 发生死锁. 
    if not is_offline and exp_manager.is_rank_zero:
        try:
            _ = logger.experiment          # 触发 wandb.init()
            print("[Train] [OK] WandB 在线初始化成功 (wandb.init() succeeded)")
        except Exception as _wandb_err:
            print(f"[Train] [X] WandB 在线初始化失败: {_wandb_err}")
            print("[Train]      自动回退到离线模式 (Falling back to offline mode)...")
            # 尝试清理失败的 wandb run
            try:
                wandb.finish()
            except Exception:
                pass
            os.environ["WANDB_MODE"] = "offline"
            is_offline = True
            
    # 如果 Rank 0 决定降级为离线模式, 在此重建 logger. 
    # 其他进程的 logger 可能仍是在线模式对象, 但在其他进程仅是 Dummy, 不影响训练. 
    if is_offline and exp_manager.is_rank_zero:
        logger = WandbLogger(
            project=cfg.project_name,
            name=_wandb_run_name,
            save_dir=run_dir,
            offline=True,
            log_model=False
        )
        print(f"[Train]   WandB 已启用离线模式 (Offline Mode)。")
        print(f"[Train]   训练将在本地生成离线日志: {run_dir}/wandb/")
        print("[Train]   训练结束后，请使用 'wandb sync' 或配套的同步脚本上传到云端。")



    # 6. -------------- 模型检查点回调(Callback), 用于保存最佳模型 --------------
    # monitor 和 monitor_mode 统一从 cfg.model 读取, 与 Wrapper 中 self.log() 和 configure_optimizers() 使用的 monitor_metric 保持一致
    _monitor = cfg.model.monitor_metric
    _monitor_mode = cfg.model.monitor_mode
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(run_dir, "checkpoints"),    # str, save path
        filename="TOP_epoch_{epoch:02d}_score_{" + _monitor + ":.4f}", # str, 带有 TOP_ 前缀的文件名
        auto_insert_metric_name=False,                   # bool, 关闭指标自动拼接, 防止基于 '/' 创建意外的子文件夹
        monitor=_monitor,
        mode=_monitor_mode,                              # str, 'min' or 'max'
        save_top_k=cfg.output.save_top_k,                # int, number of models to save
        save_last=True,                                  # bool, save last epoch
    )
    best_alias_callback = BestCheckpointAlias(
        checkpoint_callback=checkpoint_callback,
        alias_path=Path(run_dir) / "checkpoints" / "BEST.ckpt",
    )
    # 建立最初的回调列表
    callbacks = [checkpoint_callback, best_alias_callback, DatasetEpochController()]

    # ------ 额外开启周期性保存 ------
    # 从配置中获取 save_every_n_epochs (例如: 10)
    save_every_n_epochs = cfg.output.get("save_every_n_epochs", None)
    if save_every_n_epochs is not None and save_every_n_epochs > 0:
        periodic_checkpoint = PeriodicCheckpointSaver(
            dirpath=Path(run_dir) / "checkpoints",
            filename_template="PERIODIC_epoch_{epoch:02d}",  # str, 带有 PERIODIC_ 前缀的文件名
            every_n_epochs=int(save_every_n_epochs),         # int, 每隔这么多个 epoch 保存一次
        )
        callbacks.append(periodic_checkpoint)

    # LearningRateMonitor, (Callback), 学习率监控
    lr_monitor = LearningRateMonitor(logging_interval="step")
    # RichProgressBar, (Callback), 终端进度条
    rich_bar = RichProgressBar()
    
    # 将进度条与学习率监控加入 list
    callbacks.extend([lr_monitor, rich_bar])
    if cfg.train.scheduler.get("name", None) == "warmup_plateau":
        callbacks.append(
            WarmupPlateauController(
                sched_cfg=OmegaConf.to_container(cfg.train.scheduler, resolve=True),
                monitor_metric=str(cfg.model.monitor_metric),
            )
        )
        if exp_manager.is_rank_zero:
            print("[Train] warmup_plateau plateau controller enabled in train.py.")

    stop_after_lr_reductions = cfg.train.scheduler.get("stop_after_lr_reductions", None)
    if stop_after_lr_reductions is not None:
        callbacks.append(LearningRateReductionStopper(stop_after_lr_reductions=int(stop_after_lr_reductions)))
        if exp_manager.is_rank_zero:
            print(f"[Train] LR reduction stopper enabled: stop_after_lr_reductions={int(stop_after_lr_reductions)}")

    # --- 每个 epoch 内多次验证: 使用 Lightning 原生 val_check_interval ---
    val_per_epoch = cfg.train.get("val_per_epoch", 1)
    val_check_interval = _resolve_val_check_interval(val_per_epoch)
    if exp_manager.is_rank_zero and int(val_per_epoch) > 1:
        print(
            "[Train] Multi-validation per epoch enabled via native Lightning scheduling: "
            f"val_per_epoch={int(val_per_epoch)}, val_check_interval={val_check_interval:.6f}"
        )




    # '训练时测试' 这个功能暂时关闭
    # # 7. -------------- 周期性测试回调(支持多个测试集、多种回调机制) --------------(本条目暂略)
    # # dict, (Dict[str, DataLoader]), 存储测试集 DataLoader 的字典
    # test_dataloaders = {}
    # # 从 cfg.dataset.split_test 获取测试集配置
    # if "split_test" in cfg.dataset and cfg.dataset.split_test is not None:

    #     # 这要求 cfg.dataset.split_test是字典或列表(都兼容), 比如 {'test_set_name1': 'path/to/test1.json', 'test_set_name2': 'path/to/test2.json', ...}
    #     splits = OmegaConf.to_container(cfg.dataset.split_test, resolve=True)
    #     if isinstance(splits, (list, tuple)):
    #         splits = {f"test_{i}": path for i, path in enumerate(splits)}
    #     for name, split_path in splits.items():    
    #         if os.path.exists(split_path):
    #             if exp_manager.is_rank_zero:
    #                 print(f"[Train] Loading Test Split for Callback.训练时测试,测试集名字是: {name}、划分路径是 {split_path}")
    #             # 通过 Hydra 实例化测试数据集 (路径由 dataset.all_data_path_test 显式指定)
    #             ds = hydra.utils.instantiate(
    #                 cfg.dataset,
    #                 split_file=split_path,
    #                 all_data_path=cfg.dataset.all_data_path_test,
    #                 mode="test"
    #             )
    #             dl = dm._get_dataloader(ds, shuffle=False)
    #             test_dataloaders[name] = dl
    #         else:
    #             if exp_manager.is_rank_zero:
    #                 print(f"[Train] 警告: 测试集{name}或者划分文件未找到: {split_path}. 请检查路径. ")
    # else:
    #     if exp_manager.is_rank_zero:
    #         print("[Train] No 'split_test' found in dataset config. Periodic testing disabled.")
        

    # # 暂且不实现 periodic_test.py, 但在此留下接口并说明. 
    # # 回调类(训练时测试)
    # # 实例化类是 src.callbacks.periodic_test.PeriodicTestCallback ,只需传入通用参数 dataloaders_dict=test_dataloaders (如前所述的字典) 和 interval (间隔的epoch)
    # if len(test_dataloaders) > 0:
    #     # 此回调的作用是在训练过程中（特定epoch间隔）调用测试流程, 以监控全集或其它测试集的表现. 
    #     # 未来的实现应大部分调用推断程序 (src/infer.py) 中的代码逻辑（例如导入相关推断函数）, 
    #     # 从而避免在训练代码中重新写一遍推断逻辑并确保二者一致性. 
    #     #
    #     # 下面为原本的调用接口, 当前已注释: 

    #     # callback_class = hydra.utils.get_class("src.callbacks.periodic_test.PeriodicTestCallback")
    #     # periodic_eval_cb = callback_class(
    #     #     dataloaders_dict=test_dataloaders,
    #     #     interval=cfg.train.get("test_interval", 50)
    #     # )
    #     # callbacks.append(periodic_eval_cb)
    #     if exp_manager.is_rank_zero:
    #         print("[Train] TODO: callbacks.periodic_test 暂未实现. 正在按要求显式跳过周期测试回调. ")


    # # 可视化回调(暂略)
    # # 配置中支持 cfg.output.visualization 这个条目, 开关为 cfg.output.visualization.enabled
    # if cfg.output.get("visualization", {}).get("enabled", False):
    #     from src.callbacks.visualization_callback import VisualizationCallback
    #     vis_callback = VisualizationCallback(
    #         run_dir=run_dir,
    #         vis_config=cfg.output.visualization,
    #         dataset_config=cfg.dataset,
    #         model_path=cfg.model._target_
    #     )
    #     callbacks.append(vis_callback)
    #     if exp_manager.is_rank_zero:
    #         print(f"[Train] VisualizationCallback enabled, interval_plot={cfg.output.visualization.get('interval_plot', 300)}")







    # 8. -------------- 训练器 --------------
    use_distributed = (cfg.train.devices > 1 or cfg.train.nnodes > 1)
    find_unused = bool(cfg.train.get("ddp_find_unused_parameters", False))
    strategy = "auto"
    if use_distributed:
        from lightning.pytorch.strategies import DDPStrategy
        strategy = DDPStrategy(
            find_unused_parameters=find_unused,
            process_group_backend="nccl"       # 强制使用 NCCL 后端, 避免 Gloo 辅助进程组导致的网络接口问题
        )

    trainer = pl.Trainer(
        default_root_dir=run_dir, # str, save path
        accelerator=cfg.train.accelerator,                # str, hardware accelerator
        devices=cfg.train.devices,                        # int, number of GPUs
        num_nodes=cfg.train.nnodes,                       # int, number of nodes
        strategy=strategy,                                # str, distributed strategy
        max_epochs=cfg.train.max_epochs,                  # int, max epochs
        max_steps=cfg.train.get("max_steps", -1),          # int, 可选调试上限; -1 表示按 max_epochs 训练
        logger=logger,                                    # Logger
        callbacks=callbacks,                              # List[Callback]
        precision=cfg.train.precision,                    # str, mixed precision setting
        gradient_clip_val=cfg.train.gradient_clip_val,    # float, gradient clipping
        accumulate_grad_batches=cfg.train.get("accumulate_grad_batches", 1), # int, 默认 1
        check_val_every_n_epoch=cfg.train.check_val_every_n_epoch, # int
        val_check_interval=val_check_interval,            # float, validate val_per_epoch times per epoch
        limit_train_batches=cfg.train.get("limit_train_batches", None), # int|float|None, 调试时限制训练 batch 数
        limit_val_batches=cfg.train.get("limit_val_batches", None),     # int|float|None, 调试时限制验证 batch 数
        num_sanity_val_steps=cfg.train.get("num_sanity_val_steps", 2),
        use_distributed_sampler=False,
        log_every_n_steps=cfg.train.get("log_every_n_steps", 10), # int, 控制wandb记录日志的频率
    )
    



    # -------------- 8.5 自动推导全局 Batch Size 与 Accumulate Steps --------------
    if global_batch_size is not None and global_batch_size > 0:
        strict_global = bool(cfg.train.get("strict_global_batch_size", False))
        num_devices = trainer.world_size

        if batch_size_tuning_enabled:
            # ---- 分支 A: 启用自动搜索 ----
            if exp_manager.is_rank_zero:
                print(f"[Train] 开始自动探测显存，寻找最佳 Batch Size (目标 Global Batch = {global_batch_size})...")

            from lightning.pytorch.tuner import Tuner
            tuner = Tuner(trainer)
            init_val = cfg.train.get("tuning_init_batch_size", 2)
            tuning_state = _prepare_model_for_batch_size_tuning(model, verbose=exp_manager.is_rank_zero)
            try:
                tuner.scale_batch_size(
                    model,
                    datamodule=dm,
                    mode="binsearch",
                    steps_per_trial=cfg.train.get("tuning_steps_per_trial", 5),
                    init_val=init_val,
                    max_trials=25
                )
            finally:
                _restore_model_after_batch_size_tuning(model, tuning_state)

            # ======= 防护: 重新播种以消除 Tuner 的全局随机态消耗 =======
            if train_seed is not None:
                pl.seed_everything(int(train_seed), workers=bool(cfg.train.get("seed_workers", False)))
                if exp_manager.is_rank_zero:
                    print(f"[Train] Tuner探测完毕，已重新播种({int(train_seed)})以抵消Tuner执行前向传播对全局随机状态的消耗污染。")
                    
            # Tuner 只采样少量 batch; 留出余量应对真实训练中更重的 batch
            found_bs = int(dm.batch_size)
            safety_factor = float(cfg.train.get("batch_size_tuning_safety_factor", 0.7))
            safe_bs = max(1, int(found_bs * safety_factor))
            if safe_bs > found_bs:
                safe_bs = found_bs
            if found_bs > 1 and safe_bs == found_bs and safety_factor < 1.0:
                safe_bs = found_bs - 1

            # 严格对齐: 向下微调 safe_bs 使得 global_batch_size 能被 (bs * devices) 整除
            pre_align_bs = safe_bs
            safe_bs, accumulate_steps = _align_global_batch_size(
                safe_bs, num_devices, global_batch_size, strict_global,
                verbose=exp_manager.is_rank_zero,
            )

            dm.batch_size = safe_bs
            try:
                cfg.train.batch_size = safe_bs
            except Exception:
                pass
            trainer.accumulate_grad_batches = accumulate_steps

            if exp_manager.is_rank_zero:
                print(f"[Train] => Target Global Batch Size: {global_batch_size}")
                print(f"[Train] => Initial per-device guess: {init_val}")
                print(f"[Train] => Tuned per-device batch size: {found_bs}")
                print(f"[Train] => Safe per-device batch size (before align): {pre_align_bs} (safety_factor={safety_factor:.2f})")
                print(f"[Train] => Final per-device batch size: {safe_bs}")
                print(f"[Train] => Accumulate Steps: {accumulate_steps}")
                print(f"[Train] => world_size: {num_devices}")
                effective = safe_bs * num_devices * accumulate_steps
                print(f"[Train] => Effective Global Batch Size: {effective}")
                if strict_global and effective == global_batch_size:
                    print(f"[Train] => [OK] 严格对齐成功: effective == target")
                elif strict_global:
                    print(f"[Train] => [WARN] 严格对齐未能完全匹配 (effective={effective}, target={global_batch_size})")
        else:
            # ---- 分支 B: 不启用搜索, 使用 batch_size 预设值 ----
            per_device_bs = int(cfg.train.batch_size)
            pre_align_bs = per_device_bs
            per_device_bs, accumulate_steps = _align_global_batch_size(
                per_device_bs, num_devices, global_batch_size, strict_global,
                verbose=exp_manager.is_rank_zero,
            )

            if per_device_bs != pre_align_bs:
                dm.batch_size = per_device_bs
                try:
                    cfg.train.batch_size = per_device_bs
                except Exception:
                    pass
            trainer.accumulate_grad_batches = accumulate_steps

            if exp_manager.is_rank_zero:
                print(f"[Train] Batch size tuning 未启用, 使用预设 per-device batch_size")
                print(f"[Train] => Target Global Batch Size: {global_batch_size}")
                print(f"[Train] => Preset per-device batch size: {pre_align_bs}")
                if per_device_bs != pre_align_bs:
                    print(f"[Train] => Aligned per-device batch size: {per_device_bs} (strict_global_batch_size={strict_global})")
                print(f"[Train] => Accumulate Steps: {accumulate_steps}")
                print(f"[Train] => world_size: {num_devices}")
                effective = per_device_bs * num_devices * accumulate_steps
                print(f"[Train] => Effective Global Batch Size: {effective}")

    if compile_deferred:
        if exp_manager.is_rank_zero:
            print("[Train] 自动 Batch Size 探测完成, 开始执行 torch.compile(backbone)...")
        model.backbone = torch.compile(model.backbone)




    # 9. -------------- 开始训练 --------------
    if exp_manager.is_rank_zero:
        print("============================================================")
        print(f" Starting Training")
        print(f" Model: {cfg.model.name}")
        print(f" Dataset: {cfg.dataset.name}")
        print(f" PostProcess: {cfg.post_process.name}")
        print(f" Output Dir: {run_dir}")
        print("============================================================")
    try:
        # trainer.fit 触发整个 Lightning 生命周期: 
        # 1. dm.setup("fit") → 创建 train/val 数据集
        # 2. model.configure_optimizers() → 从 cfg.train 创建 optimizer 与 step 级 scheduler
        # 3. 每个 batch: model.training_step() → _extract_batch → forward → _compute_total_loss → log("train_loss/*")
        # 4. 每次 validation 结束: wrapper 聚合 payload; train.py 的 callbacks 负责 checkpoint、plateau 与 small-increment 停训
        _fix_gloo_socket_ifname()
        _log_distributed_launch_state("即将调用trainer.fit")
        trainer.fit(model, datamodule=dm)
    except Exception as e:
        print(f"[Train] Critical Exception occurred(严重异常): {e}")
        exp_manager.check_and_cleanup(error=e)
        raise e
    finally:
        # 如果没有传递 error, check_and_cleanup 视为正常退出 (会检查是否太短)
        exp_manager.check_and_cleanup()

if __name__ == "__main__":
    main()
