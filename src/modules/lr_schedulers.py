from __future__ import annotations

from typing import Any

import torch


class WarmupThenReduceLROnPlateau:
    """
    组合 step 级 warmup 与 validation 级 ReduceLROnPlateau。

    输入参数:
        - optimizer: torch.optim.Optimizer, 被调度的优化器
        - warmup_steps: int, warmup 覆盖的 optimizer step 数; 0 表示不做 warmup gate
        - warmup_start_factor: float, warmup 起始学习率比例; 建议值 0.33
        - mode: str, plateau 指标方向, 取值 "min" 或 "max"
        - factor: float, plateau 触发后学习率乘数; 建议值 0.5
        - patience: int, PyTorch 原生 patience; 3 表示第 4 次 bad validation 衰减
        - threshold: float, 指标改进阈值
        - threshold_mode: str, 阈值模式, 取值 "abs" 或 "rel"
        - cooldown: int, 衰减后的冷却 validation 次数
        - min_lr: float 或 list[float], 学习率下界
        - eps: float, ReduceLROnPlateau 的最小学习率变化阈值

    输出:
        - 对象本身保存 warmup scheduler 与 plateau scheduler; warmup 交给 Lightning step, plateau 由 wrapper 在 validation end 手动 step
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        *,
        warmup_steps: int,
        warmup_start_factor: float,
        mode: str,
        factor: float,
        patience: int,
        threshold: float,
        threshold_mode: str,
        cooldown: int,
        min_lr: float | list[float],
        eps: float,
    ) -> None:
        self.optimizer = optimizer
        self.warmup_steps = int(warmup_steps)
        if self.warmup_steps < 0:
            raise ValueError(f"warmup_steps must be >= 0, got {self.warmup_steps}.")

        # float, warmup 起始学习率比例
        start_factor = float(warmup_start_factor)
        if not (0.0 < start_factor <= 1.0):
            raise ValueError(f"warmup_start_factor must be in (0, 1], got {start_factor}.")

        if self.warmup_steps > 0 and start_factor < 1.0:
            # LinearLR, step 级 warmup 调度器, Lightning 会在每个 optimizer step 后调用
            self.warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=start_factor,
                end_factor=1.0,
                total_iters=self.warmup_steps,
            )
        else:
            # LambdaLR, 恒等调度器, 保持 Lightning lr_scheduler 接口稳定
            self.warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _: 1.0)

        # ReduceLROnPlateau, validation 级主指标停滞衰减调度器
        self.plateau_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=mode,
            factor=float(factor),
            patience=int(patience),
            threshold=float(threshold),
            threshold_mode=threshold_mode,
            cooldown=int(cooldown),
            min_lr=min_lr,
            eps=float(eps),
        )

    def lightning_warmup_config(self) -> dict[str, Any]:
        """
        返回 Lightning 管理 warmup scheduler 所需的配置字典。

        输出:
            - config: dict[str, Any], 包含 scheduler/interval/frequency/name 的 Lightning lr_scheduler 配置
        """
        return {
            "scheduler": self.warmup_scheduler,
            "interval": "step",
            "frequency": 1,
            "name": "warmup_lr",
        }

    def step_plateau(self, metric: torch.Tensor | float, *, global_step: int) -> bool:
        """
        在 validation end 后按主指标推进 plateau scheduler。

        输入参数:
            - metric: torch.Tensor 或 float, 当前 validation 聚合后的主指标值
            - global_step: int, 当前 Lightning optimizer step 计数; 小于 warmup_steps 时跳过 plateau

        输出:
            - lr_reduced: bool, 本次调用是否降低了至少一个 param group 的学习率
        """
        if int(global_step) < self.warmup_steps:
            return False

        # list[float], 每个 param group 调度前的学习率
        lr_before = [float(group["lr"]) for group in self.optimizer.param_groups]
        self.plateau_scheduler.step(float(metric))
        # list[float], 每个 param group 调度后的学习率
        lr_after = [float(group["lr"]) for group in self.optimizer.param_groups]
        return any(after < before for before, after in zip(lr_before, lr_after))

    def state_dict(self) -> dict[str, Any]:
        """
        导出手动管理的 plateau scheduler 状态。

        输出:
            - state: dict[str, Any], 可写入 Lightning checkpoint 的 plateau 状态字典
        """
        return {
            "warmup_steps": self.warmup_steps,
            "plateau_scheduler": self.plateau_scheduler.state_dict(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """
        恢复手动管理的 plateau scheduler 状态。

        输入参数:
            - state: dict[str, Any], state_dict() 导出的状态字典

        输出:
            - None, 原地恢复 plateau scheduler 状态
        """
        self.plateau_scheduler.load_state_dict(state["plateau_scheduler"])

    def get_last_lr(self) -> list[float]:
        """
        读取 optimizer 当前学习率。

        输出:
            - last_lr: list[float], 每个 param group 当前学习率
        """
        return [float(group["lr"]) for group in self.optimizer.param_groups]
