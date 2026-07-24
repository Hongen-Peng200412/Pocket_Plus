"""累计并发布 Stage1 验证集的二分类和多分类 PRAUC。

主要入口 :class:`ValidationMetricManager` 为每个启用的预测分支保存独立的
TorchMetrics 状态。二分类分支直接统计前景 PRAUC；蛋白和核酸多分类分支
忽略背景通道，把每个主链原子类别分别视为唯一正类，再对验证集中实际出现的
正类求宏平均。该模块不决定 BEST checkpoint，日志键由调用方统一消费。
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import nn
from torchmetrics import Metric
from torchmetrics.classification import BinaryAveragePrecision

from src.wrappers.voxel_point_stage1_logging import build_metric_key


@dataclass(frozen=True)
class MetricBranchSpec:
    """
    一个 Stage1 验证预测分支的 PRAUC 统计契约。

    输入参数:
        - name: str，预测分支名称，例如 atom、receptor 或 protein_mainchain。
        - enabled: bool，是否为该分支创建并更新指标状态。
        - num_classes: int，类别总数；包含背景类别。
        - class_names: tuple[str, ...]，长度为 num_classes，顺序与 logits 类别维一致。
        - thresholds: int | Sequence[float] | None，PRAUC 的概率阈值；None 表示不分箱。
        - report_per_class: bool，是否分别记录每个非背景类别。
        - macro_present_classes_only: bool，宏平均是否跳过验证集中没有正样本的类别。
    """

    name: str
    enabled: bool
    num_classes: int
    class_names: tuple[str, ...]
    thresholds: int | Sequence[float] | None
    report_per_class: bool = True
    macro_present_classes_only: bool = False


class _PositiveCount(Metric):
    """跨验证进程累计一个 one-vs-rest 类别的正样本数。"""

    def __init__(self) -> None:
        super().__init__()
        self.add_state(
            "count",
            default=torch.tensor(0, dtype=torch.int64),
            dist_reduce_fx="sum",
        )

    def update(self, target: torch.Tensor) -> None:
        self.count += target.to(device=self.count.device, dtype=torch.int64).sum()

    def compute(self) -> torch.Tensor:
        return self.count


class ValidationMetricManager(nn.Module):
    """
    管理 Stage1 验证集各预测分支的 PRAUC 状态。

    输入参数:
        - branches: Sequence[MetricBranchSpec]，需要统计的预测分支配置。
        - metric_device_policy: str，指标状态设备策略；允许 auto、cpu、gpu。

    内部重要方法：
        - _register_branch_metrics: 为一个分支注册 TorchMetrics 对象, 也就是创建 BinaryAveragePrecision(...)
        - update_branch: 用一个 batch 更新指定分支 AP/PRAUC 指标
        - compute_payload: 计算当前验证轮次的 ``日志键 -> PRAUC 标量`` 字典。
        - reset: 重置所有 TorchMetrics 状态
    """

    def __init__(self, *, branches: Sequence[MetricBranchSpec], metric_device_policy: str) -> None:
        super().__init__()
        if metric_device_policy not in {"auto", "cpu", "gpu"}:
            raise ValueError("metric_device_policy 只允许 auto、cpu 或 gpu。")
        # str, 指标设备策略
        self.metric_device_policy = str(metric_device_policy)
        # dict[str, MetricBranchSpec], 分支名到配置的映射
        self.branch_specs = {spec.name: spec for spec in branches if spec.enabled}
        # nn.ModuleDict, TorchMetrics 指标模块树
        self.metrics = nn.ModuleDict()
        # object | None, 两卡 NCCL 训练中供 CPU 指标状态同步使用的 Gloo 通信组
        self._cpu_metric_process_group: object | None = None
        for spec in self.branch_specs.values():
            self._register_branch_metrics(spec)

    def _configure_distributed_process_groups(self) -> None:
        """
        为 NCCL 多进程训练中的 CPU 指标状态配置 Gloo 通信组。

        TorchMetrics 的非分箱 PRAUC 会把可变长度状态保存在 CPU；NCCL 不能直接
        聚合 CPU 张量，因此这些指标必须改用同时支持 CPU 张量的 Gloo 通信组。

        输出:
            - None, 原地设置使用 CPU 状态的 TorchMetrics 对象
        """
        if self._cpu_metric_process_group is not None:
            return
        if not torch.distributed.is_available() or not torch.distributed.is_initialized():
            return
        if torch.distributed.get_world_size() <= 1:
            return
        if str(torch.distributed.get_backend()).lower() != "nccl":
            return
        # list[Metric], 使用 CPU 保存非分箱 PRAUC 状态的指标对象
        cpu_metrics = [
            metric for metric in self.metrics.values() if bool(metric.compute_on_cpu)
        ]
        if not cpu_metrics:
            return
        process_group = torch.distributed.new_group(backend="gloo")
        self._cpu_metric_process_group = process_group
        for metric in cpu_metrics:
            metric.process_group = process_group

    def _metric_compute_on_cpu(self, thresholds: int | Sequence[float] | None) -> bool:
        """
        判断当前 AP 指标是否应把非 binned 状态放在 CPU。

        输入参数:
            - thresholds: int | Sequence[float] | None, TorchMetrics AP 阈值配置

        输出:
            - compute_on_cpu: bool, True 表示 TorchMetrics list state 在 CPU 累积
        """
        if self.metric_device_policy == "cpu":
            return True
        if self.metric_device_policy == "gpu":
            return False
        return thresholds is None

    def _register_branch_metrics(self, spec: MetricBranchSpec) -> None:
        """
        为一个分支注册 TorchMetrics 对象。

        输入参数:
            - spec: MetricBranchSpec, 单个分支配置

        输出:
            - None, 原地写入 self.metrics: self.metrics[f"{spec.name}__class_{class_id}"] = BinaryAveragePrecision(...)
        """
        if spec.num_classes < 2:
            raise ValueError("MetricBranchSpec.num_classes 必须 >= 2。")
        if len(spec.class_names) != spec.num_classes:
            raise ValueError("MetricBranchSpec.class_names 长度必须等于 num_classes。")
        compute_on_cpu = self._metric_compute_on_cpu(spec.thresholds)
        self.metrics[f"{spec.name}__binary"] = BinaryAveragePrecision(
            thresholds=spec.thresholds,
            compute_on_cpu=compute_on_cpu,
        )
        if spec.num_classes > 2:
            for class_id in range(1, spec.num_classes):
                self.metrics[f"{spec.name}__class_{class_id}"] = BinaryAveragePrecision(
                    thresholds=spec.thresholds,
                    compute_on_cpu=compute_on_cpu,
                )
                if spec.macro_present_classes_only:
                    self.metrics[f"{spec.name}__class_{class_id}_positive_count"] = (
                        _PositiveCount()
                    )

    def update_branch(
        self,
        *,
        branch_name: str,
        logits: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
    ) -> None:
        """
        用一个批次的预测更新指定分支 PRAUC。

        输入参数:
            - branch_name: str, 分支名; 必须存在于构造时启用的 branches
            - logits: (N, C) 或 (B, C, D, H, W)，当前分支未归一化预测值。
            - target: 与 logits 空间维度对应的整数类别标签。
            - mask: bool，与 target 同形；True 表示该原子或体素参加统计。

        输出:
            - None，原地更新该预测分支的 TorchMetrics 状态。
        """
        spec = self.branch_specs[branch_name]
        if mask.sum() <= 0:
            return
        if logits.ndim == 5:
            # (N_all, C)，把批次和三个体素空间维合并，类别维保持最后。
            logits_flat = logits.permute(0, 2, 3, 4, 1).reshape(-1, int(logits.shape[1]))
        else:
            # (N_all, C)，输入已经按原子或候选实体展平。
            logits_flat = logits.reshape(-1, int(logits.shape[-1]))
        # int64, (N_all,)，与 logits_flat 第 0 维对齐的类别编号。
        target_flat = target.reshape(-1).long()
        # bool, (N_all,)，True 对应参与当前 PRAUC 的原子或体素。
        mask_flat = mask.reshape(-1).bool()
        if spec.num_classes == 2 and logits_flat.shape[1] == 1:   # 二分类
            # (M,)，有效实体的 sigmoid 前景概率。
            preds = torch.sigmoid(logits_flat[:, 0]).detach().float()[mask_flat]
            # int64, (M,)，有效实体的 0/1 标签。
            targets = target_flat[mask_flat]
            self.metrics[f"{branch_name}__binary"].update(preds, targets)  # BinaryAveragePrecision 对象
            return

        # (N_all, C)，多分类 softmax 概率；通道 0 是背景。
        prob = torch.softmax(logits_flat, dim=1).detach().float()
        for class_id in range(1, spec.num_classes):
            # (M,)，有效实体属于当前非背景类别的概率。
            preds = prob[:, class_id][mask_flat]
            # int64, (M,)，当前类别为 1、其余类别为 0 的一对其余标签。
            targets = (target_flat[mask_flat] == class_id).long()
            self.metrics[f"{branch_name}__class_{class_id}"].update(preds, targets)
            if spec.macro_present_classes_only:
                self.metrics[
                    f"{branch_name}__class_{class_id}_positive_count"
                ].update(targets)

    def compute_payload(self) -> dict[str, torch.Tensor]:
        """
        计算当前验证轮次的 PRAUC 日志字典。

        输出:
            - payload: dict[str, torch.Tensor]，Lightning/W&B 日志键到 PRAUC 标量的映射。
        """
        self._configure_distributed_process_groups()
        payload: dict[str, torch.Tensor] = {}
        for spec in self.branch_specs.values():
            # 二分类分支只发布一个前景 PRAUC。
            if spec.num_classes == 2:
                key = build_metric_key(
                    panel="val_score",
                    metric=f"{spec.name}_PRAUC",
                    num_classes=spec.num_classes,
                    scope="global",
                    task_class_name=None,
                )
                payload[key] = self.metrics[f"{spec.name}__binary"].compute()
                continue
            # 多分类分支跳过背景通道，逐正类计算一对其余 PRAUC。
            class_values: list[torch.Tensor] = []
            for class_id in range(1, spec.num_classes):
                class_name = spec.class_names[class_id]
                class_is_present = True
                if spec.macro_present_classes_only:
                    positive_count = self.metrics[
                        f"{spec.name}__class_{class_id}_positive_count"
                    ].compute()
                    class_is_present = bool(positive_count.item() > 0)
                if not class_is_present:
                    continue
                value = self.metrics[f"{spec.name}__class_{class_id}"].compute()
                class_values.append(value)
                key = build_metric_key(
                    panel="val_score",
                    metric=f"{spec.name}_PRAUC",
                    num_classes=spec.num_classes,
                    scope="global",
                    task_class_name=class_name,
                )
                if spec.report_per_class:
                    payload[key] = value
            if class_values:
                payload[f"val_score/global/{spec.name}_PRAUC_macro"] = torch.stack(class_values).mean()
        return payload

    def reset(self) -> None:
        """
        重置所有 TorchMetrics 状态。

        输出:
            - None, 原地清空 metric state
        """
        for metric in self.metrics.values():
            metric.reset()

    def state_dict(
        self,
        *args: object,
        destination: dict[str, torch.Tensor] | None = None,
        prefix: str = "",
        keep_vars: bool = False,
    ) -> dict[str, torch.Tensor]:
        """
        返回空 state_dict, 避免 validation metric 这个大变量写进 checkpoint: return {} if destination is None else destination

        输入参数:
            - args: object, nn.Module.state_dict 兼容位置参数
            - destination: dict[str, torch.Tensor] | None, PyTorch 兼容参数
            - prefix: str, PyTorch 兼容参数
            - keep_vars: bool, PyTorch 兼容参数

        输出:
            - state: dict[str, torch.Tensor], 空字典
        """
        del args, prefix, keep_vars
        return {} if destination is None else destination
