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
    常规 validation AP(PRAUC的近似)/PRAUC 分支配置。

    输入参数:
        - name: str, 分支名; 使用 atom/atom_front/receptor/voxel_ligand
        - enabled: bool, 是否启用该分支
        - num_classes: int, task 类别数; 2 表示二分类输出无 suffix 指标
        - class_names: tuple[str, ...], (num_classes,), task class 名
        - thresholds: int | Sequence[float] | None, TorchMetrics AP 阈值配置(作为近似计算 AP 的 bin)
        - report_per_class: bool, 是否把逐类别值写入日志
        - macro_present_classes_only: bool, 宏平均是否跳过验证集中没有正样本的类别
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


# 管理 Stage1 常规 validation AP/PRAUC 指标(只是 AP 或 PRAUC 哦)
class ValidationMetricManager(nn.Module):
    """
    管理 Stage1 常规 validation AP/PRAUC 指标。

    输入参数:
        - branches: Sequence[MetricBranchSpec], 可启用的 metric 分支配置列表
        - metric_device_policy: str, 指标状态设备策略; 允许 auto/cpu/gpu

    内部重要方法：
        - _register_branch_metrics: 为一个分支注册 TorchMetrics 对象, 也就是创建 BinaryAveragePrecision(...)
        - update_branch: 用一个 batch 更新指定分支 AP/PRAUC 指标
        - compute_payload: 计算当前 epoch 的常规 validation 的PRAUC指标的 payload(dict[str, torch.Tensor])
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
        for spec in self.branch_specs.values():
            self._register_branch_metrics(spec)

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
        用一个 batch 更新指定分支 AP/PRAUC 指标。

        输入参数:
            - branch_name: str, 分支名; 必须存在于构造时启用的 branches
            - logits: torch.Tensor, (N,C) 或 (B,C,D,H,W), 当前分支 logits
            - target: torch.Tensor, 与 logits 空间维度兼容, hard-label target
            - mask: torch.Tensor, 与 target 同空间维度, True 表示参与统计

        输出:
            - None, 原地更新 TorchMetrics 状态: self.metrics[f"{branch_name}__binary"].update(preds, targets) 
        """
        spec = self.branch_specs[branch_name]
        if mask.sum() <= 0:
            return
        if logits.ndim == 5:
            # torch.Tensor, (N_all,C), 展平空间后的 logits
            logits_flat = logits.permute(0, 2, 3, 4, 1).reshape(-1, int(logits.shape[1]))
        else:
            # torch.Tensor, (N_all,C), 已展平 logits
            logits_flat = logits.reshape(-1, int(logits.shape[-1]))
        # torch.Tensor, (N_all,), 展平 hard-label target
        target_flat = target.reshape(-1).long()
        # torch.Tensor, (N_all,), 展平有效统计掩码
        mask_flat = mask.reshape(-1).bool()
        if spec.num_classes == 2 and logits_flat.shape[1] == 1:   # 二分类
            # torch.Tensor, (M,), 有效位置 sigmoid 前景概率
            preds = torch.sigmoid(logits_flat[:, 0]).detach().float()[mask_flat]
            # torch.Tensor, (M,), 有效位置二分类标签
            targets = target_flat[mask_flat]
            self.metrics[f"{branch_name}__binary"].update(preds, targets)  # BinaryAveragePrecision 对象
            return

        # torch.Tensor, (N_all,C), 多分类 softmax 概率
        prob = torch.softmax(logits_flat, dim=1).detach().float()
        for class_id in range(1, spec.num_classes):
            # torch.Tensor, (M,), 有效位置当前类别概率
            preds = prob[:, class_id][mask_flat]
            # torch.Tensor, (M,), 有效位置当前类别 one-vs-rest 标签
            targets = (target_flat[mask_flat] == class_id).long()
            self.metrics[f"{branch_name}__class_{class_id}"].update(preds, targets)
            if spec.macro_present_classes_only:
                self.metrics[
                    f"{branch_name}__class_{class_id}_positive_count"
                ].update(targets)

    def compute_payload(self) -> dict[str, torch.Tensor]:
        """
        计算当前 epoch 的常规 validation 的PRAUC指标的 payload。

        输出:
            - payload: dict[str, torch.Tensor], Lightning/W&B scalar key 到标量 tensor 的映射
        """
        payload: dict[str, torch.Tensor] = {}
        for spec in self.branch_specs.values():
            # 二分类
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
            # 多分类
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
