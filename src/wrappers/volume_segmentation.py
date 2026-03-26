r""" 半通用包装器: 针对体素级别分割、使用 src\modules\losses.py """
from __future__ import annotations

import inspect
from typing import Any, Dict, Optional, Tuple

import lightning as pl
import torch
from hydra.utils import instantiate
from torch import nn
from src.utils.bias_init import init_classification_head_bias


class VolumeSegmentationWrapper(pl.LightningModule):
    """
    通用体素分割包装器 (Generic wrapper for volumetric segmentation tasks).

    主要功能 (Features):
        - 兼容多种 Batch 格式: (inputs, labels) 或 (inputs, labels, hardmask)
        - 支持通过 Hydra 动态实例化的 backbone 和 loss
        - 提供通用的验证指标 (Accuracy) 用于模型监控 (Checkpoint monitor)
        - 自动适配通道: 当使用 MultiClassFocalLossWithAlpha 且标签为二分类时，自动将 1 通道 logits 桥接为 2 通道
    """

    def __init__(
        self,
        backbone: nn.Module,
        loss: nn.Module,
        optimizer: Optional[Dict[str, Any]] = None,  # 接受 cfg.train 的配置, 存入 self.hparams，留给 configure_optimizers
        scheduler: Optional[Dict[str, Any]] = None,  # 接受 cfg.train 的配置, 存入 self.hparams，留给 configure_optimizers

        monitor_metric: str = "val/score",           # 接受 cfg.model 的配置, 存入 self.hparams，留给 configure_optimizers
        interval: str = "epoch",                     # 接受 cfg.model 的配置, 存入 self.hparams，留给 configure_optimizers
        frequency: int = 1,                          # 接受 cfg.model 的配置, 存入 self.hparams，留给 configure_optimizers
        compile: bool = False,

        class_priors: Optional[Any] = None,     # 类别先验概率
        bias_init_layer: Optional[str] = None,  # 偏置初始化层(可为None, 自动检测最后一个Conv/Linear层)
        **kwargs: Any,
    ):
        super().__init__()
        # self.save_hyperparameters(): 会自动扫描 __init__ 函数的参数列表，并将传入的值打包成一个字典，存储在模型的 self.hparams 属性中
        self.save_hyperparameters(ignore=["backbone", "loss"])
        self.backbone = backbone if isinstance(backbone, nn.Module) else instantiate(backbone)
        self.loss = loss if isinstance(loss, nn.Module) else instantiate(loss)
        
        # 从 backbone 自动获取类别数量用于初始化评测指标
        num_classes = getattr(self.backbone, "out_channels", 1)
        from torchmetrics.classification import BinaryAveragePrecision, MulticlassAveragePrecision
        # ⚡ 关键优化：compute_on_cpu=True
        # 使 torchmetrics 将 preds/targets 的内部 state 全部存储在 CPU 上，而非 GPU 上, 这样整个 epoch 累积的体素级预测不会占用 GPU 显存
        if num_classes == 1:
            self.val_pr_auc = BinaryAveragePrecision(compute_on_cpu=True)
        else:
            self.val_pr_auc = MulticlassAveragePrecision(num_classes=num_classes, average="macro", compute_on_cpu=True)
            
        # -------------- 偏置初始化 (Bias Initialization) --------------
        # class_priors:
        #   - float:  二分类正类先验概率 π (如 0.0001 表示前景占万分之一)
        #   - list:   多分类各类样本比例 (如 [1, 10, 100, 111000] 表示 A:B:C:BG = 1:10:100:111000)
        #   - None:   不进行偏置初始化（默认）
        # bias_init_layer: 指定分类头层名称，None 则自动检测最后一个 Conv/Linear 层
        if class_priors is not None:
            layer_name = init_classification_head_bias(
                self.backbone, class_priors, layer_name=bias_init_layer
            )
            print(f"[VolumeSegWrapper] Bias initialized on layer '{layer_name}'")

        if compile:
            self.backbone = torch.compile(self.backbone)
    

    # 用于forward检查输入通道数
    def _expected_input_channels(self) -> Optional[int]:
        """
        探测 backbone 期望的输入通道数 (通过查找第一个 Conv3d 层)
        """
        # 优先从自定义属性获取
        if hasattr(self.backbone, "in_channels"):
            return self.backbone.in_channels
            
        for module in self.backbone.modules():
            if isinstance(module, nn.Conv3d):
                return int(module.in_channels)
        return None


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        """
        # 自动识别通道逻辑
        expected = self._expected_input_channels()
        if expected is None:
            # 如果 backbone 声明为待自动识别 (None)
            if hasattr(self.backbone, "set_input_channels"):
                self.backbone.set_input_channels(x.shape[1])
        elif x.shape[1] != expected:
            # 如果不匹配，抛出明确错误
            raise ValueError(
                f"Backbone expects {expected} input channels, but got {x.shape[1]}."
            )
        return self.backbone(x)


    @staticmethod   # 用于step
    def _extract_batch(batch: Any) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        从不同格式的 batch 中提取数据, 根据长度算作 (x, y) 或 (x, y, hardmask)
        """
        if isinstance(batch, (list, tuple)):
            if len(batch) == 2:
                return batch[0], batch[1], None
            if len(batch) >= 3:
                return batch[0], batch[1], batch[2]
        raise TypeError("Expected batch as (inputs, labels) or (inputs, labels, hardmask).")



    def _compute_loss(
        self, logits: torch.Tensor, target: torch.Tensor, hardmask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        计算损失函数，支持动态参数传递
        """
        # 校验：1-channel logits + MultiClassFocal + 二分类 target 直接抛错
        loss_name = self.loss.__class__.__name__
        if "MultiClassFocalLossWithAlpha" in loss_name and logits.shape[1] == 1:
            # 检查 target 是否为二分类
            ignore_index = getattr(self.loss, "ignore_index", None)
            valid_target = target[target != ignore_index] if ignore_index is not None else target
            if valid_target.numel() > 0 and valid_target.max() < 2:
                raise ValueError(
                    "Error: detected 1-channel output with MultiClassFocalLoss for binary target. "
                    "Please use BinaryFocalLossWithAlpha instead in your configuration."
                )

        signature = inspect.signature(self.loss.forward)
        kwargs: Dict[str, Any] = {}
        # 如果损失函数支持 reduction 或 hardmask 参数，则显式传递
        if "reduction" in signature.parameters:    # see me:自动选择 reduction="mean"
            kwargs["reduction"] = "mean"
        if hardmask is not None and "hardmask" in signature.parameters:
            kwargs["hardmask"] = hardmask

        out = self.loss(logits, target, **kwargs)
        # 支持返回 tuple (loss, extra_info) 的损失函数
        return out[0] if isinstance(out, tuple) else out



    # 4. -------------- 训练与验证循环 (Train & Val Steps) --------------
    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        x, y, hardmask = self._extract_batch(batch)
        logits = self(x)
        loss = self._compute_loss(logits, y, hardmask)
        
        # 记录日志，sync_dist=True 保证 DDP 模式下跨卡同步
        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
        return loss

    def validation_step(self, batch: Any, batch_idx: int):
        x, y, hardmask = self._extract_batch(batch)
        logits = self(x)
        loss = self._compute_loss(logits, y, hardmask)

        # 移除 ignore_index 控制的无效像素进行评估计算
        ignore_index = getattr(self.loss, "ignore_index", None)
        if ignore_index is None:
            valid = torch.ones_like(y, dtype=torch.bool)
        else:
            valid = y != ignore_index

        # 同样用 hardmask 过滤：只评估"有原子的体素"，与 loss 的计算范围保持语义一致
        # hardmask: torch, (N, D, H, W), int64, 取值 0 或 1; 1 表示该体素有原子
        if hardmask is not None:
            valid = valid & hardmask.bool()      # see me: 构造valid排除hardmask为0的部分, 用来索引pred与target, 只算有原子的体素的PR-AUC

        if valid.sum() > 0:
            if logits.shape[1] == 1:
                # preds: torch, (N,), 有效体素处的正类预测概率，先 detach 脱离计算图、转 float32，再移到 CPU
                preds = torch.sigmoid(logits[:, 0]).detach().float()[valid].cpu()
            else:
                # preds: torch, (N, C), 有效体素处各类 logits，同样 detach + float32 + CPU
                preds = logits.movedim(1, -1).detach().float()[valid].cpu()
            # targets: torch, (N,), 有效体素处的真实标签（long），移到 CPU
            targets = y[valid].long().cpu()
            # 使用 update 保存 state 以进行真正的、整个 epoch 级别的聚合计算, preds/targets 已在 CPU 上，update 写入 CPU state，不占用 GPU 显存
            self.val_pr_auc.update(preds, targets)
        # 记录验证损失（跨卡同步均值）
        self.log("val/loss", loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        return loss



    def on_validation_epoch_end(self):
        """
        在整个验证 epoch 结束后手动计算并记录 PR-AUC。
        
        ⚡ 为何不在 validation_step 中直接 self.log(metric_object)：
          - 若直接传 metric object 给 log，Lightning 会在 epoch 结束时调用 torchmetrics 的 _sync_dist，将整个 epoch 的体素级 state 通过 all_gather 同步到所有 GPU，需要分配 world_size 倍的内存，极易 OOM。

        """
        # 临时禁用 torchmetrics 内部海量体素的分布式 all_gather 同步
        _prev_to_sync = getattr(self.val_pr_auc, "_to_sync", True)
        self.val_pr_auc._to_sync = False          # 跳过 torchmetrics 内部的分布式同步
        
        # 1. 各卡独立计算当前卡所在验证子集的 PR-AUC
        # score_local: CPU 上的标量 tensor
        score_local = self.val_pr_auc.compute()
        self.val_pr_auc._to_sync = _prev_to_sync  # 恢复原始同步设置
        # 2. 将最终的标量移动到 GPU 上，这样就能支持 Lightning 基于 NCCL 的通信同步了
        score_gpu = score_local.to(self.device)
        # 重置 state，为下一个 epoch 的累积做准备
        self.val_pr_auc.reset()
        # 3. 让 Lightning 在这最后一步把所有卡上的标量做平均。这极大地利用了所有的卡，同时也避免了 OOM。
        # 注意: 局部算 PR-AUC 然后求平均，近似于"全局精确 PR-AUC"
        self.log(self.hparams.monitor_metric, score_gpu, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)



    # 5. -------------- 优化器配置 (Optimizers & Schedulers) --------------
    # 它将由 pl.Trainer 内部自动触发: 在 train.py 中执行 trainer.fit(model, datamodule) 时，Lightning 会在开始训练前（训练循环的第一步之前）自动调用这个方法来获取优化器和调度器
    def configure_optimizers(self):  # TODO: 下一步工作是自己构造相应的优化器/学习率调度策略, 它们将参照Train1(2;3).py, 并放在 src\utils\configure_optimizers.py 以供调用.
        """
        配置优化器与学习率调整策略
        """
        # (1). 优化器
        opt_cfg = self.hparams.optimizer
        if opt_cfg is None:   # 默认优化器
            optimizer = torch.optim.AdamW(
                params=filter(lambda p: p.requires_grad, self.parameters()),
                lr=1e-4,
                weight_decay=1e-2,
            )
        else:
            # 处理 Hydra _partial_: true 产生的 functools.partial, 以及普通 DictConfig
            import functools
            trainable_params = filter(lambda p: p.requires_grad, self.parameters())  # 过滤出可训练的参数
            if isinstance(opt_cfg, functools.partial):  # _partial_: true 已将配置解析为 functools.partial, 直接调用
                optimizer = opt_cfg(params=trainable_params)
            elif callable(opt_cfg) and not hasattr(opt_cfg, "keys"):
                optimizer = opt_cfg(params=trainable_params)
            else:
                # 标准 DictConfig, 通过 Hydra 实例化
                optimizer = instantiate(opt_cfg, params=trainable_params)
                if not isinstance(optimizer, torch.optim.Optimizer):
                    raise TypeError("Failed to instantiate optimizer.")


        # (2). 学习率调度
        sched_cfg = self.hparams.scheduler
        if sched_cfg is None:
            return {"optimizer": optimizer}
        # 处理 Hydra _partial_: true 产生的 functools.partial, 以及普通 DictConfig
        import functools
        if isinstance(sched_cfg, functools.partial):   # _partial_: true 已将配置解析为 functools.partial, 直接调用
            scheduler = sched_cfg(optimizer=optimizer)
        elif callable(sched_cfg) and not hasattr(sched_cfg, "keys"):    # 其他可调用对象 (如 lambda)
            scheduler = sched_cfg(optimizer=optimizer)
        else:
            # 标准 DictConfig, 通过 Hydra 实例化
            scheduler = instantiate(sched_cfg, optimizer=optimizer)
            if hasattr(scheduler, "__call__") and not hasattr(scheduler, "step"):
                scheduler = scheduler()


        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": self.hparams.monitor_metric,
                "interval": self.hparams.interval,
                "frequency": self.hparams.frequency,
            },
        }
