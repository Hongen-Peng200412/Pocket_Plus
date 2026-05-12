import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import random
import einops


# 第一部分：focal_loss



"""
1. Focal Loss 中 $\alpha$ 随不平衡程度的变化规律
核心结论：$\alpha$ 的最优值并不随不平衡比例线性增加，[0.75, 0.25] 确实在极大范围内（1:100 到 1:100000）都接近最优。

在 RetinaNet 论文及其后续的消融实验中，观察到了以下规律：

$\gamma$ 与 $\alpha$ 的负相关性：这是最重要的发现。$\gamma$ 的作用是“自动处理不平衡（难度轴）”，$\alpha$ 的作用是“手动处理不平衡（频率轴）”。
当 $\gamma = 0$（普通交叉熵）时，最优 $\alpha$ 往往接近逆频率（如 1:1000 时，正类 $\alpha \approx 0.999$）。
当 $\gamma = 2$ 时，由于大量的简单负样本损失被压低了成千上万倍，正样本的相对贡献已经极大。此时，如果再给正样本 $\alpha=0.999$ 的权重，模型会完全无视负样本，导致 FP（假阳性）爆表。
为何 0.25 是“魔数”？：实验证明，当 $\gamma=2$ 时，设置 $\alpha=0.25$（正类）可以将重心稳定在“困难样本”上。
1:100: $\alpha=0.25$ 最优。
1:1000: $\alpha \approx 0.25 \sim 0.35$ 依然最优。
1:10000: 随着不平衡加剧，你可能需要稍微增加正类 $\alpha$（例如到 $0.4$），但绝对不是增加到 $0.99$。
总结建议：

不要使用逆频率（即 use_adaptive_alpha=True 但不加缩放）。它在 $\gamma > 0$ 时通常效果很差。
默认先用 $[0.75, 0.25]$。如果发现模型完全不关注正样本（Recall=0），则微调 $[0.75, 0.25]$ 到 $[0.6, 0.4]$，通常就能解决。
"""
class BinaryFocalLossWithAlpha(nn.Module):
    def __init__(self, from_logits=True, gamma=2.0, ignore_index=None, eps=1e-6, flatten_count=1.0
                 , alpha_tune=[0.75, 0.25], use_adaptive_alpha=False, scale=1.0):
        """
        Binary (single-channel logits) Focal Loss with adaptive alpha tuning.

        Args:
        - from_logits: if True, assume input is raw logits (NOT passed through sigmoid)
        - gamma: focusing parameter (>=0). Typical values: 1.0-3.0.
        - ignore_index: optional label value to ignore in target
        - eps: numerical epsilon to clamp probabilities/logs
        - alpha_tune:
            * None: no multiplicative tuning
            * list/tuple of length 2: interpreted as multiplicative scaling factors [scale_neg, scale_pos]
            These will multiply the base which is computed alpha per class ([0.5, 0.5] or inverse-frequency).
        Behavior:
        - Expects `logits` shape (N, 1, D, H, W) (or (N,1,...) generally).
        - Expects `target` shape (N, D, H, W) with integer class labels 0 or 1.

        Forward Args:
        - logits: Tensor, shape (N,1,...) single-channel raw logits (NOT passed through sigmoid) , or , hard(real) probabilities (if from_logits=False)
        - target: Tensor, shape (N,...) with values {0,1}, or, possibly ignore_index
        - see_channels (bool): whether to use the loss of each channel or not, the outcome is averaged across batch and position(every pixel):
            channel_loss = loss[target==c].type_as(loss).sum() / ((target==c).type_as(loss).sum() + 0.1)
        - reduction: 'mean' | 'sum' | 'none'

        """
        super().__init__()
        self.from_logits = from_logits
        self.gamma = float(gamma)
        self.ignore_index = ignore_index
        self.eps = float(eps)
        self.flatten_count = flatten_count
        # store alpha_tune raw; we'll interpret it in forward
        self.alpha_tune = alpha_tune
        self.use_adaptive_alpha = use_adaptive_alpha
        self.scale = scale

    def forward(self, logits, target, reduction='sum', hardmask=None, 
                see_channels=False, num_classes=2):
        if logits.dim() < 2:
            raise ValueError("logits must have shape (N,1,...) for binary segmentation")
        device = logits.device
        if target.dtype != torch.long:
            target_long = target.long()
        else:
            target_long = target
        target_long = target_long.to(device)
        # Build valid mask (exclude ignore_index)
        if self.ignore_index is not None:
            valid_mask = (target_long != self.ignore_index)
        else:
            valid_mask = torch.ones_like(target_long, dtype=torch.bool, device=device)
        # Compute counts per class (exclude ignored)，ensure only consider valid positions
        valid_target = target_long[valid_mask]   # 按照张量索引valid_mask, 返回一维张量
        # If no valid points, handle gracefully
        if valid_target.numel() == 0:  # numel() 是 PyTorch 张量的一个方法，用于返回张量中元素的总数
            # return zero tensor (same dtype/device)
            if reduction == 'none':
                return torch.zeros_like(logits, dtype=logits.dtype, device=device)
            else:
                return torch.tensor(0., dtype=logits.dtype, device=device)
        count_pos = (valid_target == 1).sum(dtype=torch.float32) + self.flatten_count
        count_neg = (valid_target == 0).sum(dtype=torch.float32) + self.flatten_count
        # Base alpha per class: inverse-frequency (rare class gets larger base weight)
        inv = torch.tensor([1.0 / (count_neg + self.eps), 1.0 / (count_pos + self.eps)], device=device, dtype=torch.float32)
        alpha_base = inv / inv.sum()  if self.use_adaptive_alpha else torch.tensor([0.5, 0.5], device=device, dtype=torch.float32)  # shape (2,) -> torch.tensor([alpha_neg, alpha_pos])


        if self.alpha_tune is None:
            alpha_tensor = alpha_base
        else:
            scales = torch.tensor(self.alpha_tune, device=device, dtype=torch.float32)
            alpha_tensor = alpha_base * scales
        # Ensure alpha_tensor dtype matches logits
        alpha_tensor = alpha_tensor.to(device=device).type_as(logits)
        # logits (N,1,...) ,target_long (N,...)
        if logits.shape[1] == 1:
            target_float = target_long.unsqueeze(1).to(dtype=logits.dtype)  # (N,1,...)
        else:
            raise ValueError("logits channel dim expected to be 1 for binary loss")

        # compute probabilities
        probability = torch.sigmoid(logits) if self.from_logits else logits  # (N,1,...)
        # p_t = prob if target==1 else 1-prob
        p_t = probability * target_float + (1.0 - probability) * (1.0 - target_float) # (N,1,...)
        p_t = p_t.clamp(min=self.eps, max=1.0 - self.eps) # (N,1,...)
        focal_factor = (1.0 - p_t) ** self.gamma  # (N,1,...)
        # per-element cross-entropy (stable): binary cross entropy with logits (equals -log(p_t))
        binary_cross_entropy = F.binary_cross_entropy_with_logits(input=logits, target=target_float, reduction='none') if self.from_logits else F.binary_cross_entropy(input=probability, target=target_float, reduction='none')  # (N,1,...)

        # build alpha_per_element with shape (N,1,...)
        alpha_per_element = (alpha_tensor[1] * target_float  +  alpha_tensor[0] * (1.0 - target_float)).type_as(binary_cross_entropy)
        # final loss (per element)
        loss = alpha_per_element * focal_factor * binary_cross_entropy  # (N,1,...)
        # valid_mask shape (N,...); expand to channel dim
        mask = valid_mask.unsqueeze(1).type_as(loss)  # (N,1,...)
        loss = loss * mask


        loss = loss.squeeze(dim=1)  # (N,...)最终loss
        loss *= self.scale
        effective_valid_mask = valid_mask
        if hardmask is not None:
            # torch.Tensor, `(N, ...)`, 与 loss 对齐后的硬掩码；允许调用方额外保留 1 个通道维
            hardmask_tensor = hardmask.to(device=device)
            if hardmask_tensor.dim() == loss.dim() + 1 and hardmask_tensor.shape[1] == 1:
                hardmask_tensor = hardmask_tensor.squeeze(dim=1)
            if tuple(hardmask_tensor.shape) != tuple(loss.shape):
                raise ValueError(
                    f"hardmask shape={tuple(hardmask_tensor.shape)} must match loss shape={tuple(loss.shape)}"
                )
            # torch.Tensor, `(N, ...)`, 分子仍按 hardmask 数值做逐元素加权
            loss = loss * hardmask_tensor.type_as(loss)
            # torch.Tensor, `(N, ...)`, reduction='mean' 时仅统计 hardmask 非零的位置
            effective_valid_mask = effective_valid_mask & (hardmask_tensor != 0)
        if see_channels is False:
            if reduction == 'mean':
                valid_count = effective_valid_mask.sum()
                if valid_count.item() == 0:
                    return torch.tensor(0., dtype=loss.dtype, device=device)
                return loss.sum() / valid_count.type_as(loss)
            elif reduction == 'sum':
                return loss.sum()
            elif reduction == 'none':
                return loss  # no reduction, shape (N, ...) 
        else:
            channels_loss = []
            for c in range(num_classes):  # 2
                channel_loss = loss[target==c].type_as(loss).sum() / ((target==c).type_as(loss).sum() + 0.1)
                channels_loss.append(channel_loss)
            if reduction == 'mean':
                valid_count = effective_valid_mask.sum()
                if valid_count.item() == 0:
                    return torch.tensor(0., dtype=loss.dtype, device=device), *channels_loss
                return loss.sum() / valid_count.type_as(loss), *channels_loss
            elif reduction == 'sum':
                return loss.sum(), *channels_loss
            elif reduction == 'none':
                return loss  # no reduction, shape (N, ...) 



















# 使用要求：pred格式为 Tensor, shape (N, C, ...), logits (未 softmax)，其中Channels的个数=类别数。
class MultiClassFocalLossWithAlpha(nn.Module):
    def __init__(self, from_logits=True, gamma=2.0, ignore_index=None, eps=1e-6, scale=None, 
                 flatten_count=10.0, alpha_tune=1.0, use_adaptive_alpha=False):
        """
        Focal loss for multi-class spatial/volumetric prediction (自适应 alpha 版本).

        Args:
            - from_logits (bool): which means, pred isn't pass through softmax or pred is hard probabilities(id from_logits=False)
            - gamma (float): focusing parameter (focusing 参数).
            - reduction (str, will be in forward): 'mean', 'sum', or None.
            - ignore_index (int or None): label value to ignore in loss computation.
            - eps (float): small epsilon to avoid divide-by-zero, avoiding divide-by-zero.
            - scale (float or None): scale factor for the loss: 100 mean loss *= 100

            - flatten_count (int): number of pixels to flatten for each class.
            - use_adaptive_alpha (bool): whether to use adaptive alpha or not.
            - alpha_tune (float or list): If use_adaptive_alpha is True, alpha_tune can be a list of values to tune over(after normalization then tune over).If not, alpha_tune will be used straighforwardly as loss weight for each class.

        Forward inputs:
            - pred: Tensor, shape (N, C, ...), logits (未 softmax)，其中Channels的个数=类别数。典型例子为(N, C, D, H, W)
            - target: LongTensor, shape (N, ...), values in {0..C-1} (or ignore_index)
            - num_classes: optional int, defaults to pred.shape[1]
            - hardmask: shold have the same shape as target, and values in {0,1} or None. If not None, the loss will only be computed on the pixels with hardmask=1.
            - reduction (str): 'mean'(over every pixel), 'sum', or None.
            - see_channels (bool): whether to use the loss of each channel or not, the outcome is averaged across batch and position(or, across every pixel):
                channel_loss = loss[target==c].type_as(loss).sum() / ((target==c).type_as(loss).sum() + 0.1)
        """
        super().__init__()
        # 修正1: 确保 alpha_tune 始终是 tensor，并且可以广播到 alpha_tensor
        if isinstance(alpha_tune, (list, tuple)):
            self.alpha_tune = torch.tensor(alpha_tune, dtype=torch.float32)
        else:
            self.alpha_tune = torch.tensor([alpha_tune], dtype=torch.float32) # 如果是单一值，也转为张量
        self.from_logits = from_logits
        self.gamma = float(gamma)
        self.ignore_index = ignore_index
        self.eps = float(eps)
        self.scale = scale
        self.flatten_count = flatten_count
        self.use_adaptive_alpha = use_adaptive_alpha

    def forward(self, pred, target, num_classes=None, hardmask=None, reduction='mean', see_channels=False):
        if pred.dim() < 2:
            raise ValueError("pred must have shape (N, C, ...)")
        device = pred.device
        if target.dtype != torch.long:
            target = target.long()
        # move target to same device
        target = target.to(device)
        # infer num_classes from pred if not given
        if num_classes is None:
            num_classes = pred.shape[1]
        if num_classes != pred.shape[1]:
            raise ValueError(f"num_classes ({num_classes}) must match pred.shape[1] ({pred.shape[1]})")
        # build mask to exclude ignore_index from counts and from loss
        if self.ignore_index is not None:
            valid_mask = (target != self.ignore_index)
        else:
            valid_mask = torch.ones_like(target, dtype=torch.bool, device=device)
        # 在 forward 方法中，将 alpha_tune 转移到与 pred 相同的设备上，以确保计算正常进行。
        self.alpha_tune = self.alpha_tune.to(pred.device).type_as(pred)


        # counts per class (float), excluding ignore_index
        counts = torch.stack([
            (((target == i) & valid_mask).sum(dtype=torch.float))
            for i in range(num_classes)
        ], dim=0)  # torch.stack是为了将列表堆叠为张量，shape (N,)表示每一类别的样本个数
        total = counts.sum()
        # if no valid pixels in batch (because of ignore_index), fallback to uniform counts
        if total.item() == 0:
            counts = torch.ones_like(counts)
            total = counts.sum()
        if self.use_adaptive_alpha:
            # alpha per class (sum to 1)
            alpha_tensor = 1.0 / (counts + self.flatten_count)
            alpha_tensor = alpha_tensor / alpha_tensor.sum()
            alpha_tensor = alpha_tensor * self.alpha_tune.to(alpha_tensor.device)   # shape (C,)
            # ensure same device / dtype as pred for later indexing and multiplication
            alpha_tensor = alpha_tensor.to(device=device).type_as(pred)  # float tensor
        else:
            alpha_tensor = self.alpha_tune.to(pred.device).type_as(pred)  # shape (C,)



        # log softmax over class dim, mathematically equivalent to log(softmax(x))
        log_softmax = F.log_softmax(pred, dim=1) if self.from_logits else torch.log(pred + self.eps)  # shape (N, C, ...)
        index = target.unsqueeze(1)  # 在第一(从0开始)维的位置插入新的维度, 形状变为(N,1,...)
        # torch.gather 会根据 index 张量中的值，在 input=log_softmax 的指定维度上收集相应的元素。收集到的元素会形成一个新的张量，形状为 (N, 1, ...)， 最后squeeze为(N,...)。
        log_probability = torch.gather(input=log_softmax, dim=1, index=index).squeeze(1)
        probability = log_probability.exp() # (N, ...) 无C
        cross_entropy_loss = -log_probability  # per-element cross-entropy
        # alpha per element has the same shape as target, but with values from alpha_tensor
        alpha_per_element = alpha_tensor[target]  # shape (N,...) ; 不同于张量的布尔索引，这叫花式索引, 返回的形状与target相同
        focal_term = (1.0 - probability).clamp(min=self.eps, max=1-self.eps) ** self.gamma    # clamp 确保 focal_term 不会出现负数(数学上确实不出现)，避免数值问题
        loss = alpha_per_element * focal_term * cross_entropy_loss  # per-element loss (N,...)


        # mask out ignored elements
        loss = loss * valid_mask.type_as(loss)
        valid_count = valid_mask.sum()
        if hardmask is not None:
            hardmask = hardmask.to(device=device).type_as(loss)
            loss *= hardmask
        if self.scale is not None:
            loss *= float(self.scale)


        if see_channels is False:
            if reduction == 'mean':
                if valid_count.item() == 0:
                    return torch.tensor(0., dtype=loss.dtype, device=device)
                return loss.sum() / valid_count.type_as(loss)
            elif reduction == 'sum':
                return loss.sum()
            elif reduction == 'none':
                return loss  # no reduction, shape (N, ...) 
        else:
            channels_loss = []
            for c in range(num_classes):
                channel_loss = loss[target==c].type_as(loss).sum() / ((target==c).type_as(loss).sum() + 0.1)
                channels_loss.append(channel_loss)

            if reduction == 'mean':
                if valid_count.item() == 0:
                    return torch.tensor(0., dtype=loss.dtype, device=device), *channels_loss
                return loss.sum() / valid_count.type_as(loss), *channels_loss
            elif reduction == 'sum':
                return loss.sum(), *channels_loss
            elif reduction == 'none':
                return loss  # no reduction, shape (N, ...) 
            






























class BinaryTverskyLoss(nn.Module):
    """
    二值 Tversky Loss，用于单通道二分类分割。

    Tversky 指数: TI = (TP + smooth) / (TP + α·FP + β·FN + smooth)
    Loss = 1 - TI

    当 α=β=0.5 时等价于标准 Dice Loss。
    增大 β 可以更重地惩罚 FN（漏报），适合小目标检测。

    输入参数:
        - alpha: float, FP 惩罚系数, 建议值 0.5
        - beta: float, FN 惩罚系数, 建议值 0.5
        - smooth: float, 拉普拉斯平滑项, 建议值 1.0 (nnU-Net 默认)
        - from_logits: bool, True 表示输入 logits，内部自动 sigmoid

    前向输入:
        - logits: torch.Tensor, (N, 1, ...) 或 (N, 1), 预测值
        - target: torch.Tensor, (N, ...) 或 (N,), 真值 {0, 1}
        - reduction: str, 保留参数（与 BinaryFocalLossWithAlpha 接口一致），当前忽略
        - hardmask: torch.Tensor | None, 与 target 同形状的有效区域掩码

    前向输出:
        - loss: torch.Tensor, 标量
    """

    def __init__(
        self,
        alpha: float,
        beta: float,
        smooth: float,
        from_logits: bool,
    ) -> None:
        super().__init__()
        # float, 标量, FP 惩罚系数
        self.alpha = float(alpha)
        # float, 标量, FN 惩罚系数
        self.beta = float(beta)
        # float, 标量, 拉普拉斯平滑项
        self.smooth = float(smooth)
        # bool, 是否从 logits 计算 sigmoid
        self.from_logits = bool(from_logits)

    def forward(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        reduction: str = "mean",
        hardmask: torch.Tensor | None = None,
        **_kwargs,
    ) -> torch.Tensor:
        """
        前向计算 Tversky Loss。

        输入参数:
            - logits: torch.Tensor, (N, 1, ...) 或 (N, 1)
            - target: torch.Tensor, (N, ...) 或 (N,), 真值 {0, 1}
            - reduction: str, 保留参数，与调用协议对齐，当前不影响计算
            - hardmask: torch.Tensor | None, 有效区域掩码，与 target 同形

        输出:
            - loss: torch.Tensor, 标量
        """
        device = logits.device
        if logits.shape[1] != 1:
            raise ValueError(f"BinaryTverskyLoss 期望单通道 logits，实际 shape={tuple(logits.shape)}")
        # torch.Tensor, (N, ...), 预测概率（去掉 channel 维）
        prob = torch.sigmoid(logits).squeeze(1) if self.from_logits else logits.squeeze(1)
        # torch.Tensor, (N, ...), float32 真值
        target_float = target.to(dtype=prob.dtype, device=device)

        if hardmask is not None:
            # torch.Tensor, (N, ...), float32 掩码；与 prob 对齐形状
            mask_float = hardmask.to(dtype=prob.dtype, device=device)
            if mask_float.ndim == prob.ndim + 1 and mask_float.shape[1] == 1:
                mask_float = mask_float.squeeze(1)
            # hardmask=0 的位置: prob 和 target 均置零，不贡献 TP/FP/FN
            prob = prob * mask_float
            target_float = target_float * mask_float

        # torch.Tensor, 标量, True Positive 之和
        tp = (prob * target_float).sum()
        # torch.Tensor, 标量, False Positive 之和
        fp = (prob * (1.0 - target_float)).sum()
        # torch.Tensor, 标量, False Negative 之和
        fn = ((1.0 - prob) * target_float).sum()

        # torch.Tensor, 标量, Tversky 指数
        tversky_index = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        return 1.0 - tversky_index


# ============================================================
# FocalTverskyCombinedLoss
# Focal Loss + Tversky Loss 的加权和，接口与 BinaryFocalLossWithAlpha 完全一致
# ============================================================
class FocalTverskyCombinedLoss(nn.Module):
    """
    Focal Loss + Tversky Loss 加权求和的联合损失。

    输入参数:
        - focal_weight: float, Focal 损失权重, 建议值 1.0
        - dice_weight: float, Tversky 损失权重, 建议值 1.0
        - tversky_alpha: float, Tversky FP 惩罚系数, 建议值 0.5
        - tversky_beta: float, Tversky FN 惩罚系数, 建议值 0.5
        - tversky_smooth: float, Tversky 平滑项, 建议值 1.0
        - (其余参数透传给 BinaryFocalLossWithAlpha):
            - gamma: float, Focal 聚焦参数
            - ignore_index: int | None
            - eps: float
            - scale: float
            - alpha_tune: list[float] | None
            - use_adaptive_alpha: bool
            - from_logits: bool, 建议值 True
            - flatten_count: float

    前向输入:
        - logits: torch.Tensor, (N, 1, ...)
        - target: torch.Tensor, (N, ...)
        - reduction: str, "mean" 传递给 Focal；Tversky 强制全局 mean
        - hardmask: torch.Tensor | None

    前向输出:
        - loss: torch.Tensor, 标量
    """

    def __init__(
        self,
        focal_weight: float,
        dice_weight: float,
        tversky_alpha: float,
        tversky_beta: float,
        tversky_smooth: float,
        gamma: float,
        ignore_index,
        eps: float,
        scale: float,
        alpha_tune,
        use_adaptive_alpha: bool,
        from_logits: bool = True,
        flatten_count: float = 1.0,
    ) -> None:
        super().__init__()
        # float, 标量, Focal 损失权重
        self.focal_weight = float(focal_weight)
        # float, 标量, Tversky 损失权重
        self.dice_weight = float(dice_weight)

        # BinaryFocalLossWithAlpha, Focal 损失子模块
        self.focal_loss = BinaryFocalLossWithAlpha(
            from_logits=from_logits,
            gamma=gamma,
            ignore_index=ignore_index,
            eps=eps,
            flatten_count=flatten_count,
            alpha_tune=alpha_tune,
            use_adaptive_alpha=use_adaptive_alpha,
            scale=scale,
        )
        # BinaryTverskyLoss, Tversky 损失子模块（α=β=0.5 时等价于 Dice）
        self.tversky_loss = BinaryTverskyLoss(
            alpha=tversky_alpha,
            beta=tversky_beta,
            smooth=tversky_smooth,
            from_logits=from_logits,
        )

    def forward(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        reduction: str = "mean",
        hardmask: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        前向计算 Focal + Tversky 联合损失。

        输入参数:
            - logits: torch.Tensor, (N, 1, ...) 或 (N, 1)
            - target: torch.Tensor, (N, ...) 或 (N,)
            - reduction: str, 传递给 Focal 子模块；Tversky 固定为全局 mean
            - hardmask: torch.Tensor | None

        输出:
            - loss: torch.Tensor, 标量，等于 focal_weight·focal + dice_weight·tversky
        """
        # torch.Tensor, 标量, Focal 损失值
        focal = self.focal_loss(
            logits, target, reduction=reduction, hardmask=hardmask, **kwargs
        )
        # torch.Tensor, 标量, Tversky 损失值（强制全局 mean）
        tversky = self.tversky_loss(
            logits, target, reduction="mean", hardmask=hardmask
        )
        return self.focal_weight * focal + self.dice_weight * tversky


# ============================================================
# UnifiedCompositeLoss
# 干净的 Focal + Tversky + MSE 统一复合损失
# 同时服务 atom、voxel_aux、voxel_ligand 三个分支
# ============================================================
class UnifiedCompositeLoss(nn.Module):
    """
    统一复合损失，同时服务 atom、voxel_aux、voxel_ligand 三个分支。

    总损失 = w_focal * FocalLoss + w_tversky * TverskyLoss + w_mse * MSELoss。
    若启用 voxel_ligand 损失，训练数据需要提供 ligand_dist_map。

    标签构造:
        - hard_label: torch.Tensor, (*,), int64, 硬标签(取值0或1)
            - 若 forward 传入 target: 直接使用 target 作为硬标签
            - 若 target 为 None: 使用 ligand_dist_map < hard_label_threshold 生成硬标签
        - y_soft: torch.Tensor | None, (*,), float32, 距离高斯软标签
            - 仅当 ligand_dist_map 非 None 时构造
            - focal_soft_negative_suppression=false 且 tversky_soft_target=false 且 w_mse=0 时不参与损失

    损失语义:
        - Focal: 始终使用 hard_label；focal_soft_negative_suppression 控制是否用 y_soft 弱化硬负类损失
        - Tversky: tversky_soft_target=true 时使用 y_soft，false 时使用 hard_label
        - MSE: 有 y_soft 时使用 y_soft，否则使用 hard_label；w_mse=0 时关闭

    输入参数:
        - sigma: float, 距离高斯软标签标准差, 仅 ligand_dist_map 非 None 时用于构造 y_soft
        - hard_label_threshold: float | None, 距离阈值; None=硬标签从 target 读取, float=由 ligand_dist_map 生成硬标签
        - focal_gamma: float, focal 聚焦参数, 建议值 2.0
        - focal_alpha_neg: float, 背景(负类) alpha 权重
        - focal_alpha_pos: float, 前景(正类) alpha 权重
        - focal_eps: float, 数值稳定 eps
        - tversky_alpha: float, Tversky FP 惩罚系数; 0.5 与 beta=0.5 时等价 Dice
        - tversky_beta: float, Tversky FN 惩罚系数; 0.5 与 alpha=0.5 时等价 Dice
        - tversky_smooth: float, Tversky 平滑项
        - w_focal: float, focal 损失权重
        - w_tversky: float, tversky 损失权重
        - w_mse: float, MSE 损失权重; 0 表示关闭
        - focal_soft_negative_suppression: bool, 是否用 y_soft 对硬负类 focal 损失做距离抑制
        - tversky_soft_target: bool, 是否让 Tversky 使用 y_soft 作为目标; false 时使用 hard_label

    前向输入:
        - logits: torch.Tensor, (*, 1, ...), 单通道预测 logits
        - target: torch.Tensor | None, (*, ...), 硬标签(取值0或1); None 时须提供 ligand_dist_map 与 hard_label_threshold
        - hardmask: torch.Tensor | None, (*, ...), 几何掩码(0/1)
        - valid_mask: torch.Tensor | None, (*, ...), 有效区域掩码
        - ligand_dist_map: torch.Tensor | None, (*, ...), 距离图; 用于生成 hard_label 和 y_soft

    前向输出:
        - loss: torch.Tensor, 标量, 加权后的复合损失
    """

    def __init__(
        self,
        sigma: float,
        hard_label_threshold,
        focal_gamma: float,
        focal_alpha_neg: float,
        focal_alpha_pos: float,
        focal_eps: float,
        tversky_alpha: float,
        tversky_beta: float,
        tversky_smooth: float,
        w_focal: float,
        w_tversky: float,
        w_mse: float,
        focal_soft_negative_suppression: bool = True,
        tversky_soft_target: bool = True,
    ) -> None:
        super().__init__()
        # float, 标量, 软标签高斯核 σ
        self.sigma = float(sigma)
        # float | None, 硬标签距离阈值
        self.hard_label_threshold = (
            float(hard_label_threshold) if hard_label_threshold is not None else None
        )
        # float, 标量, focal 聚焦参数
        self.focal_gamma = float(focal_gamma)
        # float, 标量, 负类 alpha 权重
        self.focal_alpha_neg = float(focal_alpha_neg)
        # float, 标量, 正类 alpha 权重
        self.focal_alpha_pos = float(focal_alpha_pos)
        # float, 标量, 数值稳定 eps
        self.focal_eps = float(focal_eps)
        # float, 标量, Tversky FP 惩罚系数
        self.tversky_alpha = float(tversky_alpha)
        # float, 标量, Tversky FN 惩罚系数
        self.tversky_beta = float(tversky_beta)
        # float, 标量, Tversky 平滑项
        self.tversky_smooth = float(tversky_smooth)
        # float, 标量, focal 权重
        self.w_focal = float(w_focal)
        # float, 标量, tversky 权重
        self.w_tversky = float(w_tversky)
        # float, MSE 损失权重
        self.w_mse = float(w_mse)
        # bool, 是否用 y_soft 对 hard_label=0 的 focal 损失做距离抑制
        self.focal_soft_negative_suppression = bool(focal_soft_negative_suppression)
        # bool, Tversky 是否使用 y_soft；False 时使用 hard_label
        self.tversky_soft_target = bool(tversky_soft_target)

    @staticmethod
    def _maybe_squeeze_channel(x: torch.Tensor, ref_ndim: int) -> torch.Tensor:
        """
        若 x 比参考维度多 1 且第 1 维大小为 1，则 squeeze(1)。

        输入参数:
            - x: torch.Tensor, 待处理张量
            - ref_ndim: int, 标量, 参考维度数

        输出:
            - x: torch.Tensor, squeeze 后的张量
        """
        if x.ndim == ref_ndim + 1 and x.shape[1] == 1:
            return x.squeeze(1)
        return x

    def forward(
        self,
        logits: torch.Tensor,
        target: torch.Tensor | None = None,
        hardmask: torch.Tensor | None = None,
        valid_mask: torch.Tensor | None = None,
        ligand_dist_map: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        前向计算统一复合损失。

        输入参数:
            - logits: torch.Tensor, (*, 1, ...), 单通道预测 logits
            - target: torch.Tensor | None, (*, ...), 硬标签 (0/1)
            - hardmask: torch.Tensor | None, (*, ...), 几何掩码
            - valid_mask: torch.Tensor | None, (*, ...), 边界有效区域
            - ligand_dist_map: torch.Tensor | None, (*, ...), 距离图

        输出:
            - loss: torch.Tensor, 标量
        """
        device = logits.device
        if logits.ndim < 2 or logits.shape[1] != 1:
            raise ValueError(
                f"UnifiedCompositeLoss 期望 logits 形状为 (*, 1, ...), "
                f"实际 shape={tuple(logits.shape)}"
            )

        # ---- 1. squeeze channel dim ----
        # torch.Tensor, (*,), 去掉 channel 维的 logits
        logits_flat = logits.squeeze(1)
        # torch.Tensor, (*,), 预测概率
        prob = torch.sigmoid(logits_flat)

        # ---- 2. 构造 hard_label 和 y_soft ----
        # torch.Tensor | None, (*,), float32, 软标签 (0~1)
        y_soft = None
        if ligand_dist_map is not None:
            # torch.Tensor, (*,), float32, 距离图 (已 squeeze)
            dist = self._maybe_squeeze_channel(
                ligand_dist_map.to(device=device, dtype=prob.dtype),
                logits_flat.ndim,
            )
            # y_soft = exp(-d^2 / (2σ^2))
            y_soft = torch.exp(-dist.pow(2) / (2.0 * self.sigma ** 2))
            if target is None:
                if self.hard_label_threshold is None:
                    raise ValueError(
                        "target=None 且 hard_label_threshold=None, "
                        "无法生成硬标签。请指定 hard_label_threshold 或传入 target。"
                    )
                # torch.Tensor, (*,), int64, 由距离阈值生成的硬标签
                hard_label = (dist < self.hard_label_threshold).long()
            else:
                hard_label = self._maybe_squeeze_channel(
                    target.to(device=device), logits_flat.ndim
                ).long()
        else:
            if target is None:
                raise ValueError("target 和 ligand_dist_map 不能同时为 None")
            hard_label = self._maybe_squeeze_channel(
                target.to(device=device), logits_flat.ndim
            ).long()

        # torch.Tensor, (*,), float32, 硬标签浮点版
        hard_float = hard_label.to(dtype=prob.dtype)




        # ---- 3. effective_mask ----
        # torch.Tensor, (*,), bool, 有效监督区域
        effective_mask = torch.ones_like(logits_flat, dtype=torch.bool)
        if valid_mask is not None:
            vm = self._maybe_squeeze_channel(
                valid_mask.to(device=device).bool(), logits_flat.ndim
            )
            effective_mask = effective_mask & vm
        if hardmask is not None:
            hm = self._maybe_squeeze_channel(
                hardmask.to(device=device), logits_flat.ndim
            )
            effective_mask = effective_mask & (hm != 0)

        # int, 有效像素数 (至少为 1, 防止除零)
        valid_count = effective_mask.sum().clamp(min=1)
        # 若无任何有效像素, 直接返回 0
        if effective_mask.sum().item() == 0:
            return torch.tensor(0.0, device=device, dtype=prob.dtype)
        # torch.Tensor, (*,), float32, 有效区域浮点掩码
        mask_float = effective_mask.to(dtype=prob.dtype)




        # ---- 4. Focal Loss ----
        # p_t = prob if target==1 else 1-prob
        # torch.Tensor, (*,), float32
        p_t = prob * hard_float + (1.0 - prob) * (1.0 - hard_float)
        p_t = p_t.clamp(min=self.focal_eps, max=1.0 - self.focal_eps)
        # torch.Tensor, (*,), float32, focal 聚焦因子
        focal_factor = (1.0 - p_t) ** self.focal_gamma
        # torch.Tensor, (*,), float32, 逐元素 BCE
        bce = F.binary_cross_entropy_with_logits(
            logits_flat, hard_float, reduction="none"
        )
        # torch.Tensor, (*,), float32, 逐元素 alpha 权重
        alpha_per_elem = (
            self.focal_alpha_pos * hard_float
            + self.focal_alpha_neg * (1.0 - hard_float)
        )
        # torch.Tensor, (*,), float32, 逐元素 focal loss
        focal_per_elem = alpha_per_elem * focal_factor * bce
        if self.focal_soft_negative_suppression and y_soft is not None:
            suppression = torch.where(
                hard_label == 1,
                torch.ones_like(y_soft),
                1.0 - y_soft,
            )
            focal_per_elem = focal_per_elem * suppression
        # torch.Tensor, 标量, masked mean focal loss
        focal_loss = (focal_per_elem * mask_float).sum() / valid_count




        # ---- 5. Tversky Loss ----
        # 在有效区域计算; mask 区域 prob=0, target=0, 不贡献 TP/FP/FN
        # torch.Tensor, (*,), float32, masked 预测概率
        masked_prob = prob * mask_float
        # torch.Tensor, (*,), float32, masked 目标
        tversky_target = y_soft if self.tversky_soft_target and y_soft is not None else hard_float
        
        masked_target = tversky_target * mask_float

        # torch.Tensor, 标量
        tp = (masked_prob * masked_target).sum()
        fp = (masked_prob * (1.0 - masked_target) * mask_float).sum()
        fn = ((1.0 - masked_prob) * masked_target * mask_float).sum()

        # torch.Tensor, 标量, Tversky 指数
        tversky_index = (tp + self.tversky_smooth) / (
            tp + self.tversky_alpha * fp + self.tversky_beta * fn + self.tversky_smooth
        )
        # torch.Tensor, 标量
        tversky_loss = 1.0 - tversky_index



        # ---- 6. MSE Loss ----
        # MSE target: 有 y_soft 时用 y_soft, 否则用 hard_float
        mse_target = y_soft if y_soft is not None else hard_float
        # torch.Tensor, (*,), float32, 逐元素 MSE
        mse_per_elem = (prob - mse_target) ** 2
        # torch.Tensor, 标量, masked mean MSE
        mse_loss = (mse_per_elem * mask_float).sum() / valid_count

        # ---- 7. 加权求和 ----
        return (
            self.w_focal * focal_loss
            + self.w_tversky * tversky_loss
            + self.w_mse * mse_loss
        )
