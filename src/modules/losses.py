import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import random
import einops


# 第一部分：加了softmask, hardmask的二分类和多分类focal_loss
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def get_WeightMask_for_numpy(label, kernel, big_is_big=True):
    """ 一般用下一个函数，这个效率低
    输入的label和mask要求是tensor, kernel形状要求(最好)全是奇数。
    运算过程为：构造一个初始全零的初始mask, 它与label相比适当延长。对label的每个取值为1的位置"套上"kernel，如果和之前的值产生冲突，那么在这一步取最大的值，表示“取最大的容忍度”。
        big_is_gig=True时(返回的可以直接作为softmask，它的单调性与hardmask一致)，最后返回的mask=1-mask。最后的的mask越小，算focal_loss损失时, 损失越小 """
    # i维最低有效坐标 = i_0 = (kernel.shape[0]-1)//2   (kernel.shape[0] = 2 * i_0 + 1)
    # i维最高有效坐标 = 最高坐标-多余数i_0 = (label.shape[0]+kernel.shape[0]-1-1) - i_0 = label.shape[0]+2 * i_0 -1 -i_0 = label.shape[0] + i_0 - 1，检验成功！
    mask = np.zeros( (label.shape[0]+kernel.shape[0]-1, label.shape[1]+kernel.shape[1]-1, label.shape[2]+kernel.shape[2]-1) )   # 顺序延长，便于检测
    i_0, j_0, k_0 = (kernel.shape[0]-1)//2, (kernel.shape[1]-1)//2, (kernel.shape[2]-1)//2
    for i in range( i_0 , label.shape[0]+i_0 ):
        for j in range( j_0 , label.shape[1]+j_0 ):
            for k in range( k_0 , label.shape[2]+k_0 ):
                # 注意坐标映射关系：现在的坐标多了 i_0
                if label[i-i_0, j-j_0, k-k_0] == 1:
                    will_change = np.maximum(kernel, mask[i:i+2*i_0+1, j:j+2*j_0+1, k:k+2*k_0+1])
                    mask[i-i_0:i+i_0+1, j-j_0:j+j_0+1, k-k_0:k+k_0+1] = will_change
    if big_is_big == True:
        mask = 1 - mask[i_0:label.shape[0]+i_0, j_0:label.shape[1]+j_0, k_0:label.shape[2]+k_0]
    elif big_is_big == False:
        mask = mask[i_0:label.shape[0]+i_0, j_0:label.shape[1]+j_0, k_0:label.shape[2]+k_0]
    return mask






def get_WeightMask(label, kernel, big_is_big=True):
    """
    label: Tensor, shape (D,H,W) binary (0/1)
    kernel: Tensor, shape (kd,kh,kw), 形状均为奇数
        运算过程为：构造一个初始全零的初始mask, 它与label相比适当延长。对label的每个取值为1的位置，把它的中心"套上"kernel，如果和之前的值产生冲突，那么在这一步总是取最大的值，表示“取最大的容忍度”。
        big_is_big=True时(返回的可以直接作为softmask，它的单调性与hardmask一致)，最后返回的mask=1-mask。最后的的mask越小，算focal_loss损失时, 损失越小
    """
    device = label.device
    dtype = kernel.dtype
    kd, kh, kw = kernel.shape   # kernel_depth, kernel_height, kernel_width
    D, H, W = label.shape   # Depth, High, Width

    mask = torch.zeros(D + kd - 1,
                       H + kh - 1,
                       W + kw - 1,
                       device=device, dtype=dtype)   # 意欲在边界处各自延长 (kd-1)//2, (kh-1)//2, (kw-1)//2 个单位
    kernel = kernel.to(device).type_as(mask)

    # 找出 label 中为 1 的位置（只遍历这些位置）
    coords = torch.nonzero(label == 1, as_tuple=False)  # shape (N,3)
    # 如果没有任何 1，直接裁剪返回（避免进入循环）
    if coords.numel() == 0:
        i0, j0, k0 = (kd - 1) // 2, (kh - 1) // 2, (kw - 1) // 2
        cropped = mask[i0:i0 + D, j0:j0 + H, k0:k0 + W] 
        return 1.0 - cropped if big_is_big else cropped

    # 遍历非零坐标并更新对应块; Python int 以便切片
    for p in coords:
        a, b, c = int(p[0].item()), int(p[1].item()), int(p[2].item())
        # max 逻辑, 最大容忍度，后果是尽可能让loss变小
        mask[a:a + kd, b:b + kh, c:c + kw] = torch.maximum(       # mask的“掩码中心”是 a+(kd-1)//2, b+(kh-1)//2, c+(kw-1)//2, 相当于原本grid的 a b c,正确
            mask[a:a + kd, b:b + kh, c:c + kw],
            kernel
        )

    # 裁剪中心区域（与原实现一致）
    i0, j0, k0 = (kd - 1) // 2, (kh - 1) // 2, (kw - 1) // 2
    cropped = mask[i0:i0 + D, j0:j0 + H, k0:k0 + W]
    return 1.0 - cropped if big_is_big else cropped



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




class BinaryFocalLossWithAlpha_and_SoftMask_HardMask(BinaryFocalLossWithAlpha):
    """
    - pred: Tensor, shape (N, C, ...), logits (未 softmax)
    - target: LongTensor, shape (N, ...), values in {0..C-1} (or ignore_index)
    - num_classes: optional int, defaults to pred.shape[1]
    - softmask: Tensor, shape (N, ...) as the same as pred, 但在函数中会unsqueeze为 (N,C,...)。注意，这个权重是"软权重", 即如果一个体素处的软权重是 alpha(强制要求属于[0,1]), 且“这个体素处label为1的情况下的损失”比“原始损失”要小（不然则等于原始损失），那么它经过softmask后损失为：
        alpha * 这个体素处label为0的情况下的损失 + (1-alpha) *  这个体素处label为1的情况下的损失。softmask生成的优先级为：手动强制输入、label+kernel调用get_WeightMask生成、默认值1.0(big_is_big)。但是一般都是输入kernel来生成
    - hardmask: Tensor, shape (N, ...) as the same as pred。注意，这个权重是"硬权重", 即如果一个体素处的硬权重是 alpha, 那么它的损失为：alpha * mask在这个体素处的值

    - see_channels (bool): whether to use the loss of each channel or not, the outcome is averaged across batch and position:
        channel_loss = loss[target==c].type_as(loss).sum() / ((target==c).type_as(loss).sum() + 0.1)
    """
    def forward(self, pred, target, softmask=None, kernel=None, hardmask=None, reduction='sum', see_channels=False, num_classes=2):
        use_origin_softmask = True if softmask is not None else False
        device = pred.device
        target = target.long().to(device)  # target must be long and binary for this subclass
        # default softmask: ones with same shape as target (no channel)
        if softmask is None and kernel is None:
            softmask = torch.ones_like(target, dtype=pred.dtype, device=device).unsqueeze(dim=1)  # (N, 1, D, H, W)
        elif softmask is None and kernel is not None:
            # apply get_WeightMask per sample; kernel must be 3D and same for all batches
            batch_masks = []
            for idx_batch in range(target.shape[0]):
                piece_mask = get_WeightMask(label=target[idx_batch], kernel=kernel, big_is_big=True).to(device)
                batch_masks.append(piece_mask)
            softmask = torch.stack(batch_masks, dim=0).unsqueeze(dim=1).type_as(pred)
        else:
            softmask = softmask.to(device).unsqueeze(dim=1).type_as(pred)

        # create dilated(膨胀) binary label: more_1_label (long)
        if use_origin_softmask:
            more_1_label = (softmask < 0.99).long().squeeze(dim=1)  # (N, D, H, W), long
        elif kernel is not None:
            batch_more1 = []
            for idx_batch in range(target.shape[0]):
                piece_dilate = get_WeightMask(label=target[idx_batch], kernel=torch.ones_like(kernel), big_is_big=False)
                # dil might be float in [0,1]; threshold to binary
                batch_more1.append((piece_dilate > 0.99).long())
            more_1_label = torch.stack(batch_more1, dim=0)  # (N, D, H, W), long
        else:
            more_1_label = target.clone()  # (N, D, H, W), long

        # get losses (none reduction) from parent
        loss = super().forward(logits=pred, target=target, reduction='none')  # (N, 1, D, H, W)
        loss_if_1 = super().forward(logits=pred, target=more_1_label, reduction='none')
        switch_0_to_1_identifier = (loss_if_1 < loss).bool()  # (N, 1, D, H, W);  switch_0_to_1_identifier 出现1等价于“要换”
        blended = loss * softmask + loss_if_1 * (1-softmask)
        loss = torch.where(condition=switch_0_to_1_identifier, input=blended, other=loss)


        loss = loss.squeeze(dim=1)  # (N, D, H, W),最终loss
        if self.ignore_index is not None:
            valid_mask = (target != self.ignore_index)
        else:
            valid_mask = torch.ones_like(target, dtype=torch.bool, device=device)
        loss = loss * valid_mask.unsqueeze(1).type_as(loss)  # (N, 1, D, H, W)
        if hardmask is not None:
            hardmask = hardmask.to(device).type_as(loss)
            hardmask = hardmask.unsqueeze(1) if hardmask.dim() == loss.dim() - 1 else hardmask
            loss = loss * hardmask
        valid_count = valid_mask.sum()

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







def binary_focal_loss_with_masks(pred, target, gamma=2.0, ignore_index=None, eps=1e-6, flatten_count=1.0, 
                                from_logits=True, alpha_tune=None, softmask=None, kernel=None, hardmask=None, reduction='sum'):
    """
    Binary Focal Loss with adaptive alpha tuning, softmask, and hardmask.
    
    Parameters:
        - from_logits: if True, assume input is raw logits (NOT passed through sigmoid)
        - pred: Tensor, shape (N,1,...), single-channel raw logits (NOT passed through sigmoid)
        - target: Tensor, shape (N,...), with values {0,1} or possibly ignore_index
        - gamma: float, focusing parameter (>=0). Typical values: 2.0, 1.0-3.0
        - ignore_index: int, optional label value to ignore in target
        - eps: float, numerical epsilon to clamp probabilities/logs
        - flatten_count: float, smoothing factor for class counts
        - alpha_tune: None, float in (0,1), or list/tuple of length 2
            - None: alpha derived from inverse class frequency
            - float: interpreted as alpha_pos, alpha_neg = 1-alpha_pos
            - list/tuple: [scale_neg, scale_pos] to multiply base alpha
        - softmask: Tensor, shape (N,...), soft weights in [0,1]. If None and kernel provided, generated via get_WeightMask
        - kernel: Tensor, shape (kd,kh,kw), used to generate softmask and dilated labels
        - hardmask: Tensor, shape (N,...) or (N,1,...), multiplicative hard weights
        - reduction: str, 'mean' | 'sum' | 'none'
    
    Returns:
        - loss: Tensor (N,1,...) (none) / scalar (mean/sum)
    """
    if pred.dim() < 2 or pred.shape[1] != 1:
        raise ValueError("pred must have shape (N,1,...) for binary segmentation")
    
    device = pred.device
    target = target.long().to(device)
    if ignore_index is not None:
        valid_mask = (target != ignore_index)
    else:
        valid_mask = torch.ones_like(target, dtype=torch.bool, device=device)
    # Compute class counts
    valid_target = target[valid_mask]  # 按照张量索引valid_mask, 返回一维张量
    if valid_target.numel() == 0:
        if reduction == 'none':
            return torch.zeros_like(pred, dtype=pred.dtype, device=device)
        return torch.tensor(0., dtype=pred.dtype, device=device)
    count_pos = (valid_target == 1).sum(dtype=torch.float32) + flatten_count
    count_neg = (valid_target == 0).sum(dtype=torch.float32) + flatten_count
    
    # Base alpha: inverse-frequency
    inv = torch.tensor([1.0 / (count_neg + eps), 1.0 / (count_pos + eps)], 
                      device=device, dtype=torch.float32)
    alpha_base = inv / inv.sum()
    # Adjust alpha based on alpha_tune
    if alpha_tune is None:
        alpha_tensor = alpha_base
    elif isinstance(alpha_tune, (float, int)):
        a_pos = max(min(float(alpha_tune), 1.0), 0.0)
        alpha_tensor = torch.tensor([1.0 - a_pos, a_pos], device=device, dtype=torch.float32)
    elif hasattr(alpha_tune, "__len__") and len(alpha_tune) == 2:
        scales = torch.tensor(alpha_tune, device=device, dtype=torch.float32)
        alpha_tensor = alpha_base * scales
        alpha_tensor = alpha_tensor / alpha_tensor.sum()
    else:
        alpha_tensor = alpha_base
    alpha_tensor = alpha_tensor.type_as(pred)
    

    # Prepare target for BCE
    target_float = target.unsqueeze(1).to(dtype=pred.dtype)   # N,1,D,H,W
    # Compute focal loss components
    probability = torch.sigmoid(pred) if from_logits else pred  # N,1,D,H,W
    p_t = probability * target_float + (1.0 - probability) * (1.0 - target_float)
    p_t = p_t.clamp(min=eps, max=1.0 - eps)
    focal_factor = (1.0 - p_t) ** gamma
    binary_cross_entropy = F.binary_cross_entropy_with_logits(input=pred, target=target_float, reduction='none') if from_logits else F.binary_cross_entropy(input=probability, target=target_float, reduction='none')  # N,1,D,H,W
    # Alpha per element
    alpha_per_element = (alpha_tensor[1] * target_float + alpha_tensor[0] * (1.0 - target_float))
    # Base loss
    loss = alpha_per_element * focal_factor * binary_cross_entropy
    loss = loss * valid_mask.unsqueeze(1).type_as(loss)
    

    use_origin_softmask = True if softmask is not None else False
    # Softmask processing
    if softmask is None and kernel is None:
        softmask = torch.ones_like(target, dtype=pred.dtype, device=device).unsqueeze(1) # N C...
    elif softmask is None and kernel is not None:
        batch_masks = []
        for idx_batch in range(target.shape[0]):
            piece_mask = get_WeightMask(label=target[idx_batch], kernel=kernel, big_is_big=True).to(device)
            batch_masks.append(piece_mask)
        softmask = torch.stack(batch_masks, dim=0).unsqueeze(1).type_as(pred)
    else:
        softmask = softmask.to(device).unsqueeze(1).type_as(pred)
    

    # Dilated labels
    if use_origin_softmask:
        more_1_label = (softmask < 0.99).long().squeeze(dim=1)  # N,D,H,W, long
    elif kernel is not None:
        batch_more1 = []
        for idx_batch in range(target.shape[0]):
            piece_dilate = get_WeightMask(label=target[idx_batch], kernel=torch.ones_like(kernel), big_is_big=False)
            batch_more1.append((piece_dilate > 0.99).long())
        more_1_label = torch.stack(batch_more1, dim=0)   # N,D,H,W, long
    else:
        more_1_label = target.clone()

    # Compute loss_if_1 (loss if target were 1)
    target_float_if_1 = more_1_label.unsqueeze(1).to(dtype=pred.dtype)   # N,1,D,H,W, float
    probability_if_1 = torch.sigmoid(pred) if from_logits else pred  # N,1,D,H,W
    p_t_if_1 = probability_if_1 * target_float_if_1 + (1.0 - probability_if_1) * (1.0 - target_float_if_1)
    p_t_if_1 = p_t_if_1.clamp(min=eps, max=1.0 - eps)
    focal_factor_if_1 = (1.0 - p_t_if_1) ** gamma
    binary_cross_entropy_if_1 = F.binary_cross_entropy_with_logits(input=pred, target=target_float_if_1, reduction='none') if from_logits else F.binary_cross_entropy(input=probability_if_1, target=target_float_if_1, reduction='none')  # N,1,D,H,W
    alpha_per_element_if_1 = (alpha_tensor[1] * target_float_if_1 + alpha_tensor[0] * (1.0 - target_float_if_1))
    loss_if_1 = alpha_per_element_if_1 * focal_factor_if_1 * binary_cross_entropy_if_1
    loss_if_1 = loss_if_1 * valid_mask.unsqueeze(1).type_as(loss)
    
    # Apply softmask
    switch_0_to_1_identifier = (loss_if_1 < loss).bool()
    blended = loss * softmask + loss_if_1 * (1.0 - softmask)
    loss = torch.where(condition=switch_0_to_1_identifier, input=blended, other=loss)
    
    # Apply hardmask
    if hardmask is not None:
        hardmask = hardmask.to(device).type_as(loss)
        hardmask = hardmask.unsqueeze(1) if hardmask.dim() == loss.dim() - 1 else hardmask
        loss = loss * hardmask
    
    # Reduction
    valid_count = valid_mask.sum()
    if reduction == 'mean':
        if valid_count.item() == 0:
            return torch.tensor(0., dtype=loss.dtype, device=device)
        return loss.sum() / valid_count.type_as(loss)
    elif reduction == 'sum':
        return loss.sum()
    else:  # 'none'
        return loss








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
            










class MultiClassFocalLossWithAlpha_and_SoftMask_HardMask(MultiClassFocalLossWithAlpha):
    """
    Extends MultiClassFocalLossWithAlpha with a per-element weight mask (softmask and hardmask) mechanism.
    Args:
        - (Inherits all parameters from MultiClassFocalLossWithAlpha), 注意也继承了参数 from_logits
        - softmask: 必须是 N,C,...
    Forward args:
        - see_channels (bool): whether to use the loss of each channel or not, the outcome is averaged across batch and position:
            channel_loss = loss[target==c].type_as(loss).sum() / ((target==c).type_as(loss).sum() + 0.1)

    我需要你将二分类时“支持softmask”和“支持hardmask”的功能推广到多分类的情况。请务必仔细领会我的意思，仔细思考构思，我一直等着你
    具体实现逻辑为：
     - 1.用户可直接指定softmask,hardmask。或通过kernel生成mask。这里的softmask,hardmask，kernel的维度也扩展为 C D H W 而不是D H W。但也需要支持输入 D H W的情况（即假定每个Channel的mask是一样的）, 和None等情况
     - 2.通过kernel生成softmask与候选label的逻辑,注意接下来我将想你着重说明数学逻辑，你可以换种方法实现，越清晰、效率越高越好：
       - （1）首先我们注意到label的维度为D H W，每个元素取值为0到C-1，表示对应体素处每个具体类别;我们可以等价地将label转换为C D H W，每个体素处取值为0或1，表示“是这个类/不是这个类”
       - （2）对变换后label的每个channel,像二分类一样膨胀即可，注意这会导致每个体素有多个候选label；softmask的每个channel也像二分类那样膨胀，使用最大容忍度策略
     - 3.最终loss的计算：
       - （1）考虑取定每个样本, 每个特定的体素，它有几种候选label
       - （2）对每种候选label，像二分类那样计算loss: 如果把原始label当作改成这个候选label后损失更小，那么和二分类时一样，这个候选label对应的候选损失= softmask的对应值 * 原始label的损失 +（1-softmask的对应值） * 改了label后的损失；如果把原始label当作改成这个候选label后损失更大，那么这个候选损失就是原本的损失
       -（3）对这个样本的这个体素点，取它的最小候选损失即可

    # 注意：这样调用子类时：
    # loss_fn = MultiClassFocalLoss_With_Alpha_and_WeightMask(
    #     gamma=5.0,  # 传入的参数会直接送给父类 MultiClassFocalLossWithAlpha 的 __init__
    #     reduction='sum',
    #     ignore_index=255
    # ) 
    # 相当于：loss_fn = MultiClassFocalLossWithAlpha(gamma=5.0, reduction='sum', ignore_index=255)
    """

    def forward(self, pred, target, softmask=None, kernel=None, hardmask=None, reduction='mean', num_classes=None, see_channels=False):
        if pred.dim() < 2:
            raise ValueError("pred must have shape (N, C, ...)")
        device = pred.device
        target = target.long().to(device)
        N = pred.shape[0]
        if num_classes is None:
            num_classes = pred.shape[1]
        C = num_classes
        
        # 计算原始的逐类别损失（不做 reduction）
        base_loss = super().forward(pred, target, num_classes=num_classes, reduction='none')  # (N,...)
        
#------------------------------- 下面开始计算逐类别损失：如果类别是c，则该样本(n)、该体素(d,h,w)的损失 ------------------------
        log_softmax = F.log_softmax(pred, dim=1) if self.from_logits else torch.log(pred + self.eps)  # (N,C,...), 对数概率
        probs = log_softmax.exp()  # (N,C,...), 真实概率
        alpha_tune = self.alpha_tune.to(device).type_as(pred)
        
        # 计算类别计数
        if self.ignore_index is not None:
            valid_mask = (target != self.ignore_index)
        else:
            valid_mask = torch.ones_like(target, dtype=torch.bool, device=device)
        counts = torch.zeros(C, dtype=torch.float32, device=device)  # C个类别的计数器
        for i in range(C):
            counts[i] = ((target == i) & valid_mask).sum(dtype=torch.float32)
        total = counts.sum()
        if total.item() == 0:
            counts = torch.ones_like(counts)
        
        # 计算 alpha
        if self.use_adaptive_alpha:
            alpha_tensor = 1.0 / (counts + self.flatten_count)
            alpha_tensor = alpha_tensor / alpha_tensor.sum()
            alpha_tensor = alpha_tensor * alpha_tune if alpha_tune.numel() > 1 else alpha_tensor * alpha_tune.item()
        else:
            alpha_tensor = alpha_tune if alpha_tune.numel() > 1 else torch.full((C,), alpha_tune.item(), device=device, dtype=torch.float32)
        alpha_tensor = alpha_tensor.type_as(pred)
        
        # 计算逐类别损失NCDHW————它的意义是：如果类别是c，则该样本(n)、该体素(d,h,w)的损失 = alpha_c * (1 - p_c)^gamma * (-log p_c)
        loss_per_assumed_class = alpha_tensor[None, :, ...].reshape(1, C, *([1] * (probs.dim() - 2))) * ((1 - probs) ** self.gamma) * (-log_softmax)  # (N,C,...)
        loss_per_assumed_class = loss_per_assumed_class.clamp(min=0)  # 防止数值问题
        
#  ---------------------- 下面开始生成 softmask 和 膨胀标签（候选标签）dilated_label，他们都将会 N C D H W... -------------------
        # 处理 softmask：注意kernel有膨胀性质————对于小张量在边缘处加0，损失仍不变（故不妨假定kernel[c]同形状,这验证了假定：kernel可作为张量输入）
        use_origin_softmask = True if softmask is not None else False
        if softmask is None and kernel is None:
            # 默认 softmask 为全 1(对损失而言big_is_big)
            softmask = torch.ones(N, C, *target.shape[1:], dtype=pred.dtype, device=device)
        elif softmask is None and kernel is not None:
            # 通过 kernel 生成 softmask
            has_per_class_kernel = (kernel.dim() == 4)   # 如果对每channel都指定了生成softmax的kernel
            softmask_list = []
            for n in range(N):   # 处理每个样本
                softmask_n = []
                for c in range(C):   # 处理每个channel
                    label_c = (target[n] == c).float()
                    kernel_c = kernel[c] if has_per_class_kernel else kernel
                    piece_softmask = get_WeightMask(label_c, kernel_c, big_is_big=True)
                    softmask_n.append(piece_softmask)
                softmask_list.append(torch.stack(softmask_n, dim=0))
            softmask = torch.stack(softmask_list, dim=0).type_as(pred)
        else:
            # 若使用输入的 softmask
            softmask = softmask.to(device).type_as(pred)
            if softmask.dim() == pred.dim() - 1:  # 若(N,...)无 C 维度
                softmask = softmask.unsqueeze(1).repeat(1, C, *( [1] * (softmask.dim() - 1)) ) # *依次为解包、数组复制。repeat只把第二个维度重复C次（别的位置参数都是1表示不重复）, 最后变成了(N,C,...)
        
        # 生成膨胀标签（候选标签）dilated_label  N,C...
        if use_origin_softmask:
            dilated_label = (softmask < 0.99).type_as(pred)
        elif kernel is not None:
            dilated_list = []
            for n in range(N):
                dilated_n = []
                for c in range(C):
                    label_c = (target[n] == c).float()
                    kernel_c = kernel[c] if kernel.dim() == 4 else kernel
                    piece_dilate = get_WeightMask(label_c, torch.ones_like(kernel_c), big_is_big=False)
                    dilated_n.append((piece_dilate > 0.99).float())
                dilated_list.append(torch.stack(dilated_n, dim=0))
            dilated_label = torch.stack(dilated_list, dim=0).type_as(pred)  # (N,C,...)
        else:
            # 无 kernel 时，候选标签仅为原始标签
            dilated_label = F.one_hot(target, num_classes=C).float().type_as(pred) # (N,...,C) -> permute to (N,C,...)
            dilated_label = dilated_label.permute(0, -1, *range(1, dilated_label.dim()-1)) if dilated_label.dim() > 2 else dilated_label.permute(0, -1)
        
# ---------------------------------------- 下面开始计算候选损失, 并用softmask进行混合(blend) ----------------------------------
        candidate_loss = base_loss.unsqueeze(1).repeat(1, C, *([1] * (base_loss.dim() - 1)))  # (N,C,...), 初始化候选损失
        change_identifier = (loss_per_assumed_class < base_loss.unsqueeze(1)) & (dilated_label > 0)  # (N,C,...),(n,c,d,h,w)表示对样本n的(d,h,w)处的体素而言, 候选标签c是否比原始标签更好
        blended = softmask * base_loss.unsqueeze(1) + (1 - softmask) * loss_per_assumed_class   # 布尔索引，仍为(N,C,...)
        candidate_loss[change_identifier] = blended[change_identifier]
        # 非膨胀标签（候选标签）设为 inf, 但一般不会出现(除非都是ignore_index)
        candidate_loss[dilated_label == 0] = float('inf')
        
        # 应用 hardmask
        if hardmask is not None:
            hardmask = hardmask.to(device).type_as(pred)
            if hardmask.dim() == pred.dim() - 1:  # (N,...) 无 C 维度
                hardmask = hardmask.unsqueeze(1).repeat(1, C, *([1] * (hardmask.dim() - 1)))
            candidate_loss = candidate_loss * hardmask
        
        # 用完softmask了,取最小候选损失就好了
        final_loss = candidate_loss.min(dim=1)[0]  # (N,...)
        final_loss[torch.isinf(final_loss)] = 0  # 正常情况下，候选损失中不会都是 inf(因为总有原始标签)，所以这不会发生
        # 应用有效掩码
        final_loss = final_loss * valid_mask.type_as(final_loss)
        

        if self.scale is not None:
            final_loss *= self.scale.to(device).type_as(final_loss)
        # 应用 reduction
        valid_count = valid_mask.sum()
        if see_channels is False:
            if reduction == 'mean':
                if valid_count.item() == 0:
                    return torch.tensor(0., dtype=final_loss.dtype, device=device)
                return final_loss.sum() / valid_count.type_as(final_loss)
            elif reduction == 'sum':
                return final_loss.sum()
            elif reduction == 'none':
                return final_loss  # no reduction, shape (N, ...) 
        else:
            channels_loss = []
            for c in range(num_classes):
                channel_loss = final_loss[target==c].type_as(final_loss).sum() / ((target==c).type_as(final_loss).sum() + 0.1)
                channels_loss.append(channel_loss)

            if reduction == 'mean':
                if valid_count.item() == 0:
                    return torch.tensor(0., dtype=final_loss.dtype, device=device), *channels_loss
                return final_loss.sum() / valid_count.type_as(final_loss), *channels_loss
            elif reduction == 'sum':
                return final_loss.sum(), *channels_loss
            elif reduction == 'none':
                return final_loss  # no reduction, shape (N, ...) 
        """
        ### MultiClassFocalLossWithAlpha_and_SoftMask_HardMask 类的整体行为

        本类扩展了基类 `MultiClassFocalLossWithAlpha`，加入 softmask 和 hardmask 机制，支持多分类 focal loss 的边界软化和硬权重调整（例如用于分割）。核心思想是为每个体素生成“候选标签”（通过标签膨胀），计算候选损失，若有益则用 softmask 混合，应用 hardmask，最终选择每体素最小损失，促进类边界容忍。

        #### 1. 用户输入的掩码和内核
        - 用户可提供 `softmask`、`hardmask`，或通过 `kernel` 生成，形状支持按类变异或共享：
        - `kernel`：形状 (C, kd, kh, kw)（类特定）或 (kd, kh, kw)（共享，kd/kh/kw 为奇数）。若提供，生成 softmask 和候选标签；若 None，无膨胀（仅原始标签为候选），softmask=1.0。
        - `softmask`：形状 (N, C, ...)（类特定）或 (N, ...)（共享，扩展到 C）。值通常 [0,1]，用于混合损失。若 None 且 kernel 提供，按类生成；否则为全 1（无混合）。
        - `hardmask`：形状 (N, C, ...) 或 (N, ...)（扩展到 C）。乘性权重应用于候选损失（例如硬忽略区域）。若 None，无效果。
        - 处理：掩码无 C 维度时，unsqueeze 并重复 C 次，移动到设备并与 pred 类型一致。

        #### 2. 通过内核生成 Softmask 和候选标签
        - 将多分类问题视为按类二分类子问题，扩展二分类的膨胀和软化逻辑：
        - **标签转换**：对每个样本 n 和类 c，将 target (N, ..., 值 0 到 C-1) 转为二进制 label_c (D, H, W)，target[n] == c 时为 1，否则 0，等价于 one-hot 但单独处理。
        - **候选标签膨胀（dilated_label）**：若 kernel 提供，对每个 label_c：
            - 用 `get_WeightMask(label_c, torch.ones_like(kernel_c), big_is_big=False)` 膨胀，ones_like(kernel_c) 为均匀膨胀内核，big_is_big=False 返回 [0,1] 最大叠加掩码。
            - 阈值化为二进制：(piece_dilate > 0.99).float()，标记 c 为候选（原始或膨胀附近）。
            - 堆叠为 (N, C, ...)，边界处可能多候选。
        - 若无 kernel，dilated_label 为 target 的 one-hot (N, C, ...)，每体素仅原始标签为候选。
        - **Softmask 生成**：若 kernel 提供且 softmask 为 None，对每个 label_c：
            - 用 `get_WeightMask(label_c, kernel_c, big_is_big=True)`，在 1 位置叠加 kernel_c，取最大（“最大容忍度”），返回 1 - cropped（值小表示高容忍，损失低）。
            - 堆叠为 (N, C, ...)，类特定或共享根据 kernel 维度。
        - **数学逻辑**：膨胀通过最大叠加扩展正区域，允许边界跨类候选。循环为清晰实现，生产中可用 3D 卷积优化。

        #### 3. 最终损失计算
        - **基损失和按假设类损失**：
        - `base_loss` (N, ...)：父类计算原始标签的 focal loss（reduction='none'）。
        - `loss_per_assumed_class` (N, C, ...)：每体素假设类 c 的 focal loss：alpha_c * (1 - p_c)^gamma * (-log p_c)，p_c 为 softmax(pred) 的概率，clamp 确保稳定性。
        - **候选损失定义与 Softmask 混合**：
        - 初始化 `candidate_loss` (N, C, ...)：base_loss 重复 C 次（候选起点）。
        - 识别有益候选：loss_per_assumed_class[c] < base_loss 且 dilated_label[c] > 0（c 为候选且切换降低损失）。
        - 混合：candidate_loss[c] = softmask[c] * base_loss + (1 - softmask[c]) * loss_per_assumed_class[c]，softmask 小则更偏向低损失。
        - 非有益或非候选位置：保持 base_loss 或 inf（非候选）。
        - **最小候选损失选择**：
        - 非候选 (dilated_label[c] == 0) 设为 inf，排除 min。
        - 应用 hardmask：candidate_loss 乘 hardmask（按类或共享），支持硬抑制/放大。
        - 每体素取 min：final_loss = candidate_loss.min(dim=1)[0] (N, ...)，选最低混合损失。inf 替换为 0（罕见）。
        - **后处理**：应用 valid_mask（排除 ignore_index），reduction：mean（sum / valid_count）、sum 或 none。
        - **关键效果**：通过允许附近类低损失，促进边界容忍，由膨胀（内核大小/形状）和 softmask 控制。若无 kernel/掩码，退化为基 focal loss。

        #### 潜在问题
        - 无硬性运行时 bug：维度通过 unsqueeze/repeat 处理；设备/类型一致；循环安全。
        - 小问题：alpha_tune 若 numel()>1 且 !=C，无检查，建议加 ValueError。
        - 效率：N*C 循环对大批量/类慢，建议用 conv3d 向量化。
        - 数值：loss_per_assumed_class 已 clamp，softmask 假设 kernel [0,1]，否则可能无效混合，建议 clamp softmask 到 [0,1]。
        """












def multi_class_focal_loss_with_masks(pred, target, gamma=2.0, ignore_index=None, eps=1e-6, flatten_count=1.0, from_logits=True, 
                                      alpha_tune=1.0, use_adaptive_alpha=True, softmask=None, kernel=None, 
                                      hardmask=None, reduction='sum', num_classes=None):
    """
    Multi-class Focal Loss with adaptive alpha tuning, softmask, and hardmask.
    
    Parameters:
        - from_logits (bool): which means, pred isn't pass through softmax or pred is hard probabilities(id from_logits=False)
        - pred: Tensor, shape (N,C,...), raw logits (NOT passed through softmax)
        - target: Tensor, shape (N,...), with values {0..C-1} or ignore_index
        - gamma: float, focusing parameter (>=0)
        - ignore_index: int, optional label value to ignore
        - eps: float, numerical epsilon
        - flatten_count: float, smoothing for class counts
        - alpha_tune: float or list/tuple, initial alpha value(s)
        - use_adaptive_alpha: bool, whether to use inverse frequency alpha
        - softmask: Tensor shapes (N,C,...) or (N,...). If None and kernel, generated per class
        - kernel: Tensor, shape (C,kd,kh,kw) or (kd,kh,kw), for generating masks per class
        - hardmask: Tensor, shape (N,C,...) or (N,...), hard weights applied to candidate losses
        - reduction: str, 'mean' | 'sum' | 'none'
        - num_classes: int, optional, defaults to pred.shape[1]
    
    Returns:
        - loss: scalar (mean/sum) , Tensor (N,...) (none)
    """
    if pred.dim() < 2:
        raise ValueError("pred must have shape (N,C,...)")
    device = pred.device
    target = target.long().to(device)
    N = pred.shape[0]
    if num_classes is None:
        num_classes = pred.shape[1]
    C = num_classes
    if isinstance(alpha_tune, (list, tuple)):
        alpha_tune_tensor = torch.tensor(alpha_tune, dtype=torch.float32, device=device)
    else:
        alpha_tune_tensor = torch.tensor([alpha_tune], dtype=torch.float32, device=device)
    # Build valid mask
    if ignore_index is not None:
        valid_mask = (target != ignore_index)
    else:
        valid_mask = torch.ones_like(target, dtype=torch.bool, device=device)
    # Class counts
    counts = torch.zeros(C, dtype=torch.float32, device=device)
    for i in range(C):
        counts[i] = ((target == i) & valid_mask).sum(dtype=torch.float32)
    total = counts.sum()
    if total.item() == 0:
        counts = torch.ones_like(counts)
    # Alpha tensor
    if use_adaptive_alpha:
        alpha_tensor = 1.0 / (counts + flatten_count)
        alpha_tensor = alpha_tensor / alpha_tensor.sum()
        # Broadcast alpha_tune_tensor if scalar
        if alpha_tune_tensor.numel() == 1:
            alpha_tensor = alpha_tensor * alpha_tune_tensor.item()
        else:
            if alpha_tune_tensor.shape[0] != C:
                raise ValueError(f"alpha_tune list must have length {C}")
            alpha_tensor = alpha_tensor * alpha_tune_tensor
    else:
        if alpha_tune_tensor.numel() == 1:
            alpha_tensor = torch.full((C,), alpha_tune_tensor.item(), device=device, dtype=torch.float32)
        else:
            if alpha_tune_tensor.shape[0] != C:
                raise ValueError(f"alpha_tune list must have length {C}")
            alpha_tensor = alpha_tune_tensor
    alpha_tensor = alpha_tensor.type_as(pred)
    


    # Compute probabilities and log_softmax
    log_softmax = F.log_softmax(pred, dim=1) if from_logits else torch.log(pred + eps)  # (N,C,...)
    probs = log_softmax.exp()  # (N,C,...)
    
    # Loss per assumed class: alpha_c * (1 - p_c)^gamma * (-log p_c)
    loss_per_assumed_class = alpha_tensor[None, :, ...].reshape(1, C, *([1] * (probs.dim() - 2))) * ((1 - probs) ** gamma) * (-log_softmax)  # (N,C,...)
    loss_per_assumed_class = loss_per_assumed_class.clamp(min=0)  # Avoid numerical issues
    
    # Original loss per voxel
    index = target.unsqueeze(1)  # (N,1,...)
    original_loss = torch.gather(loss_per_assumed_class, dim=1, index=index).squeeze(1)  # (N,...)
    
    # Generate dilated and softmask if kernel provided
    if softmask is not None:
        dilated = (softmask < 0.99).type_as(pred)
    elif softmask is None and kernel is not None:
        dilated_list = []
        softmask_list = []
        has_per_class_kernel = (kernel.dim() == 4)
        for n in range(N):    # 对样本循环
            dilated_n = []
            softmask_n = []
            for c in range(C):   # 对channel循环
                label_c = (target[n] == c).float()  # (D,H,W)
                if has_per_class_kernel:
                    kernel_c = kernel[c]
                else:
                    kernel_c = kernel
                # Dilated for candidates
                piece_dilate = get_WeightMask(label_c, torch.ones_like(kernel_c), big_is_big=False)
                dilated_n.append((piece_dilate > 0.99).float())
                # Softmask
                piece_softmask = get_WeightMask(label_c, kernel_c, big_is_big=True)
                softmask_n.append(piece_softmask)
            dilated_list.append(torch.stack(dilated_n, dim=0))  # (C, D, H, W)
            softmask_list.append(torch.stack(softmask_n, dim=0))
        dilated = torch.stack(dilated_list, dim=0).type_as(pred)  # (N,C,...), 这里的dilated就是上面的dilated_label变量(膨胀/候选标签),懒的统一了
        softmask_gen = torch.stack(softmask_list, dim=0).type_as(pred)  # (N,C,...)
    else:
        # No kernel: candidates only original class, softmask ones
        dilated = F.one_hot(target, num_classes=C).float().type_as(pred)  # (N,...,C) -> permute to (N,C,...)
        dilated = dilated.permute(0, -1, *range(1, dilated.dim()-1)) if dilated.dim() > 2 else dilated.permute(0, -1)
        softmask_gen = torch.ones_like(dilated)
    
    # Handle input softmask
    if softmask is None:
        softmask = softmask_gen
    else:
        softmask = softmask.to(device).type_as(pred)
        if softmask.dim() == pred.dim() - 1:  # (N,...) no C, expand
            softmask = softmask.unsqueeze(1).repeat(1, C, *([1] * (softmask.dim() - 1)))
    
    # Compute candidate_loss
    candidate_loss = original_loss.unsqueeze(1).repeat(1, C, *([1] * (original_loss.dim() - 1)))  # (N,C,...)
    change_identifier = (loss_per_assumed_class < original_loss.unsqueeze(1)) & (dilated > 0)
    blended = softmask * original_loss.unsqueeze(1) + (1 - softmask) * loss_per_assumed_class
    candidate_loss[change_identifier] = blended[change_identifier]
    
    # Mask non-candidates to inf for min
    candidate_loss[dilated == 0] = float('inf')
    
    # Apply hardmask if provided
    if hardmask is not None:
        hardmask = hardmask.to(device).type_as(pred)
        if hardmask.dim() == pred.dim() - 1:  # (N,...) no C
            hardmask = hardmask.unsqueeze(1).repeat(1, C, *([1] * (hardmask.dim() - 1)))
        candidate_loss = candidate_loss * hardmask
    
    # Final loss per voxel: min over candidates
    final_loss = candidate_loss.min(dim=1)[0]  # (N,...)
    final_loss[torch.isinf(final_loss)] = 0  # If no candidates (should not happen), set 0
    
    # Apply valid_mask
    final_loss = final_loss * valid_mask.type_as(final_loss)
    
    # Reduction
    valid_count = valid_mask.sum()
    if reduction == 'mean':
        if valid_count.item() == 0:
            return torch.tensor(0., dtype=final_loss.dtype, device=device)
        return final_loss.sum() / valid_count.type_as(final_loss)
    elif reduction == 'sum':
        return final_loss.sum()
    else:  # 'none'
        return final_loss









# ============== 搬过来的 ==========
class FocalLoss(nn.Module):
    """
    =============================================================================
    FocalLoss 焦点损失函数
    =============================================================================
    用于处理类别不平衡问题的损失函数。通过给难分类样本更高的权重，
    使模型更关注难分类的样本。
    
    公式: FL(p) = -alpha * (1-p)^gamma * log(p)
    
    输入参数 (Input Parameters):
    -----------------------------------------------------------------------------
    - gama: float, 默认=2, 焦点参数，控制对难分类样本的关注程度
    - eps: float, 默认=1e-6, 防止log(0)的小常数
    
    forward输入:
    - x: torch, (B, C, H, W, D), 网络输出的logits
    - y: torch, (B, H, W, D), 真实标签(整数类别)
    
    输出 (Output):
    -----------------------------------------------------------------------------
    - forward返回: torch, scalar, 焦点损失值
    =============================================================================
    """
    def __init__(self,gama=2,eps=1e-6):
        super(FocalLoss,self).__init__()
        # eps: float, 数值稳定性常数
        self.eps = eps
        # gamma: float, 焦点参数
        self.gamma = gama
    
    def forward(self,x,y):
        """
        前向传播
        
        输入参数 (Input Parameters):
        - x: torch, (B, C, H, W, D), 网络输出logits，C为类别数
        - y: torch, (B, H, W, D), 真实标签，整数类别索引
        
        输出 (Output):
        - torch, scalar, 焦点损失值
        """
        # N: int, 总样本数(体素数)
        N = y.numel()
        # M: int, 正样本数(标签>0.5的体素数)
        M = torch.sum(y>0.5)
        # p: torch, (B, C, H, W, D), softmax概率
        p = x.softmax(dim=1)
        # class_num: int, 类别数
        class_num = p.size(1)
        # class_mask: torch, (B, H, W, D, C), one-hot编码的标签
        class_mask = nn.functional.one_hot(y, class_num)
        # class_mask: torch, (B, C, H, W, D), 重排维度
        class_mask = einops.rearrange(class_mask, 'b h w d c -> b c h w d', c=class_num)
        # alpha: torch, (class_num,), 类别权重
        # 背景类权重=1，其他类权重=(N-M)/M(类别不平衡调整)
        alpha = torch.tensor([1]+[(N-M)/M]*(class_num-1),device=y.device)
        # alpha: torch, (B, H, W, D), 每个样本对应的类别权重
        alpha = alpha[y]
        # p: torch, (B, H, W, D), 真实类别对应的预测概率
        p = torch.sum(p * class_mask, dim=1)
        # loss: torch, (B, H, W, D), 每个样本的焦点损失
        loss = -alpha * (1 - p) ** self.gamma * torch.log(p + self.eps)
        return loss.mean()*(N/(N-M))


class RandomCrop(object):
    """
    =============================================================================
    RandomCrop 随机裁剪数据增强
    =============================================================================
    对3D体素数据进行随机裁剪，用于数据增强。
    如果输入尺寸小于目标尺寸，会先进行padding。
    
    输入参数 (Input Parameters):
    -----------------------------------------------------------------------------
    - output_size: int, 输出的立方体边长
    - ispadding: bool, 默认=True, 输入尺寸不足时是否进行padding
    
    __call__输入:
    - *x: tuple of numpy arrays, 每个元素形状为(D, H, W)
    
    输出 (Output):
    -----------------------------------------------------------------------------
    - __call__返回: tuple of numpy arrays, 每个元素形状为(output_size, output_size, output_size)
    =============================================================================
    """
    def __init__(self,output_size:int,ispadding:bool=True):
        assert isinstance(output_size,int)
        # output_size: tuple, 输出尺寸(立方体)
        self.output_size = (output_size,output_size,output_size)
        # ispadding: bool, 是否在必要时进行padding
        self.ispadding = ispadding
    
    def __call__(self,*x):
        """
        执行随机裁剪
        
        输入参数 (Input Parameters):
        - *x: tuple of numpy arrays, 需要同步裁剪的多个3D数组
          每个元素形状为(D, H, W)
        
        输出 (Output):
        - y: tuple of numpy arrays, 裁剪后的数组
          每个元素形状为(output_size, output_size, output_size)
        """
        # y: list, 存储裁剪结果
        y=[]
        # d, h, w: int, 输入尺寸
        d,h,w=x[0].shape
        # od, oh, ow: int, 输出尺寸
        od,oh,ow=self.output_size
        
        if self.ispadding:
            x=list(x)
            # 计算每个维度需要的padding量
            # k1: int, D维度需要padding的总量
            k1=max(od-d,0);pad1 = k1//2;pads1 = (pad1,pad1) if k1 % 2 == 0 else (pad1,pad1+1);
            # k2: int, H维度需要padding的总量
            k2 = max(oh-h, 0);pad2 = k2 // 2;pads2 = (pad2, pad2) if k2 % 2 == 0 else (pad2, pad2 + 1);
            # k3: int, W维度需要padding的总量
            k3 = max(ow-w, 0);pad3 = k3 // 2;pads3 = (pad3, pad3) if k3 % 2 == 0 else (pad3, pad3 + 1);
            for i in range(len(x)):
                # 对每个输入数组进行padding
                x[i] = np.pad(x[i],(pads1,pads2,pads3),mode='constant')
        
        # 更新padding后的尺寸
        d, h, w = x[0].shape
        # sd, sh, sw: int, 随机选择的裁剪起始位置
        sd = random.randint(0,d-od)
        sh = random.randint(0,h-oh)
        sw = random.randint(0,w-ow)
        
        for ix in x:
            # 对每个数组执行相同的裁剪
            y.append(ix[sd:sd+od,sh:sh+oh,sw:sw+ow])
        y = tuple(y)
        return y


class FocalTverskyLoss(nn.Module):
    """
    =============================================================================
    FocalTverskyLoss 焦点Tversky损失函数
    =============================================================================
    结合Focal Loss和Tversky Loss的损失函数，特别适用于处理类别不平衡的分割任务。
    Tversky Index是Dice系数的推广，允许分别控制假阳性和假阴性的权重。
    
    公式: TI = TP / (TP + alpha*FN + beta*FP)
          FTL = (1 - TI)^gamma
    
    输入参数 (Input Parameters):
    -----------------------------------------------------------------------------
    - factors: list/tensor, 每个类别的权重因子
    - alpha: float, 默认=0.9, 假阴性(漏检)的权重
    - beta: float, 默认=0.1, 假阳性(误检)的权重
    - gamma: float, 默认=0.75, 焦点参数
    - eps: float, 默认=1e-3, 数值稳定性常数
    
    forward输入:
    - y_p: torch, (B, C, H, W, D), 网络输出logits
    - y_t: torch, (B, H, W, D), 真实标签
    
    输出 (Output):
    -----------------------------------------------------------------------------
    - forward返回: torch, scalar, 焦点Tversky损失值
    =============================================================================
    """
    def __init__(self, factors,alpha=0.9, beta=0.1, gamma=0.75,eps=1e-3):
        super(FocalTverskyLoss, self).__init__()
        # alpha: float, 假阴性权重
        self.alpha = alpha
        # beta: float, 假阳性权重
        self.beta = beta
        # gamma: float, 焦点参数
        self.gamma = gamma
        # factors: list/tensor, 类别权重因子
        self.factors = factors
        # epsilon: float, 数值稳定性常数
        self.epsilon = eps
    
    def forward(self, y_p, y_t):
        """
        前向传播
        
        输入参数 (Input Parameters):
        - y_p: torch, (B, C, H, W, D), 网络输出logits，C为类别数
        - y_t: torch, (B, H, W, D), 真实标签，整数类别索引
        
        输出 (Output):
        - torch, scalar, 焦点Tversky损失值
        """
        # Ensure the predictions are in the same dimension as y_true
        # loss: float/torch, 累积损失
        loss = 0
        # y_pp: torch, (B, C, H, W, D), softmax概率
        y_pp = y_p.softmax(dim=1)
        # class_num: int, 类别数
        class_num = y_p.size(1)
        # y_tt: torch, (B, H, W, D, C), one-hot编码的标签
        y_tt = nn.functional.one_hot(y_t, class_num)
        # y_tt: torch, (B, C, H, W, D), 重排维度
        y_tt = einops.rearrange(y_tt, 'b h w d c -> b c h w d', c=class_num)
        
        # 对每个前景类别(从1开始，跳过背景类0)计算损失
        for ii in range(1,class_num):
            # y_true: torch, (B, H, W, D), 第ii类的真实标签
            y_true = y_tt[:,ii]
            # y_pred: torch, (B, H, W, D), 第ii类的预测概率
            y_pred = y_pp[:,ii]
            
            # Calculate the Tversky loss
            # tp: torch, (B,), 真阳性
            tp = (y_true * y_pred).sum(dim=(-3, -2, -1))
            # fn: torch, (B,), 假阴性(漏检)
            fn = (y_true * (1 - y_pred)).sum(dim=(-3, -2, -1))
            # fp: torch, (B,), 假阳性(误检)
            fp = ((1 - y_true) * y_pred).sum(dim=(-3, -2, -1))

            # tversky_index: torch, (B,), Tversky指数
            tversky_index = tp / (tp + self.alpha * fn + self.beta * fp + self.epsilon)

            # Calculate the Focal Tversky loss
            # 累加每个类别的加权焦点Tversky损失
            loss = loss + ((1 - tversky_index+self.epsilon).pow(self.gamma))*self.factors[ii]
        
        return loss.mean()


class AutomaticWeightedLoss(nn.Module):
    """
    =============================================================================
    AutomaticWeightedLoss 自动加权多任务损失
    =============================================================================
    实现多任务学习中的自动加权机制。使用可学习的参数来自动平衡多个损失项。
    
    公式: L = sum_i (0.5 / sigma_i^2 * L_i + log(1 + sigma_i^2))
    
    输入参数 (Input Parameters):
    -----------------------------------------------------------------------------
    - *init_params: float, 每个损失项的初始权重参数
    
    forward输入:
    - *losses: torch tensors, 多个损失值
    
    输出 (Output):
    -----------------------------------------------------------------------------
    - forward返回: torch, scalar, 加权后的总损失
    =============================================================================
    """
    def __init__(self, *init_params):
        super(AutomaticWeightedLoss, self).__init__()
        # params: torch.nn.Parameter, 可学习的权重参数
        params = torch.tensor(init_params, requires_grad=True)
        self.params = torch.nn.Parameter(params)

    def forward(self, *losses):
        """
        前向传播
        
        输入参数 (Input Parameters):
        - *losses: tuple of torch tensors, 多个损失值
        
        输出 (Output):
        - loss_sum: torch, scalar, 加权后的总损失
        """
        # loss_sum: torch, scalar, 累积损失
        loss_sum = 0
        for i, loss in enumerate(losses):
            # 根据公式计算每个损失的加权值并累加
            # 0.5/sigma^2 * loss 是数据不确定性加权
            # log(1+sigma^2) 是正则化项，防止sigma趋向无穷大
            loss_sum += 0.5 / (self.params[i] ** 2) * loss + torch.log(1 + self.params[i] ** 2)
        return loss_sum








# # 第二部分：用于ligand的二分类
# # 1️⃣ Tversky Loss - 体积重叠优化
#     # 控制假阳性(FP)和假阴性(FN)的权重————参数 α 和 β 可针对不同目标大小调整
#     # 特别适合类别不平衡问题

# # 2️⃣ Focal Loss - 梯度稳定性优化
#     # 降低易分类样本权重，关注困难样本————参数 γ 控制聚焦程度（通常2.0）
#     # 解决类别不平衡导致的训练不稳定

# # 3️⃣ Boundary Loss - 拓扑连贯性优化
#     # 使用距离图引导边界精确性
#     # 特别适合需要精确边界的医学分割
# # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# from scipy.ndimage import distance_transform_edt


# # ====================================================================== Boundary Loss  ======================================================================

# # ======================== compute_distance_map ========================
# def compute_distance_map(mask: np.ndarray, normalize: bool = False, clip: float = None) -> np.ndarray:
#     """
#     计算一张二值 mask 的 signed distance map φ_G(p)
#     - 输入和返回的形状都为 (D,H,W)。

#     参数:
#         mask: numpy array, 二值 (0 背景, 1 前景)，shape (H,W) 或 (D,H,W)
#         normalize: 是否按样本最大绝对值归一化到 [-1,1]
#         clip: 若不为 None，则先将 φ 裁剪到 [-clip, clip]（像素/体素单位）

#     返回:
#         distance_map: 前景内部 φ < 0，外部 φ > 0（phi = dist_out - dist_in）
#     """
#     mask_bool = mask.astype(bool)
#     # pos_dist: 前景像素到背景的距离（foreground distance）
#     pos_dist = distance_transform_edt(mask_bool)
#     # neg_dist: 背景像素到前景的距离（background distance）
#     neg_dist = distance_transform_edt(~mask_bool)
#     # φ = dist_out - dist_in，使得前景内部为负
#     distance_map = neg_dist.astype(np.float32) - pos_dist.astype(np.float32)

#     if clip is not None:
#         distance_map = np.clip(distance_map, -clip, clip)
#     if normalize:
#         maxv = np.max(np.abs(distance_map))
#         if maxv > 0:
#             distance_map = distance_map / maxv
#     return distance_map.astype(np.float32)



# # 用 cupy 在 GPU 上计算 distance_maps，并用 DLPack 零拷贝为 PyTorch tensor
# import cupy as cp
# import cupyx.scipy.ndimage as cnd
# from torch.utils.dlpack import from_dlpack, to_dlpack

# def compute_distance_map_cupy(mask_np, clip=None, normalize=False, save_path=None, give_torch_rather_numpy=False):
#     """
#     mask_np: numpy 二值 mask，shape 可为 (D,H,W)，dtype 任意整型或 bool
#     - clip: 裁剪到 [-clip, clip]
#     - normalize: 是否按 max-abs 归一化
#     - save_path: 若不为 None，会将返回的 numpy array 保存在该路径（.npy）
#     - give_torch_rather_numpy: True 返回 torch.cuda.Tensor；False 返回 numpy.ndarray
#     返回：根据 give_torch_rather_numpy 返回 torch.Tensor（CUDA）或 numpy.ndarray
#     """
#     # numpy -> cupy（复制到 GPU）
#     arr_gpu = cp.asarray(mask_np.astype(bool))

#     # 前景/背景距离（GPU 上的 EDT）
#     pos_dist = cnd.distance_transform_edt(arr_gpu)
#     neg_dist = cnd.distance_transform_edt(~arr_gpu)
#     distance_map_gpu = (neg_dist - pos_dist).astype(cp.float32)

#     # clip / normalize（在 GPU 上）
#     if clip is not None:
#         distance_map_gpu = cp.clip(distance_map_gpu, -clip, clip)
#     if normalize:
#         maxv = cp.max(cp.abs(distance_map_gpu))
#         if float(maxv) > 0:
#             distance_map_gpu = distance_map_gpu / maxv

#     # 根据请求返回 torch 或 numpy
#     if give_torch_rather_numpy:
#         # cupy -> torch（DLPack 零拷贝，返回 CUDA tensor）
#         dlp = distance_map_gpu.toDlpack()
#         distance_map_torch = from_dlpack(dlp)  # CUDA tensor, float32
#         if save_path is not None:
#             np.save(save_path, distance_map_torch.detach().cpu().numpy())
#         return distance_map_torch
#     else:
#         # cupy -> numpy（会从 GPU 拷贝回主机）
#         distance_map_np = cp.asnumpy(distance_map_gpu)
#         if save_path is not None:
#             np.save(save_path, distance_map_np)
#         return distance_map_np



# # =========================== compute_distance_map_batch：批量计算（返回 numpy）==============================
# def compute_distance_map_batch(masks, clip=None, normalize=False, use_cupy=True):
#     """
#     masks: numpy array, 支持 (N, D, H, W) 或 (N, 1, D, H, W)
#     返回: numpy array, shape (N, D, H, W)
#     说明：本函数返回 numpy 数组（便于后面统一转为 torch）
#     """
#     # 去掉可能的 channel 维 (N,1,D,H,W) -> (N,D,H,W)
#     if masks.ndim == 5 and masks.shape[1] == 1:
#         masks = masks.squeeze(1)

#     batch_size = masks.shape[0]
#     out_shape = masks.shape  # (N, D, H, W)
#     distance_maps = np.zeros(out_shape, dtype=np.float32)

#     for i in range(batch_size):
#         if use_cupy is True:
#             # 让compute_distance_map_cupy 返回 numpy
#             distance_maps[i] = compute_distance_map_cupy(masks[i], clip=clip, normalize=normalize,
#                                                          give_torch_rather_numpy=False)
#         else:
#             distance_maps[i] = compute_distance_map(masks[i], clip=clip, normalize=normalize)
#     return distance_maps



# # ======================== BoundaryLoss 类 ========================
# class BoundaryLoss(nn.Module):
#     def __init__(self, from_logits=True, normalize=True, clip=None, alpha=1.0, 
#                  use_precomputed=True, use_cupy=True, reduction='mean'):
#         """
#         二分类 Boundary Loss 实现（只支持3D）。distance_map: 前景内部 φ < 0，外部 φ > 0（phi = dist_out - dist_in）
#         L = sum_p phi_G(p) * S_P(p)

#         参数:
#             - reduction: 'mean' / 'sum' / 'none'，其中'none': 返回每个样本的每个像素处的损失 N D H W
#             - from_logits: 如果为 True，则把 pred 当作 logits 并在内部做 sigmoid
#             - normalize: 是否对每个样本的 distance_map 做 max-abs 归一化（避免 scale 问题）
#             - clip: 可选 float，先把 phi 裁剪到 [-clip, clip]
#             - alpha: 正类相比于负类的权重，将distance_map的负区域乘以alpha, 正区域不变
#             - use_precomputed: 如果 True，forward 中应传入预计算好的 distance_maps（推荐在 dataset 中预计算）
#             - use_cupy: 是否使用 cupy 加速计算 distance_maps

#         forward参数:
#             - pred: torch.Tensor，模型输出 logits 或 probabilities，shape 可为：(N, 1, D, H, W) 或 (N, D, H, W)
#             - target: 可选，二值 mask tensor,用于在线计算 distance_maps，当 distance_maps 未提供且 use_precomputed=False 时必须给出
#                     shape 与 pred 的空间维相同，形状必须为 (N, D, H, W)
#             - distance_maps: 可选，预计算的 signed distance maps tensor(如果不是None则优先使用，以计算BoundaryLoss).dtype float32，shape只能是 (N, D, H, W)
#                            （推荐在 Dataset 中预计算并传入以加速训练）
#             返回:根据 reduction 返回标量或每样本标量张量,注意none时返回每个样本每个体素处的损失 N D H W
#         """
#         super(BoundaryLoss, self).__init__()
#         self.from_logits = from_logits
#         self.normalize = normalize
#         self.clip = clip
#         self.alpha = alpha

#         self.use_precomputed = use_precomputed
#         self.use_cupy = use_cupy
#         self.reduction = reduction

#     def forward(self, pred, target=None, distance_maps=None):
#         # 统一 pred 形状 -> (N, *spatial)
#         if pred.dim() == 5 and pred.size(1) == 1:
#             pred = pred.squeeze(1)  # (N, D, H, W)
#         batch_size = pred.size(0)

#         # 准备 distance_maps（phi）
#         if distance_maps is None:
#             # if self.use_precomputed:
#             #     raise ValueError("use_precomputed=True 时必须在 forward 中传入预计算的 distance_maps（推荐在 Dataset 中预计算）。")
#             # 否则在线计算 distance_maps（慢），需要 target
#             if target is None:   
#                 raise ValueError("没有提供 distance_maps，且 use_precomputed=False，但 target 也为空，无法计算 distance_maps。")
#             target_np = target.detach().cpu().numpy()
#             distance_maps_np = compute_distance_map_batch(target_np, normalize=False, clip=self.clip, use_cupy=self.use_cupy)
#             distance_maps = torch.as_tensor(distance_maps_np, dtype=pred.dtype, device=pred.device)
#         else:
#             distance_maps = distance_maps.to(device=pred.device, dtype=pred.dtype)   # 转到 pred 相同 device 与 dtype


#         if self.clip is not None:
#             distance_maps = torch.clamp(distance_maps, -self.clip, self.clip)
#         if self.normalize is True:   # per-sample 归一化 phi（可选，按样本 max abs), view为在共享内存的前提下变换形状
#             flat = distance_maps.view(batch_size, -1)
#             # 在 PyTorch 中使用 torch.max(input, dim=d) 时，它返回的是一个包含两个张量的 元组 (tuple)：最大值张量 (Values Tensor): 沿指定维度 d 上的最大值。索引张量 (Indices Tensor): 最大值所在的索引位置
#             max_abs = torch.max(torch.abs(flat), dim=1)[0]  # (N, )
#             max_abs = torch.clamp(max_abs, min=1.0)
#             # reshape 以广播
#             distance_maps = distance_maps / max_abs.view(batch_size, 1,1,1,1)
#         if (self.alpha is not None) and (self.alpha != 1.0):
#             distance_maps = torch.where(distance_maps < 0, distance_maps * self.alpha, distance_maps)


#         probability = torch.sigmoid(pred) if self.from_logits else pred    # 将 pred（可能是 logits）转换为概率 probability
#         # reduction
#         if self.reduction == 'mean':
#             return (distance_maps * probability).mean()
#         elif self.reduction == 'sum':
#             return (distance_maps * probability).sum()
#         else:  # 'none' 返回每个样本每个体素处的损失 N D H W
#             return distance_maps * probability



# # ====================================================================== Tversky Loss ======================================================================
# class TverskyLoss(nn.Module):
#     def __init__(self, from_logits=True, alpha=0.7, beta=0.3, smooth=1e-5, reduction='mean'):
#         """  Tversky Loss: 优化体积重叠 - 支持2D和3D
#         参数:
#             - alpha: FP的权重，增大alpha会减少假阳性FP（提高precision）
#             - beta: FN的权重，增大beta会减少假阴性FN（提高recall）
#             - smooth: 平滑项，避免除零
#         forward 参数:
#             - pred: torch.Tensor， shape (N, 1, D, H, W) 或 (N, D, H, W)
#             - target: torch.Tensor, 只能是(N, D, H, W)
#             - reduction: 'mean' /'sum' / 'none'
#         """
#         super(TverskyLoss, self).__init__()
#         self.from_logits = from_logits
#         self.alpha = alpha
#         self.beta = beta
#         self.smooth = smooth
#         self.reduction = reduction
        
    
#     def forward(self, pred, target):
#         pred = torch.sigmoid(pred) if self.from_logits else pred  # 转换为概率
#         if pred.ndim == 5:  # N 1 D H W -> N D H W
#             pred = pred.squeeze(1)
#         if target.ndim != 4:
#             raise ValueError("target 必须是 (N, D, H, W) 形状")
        
#         # 计算TP, FP, FN, sum over (D, H, W)
#         TP = (pred * target).sum(dim=(1, 2, 3))
#         FP = (pred * (1 - target)).sum(dim=(1, 2, 3))
#         FN = ((1 - pred) * target).sum(dim=(1, 2, 3))
#         tversky_index = (TP + self.smooth) / (TP + self.alpha * FP + self.beta * FN + self.smooth)
        
#         if self.reduction =='mean':
#             tversky_loss = (1.0 - tversky_index).mean()
#         elif self.reduction =='sum':
#             tversky_loss = (1.0 - tversky_index).sum()
#         else:  # 'none' 返回每个样本每个体素处的损失 N D H W
#             tversky_loss = 1.0 - tversky_index
#         return tversky_loss



# # ============== Focal Loss ==============
# class FocalLoss(nn.Module):
#     """
#     Focal Loss: 优化梯度稳定性，关注困难样本
    
#     通过降低易分类样本的权重，使模型更关注难分类样本
#     FL(p_t) = -α(1-p_t)^γ log(p_t)
#     """
    
#     def __init__(self, from_logits=True, alpha=0.25, gamma=2.0, reduction='mean'):
#         """
#         参数:
#             alpha: 平衡因子，调节正负样本的权重 (通常 0.25)
#             gamma: 聚焦参数，调节难易样本的权重 (通常 2.0)
#                    gamma=0 时退化为标准交叉熵
#                    gamma越大，对易分类样本的抑制越强
#             reduction: 'mean' 或 'sum'
#         """
#         super(FocalLoss, self).__init__()
#         self.form_logits = from_logits
#         self.alpha = alpha
#         self.gamma = gamma
#         self.reduction = reduction
    
#     def forward(self, pred, target):
#         """
#         参数:
#             pred: torch.Tensor, shape (B, 1, H, W) 或 (B, H, W)
#                   预测概率 (sigmoid后的值)
#             target: torch.Tensor, shape (B, 1, H, W) 或 (B, H, W)
#                     ground truth (0或1)
#         """
#         pred = torch.sigmoid(pred) if self.form_logits else pred  # 转换为概率
#         if pred.ndim == 4:
#             pred = pred.squeeze(1)
#         if target.ndim == 4:
#             target = target.squeeze(1)
        
#         # 避免log(0)
#         eps = 1e-7
#         pred = torch.clamp(pred, eps, 1 - eps)
        
#         # 计算交叉熵
#         ce_loss = -target * torch.log(pred) - (1 - target) * torch.log(1 - pred)
        
#         # 计算p_t
#         p_t = torch.where(target == 1, pred, 1 - pred)
        
#         # 计算focal权重
#         focal_weight = (1 - p_t) ** self.gamma
        
#         # 计算alpha权重
#         alpha_weight = torch.where(target == 1, self.alpha, 1 - self.alpha)
        
#         # Focal Loss
#         focal_loss = alpha_weight * focal_weight * ce_loss
        
#         if self.reduction == 'mean':
#             return focal_loss.mean()
#         elif self.reduction == 'sum':
#             return focal_loss.sum()
#         else:
#             return focal_loss
# class BinaryFocalLossWithAlpha(nn.Module):
#     def __init__(self, from_logits=True, gamma=2.0, ignore_index=None, eps=1e-6, flatten_count=1.0, alpha_tune=None):
#         """
#         Binary (single-channel logits) Focal Loss with adaptive alpha tuning.

#         Notes on parameters:
#         - from_logits: if True, assume input is raw logits (NOT passed through sigmoid)
#         - gamma: focusing parameter (>=0). Typical values: 1.0-3.0.
#         - reduction: 'mean' | 'sum' | 'none'
#         - ignore_index: optional label value to ignore in target
#         - eps: numerical epsilon to clamp probabilities/logs
#         - alpha_tune:
#             * None: no multiplicative tuning (alpha derived from counts only if desired below)
#             * float in (0,1): interpreted as alpha_pos (weight used for positive class), alpha_neg = 1-alpha_pos
#             * list/tuple of length 2: interpreted as multiplicative scaling factors [scale_neg, scale_pos]
#             These will multiply the base computed alpha per class (see below).
#         Behavior:
#         - Expects `logits` shape (N, 1, D, H, W) (or (N,1,...) generally).
#         - Expects `target` shape (N, D, H, W) with integer class labels 0 or 1.
#         """
#         super().__init__()
#         self.from_logits = from_logits
#         self.gamma = float(gamma)
#         self.ignore_index = ignore_index
#         self.eps = float(eps)
#         self.flatten_count = flatten_count
#         # store alpha_tune raw; we'll interpret it in forward
#         self.alpha_tune = alpha_tune

#     def forward(self, logits, target, reduction='sum'):
#         """
#         - logits: Tensor, shape (N,1,...) single-channel raw logits (NOT passed through sigmoid)
#         - target: Tensor, shape (N,...) with values {0,1} or possibly ignore_index
#         """
#         if logits.dim() < 2:
#             raise ValueError("logits must have shape (N,1,...) for binary segmentation")
#         device = logits.device
#         if target.dtype != torch.long:
#             target_long = target.long()
#         else:
#             target_long = target
#         target_long = target_long.to(device)
#         # Build valid mask (exclude ignore_index)
#         if self.ignore_index is not None:
#             valid_mask = (target_long != self.ignore_index)
#         else:
#             valid_mask = torch.ones_like(target_long, dtype=torch.bool, device=device)
#         # Compute counts per class (exclude ignored)，ensure only consider valid positions
#         valid_target = target_long[valid_mask]   # 按照张量索引valid_mask, 返回一维张量
#         # If no valid points, handle gracefully
#         if valid_target.numel() == 0:  # numel() 是 PyTorch 张量的一个方法，用于返回张量中元素的总数
#             # return zero tensor (same dtype/device)
#             if self.reduction == 'none':
#                 return torch.zeros_like(logits, dtype=logits.dtype, device=device)
#             else:
#                 return torch.tensor(0., dtype=logits.dtype, device=device)
#         count_pos = (valid_target == 1).sum(dtype=torch.float32) + self.flatten_count
#         count_neg = (valid_target == 0).sum(dtype=torch.float32) + self.flatten_count
#         # Base alpha per class: inverse-frequency (rare class gets larger base weight)
#         inv = torch.tensor([1.0 / (count_neg + self.eps), 1.0 / (count_pos + self.eps)], device=device, dtype=torch.float32)
#         alpha_base = inv / inv.sum()  # shape (2,) -> torch.tensor([alpha_neg, alpha_pos])


#         if self.alpha_tune is None:
#             alpha_tensor = alpha_base
#         else:
#             # if alpha_tune is a scalar in (0,1): interpret as alpha_pos (weight for positive), set alpha_neg = 1 - alpha_pos
#             if isinstance(self.alpha_tune, (float, int)):
#                 a_pos = float(self.alpha_tune)
#                 a_pos = max(min(a_pos, 1.0), 0.0)  # clamp to [0,1]
#                 alpha_tensor = torch.tensor([1.0 - a_pos, a_pos], device=device, dtype=torch.float32)
#             else:
#                 # list/tuple length 2 => multiplicative scales for [neg, pos]
#                 if hasattr(self.alpha_tune, "__len__") and len(self.alpha_tune) == 2:
#                     scales = torch.tensor(self.alpha_tune, device=device, dtype=torch.float32)
#                     alpha_tensor = alpha_base * scales
#                     alpha_tensor = alpha_tensor / alpha_tensor.sum()
#                 else:
#                     alpha_tensor = alpha_base

#         # Ensure alpha_tensor dtype matches logits
#         alpha_tensor = alpha_tensor.to(device=device).type_as(logits)
#         # Previous: logits (N,1,...) ,target_long (N,...)
#         if logits.shape[1] == 1:
#             target_float = target_long.unsqueeze(1).to(dtype=logits.dtype)  # (N,1,...)
#         else:
#             # unexpected channel dim >1 but user used single-output config
#             raise ValueError("logits channel dim expected to be 1 for binary loss")

#         # compute probabilities
#         probability = torch.sigmoid(logits) if self.from_logits else logits  # (N,1,...)
#         # p_t = prob if target==1 else 1-prob
#         p_t = probability * target_float + (1.0 - probability) * (1.0 - target_float) # (N,1,...)
#         p_t = p_t.clamp(min=self.eps, max=1.0 - self.eps) # (N,1,...)
#         focal_factor = (1.0 - p_t) ** self.gamma  # (N,1,...)
#         # per-element cross-entropy (stable): binary cross entropy with logits (equals -log(p_t))
#         binary_cross_entropy = F.binary_cross_entropy_with_logits(input=logits, target=target_float, reduction='none')  # (N,1,...)

#         # build alpha_per_element with shape (N,1,...)
#         alpha_per_element = (alpha_tensor[1] * target_float  +  alpha_tensor[0] * (1.0 - target_float)).type_as(binary_cross_entropy)
#         # final loss (per element)
#         loss = alpha_per_element * focal_factor * binary_cross_entropy  # (N,1,...)
#         # valid_mask shape (N,...); expand to channel dim
#         mask = valid_mask.unsqueeze(1).type_as(loss)  # (N,1,...)
#         loss = loss * mask
#         # loss = loss * (count_pos + count_neg) / count_pos  # 将loss扩大一个系数，便于梯度计算

#         if reduction == 'mean':
#             valid_count = mask.sum()
#             if valid_count.item() == 0:
#                 return torch.tensor(0., dtype=loss.dtype, device=device)
#             return loss.sum() / valid_count
#         elif reduction == 'sum':
#             return loss.sum()
#         else:  # 'none'
#             return loss  # shape (N,1,...)






# # ====================================================================== 混合损失函数 ======================================================================
# class HybridLoss(nn.Module):
#     """     self, from_logits=True, normalize=True, clip=None, use_precomputed=True, use_cupy=True
#     三方混合损失函数
#     - L_Tversky: 优化体积重叠（处理类别不平衡）
#     - L_Focal: 优化梯度稳定性（关注困难样本）
#     - L_Boundary: 优化拓扑连贯性（保持边界精确性）
#     """
    
#     def __init__(self, reduction='mean', 
#                  lambda1=1.0, lambda2=1.0, lambda3=1.0,
#                  tversky_alpha=0.01, tversky_beta=3.0, tversky_smooth=1e-5,
#                  focal_alpha=300.0, focal_gamma=2.0, 
#                  boundary_normalize=True, boundary_clip=None, boundary_alpha=300.0, boundary_use_precomputed=True, boundary_use_cupy=True):
#         """ L_Final = λ1·L_Tversky + λ2·L_Focal + λ3·L_Boundary
#         参数:
#             - lambda1，2，3: Tversky Loss的权重，Focal Loss的权重， Boundary Loss的权重
#             - tversky_alpha, tversky_beta, tversky_smooth: 去掉前面的tversky前缀，就等于TverskyLoss()原本的参数，下面同样的逻辑

#         forward参数:
#             - pred: torch.Tensor, shape (N, 1, H, W)
#                   预测概率图 (sigmoid后)
#             - target: torch.Tensor, shape (N, 1, H, W)
#                     ground truth
#             - return_components: bool, 是否返回各个损失分量
#         返回:
#             如果 return_components=False: 返回总损失
#             如果 return_components=True: 返回 (总损失, 字典{各分量})
#         注意：输入，默认是logit(但可调)
#         """
#         super(HybridLoss, self).__init__()
        
#         self.lambda1 = lambda1
#         self.lambda2 = lambda2
#         self.lambda3 = lambda3
        
#         self.tversky_loss = TverskyLoss(alpha=tversky_alpha, beta=tversky_beta, smooth=tversky_smooth, reduction=reduction)
#         self.focal_loss = BinaryFocalLossWithAlpha(alpha_tune=focal_alpha, gamma=focal_gamma, reduction=reduction)
#         self.boundary_loss = BoundaryLoss(normalize=boundary_normalize, clip=boundary_clip, alpha=boundary_alpha, use_precomputed=boundary_use_precomputed, use_cupy=boundary_use_cupy, reduction=reduction)
        
#         print(f"混合损失函数初始化:")
#         print(f"  λ1 (Tversky)  = {lambda1:.2f} | alpha={tversky_alpha}, beta={tversky_beta}")
#         print(f"  λ2 (Focal)    = {lambda2:.2f} | alpha={focal_alpha}, gamma={focal_gamma}")
#         print(f"  λ3 (Boundary) = {lambda3:.2f}")
    
#     def forward(self, pred, target, return_components=False, distance_maps=None):
#         # 计算各个损失分量
#         loss_tversky = self.tversky_loss(pred, target)
#         loss_focal = self.focal_loss(pred, target)
#         loss_boundary = self.boundary_loss(pred, target, distance_maps)
        
#         # 加权求和
#         total_loss = (self.lambda1 * loss_tversky + 
#                       self.lambda2 * loss_focal + 
#                       self.lambda3 * loss_boundary)
        
#         if return_components:
#             components = {
#                 'total': total_loss,
#                 'tversky': loss_tversky,
#                 'focal': loss_focal,
#                 'boundary': loss_boundary
#             }
#             return total_loss, components
#         else:
#             return total_loss


# # ============== 使用示例 ==============
# if __name__ == "__main__":
#     torch.manual_seed(42)
#     np.random.seed(42)
    
#     print("=" * 70)
#     print("三方混合损失函数 (Tversky + Focal + Boundary) 使用示例")
#     print("=" * 70)
    
#     # 创建模拟数据
#     batch_size = 4
#     height, width = 128, 128
    
#     # 模拟预测（经过sigmoid的概率图）
#     pred = torch.rand(batch_size, 1, height, width)
    
#     # 模拟ground truth（二值掩码）
#     target = torch.zeros(batch_size, 1, height, width)
#     for i in range(batch_size):
#         # 创建不同大小的圆形目标
#         center = (height // 2 + np.random.randint(-20, 20), 
#                   width // 2 + np.random.randint(-20, 20))
#         radius = np.random.randint(20, 40)
#         y, x = torch.meshgrid(torch.arange(height), torch.arange(width), indexing='ij')
#         mask = ((x - center[1])**2 + (y - center[0])**2) <= radius**2
#         target[i, 0] = mask.float()
    
#     # ============== 1. 测试各个独立损失 ==============
#     print("\n1. 独立损失函数测试:")
#     print("-" * 70)
    
#     tversky_loss = TverskyLoss(alpha=0.7, beta=0.3)
#     focal_loss = FocalLoss(alpha=0.25, gamma=2.0)
#     boundary_loss = BoundaryLoss()
    
#     loss_t = tversky_loss(pred, target)
#     loss_f = focal_loss(pred, target)
#     loss_b = boundary_loss(pred, target)
    
#     print(f"  Tversky Loss:  {loss_t.item():.4f} (优化体积重叠)")
#     print(f"  Focal Loss:    {loss_f.item():.4f} (优化梯度稳定性)")
#     print(f"  Boundary Loss: {loss_b.item():.4f} (优化拓扑连贯性)")
    
#     # ============== 2. 混合损失函数 ==============
#     print("\n2. 混合损失函数测试:")
#     print("-" * 70)
    
#     # 默认配置：均衡三个损失
#     hybrid_loss = HybridLoss(
#         lambda1=1.0,  # Tversky权重
#         lambda2=1.0,  # Focal权重
#         lambda3=1.0,  # Boundary权重
#         tversky_alpha=0.7, tversky_beta=0.3,
#         focal_alpha=0.25, focal_gamma=2.0
#     )
    
#     total_loss, components = hybrid_loss(pred, target, return_components=True)
    
#     print(f"\n  总损失: {components['total']:.4f}")
#     print(f"    ├─ Tversky:  {components['tversky']:.4f} × {hybrid_loss.lambda1} = {components['tversky'] * hybrid_loss.lambda1:.4f}")
#     print(f"    ├─ Focal:    {components['focal']:.4f} × {hybrid_loss.lambda2} = {components['focal'] * hybrid_loss.lambda2:.4f}")
#     print(f"    └─ Boundary: {components['boundary']:.4f} × {hybrid_loss.lambda3} = {components['boundary'] * hybrid_loss.lambda3:.4f}")
    
#     # ============== 3. 不同权重配置示例 ==============
#     print("\n3. 不同权重配置的效果:")
#     print("-" * 70)
    
#     configs = [
#         {"name": "均衡配置", "lambda1": 1.0, "lambda2": 1.0, "lambda3": 1.0},
#         {"name": "强调边界", "lambda1": 0.5, "lambda2": 0.5, "lambda3": 2.0},
#         {"name": "强调重叠", "lambda1": 2.0, "lambda2": 0.5, "lambda3": 0.5},
#         {"name": "强调困难样本", "lambda1": 0.5, "lambda2": 2.0, "lambda3": 0.5},
#     ]
    
#     for config in configs:
#         loss_fn = HybridLoss(
#             lambda1=config["lambda1"],
#             lambda2=config["lambda2"],
#             lambda3=config["lambda3"],
#             tversky_alpha=0.7, tversky_beta=0.3,
#             focal_alpha=0.25, focal_gamma=2.0
#         )
#         total, comp = loss_fn(pred, target, return_components=True)
#         print(f"\n  {config['name']} (λ1={config['lambda1']}, λ2={config['lambda2']}, λ3={config['lambda3']}):")
#         print(f"    总损失 = {comp['total']:.4f}")
    
#     # ============== 4. 训练循环示例 ==============
#     print("\n4. 完整训练循环示例:")
#     print("-" * 70)
    
#     # 创建简单的U-Net风格模型（简化版）
#     class SimpleUNet(nn.Module):
#         def __init__(self):
#             super().__init__()
#             self.encoder = nn.Sequential(
#                 nn.Conv2d(1, 32, 3, padding=1),
#                 nn.ReLU(),
#                 nn.Conv2d(32, 64, 3, padding=1),
#                 nn.ReLU(),
#             )
#             self.decoder = nn.Sequential(
#                 nn.Conv2d(64, 32, 3, padding=1),
#                 nn.ReLU(),
#                 nn.Conv2d(32, 1, 3, padding=1),
#                 nn.Sigmoid()
#             )
        
#         def forward(self, x):
#             x = self.encoder(x)
#             x = self.decoder(x)
#             return x
    
#     model = SimpleUNet()
#     optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
#     criterion = HybridLoss(lambda1=1.0, lambda2=1.0, lambda3=1.0)
    
#     # 模拟输入
#     input_img = torch.rand(batch_size, 1, height, width)
    
#     print("\n  训练进度:")
#     for epoch in range(5):
#         optimizer.zero_grad()
#         output = model(input_img)
#         loss, comp = criterion(output, target, return_components=True)
#         loss.backward()
#         optimizer.step()
        
#         print(f"    Epoch {epoch+1}: Total={comp['total']:.4f} | "
#               f"T={comp['tversky']:.3f} | F={comp['focal']:.3f} | B={comp['boundary']:.3f}")
    
#     # ============== 5. 参数调优建议 ==============
#     print("\n5. 参数调优建议:")
#     print("-" * 70)
#     print("""
#   【权重参数 λ1, λ2, λ3】
#     • 均衡起点: λ1=λ2=λ3=1.0
#     • 边界不清晰: 增大 λ3 (如 λ3=2.0)
#     • 小目标漏检: 增大 λ1 和 β (如 λ1=2.0, β=0.7)
#     • 困难样本多: 增大 λ2 和 γ (如 λ2=2.0, γ=3.0)
  
#   【Tversky参数 α, β】
#     • 假阳性多(过分割): 增大 α (如 α=0.7, β=0.3)
#     • 假阴性多(欠分割): 增大 β (如 α=0.3, β=0.7)
#     • 小目标/稀有类: α < β
#     • 大目标/常见类: α > β
  
#   【Focal参数 α, γ】
#     • γ=0: 退化为标准BCE
#     • γ=2: 标准配置，适合大多数情况
#     • γ=3~5: 极度关注困难样本
#     • α: 正样本权重，通常设为 0.25
  
#   【应用场景】
#     • 医学图像分割: λ1=1.5, λ2=1.0, λ3=2.0 (强调边界)
#     • 小目标检测: λ1=2.0, λ2=1.5, λ3=1.0 (强调重叠)
#     • 不平衡数据: λ1=1.0, λ2=2.0, λ3=1.0 (强调困难样本)
#     """)
    
#     print("=" * 70)
#     print("✓ 混合损失函数已成功实现并测试完毕！")
#     print("=" * 70)

