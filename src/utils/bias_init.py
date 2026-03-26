"""
=============================================================================
分类头偏置初始化工具 (Classification Head Bias Initialization)
=============================================================================
针对极度不均衡的分类/分割任务，在训练开始时初始化最后一层的偏置，
使模型的初始预测分布接近数据的真实先验分布。

理论依据：
  - Lin et al., "Focal Loss for Dense Object Detection" (ICCV 2017):
    提出用 b = -log((1-π)/π) 初始化二分类分类头偏置，其中 π 是前景先验概率。
    论文指出结果对 π 的精确值不敏感，但有此初始化可显著提升训练稳定性。
  - 多分类推广（softmax 头）：b_k = log(p_k)，使 softmax(b) = p_k。

适用范围：
  - 任何以 Conv2d/Conv3d/Linear 结尾的分类头（UNet, CNN, Transformer 等）
  - 二分类（sigmoid 输出，单通道）或多分类（softmax 输出，C 通道）
"""

import math
import logging
from typing import Union, List, Optional, Sequence, Tuple

import torch
from torch import nn

logger = logging.getLogger(__name__)


# ========================== 偏置计算 ==========================

def compute_bias_binary(pi: float) -> float:
    """
    计算二分类（sigmoid 输出）的初始偏置。

    公式: b = -log((1 - π) / π)
    效果: sigmoid(b) = π，即初始时模型预测正类的概率为 π。

    Args:
        pi: 正类（前景）的先验概率。典型值：0.01（RetinaNet 默认）, 或根据数据集中前景体素占比设定。

    Returns:
        float: 偏置值b
    """
    if not 0 < pi < 1:
        raise ValueError(f"pi must be in (0, 1), got {pi}")
    return -math.log((1 - pi) / pi)


def compute_bias_multiclass(class_ratios: Sequence[float]) -> List[float]:
    """
    计算多分类（softmax 输出）的 per-class 初始偏置。

    公式: b_k = log(p_k)，其中 p_k = ratio_k / Σ(ratios)。
    效果: softmax([b_0, ..., b_{C-1}]) = [p_0, ..., p_{C-1}]。

    Args:
        class_ratios: 各类别的样本比例（不需要归一化）。
            例如 [1, 10, 100, 111000] 表示 A:B:C:BG = 1:10:100:111000, 顺序应与模型输出通道顺序一致。

    Returns:
        List[float]: 长度为 C 的偏置列表。

    Examples:
        >>> compute_bias_multiclass([1, 10, 100, 111000])
        # ≈ [-11.62, -9.32, -7.01, -0.001]
    """
    if len(class_ratios) < 2:
        raise ValueError(f"Need at least 2 classes, got {len(class_ratios)}")
    if any(r <= 0 for r in class_ratios):
        raise ValueError(f"All ratios must be positive, got {class_ratios}")

    total = sum(class_ratios)
    priors = [r / total for r in class_ratios]
    biases = [math.log(p) for p in priors]
    return biases


# ========================== 模型初始化 ==========================

def _find_last_conv_or_linear(model: nn.Module) -> Optional[Tuple[str, nn.Module]]:
    """
    通过反向遍历 model.named_modules()，找到最后一个 Conv1d/Conv2d/Conv3d/Linear 层。
    这是一种通用的启发式方法，适用于大多数 UNet/CNN/分类器。
    """
    last = None
    for name, m in model.named_modules():
        if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear)):
            last = (name, m)
    return last


def _resolve_named_layer(model: nn.Module, layer_name: str) -> Optional[Tuple[str, nn.Module]]:
    """
    Resolve layer names robustly for wrapped models (e.g. torch.compile/Distributed wrappers).
    """
    named = [(n, m) for n, m in model.named_modules() if n]

    for name, module in named:
        if name == layer_name:
            return name, module

    suffix_matches = [(n, m) for n, m in named if n.split(".")[-1] == layer_name]
    if len(suffix_matches) == 1:
        return suffix_matches[0]
    if len(suffix_matches) > 1:
        raise ValueError(
            f"Layer name '{layer_name}' is ambiguous after wrapper-prefix matching. "
            f"Matched: {[n for n, _ in suffix_matches]}"
        )
    return None


def init_classification_head_bias(
    model: nn.Module,
    class_priors: Union[float, List[float]],
    layer_name: Optional[str] = None,
) -> str:
    """
    初始化模型分类头（最后一层）的偏置，使初始预测接近数据先验分布。

    支持：
      - 二分类 sigmoid 头（out_channels=1）: class_priors 传 float π
      - 多分类 softmax 头（out_channels=C）: class_priors 传 list [r_0, ..., r_{C-1}]

    Args:
        - model: 任意 nn.Module（UNet, CNN, Transformer 等）。
        - class_priors: 
            - float: 二分类正类先验概率 π ∈ (0,1)。
            - list: 多分类各类比例（不需归一化），长度应等于输出通道数。
        - layer_name:
            - 如果指定，则查找 model 中名为 layer_name 的子模块作为分类头。
            - 如果为 None，则自动查找最后一个 Conv/Linear 层。
    
    Returns:
        str: 被初始化的层的名称（用于日志/debug）。

    Raises:
        ValueError: 如果找不到目标层，或通道数量不匹配。
    """
    # 1. 找到目标层
    if layer_name is not None:
        target = _resolve_named_layer(model, layer_name)
        if target is None:
            raise ValueError(
                f"Cannot find layer named '{layer_name}' in model. "
                "Tried exact and wrapped-prefix-aware matching. "
                f"Available: {[n for n, _ in model.named_modules() if n]}"
            )
    else:
        target = _find_last_conv_or_linear(model)
        if target is None:
            raise ValueError(
                "Cannot auto-detect classification head: "
                "no Conv or Linear layer found in model."
            )
    name, layer = target


    # 2. 确保该层有 bias
    if layer.bias is None:
        raise ValueError(
            f"Layer '{name}' ({type(layer).__name__}) has no bias parameter. "
            f"Set bias=True in the layer constructor to enable bias initialization."
        )

    # 3. 确定输出通道数
    if isinstance(layer, nn.Linear):
        out_channels = layer.out_features
    else:
        out_channels = layer.out_channels


    # 4. 计算偏置并赋值
    if isinstance(class_priors, (int, float)):
        # 二分类 sigmoid 头
        pi = float(class_priors)
        if out_channels != 1:
            raise ValueError(
                f"Binary prior (float pi={pi}) requires out_channels=1, "
                f"but layer '{name}' has out_channels={out_channels}. "
                f"For multi-class, pass a list of ratios instead."
            )
        bias_val = compute_bias_binary(pi)
        with torch.no_grad():
            layer.bias.fill_(bias_val)
        logger.info(
            f"[BiasInit] Layer '{name}': binary bias = {bias_val:.4f} "
            f"(π = {pi}, initial P(pos) ≈ {pi:.6f})"
        )

    elif isinstance(class_priors, (list, tuple)):
        # 多分类 softmax 头
        if len(class_priors) != out_channels:
            raise ValueError(
                f"class_priors has {len(class_priors)} elements, "
                f"but layer '{name}' has out_channels={out_channels}."
            )
        biases = compute_bias_multiclass(class_priors)
        with torch.no_grad():
            layer.bias.copy_(torch.tensor(biases, dtype=layer.bias.dtype))
        total = sum(class_priors)
        priors_str = ", ".join(
            f"c{i}: {r/total:.6f}" for i, r in enumerate(class_priors)
        )
        logger.info(
            f"[BiasInit] Layer '{name}': multiclass bias initialized. "
            f"Prior probs: [{priors_str}]"
        )
    else:
        raise TypeError(
            f"class_priors must be float (binary) or list (multiclass), "
            f"got {type(class_priors)}"
        )

    return name
