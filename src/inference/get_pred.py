"""
get_pred.py - 模型加载与推断模块

负责从 checkpoint 加载训练好的模型,
并对输入特征执行滑窗推断, 产出概率图。

输出概率图形状:
    - 二分类: (D, H, W), 每个体素的正类概率
    - 多分类: (C_out, D, H, W), 每个体素各类的概率 [预留]
"""

import os
import sys
from typing import Optional
import numpy as np
import torch
import tqdm
from utils.network_tools import map_segmentation, map_reconstruction


# =============================================================================
# 1. 模型加载
# =============================================================================
def load_model(
    ckpt_path: str,
    device: torch.device,
    backbone_override: dict = None,
) -> torch.nn.Module:
    """
    从 Lightning checkpoint 加载训练好的 backbone 模型。

    自动从 checkpoint 的 hyper_parameters 中推断 backbone 结构,无需手动指定 in_channels / out_channels。

    Args:
        - ckpt_path:        str,           checkpoint 文件路径 (.ckpt)
        - device:           torch.device,  推断设备
        - backbone_override: dict | None,  若提供则用此配置覆盖 checkpoint 中的 backbone 配置；
                             格式为 Hydra instantiate 所需的字典（含 _target_ 等）。
                             通常不需要，仅在 ckpt 内参数缺失时使用。

    Returns:
        - model: nn.Module, eval 模式的裸 backbone, 位于指定 device
    """
    print(f"[get_pred] 正在加载模型: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if "state_dict" not in ckpt:
        raise RuntimeError(
            f"[get_pred] checkpoint 中未找到 'state_dict'。"
            f"请确认文件是 Lightning .ckpt 格式: {ckpt_path}"
        )

    # ---- 1. 推断 backbone 结构 ----
    if backbone_override is not None:
        # 用户显式覆盖
        from hydra.utils import instantiate
        model = instantiate(backbone_override)
        print(f"[get_pred] 使用用户提供的 backbone 配置")
    elif "hyper_parameters" in ckpt and "backbone" in ckpt["hyper_parameters"]:
        # 从 checkpoint 的 hyper_parameters 中自动推断
        hp = ckpt["hyper_parameters"]
        backbone_cfg = hp["backbone"]
        from hydra.utils import instantiate
        model = instantiate(backbone_cfg)
        print(f"[get_pred] 从 checkpoint hyper_parameters 自动推断 backbone: "
              f"{backbone_cfg.get('_target_', 'unknown')}")
    else:
        # 回退: 尝试硬编码 SimpleUnet (兼容旧 checkpoint)
        print("[get_pred] ⚠️ checkpoint 中无 hyper_parameters.backbone, "
              "尝试使用默认 SimpleUnet")
        from src.model.raunet import SimpleUnet
        # 从 state_dict 推断 in_channels
        state_dict = ckpt["state_dict"]
        backbone_keys = [k for k in state_dict if k.startswith("backbone.")]
        if not backbone_keys:
            raise RuntimeError("[get_pred] state_dict 中无 backbone.* 键")
        # 找第一个卷积层的权重来推断 in_channels
        in_ch = _infer_in_channels_from_state_dict(state_dict)
        model = SimpleUnet(in_channels=in_ch, out_channels=1)

    # ---- 2. 提取并加载 backbone 权重 ----
    state_dict = ckpt["state_dict"]
    backbone_state = {
        k.replace("backbone.", ""): v
        for k, v in state_dict.items()
        if k.startswith("backbone.")
    }
    if not backbone_state:
        # 尝试不带前缀的情况 (直接 state_dict)
        backbone_state = state_dict
        print("[get_pred] ⚠️ 未找到 backbone.* 前缀, 尝试直接加载全部权重")

    # 使用 strict=False 以容忍 Lightning 包装器附带的额外键,（如 criterion.weight、loss.* 等），避免 Unexpected key(s) 报错
    load_result = model.load_state_dict(backbone_state, strict=False)
    if load_result.missing_keys:
        print(f"[get_pred] ⚠️ 缺失的键: {load_result.missing_keys}")
    if load_result.unexpected_keys:
        print(f"[get_pred] ⚠️ 多余的键 (已忽略): {load_result.unexpected_keys}")
    model.to(device)
    model.eval()

    # 打印模型摘要
    in_ch = getattr(model, "in_channels", "?")
    out_ch = getattr(model, "out_channels", "?")
    print(f"[get_pred] 模型加载成功: in_channels={in_ch}, out_channels={out_ch}, device={device}")

    return model

def _infer_in_channels_from_state_dict(state_dict: dict) -> int:
    """
    从 Lightning state_dict 中推断 backbone 输入通道数。
    查找第一个 backbone.*.weight 且形状为 5D (Conv3d) 的参数, 取其 shape[1]。

    Args:
        - state_dict: dict, Lightning checkpoint 的 state_dict

    Returns:
        - in_channels: int, 推断出的输入通道数
    """
    for key in state_dict.keys():
        if key.startswith("backbone.") and key.endswith(".weight") and "conv" in key.lower():
            w = state_dict[key]
            if w.dim() == 5:  # Conv3d: (out_ch, in_ch, kD, kH, kW)
                in_ch = w.shape[1]
                # Check if it's likely the first layer (e.g., in_channels is small, like 1, 49, 50, etc., not 256)
                if in_ch <= 128:
                    print(f"[get_pred] 从 {key} 推断 in_channels={in_ch}")
                    return in_ch
    print("[get_pred] ⚠️ 无法从 state_dict 推断 in_channels, 使用默认值 1")
    return 1




# =============================================================================
# 2. 推断 - 滑窗机制
# =============================================================================
def run_inference(
    model, device, show_progress,
    grid,
    stride, windows_size, batch_size, core_offset,
    blocks_weight=None,  # see me: 将会在这里增加高斯衰减核开关
    **kwargs
) -> np.ndarray:
    """
    Run sliding-window inference on multi-channel features and return probability maps.

    Args:
        - model: nn.Module, backbone in eval mode
        - device: torch.device
        - show_progress: bool, whether to show tqdm
        - grid: np.ndarray, float32, (C, D, H, W), 作为模型输入的特征(50)
        - stride: int, sliding window stride
        - windows_size: int, window edge length
        - batch_size: int, number of windows per batch
        - core_offset: int, crop border thickness
        - blocks_weight: np.ndarray | None, optional fusion weights (3D or 5D): (D,H,W) OR (N,C,D,H,W)

    Returns:
        - pred_prob: np.ndarray, float32
            - binary (out_channels=1): (D, H, W)
            - multi-class: (C_out, D, H, W)
    """
    # 丢掉任何外部传递下来多余的参数 kwargs, e.g. threshold
    kwargs.pop("threshold", None)
    C, D, H, W = grid.shape
    spatial_shape = np.array([D, H, W])
    # out_channels: int, 提取自模型中的目标通道数设定，找不到时默认设1。
    out_channels = getattr(model, "out_channels", None)

    # Use total volume threshold instead of per-axis comparison
    total_voxels = int(np.prod(spatial_shape))
    volume_threshold = 2 * (windows_size ** 3)
    use_sliding_window = (total_voxels > volume_threshold)

    if use_sliding_window:
        pad_d = max(0, windows_size - D)
        pad_h = max(0, windows_size - H)
        pad_w = max(0, windows_size - W)
        if pad_d > 0 or pad_h > 0 or pad_w > 0:
            grid = np.pad(grid, ((0, 0), (0, pad_d), (0, pad_h), (0, pad_w)), mode='constant')

        _, Dp, Hp, Wp = grid.shape
        spatial_shape_pad = (Dp, Hp, Wp)

        # 1. 将体积分割为多个滑窗块
        multi_ch_blocks, coords = map_segmentation(
            torch.from_numpy(grid),
            windows_size=windows_size,
            stride=stride,
        )
        n_blocks = len(multi_ch_blocks)

        # 2. 批量推断
        # pbar: tqdm.tqdm, 进度条对象，用于实时显示滑窗推断进度
        pbar = tqdm.tqdm(
            desc="  Sliding-window inference",
            total=n_blocks,
            file=sys.stdout,
            position=0,
            leave=False,
            disable=not show_progress,
        )
        
        # grid_batches: list[list[int]], 嵌套列表，每个子列表包含当前 batch 要处理的滑窗索引序列
        grid_batches = _get_batch_slices(n_blocks, batch_size)
        out_segmentation = torch.zeros(
            (n_blocks, out_channels, windows_size, windows_size, windows_size),
            device="cpu",
        )
        with torch.no_grad():
            for batch_indices in grid_batches:
                batch_input = torch.stack([multi_ch_blocks[i] for i in batch_indices], dim=0).to(device)
                batch_output = torch.sigmoid(model(batch_input)).cpu()
                out_segmentation[batch_indices] = batch_output
                pbar.update(len(batch_indices))
        pbar.close()

        # 3. Reconstruct
        blocks_np = out_segmentation.numpy()  # (N_blocks, out_channels, Dp, Hp, Wp)
        pred_prob = map_reconstruction(       # (out_channels, Dp, Hp, Wp)
            blocks=blocks_np,
            image_shape=(out_channels,) + spatial_shape_pad,
            coords=coords,
            windows_size=windows_size,
            core_offset=core_offset,
            blocks_weight=blocks_weight,
        )
        if out_channels == 1:
            pred_prob = pred_prob[0]

        # Crop padding
        if pad_d > 0 or pad_h > 0 or pad_w > 0:
            pred_prob = pred_prob[..., :D, :H, :W]

    else:
        pred_prob = _full_volume_inference(
            model, grid, device,
            out_channels=out_channels,
            show_progress=show_progress,
        )

    return pred_prob





def _full_volume_inference(
    model, grid, device, out_channels, show_progress
) -> np.ndarray:
    """
    全图推断: 当三维体积足够小时, 不需要滑窗。

    Args / Returns: 同 run_inference（内部函数）
    """
    C, D, H, W = grid.shape
    # 绝大多数 3D UNet (下采样3~4层) 要求输入尺寸能够被 16 或 32 整除, 此处将空间维强行 padding 到 32 的整数倍，推理完毕后再截取
    pad_d = (32 - D % 32) % 32
    pad_h = (32 - H % 32) % 32
    pad_w = (32 - W % 32) % 32
    if pad_d > 0 or pad_h > 0 or pad_w > 0:
        grid = np.pad(grid, ((0, 0), (0, pad_d), (0, pad_h), (0, pad_w)), mode='constant')
        
    with torch.no_grad():
        full_input = torch.from_numpy(grid)[None].to(device)   # (1, C, D_pad, H_pad, W_pad)
        output = torch.sigmoid(model(full_input))              # (1, C_out, D_pad, H_pad, W_pad)
        if out_channels == 1:
            pred_prob = output[0, 0].detach().cpu().numpy()    # (D_pad, H_pad, W_pad)
        else:
            pred_prob = output[0].detach().cpu().numpy()       # (C_out, D_pad, H_pad, W_pad)
            
    # 去除由于 pad 到 32 整数倍引入的空边，将其还原为真正的 D, H, W
    if pad_d > 0 or pad_h > 0 or pad_w > 0:
        pred_prob = pred_prob[..., :D, :H, :W]
        
    if show_progress:
        print("  [get_pred] 全图推断完成")
    return pred_prob



# =============================================================================
# 工具函数
# =============================================================================
def _get_batch_slices(total: int, batch_size: int) -> list:
    """
    将 [0, total) 切分为 batch_size 大小的索引切片列表。

    Args:
        - total:      int, 总数量
        - batch_size: int, 每批数量

    Returns:
        - slices: list[ list[int] ], 索引列表的列表
    """
    slices = []
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        slices.append(list(range(start, end)))
    return slices

