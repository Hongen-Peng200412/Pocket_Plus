"""
get_pred.py — 模型加载与点云级推断模块

负责:
    1. 从 Lightning checkpoint 加载完整 VolumePointStage1Model
    2. 对已组装好的 batch dict 执行模型前向, 返回 atom-level 原始 logits

旧版体素推断逻辑已迁移至 src/inference/legacy/get_pred_voxel.py
"""

import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _extract_backbone_cfg(config_obj: Any) -> dict[str, Any] | None:
    """
    从用户传入配置、checkpoint 超参数或训练配置快照中提取 VolumePointStage1Model 的 Hydra 配置。

    输入参数:
        - config_obj: Any, 可能是 DictConfig、dict、wrapper 级配置或 backbone 级配置

    输出:
        - backbone_cfg: dict[str, Any] | None, 可直接传给 hydra.instantiate() 的 backbone 配置
    """
    if config_obj is None:
        return None

    if OmegaConf.is_config(config_obj):
        config_obj = OmegaConf.to_container(config_obj, resolve=True)
    if not isinstance(config_obj, dict):
        return None

    # 直接就是 backbone 配置
    target_name = config_obj.get("_target_")
    if target_name is not None and "model" not in config_obj and "backbone" not in config_obj:
        return config_obj

    # 训练快照通常是全局配置，backbone 位于 cfg.model.backbone
    model_cfg = config_obj.get("model")
    if isinstance(model_cfg, dict):
        model_backbone_cfg = _extract_backbone_cfg(model_cfg)
        if model_backbone_cfg is not None:
            return model_backbone_cfg

    # wrapper 级配置通常是 cfg.model，backbone 位于其下一级
    nested_backbone_cfg = config_obj.get("backbone")
    if isinstance(nested_backbone_cfg, dict):
        direct_backbone_cfg = _extract_backbone_cfg(nested_backbone_cfg)
        if direct_backbone_cfg is not None:
            return direct_backbone_cfg

    return None


def _load_backbone_cfg_from_training_snapshot(ckpt_path: str) -> tuple[dict[str, Any] | None, str | None]:
    """
    从 checkpoint 同一 run 目录下保存的训练配置快照中恢复 backbone 配置。

    输入参数:
        - ckpt_path: str, Lightning checkpoint 路径

    输出:
        - backbone_cfg: dict[str, Any] | None, 恢复出的 backbone 配置
        - config_path: str | None, 实际命中的配置文件路径
    """
    ckpt_file = Path(ckpt_path).resolve()
    if len(ckpt_file.parents) < 2:
        return None, None

    run_dir = ckpt_file.parents[1]
    candidate_paths = (
        run_dir / "config.yaml",
        run_dir / ".hydra" / "config.yaml",
    )
    for config_path in candidate_paths:
        if not config_path.exists():
            continue
        loaded_cfg = OmegaConf.load(config_path)
        backbone_cfg = _extract_backbone_cfg(loaded_cfg)
        if backbone_cfg is not None:
            return backbone_cfg, str(config_path)

    return None, None


# =============================================================================
# 1. 模型加载
# =============================================================================
def load_model(
    ckpt_path: str,
    device: str | torch.device,
    backbone_override: dict | None = None,
) -> torch.nn.Module:
    """
    从 Lightning checkpoint 加载完整 VolumePointStage1Model。

    输入参数:
        - ckpt_path: str, checkpoint 文件路径 (.ckpt)
        - device: str | torch.device, 推断设备
        - backbone_override: dict | None, 若提供则用此配置覆盖 checkpoint 中的 backbone 配置; 格式为 Hydra instantiate 所需的字典（含 _target_ 等）

    输出:
        - model: nn.Module, eval 模式的完整 VolumePointStage1Model, 位于指定 device
    """
    print(f"[get_pred] 正在加载模型: {ckpt_path}")

    # dict, Lightning checkpoint 完整内容
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if "state_dict" not in ckpt:
        raise RuntimeError(
            f"[get_pred] checkpoint 中未找到 'state_dict'。"
            f"请确认文件是 Lightning .ckpt 格式: {ckpt_path}"
        )

    # ---- 1. 构建模型 ----
    if backbone_override is not None:
        backbone_cfg = _extract_backbone_cfg(backbone_override)
        if backbone_cfg is None:
            raise ValueError(
                "[get_pred] backbone_override 必须是 VolumePointStage1Model 配置，"
                "或包含 model.backbone / backbone 的完整训练配置。"
            )
        from hydra.utils import instantiate
        model = instantiate(backbone_cfg)
        print("[get_pred] 使用用户提供的 backbone 配置")
    elif "hyper_parameters" in ckpt and "backbone" in ckpt["hyper_parameters"]:
        hp = ckpt["hyper_parameters"]
        backbone_cfg = hp["backbone"]
        from hydra.utils import instantiate
        model = instantiate(backbone_cfg)
        target_name = backbone_cfg.get("_target_", "unknown") if isinstance(backbone_cfg, dict) else "unknown"
        print(f"[get_pred] 从 checkpoint hyper_parameters 自动推断 backbone: {target_name}")
    else:
        backbone_cfg, config_path = _load_backbone_cfg_from_training_snapshot(ckpt_path)
        if backbone_cfg is not None:
            from hydra.utils import instantiate
            model = instantiate(backbone_cfg)
            target_name = backbone_cfg.get("_target_", "unknown")
            print(f"[get_pred] 从训练配置快照恢复 backbone: {target_name}")
            print(f"[get_pred] 配置来源: {config_path}")
        else:
            raise RuntimeError(
                "[get_pred] checkpoint 中无 hyper_parameters.backbone，且未在 checkpoint 邻近目录找到可用训练配置。"
                "请提供 backbone_override，或确认 run_dir/config.yaml 存在且包含 model.backbone。"
            )

    # ---- 2. 推断 in_channels 并设置 ----
    in_channels = _infer_in_channels(ckpt)
    if hasattr(model, "set_input_channels"):
        model.set_input_channels(in_channels)
        print(f"[get_pred] 设置 model.set_input_channels({in_channels})")

    # ---- 3. 提取并加载 backbone 权重 ----
    # dict[str, torch.Tensor], 去掉 "backbone." 前缀后的权重字典
    state_dict = ckpt["state_dict"]
    backbone_state = {
        k.replace("backbone.", ""): v
        for k, v in state_dict.items()
        if k.startswith("backbone.")
    }
    if not backbone_state:
        backbone_state = state_dict
        print("[get_pred] ⚠️ 未找到 backbone.* 前缀, 尝试直接加载全部权重")

    # strict=False 以容忍 Lightning 包装器附带的额外键
    load_result = model.load_state_dict(backbone_state, strict=False)
    if load_result.missing_keys:
        print(f"[get_pred] ⚠️ 缺失的键: {load_result.missing_keys}")
    if load_result.unexpected_keys:
        print(f"[get_pred] ⚠️ 多余的键 (已忽略): {load_result.unexpected_keys}")

    model.to(device)
    model.eval()
    print(f"[get_pred] 模型加载成功, device={device}")
    return model


def _infer_in_channels(ckpt: dict) -> int:
    """
    从 Lightning checkpoint 推断 backbone 输入通道数。

    优先级:
        1. hyper_parameters 中显式记录的 in_channels
        2. 从 state_dict 第一个 Conv3d 权重推断

    输入参数:
        - ckpt: dict, Lightning checkpoint

    输出:
        - in_channels: int, 推断出的输入通道数
    """
    # 尝试从 hyper_parameters 直接获取
    hp = ckpt.get("hyper_parameters", {})
    if "in_channels" in hp:
        return int(hp["in_channels"])

    # 从 state_dict 推断
    state_dict = ckpt.get("state_dict", {})
    for key in state_dict.keys():
        if key.startswith("backbone.") and key.endswith(".weight") and "conv" in key.lower():
            w = state_dict[key]
            if w.dim() == 5:  # Conv3d: (out_ch, in_ch, kD, kH, kW)
                in_ch = w.shape[1]
                if in_ch <= 128:
                    print(f"[get_pred] 从 {key} 推断 in_channels={in_ch}")
                    return in_ch

    print("[get_pred] ⚠️ 无法推断 in_channels, 使用默认值 1")
    return 1





# =============================================================================
# 2. 点云级推断
# =============================================================================
def run_point_inference(
    model: torch.nn.Module,
    device: str | torch.device,
    batch_dict: dict[str, Any],
) -> torch.Tensor:
    """
    对一个已组装好的 batch dict 执行模型前向, 返回原始 atom logits。

    输入参数:
        - model: nn.Module, eval 模式的 VolumePointStage1Model
        - device: str | torch.device, 推断设备
        - batch_dict: dict[str, Any], 由 box_point_collate() 产出的 batch dict, 所有 tensor 已在 device 上

    输出:
        - atom_logits: torch.Tensor, (sumN, C_logit), 原始 atom 分类 logits, 未经 sigmoid
    """
    with torch.no_grad():
        # dict[str, Any], 模型前向输出
        outputs = model(batch_dict)

    # torch.Tensor, (sumN, C_logit), atom 分类 logits
    atom_logits = outputs["atom_logits"]
    return atom_logits





# =============================================================================
# 工具函数
# =============================================================================
def move_batch_to_device(
    batch_dict: dict[str, Any],
    device: str | torch.device,
) -> dict[str, Any]:
    """
    将 batch dict 中的所有 torch.Tensor 移至指定设备, 非 tensor 字段保持不变。

    输入参数:
        - batch_dict: dict[str, Any], box_point_collate 输出的 batch dict
        - device: str | torch.device, 目标设备

    输出:
        - batch_dict: dict[str, Any], tensor 已移至 device 的 batch dict (原地修改并返回)
    """
    for key, val in batch_dict.items():
        if isinstance(val, torch.Tensor):
            batch_dict[key] = val.to(device)
    return batch_dict
