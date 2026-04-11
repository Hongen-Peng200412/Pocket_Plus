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



# =============================================================================
# 1. 模型加载
# =============================================================================
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

# 总函数
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

    # ---- 0. 注入训练时的代码快照 (如果存在) ----
    # 当 checkpoint 所在 run 目录下存在 src_snapshot 时，模型类必须从快照中加载，以保证 state_dict key 与模型结构一致。
    ckpt_file = Path(ckpt_path).resolve()
    if len(ckpt_file.parents) >= 2:
        run_dir = ckpt_file.parents[1]
        snapshot_dir = run_dir / "src_snapshot"
        snapshot_src_dir = snapshot_dir / "src"
        if snapshot_src_dir.exists():
            # str, snapshot 中的 src/ 目录绝对路径
            snapshot_src_str = str(snapshot_src_dir.resolve())

            # list[str], 需要从 sys.modules 中移除的 src.model.* 缓存条目
            stale_module_keys = [
                mod_name for mod_name in sys.modules
                if mod_name == "src.model" or mod_name.startswith("src.model.")
            ]
            for mod_name in stale_module_keys:
                del sys.modules[mod_name]
            # 将 snapshot/src 插入 src 包的 __path__ 最前面, 使后续 import src.model.* 优先从 snapshot 解析
            import src as _src_pkg
            if snapshot_src_str not in _src_pkg.__path__:
                _src_pkg.__path__.insert(0, snapshot_src_str)

            print(f"[get_pred] 已注入代码快照: {snapshot_dir}")
            if stale_module_keys:
                print(f"[get_pred] 已清除 {len(stale_module_keys)} 个 src.model.* 缓存模块")

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


    # ---- 3. 提取并加载 backbone 权重 ----
    # dict[str, torch.Tensor], 仅剥掉开头 "backbone." 前缀后的权重字典
    state_dict = ckpt["state_dict"]
    backbone_state = {
        k[len("backbone."):]: v
        for k, v in state_dict.items()
        if k.startswith("backbone.")
    }
    if not backbone_state:
        backbone_state = state_dict
        print("[get_pred] ⚠️ 未找到 backbone.* 前缀, 尝试直接加载全部权重")

    # strict=True: 任何 key 不匹配立即报错, 避免静默加载随机权重
    model.load_state_dict(backbone_state, strict=True)

    model.to(device)
    model.eval()
    print(f"[get_pred] 模型加载成功, device={device}")
    return model




"""
=============================================================================
1.5 训练配置加载
=============================================================================
推理管线的参数按来源分为三类:

A. 训练契约类 —— 应自动从训练 config 的 dataset.* 继承
   ┌────────────────────┬──────────────────────────┐
   │ 参数               │ 训练 config 路径          │
   ├────────────────────┼──────────────────────────┤
   │ data_folder_names  │ dataset.data_folder_names │
   │ class_mapping      │ dataset.class_mapping     │
   │ atom_buffer_radius │ dataset.atom_buffer_radius│
   │ valid_crop_margin  │ dataset.valid_crop_margin │
   │ emdb_z_score       │ dataset.emdb_z_score      │
   │ (未来新增参数)      │ dataset.*                 │
   └────────────────────┴──────────────────────────┘
   → 由 load_training_config() 自动提取, 推理管线直接使用。
   → 如需覆盖, 可在推理 YAML 中显式设置同名字段 (_get_cfg 优先级最高)。

B. 推理专属类 —— 只存在于推理 YAML
   threshold, dist_threshold, core_decay_mode, core_offset,
   merge_mode, semantic_segment_method, dbscan_eps, dbscan_min_samples,
   box_spatial_weight_sigma_ratio, stride, windows_size, batch_size,
   output_dir, show_progress, error_dir

C. 数据处理类 —— 推理 YAML 中有, 训练 config 中不一定有
   target_voxel_size, compute_density, select_first_model
"""

def load_training_config(ckpt_path: str) -> dict[str, Any]:
    """
    从 checkpoint 所在的训练 run 目录中提取完整训练配置。

    输入参数:
        - ckpt_path: str, Lightning checkpoint 路径

    输出:
        - train_cfg: dict[str, Any], 训练时的完整配置(顶层 dict, 含 model/dataset/train 等子键)

    异常:
        - FileNotFoundError: 若无法从 checkpoint 路径推断 run 目录, 或 run 目录下无 config.yaml
    """
    ckpt_file = Path(ckpt_path).resolve()
    if len(ckpt_file.parents) < 2:
        raise FileNotFoundError(
            f"无法从 checkpoint 路径推断 run 目录: {ckpt_path}"
        )
    run_dir = ckpt_file.parents[1]
    for config_path in (run_dir / "config.yaml", run_dir / ".hydra" / "config.yaml"):
        if config_path.exists():
            loaded = OmegaConf.load(config_path)
            cfg_dict = OmegaConf.to_container(loaded, resolve=True)
            print(f"[get_pred] 已加载训练配置: {config_path}")
            return cfg_dict
    raise FileNotFoundError(
        f"在 run 目录 {run_dir} 下未找到 config.yaml 或 .hydra/config.yaml"
    )









# =============================================================================
# 2. 点云级推断
# =============================================================================
def run_point_inference(
    model: torch.nn.Module,
    device: str | torch.device,
    batch_dict: dict[str, Any],
) -> dict[str, Any]:
    """
    对一个已组装好的 batch dict 执行模型前向, 返回原始 atom logits。

    输入参数:
        - model: nn.Module, eval 模式的 VolumePointStage1Model
        - device: str | torch.device, 推断设备
        - batch_dict: dict[str, Any], 由 box_point_collate() 产出的 batch dict, 所有 tensor 已在 device 上

    输出:
        - outputs: dict[str, Any], 模型前向输出, 至少包含:
            - "atom_logits": torch.Tensor, (sumN_final, C_logit), 原始 atom 分类 logits, 未经 sigmoid
            - 以及与最终原子视图对齐的辅助字段(global_atom_indices / atom_counts / atom_coord_local_voxel / atom_is_in_core_box 等)
    """
    with torch.no_grad():
        # dict[str, Any], 模型前向输出
        outputs = model(batch_dict)

    return outputs





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
