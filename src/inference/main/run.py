from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[3]
project_root = str(PROJECT_ROOT)
if project_root in sys.path:
    sys.path.remove(project_root)
sys.path.insert(0, project_root)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(line_buffering=True)

from src.inference.get_pred import load_model, load_training_config
from src.inference.main.voxel_pipeline import run_voxel_batch, run_voxel_param_search, run_voxel_single


def _get_cfg(
    cfg_dict: dict[str, Any],
    key: str,
    required: bool = True,
) -> Any:
    """
    从推理配置中读取参数, 推理 YAML 优先于训练 dataset 配置。

    输入参数:
        - cfg_dict: dict[str, Any], Hydra 配置转成的普通字典
        - key: str, 参数名
        - required: bool, 是否必须存在

    输出:
        - value: Any, 参数值; optional 且缺失时为 None
    """
    if key in cfg_dict and cfg_dict[key] is not None and cfg_dict[key] != "???":
        return cfg_dict[key]
    train_cfg = cfg_dict.get("_train_dataset_cfg", {})
    if key in train_cfg and train_cfg[key] is not None:
        return train_cfg[key]
    if required:
        raise KeyError(
            f"参数 '{key}' 既未在推理配置中设置, 也未在训练 dataset 配置中找到。"
        )
    return None


def _get_config_name() -> str | None:
    """
    从命令行解析 --config 并从 sys.argv 移除, 避免 Hydra 报未知参数。

    输出:
        - config_name: str | None, Hydra 配置文件名(不含 .yaml)
    """
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", type=str, help="Hydra config file name (without .yaml)")
    args, _ = parser.parse_known_args()

    cleaned: list[str] = []
    skip_next = False
    for arg in sys.argv:
        if skip_next:
            skip_next = False
            continue
        if arg == "--config":
            skip_next = True
            continue
        if arg.startswith("--config="):
            continue
        cleaned.append(arg)
    sys.argv = cleaned
    return args.config


_CONFIG_NAME = _get_config_name()


@hydra.main(version_base="1.3", config_path="../../../configs/infer_or_eval", config_name=_CONFIG_NAME)
def main(cfg: DictConfig) -> None:
    """
    voxel-only ligand 推理统一入口。

    输入参数:
        - cfg: DictConfig, Hydra 加载的 voxel_single/voxel_batch/voxel_param_search 配置

    输出:
        - None, 结果写入配置指定的 output_root/cache_root
    """
    # dict[str, Any], Hydra 配置的普通字典形式
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(cfg_dict, dict):
        raise TypeError(f"Hydra 配置必须解析为 dict, 实际为 {type(cfg_dict)}")

    ckpt_path = _get_cfg(cfg_dict, "ckpt_path", required=True)
    mode = str(_get_cfg(cfg_dict, "mode", required=True))
    output_root = str(_get_cfg(cfg_dict, "output_root", required=True))
    os.makedirs(output_root, exist_ok=True)

    device_value = _get_cfg(cfg_dict, "device", required=False)
    device_str = str(device_value) if device_value is not None else ("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)

    print("=" * 72)
    print("  [inference/run] voxel-only ligand 推理")
    print(f"  mode: {mode} | config: {_CONFIG_NAME}.yaml")
    print(f"  device: {device_str} | checkpoint: {ckpt_path}")
    print("=" * 72)

    backbone_override = _get_cfg(cfg_dict, "backbone_override", required=False)
    model = load_model(str(ckpt_path), device, backbone_override=backbone_override)

    train_cfg = load_training_config(str(ckpt_path))
    cfg_dict["_train_dataset_cfg"] = train_cfg["dataset"]

    if mode == "voxel_single":
        run_voxel_single(cfg_dict, model, device)
    elif mode == "voxel_batch":
        run_voxel_batch(cfg_dict, model, device)
    elif mode == "voxel_param_search":
        run_voxel_param_search(cfg_dict, model, device)
    else:
        raise ValueError(
            f"未知 mode: {mode}; voxel-only 入口仅支持 voxel_single / voxel_batch / voxel_param_search"
        )


if __name__ == "__main__":
    main()
