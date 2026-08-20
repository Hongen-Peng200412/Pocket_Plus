# -*- coding: utf-8 -*-
"""从训练 run 的 resolved config 和代码快照恢复完整 Stage1 wrapper.

主要入口 :func:`load_stage1_wrapper` 只恢复模型和 checkpoint 生命周期. Dataset
实例由 `cli.py` 在模型代码来源已经确定后直接构造.
"""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any


# ================================================================================================


def load_stage1_wrapper(
    checkpoint_path: str | Path,
    resolved_config_path: str | Path,
    map_location: str,
    model_code_source: str,
) -> tuple[Any, Any]:
    """严格恢复一个可执行 voxel-only 和完整 centered 前向的 wrapper.

    输入参数:
        - checkpoint_path: str | Path, Lightning 完整 checkpoint, 必须包含 `state_dict`.
        - resolved_config_path: str | Path, 训练 run 的显式 resolved config 路径.
        - map_location: str, `torch.load` 的设备位置, 正式加载通常为 `cpu`.
        - model_code_source: str, ``training_snapshot`` 使用 run 内代码快照; ``current_workspace`` 使用当前工作区模型代码.

    返回值:
        - wrapper: torch.nn.Module, strict state dict, `on_load_checkpoint` 和 `eval()` 已完成.
        - resolved_cfg: OmegaConf DictConfig, 已解析插值的训练最终配置; CLI 用其中 dataset 配置构造完整图请求环境.
    """

    checkpoint = Path(checkpoint_path).resolve()
    run_directory = (
        checkpoint.parent.parent
        if checkpoint.parent.name == "checkpoints"
        else checkpoint.parent
    )
    snapshot_source = run_directory / "src_snapshot" / "src"
    if model_code_source == "training_snapshot":
        source_path = snapshot_source.resolve()
        if not (source_path / "__init__.py").is_file():
            raise FileNotFoundError(f"{source_path}: 缺少训练代码快照.")
    elif model_code_source == "current_workspace":
        source_path = None
    else:
        raise ValueError(f"未知 model_code_source: {model_code_source}.")

    config_path = Path(resolved_config_path)
    original_src_path: tuple[str, ...] | None = None
    if source_path is not None:
        # 只在模型恢复期间把 `src` 指向训练快照; 当前 V3 Dataset 稍后由 CLI 导入.
        import src

        original_src_path = tuple(str(value) for value in src.__path__)
        src.__path__[:] = [str(source_path)]
        importlib.invalidate_caches()
    try:
        import hydra
        import torch
        from omegaconf import OmegaConf

        resolved_cfg = OmegaConf.load(config_path)
        OmegaConf.resolve(resolved_cfg)
        wrapper = hydra.utils.instantiate(
            resolved_cfg.model,
            optimizer=resolved_cfg.train.optimizer,
            scheduler=resolved_cfg.train.scheduler,
            compile=False,
        )
        checkpoint_payload = torch.load(
            checkpoint,
            map_location=map_location,
            weights_only=False,
        )
        wrapper.load_state_dict(checkpoint_payload["state_dict"], strict=True)
        wrapper.on_load_checkpoint(checkpoint_payload)
        wrapper.eval()
    finally:
        if original_src_path is not None:
            # 无论 checkpoint 恢复是否成功都恢复当前工作区的 `src` 搜索路径.
            src.__path__[:] = list(original_src_path)
            importlib.invalidate_caches()
    return wrapper, resolved_cfg
