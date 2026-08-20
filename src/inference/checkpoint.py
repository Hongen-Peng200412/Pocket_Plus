# -*- coding: utf-8 -*-
"""从训练 run 的 resolved config 和代码快照恢复完整 Stage1 wrapper.

主要入口 :func:`load_stage1_wrapper` 只恢复模型和 checkpoint 生命周期. Dataset
实例由 `pipeline.py` 在模型代码来源已经确定后直接构造.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path


_ACTIVE_SNAPSHOT_SOURCE: Path | None = None


# ================================================================================================


def load_stage1_wrapper(
    checkpoint_path: str | Path,
    resolved_config_path: str | Path | None,
    map_location: str,
    allow_current_workspace_code: bool,
):
    """严格恢复一个可执行 voxel-only 和完整 centered 前向的 wrapper.

    输入参数:
        - checkpoint_path: str | Path, Lightning 完整 checkpoint, 必须包含 `state_dict`.
        - resolved_config_path: str | Path | None, 显式配置路径; None 时训练 run 内必须恰有 `config.yaml` 或 `resolved_config.yaml`.
        - map_location: str, `torch.load` 的设备位置, 正式加载通常为 `cpu`.
        - allow_current_workspace_code: bool, run 缺少 `src_snapshot/src` 时是否明确使用当前工作区代码.

    返回值:
        - wrapper: torch.nn.Module, strict state dict、`on_load_checkpoint` 和 `eval()` 已完成.
    """

    global _ACTIVE_SNAPSHOT_SOURCE

    checkpoint = Path(checkpoint_path).resolve()
    run_directory = (
        checkpoint.parent.parent
        if checkpoint.parent.name == "checkpoints"
        else checkpoint.parent
    )
    snapshot_source = run_directory / "src_snapshot" / "src"
    required_packages = ("datasets", "model", "modules", "utils", "wrappers")
    snapshot_complete = (
        snapshot_source.is_dir()
        and (snapshot_source / "__init__.py").is_file()
        and all((snapshot_source / name).is_dir() for name in required_packages)
    )
    if snapshot_complete:
        source_path: Path | None = snapshot_source.resolve()
    elif allow_current_workspace_code:
        source_path = None
    else:
        raise FileNotFoundError(
            f"{snapshot_source}: 缺少完整训练代码快照; "
            "若要使用当前工作区代码, 必须显式设置 allow_current_workspace_code=true."
        )

    if source_path is not None:
        if _ACTIVE_SNAPSHOT_SOURCE not in (None, source_path):
            raise RuntimeError(
                f"同一进程不能混用训练快照: {_ACTIVE_SNAPSHOT_SOURCE} 与 {source_path}."
            )
        imported = [
            name
            for name in sys.modules
            if any(
                name == prefix or name.startswith(f"{prefix}.")
                for prefix in (
                    "src.datasets",
                    "src.model",
                    "src.modules",
                    "src.utils",
                    "src.wrappers",
                )
            )
        ]
        if imported and _ACTIVE_SNAPSHOT_SOURCE is None:
            raise RuntimeError(
                f"加载训练快照前已经导入模型或 Dataset 模块: {sorted(imported)[:8]}."
            )
        if _ACTIVE_SNAPSHOT_SOURCE is None:
            import src

            src.__path__[:] = [str(source_path)]
            importlib.invalidate_caches()
            _ACTIVE_SNAPSHOT_SOURCE = source_path
    elif _ACTIVE_SNAPSHOT_SOURCE is not None:
        raise RuntimeError(
            f"当前进程已经加载训练快照 {_ACTIVE_SNAPSHOT_SOURCE}, 不能切换到工作区代码."
        )

    if resolved_config_path is None:
        candidates = (
            run_directory / "config.yaml",
            run_directory / "resolved_config.yaml",
        )
        existing = tuple(path for path in candidates if path.is_file())
        if len(existing) != 1:
            raise FileNotFoundError(
                f"无法唯一解析 resolved config, 请显式传入路径. 候选={candidates}."
            )
        config_path = existing[0]
    else:
        config_path = Path(resolved_config_path)

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
    return wrapper, resolved_cfg
