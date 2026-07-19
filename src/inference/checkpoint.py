"""AdaLigand Stage1 正式完整 wrapper checkpoint 加载器。"""

from __future__ import annotations

from pathlib import Path
from typing import Callable


def resolve_checkpoint_config_path(
    checkpoint_path: str | Path,
    resolved_config_path: str | Path | None,
) -> Path:
    """
    解析调用者指定或训练 run 中与 checkpoint 相邻的 resolved config。

    输入参数:
        - checkpoint_path: str | Path, 调用者明确选中的 Stage1 checkpoint
        - resolved_config_path: str | Path | None, 显式 config 路径；为 None 时只检查
          训练 run 的 `checkpoints/../config.yaml` 与 checkpoint 同目录 `config.yaml`

    输出:
        - config_path: Path, 唯一存在的 resolved YAML 路径
    """
    checkpoint = Path(checkpoint_path)
    if resolved_config_path is not None:
        config_path = Path(resolved_config_path)
        if not config_path.is_file():
            raise FileNotFoundError(f"resolved config 不存在: {config_path}")
        return config_path
    candidates = (
        checkpoint.parent.parent / "config.yaml",
        checkpoint.parent / "config.yaml",
    )
    existing = tuple(dict.fromkeys(path for path in candidates if path.is_file()))
    if len(existing) != 1:
        raise FileNotFoundError(
            "无法从 checkpoint 唯一解析相邻 resolved config；请显式传入 "
            f"resolved_config_path。候选={candidates}"
        )
    return existing[0]


def load_stage1_wrapper(
    checkpoint_path: str | Path,
    resolved_config_path: str | Path | None,
    map_location: str,
    lazy_initializer: Callable[[object, object], None] | None = None,
):
    """
    从 resolved config 严格恢复完整 Stage1 wrapper 并执行 checkpoint 生命周期。

    输入参数:
        - checkpoint_path: str | Path, 调用者选中的 `BEST.ckpt` 或等价完整 checkpoint
        - resolved_config_path: str | Path | None, resolved `config.yaml`；None 时按训练
          run 固定相邻关系解析，不计算 hash 或查询 registry
        - map_location: str, `torch.load` 的设备位置，正式 CPU 恢复传 `"cpu"`
        - lazy_initializer: Callable | None, 仅当当前模型仍含未初始化参数时由调用者
          提供的 `(wrapper,resolved_cfg)->None` 初始化函数

    输出:
        - wrapper: torch.nn.Module, strict state_dict、`on_load_checkpoint`、eval 已完成的
          完整 wrapper，而不是裸 backbone
    """
    import hydra
    import torch
    from omegaconf import OmegaConf

    checkpoint = Path(checkpoint_path)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint 不存在: {checkpoint}")
    config_path = resolve_checkpoint_config_path(checkpoint, resolved_config_path)
    resolved_cfg = OmegaConf.load(config_path)
    OmegaConf.resolve(resolved_cfg)
    if "model" not in resolved_cfg or "train" not in resolved_cfg:
        raise KeyError("resolved config 必须包含 model 与 train")
    wrapper = hydra.utils.instantiate(
        resolved_cfg.model,
        optimizer=resolved_cfg.train.optimizer,
        scheduler=resolved_cfg.train.scheduler,
        compile=False,
    )
    if lazy_initializer is not None:
        lazy_initializer(wrapper, resolved_cfg)
    checkpoint_payload = torch.load(
        str(checkpoint), map_location=map_location, weights_only=False
    )
    if not isinstance(checkpoint_payload, dict) or "state_dict" not in checkpoint_payload:
        raise TypeError("正式 Stage1 checkpoint 必须是含 state_dict 的完整 wrapper checkpoint")
    wrapper.load_state_dict(checkpoint_payload["state_dict"], strict=True)
    if not hasattr(wrapper, "on_load_checkpoint"):
        raise TypeError("正式 Stage1 wrapper 必须实现 on_load_checkpoint 生命周期")
    wrapper.on_load_checkpoint(checkpoint_payload)
    wrapper.eval()
    return wrapper
