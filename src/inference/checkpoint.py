"""AdaLigand Stage1 正式完整 wrapper checkpoint 加载器。

训练侧的 CPC1→CPC2 只恢复 model state；本模块服务推理侧，必须从 resolved config
重新实例化完整 wrapper，再 strict 加载 ``state_dict`` 并执行 ``on_load_checkpoint``。
因此推理能恢复候选阈值等 wrapper 状态，而不是只得到一个裸 backbone。
"""

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
    # tuple[Path,Path], 只允许训练 run 根与 checkpoint 同目录这两种固定相邻布局。
    candidates = (
        checkpoint.parent.parent / "config.yaml",
        checkpoint.parent / "config.yaml",
    )
    # tuple[Path,...], 去重后真实存在的候选；必须恰有一个，避免静默选错实验配置。
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
    # DictConfig, 训练时保存的完整 resolved 配置；model/train 已无 Hydra 插值歧义。
    resolved_cfg = OmegaConf.load(config_path)
    OmegaConf.resolve(resolved_cfg)
    if "model" not in resolved_cfg or "train" not in resolved_cfg:
        raise KeyError("resolved config 必须包含 model 与 train")
    # LightningModule, 由 resolved_cfg.model 重建的完整 wrapper，尚未加载参数与 runtime cache。
    wrapper = hydra.utils.instantiate(
        resolved_cfg.model,
        optimizer=resolved_cfg.train.optimizer,
        scheduler=resolved_cfg.train.scheduler,
        compile=False,
    )
    if lazy_initializer is not None:
        lazy_initializer(wrapper, resolved_cfg)
    # dict[str,Any], Lightning checkpoint；state_dict 之外还含 wrapper 的候选阈值 cache。
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
