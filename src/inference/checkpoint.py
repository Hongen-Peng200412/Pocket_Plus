"""从训练运行目录的代码快照严格恢复 Stage1 wrapper. 

主要入口 ``load_stage1_wrapper`` 优先使用 checkpoint 所属运行目录中的
``src_snapshot/src/`` 与 resolved config, 再 strict 加载 ``state_dict`` 并执行
``on_load_checkpoint``. 缺少完整快照时默认报错; 只有调用者显式允许时才使用
当前工作区代码. 同一 Python 进程不得混用两个训练运行的快照. 
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Callable


_ACTIVE_SNAPSHOT_SOURCE: Path | None = None
_REQUIRED_SNAPSHOT_PACKAGES = (
    "artifacts",
    "datasets",
    "inference",
    "model",
    "modules",
    "selector",
    "utils",
    "wrappers",
)


def _checkpoint_run_directory(checkpoint_path: str | Path) -> Path:
    """返回 checkpoint 所属的训练运行目录. """
    checkpoint = Path(checkpoint_path).resolve()
    if checkpoint.parent.name == "checkpoints":
        return checkpoint.parent.parent
    return checkpoint.parent


def resolve_checkpoint_source_path(
    checkpoint_path: str | Path,
    allow_current_workspace_code: bool,
) -> Path | None:
    """
    解析运行目录的完整 ``src`` 快照. 

    返回 ``run/src_snapshot/src``; 该目录缺失时, 只有
    ``allow_current_workspace_code=True`` 才返回 ``None``. 
    """
    source_path = _checkpoint_run_directory(checkpoint_path) / "src_snapshot" / "src"
    missing = [
        name
        for name in _REQUIRED_SNAPSHOT_PACKAGES
        if not (source_path / name).is_dir()
    ]
    if (
        source_path.is_dir()
        and (source_path / "__init__.py").is_file()
        and not missing
    ):
        return source_path.resolve()
    if allow_current_workspace_code:
        return None
    raise FileNotFoundError(
        "checkpoint 所属运行目录缺少完整代码快照: "
        f"{source_path}，缺少子包={missing}。若确认要使用当前工作区代码，"
        "必须显式设置 "
        "allow_current_workspace_code=True。"
    )


def _activate_checkpoint_source(source_path: Path | None) -> None:
    """
    让 Hydra 后续导入从唯一训练快照的 ``src`` 目录解析. 

    推理编排模块已从当前工作区导入; Dataset、模型、wrapper、损失模块
    与共享工具必须尚未导入, 才能统一从快照根目录解析. 如果其中任一
    子包已经从其他位置导入, 直接报错, 不在活跃进程中删除已加载模块. 
    """
    global _ACTIVE_SNAPSHOT_SOURCE

    if source_path is None:
        if _ACTIVE_SNAPSHOT_SOURCE is not None:
            raise RuntimeError(
                "当前进程已加载训练快照，不能再切换到工作区代码: "
                f"{_ACTIVE_SNAPSHOT_SOURCE}"
            )
        return
    source_path = source_path.resolve()
    if _ACTIVE_SNAPSHOT_SOURCE is not None:
        if _ACTIVE_SNAPSHOT_SOURCE != source_path:
            raise RuntimeError(
                "同一推理进程只能加载一套训练快照；已加载 "
                f"{_ACTIVE_SNAPSHOT_SOURCE}，本次请求 {source_path}"
            )
        return

    snapshot_packages = (
        "src.datasets",
        "src.model",
        "src.wrappers",
        "src.modules",
        "src.utils",
    )
    imported = sorted(
        name
        for name in sys.modules
        if any(name == prefix or name.startswith(f"{prefix}.") for prefix in snapshot_packages)
    )
    if imported:
        raise RuntimeError(
            "加载 checkpoint 快照前已导入模型或 wrapper 代码，无法保证快照单一性: "
            f"{imported[:10]}"
        )

    import src

    # 只保留快照根, 防止快照缺失子模块时从当前工作区静默补齐. 
    src.__path__[:] = [str(source_path)]
    importlib.invalidate_caches()
    _ACTIVE_SNAPSHOT_SOURCE = source_path


def resolve_checkpoint_config_path(
    checkpoint_path: str | Path,
    resolved_config_path: str | Path | None,
) -> Path:
    """
    解析调用者指定或训练 run 中与 checkpoint 相邻的 resolved config. 

    输入参数:
        - checkpoint_path: str | Path, 调用者明确选中的 Stage1 checkpoint
        - resolved_config_path: str | Path | None, 显式 config 路径; 为 None 时只检查
          训练运行目录的 `config.yaml` 与 `resolved_config.yaml`

    输出:
        - config_path: Path, 唯一存在的 resolved YAML 路径
    """
    checkpoint = Path(checkpoint_path)
    if resolved_config_path is not None:
        config_path = Path(resolved_config_path)
        if not config_path.is_file():
            raise FileNotFoundError(f"resolved config 不存在: {config_path}")
        return config_path
    # tuple[Path,Path], 只检查 checkpoint 所属运行目录的两个 resolved config 标准名称. 
    run_directory = _checkpoint_run_directory(checkpoint)
    candidates = (run_directory / "config.yaml", run_directory / "resolved_config.yaml")
    # tuple[Path,...], 去重后真实存在的候选; 必须恰有一个, 避免静默选错实验配置. 
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
    allow_current_workspace_code: bool = False,
):
    """
    从 resolved config 严格恢复完整 Stage1 wrapper 并执行 checkpoint 生命周期. 

    输入参数:
        - checkpoint_path: str | Path, 调用者选中的 `BEST.ckpt` 或等价完整 checkpoint
        - resolved_config_path: str | Path | None, resolved `config.yaml`; None 时按训练
          run 固定相邻关系解析, 不计算 hash 或查询 registry
        - map_location: str, `torch.load` 的设备位置, 正式 CPU 恢复传 `"cpu"`
        - lazy_initializer: Callable | None, 仅当当前模型仍含未初始化参数时由调用者
          提供的 `(wrapper,resolved_cfg)->None` 初始化函数
        - allow_current_workspace_code: bool, 缺少 ``src_snapshot/src`` 时是否
          显式允许使用当前工作区代码; 默认为 False

    输出:
        - wrapper: torch.nn.Module, strict state_dict、`on_load_checkpoint`、eval 已完成的
          完整 wrapper, 而不是裸 backbone
    """
    import hydra
    import torch
    from omegaconf import OmegaConf

    checkpoint = Path(checkpoint_path)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint 不存在: {checkpoint}")
    source_path = resolve_checkpoint_source_path(
        checkpoint,
        allow_current_workspace_code=bool(allow_current_workspace_code),
    )
    _activate_checkpoint_source(source_path)
    config_path = resolve_checkpoint_config_path(checkpoint, resolved_config_path)
    # DictConfig, 训练时保存的完整 resolved 配置; model/train 已无 Hydra 插值歧义. 
    resolved_cfg = OmegaConf.load(config_path)
    OmegaConf.resolve(resolved_cfg)
    if "model" not in resolved_cfg or "train" not in resolved_cfg:
        raise KeyError("resolved config 必须包含 model 与 train")
    # LightningModule, 由 resolved_cfg.model 重建的完整 wrapper, 尚未加载参数与 runtime cache. 
    wrapper = hydra.utils.instantiate(
        resolved_cfg.model,
        optimizer=resolved_cfg.train.optimizer,
        scheduler=resolved_cfg.train.scheduler,
        compile=False,
    )
    if lazy_initializer is not None:
        lazy_initializer(wrapper, resolved_cfg)
    # dict[str,Any], Lightning checkpoint; state_dict 之外还含 wrapper 的候选阈值 cache. 
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
