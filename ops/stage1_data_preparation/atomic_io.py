"""为 Stage1 数据准备工具提供同目录原子发布与持久化同步。

主要入口是 :func:`atomic_write_text`、:func:`atomic_save_npz` 与
:func:`fsync_directory`。本模块只处理文件系统事务，不解释密度数组、划分或
BOX pool 的科学含义。
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np


def fsync_directory(path: Path) -> None:
    """同步目录元数据，使此前的同目录原子替换在断电后可追溯。"""

    if os.name == "nt":
        # Windows 不允许用普通只读句柄打开目录；正式服务器为 Linux，Windows 仅运行逻辑测试。
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_write_text(path: Path, content: str) -> None:
    """以 UTF-8 写入完整文本，并在同目录完成原子替换和目录同步。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
        fsync_directory(path.parent)
    finally:
        temporary_path.unlink(missing_ok=True)


def atomic_write_json(path: Path, value: Any) -> None:
    """把一个 JSON 值以 UTF-8、缩进格式原子发布到 ``path``。"""

    atomic_write_text(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def atomic_save_npz(path: Path, arrays: dict[str, np.ndarray], *, compressed: bool) -> None:
    """把具名 NumPy 数组写入同目录临时 NPZ，完整同步后原子发布。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            if compressed:
                np.savez_compressed(handle, **arrays)
            else:
                np.savez(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
        fsync_directory(path.parent)
    finally:
        temporary_path.unlink(missing_ok=True)
