"""为 Stage1 数据准备工具提供同目录原子发布和持久化同步。

主要入口是 :func:`atomic_write_text`、:func:`atomic_write_json` 和 :func:`atomic_save_npz`。本模块只负责临时文件、fsync、同目录 ``os.replace`` 与目录元数据同步，不解释密度数组、split 或 BOX pool 的科学字段。
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np


def fsync_directory(path: Path) -> None:
    """同步目录元数据，使同目录原子替换在进程退出或断电后可见。

    输入参数:
        - path: Path；要同步的已存在目录；目录中的文件替换必须已经完成。

    状态变化:
        - Linux 等支持目录文件描述符的系统执行 ``os.fsync``；Windows 测试环境直接返回，因为普通只读句柄不能安全打开目录。
    """

    if os.name == "nt":
        # Windows 不允许用普通只读句柄打开目录；正式服务器为 Linux，Windows 仅运行逻辑测试。
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_write_text(path: Path, content: str) -> None:
    """以 UTF-8 写入完整文本并在同目录原子替换目标文件。

    输入参数:
        - path: Path；目标文本文件；父目录会在写入前创建。
        - content: str；要完整写入的 Unicode 文本，使用 UTF-8 和 ``\n`` 换行。

    状态变化:
        - 先在目标同目录创建临时文件并 fsync 文件内容，再用 ``os.replace`` 替换目标，最后同步父目录；失败时清理临时文件，不留下半写目标。
    """

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
    """把任意 JSON 可序列化值以 UTF-8 缩进格式原子发布到 ``path``。

    输入参数:
        - path: Path；目标 JSON 文件。
        - value: Any；可由 ``json.dumps`` 序列化的对象；中文不转义，末尾添加一个换行。

    状态变化:
        - 委托 :func:`atomic_write_text` 完成同目录临时文件、fsync、替换和目录同步。
    """

    atomic_write_text(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def atomic_save_npz(path: Path, arrays: dict[str, np.ndarray], *, compressed: bool) -> None:
    """把具名 NumPy 数组写入同目录临时 NPZ，并在完整同步后原子发布。

    输入参数:
        - path: Path；目标 NPZ 文件；父目录会在写入前创建。
        - arrays: dict[str, np.ndarray]；NPZ 字段名到数组的映射；字段语义和 dtype 由调用方契约负责。
        - compressed: bool；为真使用 ``np.savez_compressed``，否则使用非压缩 ``np.savez``。

    状态变化:
        - 临时文件写完并 fsync 后使用 ``os.replace`` 替换目标，再同步父目录；失败时删除临时文件，不修改旧目标。
    """

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
