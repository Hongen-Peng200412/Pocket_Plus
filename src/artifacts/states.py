"""发布并读取 Stage1 的 PDB 互斥租约、role 完成标记与组件超限终态。

主要入口:
    - `PdbRunningLease.acquire`: 通过原子创建 `_RUNNING` 目录抢占一个 `producer/split/pdb_id`。
    - `mark_role_complete`: 在对应 payload 已关闭并重读校验后发布 `status/<role>/_COMPLETE`。
    - `mark_blob_exceed`: 发布 `_BLOB_EXCEED`，表示 `t_F1` 层候选组件数量超过正式上限。
    - `pdb_is_consumable`: 同时检查临时租约、异常终态与消费者要求的完成标记。

本模块只管理状态文件，不生成 probability、components 或 centered payload。`_RUNNING` 是短期互斥状态，`_COMPLETE` 与 `_BLOB_EXCEED` 是原子发布的 JSON；下游只读取没有前两类阻断且全部所需 role 已完成的 PDB。
"""

from __future__ import annotations

import json
import os
import socket
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .paths import OUTPUT_ROLES, Stage1ArtifactPaths


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    """
    在目标目录内写临时 JSON，再原子替换正式文件。

    输入参数:
        - path: Path, 正式 JSON 路径
        - payload: dict[str,Any], 可由标准 JSON 编码的完整内容

    输出:
        - None, 临时文件 flush/fsync 后原子替换 `path`；发布失败时删除当前函数创建的临时文件。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary_path.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


@dataclass
class PdbRunningLease:
    """
    表示当前进程成功持有的 PDB 级 `_RUNNING` 互斥锁。

    输入参数:
        - paths: Stage1ArtifactPaths, 当前 producer/split/PDB 路径
        - owner_token: str, 调用者提供的本 worker 唯一身份

    使用约定:
        - 通过 `acquire` 创建；返回 `None` 表示其它 worker 已持有。
        - 仅释放本对象成功创建的目录，不清理或猜测陈旧锁。
    """
    paths: Stage1ArtifactPaths
    owner_token: str
    acquired: bool = False

    @classmethod
    def acquire(
        cls,
        paths: Stage1ArtifactPaths,
        owner_token: str,
    ) -> "PdbRunningLease | None":
        """
        原子抢占一个 PDB 的 `_RUNNING` 目录。

        输入参数:
            - paths: Stage1ArtifactPaths, 当前 PDB 路径
            - owner_token: str, 当前 worker 的稳定唯一标识

        输出:
            - lease: PdbRunningLease | None, 成功时为可释放租约，冲突时为 None
        """
        paths.pdb_root.mkdir(parents=True, exist_ok=True)
        try:
            paths.running_dir.mkdir()
        except FileExistsError:
            return None

        # 当前进程刚以原子 `mkdir` 获得的 PDB 级租约；`acquired=True` 只说明当前对象有权尝试释放该目录。
        lease = cls(paths=paths, owner_token=str(owner_token), acquired=True)
        # dict[str, Any], 写入 `_RUNNING/owner.json` 的持有者身份；`owner_token` 是释放时必须再次匹配的权威字段。
        owner_payload = {
            "owner_token": str(owner_token),
            "pid": int(os.getpid()),
            "host": socket.gethostname(),
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        try:
            _atomic_json(paths.running_dir / "owner.json", owner_payload)
        except BaseException:
            paths.running_dir.rmdir()
            raise
        return lease

    def release(self) -> None:
        """
        释放当前对象确实持有的 `_RUNNING`，不递归删除未知内容。

        输出:
            - None: owner token 匹配时删除 `owner.json` 与空 `_RUNNING` 目录，并把 `acquired` 置为 False
        """
        if not self.acquired:
            return
        owner_path = self.paths.running_dir / "owner.json"
        if owner_path.is_file():
            with owner_path.open("r", encoding="utf-8") as handle:
                # dict[str, Any], 磁盘上的实际租约持有者；`owner_token` 不同表示目录已不属于当前对象，禁止代替其它 worker 解锁。
                owner_payload = json.load(handle)
            if owner_payload.get("owner_token") != self.owner_token:
                raise RuntimeError("_RUNNING owner_token 已变化，拒绝释放不属于当前 worker 的锁")
            owner_path.unlink()
        self.paths.running_dir.rmdir()
        self.acquired = False

    def __enter__(self) -> "PdbRunningLease":
        """
        进入租约上下文。

        输出:
            - lease: PdbRunningLease, 当前已持有的同一租约对象
        """
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        """
        离开上下文并释放当前租约。

        输入参数:
            - exc_type: object, 上下文异常类型；无异常时为 None
            - exc: object, 上下文异常对象；无异常时为 None
            - traceback: object, 上下文异常 traceback；无异常时为 None

        输出:
            - None: 无论是否异常都调用 `release()`，不吞掉异常
        """
        self.release()


def is_role_complete(paths: Stage1ArtifactPaths, output_role: str) -> bool:
    """
    返回指定 role 是否已有正式 `_COMPLETE` 标记。

    输入参数:
        - paths: Stage1ArtifactPaths, 当前 producer/split/PDB 路径对象
        - output_role: str, `OUTPUT_ROLES` 中的正式 role

    输出:
        - is_complete: bool, 对应完成标记文件存在时为 True
    """
    return paths.role_complete_path(output_role).is_file()


def mark_role_complete(paths: Stage1ArtifactPaths, output_role: str) -> None:
    """
    在 payload 已关闭并校验后原子发布 role `_COMPLETE`。

    输入参数:
        - paths: Stage1ArtifactPaths, 当前 PDB 路径
        - output_role: str, `OUTPUT_ROLES` 中已经完成的 role

    输出:
        - None, 写入包含 `output_role` 和 UTC 完成时间的 JSON 标记；本函数不检查或生成对应 payload。
    """
    if output_role not in OUTPUT_ROLES:
        raise ValueError(f"未知 output_role={output_role!r}")
    _atomic_json(
        paths.role_complete_path(output_role),
        {
            "output_role": output_role,
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        },
    )


def mark_blob_exceed(
    paths: Stage1ArtifactPaths,
    n_f1_eligible: int,
    limit: int,
) -> None:
    """
    发布与正常下游完成互斥的 `_BLOB_EXCEED` 终态。

    输入参数:
        - paths: Stage1ArtifactPaths, 当前 PDB 路径
        - n_f1_eligible: int, `t_F1` 层正式 eligible component 数
        - limit: int, 允许继续组件/居中生产的上限，正式值为 200

    输出:
        - None, 写入包含 `N_F1_eligible` 和 `limit` 的 JSON 终态；本函数不删除已有正常产物。
    """
    _atomic_json(
        paths.blob_exceed_path,
        {"N_F1_eligible": int(n_f1_eligible), "limit": int(limit)},
    )


def pdb_is_consumable(
    paths: Stage1ArtifactPaths,
    required_roles: tuple[str, ...],
) -> bool:
    """
    判断一个 PDB 是否可被下游作为完整样本读取。

    输入参数:
        - paths: Stage1ArtifactPaths, 当前 PDB 路径
        - required_roles: tuple[str, ...], 消费者需要的全部 role, 如 F1_centered 等；每项必须能由 `Stage1ArtifactPaths.role_complete_path` 解析。

    输出:
        - consumable: bool, 无 `_RUNNING/_BLOB_EXCEED` 且全部 role 完成时为 True
    """
    if paths.running_dir.exists() or paths.blob_exceed_path.exists():
        return False
    return all(is_role_complete(paths, role) for role in required_roles)
