"""把历史 Lightning checkpoint 的 ModelCheckpoint 状态迁移到新运行目录."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from collections.abc import Mapping, MutableMapping
from pathlib import Path
from typing import Any

import torch


CHECKPOINT_PATH_FIELDS = (
    "best_model_path",
    "kth_best_model_path",
    "last_model_path",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    return parser


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_recorded_path(value: str, source_dir: Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = source_dir / path
    return path.resolve()


def _copy_checkpoint_artifact(
    source_path: Path,
    *,
    source_checkpoint: Path,
    output_checkpoint: Path,
    destination_dir: Path,
    copied: dict[str, dict[str, str]],
) -> Path:
    if source_path == source_checkpoint:
        return output_checkpoint
    if not source_path.is_file():
        raise FileNotFoundError(f"ModelCheckpoint 引用的文件不存在：{source_path}")
    destination_path = destination_dir / source_path.name
    source_sha256 = _sha256_file(source_path)
    if destination_path.exists():
        if _sha256_file(destination_path) != source_sha256:
            raise FileExistsError(f"目标 checkpoint 已存在且内容不同：{destination_path}")
    else:
        shutil.copy2(source_path, destination_path)
    copied[str(source_path)] = {
        "destination": str(destination_path),
        "sha256": source_sha256,
    }
    return destination_path


def _rebase_callback_state(
    state: MutableMapping[str, Any],
    *,
    source_checkpoint: Path,
    output_checkpoint: Path,
    destination_dir: Path,
    copied: dict[str, dict[str, str]],
) -> None:
    source_dir = Path(str(state.get("dirpath") or source_checkpoint.parent)).resolve()
    state["dirpath"] = str(destination_dir)
    for field_name in CHECKPOINT_PATH_FIELDS:
        recorded_path = str(state.get(field_name) or "")
        if not recorded_path:
            continue
        source_path = _resolve_recorded_path(recorded_path, source_dir)
        state[field_name] = str(
            _copy_checkpoint_artifact(
                source_path,
                source_checkpoint=source_checkpoint,
                output_checkpoint=output_checkpoint,
                destination_dir=destination_dir,
                copied=copied,
            )
        )

    best_k_models = state.get("best_k_models")
    if isinstance(best_k_models, Mapping):
        rebased_best_k_models = {}
        for recorded_path, score in best_k_models.items():
            source_path = _resolve_recorded_path(str(recorded_path), source_dir)
            destination_path = _copy_checkpoint_artifact(
                source_path,
                source_checkpoint=source_checkpoint,
                output_checkpoint=output_checkpoint,
                destination_dir=destination_dir,
                copied=copied,
            )
            rebased_best_k_models[str(destination_path)] = score
        state["best_k_models"] = rebased_best_k_models


def rebase_checkpoint(source: Path, destination_dir: Path) -> Path:
    """复制历史 top-k 并发布已迁移回调路径的完整 checkpoint."""

    source_checkpoint = source.expanduser().resolve()
    if not source_checkpoint.is_file():
        raise FileNotFoundError(f"源 checkpoint 不存在：{source_checkpoint}")
    destination_dir = destination_dir.expanduser().resolve()
    destination_dir.mkdir(parents=True, exist_ok=True)
    output_checkpoint = destination_dir / "resume_state.ckpt"
    manifest_path = destination_dir / "resume_state_manifest.json"

    payload = torch.load(source_checkpoint, map_location="cpu")
    callback_states = payload.get("callbacks") if isinstance(payload, Mapping) else None
    if not isinstance(callback_states, MutableMapping):
        raise ValueError("源 checkpoint 缺少 Lightning callbacks 状态。")

    copied: dict[str, dict[str, str]] = {}
    rebased_callback_keys = []
    for callback_key, state in callback_states.items():
        if not isinstance(state, MutableMapping):
            continue
        if not {"dirpath", "best_model_path", "best_k_models"}.issubset(state):
            continue
        _rebase_callback_state(
            state,
            source_checkpoint=source_checkpoint,
            output_checkpoint=output_checkpoint,
            destination_dir=destination_dir,
            copied=copied,
        )
        rebased_callback_keys.append(str(callback_key))
    if not rebased_callback_keys:
        raise ValueError("源 checkpoint 中没有可迁移的 ModelCheckpoint 状态。")

    temporary_checkpoint = output_checkpoint.with_suffix(".ckpt.tmp")
    torch.save(payload, temporary_checkpoint)
    os.replace(temporary_checkpoint, output_checkpoint)
    manifest = {
        "schema_version": 1,
        "source_checkpoint": str(source_checkpoint),
        "source_sha256": _sha256_file(source_checkpoint),
        "output_checkpoint": str(output_checkpoint),
        "output_sha256": _sha256_file(output_checkpoint),
        "rebased_callback_keys": rebased_callback_keys,
        "copied_checkpoint_artifacts": copied,
        "random_augmentation_state": "not_present_in_source_checkpoint",
    }
    temporary_manifest = manifest_path.with_suffix(".json.tmp")
    temporary_manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.replace(temporary_manifest, manifest_path)
    return output_checkpoint


def main() -> None:
    arguments = build_parser().parse_args()
    print(rebase_checkpoint(arguments.source, arguments.destination))


if __name__ == "__main__":
    main()
