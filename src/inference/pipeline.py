# -*- coding: utf-8 -*-
"""执行一个 PDB 的 Stage1 V3 概率图、blob、centered 和评分发布.

本模块提供按正式产物边界划分的三个入口. 清单遍历、calibration 搜索和并发的
跨 PDB 调度留在 ``cli.py``, 避免把命令流程藏进多层 runner 包装.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .artifacts import (
    Stage1ArtifactPaths,
    load_stage1_npz,
    publish_stage1_artifact,
    publish_stage1_json,
)
from .centered import infer_centered_boxes
from .full_map import infer_full_map
from .scoring import score_centered_candidates, select_centered_candidates


# ================================================================================================


def produce_probability_map(
    paths: Stage1ArtifactPaths,
    dataset: Any,
    collator: Any,
    wrapper: Any,
    device: str,
    window_config: Mapping[str, Any],
) -> dict[str, np.ndarray]:
    """运行滑窗并原子发布一个 PDB 的 ``probability_map.npz``."""

    result = infer_full_map(
        dataset=dataset,
        collator=collator,
        wrapper=wrapper,
        pdb_id=paths.pdb_id,
        device=device,
        stride_zyx=window_config["stride_zyx"],
        sigma=float(window_config["gaussian_sigma"]),
        window_batch_size=int(window_config["batch_size"]),
        window_workers=int(window_config["workers"]),
        prefetch_batches=int(window_config["prefetch_batches"]),
        precision=str(window_config["precision"]),
        pending_fusion_batches=int(window_config["pending_fusion_batches"]),
    )
    arrays = {
        "probability_map": result.probability_map.astype(np.float32, copy=False),
        "full_shape_zyx": np.asarray(result.probability_map.shape, dtype=np.int64),
        "origin_xyz": result.origin_xyz.astype(np.float32, copy=False),
        "voxel_size_xyz": result.voxel_size_xyz.astype(np.float32, copy=False),
        "window_shape_zyx": np.asarray((80, 80, 80), dtype=np.uint8),
        "stride_zyx": np.asarray(window_config["stride_zyx"], dtype=np.int32),
        "gaussian_sigma": np.asarray(window_config["gaussian_sigma"], dtype=np.float32),
        "window_count": np.asarray(result.window_count, dtype=np.int32),
    }
    publish_stage1_artifact(
        paths.artifact("probability"),
        arrays,
        paths.complete("probability"),
    )
    return arrays


def produce_centered_role(
    paths: Stage1ArtifactPaths,
    dataset: Any,
    collator: Any,
    wrapper: Any,
    device: str,
    blobs: Mapping[str, np.ndarray],
    centered_role: str,
    min_voxels: int,
    centered_config: Mapping[str, Any],
    blob_limit: int | None,
    enforce_blob_limit: bool | None,
    publish_complete: bool,
) -> dict[str, np.ndarray] | None:
    """读取调用者已并行生成的 F1/F3 blobs 并发布 centered 文件.

    F3 中满足 ``fits_centered_box`` 和 ``min_voxels`` 的候选数严格大于
    ``blob_limit`` 时写 ``_BLOB_EXCEED``. ``enforce_blob_limit`` 为真则返回
    ``None`` 且不发布 centered; 为假时保留标记并继续.
    """

    eligible = np.asarray(blobs["fits_centered_box"], dtype=np.bool_) & (
        np.asarray(blobs["voxel_count"], dtype=np.int64) >= int(min_voxels)
    )
    if centered_role == "F3_centered" and int(eligible.sum()) > int(blob_limit):
        publish_stage1_json(
            paths.blob_exceed,
            {
                "pdb_id": paths.pdb_id,
                "centered_role": centered_role,
                "eligible_blob_count": int(eligible.sum()),
                "limit": int(blob_limit),
                "enforced": bool(enforce_blob_limit),
            },
        )
        if enforce_blob_limit:
            paths.artifact(centered_role).unlink(missing_ok=True)
            paths.complete(centered_role).unlink(missing_ok=True)
            return None
    elif centered_role == "F3_centered":
        Path(paths.blob_exceed).unlink(missing_ok=True)

    probability = load_stage1_npz(
        paths.artifact("probability"),
        ("probability_map", "origin_xyz", "voxel_size_xyz"),
    )
    arrays = infer_centered_boxes(
        dataset=dataset,
        collator=collator,
        wrapper=wrapper,
        pdb_id=paths.pdb_id,
        producer=paths.stage1_model_name,
        blobs=blobs,
        full_probability=probability["probability_map"],
        origin_xyz=probability["origin_xyz"],
        voxel_size_xyz=probability["voxel_size_xyz"],
        device=device,
        precision=str(centered_config["precision"]),
        centered_batch_size=int(centered_config["batch_size"]),
        centered_workers=int(centered_config["workers"]),
        prefetch_batches=int(centered_config["prefetch_batches"]),
        pending_cpu_batches=int(centered_config["pending_cpu_batches"]),
        min_voxels=int(min_voxels),
        centered_forward=str(centered_config["forward"]),
        save_voxel_final=bool(centered_config["save_voxel_final"]),
        save_dense48=bool(centered_config["save_dense48"]),
    )
    if not publish_complete:
        paths.complete(centered_role).unlink(missing_ok=True)
    publish_stage1_artifact(
        paths.artifact(centered_role),
        arrays,
        paths.complete(centered_role) if publish_complete else None,
    )
    return arrays


def score_and_publish_centered(
    paths: Stage1ArtifactPaths,
    centered_role: str,
    score_mode: str,
    score_parameters: Mapping[str, float],
    score_threshold: float,
    min_voxels: int,
) -> dict[str, np.ndarray]:
    """计算冻结分数和选择标志, 原子替换同一 centered 正式路径."""

    centered = load_stage1_npz(
        paths.artifact(centered_role),
        (
            "source_probability_mean",
            "voxel_offsets",
            "voxel_index_local_zyx",
        ),
    )
    score = score_centered_candidates(
        centered,
        score_mode=score_mode,
        score_parameters=score_parameters,
    )
    selected = select_centered_candidates(
        centered,
        score=score,
        score_threshold=float(score_threshold),
        min_voxels=int(min_voxels),
    )
    publish_stage1_artifact(
        paths.artifact(centered_role),
        selected,
        paths.complete(centered_role),
    )
    return selected
