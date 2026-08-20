# -*- coding: utf-8 -*-
"""执行一个 PDB 的 Stage1 V3 概率图, blob, centered 和评分发布.

本模块提供按正式产物边界划分的入口. 清单遍历, calibration 搜索和跨 PDB
并发调度由 ``workflow.py`` 直接编排; ``cli.py`` 只负责解析命令与构造依赖.
"""

from __future__ import annotations

from concurrent.futures import Executor, Future
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
    publisher: Executor,
) -> tuple[dict[str, np.ndarray], Future[None]]:
    """运行滑窗并异步发布一个 PDB 的概率产物.

    Dataset, collator 与 wrapper 直接传给 :func:`infer_full_map`;
    ``window_config`` 必须显式提供 stride, sigma, batch, 线程, 预取, 精度和
    融合队列深度.

    返回的数组只含 ``probability_map``,``origin_xyz`` 和
    ``voxel_size_xyz`` 三个科学字段. ``Future`` 完成时, 科学 NPZ, 窗口
    ``geometry.json``, 性能 JSON 和 ``status/probability/_COMPLETE`` 已按该
    顺序发布. NPZ 压缩由 ``publisher`` 执行, 不阻塞下一个 PDB 的 GPU 前向.
    """

    paths.complete("probability").unlink(missing_ok=True)
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
        "origin_xyz": result.origin_xyz.astype(np.float32, copy=False),
        "voxel_size_xyz": result.voxel_size_xyz.astype(np.float32, copy=False),
    }
    if not all(np.isfinite(value).all() for value in arrays.values()):
        raise ValueError(f"{paths.pdb_id}: 完整图概率或几何包含 NaN/Inf.")
    geometry = {
        "full_shape_zyx": [int(value) for value in result.probability_map.shape],
        "origin_xyz": [float(value) for value in result.origin_xyz],
        "voxel_size_xyz": [float(value) for value in result.voxel_size_xyz],
        "window_shape_zyx": [80, 80, 80],
        "stride_zyx": [int(value) for value in window_config["stride_zyx"]],
        "gaussian_sigma": float(window_config["gaussian_sigma"]),
        "window_count": int(result.window_count),
    }
    performance = {
        "wall_seconds": float(result.wall_seconds),
        "materialize_wait_seconds": float(result.materialize_wait_seconds),
        "fusion_wait_seconds": float(result.fusion_wait_seconds),
    }

    def publish() -> None:
        publish_stage1_json(paths.pdb_root / "probability" / "geometry.json", geometry)
        publish_stage1_json(paths.performance("probability"), performance)
        publish_stage1_artifact(
            paths.artifact("probability"),
            arrays,
            paths.complete("probability"),
        )

    return arrays, publisher.submit(publish)


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
    selection: Mapping[str, Any] | None,
    publisher: Executor,
) -> Future[dict[str, np.ndarray]]:
    """读取调用者已并行生成的 F1/F3 blobs 并发布 centered 文件.

    ``blobs`` 是当前角色已经发布或刚从 CPU future 返回的字段映射. F3 中
    满足 ``fits_centered_box`` 和 ``min_voxels`` 的候选数严格大于
    ``blob_limit`` 时写 ``_BLOB_EXCEED`` 并继续生产, 不建立特殊跳过终态;
    该 JSON 精确保存 `pdb_id`, `centered_role`, `eligible_blob_count` 和 `limit`.
    返回 centered 发布 Future. ``selection=None`` 用于 calibration 搜索前的
    临时候选归档, 只发布未评分 NPZ, 不建立完成标记. 传入冻结选择映射时,
    Future 在首次压缩前写入 ``score`` 与 ``selected``; 完成时 centered NPZ,
    五键性能 JSON 和角色 ``_COMPLETE`` 均已发布. 浮点字段在提交压缩前直接
    检查 NaN/Inf. 性能 JSON 精确保存 `wall_seconds`,
    `materialize_wait_seconds`, `cpu_arrange_wait_seconds`, `batch_count` 和
    `entry_count`; 正式完成标记保存 `output_role` 与 `completed_at_utc`.
    """

    paths.complete(centered_role).unlink(missing_ok=True)
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
            },
        )
    elif centered_role == "F3_centered":
        Path(paths.blob_exceed).unlink(missing_ok=True)

    probability = load_stage1_npz(
        paths.artifact("probability"),
        ("probability_map", "origin_xyz", "voxel_size_xyz"),
    )
    packed, performance = infer_centered_boxes(
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
        packer=publisher,
    )

    def publish() -> dict[str, np.ndarray]:
        arrays = packed.result()
        if selection is not None:
            score = score_centered_candidates(
                arrays,
                score_mode=str(selection["score_mode"]),
                score_parameters=selection["score_parameters"],
            )
            arrays = select_centered_candidates(
                arrays,
                score=score,
                score_threshold=float(selection["score_threshold"]),
                min_voxels=int(selection["min_voxels"]),
            )
        for name, value in arrays.items():
            if np.issubdtype(value.dtype, np.floating) and not np.isfinite(value).all():
                raise ValueError(f"{paths.pdb_id}:{centered_role}:{name} 包含 NaN/Inf.")
        paths.complete(centered_role).unlink(missing_ok=True)
        publish_stage1_json(paths.performance(centered_role), performance)
        publish_stage1_artifact(
            paths.artifact(centered_role),
            arrays,
            None if selection is None else paths.complete(centered_role),
        )
        return arrays

    return publisher.submit(publish)
