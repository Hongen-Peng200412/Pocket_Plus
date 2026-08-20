# -*- coding: utf-8 -*-
"""从完整概率图提取并发布单阈值 26-连通区域.

主要入口 :func:`extract_probability_blobs` 返回全部区域的稳定稀疏表,
:func:`publish_probability_blobs` 把该表写入 `F1_blobs.npz` 或
`F3_blobs.npz`. 本模块不按 `min_voxels` 删除区域.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from .artifacts import Stage1ArtifactPaths, load_stage1_npz, publish_stage1_artifact


# ================================================================================================


def extract_probability_blobs(
    probability_map: np.ndarray,
    threshold: float,
) -> dict[str, np.ndarray]:
    """按平均概率降序和最小全图线性索引升序整理全部 26-连通区域."""

    probability = np.asarray(probability_map, dtype=np.float32)
    labels, count = ndimage.label(
        probability >= np.float32(threshold),
        structure=np.ones((3, 3, 3), dtype=np.uint8),
    )
    records: list[tuple[float, int, np.ndarray, np.ndarray, bool, np.ndarray]] = []
    full_shape = np.asarray(probability.shape, dtype=np.int64)
    for label_id in range(1, int(count) + 1):
        coordinates = np.argwhere(labels == label_id).astype(np.int32)
        linear = np.ravel_multi_index(coordinates.T, probability.shape)
        order = np.argsort(linear, kind="stable")
        coordinates = coordinates[order]
        values = probability[tuple(coordinates.T)].astype(np.float32)
        minimum = coordinates.min(axis=0).astype(np.int64)
        maximum = coordinates.max(axis=0).astype(np.int64)
        requested = np.rint(
            coordinates.astype(np.float64).mean(axis=0) + 0.5 - 40.0
        ).astype(np.int64)
        start = np.clip(requested, 0, full_shape - 80)
        fits = bool(np.all(minimum >= start) and np.all(maximum < start + 80))
        records.append(
            (
                float(values.mean(dtype=np.float64)),
                int(linear[order[0]]),
                coordinates,
                values,
                fits,
                start.astype(np.int32) if fits else np.full(3, -1, dtype=np.int32),
            )
        )
    records.sort(key=lambda record: (-record[0], record[1]))
    counts = np.asarray([record[2].shape[0] for record in records], dtype=np.int32)
    offsets = np.concatenate(
        (np.zeros(1, dtype=np.int64), np.cumsum(counts, dtype=np.int64))
    )
    return {
        "blob_index": np.arange(len(records), dtype=np.int32),
        "voxel_offsets": offsets,
        "voxel_index_global_zyx": (
            np.concatenate([record[2] for record in records], axis=0)
            if records
            else np.empty((0, 3), dtype=np.int32)
        ),
        "source_probability": (
            np.concatenate([record[3] for record in records], axis=0)
            if records
            else np.empty((0,), dtype=np.float32)
        ),
        "source_probability_mean": np.asarray(
            [record[0] for record in records],
            dtype=np.float32,
        ),
        "voxel_count": counts,
        "fits_centered_box": np.asarray(
            [record[4] for record in records],
            dtype=np.bool_,
        ),
        "centered_box_start_zyx": (
            np.stack([record[5] for record in records], axis=0)
            if records
            else np.empty((0, 3), dtype=np.int32)
        ),
        "source_threshold_value": np.asarray([threshold], dtype=np.float32),
    }


def publish_probability_blobs(
    paths: Stage1ArtifactPaths,
    role: str,
    threshold: float,
) -> dict[str, np.ndarray]:
    """读取已完成概率图, 发布 F1 或 F3 的全部连通区域并返回同一数组映射."""

    probability = load_stage1_npz(
        paths.artifact("probability"),
        ("probability_map",),
    )["probability_map"]
    arrays = extract_probability_blobs(probability, threshold)
    publish_stage1_artifact(
        paths.artifact(role),
        arrays,
        paths.complete(role),
    )
    return arrays
