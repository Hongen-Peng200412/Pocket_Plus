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


def extract_probability_blobs(
    probability_map: np.ndarray,
    threshold: float,
) -> dict[str, np.ndarray]:
    """提取并稳定排序一个阈值下的全部 26 邻域连通区域.

    输入参数:
        - probability_map: float32 ``(D, H, W)``, 完整图 ZYX 配体概率.
        - threshold: float, 包含端点的概率阈值.

    返回值:
        - blob_index: int32 ``(N_blob,)``, 排序后的连续身份.
        - voxel_offsets: int64 ``(N_blob+1,)``, 切分两个逐体素值表.
        - voxel_index_global_zyx: int32 ``(L_voxel, 3)``, 完整图 ZYX 索引.
        - source_probability: float32 ``(L_voxel,)``, 来源体素的完整图概率.
        - source_probability_mean: float32 ``(N_blob,)``, 各区域平均概率.
        - voxel_count: int32 ``(N_blob,)``, 各区域体素数.
        - fits_centered_box: bool ``(N_blob,)``, 包围盒是否可被合法 80³ BOX 容纳.
        - centered_box_start_zyx: int32 ``(N_blob, 3)``, 合法 BOX 起点; 不可容纳时为 ``-1``.
        - source_threshold_value: float32 ``(1,)``, 本次阈值.

    区域按平均概率降序, 再按最小完整图 C-order 线性索引升序. 本函数不按
    ``min_voxels`` 删除区域.
    """

    probability = np.asarray(probability_map, dtype=np.float32)
    labels, count = ndimage.label(
        probability >= np.float32(threshold),
        structure=np.ones((3, 3, 3), dtype=np.uint8),
    )
    records: list[tuple[float, int, np.ndarray, np.ndarray, bool, np.ndarray]] = []
    full_shape = np.asarray(probability.shape, dtype=np.int64)
    linear_positive = np.flatnonzero(labels.reshape(-1)).astype(np.int64)
    positive_label = labels.reshape(-1)[linear_positive]
    label_order = np.argsort(positive_label, kind="stable")
    linear_by_label = linear_positive[label_order]
    label_counts = np.bincount(
        positive_label,
        minlength=int(count) + 1,
    )[1:]
    label_offsets = np.concatenate(
        (np.zeros(1, dtype=np.int64), np.cumsum(label_counts, dtype=np.int64))
    )
    probability_flat = probability.reshape(-1)
    for label_id in range(1, int(count) + 1):
        begin = int(label_offsets[label_id - 1])
        end = int(label_offsets[label_id])
        linear = linear_by_label[begin:end]
        coordinates = np.column_stack(
            np.unravel_index(linear, probability.shape)
        ).astype(np.int32)
        values = probability_flat[linear].astype(np.float32)
        minimum = coordinates.min(axis=0).astype(np.int64)
        maximum = coordinates.max(axis=0).astype(np.int64)
        requested = np.rint(
            coordinates.astype(np.float64).mean(axis=0) + 0.5 - 40.0
        ).astype(np.int64)
        lowest_start = np.maximum(0, maximum - 79)
        highest_start = np.minimum(minimum, full_shape - 80)
        fits = bool(np.all(lowest_start <= highest_start))
        start = np.clip(requested, lowest_start, highest_start) if fits else requested
        source_mean = np.float32(values.mean(dtype=np.float64))
        records.append(
            (
                float(source_mean),
                int(linear[0]),
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


# ================================================================================================


def publish_probability_blobs(
    paths: Stage1ArtifactPaths,
    role: str,
    threshold: float,
    probability_map: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """发布 F1 或 F3 的全部连通区域并返回同一数组映射.

    ``probability_map`` 为完整图 ``float32 (D, H, W)``. GPU 流水刚生成完整图时,
    调用者直接传入该数组, 使 CPU 连通区域提取与概率 NPZ 压缩并行. 复用已有
    概率产物时传入 ``None``, 本函数从 ``probability_map.npz`` 读取同名字段.
    """

    probability = (
        load_stage1_npz(
            paths.artifact("probability"),
            ("probability_map",),
        )["probability_map"]
        if probability_map is None
        else np.asarray(probability_map, dtype=np.float32)
    )
    arrays = extract_probability_blobs(probability, threshold)
    publish_stage1_artifact(
        paths.artifact(role),
        arrays,
        paths.complete(role),
    )
    return arrays
