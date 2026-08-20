# -*- coding: utf-8 -*-
"""执行一个 PDB 的 Stage1 V3 概率图与 centered 评分发布.

主要入口 :func:`produce_probability_map` 发布 `probability_map.npz`, `geometry.json`,
`performance.json` 和概率完成标记; :func:`produce_centered_role` 发布
`F1_basic.npz` 或 `F3_centered.npz`, centered 性能 JSON 和可选完成标记. 概率 NPZ
核心字段为 `probability_map`, `origin_xyz`, `voxel_size_xyz`; centered 字段由
``centered.pack_centered_entries`` 定义. 跨 PDB 调度由 ``workflow.py`` 直接编排.
"""

from __future__ import annotations

from concurrent.futures import Executor, Future
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
from .scoring import score_centered_candidates


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

    输入参数:
        - paths: 当前 producer, 数据划分和 PDB 的正式产物路径.
        - dataset: Stage1Dataset, 直接传给 :func:`infer_full_map`.
        - collator: Stage1Collator, 直接传给 :func:`infer_full_map`.
        - wrapper: Stage1 模型包装器, 直接传给 :func:`infer_full_map`.
        - device: 字符串, 模型前向设备.
        - window_config.stride_zyx: 三个整数, 完整图 ZYX 滑窗步长.
        - window_config.gaussian_sigma: float, 规范化滑窗坐标中的 Gaussian 标准差.
        - window_config.batch_size: int, 一次完整图模型前向的窗口数.
        - window_config.workers: int, CPU 窗口物化线程数.
        - window_config.prefetch_batches: int, 尚未送入模型的最大预取 batch 数.
        - window_config.precision: 字符串, GPU autocast 精度.
        - window_config.pending_fusion_batches: int, 尚未完成 CPU 融合的最大 batch 数.
        - publisher: 跨 PDB 共享的 CPU Executor, 用于概率 NPZ 压缩和四份文件发布.

    返回值:
        - arrays.probability_map: float32 `(D, H, W)`, 完整图 ZYX 配体概率.
        - arrays.origin_xyz: float32 `(3,)`, 完整图世界 XYZ 角点坐标, 单位 Å.
        - arrays.voxel_size_xyz: float32 `(3,)`, 世界 XYZ 体素尺寸, 单位 Å/voxel.
        - publish_future: `Future[None]`; 完成时概率几何, 性能, NPZ 和完成标记均已发布.

    ``Future`` 完成时, 窗口 ``geometry.json`` 和性能 JSON 已先发布, 科学 NPZ
    已原子替换, 最终 ``status/probability/_COMPLETE`` 已建立. 上述边界包含四个
    文件. NPZ 压缩由 ``publisher`` 执行,
    不阻塞下一个 PDB 的 GPU 前向.
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
        """在 publisher 线程中依次发布外层已冻结的几何, 性能, 概率 NPZ 和完成标记; 成功时返回 None."""

        publish_stage1_json(paths.pdb_root / "probability" / "geometry.json", geometry)
        publish_stage1_json(
            paths.pdb_root / "status" / "probability" / "performance.json",
            performance,
        )
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

    输入参数:
        - paths: 当前 producer, 数据划分和 PDB 的正式产物路径.
        - dataset: Stage1Dataset, 直接传给 :func:`infer_centered_boxes`.
        - collator: Stage1Collator, 直接传给 :func:`infer_centered_boxes`.
        - wrapper: Stage1 模型包装器, 直接传给 :func:`infer_centered_boxes`.
        - device: 字符串, 模型前向设备.
        - blobs.voxel_offsets: int64 `(N_blob + 1,)`, 同步切分 `voxel_index_global_zyx` 与 `source_probability`.
        - blobs.voxel_index_global_zyx: int32 `(L_voxel, 3)`, 完整图 ZYX 来源体素索引.
        - blobs.source_probability: float32 `(L_voxel,)`, 与来源体素逐项对齐的完整图概率.
        - blobs.source_probability_mean: float32 `(N_blob,)`, 每个来源 blob 的平均概率.
        - blobs.voxel_count: int32 `(N_blob,)`, 每个来源 blob 的体素数.
        - blobs.fits_centered_box: bool `(N_blob,)`, 是否存在可容纳来源 blob 的合法 80³ BOX.
        - blobs.centered_box_start_zyx: int32 `(N_blob, 3)`, 完整图 ZYX BOX 起点.
        - blobs.source_threshold_value: float32 `(1,)`, 当前角色的语义概率阈值.
        - centered_role: 字符串, `F1_basic` 或 `F3_centered`.
        - min_voxels: 整数, 进入 centered 推理的来源 blob 最小体素数, 包含端点.
        - centered_config.precision: 字符串, GPU autocast 精度.
        - centered_config.batch_size: int, 一次 centered 模型前向的候选数.
        - centered_config.workers: int, CPU centered 请求物化线程数.
        - centered_config.prefetch_batches: int, 尚未送入模型的最大预取 batch 数.
        - centered_config.pending_cpu_batches: int, 尚未完成 CPU 整理的最大 batch 数.
        - centered_config.forward: 字符串, `voxel_only` 或 `full`.
        - centered_config.save_voxel_final: bool, 是否保存 V 学习特征.
        - centered_config.save_dense48: bool, 是否读取完整概率图并保存三张 48³ 稠密数组.
        - blob_limit: 整数或 None; F3 记录 `_BLOB_EXCEED` 的显式候选数阈值, F1 传 None.
        - selection.score_mode: 字符串, `source_mean` 或 `find_gaussian`; selection 为 None 时不存在.
        - selection.score_parameters.tau_angstrom: float, Find Gaussian 距离标准差, 单位 Å.
        - selection.score_parameters.lambda_positive: float, Find Gaussian 正项系数.
        - selection.score_parameters.lambda_negative: float, Find Gaussian 负项系数.
        - selection.score_threshold: float, 最终候选分数下限, 包含端点.
        - selection.min_voxels: int, 最终来源 blob 最小体素数, 包含端点.
        - selection=None: calibration 搜索前的临时 centered 不评分且不建立完成标记; source_mean 模式的 `score_parameters` 是空映射.
        - publisher: 跨 PDB 共享的 CPU Executor, 用于字段打包, NPZ 压缩和文件发布.

    返回值:
        - publish_future: `Future[dict[str, np.ndarray]]`; 完成时 centered 数组已发布并作为结果返回.

    `selection=None` 用于 calibration 搜索前的临时候选
    归档, 只发布未评分 NPZ, 不建立完成标记. 传入冻结选择映射时, Future 在
    首次压缩前写入 `score` 与 `selected`.

    发布文件:
        - `centered/<centered_role>.npz`: 保存 :func:`pack_centered_entries` 定义的字段.
        - `status/<centered_role>/performance.json`: 保存 :func:`infer_centered_boxes` 返回的五个性能字段.
        - `status/<centered_role>/_COMPLETE`: 正式选择参数存在时由 :func:`publish_stage1_artifact` 建立.
        - `status/F3_centered/_BLOB_EXCEED`: eligible 数量严格大于 blob_limit 时建立; 该事实不终止生产.

    `_BLOB_EXCEED` 字段:
        - pdb_id: str, 当前小写 PDB 标识.
        - centered_role: str, 固定为 F3_centered.
        - eligible_blob_count: int, 满足 BOX 包络和当前 min_voxels 的候选数.
        - limit: int, 当前配置的 blob_limit.

    浮点字段在提交压缩前直接检查 NaN/Inf.
    """

    paths.complete(centered_role).unlink(missing_ok=True)
    eligible = np.asarray(blobs["fits_centered_box"], dtype=np.bool_) & (
        np.asarray(blobs["voxel_count"], dtype=np.int64) >= int(min_voxels)
    )
    if centered_role == "F3_centered" and int(eligible.sum()) > int(blob_limit):
        publish_stage1_json(
            paths.pdb_root / "status" / "F3_centered" / "_BLOB_EXCEED",
            {
                "pdb_id": paths.pdb_id,
                "centered_role": centered_role,
                "eligible_blob_count": int(eligible.sum()),
                "limit": int(blob_limit),
            },
        )
    elif centered_role == "F3_centered":
        (paths.pdb_root / "status" / "F3_centered" / "_BLOB_EXCEED").unlink(
            missing_ok=True
        )

    save_dense48 = bool(centered_config["save_dense48"])
    probability_fields = ["origin_xyz", "voxel_size_xyz"]
    if save_dense48:
        probability_fields.insert(0, "probability_map")
    probability = load_stage1_npz(paths.artifact("probability"), probability_fields)
    full_probability = probability["probability_map"] if save_dense48 else None
    packed, performance = infer_centered_boxes(
        dataset=dataset,
        collator=collator,
        wrapper=wrapper,
        pdb_id=paths.pdb_id,
        producer=paths.stage1_model_name,
        blobs=blobs,
        full_probability=full_probability,
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
        save_dense48=save_dense48,
        packer=publisher,
    )

    def publish() -> dict[str, np.ndarray]:
        """在 publisher 线程中等待外层 centered 打包, 写入冻结选择字段, 再发布性能, NPZ 和可选完成标记; 返回与 NPZ 一致的数组映射."""

        arrays = packed.result()
        if selection is not None:
            # 分数与体素数共同决定 selected, 两个阈值都包含端点.
            score = score_centered_candidates(
                arrays,
                score_mode=str(selection["score_mode"]),
                score_parameters=selection["score_parameters"],
            )
            arrays["score"] = np.asarray(score, dtype=np.float32)
            voxel_count = np.diff(np.asarray(arrays["voxel_offsets"], dtype=np.int64))
            arrays["selected"] = (
                arrays["score"] >= np.float32(selection["score_threshold"])
            ) & (voxel_count >= int(selection["min_voxels"]))
        for name, value in arrays.items():
            if np.issubdtype(value.dtype, np.floating) and not np.isfinite(value).all():
                raise ValueError(f"{paths.pdb_id}:{centered_role}:{name} 包含 NaN/Inf.")
        paths.complete(centered_role).unlink(missing_ok=True)
        publish_stage1_json(
            paths.pdb_root / "status" / centered_role / "performance.json",
            performance,
        )
        publish_stage1_artifact(
            paths.artifact(centered_role),
            arrays,
            None if selection is None else paths.complete(centered_role),
        )
        return arrays

    return publisher.submit(publish)
