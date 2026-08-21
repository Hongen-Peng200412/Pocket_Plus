# -*- coding: utf-8 -*-
"""编排 Stage1 V3 的五个可独立提交阶段.

主要入口依次是 :func:`run_probability_stage`, :func:`run_blobs_stage`,
:func:`run_centered_stage`, :func:`run_tune_stage` 和 :func:`run_evaluate_stage`.
前三个入口发布逐 PDB NPZ 与角色完成标记; tune 发布独立选择参数 JSON;
evaluate 发布逐 PDB 评估 NPZ, 数据划分 JSONL 与指标 JSON. 本模块直接连接
科学函数与文件发布, 不建立额外 workflow, 身份对象或 producer 包装层.

probability NPZ 的三个字段由 :func:`run_probability_stage` 列出; blobs 字段
沿用 :func:`extract_probability_blobs`; centered 字段沿用
:func:`infer_centered_boxes` 与 :func:`pack_centered_entries`; selection 字段沿用
:func:`tune_centered_selection`; 评估字段沿用 :func:`evaluate_centered_pdb` 和
:func:`aggregate_stage1_metrics`.
"""

from __future__ import annotations

from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .artifacts import (
    Stage1ArtifactPaths,
    f_alpha_tag,
    load_stage1_npz,
    publish_stage1_artifact,
    publish_stage1_json,
    publish_stage1_jsonl,
)
from .blobs import extract_probability_blobs
from .calibration import calibrate_semantic_thresholds, tune_centered_selection
from .centered import infer_centered_boxes
from .evaluation import (
    aggregate_stage1_metrics,
    evaluate_centered_pdb,
    load_occurrence_voxels,
)
from .full_map import infer_full_map
from .scoring import score_centered_candidates


CENTERED_BLOB_LIMIT = 1000

_CENTERED_BLOB_FIELDS = (
    "blob_index",
    "voxel_offsets",
    "voxel_index_global_zyx",
    "source_probability",
    "source_probability_mean",
    "voxel_count",
    "fits_centered_box",
    "centered_box_start_zyx",
    "source_threshold_value",
)


# ================================================================================================


def run_probability_stage(
    config: Any,
    dataset: Any,
    collator: Any,
    wrapper: Any,
    device: str,
    producer: str,
    split: str,
    pdb_ids: Sequence[str],
    output_root: str | Path,
    overwrite: bool,
) -> None:
    """生成并异步发布一个 PDB 清单的完整图概率.

    `dataset`, `collator` 和 `wrapper` 直接传给 `infer_full_map()`. `config.window`
    显式提供 stride, Gaussian sigma, batch, 线程, 预取, 混合精度和融合队列.
    每个 PDB 发布 `probability_map.npz`, `geometry.json`, `performance.json` 和
    `status/probability/_COMPLETE`. 默认跳过已有完成标记; `overwrite=True` 只
    撤销并重算 probability 阶段. 概率 NPZ 压缩与下一个 PDB 的 GPU 前向重叠.

    输入参数:
        - config: OmegaConf 配置, 读取 `window`, `publish_workers` 和 `pending_probability_pdbs`.
        - dataset: Stage1Dataset, 以 `full_map` 模式物化当前 PDB 的滑窗请求.
        - collator: `dataset.collate_fn`, 把相邻滑窗拼成模型批次.
        - wrapper: 已恢复权重并移到 `device` 的 Stage1 模型包装器.
        - device: str, 模型前向设备.
        - producer: str, 当前模型产物目录名.
        - split: str, 当前 PDB 清单的数据划分名.
        - pdb_ids: Sequence[str], 按执行顺序排列的小写 PDB 标识.
        - output_root: str | Path, Stage1 V3 产物根目录.
        - overwrite: bool, 是否重算已有 probability 完成标记的 PDB.

    本阶段成功时无返回值. 每个未跳过 PDB 的完成标记只在概率 NPZ 已原子替换后发布.

    正式产物字段:
        - probability_map: float32 `(D, H, W)`, 完整图 ZYX 配体概率.
        - origin_xyz: float32 `(3,)`, 完整图世界 XYZ 角点坐标, 单位 Å.
        - voxel_size_xyz: float32 `(3,)`, 世界 XYZ 体素尺寸, 单位 Å/voxel.
        - geometry.full_shape_zyx: int[3], 完整图 ZYX 形状.
        - geometry.origin_xyz: float[3], 与 NPZ `origin_xyz` 相同.
        - geometry.voxel_size_xyz: float[3], 与 NPZ `voxel_size_xyz` 相同.
        - geometry.window_shape_zyx: int[3], 固定为 `[80, 80, 80]`.
        - geometry.stride_zyx: int[3], 当前配置的完整图 ZYX 步长.
        - geometry.gaussian_sigma: float, 归一化滑窗 Gaussian 标准差.
        - geometry.window_count: int, 当前 PDB 实际前向窗口数.
        - performance.wall_seconds: float, 当前 PDB 完整图推理墙钟秒数.
        - performance.materialize_wait_seconds: float, 等待 CPU 请求物化的累计秒数.
        - performance.fusion_wait_seconds: float, 等待 CPU 有序融合的累计秒数.
    """

    def publish_probability(
        paths: Stage1ArtifactPaths,
        arrays: Mapping[str, np.ndarray],
        geometry: Mapping[str, object],
        performance: Mapping[str, object],
    ) -> None:
        """在线程池中发布一个 PDB 的几何, 性能, 概率 NPZ 和完成标记.

        `paths` 决定同一 PDB 的全部目标路径. `arrays` 是概率 NPZ 的三个正式
        数组 `probability_map`, `origin_xyz`, `voxel_size_xyz`. `geometry` 保存
        完整图形状, 世界几何, 80³ 窗口, stride, Gaussian sigma 和窗口数;
        `performance` 保存墙钟, 物化等待和融合等待秒数. 成功时无返回值.
        """

        publish_stage1_json(paths.pdb_root / "probability" / "geometry.json", geometry)
        publish_stage1_json(
            paths.pdb_root / "status" / "probability" / "performance.json",
            performance,
        )
        publish_stage1_artifact(
            paths.artifact("probability"), arrays, paths.complete("probability")
        )

    pending: deque[Future[None]] = deque()
    with ThreadPoolExecutor(
        max_workers=int(config.publish_workers),
        thread_name_prefix="stage1-probability-publish",
    ) as publisher:
        for pdb_id in pdb_ids:
            paths = Stage1ArtifactPaths(output_root, producer, split, pdb_id)
            if paths.complete("probability").is_file() and not overwrite:
                continue
            paths.complete("probability").unlink(missing_ok=True)
            result = infer_full_map(
                dataset=dataset,
                collator=collator,
                wrapper=wrapper,
                pdb_id=pdb_id,
                device=device,
                stride_zyx=config.window.stride_zyx,
                sigma=float(config.window.gaussian_sigma),
                window_batch_size=int(config.window.batch_size),
                window_workers=int(config.window.workers),
                prefetch_batches=int(config.window.prefetch_batches),
                precision=str(config.window.precision),
                pending_fusion_batches=int(config.window.pending_fusion_batches),
            )
            arrays = {
                "probability_map": result.probability_map.astype(
                    np.float32, copy=False
                ),
                "origin_xyz": result.origin_xyz.astype(np.float32, copy=False),
                "voxel_size_xyz": result.voxel_size_xyz.astype(np.float32, copy=False),
            }
            geometry = {
                "full_shape_zyx": [
                    int(value) for value in result.probability_map.shape
                ],
                "origin_xyz": [float(value) for value in result.origin_xyz],
                "voxel_size_xyz": [float(value) for value in result.voxel_size_xyz],
                "window_shape_zyx": [80, 80, 80],
                "stride_zyx": [int(value) for value in config.window.stride_zyx],
                "gaussian_sigma": float(config.window.gaussian_sigma),
                "window_count": int(result.window_count),
            }
            performance = {
                "wall_seconds": float(result.wall_seconds),
                "materialize_wait_seconds": float(result.materialize_wait_seconds),
                "fusion_wait_seconds": float(result.fusion_wait_seconds),
            }
            pending.append(
                publisher.submit(
                    publish_probability, paths, arrays, geometry, performance
                )
            )
            if len(pending) >= int(config.pending_probability_pdbs):
                pending.popleft().result()
        while pending:
            pending.popleft().result()


def run_blobs_stage(
    config: Any,
    data_root: str | Path | None,
    producer: str,
    split: str,
    pdb_ids: Sequence[str],
    output_root: str | Path,
    alpha: float,
    semantic_threshold: float | None,
    fit_semantic: bool,
    overwrite: bool,
) -> dict[str, object] | None:
    """冻结或读取语义阈值, 再并行发布动态 F-alpha blobs.

    `fit_semantic=True` 时逐 PDB 读取 probability 与 `union_mask.npy`, 按
    calibration 全集 micro F-alpha 写出 `calibration/F{alpha}_semantic.json`
    和 `F{alpha}_semantic_scan.npz`; `data_root` 此时是 Stage1 V3 数据根目录.
    其他调用直接使用显式 `semantic_threshold`. 每个 PDB 的 blobs 文件保存
    阈值下全部 26 邻域连通区域, 不应用最小体素数. 默认跳过已有角色完成标记;
    `overwrite=True` 只重算当前 blobs 角色. 返回值只在本次拟合语义阈值时存在.

    输入参数:
        - config: OmegaConf 配置, 读取语义阈值网格和 blob CPU 线程数.
        - data_root: str | Path | None, 语义拟合时读取 `union_mask.npy` 的数据根目录.
        - producer: str, 当前模型产物目录名.
        - split: str, 当前 PDB 清单的数据划分名.
        - pdb_ids: Sequence[str], 按执行顺序排列的小写 PDB 标识.
        - output_root: str | Path, 已有 probability 和新增 blobs 的共同根目录.
        - alpha: float, 路径标签和语义 F-alpha 使用的正浮点参数.
        - semantic_threshold: float | None, 不拟合时直接使用的概率阈值.
        - fit_semantic: bool, 是否从当前完整清单拟合语义阈值.
        - overwrite: bool, 是否重算已有 blobs 完成标记的 PDB.

    返回值:
        - semantic_result: dict[str, object] | None, 本次拟合的阈值摘要; 直接使用已有阈值时为 None.
    """

    alpha_tag = f_alpha_tag(alpha)
    blob_role = f"{alpha_tag}_blobs"
    semantic_result: dict[str, object] | None = None
    threshold = semantic_threshold
    if fit_semantic:
        probability_and_target = (
            (
                load_stage1_npz(
                    Stage1ArtifactPaths(output_root, producer, split, pdb_id).artifact(
                        "probability"
                    ),
                    ("probability_map",),
                )["probability_map"],
                np.load(
                    Path(data_root) / "density" / pdb_id / "union_mask.npy",
                    mmap_mode="r",
                    allow_pickle=False,
                )[0],
            )
            for pdb_id in pdb_ids
        )
        semantic_result = calibrate_semantic_thresholds(
            probability_and_target,
            denominator=int(config.calibration.semantic_denominator),
            alpha=float(alpha),
        )
        scan = semantic_result.pop("scan")
        calibration_root = Path(output_root) / producer / "calibration"
        publish_stage1_artifact(
            calibration_root / f"{alpha_tag}_semantic_scan.npz", scan, None
        )
        publish_stage1_json(
            calibration_root / f"{alpha_tag}_semantic.json", semantic_result
        )
        threshold = float(semantic_result["threshold_value"])

    def publish_blobs(pdb_id: str) -> None:
        """在线程池中读取一个完整概率图并发布其动态 F-alpha 连通区域.

        `pdb_id` 是当前小写 PDB 标识. 函数捕获已冻结的路径参数, 角色和阈值,
        成功时无返回值; blobs 完成标记只在 NPZ 已原子替换后发布.
        """

        paths = Stage1ArtifactPaths(output_root, producer, split, pdb_id)
        probability = load_stage1_npz(
            paths.artifact("probability"), ("probability_map",)
        )["probability_map"]
        arrays = extract_probability_blobs(probability, float(threshold))
        publish_stage1_artifact(
            paths.artifact(blob_role), arrays, paths.complete(blob_role)
        )

    pending: deque[Future[None]] = deque()
    with ThreadPoolExecutor(
        max_workers=int(config.blob_workers),
        thread_name_prefix="stage1-blobs",
    ) as executor:
        for pdb_id in pdb_ids:
            paths = Stage1ArtifactPaths(output_root, producer, split, pdb_id)
            if paths.complete(blob_role).is_file() and not overwrite:
                continue
            paths.complete(blob_role).unlink(missing_ok=True)
            pending.append(executor.submit(publish_blobs, pdb_id))
            if len(pending) >= int(config.blob_workers):
                pending.popleft().result()
        while pending:
            pending.popleft().result()
    return semantic_result


def run_centered_stage(
    config: Any,
    dataset: Any,
    collator: Any,
    wrapper: Any,
    device: str,
    producer: str,
    split: str,
    pdb_ids: Sequence[str],
    output_root: str | Path,
    alpha: float,
    forward_min_voxels: int | None,
    selection: Mapping[str, object] | None,
    overwrite: bool,
    score_only: bool,
) -> None:
    """生成动态 F-alpha centered, 或只更新已有文件的选择字段.

    正常模式读取 `F{alpha}_blobs.npz`, 对 `fits_centered_box=True` 且来源体素数
    达到 `forward_min_voxels` 的候选执行完整模型前向. 所有 producer 保存共同
    字段, `voxel_final`, auxiliary 和三张 48³ 稠密数组; Find 另外保存 A/P 表. 来源
    `blob_index` 数量严格大于 `CENTERED_BLOB_LIMIT` 时只写 `_BLOB_EXCEED` 并
    跳过当前 PDB. `selection` 存在时首次发布即写入 `score` 与 `selected`.

    `score_only=True` 时不读取 Dataset 或 wrapper, 只重用已有 centered 数组并
    替换 `score` 与 `selected`; 其他字段, 候选轴和 offsets 保持不变. 默认正常
    模式跳过已有完成标记; `overwrite=True` 只重跑 centered 阶段.

    输入参数:
        - config: OmegaConf 配置, 读取 centered 批次, 线程, 预取和发布队列参数.
        - dataset: Stage1Dataset | None, 正常模式使用; score-only 显式传入 None.
        - collator: callable | None, 正常模式使用的 Dataset 批次拼装函数.
        - wrapper: Stage1 模型包装器 | None, 正常模式执行完整前向.
        - device: str, 正常模式的模型前向设备.
        - producer: str, 决定产物目录与 U-Net/Find 字段集合.
        - split: str, 当前 PDB 清单的数据划分名.
        - pdb_ids: Sequence[str], 按执行顺序排列的小写 PDB 标识.
        - output_root: str | Path, probability, blobs 和 centered 的共同根目录.
        - alpha: float, 来源 blobs 与目标 centered 的 F-alpha 参数.
        - forward_min_voxels: int | None, 正常模式进入 GPU 的来源体素数下限.
        - selection: Mapping[str, object] | None, 可选 basic 或 Gaussian 冻结参数; 分别含固定 `prefiltered_min_voxel` 和搜索所得 `min_voxels`.
        - overwrite: bool, 正常模式是否重算已有 centered 完成标记.
        - score_only: bool, 是否只替换已有 centered 的 `score` 和 `selected`.

    本阶段成功时无返回值. 来源 blob 数严格大于 1000 的 PDB 只发布
    `_BLOB_EXCEED`, 不发布 centered NPZ 或 `_COMPLETE`.
    """

    alpha_tag = f_alpha_tag(alpha)
    blob_role = f"{alpha_tag}_blobs"
    centered_role = f"{alpha_tag}_centered"
    if score_only:
        for pdb_id in pdb_ids:
            paths = Stage1ArtifactPaths(output_root, producer, split, pdb_id)
            if (
                not paths.artifact(centered_role).is_file()
                and paths.blob_exceed(centered_role).is_file()
            ):
                print(f"{pdb_id}: centered skipped because _BLOB_EXCEED")
                continue
            arrays = load_stage1_npz(paths.artifact(centered_role), None)
            score = score_centered_candidates(
                arrays,
                score_mode=str(selection["score_mode"]),
                score_parameters=selection["score_parameters"],
            )
            arrays["score"] = np.asarray(score, dtype=np.float32)
            voxel_count = np.diff(np.asarray(arrays["voxel_offsets"], dtype=np.int64))
            arrays["selected"] = (
                (arrays["score"] >= np.float32(selection["score_threshold"]))
                & (voxel_count >= int(selection["prefiltered_min_voxel"]))
                & (voxel_count >= int(selection["min_voxels"]))
            )
            paths.complete(centered_role).unlink(missing_ok=True)
            publish_stage1_artifact(
                paths.artifact(centered_role), arrays, paths.complete(centered_role)
            )
        return

    def publish_centered(
        paths: Stage1ArtifactPaths,
        packed: Future[dict[str, np.ndarray]],
        performance: Mapping[str, float | int],
    ) -> dict[str, np.ndarray]:
        """在线程池中等待字段打包, 写入可选选择字段并发布 centered 文件.

        `paths` 和 `performance` 属于同一 PDB. `packed` 是 centered CPU 整理
        线程返回的 Future; 其字段完整遵循 :func:`pack_centered_entries`.
        `selection` 存在时新增 float32 `score (N_candidate,)` 和 bool
        `selected (N_candidate,)`. 返回值是已发布的数组映射, 供外层队列传播
        线程异常.
        """

        arrays = packed.result()
        if selection is not None:
            score = score_centered_candidates(
                arrays,
                score_mode=str(selection["score_mode"]),
                score_parameters=selection["score_parameters"],
            )
            arrays["score"] = np.asarray(score, dtype=np.float32)
            voxel_count = np.diff(np.asarray(arrays["voxel_offsets"], dtype=np.int64))
            arrays["selected"] = (
                (arrays["score"] >= np.float32(selection["score_threshold"]))
                & (voxel_count >= int(selection["prefiltered_min_voxel"]))
                & (voxel_count >= int(selection["min_voxels"]))
            )
        publish_stage1_json(
            paths.pdb_root / "status" / centered_role / "performance.json",
            performance,
        )
        publish_stage1_artifact(
            paths.artifact(centered_role), arrays, paths.complete(centered_role)
        )
        return arrays

    pending: deque[Future[dict[str, np.ndarray]]] = deque()
    with ThreadPoolExecutor(
        max_workers=int(config.publish_workers),
        thread_name_prefix="stage1-centered-publish",
    ) as publisher:
        for pdb_id in pdb_ids:
            paths = Stage1ArtifactPaths(output_root, producer, split, pdb_id)
            if paths.complete(centered_role).is_file() and not overwrite:
                continue
            paths.complete(centered_role).unlink(missing_ok=True)
            blobs = load_stage1_npz(paths.artifact(blob_role), _CENTERED_BLOB_FIELDS)
            source_blob_count = int(np.asarray(blobs["blob_index"]).size)
            if source_blob_count > CENTERED_BLOB_LIMIT:
                publish_stage1_json(
                    paths.blob_exceed(centered_role),
                    {
                        "pdb_id": pdb_id,
                        "centered_role": centered_role,
                        "source_blob_count": source_blob_count,
                        "limit": CENTERED_BLOB_LIMIT,
                    },
                )
                continue
            probability = load_stage1_npz(
                paths.artifact("probability"),
                ("probability_map", "origin_xyz", "voxel_size_xyz"),
            )
            packed, performance = infer_centered_boxes(
                dataset=dataset,
                collator=collator,
                wrapper=wrapper,
                pdb_id=pdb_id,
                producer=producer,
                blobs=blobs,
                full_probability=probability["probability_map"],
                origin_xyz=probability["origin_xyz"],
                voxel_size_xyz=probability["voxel_size_xyz"],
                device=device,
                precision=str(config.centered.precision),
                centered_batch_size=int(config.centered.batch_size),
                centered_workers=int(config.centered.workers),
                prefetch_batches=int(config.centered.prefetch_batches),
                pending_cpu_batches=int(config.centered.pending_cpu_batches),
                forward_min_voxels=int(forward_min_voxels),
                packer=publisher,
            )
            pending.append(
                publisher.submit(publish_centered, paths, packed, performance)
            )
            if len(pending) >= int(config.pending_centered_pdbs):
                pending.popleft().result()
        while pending:
            pending.popleft().result()


def run_tune_stage(
    config: Any,
    data_root: str | Path,
    producer: str,
    split: str,
    pdb_ids: Sequence[str],
    output_root: str | Path,
    alpha: float,
    score_mode: str,
    objective_beta: float,
    prefiltered_min_voxel: int,
) -> dict[str, object]:
    """从 blobs 或 centered 事实冻结独立的 basic/Gaussian 选择参数.

    `score_mode='basic'` 读取动态 blobs, 包括 `fits_centered_box=False` 的区域.
    两种模式都在尝试任何参数之前固定 `prefiltered_min_voxel`; 体素数低于该值
    的候选在所有参数组合中保持未入选. `score_mode='gaussian'` 读取动态 centered
    的 A 原子表; 只有 `_BLOB_EXCEED` 而没有 centered 的 PDB 被直接跳过并在
    标准输出说明原因. 目标是 semantic, coverage@0.3 和 one-to-one@0.3 三项
    micro F-beta 之和. 返回映射同时原子发布到 `calibration/F{alpha}_{mode}.json`.

    输入参数:
        - config: OmegaConf 配置, 读取 Gaussian 网格, 最小体素数和评估阈值.
        - data_root: str | Path, 读取每个 PDB `ligand_area.npz` 的数据根目录.
        - producer: str, 当前模型产物目录名.
        - split: str, 必须完整消费的数据划分名.
        - pdb_ids: Sequence[str], 按评估顺序排列的小写 PDB 标识.
        - output_root: str | Path, blobs/centered 与 calibration JSON 的共同根目录.
        - alpha: float, 输入产物和输出参数文件使用的 F-alpha 参数.
        - score_mode: str, `basic` 或 `gaussian`.
        - objective_beta: float, 三项评估目标共同使用的 F-beta 参数.
        - prefiltered_min_voxel: int, tune 前固定的来源 blob 体素数下限, 包含端点; 与搜索列表中的 `min_voxels` 相互独立.

    返回值:
        - selection: dict[str, object], 字段完整遵循 :func:`tune_centered_selection`; pipeline 只在顶层增加 float `alpha`.
    """

    alpha_tag = f_alpha_tag(alpha)
    blob_role = f"{alpha_tag}_blobs"
    centered_role = f"{alpha_tag}_centered"
    items: list[tuple[str, Mapping[str, np.ndarray]]] = []
    ground_truth: dict[str, tuple[np.ndarray, Sequence[np.ndarray], Sequence[int]]] = {}
    for pdb_id in pdb_ids:
        paths = Stage1ArtifactPaths(output_root, producer, split, pdb_id)
        if score_mode == "basic":
            blobs = load_stage1_npz(
                paths.artifact(blob_role),
                (
                    "blob_index",
                    "source_probability_mean",
                    "voxel_offsets",
                    "voxel_index_global_zyx",
                ),
            )
            candidate_count = int(np.asarray(blobs["blob_index"]).size)
            candidate = {
                "source_blob_index": np.asarray(blobs["blob_index"], dtype=np.int32),
                "source_probability_mean": np.asarray(
                    blobs["source_probability_mean"], dtype=np.float32
                ),
                "voxel_offsets": np.asarray(blobs["voxel_offsets"], dtype=np.int64),
                "voxel_index_local_zyx": np.asarray(
                    blobs["voxel_index_global_zyx"], dtype=np.int32
                ),
                "box_start_zyx": np.zeros((candidate_count, 3), dtype=np.int32),
            }
        else:
            if (
                not paths.artifact(centered_role).is_file()
                and paths.blob_exceed(centered_role).is_file()
            ):
                print(f"{pdb_id}: tune skipped because _BLOB_EXCEED")
                continue
            candidate = load_stage1_npz(
                paths.artifact(centered_role),
                (
                    "source_blob_index",
                    "source_probability_mean",
                    "voxel_offsets",
                    "voxel_index_local_zyx",
                    "box_start_zyx",
                    "A_offsets",
                    "A_coord_local_xyz",
                    "A_probability",
                    "voxel_size_world",
                ),
            )
        items.append((pdb_id, candidate))
        ground_truth[pdb_id] = load_occurrence_voxels(
            Path(data_root) / "density" / pdb_id / "ligand_area.npz"
        )
    selection = tune_centered_selection(
        centered_items=items,
        ground_truth_by_pdb=ground_truth,
        score_mode=score_mode,
        score_parameter_grid=(
            config.calibration.gaussian_grid if score_mode == "gaussian" else None
        ),
        refinement_multipliers=(
            config.calibration.gaussian_refinement if score_mode == "gaussian" else None
        ),
        prefiltered_min_voxel=int(prefiltered_min_voxel),
        min_voxel_values=config.calibration.min_voxel_values,
        objective_beta=float(objective_beta),
        coverage_thresholds=config.evaluation.coverage_thresholds,
        topk_values=config.evaluation.topk_values,
    )
    selection = {"alpha": float(alpha), **selection}
    publish_stage1_json(
        Path(output_root) / producer / "calibration" / f"{alpha_tag}_{score_mode}.json",
        selection,
    )
    return selection


def run_evaluate_stage(
    config: Any,
    data_root: str | Path,
    producer: str,
    split: str,
    pdb_ids: Sequence[str],
    output_root: str | Path,
    alpha: float,
    artifact: str,
    evaluation_name: str,
    selection: Mapping[str, object] | None,
) -> dict[str, object]:
    """按显式候选范围和评估名称发布 Stage1 评估事实.

    `selection` 为参数映射时按 basic 或 Gaussian 参数重算 `score` 与
    `selected`; 显式为 ``None`` 时不做二次打分, 以来源概率均值稳定排序并把
    当前 artifact 中的全部候选纳入指标. blobs 使用完整图稀疏坐标,
    centered 使用 80³ BOX 局部坐标. centered 文件因 `_BLOB_EXCEED` 缺失时
    跳过该 PDB 并在标准输出说明原因; 不建立额外跳过清单或完成状态.

    输入参数:
        - config: OmegaConf 配置, 读取 coverage 阈值和 top-K 列表.
        - data_root: str | Path, 读取每个 PDB `ligand_area.npz` 的数据根目录.
        - producer: str, 当前模型产物目录名.
        - split: str, 必须完整消费的数据划分名.
        - pdb_ids: Sequence[str], 按评估顺序排列的小写 PDB 标识.
        - output_root: str | Path, 输入候选和输出 evaluation 的共同根目录.
        - alpha: float, 用于定位输入候选的 F-alpha 参数.
        - artifact: str, `blobs` 或 `centered`.
        - evaluation_name: str, 本次评估的显式文件名, 用于让不同参数结果并存.
        - selection: Mapping[str, object] | None, 已冻结的评分参数与体素门槛; ``None`` 表示纳入 artifact 中的全部候选.

    返回值:
        - global_metrics: dict[str, object], 字段完整遵循 :func:`aggregate_stage1_metrics`, 对未跳过 PDB 聚合.
    """

    alpha_tag = f_alpha_tag(alpha)
    score_mode = (
        "all_candidates" if selection is None else str(selection["score_mode"])
    )
    role = f"{alpha_tag}_{artifact}"
    evaluations = []
    jsonl_rows = []
    for pdb_id in pdb_ids:
        paths = Stage1ArtifactPaths(output_root, producer, split, pdb_id)
        if (
            artifact == "centered"
            and not paths.artifact(role).is_file()
            and paths.blob_exceed(role).is_file()
        ):
            print(f"{pdb_id}: evaluate skipped because _BLOB_EXCEED")
            continue
        if artifact == "blobs":
            arrays = load_stage1_npz(
                paths.artifact(role),
                (
                    "blob_index",
                    "source_probability_mean",
                    "voxel_offsets",
                    "voxel_index_global_zyx",
                ),
            )
            candidate_count = int(np.asarray(arrays["blob_index"]).size)
            candidate = {
                "source_blob_index": np.asarray(arrays["blob_index"], dtype=np.int32),
                "source_probability_mean": np.asarray(
                    arrays["source_probability_mean"], dtype=np.float32
                ),
                "voxel_offsets": np.asarray(arrays["voxel_offsets"], dtype=np.int64),
                "voxel_index_local_zyx": np.asarray(
                    arrays["voxel_index_global_zyx"], dtype=np.int32
                ),
                "box_start_zyx": np.zeros((candidate_count, 3), dtype=np.int32),
            }
        else:
            fields = [
                "source_blob_index",
                "source_probability_mean",
                "voxel_offsets",
                "voxel_index_local_zyx",
                "box_start_zyx",
            ]
            if score_mode == "gaussian":
                fields.extend(
                    (
                        "A_offsets",
                        "A_coord_local_xyz",
                        "A_probability",
                        "voxel_size_world",
                    )
                )
            candidate = load_stage1_npz(paths.artifact(role), fields)
        # float32, (N_candidate,), 全候选模式的稳定排序值; 参数过滤模式会在下方替换.
        candidate["score"] = np.asarray(
            candidate["source_probability_mean"], dtype=np.float32
        )
        # bool, (N_candidate,), 全候选模式全部纳入; 参数过滤模式会在下方替换.
        candidate["selected"] = np.ones(candidate["score"].shape, dtype=np.bool_)
        if selection is not None:
            score = score_centered_candidates(
                candidate,
                score_mode=score_mode,
                score_parameters=selection["score_parameters"],
            )
            candidate["score"] = np.asarray(score, dtype=np.float32)
            voxel_count = np.diff(
                np.asarray(candidate["voxel_offsets"], dtype=np.int64)
            )
            candidate["selected"] = (
                (candidate["score"] >= np.float32(selection["score_threshold"]))
                & (voxel_count >= int(selection["prefiltered_min_voxel"]))
                & (voxel_count >= int(selection["min_voxels"]))
            )
        occurrence_id, occurrence_rows, full_shape = load_occurrence_voxels(
            Path(data_root) / "density" / pdb_id / "ligand_area.npz"
        )
        evaluation = evaluate_centered_pdb(
            pdb_id=pdb_id,
            centered=candidate,
            occurrence_id=occurrence_id,
            occurrence_voxel_zyx=occurrence_rows,
            full_shape_zyx=full_shape,
            coverage_thresholds=config.evaluation.coverage_thresholds,
            topk_values=config.evaluation.topk_values,
        )
        evaluations.append(evaluation)
        publish_stage1_artifact(
            paths.pdb_root / "evaluation" / f"{evaluation_name}.npz",
            {
                "coverage_thresholds": evaluation.coverage_thresholds,
                "topk_values": evaluation.topk_values,
                "occurrence_id": evaluation.occurrence_id,
                "source_blob_index": evaluation.source_blob_index,
                "candidate_score": evaluation.candidate_score,
                "candidate_selected": evaluation.candidate_selected,
                "intersections": evaluation.intersections,
                "pred_sizes": evaluation.pred_sizes,
                "gt_sizes": evaluation.gt_sizes,
                "candidate_semantic_tp": evaluation.candidate_semantic_tp,
                "semantic_tp": np.asarray(evaluation.semantic_tp, dtype=np.int64),
                "semantic_fp": np.asarray(evaluation.semantic_fp, dtype=np.int64),
                "semantic_fn": np.asarray(evaluation.semantic_fn, dtype=np.int64),
                "coverage_pred_hit_mask": evaluation.coverage_pred_hit_mask,
                "coverage_gt_hit_mask": evaluation.coverage_gt_hit_mask,
                "one_to_one_match_offsets": evaluation.one_to_one_match_offsets,
                "one_to_one_match_pred_index": evaluation.one_to_one_match_pred_index,
                "one_to_one_match_gt_index": evaluation.one_to_one_match_gt_index,
                "topk_winning_candidate_rank": evaluation.topk_winning_candidate_rank,
                "topk_winning_occurrence_index": evaluation.topk_winning_occurrence_index,
            },
            None,
        )
        single_metrics = aggregate_stage1_metrics(
            (evaluation,),
            coverage_thresholds=config.evaluation.coverage_thresholds,
            topk_values=config.evaluation.topk_values,
        )
        jsonl_rows.append({"pdb_id": pdb_id, **single_metrics})
    global_metrics = aggregate_stage1_metrics(
        evaluations,
        coverage_thresholds=config.evaluation.coverage_thresholds,
        topk_values=config.evaluation.topk_values,
    )
    evaluation_root = Path(output_root) / producer / split / "evaluation"
    publish_stage1_jsonl(evaluation_root / f"{evaluation_name}.jsonl", jsonl_rows)
    publish_stage1_json(
        evaluation_root / f"{evaluation_name}.metrics.json", global_metrics
    )
    return global_metrics
