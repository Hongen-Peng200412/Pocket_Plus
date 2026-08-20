# -*- coding: utf-8 -*-
"""编排 Stage1 V3 calibration 与冻结参数推理.

主要入口 :func:`run_calibration_workflow` 发布 `calibration/stage1_v3.json`,
`semantic_threshold_scan.npz`, `stage1_v3.metrics.json` 和 calibration 完成标记;
:func:`run_frozen_workflow` 使用冻结参数发布 validation 或 train 的逐 PDB 五类科学
NPZ 与评估文件. :func:`evaluate_and_publish_role` 保存候选/occurrence 交集 NPZ,
逐 PDB JSONL 和跨 PDB metrics JSON. 命令解析与 Dataset 构造留在 ``cli.py``.
"""

from __future__ import annotations

from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import numpy as np

from .artifacts import (
    Stage1ArtifactPaths,
    load_stage1_npz,
    publish_stage1_artifact,
    publish_stage1_json,
    publish_stage1_jsonl,
)
from .blobs import publish_probability_blobs
from .calibration import calibrate_semantic_thresholds, tune_centered_selection
from .evaluation import (
    aggregate_stage1_metrics,
    evaluate_centered_pdb,
    load_occurrence_voxels,
)
from .pipeline import (
    produce_centered_role,
    produce_probability_map,
)


_CENTERED_BLOB_FIELDS = (
    "voxel_offsets",
    "voxel_index_global_zyx",
    "source_probability",
    "source_probability_mean",
    "voxel_count",
    "fits_centered_box",
    "centered_box_start_zyx",
    "source_threshold_value",
)
_EVALUATION_CENTERED_FIELDS = (
    "source_blob_index",
    "score",
    "selected",
    "voxel_offsets",
    "voxel_index_local_zyx",
    "box_start_zyx",
)


# ================================================================================================


def evaluate_and_publish_role(
    config: Any,
    dataset_root: str | Path,
    output_root: str | Path,
    producer: str,
    split: str,
    pdb_ids: Sequence[str],
    centered_role: str,
) -> dict[str, object]:
    """读取一个冻结 centered 角色并发布逐 PDB 事实与数据划分汇总.

    输入参数:
        - config: 推理配置; 本入口读取 `evaluation.coverage_thresholds` 和 `evaluation.topk_values`.
        - dataset_root: Stage1 V3 数据根目录; 真实 occurrence 位于 `density/<pdb>/ligand_area.npz`.
        - output_root: 当前科学配置版本的推理产物根目录.
        - producer: 模型产物目录名, 如 `unet_c1` 或 `Find_1`.
        - split: 当前数据划分目录名, 如 `calibration`, `validation` 或 `train`.
        - pdb_ids: 有序且无重复的小写 PDB 标识序列.
        - centered_role: `F1_basic` 或 `F3_centered`.

    ``config.evaluation`` 必须提供有序 `coverage_thresholds` 和 `topk_values`;
    ``dataset_root/density/<pdb>/ligand_area.npz`` 提供真实 occurrence;
    ``output_root/producer/split/<pdb>`` 提供已冻结 centered NPZ 及其 `_COMPLETE`.

    输入的每个 PDB 必须已有带完成标记的冻结 centered NPZ. 逐 PDB 评估 NPZ
    保存候选与真实 occurrence 的交集矩阵, 双向覆盖命中掩码, 各阈值一对一
    匹配下标以及 top-K 获胜下标. 数据划分目录另存逐 PDB JSONL 和 micro/PDB
    等权 macro 指标; 缺少任一正式文件或完成标记时直接失败. 返回值是与
    `<centered_role>.metrics.json` 相同的跨 PDB 指标映射. 本入口生成:

    - `<pdb>/evaluation/<centered_role>.npz`: 保存 :class:`PdbEvaluation` 的正式落盘字段; 精确字段表见 BOX-level 数据契约第 9.1 节.
    - `<split>/evaluation/<centered_role>.jsonl`: 每个 PDB 一条 :func:`aggregate_stage1_metrics` 指标记录并增加 `pdb_id`.
    - `<split>/evaluation/<centered_role>.metrics.json`: 保存跨 PDB 指标映射.

    评估文件不是生产角色, 不建立 `_COMPLETE`.
    """

    evaluations = []
    jsonl_rows = []
    for pdb_id in pdb_ids:
        paths = Stage1ArtifactPaths(output_root, producer, split, pdb_id)
        if not paths.complete(centered_role).is_file():
            raise FileNotFoundError(paths.complete(centered_role))
        selected = load_stage1_npz(
            paths.artifact(centered_role), _EVALUATION_CENTERED_FIELDS
        )
        occurrence_id, occurrence_rows, full_shape = load_occurrence_voxels(
            Path(dataset_root) / "density" / pdb_id / "ligand_area.npz"
        )
        evaluation = evaluate_centered_pdb(
            pdb_id=pdb_id,
            centered=selected,
            occurrence_id=occurrence_id,
            occurrence_voxel_zyx=occurrence_rows,
            full_shape_zyx=full_shape,
            coverage_thresholds=config.evaluation.coverage_thresholds,
            topk_values=config.evaluation.topk_values,
        )
        evaluations.append(evaluation)
        publish_stage1_artifact(
            paths.pdb_root / "evaluation" / f"{centered_role}.npz",
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
    publish_stage1_jsonl(evaluation_root / f"{centered_role}.jsonl", jsonl_rows)
    publish_stage1_json(
        evaluation_root / f"{centered_role}.metrics.json", global_metrics
    )
    return global_metrics


def run_calibration_workflow(
    config: Any,
    dataset: Any,
    collator: Any,
    wrapper: Any,
    device: str,
    producer: str,
    split: str,
    pdb_ids: Sequence[str],
    output_root: str | Path,
    checkpoint_path: str,
) -> dict[str, object]:
    """生成 calibration 产物并最后发布 checkpoint 路径与冻结参数.

    输入参数:
        - config.window: 完整图滑窗字段映射, 字段契约由 `produce_probability_map` 定义.
        - config.roles.<role>.blob_role: 字符串, 当前角色的 blobs 产物名.
        - config.roles.<role>.centered_role: 字符串, 当前角色的 centered 产物名.
        - config.roles.<role>.semantic_role: 字符串, 当前角色使用的 F-beta 语义阈值名.
        - config.roles.<role>.min_voxel_values: 整数序列, 校准搜索的来源 blob 最小体素数轴.
        - config.roles.<role>.objective_beta: float, centered 校准目标使用的 beta.
        - config.roles.<role>.score_mode: producer 到 `source_mean` 或 `find_gaussian` 的映射.
        - config.roles.<role>.score_parameter_grid.tau_angstrom: Gaussian 距离标准差序列或 None.
        - config.roles.<role>.score_parameter_grid.lambda_positive: Gaussian 正项系数序列或 None.
        - config.roles.<role>.score_parameter_grid.lambda_negative: Gaussian 负项系数序列或 None.
        - config.roles.<role>.score_parameter_grid.gauss_score_min: Gaussian 分数下限序列或 None.
        - config.roles.<role>.blob_limit: int 或 None, 单个 PDB 允许进入 centered 的最大 blob 数.
        - config.roles.<role>.centered: centered 推理映射, 字段契约由 `produce_centered_role` 定义.
        - config.calibration.semantic_denominator: int, 完整图语义阈值网格分母.
        - config.calibration.gaussian_refinement.lambda: Find Gaussian 正负系数的第二阶段乘数序列.
        - config.calibration.gaussian_refinement.score_threshold: Find Gaussian 分数下限的第二阶段乘数序列.
        - config.evaluation.coverage_thresholds: 浮点数序列, 双向覆盖率阈值轴.
        - config.evaluation.topk_values: 整数序列, top-K 评估的 K 轴.
        - config.blob_workers: int, 跨 PDB 连通区域线程数.
        - config.publish_workers: int, NPZ 打包与发布线程数.
        - config.pending_probability_pdbs: int, 尚未完成概率发布的最大 PDB 数.
        - config.pending_centered_pdbs: int, 尚未完成 centered 发布的最大 PDB 数.
        - dataset: Stage1Dataset, 物化当前清单的完整图滑窗与 centered BOX.
        - collator: Stage1Collator, 拼装模型批次.
        - wrapper: 已加载 checkpoint 的 Stage1 模型包装器.
        - device: 字符串, 模型前向设备.
        - producer: 模型产物目录名, 如 `unet_c1` 或 `Find_1`.
        - split: calibration 使用的数据划分目录名.
        - pdb_ids: 有序且无重复的小写 calibration PDB 标识序列.
        - output_root: 当前科学配置版本的推理产物根目录.
        - checkpoint_path: 当前命令显式使用的 checkpoint 规范化绝对路径.

    ``dataset`` 必须能物化 ``pdb_ids`` 的完整图和 centered 请求. 同一 checkpoint
    的不同科学参数由不同 `output_root` 目录区分. 每次调用先撤销旧 calibration
    `_COMPLETE` 并重新计算所有 PDB 概率.

    概率 NPZ 压缩与下一个 PDB 的 GPU 滑窗并行, 同时只保留配置规定数量的
    待发布完整图. 语义直方图逐 PDB读取正式 NPZ. centered 阶段让 CPU blob
    future, GPU 前向和前一个 PDB 的 NPZ 压缩重叠. 校准参数, 完整语义扫描,
    汇总指标全部完成后, 最后发布 calibration ``_COMPLETE`` JSON. 校准搜索
    使用最小候选阈值生成临时 centered; 临时文件不保存 V 学习特征或 48³
    稠密数组. 冻结最终 `min_voxels` 后重新生成完整正式 centered, 因而低于
    最终阈值的 blob 只保留在 blobs 文件.

    返回字段与 `calibration/stage1_v3.json` 相同:
        - checkpoint_path: 字符串, 当前命令使用的 checkpoint 规范化绝对路径.
        - semantic: :func:`calibrate_semantic_thresholds` 返回值去掉 `scan` 后的语义阈值摘要.
        - roles.F1_basic.source_threshold: float, F1 blobs 使用的完整图概率阈值.
        - roles.F1_basic.selection: :func:`tune_centered_selection` 返回的基本模式冻结参数.
        - roles.F3_centered.source_threshold: float, F3 blobs 使用的完整图概率阈值.
        - roles.F3_centered.selection: :func:`tune_centered_selection` 返回的完整模式冻结参数.

    本入口另生成 `semantic_threshold_scan.npz`,
    `stage1_v3.metrics.json` 以及逐 PDB 五类科学文件, 性能 JSON 和评估文件.
    最后的 `calibration/_COMPLETE` 精确保存 `checkpoint_path` 和
    `result_scope='calibration_fitted'`; 任一前置发布失败都不会建立该标记.
    """

    calibration_root = Path(output_root) / producer / "calibration"
    (calibration_root / "_COMPLETE").unlink(missing_ok=True)
    # 第一阶段逐 PDB 运行 GPU 滑窗, 同时让前一 PDB 的概率 NPZ 在 CPU 线程压缩.
    probability_publish: deque[Future[None]] = deque()
    with ThreadPoolExecutor(
        max_workers=int(config.publish_workers),
        thread_name_prefix="stage1-publish",
    ) as publisher:
        for pdb_id in pdb_ids:
            paths = Stage1ArtifactPaths(output_root, producer, split, pdb_id)
            _, future = produce_probability_map(
                paths=paths,
                dataset=dataset,
                collator=collator,
                wrapper=wrapper,
                device=device,
                window_config=config.window,
                publisher=publisher,
            )
            probability_publish.append(future)
            if len(probability_publish) >= int(config.pending_probability_pdbs):
                probability_publish.popleft().result()
        while probability_publish:
            probability_publish.popleft().result()

    def probability_and_target() -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """惰性读取语义阈值校准所需的逐 PDB 完整图概率和真实配体并集.

        返回值:
            - probability: float32 `(D, H, W)`, 当前 PDB 的完整图 ZYX 配体概率.
            - target: bool `(D, H, W)`, 当前 PDB 的真实配体并集; True 表示属于至少一个 ligand occurrence, False 表示背景.

        读取顺序与 `pdb_ids` 一致, 因而 calibration 全部完整图不会同时常驻内存.
        """

        for pdb_id in pdb_ids:
            paths = Stage1ArtifactPaths(output_root, producer, split, pdb_id)
            probability = load_stage1_npz(
                paths.artifact("probability"), ("probability_map",)
            )["probability_map"]
            target = np.load(
                Path(dataset.root) / "density" / pdb_id / "union_mask.npy",
                mmap_mode="r",
                allow_pickle=False,
            )[0]
            yield probability, target

    # 概率阈值先于 blob 与 centered 参数冻结, F1 和 F3 共用同一次完整图扫描.
    semantic_with_scan = calibrate_semantic_thresholds(
        probability_and_target(),
        denominator=int(config.calibration.semantic_denominator),
        betas=(1.0, 3.0),
    )
    semantic_scan = semantic_with_scan["scan"]
    semantic = {
        name: value for name, value in semantic_with_scan.items() if name != "scan"
    }
    calibration_payload: dict[str, Any] = {
        "checkpoint_path": str(checkpoint_path),
        "semantic": semantic,
        "roles": {},
    }
    role_metrics: dict[str, object] = {}
    centered_pdb_ids_by_role: dict[str, tuple[str, ...]] = {}

    # 每个角色先按最宽松的 min_voxels 生成校准 centered, 供后续选择参数搜索.
    for role_name, role_config in config.roles.items():
        blob_role = str(role_config.blob_role)
        centered_role = str(role_config.centered_role)
        semantic_threshold = float(
            semantic["thresholds"][str(role_config.semantic_role)]["value"]
        )
        generation_min_voxels = min(
            int(value) for value in role_config.min_voxel_values
        )
        search_centered_config = dict(role_config.centered)
        search_centered_config["save_voxel_final"] = False
        search_centered_config["save_dense48"] = False
        centered_pdb_ids: list[str] = []
        centered_publish: deque[Future[dict[str, np.ndarray]]] = deque()
        with ThreadPoolExecutor(
            max_workers=int(config.blob_workers),
            thread_name_prefix="stage1-blobs",
        ) as blob_executor, ThreadPoolExecutor(
            max_workers=int(config.publish_workers),
            thread_name_prefix="stage1-publish",
        ) as publisher:
            blob_futures: deque[Future[dict[str, np.ndarray]]] = deque()
            next_blob_index = 0
            while next_blob_index < min(int(config.blob_workers), len(pdb_ids)):
                next_pdb_id = pdb_ids[next_blob_index]
                blob_futures.append(
                    blob_executor.submit(
                        publish_probability_blobs,
                        Stage1ArtifactPaths(output_root, producer, split, next_pdb_id),
                        blob_role,
                        semantic_threshold,
                        None,
                    )
                )
                next_blob_index += 1
            for pdb_id in pdb_ids:
                blobs = blob_futures.popleft().result()
                if next_blob_index < len(pdb_ids):
                    next_pdb_id = pdb_ids[next_blob_index]
                    blob_futures.append(
                        blob_executor.submit(
                            publish_probability_blobs,
                            Stage1ArtifactPaths(
                                output_root, producer, split, next_pdb_id
                            ),
                            blob_role,
                            semantic_threshold,
                            None,
                        )
                    )
                    next_blob_index += 1
                paths = Stage1ArtifactPaths(output_root, producer, split, pdb_id)
                produced = produce_centered_role(
                    paths=paths,
                    dataset=dataset,
                    collator=collator,
                    wrapper=wrapper,
                    device=device,
                    blobs=blobs,
                    centered_role=centered_role,
                    min_voxels=generation_min_voxels,
                    centered_config=search_centered_config,
                    blob_limit=(
                        None
                        if role_config.blob_limit is None
                        else int(role_config.blob_limit)
                    ),
                    selection=None,
                    publisher=publisher,
                )
                centered_pdb_ids.append(pdb_id)
                centered_publish.append(produced)
                if len(centered_publish) >= int(config.pending_centered_pdbs):
                    centered_publish.popleft().result()
            while centered_publish:
                centered_publish.popleft().result()
        centered_pdb_ids_by_role[str(role_name)] = tuple(centered_pdb_ids)

    for role_name, role_config in config.roles.items():
        centered_role = str(role_config.centered_role)
        centered_pdb_ids = centered_pdb_ids_by_role[str(role_name)]
        semantic_threshold = float(
            semantic["thresholds"][str(role_config.semantic_role)]["value"]
        )
        ground_truth = {
            pdb_id: load_occurrence_voxels(
                Path(dataset.root) / "density" / pdb_id / "ligand_area.npz"
            )
            for pdb_id in centered_pdb_ids
        }

        score_mode = str(role_config.score_mode[producer])
        centered_fields = (
            "source_blob_index",
            "source_probability_mean",
            "voxel_offsets",
            "voxel_index_local_zyx",
            "box_start_zyx",
        )
        if score_mode == "find_gaussian":
            centered_fields += (
                "A_offsets",
                "A_coord_local_xyz",
                "A_probability",
                "voxel_size_world",
            )

        def centered_items() -> Iterator[tuple[str, dict[str, np.ndarray]]]:
            """惰性读取当前角色校准搜索实际使用的逐 PDB centered 字段.

            返回值:
                - pdb_id: 字符串, 当前小写 PDB 标识.
                - centered.source_blob_index: int32 `(N_candidate,)`, 来源 blob 编号.
                - centered.source_probability_mean: float32 `(N_candidate,)`, 来源 blob 平均概率.
                - centered.voxel_offsets: int64 `(N_candidate + 1,)`, 切分来源稀疏体素坐标.
                - centered.voxel_index_local_zyx: int16 `(L_voxel, 3)`, 候选 BOX 内 ZYX 体素索引.
                - centered.box_start_zyx: int32 `(N_candidate, 3)`, 候选 BOX 在完整图中的 ZYX 起点.
                - centered.A_offsets: Find Gaussian 专用 int64 `(N_candidate + 1,)`, 切分 A 原子表.
                - centered.A_coord_local_xyz: Find Gaussian 专用 float32 `(N_A, 3)`, BOX 局部 XYZ 原子坐标.
                - centered.A_probability: Find Gaussian 专用 float32 `(N_A,)`, A 原子概率.
                - centered.voxel_size_world: Find Gaussian 专用 float32 `(N_candidate, 3)`, 世界 XYZ 体素尺寸.

            V, P 与 48³ 稠密数组不解压.
            """

            for pdb_id in centered_pdb_ids:
                paths = Stage1ArtifactPaths(output_root, producer, split, pdb_id)
                yield pdb_id, load_stage1_npz(
                    paths.artifact(centered_role),
                    centered_fields,
                )

        score_grid = (
            None
            if role_config.score_parameter_grid is None
            else role_config.score_parameter_grid
        )
        refinement = (
            config.calibration.gaussian_refinement
            if score_mode == "find_gaussian"
            else None
        )
        # 搜索冻结后, 同一角色按最终 min_voxels 重新生成并首次发布正式 centered.
        best = tune_centered_selection(
            centered_items=centered_items(),
            ground_truth_by_pdb=ground_truth,
            score_mode=score_mode,
            score_parameter_grid=score_grid,
            refinement_multipliers=refinement,
            min_voxel_values=role_config.min_voxel_values,
            objective_beta=float(role_config.objective_beta),
            coverage_thresholds=config.evaluation.coverage_thresholds,
            topk_values=config.evaluation.topk_values,
        )
        calibration_payload["roles"][str(role_name)] = {
            "source_threshold": semantic_threshold,
            "selection": best,
        }
        final_centered: deque[Future[dict[str, np.ndarray]]] = deque()
        with ThreadPoolExecutor(
            max_workers=int(config.publish_workers),
            thread_name_prefix="stage1-publish",
        ) as publisher:
            for pdb_id in centered_pdb_ids:
                paths = Stage1ArtifactPaths(output_root, producer, split, pdb_id)
                final_centered.append(
                    produce_centered_role(
                        paths=paths,
                        dataset=dataset,
                        collator=collator,
                        wrapper=wrapper,
                        device=device,
                        blobs=load_stage1_npz(
                            paths.artifact(str(role_config.blob_role)),
                            _CENTERED_BLOB_FIELDS,
                        ),
                        centered_role=centered_role,
                        min_voxels=int(best["min_voxels"]),
                        centered_config=role_config.centered,
                        blob_limit=(
                            None
                            if role_config.blob_limit is None
                            else int(role_config.blob_limit)
                        ),
                        selection=best,
                        publisher=publisher,
                    )
                )
                if len(final_centered) >= int(config.pending_centered_pdbs):
                    final_centered.popleft().result()
            while final_centered:
                final_centered.popleft().result()
        role_metrics[str(role_name)] = evaluate_and_publish_role(
            config=config,
            dataset_root=dataset.root,
            output_root=output_root,
            producer=producer,
            split=split,
            pdb_ids=centered_pdb_ids,
            centered_role=centered_role,
        )

    # 全部角色的正式 centered 与评估完成后, 最后发布校准参数及完成标记.
    publish_stage1_artifact(
        calibration_root / "semantic_threshold_scan.npz", semantic_scan, None
    )
    publish_stage1_json(
        calibration_root / "stage1_v3.metrics.json",
        {"checkpoint_path": str(checkpoint_path), "roles": role_metrics},
    )
    publish_stage1_json(calibration_root / "stage1_v3.json", calibration_payload)
    publish_stage1_json(
        calibration_root / "_COMPLETE",
        {"checkpoint_path": str(checkpoint_path), "result_scope": "calibration_fitted"},
    )
    return calibration_payload


def run_frozen_workflow(
    config: Any,
    calibration_payload: Mapping[str, Any],
    calibration_complete: Mapping[str, Any],
    dataset: Any,
    collator: Any,
    wrapper: Any,
    device: str,
    producer: str,
    split: str,
    pdb_ids: Sequence[str],
    output_root: str | Path,
    checkpoint_path: str,
) -> dict[str, dict[str, object]]:
    """用同源冻结参数生成 validation 或 train 的五类正式产物.

    输入参数:
        - config.window: 完整图滑窗字段映射, 字段契约由 `produce_probability_map` 定义.
        - config.roles.<role>.blob_role: 字符串, 当前角色的 blobs 产物名.
        - config.roles.<role>.centered_role: 字符串, 当前角色的 centered 产物名.
        - config.roles.<role>.blob_limit: int 或 None, 单个 PDB 允许进入 centered 的最大 blob 数.
        - config.roles.<role>.centered: centered 推理映射, 字段契约由 `produce_centered_role` 定义.
        - config.evaluation.coverage_thresholds: 浮点数序列, 双向覆盖率阈值轴.
        - config.evaluation.topk_values: 整数序列, top-K 评估的 K 轴.
        - config.blob_workers: int, 跨 PDB 连通区域线程数.
        - config.publish_workers: int, NPZ 打包与发布线程数.
        - config.pending_probability_pdbs: int, 尚未完成概率与 blobs 发布的最大 PDB 数.
        - config.pending_centered_pdbs: int, 尚未完成 centered 发布的最大 PDB 数.
        - calibration_payload.checkpoint_path: 字符串, calibration 使用的 checkpoint 规范化绝对路径.
        - calibration_payload.roles.<role>.source_threshold: float, 当前角色 blobs 使用的完整图概率阈值.
        - calibration_payload.roles.<role>.selection: centered 冻结选择映射, 字段契约由 `produce_centered_role` 定义.
        - calibration_complete.checkpoint_path: 字符串, calibration 完成标记记录的 checkpoint 规范化绝对路径.
        - calibration_complete.result_scope: 字符串, 必须为 `calibration_fitted`.
        - dataset: Stage1Dataset, 物化当前清单的完整图滑窗与 centered BOX.
        - collator: Stage1Collator, 拼装模型批次.
        - wrapper: 已加载 checkpoint 的 Stage1 模型包装器.
        - device: 字符串, 模型前向设备.
        - producer: 模型产物目录名, 如 `unet_c1` 或 `Find_1`.
        - split: `validation` 或 `train`.
        - pdb_ids: 有序且无重复的小写 PDB 标识序列.
        - output_root: 当前科学配置版本的推理产物根目录.
        - checkpoint_path: 当前命令显式使用的 checkpoint 规范化绝对路径.

    ``calibration_payload`` 来自 `calibration/stage1_v3.json`;
    ``calibration_complete`` 来自同目录 `_COMPLETE`. 两者的 checkpoint 路径必须与
    当前命令的 `checkpoint_path` 完全相同, 且完成标记结果范围必须是
    `calibration_fitted`. ``pdb_ids`` 由 CLI 保证小写且无重复.

    calibration 和当前命令必须使用同一 checkpoint 规范化路径. 不同科学参数
    通过不同 `output_root` 目录区分. 每个完整图返回后, 概率压缩及 F1/F3 CPU 连通区域
    立即并行, GPU 开始下一 PDB. centered 阶段同样让前一个 PDB 的压缩与
    下一个 PDB 的 GPU 前向重叠. 两处待发布队列都有显式 PDB 数上限. 冻结
    selection 在 centered 首次压缩前写入 `score` 与 `selected`, 不再完整解压
    并二次压缩大型 F3 文件.

    每个 PDB 生成 probability, F1/F3 blobs, F1 basic 和 F3 centered NPZ,
    对应性能 JSON, 几何 JSON, 角色 `_COMPLETE`, 可选 `_BLOB_EXCEED` 事实和
    两类评估 NPZ. 数据划分目录生成两类 JSONL 与 metrics JSON. 返回值按
    F1/F3 角色保存同一份汇总指标映射. 返回映射精确包含 `F1_basic` 和
    `F3_centered`; 每个值是 :func:`aggregate_stage1_metrics` 定义的跨 PDB 指标.
    """

    if str(calibration_payload["checkpoint_path"]) != str(checkpoint_path):
        raise ValueError("calibration 与当前命令使用的 checkpoint 路径不一致.")
    if str(calibration_complete["checkpoint_path"]) != str(checkpoint_path):
        raise ValueError("calibration 完成标记与当前命令使用的 checkpoint 路径不一致.")
    if calibration_complete["result_scope"] != "calibration_fitted":
        raise ValueError("calibration 完成标记缺少 calibration_fitted 结果范围.")
    pending_probability: deque[
        tuple[Future[None], tuple[Future[dict[str, np.ndarray]], ...]]
    ] = deque()
    with ThreadPoolExecutor(
        max_workers=int(config.blob_workers),
        thread_name_prefix="stage1-blobs",
    ) as blob_executor, ThreadPoolExecutor(
        max_workers=int(config.publish_workers),
        thread_name_prefix="stage1-publish",
    ) as publisher:
        for pdb_id in pdb_ids:
            paths = Stage1ArtifactPaths(output_root, producer, split, pdb_id)
            probability, probability_future = produce_probability_map(
                paths=paths,
                dataset=dataset,
                collator=collator,
                wrapper=wrapper,
                device=device,
                window_config=config.window,
                publisher=publisher,
            )
            blob_rows = []
            for role_name, role_config in config.roles.items():
                source_threshold = float(
                    calibration_payload["roles"][role_name]["source_threshold"]
                )
                future = blob_executor.submit(
                    publish_probability_blobs,
                    paths,
                    str(role_config.blob_role),
                    source_threshold,
                    probability["probability_map"],
                )
                blob_rows.append(future)
            pending_probability.append((probability_future, tuple(blob_rows)))
            if len(pending_probability) >= int(config.pending_probability_pdbs):
                oldest_probability, oldest_blobs = pending_probability.popleft()
                oldest_probability.result()
                for future in oldest_blobs:
                    future.result()
        while pending_probability:
            oldest_probability, oldest_blobs = pending_probability.popleft()
            oldest_probability.result()
            for future in oldest_blobs:
                future.result()

        centered_pdb_ids_by_role: dict[str, tuple[str, ...]] = {}
        for role_name, role_config in config.roles.items():
            centered_role = str(role_config.centered_role)
            selection = calibration_payload["roles"][role_name]["selection"]
            centered_publish: deque[Future[dict[str, np.ndarray]]] = deque()
            centered_pdb_ids: list[str] = []
            for pdb_id in pdb_ids:
                paths = Stage1ArtifactPaths(output_root, producer, split, pdb_id)
                produced = produce_centered_role(
                    paths=paths,
                    dataset=dataset,
                    collator=collator,
                    wrapper=wrapper,
                    device=device,
                    blobs=load_stage1_npz(
                        paths.artifact(str(role_config.blob_role)),
                        _CENTERED_BLOB_FIELDS,
                    ),
                    centered_role=centered_role,
                    min_voxels=int(selection["min_voxels"]),
                    centered_config=role_config.centered,
                    blob_limit=(
                        None
                        if role_config.blob_limit is None
                        else int(role_config.blob_limit)
                    ),
                    selection=selection,
                    publisher=publisher,
                )
                centered_pdb_ids.append(pdb_id)
                centered_publish.append(produced)
                if len(centered_publish) >= int(config.pending_centered_pdbs):
                    centered_publish.popleft().result()
            while centered_publish:
                centered_publish.popleft().result()
            centered_pdb_ids_by_role[str(role_name)] = tuple(centered_pdb_ids)

        role_metrics: dict[str, dict[str, object]] = {}
        for role_name, role_config in config.roles.items():
            centered_role = str(role_config.centered_role)
            selection = calibration_payload["roles"][role_name]["selection"]
            role_metrics[str(role_name)] = evaluate_and_publish_role(
                config=config,
                dataset_root=dataset.root,
                output_root=output_root,
                producer=producer,
                split=split,
                pdb_ids=centered_pdb_ids_by_role[str(role_name)],
                centered_role=centered_role,
            )
    return role_metrics
