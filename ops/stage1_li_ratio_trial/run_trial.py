# -*- coding: utf-8 -*-
"""在 calibration/validation PDB 清单上执行 Li blobs 与 basic_ratio 实验.

本入口复用既有完整图 probability, 为每个 PDB 发布唯一 `Li_blobs.npz`.
calibration 分别以 objective_beta=1 和 2 搜索 basic_ratio 选择参数并评估;
validation 冻结复用这两份参数. 科学计算来自 `src.inference`; 本模块只连接文件
路径、并行 PDB 任务与正式发布入口.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping, Sequence

import numpy as np
from omegaconf import OmegaConf

from src.inference.artifacts import (
    Stage1ArtifactPaths,
    load_stage1_npz,
    publish_stage1_artifact,
    publish_stage1_json,
)
from src.inference.blobs import extract_probability_blobs, li_probability_threshold
from src.inference.calibration import tune_centered_selection
from src.inference.evaluation import load_occurrence_voxels
from src.inference.pipeline import run_evaluate_stage


def publish_li_blobs(
    output_root: Path,
    producer: str,
    split: str,
    pdb_id: str,
    denominator: int,
) -> dict[str, float | int | str]:
    """从一个既有 probability 产物发布同一 PDB 的 Li 连通区域.

    输入参数:
        - output_root: Path, Stage1 `artifacts` 根目录.
        - producer: str, 当前模型产物目录名.
        - split: str, 当前 PDB 清单的数据划分名.
        - pdb_id: str, 当前小写 PDB 标识.
        - denominator: int, Li 原始阈值向上量化时采用的概率网格分母.

    返回字段:
        - pdb_id: str, 当前 PDB 标识.
        - blob_count: int, Li 截断后全部 26 连通区域数量.
        - li_threshold_raw: float, Li 迭代原始阈值.
        - li_threshold_grid_index: int, 向上量化后的概率网格编号.
        - li_threshold_applied: float, 连通区域提取实际使用的包含端点阈值.

    本函数不使用旧 component forest, 也不按体素数删除连通区域. NPZ 成功替换后
    才发布 `status/Li_blobs/_COMPLETE`.
    """
    paths = Stage1ArtifactPaths(output_root, producer, split, pdb_id)
    # probability_map: float32, (D, H, W), 已有完整图配体概率; 本实验不重跑模型前向.
    probability_map = load_stage1_npz(
        paths.artifact("probability"),
        ("probability_map",),
    )["probability_map"]
    # 三个 Li 数值依次是原始阈值, 网格编号和向上量化后的实际阈值.
    raw_threshold, grid_index, applied_threshold = li_probability_threshold(
        probability_map,
        denominator,
    )
    # arrays 保存 applied_threshold 下的全部 26 连通区域及其稳定来源均值顺序.
    arrays = extract_probability_blobs(probability_map, applied_threshold)
    arrays.update(
        {
            "li_threshold_raw": np.asarray([raw_threshold], dtype=np.float32),
            "li_threshold_grid_index": np.asarray([grid_index], dtype=np.int32),
            "li_threshold_applied": np.asarray(
                [applied_threshold], dtype=np.float32
            ),
        }
    )
    publish_stage1_artifact(
        paths.artifact("Li_blobs"),
        arrays,
        paths.complete("Li_blobs"),
    )
    return {
        "pdb_id": pdb_id,
        "blob_count": int(arrays["blob_index"].size),
        "li_threshold_raw": raw_threshold,
        "li_threshold_grid_index": grid_index,
        "li_threshold_applied": applied_threshold,
    }


def load_li_calibration_item(
    output_root: Path,
    data_root: Path,
    producer: str,
    split: str,
    pdb_id: str,
) -> tuple[
    str,
    Mapping[str, np.ndarray],
    tuple[np.ndarray, tuple[np.ndarray, ...], tuple[int, int, int]],
]:
    """读取一个 PDB 的 Li blobs 并转换为通用 basic 校准候选.

    输入参数:
        - output_root: Path, 含 `Li_blobs.npz` 的 Stage1 `artifacts` 根目录.
        - data_root: Path, 含 `density/<pdb_id>/ligand_area.npz` 的训练数据根目录.
        - producer: str, 当前模型产物目录名.
        - split: str, 当前 PDB 清单的数据划分名.
        - pdb_id: str, 当前小写 PDB 标识.

    返回值:
        - pdb_id: str, 当前 PDB 标识.
        - candidate: 数组映射, 以完整图 ZYX 稀疏体素表达 Li 候选; BOX 起点统一为零.
        - ground_truth: occurrence 标识、逐 occurrence 完整图体素与完整图 ZYX 形状.
    """
    paths = Stage1ArtifactPaths(output_root, producer, split, pdb_id)
    blobs = load_stage1_npz(
        paths.artifact("Li_blobs"),
        (
            "blob_index",
            "source_probability_mean",
            "voxel_offsets",
            "voxel_index_global_zyx",
        ),
    )
    # candidate_count 定义通用 basic 候选轴; 每项与 blob_index 一一对应.
    candidate_count = int(blobs["blob_index"].size)
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
    ground_truth = load_occurrence_voxels(
        data_root / "density" / pdb_id / "ligand_area.npz"
    )
    return pdb_id, candidate, ground_truth


def publish_li_split(
    output_root: Path,
    producer: str,
    split: str,
    pdb_ids: Sequence[str],
    denominator: int,
    workers: int,
) -> None:
    """并行发布一个完整 PDB 清单的 Li blobs 并保持清单顺序收口.

    输入参数:
        - output_root: Path, Stage1 `artifacts` 根目录.
        - producer: str, 当前模型产物目录名.
        - split: str, `calibration` 或 `validation`.
        - pdb_ids: Sequence[str], 当前数据划分的完整小写 PDB 清单.
        - denominator: int, Li 原始阈值向上量化时采用的概率网格分母.
        - workers: int, 同时处理的 PDB 数量.

    成功时无返回值. 标准输出记录 PDB 数、总连通区域数和墙钟时间.
    """
    started = perf_counter()
    with ThreadPoolExecutor(max_workers=int(workers)) as executor:
        blob_futures = [
            executor.submit(
                publish_li_blobs,
                output_root,
                producer,
                split,
                pdb_id,
                denominator,
            )
            for pdb_id in pdb_ids
        ]
        # blob_summaries 按数据划分清单顺序收口, 并发完成顺序不改变记录顺序.
        blob_summaries = [future.result() for future in blob_futures]
    print(
        f"[Li trial] {split} Li blobs 发布完成: "
        f"pdb_count={len(blob_summaries)}, "
        f"blob_count={sum(int(item['blob_count']) for item in blob_summaries)}, "
        f"seconds={perf_counter() - started:.3f}"
    )


def run_li_calibration(
    config: Any,
    pdb_ids: Sequence[str],
    data_root: Path,
    output_root: Path,
    producer: str,
    split: str,
    denominator: int,
    prefiltered_min_voxel: int,
) -> None:
    """发布 Li blobs, 搜索 F1/F2 比例参数并评估 calibration 清单.

    输入参数:
        - config: OmegaConf 配置, 读取 calibration 线程数、最终体素数列表和评估轴.
        - pdb_ids: Sequence[str], 按正式 calibration 顺序排列的小写 PDB 标识.
        - data_root: Path, Stage1 V3 训练数据根目录.
        - output_root: Path, 本次隔离实验的 Stage1 `artifacts` 根目录.
        - producer: str, 当前模型产物目录名.
        - split: str, 当前固定为 calibration.
        - denominator: int, Li 阈值向上量化的概率网格分母.
        - prefiltered_min_voxel: int, 比例总体建立前固定的来源体素数下限.

    本函数依次发布 `Li_blobs`, `Li_F1_basic_ratio.json`,
    `Li_F2_basic_ratio.json` 及两组逐 PDB/汇总评估. 不生成 centered 产物.
    """
    workers = int(config.calibration.workers)
    publish_li_split(
        output_root,
        producer,
        split,
        pdb_ids,
        denominator,
        workers,
    )

    loading_started = perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as executor:
        item_futures = [
            executor.submit(
                load_li_calibration_item,
                output_root,
                data_root,
                producer,
                split,
                pdb_id,
            )
            for pdb_id in pdb_ids
        ]
        loaded_items = [future.result() for future in item_futures]
    # centered_items 和 ground_truth 保持同一 calibration PDB 顺序与候选轴.
    centered_items = [(pdb_id, candidate) for pdb_id, candidate, _ in loaded_items]
    ground_truth = {
        pdb_id: pdb_ground_truth
        for pdb_id, _, pdb_ground_truth in loaded_items
    }
    print(
        "[Li trial] 校准事实输入加载完成: "
        f"pdb_count={len(centered_items)}, "
        f"seconds={perf_counter() - loading_started:.3f}"
    )

    for objective_beta in (1.0, 2.0):
        # beta_tag 同时决定选择参数文件和评估名称中的 F1/F2 标签.
        beta_tag = f"F{int(objective_beta)}"
        # selection 使用同一 Li 候选事实, 仅 objective_beta 在 F1 与 F2 两轮间变化.
        selection = tune_centered_selection(
            centered_items=centered_items,
            ground_truth_by_pdb=ground_truth,
            score_mode="basic_ratio",
            score_parameter_grid=None,
            refinement_multipliers=None,
            prefiltered_min_voxel=int(prefiltered_min_voxel),
            min_voxel_values=config.calibration.min_voxel_values,
            objective_beta=objective_beta,
            coverage_thresholds=config.evaluation.coverage_thresholds,
            topk_values=config.evaluation.topk_values,
            workers=workers,
        )
        selection = {
            "candidate_role": "Li_blobs",
            "threshold_method": "li",
            "li_denominator": int(denominator),
            **selection,
        }
        # tuning_path 位于 producer 级 tuning 目录, 不与逐 PDB calibration 产物混放.
        tuning_path = (
            output_root
            / producer
            / "tuning"
            / f"Li_{beta_tag}_basic_ratio.json"
        )
        publish_stage1_json(tuning_path, selection)
        # evaluation_name 让 F1 与 F2 的逐 PDB、JSONL 和汇总指标文件稳定并存.
        evaluation_name = f"li_{beta_tag.lower()}_blobs_basic_ratio_selected"
        # metrics 同时含候选 micro/macro 指标及完整概率图 micro/macro PRAUC.
        metrics = run_evaluate_stage(
            config=config,
            data_root=data_root,
            producer=producer,
            split=split,
            pdb_ids=pdb_ids,
            output_root=output_root,
            role="Li_blobs",
            artifact="blobs",
            evaluation_name=evaluation_name,
            selection=selection,
        )
        print(
            f"[Li trial] {beta_tag} 调参与评估完成: "
            f"objective={selection['objective']:.8f}, "
            f"score_ratio_threshold={selection['score_ratio_threshold']:.12g}, "
            f"min_voxels={selection['min_voxels']}, "
            f"semantic_macro_{beta_tag.lower()}="
            f"{metrics[f'semantic_macro_{beta_tag.lower()}']:.8f}"
        )


def run_li_validation(
    config: Any,
    pdb_ids: Sequence[str],
    data_root: Path,
    output_root: Path,
    producer: str,
    split: str,
    denominator: int,
) -> None:
    """发布 validation Li blobs 并冻结复用 calibration F1/F2 选择参数.

    输入参数:
        - config: OmegaConf 配置, 读取 Li PDB 并行数和评估轴.
        - pdb_ids: Sequence[str], 按正式 validation 顺序排列的小写 PDB 标识.
        - data_root: Path, Stage1 V3 训练数据根目录.
        - output_root: Path, 本次隔离实验的 Stage1 `artifacts` 根目录.
        - producer: str, 当前模型产物目录名.
        - split: str, 当前固定为 validation.
        - denominator: int, Li 阈值向上量化的概率网格分母.

    本函数不搜索 validation 参数. 它从 producer 级 `tuning/` 读取 calibration
    已冻结的 F1/F2 basic_ratio JSON, 再发布两组 validation 评估.
    """
    publish_li_split(
        output_root,
        producer,
        split,
        pdb_ids,
        denominator,
        int(config.calibration.workers),
    )
    for beta_tag in ("F1", "F2"):
        # beta_tag 定位 calibration 已冻结的选择参数及对应 validation 评估名称.
        tuning_path = (
            output_root
            / producer
            / "tuning"
            / f"Li_{beta_tag}_basic_ratio.json"
        )
        # selection 完整复用 calibration JSON; validation 不替换任何参数字段.
        selection = json.loads(tuning_path.read_text(encoding="utf-8"))
        evaluation_name = f"li_{beta_tag.lower()}_blobs_basic_ratio_selected"
        # metrics 同时含 validation 候选 micro/macro 指标及完整图 PRAUC.
        metrics = run_evaluate_stage(
            config=config,
            data_root=data_root,
            producer=producer,
            split=split,
            pdb_ids=pdb_ids,
            output_root=output_root,
            role="Li_blobs",
            artifact="blobs",
            evaluation_name=evaluation_name,
            selection=selection,
        )
        print(
            f"[Li trial] validation {beta_tag} 评估完成: "
            f"score_ratio_threshold={selection['score_ratio_threshold']:.12g}, "
            f"min_voxels={selection['min_voxels']}, "
            f"semantic_macro_{beta_tag.lower()}="
            f"{metrics[f'semantic_macro_{beta_tag.lower()}']:.8f}"
        )


def main() -> None:
    """解析隔离实验路径与固定科学参数并运行一个完整数据划分.

    命令必须显式提供 phase、配置、PDB 清单、数据根、产物根、producer、split、
    Li 网格分母和固定预过滤门槛. calibration 调用 :func:`run_li_calibration`;
    validation 调用 :func:`run_li_validation`.
    """
    parser = argparse.ArgumentParser(description="Stage1 Li basic_ratio trial")
    parser.add_argument(
        "--phase", choices=("calibration", "validation"), required=True
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--pdb-json", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--producer", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--li-denominator", type=int, required=True)
    parser.add_argument("--prefiltered-min-voxel", type=int, required=True)
    arguments = parser.parse_args()

    config = OmegaConf.load(Path(arguments.config))
    OmegaConf.resolve(config)
    pdb_ids = tuple(
        str(value).strip().lower()
        for value in json.loads(
            Path(arguments.pdb_json).read_text(encoding="utf-8")
        )
    )
    if arguments.phase == "calibration":
        run_li_calibration(
            config=config,
            pdb_ids=pdb_ids,
            data_root=Path(arguments.data_root),
            output_root=Path(arguments.output_root),
            producer=str(arguments.producer),
            split=str(arguments.split),
            denominator=int(arguments.li_denominator),
            prefiltered_min_voxel=int(arguments.prefiltered_min_voxel),
        )
    else:
        run_li_validation(
            config=config,
            pdb_ids=pdb_ids,
            data_root=Path(arguments.data_root),
            output_root=Path(arguments.output_root),
            producer=str(arguments.producer),
            split=str(arguments.split),
            denominator=int(arguments.li_denominator),
        )


if __name__ == "__main__":
    main()
