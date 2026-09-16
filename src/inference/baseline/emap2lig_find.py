# -*- coding: utf-8 -*-
"""把 Emap2lig 官方 Find 产物接入 Pocket Plus Stage1 评估契约。"""

from __future__ import annotations

import argparse
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import fields
from pathlib import Path
from typing import Any, Sequence

import mrcfile
import numpy as np
from scipy import ndimage

from src.inference.artifacts import (
    publish_stage1_artifact,
    publish_stage1_json,
    publish_stage1_jsonl,
)
from src.inference.evaluation import (
    PdbEvaluation,
    aggregate_semantic_prauc,
    aggregate_stage1_metrics,
    evaluate_centered_pdb,
    load_occurrence_voxels,
    semantic_prauc_histogram,
)


COVERAGE_THRESHOLDS = (0.3, 0.5, 0.6)
TOPK_VALUES = (3, 4, 5)
EVALUATION_NAME = "official_find_li"


def _load_pdb_ids(path: str | Path) -> list[str]:
    """读取 schema-v1 数据划分对象中的有序 PDB 标识。

    输入参数:
        - path: str | Path，必须包含字符串数组 ``pdb_ids`` 的 JSON 文件。

    返回值:
        - pdb_ids: list[str]，按文件原顺序规范为小写的唯一 PDB 标识。
    """

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    # pdb_ids: 长度 N_pdb 的正式评估顺序。
    pdb_ids = [str(value).lower() for value in payload["pdb_ids"]]
    if len(pdb_ids) != len(set(pdb_ids)):
        raise ValueError(f"{path}: pdb_ids 含重复项。")
    return pdb_ids


def _sha256(path: str | Path) -> str:
    """计算文件内容的 SHA-256 十六进制摘要。"""

    digest = hashlib.sha256()
    with Path(path).open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_official_mrc(path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """读取 Emap2lig 自身写出的标准轴序 MRC。

    输入参数:
        - path: str | Path，官方 ``ligand.mrc`` 或 ``mask_N.mrc`` 路径。

    返回值:
        - grid: np.ndarray ``(D,H,W)``，ZYX 轴顺序的独立数组。
        - voxel_size_xyz: float32 ``(3,)``，世界 XYZ 体素尺寸。
        - origin_xyz: float32 ``(3,)``，零索引体素网格角点的世界 XYZ 坐标。

    Emap2lig 的 :func:`to_mrc` 新建标准 ``mapc=1,mapr=2,maps=3`` 文件并把
    ``MapObject.global_origin`` 写入 header origin。本适配器只消费这类官方输出；
    轴声明不符时直接停止，不猜测排列。
    """

    with mrcfile.open(path, mode="r", permissive=True) as archive:
        # axis_order: MRC 列、行、层分别对应的世界轴编号。
        axis_order = (
            int(archive.header.mapc),
            int(archive.header.mapr),
            int(archive.header.maps),
        )
        if axis_order != (1, 2, 3):
            raise ValueError(f"{path}: 非标准 MRC 轴声明 {axis_order}。")
        # grid: float32 (D,H,W)，复制后不依赖已关闭的 MRC 文件句柄。
        grid = np.asarray(archive.data, dtype=np.float32).copy()
        # voxel_size_xyz: float32 (3,)，MRC 世界 XYZ 三轴体素尺寸。
        voxel_size_xyz = np.asarray(archive.voxel_size.tolist(), dtype=np.float32)
        # origin_xyz: float32 (3,)，Emap2lig 写入的世界坐标网格角点。
        origin_xyz = np.asarray(archive.header.origin.tolist(), dtype=np.float32)
    return grid, voxel_size_xyz, origin_xyz


def _map_probability(
    source: np.ndarray,
    source_voxel_size_xyz: np.ndarray,
    source_origin_xyz: np.ndarray,
    target_shape_zyx: Sequence[int],
    target_voxel_size_xyz: np.ndarray,
    target_origin_xyz: np.ndarray,
    chunk_depth: int,
) -> np.ndarray:
    """按体素中心世界坐标把官方概率线性映射到 V3 真值网格。

    输入参数:
        - source: float32 ``(D_src,H_src,W_src)``，Emap2lig ligand probability。
        - source_voxel_size_xyz/source_origin_xyz: float32 ``(3,)``，源网格几何。
        - target_shape_zyx: 三个整数，Pocket Plus 目标 ZYX 形状。
        - target_voxel_size_xyz/target_origin_xyz: float32 ``(3,)``，目标网格几何。
        - chunk_depth: int，沿目标 Z 轴的映射分块深度。

    返回值:
        - mapped: float32 ``(D,H,W)``，目标网格上的线性插值概率；源范围外为 0。
    """

    # target_shape: 三个整数，目标完整图 ZYX 体素数。
    target_shape = tuple(int(value) for value in target_shape_zyx)
    if (
        source.shape == target_shape
        and np.array_equal(source_voxel_size_xyz, target_voxel_size_xyz)
        and np.array_equal(source_origin_xyz, target_origin_xyz)
    ):
        return np.asarray(source, dtype=np.float32).copy()

    # mapped: float32 (D,H,W)，逐 Z 分块填充的目标概率图。
    mapped = np.zeros(target_shape, dtype=np.float32)
    # target_y, target_x: float32 (H,W)，目标网格 YX 整数索引。
    target_y, target_x = np.mgrid[0 : target_shape[1], 0 : target_shape[2]].astype(
        np.float32,
        copy=False,
    )
    for begin in range(0, target_shape[0], int(chunk_depth)):
        # end: 当前目标 Z 分块的右开端点。
        end = min(begin + int(chunk_depth), target_shape[0])
        # target_z: float32 (D_chunk,1,1)，当前分块的目标 Z 索引。
        target_z = np.arange(begin, end, dtype=np.float32)[:, None, None]
        # source_z/source_y/source_x: 目标体素中心在源网格中的浮点 ZYX 索引。
        source_z = (
            target_origin_xyz[2]
            + (target_z + 0.5) * target_voxel_size_xyz[2]
            - source_origin_xyz[2]
        ) / source_voxel_size_xyz[2] - 0.5
        source_y = (
            target_origin_xyz[1]
            + (target_y[None, :, :] + 0.5) * target_voxel_size_xyz[1]
            - source_origin_xyz[1]
        ) / source_voxel_size_xyz[1] - 0.5
        source_x = (
            target_origin_xyz[0]
            + (target_x[None, :, :] + 0.5) * target_voxel_size_xyz[0]
            - source_origin_xyz[0]
        ) / source_voxel_size_xyz[0] - 0.5
        # coordinates: 三个可广播数组，依次是源 Z、Y、X 浮点索引。
        coordinates = np.broadcast_arrays(source_z, source_y, source_x)
        mapped[begin:end] = ndimage.map_coordinates(
            source,
            coordinates,
            order=1,
            mode="constant",
            cval=0.0,
        )
    return mapped


def _numbered_mask_paths(directory: str | Path) -> list[tuple[int, Path]]:
    """按官方 ``mask_N`` 数字编号返回实例 MRC 路径。"""

    numbered = []
    for path in Path(directory).glob("mask_*.mrc"):
        suffix = path.stem.removeprefix("mask_")
        if suffix.isdigit():
            numbered.append((int(suffix), path))
    numbered.sort(key=lambda item: item[0])
    return numbered


def _map_official_candidates(
    mask_paths: Sequence[tuple[int, Path]],
    probability: np.ndarray,
    probability_voxel_size_xyz: np.ndarray,
    probability_origin_xyz: np.ndarray,
    target_shape_zyx: Sequence[int],
    target_voxel_size_xyz: np.ndarray,
    target_origin_xyz: np.ndarray,
) -> dict[str, np.ndarray]:
    """保留官方 blob 身份，把每个实例稀疏映射到目标网格。

    每个官方 blob 的分数是其原生 mask 体素在原生 ligand probability 中的
    算术平均。实例独立进行最近体素映射；不同实例映射后可以重叠，不再次执行
    连通域拆分或合并。
    """

    # blob_indices: int32 (N_candidate,)，官方 mask_N 的数字编号。
    blob_indices: list[int] = []
    # scores: float32 (N_candidate,)，原生概率图上的实例平均概率。
    scores: list[float] = []
    # target_rows: 长度 N_candidate 的列表，每项是唯一目标 ZYX 稀疏坐标。
    target_rows: list[np.ndarray] = []
    # target_shape: int64 (3,)，目标 ZYX 边界检查所需形状。
    target_shape = np.asarray(target_shape_zyx, dtype=np.int64)

    for blob_index, mask_path in mask_paths:
        # mask: float32 (D_blob,H_blob,W_blob)，官方裁剪后的单实例二值网格。
        mask, mask_voxel_size_xyz, mask_origin_xyz = _read_official_mrc(mask_path)
        # mask_zyx: int64 (K,3)，官方实例中非零体素的局部 ZYX 索引。
        mask_zyx = np.argwhere(mask > 0).astype(np.int64, copy=False)
        if mask_zyx.size == 0:
            raise ValueError(f"{mask_path}: 官方实例 mask 为空。")
        # world_xyz: float64 (K,3)，官方实例体素中心的世界 XYZ 坐标。
        world_xyz = mask_origin_xyz + (
            mask_zyx[:, ::-1].astype(np.float64) + 0.5
        ) * mask_voxel_size_xyz
        # probability_xyz: int64 (K,3)，同一世界坐标在原生概率图中的最近 XYZ 体素。
        probability_xyz = np.rint(
            (world_xyz - probability_origin_xyz) / probability_voxel_size_xyz - 0.5
        ).astype(np.int64)
        # probability_zyx: int64 (K,3)，用于索引原生概率图的 ZYX 坐标。
        probability_zyx = probability_xyz[:, ::-1]
        # probability_shape: int64 (3,)，原生概率图 ZYX 形状。
        probability_shape = np.asarray(probability.shape, dtype=np.int64)
        if np.any((probability_zyx < 0) | (probability_zyx >= probability_shape)):
            raise ValueError(f"{mask_path}: 实例体素超出原生概率图范围。")
        # score: float，当前官方实例在原生 ligand probability 中的平均概率。
        score = float(probability[tuple(probability_zyx.T)].mean(dtype=np.float64))

        # target_xyz: int64 (K,3)，世界坐标在 Pocket Plus 网格中的最近 XYZ 体素。
        target_xyz = np.rint(
            (world_xyz - target_origin_xyz) / target_voxel_size_xyz - 0.5
        ).astype(np.int64)
        # target_zyx: int64 (K,3)，目标网格 ZYX 坐标，允许不同实例之间重叠。
        target_zyx = target_xyz[:, ::-1]
        # in_bounds: bool (K,)，实例体素是否落在目标完整图内。
        in_bounds = np.all((target_zyx >= 0) & (target_zyx < target_shape), axis=1)
        # unique_target_zyx: int32 (K_unique,3)，当前实例内部去重后的目标体素。
        unique_target_zyx = np.unique(target_zyx[in_bounds], axis=0).astype(
            np.int32,
            copy=False,
        )
        if unique_target_zyx.size == 0:
            raise ValueError(f"{mask_path}: 映射后没有目标网格体素。")
        blob_indices.append(blob_index)
        scores.append(score)
        target_rows.append(unique_target_zyx)

    # counts: int64 (N_candidate,)，每个映射实例的唯一目标体素数。
    counts = np.asarray([rows.shape[0] for rows in target_rows], dtype=np.int64)
    # offsets: int64 (N_candidate+1,)，切分拼接稀疏坐标的右开端点。
    offsets = np.concatenate(
        (np.zeros(1, dtype=np.int64), np.cumsum(counts, dtype=np.int64))
    )
    # coordinates: int32 (L_voxel,3)，按官方编号顺序拼接的目标 ZYX 坐标。
    coordinates = (
        np.concatenate(target_rows, axis=0)
        if target_rows
        else np.empty((0, 3), dtype=np.int32)
    )
    # score_array: float32 (N_candidate,)，与官方编号轴对齐的冻结分数。
    score_array = np.asarray(scores, dtype=np.float32)
    return {
        "source_blob_index": np.asarray(blob_indices, dtype=np.int32),
        "source_probability_mean": score_array,
        "score": score_array,
        "selected": np.ones(score_array.shape, dtype=np.bool_),
        "prauc_eligible": np.ones(score_array.shape, dtype=np.bool_),
        "voxel_offsets": offsets,
        "voxel_index_global_zyx": coordinates,
        "voxel_count": counts.astype(np.int32, copy=False),
    }


def _evaluation_arrays(
    evaluation: PdbEvaluation,
    prauc_histogram: np.ndarray,
) -> dict[str, np.ndarray]:
    """把逐 PDB 评估事实转换为无 object dtype 的 NPZ 字段。"""

    arrays: dict[str, np.ndarray] = {}
    for field in fields(PdbEvaluation):
        if field.name == "pdb_id":
            continue
        arrays[field.name] = np.asarray(getattr(evaluation, field.name))
    arrays["semantic_prauc_histogram"] = np.asarray(
        prauc_histogram,
        dtype=np.int64,
    )
    return arrays


def _load_evaluation(path: str | Path, pdb_id: str) -> tuple[PdbEvaluation, np.ndarray]:
    """从适配器正式 NPZ 恢复一个 PDB 的完整评估事实。"""

    with np.load(path, allow_pickle=False) as archive:
        values: dict[str, Any] = {"pdb_id": pdb_id}
        for field in fields(PdbEvaluation):
            if field.name == "pdb_id":
                continue
            value = np.asarray(archive[field.name])
            values[field.name] = int(value) if value.ndim == 0 else value
        # histogram: int64 (2,1024)，当前 PDB 的完整概率图 PRAUC 计数。
        histogram = np.asarray(archive["semantic_prauc_histogram"], dtype=np.int64)
    return PdbEvaluation(**values), histogram


def _evaluate_one_pdb(
    pdb_id: str,
    raw_find_root: str,
    data_root: str,
    split_root: str,
    chunk_depth: int,
) -> str:
    """映射并评估一个 PDB，原子发布概率、候选和交集事实。"""

    # source_root: 当前 PDB 的原始 Emap2lig 官方输出目录。
    source_root = Path(raw_find_root) / pdb_id
    # probability_path: 官方完整 ligand probability MRC。
    probability_path = source_root / "find_maps" / "ligand.mrc"
    # mask_paths: 按官方编号排序的全部 >=32 体素实例 MRC。
    mask_paths = _numbered_mask_paths(source_root / "find_blobs")
    if not probability_path.is_file():
        raise FileNotFoundError(probability_path)
    if not mask_paths:
        raise FileNotFoundError(f"{source_root}: 没有官方 mask_N.mrc。")

    # density_root: 当前 PDB 的 Pocket Plus V3 真值目录。
    density_root = Path(data_root) / "density" / pdb_id
    with np.load(density_root / "ligand_area.npz", allow_pickle=False) as ligand_area:
        # target_shape_zyx: int64 (3,)，真值完整图 ZYX 形状。
        target_shape_zyx = np.asarray(ligand_area["grid_shape_zyx"], dtype=np.int64)
        # target_voxel_size_xyz: float32 (3,)，真值网格世界 XYZ 体素尺寸。
        target_voxel_size_xyz = np.asarray(
            ligand_area["voxel_size_xyz"], dtype=np.float32
        )
        # target_origin_xyz: float32 (3,)，真值网格角点世界坐标。
        target_origin_xyz = np.asarray(ligand_area["origin_xyz"], dtype=np.float32)

    # source_probability: float32 (D_src,H_src,W_src)，官方原生 ligand probability。
    source_probability, source_voxel_size_xyz, source_origin_xyz = _read_official_mrc(
        probability_path
    )
    # mapped_probability: float32 (D,H,W)，目标 V3 网格上的线性插值概率。
    mapped_probability = _map_probability(
        source_probability,
        source_voxel_size_xyz,
        source_origin_xyz,
        target_shape_zyx,
        target_voxel_size_xyz,
        target_origin_xyz,
        chunk_depth,
    )
    if not np.isfinite(mapped_probability).all():
        raise ValueError(f"{pdb_id}: 映射概率含非有限值。")
    # candidates: 官方实例身份不变的目标稀疏坐标与原生平均概率。
    candidates = _map_official_candidates(
        mask_paths,
        source_probability,
        source_voxel_size_xyz,
        source_origin_xyz,
        target_shape_zyx,
        target_voxel_size_xyz,
        target_origin_xyz,
    )

    # mapped_root: 当前 PDB 的目标网格概率与稀疏实例目录。
    mapped_root = Path(split_root) / "mapped" / pdb_id
    publish_stage1_artifact(
        mapped_root / "probability_map.npz",
        {
            "probability_map": mapped_probability,
            "origin_xyz": target_origin_xyz,
            "voxel_size_xyz": target_voxel_size_xyz,
        },
        None,
    )
    publish_stage1_artifact(
        mapped_root / "official_blobs.npz",
        {
            **candidates,
            "origin_xyz": target_origin_xyz,
            "voxel_size_xyz": target_voxel_size_xyz,
            "grid_shape_zyx": target_shape_zyx,
        },
        None,
    )

    occurrence_id, occurrence_rows, full_shape = load_occurrence_voxels(
        density_root / "ligand_area.npz"
    )
    # centered: evaluate_centered_pdb 消费的完整图稀疏候选视图。
    centered = {
        "source_blob_index": candidates["source_blob_index"],
        "score": candidates["score"],
        "selected": candidates["selected"],
        "prauc_eligible": candidates["prauc_eligible"],
        "voxel_offsets": candidates["voxel_offsets"],
        "voxel_index_local_zyx": candidates["voxel_index_global_zyx"],
        "box_start_zyx": np.zeros(
            (candidates["source_blob_index"].size, 3), dtype=np.int32
        ),
    }
    # evaluation: 当前 PDB 的完整语义、交集、匹配和 top-K 事实。
    evaluation = evaluate_centered_pdb(
        pdb_id=pdb_id,
        centered=centered,
        occurrence_id=occurrence_id,
        occurrence_voxel_zyx=occurrence_rows,
        full_shape_zyx=full_shape,
        coverage_thresholds=COVERAGE_THRESHOLDS,
        topk_values=TOPK_VALUES,
    )
    # union_mask: bool (D,H,W)，所有真实 occurrence 的体素并集。
    union_mask = np.load(
        density_root / "union_mask.npy",
        mmap_mode="r",
        allow_pickle=False,
    )[0]
    # histogram: int64 (2,1024)，完整概率图的语义 PRAUC 计数。
    histogram = semantic_prauc_histogram(mapped_probability, union_mask)
    publish_stage1_artifact(
        Path(split_root) / "evaluation" / "per_pdb" / f"{pdb_id}.npz",
        _evaluation_arrays(evaluation, histogram),
        None,
    )
    publish_stage1_json(
        mapped_root / "mapping.json",
        {
            "pdb_id": pdb_id,
            "official_probability": str(probability_path),
            "official_blob_count": len(mask_paths),
            "mapped_blob_count": int(candidates["source_blob_index"].size),
            "source_shape_zyx": list(source_probability.shape),
            "source_voxel_size_xyz": source_voxel_size_xyz.tolist(),
            "source_origin_xyz": source_origin_xyz.tolist(),
            "target_shape_zyx": target_shape_zyx.tolist(),
            "target_voxel_size_xyz": target_voxel_size_xyz.tolist(),
            "target_origin_xyz": target_origin_xyz.tolist(),
        },
    )
    return pdb_id


def _aggregate_split(
    pdb_ids: Sequence[str],
    parent_split_root: str | Path,
    output_split_root: str | Path,
) -> tuple[Path, Path]:
    """从逐 PDB 正式事实聚合一个数据划分的 JSON 与 JSONL。"""

    evaluations = []
    histograms = []
    jsonl_rows = []
    for pdb_id in pdb_ids:
        # evaluation_path: 父 test_0 中当前 PDB 的不可变评估事实。
        evaluation_path = (
            Path(parent_split_root) / "evaluation" / "per_pdb" / f"{pdb_id}.npz"
        )
        evaluation, histogram = _load_evaluation(evaluation_path, pdb_id)
        evaluations.append(evaluation)
        histograms.append(histogram)
        # single_metrics: 当前 PDB 使用生产聚合函数得到的单项 JSONL 指标。
        single_metrics = aggregate_stage1_metrics(
            (evaluation,), COVERAGE_THRESHOLDS, TOPK_VALUES
        )
        single_metrics.update(aggregate_semantic_prauc((histogram,)))
        jsonl_rows.append({"pdb_id": pdb_id, **single_metrics})

    # metrics: 当前数据划分的全部标准 Stage1 指标。
    metrics = aggregate_stage1_metrics(
        evaluations,
        COVERAGE_THRESHOLDS,
        TOPK_VALUES,
    )
    metrics.update(aggregate_semantic_prauc(histograms))
    # evaluation_root: 当前数据划分的全局机器可读结果目录。
    evaluation_root = Path(output_split_root) / "evaluation"
    metrics_path = evaluation_root / f"{EVALUATION_NAME}.metrics.json"
    jsonl_path = evaluation_root / f"{EVALUATION_NAME}.jsonl"
    publish_stage1_json(metrics_path, metrics)
    publish_stage1_jsonl(jsonl_path, jsonl_rows)
    return metrics_path, jsonl_path


def run_evaluation(args: argparse.Namespace) -> None:
    """评估完整 test_0，并从同一批逐 PDB 事实派生 test_1。"""

    # test0_ids: 179 个正式前向与逐 PDB 评估对象。
    test0_ids = _load_pdb_ids(args.test0_json)
    # test1_ids: 149 个 test_0 保序子集对象。
    test1_ids = _load_pdb_ids(args.test1_json)
    if [pdb_id for pdb_id in test0_ids if pdb_id in set(test1_ids)] != test1_ids:
        raise ValueError("test_1 不是 test_0 的严格保序子集。")

    # test0_root: 唯一执行官方 Find 和几何映射的数据划分根。
    test0_root = Path(args.output_root) / "held_out_test_0"
    # worker_args: 每个 PDB 独立映射与评估所需的不可变参数。
    worker_args = [
        (
            pdb_id,
            str(test0_root / "raw_find_outputs"),
            str(args.data_root),
            str(test0_root),
            int(args.chunk_depth),
        )
        for pdb_id in test0_ids
    ]
    with ProcessPoolExecutor(max_workers=int(args.workers)) as executor:
        completed = list(executor.map(_evaluate_one_pdb_star, worker_args))
    if completed != test0_ids:
        raise RuntimeError("test_0 并行评估返回顺序与清单不一致。")

    test0_metrics, test0_jsonl = _aggregate_split(test0_ids, test0_root, test0_root)
    # test1_root: 只保存派生汇总与 provenance，不复制概率、候选或逐 PDB NPZ。
    test1_root = Path(args.output_root) / "held_out_test_1"
    test1_metrics, test1_jsonl = _aggregate_split(test1_ids, test0_root, test1_root)
    # parent_hashes: test_1 每个成员引用的 test_0 逐 PDB 事实哈希。
    parent_hashes = {
        pdb_id: _sha256(test0_root / "evaluation" / "per_pdb" / f"{pdb_id}.npz")
        for pdb_id in test1_ids
    }
    publish_stage1_json(
        test1_root / "evaluation" / f"{EVALUATION_NAME}.provenance.json",
        {
            "derivation": "ordered_subset_of_held_out_test_0",
            "parent_split_root": str(test0_root),
            "test_0_json": str(args.test0_json),
            "test_0_json_sha256": _sha256(args.test0_json),
            "test_1_json": str(args.test1_json),
            "test_1_json_sha256": _sha256(args.test1_json),
            "pdb_ids": test1_ids,
            "pdb_count": len(test1_ids),
            "parent_evaluation_sha256": parent_hashes,
            "parent_metrics": str(test0_metrics),
            "parent_metrics_sha256": _sha256(test0_metrics),
            "parent_jsonl": str(test0_jsonl),
            "parent_jsonl_sha256": _sha256(test0_jsonl),
            "metrics_sha256": _sha256(test1_metrics),
            "jsonl_sha256": _sha256(test1_jsonl),
            "adapter_sha256": _sha256(__file__),
            "evaluation_module_sha256": _sha256(Path(__file__).parents[1] / "evaluation.py"),
            "repeated_probability_or_find": False,
        },
    )


def _evaluate_one_pdb_star(arguments: tuple[str, str, str, str, int]) -> str:
    """把进程池单个元组参数展开给 :func:`_evaluate_one_pdb`。"""

    return _evaluate_one_pdb(*arguments)


def parse_args() -> argparse.Namespace:
    """解析 Emap2lig held-out 标准评估参数。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test0-json", required=True)
    parser.add_argument("--test1-json", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--chunk-depth", type=int, default=8)
    return parser.parse_args()


if __name__ == "__main__":
    run_evaluation(parse_args())

