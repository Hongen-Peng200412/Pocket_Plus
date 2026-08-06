"""在 calibration 集合扫描 Gauss scorer 四个正参数并汇总正式指标。"""

from __future__ import annotations

import argparse
import itertools
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from src.artifacts import Stage1ArtifactPaths, atomic_write_json, load_npz_strict
from src.artifacts.states import is_role_complete
from src.evaluation import (
    aggregate_instance_counts,
    evaluate_instance_overlap_counts,
    evaluate_topk_overlap_counts,
)
from src.inference.Gauss_Scorer import (
    compute_centered_gauss_terms,
    compute_li_centered_gauss_terms,
)
from src.inference.assembly import AGOccurrenceVoxelLoader, load_pdb_id_list


@dataclass(frozen=True)
class PdbFacts:
    """保存一个 calibration PDB 对全部参数组合复用的稀疏评估事实。"""

    pdb_id: str
    forest_arrays: Mapping[str, np.ndarray]
    candidate_rows: np.ndarray
    candidate_voxels: tuple[np.ndarray, ...]
    probability_mean: np.ndarray
    positive_terms_by_tau: Mapping[float, np.ndarray]
    negative_terms_by_tau: Mapping[float, np.ndarray]
    occurrence_voxels: tuple[np.ndarray, ...]
    intersections: np.ndarray
    pred_sizes: np.ndarray
    gt_sizes: np.ndarray
    target_union: np.ndarray


def _load_grid(path: str | Path) -> dict[str, Any]:
    """读取第一阶段粗网格或第二阶段 5×5×1×15 精修网格。"""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    limits = {
        "lambda_positive": (2, 5),
        "lambda_negative": (2, 5),
        "tau_angstrom": (1, 4),
        "gauss_score_min": (2, 15),
    }
    for field, (minimum_count, maximum_count) in limits.items():
        values = payload.get(field)
        if not isinstance(values, list) or not minimum_count <= len(values) <= maximum_count:
            raise ValueError(
                f"{field} 必须包含 {minimum_count} 至 {maximum_count} 个候选值"
            )
        floats = [float(value) for value in values]
        if len(set(floats)) != len(floats) or any(
            not np.isfinite(value) or value <= 0.0 for value in floats
        ):
            raise ValueError(f"{field} 必须是唯一的有限正数")
        payload[field] = floats
    cutoff = float(payload.get("distance_cutoff_angstrom", 5.0))
    if not np.isfinite(cutoff) or cutoff <= 0.0:
        raise ValueError("distance_cutoff_angstrom 必须是有限正数")
    payload["distance_cutoff_angstrom"] = cutoff
    payload["include_baseline"] = bool(payload.get("include_baseline", True))
    return payload


def enumerate_configs(grid: Mapping[str, Any]) -> list[dict[str, Any]]:
    """产生一个无额外过滤基线和四轴笛卡尔积配置。"""

    configs: list[dict[str, Any]] = []
    if bool(grid.get("include_baseline", True)):
        configs.append({"config_index": 0, "kind": "baseline"})
    start_index = len(configs)
    for index, values in enumerate(
        itertools.product(
            grid["lambda_positive"],
            grid["lambda_negative"],
            grid["tau_angstrom"],
            grid["gauss_score_min"],
        ),
        start=start_index,
    ):
        lambda_positive, lambda_negative, tau_angstrom, gauss_score_min = values
        configs.append(
            {
                "config_index": index,
                "kind": "gauss",
                "lambda_positive": float(lambda_positive),
                "lambda_negative": float(lambda_negative),
                "tau_angstrom": float(tau_angstrom),
                "gauss_score_min": float(gauss_score_min),
                "distance_cutoff_angstrom": float(grid["distance_cutoff_angstrom"]),
            }
        )
    return configs


def build_refinement_grid(selected_parameters: Mapping[str, Any]) -> dict[str, Any]:
    """围绕第一阶段最优值生成固定 5×5×1×15 第二阶段局部网格。"""

    positive = float(selected_parameters["lambda_positive"])
    negative = float(selected_parameters["lambda_negative"])
    tau = float(selected_parameters["tau_angstrom"])
    score_min = float(selected_parameters["gauss_score_min"])
    if any(
        not np.isfinite(value) or value <= 0.0
        for value in (positive, negative, tau, score_min)
    ):
        raise ValueError("第一阶段最优参数必须全部为有限正数")
    return {
        "search_stage": "refinement",
        "include_baseline": False,
        "lambda_positive": [
            float(positive * factor) for factor in (0.8, 0.9, 1.0, 1.1, 1.2)
        ],
        "lambda_negative": [
            float(negative * factor) for factor in (0.8, 0.9, 1.0, 1.1, 1.2)
        ],
        "tau_angstrom": [tau],
        "gauss_score_min": [
            float(score_min * step / 10.0) for step in range(3, 18)
        ],
        "distance_cutoff_angstrom": float(
            selected_parameters.get("distance_cutoff_angstrom", 5.0)
        ),
    }


def _load_pdb_facts(
    *,
    pdb_id: str,
    data_root: Path,
    output_root: Path,
    producer: str,
    centered_role: str,
    probability_output_root: Path | None,
    tau_values: Sequence[float],
    distance_cutoff_angstrom: float,
) -> PdbFacts:
    """读取一个已完成 F1 PDB，并预计算节点/GT 交集与各 tau 高斯项。"""

    paths = Stage1ArtifactPaths(output_root, producer, "calibration", pdb_id)
    required_roles = (
        ("Li_centered",)
        if centered_role == "Li_centered"
        else ("probability", "components", centered_role)
    )
    missing_roles = [role for role in required_roles if not is_role_complete(paths, role)]
    if paths.running_dir.exists() or missing_roles:
        raise RuntimeError(f"{pdb_id}: calibration F1 尚不可消费，缺少 {missing_roles}")
    centered = load_npz_strict(paths.centered_npz(centered_role))
    source_paths = paths
    if centered_role == "Li_centered":
        if probability_output_root is None:
            raise ValueError("Li-centered 调参必须提供 probability_output_root")
        source_paths = Stage1ArtifactPaths(
            probability_output_root, producer, "calibration", pdb_id
        )
    with source_paths.probability_geometry_json.open("r", encoding="utf-8") as handle:
        geometry = json.load(handle)
    full_shape = tuple(int(value) for value in geometry["full_shape_zyx"])

    positive_by_tau: dict[float, np.ndarray] = {}
    negative_by_tau: dict[float, np.ndarray] = {}
    if centered_role == "Li_centered":
        candidate_rows = np.arange(
            np.asarray(centered["centered_box_index"]).size, dtype=np.int64
        )
        for tau in tau_values:
            positive, negative = compute_li_centered_gauss_terms(
                centered,
                tau_angstrom=float(tau),
                distance_cutoff_angstrom=distance_cutoff_angstrom,
            )
            positive_by_tau[float(tau)] = positive
            negative_by_tau[float(tau)] = negative
        offsets = np.asarray(centered["voxel_offsets"], dtype=np.int64)
        local_values = np.asarray(centered["voxel_index_local_zyx"], dtype=np.int64)
        starts = np.asarray(centered["box_start_zyx"], dtype=np.int64)
        candidate_voxels = tuple(
            np.ravel_multi_index(
                (local_values[int(offsets[row]) : int(offsets[row + 1])] + starts[row]),
                full_shape,
            )
            for row in candidate_rows
        )
        probability_mean = np.asarray(
            centered["source_probability_mean"], dtype=np.float64
        )
    else:
        forest = load_npz_strict(paths.forest_npz)
        candidate_rows: np.ndarray | None = None
        for tau in tau_values:
            positive, negative = compute_centered_gauss_terms(
                forest_arrays=forest,
                centered_arrays=centered,
                full_shape_zyx=full_shape,
                tau_angstrom=float(tau),
                distance_cutoff_angstrom=distance_cutoff_angstrom,
                centered_role=centered_role,
            )
            rows = np.flatnonzero(np.isfinite(positive) & np.isfinite(negative)).astype(np.int64)
            if candidate_rows is None:
                candidate_rows = rows
            elif not np.array_equal(candidate_rows, rows):
                raise ValueError(f"{pdb_id}: 不同 tau 的 centered 节点集合不一致")
            positive_by_tau[float(tau)] = positive[rows]
            negative_by_tau[float(tau)] = negative[rows]
        assert candidate_rows is not None
        offsets = np.asarray(forest["node_voxel_offsets"], dtype=np.int64)
        values = np.asarray(forest["node_voxel_global_linear_index"], dtype=np.int64)
        candidate_voxels = tuple(
            values[int(offsets[row]) : int(offsets[row + 1])] for row in candidate_rows
        )
        probability_mean = np.asarray(forest["probability_mean"], dtype=np.float64)[candidate_rows]
    occurrence_map = AGOccurrenceVoxelLoader(data_root)(pdb_id, full_shape)
    occurrence_voxels = tuple(occurrence_map.values())
    intersections = np.asarray(
        [
            [
                np.intersect1d(prediction, target, assume_unique=True).size
                for target in occurrence_voxels
            ]
            for prediction in candidate_voxels
        ],
        dtype=np.int64,
    ).reshape(len(candidate_voxels), len(occurrence_voxels))
    target_union = (
        np.unique(np.concatenate(occurrence_voxels))
        if occurrence_voxels
        else np.empty(0, dtype=np.int64)
    )
    return PdbFacts(
        pdb_id=pdb_id,
        forest_arrays={} if centered_role == "Li_centered" else forest,
        candidate_rows=candidate_rows,
        candidate_voxels=candidate_voxels,
        probability_mean=probability_mean,
        positive_terms_by_tau=positive_by_tau,
        negative_terms_by_tau=negative_by_tau,
        occurrence_voxels=occurrence_voxels,
        intersections=intersections,
        pred_sizes=np.asarray([value.size for value in candidate_voxels], dtype=np.int64),
        gt_sizes=np.asarray([value.size for value in occurrence_voxels], dtype=np.int64),
        target_union=target_union,
    )


def _selected_and_scores(facts: PdbFacts, config: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """为一个配置返回候选级保留掩码和排名分数。"""

    if config["kind"] == "baseline":
        return np.ones(facts.candidate_rows.size, dtype=np.bool_), facts.probability_mean
    tau = float(config["tau_angstrom"])
    scores = (
        facts.probability_mean
        + float(config["lambda_positive"]) * facts.positive_terms_by_tau[tau]
        - float(config["lambda_negative"]) * facts.negative_terms_by_tau[tau]
    )
    return scores >= float(config["gauss_score_min"]), scores


def evaluate_config(facts_by_pdb: Sequence[PdbFacts], config: Mapping[str, Any]) -> dict[str, Any]:
    """对一个参数配置汇总语义、实例与 top-K 指标。"""

    semantic_tp = semantic_fp = semantic_fn = 0
    semantic_macro: list[float] = []
    instance_counts = []
    topk_totals: dict[str, int] = {}

    for facts in facts_by_pdb:
        selected, scores = _selected_and_scores(facts, config)
        selected_rows = np.flatnonzero(selected)
        selected_voxels = [facts.candidate_voxels[row] for row in selected_rows]
        pred_union = (
            np.unique(np.concatenate(selected_voxels))
            if selected_voxels
            else np.empty(0, dtype=np.int64)
        )
        tp = int(np.intersect1d(pred_union, facts.target_union, assume_unique=True).size)
        fp = int(pred_union.size) - tp
        fn = int(facts.target_union.size) - tp
        semantic_tp += tp
        semantic_fp += fp
        semantic_fn += fn
        denominator = 2 * tp + fp + fn
        semantic_macro.append(0.0 if denominator == 0 else (2.0 * tp) / denominator)

        instance_counts.append(
            evaluate_instance_overlap_counts(
                intersections=facts.intersections[selected],
                pred_sizes=facts.pred_sizes[selected],
                gt_sizes=facts.gt_sizes,
                coverage_thresholds=(0.3, 0.5),
            )
        )
        topk = evaluate_topk_overlap_counts(
            intersections=facts.intersections[selected],
            pred_sizes=facts.pred_sizes[selected],
            gt_sizes=facts.gt_sizes,
            candidate_scores=scores[selected_rows],
            topk_values=(3, 4, 5),
            coverage_thresholds=(0.3, 0.5),
        )
        for field, value in topk.items():
            topk_totals[field] = topk_totals.get(field, 0) + int(value)

    total_instance = aggregate_instance_counts(instance_counts)
    metrics: dict[str, Any] = total_instance.metrics()
    micro_denominator = 2 * semantic_tp + semantic_fp + semantic_fn
    metrics.update(
        {
            "n_evaluated_pdb": len(facts_by_pdb),
            "semantic_dice_micro": 0.0
            if micro_denominator == 0
            else (2.0 * semantic_tp) / micro_denominator,
            "semantic_dice_macro": float(np.mean(semantic_macro)),
            "semantic_tp": semantic_tp,
            "semantic_fp": semantic_fp,
            "semantic_fn": semantic_fn,
        }
    )
    topk_denominator = int(topk_totals.get("n_topk_eligible_pdb", 0))
    metrics.update(topk_totals)
    for field, value in tuple(topk_totals.items()):
        if field.startswith("top") and "_success_" in field:
            ratio_field = field.replace("_success_", "_success_ratio_")
            metrics[ratio_field] = (
                0.0 if topk_denominator == 0 else value / topk_denominator
            )
    metrics["objective"] = (
        float(metrics["semantic_dice_micro"])
        + float(metrics["coverage_f1_0p3"])
        + float(metrics["one_to_one_f1_0p3"])
    )
    return {"config": dict(config), "metrics": metrics}


def _evaluate_shard(arguments: argparse.Namespace) -> int:
    """预计算全部 PDB 事实并评估当前数组元素负责的配置。"""

    grid = _load_grid(arguments.grid_json)
    configs = enumerate_configs(grid)
    if arguments.task_count <= 0 or not 0 <= arguments.task_index < arguments.task_count:
        raise ValueError("task_index 必须位于 [0, task_count)")
    assigned = [
        config
        for config in configs
        if int(config["config_index"]) % arguments.task_count == arguments.task_index
    ]
    pdb_ids = load_pdb_id_list(arguments.pdb_list)
    output_root = Path(arguments.output_root)
    evaluated_ids: list[str] = []
    blob_exceed_ids: list[str] = []
    facts: list[PdbFacts] = []
    for pdb_id in pdb_ids:
        paths = Stage1ArtifactPaths(output_root, arguments.producer, "calibration", pdb_id)
        if paths.blob_exceed_path.is_file():
            blob_exceed_ids.append(pdb_id)
            if not arguments.evaluate_on_blob_exceed:
                continue
        try:
            facts.append(
                _load_pdb_facts(
                    pdb_id=pdb_id,
                    data_root=Path(arguments.data_root),
                    output_root=output_root,
                    producer=arguments.producer,
                    centered_role=arguments.centered_role,
                    probability_output_root=None
                    if arguments.probability_output_root is None
                    else Path(arguments.probability_output_root),
                    tau_values=grid["tau_angstrom"],
                    distance_cutoff_angstrom=float(grid["distance_cutoff_angstrom"]),
                )
            )
        except RuntimeError:
            if paths.blob_exceed_path.is_file():
                continue
            raise
        evaluated_ids.append(pdb_id)
    results = [evaluate_config(facts, config) for config in assigned]
    part_path = Path(arguments.result_root) / "parts" / f"part_{arguments.task_index:03d}.json"
    atomic_write_json(
        part_path,
        {
            "schema_version": 1,
            "grid_definition": grid,
            "centered_role": arguments.centered_role,
            "evaluate_on_blob_exceed": bool(arguments.evaluate_on_blob_exceed),
            "task_index": arguments.task_index,
            "task_count": arguments.task_count,
            "n_total_configs": len(configs),
            "evaluated_pdb_ids": evaluated_ids,
            "blob_exceed_pdb_ids": blob_exceed_ids,
            "results": results,
        },
    )
    return 0


def _merge(arguments: argparse.Namespace) -> int:
    """合并全部数组结果，报告整体最优与正参数范围内最优并发布 calibration。"""

    grid = _load_grid(arguments.grid_json)
    configs = enumerate_configs(grid)
    part_paths = sorted((Path(arguments.result_root) / "parts").glob("part_*.json"))
    if not part_paths:
        raise FileNotFoundError("没有找到 Gauss scorer 参数扫描分片结果")
    results: list[dict[str, Any]] = []
    identity: tuple[tuple[str, ...], tuple[str, ...]] | None = None
    for path in part_paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload["grid_definition"] != grid:
            raise ValueError(f"{path}: grid_definition 不一致")
        if payload["centered_role"] != arguments.centered_role:
            raise ValueError(f"{path}: centered_role 不一致")
        current_identity = (
            tuple(payload["evaluated_pdb_ids"]),
            tuple(payload["blob_exceed_pdb_ids"]),
        )
        if identity is None:
            identity = current_identity
        elif identity != current_identity:
            raise ValueError(f"{path}: calibration PDB 集合不一致")
        results.extend(payload["results"])
    by_index = {int(item["config"]["config_index"]): item for item in results}
    expected = set(range(len(configs)))
    if set(by_index) != expected or len(by_index) != len(results):
        raise ValueError("Gauss scorer 参数扫描结果没有精确覆盖全部配置")
    ordered = [by_index[index] for index in range(len(configs))]
    def ranking_key(item: Mapping[str, Any]) -> tuple[float, float, float, float, int]:
        """按目标函数、三项组成指标和稳定配置编号选择唯一结果。"""

        metrics = item["metrics"]
        return (
            float(metrics["objective"]),
            float(metrics["one_to_one_f1_0p3"]),
            float(metrics["coverage_f1_0p3"]),
            float(metrics["semantic_dice_micro"]),
            -int(item["config"]["config_index"]),
        )

    best_overall = max(ordered, key=ranking_key)
    best_gauss = max(
        (item for item in ordered if item["config"]["kind"] == "gauss"),
        key=ranking_key,
    )
    result_root = Path(arguments.result_root)
    atomic_write_json(
        result_root / "tuning_results.json",
        {
            "schema_version": 1,
            "objective_definition": "semantic_dice_micro + coverage_f1_0p3 + one_to_one_f1_0p3",
            "grid_definition": grid,
            "centered_role": arguments.centered_role,
            "best_overall": best_overall,
            "best_gauss": best_gauss,
            "results": ordered,
        },
    )
    selected = dict(best_gauss["config"])
    selected.pop("config_index", None)
    selected.pop("kind", None)
    calibration_payload: dict[str, Any] = {
            "schema_version": 1,
            "producer": arguments.producer,
            "split": "calibration",
            "centered_role": arguments.centered_role,
            "objective_definition": "semantic_dice_micro + coverage_f1_0p3 + one_to_one_f1_0p3",
            "grid_definition": grid,
            "selected_parameters": selected,
            "selected_metrics": best_gauss["metrics"],
            "best_overall_kind": best_overall["config"]["kind"],
            "evaluated_pdb_ids": list(identity[0]) if identity else [],
            "blob_exceed_pdb_ids": list(identity[1]) if identity else [],
        }
    baseline = next(
        (item for item in ordered if item["config"]["kind"] == "baseline"), None
    )
    if baseline is not None:
        calibration_payload["baseline_metrics"] = baseline["metrics"]
    atomic_write_json(arguments.calibration_json, calibration_payload)
    return 0


def _build_refinement_grid(arguments: argparse.Namespace) -> int:
    """从第一阶段冻结结果生成第二阶段固定局部网格。"""

    phase1 = json.loads(Path(arguments.phase1_calibration_json).read_text(encoding="utf-8"))
    grid = build_refinement_grid(phase1["selected_parameters"])
    atomic_write_json(arguments.output_grid_json, grid)
    return 0


def build_parser() -> argparse.ArgumentParser:
    """声明两阶段参数网格生成、数组评估与结果合并入口。"""

    parser = argparse.ArgumentParser(description="Find_0 Gauss scorer calibration 参数扫描")
    subparsers = parser.add_subparsers(dest="command", required=True)
    evaluate = subparsers.add_parser("evaluate-shard")
    evaluate.add_argument("--pdb-list", required=True)
    evaluate.add_argument("--data-root", required=True)
    evaluate.add_argument("--output-root", required=True)
    evaluate.add_argument("--probability-output-root")
    evaluate.add_argument("--producer", default="Find_0", choices=("Find_0",))
    evaluate.add_argument(
        "--centered-role",
        default="F1_centered",
        choices=(
            "F_1_2_centered",
            "F_2_3_centered",
            "F_4_5_centered",
            "F1_centered",
            "F_5_4_centered",
            "F_3_2_centered",
            "F_2_centered",
            "Li_centered",
        ),
    )
    evaluate.add_argument("--evaluate-on-blob-exceed", action="store_true")
    evaluate.add_argument("--grid-json", required=True)
    evaluate.add_argument("--result-root", required=True)
    evaluate.add_argument("--task-index", type=int, required=True)
    evaluate.add_argument("--task-count", type=int, required=True)
    merge = subparsers.add_parser("merge")
    merge.add_argument("--grid-json", required=True)
    merge.add_argument("--result-root", required=True)
    merge.add_argument("--calibration-json", required=True)
    merge.add_argument("--producer", default="Find_0", choices=("Find_0",))
    merge.add_argument(
        "--centered-role",
        default="F1_centered",
        choices=(
            "F_1_2_centered",
            "F_2_3_centered",
            "F_4_5_centered",
            "F1_centered",
            "F_5_4_centered",
            "F_3_2_centered",
            "F_2_centered",
            "Li_centered",
        ),
    )
    refinement = subparsers.add_parser("build-refinement-grid")
    refinement.add_argument("--phase1-calibration-json", required=True)
    refinement.add_argument("--output-grid-json", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """执行参数扫描分片或合并。"""

    arguments = build_parser().parse_args(argv)
    if arguments.command == "evaluate-shard":
        return _evaluate_shard(arguments)
    if arguments.command == "build-refinement-grid":
        return _build_refinement_grid(arguments)
    return _merge(arguments)


if __name__ == "__main__":
    raise SystemExit(main())
