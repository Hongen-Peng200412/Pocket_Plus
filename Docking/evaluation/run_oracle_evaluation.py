from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

DOCKING_DIR = Path(__file__).resolve().parents[1]
if str(DOCKING_DIR) not in sys.path:
    sys.path.insert(0, str(DOCKING_DIR))

from docking_pipeline.config import MatchingOptions, ServerPaths
from docking_pipeline.io_utils import write_json
from docking_pipeline.matching import RawPairFeatures, build_pair_scores
from run_evaluation import RMSD_THRESHOLDS, direct_rmsd, first_output_ligand_xyz, load_truth_instances, write_csv


TOP_K_VALUES = (1, 3, 5)
TOP_PERCENT_VALUES = (0.10, 0.25)


def main() -> None:
    """
    评估 oracle easy20 的真实 matching 与 pose 质量。

    输入参数:
        - `--eval-run-id`: str, 当前严格评价输出目录名称
        - `--oracle-run-id`: str, 已完成 oracle pipeline run ID

    输出:
        - None; 在 `evaluation_runs/{eval_run_id}` 下写出逐 pose、逐 assignment、逐 rank 与汇总 JSON/CSV
    """
    parser = argparse.ArgumentParser(description="严格评估 oracle docking 的 assignment 与 RMSD")
    parser.add_argument("--eval-run-id", required=True, help="严格评价 run ID")
    parser.add_argument("--oracle-run-id", required=True, help="oracle pipeline run ID")
    args = parser.parse_args()

    paths = ServerPaths.default()
    output_dir = paths.allowed_root / "evaluation_runs" / args.eval_run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    samples = load_oracle_samples(paths, args.oracle_run_id)
    truth_rows = load_truth_instances(paths, [str(sample["pdb_id"]) for sample in samples])
    truth_lookup = {
        (str(row["sample_id"]), str(row["ligand_label"])): row
        for row in truth_rows
        if not row.get("truth_error")
    }

    variant_rows: list[dict[str, Any]] = []
    truth_pair_pose_rows: list[dict[str, Any]] = []
    strict_selection_rows: list[dict[str, Any]] = []
    assignment_rows: list[dict[str, Any]] = []
    rank_rows: list[dict[str, Any]] = []
    for sample in samples:
        sample_id = str(sample["pdb_id"])
        for variant in sample.get("variants", []):
            variant_rows.append(variant_flow_row(sample_id, variant))
            context = variant_context(sample_id, variant)
            result_lookup = index_variant_results(variant)
            truth_pair_pose_rows.extend(
                build_truth_pair_pose_rows(context, result_lookup, truth_lookup)
            )
            if variant.get("mode") == "identity":
                strict_selection_rows.extend(
                    build_identity_selection_rows(context, result_lookup, truth_lookup)
                )
                continue
            pair_scores = rebuild_pair_scores(context, result_lookup)
            current_assignments = variant.get("assignments", [])
            assignment_rows.extend(
                build_assignment_rows(context, current_assignments, pair_scores)
            )
            rank_rows.extend(build_rank_rows(context, current_assignments, pair_scores))
            strict_selection_rows.extend(
                build_hungarian_selection_rows(context, current_assignments, result_lookup, truth_lookup)
            )

    summary = {
        "eval_run_id": args.eval_run_id,
        "oracle_run_id": args.oracle_run_id,
        "num_samples": len(samples),
        "flow_summary": summarize_flow(variant_rows),
        "truth_pair_pose_summary": summarize_pose_rows(truth_pair_pose_rows, strict_denominator=False),
        "strict_selected_pose_summary": summarize_pose_rows(strict_selection_rows, strict_denominator=True),
        "assignment_summary": summarize_assignments(assignment_rows),
        "rank_summary": summarize_ranks(rank_rows),
        "grouped_by_task_receptor": grouped_summary(strict_selection_rows, assignment_rows, rank_rows),
        "notes": [
            "流程跑通不等于真正对接成功；严格成功要求选中真实 ligand occurrence 且 pose RMSD 达到阈值。",
            "truth_pair_pose_summary 回答真实 pair 已经跑过时 Rosetta pose 能否接近 native；不惩罚 Hungarian 选错 ligand。",
            "strict_selected_pose_summary 对 Hungarian 任务将错误 ligand、未匹配或失败 job 都计为失败。",
            "RMSD 当前使用 direct atom-order；存在 atom-count 或解析 warning 的 pose 不纳入可靠 RMSD 命中。",
        ],
    }
    write_csv(output_dir / "variants.csv", variant_rows)
    write_csv(output_dir / "truth_pair_pose_metrics.csv", truth_pair_pose_rows)
    write_csv(output_dir / "strict_selected_pose_metrics.csv", strict_selection_rows)
    write_csv(output_dir / "assignment_metrics.csv", assignment_rows)
    write_csv(output_dir / "truth_pair_rank_metrics.csv", rank_rows)
    write_csv(output_dir / "grouped_metrics.csv", summary["grouped_by_task_receptor"])
    write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def load_oracle_samples(paths: ServerPaths, oracle_run_id: str) -> list[dict[str, Any]]:
    """
    读取 oracle run 下每个样本的最终 summary。

    输入参数:
        - paths: ServerPaths, 服务器路径配置
        - oracle_run_id: str, oracle pipeline run ID

    输出:
        - samples: list[dict[str, Any]], 每项为一个样本 summary，包含 `variants` 列表
    """
    samples_dir = paths.allowed_root / "pipeline_runs" / oracle_run_id / "samples"
    return [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(samples_dir.glob("*/audit/summary.json"))
    ]


def variant_context(sample_id: str, variant: dict[str, Any]) -> dict[str, Any]:
    """
    构造当前 oracle variant 的固定 truth site 元数据。

    输入参数:
        - sample_id: str, 当前样本 ID
        - variant: dict[str, Any], oracle variant summary

    输出:
        - context: dict[str, Any], 包含 task、variant、site 到真实 ligand 的映射与 receptor scopes
    """
    site_truth = {
        site_id(site): str(site["truth_ligand_label"])
        for site in variant.get("sites", [])
    }
    receptor_scopes = sorted(
        {
            str(result["job"]["receptor"]["name"])
            for result in variant.get("results", [])
        }
    )
    return {
        "sample_id": sample_id,
        "task": str(variant.get("task", "")),
        "variant_id": str(variant.get("variant_id", "")),
        "mode": str(variant.get("mode", "")),
        "radius": variant.get("radius", ""),
        "seed": variant.get("seed", ""),
        "site_truth": site_truth,
        "receptor_scopes": receptor_scopes,
    }


def site_id(site: dict[str, Any]) -> str:
    """将 JSON site 记录转换为 runner 使用的稳定 `siteNNN` 标签。"""
    return f"site{int(site['instance_id']):03d}"


def index_variant_results(variant: dict[str, Any]) -> dict[tuple[str, str, str], dict[str, Any]]:
    """
    按 `(site_id, ligand_label, receptor_scope)` 索引 variant 的 Rosetta 结果。

    输入参数:
        - variant: dict[str, Any], oracle variant summary

    输出:
        - indexed: dict[tuple[str, str, str], dict[str, Any]], Rosetta result 查询表
    """
    indexed: dict[tuple[str, str, str], dict[str, Any]] = {}
    for result in variant.get("results", []):
        job = result["job"]
        key = (
            site_id(job["site"]),
            str(job["ligand"]["label"]),
            str(job["receptor"]["name"]),
        )
        indexed[key] = result
    return indexed


def build_truth_pair_pose_rows(
    context: dict[str, Any],
    results: dict[tuple[str, str, str], dict[str, Any]],
    truth_lookup: dict[tuple[str, str], dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    计算每个 site 对应真实 ligand pair 的 pose RMSD，不考虑 assignment 是否选中该 pair。

    输入参数:
        - context: dict[str, Any], 当前 variant 的 truth site 元数据
        - results: dict, 当前 variant 的 Rosetta 结果查询表
        - truth_lookup: dict, `(sample_id, ligand_label)` 到 native ligand 的映射

    输出:
        - rows: list[dict[str, Any]], 每个真实 site 与 receptor 一行的 RMSD 明细
    """
    rows: list[dict[str, Any]] = []
    for current_site, truth_ligand in context["site_truth"].items():
        truth = truth_lookup.get((context["sample_id"], truth_ligand))
        for receptor_scope in context["receptor_scopes"]:
            result = results.get((current_site, truth_ligand, receptor_scope))
            row = common_selection_row(context, current_site, receptor_scope, truth_ligand, truth_ligand, "truth_pair")
            row["assignment_correct"] = True
            rows.append(add_pose_metrics(row, result, truth))
    return rows


def build_identity_selection_rows(
    context: dict[str, Any],
    results: dict[tuple[str, str, str], dict[str, Any]],
    truth_lookup: dict[tuple[str, str], dict[str, Any]],
) -> list[dict[str, Any]]:
    """构造 identity 任务的严格选择结果；身份由实验定义直接给定。"""
    return build_truth_pair_pose_rows(context, results, truth_lookup)


def build_hungarian_selection_rows(
    context: dict[str, Any],
    assignments: list[dict[str, Any]],
    results: dict[tuple[str, str, str], dict[str, Any]],
    truth_lookup: dict[tuple[str, str], dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    构造 Hungarian 任务选中结果的严格 pose 明细。

    错误 ligand 或虚拟节点不具有“正确 pose”含义，直接记为严格失败。
    """
    rows: list[dict[str, Any]] = []
    for assignment in assignments:
        receptor_scope = str(assignment["receptor_scope"])
        selected = {str(pair["site_id"]): str(pair["ligand_label"]) for pair in assignment.get("pairs", [])}
        for current_site, truth_ligand in context["site_truth"].items():
            selected_ligand = selected.get(current_site, "")
            correct = selected_ligand == truth_ligand
            row = common_selection_row(
                context,
                current_site,
                receptor_scope,
                truth_ligand,
                selected_ligand,
                "hungarian_selected",
            )
            row["assignment_correct"] = correct
            if not correct:
                row.update(
                    {
                        "pose_success": False,
                        "rmsd": "",
                        "rmsd_method": "",
                        "rmsd_warning": "wrong_or_missing_assignment",
                    }
                )
                for threshold in RMSD_THRESHOLDS:
                    row[f"rmsd_le_{threshold:g}A"] = False
                rows.append(row)
                continue
            truth = truth_lookup.get((context["sample_id"], truth_ligand))
            result = results.get((current_site, selected_ligand, receptor_scope))
            rows.append(add_pose_metrics(row, result, truth))
    return rows


def common_selection_row(
    context: dict[str, Any],
    current_site: str,
    receptor_scope: str,
    truth_ligand: str,
    selected_ligand: str,
    selection_source: str,
) -> dict[str, Any]:
    """生成逐 site 评价表中的公共定位字段。"""
    return {
        "sample_id": context["sample_id"],
        "task": context["task"],
        "variant_id": context["variant_id"],
        "mode": context["mode"],
        "radius": context["radius"],
        "seed": context["seed"],
        "site_id": current_site,
        "receptor_scope": receptor_scope,
        "truth_ligand_label": truth_ligand,
        "selected_ligand_label": selected_ligand,
        "selection_source": selection_source,
    }


def add_pose_metrics(
    row: dict[str, Any],
    result: dict[str, Any] | None,
    truth: dict[str, Any] | None,
) -> dict[str, Any]:
    """
    将一个正确身份 pair 的 Rosetta pose 转为 RMSD 指标。

    输入参数:
        - row: dict[str, Any], 当前输出行的定位字段
        - result: dict[str, Any] 或 None, 当前 pair 的 Rosetta 执行结果
        - truth: dict[str, Any] 或 None, native ligand 坐标来源

    输出:
        - row: dict[str, Any], 添加流程状态、RMSD、warning 与阈值命中字段后的输出行
    """
    if result is None or truth is None or not result.get("success"):
        row.update(
            {
                "pose_success": False,
                "rmsd": "",
                "rmsd_method": "",
                "rmsd_warning": "missing_truth_or_pose_not_runnable",
            }
        )
        for threshold in RMSD_THRESHOLDS:
            row[f"rmsd_le_{threshold:g}A"] = False
        return row
    job = result["job"]
    try:
        from docking_pipeline.shape_scoring import mol2_heavy_atom_xyz

        truth_xyz = mol2_heavy_atom_xyz(Path(str(truth["mol2_path"])))
        pose_xyz = first_output_ligand_xyz(Path(str(job["output_dir"])), str(job["ligand"]["rosetta_name"]))
        rmsd, warning = direct_rmsd(truth_xyz, pose_xyz)
        row.update(
            {
                "pose_success": True,
                "rmsd": rmsd,
                "rmsd_method": "direct_atom_order",
                "rmsd_warning": warning,
            }
        )
        for threshold in RMSD_THRESHOLDS:
            row[f"rmsd_le_{threshold:g}A"] = warning == "" and rmsd <= threshold
    except Exception as exc:  # noqa: BLE001 - 评价必须保留 warning 后继续统计其他 pose
        row.update(
            {
                "pose_success": False,
                "rmsd": "",
                "rmsd_method": "direct_atom_order",
                "rmsd_warning": f"{type(exc).__name__}: {exc}",
            }
        )
        for threshold in RMSD_THRESHOLDS:
            row[f"rmsd_le_{threshold:g}A"] = False
    return row


def rebuild_pair_scores(
    context: dict[str, Any],
    results: dict[tuple[str, str, str], dict[str, Any]],
) -> dict[str, dict[tuple[str, str], Any]]:
    """
    按 oracle runner 的当前成本定义重建全部真实 pair 分数。

    输入参数:
        - context: dict[str, Any], 当前 variant 元数据
        - results: dict, 当前 variant 的 Rosetta result 查询表

    输出:
        - scores: dict[str, dict[tuple[str, str], PairScore]], 按 receptor scope 索引的成本矩阵
    """
    ligand_labels = sorted({key[1] for key in results})
    scores: dict[str, dict[tuple[str, str], Any]] = {}
    for receptor_scope in context["receptor_scopes"]:
        features: list[RawPairFeatures] = []
        for current_site in sorted(context["site_truth"]):
            for ligand_label in ligand_labels:
                result = results.get((current_site, ligand_label, receptor_scope))
                score_values = {} if result is None else result.get("score_values", {})
                d_g = safe_float(score_values.get("dG"), 9999.0)
                features.append(
                    RawPairFeatures(
                        site_id=current_site,
                        ligand_label=ligand_label,
                        receptor_scope=receptor_scope,
                        dG=d_g,
                        shape=0.0,
                    )
                )
        scores[receptor_scope] = {
            (score.site_id, score.ligand_label): score
            for score in build_pair_scores(features, MatchingOptions.current())
        }
    return scores


def build_assignment_rows(
    context: dict[str, Any],
    assignments: list[dict[str, Any]],
    pair_scores: dict[str, dict[tuple[str, str], Any]],
) -> list[dict[str, Any]]:
    """
    统计 Hungarian assignment 是否为真实 site-ligand occurrence 匹配。

    输出中同时记录真实 assignment cost 与当前被选 assignment cost 的比值。
    """
    rows: list[dict[str, Any]] = []
    for assignment in assignments:
        receptor_scope = str(assignment["receptor_scope"])
        selected = {str(pair["site_id"]): str(pair["ligand_label"]) for pair in assignment.get("pairs", [])}
        score_map = pair_scores[receptor_scope]
        truth_cost = sum(
            float(score_map[(current_site, truth_ligand)].combined_cost)
            for current_site, truth_ligand in context["site_truth"].items()
        )
        optimal_cost = float(assignment.get("total_cost", 0.0))
        correct_count = sum(selected.get(current_site, "") == truth_ligand for current_site, truth_ligand in context["site_truth"].items())
        num_sites = len(context["site_truth"])
        rows.append(
            {
                "sample_id": context["sample_id"],
                "task": context["task"],
                "variant_id": context["variant_id"],
                "radius": context["radius"],
                "seed": context["seed"],
                "receptor_scope": receptor_scope,
                "solver": assignment.get("solver", ""),
                "num_truth_sites": num_sites,
                "num_correct_pairs": correct_count,
                "assignment_accuracy": correct_count / num_sites if num_sites else "",
                "num_unmatched_sites": len(assignment.get("unmatched_sites", [])),
                "num_unmatched_ligands": len(assignment.get("unmatched_ligands", [])),
                "truth_assignment_cost": truth_cost,
                "optimal_assignment_cost": optimal_cost,
                "truth_to_optimal_cost_ratio": truth_cost / optimal_cost if optimal_cost > 1e-12 else "",
                "truth_minus_optimal_cost": truth_cost - optimal_cost,
            }
        )
    return rows


def build_rank_rows(
    context: dict[str, Any],
    assignments: list[dict[str, Any]],
    pair_scores: dict[str, dict[tuple[str, str], Any]],
) -> list[dict[str, Any]]:
    """按每个真实 site 的候选 ligand 成本排序，记录真实 pair 的 top-k/top-k% 位置。"""
    rows: list[dict[str, Any]] = []
    for assignment in assignments:
        receptor_scope = str(assignment["receptor_scope"])
        score_map = pair_scores[receptor_scope]
        for current_site, truth_ligand in context["site_truth"].items():
            candidates = sorted(
                (score for (site, _), score in score_map.items() if site == current_site),
                key=lambda score: float(score.combined_cost),
            )
            rank = next(
                index
                for index, score in enumerate(candidates, start=1)
                if score.ligand_label == truth_ligand
            )
            row = {
                "sample_id": context["sample_id"],
                "task": context["task"],
                "variant_id": context["variant_id"],
                "radius": context["radius"],
                "seed": context["seed"],
                "receptor_scope": receptor_scope,
                "site_id": current_site,
                "truth_ligand_label": truth_ligand,
                "num_candidates": len(candidates),
                "truth_pair_rank": rank,
                "truth_pair_rank_percent": rank / len(candidates) if candidates else "",
                "truth_pair_cost": float(score_map[(current_site, truth_ligand)].combined_cost),
            }
            for top_k in TOP_K_VALUES:
                row[f"top_{top_k}"] = rank <= top_k
            for top_percent in TOP_PERCENT_VALUES:
                row[f"top_{int(top_percent * 100)}pct"] = rank / len(candidates) <= top_percent if candidates else False
            rows.append(row)
    return rows


def variant_flow_row(sample_id: str, variant: dict[str, Any]) -> dict[str, Any]:
    """将一个 oracle variant 压平成流程统计行。"""
    return {
        "sample_id": sample_id,
        "task": variant.get("task", ""),
        "variant_id": variant.get("variant_id", ""),
        "mode": variant.get("mode", ""),
        "radius": variant.get("radius", ""),
        "seed": variant.get("seed", ""),
        "num_sites": variant.get("num_sites", ""),
        "num_jobs": variant.get("num_jobs", ""),
        "num_success": variant.get("num_success", ""),
    }


def summarize_flow(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """汇总全部 variant 的 Rosetta 流程跑通情况。"""
    num_jobs = sum(int(row["num_jobs"]) for row in rows)
    num_success = sum(int(row["num_success"]) for row in rows)
    return {
        "num_variants": len(rows),
        "num_jobs": num_jobs,
        "num_success": num_success,
        "runnable_rate": num_success / num_jobs if num_jobs else None,
    }


def summarize_pose_rows(rows: list[dict[str, Any]], strict_denominator: bool) -> dict[str, Any]:
    """
    汇总 pose RMSD。

    `strict_denominator=True` 时，错误 assignment、未跑通 pose 与带 warning pose 均保留在阈值成功率分母中。
    """
    reliable = [row for row in rows if row.get("rmsd_warning", "") == "" and row.get("rmsd") not in {"", None}]
    denominator = rows if strict_denominator else reliable
    summary: dict[str, Any] = {
        "num_rows": len(rows),
        "num_reliable_rmsd": len(reliable),
        "num_assignment_correct": sum(bool(row.get("assignment_correct")) for row in rows),
    }
    for threshold in RMSD_THRESHOLDS:
        hits = sum(bool(row.get(f"rmsd_le_{threshold:g}A")) for row in rows)
        summary[f"rmsd_le_{threshold:g}A"] = hits / len(denominator) if denominator else None
        summary[f"num_rmsd_le_{threshold:g}A"] = hits
    return summary


def summarize_assignments(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """汇总 Hungarian variant 的 occurrence 级 assignment accuracy。"""
    num_pairs = sum(int(row["num_truth_sites"]) for row in rows)
    num_correct = sum(int(row["num_correct_pairs"]) for row in rows)
    return {
        "num_assignments": len(rows),
        "num_truth_pairs": num_pairs,
        "num_correct_pairs": num_correct,
        "pair_accuracy": num_correct / num_pairs if num_pairs else None,
        "solver_counts": count_values(rows, "solver"),
    }


def summarize_ranks(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """汇总真实 pair 在每个 site 候选列表中的 top-k/top-k% 命中率。"""
    summary: dict[str, Any] = {"num_truth_pair_ranks": len(rows)}
    for top_k in TOP_K_VALUES:
        summary[f"top_{top_k}_rate"] = mean_bool(rows, f"top_{top_k}")
    for top_percent in TOP_PERCENT_VALUES:
        summary[f"top_{int(top_percent * 100)}pct_rate"] = mean_bool(rows, f"top_{int(top_percent * 100)}pct")
    return summary


def grouped_summary(
    strict_rows: list[dict[str, Any]],
    assignment_rows: list[dict[str, Any]],
    rank_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """按 `(task, receptor_scope)` 给出用户可直接比较的严格评价汇总。"""
    strict_groups = group_by_keys(strict_rows, ("task", "receptor_scope"))
    assignment_groups = group_by_keys(assignment_rows, ("task", "receptor_scope"))
    rank_groups = group_by_keys(rank_rows, ("task", "receptor_scope"))
    rows: list[dict[str, Any]] = []
    for key in sorted(set(strict_groups) | set(assignment_groups) | set(rank_groups)):
        pose_summary = summarize_pose_rows(strict_groups.get(key, []), strict_denominator=True)
        assignment_summary = summarize_assignments(assignment_groups.get(key, []))
        rank_summary = summarize_ranks(rank_groups.get(key, []))
        rows.append(
            {
                "task": key[0],
                "receptor_scope": key[1],
                **pose_summary,
                "assignment_pair_accuracy": assignment_summary.get("pair_accuracy"),
                "truth_top_1_rate": rank_summary.get("top_1_rate"),
                "truth_top_3_rate": rank_summary.get("top_3_rate"),
                "truth_top_10pct_rate": rank_summary.get("top_10pct_rate"),
            }
        )
    return rows


def group_by_keys(rows: list[dict[str, Any]], keys: tuple[str, ...]) -> dict[tuple[str, ...], list[dict[str, Any]]]:
    """按多个字段分组。"""
    grouped: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(str(row.get(key, "")) for key in keys)].append(row)
    return grouped


def count_values(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    """统计一个字段的值分布。"""
    counts: dict[str, int] = {}
    for row in rows:
        value = str(row.get(key, ""))
        counts[value] = counts.get(value, 0) + 1
    return counts


def mean_bool(rows: list[dict[str, Any]], key: str) -> float | None:
    """计算 bool 指标均值；没有行时返回 None。"""
    return sum(bool(row.get(key)) for row in rows) / len(rows) if rows else None


def safe_float(value: Any, default: float) -> float:
    """将 score 字段转换为 float；缺失或无法解析时返回显式惩罚值。"""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


if __name__ == "__main__":
    main()
