from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

DOCKING_DIR = Path(__file__).resolve().parents[1]
if str(DOCKING_DIR) not in sys.path:
    sys.path.insert(0, str(DOCKING_DIR))

from docking_pipeline.config import ServerPaths
from docking_pipeline.io_utils import read_ligand_candidates, write_json
from docking_pipeline.shape_scoring import mol2_heavy_atom_xyz


SITE_HIT_THRESHOLDS = (3.0, 4.0, 8.0)
RMSD_THRESHOLDS = (2.0, 3.0, 5.0)


def main() -> None:
    """
    评估已有 docking run 的前置位点和 pose 几何质量. 

    输入参数:
        - CLI 参数, 包括 eval_run_id、一个或多个 docking_run_id、是否限制样本数

    输出:
        - None; 评估表写入 `/home/penghongen/分子对接尝试/evaluation_runs/{eval_run_id}`
    """
    parser = argparse.ArgumentParser(description="评估 Pocket Plus docking run 的真实几何质量")
    parser.add_argument("--eval-run-id", required=True, help="evaluation run ID, 用于隔离输出目录")
    parser.add_argument("--docking-run-id", action="append", required=True, help="要读取的 docking run ID, 可重复传入")
    parser.add_argument("--max-samples", type=int, help="只评估前 N 个可发现样本, 用于小规模验证")
    args = parser.parse_args()

    paths = ServerPaths.default()
    output_dir = paths.allowed_root / "evaluation_runs" / args.eval_run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    sample_rows, pred_site_rows, pose_rows, assignment_rows, skipped_rows = load_docking_tables(paths, args.docking_run_id)
    if args.max_samples is not None:
        keep = {row["sample_id"] for row in sample_rows[: args.max_samples]}
        sample_rows = [row for row in sample_rows if row["sample_id"] in keep]
        pred_site_rows = [row for row in pred_site_rows if row["sample_id"] in keep]
        pose_rows = [row for row in pose_rows if row["sample_id"] in keep]
        assignment_rows = [row for row in assignment_rows if row["sample_id"] in keep]
        skipped_rows = [row for row in skipped_rows if row["sample_id"] in keep]

    sample_ids = sorted({row["sample_id"] for row in sample_rows})
    truth_rows = load_truth_instances(paths, sample_ids)
    site_hit_rows, site_hit_summary = evaluate_site_hits(truth_rows, pred_site_rows)
    pose_rmsd_rows, pose_rmsd_summary = evaluate_pose_rmsd(truth_rows, pose_rows)
    rank_rows, rank_summary = evaluate_rank_metrics(assignment_rows, truth_rows)

    write_csv(output_dir / "samples.csv", sample_rows)
    write_csv(output_dir / "truth_instances.csv", truth_rows)
    write_csv(output_dir / "pred_sites.csv", pred_site_rows)
    write_csv(output_dir / "pose_results.csv", pose_rows)
    write_csv(output_dir / "assignment_pairs.csv", assignment_rows)
    write_csv(output_dir / "site_hit_metrics.csv", site_hit_rows)
    write_csv(output_dir / "pose_rmsd_metrics.csv", pose_rmsd_rows)
    write_csv(output_dir / "rank_metrics.csv", rank_rows)
    write_csv(output_dir / "skipped_samples.csv", skipped_rows)

    summary = {
        "eval_run_id": args.eval_run_id,
        "docking_run_ids": args.docking_run_id,
        "num_samples_seen": len(sample_rows),
        "num_skipped_or_pending": len(skipped_rows),
        "num_truth_instances": len(truth_rows),
        "num_pred_sites": len(pred_site_rows),
        "num_pose_results": len(pose_rows),
        "num_assignment_pairs": len(assignment_rows),
        "site_hit_summary": site_hit_summary,
        "pose_rmsd_summary": pose_rmsd_summary,
        "rank_summary": rank_summary,
        "notes": [
            "流程跑通不等于对接成功; 真正成功率由本 evaluation 的 center hit、RMSD 与 rank/top k 指标定义。",
            "当前 RMSD 第一版使用 direct atom-order 对齐; 不可靠条目会写入 rmsd_warning。",
        ],
    }
    write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def load_docking_tables(paths: ServerPaths, docking_run_ids: list[str]) -> tuple[list[dict[str, Any]], ...]:
    """
    从一个或多个 docking run 的 audit 目录提取基础表. 

    输入参数:
        - paths: ServerPaths, 服务器路径配置
        - docking_run_ids: list[str], 要读取的 run ID 列表

    输出:
        - tables: tuple[list[dict[str, Any]], ...], 依次为样本、预测位点、pose、assignment、跳过样本表
    """
    sample_rows: list[dict[str, Any]] = []
    pred_site_rows: list[dict[str, Any]] = []
    pose_rows: list[dict[str, Any]] = []
    assignment_rows: list[dict[str, Any]] = []
    skipped_rows: list[dict[str, Any]] = []

    for run_id in docking_run_ids:
        samples_dir = paths.allowed_root / "pipeline_runs" / run_id / "samples"
        if not samples_dir.exists():
            skipped_rows.append({"run_id": run_id, "sample_id": "", "reason": "missing_samples_dir"})
            continue
        for sample_dir in sorted(path for path in samples_dir.iterdir() if path.is_dir()):
            audit_dir = sample_dir / "audit"
            sample_id = sample_dir.name.lower()
            summary_path = audit_dir / "summary.json"
            error_path = audit_dir / "error.json"
            if not summary_path.exists():
                skipped_rows.append(
                    {
                        "run_id": run_id,
                        "sample_id": sample_id,
                        "reason": "sample_error" if error_path.exists() else "pending_no_summary",
                    }
                )
                continue
            summary = read_json(summary_path)
            sample_rows.append(sample_summary_row(run_id, sample_id, summary))
            pred_site_rows.extend(pred_site_table(run_id, sample_id, summary))
            pose_rows.extend(pose_result_table(run_id, sample_id, read_json(audit_dir / "results.json") if (audit_dir / "results.json").exists() else []))
            assignment_rows.extend(
                assignment_pair_table(run_id, sample_id, read_json(audit_dir / "assignments.json") if (audit_dir / "assignments.json").exists() else [])
            )
    return sample_rows, pred_site_rows, pose_rows, assignment_rows, skipped_rows


def load_truth_instances(paths: ServerPaths, sample_ids: list[str]) -> list[dict[str, Any]]:
    """
    从 ligand mapping 和 native mol2 构建 GT ligand instance 表. 

    输入参数:
        - paths: ServerPaths, 服务器路径配置
        - sample_ids: list[str], 需要构建 GT 的样本 ID

    输出:
        - rows: list[dict[str, Any]], 每行是一个真实可对接 ligand instance
    """
    rows: list[dict[str, Any]] = []
    for sample_id in sample_ids:
        try:
            ligands = read_ligand_candidates(paths.ligand_mapping_csv, sample_id)
        except Exception as exc:  # noqa: BLE001 - evaluation 需要记录坏样本并继续
            rows.append({"sample_id": sample_id, "truth_error": f"{type(exc).__name__}: {exc}"})
            continue
        for ligand in ligands:
            try:
                coords = mol2_heavy_atom_xyz(ligand.mol2_path)
                center = coords.mean(axis=0)
                radius = float(np.linalg.norm(coords - center, axis=1).max()) if coords.size else math.nan
                rows.append(
                    {
                        "sample_id": sample_id,
                        "ccd_id": ligand.ccd_id,
                        "ligand_label": ligand.label,
                        "mol2_path": str(ligand.mol2_path),
                        "heavy_atoms": ligand.heavy_atoms,
                        "internal_metals": ";".join(ligand.internal_metals),
                        "truth_center_x": float(center[0]),
                        "truth_center_y": float(center[1]),
                        "truth_center_z": float(center[2]),
                        "truth_radius": radius,
                        "truth_error": "",
                    }
                )
            except Exception as exc:  # noqa: BLE001
                rows.append(
                    {
                        "sample_id": sample_id,
                        "ccd_id": ligand.ccd_id,
                        "ligand_label": ligand.label,
                        "mol2_path": str(ligand.mol2_path),
                        "heavy_atoms": ligand.heavy_atoms,
                        "internal_metals": ";".join(ligand.internal_metals),
                        "truth_error": f"{type(exc).__name__}: {exc}",
                    }
                )
    return rows


def evaluate_site_hits(truth_rows: list[dict[str, Any]], pred_site_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """
    计算预测 instance center 到 GT center 的宽松命中和一一匹配命中. 

    输入参数:
        - truth_rows: list[dict[str, Any]], GT ligand instance 表
        - pred_site_rows: list[dict[str, Any]], 预测 site 表

    输出:
        - result: tuple[list[dict[str, Any]], dict[str, Any]], 明细行与汇总; 明细仍以 GT ligand instance 为主行
    """
    rows: list[dict[str, Any]] = []
    by_sample_truth = group_by(truth_rows, "sample_id")
    by_sample_pred = group_by(pred_site_rows, "sample_id")
    total_truths = 0
    total_preds = 0
    total_matched_pairs = 0
    loose_hits = {threshold: 0 for threshold in SITE_HIT_THRESHOLDS}
    hungarian_hits = {threshold: 0 for threshold in SITE_HIT_THRESHOLDS}
    solver_counts: dict[str, int] = {}
    for sample_id in sorted(set(by_sample_truth) | set(by_sample_pred)):
        truths = [row for row in by_sample_truth.get(sample_id, []) if not row.get("truth_error")]
        preds = by_sample_pred.get(sample_id, [])
        total_truths += len(truths)
        total_preds += len(preds)
        matched_by_truth, solver = match_site_centers(truths, preds)
        solver_counts[solver] = solver_counts.get(solver, 0) + 1
        total_matched_pairs += len(matched_by_truth)
        for truth in truths:
            best = nearest_pred_site(truth, preds)
            matched = matched_by_truth.get(id(truth), {})
            row = {
                "sample_id": sample_id,
                "truth_ligand_label": truth.get("ligand_label", ""),
                "truth_ccd_id": truth.get("ccd_id", ""),
                "nearest_pred_site_id": best.get("site_id", ""),
                "nearest_center_distance": best.get("distance", ""),
                "hungarian_pred_site_id": matched.get("site_id", ""),
                "hungarian_center_distance": matched.get("distance", ""),
            }
            for threshold in SITE_HIT_THRESHOLDS:
                loose_hit = bool(best and best["distance"] <= threshold)
                hungarian_hit = bool(matched and matched["distance"] <= threshold)
                row[f"hit_le_{threshold:g}A"] = loose_hit
                row[f"hungarian_hit_le_{threshold:g}A"] = hungarian_hit
                loose_hits[threshold] += int(loose_hit)
                hungarian_hits[threshold] += int(hungarian_hit)
            rows.append(row)
    summary: dict[str, Any] = {
        "num_truth_instances": total_truths,
        "num_pred_sites": total_preds,
        "num_hungarian_matched_pairs": total_matched_pairs,
        "hungarian_solver_counts": solver_counts,
    }
    for threshold in SITE_HIT_THRESHOLDS:
        summary[f"hit_le_{threshold:g}A"] = loose_hits[threshold] / total_truths if total_truths else None
        summary[f"hungarian_recall_le_{threshold:g}A"] = hungarian_hits[threshold] / total_truths if total_truths else None
        summary[f"hungarian_precision_le_{threshold:g}A"] = hungarian_hits[threshold] / total_preds if total_preds else None
    return rows, summary


def match_site_centers(truths: list[dict[str, Any]], preds: list[dict[str, Any]]) -> tuple[dict[int, dict[str, Any]], str]:
    """
    对同一样本内的 GT center 和预测 center 做一一距离最小匹配. 

    输入参数:
        - truths: list[dict[str, Any]], 同一样本内有效 GT ligand instance 行
        - preds: list[dict[str, Any]], 同一样本内 selected pred site 行

    输出:
        - result: tuple[dict[int, dict[str, Any]], str], 以 `id(truth)` 为键的匹配预测 site 与 solver 名称
    """
    if not truths or not preds:
        return {}, "none_empty_side"
    cost = site_center_distance_matrix(truths, preds)
    row_indices, col_indices, solver = solve_rectangular_assignment(cost)
    matched: dict[int, dict[str, Any]] = {}
    for row_index, col_index in zip(row_indices, col_indices):
        pred = dict(preds[int(col_index)])
        pred["distance"] = float(cost[int(row_index), int(col_index)])
        matched[id(truths[int(row_index)])] = pred
    return matched, solver


def site_center_distance_matrix(truths: list[dict[str, Any]], preds: list[dict[str, Any]]) -> np.ndarray:
    """
    构建 GT center 到预测 center 的欧氏距离矩阵. 

    输入参数:
        - truths: list[dict[str, Any]], 长度 T, 每项包含 truth_center_x/y/z
        - preds: list[dict[str, Any]], 长度 P, 每项包含 pred_center_x/y/z

    输出:
        - distance: np.ndarray, (T, P), 单位 Å 的中心距离矩阵
    """
    truth_xyz = np.asarray([[row["truth_center_x"], row["truth_center_y"], row["truth_center_z"]] for row in truths], dtype=float)
    pred_xyz = np.asarray([[row["pred_center_x"], row["pred_center_y"], row["pred_center_z"]] for row in preds], dtype=float)
    diff = truth_xyz[:, None, :] - pred_xyz[None, :, :]
    return np.linalg.norm(diff, axis=2)


def solve_rectangular_assignment(cost: np.ndarray) -> tuple[np.ndarray, np.ndarray, str]:
    """
    求解矩形 Hungarian assignment, 返回参与一一匹配的行列索引. 

    输入参数:
        - cost: np.ndarray, (T, P), GT 与预测 site 的距离成本矩阵

    输出:
        - result: tuple[np.ndarray, np.ndarray, str], 匹配到的行索引、列索引和 solver 名称
    """
    try:
        from scipy.optimize import linear_sum_assignment

        row_indices, col_indices = linear_sum_assignment(cost)
        return row_indices, col_indices, "scipy_hungarian"
    except ImportError:
        if min(cost.shape) > 16:
            raise
        row_indices, col_indices = solve_rectangular_assignment_dp(cost)
        return row_indices, col_indices, "dp_rectangular"


def solve_rectangular_assignment_dp(cost: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    用 bitmask DP 求解小规模矩形 assignment. 

    输入参数:
        - cost: np.ndarray, (T, P), GT 与预测 site 的距离成本矩阵; `min(T, P) <= 16`

    输出:
        - result: tuple[np.ndarray, np.ndarray], 匹配到的行索引与列索引
    """
    transpose = cost.shape[0] > cost.shape[1]
    work_cost = cost.T if transpose else cost
    num_left, num_right = work_cost.shape
    states: dict[int, tuple[float, list[tuple[int, int]]]] = {0: (0.0, [])}
    for left_index in range(num_left):
        next_states: dict[int, tuple[float, list[tuple[int, int]]]] = {}
        for mask, (current_cost, current_pairs) in states.items():
            for right_index in range(num_right):
                bit = 1 << right_index
                if mask & bit:
                    continue
                next_mask = mask | bit
                next_cost = current_cost + float(work_cost[left_index, right_index])
                if next_mask not in next_states or next_cost < next_states[next_mask][0]:
                    next_states[next_mask] = (next_cost, current_pairs + [(left_index, right_index)])
        states = next_states
    _, best_pairs = min(states.values(), key=lambda item: item[0])
    if transpose:
        row_indices = np.asarray([right for left, right in best_pairs], dtype=int)
        col_indices = np.asarray([left for left, right in best_pairs], dtype=int)
    else:
        row_indices = np.asarray([left for left, right in best_pairs], dtype=int)
        col_indices = np.asarray([right for left, right in best_pairs], dtype=int)
    return row_indices, col_indices


def evaluate_pose_rmsd(truth_rows: list[dict[str, Any]], pose_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """
    计算 docked pose 到同标签 GT ligand 的第一版 RMSD. 

    输入参数:
        - truth_rows: list[dict[str, Any]], GT ligand instance 表
        - pose_rows: list[dict[str, Any]], Rosetta pose 结果表

    输出:
        - result: tuple[list[dict[str, Any]], dict[str, Any]], RMSD 明细与汇总
    """
    truth_by_key = {(row.get("sample_id"), row.get("ligand_label")): row for row in truth_rows if not row.get("truth_error")}
    rows: list[dict[str, Any]] = []
    for pose in pose_rows:
        row = dict(pose)
        truth = truth_by_key.get((pose.get("sample_id"), pose.get("ligand_label")))
        if not truth or not pose.get("success"):
            row.update({"rmsd": "", "rmsd_method": "", "rmsd_warning": "missing_truth_or_pose_not_runnable"})
            rows.append(row)
            continue
        try:
            truth_xyz = mol2_heavy_atom_xyz(Path(str(truth["mol2_path"])))
            pose_xyz = first_output_ligand_xyz(Path(str(pose.get("output_dir", ""))), str(pose.get("rosetta_name", "")))
            rmsd, warning = direct_rmsd(truth_xyz, pose_xyz)
            row.update({"rmsd": rmsd, "rmsd_method": "direct_atom_order", "rmsd_warning": warning})
            for threshold in RMSD_THRESHOLDS:
                row[f"rmsd_le_{threshold:g}A"] = warning == "" and rmsd <= threshold
        except Exception as exc:  # noqa: BLE001
            row.update({"rmsd": "", "rmsd_method": "direct_atom_order", "rmsd_warning": f"{type(exc).__name__}: {exc}"})
        rows.append(row)
    valid = [float(row["rmsd"]) for row in rows if row.get("rmsd") not in {"", None} and row.get("rmsd_warning", "") == ""]
    summary: dict[str, Any] = {"num_valid_rmsd": len(valid), "num_pose_rows": len(rows)}
    for threshold in RMSD_THRESHOLDS:
        summary[f"rmsd_le_{threshold:g}A"] = sum(value <= threshold for value in valid) / len(valid) if valid else None
    return rows, summary


def evaluate_rank_metrics(assignment_rows: list[dict[str, Any]], truth_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """
    计算真实 ligand 在当前 assignment pairs 中的 rank/top k 接近程度. 

    输入参数:
        - assignment_rows: list[dict[str, Any]], assignment pair 表
        - truth_rows: list[dict[str, Any]], GT ligand instance 表

    输出:
        - result: tuple[list[dict[str, Any]], dict[str, Any]], rank 明细与汇总
    """
    truth_labels = {(row.get("sample_id"), row.get("ligand_label")) for row in truth_rows if not row.get("truth_error")}
    rows: list[dict[str, Any]] = []
    grouped = group_by(assignment_rows, "rank_group_id")
    for group_id, group_rows in grouped.items():
        ranked = sorted(group_rows, key=lambda row: safe_float(row.get("combined_cost"), math.inf))
        total = len(ranked)
        for rank, row in enumerate(ranked, start=1):
            is_truth_ligand = (row.get("sample_id"), row.get("ligand_label")) in truth_labels
            out = dict(row)
            out["rank"] = rank
            out["rank_percent"] = rank / total if total else ""
            out["is_truth_ligand_label"] = is_truth_ligand
            out["top1"] = rank <= 1
            out["top3"] = rank <= 3
            out["top10_percent"] = rank / total <= 0.10 if total else False
            rows.append(out)
    truth_rank_rows = [row for row in rows if row.get("is_truth_ligand_label")]
    summary = {
        "num_rank_rows": len(rows),
        "num_truth_rank_rows": len(truth_rank_rows),
        "truth_top1_rate": mean_bool(truth_rank_rows, "top1"),
        "truth_top3_rate": mean_bool(truth_rank_rows, "top3"),
        "truth_top10_percent_rate": mean_bool(truth_rank_rows, "top10_percent"),
    }
    return rows, summary


def sample_summary_row(run_id: str, sample_id: str, summary: dict[str, Any]) -> dict[str, Any]:
    """把 summary.json 压平成样本级表行. """
    return {
        "run_id": run_id,
        "sample_id": sample_id,
        "status": summary.get("status", ""),
        "dry_run": summary.get("dry_run", ""),
        "num_selected_sites": summary.get("num_selected_sites", ""),
        "num_ligands": summary.get("num_ligands", ""),
        "num_jobs": summary.get("num_jobs", ""),
        "num_results": summary.get("num_results", ""),
        "num_runnable_jobs": summary.get("num_success", ""),
        "work_dir": summary.get("work_dir", ""),
    }


def pred_site_table(run_id: str, sample_id: str, summary: dict[str, Any]) -> list[dict[str, Any]]:
    """从 summary.json 提取进入 docking 的预测位点表. """
    rows: list[dict[str, Any]] = []
    for site in summary.get("selected_sites", []):
        xyz = site.get("center_world_xyz") or ("", "", "")
        rows.append(
            {
                "run_id": run_id,
                "sample_id": sample_id,
                "site_id": f"site{int(site.get('instance_id')):03d}" if site.get("instance_id") is not None else "",
                "instance_id": site.get("instance_id", ""),
                "pred_center_x": xyz[0],
                "pred_center_y": xyz[1],
                "pred_center_z": xyz[2],
                "score_mean": site.get("score_mean", ""),
                "score_max": site.get("score_max", ""),
                "voxel_count": site.get("voxel_count", ""),
            }
        )
    return rows


def pose_result_table(run_id: str, sample_id: str, results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """从 results.json 提取 Rosetta job / pose 级表. """
    rows: list[dict[str, Any]] = []
    for result in results:
        job = result.get("job", {})
        ligand = job.get("ligand", {})
        site = job.get("site", {})
        receptor = job.get("receptor", {})
        score = result.get("score_values", {})
        output_dir = job.get("output_dir", "")
        rows.append(
            {
                "run_id": run_id,
                "sample_id": sample_id,
                "site_id": f"site{int(site.get('instance_id')):03d}" if site.get("instance_id") is not None else "",
                "ligand_label": ligand.get("label", ""),
                "ccd_id": ligand.get("ccd_id", ""),
                "rosetta_name": ligand.get("rosetta_name", ""),
                "heavy_atoms": ligand.get("heavy_atoms", ""),
                "internal_metals": ";".join(ligand.get("internal_metals", [])),
                "receptor_source": receptor.get("name", ""),
                "success": bool(result.get("success")),
                "returncode": result.get("returncode", ""),
                "seconds": result.get("seconds", ""),
                "dG": score.get("dG", ""),
                "dH": score.get("dH", ""),
                "lig_dens": score.get("lig_dens", ""),
                "ligscore": score.get("ligscore", ""),
                "total_score": score.get("total_score", ""),
                "score": score.get("score", ""),
                "num_decoys": result.get("decoy_summary", {}).get("num_decoys", ""),
                "output_dir": output_dir,
                "stdout_log": result.get("stdout_log", ""),
                "stderr_log": result.get("stderr_log", ""),
            }
        )
    return rows


def assignment_pair_table(run_id: str, sample_id: str, assignments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """从 assignments.json 提取当前 matching 输出的 pair 表. """
    rows: list[dict[str, Any]] = []
    for assignment_index, assignment in enumerate(assignments):
        receptor_scope = assignment.get("receptor_scope", "")
        group_id = f"{run_id}:{sample_id}:{receptor_scope}:{assignment_index}"
        for pair in assignment.get("pairs", []):
            rows.append(
                {
                    "run_id": run_id,
                    "sample_id": sample_id,
                    "rank_group_id": group_id,
                    "assignment_index": assignment_index,
                    "receptor_scope": receptor_scope,
                    "solver": assignment.get("solver", ""),
                    "total_cost": assignment.get("total_cost", ""),
                    "site_id": pair.get("site_id", ""),
                    "ligand_label": pair.get("ligand_label", ""),
                    "combined_cost": pair.get("combined_cost", ""),
                    "terms_json": json.dumps(pair.get("terms", {}), ensure_ascii=False, sort_keys=True),
                }
            )
    return rows


def first_output_ligand_xyz(output_dir: Path, rosetta_name: str) -> np.ndarray:
    """
    从 Rosetta 输出 PDB 中读取第一个 ligand 坐标. 

    输入参数:
        - output_dir: Path, 当前 Rosetta job 输出目录
        - rosetta_name: str, ligand 在 Rosetta 中的 residue 名

    输出:
        - coords: np.ndarray, (N, 3), ligand 重原子坐标
    """
    pdb_files = sorted(output_dir.glob("*.pdb"))
    if not pdb_files:
        raise FileNotFoundError(f"no output pdb in {output_dir}")
    coords: list[list[float]] = []
    for line in pdb_files[0].read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.startswith(("ATOM", "HETATM")):
            continue
        residue_name = line[17:20].strip()
        atom_name = line[12:16].strip()
        element = line[76:78].strip() or atom_name[:1]
        if residue_name != rosetta_name or element.upper().startswith("H"):
            continue
        coords.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
    if not coords:
        raise ValueError(f"no ligand atoms for residue {rosetta_name} in {pdb_files[0]}")
    return np.asarray(coords, dtype=float)


def direct_rmsd(truth_xyz: np.ndarray, pose_xyz: np.ndarray) -> tuple[float, str]:
    """按 atom order 直接计算 RMSD, 并返回可靠性 warning. """
    if truth_xyz.shape != pose_xyz.shape:
        n = min(len(truth_xyz), len(pose_xyz))
        if n == 0:
            return math.nan, "empty_truth_or_pose_atoms"
        diff = truth_xyz[:n] - pose_xyz[:n]
        return float(np.sqrt(np.mean(np.sum(diff * diff, axis=1)))), f"atom_count_mismatch_truth_{len(truth_xyz)}_pose_{len(pose_xyz)}"
    diff = truth_xyz - pose_xyz
    return float(np.sqrt(np.mean(np.sum(diff * diff, axis=1)))), ""


def nearest_pred_site(truth: dict[str, Any], preds: list[dict[str, Any]]) -> dict[str, Any]:
    """返回离一个 GT center 最近的预测 site. """
    if not preds:
        return {}
    truth_xyz = np.asarray([truth["truth_center_x"], truth["truth_center_y"], truth["truth_center_z"]], dtype=float)
    best: dict[str, Any] = {}
    best_distance = math.inf
    for pred in preds:
        pred_xyz = np.asarray([pred["pred_center_x"], pred["pred_center_y"], pred["pred_center_z"]], dtype=float)
        distance = float(np.linalg.norm(truth_xyz - pred_xyz))
        if distance < best_distance:
            best_distance = distance
            best = dict(pred)
            best["distance"] = distance
    return best


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """写出 CSV; 空表也保留表头占位文件. """
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def read_json(path: Path) -> Any:
    """读取 JSON 文件. """
    return json.loads(path.read_text(encoding="utf-8"))


def group_by(rows: list[dict[str, Any]], key: str) -> dict[str, list[dict[str, Any]]]:
    """按一个字段分组. """
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row.get(key, "")), []).append(row)
    return grouped


def mean_bool(rows: list[dict[str, Any]], key: str) -> float | None:
    """计算 bool 字段均值; 空列表返回 None. """
    if not rows:
        return None
    return sum(bool(row.get(key)) for row in rows) / len(rows)


def safe_float(value: Any, default: float) -> float:
    """把表字段转成 float; 失败时返回 default. """
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


if __name__ == "__main__":
    main()
