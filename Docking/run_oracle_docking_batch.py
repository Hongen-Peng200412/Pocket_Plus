from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import traceback
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np

DOCKING_DIR = Path(__file__).resolve().parent
if str(DOCKING_DIR) not in sys.path:
    sys.path.insert(0, str(DOCKING_DIR))

from docking_pipeline.config import MatchingOptions, RosettaOptions, ServerPaths
from docking_pipeline.io_utils import (
    load_cache_meta,
    load_resolution_table,
    read_ligand_candidates,
    write_json,
)
from docking_pipeline.matching import RawPairFeatures, build_pair_scores, solve_assignment_with_virtual_nodes
from docking_pipeline.records import AssignmentResult, DockingResult, InferenceSite, LigandCandidate
from docking_pipeline.rosetta import (
    build_docking_job,
    make_complex_pdb,
    run_molfile_to_params,
    run_rosetta_job,
    translate_ligand_to_site,
    write_galiganddock_xml,
)
from docking_pipeline.runner import _prepare_receptors, _to_jsonable
from docking_pipeline.shape_scoring import mol2_heavy_atom_xyz


TASKS = ("true_center_identity", "offset_center_identity", "true_center_hungarian", "offset_center_hungarian")


def main() -> None:
    """
    运行真实中心 / 偏移中心 oracle docking 批处理。

    输入参数:
        - CLI 参数, 包括 run_id、样本列表、四类任务、offset 半径、seed 数和 nstruct

    输出:
        - None; 输出写入 `/home/penghongen/分子对接尝试/pipeline_runs/{run_id}`
    """
    parser = argparse.ArgumentParser(description="运行 Pocket Plus oracle/easy20 docking 实验")
    parser.add_argument("--run-id", required=True, help="pipeline run ID")
    parser.add_argument("--sample-list", required=True, help="样本列表文件或逗号分隔样本 ID")
    parser.add_argument("--tasks", default=",".join(TASKS), help="逗号分隔任务名")
    parser.add_argument("--offset-radii", default="4,6,8", help="offset 半径列表, 单位 Å")
    parser.add_argument("--offset-seeds", type=int, default=3, help="每个半径的偏移 seed 数")
    parser.add_argument("--nstruct", type=int, default=5, help="每个 Rosetta job 的 decoy 数")
    parser.add_argument("--jobs", type=int, default=1, help="样本级 joblib 并发数")
    parser.add_argument("--receptors", default="true_receptor,cryoatom_receptor", help="逗号分隔 receptor 来源")
    parser.add_argument("--shard-id", help="array 分片 ID; 设置后只写 shard summary, 避免并发覆盖总表")
    parser.add_argument("--plain-assignment", action="store_true", help="Hungarian 任务使用普通矩形匹配, 默认使用虚拟节点")
    parser.add_argument("--dry-run", action="store_true", help="只生成输入和审计, 不执行 Rosetta")
    args = parser.parse_args()

    paths = ServerPaths.default()
    run_root = paths.allowed_root / "pipeline_runs" / args.run_id
    (run_root / "config").mkdir(parents=True, exist_ok=True)
    (run_root / "tables").mkdir(parents=True, exist_ok=True)
    sample_ids = resolve_sample_ids(args.sample_list)
    task_names = [item.strip() for item in args.tasks.split(",") if item.strip()]
    offset_radii = [float(item) for item in args.offset_radii.split(",") if item.strip()]
    receptor_names = [item.strip() for item in args.receptors.split(",") if item.strip()]
    config = {
        "run_id": args.run_id,
        "sample_ids": sample_ids,
        "tasks": task_names,
        "offset_radii": offset_radii,
        "offset_seeds": args.offset_seeds,
        "nstruct": args.nstruct,
        "jobs": args.jobs,
        "receptors": receptor_names,
        "dry_run": args.dry_run,
        "plain_assignment": args.plain_assignment,
    }
    config_dir = run_root / "config"
    tables_dir = run_root / "tables"
    if args.shard_id:
        config_dir = config_dir / "shards"
        tables_dir = tables_dir / "shards"
        config["shard_id"] = args.shard_id
    config_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    config_name = "run_config.json" if not args.shard_id else f"{args.shard_id}_run_config.json"
    sample_list_name = "sample_list.txt" if not args.shard_id else f"{args.shard_id}_sample_list.txt"
    write_json(config_dir / config_name, config)
    (config_dir / sample_list_name).write_text("\n".join(sample_ids) + "\n", encoding="utf-8")

    rosetta_options = replace(RosettaOptions.smoke(), nstruct=args.nstruct)
    payloads = [
        {
            "pdb_id": sample_id,
            "paths": paths,
            "rosetta_options": rosetta_options,
            "matching_options": MatchingOptions.current(),
            "run_id": args.run_id,
            "task_names": task_names,
            "offset_radii": offset_radii,
            "offset_seeds": args.offset_seeds,
            "receptor_names": receptor_names,
            "use_virtual_nodes": not args.plain_assignment,
            "dry_run": args.dry_run,
        }
        for sample_id in sample_ids
    ]
    summaries = parallel_map(payloads, args.jobs)
    batch_summary = {
        "run_id": args.run_id,
        "num_samples": len(summaries),
        "num_ok": sum(item.get("status") == "ok" for item in summaries),
        "num_failed": sum(item.get("status") == "failed" for item in summaries),
        "num_variants": sum(int(item.get("num_variants", 0)) for item in summaries),
        "num_jobs": sum(int(item.get("num_jobs", 0)) for item in summaries),
        "num_success": sum(int(item.get("num_success", 0)) for item in summaries),
        "samples": summaries,
    }
    summary_name = "batch_summary.json" if not args.shard_id else f"{args.shard_id}_summary.json"
    write_json(tables_dir / summary_name, batch_summary)
    print(json.dumps(batch_summary, ensure_ascii=False, indent=2))


def run_one(payload: dict[str, Any]) -> dict[str, Any]:
    """运行单样本 oracle 实验并捕获样本级异常。"""
    pdb_id = str(payload["pdb_id"]).lower()
    paths: ServerPaths = payload["paths"]
    run_id = str(payload["run_id"])
    sample_dir = paths.allowed_root / "pipeline_runs" / run_id / "samples" / pdb_id
    try:
        return run_oracle_sample(**payload)
    except Exception as exc:  # noqa: BLE001 - 批处理边界需要审计单样本失败
        audit_dir = sample_dir / "audit"
        audit_dir.mkdir(parents=True, exist_ok=True)
        error = {
            "pdb_id": pdb_id,
            "status": "failed",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        write_json(audit_dir / "error.json", error)
        return error


def run_oracle_sample(
    pdb_id: str,
    paths: ServerPaths,
    rosetta_options: RosettaOptions,
    matching_options: MatchingOptions,
    run_id: str,
    task_names: list[str],
    offset_radii: list[float],
    offset_seeds: int,
    receptor_names: list[str],
    use_virtual_nodes: bool,
    dry_run: bool,
) -> dict[str, Any]:
    """
    运行单样本四类 oracle/easy docking 实验。

    输入参数:
        - pdb_id: str, 小写 PDB ID
        - paths: ServerPaths, 服务器路径配置
        - rosetta_options: RosettaOptions, Rosetta 参数
        - matching_options: MatchingOptions, assignment 参数
        - run_id: str, 输出 run ID
        - task_names: list[str], 要运行的任务名
        - offset_radii: list[float], offset 半径
        - offset_seeds: int, 每个半径的 seed 数
        - receptor_names: list[str], receptor 来源白名单
        - use_virtual_nodes: bool, Hungarian 任务是否使用虚拟节点
        - dry_run: bool, 是否只生成输入

    输出:
        - summary: dict[str, Any], 样本级汇总
    """
    pdb_id = pdb_id.lower()
    sample_dir = paths.allowed_root / "pipeline_runs" / run_id / "samples" / pdb_id
    sample_dir.mkdir(parents=True, exist_ok=True)
    meta = load_cache_meta(paths.inference_root / pdb_id)
    resolution = load_resolution_table(paths.resolution_csv)[pdb_id]["resolution"]
    ligands = read_ligand_candidates(paths.ligand_mapping_csv, pdb_id)
    truth_sites = truth_ligand_sites(ligands)
    variants = build_variants(task_names, truth_sites, offset_radii, offset_seeds)
    if not ligands or not truth_sites:
        (sample_dir / "audit").mkdir(parents=True, exist_ok=True)
        summary = {"pdb_id": pdb_id, "status": "skipped_no_ligands", "num_jobs": 0, "num_success": 0, "num_variants": 0}
        write_json(sample_dir / "audit" / "summary.json", summary)
        return summary

    variant_summaries: list[dict[str, Any]] = []
    for variant in variants:
        variant_summaries.append(
            run_variant(
                pdb_id,
                paths,
                meta,
                sample_dir,
                variant,
                ligands,
                resolution,
                rosetta_options,
                matching_options,
                receptor_names,
                use_virtual_nodes,
                dry_run,
            )
        )
    summary = {
        "pdb_id": pdb_id,
        "status": "ok",
        "dry_run": dry_run,
        "work_dir": str(sample_dir),
        "rosetta_options": asdict(rosetta_options),
        "matching_options": asdict(matching_options),
        "ligands": [_to_jsonable(ligand) for ligand in ligands],
        "num_ligands": len(ligands),
        "num_variants": len(variant_summaries),
        "num_jobs": sum(int(item["num_jobs"]) for item in variant_summaries),
        "num_success": sum(int(item["num_success"]) for item in variant_summaries),
        "variants": variant_summaries,
    }
    audit_dir = sample_dir / "audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    write_json(audit_dir / "summary.json", _to_jsonable(summary))
    return summary


def run_variant(
    pdb_id: str,
    paths: ServerPaths,
    meta: dict[str, Any],
    sample_dir: Path,
    variant: dict[str, Any],
    ligands: list[LigandCandidate],
    resolution: float,
    rosetta_options: RosettaOptions,
    matching_options: MatchingOptions,
    receptor_names: list[str],
    use_virtual_nodes: bool,
    dry_run: bool,
) -> dict[str, Any]:
    """
    运行一个 task/variant 的 Rosetta job 矩阵。

    输入参数:
        - variant: dict[str, Any], 包含 task、variant_id、sites、mode 等字段

    输出:
        - summary: dict[str, Any], variant 级摘要
    """
    variant_dir = sample_dir / "variants" / str(variant["task"]) / str(variant["variant_id"])
    for sub in ("audit", "inputs/ligands", "inputs/true_receptor", "inputs/cryoatom_receptor", "inputs/complexes", "params", "xml", "logs", "outputs"):
        (variant_dir / sub).mkdir(parents=True, exist_ok=True)
    write_galiganddock_xml(variant_dir / "xml" / "gadock_density_smoke.xml", resolution, rosetta_options)
    receptors = [receptor for receptor in _prepare_receptors(pdb_id, paths, meta, variant_dir) if receptor.name in receptor_names]
    params_paths = {}
    for ligand in ligands:
        mol2_copy = variant_dir / "inputs" / "ligands" / f"{ligand.label}.mol2"
        shutil.copy2(ligand.mol2_path, mol2_copy)
        params_paths[ligand.label] = run_molfile_to_params(paths, ligand, mol2_copy, variant_dir / "params")

    jobs = []
    for site_record in variant["sites"]:
        site = site_record["site"]
        allowed_ligands = [site_record["ligand"]] if variant["mode"] == "identity" else ligands
        for ligand in allowed_ligands:
            ligand_pdb = variant_dir / "params" / f"{ligand.rosetta_name}_0001.pdb"
            translated = variant_dir / "inputs" / "complexes" / f"{ligand.label}_{site.site_id}_translated.pdb"
            translate_ligand_to_site(ligand_pdb, translated, site)
            for receptor in receptors:
                job = build_docking_job(pdb_id, site, ligand, receptor, variant_dir)
                make_complex_pdb(receptor.pdb_path, translated, job.complex_pdb)
                jobs.append(job)
    results = [] if dry_run else [run_rosetta_job(paths, job, Path(meta["map_path"]), resolution, rosetta_options) for job in jobs]
    assignments = [] if dry_run or variant["mode"] == "identity" else assign_variant(results, matching_options, use_virtual_nodes)
    summary = {
        "task": variant["task"],
        "variant_id": variant["variant_id"],
        "mode": variant["mode"],
        "radius": variant.get("radius", ""),
        "seed": variant.get("seed", ""),
        "num_sites": len(variant["sites"]),
        "num_jobs": len(jobs),
        "num_results": len(results),
        "num_success": sum(result.success for result in results),
        "work_dir": str(variant_dir),
        "sites": [_to_jsonable(item["site"]) | {"truth_ligand_label": item["ligand"].label} for item in variant["sites"]],
        "results": _to_jsonable(results),
        "assignments": _to_jsonable(assignments),
    }
    write_json(variant_dir / "audit" / "summary.json", _to_jsonable(summary))
    write_json(variant_dir / "audit" / "results.json", _to_jsonable(results))
    write_json(variant_dir / "audit" / "assignments.json", _to_jsonable(assignments))
    return summary


def truth_ligand_sites(ligands: list[LigandCandidate]) -> list[dict[str, Any]]:
    """
    为每个真实 ligand 构造一个真实中心 site。

    输入参数:
        - ligands: list[LigandCandidate], 当前样本 dockable ligand 列表

    输出:
        - sites: list[dict[str, Any]], 每项包含 ligand 与对应 InferenceSite
    """
    sites: list[dict[str, Any]] = []
    for index, ligand in enumerate(ligands, start=1):
        coords = mol2_heavy_atom_xyz(ligand.mol2_path)
        center = tuple(float(value) for value in coords.mean(axis=0))
        site = InferenceSite(
            instance_id=index,
            center_world_xyz=center,
            score_mean=1.0,
            score_max=1.0,
            voxel_count=int(len(coords)),
        )
        sites.append({"ligand": ligand, "site": site})
    return sites


def build_variants(task_names: list[str], truth_sites: list[dict[str, Any]], offset_radii: list[float], offset_seeds: int) -> list[dict[str, Any]]:
    """
    根据任务名生成 task/variant 列表。

    输入参数:
        - task_names: list[str], 四类任务名
        - truth_sites: list[dict[str, Any]], 真实中心 sites
        - offset_radii: list[float], offset 半径
        - offset_seeds: int, 每个半径 seed 数

    输出:
        - variants: list[dict[str, Any]], 每个 variant 独立运行 assignment
    """
    variants: list[dict[str, Any]] = []
    for task in task_names:
        if task == "true_center_identity":
            variants.append({"task": task, "variant_id": "center", "mode": "identity", "sites": truth_sites})
        elif task == "true_center_hungarian":
            variants.append({"task": task, "variant_id": "center", "mode": "hungarian", "sites": truth_sites})
        elif task in {"offset_center_identity", "offset_center_hungarian"}:
            mode = "identity" if task.endswith("identity") else "hungarian"
            for radius in offset_radii:
                for seed in range(offset_seeds):
                    variants.append(
                        {
                            "task": task,
                            "variant_id": f"r{radius:g}_seed{seed}",
                            "mode": mode,
                            "radius": radius,
                            "seed": seed,
                            "sites": offset_sites(truth_sites, radius, seed),
                        }
                    )
        else:
            raise ValueError(f"unknown task: {task}")
    return variants


def offset_sites(truth_sites: list[dict[str, Any]], radius: float, seed: int) -> list[dict[str, Any]]:
    """对同一样本内每个真实中心生成可复现的球内均匀偏移。"""
    rng = np.random.default_rng(seed)
    sites: list[dict[str, Any]] = []
    for item in truth_sites:
        site: InferenceSite = item["site"]
        direction = rng.normal(size=3)
        direction = direction / np.linalg.norm(direction)
        distance = radius * (rng.random() ** (1.0 / 3.0))
        center = np.asarray(site.center_world_xyz, dtype=float) + direction * distance
        shifted = replace_site_center(site, tuple(float(value) for value in center))
        sites.append({"ligand": item["ligand"], "site": shifted})
    return sites


def replace_site_center(site: InferenceSite, center: tuple[float, float, float]) -> InferenceSite:
    """返回同一 instance_id 但中心不同的 site。"""
    return InferenceSite(
        instance_id=site.instance_id,
        center_world_xyz=center,
        score_mean=site.score_mean,
        score_max=site.score_max,
        voxel_count=site.voxel_count,
    )


def assign_variant(results: list[DockingResult], options: MatchingOptions, use_virtual_nodes: bool) -> list[AssignmentResult]:
    """对一个 variant 内每个 receptor scope 构造 assignment。"""
    site_ids = sorted({result.job.site.site_id for result in results})
    ligand_labels = sorted({result.job.ligand.label for result in results})
    receptor_scopes = sorted({result.job.receptor.name for result in results})
    assignments: list[AssignmentResult] = []
    for scope in receptor_scopes:
        features = []
        for site_id in site_ids:
            for ligand_label in ligand_labels:
                features.append(
                    RawPairFeatures(
                        site_id=site_id,
                        ligand_label=ligand_label,
                        receptor_scope=scope,
                        dG=aggregate_dg(results, scope, site_id, ligand_label),
                        shape=0.0,
                    )
                )
        scores = build_pair_scores(features, options)
        if use_virtual_nodes:
            assignments.append(
                solve_assignment_with_virtual_nodes(
                    scores,
                    site_ids,
                    ligand_labels,
                    site_ignore_costs={site_id: options.ignore_site_base_cost for site_id in site_ids},
                    ligand_missing_costs={ligand_label: options.missing_ligand_base_cost for ligand_label in ligand_labels},
                )
            )
        else:
            from docking_pipeline.matching import solve_assignment

            assignments.append(solve_assignment(scores, site_ids, ligand_labels))
    return assignments


def aggregate_dg(results: list[DockingResult], scope: str, site_id: str, ligand_label: str) -> float:
    """聚合一个 scope 内同一 site-ligand pair 的 best dG。"""
    values = [
        value
        for result in results
        if result.job.receptor.name == scope and result.job.site.site_id == site_id and result.job.ligand.label == ligand_label
        for value in [result.numeric_score("dG")]
        if value is not None
    ]
    return 9999.0 if not values else sum(values) / len(values)


def resolve_sample_ids(sample_list: str) -> list[str]:
    """解析样本列表参数。"""
    path = Path(sample_list)
    if path.exists():
        return [line.strip().lower() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [item.strip().lower() for item in sample_list.split(",") if item.strip()]


def parallel_map(payloads: list[dict[str, Any]], jobs: int) -> list[dict[str, Any]]:
    """使用 joblib 做样本级并行; joblib 不可用时顺序执行。"""
    if jobs <= 1:
        return [run_one(payload) for payload in payloads]
    try:
        from joblib import Parallel, delayed
    except ImportError:
        return [run_one(payload) for payload in payloads]
    return Parallel(n_jobs=jobs, backend="loky")(delayed(run_one)(payload) for payload in payloads)


if __name__ == "__main__":
    main()
