from __future__ import annotations

from dataclasses import asdict, is_dataclass
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from .config import MatchingOptions, RosettaOptions, ServerPaths
from .instance_postprocess import InstancePostprocessOptions, postprocess_sites
from .io_utils import (
    cif_to_receptor_pdb,
    load_cache_meta,
    load_instance_label,
    load_resolution_table,
    load_sites,
    load_voxel_transform,
    read_ligand_candidates,
    write_json,
)
from .matching import RawPairFeatures, build_pair_scores, solve_assignment, solve_assignment_with_virtual_nodes
from .records import AssignmentResult, DockingResult, ReceptorSource
from .rosetta import (
    build_docking_job,
    make_complex_pdb,
    run_molfile_to_params,
    run_rosetta_job,
    translate_ligand_to_site,
    write_galiganddock_xml,
)
from .shape_scoring import instance_voxel_xyz, mol2_heavy_atom_xyz, radial_shape_score


def run_sample_smoke(pdb_id: str, paths: ServerPaths, rosetta_options: RosettaOptions, matching_options: MatchingOptions) -> dict[str, object]:
    """
    运行一个样本的低成本 docking smoke pipeline。
    输入参数:
        - pdb_id: str, 当前样本 PDB ID
        - paths: ServerPaths, 服务器路径配置
        - rosetta_options: RosettaOptions, Rosetta 低成本参数
        - matching_options: MatchingOptions, 匹配权重配置

    输出:
        - summary: dict[str, object], 包含:
            - "pdb_id": str, 样本 ID
            - "work_dir": str, 样本工作目录
            - "num_jobs": int, Rosetta job 数
            - "num_success": int, 流程成功 job 数
            - "assignments": list[AssignmentResult], 每个 receptor scope 的匹配结果
    """
    return run_sample(
        pdb_id=pdb_id,
        paths=paths,
        rosetta_options=rosetta_options,
        matching_options=matching_options,
        instance_options=InstancePostprocessOptions.conservative(),
        run_id=f"smoke_{pdb_id.lower()}",
        use_virtual_nodes=True,
        dry_run=False,
    )


def run_sample(
    pdb_id: str,
    paths: ServerPaths,
    rosetta_options: RosettaOptions,
    matching_options: MatchingOptions,
    instance_options: InstancePostprocessOptions,
    run_id: str,
    use_virtual_nodes: bool,
    dry_run: bool,
    max_sites_per_sample: int | None = None,
) -> dict[str, object]:
    """
    运行一个样本的可审计 docking pipeline。

    输入参数:
        - pdb_id: str, 当前样本 PDB ID
        - paths: ServerPaths, 服务器路径配置
        - rosetta_options: RosettaOptions, Rosetta 参数
        - matching_options: MatchingOptions, 匹配权重和虚拟节点基础成本
        - instance_options: InstancePostprocessOptions, 推理 instance 后处理参数
        - run_id: str, 当前批处理运行 ID, 用于隔离输出目录
        - use_virtual_nodes: bool, 是否使用虚拟节点 assignment
        - dry_run: bool, 是否只生成输入和审计, 不执行 Rosetta
        - max_sites_per_sample: int 或 None, 每个样本最多进入 docking 的 site 数; None 表示不截断

    输出:
        - summary: dict[str, object], 包含样本、输出目录、job 数、成功数、后处理和匹配摘要
    """
    pdb_id = pdb_id.lower()
    sample_dir = paths.inference_root / pdb_id
    run_root = paths.allowed_root / "pipeline_runs" / run_id
    work_dir = run_root / "samples" / pdb_id
    for sub in ["audit", "inputs/ligands", "inputs/true_receptor", "inputs/cryoatom_receptor", "inputs/complexes", "params", "xml", "logs", "outputs"]:
        (work_dir / sub).mkdir(parents=True, exist_ok=True)

    sites = load_sites(sample_dir)
    label = load_instance_label(sample_dir)
    origin, voxel_size = load_voxel_transform(sample_dir)
    postprocess = postprocess_sites(sites, label, origin, voxel_size, instance_options)
    selected_sites = _select_sites(postprocess.sites, max_sites_per_sample)
    meta = load_cache_meta(sample_dir)
    ligands = read_ligand_candidates(paths.ligand_mapping_csv, pdb_id)
    resolution = load_resolution_table(paths.resolution_csv)[pdb_id]["resolution"]
    write_galiganddock_xml(work_dir / "xml" / "gadock_density_smoke.xml", resolution, rosetta_options)

    if not selected_sites or not ligands:
        summary = _sample_summary(
            pdb_id=pdb_id,
            work_dir=work_dir,
            rosetta_options=rosetta_options,
            matching_options=matching_options,
            instance_options=instance_options,
            postprocess=postprocess.audit_dict(),
            selected_sites=selected_sites,
            max_sites_per_sample=max_sites_per_sample,
            ligands=ligands,
            results=[],
            assignments=[],
            planned_job_count=0,
            dry_run=dry_run,
            status="skipped_no_sites_or_ligands",
        )
        write_json(work_dir / "audit" / "summary.json", _to_jsonable(summary))
        return summary

    receptor_sources = _prepare_receptors(pdb_id, paths, meta, work_dir)
    params_paths = {}
    for ligand in ligands:
        mol2_copy = work_dir / "inputs" / "ligands" / f"{ligand.label}.mol2"
        shutil.copy2(ligand.mol2_path, mol2_copy)
        params_paths[ligand.label] = run_molfile_to_params(paths, ligand, mol2_copy, work_dir / "params")

    jobs = []
    for site in selected_sites:
        for ligand in ligands:
            ligand_pdb = work_dir / "params" / f"{ligand.rosetta_name}_0001.pdb"
            translated = work_dir / "inputs" / "complexes" / f"{ligand.label}_{site.site_id}_translated.pdb"
            translate_ligand_to_site(ligand_pdb, translated, site)
            for receptor in receptor_sources:
                job = build_docking_job(pdb_id, site, ligand, receptor, work_dir)
                make_complex_pdb(receptor.pdb_path, translated, job.complex_pdb)
                jobs.append(job)

    results = [] if dry_run else [
        run_rosetta_job(paths, job, Path(meta["map_path"]), resolution, rosetta_options)
        for job in jobs
    ]
    shape_scores = _shape_scores(postprocess.label, origin, voxel_size, selected_sites, ligands)
    assignments = [] if dry_run else _assign(results, shape_scores, matching_options, use_virtual_nodes)
    summary = _sample_summary(
        pdb_id=pdb_id,
        work_dir=work_dir,
        rosetta_options=rosetta_options,
        matching_options=matching_options,
        instance_options=instance_options,
        postprocess=postprocess.audit_dict(),
        selected_sites=selected_sites,
        max_sites_per_sample=max_sites_per_sample,
        ligands=ligands,
        results=results,
        assignments=assignments,
        planned_job_count=len(jobs),
        dry_run=dry_run,
        status="ok",
    )
    write_json(work_dir / "audit" / "summary.json", _to_jsonable(summary))
    write_json(work_dir / "audit" / "postprocess.json", postprocess.audit_dict())
    write_json(work_dir / "audit" / "results.json", _to_jsonable(results))
    write_json(work_dir / "audit" / "assignments.json", _to_jsonable(assignments))
    return summary


def _prepare_receptors(pdb_id: str, paths: ServerPaths, meta: dict[str, object], work_dir: Path) -> list[ReceptorSource]:
    """
    准备 true/cryoatom 两套 receptor PDB。
    输入参数:
        - pdb_id: str, 当前样本 PDB ID
        - paths: ServerPaths, 服务器路径配置
        - meta: dict[str, object], infer cache meta_json
        - work_dir: Path, 当前样本工作目录

    输出:
        - receptors: list[ReceptorSource], 两套 receptor 来源记录
    """
    true_cif = paths.true_receptor_root / f"{pdb_id.upper()}.cif"
    cryo_cif = Path(str(meta["cif_path"]))
    true_pdb = work_dir / "inputs" / "true_receptor" / f"{pdb_id}_true_receptor.pdb"
    cryo_pdb = work_dir / "inputs" / "cryoatom_receptor" / f"{pdb_id}_cryoatom_receptor.pdb"
    cif_to_receptor_pdb(true_cif, true_pdb)
    cif_to_receptor_pdb(cryo_cif, cryo_pdb)
    return [
        ReceptorSource("true_receptor", true_cif, true_pdb),
        ReceptorSource("cryoatom_receptor", cryo_cif, cryo_pdb),
    ]


def _shape_scores(label: np.ndarray, origin: np.ndarray, voxel_size: np.ndarray, sites, ligands) -> dict[tuple[str, str], float]:
    """
    计算当前第一版网络径向 shape score。
    输入参数:
        - label: np.ndarray, (D, H, W), int, 后处理后的 instance 标签图
        - origin: np.ndarray, (3,), 体素世界坐标原点
        - voxel_size: np.ndarray, (3,), xyz 体素大小
        - sites: list[InferenceSite], 预测位点
        - ligands: list[LigandCandidate], 候选 ligand

    输出:
        - scores: dict[tuple[str, str], float], key 为 `(site_id, ligand_label)`
    """
    bins = np.linspace(0.0, 12.0, 13)
    scores: dict[tuple[str, str], float] = {}
    for site in sites:
        pred_xyz = instance_voxel_xyz(label, site.instance_id, origin, voxel_size)
        for ligand in ligands:
            ligand_xyz = mol2_heavy_atom_xyz(ligand.mol2_path)
            scores[(site.site_id, ligand.label)] = radial_shape_score(pred_xyz, ligand_xyz, bins)["network_shape_score"]
    return scores


def _assign(
    results: list[DockingResult],
    shape_scores: dict[tuple[str, str], float],
    options: MatchingOptions,
    use_virtual_nodes: bool,
) -> list[AssignmentResult]:
    """
    对每个 receptor scope 构造 matching。
    输入参数:
        - results: list[DockingResult], Rosetta 结果列表
        - shape_scores: dict[tuple[str, str], float], 网络 shape 成本
        - options: MatchingOptions, 匹配参数
        - use_virtual_nodes: bool, 是否使用虚拟节点匹配

    输出:
        - assignments: list[AssignmentResult], 包含 true/cryoatom/mean 三种 scope
    """
    site_ids = sorted({result.job.site.site_id for result in results})
    ligand_labels = sorted({result.job.ligand.label for result in results})
    assignments: list[AssignmentResult] = []
    for scope in ["true_receptor", "cryoatom_receptor", "mean_receptor"]:
        features: list[RawPairFeatures] = []
        for site_id in site_ids:
            for ligand_label in ligand_labels:
                dG = _aggregate_dg(results, scope, site_id, ligand_label)
                features.append(
                    RawPairFeatures(
                        site_id=site_id,
                        ligand_label=ligand_label,
                        receptor_scope=scope,
                        dG=dG,
                        shape=shape_scores[(site_id, ligand_label)],
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
            assignments.append(solve_assignment(scores, site_ids, ligand_labels))
    return assignments


def _aggregate_dg(results: list[DockingResult], scope: str, site_id: str, ligand_label: str) -> float:
    """按 receptor scope 聚合一个 site-ligand pair 的 dG。"""
    values: list[float] = []
    for result in results:
        if result.job.site.site_id != site_id or result.job.ligand.label != ligand_label:
            continue
        if scope != "mean_receptor" and result.job.receptor.name != scope:
            continue
        value = result.numeric_score("dG")
        if value is not None:
            values.append(value)
    return 9999.0 if not values else sum(values) / len(values)


def _sample_summary(
    pdb_id: str,
    work_dir: Path,
    rosetta_options: RosettaOptions,
    matching_options: MatchingOptions,
    instance_options: InstancePostprocessOptions,
    postprocess: dict[str, Any],
    selected_sites: tuple,
    max_sites_per_sample: int | None,
    ligands: list,
    results: list[DockingResult],
    assignments: list[AssignmentResult],
    planned_job_count: int,
    dry_run: bool,
    status: str,
) -> dict[str, object]:
    """
    构造样本级审计摘要。

    输入参数:
        - pdb_id: str, 当前样本 PDB ID
        - work_dir: Path, 当前样本输出目录
        - rosetta_options: RosettaOptions, Rosetta 参数
        - matching_options: MatchingOptions, 匹配参数
        - instance_options: InstancePostprocessOptions, instance 后处理参数
        - postprocess: dict[str, Any], 后处理审计字典
        - selected_sites: tuple[InferenceSite, ...], 最终进入 docking 的 site
        - max_sites_per_sample: int 或 None, site 截断上限
        - ligands: list[LigandCandidate], 当前样本可对接候选 ligand
        - results: list[DockingResult], Rosetta 运行结果
        - assignments: list[AssignmentResult], 匹配结果
        - planned_job_count: int, 根据 site/ligand/receptor 矩阵计划生成的 job 数
        - dry_run: bool, 是否为 dry-run
        - status: str, 样本级状态

    输出:
        - summary: dict[str, object], JSON 友好的样本摘要
    """
    return {
        "pdb_id": pdb_id,
        "status": status,
        "dry_run": dry_run,
        "work_dir": str(work_dir),
        "rosetta_options": asdict(rosetta_options),
        "matching_options": asdict(matching_options),
        "instance_options": asdict(instance_options),
        "postprocess": postprocess,
        "max_sites_per_sample": max_sites_per_sample,
        "selected_sites": [_to_jsonable(site) for site in selected_sites],
        "num_selected_sites": len(selected_sites),
        "ligands": [_to_jsonable(ligand) for ligand in ligands],
        "num_ligands": len(ligands),
        "num_jobs": planned_job_count,
        "num_results": len(results),
        "num_success": sum(result.success for result in results),
        "results": _to_jsonable(results),
        "assignments": _to_jsonable(assignments),
    }


def _to_jsonable(value):
    """把 dataclass、Path、tuple 等对象转换成 JSON 友好结构。"""
    if is_dataclass(value):
        return _to_jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _select_sites(sites: tuple, max_sites_per_sample: int | None) -> tuple:
    """
    按置信度选择进入 docking 的 site。

    输入参数:
        - sites: tuple[InferenceSite, ...], 后处理后的候选 site
        - max_sites_per_sample: int 或 None, 最多保留 site 数; None 表示全部保留

    输出:
        - selected: tuple[InferenceSite, ...], 按原 instance_id 排序的入选 site
    """
    if max_sites_per_sample is None or len(sites) <= max_sites_per_sample:
        return tuple(sites)
    ranked = sorted(sites, key=lambda site: (site.score_max, site.score_mean, site.voxel_count), reverse=True)
    return tuple(sorted(ranked[:max_sites_per_sample], key=lambda site: site.instance_id))
