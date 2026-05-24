from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import subprocess
import time
from pathlib import Path

import numpy as np

from .config import RosettaOptions, ServerPaths
from .io_utils import read_scorefile_rows
from .records import DockingJob, DockingResult, InferenceSite, LigandCandidate, ReceptorSource


def write_galiganddock_xml(path: Path, resolution: float, options: RosettaOptions) -> None:
    """
    写入当前已验证的低成本 GALigandDock XML。
    输入参数:
        - path: Path, XML 输出路径
        - resolution: float, 当前样本 map 分辨率
        - options: RosettaOptions, Rosetta smoke test 参数

    输出:
        - None
    """
    text = f"""<ROSETTASCRIPTS>
  <SCOREFXNS>
    <ScoreFunction name="ga" weights="{options.scorefxn}"/>
  </SCOREFXNS>
  <MOVERS>
    <GALigandDock name="dock" scorefxn="ga" scorefxn_relax="ga"
      runmode="VSX" reference_pool="map" grid_radius="{options.grid_radius}"
      grid_step="{options.grid_step}" padding="{options.padding}" sidechains="none"
      nrelax="1" nreport="1" estimate_dG="true" entropy_method="Simple"
      local_resolution="{resolution}" skeleton_radius="{options.skeleton_radius}"
      method_for_radius="fixed" random_oversample="1.0" reference_oversample="1.0"
      reference_frac="0.8" debug_report="true" use_pharmacophore="false">
      <Stage repeats="1" npool="{options.npool}" smoothing="0.5" elec_scale="1.0"
        pmut="0.2" rmsdthreshold="2.0" maxiter="{options.maxiter}"
        pack_cycles="{options.pack_cycles}"/>
    </GALigandDock>
  </MOVERS>
  <PROTOCOLS>
    <Add mover_name="dock"/>
  </PROTOCOLS>
  <OUTPUT scorefxn="ga"/>
</ROSETTASCRIPTS>
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def run_molfile_to_params(paths: ServerPaths, ligand: LigandCandidate, mol2_copy: Path, params_dir: Path) -> Path:
    """
    为一个 mol2 生成 Rosetta params。
    输入参数:
        - paths: ServerPaths, 服务器路径配置
        - ligand: LigandCandidate, 当前候选 ligand
        - mol2_copy: Path, 已复制到允许目录的 mol2 文件
        - params_dir: Path, params 输出目录

    输出:
        - params_path: Path, 生成的 `.params` 文件路径
    """
    params_dir.mkdir(parents=True, exist_ok=True)
    command = [
        "python3",
        str(paths.molfile_to_params),
        "-n",
        ligand.rosetta_name,
        "-p",
        ligand.rosetta_name,
        str(mol2_copy),
    ]
    subprocess.run(command, cwd=params_dir, check=True, text=True, capture_output=True)
    return params_dir / f"{ligand.rosetta_name}.params"


def translate_ligand_to_site(ligand_pdb: Path, output_pdb: Path, site: InferenceSite) -> None:
    """
    将 ligand conformer 平移到预测 site 中心。
    输入参数:
        - ligand_pdb: Path, `molfile_to_params.py` 生成的 ligand PDB
        - output_pdb: Path, 平移后的 ligand PDB
        - site: InferenceSite, 目标预测位点

    输出:
        - None
    """
    lines = ligand_pdb.read_text(encoding="ascii").splitlines()
    coords = _pdb_coords(lines)
    delta = np.asarray(site.center_world_xyz, dtype=float) - coords.mean(axis=0)
    output_lines: list[str] = []
    for line in lines:
        if not line.startswith(("ATOM", "HETATM")):
            continue
        xyz = np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])]) + delta
        output_lines.append(line[:21] + "X" + f"{999:4d}" + line[26:30] + f"{xyz[0]:8.3f}{xyz[1]:8.3f}{xyz[2]:8.3f}" + line[54:])
    output_lines.append("TER")
    output_pdb.parent.mkdir(parents=True, exist_ok=True)
    output_pdb.write_text("\n".join(output_lines) + "\n", encoding="ascii")


def make_complex_pdb(receptor_pdb: Path, ligand_pdb: Path, output_pdb: Path) -> None:
    """
    拼接 receptor PDB 和已经平移的 ligand PDB。
    输入参数:
        - receptor_pdb: Path, receptor-only PDB
        - ligand_pdb: Path, 平移后的 ligand PDB
        - output_pdb: Path, complex PDB 输出路径

    输出:
        - None
    """
    receptor_lines = [line for line in receptor_pdb.read_text(encoding="ascii").splitlines() if not line.startswith("END")]
    if receptor_lines and receptor_lines[-1] != "TER":
        receptor_lines.append("TER")
    ligand_lines = [line for line in ligand_pdb.read_text(encoding="ascii").splitlines() if not line.startswith("END")]
    output_pdb.parent.mkdir(parents=True, exist_ok=True)
    output_pdb.write_text("\n".join(receptor_lines + ligand_lines + ["END"]) + "\n", encoding="ascii")


def build_docking_job(
    pdb_id: str,
    site: InferenceSite,
    ligand: LigandCandidate,
    receptor: ReceptorSource,
    work_dir: Path,
) -> DockingJob:
    """
    构造一个 docking job 的路径规格。
    输入参数:
        - pdb_id: str, 当前样本 PDB ID
        - site: InferenceSite, 当前预测位点
        - ligand: LigandCandidate, 当前候选 ligand
        - receptor: ReceptorSource, 当前 receptor
        - work_dir: Path, 当前样本工作目录

    输出:
        - job: DockingJob, 可交给 `run_rosetta_job` 执行的 job 规格
    """
    complex_pdb = work_dir / "inputs" / "complexes" / f"{pdb_id}_{receptor.name}_{ligand.label}_{site.site_id}.pdb"
    return DockingJob(
        pdb_id=pdb_id,
        site=site,
        ligand=ligand,
        receptor=receptor,
        complex_pdb=complex_pdb,
        params_path=work_dir / "params" / f"{ligand.rosetta_name}.params",
        xml_path=work_dir / "xml" / "gadock_density_smoke.xml",
        output_dir=work_dir / "outputs" / site.site_id / f"{receptor.name}_{ligand.label}",
        scorefile_name=f"{receptor.name}_{ligand.label}_{site.site_id}_score.sc",
    )


def rosetta_command(paths: ServerPaths, job: DockingJob, map_path: Path, resolution: float, options: RosettaOptions) -> list[str]:
    """
    构造 Rosetta 命令。
    输入参数:
        - paths: ServerPaths, 服务器路径配置
        - job: DockingJob, 当前 job 输入规格
        - map_path: Path, 真实 EMDB map
        - resolution: float, map 分辨率
        - options: RosettaOptions, Rosetta 参数

    输出:
        - command: list[str], 可传给 `subprocess.run` 的命令列表
    """
    return [
        str(paths.rosetta_bin),
        "-database",
        str(paths.rosetta_database),
        "-s",
        str(job.complex_pdb),
        "-parser:protocol",
        str(job.xml_path),
        "-extra_res_fa",
        str(job.params_path),
        "-edensity:mapfile",
        str(map_path),
        "-edensity:mapreso",
        str(resolution),
        "-edensity:cryoem_scatterers",
        "true",
        "-corrections::gen_potential",
        "true",
        "-out:path:all",
        str(job.output_dir),
        "-out:prefix",
        f"{job.receptor.name}_{job.ligand.label}_{job.site.site_id}_",
        "-out:file:scorefile",
        job.scorefile_name,
        "-nstruct",
        str(options.nstruct),
        "-overwrite",
    ]


def run_rosetta_job(paths: ServerPaths, job: DockingJob, map_path: Path, resolution: float, options: RosettaOptions) -> DockingResult:
    """
    执行一个 Rosetta job 并解析 scorefile。
    输入参数:
        - paths: ServerPaths, 服务器路径配置
        - job: DockingJob, 当前 job 输入规格
        - map_path: Path, 真实 EMDB map
        - resolution: float, map 分辨率
        - options: RosettaOptions, Rosetta 参数

    输出:
        - result: DockingResult, 包含进程状态、日志路径和 scorefile 字段
    """
    job.output_dir.mkdir(parents=True, exist_ok=True)
    log_dir = job.output_dir.parents[2] / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    stdout_log = log_dir / f"dock_{job.site.site_id}_{job.receptor.name}_{job.ligand.label}.stdout.log"
    stderr_log = log_dir / f"dock_{job.site.site_id}_{job.receptor.name}_{job.ligand.label}.stderr.log"
    command = rosetta_command(paths, job, map_path, resolution, options)
    start = time.time()
    process = subprocess.run(command, cwd=job.output_dir, text=True, capture_output=True)
    seconds = time.time() - start
    stdout_log.write_text(process.stdout, encoding="utf-8", errors="replace")
    stderr_log.write_text(process.stderr, encoding="utf-8", errors="replace")
    scorefile = job.output_dir / job.scorefile_name
    score_rows = read_scorefile_rows(scorefile) if scorefile.exists() else []
    values = _best_score_row(score_rows)
    decoy_summary = _decoy_summary(score_rows)
    output_pdb_exists = any(job.output_dir.glob("*.pdb"))
    return DockingResult(
        job=job,
        success=process.returncode == 0 and bool(score_rows) and output_pdb_exists,
        returncode=process.returncode,
        seconds=seconds,
        score_values=values,
        stdout_log=stdout_log,
        stderr_log=stderr_log,
        score_rows=score_rows,
        decoy_summary=decoy_summary,
    )


def run_rosetta_jobs(
    paths: ServerPaths,
    jobs: list[DockingJob],
    map_path: Path,
    resolution: float,
    options: RosettaOptions,
    rosetta_jobs: int,
) -> list[DockingResult]:
    """
    在一个样本内并行执行相互独立的 Rosetta docking jobs。

    输入参数:
        - paths: ServerPaths, 服务器路径配置
        - jobs: list[DockingJob], 已准备好输入且输出路径互不冲突的 Rosetta jobs
        - map_path: Path, 当前样本真实 EMDB map
        - resolution: float, 当前样本 map 分辨率
        - options: RosettaOptions, Rosetta 参数
        - rosetta_jobs: int, 同一样本内允许同时运行的 Rosetta 子进程数; 应不超过申请 CPU 数

    输出:
        - results: list[DockingResult], 与 `jobs` 输入顺序一致的执行结果
    """
    if rosetta_jobs <= 1 or len(jobs) <= 1:
        return [run_rosetta_job(paths, job, map_path, resolution, options) for job in jobs]
    max_workers = min(rosetta_jobs, len(jobs))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        return list(executor.map(lambda job: run_rosetta_job(paths, job, map_path, resolution, options), jobs))


def _pdb_coords(lines: list[str]) -> np.ndarray:
    """
    从 PDB 行中提取坐标。
    输入参数:
        - lines: list[str], PDB 文本行

    输出:
        - coords: np.ndarray, (N, 3), ATOM/HETATM 坐标
    """
    coords = [
        [float(line[30:38]), float(line[38:46]), float(line[46:54])]
        for line in lines
        if line.startswith(("ATOM", "HETATM"))
    ]
    return np.asarray(coords, dtype=float)


def _best_score_row(rows: list[dict[str, str]]) -> dict[str, str]:
    """
    从所有 decoy 中选择默认用于 matching 的 best row。

    输入参数:
        - rows: list[dict[str, str]], scorefile 中所有 decoy 字段

    输出:
        - row: dict[str, str], 默认按 `dG` 最低选择; 无 `dG` 时按 `total_score` 或 `score` 选择
    """
    if not rows:
        return {}
    for key in ["dG", "total_score", "score"]:
        numeric_rows = [(row, _safe_float(row.get(key))) for row in rows]
        numeric_rows = [(row, value) for row, value in numeric_rows if value is not None]
        if numeric_rows:
            return min(numeric_rows, key=lambda item: item[1])[0]
    return rows[-1]


def _decoy_summary(rows: list[dict[str, str]]) -> dict[str, float]:
    """
    汇总所有 decoy 的数值字段。

    输入参数:
        - rows: list[dict[str, str]], scorefile 中所有 decoy 字段

    输出:
        - summary: dict[str, float], 包含 `num_decoys` 以及每个数值字段的 best/mean/std
    """
    summary: dict[str, float] = {"num_decoys": float(len(rows))}
    if not rows:
        return summary
    keys = sorted({key for row in rows for key in row})
    for key in keys:
        values = [_safe_float(row.get(key)) for row in rows]
        numeric_values = np.asarray([value for value in values if value is not None], dtype=float)
        if numeric_values.size == 0:
            continue
        summary[f"{key}_best"] = float(numeric_values.min())
        summary[f"{key}_mean"] = float(numeric_values.mean())
        summary[f"{key}_std"] = float(numeric_values.std())
    return summary


def _safe_float(value: str | None) -> float | None:
    """将 scorefile 字段转成 float, 非数值字段返回 None。"""
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        return None
