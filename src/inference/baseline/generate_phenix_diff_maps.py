from __future__ import annotations

"""
phenix 差图 baseline 预生成脚本(committed, 独立 + 幂等 + 只产出)。

设计(详见 docs/test_pipline/phenix.md 与 收尾实施计划.md):
    - phenix 是六组 baseline 里唯一依赖外部二进制的一组, 最脆弱、最依赖服务器环境;
      故由本脚本先在服务器把每个样本的差图跑好并对齐到 cache 网格, build_baseline_cache 只消费。
    - 路径传递走 A2 约定派生: 不写 JSON 字段, 生成端与消费端共用 build_baseline_cache.derive_phenix_map_path。
    - receptor-only 剔除只作用于喂 phenix 的 model 副本, 且仅当 model 源是 cif_gt_path(stardard)时剔除;
      strict 的 cif_path 本就无配体, 直接喂。phenix 实测可吃 mmCIF, 不转 PDB。
    - resolution 逐样本优先级: JSON 显式字段 resolution > CSV 查表 > 默认 3.5(warn)。

分层: 核心 generate_one_phenix_diff_map() 可被用户层薄壳直接 import 复用(单输入→单差图);
      本文件的批壳只负责遍历 protein_40/protein_110 × 两系统。

用法:
    python src/inference/baseline/generate_phenix_diff_maps.py \
        --phenix_output_root /home/penghongen/My_Project/EVAL_OUT/phenix_diff_maps \
        [--systems stardard strict] [--val_json ...] [--test_json ...] \
        [--phenix_bin ...] [--resolution_csv ...] [--no_skip_existing] \
        [--shard_index 0 --num_shards 30]
"""

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
project_root = str(PROJECT_ROOT)
if project_root in sys.path:
    sys.path.remove(project_root)
sys.path.insert(0, project_root)

from processedPDB_EMDB_binder.utils.mrc_tools import load_map, make_model_grid
from src.inference.parse_input import load_from_raw_cif
from src.inference.utils.receptor_strip import extract_receptor_cif
from src.inference.utils.utils import write_grid_as_map
from src.inference.baseline.build_baseline_cache import derive_phenix_map_path

# str, phenix 差图二进制默认路径(服务器); 换机器用 --phenix_bin 覆盖
DEFAULT_PHENIX_BIN = "/home/yangjy/software/phenix/build/bin/phenix.real_space_diff_map"
# str, EMDB 分辨率表默认路径; 列含 emdb_id,resolution
DEFAULT_RESOLUTION_CSV = "/home/penghongen/My_Project/Data/EMDB_PDB_resolution_3.5.csv"
# str, 验证集(protein_40) / 测试集(protein_110)样本列表默认路径
DEFAULT_VAL_JSON = "/home/penghongen/My_Project/Pocket_Plus/src/inference/utils/protein_40.json"
DEFAULT_TEST_JSON = "/home/penghongen/My_Project/Pocket_Plus/src/inference/utils/protein_110.json"
# str, phenix 预生成差图根目录默认路径(与 run_baseline_two_stage 默认一致)
DEFAULT_PHENIX_OUTPUT_ROOT = "/home/penghongen/My_Project/EVAL_OUT/phenix_diff_maps"
# float, resolution 兜底值; 仅在 JSON 字段与 CSV 都查不到时使用(非正式结论)
RESOLUTION_FALLBACK = 3.5
# list[str], 两个系统
SYSTEMS = ["stardard", "strict"]
# dict[str, dict[str, Any]], load_from_raw_cif 的参考网格 density 配置(只触发 exp 通道, 不需 sim)
_REFERENCE_GRID_DENSITY_CONFIG = {
    "clip_percentile": (0.001, 0.999),
    "fit_mask_percentile": 0.003,
    "enabled_channels": ["exp_clipnorm_nopost"],
}


def load_resolution_table(csv_path: str) -> dict[str, float]:
    """
    读取 EMDB 分辨率表。

    输入参数:
        - csv_path: str, CSV 路径, 需含 emdb_id 与 resolution 列

    输出:
        - table: dict[str, float], 标准化 EMDB ID(形如 EMD-33513)到 resolution 的映射; 文件不存在则返回空表并 warn
    """
    # dict[str, float], EMDB ID -> resolution
    table: dict[str, float] = {}
    if not os.path.exists(csv_path):
        print(f"[resolution] 警告: CSV 不存在 {csv_path}, 将仅依赖 JSON 字段或 3.5 兜底")
        return table
    with open(csv_path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            # str, 标准化 EMDB ID
            emdb_id = str(row["emdb_id"]).strip().upper().replace("_", "-")
            if not emdb_id:
                continue
            try:
                table[emdb_id] = float(row["resolution"])
            except (KeyError, ValueError):
                continue
    return table


def emdb_id_from_map_path(map_path: str) -> str:
    """
    从密度图路径推导标准化 EMDB ID。

    输入参数:
        - map_path: str, 样本实验密度图路径, 文件名通常为 emd_33513.map

    输出:
        - emdb_id: str, 标准化 ID, 形如 EMD-33513
    """
    # str, 文件名 stem(小写)
    stem = Path(map_path).stem.lower()
    if stem.startswith("emd_"):
        return "EMD-" + stem.split("_", 1)[1]
    if stem.startswith("emd-"):
        return "EMD-" + stem.split("-", 1)[1]
    return stem.upper()


def resolve_sample_resolution(sample: dict[str, Any], table: dict[str, float]) -> tuple[float, str]:
    """
    逐样本解析 phenix resolution: JSON 字段 resolution > CSV 查表 > 3.5 兜底。

    输入参数:
        - sample: dict[str, Any], 样本条目(直接来自 JSON, 保留全部原始字段)
        - table: dict[str, float], EMDB ID 到 resolution 的本地表

    输出:
        - resolution: float, 传给 phenix 的分辨率
        - source: str, 来源: json / csv / fallback_3.5
    """
    if sample.get("resolution") is not None:
        return float(sample["resolution"]), "json"
    # str, 当前样本 EMDB ID
    emdb_id = emdb_id_from_map_path(str(sample["map_path"]))
    if emdb_id in table:
        return table[emdb_id], "csv"
    print(f"[resolution] 警告: 样本 map={sample['map_path']} 的 resolution 未在 JSON/CSV 找到, 兜底用 {RESOLUTION_FALLBACK}(非正式结论)")
    return float(RESOLUTION_FALLBACK), "fallback_3.5"


def _newest_map_file(work_dir: Path, start_time: float) -> Path:
    """
    从 phenix 工作目录中找出本次命令新生成的 map 文件。

    输入参数:
        - work_dir: Path, phenix 工作目录
        - start_time: float, 命令启动前时间戳

    输出:
        - map_path: Path, 最新的 .map/.mrc/.ccp4 文件
    """
    # list[Path], 候选输出文件
    candidates: list[Path] = []
    for suffix in ("*.map", "*.mrc", "*.ccp4"):
        candidates.extend(work_dir.glob(suffix))
    # list[Path], 本次命令后新生成的文件
    fresh = [path for path in candidates if path.stat().st_mtime >= start_time - 1.0]
    if not fresh:
        raise FileNotFoundError(f"phenix 未在 {work_dir} 生成 map/mrc/ccp4 文件")
    return max(fresh, key=lambda path: path.stat().st_mtime)


def generate_one_phenix_diff_map(
    structure_path: str,
    map_path: str,
    resolution: float,
    out_path: str,
    phenix_bin: str,
    work_dir: str,
    strip_hetatm: bool,
) -> dict[str, Any]:
    """
    为单个样本生成并对齐 phenix 差图(核心函数, 可被用户层薄壳复用)。

    流程: (可选)receptor 剔除 -> phenix.real_space_diff_map -> 找新图 -> make_model_grid 对齐到 cache 网格
          -> 断言 shape/origin 一致 -> 写出 out_path。

    输入参数:
        - structure_path: str, phenix model 结构路径(.cif/.pdb)
        - map_path: str, 实验密度图路径
        - resolution: float, phenix resolution 参数
        - out_path: str, 对齐后差图输出路径(.mrc)
        - phenix_bin: str, phenix.real_space_diff_map 可执行文件路径
        - work_dir: str, 本样本 phenix 工作目录(放 model 副本/原始输出/日志)
        - strip_hetatm: bool, 是否先剔除所有 HETATM 得到 receptor-only mmCIF 再喂 phenix(stardard 为 True)

    输出:
        - stats: dict[str, Any], 含 out_path/aligned_shape/origin/min/max/mean/std 等诊断信息
    """
    work_path = Path(work_dir)
    work_path.mkdir(parents=True, exist_ok=True)

    # str, 实际喂给 phenix 的 model 路径; stardard 先剔 HETATM, strict 直接用原结构
    if strip_hetatm:
        model_path = str(work_path / "phenix_model_receptor.cif")
        _, success, error_msg, n_res, _ = extract_receptor_cif(structure_path, model_path)
        if not success:
            raise RuntimeError(f"receptor 剔除失败: {structure_path}: {error_msg}")
    else:
        model_path = structure_path

    # list[str], phenix 命令
    command = [str(phenix_bin), str(model_path), str(map_path), f"resolution={float(resolution):.4f}"]
    # float, 命令启动前时间戳(用于定位新输出)
    start_time = time.time()
    completed = subprocess.run(command, cwd=str(work_path), text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    (work_path / "phenix.stdout.txt").write_text(completed.stdout, encoding="utf-8")
    (work_path / "phenix.stderr.txt").write_text(completed.stderr, encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(f"phenix 失败 rc={completed.returncode}: {map_path}")

    # Path, phenix 原始差图输出
    raw_diff_path = _newest_map_file(work_path, start_time)

    # dict[str, Any], 参考网格(origin/voxel_size/full_shape_zyx 只依赖 map_path)
    reference = load_from_raw_cif(
        cif_path=str(model_path),
        map_path=str(map_path),
        sim_map_path=None,
        target_voxel_size=1.0,
        compute_density=False,
        select_first_model=True,
        error_dir=str(work_path / "errors"),
        density_channel_config=_REFERENCE_GRID_DENSITY_CONFIG,
    )
    # np.ndarray/np.ndarray/np.ndarray, phenix 原始差图的体素/体素大小/原点
    diff_grid, diff_voxel_size, diff_origin = load_map(str(raw_diff_path))
    # np.ndarray, (D,H,W), 重采样到 target_voxel_size=1.0 的差图; 同时拿到重采样后的 voxel_size/origin
    diff_resampled, _, diff_resampled_origin = make_model_grid(diff_grid, diff_voxel_size, diff_origin, target_voxel_size=1.0)

    # tuple[int, ...], 期望网格形状(与 cache resampled_emdb 一致)
    expected_shape = tuple(int(v) for v in reference["full_shape_zyx"])
    if tuple(int(v) for v in diff_resampled.shape) != expected_shape:
        raise ValueError(f"phenix 重采样 shape={diff_resampled.shape} 与 cache 网格 {expected_shape} 不一致: {map_path}")
    if not np.allclose(np.asarray(diff_resampled_origin, dtype=float), np.asarray(reference["origin"], dtype=float), atol=1e-3):
        raise ValueError(f"phenix 重采样 origin={diff_resampled_origin} 与 cache origin={reference['origin']} 不一致: {map_path}")

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    write_grid_as_map(np.asarray(diff_resampled, dtype=np.float32), str(out_path), reference["origin"], reference["voxel_size"])
    return {
        "out_path": str(out_path),
        "raw_diff_path": str(raw_diff_path),
        "model_path": str(model_path),
        "aligned_shape": [int(v) for v in diff_resampled.shape],
        "min": float(np.nanmin(diff_resampled)),
        "max": float(np.nanmax(diff_resampled)),
        "mean": float(np.nanmean(diff_resampled)),
        "std": float(np.nanstd(diff_resampled)),
    }


def _load_samples(json_path: str) -> list[dict[str, Any]]:
    """
    直接读取样本 JSON(保留 resolution 等全部原始字段, 不走会丢字段的 load_raw_pairs)。

    输入参数:
        - json_path: str, protein_40/protein_110 JSON 路径

    输出:
        - samples: list[dict[str, Any]], 每项补 sample_name(缺省取 cif_path 的 stem)
    """
    with open(json_path, "r", encoding="utf-8-sig") as handle:
        # list[dict[str, Any]], 原始样本条目
        items = json.load(handle)
    # list[dict[str, Any]], 补 sample_name 后的样本
    samples: list[dict[str, Any]] = []
    for item in items:
        sample = dict(item)
        if sample.get("sample_name") is None:
            sample["sample_name"] = Path(str(sample["cif_path"])).stem
        samples.append(sample)
    return samples


def build_phenix_tasks(samples: list[dict[str, Any]], systems: list[str]) -> list[dict[str, Any]]:
    """
    构造 phenix 生成任务表, 任务身份为 (system, sample_name)。

    输入参数:
        - samples: list[dict[str, Any]], protein_40 与 protein_110 合并后的样本; 同名样本复用第一条
        - systems: list[str], 要生成的系统名列表, 通常为 ["stardard", "strict"]

    输出:
        - tasks: list[dict[str, Any]], 每项包含 system 与 sample, 可直接进入分片或完整性检查
    """
    # dict[str, dict[str, Any]], sample_name -> 首次出现的样本条目
    sample_by_name: dict[str, dict[str, Any]] = {}
    for sample in samples:
        sample_name = str(sample["sample_name"])
        if sample_name not in sample_by_name:
            sample_by_name[sample_name] = sample

    # list[dict[str, Any]], 固定顺序任务表; 先按样本, 再交错 system
    tasks: list[dict[str, Any]] = []
    for sample in sample_by_name.values():
        for system in systems:
            tasks.append({"system": str(system), "sample": sample})
    return tasks


def resolve_shard_from_args_or_slurm(shard_index: int | None, num_shards: int | None) -> tuple[int | None, int | None]:
    """
    解析显式 shard 参数或 Slurm array 环境变量。

    输入参数:
        - shard_index: int|None, 当前分片编号; 显式传入时按 0-based 解释
        - num_shards: int|None, 分片总数; 显式传入时需与 shard_index 成对使用

    输出:
        - shard: tuple[int|None, int|None], 未启用分片时返回 (None, None)
    """
    if shard_index is not None or num_shards is not None:
        if shard_index is None or num_shards is None:
            raise ValueError("--shard_index 与 --num_shards 必须成对传入")
        if int(num_shards) <= 0:
            raise ValueError(f"--num_shards 必须为正整数, 实际为 {num_shards}")
        if int(shard_index) < 0 or int(shard_index) >= int(num_shards):
            raise ValueError(f"--shard_index={shard_index} 超出 [0, {int(num_shards) - 1}]")
        return int(shard_index), int(num_shards)

    if os.environ.get("SLURM_ARRAY_TASK_ID") is None:
        return None, None

    # Slurm 允许 --array=1-30, 因此用 TASK_MIN 平移成 0-based 分片编号。
    task_id = int(os.environ["SLURM_ARRAY_TASK_ID"])
    task_min = int(os.environ.get("SLURM_ARRAY_TASK_MIN", "0"))
    task_count = int(os.environ.get("SLURM_ARRAY_TASK_COUNT", "1"))
    resolved_index = task_id - task_min
    if resolved_index < 0 or resolved_index >= task_count:
        raise ValueError(
            f"SLURM array 编号不一致: task_id={task_id}, task_min={task_min}, task_count={task_count}, "
            f"解析得到 shard_index={resolved_index}"
        )
    return resolved_index, task_count


def summary_path_for_run(phenix_output_root: str, shard_index: int | None, num_shards: int | None) -> str:
    """
    生成本次运行独占写入的 summary 路径。

    输入参数:
        - phenix_output_root: str, phenix 差图根目录
        - shard_index: int|None, 当前分片编号; None 表示非 array 运行
        - num_shards: int|None, 分片总数; None 表示非 array 运行

    输出:
        - summary_path: str, JSON 摘要路径
    """
    if shard_index is None or num_shards is None:
        return os.path.join(str(phenix_output_root), "generate_summary.json")
    return os.path.join(str(phenix_output_root), "_shard_summaries", f"generate_summary_shard_{shard_index:04d}_of_{num_shards:04d}.json")


def _structure_source_for_system(sample: dict[str, Any], system: str) -> str:
    """
    按系统选择 phenix model 的结构来源。

    输入参数:
        - sample: dict[str, Any], 样本条目
        - system: str, stardard 用 cif_gt_path(含配体, 需剔除); strict 用 cif_path(无配体)

    输出:
        - structure_path: str, 选中的结构路径
    """
    # str, 结构字段名
    field = "cif_gt_path" if system == "stardard" else "cif_path"
    if sample.get(field) is None:
        raise KeyError(f"样本 {sample.get('sample_name')} 缺少 {field}(system={system})")
    return str(sample[field])


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="预生成并对齐 phenix 差图(protein_40/protein_110 × 两系统)。")
    parser.add_argument("--phenix_output_root", default=DEFAULT_PHENIX_OUTPUT_ROOT, help="对齐差图根目录(消费端按同规则派生)。")
    parser.add_argument("--systems", nargs="*", default=SYSTEMS, help="要生成的系统, 默认 stardard strict。")
    parser.add_argument("--val_json", default=DEFAULT_VAL_JSON, help="protein_40 样本列表 JSON。")
    parser.add_argument("--test_json", default=DEFAULT_TEST_JSON, help="protein_110 样本列表 JSON。")
    parser.add_argument("--extra_test_json", action="append", default=[], help="追加样本列表 JSON, 支持 label=/path/to/json 或直接路径。")
    parser.add_argument("--phenix_bin", default=DEFAULT_PHENIX_BIN, help="phenix.real_space_diff_map 可执行文件路径。")
    parser.add_argument("--resolution_csv", default=DEFAULT_RESOLUTION_CSV, help="EMDB 分辨率 CSV(emdb_id,resolution)。")
    parser.add_argument("--no_skip_existing", action="store_true", help="不跳过已存在的对齐差图(默认幂等跳过)。")
    parser.add_argument("--shard_index", type=int, default=None, help="当前分片编号(0-based); 未传时自动读取 SLURM_ARRAY_TASK_ID。")
    parser.add_argument("--num_shards", type=int, default=None, help="分片总数; 未传时自动读取 SLURM_ARRAY_TASK_COUNT。")
    args = parser.parse_args(argv)

    # bool, 是否幂等跳过已存在产物
    skip_existing = not bool(args.no_skip_existing)
    # dict[str, float], EMDB 分辨率表
    resolution_table = load_resolution_table(str(args.resolution_csv))
    # list[str], 追加样本列表 JSON 路径; label=path 只取 path
    extra_jsons = [str(value).split("=", 1)[-1] for value in args.extra_test_json]
    # list[dict[str, Any]], 合并 val/test/extra 的样本(按 system+sample_name 去重, 同名复用同一差图)
    all_samples = _load_samples(str(args.val_json)) + _load_samples(str(args.test_json))
    for extra_json in extra_jsons:
        all_samples.extend(_load_samples(extra_json))
    # tuple[int|None, int|None], 当前 array 分片设置; None 表示单进程全量串行
    shard_index, num_shards = resolve_shard_from_args_or_slurm(args.shard_index, args.num_shards)
    # list[dict[str, Any]], 完整任务表, 任务身份为 (system, sample_name)
    all_tasks = build_phenix_tasks(all_samples, [str(v) for v in args.systems])
    # list[dict[str, Any]], 当前进程负责的任务切片; 轮转分片可减轻慢样本拖尾
    selected_tasks = all_tasks if shard_index is None or num_shards is None else all_tasks[shard_index::num_shards]
    print(
        "[generate_phenix_diff_maps] "
        f"total_tasks={len(all_tasks)} selected_tasks={len(selected_tasks)} "
        f"shard_index={shard_index} num_shards={num_shards}"
    )

    # list[dict[str, Any]], 全部样本生成统计
    run_rows: list[dict[str, Any]] = []
    # list[dict[str, Any]], 已存在而跳过的样本记录
    skipped_rows: list[dict[str, Any]] = []
    # list[dict[str, Any]], 失败样本记录
    failures: list[dict[str, Any]] = []
    for task in selected_tasks:
        # str, 当前系统; dict[str, Any], 当前样本条目
        system = str(task["system"])
        sample = task["sample"]
        # str, 样本名
        sample_name = str(sample["sample_name"])
        # str, 对齐差图输出路径(A2 约定派生)
        out_path = derive_phenix_map_path(str(args.phenix_output_root), system, sample_name)
        if skip_existing and os.path.exists(out_path):
            skipped_rows.append({"system": system, "sample_name": sample_name, "out_path": out_path})
            print(f"[skip] system={system} sample={sample_name} 已存在 {out_path}")
            continue
        try:
            # str, phenix model 结构来源
            structure_path = _structure_source_for_system(sample, system)
            # float/str, 分辨率与来源
            resolution, resolution_source = resolve_sample_resolution(sample, resolution_table)
            # str, 本样本 phenix 工作目录
            work_dir = str(Path(out_path).parent / "_phenix_work")
            stats = generate_one_phenix_diff_map(
                structure_path=structure_path,
                map_path=str(sample["map_path"]),
                resolution=resolution,
                out_path=out_path,
                phenix_bin=str(args.phenix_bin),
                work_dir=work_dir,
                strip_hetatm=(system == "stardard"),
            )
            stats.update({"system": system, "sample_name": sample_name, "resolution": float(resolution), "resolution_source": resolution_source})
            run_rows.append(stats)
            print(f"[ok] system={system} sample={sample_name} resolution={resolution}({resolution_source}) -> {out_path}")
        except Exception as exc:
            failures.append({"system": system, "sample_name": sample_name, "out_path": out_path, "error": str(exc)})
            print(f"[fail] system={system} sample={sample_name}: {exc}")

    # 落一份生成统计摘要(便于排查与写进复现说明)
    os.makedirs(str(args.phenix_output_root), exist_ok=True)
    summary_path = summary_path_for_run(str(args.phenix_output_root), shard_index, num_shards)
    os.makedirs(str(Path(summary_path).parent), exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "shard_index": shard_index,
                "num_shards": num_shards,
                "total_tasks": len(all_tasks),
                "selected_tasks": len(selected_tasks),
                "generated": run_rows,
                "skipped": skipped_rows,
                "failures": failures,
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )
    print(f"[generate_phenix_diff_maps] done. 成功 {len(run_rows)}, 跳过 {len(skipped_rows)}, 失败 {len(failures)}, 摘要: {summary_path}")


if __name__ == "__main__":
    main()
