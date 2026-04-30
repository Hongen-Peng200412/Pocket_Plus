# -*- coding: utf-8 -*-
"""
================================================================================
生成 Chimera 命令文件：用纯受体 (CIF_3.5_atom) 按真实分辨率生成模拟密度图
================================================================================

核心流程:
  1. 读取 CSV (EMDB_ID, Resolution, PDB_ID)，按行展开多 PDB 情况
  2. 验证受体 CIF 文件存在，不存在则跳过并记录
  3. 分批生成 .cmd Chimera 命令文件（每批 ~100 条）
  4. 生成 SLURM 提交脚本（支持 sbatch 批量提交）
  5. 导出 manifest.csv（任务清单，含成功/失败统计）

输出文件结构:
  {OUTPUT_DIR}/
      cmd/                  ← Chimera 命令文件
          batch_000.cmd
          batch_001.cmd
          ...
      slurm/                ← SLURM 提交脚本
          run_all.sh         ← 一键提交全部批次
          submit_batch_XXX.sh
      manifest.csv           ← 任务清单
      log/                   ← SLURM 日志输出目录（需上传至服务器）

用法:
  python gen_chimera_cmds.py

  生成完毕后，将 cmd/ slurm/ log/ 上传至服务器对应目录，
  在服务器上执行: bash slurm/run_all.sh
================================================================================
"""

import os
import csv
import re
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from datetime import datetime
import warnings

# ============================================================================
# 服务器端路径配置（需上传至服务器后保持路径一致）
# ============================================================================

# str, EMDB-PDB 分辨率映射 CSV（包含列: emdb_id, resolution, fitted_pdbs）
# 服务器路径: /storage/penghongen/EMDB_PDB_resolution_3.5.csv
# 本地备份路径（本地运行时使用此路径）
CSV_PATH = r"C:\Users\15919\OneDrive\My_Project\Pocket_Plus\EMDB_PDB_resolution_3.5.csv"

# str, 纯受体 CIF 文件所在目录（get_receptor_from_PDB.py 的输出）
RECEPTOR_CIF_DIR = "/storage/chenzhaoyang/cryo_em/CIF_3.5_atom"

# str, 原始 EMDB 密度图所在目录
EMDB_MAP_DIR = "/storage/chenzhaoyang/cryo_em/EMDB_3.5"

# str, 模拟密度图输出目录（分子图将保存至此路径下）
SIMU_OUTPUT_DIR = "/storage/penghongen/simulated_receptor_map"

# str, 服务器上 output 目录的绝对路径
# 本地 output/ 由 run_sync.bat 自动同步至此路径，与本地目录结构完全一致
SERVER_OUTPUT_DIR = "/home/penghongen/My_Project/Pocket_Plus/Bundle_of_Maps/simulated_map/output"

# ============================================================================
# Chimera 命令模板（每条样本一条命令）
#
# 语法说明:
#   open ...                 — 打开纯受体 CIF
#   open ...                 — 打开原始 EMDB 密度图（定义目标网格）
#   volume #1 step 1         — 设置密度图采样步长为 1 Å
#   molmap #0 X onGrid #1     — 将受体结构按分辨率 X 生成模拟密度图，
#                               使用原始 EMDB map 的网格（对齐）
#   volume #2 save ...       — 将模拟图保存为 MRC 文件
#   close all                — 关闭当前样本，释放内存
# ============================================================================
CHIMERA_CMD_TEMPLATE = (
    "open {receptor_cif_path}\n"
    "open {emdb_map_path}\n"
    "volume #1 step 1\n"
    "molmap #0 {resolution} onGrid #1\n"
    "volume #2 save {simu_map_path}\n"
    "close all\n"
)

# 单批次 Chimera 命令文件末尾的统一结束语
CHIMERA_STOP_CMD = "stop\n"

# ============================================================================
# 生成配置
# ============================================================================

# int, 每批 Chimera 命令的最大条数（建议值 100，超出过长的任务可降低此值）
BATCH_SIZE = 100

# str, 本地输出根目录（生成的文件在此目录下，生成完毕后再上传至服务器）
OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "output"
)


# ============================================================================
# 工具函数
# ============================================================================

def normalize_emdb_id(emdb_id: str) -> str:
    """
    将 EMDB ID 标准化为文件名字格式.

    输入参数:
        - emdb_id: str, 标量, 原始 EMDB ID，如 "EMD-63092" 或 "emd-63092"

    输出:
        - str, 标量, 文件名格式，如 "emd_63092"
    """
    # 去除空格，转为小写，去除 "EMD-" / "emd-" 前缀，拼回 emd_ 前缀
    return "emd_" + emdb_id.strip().lower().replace("emd-", "").replace("EMD-", "")


def normalize_pdb_id(pdb_id: str) -> str:
    """
    将 PDB ID 标准化为大写格式.

    输入参数:
        - pdb_id: str, 标量, 原始 PDB ID，如 "9lhb" 或 "9Lhb"

    输出:
        - str, 标量, 大写格式，如 "9LHB"
    """
    return pdb_id.strip().upper()


def expand_csv_rows(csv_path: str) -> List[Dict]:
    """
    读取 CSV 文件并展开多 PDB 条目.

    原始 CSV 中 fitted_pdbs 列可能包含逗号分隔的多个 PDB（如 "6j8g,6j8h"），
    本函数将每行展开为多个独立条目，每个条目对应一个 PDB-EMDB 对。

    输入参数:
        - csv_path: str, CSV 文件路径

    输出:
        - list[dict], 展开后的条目列表，每个 dict 含:
            - emdb_id: str, 标准化后的 EMDB ID
            - resolution: float, 分辨率（Å）
            - pdb_id: str, 标准化后的 PDB ID
    """
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # str, 原始 EMDB ID
            raw_emdb = row.get("emdb_id", "").strip()
            # str, 原始 Resolution 字符串
            raw_res = row.get("resolution", "").strip()
            # str, 原始 PDB ID 列表（逗号分隔）
            raw_pdbs = row.get("fitted_pdbs", "").strip()

            # 跳过无效行
            if not raw_emdb or not raw_pdbs:
                continue
            if raw_emdb.lower() in ("nan", "") or raw_pdbs.lower() in ("nan", ""):
                continue

            # float, 分辨率（异常值跳过）
            try:
                resolution = float(raw_res)
            except ValueError:
                continue

            # list[str], 按逗号分割 PDB ID
            pdb_list = [p.strip() for p in raw_pdbs.split(",")]
            # str, 标准化后的 EMDB ID（整个行共用一个）
            emdb_id_norm = normalize_emdb_id(raw_emdb)

            for pdb in pdb_list:
                if not pdb or pdb.lower() == "nan":
                    continue
                pdb_norm = normalize_pdb_id(pdb)
                rows.append({
                    "emdb_id": emdb_id_norm,
                    "resolution": resolution,
                    "pdb_id": pdb_norm,
                })
    return rows




# ============================================================================
# 核心：生成 Chimera 命令文件
# ============================================================================

def build_chimera_commands(
    validated_rows: List[Dict],
    receptor_dir: str,
    emdb_dir: str,
    simu_output_dir: str,
) -> List[str]:
    """
    将有效条目转换为 Chimera 命令列表.

    输入参数:
        - validated_rows: list[dict], 通过验证的条目列表（每条含 emdb_id, resolution, pdb_id）
        - receptor_dir: str, 标量, 受体 CIF 目录
        - emdb_dir: str, 标量, EMDB map 目录
        - simu_output_dir: str, 标量, 模拟密度图输出目录

    输出:
        - list[str], Chimera 命令字符串列表（每个元素为一条样本的完整命令块）
    """
    commands = []
    for row in validated_rows:
        emdb_id = row["emdb_id"]
        resolution = row["resolution"]
        pdb_id = row["pdb_id"]
        # Chimera 命令中的路径必须使用正斜杠（Linux 服务器路径）
        # str, 纯受体 CIF 路径
        receptor_cif_path = f"{receptor_dir}/{pdb_id}.cif".replace("\\", "/")
        # str, 原始 EMDB 密度图路径（尝试 .map 后缀）
        emdb_map_path = f"{emdb_dir}/{emdb_id}.map".replace("\\", "/")
        # str, 模拟密度图输出路径
        simu_map_path = f"{simu_output_dir}/{emdb_id}.mrc".replace("\\", "/")

        cmd = CHIMERA_CMD_TEMPLATE.format(
            receptor_cif_path=receptor_cif_path,
            emdb_map_path=emdb_map_path,
            resolution=resolution,
            simu_map_path=simu_map_path,
        )
        commands.append(cmd)

    return commands


def write_batch_cmd_files(
    commands: List[str],
    output_dir: str,
    batch_size: int,
) -> Tuple[List[str], List[Tuple[int, int]]]:
    """
    将命令列表分批写入多个 .cmd 文件.

    输入参数:
        - commands: list[str], Chimera 命令列表, 由 build_chimera_commands 生成
        - output_dir: str, 标量, 输出目录（.cmd 文件写入此目录）
        - batch_size: int, 标量, 每批最大命令条数

    输出:
        - list[str]: 生成的文件路径列表
        - list[tuple[int, int]]: 每个批次对应的 (start_idx, end_idx)
    """
    cmd_dir = os.path.join(output_dir, "cmd")
    os.makedirs(cmd_dir, exist_ok=True)

    batch_paths = []
    batch_ranges = []

    for batch_idx in range(0, len(commands), batch_size):
        batch_commands = commands[batch_idx:batch_idx + batch_size]
        # str, 批次编号（补零 3 位）
        batch_label = f"{batch_idx // batch_size:03d}"
        # str, 文件名如 batch_000.cmd
        batch_filename = f"batch_{batch_label}.cmd"
        batch_path = os.path.join(cmd_dir, batch_filename)

        with open(batch_path, "w", encoding="utf-8") as f:
            for cmd in batch_commands:
                f.write(cmd + "\n\n")
            f.write(CHIMERA_STOP_CMD)

        batch_paths.append(batch_path)
        batch_ranges.append((batch_idx, batch_idx + len(batch_commands)))

    return batch_paths, batch_ranges


# ============================================================================
# 核心：生成 SLURM 提交脚本
# ============================================================================

def write_slurm_scripts(
    batch_paths: List[str],
    output_dir: str,
    chimera_bin: str,
    server_output_dir: str,
) -> Tuple[str, List[str]]:
    """
    生成 SLURM 提交脚本.

    脚本文件本身写入本地 output_dir（供上传），但脚本内容里所有路径（cmd 文件、log/err 日志、sbatch 路径）均使用 server_output_dir 下的服务器端 Linux 绝对路径，确保上传后可直接在服务器执行。

    输入参数:
        - batch_paths: list[str], 本地 .cmd 文件路径列表（仅用于提取文件名）
        - output_dir: str, 标量, 本地输出根目录（脚本文件写入此目录）
        - chimera_bin: str, 标量, 服务器上 Chimera 可执行文件路径
        - server_output_dir: str, 标量, 服务器上 output 目录的绝对路径脚本内容里的 cmd/log/err 路径均以此为根目录拼接

    输出:
        - str, 本地 run_all.sh 路径
        - list[str], 本地各个 submit_batch_XXX.sh 路径
    """
    # str, 本地写文件用的目录
    slurm_dir = os.path.join(output_dir, "slurm")
    os.makedirs(slurm_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "log"), exist_ok=True)  # 占位，上传后用

    individual_scripts = []         # list[str], 本地脚本路径列表
    server_script_paths = []        # list[str], 服务器侧脚本路径列表（供 run_all.sh 引用）

    for batch_path in batch_paths:
        # str, 批次文件名（无路径、无扩展名），如 "batch_000"
        batch_name = Path(batch_path).stem

        # 服务器侧路径（全部使用正斜杠）
        # str, 服务器上 .cmd 文件路径
        server_cmd_file  = f"{server_output_dir}/cmd/{batch_name}.cmd"
        # str, 服务器上 stdout 日志路径
        server_log_file  = f"{server_output_dir}/log/{batch_name}.log"
        # str, 服务器上 stderr 日志路径
        server_err_file  = f"{server_output_dir}/log/{batch_name}.err"
        # str, 服务器上本脚本自身路径（供 run_all.sh 中 sbatch 引用）
        server_script_path = f"{server_output_dir}/slurm/submit_{batch_name}.sh"

        # str, 本地脚本写入路径
        local_script_path = os.path.join(slurm_dir, f"submit_{batch_name}.sh")

        # SLURM 参数说明:
        #   -p cpu        分区名（cpu 队列）
        #   --qos=Cpu96   QOS 名称
        #   -N 1          单节点（CPU 运算不建议跨节点）
        #   --ntasks=1    1 个主进程
        #   --cpus-per-task=1  单核（Chimera 主要受 I/O 限制）
        slurm_content = (
            "#!/bin/bash\n"
            "#SBATCH -o {log_file}\n"
            "#SBATCH -e {err_file}\n"
            "#SBATCH -p cpu\n"
            "#SBATCH --qos=Cpu96\n"
            "#SBATCH -J {job_name}\n"
            "#SBATCH -N 1\n"
            "#SBATCH --ntasks=1\n"
            "#SBATCH --cpus-per-task=1\n"
            "\n"
            "{chimera_bin} {cmd_file}\n"
        ).format(
            job_name=f"simu_{batch_name}",
            log_file=server_log_file,
            err_file=server_err_file,
            chimera_bin=chimera_bin,
            cmd_file=server_cmd_file,
        )

        with open(local_script_path, "w", encoding="utf-8", newline="\n") as f:
            f.write(slurm_content)

        individual_scripts.append(local_script_path)
        server_script_paths.append(server_script_path)

    # 生成 run_all.sh：按顺序提交所有批次（sbatch 引用服务器侧路径）
    run_all_path = os.path.join(slurm_dir, "run_all.sh")
    run_all_content = "#!/bin/bash\n"
    run_all_content += "# " + "=" * 70 + "\n"
    run_all_content += "# 一键提交所有 Chimera 模拟密度图生成批次\n"
    run_all_content += f"# 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    run_all_content += f"# 服务器 output 目录: {server_output_dir}\n"
    run_all_content += "# " + "=" * 70 + "\n\n"
    run_all_content += "set -e\n\n"

    for local_sp, server_sp in zip(individual_scripts, server_script_paths):
        batch_name = Path(local_sp).stem.replace("submit_", "")
        run_all_content += f"# 提交批次 {batch_name}\n"
        run_all_content += f"sbatch {server_sp}\n"
        run_all_content += f"echo 'Submitted {batch_name}'\n\n"

    run_all_content += "echo '========================================'\n"
    run_all_content += "echo 'All batches submitted. Check log/ for status.'\n"
    run_all_content += "echo '========================================'\n"

    with open(run_all_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(run_all_content)

    return run_all_path, individual_scripts


# ============================================================================
# 核心：生成 manifest.csv
# ============================================================================

def write_manifest(
    validated_rows: List[Dict],
    batch_ranges: List[Tuple[int, int]],
    simu_output_dir: str,
    output_dir: str,
) -> str:
    """
    生成 manifest.csv 任务清单文件.

    输入参数:
        - validated_rows: list[dict], 有效条目列表
        - batch_ranges: list[tuple[int, int]], 每批覆盖的索引范围
        - simu_output_dir: str, 标量, 模拟密度图输出目录
        - output_dir: str, 标量, 本地输出根目录

    输出:
        - str, manifest.csv 文件路径
    """
    manifest_path = os.path.join(output_dir, "manifest.csv")

    # list[dict], manifest 行数据
    manifest_rows = []
    for row in validated_rows:
        # 确定该条目属于哪个批次
        row_idx = validated_rows.index(row)
        batch_idx = next(
            (i for i, (start, end) in enumerate(batch_ranges) if start <= row_idx < end),
            -1
        )
        batch_label = f"{batch_idx:03d}"

        # 服务器路径使用正斜杠（避免 Windows os.path.join 生成反斜杠）
        manifest_rows.append({
            "emdb_id": row["emdb_id"],
            "pdb_id": row["pdb_id"],
            "resolution": row["resolution"],
            "simu_map_path": f"{simu_output_dir}/{row['emdb_id']}.mrc".replace("\\", "/"),
            "batch": batch_label,
        })

    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["emdb_id", "pdb_id", "resolution", "simu_map_path", "batch"]
        )
        writer.writeheader()
        writer.writerows(manifest_rows)

    return manifest_path


# ============================================================================
# 主流程
# ============================================================================

def main(
    csv_path: str = CSV_PATH,
    receptor_cif_dir: str = RECEPTOR_CIF_DIR,
    emdb_map_dir: str = EMDB_MAP_DIR,
    simu_output_dir: str = SIMU_OUTPUT_DIR,
    output_dir: str = OUTPUT_DIR,
    batch_size: int = BATCH_SIZE,
):
    """主流程：读取 CSV → 验证文件 → 生成命令文件 + SLURM 脚本 + manifest."""
    print("=" * 70)
    print("Chimera 模拟密度图命令生成器")
    print("=" * 70)
    print(f"CSV 路径:           {csv_path}")
    print(f"受体 CIF 目录:      {receptor_cif_dir}")
    print(f"EMDB Map 目录:      {emdb_map_dir}")
    print(f"模拟图输出目录:     {simu_output_dir}")
    print(f"本地输出根目录:     {output_dir}")
    print(f"批次大小:           {batch_size}")
    print("=" * 70)

    # --- Step 1: 读取 CSV 并展开多 PDB 条目 ---
    print("\n[Step 1] 读取 CSV 并展开多 PDB 条目 ...")
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"CSV 文件不存在: {CSV_PATH}")

    raw_rows = expand_csv_rows(CSV_PATH)
    print(f"  展开后总条目数: {len(raw_rows)}")

    # --- Step 2: 跳过服务器端文件验证（本地无法访问服务器路径）---
    # CIF 和 EMDB Map 均在服务器上，本地生成命令时无需验证其存在性。
    # Chimera 在服务器执行时若文件缺失会自动报错，可通过 .err 日志定位。
    print("\n[Step 2] 跳过服务器端文件验证（本地模式）...")
    print(f"  ⚠ 本地无法访问服务器路径，所有 CSV 条目将全量生成命令。")
    print(f"  全量条目数: {len(raw_rows)}")
    # list[dict], 视所有条目为有效，命令按 CSV 全量生成
    validated_rows = raw_rows
    skipped_receptor = []  # list[dict], 本地模式下不校验，保持为空
    skipped_emdb = []      # list[dict], 本地模式下不校验，保持为空

    # --- Step 3: 生成 Chimera 命令列表 ---
    print("\n[Step 3] 生成 Chimera 命令 ...")
    commands = build_chimera_commands(
        validated_rows=validated_rows,
        receptor_dir=RECEPTOR_CIF_DIR,
        emdb_dir=EMDB_MAP_DIR,
        simu_output_dir=SIMU_OUTPUT_DIR,
    )
    print(f"  命令条数: {len(commands)}")

    # --- Step 4: 分批写入 .cmd 文件 ---
    print("\n[Step 4] 分批写入 Chimera 命令文件 ...")
    batch_paths, batch_ranges = write_batch_cmd_files(
        commands=commands,
        output_dir=OUTPUT_DIR,
        batch_size=BATCH_SIZE,
    )
    print(f"  生成批次数: {len(batch_paths)}")
    for bp in batch_paths:
        print(f"    {bp}")

    # --- Step 5: 生成 SLURM 提交脚本 ---
    print("\n[Step 5] 生成 SLURM 提交脚本 ...")
    # str, Chimera 在服务器上的二进制路径
    chimera_bin = "/home/chengbin/.local/UCSF-Chimera64-1.18/bin/chimera"
    run_all_path, individual_scripts = write_slurm_scripts(
        batch_paths=batch_paths,
        output_dir=OUTPUT_DIR,
        chimera_bin=chimera_bin,
        server_output_dir=SERVER_OUTPUT_DIR,
    )
    print(f"  run_all.sh:     {run_all_path}")
    for sp in individual_scripts:
        print(f"  {sp}")

    # --- Step 6: 生成 manifest.csv ---
    print("\n[Step 6] 生成 manifest.csv ...")
    manifest_path = write_manifest(
        validated_rows=validated_rows,
        batch_ranges=batch_ranges,
        simu_output_dir=SIMU_OUTPUT_DIR,
        output_dir=OUTPUT_DIR,
    )
    print(f"  manifest.csv:   {manifest_path}")

    # --- 统计摘要 ---
    print("\n" + "=" * 70)
    print("生成完毕 / Generation Complete")
    print("=" * 70)
    print(f"有效样本数:     {len(validated_rows)}")
    print(f"批次数量:       {len(batch_paths)}")
    print(f"跳过(受体缺失): {len(skipped_receptor)}")
    print(f"跳过(EMDB缺失): {len(skipped_emdb)}")
    print("=" * 70)
    print(f"\n输出根目录: {OUTPUT_DIR}")
    print("\n上传至服务器后，在服务器上执行:")
    print(f"  bash {run_all_path}")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="生成 Chimera 命令文件：为纯受体按真实分辨率生成模拟密度图"
    )
    parser.add_argument(
        "--csv", type=str, default=CSV_PATH,
        help="EMDB_PDB_resolution CSV 文件路径（默认: CSV_PATH 常量）"
    )
    parser.add_argument(
        "--receptor_dir", type=str, default=RECEPTOR_CIF_DIR,
        help="纯受体 CIF 文件所在目录（默认: RECEPTOR_CIF_DIR 常量）"
    )
    parser.add_argument(
        "--emdb_dir", type=str, default=EMDB_MAP_DIR,
        help="原始 EMDB 密度图所在目录（默认: EMDB_MAP_DIR 常量）"
    )
    parser.add_argument(
        "--output_dir", type=str, default=OUTPUT_DIR,
        help="本地输出根目录（默认: OUTPUT_DIR 常量）"
    )
    parser.add_argument(
        "--batch_size", type=int, default=BATCH_SIZE,
        help=f"每批 Chimera 命令条数（默认: {BATCH_SIZE}）"
    )
    parser.add_argument(
        "--simu_output_dir", type=str, default=SIMU_OUTPUT_DIR,
        help="模拟密度图在服务器上的输出目录（默认: SIMU_OUTPUT_DIR 常量）"
    )

    args = parser.parse_args()

    main(
        csv_path=args.csv,
        receptor_cif_dir=args.receptor_dir,
        emdb_map_dir=args.emdb_dir,
        simu_output_dir=args.simu_output_dir,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
    )