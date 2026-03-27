# -*- coding: utf-8 -*-
"""
================================================================================
从 PDB/CIF 结构文件中提取受体 (剔除所有 HETATM), 写出为 mmCIF
================================================================================

包含两个核心函数:
  1. extract_receptor_cif()  —— 解析单个结构文件, 剔除 HETATM, 写出 CIF
  2. batch_extract_receptor_cif() —— 批处理整个目录, 并打印统计摘要

用法:
  python get_receptor_from_PDB.py --input_dir <pdb_dir> --output_dir <out_dir>
================================================================================
"""

import os
import sys
import argparse
import warnings
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
from datetime import datetime

from Bio.PDB import PDBParser, MMCIFParser, MMCIFIO
from Bio.PDB.PDBIO import Select
from joblib import Parallel, delayed


# ============================================================================
# 核心: 残基级 HETATM 过滤器 / Residue-Level HETATM Filter
# ============================================================================

class ReceptorOnlySelect(Select):
    """
    Biopython Select 子类: 仅接受非 HETATM 的残基 (受体主链).

    过滤逻辑:
        - het_flag == ' ' (标准 ATOM 记录) → 保留
        - het_flag != ' ' (HETATM 记录, 包括水/配体/离子/修饰残基) → 剔除
    """

    def __init__(self):
        super().__init__()
        # int, 标量, 统计被保留的受体残基数
        self.n_accepted_residues = 0
        # int, 标量, 统计被保留的受体原子数
        self.n_accepted_atoms = 0

    def accept_residue(self, residue):
        """
        判断残基是否为受体 (非 HETATM).

        输入参数:
            - residue: Bio.PDB.Residue.Residue, Biopython 残基对象

        输出:
            - int, 1 表示保留, 0 表示剔除
        """
        # str, HETATM 标识; ' ' 表示标准 ATOM, 'H_xxx' 或 'W' 表示 HETATM
        het_flag = residue.id[0]
        if het_flag == ' ':
            self.n_accepted_residues += 1
            return 1
        return 0

    def accept_atom(self, atom):
        """
        对于已接受的残基, 统计原子数并全部保留.

        输入参数:
            - atom: Bio.PDB.Atom.Atom, Biopython 原子对象

        输出:
            - int, 始终返回 1 (全部保留)
        """
        self.n_accepted_atoms += 1
        return 1


# ============================================================================
# 函数 1: 单样本提取 / Single Sample Extraction
# ============================================================================

def extract_receptor_cif(input_pdb_path: str,
                         output_cif_path: str):
    """
    解析 PDB/CIF 结构文件, 剔除所有 HETATM 组分, 仅保留受体主链, 写出为 mmCIF.

    输入参数:
        - input_pdb_path: str, 输入结构文件路径 (.pdb / .cif / .mmcif)
        - output_cif_path: str, 输出 CIF 文件路径

    输出:
        - sample_id: str, 自动推断的样本名 (文件名 stem)
        - success: bool, 是否成功
        - error_msg: str 或 None, 错误信息
        - n_receptor_residues: int, 写出的受体残基数量 (失败时为 0)
        - n_receptor_atoms: int, 写出的受体原子数量 (失败时为 0)
    """
    # str, 从文件名自动推断的样本名
    sample_id = Path(input_pdb_path).stem

    # --- 1. 检查输入文件 ---
    if not Path(input_pdb_path).exists():
        return sample_id, False, f"文件不存在: {input_pdb_path}", 0, 0

    # --- 2. 选择解析器 ---
    # str, 文件后缀 (小写)
    file_ext = Path(input_pdb_path).suffix.lower()
    if file_ext in ['.pdb']:
        parser = PDBParser(QUIET=True)
    elif file_ext in ['.cif', '.mmcif']:
        parser = MMCIFParser(QUIET=True)
    else:
        return sample_id, False, f"不支持的文件格式: {file_ext}", 0, 0

    # --- 3. 解析结构 ---
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Bio.PDB.Structure.Structure, 解析后的结构对象
            structure = parser.get_structure(sample_id, input_pdb_path)
    except Exception as e:
        return sample_id, False, f"解析失败: {type(e).__name__}: {e}", 0, 0

    # --- 4. 用 MMCIFIO 写出受体 ---
    try:
        # 创建输出目录
        Path(output_cif_path).parent.mkdir(parents=True, exist_ok=True)

        # ReceptorOnlySelect, 残基级过滤器（剔除所有 HETATM）
        selector = ReceptorOnlySelect()

        io = MMCIFIO()
        io.set_structure(structure)
        io.save(output_cif_path, select=selector)   # 调包

        # int, 保留的受体残基数
        n_residues = selector.n_accepted_residues
        # int, 保留的受体原子数
        n_atoms = selector.n_accepted_atoms

        if n_residues == 0:
            return sample_id, False, "过滤后无受体残基", 0, 0

        return sample_id, True, None, n_residues, n_atoms

    except Exception as e:
        return sample_id, False, f"写出失败: {type(e).__name__}: {e}", 0, 0


# ============================================================================
# 函数 2: 批处理 / Batch Processing
# ============================================================================

def batch_extract_receptor_cif(input_dir: str,
                               output_dir: str,
                               overwrite: bool,
                               n_jobs: int):
    """
    批处理: 遍历目录中所有 PDB/CIF 文件, 调用 extract_receptor_cif, 并打印统计摘要.

    输入参数:
        - input_dir: str, 包含 PDB/CIF 文件的目录路径
        - output_dir: str, 输出 CIF 文件的目录路径
        - overwrite: bool, 是否覆盖已有的输出文件
        - n_jobs: int, 并行进程数, 建议值 1

    输出:
        - results: list[tuple], 每个元素为 (sample_id, success, error_msg, n_residues, n_atoms)
    """
    # --- 1. 扫描输入目录 ---
    # Path, 输入目录
    input_path = Path(input_dir)
    if not input_path.exists():
        raise FileNotFoundError(f"输入目录不存在: {input_dir}")

    # list[str], 扫描到的结构文件路径列表 (按文件名排序)
    extensions = ['.pdb', '.cif', '.mmcif']
    file_list = []
    for ext in extensions:
        file_list.extend([str(p) for p in input_path.glob(f"*{ext}")])
    file_list.sort()

    print("=" * 70)
    print("批处理: 从 PDB/CIF 提取受体结构")
    print("=" * 70)
    print(f"输入目录:   {input_dir}")
    print(f"输出目录:   {output_dir}")
    print(f"文件总数:   {len(file_list)}")
    print(f"并行进程:   {n_jobs}")
    print(f"覆盖模式:   {overwrite}")
    print("=" * 70)

    if len(file_list) == 0:
        print("未找到任何 PDB/CIF 文件.")
        return []

    # Path, 输出目录
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # --- 2. 构建任务列表 (检查覆盖) ---
    def _process_one(input_path: str):
        """
        处理单个文件的包装函数, 含覆盖检查.

        输入参数:
            - input_path: str, 输入文件路径

        输出:
            - tuple, (sample_id, success, error_msg, n_residues, n_atoms)
        """
        # str, 样本名
        sample_id = Path(input_path).stem
        # str, 输出路径
        out_path = str(Path(output_dir) / f"{sample_id}.cif")

        # 检查是否已存在
        if not overwrite and Path(out_path).exists():
            return sample_id, False, "已存在 (跳过)", 0, 0

        return extract_receptor_cif(input_path, out_path)

    # --- 3. 并行处理 ---
    # list[tuple], 处理结果列表
    results = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(_process_one)(fp) for fp in file_list
    )

    # --- 4. 打印统计摘要 ---
    _print_summary(results)

    return results


# ============================================================================
# 统计摘要 / Summary Statistics
# ============================================================================

def _print_summary(results: List[Tuple]):
    """
    打印批处理统计摘要, 包含成功/失败/跳过计数和受体残基/原子数量的分位数统计.

    输入参数:
        - results: list[tuple], 每个元素为 (sample_id, success, error_msg, n_residues, n_atoms)
    """
    # int, 各类别计数
    n_success = sum(1 for _, success, err, _, _ in results if success)
    n_skipped = sum(1 for _, success, err, _, _ in results if not success and err == "已存在 (跳过)")
    n_failed = len(results) - n_success - n_skipped

    print("\n" + "=" * 70)
    print("处理完成 / Processing Complete")
    print("=" * 70)
    print(f"总文件数:   {len(results)}")
    print(f"成功:       {n_success}")
    print(f"跳过:       {n_skipped}")
    print(f"失败:       {n_failed}")
    print("=" * 70)

    # --- 受体残基/原子数量统计 (仅成功样本) ---
    if n_success > 0:
        # np.ndarray, (n_success,), 成功样本的受体残基数
        residue_counts = np.array([r[3] for r in results if r[1]])
        # np.ndarray, (n_success,), 成功样本的受体原子数
        atom_counts = np.array([r[4] for r in results if r[1]])

        # list[float], 分位数节点: 0.1, 0.2, ..., 1.0
        quantile_points = [i / 10.0 for i in range(1, 11)]

        print(f"\n受体残基数量统计 (n={n_success}):")
        print(f"  平均值:   {residue_counts.mean():.1f}")
        print(f"  分位数:")
        # np.ndarray, (10,), 受体残基数的分位数
        res_quantiles = np.quantile(residue_counts, quantile_points)
        for q, v in zip(quantile_points, res_quantiles):
            print(f"    {q:.1f}:  {v:.0f}")

        print(f"\n受体原子数量统计 (n={n_success}):")
        print(f"  平均值:   {atom_counts.mean():.1f}")
        print(f"  分位数:")
        # np.ndarray, (10,), 受体原子数的分位数
        atom_quantiles = np.quantile(atom_counts, quantile_points)
        for q, v in zip(quantile_points, atom_quantiles):
            print(f"    {q:.1f}:  {v:.0f}")

    # --- 失败样本列表 ---
    if n_failed > 0:
        print(f"\n失败样本:")
        for sample_id, success, error_msg, _, _ in results:
            if not success and error_msg != "已存在 (跳过)":
                print(f"  - {sample_id}: {error_msg}")
    print("=" * 70)


# ============================================================================
# CLI 入口 / CLI Entry Point
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="从 PDB/CIF 文件批量提取受体结构 (剔除 HETATM), 输出为 mmCIF"
    )
    parser.add_argument(
        "--input_dir", type=str, required=True,
        help="包含 PDB/CIF 文件的输入目录"
    )
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="输出 CIF 文件的目录"
    )
    parser.add_argument(
        "--overwrite", action=argparse.BooleanOptionalAction, default=False,
        help="覆盖已有输出文件 (默认跳过); 用 --overwrite 开启, --no-overwrite 关闭"
    )
    parser.add_argument(
        "--n_jobs", type=int, default=1,
        help="并行进程数 (默认 1, -1 = 全部 CPU)"
    )

    args = parser.parse_args()

    batch_extract_receptor_cif(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        overwrite=args.overwrite,
        n_jobs=args.n_jobs,
    )
