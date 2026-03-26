# -*- coding: utf-8 -*-
"""
================================================================================
全流程处理脚本 - 解析 + 特征提取 + 打标签 / Full Pipeline: Parse + Features + Labels
================================================================================

支持两种工作模式 / Two working modes:
  Mode A (--mode full): 从零开始，解析 PDB/CIF → 提取特征 → 打标签
    输入: PDB/CIF 文件目录
    输出: candidates.npz + atoms.npz + residues.npz + graph.npz + labels.npz

  Mode B (--mode label_only): 仅打标签（特征已提取完毕）
    输入: 已有 .npz 文件目录（包含 candidates.npz）
    输出: labels.npz

- joblib 多核并行
- 分片处理大型数据集
- 详细错误日志
================================================================================
"""

import os
import sys
import argparse
import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple
from datetime import datetime
from joblib import Parallel, delayed
import traceback
import shutil

# 添加 Make_Data/ 到路径（使 PDB_processor 和 labels 均可导入）
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Part 1 模块（解析 + 特征）
from PDB_processor.parser import parse_structure
from PDB_processor.features.atom_features import compute_atom_features, save_atoms_npz
from PDB_processor.features.residue_features import compute_residue_features, save_residues_npz
from PDB_processor.geometry.local_frames import compute_local_frames
from PDB_processor.geometry.graph_builder import save_graph_npz
from PDB_processor.ligand_candidates import save_candidates_npz, load_candidates_npz
from PDB_processor.error_logger import (
    ProcessingError,
    return_error_info,
    return_error_and_raise,
    ErrorType,
)
from PDB_processor.config import GRAPH_CUTOFF

# Part 2 模块（筛选 + 打标签）
from labels.ligand_filter import (
    LigandFilterConfig, filter_and_classify, get_pocket_class_name_map,
)
from labels.filter_config import (
    get_filter_preset,
    list_filter_preset_names,
    get_default_filter_preset_name,
)
from labels.instance_labels import compute_binding_labels, save_labels_npz


def _normalize_filter_preset_name(value: str) -> str:
    """
    规范化命令行传入的预设名（去首尾空格并转小写）。
    """
    return value.strip().lower()


def _cleanup_sample_dir(sample_output_dir: Path, sample_id: str, reason: str) -> None:
    """
    Remove partially generated sample directory(sample_output_dir) and print a clear hint.
    """
    if sample_output_dir.exists():
        shutil.rmtree(sample_output_dir, ignore_errors=True)
        print(f"[Cleanup] Removed sample dir for {sample_id}: {sample_output_dir} ({reason})")


# ============================================================================
# 核心处理函数 / Core Processing Functions
# ============================================================================

def label_single_sample(
    sample_dir: str,
    error_dir: str,
    overwrite: bool = False,
    filter_config: LigandFilterConfig = None,
    require_ligand: bool = True,
) -> Tuple[str, bool, Optional[str]]:
    """
    Mode B: 对已有的 .npz 文件目录打标签（不重新解析）。

    输入参数 / Input:
        - sample_dir: str, 样本目录路径（包含 candidates.npz 和 atoms.npz）
        - error_dir: str, 错误日志目录
        - overwrite: bool, 是否覆盖已有 labels.npz
        - binding_threshold: float, 结合位点距离阈值 (Å)
        - filter_config: LigandFilterConfig, 筛选配置
        - require_ligand: bool, 若为 True 且无配体则跳过

    输出 / Output:
        - sample_id: str, 样本 ID
        - success: bool, 是否成功
        - error_msg: str 或 None, 错误信息
    """
    # str, 样本 ID（从目录名提取）
    sample_id = Path(sample_dir).name
    # LigandFilterConfig, 筛选配置
    if filter_config is None:
        return_error_and_raise(
            file_path=sample_dir,
            line=-1,
            error_type=ErrorType.INVALID_CONFIGURATION,
            error_detail="filter_config is None in label_single_sample().",
            output_dir=error_dir,
            sample_id=sample_id,
        )
    # str, 输出文件路径
    labels_path = str(Path(sample_dir) / "labels.npz")
    if not overwrite and os.path.exists(labels_path):
        return sample_id, False, "Already exists"
    # str, candidates.npz 路径
    candidates_path = str(Path(sample_dir) / "candidates.npz")
    if not os.path.exists(candidates_path):
        return sample_id, False, "candidates.npz not found"

    try:
        # ===== 1. 加载候选配体 / Load candidates =====
        # list[LigandCandidate], 候选配体列表
        # int, 水分子数量
        candidates, water_count = load_candidates_npz(candidates_path)
        if require_ligand and len(candidates) == 0:
            return sample_id, False, "No candidates in candidates.npz"


        # ===== 2. 加载原子坐标（用于计算距离）/ Load atom coords =====
        # 需要从 atoms.npz 加载原子坐标，构造轻量 ParsedStructure
        atoms_path = str(Path(sample_dir) / "atoms.npz")
        if not os.path.exists(atoms_path):
            return sample_id, False, "atoms.npz not found"

        # 使用 _MinimalParsedData 代替完整 ParsedStructure（避免重新解析）
        atoms_data = np.load(atoms_path, allow_pickle=True)
        # np.ndarray, (N_atoms, 3), float32, 原子坐标
        atom_coords = atoms_data['coords']
        # 构造最小化 ParsedStructure（只需 atom_coords）
        from PDB_processor.parser import ParsedStructure
        # 创建一个只含 atom_coords 的最小 ParsedStructure
        parsed_data = ParsedStructure.__new__(ParsedStructure)
        parsed_data.atom_coords = atom_coords


        # ===== 3. 筛选 + 分类 / Filter and classify =====
        # list[LigandCandidate], 通过筛选的候选
        # dict[int, tuple[int, str]], candidate_id → (class_id, class_name)
        # list[tuple[int, str]], 被排除的候选及原因
        selected, pocket_class_map, excluded = filter_and_classify(candidates, filter_config)
        if require_ligand and len(selected) == 0:
            return_error_info(
                file_path=sample_dir,
                line=-1,
                error_type=ErrorType.NO_LIGAND,
                error_detail=(
                    f"No ligand after filtering ({len(candidates)} candidates, all excluded)"
                ),
                output_dir=error_dir,
                sample_id=sample_id,
            )
            return sample_id, False, (
                f"No ligand after filtering ({len(candidates)} candidates, all excluded)"
            )
        # dict[int, str], 口袋类别名称映射
        pocket_class_names = get_pocket_class_name_map(filter_config)


        # ===== 4. 计算标签 / Compute labels =====
        binding_labels = compute_binding_labels(
            parsed_data,
            selected_candidates=selected,
            pocket_class_map=pocket_class_map,
            error_dir=error_dir,
            sample_id=sample_id,
            require_binding_site=require_ligand,
        )
        if binding_labels is None:
            return sample_id, False, "No binding site"


        # ===== 5. 保存标签 / Save labels =====
        save_labels_npz(
            parsed_data,
            binding_labels,
            selected_candidates=selected,
            pocket_class_names=pocket_class_names,
            output_path=labels_path,
        )

        return sample_id, True, None

    except Exception as e:
        error_msg = f"{type(e).__name__}: {str(e)}"
        if isinstance(e, ProcessingError) and e.logged:
            return sample_id, False, error_msg
        return_error_info(
            sample_dir, -1, ErrorType.PARSE_ERROR,
            f"Unexpected error: {error_msg}\n{traceback.format_exc()}",
            error_dir, sample_id
        )
        return sample_id, False, error_msg


def process_and_label_single_file(
    input_path: str,
    output_dir: str,
    error_dir: str,
    overwrite: bool = False,

    graph_cutoff: float = GRAPH_CUTOFF,
    require_ligand: bool = True,
    compute_density: bool = True,
    select_first_model: bool = False,
    filter_config: LigandFilterConfig = None,
) -> Tuple[str, bool, Optional[str]]:
    """
    Mode A: 从零开始处理单个结构文件（解析 + 特征提取 + 打标签）。
    Mode A: Full pipeline for a single structure file (parse + features + labels).

    输入参数 / Input:
        - input_path: str, 输入 PDB/CIF 文件路径
        - output_dir: str, 输出根目录
        - error_dir: str, 错误日志目录

        - overwrite: bool, 若为 True 则覆盖已有的同名文件
        - graph_cutoff: float, 图边距离截断 (Å)
        - require_ligand: bool, 是否要求配体存在（无配体则跳过）
        - compute_density: bool, 是否计算原子局部密度特征
        - select_first_model: bool, 多 model 时是否仅取第一个
        - filter_config: LigandFilterConfig 或 None, 筛选配置

    输出 / Output:
        - sample_id: str, 样本 ID
        - success: bool, 是否成功
        - error_msg: str 或 None, 错误信息
    """
    # str, 样本 ID（从文件名提取）
    sample_id = Path(input_path).stem
    # Path, 样本输出目录（用于失败清理）
    sample_output_dir = Path(output_dir) / sample_id
    # LigandFilterConfig, 筛选配置
    if filter_config is None:
        return_error_and_raise(
            file_path=input_path,
            line=-1,
            error_type=ErrorType.INVALID_CONFIGURATION,
            error_detail="filter_config is None in process_and_label_single_file().",
            output_dir=error_dir,
            sample_id=sample_id,
        )

    try:
        # ===== 1. 解析结构 / Parse structure =====
        parsed_data = parse_structure(
            input_path,
            error_dir,
            sample_id,
            require_ligand=require_ligand,
            select_first_model=select_first_model
        )
        if parsed_data is None:
            return sample_id, False, "Parse failed"
        elif parsed_data == ">1 model":
            return sample_id, False, ">1 model"


        # ===== 2. 创建输出目录 / Create output directory =====
        if not overwrite and os.path.exists(sample_output_dir):
            return sample_id, False, "Already exists"
        sample_output_dir.mkdir(parents=True, exist_ok=True)

        # ===== 3. 保存候选配体属性 / Save candidate attributes =====
        save_candidates_npz(
            parsed_data.ligand_candidates,
            parsed_data.water_count,
            str(sample_output_dir / "candidates.npz")
        )



        # ===== 4. 计算原子特征 / Compute atom features =====
        atom_features = compute_atom_features(parsed_data, compute_density=compute_density)
        save_atoms_npz(parsed_data, atom_features, str(sample_output_dir / "atoms.npz"))
        # ===== 5. 计算残基特征和局部坐标系 / Compute residue features and local frames =====
        residue_features = compute_residue_features(parsed_data)
        local_frames, frames_mask = compute_local_frames(
            parsed_data,
            error_dir=error_dir,
            sample_id=sample_id,
            file_path=input_path,
        )
        save_residues_npz(
            parsed_data, residue_features, local_frames, frames_mask,
            str(sample_output_dir / "residues.npz")
        )
        # ===== 6. 保存图结构 / Save graph structure =====
        save_graph_npz(parsed_data, graph_cutoff, str(sample_output_dir / "graph.npz"))



        # ===== 7.筛选配体 + 分配口袋类别 / Filter candidates + assign pocket classes =====
        # list[LigandCandidate], 通过筛选的候选
        # dict[int, tuple[int, str]], candidate_id → (class_id, class_name)
        # list[tuple[int, str]], 被排除的候选及原因
        selected, pocket_class_map, excluded = filter_and_classify(
            parsed_data.ligand_candidates, filter_config
        )
        if require_ligand and len(selected) == 0:
            return_error_info(
                file_path=input_path,
                line=-1,
                error_type=ErrorType.NO_LIGAND,
                error_detail=(
                    f"No ligand after filtering ({len(parsed_data.ligand_candidates)} candidates, all excluded)"
                ),
                output_dir=error_dir,
                sample_id=sample_id,
            )
            _cleanup_sample_dir(
                sample_output_dir,
                sample_id,
                "No ligand remained after filtering",
            )
            return sample_id, False, (
                f"No ligand after filtering ({len(parsed_data.ligand_candidates)} candidates, all excluded)"
            )
        # dict[int, str], 口袋类别名称映射
        pocket_class_names = get_pocket_class_name_map(filter_config)



        # ===== 8. 计算并保存标签 / Compute and save labels =====
        binding_labels = compute_binding_labels(
            parsed_data,
            selected_candidates=selected,
            pocket_class_map=pocket_class_map,
            error_dir=error_dir,
            sample_id=sample_id,
            require_binding_site=require_ligand,
        )
        if binding_labels is None:
            _cleanup_sample_dir(
                sample_output_dir,
                sample_id,
                "No binding site after ligand filtering",
            )
            return sample_id, False, "No binding site"
        save_labels_npz(
            parsed_data,
            binding_labels,
            selected_candidates=selected,
            pocket_class_names=pocket_class_names,
            output_path=str(sample_output_dir / "labels.npz"),
        )

        return sample_id, True, None

    except Exception as e:
        error_msg = f"{type(e).__name__}: {str(e)}"
        _cleanup_sample_dir(sample_output_dir, sample_id, error_msg)
        if isinstance(e, ProcessingError) and e.logged:
            return sample_id, False, error_msg
        return_error_info(
            input_path, -1, ErrorType.PARSE_ERROR,
            f"Unexpected error: {error_msg}\n{traceback.format_exc()}",
            error_dir, sample_id
        )
        return sample_id, False, error_msg


# ============================================================================
# 工具函数 / Utility Functions
# ============================================================================

def get_pdb_file_list(
    input_dir: str,
    extensions: List[str] = ['.pdb', '.cif', '.mmcif'],
    error_dir: Optional[str] = None,
) -> List[str]:
    """
    获取目录下所有结构文件 / Get all structure files in directory.

    输入参数 / Input:
        - input_dir: str, 输入目录
        - extensions: list[str], 文件扩展名列表

    输出 / Output:
        - list[str], 文件路径列表（已排序）
    """
    # Path, 输入目录路径
    input_path = Path(input_dir)
    if not input_path.exists():
        detail = f"Input directory does not exist: {input_dir}"
        if error_dir is not None:
            return_error_and_raise(
                file_path=input_dir,
                line=-1,
                error_type=ErrorType.INVALID_CONFIGURATION,
                error_detail=detail,
                output_dir=error_dir,
                sample_id="global",
            )
        raise ProcessingError(ErrorType.INVALID_CONFIGURATION, detail)
    # list[str], 文件列表
    file_list = []
    for ext in extensions:
        file_list.extend([str(p) for p in input_path.glob(f"*{ext}")])
    file_list.sort()
    return file_list



def get_sample_dir_list(
    processed_dir: str,
    error_dir: Optional[str] = None,
) -> List[str]:
    """
    获取已处理目录下所有样本子目录（包含 candidates.npz）。

    输入参数 / Input:
        - processed_dir: str, 已处理数据根目录(含有一系列 以pdb_id命名的子文件夹, 内部有关于这个样本的一系列.npz)

    输出 / Output:
        - list[str], 样本目录路径列表（已排序）
    """
    processed_path = Path(processed_dir)
    if not processed_path.exists():
        detail = f"Processed directory does not exist: {processed_dir}"
        if error_dir is not None:
            return_error_and_raise(
                file_path=processed_dir,
                line=-1,
                error_type=ErrorType.INVALID_CONFIGURATION,
                error_detail=detail,
                output_dir=error_dir,
                sample_id="global",
            )
        raise ProcessingError(ErrorType.INVALID_CONFIGURATION, detail)
    sample_dirs = [
        str(d) for d in processed_path.iterdir()
        if d.is_dir() and (d / "candidates.npz").exists()
    ]
    sample_dirs.sort()
    return sample_dirs


def apply_sharding(
    item_list: List[str],
    part_id: int,
    total_parts: int,
    error_dir: Optional[str] = None,
) -> List[str]:
    """
    应用分片策略 / Apply sharding strategy.

    输入参数 / Input:
        - item_list: list[str], 完整列表
        - part_id: int, 当前分片 ID (0-indexed)
        - total_parts: int, 总分片数

    输出 / Output:
        - list[str], 当前分片的列表
    """
    if total_parts <= 1:
        return item_list
    if part_id < 0 or part_id >= total_parts:
        detail = f"Invalid part_id {part_id} for {total_parts} parts"
        if error_dir is not None:
            return_error_and_raise(
                file_path="",
                line=-1,
                error_type=ErrorType.INVALID_CONFIGURATION,
                error_detail=detail,
                output_dir=error_dir,
                sample_id="global",
            )
        raise ProcessingError(ErrorType.INVALID_CONFIGURATION, detail)
    # 计算每个分片的起止索引
    n = len(item_list)
    shard_size = n // total_parts
    remainder = n % total_parts
    # 前 remainder 个分片多分一个
    if part_id < remainder:
        start = part_id * (shard_size + 1)
        end = start + shard_size + 1
    else:
        start = remainder * (shard_size + 1) + (part_id - remainder) * shard_size
        end = start + shard_size
    return item_list[start:end]


def print_summary(results: List[Tuple], elapsed: float, error_dir: Optional[str] = None) -> None:
    """打印处理结果摘要 / Print processing summary."""
    n_success = sum(1 for _, success, _ in results if success)
    n_failed = len(results) - n_success
    print("=" * 70)
    print("Processing Complete / 处理完成")
    print("=" * 70)
    print(f"Total:    {len(results)}")
    print(f"Success:  {n_success}")
    print(f"Failed:   {n_failed}")
    if len(results) > 0:
        print(f"Time:     {elapsed:.2f}s ({elapsed / len(results):.3f}s/item)")
    print("=" * 70)
    if n_failed > 0:
        print("Failed samples:")
        for sample_id, success, error_msg in results:
            if not success and error_msg != "Already exists":
                print(f"  - {sample_id}: {error_msg}")
        if error_dir is not None:
            print(f"\nError logs saved to: {Path(error_dir) / 'error_logs'}")




# ============================================================================
# 主函数 / Main Function
# ============================================================================

def main():
    """主函数 / Main function."""
    parser = argparse.ArgumentParser(
        description="Full Pipeline: Parse + Features + Labels / 全流程：解析 + 特征 + 打标签"
    )
    # list[str], 从 labels/filter_config.py 自动发现的全部预设名
    available_filter_presets = list_filter_preset_names()
    if len(available_filter_presets) == 0:
        raise ValueError(
            "未检测到任何配体筛选预设，请先在 labels/filter_config.py 中定义 *_PRESET。"
        )
    # str, 默认预设名（优先 binary）
    default_filter_preset = get_default_filter_preset_name()

    # ========================================== 工作模式 ==========================================
    parser.add_argument(
        "--mode", type=str, default='full',
        choices=['full', 'label_only'],
        help=(
            "工作模式: "
            "full=从零开始（解析+特征+打标签）; "
            "label_only=仅打标签（需已有 candidates.npz）"
        )
    )

    # ========================================== 输入/输出/报错 ==========================================
    parser.add_argument(
        "--input_dir", type=str, required=True,
        help=(
            "Mode full的时候: PDB/CIF 文件目录; "
            "Mode label_only的时候: 已处理的样本根目录（含各样本子目录）"
        )
    )
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="输出根目录（Mode full 时使用; label_only 时 labels.npz 写入各样本子目录）"
    )
    parser.add_argument(
        "--error_dir", type=str, required=True,
        help="错误日志目录"
    )
    parser.add_argument(
        "--overwrite", action=argparse.BooleanOptionalAction, default=False,
        help="覆盖已有输出文件 (默认跳过); 用 --overwrite 开启, --no-overwrite 关闭"
    )



    # ========================================== 分片 ==========================================
    parser.add_argument("--part_id", type=int, default=0,
                        help="当前分片 ID (0-indexed)")
    parser.add_argument("--total_parts", type=int, default=1,
                        help="总分片数")
    # 并行
    parser.add_argument("--n_jobs", type=int, default=1,
                        help="并行进程数 (-1 = 全部 CPU)")



    # ========================================== 处理参数 ==========================================
    parser.add_argument("--graph_cutoff", type=float, default=GRAPH_CUTOFF,
                        help=f"图边距离截断 (默认 {GRAPH_CUTOFF}Å)")
    parser.add_argument("--no_require_ligand", action="store_true",
                        help="不要求配体存在（处理所有文件）")
    parser.add_argument("--no_compute_density", action="store_true",
                        help="跳过密度特征计算（更快）")
    parser.add_argument("--select_first_model", action="store_true",
                        help="多 model 时仅取第一个")
    parser.add_argument(
        "--filter_preset", "--class_name",
        dest="filter_preset",
        type=_normalize_filter_preset_name,
        default=default_filter_preset,
        choices=available_filter_presets,
        help=(
            "筛选预设名（自动从 labels/filter_config.py 读取）；"
            "可在 sbatch 用 CLASS_NAME 变量传入。"
        )
    )

    args = parser.parse_args()

    # =========================================================================
    # 初始化 / Initialize
    # =========================================================================
    print("=" * 70)
    print("Full Pipeline / 全流程处理脚本")
    print("=" * 70)
    print(f"Mode:   {args.mode}")
    print(f"Input:  {args.input_dir}")
    print(f"Output: {args.output_dir}")
    print(f"Shard:  {args.part_id + 1}/{args.total_parts}")
    print(f"Jobs:   {args.n_jobs}")

    # LigandFilterConfig, 按预设名读取筛选配置
    filter_config = get_filter_preset(args.filter_preset)
    if filter_config is None:
        raise ValueError(
            f"未知筛选预设: {args.filter_preset}；可选值: {available_filter_presets}"
        )
    print(f"Filter: {args.filter_preset}")
    print("=" * 70)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.error_dir).mkdir(parents=True, exist_ok=True)

    # =========================================================================
    # 获取任务列表 / Get task list
    # =========================================================================
    if args.mode == 'full':
        # Mode A: 从 PDB/CIF 文件目录获取文件列表
        all_items = get_pdb_file_list(args.input_dir, error_dir=args.error_dir)
        print(f"Total PDB/CIF files found: {len(all_items)}")
    else:
        # Mode B: 从已处理目录获取样本子目录列表
        all_items = get_sample_dir_list(args.input_dir, error_dir=args.error_dir)
        print(f"Total sample directories found: {len(all_items)}")

    # 应用分片
    shard_items = apply_sharding(
        all_items,
        args.part_id,
        args.total_parts,
        error_dir=args.error_dir,
    )
    print(f"Items in this shard: {len(shard_items)}")

    if len(shard_items) == 0:
        print("No items to process in this shard.")
        return

    # =========================================================================
    # 并行处理 / Parallel processing
    # =========================================================================
    start_time = datetime.now()

    if args.mode == 'full':
        results = Parallel(n_jobs=args.n_jobs, verbose=10)(
            delayed(process_and_label_single_file)(
                file_path,
                args.output_dir,
                args.error_dir,
                overwrite=args.overwrite,
                graph_cutoff=args.graph_cutoff,
                require_ligand=not args.no_require_ligand,
                compute_density=not args.no_compute_density,
                select_first_model=args.select_first_model,
                filter_config=filter_config,
            )
            for file_path in shard_items
        )
    else:
        results = Parallel(n_jobs=args.n_jobs, verbose=10)(
            delayed(label_single_sample)(
                sample_dir,
                args.error_dir,
                overwrite=args.overwrite,
                filter_config=filter_config,
                require_ligand=not args.no_require_ligand,
            )
            for sample_dir in shard_items
        )

    elapsed = (datetime.now() - start_time).total_seconds()
    print_summary(results, elapsed, error_dir=args.error_dir)


if __name__ == "__main__":
    main()




# see me
"""
================================================================================
Output Data Structure / 输出数据结构
================================================================================

Detailed description of .npz files generated by this script.
本脚本生成的 .npz 文件详细说明。

1. atoms.npz (Atom-level Features / 原子级特征)
--------------------------------------------------------------------------------
coords                                              # np.ndarray, (N_atoms, 3), float32, 原子的笛卡尔坐标
features                                            # np.ndarray, (N_atoms, 49), float32, 原子级特征向量 (元素+残基类型+理化性质+质量+密度)
elements                                            # np.ndarray, (N_atoms,), str, 元素符号 (如 'C', 'N')
res_indices                                         # np.ndarray, (N_atoms,), int32, 原子所属的残基全局索引 (0 ~ N_res-1), 残基索引的先后顺序就是下面 residues.npz 的索引顺序
chain_indices                                       # np.ndarray, (N_atoms,), int32, 每个原子所属的链索引 (0 ~ N_chains-1)
res_names                                           # np.ndarray, (N_atoms,), str, 残基三字母代码 (如 'ALA', 'G')
atom_names                                          # np.ndarray, (N_atoms,), str, 原子名称 (如 'CA', 'CB')



2. residues.npz (Residue-level Features / 残基级特征)
--------------------------------------------------------------------------------
coords                                              # np.ndarray, (N_res, 3), float32, 代表原子坐标 (蛋白质为 CA，核苷酸为 C4')
features                                            # np.ndarray, (N_res, 33), float32, 残基级特征向量 (类型 One-Hot + 理化性质)
names                                               # np.ndarray, (N_res,), str, 残基名称 (如 'ALA')
types                                               # np.ndarray, (N_res,), str, 残基类型：'protein' (蛋白质) 或 'nucleotide' (核苷酸)
chain_indices                                       # np.ndarray, (N_res,), int32, 每个残基的链索引 (0 ~ N_chains-1)
seq_numbers                                         # np.ndarray, (N_res,), int32, PDB 文件中的残基序列号(未必从0开始, 用来衡量同一条链上残基的1D距离)
local_frames                                        # np.ndarray, (N_res, 3, 3), float32, 局部坐标系旋转矩阵 (列向量为 X、Y、Z 轴)
frames_mask                                         # np.ndarray, (N_res,), bool, 局部坐标系有效性掩码 (如果骨架原子存在则为 True)
backbone_complete                                   # np.ndarray, (N_res,), bool, 所有骨架原子是否都存在



3. graph.npz (Graph Structure / 图结构)
--------------------------------------------------------------------------------
edge_row                                            # np.ndarray, (N_edges,), int32, 源节点索引 (0 到 N_atoms-1)
edge_col                                            # np.ndarray, (N_edges,), int32, 目标节点索引 (0 到 N_atoms-1)
edge_dist                                           # np.ndarray, (N_edges,), float32, 连接节点之间的欧几里得距离
edge_weight                                         # np.ndarray, (N_edges,), float32, 边权重 (默认为 1 / (distance + epsilon))
num_atoms                                           # int, 原子总数
num_residues                                        # int, 残基总数
cutoff                                              # float, 用于构建图的距离截断值 (如 10.0)



4. candidates.npz (候选配体属性 / Candidate Ligand Attributes) [Part 1 产生]
--------------------------------------------------------------------------------
n_candidates                                        # int, 候选配体数量 (不含水)
water_count                                         # int, 被排除的水分子数
resnames                                            # np.ndarray, (N_cand,), str,   CCD 残基名 (按 candidate_id 升序, 与下方各字段一一对应)
chain_ids                                           # np.ndarray, (N_cand,), str,   链 ID
res_ids                                             # np.ndarray, (N_cand,), int32, 残基序号
n_heavy_atoms                                       # np.ndarray, (N_cand,), int32, 重原子数
is_metal_ion                                        # np.ndarray, (N_cand,), bool,  金属离子标志
is_peptide_like                                     # np.ndarray, (N_cand,), bool,  标准 AA 类 HETATM
is_nucleotide_like                                  # np.ndarray, (N_cand,), bool,  标准核苷酸类 HETATM
is_covalent                                         # np.ndarray, (N_cand,), bool,  共价连接标志
polymer_length                                      # np.ndarray, (N_cand,), int32, 聚合物链长
centers                                             # np.ndarray, (N_cand, 3), float32, 候选配体重心
candidate_coords_{i}                                # np.ndarray, (M_i, 3), float32, 第 i 个候选配体的原子坐标 (i = candidate_id, 例如 candidate_coords_0)



5. labels.npz (Labels & Ground Truth / 标签与真值) [Part 2 产生; 经 filter_and_classify 筛选后只保留通过筛选的配体]
--------------------------------------------------------------------------------
# ---- 逐原子字段 (N_atoms = 蛋白质/核酸原子总数, 不含配体原子) ----
instance_ids                                        # np.ndarray, (N_atoms,), int32,  每个原子的结合位点实例 ID (实例ID = candidate_id; 背景为 -1;)
ligand_ids                                          # np.ndarray, (N_atoms,), int32,  每个原子最近配体的 candidate_id (无论距离远近; 背景原子也有值)
distances                                           # np.ndarray, (N_atoms,), float32, 每个原子到最近配体原子的距离 (Å)
binding_mask                                        # np.ndarray, (N_atoms,), bool,   距最近配体 ≤ binding_threshold (默认 4.5Å) 则为 True
pocket_class_ids                                    # np.ndarray, (N_atoms,), int32,  口袋类别 ID (0=非口袋/背景, 1=可成药, 2=金属离子, etc.)

# ---- 逐配体字段 (N_ligands = 通过筛选的配体数量; 各字段均按 candidate_id 升序排列, 彼此逐行对应) ----
num_ligands                                         # int,                            通过筛选的配体数量
pocket_centers                                      # np.ndarray, (N_ligands, 3), float32, 结合口袋的几何中心 (由该配体周围结合蛋白原子计算)
ligand_resnames                                     # np.ndarray, (N_ligands,), str,  各配体的残基名称 (CCD code)
ligand_candidate_ids                                # np.ndarray, (N_ligands,), int32, 各配体的原始 candidate_id, 如 pocket_centers[i] 对应的配体 candidate_id 就是 ligand_candidate_ids[i]
ligand_coords_{id}                                  # np.ndarray, (M_id, 3), float32, candidate_id 为 {id} 的配体的全部重原子坐标 (id = candidate_id)
pocket_class_name_map                               # np.ndarray, str (scalar),       类别 ID→名称映射字符串 (格式: "0:background,1:druggable,...")
"""




# NOTE: 这是在推断是用的
def get_features_when_infer(
    input_path: str,
    error_dir: str = None,
    graph_cutoff: float = GRAPH_CUTOFF,
    compute_density: bool = True, 
    select_first_model: bool = False
) -> Optional[Tuple[dict, dict, dict]]:
    """
    推断时提取单个结构文件的三类特征，不保存任何文件、不计算标签。

    输入参数 / Input:
        - input_path: str, 输入的 PDB/CIF 文件路径
        - error_dir: str, 错误日志存放目录
        - graph_cutoff: float, 图边距离截断 (Å)
        - compute_density: bool, 是否计算原子局部密度特征
        - select_first_model: bool, 多模型时是否仅取第一个模型

    输出 / Output:
        成功时返回 (atoms_dict, residues_dict, graph_dict)，失败返回 None。

        - atoms_dict 的键 / keys of atoms_dict:
            - coords          - np.ndarray, (N_atoms, 3),       float32, 原子坐标
            - features        - np.ndarray, (N_atoms, 49),      float32, 原子特征向量
            - elements        - np.ndarray, (N_atoms,),          str,    元素符号
            - res_indices     - np.ndarray, (N_atoms,),          int32,  所属残基索引
            - chain_indices   - np.ndarray, (N_atoms,),          int32,  所属链索引
            - res_names       - np.ndarray, (N_atoms,),          str,    残基名称
            - atom_names      - np.ndarray, (N_atoms,),          str,    原子名称

        - residues_dict 的键 / keys of residues_dict:
            - coords            - np.ndarray, (N_res, 3),        float32, 残基代表原子坐标
            - features          - np.ndarray, (N_res, 33),       float32, 残基特征向量
            - names             - np.ndarray, (N_res,),           str,    残基名称
            - types             - np.ndarray, (N_res,),           str,    残基类型 ('protein' / 'nucleotide')
            - chain_indices     - np.ndarray, (N_res,),           int32,  链索引
            - seq_numbers       - np.ndarray, (N_res,),           int32,  PDB 序列号
            - local_frames      - np.ndarray, (N_res, 3, 3),     float32, 局部坐标系旋转矩阵
            - frames_mask       - np.ndarray, (N_res,),           bool,   坐标系有效性掩码
            - backbone_complete - np.ndarray, (N_res,),           bool,   骨架完整性标记

        - graph_dict 的键 / keys of graph_dict:
            - edge_row      - np.ndarray, (N_edges,), int32,   源节点索引
            - edge_col      - np.ndarray, (N_edges,), int32,   目标节点索引
            - edge_dist     - np.ndarray, (N_edges,), float32, 边距离
            - edge_weight   - np.ndarray, (N_edges,), float32, 边权重 (1 / (dist + ε))
            - num_atoms     - int,   原子总数
            - num_residues  - int,   残基总数
            - cutoff        - float, 距离截断值
    """
    # ------------------------------------------------------------------
    # 导入图边构建函数 (仅在此函数中需要)
    # ------------------------------------------------------------------
    from PDB_processor.geometry.graph_builder import build_graph_edges_sparse
    # str, 样本 ID，从文件名提取 / sample ID derived from filename
    sample_id = Path(input_path).stem

    try:
        # 1. 解析结构 / Parse structure
        parsed_data = parse_structure(
            input_path,
            error_dir,
            sample_id,
            require_ligand=False,  #    推断时 require_ligand=False，不要求配体存在
            select_first_model=select_first_model,
        )
        if parsed_data is None:
            # 解析失败，错误信息已由 parse_structure 内部写入 error_dir
            print(f"[get_features_when_infer] Parse failed for {sample_id}")
            return None

        # 2. 原子级特征 / Atom-level features
        # np.ndarray, (N_atoms, 49), float32, 原子特征矩阵
        atom_feat = compute_atom_features(
            parsed_data,
            compute_density=compute_density,
        )

        # dict, 与 atoms.npz 内容完全对齐
        atoms_dict = {
            "coords":        parsed_data.atom_coords,                           # np.ndarray, (N_atoms, 3), float32
            "features":      atom_feat,                                         # np.ndarray, (N_atoms, 49), float32
            "elements":      np.array(parsed_data.atom_elements, dtype=object), # np.ndarray, (N_atoms,), str
            "res_indices":   parsed_data.atom_res_indices,                      # np.ndarray, (N_atoms,), int32
            "chain_indices": parsed_data.atom_chain_indices,                    # np.ndarray, (N_atoms,), int32
            "res_names":     np.array(parsed_data.atom_res_names, dtype=object),# np.ndarray, (N_atoms,), str
            "atom_names":    np.array(parsed_data.atom_names, dtype=object),    # np.ndarray, (N_atoms,), str
        }

        # 3. 残基级特征 + 局部坐标系 / Residue-level features + local frames
        # np.ndarray, (N_res, 33), float32, 残基特征矩阵
        res_feat = compute_residue_features(parsed_data)
        # np.ndarray, (N_res, 3, 3), float32, 局部坐标系旋转矩阵
        # np.ndarray, (N_res,), bool, 坐标系有效性掩码
        local_frames, frames_mask = compute_local_frames(
            parsed_data,
            error_dir=error_dir,
            sample_id=sample_id,
            file_path=input_path,
        )

        # dict, 与 residues.npz 内容完全对齐
        residues_dict = {
            "coords":            parsed_data.res_coords,                             # np.ndarray, (N_res, 3), float32
            "features":          res_feat,                                           # np.ndarray, (N_res, 33), float32
            "names":             np.array(parsed_data.res_names, dtype=object),       # np.ndarray, (N_res,), str
            "types":             np.array(parsed_data.res_types, dtype=object),       # np.ndarray, (N_res,), str
            "chain_indices":     parsed_data.res_chain_indices,                       # np.ndarray, (N_res,), int32
            "seq_numbers":       parsed_data.res_seq_numbers,                         # np.ndarray, (N_res,), int32
            "local_frames":      local_frames,                                        # np.ndarray, (N_res, 3, 3), float32
            "frames_mask":       frames_mask,                                         # np.ndarray, (N_res,), bool
            "backbone_complete": parsed_data.backbone_complete_mask,                   # np.ndarray, (N_res,), bool
        }

        # ==============================================================
        # 4. 图结构 / Graph structure
        # ==============================================================
        # np.ndarray, (N_edges,), int32, 源节点索引
        # np.ndarray, (N_edges,), int32, 目标节点索引
        # np.ndarray, (N_edges,), float32, 边距离
        row_idx, col_idx, distances = build_graph_edges_sparse(
            parsed_data.atom_coords, graph_cutoff
        )
        # np.ndarray, (N_edges,), float32, 边权重 = 1 / (距离 + epsilon)
        weights = 1.0 / (distances + 1e-6)

        # dict, 与 graph.npz 内容完全对齐
        graph_dict = {
            "edge_row":    row_idx.astype(np.int32),      # np.ndarray, (N_edges,), int32
            "edge_col":    col_idx.astype(np.int32),      # np.ndarray, (N_edges,), int32
            "edge_dist":   distances.astype(np.float32),  # np.ndarray, (N_edges,), float32
            "edge_weight": weights.astype(np.float32),    # np.ndarray, (N_edges,), float32
            "num_atoms":   len(parsed_data.atom_coords),  # int, 原子总数
            "num_residues":len(parsed_data.res_names),    # int, 残基总数
            "cutoff":      graph_cutoff,                  # float, 距离截断值
        }

        return atoms_dict, residues_dict, graph_dict

    except Exception as e:
        # 捕获未预期的异常并打印 / Catch unexpected exceptions
        error_msg = f"{type(e).__name__}: {str(e)}"
        print(f"[get_features_when_infer] Error for {sample_id}: {error_msg}")
        traceback.print_exc()
        if isinstance(e, ProcessingError) and e.logged:
            return None
        if error_dir is not None:
            return_error_info(
                file_path=input_path,
                line=-1,
                error_type=ErrorType.PARSE_ERROR,
                error_detail=f"Failed in get_features_when_infer: {error_msg}",
                output_dir=error_dir,
                sample_id=sample_id,
            )
        return None
