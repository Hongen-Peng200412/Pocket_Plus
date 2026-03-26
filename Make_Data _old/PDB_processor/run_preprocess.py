"""
================================================================================
统一预处理系统 - 主运行脚本 / Unified Preprocessing System - Main Runner
================================================================================

主入口点，处理整个数据集:
- 支持 PDB 和 mmCIF 格式
- joblib 多核并行
- 分片处理大型数据集
- 详细错误日志

Main entry point for processing entire datasets.
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

# 添加父目录到路径 / Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PDB_processor.parser import parse_structure
from PDB_processor.features.atom_features import compute_atom_features, save_atoms_npz
from PDB_processor.features.residue_features import compute_residue_features, save_residues_npz
from PDB_processor.geometry.local_frames import compute_local_frames
from PDB_processor.geometry.graph_builder import save_graph_npz
from PDB_processor.labels.instance_labels import compute_binding_labels, save_labels_npz
from PDB_processor.error_logger import return_error_info, ErrorType
from PDB_processor.config import GRAPH_CUTOFF, BINDING_THRESHOLD


def process_single_file(
    input_path: str,
    output_dir: str,
    error_dir: str, 
    overwrite: bool = False,

    binding_threshold: float = BINDING_THRESHOLD,
    graph_cutoff: float = GRAPH_CUTOFF,
    require_ligand: bool = True,
    compute_density: bool = True, 
    select_first_model: bool = False
) -> Tuple[str, bool, Optional[str]]:
    """
    处理单个结构文件
    Process a single structure file
    
    输入参数 / Input:
        - input_path: str, 输入文件路径
        - output_dir: str, 输出根目录
        - error_dir: str, 错误日志目录
        - overwrite: boll, 若为True则覆盖掉已有的同名文件(不然直接跳过)

        - binding_threshold: float, 结合位点距离阈值
        - graph_cutoff: float, 图边距离截断
        - require_ligand: bool, 是否要求配体存在
        - compute_density: bool, 是否计算密度特征
    
    输出 / Output:
        - sample_id: str, 样本ID
        - success: bool, 是否成功
        - error_msg: str 或 None, 错误信息
    """
    # str, 样本ID (从文件名提取)
    sample_id = Path(input_path).stem
    try:
        # 1. 解析结构 / Parse structure
        parsed_data = parse_structure(
            input_path,
            error_dir,
            sample_id,
            require_ligand=require_ligand,
            select_first_model=select_first_model
        )
        if parsed_data is None:
            # 解析失败 (错误已记录)
            return sample_id, False, "Parse failed"
        elif parsed_data == ">1 model":
            return sample_id, False, ">1 model"
        

        # 2. 创建输出目录 / Create output directory
        # Path, 样本输出目录
        sample_output_dir = Path(output_dir) / sample_id
        if not overwrite and os.path.exists(sample_output_dir):
            return sample_id, False, "Already exists"
        sample_output_dir.mkdir(parents=True, exist_ok=True)
        

        # 3. 计算原子特征 / Compute atom features
        atom_features = compute_atom_features(
            parsed_data,
            compute_density=compute_density
        )
        # 保存 atoms.npz
        save_atoms_npz(
            parsed_data,
            atom_features,
            str(sample_output_dir / "atoms.npz")
        )
        

        # 4. 计算残基特征和局部坐标系 / Compute residue features and local frames
        residue_features = compute_residue_features(parsed_data)
        local_frames, frames_mask = compute_local_frames(parsed_data)
        # 保存 residues.npz
        save_residues_npz(
            parsed_data,
            residue_features,
            local_frames,
            frames_mask,
            str(sample_output_dir / "residues.npz")
        )
        
        
        # 5. 保存图结构 / Save graph structure
        save_graph_npz(
            parsed_data,
            graph_cutoff,
            str(sample_output_dir / "graph.npz")
        )
        
        
        # 6. 计算并保存标签 / Compute and save labels
        binding_labels = compute_binding_labels(
            parsed_data,
            binding_threshold=binding_threshold,
            output_dir=error_dir,
            sample_id=sample_id,
            require_binding_site=require_ligand
        )
        if binding_labels is None:
            # 无结合位点 (错误已记录)
            # 删除已创建的文件
            import shutil
            shutil.rmtree(sample_output_dir)
            return sample_id, False, "No binding site"
        save_labels_npz(
            parsed_data,
            binding_labels,
            str(sample_output_dir / "labels.npz")
        )
        
        return sample_id, True, None
        
    except Exception as e:
        # 捕获未预期的异常
        error_msg = f"{type(e).__name__}: {str(e)}"
        return_error_info(
            input_path, -1, ErrorType.PARSE_ERROR,
            f"Unexpected error: {error_msg}\n{traceback.format_exc()}",
            error_dir, sample_id
        )
        return sample_id, False, error_msg



def get_file_list(
    input_dir: str,
    extensions: List[str] = ['.pdb', '.cif', '.mmcif']
) -> List[str]:
    """
    获取目录下所有结构文件
    Get all structure files in directory
    
    输入参数 / Input:
        - input_dir: str, 输入目录
        - extensions: list[str], 文件扩展名列表
    
    输出 / Output:
        - file_list: list[str], 文件路径列表
    """
    # Path, 输入目录路径
    input_path = Path(input_dir)
    if not input_path.exists():
        raise ValueError(f"Input directory does not exist: {input_dir}")
    # list[str], 文件列表
    file_list = []
    for ext in extensions:
        file_list.extend([str(p) for p in input_path.glob(f"*{ext}")])
    # 排序确保一致性
    file_list.sort()
    return file_list





def apply_sharding(
    file_list: List[str],
    part_id: int,
    total_parts: int
) -> List[str]:
    """
    应用分片策略
    Apply sharding strategy
    
    输入参数 / Input:
        - file_list: list[str], 完整文件列表
        - part_id: int, 当前分片ID (0-indexed)
        - total_parts: int, 总分片数
    
    输出 / Output:
        - shard_list: list[str], 当前分片的文件列表
    """
    if total_parts <= 1:
        return file_list
    if part_id < 0 or part_id >= total_parts:
        raise ValueError(f"Invalid part_id {part_id} for {total_parts} parts")
    # 计算每个分片的起止索引
    n = len(file_list)
    shard_size = n // total_parts
    remainder = n % total_parts
    # 前 remainder 个分片多分一个
    if part_id < remainder:
        start = part_id * (shard_size + 1)
        end = start + shard_size + 1
    else:
        start = remainder * (shard_size + 1) + (part_id - remainder) * shard_size
        end = start + shard_size
    return file_list[start:end]





def main():
    """
    主函数
    Main function
    """
    parser = argparse.ArgumentParser(
        description="Unified Preprocessing System for Protein/RNA Structures"
    )
    
    # 输入/输出/报错
    parser.add_argument(
        "--input_dir", type=str, required=True,
        help="Input directory containing PDB/CIF files"
    )
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="Output directory for processed data"
    )
    parser.add_argument(
        "--error_dir", type=str, required=True,
        help="Directory to save error information"
    )
    parser.add_argument(
        "--overwrite", action=argparse.BooleanOptionalAction, default=False,
        help="覆盖已有输出文件 (默认跳过); 用 --overwrite 开启, --no-overwrite 关闭"
    )
    


    # 分片
    parser.add_argument(
        "--part_id", type=int, default=0,
        help="Current partition ID (0-indexed, for distributed processing)"
    )
    parser.add_argument(
        "--total_parts", type=int, default=1,
        help="Total number of partitions"
    )
    
    # 并行
    parser.add_argument(
        "--n_jobs", type=int, default=1,
        help="Number of parallel jobs (-1 for all CPUs)"
    )
    


    # 参数
    parser.add_argument(
        "--binding_threshold", type=float, default=BINDING_THRESHOLD,
        help=f"Binding site distance threshold (default: {BINDING_THRESHOLD}Å)"
    )
    parser.add_argument(
        "--graph_cutoff", type=float, default=GRAPH_CUTOFF,
        help=f"Graph edge distance cutoff (default: {GRAPH_CUTOFF}Å)"
    )
    parser.add_argument(
        "--no_require_ligand", action="store_true",
        help="Do not require ligand in structure (process all files)"
    )
    parser.add_argument(
        "--no_compute_density", action="store_true",
        help="Skip density feature computation (faster)"
    )
    parser.add_argument(
        "--select_first_model", action="store_true",
        help="Select first model from structure (skip structures with multiple models)"
    )
    
    args = parser.parse_args()
    
    # =========================================================================
    # 初始化 / Initialize
    # =========================================================================
    print("=" * 70)
    print("Unified Preprocessing System / 统一预处理系统")
    print("=" * 70)
    print(f"Input:  {args.input_dir}")
    print(f"Output: {args.output_dir}")
    print(f"Shard:  {args.part_id + 1}/{args.total_parts}")
    print(f"Jobs:   {args.n_jobs}")
    print("=" * 70)
    
    # 创建输出目录
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # =========================================================================
    # 获取文件列表 / Get file list
    # =========================================================================
    all_files = get_file_list(args.input_dir)
    print(f"Total files found: {len(all_files)}")
    
    # 应用分片
    shard_files = apply_sharding(all_files, args.part_id, args.total_parts)
    print(f"Files in this shard: {len(shard_files)}")
    
    if len(shard_files) == 0:
        print("No files to process in this shard.")
        return
    
    # =========================================================================
    # 处理文件 / Process files
    # =========================================================================
    start_time = datetime.now()
    
    # 并行处理
    results = Parallel(n_jobs=args.n_jobs, verbose=10)(
        delayed(process_single_file)(
            file_path,
            args.output_dir,
            args.error_dir,
            overwrite=args.overwrite,
            binding_threshold=args.binding_threshold,
            graph_cutoff=args.graph_cutoff,
            require_ligand=not args.no_require_ligand,
            compute_density=not args.no_compute_density
        )
        for file_path in shard_files
    )
    
    # =========================================================================
    # 统计结果 / Summarize results
    # =========================================================================
    end_time = datetime.now()
    elapsed = (end_time - start_time).total_seconds()
    
    n_success = sum(1 for _, success, _ in results if success)
    n_failed = len(results) - n_success
    
    print("=" * 70)
    print("Processing Complete / 处理完成")
    print("=" * 70)
    print(f"Total:    {len(results)}")
    print(f"Success:  {n_success}")
    print(f"Failed:   {n_failed}")
    print(f"Time:     {elapsed:.2f}s ({elapsed / len(results):.3f}s/file)")
    print("=" * 70)
    
    # 列出失败样本
    if n_failed > 0:
        print("Failed samples:")
        for sample_id, success, error_msg in results:
            if not success and error_msg != "Already exists":
                print(f"  - {sample_id}: {error_msg}")
        print(f"\nError logs saved to: {Path(args.output_dir) / 'error_logs'}")


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

4. labels.npz (Labels & Ground Truth / 标签与真值)
--------------------------------------------------------------------------------
instance_ids                                        # np.ndarray, (N_atoms,), int32, 每个原子的结合位点实例 ID (背景为 -1)。与配体 ID 相同
ligand_ids                                          # np.ndarray, (N_atoms,), int32, 每个原子最近的配体 ID (即使距离很远)
distances                                           # np.ndarray, (N_atoms,), float32, 到最近配体原子的距离
binding_mask                                        # np.ndarray, (N_atoms,), bool, 如果最近配体距离 < 阈值 binding_threshold (4.5Å) 则为 True
num_ligands                                         # int, 结构中的配体数量
pocket_centers                                      # np.ndarray, (N_ligands, 3), float32, 结合口袋的几何中心(所有相应结合原子的中心, 空则回退至配体中心)
ligand_resnames                                     # np.ndarray, (N_ligands,), str, 配体的残基名称
ligand_coords_{id}                                  # np.ndarray, (N_lig_atoms, 3), float32, 第{id} 的配体的全部原子的坐标 (例如 ligand_coords_0)
"""


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

        atoms_dict 的键 / keys of atoms_dict:
            coords          - np.ndarray, (N_atoms, 3),       float32, 原子坐标
            features        - np.ndarray, (N_atoms, 49),      float32, 原子特征向量
            elements        - np.ndarray, (N_atoms,),          str,    元素符号
            res_indices     - np.ndarray, (N_atoms,),          int32,  所属残基索引
            chain_indices   - np.ndarray, (N_atoms,),          int32,  所属链索引
            res_names       - np.ndarray, (N_atoms,),          str,    残基名称
            atom_names      - np.ndarray, (N_atoms,),          str,    原子名称

        residues_dict 的键 / keys of residues_dict:
            coords            - np.ndarray, (N_res, 3),        float32, 残基代表原子坐标
            features          - np.ndarray, (N_res, 33),       float32, 残基特征向量
            names             - np.ndarray, (N_res,),           str,    残基名称
            types             - np.ndarray, (N_res,),           str,    残基类型 ('protein' / 'nucleotide')
            chain_indices     - np.ndarray, (N_res,),           int32,  链索引
            seq_numbers       - np.ndarray, (N_res,),           int32,  PDB 序列号
            local_frames      - np.ndarray, (N_res, 3, 3),     float32, 局部坐标系旋转矩阵
            frames_mask       - np.ndarray, (N_res,),           bool,   坐标系有效性掩码
            backbone_complete - np.ndarray, (N_res,),           bool,   骨架完整性标记

        graph_dict 的键 / keys of graph_dict:
            edge_row      - np.ndarray, (N_edges,), int32,   源节点索引
            edge_col      - np.ndarray, (N_edges,), int32,   目标节点索引
            edge_dist     - np.ndarray, (N_edges,), float32, 边距离
            edge_weight   - np.ndarray, (N_edges,), float32, 边权重 (1 / (dist + ε))
            num_atoms     - int,   原子总数
            num_residues  - int,   残基总数
            cutoff        - float, 距离截断值
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
        local_frames, frames_mask = compute_local_frames(parsed_data)

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
        raise ValueError(f"Failed to parse PDB file: {sample_id}, 错误日志见 {error_dir}")
