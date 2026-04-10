# -*- coding: utf-8 -*-
"""
================================================================================
数据统计脚本 / Data Statistics Script
================================================================================
本脚本用于处理解析完成的 PDB 提取特征结果，并计算出：
1. 过滤前的各类候选配体数目分布
2. 过滤后的各类口袋（实例维度）的分别数量、口袋大小（原子数）
3. 各类口袋占整体受体的比例

仅需由一个进程最终调用一次。
================================================================================
"""

import os
import argparse
import numpy as np
from pathlib import Path
from joblib import Parallel, delayed
import sys

def process_single_sample(sample_dir: str):
    """
    处理单个样本目录，提取它在过滤前、过滤后的配体和口袋特征供统计汇总。

    输入参数:
        - sample_dir: str, 标量, 单个样本(如特定 PDB ID)经过解析后的输出子目录路径

    输出:
        - pre_filter_counts: dict[str, int], 各类特定标志(如是金属离子等)的候选配体数
        - pre_filter_other_count: int, 标量, 剩余候选配体数
        - pre_filter_total: int, 标量, `candidates.npz` 中记录的该样本候选配体数量

        - pocket_atom_sizes: dict[str, list[int]], 该样本中各类保留口袋内部【每个实例】独立包含的原子个数字典: pocket_atom_sizes={class_name: [size1, size2, ...]}
        - class_name_atom_counts: dict[str, int], 过滤后每种口袋的受体原子数: class_name_atom_counts={class_name: atom_count}
        - total_receptor_atoms: int, 标量, `labels.npz` 中当前受体具有的蛋白/核酸骨架总原子数
    """
    candidates_path = os.path.join(sample_dir, 'candidates.npz')
    labels_path = os.path.join(sample_dir, 'labels.npz')
    
    # ---------------- 1. 解析 candidates.npz (过滤前) ----------------
    pre_filter_counts = {}
    pre_filter_other_count = 0
    pre_filter_total = 0
    
    if os.path.exists(candidates_path):
        data = np.load(candidates_path, allow_pickle=True)
        if 'n_candidates' in data and data['n_candidates'] > 0:
            n = int(data['n_candidates'])
            pre_filter_total = n
            
            keys_to_check = ['is_metal_ion', 'is_peptide_like', 'is_nucleotide_like']
            for k in keys_to_check:
                pre_filter_counts[k] = 0
                if k in data:
                    pre_filter_counts[k] = int(np.sum(data[k]))
                    
            # 通过向量化逻辑计算 'other'
            arrs = []
            for k in keys_to_check:
                if k in data:
                    arrs.append(data[k])
                else:
                    arrs.append(np.zeros(n, dtype=bool))
            
            if arrs:
                any_mask = np.logical_or.reduce(arrs)
                pre_filter_other_count = int(np.sum(~any_mask))
            else:
                pre_filter_other_count = n
                    


    # ---------------- 2. 解析 labels.npz (过滤后) ----------------
    # pocket_atom_sizes: {class_name: [size1, size2, ...]} - 记录每个独立 instance 的原子数
    pocket_atom_sizes = {}
    # pocket_proportion: {class_name: float_count} - 记录每个 class 原子的数量，便于后续算比例
    class_name_atom_counts = {}
    total_receptor_atoms = 0
    
    if os.path.exists(labels_path):
        data = np.load(labels_path, allow_pickle=True)
        if 'instance_ids' in data and 'pocket_class_ids' in data and 'binding_mask' in data:
            instance_ids = data['instance_ids']
            pocket_class_ids = data['pocket_class_ids']
            binding_mask = data['binding_mask']
            total_receptor_atoms = len(instance_ids)
            
            # 提取类别映射 (0:background,1:druggable，等)
            class_map = {}
            if 'pocket_class_name_map' in data:
                raw_map = str(data['pocket_class_name_map'])  # "0:background,1:druggable,..."
                try:
                    for part in raw_map.split(','):
                        if ':' in part:
                            k, v = part.split(':')
                            class_map[int(k)] = v
                except:
                    pass
            if not class_map:
                class_map = {1: 'pocket_1'}
            
            # 聚类原子 -> instance_id 以区分口袋实例维度（Instance-Level）
            instance_atom_cnt = {}
            instance_class = {}
            
            for i in range(total_receptor_atoms):
                iid = instance_ids[i]
                if iid != -1 and binding_mask[i]:
                    cid = pocket_class_ids[i]
                    instance_atom_cnt[iid] = instance_atom_cnt.get(iid, 0) + 1
                    instance_class[iid] = cid
                    
                    cname = class_map.get(cid, f"class_{cid}")
                    if cname != "background":
                        class_name_atom_counts[cname] = class_name_atom_counts.get(cname, 0) + 1
            
            for iid, count in instance_atom_cnt.items():
                cid = instance_class[iid]
                cname = class_map.get(cid, f"class_{cid}")
                if cname != "background":
                    if cname not in pocket_atom_sizes:
                        pocket_atom_sizes[cname] = []
                    pocket_atom_sizes[cname].append(count)

    return pre_filter_counts, pre_filter_other_count, pre_filter_total, pocket_atom_sizes, class_name_atom_counts, total_receptor_atoms


def print_stats(name, array):
    """形式化输出统计信息。计算各个十分位，均值及方差"""
    if len(array) == 0:
        print(f"  [{name}]: 无数据 (0)")
        return
        
    arr = np.array(array)
    mean = np.mean(arr)
    var = np.var(arr)
    p0 = np.min(arr)
    p10 = np.percentile(arr, 10)
    p20 = np.percentile(arr, 20)
    p30 = np.percentile(arr, 30)
    p40 = np.percentile(arr, 40)
    p50 = np.median(arr)
    p60 = np.percentile(arr, 60)
    p70 = np.percentile(arr, 70)
    p80 = np.percentile(arr, 80)
    p90 = np.percentile(arr, 90)
    p100 = np.max(arr)
    
    print(f"  [{name}]")
    print(f"    - 数据点数 / Count : {len(arr)}")
    print(f"    - 均值 / Mean      : {mean:.4f}")
    print(f"    - 方差 / Variance  : {var:.4f}")
    print(f"    - 十分位数分布 / Deciles:")
    print(f"        0% (Min) = {p0:.2f} \t 10% = {p10:.2f} \t 20% = {p20:.2f} \t 30% = {p30:.2f} \t 40% = {p40:.2f}")
    print(f"        50%(Med) = {p50:.2f} \t 60% = {p60:.2f} \t 70% = {p70:.2f} \t 80% = {p80:.2f} \t 90% = {p90:.2f}")
    print(f"        100%(Max)= {p100:.2f}")
    print("-" * 60)


def main():
    parser = argparse.ArgumentParser(description="PDB 配体与口袋数据统计分析")
    parser.add_argument("--input_dir", type=str, required=True, help="解析后的样本输出大目录 (如 parsed_pdb/)")
    parser.add_argument("--n_jobs", type=int, default=8, help="并行线程数 (默认8)")
    args = parser.parse_args()
    
    input_path = Path(args.input_dir)
    if not input_path.exists():
        print(f"[Error] 输入目录不存在 / Directory not found: {args.input_dir}")
        sys.exit(1)
        
    sample_dirs = [str(d) for d in input_path.iterdir() if d.is_dir() and (d / "candidates.npz").exists()]
    print(f"[*] 发现有效样本数目: {len(sample_dirs)}")
    
    if len(sample_dirs) == 0:
        print("[!] 没有可供分析的样本，退出。")
        sys.exit(0)

    print(f"[*] 开始并行加载数据并计算属性 (n_jobs={args.n_jobs})...")
    results = Parallel(n_jobs=args.n_jobs, verbose=5)(
        delayed(process_single_sample)(sdir) for sdir in sample_dirs
    )
    


    # ---------------- 准备全局容器 ----------------
    # 1. 过滤前的候选分布
    global_pre_filter_type_counts = {} # 每个样本里的类型的配体总个数数组, { 'is_metal_ion': [1, 2, 0, ...], ... }
    global_pre_filter_other = []
    global_pre_filter_total = []

    # 2. 过滤后各个 instance 的大小 (跨样本平铺)
    global_pocket_atom_sizes = {} # { 'druggable': [大小1, 大小2, ...], ... }
    
    # 3. 过滤后的某个特定 pocket class 每样本出现的数量
    global_pocket_nums = {} # { 'druggable': [0, 1, 2, ...], ... }
    
    # 4. 各类占总原子数比例
    global_class_proportion = {} # { 'druggable': [0.0, 0.1, ...], ... }

    # 首先搜集所有的类别，预分配空数组，保证0值填充（如果样本中没有该类，需置0）
    all_is_keys = set()
    all_class_names = set()
    
    for (pre_cnt, _, _, p_sizes, c_counts, _) in results:
        all_is_keys.update(pre_cnt.keys())
        all_class_names.update(p_sizes.keys())
        all_class_names.update(c_counts.keys())
        
    for k in all_is_keys:
        global_pre_filter_type_counts[k] = []
    for k in all_class_names:
        if k != "background":
            global_pocket_atom_sizes[k] = []
            global_pocket_nums[k] = []
            global_class_proportion[k] = []
            
    # ------ 二次遍历，填充值 ------
    for (pre_cnt, pre_other, pre_tot, p_sizes, c_counts, tot_atoms) in results:
        global_pre_filter_total.append(pre_tot)
        global_pre_filter_other.append(pre_other)
        for k in all_is_keys:
            global_pre_filter_type_counts[k].append(pre_cnt.get(k, 0))
            
        for k in all_class_names:
            if k == "background":
                continue
            # 单样本内各类口袋数量（有该类型的存在才计数，不存在则0）
            nums = len(p_sizes.get(k, []))
            global_pocket_nums[k].append(nums)
            
            # 各个独立口袋的原子容量，跨样本合并（实例维度，只有存在才合并进去）
            for sz in p_sizes.get(k, []):
                global_pocket_atom_sizes[k].append(sz)
                
            # 各类占据比例（样本维度）
            prop = c_counts.get(k, 0) / tot_atoms if tot_atoms > 0 else 0.0
            global_class_proportion[k].append(prop)

    # ---------------- 输出阶段 ----------------
    print("\n" + "=" * 90)
    print("                 统计汇总输出 / STATISTICAL RESULTS SUMMARY")
    print("=" * 90)
    print(f"总计评估的样本文件夹个数: {len(results)}\n")
    
    print("\n【 一、 过滤前：样本各类候选配体数目 (Pre-filter candidate ligand counts per sample) 】")
    print("  -> 含义：读取 candidates.npz。对于每个独立的 pdb 样本，统计各个 `is_` 布尔分类标志等于 True")
    print("     的候选配体总数。下列结果反映的是这个数目在跨所有样本上的“均值”、“方差”和“各十分位数值”。")
    print("     （比如，如果某类的 P50 是 3，表明有一半的 pdb 样本在这类标志的数量上 ≤ 3）\n")
    print_stats("Total Candidates (总候选配体个数)", global_pre_filter_total)
    for k, v in global_pre_filter_type_counts.items():
        print_stats(f"分类: {k}", v)
    print_stats("分类: other (皆未满足任何 is_ 标志)", global_pre_filter_other)
    
    
    print("\n【 二、 过滤后：样本内各类配体(口袋)的数量分布 (Post-filter pocket count per sample) 】")
    print("  -> 含义：读取 labels.npz。统计经过 filter_config.py 规则筛选出各类真正的合法口袋后，单")
    print("     个样本中拥有属于同一类别名称 `pocket_class_name` 的口袋的总个数。\n")
    for k, v in global_pocket_nums.items():
        print_stats(f"数量: {k}", v)
        
        
    print("\n【 三、 过滤后：各类别“个体口袋”分别包含的原子数目 (Atoms per unique pocket instance) 】")
    print("  -> 含义：跨所有样本聚合，考察全部被判定为类 C 的“每一个独立的口袋个体（由 instance_id ")
    print("     区分）”里，独立含有了多少个结合原子。这里的统计单位是【每个口袋】，并非【每个样本】。\n")
    for k, v in global_pocket_atom_sizes.items():
        print_stats(f"原子个数值: {k}", v)
        
        
    print("\n【 四、 过滤后：各类口袋的结合原子占整个受体骨架总原子的比例 (Proportion of atoms in class) 】")
    print("  -> 含义：按单样本计算，样本中归属于类 C 的特异性结合原子个数总和，除以该受体总氨基酸/核")
    print("     苷酸原本具有的总原子个数的比例分布（即样本占比）。\n")
    for k, v in global_class_proportion.items():
        print_stats(f"原子比例: {k} (%)", [x * 100 for x in v])

    print("=" * 90)
    print("统计正常结束。 / Statistics completed successfully.\n")

if __name__ == "__main__":
    main()
