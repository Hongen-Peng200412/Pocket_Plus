"""
check.py — BOX 数据检查与清理脚本
===================================
功能:
  1. 检查所有 BOX 的空间维度是否为 (72,72,72), 输出并删除异常样本, 同步更新 split json
  2. 统计各原始类别口袋体素数 / 有原子的总体素数 (全局汇总)

用法:
  python check.py                 # 在 Linux 服务器上运行 (默认使用所有可用 CPU 核)
  python check.py --n_jobs 16     # 指定使用 16 个进程
  python check.py --dry-run       # 仅检查不删除 (安全模式)
"""
import argparse
import json
import os
import sys
from collections import defaultdict
import numpy as np
from joblib import Parallel, delayed

# ======================== 配置区 ========================
DATA_ROOT = "/storage/penghongen/Pocket_classic/v_0"
BOX_FOLDERS = ["emdb_BOX", "pdb_feature_BOX", "pdb_label_BOX"]
CLASS_FOLDERS = ["small_molecule", "metal_ion", "peptide", "nucleic"]
EXPECTED_SPATIAL = (72, 72, 72)
SPLIT_ROOT = os.path.join(DATA_ROOT, "split", "split_3")
SPLIT_NAMES = ["train.json", "val.json", "test.json"]

ORIGINAL_CLASSES = {
    1: "metal_ion",
    2: "peptide",
    3: "nucleic",
    4: "small_molecule",
}

# ======================================================================
#  并行计算的 Worker 函数
# ======================================================================
def _check_one_file(idx, total, fpath, folder, cls, fn):
    """用于 Part 1: 检查单个文件的空间维度"""
    # 进度提示: 每 1% 打印一次
    if total >= 100 and idx % max(1, total // 100) == 0:
        print(f"  [进度] Part 1 维度检查: {idx}/{total} ({idx/total*100:.1f}%)", flush=True)

    try:
        data = np.load(fpath)
        grid = data["grid"]
        spatial = grid.shape[-3:]
        data.close()
        if spatial != EXPECTED_SPATIAL:
            return (cls, fn[:-4], f"shape={grid.shape}, 空间维度={spatial} != {EXPECTED_SPATIAL}")
    except Exception as e:
        return (cls, fn[:-4], str(e))
    return None

def _stat_one_sample(idx, total, label_path, feature_path):
    """用于 Part 2: 统计单个样本的体素分布"""
    # 进度提示: 每 1% 打印一次
    if total >= 100 and idx % max(1, total // 100) == 0:
        print(f"  [进度] Part 2 体素统计: {idx}/{total} ({idx/total*100:.1f}%)", flush=True)

    try:
        label_grid = np.load(label_path)["grid"]
        feature_grid = np.load(feature_path)["grid"]

        # hardmask = (D, H, W)
        hardmask = np.any(feature_grid != 0, axis=0)
        atom_count = int(np.sum(hardmask))

        if atom_count == 0:
            return None
        
        total_voxels = int(np.prod(label_grid.shape[-3:]))
        atom_ratio = atom_count / total_voxels

        label_map = np.round(label_grid[0]).astype(np.int32)
        unique_ids = set(np.unique(label_map)) - {0}

        class_voxels = {}
        for cid in unique_ids:
            class_voxels[cid] = int(np.sum(label_map == cid))

        return {
            'atom_count': atom_count,
            'atom_ratio': atom_ratio,
            'class_voxels': class_voxels,
            'unique_ids': unique_ids
        }
    except Exception as e:
        return {"error": f"{label_path}: {e}"}

# ======================================================================
#  Part 1: 检查 BOX 空间维度, 删除异常样本, 更新 split json
# ======================================================================
def check_and_clean(n_jobs: int = -1, dry_run: bool = False):
    print("=" * 70)
    print("  Part 1: 检查 BOX 空间维度 (多进程)")
    print("=" * 70)

    # 1. 收集所有任务
    tasks = []
    for folder in BOX_FOLDERS:
        for cls in CLASS_FOLDERS:
            dir_path = os.path.join(DATA_ROOT, folder, cls)
            if not os.path.isdir(dir_path):
                continue
            for fn in sorted(os.listdir(dir_path)):
                if fn.endswith(".npz"):
                    fpath = os.path.join(dir_path, fn)
                    tasks.append((fpath, folder, cls, fn))

    total = len(tasks)
    if total == 0:
        print("  [提示] 未找到任何 .npz 文件，请检查 DATA_ROOT。")
        return

    print(f"  [信息] 共发现 {total} 个 .npz 文件即将被检查。")
    
    # 2. 并行执行任务
    results = Parallel(n_jobs=n_jobs)(
        delayed(_check_one_file)(i + 1, total, fpath, folder, cls, fn)
        for i, (fpath, folder, cls, fn) in enumerate(tasks)
    )

    # 3. 汇总异常
    bad_samples = defaultdict(set)
    total_bad_files = 0
    for res in results:
        if res is not None:
            cls, name, err = res
            bad_samples[cls].add(name)
            total_bad_files += 1
            print(f"  [异常] {cls}/{name}.npz: {err}")

    # 异常样本指在任一文件夹中发生异常的独立样本数
    unique_bad_samples = sum(len(v) for v in bad_samples.values())
    print(f"\n  检查完毕: 累计检查 {total} 个文件, 发现 {unique_bad_samples} 个异常样本（共涉及 {total_bad_files} 个异常切片）。")

    if unique_bad_samples == 0:
        print("  ✅ 所有样本维度正常，无需清理。")
        return

    # 打印异常样本
    print(f"\n  异常样本列表:")
    for cls in CLASS_FOLDERS:
        if cls not in bad_samples: continue
        for name in sorted(bad_samples[cls]):
            print(f"    {cls}/{name}")

    if dry_run:
        print(f"\n  [DRY-RUN] 仅检查模式，未删除/修改任何文件。")
        return

    # 删除异常文件
    deleted_count = 0
    for cls, names in bad_samples.items():
        for name in sorted(names):
            for folder in BOX_FOLDERS:
                fpath = os.path.join(DATA_ROOT, folder, cls, f"{name}.npz")
                if os.path.exists(fpath):
                    os.remove(fpath)
                    deleted_count += 1
                    print(f"  [删除] {fpath}")
    print(f"\n  已删除 {deleted_count} 个文件")

    # 更新 split json
    print(f"\n  更新 split json 文件...")
    for cls, names in bad_samples.items():
        if len(names) == 0: continue
        for split_name in SPLIT_NAMES:
            json_path = os.path.join(SPLIT_ROOT, cls, split_name)
            if not os.path.exists(json_path): continue
            
            with open(json_path, "r", encoding="utf-8") as f:
                entries = json.load(f)
            original_len = len(entries)
            entries = [e for e in entries if e not in names]
            removed = original_len - len(entries)
            
            if removed > 0:
                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(entries, f, indent=2, ensure_ascii=False)
                print(f"  [更新] {json_path}: 移除 {removed} 条 (剩余 {len(entries)} 条)")

    print(f"\n  Part 1 结束。")


# ======================================================================
#  Part 2: 统计各原始类别体素占比 (全局汇总)
# ======================================================================
def compute_voxel_stats(n_jobs: int = -1):
    print("\n" + "=" * 70)
    print("  Part 2: 统计各原始类别体素占比 (多进程)")
    print("=" * 70)

    # 1. 收集任务
    tasks = []
    for cls_folder in CLASS_FOLDERS:
        label_dir = os.path.join(DATA_ROOT, "pdb_label_BOX", cls_folder)
        feature_dir = os.path.join(DATA_ROOT, "pdb_feature_BOX", cls_folder)
        if not os.path.isdir(label_dir): continue
        
        for fn in sorted(os.listdir(label_dir)):
            if fn.endswith(".npz"):
                feature_path = os.path.join(feature_dir, fn)
                if os.path.exists(feature_path):
                    tasks.append((os.path.join(label_dir, fn), feature_path))

    total = len(tasks)
    if total == 0:
        print("  [提示] 未找到任何有效样本用于统计。")
        return
    print(f"  [信息] 需要处理的样本总数: {total}")

    # 2. 并行执行
    results = Parallel(n_jobs=n_jobs)(
        delayed(_stat_one_sample)(i + 1, total, label_path, feature_path)
        for i, (label_path, feature_path) in enumerate(tasks)
    )

    # 3. 结果汇总
    global_atom_voxels = 0
    total_valid_samples = 0
    per_sample_atom_ratio = []
    global_class_voxels = defaultdict(int)
    per_sample_class_ratio = defaultdict(list)
    discovered_class_ids = set()

    for res in results:
        if res is None:
            continue
        if "error" in res:
            print(f"  [错误] {res['error']}")
            continue
            
        total_valid_samples += 1
        global_atom_voxels += res['atom_count']
        per_sample_atom_ratio.append(res['atom_ratio'])
        
        discovered_class_ids.update(res['unique_ids'])
        for cid, count in res['class_voxels'].items():
            global_class_voxels[cid] += count
            per_sample_class_ratio[cid].append(count / res['atom_count'])
            
        # 兼容当前样本没有的类别存 0.0
        for cid in ORIGINAL_CLASSES:
            if cid not in res['unique_ids']:
                per_sample_class_ratio[cid].append(0.0)

    # 4. 打印报告
    print(f"\n{'=' * 70}")
    print(f"  全局统计结果 (共 {total_valid_samples} 个有效样本)")
    print(f"{'=' * 70}")
    print(f"  有原子体素总数: {global_atom_voxels:,}")

    if per_sample_atom_ratio:
        arr = np.array(per_sample_atom_ratio)
        print(f"\n  --- 稀疏度: 有原子体素 / 总体素 ({EXPECTED_SPATIAL[0]}^3 = {EXPECTED_SPATIAL[0]**3:,}) ---")
        _print_distribution(arr)

    all_class_ids = sorted(set(ORIGINAL_CLASSES.keys()) | discovered_class_ids)

    print(f"\n  --- 各原始类别: 口袋体素数 / 有原子体素数 ---")
    print(f"  {'类别ID':<8} {'类别名':<16} {'全局体素数':>14} {'全局比例':>12} "
          f"{'样本均值':>12} {'样本中位数':>12} {'样本P90':>12}")
    print(f"  {'-' * 88}")

    for cid in all_class_ids:
        cname = ORIGINAL_CLASSES.get(cid, f"class_{cid}")
        gc = global_class_voxels.get(cid, 0)
        global_ratio = gc / global_atom_voxels if global_atom_voxels > 0 else 0

        if per_sample_class_ratio[cid]:
            arr = np.array(per_sample_class_ratio[cid])
            mean_r = float(np.mean(arr))
            median_r = float(np.median(arr))
            p90_r = float(np.percentile(arr, 90))
        else:
            mean_r = median_r = p90_r = 0.0

        print(f"  {cid:<8} {cname:<16} {gc:>14,} {global_ratio:>12.6f} "
              f"{mean_r:>12.6f} {median_r:>12.6f} {p90_r:>12.6f}")

    bg_voxels = global_atom_voxels - sum(global_class_voxels.values())
    bg_ratio = bg_voxels / global_atom_voxels if global_atom_voxels > 0 else 0
    print(f"  {'0':<8} {'background':<16} {bg_voxels:>14,} {bg_ratio:>12.6f}")

    print(f"\n  Part 2 结束。")


def _print_distribution(arr: np.ndarray, prefix: str = "    "):
    print(f"{prefix}均值   = {np.mean(arr):.6f}")
    print(f"{prefix}标准差 = {np.std(arr):.6f}")
    print(f"{prefix}最小值 = {np.min(arr):.6f}")
    print(f"{prefix}P25    = {np.percentile(arr, 25):.6f}")
    print(f"{prefix}中位数 = {np.median(arr):.6f}")
    print(f"{prefix}P75    = {np.percentile(arr, 75):.6f}")
    print(f"{prefix}P90    = {np.percentile(arr, 90):.6f}")
    print(f"{prefix}最大值 = {np.max(arr):.6f}")


# ======================================================================
#  主入口
# ======================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BOX 数据检查与清理脚本")
    parser.add_argument("--dry-run", action="store_true", help="仅检查不删除/修改任何文件")
    parser.add_argument("--skip-check", action="store_true", help="跳过 Part 1 (维度检查)")
    parser.add_argument("--skip-stats", action="store_true", help="跳过 Part 2 (统计)")
    parser.add_argument("--n_jobs", type=int, default=-1, help="并行线程/进程数 (默认 -1 即使用所有可用核)")
    args = parser.parse_args()

    if not args.skip_check:
        check_and_clean(n_jobs=args.n_jobs, dry_run=args.dry_run)

    if not args.skip_stats:
        compute_voxel_stats(n_jobs=args.n_jobs)

    print("\n✅ 全部完成。")
