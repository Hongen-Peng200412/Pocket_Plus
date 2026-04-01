import os
import sys
import re
import numpy as np
import torch
import rootutils
import json

ROOT = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from pathlib import Path
POCKET_ROOT = Path(__file__).resolve().parent.parent.parent  # Pocket/
if str(POCKET_ROOT) in sys.path:
    sys.path.remove(str(POCKET_ROOT))
sys.path.insert(0, str(POCKET_ROOT))



# 导入推断模块
from src.inference.get_pred import load_model, run_inference

from src.inference.evaluator import print_metrics
from src.inference.utils.utils import (
    write_batch_excel,
    write_param_search_excel,
    generate_param_grid,
)



# =============================================================================
# 1. 从已处理的 .npz 目录加载单个样本
# =============================================================================
from src.datasets.box_geometry import build_hardmask_from_world_coordinates
from src.inference.parse_input import apply_class_mapping


_BOX_SAMPLE_NAME_PATTERN = re.compile(
    r"^(?P<pdb_id>.+)_(?P<instance_id>-?\d+)_(?P<rxx>-?\d+)_(?P<ryy>-?\d+)_(?P<rzz>-?\d+)(?P<center>_C)?$"
)


def _parse_box_sample_name(sample_name: str) -> dict[str, str]:
    """
    从 BOX 样本名中解析出结构级 `pdb_id`。

    输入参数:
        - sample_name: str, 标量, BOX 样本名, 例如 `9f3f_0_0_0_0_C`

    输出:
        - dict[str, str], 仅包含:
            - "pdb_id": str, 小写结构 ID
    """
    matched = _BOX_SAMPLE_NAME_PATTERN.match(sample_name)
    if matched is None:
        raise ValueError(f"[pipline_for_box] 非法 sample_name: {sample_name}")
    return {"pdb_id": matched.group("pdb_id").lower()}


def _load_atom_coords_for_box(sample_root_path: str, sample_name: str) -> np.ndarray:
    """
    读取 BOX 对应结构的全局原子世界坐标。

    输入参数:
        - sample_root_path: str, 标量, 结构缓存根目录, 要求其下存在 `pdb_id/atoms.npz`
        - sample_name: str, 标量, BOX 样本名

    输出:
        - atom_coords_world: np.ndarray, (N_atom, 3), float32, 全局原子世界坐标
    """
    # str, 标量, 从 BOX 样本名解析出的结构 ID
    pdb_id = _parse_box_sample_name(sample_name)["pdb_id"]
    # Path, 标量, atoms.npz 的绝对路径
    atoms_path = Path(sample_root_path) / pdb_id / "atoms.npz"
    if not atoms_path.exists():
        raise FileNotFoundError(f"[pipline_for_box] 未找到 atoms.npz: {atoms_path}")

    with np.load(atoms_path) as npz_file:
        # np.ndarray, (N_atom, 3), float32, 全局原子世界坐标
        atom_coords_world = np.asarray(npz_file["coords"], dtype=np.float32)

    if atom_coords_world.ndim != 2 or atom_coords_world.shape[1] != 3:
        raise ValueError(
            f"[pipline_for_box] atoms.npz 中的 coords 形状非法: {atom_coords_world.shape}, path={atoms_path}"
        )
    return atom_coords_world.astype(np.float32, copy=False)


def voxel_evaluate(
    pred_label: np.ndarray,
    gt_label: np.ndarray,
    hardmask: np.ndarray = None,
    positive_class: int = 1,
) -> dict:
    """
    在体素级别计算语义分割的评估指标。

    # 输入参数:
        - pred_label: np.ndarray, 形状 (D, H, W), 预测标签
        - gt_label: np.ndarray, 形状 (D, H, W), 真实标签
        - hardmask: np.ndarray | None, 形状 (D, H, W), 1=参与评估, 0=忽略
        - positive_class: int, 正类 ID, 默认 1

    # 输出:
        - metrics: dict, 包含:
            - "precision": float
            - "recall": float
            - "f1": float
            - "iou": float
            - "tp": int
            - "fp": int
            - "fn": int
            - "tn": int
            - "num_eval": int
    """
    if hardmask is not None:
        mask = hardmask.astype(bool)
    else:
        mask = np.ones(pred_label.shape, dtype=bool)

    pred_flat = pred_label[mask]
    gt_flat = gt_label[mask]
    num_eval = int(pred_flat.size)
    if num_eval == 0:
        return {
            "mode": "voxel",
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "iou": 0.0,
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "tn": 0,
            "num_eval": 0,
        }

    # np.ndarray, 形状 (N_eval,), bool
    pred_pos = (pred_flat == positive_class)
    gt_pos = (gt_flat == positive_class)

    tp = int(np.sum(pred_pos & gt_pos))
    fp = int(np.sum(pred_pos & ~gt_pos))
    fn = int(np.sum(~pred_pos & gt_pos))
    tn = int(np.sum(~pred_pos & ~gt_pos))

    # 若模型完美拒绝了没有任何 GT 的负样本，赋予满分
    if tp == 0 and fp == 0 and fn == 0 and tn > 0:
        precision = 1.0
        recall = 1.0
        f1 = 1.0
        iou = 1.0
    else:
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)
              if (precision + recall) > 0 else 0.0)
        iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0

    return {
        "mode": "voxel",
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "iou": iou,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "num_eval": num_eval,
    }
def load_from_npz_dirs(
    all_data_path: str,
    class_folder: str,              # ["small_molecule", "metal_ion", "peptide", "nucleic"]
    data_folder_names: list = None, # ["emdb_BOX", "pdb_feature_BOX", "pdb_label_BOX"]
    sample_name: str = None,
    sample_root_path: str = None,
    class_mapping: list = None,
) -> dict:
    """
    从已处理的 .npz 目录（emdb_BOX / pdb_feature_BOX / pdb_label_BOX）加载单个样本，拼接特征、提取标签和 hardmask。
    此函数复用了 MyDatasets.__getitem__() 的核心加载逻辑，但不做数据增强。

    Args:
        - all_data_path:     str,       数据集根目录（如 "/storage/.../Pocket_classic/v_0"）
        - class_folder:      str,       类别子文件夹名（如 "small_molecule"）
        - data_folder_names: list[str], 数据文件夹列表, 默认 ["emdb_BOX", "pdb_feature_BOX", "pdb_label_BOX"]。
                             see me: 1.含 "label" 字段的文件夹被视为标签, 含 "emdb" 字段的文件夹做 z-score 归一化, 其余直接拼接为特征。
                                   2.含 "emdb" 的文件夹仍建议排在最前面，以保持与训练配置一致的输入通道顺序
        - sample_name:       str,       样本名（不含 .npz 后缀，如 "9f3f_0_0_0_0_C"）
        - sample_root_path:  str,       结构缓存根目录, 用于读取 `pdb_id/atoms.npz` 并基于原子几何位置生成 hardmask
        - class_mapping:     list[int] | None, 标签类别映射表, 例如 [0, 1, 1, 1, 1] 将 4 类口袋合并为 1 类。

    Returns:
        - result: dict, 包含以下键:
            - "grid":       np.ndarray, float32, (C, D, H, W), 拼接后的完整特征网格
            - "label":      np.ndarray, int64,   (D, H, W),    标签（若存在标签文件夹）; 若不存在标签文件夹则此键不存在
            - "hardmask":   np.ndarray, int64,   (D, H, W),    几何定义的硬掩膜（1=至少有一个原子落入该体素）
            - "emdb_channels": int, EMDB 密度图占据的通道数
            - "sample_name": str, 样本名
            - "class_folder": str, 类别文件夹名
    """
    if sample_root_path is None:
        raise ValueError("[pipline_for_box] load_from_npz_dirs 现在要求显式传入 sample_root_path，用于几何 hardmask 生成")

    grid_parts = []                # list[np.ndarray], 特征部分，后续拼接
    label = None                   # np.ndarray | None, 标签
    emdb_channels = 0              # int, EMDB 密度图特征占的通道数
    box_origin_world = None        # np.ndarray | None, (3,), 当前 BOX 的世界坐标原点
    voxel_size_world = None        # np.ndarray | None, (3,), 当前 BOX 的体素尺寸
    box_shape_zyx = None           # np.ndarray | None, (3,), 当前 BOX 的空间大小

    # 检查 emdb 文件夹是否在最前面
    non_emdb_seen = False
    for folder_name in data_folder_names:
        if "emdb" in folder_name:
            if non_emdb_seen:
                raise ValueError(f"含有 'emdb' 的特征文件夹必须排在最前面，但当前配置为: {data_folder_names}")
        else:
            non_emdb_seen = True

    for folder_name in data_folder_names:
        npz_path = os.path.join(
            all_data_path, folder_name, class_folder, sample_name + ".npz"
        )

        # ---- 标签分支 ----
        if "label" in folder_name:
            if not os.path.exists(npz_path):
                # 标签文件不存在 → 跳过（标签可选）
                continue
            with np.load(npz_path) as npz_file:
                raw_label = np.asarray(npz_file["grid"])   # np.ndarray, (1,D,H,W) 或 (D,H,W)
                if box_origin_world is None:
                    box_origin_world = np.asarray(npz_file["origin"], dtype=np.float32).reshape(3)
                    voxel_size_world = np.asarray(npz_file["voxel_size"], dtype=np.float32).reshape(3)
                    box_shape_zyx = np.asarray(raw_label.shape[-3:], dtype=np.int64)
            if raw_label.ndim == 4:
                raw_label = raw_label[0]             # (1,D,H,W) → (D,H,W)
            if class_mapping is not None:
                raw_label = apply_class_mapping(raw_label, class_mapping)
            label = np.round(raw_label).astype(np.int64)  # np.ndarray, int64, (D,H,W)
            continue

        # ---- 特征分支 ----
        if not os.path.exists(npz_path):
            raise FileNotFoundError( f"[parse_input] 特征文件不存在: {npz_path}")
        with np.load(npz_path) as npz_file:
            _grid = np.asarray(npz_file["grid"])   # np.ndarray, (C_k, D, H, W), float
            if box_origin_world is None:
                box_origin_world = np.asarray(npz_file["origin"], dtype=np.float32).reshape(3)
                voxel_size_world = np.asarray(npz_file["voxel_size"], dtype=np.float32).reshape(3)
                box_shape_zyx = np.asarray(_grid.shape[-3:], dtype=np.int64)
        # 对 EMDB 密度图做 z-score 归一化
        if "emdb" in folder_name:
            _grid = (_grid - np.mean(_grid)) / (np.std(_grid) + 1e-8)
            emdb_channels += _grid.shape[0]
        grid_parts.append(_grid)

    # 拼接特征通道
    # NOTE：拼接顺序与 data_folder_names 的遍历顺序一致。
    #       hardmask 已切换为几何定义，不再依赖特征通道的非零模式；
    #       这里仍保留现有目录顺序约束，仅用于保持与训练阶段一致的输入通道布局。
    if not grid_parts:
        raise RuntimeError(
            f"[parse_input] 未找到任何特征文件: "
            f"all_data_path={all_data_path}, class_folder={class_folder}, "
            f"sample_name={sample_name}"
        )
    if box_origin_world is None or voxel_size_world is None or box_shape_zyx is None:
        raise RuntimeError(
            f"[pipline_for_box] 未能从 BOX npz 读取几何元信息: "
            f"all_data_path={all_data_path}, class_folder={class_folder}, sample_name={sample_name}"
        )
    grid = np.concatenate(grid_parts, axis=0).astype(np.float32)
    # np.ndarray, float32, (C, D, H, W), 其中 C = sum(各特征文件夹的通道数)

    # np.ndarray, (N_atom, 3), float32, 结构级全局原子世界坐标
    atom_coords_world = _load_atom_coords_for_box(sample_root_path=sample_root_path, sample_name=sample_name)
    # np.ndarray, int64, (D, H, W), 几何定义的 hardmask
    hardmask = build_hardmask_from_world_coordinates(
        atom_coords_world=atom_coords_world,
        box_origin_world=box_origin_world,
        voxel_size_world=voxel_size_world,
        box_shape_zyx=box_shape_zyx,
    )

    result = {
        "grid":          grid,
        "hardmask":      hardmask,
        "emdb_channels": emdb_channels,
        "sample_name":   sample_name,
        "class_folder":  class_folder,
    }
    if label is not None:
        result["label"] = label

    return result



# =============================================================================
# 2. 扫描数据目录，为批量推断/评估准备数据条目
# =============================================================================
def prepare_data_entries(
    all_data_path: str,
    class_folder_names: list,       # ["small_molecule", "metal_ion", "peptide", "nucleic"]
    data_folder_names: list = None, # ["emdb_BOX", "pdb_feature_BOX", "pdb_label_BOX"]
    split_files: list = None,
) -> list:
    """
    扫描数据根目录和 split JSON 文件，定位每个样本的路径信息。

    Args:
        - all_data_path:      str,        数据集根目录
        - class_folder_names: list[str],  类别子文件夹名列表, 默认 ["small_molecule", "metal_ion", "peptide", "nucleic"]
        - data_folder_names:  list[str],  数据文件夹列表，默认 ["emdb_BOX", "pdb_feature_BOX", "pdb_label_BOX"]
        - split_files:        list[str] | None, JSON 划分文件列表, 若为 None，则扫描每个 class_folder 下的全部 .npz 文件. 
                                          see me: split_files 需要与 class_folder_names 长度相同且一一对应
                              
    Returns:
        - entries: list[dict], 每项包含：
            - "all_data_path":      str,        数据根目录
            - "class_folder":       str,        类别文件夹名
            - "data_folder_names":  list[str],  数据文件夹列表
            - "sample_name":        str,        样本名（不含 .npz）
            - "has_label":          bool,       是否存在标签文件

    打印统计信息：总条目数、含标签数、各类别分布。
    """
    # 找出哪个文件夹是标签文件夹
    label_folder = None
    feature_folder_for_scan = None
    for fn in data_folder_names:
        if "label" in fn:
            label_folder = fn
        elif feature_folder_for_scan is None:
            feature_folder_for_scan = fn  # 用第一个非标签文件夹来扫描样本名

    entries = []
    skipped_samples = []   # list[(class_folder, sample_name, reason)], 记录被跳过的样本

    for cls_idx, class_folder in enumerate(class_folder_names):
        # 获取样本名列表
        if split_files is not None:
            # NOTE: split_files 需要与 class_folder_names 长度相同且一一对应
            split_path = split_files[cls_idx] 
            if not os.path.exists(split_path):
                print(f"[parse_input] ⚠️ split 文件不存在，跳过: {split_path}")
                continue
            with open(split_path, "r", encoding="utf-8") as f:
                sample_names = json.load(f)  # list[str]
        else:
            # 未指定 split → 扫描文件夹下的全部 .npz 文件
            scan_dir = os.path.join(all_data_path, feature_folder_for_scan, class_folder)
            if not os.path.isdir(scan_dir):
                print(f"[parse_input] ⚠️ 目录不存在，跳过: {scan_dir}")
                continue
            sample_names = sorted([
                f[:-4] for f in os.listdir(scan_dir)
                if f.endswith(".npz")
            ])

        for sample_name in sample_names: 
            # 验证所有特征文件是否实际存在于磁盘
            missing_reason = ""
            for folder_name in data_folder_names:
                if "label" in folder_name:
                    continue
                feat_path = os.path.join(
                    all_data_path, folder_name, class_folder, sample_name + ".npz"
                )
                if not os.path.exists(feat_path):
                    missing_reason = f"特征文件不存在: {feat_path}"
                    break

            if missing_reason:
                print(f"[parse_input] ⚠️ split 中样本 '{class_folder}/{sample_name}' "
                      f"在磁盘上未找到，跳过 ({missing_reason})")
                skipped_samples.append((class_folder, sample_name, missing_reason))
                continue

            # 检查标签是否存在
            has_label = False
            if label_folder is not None:
                label_path = os.path.join(
                    all_data_path, label_folder, class_folder, sample_name + ".npz"
                )
                has_label = os.path.exists(label_path)

            entries.append({
                "all_data_path":     all_data_path,
                "class_folder":      class_folder,
                "data_folder_names": data_folder_names,
                "sample_name":       sample_name,
                "has_label":         has_label,
            })

    # 打印统计
    with_label = sum(1 for e in entries if e["has_label"])
    print(f"[parse_input] 共找到 {len(entries)} 条数据条目 "
          f"({with_label} 个含标签)")
    # 各类别分布
    from collections import Counter
    cls_counts = Counter(e["class_folder"] for e in entries)
    for cls_name, count in cls_counts.items():
        print(f"  {cls_name}: {count} 条")

    # 打印跳过的样本汇总
    if skipped_samples:
        print(f"[parse_input] ⚠️ 共跳过 {len(skipped_samples)} 个样本:")
        for cls, name, reason in skipped_samples:
            print(f"  跳过: {cls}/{name} — {reason}")

    return entries


















# =============================================================================
# 一. 对于BOX
# =============================================================================
# 单个BOX的推断+(提供label时)评估
def run_single_pipeline(
    model: torch.nn.Module,
    device: torch.device,
    all_data_path: str,
    sample_root_path: str,
    class_folder: str,
    sample_name: str,
    data_folder_names: list,
    class_mapping: list = None,
    stride: int = 32,
    windows_size: int = 48,
    batch_size: int = 1,
    threshold: float = 0.5,
    core_offset: int = 2,
    output_dir: str = None,
    show_progress: bool = True,
) -> dict:
    """
    单样本推断流水线: parse_input → get_pred → postprocess → (可选)evaluator。

    Args:
        - model:             nn.Module,    已加载的 backbone
        - device:            torch.device, 推断设备
        - all_data_path:     str,          数据根目录
        - sample_root_path:  str,          结构缓存根目录, 用于几何 hardmask 生成
        - class_folder:      str,          类别文件夹
        - sample_name:       str,          样本名
        - data_folder_names: list[str],    数据文件夹列表
        - class_mapping:     list[int] | None, 标签类别映射
        - stride:            int,          滑窗步幅
        - windows_size:      int,          滑窗边长
        - batch_size:        int,          推断 batch size
        - threshold:         float,        语义分割阈值
        - output_dir:        str | None,   输出目录（保存概率图等）; None 则不保存
        - show_progress:     bool,         是否显示进度条

    Returns:
        - result: dict, 包含：
            - "sample_name":   str
            - "class_folder":  str
            - "pred_prob":     np.ndarray, (D,H,W), 概率图
            - "pred_label":    np.ndarray, (D,H,W), 预测标签
            - "hardmask":      np.ndarray, (D,H,W), 硬掩膜
            - "metrics":       dict | None, 评估指标（有 GT 时）
            - "error":         str | None, 错误信息（推断失败时）
    """
    result = {
        "sample_name":  sample_name,
        "class_folder": class_folder,
        "metrics":      None,
        "error":        None,
    }

    try:
        # ---- 1. 加载数据 ----
        data = load_from_npz_dirs(
            all_data_path=all_data_path,
            class_folder=class_folder,
            sample_name=sample_name,
            sample_root_path=sample_root_path,
            data_folder_names=data_folder_names,
            class_mapping=class_mapping,
        )
        grid     = data["grid"]       # (C, D, H, W)
        hardmask = data["hardmask"]   # (D, H, W)
        label    = data.get("label")  # (D, H, W) or None

        # ---- 2. 推断 ----
        pred_prob = run_inference(
            model=model, device=device, show_progress=show_progress, grid=grid,
            stride=stride,
            windows_size=windows_size,
            batch_size=batch_size,
            core_offset=core_offset,
        )
        # np.ndarray, float32, (D, H, W)

        # ---- 3. 后处理 (直接进行二值化) ----
        pred_label = (pred_prob >= threshold).astype(np.int64)
        if hardmask is not None:
             pred_label = pred_label * hardmask
        # np.ndarray, int64, (D, H, W)

        result["pred_prob"]  = pred_prob
        result["pred_label"] = pred_label
        result["hardmask"]   = hardmask

        # ---- 4. 评估 (若有 GT) ----
        if label is not None:
            metrics = voxel_evaluate(
                pred_label, label, hardmask=hardmask,
            )
            result["metrics"] = metrics

        # ---- 5. 保存结果 ----
        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)
            # 保存概率图
            np.savez_compressed(
                os.path.join(output_dir, "pred_prob.npz"),
                pred_prob=pred_prob,
            )
            # 保存预测标签
            np.savez_compressed(
                os.path.join(output_dir, "pred_label.npz"),
                pred_label=pred_label,
            )
    except Exception as e:
        result["error"] = str(e)
        print(f"  ❌ 推断失败: {e}")
    return result

# 批量 BOX 的推断 + (提供label时)评估
def run_batch(
    model: torch.nn.Module,
    device: torch.device,
    entries: list,
    infer_params: dict,
    sample_root_path: str,
    class_mapping: list = None,
    output_root: str = None,
    save_results: bool = True,
) -> list:
    """
    批量推断 + 可选评估。对所有样本逐一执行 run_single_pipeline()。

    Args:
        - model:         nn.Module,    已加载的 backbone
        - device:        torch.device
        - entries:       list[dict],   prepare_data_entries() 的返回值, 位于 Pocket\src\inference\parse_input.py
        - infer_params:  dict,         推断参数 (stride / windows_size / batch_size / threshold)
        - sample_root_path: str,       结构缓存根目录, 用于几何 hardmask 生成
        - class_mapping: list[int] | None
        - output_root:   str | None,   输出根目录; 每个样本在此下建子目录
        - save_results:  bool,         是否保存每个样本的结果到磁盘

    Returns:
        - all_results: list[dict], 每项与 run_single_pipeline() 返回值相同, 额外添加 "num_pred_pos" (预测正类体素数) 字段
    """
    all_results = []

    for i, entry in enumerate(entries):
        name = entry["sample_name"]
        cls = entry["class_folder"]
        print(f"\n[Batch {i+1}/{len(entries)}] === {cls}/{name} ===")

        sample_output_dir = None
        if output_root and save_results:
            sample_output_dir = os.path.join(output_root, cls, name)

        result = run_single_pipeline(
            model=model,
            device=device,
            all_data_path=entry["all_data_path"],
            sample_root_path=sample_root_path,
            class_folder=cls,
            sample_name=name,
            data_folder_names=entry["data_folder_names"],
            class_mapping=class_mapping,
            output_dir=sample_output_dir,
            **infer_params,
        )

        # 添加统计信息
        if result["error"] is None:
            result["num_pred_pos"] = int(np.sum(result.get("pred_label", 0) > 0))

            if result["metrics"] is not None:
                m = result["metrics"]
                print(f"  Precision={m['precision']:.4f}  Recall={m['recall']:.4f}  "
                      f"F1={m['f1']:.4f}  IoU={m['iou']:.4f}  "
                      f"预测正类={result['num_pred_pos']}")
            else:
                print(f"  推断完成 (无 GT, 不评估), 预测正类体素数={result['num_pred_pos']}")
        else:
            result["num_pred_pos"] = 0

        # 释放大数组节省内存 (结果已保存到磁盘)
        for key in ["pred_prob", "pred_label", "hardmask"]:
            result.pop(key, None)

        all_results.append(result)

    return all_results

# 关于BOX的网格调参搜索
def run_param_search(
    model: torch.nn.Module,
    device: torch.device,
    entries: list,
    param_sweep_cfg: list,
    base_params: dict,
    sample_root_path: str,
    class_mapping: list = None,
    output_root: str = "param_search_output",
) -> None:
    """
    网格调参搜索: 对每组参数执行批量推断, 以 avg_F1 为优化目标。

    Args:
        - model:           nn.Module
        - device:          torch.device
        - entries:         list[dict], prepare_data_entries() 返回值, 位于 Pocket\src\inference\parse_input.py
        - param_sweep_cfg: list[dict], sweep 配置
        - base_params:     dict, 基准推断参数. param_sweep_cfg没有参数值时将作为默认值
        - sample_root_path: str, 结构缓存根目录, 用于几何 hardmask 生成
        - class_mapping:   list[int] | None
        - output_root:     str, Excel 写出目录
    """
    param_grid = generate_param_grid(param_sweep_cfg)
    param_names = list(param_grid[0].keys()) if param_grid else []
    summary = []

    # 优化 param_search：判断是否只搜 `threshold`, 如果仅对 threshold 进行后处理参数搜索，则无需每次重新做 GPU 耗时滑窗推断，复用概率图。
    only_sweep_threshold = all(name == "threshold" for name in param_names)

    if only_sweep_threshold:
        print("\n[ParamSearch] 💡 检测到仅搜索 threshold，启用推断缓存加速！")
        cached_probs = []  # list[dict]
        print("[ParamSearch] --- 阶段 1: 全量推断缓存 ---")
        for i, entry in enumerate(entries):
            name = entry["sample_name"]
            cls = entry["class_folder"]
            # 用 base_params 拿一次推断全图
            from src.inference.get_pred import run_inference

            try:
                data = load_from_npz_dirs(
                    all_data_path=entry["all_data_path"],
                    class_folder=cls,
                    sample_name=name,
                    sample_root_path=sample_root_path,
                    data_folder_names=entry["data_folder_names"],
                    class_mapping=class_mapping,
                )
                pred_prob = run_inference(
                    model=model, device=device, show_progress=False, grid=data["grid"],
                    stride=base_params["stride"],
                    windows_size=base_params["windows_size"],
                    batch_size=base_params["batch_size"],
                    core_offset=base_params["core_offset"],
                )
                cached_probs.append({
                    "entry": entry,
                    "pred_prob": pred_prob,
                    "hardmask": data["hardmask"],
                    "label": data.get("label"),
                })
            except Exception as e:
                print(f"  ❌ 推断失败 {cls}/{name}: {e}")
                cached_probs.append({"error": str(e), "entry": entry})
        
        print("\n[ParamSearch] --- 阶段 2: 阈值组合评估 ---")
        for idx, override_params in enumerate(param_grid):
            eval_results = []
            th = override_params["threshold"]
            print(f"[{idx+1}/{len(param_grid)}] 组合 threshold={th:.4f}")

            for cache in cached_probs:
                if "error" in cache: continue
                if cache["label"] is None: continue

                pred_label = (cache["pred_prob"] >= th).astype(np.int64)
                if cache.get("hardmask") is not None:
                    pred_label = pred_label * cache["hardmask"]
                metrics = voxel_evaluate(pred_label, cache["label"], hardmask=cache["hardmask"])
                eval_results.append(metrics)
            
            if eval_results:
                avg_p    = float(np.mean([r["precision"] for r in eval_results]))
                avg_r    = float(np.mean([r["recall"]    for r in eval_results]))
                avg_f1   = float(np.mean([r["f1"]        for r in eval_results]))
                avg_iou  = float(np.mean([r["iou"]       for r in eval_results]))
            else:
                avg_p = avg_r = avg_f1 = avg_iou = 0.0

            row = {
                **override_params,
                "avg_P": avg_p, "avg_R": avg_r,
                "avg_F1": avg_f1, "avg_IoU": avg_iou,
            }
            summary.append(row)
            print(f"  → avg_P={avg_p:.4f}  avg_R={avg_r:.4f}  "
                  f"avg_F1={avg_f1:.4f}  avg_IoU={avg_iou:.4f}")

    else:
        # 不仅有threshold这个参数，回退为全量重算
        for idx, override_params in enumerate(param_grid):
            infer_params = {**base_params, **override_params}
            param_str = "  ".join(f"{k}={v}" for k, v in override_params.items())
            print(f"\n{'=' * 60}")
            print(f"[ParamSearch {idx+1}/{len(param_grid)}] 参数组合: {param_str}")

            results = run_batch(
                model, device, entries,
                infer_params=infer_params,
                sample_root_path=sample_root_path,
                class_mapping=class_mapping,
                output_root=None,   # 调参时不保存到磁盘
                save_results=False,
            )

            eval_results = [r["metrics"] for r in results if r.get("metrics") is not None]
            
            if eval_results:
                avg_p    = float(np.mean([r["precision"] for r in eval_results]))
                avg_r    = float(np.mean([r["recall"]    for r in eval_results]))
                avg_f1   = float(np.mean([r["f1"]        for r in eval_results]))
                avg_iou  = float(np.mean([r["iou"]       for r in eval_results]))
            else:
                avg_p = avg_r = avg_f1 = avg_iou = 0.0

            row = {
                **override_params,
                "avg_P": avg_p, "avg_R": avg_r,
                "avg_F1": avg_f1, "avg_IoU": avg_iou,
            }
            summary.append(row)
            print(f"  → avg_P={avg_p:.4f}  avg_R={avg_r:.4f}  "
                  f"avg_F1={avg_f1:.4f}  avg_IoU={avg_iou:.4f}")

    # 按 avg_F1 降序
    summary.sort(key=lambda r: r["avg_F1"], reverse=True)
    best = summary[0]
    # 写 Excel
    _write_param_search_excel = write_param_search_excel  # 别名，兼容历史调用
    _write_param_search_excel(summary, param_names, output_root)
    print("\n" + "=" * 60)
    print("  [ParamSearch] 🏆 最优参数组合:")
    for k, v in best.items():
        print(f"    {k}: {round(v, 4) if isinstance(v, float) else v}")
    print("=" * 60)





















# ----------------------------------------------- BOX -----------------------------------------------
def run_single_mode(cfg_dict, model, device, all_data_path,
                     data_folder_names, class_mapping, infer_params, output_root):
    """mode=single: 单样本推断 + 可选评估"""
    class_folder = cfg_dict.get("class_folder", None)
    sample_name  = cfg_dict.get("sample_name", None)
    sample_root_path = cfg_dict.get("sample_root_path", None)
    if not class_folder or not sample_name:
        raise ValueError(
            "[错误] mode=single 需要指定 class_folder 和 sample_name。"
            "用法: +class_folder=\"small_molecule\" +sample_name=\"9f3f_0_0_0_0_C\""
        )
    result = run_single_pipeline(
        model=model,
        device=device,
        all_data_path=all_data_path,
        sample_root_path=sample_root_path,
        class_folder=class_folder,
        sample_name=sample_name,
        data_folder_names=data_folder_names,
        class_mapping=class_mapping,
        output_dir=os.path.join(output_root, class_folder, sample_name),
        **infer_params,
    )
    if result["error"]:
        print(f"\n❌ 推断失败: {result['error']}")
        return
    # 打印概率图统计
    pred_prob = result["pred_prob"]
    print(f"\n[Single] 概率图统计:")
    print(f"  shape = {pred_prob.shape}")
    print(f"  min={pred_prob.min():.6f}  max={pred_prob.max():.6f}  "
          f"mean={pred_prob.mean():.6f}")
    for q in [0.5, 0.9, 0.95, 0.99]:
        print(f"  {q*100:.0f}% 分位数 = {np.quantile(pred_prob, q):.6f}")

    if result["metrics"]:
        print(f"\n[Single] 评估结果:")
        print_metrics(result["metrics"])
    else:
        print(f"\n[Single] 无 GT 标签, 跳过评估")
    print(f"\n✅ 单样本推断完毕!")

def run_batch_mode(cfg_dict, model, device, all_data_path,
                    data_folder_names, class_folder_names, class_mapping,
                    infer_params, output_root):
    """mode=batch: 批量推断 + 可选评估 + Excel 汇总"""
    sample_root_path = cfg_dict.get("sample_root_path", None)
    split_files = cfg_dict.get("split_files", None)
    if split_files is not None and not isinstance(split_files, list):
        split_files = list(split_files)

    entries = prepare_data_entries(
        all_data_path=all_data_path,
        class_folder_names=class_folder_names,
        split_files=split_files,
        data_folder_names=data_folder_names,
    )
    if not entries:
        raise RuntimeError("[错误] 没有找到任何数据条目")

    results = run_batch(
        model, device, entries,
        infer_params=infer_params,
        sample_root_path=sample_root_path,
        class_mapping=class_mapping,
        output_root=output_root,
        save_results=True,
    )

    write_batch_excel(results, output_root)
    print("\n🎉 批量推断完毕！\n")

def run_param_search_mode(cfg_dict, model, device, all_data_path,
                           data_folder_names, class_folder_names, class_mapping,
                           base_params, output_root):
    """mode=param_search: 网格调参搜索"""
    sample_root_path = cfg_dict.get("sample_root_path", None)
    split_files = cfg_dict.get("split_files", None)
    if split_files is not None and not isinstance(split_files, list):
        split_files = list(split_files)

    param_sweep_cfg = cfg_dict.get("param_sweep", None)
    if not param_sweep_cfg:
        raise ValueError(
            "[错误] mode=param_search 需要配置 param_sweep 节。"
        )

    entries = prepare_data_entries(
        all_data_path=all_data_path,
        class_folder_names=class_folder_names,
        split_files=split_files,
        data_folder_names=data_folder_names,
    )
    if not entries:
        raise RuntimeError("[错误] 没有找到任何数据条目")

    run_param_search(
        model, device, entries,
        param_sweep_cfg=param_sweep_cfg,
        base_params=base_params,
        sample_root_path=sample_root_path,
        class_mapping=class_mapping,
        output_root=output_root,
    )
    print("\n🎉 网格调参搜索完毕！\n")

