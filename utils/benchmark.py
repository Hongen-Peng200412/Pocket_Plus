import os
import numpy as np


SAMPLE_ROOT_PATH = "/home/penghongen/My_Project/Data/DATA_v1/parsed_pdb"
CLASS_MAPPING = [0,1,0,0,1]
def fetch_atom_gt(pdb_id: str):
    """ 返回{pdb_id}的所有正类原子坐标 """
    label_npz_path = os.path.join(SAMPLE_ROOT_PATH, pdb_id.lower(), "labels.npz")
    atoms_npz_path = os.path.join(SAMPLE_ROOT_PATH, pdb_id.lower(), "atoms.npz")
    pocket_class_ids = np.load(label_npz_path)["pocket_class_id"]  # (N_atoms, )
    atom_coords = np.load(atoms_npz_path)["coords"]               # (N_atoms, 3)

    mapped_ids = np.zeros_like(pocket_class_ids)
    for old_id, new_id in enumerate(CLASS_MAPPING):
        mapped_ids[pocket_class_ids == old_id] = new_id

    atom_gt = atom_coords[mapped_ids > 0]

    return atom_gt














def _compute_hit_mask(
    src_points: np.ndarray,
    tgt_points: np.ndarray,
    dist_threshold: float,
    chunk_size: int = 2048,
) -> np.ndarray:
    """
    计算 src_points 中每个点是否被 tgt_points 命中（存在任意 tgt_points 距离 <= dist_threshold）。

    输入参数:
        - src_points: np.ndarray, (N_src, 3), 源点云 (世界坐标, 单位 Å)
        - tgt_points: np.ndarray, (N_tgt, 3), 目标点云 (世界坐标, 单位 Å)
        - dist_threshold: float, 距离阈值 (Å)
        - chunk_size: int, 分块大小, 避免一次性占用过大内存

    输出:
        - hit_mask: np.ndarray, (N_src,), bool, 每个源点是否命中
    """
    if src_points.size == 0 or tgt_points.size == 0:
        return np.zeros((src_points.shape[0],), dtype=bool)

    # float, 命中阈值的平方
    dist2_threshold = float(dist_threshold) ** 2
    # np.ndarray, (N_src,), bool, 每个源点是否命中
    hit_mask = np.zeros((src_points.shape[0],), dtype=bool)

    for start in range(0, src_points.shape[0], chunk_size):
        end = start + chunk_size
        sub_points = src_points[start:end]
        # np.ndarray, (M, N_tgt, 3), 点对差值
        diff = sub_points[:, None, :] - tgt_points[None, :, :]
        # np.ndarray, (M, N_tgt), 平方距离
        dist2 = np.sum(diff * diff, axis=2)
        hit_mask[start:end] = np.any(dist2 <= dist2_threshold, axis=1)

    return hit_mask


def _count_hits(
    src_points: np.ndarray,
    tgt_points: np.ndarray,
    dist_threshold: float,
    chunk_size: int = 2048,
) -> int:
    """
    统计 src_points 中被命中的点数（存在任意 tgt_points 距离 <= dist_threshold）。

    输入参数:
        - src_points: np.ndarray, (N_src, 3), 源点云 (世界坐标, 单位 Å)
        - tgt_points: np.ndarray, (N_tgt, 3), 目标点云 (世界坐标, 单位 Å)
        - dist_threshold: float, 距离阈值 (Å)
        - chunk_size: int, 分块大小, 避免一次性占用过大内存

    输出:
        - hit_count: int, 命中点数量
    """
    return int(np.sum(_compute_hit_mask(src_points, tgt_points, dist_threshold, chunk_size)))







# NOTE: 注意，这是需要调用的最终函数. 跑两次, dist_threshold 依次取 0.1, 3.0
def semantic_evaluate(
    pred_atom_coords: np.ndarray,
    pdb_id: str, 
    dist_threshold: float,
) -> dict:
    """
    基于点云进行语义分割评估（预测正类原子坐标 vs GT 正类原子坐标 → 距离命中）。

    输入参数:
        - pred_atom_coords:  np.ndarray, float32, (N_pred, 3), 预测为正类的原子坐标. 由 postprocess 模块产出. 世界坐标, 顺序为xyz, 单位 Å
        - pdb_id:            大写小写都可
        - dist_threshold:    float, 评估的距离阈值 (Å): 预测正类与真实正类 hit ↔ 距离 <= dist_threshold

    输出:
        - metrics: dict, 包含:
            - "precision":   float, 预测命中率
            - "recall":      float, 真实命中率
            - "f1":          float, F1 分数
            - "iou":         float, IoU 分数（基于 tp/fp/fn 计算）
            - "num_pred":    int,   预测正类原子数
            - "num_gt":      int,   真实正类原子数
            - "hit_pred":    int,   被命中的预测原子数
            - "hit_gt":      int,   被命中的真实原子数

    逻辑:
        - 根据 dist_threshold, 以及 (预测正类点云, 真实正类点云) 确定 hit 的原子数, 进而算各类指标
    """
    # np.ndarray, float32, (N_pred, 3), 预测正类点云
    pred_points = np.asarray(pred_atom_coords, dtype=np.float32).reshape(-1, 3)
    # np.ndarray, float32, (N_gt, 3), GT 正类点云
    gt_points = fetch_atom_gt(pdb_id)

    # int, 预测正类原子数
    num_pred = int(pred_points.shape[0])
    # int, GT 正类原子数
    num_gt = int(gt_points.shape[0])

    # 特殊情况: 没有任何正类且模型也没预测出正类 → 视为完美
    if num_pred == 0 and num_gt == 0:
        return {
            "mode": "point",
            "precision": 1.0,
            "recall": 1.0,
            "f1": 1.0,
            "iou": 1.0,
            "num_pred": 0,
            "num_gt": 0,
            "hit_pred": 0,
            "hit_gt": 0,
        }

    # int, 被命中的预测原子数
    hit_pred = _count_hits(pred_points, gt_points, dist_threshold)
    # int, 被命中的真实原子数
    hit_gt = _count_hits(gt_points, pred_points, dist_threshold)

    precision = hit_pred / num_pred if num_pred > 0 else 0.0
    recall = hit_gt / num_gt if num_gt > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)

    # 以 hit_pred/ hit_gt 构造 tp / fp / fn
    tp = hit_pred
    fp = num_pred - hit_pred
    fn = num_gt - hit_gt
    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0

    return {
        "mode": "point",
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "iou": float(iou),
        "num_pred": num_pred,
        "num_gt": num_gt,
        "hit_pred": int(hit_pred),
        "hit_gt": int(hit_gt),
    }

