from __future__ import annotations

from pathlib import Path

import numpy as np


def mol2_heavy_atom_xyz(path: Path) -> np.ndarray:
    """
    读取 mol2 重原子坐标。
    输入参数:
        - path: Path, mol2 文件路径

    输出:
        - coords: np.ndarray, (N, 3), mol2 ATOM block 中非氢原子坐标
    """
    coords: list[list[float]] = []
    in_atom_block = False
    for line in path.read_text(errors="replace").splitlines():
        if line.startswith("@<TRIPOS>ATOM"):
            in_atom_block = True
            continue
        if line.startswith("@<TRIPOS>") and in_atom_block:
            break
        if in_atom_block and line.strip():
            parts = line.split()
            if not parts[5].upper().startswith("H"):
                coords.append([float(parts[2]), float(parts[3]), float(parts[4])])
    return np.asarray(coords, dtype=float)


def instance_voxel_xyz(label: np.ndarray, instance_id: int, origin: np.ndarray, voxel_size: np.ndarray) -> np.ndarray:
    """
    将 instance label 中的一个 instance 转成近似世界坐标点云。
    输入参数:
        - label: np.ndarray, (D, H, W), int, instance 编号图
        - instance_id: int, 要提取的 instance 编号
        - origin: np.ndarray, (3,), cache origin
        - voxel_size: np.ndarray, (3,), cache voxel_size

    输出:
        - coords: np.ndarray, (N, 3), xyz 顺序的体素中心近似坐标
    """
    zyx = np.argwhere(label == instance_id)
    return np.stack(
        [
            zyx[:, 2] * voxel_size[2] + origin[2],
            zyx[:, 1] * voxel_size[1] + origin[1],
            zyx[:, 0] * voxel_size[0] + origin[0],
        ],
        axis=1,
    )


def radial_shape_score(pred_xyz: np.ndarray, ligand_xyz: np.ndarray, bins: np.ndarray) -> dict[str, float]:
    """
    计算当前第一版径向 shape score。
    输入参数:
        - pred_xyz: np.ndarray, (N, 3), 网络预测 instance 的点云
        - ligand_xyz: np.ndarray, (M, 3), ligand mol2 或 docking pose 的点云
        - bins: np.ndarray, (K,), 半径直方图 bin 边界

    输出:
        - score: dict[str, float], 包含:
            - "network_shape_score": float, 综合径向形状成本, 越低越好
            - "radial_l1": float, 归一化半径直方图 L1 差异的一半
            - "p90_radius_delta": float, 90% 半径相对差异
            - "pred_r90": float, 预测点云 90% 半径
            - "ligand_r90": float, ligand 点云 90% 半径
    """
    pred_hist, pred_r90 = _radial_histogram(pred_xyz, bins)
    ligand_hist, ligand_r90 = _radial_histogram(ligand_xyz, bins)
    radial_l1 = float(np.abs(pred_hist - ligand_hist).sum() / 2.0)
    p90_radius_delta = float(abs(pred_r90 - ligand_r90) / max(1.0, pred_r90))
    return {
        "network_shape_score": 0.7 * radial_l1 + 0.3 * p90_radius_delta,
        "radial_l1": radial_l1,
        "p90_radius_delta": p90_radius_delta,
        "pred_r90": float(pred_r90),
        "ligand_r90": float(ligand_r90),
    }


def _radial_histogram(points: np.ndarray, bins: np.ndarray) -> tuple[np.ndarray, float]:
    """
    计算点云到自身中心的半径分布。
    输入参数:
        - points: np.ndarray, (N, 3), 输入点云
        - bins: np.ndarray, (K,), 半径 bin 边界

    输出:
        - result: tuple[np.ndarray, float], 包含归一化直方图和 90% 半径
    """
    center = points.mean(axis=0)
    radius = np.linalg.norm(points - center, axis=1)
    hist, _ = np.histogram(radius, bins=bins)
    hist = hist.astype(float)
    return hist / hist.sum(), float(np.percentile(radius, 90))

