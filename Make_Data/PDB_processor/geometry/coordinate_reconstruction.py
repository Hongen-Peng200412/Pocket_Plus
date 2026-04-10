"""
原子坐标重建模块

基于理想几何和相邻残基插值重建缺失的主链原子坐标
"""

import numpy as np
from typing import Optional


def reconstruct_protein_backbone(
    residues: list[dict],
    res_idx: int,
    missing_atoms: list[str]
) -> dict[str, np.ndarray]:
    """
    重建蛋白质残基的缺失主链原子坐标。

    输入参数:
        - residues: list[dict], 可变长度, 所有残基的字典列表
        - res_idx: int, 标量, 当前残基在列表中的索引
        - missing_atoms: list[str], 可变长度, 缺失的原子名称列表, 如 ['N', 'CA']

    输出:
        - reconstructed: dict[str, np.ndarray], 无固定形状, 键为原子名, 值为 (3,) 坐标数组
            若无法重建, 返回空字典
    """
    # 边界检查: 空列表或越界索引
    if len(residues) == 0 or res_idx < 0 or res_idx >= len(residues):
        return {}

    # dict[str, np.ndarray], 无固定形状, 存储重建的原子坐标
    reconstructed = {}

    # dict, 无固定形状, 当前残基数据
    res = residues[res_idx]

    # 理想几何参数 (Engh & Huber, 1991)
    # float, 标量, CA-C 键长 (Angstrom)
    CA_C_BOND = 1.52
    # float, 标量, C-N 键长 (Angstrom)
    C_N_BOND = 1.33
    # float, 标量, N-CA 键长 (Angstrom)
    N_CA_BOND = 1.46

    # 尝试重建 CA (使用前后残基线性插值)
    if 'CA' in missing_atoms:
        # Optional[np.ndarray], (3,) 或 None, 前一残基 CA 坐标
        prev_ca = _get_prev_ca(residues, res_idx)
        # Optional[np.ndarray], (3,) 或 None, 后一残基 CA 坐标
        next_ca = _get_next_ca(residues, res_idx)

        if prev_ca is not None and next_ca is not None:
            # np.ndarray, (3,), 插值得到的 CA 坐标 (强制 asarray 以兼容 Python list 输入)
            reconstructed['CA'] = (np.asarray(prev_ca, dtype=np.float64) + np.asarray(next_ca, dtype=np.float64)) / 2.0
        elif prev_ca is not None:
            # np.ndarray, (3,), 基于前一残基外推的 CA 坐标
            reconstructed['CA'] = np.asarray(prev_ca, dtype=np.float64) + np.array([3.8, 0.0, 0.0])
        elif next_ca is not None:
            # np.ndarray, (3,), 基于后一残基外推的 CA 坐标
            reconstructed['CA'] = np.asarray(next_ca, dtype=np.float64) - np.array([3.8, 0.0, 0.0])

    # np.ndarray 或 None, (3,) 或 None, 当前 CA 坐标 (原始或重建)
    ca_raw = reconstructed.get('CA', res.get('ca_coord'))
    ca_coord = np.asarray(ca_raw, dtype=np.float64) if ca_raw is not None else None

    # 尝试重建 N (基于 CA 和理想几何)
    if 'N' in missing_atoms and ca_coord is not None:
        # Optional[np.ndarray], (3,) 或 None, 前一残基 C 坐标
        prev_c = _get_prev_c(residues, res_idx)

        if prev_c is not None:
            # np.ndarray, (3,), 从前一残基 C 到当前 CA 的方向向量
            prev_c_arr = np.asarray(prev_c, dtype=np.float64)
            direction = ca_coord - prev_c_arr
            # float, 标量, 方向向量的模长
            dist = np.linalg.norm(direction)

            if dist > 1e-6:
                # np.ndarray, (3,), 归一化方向向量
                direction = direction / dist
                # np.ndarray, (3,), 重建的 N 坐标 (位于 C 和 CA 之间)
                reconstructed['N'] = prev_c_arr + direction * C_N_BOND
        else:
            # np.ndarray, (3,), 基于 CA 的默认 N 位置
            reconstructed['N'] = ca_coord - np.array([N_CA_BOND, 0.0, 0.0])

    # 尝试重建 C (基于 CA 和理想几何)
    if 'C' in missing_atoms and ca_coord is not None:
        # Optional[np.ndarray], (3,) 或 None, 后一残基 N 坐标
        next_n = _get_next_n(residues, res_idx)

        if next_n is not None:
            # np.ndarray, (3,), 从当前 CA 到下一残基 N 的方向向量
            next_n_arr = np.asarray(next_n, dtype=np.float64)
            direction = next_n_arr - ca_coord
            # float, 标量, 方向向量的模长
            dist = np.linalg.norm(direction)

            if dist > 1e-6:
                # np.ndarray, (3,), 归一化方向向量
                direction = direction / dist
                # np.ndarray, (3,), 重建的 C 坐标 (位于 CA 和下一残基 N 之间)
                reconstructed['C'] = ca_coord + direction * CA_C_BOND
        else:
            # np.ndarray, (3,), 基于 CA 的默认 C 位置
            reconstructed['C'] = ca_coord + np.array([CA_C_BOND, 0.0, 0.0])

    return reconstructed


def reconstruct_nucleotide_backbone(
    residues: list[dict],
    res_idx: int,
    missing_atoms: list[str]
) -> dict[str, np.ndarray]:
    """
    重建核酸残基的缺失主链原子坐标。

    输入参数:
        - residues: list[dict], 可变长度, 所有残基的字典列表
        - res_idx: int, 标量, 当前残基在列表中的索引
        - missing_atoms: list[str], 可变长度, 缺失的原子名称列表, 如 ["C4'", "C1'"]

    输出:
        - reconstructed: dict[str, np.ndarray], 无固定形状, 键为原子名, 值为 (3,) 坐标数组
            若无法重建, 返回空字典
    """
    # 边界检查: 空列表或越界索引
    if len(residues) == 0 or res_idx < 0 or res_idx >= len(residues):
        return {}

    # dict[str, np.ndarray], 无固定形状, 存储重建的原子坐标
    reconstructed = {}

    # dict, 无固定形状, 当前残基数据
    res = residues[res_idx]

    # 理想几何参数
    # float, 标量, C4'-C1' 距离 (Angstrom)
    C4_C1_DIST = 2.5
    # float, 标量, C1'-N1/N9 距离 (Angstrom)
    C1_N_DIST = 1.48
    # float, 标量, 核酸主链间距 (Angstrom)
    BACKBONE_SPACING = 5.9

    # 尝试重建 C4' (使用前后核苷酸线性插值)
    if "C4'" in missing_atoms:
        # Optional[np.ndarray], (3,) 或 None, 前一核苷酸 C4' 坐标
        prev_c4 = _get_prev_c4(residues, res_idx)
        # Optional[np.ndarray], (3,) 或 None, 后一核苷酸 C4' 坐标
        next_c4 = _get_next_c4(residues, res_idx)

        if prev_c4 is not None and next_c4 is not None:
            # np.ndarray, (3,), 插值得到的 C4' 坐标 (强制 asarray 以兼容 Python list 输入)
            reconstructed["C4'"] = (np.asarray(prev_c4, dtype=np.float64) + np.asarray(next_c4, dtype=np.float64)) / 2.0
        elif prev_c4 is not None:
            # np.ndarray, (3,), 基于前一核苷酸外推的 C4' 坐标
            reconstructed["C4'"] = np.asarray(prev_c4, dtype=np.float64) + np.array([BACKBONE_SPACING, 0.0, 0.0])
        elif next_c4 is not None:
            # np.ndarray, (3,), 基于后一核苷酸外推的 C4' 坐标
            reconstructed["C4'"] = np.asarray(next_c4, dtype=np.float64) - np.array([BACKBONE_SPACING, 0.0, 0.0])

    # np.ndarray 或 None, (3,) 或 None, 当前 C4' 坐标 (原始或重建)
    c4_raw = reconstructed.get("C4'", res.get("c4p_coord"))
    c4_coord = np.asarray(c4_raw, dtype=np.float64) if c4_raw is not None else None

    # 尝试重建 C1' (基于 C4' 和糖环几何)
    if "C1'" in missing_atoms and c4_coord is not None:
        # np.ndarray, (3,), 基于 C4' 的默认 C1' 位置
        reconstructed["C1'"] = c4_coord + np.array([0.0, C4_C1_DIST, 0.0])

    # np.ndarray 或 None, (3,) 或 None, 当前 C1' 坐标 (原始或重建)
    c1_raw = reconstructed.get("C1'", res.get("c1p_coord"))
    c1_coord = np.asarray(c1_raw, dtype=np.float64) if c1_raw is not None else None

    # 尝试重建 N1/N9 (基于 C1' 和碱基类型)
    if ("N1" in missing_atoms or "N9" in missing_atoms) and c1_coord is not None:
        # str, 标量, 残基名称
        res_name = res.get('name', '')

        if res_name in {'A', 'G', 'DA', 'DG'}:
            # 嘌呤使用 N9
            # np.ndarray, (3,), 重建的 N9 坐标
            reconstructed["N9"] = c1_coord + np.array([0.0, 0.0, C1_N_DIST])
        elif res_name in {'U', 'C', 'T', 'DT', 'DC'}:
            # 嘧啶使用 N1
            # np.ndarray, (3,), 重建的 N1 坐标
            reconstructed["N1"] = c1_coord + np.array([0.0, 0.0, C1_N_DIST])

    return reconstructed


def _get_prev_ca(residues: list[dict], res_idx: int) -> Optional[np.ndarray]:
    """
    获取前一残基的 CA 坐标。

    输入参数:
        - residues: list[dict], 可变长度, 所有残基的字典列表
        - res_idx: int, 标量, 当前残基在列表中的索引

    输出:
        - ca_coord: np.ndarray 或 None, (3,) 或 None, 前一残基 CA 坐标
    """
    if res_idx > 0:
        # dict, 无固定形状, 前一残基数据
        prev_res = residues[res_idx - 1]
        if prev_res.get('type') == 'protein':
            return prev_res.get('ca_coord')
    return None


def _get_next_ca(residues: list[dict], res_idx: int) -> Optional[np.ndarray]:
    """
    获取后一残基的 CA 坐标。

    输入参数:
        - residues: list[dict], 可变长度, 所有残基的字典列表
        - res_idx: int, 标量, 当前残基在列表中的索引

    输出:
        - ca_coord: np.ndarray 或 None, (3,) 或 None, 后一残基 CA 坐标
    """
    if res_idx < len(residues) - 1:
        # dict, 无固定形状, 后一残基数据
        next_res = residues[res_idx + 1]
        if next_res.get('type') == 'protein':
            return next_res.get('ca_coord')
    return None


def _get_prev_c(residues: list[dict], res_idx: int) -> Optional[np.ndarray]:
    """
    获取前一残基的 C 坐标。

    输入参数:
        - residues: list[dict], 可变长度, 所有残基的字典列表
        - res_idx: int, 标量, 当前残基在列表中的索引

    输出:
        - c_coord: np.ndarray 或 None, (3,) 或 None, 前一残基 C 坐标
    """
    if res_idx > 0:
        # dict, 无固定形状, 前一残基数据
        prev_res = residues[res_idx - 1]
        if prev_res.get('type') == 'protein':
            return prev_res.get('c_coord')
    return None


def _get_next_n(residues: list[dict], res_idx: int) -> Optional[np.ndarray]:
    """
    获取后一残基的 N 坐标。

    输入参数:
        - residues: list[dict], 可变长度, 所有残基的字典列表
        - res_idx: int, 标量, 当前残基在列表中的索引

    输出:
        - n_coord: np.ndarray 或 None, (3,) 或 None, 后一残基 N 坐标
    """
    if res_idx < len(residues) - 1:
        # dict, 无固定形状, 后一残基数据
        next_res = residues[res_idx + 1]
        if next_res.get('type') == 'protein':
            return next_res.get('n_coord')
    return None


def _get_prev_c4(residues: list[dict], res_idx: int) -> Optional[np.ndarray]:
    """
    获取前一核苷酸的 C4' 坐标。

    输入参数:
        - residues: list[dict], 可变长度, 所有残基的字典列表
        - res_idx: int, 标量, 当前残基在列表中的索引

    输出:
        - c4_coord: np.ndarray 或 None, (3,) 或 None, 前一核苷酸 C4' 坐标
    """
    if res_idx > 0:
        # dict, 无固定形状, 前一残基数据
        prev_res = residues[res_idx - 1]
        if prev_res.get('type') == 'nucleotide':
            return prev_res.get("c4p_coord")
    return None


def _get_next_c4(residues: list[dict], res_idx: int) -> Optional[np.ndarray]:
    """
    获取后一核苷酸的 C4' 坐标。

    输入参数:
        - residues: list[dict], 可变长度, 所有残基的字典列表
        - res_idx: int, 标量, 当前残基在列表中的索引

    输出:
        - c4_coord: np.ndarray 或 None, (3,) 或 None, 后一核苷酸 C4' 坐标
    """
    if res_idx < len(residues) - 1:
        # dict, 无固定形状, 后一残基数据
        next_res = residues[res_idx + 1]
        if next_res.get('type') == 'nucleotide':
            return next_res.get("c4p_coord")
    return None
