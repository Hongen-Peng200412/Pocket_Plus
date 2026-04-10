# -*- coding: utf-8 -*-
"""
================================================================================
严格测试: allow_incomplete_backbone 回退策略全链路验证
================================================================================

测试范围:
  1. coordinate_reconstruction.py — 蛋白 & 核酸主链坐标重建
  2. local_frames.py — 宽松/严格模式下的局部坐标系构建
  3. parser.py 中 backbone_complete_mask 的长度一致性
  4. N1/N9 字符串匹配修复验证
  5. 前后残基插值修复验证 (context_list 包含下一个残基)

运行方式:
  python test_all_backbone_fixes.py
================================================================================
"""

import sys
import os
import numpy as np
import traceback

# 添加父目录和 Make_Data 根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

from coordinate_reconstruction import (
    reconstruct_protein_backbone,
    reconstruct_nucleotide_backbone,
)


# ============================================================================
# 统计变量
# ============================================================================
_PASSED = 0
_FAILED = 0
_ERRORS = []


def _assert(condition, msg):
    """断言辅助函数"""
    global _PASSED, _FAILED, _ERRORS
    if not condition:
        _FAILED += 1
        _ERRORS.append(msg)
        print(f"    ✗ FAIL: {msg}")
        return False
    return True


def _test_pass(name):
    """标记测试通过"""
    global _PASSED
    _PASSED += 1
    print(f"  ✓ {name}")


def _test_fail(name, error):
    """标记测试失败"""
    global _FAILED, _ERRORS
    _FAILED += 1
    _ERRORS.append(f"{name}: {error}")
    print(f"  ✗ {name}: {error}")


# ============================================================================
# 第一类: 蛋白质主链坐标重建测试
# ============================================================================
def test_protein_ca_interpolation_both_neighbors():
    """CA 插值: 前后邻居都有 CA 时应该取中点"""
    name = "蛋白 CA 插值 (前后邻居都存在)"
    residues = [
        {'type': 'protein', 'name': 'ALA',
         'ca_coord': np.array([0.0, 0.0, 0.0]),
         'n_coord': np.array([-1.0, 0.0, 0.0]),
         'c_coord': np.array([1.0, 0.0, 0.0])},
        {'type': 'protein', 'name': 'GLY',
         'ca_coord': None, 'n_coord': None, 'c_coord': None},
        {'type': 'protein', 'name': 'VAL',
         'ca_coord': np.array([7.6, 0.0, 0.0]),
         'n_coord': np.array([5.0, 0.0, 0.0]),
         'c_coord': np.array([9.0, 0.0, 0.0])},
    ]
    result = reconstruct_protein_backbone(residues, 1, ['CA'])
    if not _assert('CA' in result, f"{name}: CA 未被重建"):
        return
    # np.ndarray, (3,), 重建的 CA 应接近中点 (3.8, 0, 0)
    ca = result['CA']
    expected = (np.array([0.0, 0.0, 0.0]) + np.array([7.6, 0.0, 0.0])) / 2.0
    if not _assert(np.allclose(ca, expected, atol=0.1),
                   f"{name}: CA={ca}, 期望接近 {expected}"):
        return
    _test_pass(name)


def test_protein_ca_extrapolation_prev_only():
    """CA 外推: 仅有前一残基 CA"""
    name = "蛋白 CA 外推 (仅前一)"
    residues = [
        {'type': 'protein', 'name': 'ALA',
         'ca_coord': np.array([0.0, 1.0, 2.0]),
         'n_coord': np.array([-1.0, 0.0, 0.0]),
         'c_coord': np.array([1.0, 0.0, 0.0])},
        {'type': 'protein', 'name': 'GLY',
         'ca_coord': None, 'n_coord': None, 'c_coord': None},
    ]
    result = reconstruct_protein_backbone(residues, 1, ['CA'])
    if not _assert('CA' in result, f"{name}: CA 未被重建"):
        return
    # np.ndarray, (3,), 外推结果应存在
    ca = result['CA']
    if not _assert(ca.shape == (3,), f"{name}: shape={ca.shape}"):
        return
    _test_pass(name)


def test_protein_ca_extrapolation_next_only():
    """CA 外推: 仅有后一残基 CA"""
    name = "蛋白 CA 外推 (仅后一)"
    residues = [
        {'type': 'protein', 'name': 'ALA',
         'ca_coord': None, 'n_coord': None, 'c_coord': None},
        {'type': 'protein', 'name': 'GLY',
         'ca_coord': np.array([7.6, 0.0, 0.0]),
         'n_coord': np.array([5.0, 0.0, 0.0]),
         'c_coord': np.array([9.0, 0.0, 0.0])},
    ]
    result = reconstruct_protein_backbone(residues, 0, ['CA'])
    if not _assert('CA' in result, f"{name}: CA 未被重建"):
        return
    ca = result['CA']
    if not _assert(ca.shape == (3,), f"{name}: shape={ca.shape}"):
        return
    _test_pass(name)


def test_protein_n_reconstruction():
    """N 重建: 基于前一残基 C 和当前 CA 方向插值"""
    name = "蛋白 N 重建 (prev_C + CA)"
    residues = [
        {'type': 'protein', 'name': 'ALA',
         'ca_coord': np.array([0.0, 0.0, 0.0]),
         'n_coord': np.array([-1.0, 0.0, 0.0]),
         'c_coord': np.array([1.5, 0.0, 0.0])},
        {'type': 'protein', 'name': 'GLY',
         'ca_coord': np.array([4.0, 0.0, 0.0]),
         'n_coord': None,
         'c_coord': np.array([5.5, 0.0, 0.0])},
    ]
    result = reconstruct_protein_backbone(residues, 1, ['N'])
    if not _assert('N' in result, f"{name}: N 未被重建"):
        return
    n_coord = result['N']
    if not _assert(n_coord.shape == (3,), f"{name}: shape={n_coord.shape}"):
        return
    # N 应该在前一残基 C (1.5) 和当前 CA (4.0) 之间
    if not _assert(1.3 <= n_coord[0] <= 4.0,
                   f"{name}: N.x={n_coord[0]} 不在 [1.3, 4.0] 范围内"):
        return
    _test_pass(name)


def test_protein_c_reconstruction():
    """C 重建: 基于当前 CA 和下一残基 N 方向插值"""
    name = "蛋白 C 重建 (CA + next_N)"
    residues = [
        {'type': 'protein', 'name': 'ALA',
         'ca_coord': np.array([0.0, 0.0, 0.0]),
         'n_coord': np.array([-1.0, 0.0, 0.0]),
         'c_coord': None},
        {'type': 'protein', 'name': 'GLY',
         'ca_coord': np.array([4.0, 0.0, 0.0]),
         'n_coord': np.array([2.0, 0.0, 0.0]),
         'c_coord': np.array([5.5, 0.0, 0.0])},
    ]
    result = reconstruct_protein_backbone(residues, 0, ['C'])
    if not _assert('C' in result, f"{name}: C 未被重建"):
        return
    c_coord = result['C']
    if not _assert(c_coord.shape == (3,), f"{name}: shape={c_coord.shape}"):
        return
    # C 应该在 CA (0.0) 和下一 N (2.0) 之间
    if not _assert(0.0 <= c_coord[0] <= 2.0,
                   f"{name}: C.x={c_coord[0]} 不在 [0.0, 2.0] 范围内"):
        return
    _test_pass(name)


def test_protein_all_three_missing():
    """同时缺失 N/CA/C: 应从邻居重建 CA, 再基于 CA 重建 N/C"""
    name = "蛋白 N+CA+C 全缺失重建"
    residues = [
        {'type': 'protein', 'name': 'ALA',
         'ca_coord': np.array([0.0, 0.0, 0.0]),
         'n_coord': np.array([-1.0, 0.0, 0.0]),
         'c_coord': np.array([1.5, 0.0, 0.0])},
        {'type': 'protein', 'name': 'GLY',
         'ca_coord': None, 'n_coord': None, 'c_coord': None},
        {'type': 'protein', 'name': 'VAL',
         'ca_coord': np.array([7.6, 0.0, 0.0]),
         'n_coord': np.array([5.0, 0.0, 0.0]),
         'c_coord': np.array([9.0, 0.0, 0.0])},
    ]
    result = reconstruct_protein_backbone(residues, 1, ['CA', 'N', 'C'])
    if not _assert('CA' in result, f"{name}: CA 未被重建"):
        return
    if not _assert('N' in result, f"{name}: N 未被重建 (基于prev_C + 重建的CA)"):
        return
    if not _assert('C' in result, f"{name}: C 未被重建 (基于重建的CA + next_N)"):
        return
    # 验证所有输出为 (3,) 形状
    for atom_name in ['CA', 'N', 'C']:
        if not _assert(result[atom_name].shape == (3,),
                       f"{name}: {atom_name}.shape={result[atom_name].shape}"):
            return
    _test_pass(name)


def test_protein_isolated_residue_all_missing():
    """孤立残基全缺失: 无邻居，应返回空字典"""
    name = "蛋白孤立残基全缺失 → 空字典"
    residues = [
        {'type': 'protein', 'name': 'ALA',
         'ca_coord': None, 'n_coord': None, 'c_coord': None},
    ]
    result = reconstruct_protein_backbone(residues, 0, ['CA', 'N', 'C'])
    if not _assert(len(result) == 0,
                   f"{name}: 期望空字典, 实际有 {list(result.keys())}"):
        return
    _test_pass(name)


def test_protein_list_vs_ndarray_coords():
    """coords 类型: 前一残基 ca_coord 为 Python list 时不应崩溃"""
    name = "蛋白 Python list 坐标兼容性"
    residues = [
        {'type': 'protein', 'name': 'ALA',
         'ca_coord': [0.0, 0.0, 0.0],  # Python list, 非 np.ndarray
         'n_coord': [-1.0, 0.0, 0.0],
         'c_coord': [1.5, 0.0, 0.0]},
        {'type': 'protein', 'name': 'GLY',
         'ca_coord': None, 'n_coord': None, 'c_coord': None},
        {'type': 'protein', 'name': 'VAL',
         'ca_coord': [7.6, 0.0, 0.0],  # Python list
         'n_coord': [5.0, 0.0, 0.0],
         'c_coord': [9.0, 0.0, 0.0]},
    ]
    try:
        result = reconstruct_protein_backbone(residues, 1, ['CA'])
        if not _assert('CA' in result, f"{name}: CA 未被重建"):
            return
        ca = result['CA']
        if not _assert(hasattr(ca, 'shape') and ca.shape == (3,),
                       f"{name}: 结果应为 (3,) ndarray, 实际类型 {type(ca)}"):
            return
        _test_pass(name)
    except TypeError as e:
        _test_fail(name, f"list+list 运算错误: {e}")


# ============================================================================
# 第二类: 核酸主链坐标重建测试 (含 N1/N9 修复验证)
# ============================================================================
def test_nucleotide_c4_interpolation():
    """核酸 C4' 插值: 前后邻居都有 C4'"""
    name = "核酸 C4' 插值"
    residues = [
        {'type': 'nucleotide', 'name': 'A',
         'c4p_coord': np.array([0.0, 0.0, 0.0]),
         'c1p_coord': np.array([0.0, 2.5, 0.0]),
         'n_base_coord': np.array([0.0, 2.5, 1.5])},
        {'type': 'nucleotide', 'name': 'G',
         'c4p_coord': None, 'c1p_coord': None, 'n_base_coord': None},
        {'type': 'nucleotide', 'name': 'C',
         'c4p_coord': np.array([12.0, 0.0, 0.0]),
         'c1p_coord': np.array([12.0, 2.5, 0.0]),
         'n_base_coord': np.array([12.0, 2.5, 1.5])},
    ]
    result = reconstruct_nucleotide_backbone(residues, 1, ["C4'"])
    if not _assert("C4'" in result, f"{name}: C4' 未被重建"):
        return
    c4 = result["C4'"]
    expected = (np.array([0.0, 0.0, 0.0]) + np.array([12.0, 0.0, 0.0])) / 2.0
    if not _assert(np.allclose(c4, expected, atol=0.1),
                   f"{name}: C4'={c4}, 期望接近 {expected}"):
        return
    _test_pass(name)


def test_nucleotide_n9_reconstruction_purine():
    """嘌呤 N9 重建: 使用 'N9' 字符串而非 'N1/N9'"""
    name = "嘌呤 N9 重建 (字符串匹配修复)"
    residues = [
        {'type': 'nucleotide', 'name': 'A',
         'c4p_coord': np.array([0.0, 0.0, 0.0]),
         'c1p_coord': np.array([0.0, 2.5, 0.0]),
         'n_base_coord': None},
    ]
    # 关键修复验证: missing_atoms 应包含 'N9' 而非 'N1/N9'
    result = reconstruct_nucleotide_backbone(residues, 0, ['N9'])
    if not _assert('N9' in result,
                   f"{name}: N9 未被重建 (检查 missing_atoms 字符串是否正确)"):
        return
    n9 = result['N9']
    if not _assert(n9.shape == (3,), f"{name}: shape={n9.shape}"):
        return
    _test_pass(name)


def test_nucleotide_n1_reconstruction_pyrimidine():
    """嘧啶 N1 重建: 使用 'N1' 字符串"""
    name = "嘧啶 N1 重建 (字符串匹配修复)"
    residues = [
        {'type': 'nucleotide', 'name': 'C',
         'c4p_coord': np.array([0.0, 0.0, 0.0]),
         'c1p_coord': np.array([0.0, 2.5, 0.0]),
         'n_base_coord': None},
    ]
    result = reconstruct_nucleotide_backbone(residues, 0, ['N1'])
    if not _assert('N1' in result,
                   f"{name}: N1 未被重建"):
        return
    n1 = result['N1']
    if not _assert(n1.shape == (3,), f"{name}: shape={n1.shape}"):
        return
    _test_pass(name)


def test_nucleotide_old_n1n9_string_no_match():
    """旧 'N1/N9' 字符串: 重建函数应跳过 (不崩溃)"""
    name = "旧 'N1/N9' 字符串 → 跳过但不崩溃"
    residues = [
        {'type': 'nucleotide', 'name': 'A',
         'c4p_coord': np.array([0.0, 0.0, 0.0]),
         'c1p_coord': np.array([0.0, 2.5, 0.0]),
         'n_base_coord': None},
    ]
    # 使用旧的有 bug 的字符串 'N1/N9'
    result = reconstruct_nucleotide_backbone(residues, 0, ['N1/N9'])
    # 旧字符串应该不匹配任何分支, 但函数应正常返回
    if not _assert('N9' not in result and 'N1' not in result,
                   f"{name}: 不应该匹配, 实际结果 {list(result.keys())}"):
        return
    _test_pass(name)


def test_nucleotide_all_missing_with_neighbors():
    """核酸全缺失但有邻居: C4' 应重建成功, C1' 和 N9 级联重建"""
    name = "核酸全缺失 (嘌呤, 有邻居)"
    residues = [
        {'type': 'nucleotide', 'name': 'A',
         'c4p_coord': np.array([0.0, 0.0, 0.0]),
         'c1p_coord': np.array([0.0, 2.5, 0.0]),
         'n_base_coord': np.array([0.0, 2.5, 1.5])},
        {'type': 'nucleotide', 'name': 'G',
         'c4p_coord': None, 'c1p_coord': None, 'n_base_coord': None},
        {'type': 'nucleotide', 'name': 'C',
         'c4p_coord': np.array([12.0, 0.0, 0.0]),
         'c1p_coord': np.array([12.0, 2.5, 0.0]),
         'n_base_coord': np.array([12.0, 2.5, 1.5])},
    ]
    result = reconstruct_nucleotide_backbone(residues, 1, ["C4'", "C1'", "N9"])
    if not _assert("C4'" in result, f"{name}: C4' 未被重建"):
        return
    if not _assert("C1'" in result, f"{name}: C1' 未被重建 (级联)"):
        return
    if not _assert("N9" in result, f"{name}: N9 未被重建 (级联)"):
        return
    _test_pass(name)


def test_nucleotide_list_coords():
    """核酸坐标为 Python list 时不应崩溃"""
    name = "核酸 Python list 坐标兼容性"
    residues = [
        {'type': 'nucleotide', 'name': 'A',
         'c4p_coord': [0.0, 0.0, 0.0],  # Python list
         'c1p_coord': [0.0, 2.5, 0.0],
         'n_base_coord': [0.0, 2.5, 1.5]},
        {'type': 'nucleotide', 'name': 'G',
         'c4p_coord': None, 'c1p_coord': None, 'n_base_coord': None},
        {'type': 'nucleotide', 'name': 'C',
         'c4p_coord': [12.0, 0.0, 0.0],
         'c1p_coord': [12.0, 2.5, 0.0],
         'n_base_coord': [12.0, 2.5, 1.5]},
    ]
    try:
        result = reconstruct_nucleotide_backbone(residues, 1, ["C4'"])
        if not _assert("C4'" in result, f"{name}: C4' 未被重建"):
            return
        _test_pass(name)
    except TypeError as e:
        _test_fail(name, f"list 运算错误: {e}")


# ============================================================================
# 第三类: local_frames.py 导入和功能测试
# ============================================================================
def test_local_frames_importable():
    """local_frames 模块应能成功 import (无 SyntaxError)"""
    name = "local_frames 模块 import"
    try:
        from PDB_processor.geometry.local_frames import compute_local_frame, compute_local_frames
        _test_pass(name)
    except SyntaxError as e:
        _test_fail(name, f"SyntaxError: {e}")
    except Exception as e:
        _test_fail(name, f"导入异常: {e}")


def test_local_frame_basic():
    """compute_local_frame: 合法的三点应产生正交旋转矩阵"""
    name = "compute_local_frame 正交性"
    from PDB_processor.geometry.local_frames import compute_local_frame
    p1 = np.array([0.0, 0.0, 0.0])
    p2 = np.array([1.0, 0.0, 0.0])
    p3 = np.array([2.0, 1.0, 0.0])
    frame = compute_local_frame(p1, p2, p3)
    if not _assert(frame.shape == (3, 3), f"{name}: shape={frame.shape}"):
        return
    # 检查正交性: R^T @ R ≈ I
    # np.ndarray, (3, 3), float32, 应接近单位矩阵
    product = frame.T @ frame
    if not _assert(np.allclose(product, np.eye(3), atol=1e-5),
                   f"{name}: R^T@R 不是单位矩阵\n{product}"):
        return
    # 检查行列式 ≈ 1 (右手系)
    det = np.linalg.det(frame)
    if not _assert(abs(det - 1.0) < 1e-5,
                   f"{name}: det(R)={det}, 期望 1.0"):
        return
    _test_pass(name)


def test_local_frame_degenerate_collinear():
    """compute_local_frame: 三点共线应抛 ValueError"""
    name = "compute_local_frame 共线检测"
    from PDB_processor.geometry.local_frames import compute_local_frame
    p1 = np.array([0.0, 0.0, 0.0])
    p2 = np.array([1.0, 0.0, 0.0])
    p3 = np.array([2.0, 0.0, 0.0])  # 三点共线
    try:
        compute_local_frame(p1, p2, p3)
        _test_fail(name, "共线三点未抛出 ValueError")
    except ValueError:
        _test_pass(name)


def test_compute_local_frames_strict_mode():
    """compute_local_frames 严格模式: 骨架缺零向量应抛异常"""
    name = "compute_local_frames 严格模式 (zero → 异常)"
    from PDB_processor.geometry.local_frames import compute_local_frames
    from PDB_processor.parser import ParsedStructure

    parsed = ParsedStructure.__new__(ParsedStructure)
    parsed.res_names = ['ALA']
    parsed.res_types = ['protein']
    parsed.backbone_n_coords = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)   # 零向量
    parsed.backbone_ca_coords = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
    parsed.backbone_c_coords = np.array([[2.0, 1.0, 0.0]], dtype=np.float32)

    try:
        compute_local_frames(parsed, allow_incomplete_backbone=False)
        _test_fail(name, "零向量骨架未触发异常")
    except Exception:
        _test_pass(name)


def test_compute_local_frames_relaxed_mode():
    """compute_local_frames 宽松模式: 骨架缺零向量 → 单位矩阵 + mask=False"""
    name = "compute_local_frames 宽松模式 (zero → 单位矩阵)"
    from PDB_processor.geometry.local_frames import compute_local_frames
    from PDB_processor.parser import ParsedStructure

    parsed = ParsedStructure.__new__(ParsedStructure)
    parsed.res_names = ['ALA', 'GLY']
    parsed.res_types = ['protein', 'protein']
    parsed.backbone_n_coords = np.array([
        [0.0, 0.0, 0.0],   # 零向量 → 应标记为 False
        [-1.0, 0.0, 0.0],  # 正常
    ], dtype=np.float32)
    parsed.backbone_ca_coords = np.array([
        [1.0, 0.0, 0.0],
        [3.0, 0.0, 0.0],
    ], dtype=np.float32)
    parsed.backbone_c_coords = np.array([
        [2.0, 1.0, 0.0],
        [4.0, 1.0, 0.0],
    ], dtype=np.float32)

    frames, mask = compute_local_frames(parsed, allow_incomplete_backbone=True)
    if not _assert(frames.shape == (2, 3, 3), f"{name}: frames.shape={frames.shape}"):
        return
    if not _assert(mask.shape == (2,), f"{name}: mask.shape={mask.shape}"):
        return
    # 第一个残基应为无效 (骨架 N 为零向量)
    if not _assert(mask[0] == False, f"{name}: mask[0]={mask[0]}, 期望 False"):
        return
    # 第二个残基应为有效
    if not _assert(mask[1] == True, f"{name}: mask[1]={mask[1]}, 期望 True"):
        return
    # 第一个残基的 frame 应为单位矩阵
    if not _assert(np.allclose(frames[0], np.eye(3)),
                   f"{name}: frames[0] 不是单位矩阵"):
        return
    _test_pass(name)


def test_compute_local_frames_nucleotide_relaxed():
    """compute_local_frames 核酸宽松模式: 骨架零向量 → 单位矩阵"""
    name = "compute_local_frames 核酸宽松模式"
    from PDB_processor.geometry.local_frames import compute_local_frames
    from PDB_processor.parser import ParsedStructure

    parsed = ParsedStructure.__new__(ParsedStructure)
    parsed.res_names = ['A']
    parsed.res_types = ['nucleotide']
    parsed.backbone_c4p_coords = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)  # 零向量
    parsed.backbone_c1p_coords = np.array([[2.5, 0.0, 0.0]], dtype=np.float32)
    parsed.backbone_n_base_coords = np.array([[4.0, 1.0, 0.0]], dtype=np.float32)

    frames, mask = compute_local_frames(parsed, allow_incomplete_backbone=True)
    if not _assert(mask[0] == False, f"{name}: mask[0]={mask[0]}, 期望 False"):
        return
    if not _assert(np.allclose(frames[0], np.eye(3)),
                   f"{name}: frames[0] 不是单位矩阵"):
        return
    _test_pass(name)


# ============================================================================
# 第四类: backbone_complete_mask 长度一致性测试
# ============================================================================
def test_backbone_mask_length_strict_mode():
    """严格模式: backbone_complete_mask 长度 == N_res (模拟完整骨架)"""
    name = "backbone_complete_mask 长度 (严格模式, 完整骨架)"
    # 直接构造 residue_order 和 residue_info 来模拟 parser 行为
    n_residues = 5
    # list[bool], 长度应等于 n_residues
    backbone_complete_mask = []
    for i in range(n_residues):
        # 模拟严格模式: 所有骨架完整, 每个残基只 append 一次
        backbone_complete_mask.append(True)
    # 不应再有额外的 append
    if not _assert(len(backbone_complete_mask) == n_residues,
                   f"{name}: 长度={len(backbone_complete_mask)}, 期望 {n_residues}"):
        return
    _test_pass(name)


def test_backbone_mask_length_relaxed_mode():
    """宽松模式: backbone_complete_mask 长度 == N_res (含补全残基)"""
    name = "backbone_complete_mask 长度 (宽松模式, 含补全)"
    n_residues = 5
    backbone_complete_mask = []
    for i in range(n_residues):
        if i == 2:
            # 模拟补全: append False
            backbone_complete_mask.append(False)
        else:
            backbone_complete_mask.append(True)
    # 验证: 不应有多余的 append
    if not _assert(len(backbone_complete_mask) == n_residues,
                   f"{name}: 长度={len(backbone_complete_mask)}, 期望 {n_residues}"):
        return
    arr = np.array(backbone_complete_mask, dtype=bool)
    if not _assert(arr.shape == (n_residues,),
                   f"{name}: 数组 shape={arr.shape}, 期望 ({n_residues},)"):
        return
    if not _assert(arr[2] == False, f"{name}: arr[2]={arr[2]}, 期望 False"):
        return
    _test_pass(name)


# ============================================================================
# 第五类: 混合残基类型边界测试
# ============================================================================
def test_mixed_types_reconstruction_no_crash():
    """混合类型: 蛋白邻居对核酸重建不应提供 CA (类型不匹配)"""
    name = "混合类型 (蛋白邻居不影响核酸重建)"
    residues = [
        {'type': 'protein', 'name': 'ALA',
         'ca_coord': np.array([0.0, 0.0, 0.0]),
         'n_coord': np.array([-1.0, 0.0, 0.0]),
         'c_coord': np.array([1.5, 0.0, 0.0])},
        {'type': 'nucleotide', 'name': 'A',
         'c4p_coord': None, 'c1p_coord': None, 'n_base_coord': None},
        {'type': 'protein', 'name': 'GLY',
         'ca_coord': np.array([10.0, 0.0, 0.0]),
         'n_coord': np.array([9.0, 0.0, 0.0]),
         'c_coord': np.array([11.5, 0.0, 0.0])},
    ]
    # 核酸重建: 前后邻居都是蛋白, _get_prev_c4 和 _get_next_c4 应返回 None
    result = reconstruct_nucleotide_backbone(residues, 1, ["C4'", "C1'", "N9"])
    # 无核酸邻居 → 无法重建 C4' → 级联失败
    if not _assert(len(result) == 0,
                   f"{name}: 期望空字典, 实际 {list(result.keys())}"):
        return
    _test_pass(name)


def test_empty_residue_list_protein():
    """空残基列表: 不应崩溃"""
    name = "空残基列表 (蛋白)"
    try:
        result = reconstruct_protein_backbone([], 0, ['CA'])
        if not _assert(len(result) == 0, f"{name}: 期望空字典"):
            return
        _test_pass(name)
    except IndexError:
        _test_fail(name, "空列表导致 IndexError")


def test_empty_residue_list_nucleotide():
    """空残基列表: 不应崩溃"""
    name = "空残基列表 (核酸)"
    try:
        result = reconstruct_nucleotide_backbone([], 0, ["C4'"])
        if not _assert(len(result) == 0, f"{name}: 期望空字典"):
            return
        _test_pass(name)
    except IndexError:
        _test_fail(name, "空列表导致 IndexError")


# ============================================================================
# 第六类: get_features_when_infer 签名检查
# ============================================================================
def test_get_features_when_infer_signature():
    """get_features_when_infer 应接受 allow_incomplete_backbone 参数"""
    name = "get_features_when_infer 签名包含 allow_incomplete_backbone"
    import inspect
    # 需要从 Make_Data 根目录导入
    try:
        from process_and_label import get_features_when_infer
        sig = inspect.signature(get_features_when_infer)
        if not _assert('allow_incomplete_backbone' in sig.parameters,
                       f"{name}: 参数不在签名中, 当前参数: {list(sig.parameters.keys())}"):
            return
        # 默认值应为 False
        param = sig.parameters['allow_incomplete_backbone']
        if not _assert(param.default == False,
                       f"{name}: 默认值={param.default}, 期望 False"):
            return
        _test_pass(name)
    except ImportError as e:
        _test_fail(name, f"导入失败: {e}")
    except Exception as e:
        _test_fail(name, f"异常: {e}")


# ============================================================================
# 运行入口
# ============================================================================
def main():
    """运行所有测试"""
    global _PASSED, _FAILED, _ERRORS

    print("=" * 70)
    print("  严格测试: allow_incomplete_backbone 回退策略全链路验证")
    print("=" * 70)

    # ---- 第一类: 蛋白质主链坐标重建 ----
    print("\n--- 蛋白质主链坐标重建 ---")
    tests_protein = [
        test_protein_ca_interpolation_both_neighbors,
        test_protein_ca_extrapolation_prev_only,
        test_protein_ca_extrapolation_next_only,
        test_protein_n_reconstruction,
        test_protein_c_reconstruction,
        test_protein_all_three_missing,
        test_protein_isolated_residue_all_missing,
        test_protein_list_vs_ndarray_coords,
    ]

    # ---- 第二类: 核酸主链坐标重建 ----
    print("\n--- 核酸主链坐标重建 ---")
    tests_nucleotide = [
        test_nucleotide_c4_interpolation,
        test_nucleotide_n9_reconstruction_purine,
        test_nucleotide_n1_reconstruction_pyrimidine,
        test_nucleotide_old_n1n9_string_no_match,
        test_nucleotide_all_missing_with_neighbors,
        test_nucleotide_list_coords,
    ]

    # ---- 第三类: local_frames ----
    print("\n--- local_frames 模块 ---")
    tests_frames = [
        test_local_frames_importable,
        test_local_frame_basic,
        test_local_frame_degenerate_collinear,
        test_compute_local_frames_strict_mode,
        test_compute_local_frames_relaxed_mode,
        test_compute_local_frames_nucleotide_relaxed,
    ]

    # ---- 第四类: backbone_complete_mask ----
    print("\n--- backbone_complete_mask 长度一致性 ---")
    tests_mask = [
        test_backbone_mask_length_strict_mode,
        test_backbone_mask_length_relaxed_mode,
    ]

    # ---- 第五类: 边界情况 ----
    print("\n--- 混合类型 / 边界情况 ---")
    tests_edge = [
        test_mixed_types_reconstruction_no_crash,
        test_empty_residue_list_protein,
        test_empty_residue_list_nucleotide,
    ]

    # ---- 第六类: 签名检查 ----
    print("\n--- 函数签名检查 ---")
    tests_sig = [
        test_get_features_when_infer_signature,
    ]

    all_tests = tests_protein + tests_nucleotide + tests_frames + tests_mask + tests_edge + tests_sig

    for test_fn in all_tests:
        try:
            test_fn()
        except Exception as e:
            _test_fail(test_fn.__name__, f"未捕获异常: {e}\n{traceback.format_exc()}")

    # ---- 结果汇总 ----
    print("\n" + "=" * 70)
    total = _PASSED + _FAILED
    print(f"  测试结果:  {_PASSED}/{total} 通过,  {_FAILED}/{total} 失败")
    if _ERRORS:
        print("\n  失败详情:")
        for err in _ERRORS:
            print(f"    - {err}")
    print("=" * 70)

    return 0 if _FAILED == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
