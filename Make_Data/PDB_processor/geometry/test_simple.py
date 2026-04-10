"""
坐标重建模块简单测试

不依赖 pytest，直接运行测试
"""

import sys
import os

# 添加父目录到路径以便导入模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from coordinate_reconstruction import (
    reconstruct_protein_backbone,
    reconstruct_nucleotide_backbone,
)


def test_reconstruct_ca_with_neighbors():
    """测试使用前后残基插值重建 CA"""
    print("测试: 使用前后残基插值重建 CA...")

    # list[dict], 可变长度, 模拟三个连续残基
    residues = [
        {
            'type': 'protein',
            'name': 'ALA',
            'ca_coord': np.array([0.0, 0.0, 0.0]),
            'n_coord': np.array([-1.0, 0.0, 0.0]),
            'c_coord': np.array([1.0, 0.0, 0.0]),
        },
        {
            'type': 'protein',
            'name': 'GLY',
            'ca_coord': None,  # 缺失 CA
            'n_coord': np.array([2.0, 0.0, 0.0]),
            'c_coord': np.array([4.0, 0.0, 0.0]),
        },
        {
            'type': 'protein',
            'name': 'VAL',
            'ca_coord': np.array([6.0, 0.0, 0.0]),
            'n_coord': np.array([5.0, 0.0, 0.0]),
            'c_coord': np.array([7.0, 0.0, 0.0]),
        },
    ]

    # dict[str, np.ndarray], 无固定形状, 重建结果
    result = reconstruct_protein_backbone(residues, 1, ['CA'])

    assert 'CA' in result, "应该成功重建 CA"
    # np.ndarray, (3,), 重建的 CA 坐标
    reconstructed_ca = result['CA']
    assert reconstructed_ca.shape == (3,), f"CA 坐标形状应为 (3,), 实际为 {reconstructed_ca.shape}"
    assert 2.0 <= reconstructed_ca[0] <= 4.0, f"CA x 坐标应在 [2.0, 4.0] 范围内, 实际为 {reconstructed_ca[0]}"

    print(f"  ✓ 成功重建 CA: {reconstructed_ca}")


def test_reconstruct_n_with_prev_c():
    """测试使用前一残基 C 和当前 CA 重建 N"""
    print("测试: 使用前一残基 C 重建 N...")

    # list[dict], 可变长度, 模拟两个连续残基
    residues = [
        {
            'type': 'protein',
            'name': 'ALA',
            'ca_coord': np.array([0.0, 0.0, 0.0]),
            'n_coord': np.array([-1.0, 0.0, 0.0]),
            'c_coord': np.array([1.5, 0.0, 0.0]),
        },
        {
            'type': 'protein',
            'name': 'GLY',
            'ca_coord': np.array([3.0, 0.0, 0.0]),
            'n_coord': None,  # 缺失 N
            'c_coord': np.array([4.5, 0.0, 0.0]),
        },
    ]

    # dict[str, np.ndarray], 无固定形状, 重建结果
    result = reconstruct_protein_backbone(residues, 1, ['N'])

    assert 'N' in result, "应该成功重建 N"
    # np.ndarray, (3,), 重建的 N 坐标
    reconstructed_n = result['N']
    assert reconstructed_n.shape == (3,), f"N 坐标形状应为 (3,), 实际为 {reconstructed_n.shape}"
    assert 1.5 <= reconstructed_n[0] <= 3.0, f"N x 坐标应在 [1.5, 3.0] 范围内, 实际为 {reconstructed_n[0]}"

    print(f"  ✓ 成功重建 N: {reconstructed_n}")


def test_reconstruct_multiple_atoms():
    """测试同时重建多个缺失原子"""
    print("测试: 同时重建多个缺失原子...")

    # list[dict], 可变长度, 模拟三个连续残基
    residues = [
        {
            'type': 'protein',
            'name': 'ALA',
            'ca_coord': np.array([0.0, 0.0, 0.0]),
            'n_coord': np.array([-1.0, 0.0, 0.0]),
            'c_coord': np.array([1.5, 0.0, 0.0]),
        },
        {
            'type': 'protein',
            'name': 'GLY',
            'ca_coord': None,  # 缺失 CA
            'n_coord': None,  # 缺失 N
            'c_coord': None,  # 缺失 C
        },
        {
            'type': 'protein',
            'name': 'VAL',
            'ca_coord': np.array([6.0, 0.0, 0.0]),
            'n_coord': np.array([4.5, 0.0, 0.0]),
            'c_coord': np.array([7.5, 0.0, 0.0]),
        },
    ]

    # dict[str, np.ndarray], 无固定形状, 重建结果
    result = reconstruct_protein_backbone(residues, 1, ['CA', 'N', 'C'])

    assert 'CA' in result, "应该至少重建 CA"
    assert result['CA'].shape == (3,), "CA 坐标形状应为 (3,)"

    print(f"  ✓ 成功重建 {len(result)} 个原子: {list(result.keys())}")


def test_reconstruct_c4_with_neighbors():
    """测试使用前后核苷酸插值重建 C4'"""
    print("测试: 使用前后核苷酸插值重建 C4'...")

    # list[dict], 可变长度, 模拟三个连续核苷酸
    residues = [
        {
            'type': 'nucleotide',
            'name': 'A',
            'c4p_coord': np.array([0.0, 0.0, 0.0]),
            'c1p_coord': np.array([0.0, 2.5, 0.0]),
            'n_base_coord': np.array([0.0, 2.5, 1.5]),
        },
        {
            'type': 'nucleotide',
            'name': 'G',
            'c4p_coord': None,  # 缺失 C4'
            'c1p_coord': np.array([6.0, 2.5, 0.0]),
            'n_base_coord': np.array([6.0, 2.5, 1.5]),
        },
        {
            'type': 'nucleotide',
            'name': 'C',
            'c4p_coord': np.array([12.0, 0.0, 0.0]),
            'c1p_coord': np.array([12.0, 2.5, 0.0]),
            'n_base_coord': np.array([12.0, 2.5, 1.5]),
        },
    ]

    # dict[str, np.ndarray], 无固定形状, 重建结果
    result = reconstruct_nucleotide_backbone(residues, 1, ["C4'"])

    assert "C4'" in result, "应该成功重建 C4'"
    # np.ndarray, (3,), 重建的 C4' 坐标
    reconstructed_c4 = result["C4'"]
    assert reconstructed_c4.shape == (3,), f"C4' 坐标形状应为 (3,), 实际为 {reconstructed_c4.shape}"
    assert 4.0 <= reconstructed_c4[0] <= 8.0, f"C4' x 坐标应在 [4.0, 8.0] 范围内, 实际为 {reconstructed_c4[0]}"

    print(f"  ✓ 成功重建 C4': {reconstructed_c4}")


def test_no_reconstruction_possible():
    """测试无法重建的情况"""
    print("测试: 无法重建的情况...")

    # list[dict], 可变长度, 单个残基且缺失所有主链原子
    residues = [
        {
            'type': 'protein',
            'name': 'ALA',
            'ca_coord': None,
            'n_coord': None,
            'c_coord': None,
        },
    ]

    # dict[str, np.ndarray], 无固定形状, 重建结果
    result = reconstruct_protein_backbone(residues, 0, ['CA', 'N', 'C'])

    assert len(result) == 0, "无法重建时应返回空字典"

    print("  ✓ 正确返回空字典")


def main():
    """运行所有测试"""
    print("=" * 60)
    print("坐标重建模块测试")
    print("=" * 60)

    tests = [
        test_reconstruct_ca_with_neighbors,
        test_reconstruct_n_with_prev_c,
        test_reconstruct_multiple_atoms,
        test_reconstruct_c4_with_neighbors,
        test_no_reconstruction_possible,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"  ✗ 失败: {e}")
            failed += 1
        except Exception as e:
            print(f"  ✗ 错误: {e}")
            failed += 1
        print()

    print("=" * 60)
    print(f"测试结果: {passed} 通过, {failed} 失败")
    print("=" * 60)

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
