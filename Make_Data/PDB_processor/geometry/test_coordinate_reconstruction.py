"""
坐标重建模块测试

测试蛋白质和核酸主链原子的坐标补全功能
"""

import numpy as np
import pytest
from coordinate_reconstruction import (
    reconstruct_protein_backbone,
    reconstruct_nucleotide_backbone,
)


class TestProteinBackboneReconstruction:
    """蛋白质主链坐标重建测试"""

    def test_reconstruct_ca_with_neighbors(self):
        """测试使用前后残基插值重建 CA"""
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

        assert 'CA' in result
        # np.ndarray, (3,), 重建的 CA 坐标应该在前后残基 CA 的中点附近
        reconstructed_ca = result['CA']
        assert reconstructed_ca.shape == (3,)
        # 验证插值结果在合理范围内
        assert 2.0 <= reconstructed_ca[0] <= 4.0

    def test_reconstruct_n_with_prev_c(self):
        """测试使用前一残基 C 和当前 CA 重建 N"""
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

        assert 'N' in result
        # np.ndarray, (3,), 重建的 N 坐标
        reconstructed_n = result['N']
        assert reconstructed_n.shape == (3,)
        # N 应该在前一残基 C 和当前 CA 之间
        assert 1.5 <= reconstructed_n[0] <= 3.0

    def test_reconstruct_c_with_next_n(self):
        """测试使用当前 CA 和下一残基 N 重建 C"""
        # list[dict], 可变长度, 模拟两个连续残基
        residues = [
            {
                'type': 'protein',
                'name': 'ALA',
                'ca_coord': np.array([0.0, 0.0, 0.0]),
                'n_coord': np.array([-1.0, 0.0, 0.0]),
                'c_coord': None,  # 缺失 C
            },
            {
                'type': 'protein',
                'name': 'GLY',
                'ca_coord': np.array([3.0, 0.0, 0.0]),
                'n_coord': np.array([1.8, 0.0, 0.0]),
                'c_coord': np.array([4.5, 0.0, 0.0]),
            },
        ]

        # dict[str, np.ndarray], 无固定形状, 重建结果
        result = reconstruct_protein_backbone(residues, 0, ['C'])

        assert 'C' in result
        # np.ndarray, (3,), 重建的 C 坐标
        reconstructed_c = result['C']
        assert reconstructed_c.shape == (3,)
        # C 应该在当前 CA 和下一残基 N 之间
        assert 0.0 <= reconstructed_c[0] <= 1.8

    def test_reconstruct_multiple_atoms(self):
        """测试同时重建多个缺失原子"""
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

        # 应该至少重建 CA
        assert 'CA' in result
        assert result['CA'].shape == (3,)

    def test_first_residue_missing_ca(self):
        """测试首残基缺失 CA 的情况"""
        # list[dict], 可变长度, 模拟两个残基
        residues = [
            {
                'type': 'protein',
                'name': 'ALA',
                'ca_coord': None,  # 首残基缺失 CA
                'n_coord': np.array([-1.0, 0.0, 0.0]),
                'c_coord': np.array([1.5, 0.0, 0.0]),
            },
            {
                'type': 'protein',
                'name': 'GLY',
                'ca_coord': np.array([3.0, 0.0, 0.0]),
                'n_coord': np.array([1.8, 0.0, 0.0]),
                'c_coord': np.array([4.5, 0.0, 0.0]),
            },
        ]

        # dict[str, np.ndarray], 无固定形状, 重建结果
        result = reconstruct_protein_backbone(residues, 0, ['CA'])

        # 应该能基于后一残基外推
        assert 'CA' in result
        assert result['CA'].shape == (3,)

    def test_last_residue_missing_ca(self):
        """测试末残基缺失 CA 的情况"""
        # list[dict], 可变长度, 模拟两个残基
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
                'ca_coord': None,  # 末残基缺失 CA
                'n_coord': np.array([1.8, 0.0, 0.0]),
                'c_coord': np.array([4.5, 0.0, 0.0]),
            },
        ]

        # dict[str, np.ndarray], 无固定形状, 重建结果
        result = reconstruct_protein_backbone(residues, 1, ['CA'])

        # 应该能基于前一残基外推
        assert 'CA' in result
        assert result['CA'].shape == (3,)

    def test_no_reconstruction_possible(self):
        """测试无法重建的情况"""
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

        # 无法重建时应返回空字典
        assert len(result) == 0


class TestNucleotideBackboneReconstruction:
    """核酸主链坐标重建测试"""

    def test_reconstruct_c4_with_neighbors(self):
        """测试使用前后核苷酸插值重建 C4'"""
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

        assert "C4'" in result
        # np.ndarray, (3,), 重建的 C4' 坐标
        reconstructed_c4 = result["C4'"]
        assert reconstructed_c4.shape == (3,)
        # 验证插值结果在合理范围内
        assert 4.0 <= reconstructed_c4[0] <= 8.0

    def test_reconstruct_c1_from_c4(self):
        """测试基于 C4' 重建 C1'"""
        # list[dict], 可变长度, 模拟单个核苷酸
        residues = [
            {
                'type': 'nucleotide',
                'name': 'A',
                'c4p_coord': np.array([0.0, 0.0, 0.0]),
                'c1p_coord': None,  # 缺失 C1'
                'n_base_coord': np.array([0.0, 2.5, 1.5]),
            },
        ]

        # dict[str, np.ndarray], 无固定形状, 重建结果
        result = reconstruct_nucleotide_backbone(residues, 0, ["C1'"])

        assert "C1'" in result
        # np.ndarray, (3,), 重建的 C1' 坐标
        reconstructed_c1 = result["C1'"]
        assert reconstructed_c1.shape == (3,)
        # C1' 应该距离 C4' 约 2.5Å
        distance = np.linalg.norm(reconstructed_c1 - residues[0]['c4p_coord'])
        assert 2.0 <= distance <= 3.0

    def test_reconstruct_n9_for_purine(self):
        """测试为嘌呤重建 N9"""
        # list[dict], 可变长度, 模拟嘌呤核苷酸
        residues = [
            {
                'type': 'nucleotide',
                'name': 'A',  # 嘌呤
                'c4p_coord': np.array([0.0, 0.0, 0.0]),
                'c1p_coord': np.array([0.0, 2.5, 0.0]),
                'n_base_coord': None,  # 缺失 N9
            },
        ]

        # dict[str, np.ndarray], 无固定形状, 重建结果
        result = reconstruct_nucleotide_backbone(residues, 0, ["N9"])

        assert "N9" in result
        # np.ndarray, (3,), 重建的 N9 坐标
        reconstructed_n9 = result["N9"]
        assert reconstructed_n9.shape == (3,)

    def test_reconstruct_n1_for_pyrimidine(self):
        """测试为嘧啶重建 N1"""
        # list[dict], 可变长度, 模拟嘧啶核苷酸
        residues = [
            {
                'type': 'nucleotide',
                'name': 'C',  # 嘧啶
                'c4p_coord': np.array([0.0, 0.0, 0.0]),
                'c1p_coord': np.array([0.0, 2.5, 0.0]),
                'n_base_coord': None,  # 缺失 N1
            },
        ]

        # dict[str, np.ndarray], 无固定形状, 重建结果
        result = reconstruct_nucleotide_backbone(residues, 0, ["N1"])

        assert "N1" in result
        # np.ndarray, (3,), 重建的 N1 坐标
        reconstructed_n1 = result["N1"]
        assert reconstructed_n1.shape == (3,)

    def test_reconstruct_multiple_nucleotide_atoms(self):
        """测试同时重建多个缺失原子"""
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
                'c1p_coord': None,  # 缺失 C1'
                'n_base_coord': None,  # 缺失 N9
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
        result = reconstruct_nucleotide_backbone(residues, 1, ["C4'", "C1'", "N9"])

        # 应该至少重建 C4'
        assert "C4'" in result


class TestEdgeCases:
    """边界情况测试"""

    def test_empty_residue_list(self):
        """测试空残基列表"""
        # list[dict], 可变长度, 空列表
        residues = []

        # dict[str, np.ndarray], 无固定形状, 重建结果
        result = reconstruct_protein_backbone(residues, 0, ['CA'])

        # 应返回空字典
        assert len(result) == 0

    def test_invalid_residue_index(self):
        """测试无效的残基索引"""
        # list[dict], 可变长度, 单个残基
        residues = [
            {
                'type': 'protein',
                'name': 'ALA',
                'ca_coord': np.array([0.0, 0.0, 0.0]),
                'n_coord': np.array([-1.0, 0.0, 0.0]),
                'c_coord': np.array([1.5, 0.0, 0.0]),
            },
        ]

        # 测试越界索引
        with pytest.raises(IndexError):
            reconstruct_protein_backbone(residues, 5, ['CA'])

    def test_mixed_residue_types(self):
        """测试混合残基类型"""
        # list[dict], 可变长度, 蛋白质和核酸混合
        residues = [
            {
                'type': 'protein',
                'name': 'ALA',
                'ca_coord': np.array([0.0, 0.0, 0.0]),
                'n_coord': np.array([-1.0, 0.0, 0.0]),
                'c_coord': np.array([1.5, 0.0, 0.0]),
            },
            {
                'type': 'nucleotide',
                'name': 'A',
                'c4p_coord': None,
                'c1p_coord': np.array([6.0, 2.5, 0.0]),
                'n_base_coord': np.array([6.0, 2.5, 1.5]),
            },
            {
                'type': 'protein',
                'name': 'GLY',
                'ca_coord': np.array([12.0, 0.0, 0.0]),
                'n_coord': np.array([11.0, 0.0, 0.0]),
                'c_coord': np.array([13.5, 0.0, 0.0]),
            },
        ]

        # 尝试为核酸重建 C4'，但相邻是蛋白质
        # dict[str, np.ndarray], 无固定形状, 重建结果
        result = reconstruct_nucleotide_backbone(residues, 1, ["C4'"])

        # 由于相邻残基类型不匹配，可能无法重建
        # 但函数应该不会崩溃


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
