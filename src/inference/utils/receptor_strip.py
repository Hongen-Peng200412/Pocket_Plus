# -*- coding: utf-8 -*-
"""
从 PDB/CIF 结构中剔除所有 HETATM(配体/水/离子/修饰残基), 仅保留 polymer 受体残基, 写出 mmCIF。

共享 util: 由 phenix 差图预生成脚本与 Bundle_of_Maps/simulated_map/get_receptor_from_PDB.py 共同 import,
避免重复实现 receptor 剔除逻辑。
"""

import warnings
from pathlib import Path

from Bio.PDB import MMCIFParser, PDBParser, MMCIFIO
from Bio.PDB.PDBIO import Select


class ReceptorOnlySelect(Select):
    """
    Biopython Select 子类: 仅接受非 HETATM 的残基(受体 polymer)。

    过滤逻辑:
        - het_flag == ' ' (标准 ATOM 记录) → 保留
        - het_flag != ' ' (HETATM, 含水/配体/离子/修饰残基) → 剔除
    """

    def __init__(self):
        super().__init__()
        # int, 标量, 统计被保留的受体残基数
        self.n_accepted_residues = 0
        # int, 标量, 统计被保留的受体原子数
        self.n_accepted_atoms = 0

    def accept_residue(self, residue):
        """
        判断残基是否为受体(非 HETATM)。

        输入参数:
            - residue: Bio.PDB.Residue.Residue, Biopython 残基对象

        输出:
            - int, 1 表示保留, 0 表示剔除
        """
        # str, HETATM 标识; ' ' 表示标准 ATOM, 'H_xxx'/'W' 表示 HETATM
        het_flag = residue.id[0]
        if het_flag == ' ':
            self.n_accepted_residues += 1
            return 1
        return 0

    def accept_atom(self, atom):
        """
        对已接受的残基统计原子数并全部保留。

        输入参数:
            - atom: Bio.PDB.Atom.Atom, Biopython 原子对象

        输出:
            - int, 始终返回 1(全部保留)
        """
        self.n_accepted_atoms += 1
        return 1


def extract_receptor_cif(input_pdb_path: str, output_cif_path: str):
    """
    解析 PDB/CIF 结构, 剔除所有 HETATM, 仅保留 polymer 受体残基, 写出 mmCIF。

    输入参数:
        - input_pdb_path: str, 输入结构文件路径(.pdb / .cif / .mmcif)
        - output_cif_path: str, 输出 CIF 文件路径

    输出:
        - sample_id: str, 自动推断的样本名(文件名 stem)
        - success: bool, 是否成功
        - error_msg: str | None, 错误信息
        - n_receptor_residues: int, 写出的受体残基数(失败时为 0)
        - n_receptor_atoms: int, 写出的受体原子数(失败时为 0)
    """
    # str, 从文件名推断的样本名
    sample_id = Path(input_pdb_path).stem

    if not Path(input_pdb_path).exists():
        return sample_id, False, f"文件不存在: {input_pdb_path}", 0, 0

    # str, 文件后缀(小写)
    file_ext = Path(input_pdb_path).suffix.lower()
    if file_ext == '.pdb':
        parser = PDBParser(QUIET=True)
    elif file_ext in ('.cif', '.mmcif'):
        parser = MMCIFParser(QUIET=True)
    else:
        return sample_id, False, f"不支持的文件格式: {file_ext}", 0, 0

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Bio.PDB.Structure.Structure, 解析后的结构对象
            structure = parser.get_structure(sample_id, input_pdb_path)
    except Exception as e:
        return sample_id, False, f"解析失败: {type(e).__name__}: {e}", 0, 0

    try:
        Path(output_cif_path).parent.mkdir(parents=True, exist_ok=True)
        # ReceptorOnlySelect, 残基级 HETATM 过滤器
        selector = ReceptorOnlySelect()
        io = MMCIFIO()
        io.set_structure(structure)
        io.save(output_cif_path, select=selector)
        # int, 保留的受体残基/原子数
        n_residues = selector.n_accepted_residues
        n_atoms = selector.n_accepted_atoms
        if n_residues == 0:
            return sample_id, False, "过滤后无受体残基", 0, 0
        return sample_id, True, None, n_residues, n_atoms
    except Exception as e:
        return sample_id, False, f"写出失败: {type(e).__name__}: {e}", 0, 0
