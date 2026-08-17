# -*- coding: utf-8 -*-
"""从 PDB/CIF 结构中剔除 HETATM，并写出只含 polymer 受体的 mmCIF。

本模块被推理和密度图预处理入口共用；``ReceptorOnlySelect`` 按 Biopython 残基的 ``het_flag`` 过滤标准 ATOM 与 HETATM，:func:`extract_receptor_cif` 负责选择解析器、写出文件并返回可记录的统计字段。
"""

import warnings
from pathlib import Path

from Bio.PDB import MMCIFParser, PDBParser, MMCIFIO
from Bio.PDB.PDBIO import Select


class ReceptorOnlySelect(Select):
    """保留标准 ATOM 残基并统计输出的受体残基和原子数。

    过滤规则:
        - ``residue.id[0] == " "``：标准 ATOM polymer 残基，接受并计数。
        - ``residue.id[0] != " "``：HETATM 残基，包括水、配体、离子和修饰残基，拒绝。
        - 已接受残基中的每个 atom 都接受；原子计数由 :meth:`accept_atom` 累加。

    状态字段:
        - n_accepted_residues: int；已接受的 polymer 残基数。
        - n_accepted_atoms: int；已接受的原子数。
    """

    def __init__(self):
        super().__init__()
        # int；统计已保留的 polymer 受体残基数。
        self.n_accepted_residues = 0
        # int；统计已保留的受体原子数。
        self.n_accepted_atoms = 0

    def accept_residue(self, residue):
        """按 Biopython ``het_flag`` 判断残基是否是 polymer 受体。

        输入参数:
            - residue: Bio.PDB.Residue.Residue；待过滤的 Biopython 残基对象。

        返回值:
            - accepted: int；``1`` 表示 ``residue.id[0] == " "`` 并保留，``0`` 表示 HETATM 并剔除。

        状态变化:
            - 接受标准 ATOM 残基时将 ``n_accepted_residues`` 加一。
        """
        # str；残基 HETATM 标识；空格表示标准 ATOM，``H_*`` 或 ``W`` 表示 HETATM。
        het_flag = residue.id[0]
        if het_flag == ' ':
            self.n_accepted_residues += 1
            return 1
        return 0

    def accept_atom(self, atom):
        """接受已保留残基中的原子并累加输出原子数。

        输入参数:
            - atom: Bio.PDB.Atom.Atom；Biopython 当前原子对象；选择规则不读取其坐标或名称。

        返回值:
            - accepted: int；始终为 ``1``。

        状态变化:
            - ``n_accepted_atoms`` 加一。
        """
        self.n_accepted_atoms += 1
        return 1


def extract_receptor_cif(input_pdb_path: str, output_cif_path: str):
    """解析结构、过滤 HETATM 并写出 mmCIF，同时返回稳定的处理摘要。

    输入参数:
        - input_pdb_path: str；输入 ``.pdb``、``.cif`` 或 ``.mmcif`` 文件路径。
        - output_cif_path: str；只含 polymer 受体的输出 mmCIF 路径；父目录自动创建。

    返回值:
        - sample_id: str；输入文件 stem。
        - success: bool；解析、过滤和写出都成功且至少保留一个受体残基时为真。
        - error_msg: str | None；失败原因；成功时为 ``None``。
        - n_receptor_residues: int；写出的受体残基数；失败时为 0。
        - n_receptor_atoms: int；写出的受体原子数；失败时为 0。

    失败语义:
        - 输入不存在、后缀不支持、Biopython 解析失败、写出失败或过滤后没有受体残基时返回失败摘要，不向调用方抛出结构处理异常。
    """
    # str；从输入文件名推断的样本 identity。
    sample_id = Path(input_pdb_path).stem

    if not Path(input_pdb_path).exists():
        return sample_id, False, f"文件不存在: {input_pdb_path}", 0, 0

    # str；输入文件后缀的小写形式，用于选择 PDB 或 mmCIF 解析器。
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
            # Bio.PDB.Structure.Structure；Biopython 解析后的完整结构对象。
            structure = parser.get_structure(sample_id, input_pdb_path)
    except Exception as e:
        return sample_id, False, f"解析失败: {type(e).__name__}: {e}", 0, 0

    try:
        Path(output_cif_path).parent.mkdir(parents=True, exist_ok=True)
        # ReceptorOnlySelect；按残基 HETATM 标志过滤并统计输出对象。
        selector = ReceptorOnlySelect()
        io = MMCIFIO()
        io.set_structure(structure)
        io.save(output_cif_path, select=selector)
        # int；过滤器统计的受体残基数和原子数。
        n_residues = selector.n_accepted_residues
        n_atoms = selector.n_accepted_atoms
        if n_residues == 0:
            return sample_id, False, "过滤后无受体残基", 0, 0
        return sample_id, True, None, n_residues, n_atoms
    except Exception as e:
        return sample_id, False, f"写出失败: {type(e).__name__}: {e}", 0, 0
