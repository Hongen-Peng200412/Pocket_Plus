from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class SampleRef:
    """
    一个待处理 PDB 样本的引用。

    输入字段:
        - emdb_id: str, raw.json 中映射到该 PDB 的 EMDB ID; 若未使用 raw.json 过滤则为空字符串
        - pdb_id: str, 小写 PDB ID, 如 '5gmk'
        - pdb_id_upper: str, 大写 PDB ID, 如 '5GMK'
        - parsed_dir: Path, Make_Data parsed_pdb/{pdb_id} 目录
    """

    emdb_id: str
    pdb_id: str
    pdb_id_upper: str
    parsed_dir: Path


@dataclass
class MakeDataLigand:
    """
    Make_Data 中一个 ligand candidate 的结构化记录。

    输入字段:
        - pdb_id: str, 小写 PDB ID
        - candidate_id: int, Make_Data candidate ID
        - ligand_class_id: int, labels.npz 中的 ligand class ID
        - ccd_id: str, 配体 CCD ID, 来自 candidates.npz/resnames
        - chain_id: str, Make_Data chain ID
        - res_id: int, Make_Data residue number
        - insertion_code: str, 插入码; 缺失时为空字符串
        - make_data_heavy_atoms: int, Make_Data 记录的重原子数
        - make_data_coords: np.ndarray, (N_heavy, 3), Make_Data 解析出的配体重原子坐标
    """

    pdb_id: str
    candidate_id: int
    ligand_class_id: int
    ccd_id: str
    chain_id: str
    res_id: int
    insertion_code: str
    make_data_heavy_atoms: int
    make_data_coords: np.ndarray = field(repr=False)


@dataclass
class NonpolySchemeRow:
    """
    RCSB full CIF 中 _pdbx_nonpoly_scheme 的一行。

    输入字段:
        - asym_id: str, RCSB label asym ID, 用于 ModelServer ligand endpoint
        - mon_id: str, CCD ID
        - pdb_seq_num: str, RCSB 页面可见 residue number
        - auth_seq_num: str, 作者 residue number
        - pdb_mon_id: str, PDB 显示的 ligand ID
        - auth_mon_id: str, 作者 ligand ID
        - pdb_strand_id: str, RCSB/PDB chain ID, 与 Make_Data chain_id 对齐
        - pdb_ins_code: str, insertion code
        - source_scheme: str, 来源 scheme, 取值为 nonpoly 或 branch
    """

    asym_id: str
    mon_id: str
    pdb_seq_num: str
    auth_seq_num: str
    pdb_mon_id: str
    auth_mon_id: str
    pdb_strand_id: str
    pdb_ins_code: str
    source_scheme: str = ""


@dataclass
class RCSBInstanceMatch:
    """
    Make_Data ligand 对齐到 RCSB ligand instance 后的结果。

    输入字段:
        - matched: bool, 是否唯一匹配成功
        - match_method: str, PDB_STRAND_PDB_SEQ / PDB_STRAND_AUTH_SEQ / FAILED
        - row: NonpolySchemeRow | None, 匹配到的 nonpoly scheme 行
    """

    matched: bool
    match_method: str
    row: NonpolySchemeRow | None


@dataclass
class RCSBAtomSiteLigand:
    """
    从 RCSB full CIF 的 _atom_site 中取出的 ligand instance 重原子。

    输入字段:
        - coords: np.ndarray, (N_heavy, 3), RCSB CIF 中该 ligand instance 的重原子坐标
        - elements: list[str], 长度 N_heavy, 每个重原子的元素符号
        - atom_names: list[str], 长度 N_heavy, 每个重原子的 atom name
    """

    coords: np.ndarray
    elements: list[str]
    atom_names: list[str]


@dataclass
class ChemCompDescriptors:
    """
    RCSB chemical component descriptor 的核心字段。

    输入字段:
        - smiles: str, RCSB 常规 SMILES
        - smiles_stereo: str, RCSB stereo SMILES
        - inchi: str, InChI
        - inchikey: str, InChIKey
        - formula: str, 化学式
        - formula_weight: str, 分子量字符串; RCSB JSON 中可能为 float, 写出时统一字符串化
    """

    smiles: str
    smiles_stereo: str
    inchi: str
    inchikey: str
    formula: str
    formula_weight: str


@dataclass
class Mol2Info:
    """
    TRIPOS mol2 文件的轻量解析结果。

    输入字段:
        - atom_count: int, mol2 总原子数
        - heavy_atom_count: int, 非 H 原子数
        - hydrogen_count: int, H 原子数
        - bond_count: int, BOND section 条目数
        - has_charge_field: bool, ATOM 行是否都有可解析 charge
        - heavy_coords: np.ndarray, (N_heavy, 3), mol2 重原子坐标
        - heavy_elements: list[str], 长度 N_heavy, mol2 重原子元素
        - raw_has_tripos_molecule: bool, 文件是否包含 @<TRIPOS>MOLECULE
    """

    atom_count: int
    heavy_atom_count: int
    hydrogen_count: int
    bond_count: int
    has_charge_field: bool
    heavy_coords: np.ndarray = field(repr=False)
    heavy_elements: list[str] = field(default_factory=list)
    raw_has_tripos_molecule: bool = False


@dataclass
class ValidationMetrics:
    """
    一个 ligand pair 的多角度校验指标。

    输入字段:
        - make_data_heavy_atoms: int, Make_Data candidate 重原子数
        - rcsb_cif_heavy_atoms: int, RCSB full CIF 中该 ligand instance 的重原子数
        - mol2_heavy_atoms: int, native mol2 重原子数
        - make_cif_coord_median: float | None, Make_Data vs RCSB CIF 最近邻 median
        - make_cif_coord_max: float | None, Make_Data vs RCSB CIF 最近邻 max
        - cif_mol2_coord_median: float | None, RCSB CIF vs mol2 同元素最近邻 median
        - cif_mol2_coord_max: float | None, RCSB CIF vs mol2 同元素最近邻 max
        - coord_status: str, PASS / FAIL / SKIPPED
        - validation_errors: list[str], 校验失败原因列表
    """

    make_data_heavy_atoms: int = 0
    rcsb_cif_heavy_atoms: int = 0
    mol2_heavy_atoms: int = 0
    make_cif_coord_median: float | None = None
    make_cif_coord_max: float | None = None
    cif_mol2_coord_median: float | None = None
    cif_mol2_coord_max: float | None = None
    coord_status: str = "SKIPPED"
    validation_errors: list[str] = field(default_factory=list)


@dataclass
class ToolReadiness:
    """
    三个 docking/identification 工具的输入格式合规性初筛。

    输入字段:
        - dockem_input_status: str, DockEM 输入格式状态
        - emerald_id_input_status: str, EMERALD-ID ligand library 条目状态
        - pocketxmol_input_status: str, PocketXMol 输入状态
        - docking_ready_warning: str, 分号分隔的 warning
    """

    dockem_input_status: str = ""
    emerald_id_input_status: str = ""
    pocketxmol_input_status: str = ""
    docking_ready_warning: str = ""


@dataclass
class MappingRow:
    """
    ligand_mapping.csv/jsonl 的一行。

    输出:
        - row: dict[str, Any], 字段与 Ligand/notes_of_dataset.md 中 ligand_mapping 字段表一致。
    """

    emdb_id: str = ""
    pdb_id: str = ""
    pdb_id_upper: str = ""
    candidate_id: int = -1
    ligand_class_id: int = -1
    ccd_id: str = ""
    chain_id: str = ""
    res_id: int = -1
    insertion_code: str = ""
    status: str = ""
    match_method: str = ""
    label_asym_id: str = ""
    pdb_strand_id: str = ""
    pdb_seq_num: str = ""
    auth_seq_num: str = ""
    pdb_ins_code: str = ""
    smiles: str = ""
    smiles_stereo: str = ""
    inchi: str = ""
    inchikey: str = ""
    full_cif_path: str = ""
    chemcomp_json_path: str = ""
    native_mol2_path: str = ""
    make_data_heavy_atoms: int = 0
    rcsb_cif_heavy_atoms: int = 0
    mol2_total_atoms: int = 0
    mol2_heavy_atoms: int = 0
    mol2_hydrogen_count: int = 0
    mol2_bond_count: int = 0
    mol2_has_charge_field: bool = False
    make_cif_coord_median: float | None = None
    make_cif_coord_max: float | None = None
    cif_mol2_coord_median: float | None = None
    cif_mol2_coord_max: float | None = None
    coord_status: str = ""
    validation_errors: str = ""
    docking_ready_warning: str = ""
    dockem_input_status: str = ""
    emerald_id_input_status: str = ""
    pocketxmol_input_status: str = ""
    download_url_full_cif: str = ""
    download_url_chemcomp: str = ""
    download_url_mol2: str = ""
    error_message: str = ""
    source_file: str = ""
    source_part_index: int = -1
    source_part_count: int = -1
    validation_detail: dict[str, Any] = field(default_factory=dict)
    download_attempts: dict[str, int] = field(default_factory=dict)
    rcsb_nonpoly_scheme_row: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """
        将 MappingRow 转为可写出 CSV/JSONL 的字典。

        输出:
            - result: dict[str, Any], 键为 mapping 字段名, 值为标量或嵌套字典
        """

        return asdict(self)


@dataclass
class PDBProcessResult:
    """
    单个 PDB worker 的处理结果。

    输入字段:
        - sample: SampleRef, 当前 PDB 样本引用
        - rows: list[MappingRow], 当前 PDB 产生的所有 mapping 行
    """

    sample: SampleRef
    rows: list[MappingRow]
