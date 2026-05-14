from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from Bio.PDB.MMCIF2Dict import MMCIF2Dict

from .io_utils import normalize_missing
from .models import MakeDataLigand, NonpolySchemeRow, RCSBAtomSiteLigand, RCSBInstanceMatch


def as_list(value: Any) -> list[Any]:
    """
    将 MMCIF2Dict 的标量或列表字段统一为列表。

    输入参数:
        - value: Any, MMCIF2Dict 返回的字段值

    输出:
        - values: list[Any], 列表形式字段值
    """

    if isinstance(value, list):
        return value
    return [value]


def load_cif_dict(cif_path: Path) -> dict[str, list[Any]]:
    """
    读取 RCSB full CIF 为字段字典。

    输入参数:
        - cif_path: Path, full CIF 文件路径

    输出:
        - cif_dict: dict[str, list[Any]], key 为 mmCIF 字段名, value 统一为列表
    """

    raw = MMCIF2Dict(str(cif_path))
    return {key: as_list(value) for key, value in raw.items()}


def parse_nonpoly_scheme(cif_dict: dict[str, list[Any]]) -> list[NonpolySchemeRow]:
    """
    解析 _pdbx_nonpoly_scheme 为结构化行。

    输入参数:
        - cif_dict: dict[str, list[Any]], load_cif_dict 返回的 CIF 字段字典

    输出:
        - rows: list[NonpolySchemeRow], 每个非聚合物 ligand instance 一行
    """

    key = "_pdbx_nonpoly_scheme.asym_id"
    if key not in cif_dict:
        return []
    n_rows = len(cif_dict[key])
    rows: list[NonpolySchemeRow] = []
    for i in range(n_rows):
        rows.append(
            NonpolySchemeRow(
                asym_id=normalize_missing(cif_dict.get("_pdbx_nonpoly_scheme.asym_id", [""] * n_rows)[i]),
                mon_id=normalize_missing(cif_dict.get("_pdbx_nonpoly_scheme.mon_id", [""] * n_rows)[i]).upper(),
                pdb_seq_num=normalize_missing(cif_dict.get("_pdbx_nonpoly_scheme.pdb_seq_num", [""] * n_rows)[i]),
                auth_seq_num=normalize_missing(cif_dict.get("_pdbx_nonpoly_scheme.auth_seq_num", [""] * n_rows)[i]),
                pdb_mon_id=normalize_missing(cif_dict.get("_pdbx_nonpoly_scheme.pdb_mon_id", [""] * n_rows)[i]).upper(),
                auth_mon_id=normalize_missing(cif_dict.get("_pdbx_nonpoly_scheme.auth_mon_id", [""] * n_rows)[i]).upper(),
                pdb_strand_id=normalize_missing(cif_dict.get("_pdbx_nonpoly_scheme.pdb_strand_id", [""] * n_rows)[i]),
                pdb_ins_code=normalize_missing(cif_dict.get("_pdbx_nonpoly_scheme.pdb_ins_code", [""] * n_rows)[i]),
            )
        )
    return rows


def match_make_data_ligand_to_rcsb(ligand: MakeDataLigand, rows: list[NonpolySchemeRow]) -> RCSBInstanceMatch:
    """
    将 Make_Data ligand instance 唯一匹配到 RCSB nonpoly scheme 行。

    输入参数:
        - ligand: MakeDataLigand, Make_Data ligand candidate
        - rows: list[NonpolySchemeRow], RCSB nonpoly scheme 行

    输出:
        - match: RCSBInstanceMatch, 匹配结果; 成功时 row 不为 None
    """

    def comp_match(row: NonpolySchemeRow) -> bool:
        return ligand.ccd_id in {row.mon_id, row.pdb_mon_id, row.auth_mon_id}

    def ins_match(row: NonpolySchemeRow) -> bool:
        return row.pdb_ins_code == ligand.insertion_code

    pdb_seq_hits = [
        row
        for row in rows
        if comp_match(row)
        and row.pdb_strand_id == ligand.chain_id
        and row.pdb_seq_num == str(ligand.res_id)
        and ins_match(row)
    ]
    if len(pdb_seq_hits) == 1:
        return RCSBInstanceMatch(matched=True, match_method="PDB_STRAND_PDB_SEQ", row=pdb_seq_hits[0])

    auth_seq_hits = [
        row
        for row in rows
        if comp_match(row)
        and row.pdb_strand_id == ligand.chain_id
        and row.auth_seq_num == str(ligand.res_id)
        and ins_match(row)
    ]
    if len(auth_seq_hits) == 1:
        return RCSBInstanceMatch(matched=True, match_method="PDB_STRAND_AUTH_SEQ", row=auth_seq_hits[0])

    return RCSBInstanceMatch(matched=False, match_method="FAILED", row=None)


def extract_rcsb_atom_site_ligand(cif_dict: dict[str, list[Any]], ligand: MakeDataLigand, match: RCSBInstanceMatch) -> RCSBAtomSiteLigand:
    """
    从 _atom_site 中提取 RCSB ligand instance 重原子。

    输入参数:
        - cif_dict: dict[str, list[Any]], load_cif_dict 返回的 CIF 字段字典
        - ligand: MakeDataLigand, Make_Data ligand candidate
        - match: RCSBInstanceMatch, 已匹配的 RCSB nonpoly row

    输出:
        - atom_site: RCSBAtomSiteLigand, RCSB CIF 中该 ligand instance 的重原子坐标和元素
    """

    if not match.row or "_atom_site.id" not in cif_dict:
        return RCSBAtomSiteLigand(coords=np.zeros((0, 3), dtype=float), elements=[], atom_names=[])

    n_rows = len(cif_dict["_atom_site.id"])
    label_asym = cif_dict.get("_atom_site.label_asym_id", [""] * n_rows)
    label_comp = cif_dict.get("_atom_site.label_comp_id", [""] * n_rows)
    auth_comp = cif_dict.get("_atom_site.auth_comp_id", [""] * n_rows)
    auth_seq = cif_dict.get("_atom_site.auth_seq_id", [""] * n_rows)
    label_seq = cif_dict.get("_atom_site.label_seq_id", [""] * n_rows)
    x_vals = cif_dict.get("_atom_site.Cartn_x", [""] * n_rows)
    y_vals = cif_dict.get("_atom_site.Cartn_y", [""] * n_rows)
    z_vals = cif_dict.get("_atom_site.Cartn_z", [""] * n_rows)
    type_symbol = cif_dict.get("_atom_site.type_symbol", [""] * n_rows)
    atom_name = cif_dict.get("_atom_site.label_atom_id", [""] * n_rows)
    alt_id = cif_dict.get("_atom_site.label_alt_id", ["."] * n_rows)

    coords: list[list[float]] = []
    elements: list[str] = []
    atom_names: list[str] = []

    for i in range(n_rows):
        comp_ids = {
            normalize_missing(label_comp[i]).upper(),
            normalize_missing(auth_comp[i]).upper(),
        }
        seq_ids = {
            normalize_missing(auth_seq[i]),
            normalize_missing(label_seq[i]),
            match.row.pdb_seq_num,
            match.row.auth_seq_num,
        }
        alt = normalize_missing(alt_id[i])
        elem = normalize_missing(type_symbol[i]).title()
        if normalize_missing(label_asym[i]) != match.row.asym_id:
            continue
        if ligand.ccd_id not in comp_ids:
            continue
        if str(ligand.res_id) not in seq_ids and match.row.pdb_seq_num not in seq_ids:
            continue
        if alt and alt not in {"A", "1"}:
            continue
        if elem.upper() == "H":
            continue
        coords.append([float(x_vals[i]), float(y_vals[i]), float(z_vals[i])])
        elements.append(elem)
        atom_names.append(normalize_missing(atom_name[i]))

    coord_array = np.asarray(coords, dtype=float) if coords else np.zeros((0, 3), dtype=float)
    return RCSBAtomSiteLigand(coords=coord_array, elements=elements, atom_names=atom_names)
