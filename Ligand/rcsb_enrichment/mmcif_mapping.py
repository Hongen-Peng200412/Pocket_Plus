from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from Bio.PDB.MMCIF2Dict import MMCIF2Dict

from .io_utils import normalize_missing
from .models import MakeDataLigand, NonpolySchemeRow, RCSBAtomSiteLigand, RCSBInstanceMatch
from .validation import nearest_neighbor_stats


def as_list(value: Any) -> list[Any]:
    """
    将 MMCIF2Dict 的标量或列表字段统一为列表. 

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
    读取 RCSB full CIF 为字段字典. 

    输入参数:
        - cif_path: Path, full CIF 文件路径

    输出:
        - cif_dict: dict[str, list[Any]], key 为 mmCIF 字段名, value 统一为列表
    """

    raw = MMCIF2Dict(str(cif_path))
    return {key: as_list(value) for key, value in raw.items()}


def parse_nonpoly_scheme(cif_dict: dict[str, list[Any]]) -> list[NonpolySchemeRow]:
    """
    解析 _pdbx_nonpoly_scheme 为结构化行. 

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
                source_scheme="nonpoly",
            )
        )
    rows.extend(parse_branch_scheme(cif_dict))
    return rows


def parse_branch_scheme(cif_dict: dict[str, list[Any]]) -> list[NonpolySchemeRow]:
    """
    解析 _pdbx_branch_scheme 为与 nonpoly scheme 兼容的结构化行. 

    输入参数:
        - cif_dict: dict[str, list[Any]], load_cif_dict 返回的 CIF 字段字典

    输出:
        - rows: list[NonpolySchemeRow], 每个支链糖/支链配体 monomer 一行
    """

    key = "_pdbx_branch_scheme.asym_id"
    if key not in cif_dict:
        return []
    n_rows = len(cif_dict[key])
    rows: list[NonpolySchemeRow] = []
    for i in range(n_rows):
        rows.append(
            NonpolySchemeRow(
                asym_id=normalize_missing(cif_dict.get("_pdbx_branch_scheme.asym_id", [""] * n_rows)[i]),
                mon_id=normalize_missing(cif_dict.get("_pdbx_branch_scheme.mon_id", [""] * n_rows)[i]).upper(),
                pdb_seq_num=normalize_missing(cif_dict.get("_pdbx_branch_scheme.pdb_seq_num", [""] * n_rows)[i]),
                auth_seq_num=normalize_missing(cif_dict.get("_pdbx_branch_scheme.auth_seq_num", [""] * n_rows)[i]),
                pdb_mon_id=normalize_missing(cif_dict.get("_pdbx_branch_scheme.pdb_mon_id", [""] * n_rows)[i]).upper(),
                auth_mon_id=normalize_missing(cif_dict.get("_pdbx_branch_scheme.auth_mon_id", [""] * n_rows)[i]).upper(),
                pdb_strand_id=normalize_missing(cif_dict.get("_pdbx_branch_scheme.pdb_asym_id", [""] * n_rows)[i]),
                pdb_ins_code="",
                source_scheme="branch",
            )
        )
    return rows


def match_make_data_ligand_to_rcsb(ligand: MakeDataLigand, rows: list[NonpolySchemeRow]) -> RCSBInstanceMatch:
    """
    将 Make_Data ligand instance 唯一匹配到 RCSB nonpoly scheme 行. 

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
        method = "BRANCH_PDB_ASYM_PDB_SEQ" if pdb_seq_hits[0].source_scheme == "branch" else "PDB_STRAND_PDB_SEQ"
        return RCSBInstanceMatch(matched=True, match_method=method, row=pdb_seq_hits[0])

    auth_seq_hits = [
        row
        for row in rows
        if comp_match(row)
        and row.pdb_strand_id == ligand.chain_id
        and row.auth_seq_num == str(ligand.res_id)
        and ins_match(row)
    ]
    if len(auth_seq_hits) == 1:
        method = "BRANCH_PDB_ASYM_AUTH_SEQ" if auth_seq_hits[0].source_scheme == "branch" else "PDB_STRAND_AUTH_SEQ"
        return RCSBInstanceMatch(matched=True, match_method=method, row=auth_seq_hits[0])

    return RCSBInstanceMatch(matched=False, match_method="FAILED", row=None)


def extract_rcsb_atom_site_ligand(cif_dict: dict[str, list[Any]], ligand: MakeDataLigand, match: RCSBInstanceMatch) -> RCSBAtomSiteLigand:
    """
    从 _atom_site 中提取 RCSB ligand instance 重原子. 

    输入参数:
        - cif_dict: dict[str, list[Any]], load_cif_dict 返回的 CIF 字段字典
        - ligand: MakeDataLigand, Make_Data ligand candidate
        - match: RCSBInstanceMatch, 已匹配的 RCSB nonpoly row

    输出:
        - atom_site: RCSBAtomSiteLigand, RCSB CIF 中该 ligand instance 的重原子坐标和元素
    """

    if not match.row or "_atom_site.id" not in cif_dict:
        return RCSBAtomSiteLigand(coords=np.zeros((0, 3), dtype=float), elements=[], atom_names=[])

    groups = atom_site_groups_for_comp(cif_dict, ligand.ccd_id)
    if match.match_method == "ATOM_SITE_COORD":
        coord_candidates = []
        for group in groups:
            if len(group["coords"]) != ligand.make_data_heavy_atoms:
                continue
            median_dist, max_dist = nearest_neighbor_stats(ligand.make_data_coords, group["coords"])
            if median_dist is None or max_dist is None:
                continue
            if median_dist <= 0.05 and max_dist <= 0.20:
                coord_candidates.append(group)
        if len(coord_candidates) == 1:
            group = coord_candidates[0]
            return RCSBAtomSiteLigand(coords=group["coords"], elements=group["elements"], atom_names=group["atom_names"])

    target_seq_ids = {
        value
        for value in {
            str(ligand.res_id),
            match.row.pdb_seq_num,
        }
        if value
    }
    for group in groups:
        atom_seq_ids = {group["label_seq_id"], group["auth_seq_id"]}
        if group["label_asym_id"] != match.row.asym_id:
            continue
        if match.row.pdb_ins_code and group["ins_code"] != match.row.pdb_ins_code:
            continue
        if not atom_seq_ids.intersection(target_seq_ids):
            continue
        return RCSBAtomSiteLigand(coords=group["coords"], elements=group["elements"], atom_names=group["atom_names"])

    return RCSBAtomSiteLigand(coords=np.zeros((0, 3), dtype=float), elements=[], atom_names=[])


def preferred_alt_ids(raw_alt_ids: list[str]) -> set[str]:
    """
    涓€涓?atom_site instance 鍐呴€夋嫨鍗曚釜 altLoc 鏋勮薄銆?
    杈撳叆鍙傛暟:
        - raw_alt_ids: list[str], 鍘熷 altLoc 鍒楄〃

    杈撳嚭:
        - alt_ids: set[str], 搴斾繚鐣欑殑 altLoc 鍊?
    """

    alt_id_set = set(normalize_missing(value) for value in raw_alt_ids)
    if "" in alt_id_set:
        for preferred in ("A", "1"):
            if preferred in alt_id_set:
                return {"", preferred}
        alternatives = sorted(value for value in alt_id_set if value)
        return {"", alternatives[0]} if alternatives else {""}
    for preferred in ("A", "1"):
        if preferred in alt_id_set:
            return {preferred}
    return {sorted(alt_id_set)[0]} if alt_id_set else {""}


def atom_site_groups_for_comp(cif_dict: dict[str, list[Any]], ccd_id: str) -> list[dict[str, Any]]:
    """
    从 _atom_site 中按 ligand instance 分组. 

    输入参数:
        - cif_dict: dict[str, list[Any]], load_cif_dict 返回的 CIF 字段字典
        - ccd_id: str, CCD ID

    输出:
        - groups: list[dict[str, Any]], 每个 dict 表示一个 atom_site 分组, 包含 label/auth asym、seq、坐标和元素
    """

    if "_atom_site.id" not in cif_dict:
        return []

    n_rows = len(cif_dict["_atom_site.id"])
    label_asym = cif_dict.get("_atom_site.label_asym_id", [""] * n_rows)
    auth_asym = cif_dict.get("_atom_site.auth_asym_id", [""] * n_rows)
    label_comp = cif_dict.get("_atom_site.label_comp_id", [""] * n_rows)
    auth_comp = cif_dict.get("_atom_site.auth_comp_id", [""] * n_rows)
    label_seq = cif_dict.get("_atom_site.label_seq_id", [""] * n_rows)
    auth_seq = cif_dict.get("_atom_site.auth_seq_id", [""] * n_rows)
    ins_code = cif_dict.get("_atom_site.pdbx_PDB_ins_code", [""] * n_rows)
    x_vals = cif_dict.get("_atom_site.Cartn_x", [""] * n_rows)
    y_vals = cif_dict.get("_atom_site.Cartn_y", [""] * n_rows)
    z_vals = cif_dict.get("_atom_site.Cartn_z", [""] * n_rows)
    type_symbol = cif_dict.get("_atom_site.type_symbol", [""] * n_rows)
    atom_name = cif_dict.get("_atom_site.label_atom_id", [""] * n_rows)
    alt_id = cif_dict.get("_atom_site.label_alt_id", ["."] * n_rows)

    grouped: dict[tuple[str, str, str, str, str], dict[str, Any]] = {}
    for i in range(n_rows):
        comp_ids = {normalize_missing(label_comp[i]).upper(), normalize_missing(auth_comp[i]).upper()}
        elem = normalize_missing(type_symbol[i]).title()
        alt = normalize_missing(alt_id[i])
        if ccd_id.upper() not in comp_ids:
            continue
        if elem.upper() == "H":
            continue

        key = (
            normalize_missing(label_asym[i]),
            normalize_missing(auth_asym[i]),
            normalize_missing(label_seq[i]),
            normalize_missing(auth_seq[i]),
            normalize_missing(ins_code[i]),
        )
        group = grouped.setdefault(
            key,
            {
                "label_asym_id": key[0],
                "auth_asym_id": key[1],
                "label_seq_id": key[2],
                "auth_seq_id": key[3],
                "ins_code": key[4],
                "coords": [],
                "elements": [],
                "atom_names": [],
                "alt_ids": [],
            },
        )
        group["coords"].append([float(x_vals[i]), float(y_vals[i]), float(z_vals[i])])
        group["elements"].append(elem)
        group["atom_names"].append(normalize_missing(atom_name[i]))
        group["alt_ids"].append(alt)

    groups: list[dict[str, Any]] = []
    for group in grouped.values():
        keep_alt_ids = preferred_alt_ids(group["alt_ids"])
        keep_indices = [i for i, value in enumerate(group["alt_ids"]) if value in keep_alt_ids]
        group["coords"] = np.asarray([group["coords"][i] for i in keep_indices], dtype=float)
        group["elements"] = [group["elements"][i] for i in keep_indices]
        group["atom_names"] = [group["atom_names"][i] for i in keep_indices]
        group["alt_ids"] = [group["alt_ids"][i] for i in keep_indices]
        groups.append(group)
    return groups


def match_atom_site_ligand(ligand: MakeDataLigand, cif_dict: dict[str, list[Any]]) -> RCSBInstanceMatch:
    """
    当 scheme 表缺失时, 用 _atom_site 直接兜底匹配 ligand instance. 

    输入参数:
        - ligand: MakeDataLigand, Make_Data ligand candidate
        - cif_dict: dict[str, list[Any]], load_cif_dict 返回的 CIF 字段字典

    输出:
        - match: RCSBInstanceMatch, 成功时 row 为由 atom_site 合成的 NonpolySchemeRow
    """

    groups = atom_site_groups_for_comp(cif_dict, ligand.ccd_id)
    res_id = str(ligand.res_id)

    coord_candidates = []
    for group in groups:
        if len(group["coords"]) != ligand.make_data_heavy_atoms:
            continue
        median_dist, max_dist = nearest_neighbor_stats(ligand.make_data_coords, group["coords"])
        if median_dist is None or max_dist is None:
            continue
        if median_dist <= 0.05 and max_dist <= 0.20:
            coord_candidates.append(group)
    if len(coord_candidates) == 1:
        return _atom_site_group_to_match(ligand, coord_candidates[0], "ATOM_SITE_COORD")

    exact_rules = [
        ("ATOM_SITE_AUTH_ASYM_AUTH_SEQ", lambda group: group["auth_asym_id"] == ligand.chain_id and group["auth_seq_id"] == res_id),
        ("ATOM_SITE_AUTH_ASYM_LABEL_SEQ", lambda group: group["auth_asym_id"] == ligand.chain_id and group["label_seq_id"] == res_id),
        ("ATOM_SITE_LABEL_ASYM_LABEL_SEQ", lambda group: group["label_asym_id"] == ligand.chain_id and group["label_seq_id"] == res_id),
        ("ATOM_SITE_LABEL_ASYM_AUTH_SEQ", lambda group: group["label_asym_id"] == ligand.chain_id and group["auth_seq_id"] == res_id),
    ]
    for method, predicate in exact_rules:
        exact_groups = [
            group
            for group in groups
            if predicate(group) and (not ligand.insertion_code or group["ins_code"] == ligand.insertion_code)
        ]
        if len(exact_groups) == 1:
            return _atom_site_group_to_match(ligand, exact_groups[0], method)

    return RCSBInstanceMatch(matched=False, match_method="FAILED", row=None)


def _atom_site_group_to_match(ligand: MakeDataLigand, group: dict[str, Any], method: str) -> RCSBInstanceMatch:
    """
    将 atom_site 分组包装成 RCSBInstanceMatch. 

    输入参数:
        - ligand: MakeDataLigand, Make_Data ligand candidate
        - group: dict[str, Any], atom_site_groups_for_comp 返回的一个分组
        - method: str, 匹配方法名

    输出:
        - match: RCSBInstanceMatch, row 为合成 NonpolySchemeRow
    """

    res_id = str(ligand.res_id)
    pdb_seq_num = group["auth_seq_id"] if group["auth_seq_id"] == res_id else group["label_seq_id"]
    auth_seq_num = group["auth_seq_id"] or group["label_seq_id"]
    row = NonpolySchemeRow(
        asym_id=group["label_asym_id"],
        mon_id=ligand.ccd_id,
        pdb_seq_num=pdb_seq_num,
        auth_seq_num=auth_seq_num,
        pdb_mon_id=ligand.ccd_id,
        auth_mon_id=ligand.ccd_id,
        pdb_strand_id=group["auth_asym_id"] or group["label_asym_id"],
        pdb_ins_code=group["ins_code"],
        source_scheme="atom_site",
    )
    return RCSBInstanceMatch(matched=True, match_method=method, row=row)
