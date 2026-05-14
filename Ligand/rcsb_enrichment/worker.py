from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from . import config
from .io_utils import load_make_data_ligands
from .mmcif_mapping import extract_rcsb_atom_site_ligand, load_cif_dict, match_make_data_ligand_to_rcsb, parse_nonpoly_scheme
from .models import ChemCompDescriptors, MappingRow, Mol2Info, PDBProcessResult, SampleRef
from .mol2_parser import parse_mol2
from .rcsb_client import RCSBClient
from .validation import evaluate_tool_readiness, validate_ligand_pair


def empty_mol2_info() -> Mol2Info:
    """
    构造空 mol2 信息。

    输出:
        - info: Mol2Info, 所有计数为 0 的占位对象
    """

    import numpy as np

    return Mol2Info(
        atom_count=0,
        heavy_atom_count=0,
        hydrogen_count=0,
        bond_count=0,
        has_charge_field=False,
        heavy_coords=np.zeros((0, 3), dtype=float),
        heavy_elements=[],
        raw_has_tripos_molecule=False,
    )


def load_descriptors(path: Path) -> ChemCompDescriptors:
    """
    从 RCSB chemcomp JSON 中读取 SMILES/InChI 描述符。

    输入参数:
        - path: Path, chemcomp JSON 文件路径

    输出:
        - descriptors: ChemCompDescriptors, 化学描述符字段
    """

    data = json.loads(path.read_text(encoding="utf-8"))
    descriptor = data.get("rcsb_chem_comp_descriptor", {}) or {}
    chem_comp = data.get("chem_comp", {}) or {}
    return ChemCompDescriptors(
        smiles=descriptor.get("SMILES", "") or "",
        smiles_stereo=descriptor.get("SMILES_stereo", "") or "",
        inchi=descriptor.get("InChI", "") or "",
        inchikey=descriptor.get("InChIKey", "") or "",
        formula=str(chem_comp.get("formula", "") or ""),
        formula_weight=str(chem_comp.get("formula_weight", "") or ""),
    )


def base_row(sample: SampleRef, ligand) -> MappingRow:
    """
    构造包含 Make_Data 基本字段的 MappingRow。

    输入参数:
        - sample: SampleRef, 当前 PDB 样本引用
        - ligand: MakeDataLigand, 当前 Make_Data ligand candidate

    输出:
        - row: MappingRow, 已填充本地字段的 mapping 行
    """

    return MappingRow(
        emdb_id=sample.emdb_id,
        pdb_id=sample.pdb_id,
        pdb_id_upper=sample.pdb_id_upper,
        candidate_id=ligand.candidate_id,
        ligand_class_id=ligand.ligand_class_id,
        ccd_id=ligand.ccd_id,
        chain_id=ligand.chain_id,
        res_id=ligand.res_id,
        insertion_code=ligand.insertion_code,
        make_data_heavy_atoms=ligand.make_data_heavy_atoms,
        download_url_full_cif=config.RCSB_FULL_CIF_URL.format(pdb_id=sample.pdb_id_upper),
        download_url_chemcomp=config.RCSB_CHEMCOMP_URL.format(ccd_id=ligand.ccd_id),
    )


def process_one_pdb(sample: SampleRef, output_root: Path, force_download: bool, max_retries: int, retry_sleep: float) -> PDBProcessResult:
    """
    处理一个 PDB 样本。

    输入参数:
        - sample: SampleRef, 待处理 PDB 样本
        - output_root: Path, 输出根目录
        - force_download: bool, 是否覆盖已有下载文件
        - max_retries: int, 下载最大重试次数
        - retry_sleep: float, 重试间隔秒数

    输出:
        - result: PDBProcessResult, 当前 PDB 的所有 mapping 行
    """

    client = RCSBClient(output_root=output_root, force_download=force_download, max_retries=max_retries, retry_sleep=retry_sleep)
    rows: list[MappingRow] = []
    full_cif = client.download_full_cif(sample.pdb_id_upper)

    try:
        cif_dict = load_cif_dict(full_cif.path)
        nonpoly_rows = parse_nonpoly_scheme(cif_dict)
        ligands = load_make_data_ligands(sample.parsed_dir)
    except Exception as exc:
        rows.append(
            MappingRow(
                emdb_id=sample.emdb_id,
                pdb_id=sample.pdb_id,
                pdb_id_upper=sample.pdb_id_upper,
                candidate_id=-1,
                status=config.PDB_LEVEL_FAILED,
                full_cif_path=str(full_cif.path),
                download_url_full_cif=full_cif.url,
                download_attempts={"full_cif": full_cif.attempts},
                error_message=repr(exc) if not full_cif.error_message else full_cif.error_message,
            )
        )
        return PDBProcessResult(sample=sample, rows=rows)

    for ligand in ligands:
        row = base_row(sample, ligand)
        row.full_cif_path = str(full_cif.path)
        row.download_attempts["full_cif"] = full_cif.attempts

        if ligand.ligand_class_id != config.TARGET_LIGAND_CLASS_ID:
            row.status = config.SKIPPED_NON_SMALL_MOLECULE
            rows.append(row)
            continue

        match = match_make_data_ligand_to_rcsb(ligand, nonpoly_rows)
        row.match_method = match.match_method
        if not match.matched or match.row is None:
            row.status = config.RCSB_INSTANCE_MATCH_FAILED
            row.error_message = "无法用 ccd_id + chain_id + res_id + insertion_code 唯一匹配 _pdbx_nonpoly_scheme。"
            rows.append(row)
            continue

        row.label_asym_id = match.row.asym_id
        row.pdb_strand_id = match.row.pdb_strand_id
        row.pdb_seq_num = match.row.pdb_seq_num
        row.auth_seq_num = match.row.auth_seq_num
        row.pdb_ins_code = match.row.pdb_ins_code
        row.rcsb_nonpoly_scheme_row = asdict(match.row)

        chemcomp = client.download_chemcomp(ligand.ccd_id)
        row.chemcomp_json_path = str(chemcomp.path)
        row.download_attempts["chemcomp"] = chemcomp.attempts
        if chemcomp.error_message:
            row.status = config.RCSB_CHEMCOMP_DOWNLOAD_FAILED
            row.error_message = chemcomp.error_message
            rows.append(row)
            continue

        descriptors = load_descriptors(chemcomp.path)
        row.smiles = descriptors.smiles
        row.smiles_stereo = descriptors.smiles_stereo
        row.inchi = descriptors.inchi
        row.inchikey = descriptors.inchikey
        if not descriptors.smiles and not descriptors.smiles_stereo:
            row.status = config.RCSB_SMILES_MISSING
            row.error_message = "RCSB chemcomp descriptor 中没有 SMILES 或 SMILES_stereo。"
            rows.append(row)
            continue

        mol2 = client.download_native_mol2(sample.pdb_id, match.row.asym_id, match.row.pdb_seq_num, ligand.ccd_id, ligand.candidate_id)
        row.native_mol2_path = str(mol2.path)
        row.download_url_mol2 = mol2.url
        row.download_attempts["mol2"] = mol2.attempts
        if mol2.error_message:
            row.status = config.RCSB_NATIVE_MOL2_MISSING
            row.error_message = mol2.error_message
            rows.append(row)
            continue

        mol2_info = parse_mol2(mol2.path)
        row.mol2_total_atoms = mol2_info.atom_count
        row.mol2_heavy_atoms = mol2_info.heavy_atom_count
        row.mol2_hydrogen_count = mol2_info.hydrogen_count
        row.mol2_bond_count = mol2_info.bond_count
        row.mol2_has_charge_field = mol2_info.has_charge_field
        if not mol2_info.raw_has_tripos_molecule or mol2_info.atom_count == 0:
            row.status = config.RCSB_NATIVE_MOL2_MISSING
            row.error_message = "RCSB ModelServer 返回的 mol2 不包含有效 @<TRIPOS>MOLECULE / ATOM section。"
            rows.append(row)
            continue

        rcsb_atom_site = extract_rcsb_atom_site_ligand(cif_dict, ligand, match)
        metrics = validate_ligand_pair(ligand, rcsb_atom_site, mol2_info, descriptors)
        row.rcsb_cif_heavy_atoms = metrics.rcsb_cif_heavy_atoms
        row.make_cif_coord_median = metrics.make_cif_coord_median
        row.make_cif_coord_max = metrics.make_cif_coord_max
        row.cif_mol2_coord_median = metrics.cif_mol2_coord_median
        row.cif_mol2_coord_max = metrics.cif_mol2_coord_max
        row.coord_status = metrics.coord_status
        row.validation_errors = ";".join(metrics.validation_errors)
        row.validation_detail = {
            "make_data_heavy_atoms": metrics.make_data_heavy_atoms,
            "rcsb_cif_heavy_atoms": metrics.rcsb_cif_heavy_atoms,
            "mol2_heavy_atoms": metrics.mol2_heavy_atoms,
            "make_cif_coord_median": metrics.make_cif_coord_median,
            "make_cif_coord_max": metrics.make_cif_coord_max,
            "cif_mol2_coord_median": metrics.cif_mol2_coord_median,
            "cif_mol2_coord_max": metrics.cif_mol2_coord_max,
            "coord_status": metrics.coord_status,
            "validation_errors": metrics.validation_errors,
        }
        row.status = config.PASS_HIGH if not metrics.validation_errors else config.VALIDATION_FAILED

        readiness = evaluate_tool_readiness(mol2_info, bool(row.smiles or row.smiles_stereo), row.status == config.PASS_HIGH)
        row.dockem_input_status = readiness.dockem_input_status
        row.emerald_id_input_status = readiness.emerald_id_input_status
        row.pocketxmol_input_status = readiness.pocketxmol_input_status
        row.docking_ready_warning = readiness.docking_ready_warning
        rows.append(row)

    return PDBProcessResult(sample=sample, rows=rows)
