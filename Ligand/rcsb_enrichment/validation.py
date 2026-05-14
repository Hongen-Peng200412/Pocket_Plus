from __future__ import annotations

from collections import Counter

import numpy as np
from scipy.spatial import cKDTree

from . import config
from .models import ChemCompDescriptors, MakeDataLigand, Mol2Info, RCSBAtomSiteLigand, ToolReadiness, ValidationMetrics


def element_counts(elements: list[str]) -> dict[str, int]:
    """
    统计元素组成。

    输入参数:
        - elements: list[str], 元素符号列表

    输出:
        - counts: dict[str, int], 元素到出现次数的映射
    """

    return dict(sorted(Counter(elements).items()))


def nearest_neighbor_stats(query_coords: np.ndarray, target_coords: np.ndarray) -> tuple[float | None, float | None]:
    """
    计算无元素约束的最近邻距离统计。

    输入参数:
        - query_coords: np.ndarray, (N, 3), 查询坐标
        - target_coords: np.ndarray, (M, 3), 目标坐标

    输出:
        - median_dist: float | None, 最近邻距离 median
        - max_dist: float | None, 最近邻距离 max
    """

    if len(query_coords) == 0 or len(target_coords) == 0:
        return None, None
    tree = cKDTree(target_coords)
    dists, _ = tree.query(query_coords, k=1)
    return float(np.median(dists)), float(np.max(dists))


def nearest_same_element_stats(query_coords: np.ndarray, query_elements: list[str], target_coords: np.ndarray, target_elements: list[str]) -> tuple[float | None, float | None]:
    """
    计算同元素约束的最近邻距离统计。

    输入参数:
        - query_coords: np.ndarray, (N, 3), 查询坐标
        - query_elements: list[str], 长度 N, 查询坐标对应元素
        - target_coords: np.ndarray, (M, 3), 目标坐标
        - target_elements: list[str], 长度 M, 目标坐标对应元素

    输出:
        - median_dist: float | None, 同元素最近邻距离 median
        - max_dist: float | None, 同元素最近邻距离 max
    """

    distances: list[float] = []
    for element in sorted(set(query_elements)):
        query_idx = [i for i, current in enumerate(query_elements) if current == element]
        target_idx = [i for i, current in enumerate(target_elements) if current == element]
        if not query_idx or len(query_idx) > len(target_idx):
            return None, None
        tree = cKDTree(target_coords[target_idx])
        dists, _ = tree.query(query_coords[query_idx], k=1)
        distances.extend(float(value) for value in np.atleast_1d(dists))
    if not distances:
        return None, None
    return float(np.median(distances)), float(np.max(distances))


def validate_ligand_pair(make_data_ligand: MakeDataLigand, rcsb_atom_site: RCSBAtomSiteLigand, mol2_info: Mol2Info, descriptors: ChemCompDescriptors) -> ValidationMetrics:
    """
    执行 Make_Data、RCSB CIF 和 RCSB native mol2 的一致性校验。

    输入参数:
        - make_data_ligand: MakeDataLigand, Make_Data ligand candidate
        - rcsb_atom_site: RCSBAtomSiteLigand, RCSB CIF 中同一 ligand instance
        - mol2_info: Mol2Info, native mol2 解析结果
        - descriptors: ChemCompDescriptors, RCSB chemical component descriptor; 当前用于确认 SMILES 存在

    输出:
        - metrics: ValidationMetrics, 重原子数、元素组成、坐标距离和错误列表
    """

    metrics = ValidationMetrics(
        make_data_heavy_atoms=make_data_ligand.make_data_heavy_atoms,
        rcsb_cif_heavy_atoms=len(rcsb_atom_site.elements),
        mol2_heavy_atoms=mol2_info.heavy_atom_count,
    )
    metrics.make_cif_coord_median, metrics.make_cif_coord_max = nearest_neighbor_stats(
        make_data_ligand.make_data_coords,
        rcsb_atom_site.coords,
    )
    metrics.cif_mol2_coord_median, metrics.cif_mol2_coord_max = nearest_same_element_stats(
        rcsb_atom_site.coords,
        rcsb_atom_site.elements,
        mol2_info.heavy_coords,
        mol2_info.heavy_elements,
    )

    errors: list[str] = []
    if not descriptors.smiles and not descriptors.smiles_stereo:
        errors.append("RCSB_SMILES_EMPTY")
    if make_data_ligand.make_data_heavy_atoms != len(rcsb_atom_site.elements):
        errors.append("MAKE_DATA_VS_RCSB_CIF_HEAVY_COUNT_MISMATCH")
    if len(rcsb_atom_site.elements) != mol2_info.heavy_atom_count:
        errors.append("RCSB_CIF_VS_MOL2_HEAVY_COUNT_MISMATCH")
    if element_counts(rcsb_atom_site.elements) != element_counts(mol2_info.heavy_elements):
        errors.append("RCSB_CIF_VS_MOL2_ELEMENT_MISMATCH")
    if metrics.make_cif_coord_median is None or metrics.make_cif_coord_max is None:
        errors.append("MAKE_DATA_VS_RCSB_CIF_COORD_MISSING")
    elif metrics.make_cif_coord_median > config.COORD_MEDIAN_THRESHOLD or metrics.make_cif_coord_max > config.COORD_MAX_THRESHOLD:
        errors.append("MAKE_DATA_VS_RCSB_CIF_COORD_MISMATCH")
    if metrics.cif_mol2_coord_median is None or metrics.cif_mol2_coord_max is None:
        errors.append("RCSB_CIF_VS_MOL2_COORD_MISSING")
    elif metrics.cif_mol2_coord_median > config.COORD_MEDIAN_THRESHOLD or metrics.cif_mol2_coord_max > config.COORD_MAX_THRESHOLD:
        errors.append("RCSB_CIF_VS_MOL2_COORD_MISMATCH")

    metrics.validation_errors = errors
    metrics.coord_status = "PASS" if not errors else "FAIL"
    return metrics


def evaluate_tool_readiness(mol2_info: Mol2Info | None, has_smiles: bool, pair_valid: bool) -> ToolReadiness:
    """
    判断 DockEM、EMERALD-ID 和 PocketXMol 的输入格式状态。

    输入参数:
        - mol2_info: Mol2Info | None, native mol2 解析结果; 缺失时为 None
        - has_smiles: bool, 是否有 RCSB SMILES 或 stereo SMILES
        - pair_valid: bool, pair 是否通过 PASS_HIGH 校验

    输出:
        - readiness: ToolReadiness, 三个工具的格式状态和 warning
    """

    if not pair_valid or mol2_info is None:
        return ToolReadiness(
            dockem_input_status=config.NOT_READY_NO_VALID_NATIVE_MOL2_PAIR,
            emerald_id_input_status=config.NOT_READY_NO_VALID_NATIVE_MOL2_PAIR,
            pocketxmol_input_status=config.NOT_READY_NO_VALID_SMILES_OR_STRUCTURE,
        )

    warnings: list[str] = []
    if mol2_info.hydrogen_count == 0:
        warnings.append("MOL2_HAS_NO_HYDROGEN")
    if mol2_info.bond_count == 0:
        warnings.append("MOL2_HAS_NO_BONDS")
    if not mol2_info.has_charge_field:
        warnings.append("MOL2_HAS_NO_COMPLETE_CHARGE_FIELD")

    return ToolReadiness(
        dockem_input_status=config.FORMAT_OK_WITH_WARNING if warnings else config.FORMAT_OK,
        emerald_id_input_status=config.FORMAT_OK_FOR_LIBRARY_ENTRY,
        pocketxmol_input_status=config.FORMAT_OK_SMILES_AND_STRUCTURE if has_smiles else config.NOT_READY_NO_VALID_SMILES_OR_STRUCTURE,
        docking_ready_warning=";".join(warnings),
    )
