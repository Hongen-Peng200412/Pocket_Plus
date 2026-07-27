from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from .models import Mol2Info


ELEMENTS = {
    "H", "HE", "LI", "BE", "B", "C", "N", "O", "F", "NE",
    "NA", "MG", "AL", "SI", "P", "S", "CL", "AR", "K", "CA",
    "SC", "TI", "V", "CR", "MN", "FE", "CO", "NI", "CU", "ZN",
    "GA", "GE", "AS", "SE", "BR", "KR", "RB", "SR", "Y", "ZR",
    "NB", "MO", "TC", "RU", "RH", "PD", "AG", "CD", "IN", "SN",
    "SB", "TE", "I", "XE", "CS", "BA", "LA", "CE", "PR", "ND",
    "PM", "SM", "EU", "GD", "TB", "DY", "HO", "ER", "TM", "YB",
    "LU", "HF", "TA", "W", "RE", "OS", "IR", "PT", "AU", "HG",
    "TL", "PB", "BI", "PO", "AT", "RN", "FR", "RA", "AC", "TH",
    "PA", "U", "NP", "PU", "AM", "CM", "BK", "CF", "ES", "FM",
    "MD", "NO", "LR", "RF", "DB", "SG", "BH", "HS", "MT", "DS",
    "RG", "CN", "NH", "FL", "MC", "LV", "TS", "OG",
}


def infer_element(atom_name: str, atom_type: str) -> str:
    """
    从 mol2 atom name 和 atom type 中推断元素符号. 

    输入参数:
        - atom_name: str, mol2 ATOM 行中的 atom name
        - atom_type: str, mol2 ATOM 行中的 atom type, 如 C.3 / N.am

    输出:
        - element: str, 元素符号, 如 C / N / Cl
    """

    base = re.sub(r"[^A-Za-z]", "", atom_type.split(".")[0].strip()).upper()
    if base:
        if len(base) >= 2 and base[:2] in ELEMENTS:
            return base[:2].title()
        if base[:1] in ELEMENTS:
            return base[:1].upper()

    name = re.sub(r"[^A-Za-z]", "", atom_name.strip()).upper()
    if len(name) >= 2 and name[:2] in ELEMENTS:
        return name[:2].title()
    return name[:1].upper()


def parse_mol2(mol2_path: Path) -> Mol2Info:
    """
    解析 TRIPOS mol2 文件的 ATOM 和 BOND section. 

    输入参数:
        - mol2_path: Path, RCSB native ligand mol2 文件路径

    输出:
        - info: Mol2Info, atom/bond/charge/坐标的轻量解析结果
    """

    text = mol2_path.read_text(encoding="utf-8", errors="replace")
    atoms: list[dict[str, object]] = []
    bond_count = 0
    section = ""
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("@<TRIPOS>"):
            section = stripped.removeprefix("@<TRIPOS>").upper()
            continue
        if not stripped or stripped.startswith("#"):
            continue
        if section == "ATOM":
            parts = stripped.split()
            if len(parts) < 6:
                continue
            charge = None
            if len(parts) >= 9:
                try:
                    charge = float(parts[8])
                except ValueError:
                    charge = None
            element = infer_element(parts[1], parts[5])
            atoms.append(
                {
                    "coord": [float(parts[2]), float(parts[3]), float(parts[4])],
                    "element": element,
                    "charge": charge,
                }
            )
        elif section == "BOND":
            bond_count += 1

    coords = np.asarray([atom["coord"] for atom in atoms], dtype=float) if atoms else np.zeros((0, 3), dtype=float)
    elements = [str(atom["element"]) for atom in atoms]
    heavy_mask = np.asarray([element.upper() != "H" for element in elements], dtype=bool)
    charges = [atom["charge"] for atom in atoms]
    heavy_coords = coords[heavy_mask] if len(coords) else np.zeros((0, 3), dtype=float)
    heavy_elements = [element for element in elements if element.upper() != "H"]
    return Mol2Info(
        atom_count=len(atoms),
        heavy_atom_count=len(heavy_elements),
        hydrogen_count=len(elements) - len(heavy_elements),
        bond_count=bond_count,
        has_charge_field=bool(atoms) and all(charge is not None for charge in charges),
        heavy_coords=heavy_coords,
        heavy_elements=heavy_elements,
        raw_has_tripos_molecule="@<TRIPOS>MOLECULE" in text,
    )
