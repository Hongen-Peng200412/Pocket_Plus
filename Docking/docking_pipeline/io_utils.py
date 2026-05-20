from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from .records import InferenceSite, LigandCandidate


STANDALONE_METAL_CCD = {"MG", "MN", "ZN", "CA", "NA", "K", "FE", "CU", "CO", "NI", "CD", "HG", "CL", "BR", "IOD"}
MOL2_METAL_TYPES = {"FE", "MG", "MN", "ZN", "CA", "NA", "K", "CU", "CO", "NI", "CD", "HG", "MO", "W", "V"}
ROSETTA_NAME_DIGITS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def read_json(path: Path) -> Any:
    """
    读取 JSON 文件。
    输入参数:
        - path: Path, JSON 文件路径

    输出:
        - data: Any, `json.load` 返回的结构化对象
    """
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    """
    写入缩进 JSON 文件。
    输入参数:
        - path: Path, 输出 JSON 文件路径
        - data: Any, 可 JSON 序列化对象

    输出:
        - None
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def load_sites(sample_dir: Path) -> list[InferenceSite]:
    """
    从 `voxel_candidates.json` 读取预测 site。
    输入参数:
        - sample_dir: Path, 单个样本推理输出目录

    输出:
        - sites: list[InferenceSite], 按 instance_id 排序的预测位点列表
    """
    raw_sites = read_json(sample_dir / "voxel_candidates.json")
    sites = [
        InferenceSite(
            instance_id=int(item["instance_id"]),
            center_world_xyz=tuple(float(x) for x in item["center_world_xyz"]),
            score_mean=float(item["score_mean"]),
            score_max=float(item["score_max"]),
            voxel_count=int(item["voxel_count"]),
        )
        for item in raw_sites
    ]
    return sorted(sites, key=lambda site: site.instance_id)


def load_cache_meta(sample_dir: Path) -> dict[str, Any]:
    """
    读取 infer cache 中的 `meta_json`。
    输入参数:
        - sample_dir: Path, 单个样本推理输出目录, 其中必须包含 `summary.json`

    输出:
        - meta: dict[str, Any], cache 中解析出的 `meta_json`
    """
    summary = read_json(sample_dir / "summary.json")
    cache = np.load(summary["cache_path"], allow_pickle=True)
    return json.loads(cache["meta_json"].item())


def load_instance_label(sample_dir: Path) -> np.ndarray:
    """
    读取过滤后的 instance label。
    输入参数:
        - sample_dir: Path, 单个样本推理输出目录

    输出:
        - label: np.ndarray, (D, H, W), int, instance 编号图, 0 表示背景
    """
    npz = np.load(sample_dir / "instance_label_filtered.npz")
    key = "instance_label" if "instance_label" in npz.files else npz.files[0]
    return npz[key]


def load_voxel_transform(sample_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """
    读取体素到世界坐标的简化变换参数。
    输入参数:
        - sample_dir: Path, 单个样本推理输出目录

    输出:
        - transform: tuple[np.ndarray, np.ndarray], 包含:
            - origin: np.ndarray, (3,), cache 中的 origin
            - voxel_size: np.ndarray, (3,), cache 中的 voxel_size
    """
    summary = read_json(sample_dir / "summary.json")
    cache = np.load(summary["cache_path"], allow_pickle=True)
    return np.asarray(cache["origin"], dtype=float), np.asarray(cache["voxel_size"], dtype=float)


def load_resolution_table(path: Path) -> dict[str, dict[str, Any]]:
    """
    读取 EMDB-PDB-resolution CSV。
    输入参数:
        - path: Path, `/storage/penghongen/EMDB_PDB_resolution_3.5.csv`

    输出:
        - table: dict[str, dict[str, Any]], key 为小写 PDB ID, value 含 `resolution` 和 `emdb_id`
    """
    table: dict[str, dict[str, Any]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            for pdb_id in (row.get("fitted_pdbs") or "").split(","):
                key = pdb_id.strip().lower()
                if key:
                    table[key] = {"resolution": float(row["resolution"]), "emdb_id": row["emdb_id"]}
    return table


def mol2_internal_metals(path: Path) -> tuple[str, ...]:
    """
    从 mol2 atom type 字段检测 ligand 内部金属元素。
    输入参数:
        - path: Path, mol2 文件路径

    输出:
        - metals: tuple[str, ...], 去重排序后的金属元素列表
    """
    metals: set[str] = set()
    in_atom_block = False
    for line in path.read_text(errors="replace").splitlines():
        if line.startswith("@<TRIPOS>ATOM"):
            in_atom_block = True
            continue
        if line.startswith("@<TRIPOS>") and in_atom_block:
            break
        if in_atom_block and line.strip():
            parts = line.split()
            atom_type = parts[5].split(".")[0].upper()
            if atom_type in MOL2_METAL_TYPES:
                metals.add(atom_type)
    return tuple(sorted(metals))


def rosetta_ligand_name(index: int) -> str:
    """
    为 ligand 生成不碰撞 CCD 内置名的三字符 Rosetta residue 名称。

    输入参数:
        - index: int, 从 1 开始的 ligand 序号

    输出:
        - name: str, 长度为 3 的 Rosetta residue 名称, 例如 `L01`
    """
    if index < 100:
        return f"L{index:02d}"
    high = index // len(ROSETTA_NAME_DIGITS)
    low = index % len(ROSETTA_NAME_DIGITS)
    return f"Z{ROSETTA_NAME_DIGITS[high]}{ROSETTA_NAME_DIGITS[low]}"


def read_ligand_candidates(mapping_csv: Path, pdb_id: str) -> list[LigandCandidate]:
    """
    从 ligand mapping CSV 中读取当前样本可用于 docking 的候选 mol2。
    输入参数:
        - mapping_csv: Path, ligand mapping CSV
        - pdb_id: str, 当前样本 PDB ID, 大小写均可

    输出:
        - ligands: list[LigandCandidate], 已过滤独立金属离子和缺失 mol2 的候选列表
    """
    rows: list[dict[str, str]] = []
    with mapping_csv.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("pdb_id", "").lower() == pdb_id.lower():
                rows.append(row)

    valid_rows: list[dict[str, str]] = []
    for row in rows:
        mol2_path = Path(row.get("native_mol2_path", ""))
        heavy_atoms = int(row.get("mol2_heavy_atoms") or 0)
        ccd_id = row.get("ccd_id", "").upper()
        if row.get("status") != "PASS_HIGH":
            continue
        if row.get("emerald_id_input_status") != "FORMAT_OK_FOR_LIBRARY_ENTRY":
            continue
        if not mol2_path.exists() or heavy_atoms <= 1 or ccd_id in STANDALONE_METAL_CCD:
            continue
        valid_rows.append(row)

    ccd_counts: dict[str, int] = {}
    for row in valid_rows:
        ccd_id = row.get("ccd_id", "").upper()
        ccd_counts[ccd_id] = ccd_counts.get(ccd_id, 0) + 1

    ligands: list[LigandCandidate] = []
    for row in valid_rows:
        mol2_path = Path(row.get("native_mol2_path", ""))
        heavy_atoms = int(row.get("mol2_heavy_atoms") or 0)
        ccd_id = row.get("ccd_id", "").upper()
        ligand_index = len(ligands) + 1
        label = f"{ccd_id}_{ligand_index:02d}" if ccd_counts[ccd_id] > 1 else ccd_id
        rosetta_name = rosetta_ligand_name(ligand_index)
        ligands.append(
            LigandCandidate(
                pdb_id=pdb_id.lower(),
                ccd_id=ccd_id,
                label=label,
                rosetta_name=rosetta_name,
                mol2_path=mol2_path,
                heavy_atoms=heavy_atoms,
                internal_metals=mol2_internal_metals(mol2_path),
            )
        )
    return ligands


def cif_to_receptor_pdb(cif_path: Path, pdb_path: Path) -> int:
    """
    将 atom_site CIF 转成 Rosetta 更稳妥可读的 receptor-only PDB。
    输入参数:
        - cif_path: Path, 只读 CIF 来源
        - pdb_path: Path, 允许目录内输出 PDB

    输出:
        - atom_count: int, 写出的 ATOM 数量
    """
    lines: list[str] = []
    serial = 1
    last_chain: str | None = None
    with cif_path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if not line.startswith(("ATOM", "HETATM")):
                continue
            parts = line.split()
            if parts[0] == "HETATM":
                continue
            element = parts[2]
            atom = parts[3]
            alt = parts[4]
            resn = parts[5]
            chain = parts[6][0]
            resseq = int(parts[8])
            x, y, z = (float(parts[10]), float(parts[11]), float(parts[12]))
            occ, bfac = (float(parts[13]), float(parts[14]))
            if last_chain is not None and chain != last_chain:
                lines.append("TER")
            last_chain = chain
            atom_name = f" {atom[:4]:<3}" if len(element) == 1 and len(atom) < 4 else f"{atom[:4]:>4}"
            altloc = " " if alt in {".", "?"} else alt[0]
            lines.append(
                f"ATOM  {serial:5d} {atom_name}{altloc}{resn:>3} {chain:1}{resseq:4d}    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}{occ:6.2f}{bfac:6.2f}          {element:>2}"
            )
            serial += 1
    lines.extend(["TER", "END"])
    pdb_path.parent.mkdir(parents=True, exist_ok=True)
    pdb_path.write_text("\n".join(lines) + "\n", encoding="ascii")
    return serial - 1


def read_scorefile(path: Path) -> dict[str, str]:
    """
    读取 Rosetta scorefile 的最后一条 SCORE 记录。
    输入参数:
        - path: Path, scorefile 路径

    输出:
        - values: dict[str, str], key 为 scorefile 表头字段, value 为最后一条结果
    """
    rows = [line.split() for line in path.read_text(encoding="utf-8", errors="replace").splitlines() if line.startswith("SCORE:")]
    if len(rows) < 2:
        return {}
    return dict(zip(rows[0][1:], rows[-1][1:]))


def read_scorefile_rows(path: Path) -> list[dict[str, str]]:
    """
    读取 Rosetta scorefile 的所有 decoy SCORE 记录。

    输入参数:
        - path: Path, scorefile 文件路径

    输出:
        - rows: list[dict[str, str]], 每项为一个 decoy 的 score 字段字典; scorefile 不完整时返回空列表
    """
    lines = [line.split() for line in path.read_text(encoding="utf-8", errors="replace").splitlines() if line.startswith("SCORE:")]
    if len(lines) < 2:
        return []
    header = lines[0][1:]
    return [dict(zip(header, row[1:])) for row in lines[1:]]
