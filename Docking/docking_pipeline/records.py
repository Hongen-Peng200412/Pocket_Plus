from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class InferenceSite:
    """
    网络推理得到的单个 ligand instance。
    输入参数:
        - instance_id: int, `voxel_candidates.json` 中的 instance 编号
        - center_world_xyz: tuple[float, float, float], 世界坐标系下的预测中心
        - score_mean: float, instance 内体素预测分数均值
        - score_max: float, instance 内体素预测分数最大值
        - voxel_count: int, instance 体素数量

    输出:
        - InferenceSite: 不可变记录对象, 用于生成 docking center 与匹配特征
    """

    instance_id: int
    center_world_xyz: tuple[float, float, float]
    score_mean: float
    score_max: float
    voxel_count: int

    @property
    def site_id(self) -> str:
        """返回稳定 site 标签, 例如 `site001`。"""
        return f"site{self.instance_id:03d}"


@dataclass(frozen=True)
class LigandCandidate:
    """
    单个候选 mol2 记录。
    输入参数:
        - pdb_id: str, 样本 PDB ID, 小写
        - ccd_id: str, RCSB CCD ligand 名称
        - label: str, 当前流程内部唯一标签, 例如 `ATP_01`
        - rosetta_name: str, Rosetta 三字符 residue 名称, 例如 `L01`
        - mol2_path: Path, 原始 mol2 文件路径
        - heavy_atoms: int, mol2 重原子数
        - internal_metals: tuple[str, ...], ligand 内部金属元素, 不含独立金属离子条目

    输出:
        - LigandCandidate: 不可变记录对象, 用于 params 生成和矩阵构造
    """

    pdb_id: str
    ccd_id: str
    label: str
    rosetta_name: str
    mol2_path: Path
    heavy_atoms: int
    internal_metals: tuple[str, ...] = ()


@dataclass(frozen=True)
class ReceptorSource:
    """
    receptor 输入来源。
    输入参数:
        - name: str, `true_receptor` 或 `cryoatom_receptor`
        - cif_path: Path, 只读 CIF 来源路径
        - pdb_path: Path, 允许目录内派生出的 Rosetta 可读 PDB

    输出:
        - ReceptorSource: 不可变记录对象, 用于 complex 生成
    """

    name: str
    cif_path: Path
    pdb_path: Path


@dataclass(frozen=True)
class DockingJob:
    """
    一个 Rosetta docking job 的完整输入规格。
    输入参数:
        - pdb_id: str, 样本 PDB ID
        - site: InferenceSite, 当前预测位点
        - ligand: LigandCandidate, 当前候选 ligand
        - receptor: ReceptorSource, 当前 receptor 来源
        - complex_pdb: Path, 允许目录内的 receptor-ligand 复合物 PDB
        - params_path: Path, Rosetta ligand params 文件
        - xml_path: Path, RosettaScripts XML
        - output_dir: Path, 当前 job 输出目录
        - scorefile_name: str, scorefile 文件名

    输出:
        - DockingJob: 不可变记录对象, 用于命令构造和结果追踪
    """

    pdb_id: str
    site: InferenceSite
    ligand: LigandCandidate
    receptor: ReceptorSource
    complex_pdb: Path
    params_path: Path
    xml_path: Path
    output_dir: Path
    scorefile_name: str


@dataclass
class DockingResult:
    """
    Rosetta job 的解析结果。
    输入参数:
        - job: DockingJob, 对应输入规格
        - success: bool, 是否满足流程成功定义中的进程成功
        - returncode: int, Rosetta 进程返回码
        - seconds: float, 运行耗时
        - score_values: dict[str, str], best decoy 的 scorefile 字段
        - score_rows: list[dict[str, str]], scorefile 中所有 decoy 的字段
        - decoy_summary: dict[str, float], 数值字段的 best/mean/std 聚合
        - stdout_log: Path, stdout 日志路径
        - stderr_log: Path, stderr 日志路径

    输出:
        - DockingResult: 可变记录对象, 便于后处理补充字段
    """

    job: DockingJob
    success: bool
    returncode: int
    seconds: float
    score_values: dict[str, str]
    stdout_log: Path
    stderr_log: Path
    score_rows: list[dict[str, str]] = field(default_factory=list)
    decoy_summary: dict[str, float] = field(default_factory=dict)

    def numeric_score(self, key: str) -> float | None:
        """从 `score_values` 中读取一个浮点字段, 字段不存在时返回 None。"""
        value = self.score_values.get(key)
        return None if value is None else float(value)


@dataclass(frozen=True)
class PairScore:
    """
    一个 site-ligand pair 的匹配成本组成。
    输入参数:
        - site_id: str, 预测位点标签
        - ligand_label: str, ligand 候选标签
        - receptor_scope: str, `true_receptor`、`cryoatom_receptor` 或聚合范围
        - terms: dict[str, float], 成本项字典, 例如 `dG`、`shape`
        - combined_cost: float, 已归一化并加权后的总成本, 越低越好

    输出:
        - PairScore: 不可变记录对象, 用于 assignment
    """

    site_id: str
    ligand_label: str
    receptor_scope: str
    terms: dict[str, float]
    combined_cost: float


@dataclass(frozen=True)
class AssignmentResult:
    """
    site-ligand 匹配结果。
    输入参数:
        - receptor_scope: str, 匹配对应的 receptor 分数范围
        - total_cost: float, assignment 总成本
        - pairs: tuple[PairScore, ...], 真实 site-ligand 匹配边
        - unmatched_sites: tuple[str, ...], 未匹配预测位点
        - unmatched_ligands: tuple[str, ...], 未匹配 ligand
        - virtual_edges: tuple[dict[str, Any], ...], 涉及虚拟节点的边, 用于审计
        - solver: str, 本次 assignment 实际使用的求解器名称

    输出:
        - AssignmentResult: 不可变记录对象, 用于写入审计摘要
    """

    receptor_scope: str
    total_cost: float
    pairs: tuple[PairScore, ...]
    unmatched_sites: tuple[str, ...] = ()
    unmatched_ligands: tuple[str, ...] = ()
    virtual_edges: tuple[dict[str, Any], ...] = field(default_factory=tuple)
    solver: str = ""
