from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ServerPaths:
    """
    服务器路径约定. 
    输入参数:
        - allowed_root: Path, 本任务唯一允许写入的服务器目录
        - inference_root: Path, 推理结果目录
        - ligand_mapping_csv: Path, ligand enrichment mapping CSV
        - resolution_csv: Path, EMDB-PDB-resolution CSV
        - rosetta_bin: Path, 已验证的 Rosetta wrapper
        - rosetta_database: Path, Rosetta database
        - molfile_to_params: Path, Rosetta ligand 参数脚本
        - true_receptor_root: Path, 去配体真实 receptor CIF 目录
        - cryoatom_root: Path, cryoatom 预测结构根目录

    输出:
        - ServerPaths: 不可变配置对象
    """

    allowed_root: Path
    inference_root: Path
    ligand_mapping_csv: Path
    resolution_csv: Path
    rosetta_bin: Path
    rosetta_database: Path
    molfile_to_params: Path
    true_receptor_root: Path
    cryoatom_root: Path

    @staticmethod
    def default() -> "ServerPaths":
        """返回当前服务器上已经验证过的路径配置. """
        allowed = Path("/home/penghongen") / "\u5206\u5b50\u5bf9\u63a5\u5c1d\u8bd5"
        rosetta_main = Path("/home/penghongen/software/rosetta/source_build/rosetta.source.release-430/main")
        return ServerPaths(
            allowed_root=allowed,
            inference_root=Path(
                "/home/penghongen/My_Project/feedback_plus/infer_out/"
                "ligand_base2_new/stage2_threshold_component_policy"
            ),
            ligand_mapping_csv=Path("/storage/penghongen/CIF_Ligand/mapping/ligand_mapping.csv"),
            resolution_csv=Path("/storage/penghongen/EMDB_PDB_resolution_3.5.csv"),
            rosetta_bin=Path("/home/penghongen/software/rosetta/bin/rosetta_scripts_430"),
            rosetta_database=rosetta_main / "database",
            molfile_to_params=rosetta_main / "source/scripts/python/public/molfile_to_params.py",
            true_receptor_root=Path("/storage/chenzhaoyang/cryo_em/CIF_3.5_atom"),
            cryoatom_root=Path("/storage/chenzhaoyang/cryo_em/result_split"),
        )


@dataclass(frozen=True)
class RosettaOptions:
    """
    Rosetta smoke test 参数. 
    输入参数:
        - nstruct: int, 每个 job 输出构象数; 当前 smoke test 使用 1
        - scorefxn: str, Rosetta scorefunction 名称
        - grid_radius: float, GALigandDock grid radius
        - grid_step: float, GALigandDock grid spacing
        - padding: float, grid padding
        - skeleton_radius: float, EM density skeleton 搜索半径
        - npool: int, 低成本初始 pool 大小
        - maxiter: int, minimizer 最大迭代数
        - pack_cycles: int, pack 周期数

    输出:
        - RosettaOptions: 不可变配置对象
    """

    nstruct: int
    scorefxn: str
    grid_radius: float
    grid_step: float
    padding: float
    skeleton_radius: float
    npool: int
    maxiter: int
    pack_cycles: int

    @staticmethod
    def smoke() -> "RosettaOptions":
        """返回当前已跑通的低成本 Rosetta 参数. """
        return RosettaOptions(
            nstruct=1,
            scorefxn="beta_genpot",
            grid_radius=8.0,
            grid_step=0.5,
            padding=4.0,
            skeleton_radius=8.0,
            npool=12,
            maxiter=20,
            pack_cycles=5,
        )


@dataclass(frozen=True)
class MatchingOptions:
    """
    匹配成本参数. 
    输入参数:
        - docking_weight: float, Rosetta docking 分数归一化后的权重
        - shape_weight: float, 网络形状分数归一化后的权重
        - ignore_site_base_cost: float, 真实预测 instance 匹配虚拟 ligand 的基础成本
        - missing_ligand_base_cost: float, 虚拟 instance 匹配真实 ligand 的基础成本

    输出:
        - MatchingOptions: 不可变配置对象
    """

    docking_weight: float
    shape_weight: float
    ignore_site_base_cost: float
    missing_ligand_base_cost: float

    @staticmethod
    def current() -> "MatchingOptions":
        """返回当前实验使用的成本权重. """
        return MatchingOptions(
            docking_weight=0.7,
            shape_weight=0.3,
            ignore_site_base_cost=1.0,
            missing_ligand_base_cost=1.0,
        )

