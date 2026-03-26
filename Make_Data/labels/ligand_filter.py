# -*- coding: utf-8 -*-
"""
================================================================================
配体筛选与口袋分类 / Ligand Filtering & Pocket Classification
================================================================================

Part 2 核心模块：
  1. 每条 PocketClassRule 决定了一组规则:
     - 候选配体按规则列表顺序依次检查
     - 第一条满足条件的规则生效，该候选被分配对应的 class_id
     - 不匹配任何规则的候选直接排除（背景，不产生口袋）
  2. LigandFilterConfig 是规则列表的容器（壳子）
  3. filter_and_classify() 一步完成筛选+分类

================================================================================
"""
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import sys
from pathlib import Path

# 绝对导入（labels/ 是 Make_Data/ 下的顶层包）
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from PDB_processor.ligand_candidates import LigandCandidate


# ============================================================================
# 数据结构 / Data Structures
# ============================================================================

@dataclass
class PocketClassRule:
    """
    口袋类别规则。

    一个候选配体必须满足本规则的所有条件，才被归入 class_id 类别。
    规则按照 LigandFilterConfig.rules 列表的顺序检查，第一条匹配的规则生效。不匹配任何规则的候选直接排除（不产生口袋）。

    NOTE：不同规则可以共享同一个 class_id（同一种口袋类别可由多条规则产生）, 或者说不同规则可以共享同一个 class_id（即通往同一类别的不同路径），但同一 class_id 对应的 class_name 必须一致（class_id 是类别的唯一标识）, 先出现的规则具有优先权，但这只影响哪条规则被记录，不影响最终类别。

    ============================== 字段说明 ==============================

    分类标签 / Classification Label:
        - class_id:   int, 口袋类别 ID (>0; 0 保留给背景/非口袋)
        - class_name: str, 类别名称 (如 'druggable', 'metal_ion')
        - binding_threshold: float, 结合位点距离阈值 (埃)。含义: 距最近配体该距离内的蛋白原子视为属于结合口袋。

    布尔标志三值开关 / Boolean Flag Three-way Switches:
        每个开关均为 Optional[bool]，语义如下：
          True  → 候选必须具有该属性，否则拒绝
          False → 候选必须不具有该属性，否则拒绝
          None  → 不约束（默认）

        - require_metal_ion:       Optional[bool], 单独出现的金属离子约束
        - require_peptide_like:    Optional[bool], 标准多肽类 HETATM 约束
        - require_nucleotide_like: Optional[bool], 标准核苷酸类 HETATM 约束
        - require_covalent:        Optional[bool], 共价连接约束

    聚合物长度限制 / Polymer Length Limits:
        仅对对应属性为 True 的候选生效（None=不限）。
        - min_peptide_length: Optional[int], 最小氨基酸聚合链长
        - max_peptide_length: Optional[int], 最大氨基酸聚合链长
        - min_nucleic_length: Optional[int], 最小核苷酸聚合链长
        - max_nucleic_length: Optional[int], 最大核苷酸聚合链长

    保留接口 (当前不生效) / Reserved (currently inactive):
        - min_mw / max_mw:                Optional[float], 分子量范围
        - min_heavy_atoms / max_heavy_atoms: Optional[int], 重原子数范围
        - require_organic_only:           bool, 仅含有机元素
        - use_af3_ligand_exclusion:       bool, 使用 AF3 排除列表
        - min_contact_residues:           Optional[int], 最小接触残基数
        - max_resname_occurrences:        Optional[int], 最大 resname 出现次数
    """

    # ========================= 分类标签 =========================
    # int, 口袋类别 ID (>0)
    class_id: int
    # str, 类别名称
    class_name: str

    # float, 结合位点距离阈值 (埃)
    # 含义: 距最近配体该距离内的蛋白原子视为属于结合口袋
    binding_threshold: float = 4.0

    # ========================= 布尔标志三值开关 =========================
    # Optional[bool], 金属离子约束 (True=必须是, False=必须不是, None=不约束)
    require_metal_ion: Optional[bool] = None
    # Optional[bool], 标准氨基酸类 HETATM 约束
    require_peptide_like: Optional[bool] = None
    # Optional[bool], 标准核苷酸类 HETATM 约束
    require_nucleotide_like: Optional[bool] = None
    # Optional[bool], 共价连接约束
    require_covalent: Optional[bool] = None

    # ========================= 聚合物长度限制 =========================
    # Optional[int], 最小允许的肽链长度; None=不限; 仅对 is_peptide_like=True 的候选生效
    min_peptide_length: Optional[int] = None
    # Optional[int], 最大允许的肽链长度; None=不限; 仅对 is_peptide_like=True 的候选生效
    max_peptide_length: Optional[int] = None
    # Optional[int], 最小允许的核酸链长度; None=不限; 仅对 is_nucleotide_like=True 的候选生效
    min_nucleic_length: Optional[int] = None
    # Optional[int], 最大允许的核酸链长度; None=不限; 仅对 is_nucleotide_like=True 的候选生效
    max_nucleic_length: Optional[int] = None

    # # ========================= 保留接口 (当前不生效) =========================
    # min_mw: Optional[float] = None
    # max_mw: Optional[float] = None
    # min_heavy_atoms: Optional[int] = None
    # max_heavy_atoms: Optional[int] = None
    # require_organic_only: bool = False
    # use_af3_ligand_exclusion: bool = False
    # min_contact_residues: Optional[int] = None
    # max_resname_occurrences: Optional[int] = None

    def accepts(self, candidate: LigandCandidate) -> Optional[str]:
        """
        检查候选配体是否满足本规则的全部条件。

        输入参数 / Input:
            - candidate: LigandCandidate, 候选配体

        输出 / Output:

            - None: 候选满足本规则（接受）
            - str:  候选不满足本规则的原因（拒绝）
        """
        # ------ 布尔标志三值开关 ------
        if self.require_metal_ion is not None and candidate.is_metal_ion != self.require_metal_ion:
            return f"rule[{self.class_name}] reject: require_metal_ion={self.require_metal_ion}"
        if self.require_peptide_like is not None and candidate.is_peptide_like != self.require_peptide_like:
            return f"rule[{self.class_name}] reject: require_peptide_like={self.require_peptide_like}"
        if self.require_nucleotide_like is not None and candidate.is_nucleotide_like != self.require_nucleotide_like:
            return f"rule[{self.class_name}] reject: require_nucleotide_like={self.require_nucleotide_like}"
        if self.require_covalent is not None and candidate.is_covalent != self.require_covalent:
            return f"rule[{self.class_name}] reject: require_covalent={self.require_covalent}"

        # ------ 聚合物长度限制 ------
        # 肽链长度（仅对 is_peptide_like=True 的候选生效）
        if candidate.is_peptide_like:
            if self.min_peptide_length is not None and candidate.polymer_length < self.min_peptide_length:
                return (f"rule[{self.class_name}] reject: peptide polymer_length="
                        f"{candidate.polymer_length} < min={self.min_peptide_length}")
            if self.max_peptide_length is not None and candidate.polymer_length > self.max_peptide_length:
                return (f"rule[{self.class_name}] reject: peptide polymer_length="
                        f"{candidate.polymer_length} > max={self.max_peptide_length}")
        # 核酸链长度（仅对 is_nucleotide_like=True 的候选生效）
        if candidate.is_nucleotide_like:
            if self.min_nucleic_length is not None and candidate.polymer_length < self.min_nucleic_length:
                return (f"rule[{self.class_name}] reject: nucleic polymer_length="
                        f"{candidate.polymer_length} < min={self.min_nucleic_length}")
            if self.max_nucleic_length is not None and candidate.polymer_length > self.max_nucleic_length:
                return (f"rule[{self.class_name}] reject: nucleic polymer_length="
                        f"{candidate.polymer_length} > max={self.max_nucleic_length}")

        # # ------ 保留接口 (当前不生效) ------
        # if self.min_mw is not None and candidate.molecular_weight is not None:
        #     if candidate.molecular_weight < self.min_mw:
        #         return f"rule[{self.class_name}] reject: MW < min_mw"
        # ...

        # 全部通过，接受
        return None


@dataclass
class LigandFilterConfig:
    """
    配体筛选配置————只是 PocketClassRule 列表的容器（壳子）。

    候选配体按 rules 列表顺序依次检查：
      - 第一条 accepts() 返回 None 的规则生效，候选被分配该规则的 class_id
      - 不匹配任何规则的候选直接排除（不产生口袋，背景）

    字段说明 / Fields:
        - rules: list[PocketClassRule], 有序规则列表（先匹配先生效）
    """
    # list[PocketClassRule], 有序规则列表
    rules: List[PocketClassRule] = field(default_factory=list)


# 预设配置统一维护在 labels/filter_config.py, 本模块只保留筛选规则定义与核心筛选逻辑，避免配置散落在多个文件。

def filter_and_classify(
    candidates: List[LigandCandidate],
    config: LigandFilterConfig,
) -> Tuple[List[LigandCandidate], Dict[int, Tuple[int, str, float]], List[Tuple[int, str]]]:
    """
    一步完成筛选 + 分类：对每个候选配体按规则列表顺序检查，
    第一条匹配的规则决定其口袋类别；不匹配任何规则的候选直接排除。

    输入参数 / Input:
        - candidates: list[LigandCandidate], Part 1 产出的全量候选列表
        - config: LigandFilterConfig, 筛选配置（规则列表）

    输出 / Output:
        - selected:         list[LigandCandidate], 通过筛选的候选列表（按原始顺序）
        - pocket_class_map: dict[int, tuple[int, str, float]], candidate_id → (class_id, class_name, binding_threshold)
        - excluded:         list[tuple[int, str]], 被排除的候选 (candidate_id, 排除原因)
    """
    # list[LigandCandidate], 通过筛选的候选
    selected = []
    # dict[int, tuple[int, str, float]], candidate_id → (class_id, class_name, binding_threshold)
    pocket_class_map = {}
    # list[tuple[int, str]], 被排除的候选及原因
    excluded = []

    for candidate in candidates:
        # bool, 是否匹配到某条规则
        matched = False
        for rule in config.rules:
            # Optional[str], None=接受, str=拒绝原因
            reject_reason = rule.accepts(candidate)
            if reject_reason is None:
                # 匹配成功
                selected.append(candidate)
                pocket_class_map[candidate.candidate_id] = (rule.class_id, rule.class_name, rule.binding_threshold)
                matched = True
                break

        if not matched:
            excluded.append((
                candidate.candidate_id,
                f"no rule matched (resname={candidate.resname})"
            ))

    return selected, pocket_class_map, excluded


def get_pocket_class_name_map(config: LigandFilterConfig) -> Dict[int, str]:
    """
    从规则列表中提取口袋类别 ID → 名称的映射, 总是包含 0='background'。

    输入参数 / Input:
        - config: LigandFilterConfig, 筛选配置

    输出 / Output:
        - dict[int, str], {class_id: class_name}, 总是包含 0='background'
    """
    # dict[int, str], ID → 名称
    name_map = {0: 'background'}
    for rule in config.rules:
        if rule.class_id in name_map:
            # 同一 class_id 的多条规则是“通往同一类别的不同路径”，class_name 必须一致 ; 若不一致，说明用户配置有误（同一类别 ID 被赋予了两个不同名称）
            existing = name_map[rule.class_id]
            if existing != rule.class_name:
                raise ValueError(
                    f"PocketClassRule 配置错误: class_id={rule.class_id} "
                    f"对应了两个不同的 class_name: '{existing}' 和 '{rule.class_name}'。"
                    f"同一 class_id 的所有规则必须使用相同的 class_name。"
                )
            # class_name 相同，无需重复写入
        else:
            name_map[rule.class_id] = rule.class_name
    return name_map
