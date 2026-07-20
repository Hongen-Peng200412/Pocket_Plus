"""统一导出 AdaLigand Stage1 组件森林、候选谱系组及重叠基础事实接口。

主要入口:
    - `build_component_forest`: 从完整图概率和冻结阈值构造多层 26-连通组件森林。
    - `enumerate_clgs`: 在每棵树的 active 工作副本上枚举 Candidate Lineage Group（候选谱系组，CLG）。
    - `build_candidate_occurrence_overlap`: 计算候选组件与真实配体 occurrence 的非零 voxel 交集计数。

本模块只汇总公共符号，不读取或写入 forest、CLG、overlap 产物。
"""

from .clg import CLGEnumerationConfig, CLGEnumerationResult, enumerate_clgs
from .forest import build_component_forest
from .overlap import build_candidate_occurrence_overlap
from .structures import (
    CLG,
    ComponentForest,
    ComponentNode,
    ComponentTree,
    WorkingTree,
)

__all__ = [
    "CLG",
    "CLGEnumerationConfig",
    "CLGEnumerationResult",
    "ComponentForest",
    "ComponentNode",
    "ComponentTree",
    "WorkingTree",
    "build_candidate_occurrence_overlap",
    "build_component_forest",
    "enumerate_clgs",
]
