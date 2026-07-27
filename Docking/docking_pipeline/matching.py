from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import MatchingOptions
from .records import AssignmentResult, PairScore


@dataclass(frozen=True)
class RawPairFeatures:
    """
    未归一化的 site-ligand pair 特征. 
    输入参数:
        - site_id: str, site 标签
        - ligand_label: str, ligand 标签
        - receptor_scope: str, receptor 分数范围
        - dG: float, Rosetta dG, 越低越好
        - shape: float, 网络形状成本, 越低越好

    输出:
        - RawPairFeatures: 不可变特征记录
    """

    site_id: str
    ligand_label: str
    receptor_scope: str
    dG: float
    shape: float


def build_pair_scores(features: list[RawPairFeatures], options: MatchingOptions) -> list[PairScore]:
    """
    将原始 pair 特征归一化并合成为匹配成本. 
    输入参数:
        - features: list[RawPairFeatures], 同一个 receptor_scope 下的 pair 特征
        - options: MatchingOptions, 匹配权重

    输出:
        - scores: list[PairScore], 可直接进入 assignment 的成本记录
    """
    dg_norm = _minmax([item.dG for item in features])
    shape_norm = _minmax([item.shape for item in features])
    scores: list[PairScore] = []
    for item, dg_value, shape_value in zip(features, dg_norm, shape_norm):
        combined = options.docking_weight * dg_value + options.shape_weight * shape_value
        scores.append(
            PairScore(
                site_id=item.site_id,
                ligand_label=item.ligand_label,
                receptor_scope=item.receptor_scope,
                terms={"dG": item.dG, "shape": item.shape, "dG_norm": dg_value, "shape_norm": shape_value},
                combined_cost=combined,
            )
        )
    return scores


def solve_assignment(scores: list[PairScore], site_ids: list[str], ligand_labels: list[str]) -> AssignmentResult:
    """
    求解当前 site-ligand assignment. 
    输入参数:
        - scores: list[PairScore], 真实 site-ligand 边成本
        - site_ids: list[str], 真实预测 site 标签
        - ligand_labels: list[str], 真实 ligand 标签

    输出:
        - result: AssignmentResult, 当 `len(site_ids) != len(ligand_labels)` 时允许较多一侧 unmatched
    """
    receptor_scope = scores[0].receptor_scope
    score_map = {(item.site_id, item.ligand_label): item for item in scores}
    if len(site_ids) <= len(ligand_labels):
        return _assign_sites_to_ligands(score_map, receptor_scope, site_ids, ligand_labels)
    return _assign_ligands_to_subset_of_sites(score_map, receptor_scope, site_ids, ligand_labels)


def solve_assignment_with_virtual_nodes(
    scores: list[PairScore],
    site_ids: list[str],
    ligand_labels: list[str],
    site_ignore_costs: dict[str, float],
    ligand_missing_costs: dict[str, float],
) -> AssignmentResult:
    """
    使用虚拟节点求解方阵 assignment. 
    输入参数:
        - scores: list[PairScore], 真实 site-ligand 边成本
        - site_ids: list[str], 真实预测 site 标签
        - ligand_labels: list[str], 真实 ligand 标签
        - site_ignore_costs: dict[str, float], 真实 site 匹配虚拟 ligand 的成本
        - ligand_missing_costs: dict[str, float], 虚拟 site 匹配真实 ligand 的成本

    输出:
        - result: AssignmentResult, 包含真实匹配、未匹配项与虚拟边审计信息
    """
    receptor_scope = scores[0].receptor_scope
    score_map = {(item.site_id, item.ligand_label): item for item in scores}
    chosen, total, solver = _solve_virtual_assignment_edges(score_map, site_ids, ligand_labels, site_ignore_costs, ligand_missing_costs)
    real_pairs: list[PairScore] = []
    virtual_edges: list[dict[str, object]] = []
    unmatched_sites: list[str] = []
    unmatched_ligands: list[str] = []
    for site, ligand, cost in chosen:
        if site in site_ids and ligand in ligand_labels:
            real_pairs.append(score_map[(site, ligand)])
        elif site in site_ids:
            unmatched_sites.append(site)
            virtual_edges.append({"site": site, "ligand": ligand, "cost": cost, "kind": "ignore_site"})
        elif ligand in ligand_labels:
            unmatched_ligands.append(ligand)
            virtual_edges.append({"site": site, "ligand": ligand, "cost": cost, "kind": "missing_ligand"})
    return AssignmentResult(
        receptor_scope=receptor_scope,
        total_cost=total,
        pairs=tuple(real_pairs),
        unmatched_sites=tuple(unmatched_sites),
        unmatched_ligands=tuple(unmatched_ligands),
        virtual_edges=tuple(virtual_edges),
        solver=solver,
    )


def _solve_virtual_assignment_edges(
    score_map: dict[tuple[str, str], PairScore],
    site_ids: list[str],
    ligand_labels: list[str],
    site_ignore_costs: dict[str, float],
    ligand_missing_costs: dict[str, float],
) -> tuple[list[tuple[str, str, float]], float, str]:
    """
    求解带虚拟节点的 assignment 边. 

    输入参数:
        - score_map: dict[tuple[str, str], PairScore], 真实 site-ligand 边成本
        - site_ids: list[str], 真实预测 site 标签
        - ligand_labels: list[str], 真实 ligand 标签
        - site_ignore_costs: dict[str, float], 真实 site 匹配虚拟 ligand 的成本
        - ligand_missing_costs: dict[str, float], 虚拟 site 匹配真实 ligand 的成本

    输出:
        - result: tuple[list[tuple[str, str, float]], float, str], 选中边、总成本和实际 solver 名称
    """
    n = max(len(site_ids), len(ligand_labels))
    padded_sites = site_ids + [f"__virtual_site_{i:03d}" for i in range(n - len(site_ids))]
    padded_ligands = ligand_labels + [f"__virtual_ligand_{i:03d}" for i in range(n - len(ligand_labels))]
    if n <= 16:
        chosen, total = _assignment_dp(
            padded_sites,
            padded_ligands,
            lambda site, ligand: _virtual_aware_cost(site, ligand, score_map, site_ignore_costs, ligand_missing_costs),
        )
        return chosen, total, "dp_virtual"
    chosen, total = _assignment_scipy(padded_sites, padded_ligands, score_map, site_ignore_costs, ligand_missing_costs)
    return chosen, total, "scipy_hungarian"


def _assignment_scipy(
    padded_sites: list[str],
    padded_ligands: list[str],
    score_map: dict[tuple[str, str], PairScore],
    site_ignore_costs: dict[str, float],
    ligand_missing_costs: dict[str, float],
) -> tuple[list[tuple[str, str, float]], float]:
    """
    使用 scipy 的 Hungarian solver 求解较大方阵. 

    输入参数:
        - padded_sites: list[str], 补齐后的左侧 site 节点
        - padded_ligands: list[str], 补齐后的右侧 ligand 节点
        - score_map: dict[tuple[str, str], PairScore], 真实边成本
        - site_ignore_costs: dict[str, float], 忽略 site 成本
        - ligand_missing_costs: dict[str, float], 漏掉 ligand 成本

    输出:
        - result: tuple[list[tuple[str, str, float]], float], 选中边与总成本
    """
    from scipy.optimize import linear_sum_assignment

    cost = np.asarray(
        [
            [
                _virtual_aware_cost(site, ligand, score_map, site_ignore_costs, ligand_missing_costs)
                for ligand in padded_ligands
            ]
            for site in padded_sites
        ],
        dtype=float,
    )
    row_indices, col_indices = linear_sum_assignment(cost)
    chosen = [
        (padded_sites[row], padded_ligands[col], float(cost[row, col]))
        for row, col in zip(row_indices, col_indices)
    ]
    return chosen, float(cost[row_indices, col_indices].sum())




def _minmax(values: list[float]) -> list[float]:
    """对一组数做 min-max 归一化, 常数列返回全 0. """
    lower = min(values)
    upper = max(values)
    if abs(upper - lower) < 1e-12:
        return [0.0 for _ in values]
    return [(value - lower) / (upper - lower) for value in values]


def _assign_sites_to_ligands(
    score_map: dict[tuple[str, str], PairScore],
    receptor_scope: str,
    site_ids: list[str],
    ligand_labels: list[str],
) -> AssignmentResult:
    """处理 site 数不多于 ligand 数的矩形 assignment. """
    chosen, total = _assignment_dp(
        site_ids,
        ligand_labels,
        lambda site, ligand: score_map[(site, ligand)].combined_cost,
    )
    pairs = tuple(score_map[(site, ligand)] for site, ligand, _ in chosen)
    matched_ligands = {ligand for _, ligand, _ in chosen}
    return AssignmentResult(
        receptor_scope=receptor_scope,
        total_cost=total,
        pairs=pairs,
        unmatched_ligands=tuple(label for label in ligand_labels if label not in matched_ligands),
        solver="dp_rectangular",
    )


def _assign_ligands_to_subset_of_sites(
    score_map: dict[tuple[str, str], PairScore],
    receptor_scope: str,
    site_ids: list[str],
    ligand_labels: list[str],
) -> AssignmentResult:
    """处理 site 数多于 ligand 数的矩形 assignment. """
    chosen, total = _assignment_dp(
        ligand_labels,
        site_ids,
        lambda ligand, site: score_map[(site, ligand)].combined_cost,
    )
    pairs = tuple(score_map[(site, ligand)] for ligand, site, _ in chosen)
    matched_sites = {site for _, site, _ in chosen}
    return AssignmentResult(
        receptor_scope=receptor_scope,
        total_cost=total,
        pairs=pairs,
        unmatched_sites=tuple(site for site in site_ids if site not in matched_sites),
        solver="dp_rectangular",
    )


def _assignment_dp(
    left_nodes: list[str],
    right_nodes: list[str],
    cost_fn,
) -> tuple[list[tuple[str, str, float]], float]:
    """
    用 bitmask DP 求解小到中等规模 assignment. 
    输入参数:
        - left_nodes: list[str], 左侧节点, 数量不应大于右侧
        - right_nodes: list[str], 右侧节点
        - cost_fn: Callable, 输入一个 left 与 right, 返回成本

    输出:
        - result: tuple[list[tuple[str, str, float]], float], 包含选中边和总成本
    """
    states: dict[int, tuple[float, list[tuple[str, str, float]]]] = {0: (0.0, [])}
    for left in left_nodes:
        next_states: dict[int, tuple[float, list[tuple[str, str, float]]]] = {}
        for mask, (current_cost, current_edges) in states.items():
            for right_index, right in enumerate(right_nodes):
                bit = 1 << right_index
                if mask & bit:
                    continue
                edge_cost = float(cost_fn(left, right))
                new_mask = mask | bit
                new_cost = current_cost + edge_cost
                if new_mask not in next_states or new_cost < next_states[new_mask][0]:
                    next_states[new_mask] = (new_cost, current_edges + [(left, right, edge_cost)])
        states = next_states
    best_cost, best_edges = min(states.values(), key=lambda item: item[0])
    return best_edges, best_cost


def _virtual_aware_cost(
    site: str,
    ligand: str,
    score_map: dict[tuple[str, str], PairScore],
    site_ignore_costs: dict[str, float],
    ligand_missing_costs: dict[str, float],
) -> float:
    """返回真实边或虚拟边成本. """
    site_is_virtual = site.startswith("__virtual_site_")
    ligand_is_virtual = ligand.startswith("__virtual_ligand_")
    if site_is_virtual and ligand_is_virtual:
        return 0.0
    if ligand_is_virtual:
        return site_ignore_costs[site]
    if site_is_virtual:
        return ligand_missing_costs[ligand]
    return score_map[(site, ligand)].combined_cost
