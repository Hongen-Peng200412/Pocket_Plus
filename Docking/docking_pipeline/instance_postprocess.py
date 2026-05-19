from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from .records import InferenceSite


@dataclass(frozen=True)
class InstancePostprocessOptions:
    """
    推理 instance 后处理参数。

    输入参数:
        - min_voxels: int, 最小体素数; 小于该值的 instance 会在 docking 前被过滤
        - enable_merge: bool, 是否真正执行合并; False 时只保留过滤结果
        - merge_min_voxel_distance: float, 两个 instance 最近体素世界坐标距离阈值
        - merge_center_distance: float, 两个 instance 中心世界坐标距离阈值
        - merge_max_bbox_span_increase: float, 合并后包围盒对角线允许增加的最大值

    输出:
        - InstancePostprocessOptions: 不可变参数对象
    """

    min_voxels: int
    enable_merge: bool
    merge_min_voxel_distance: float
    merge_center_distance: float
    merge_max_bbox_span_increase: float

    @staticmethod
    def conservative() -> "InstancePostprocessOptions":
        """返回低误删倾向的经验参数。"""
        return InstancePostprocessOptions(
            min_voxels=30,
            enable_merge=True,
            merge_min_voxel_distance=2.5,
            merge_center_distance=6.0,
            merge_max_bbox_span_increase=8.0,
        )


@dataclass(frozen=True)
class PostprocessResult:
    """
    instance 后处理结果。

    输入参数:
        - sites: tuple[InferenceSite, ...], 后处理后进入 docking 的预测位点
        - label: np.ndarray, (D, H, W), int, 合并后用于 shape score 的 instance 标签图
        - raw_sites: tuple[InferenceSite, ...], 原始预测位点
        - filtered_sites: tuple[dict[str, Any], ...], 被最小体素数过滤的位点记录
        - merge_candidates: tuple[dict[str, Any], ...], 所有两两合并候选及其特征
        - merges: tuple[dict[str, Any], ...], 实际合并记录
        - options: InstancePostprocessOptions, 本次使用的后处理参数

    输出:
        - PostprocessResult: 后处理结果对象, 可用于 docking 和审计落盘
    """

    sites: tuple[InferenceSite, ...]
    label: np.ndarray
    raw_sites: tuple[InferenceSite, ...]
    filtered_sites: tuple[dict[str, Any], ...]
    merge_candidates: tuple[dict[str, Any], ...]
    merges: tuple[dict[str, Any], ...]
    options: InstancePostprocessOptions

    def audit_dict(self) -> dict[str, Any]:
        """
        转成 JSON 友好的审计字典。

        输出:
            - audit: dict[str, Any], 包含参数、原始数量、过滤记录、合并候选、实际合并和输出位点
        """
        return {
            "options": asdict(self.options),
            "num_raw_sites": len(self.raw_sites),
            "num_output_sites": len(self.sites),
            "raw_sites": [_site_dict(site) for site in self.raw_sites],
            "output_sites": [_site_dict(site) for site in self.sites],
            "filtered_sites": list(self.filtered_sites),
            "merge_candidates": list(self.merge_candidates),
            "merges": list(self.merges),
        }


def postprocess_sites(
    sites: list[InferenceSite],
    label: np.ndarray,
    origin: np.ndarray,
    voxel_size: np.ndarray,
    options: InstancePostprocessOptions,
) -> PostprocessResult:
    """
    对推理 instance 执行最小体素数过滤和保守合并。

    输入参数:
        - sites: list[InferenceSite], 原始预测位点列表
        - label: np.ndarray, (D, H, W), int, 原始 instance 标签图, 0 表示背景
        - origin: np.ndarray, (3,), 体素到世界坐标转换的原点
        - voxel_size: np.ndarray, (3,), xyz 方向体素大小
        - options: InstancePostprocessOptions, 后处理参数

    输出:
        - result: PostprocessResult, 包含后处理位点、合并后标签图和审计记录
    """
    kept_sites: list[InferenceSite] = []
    filtered_sites: list[dict[str, Any]] = []
    for site in sites:
        if site.voxel_count < options.min_voxels:
            filtered_sites.append({**_site_dict(site), "reason": "below_min_voxels"})
        else:
            kept_sites.append(site)

    if not options.enable_merge or len(kept_sites) < 2:
        return PostprocessResult(
            sites=tuple(kept_sites),
            label=label.copy(),
            raw_sites=tuple(sites),
            filtered_sites=tuple(filtered_sites),
            merge_candidates=(),
            merges=(),
            options=options,
        )

    coords_by_id = {
        site.instance_id: _instance_world_xyz(label, site.instance_id, origin, voxel_size)
        for site in kept_sites
    }
    merge_candidates = _build_merge_candidates(kept_sites, coords_by_id, options)
    accepted = [item for item in merge_candidates if item["should_merge"]]
    groups = _merge_groups([site.instance_id for site in kept_sites], accepted)
    merged_label, merged_sites, merges = _apply_merge_groups(label, kept_sites, coords_by_id, groups)
    return PostprocessResult(
        sites=tuple(sorted(merged_sites, key=lambda site: site.instance_id)),
        label=merged_label,
        raw_sites=tuple(sites),
        filtered_sites=tuple(filtered_sites),
        merge_candidates=tuple(merge_candidates),
        merges=tuple(merges),
        options=options,
    )


def estimate_job_count(num_sites: int, num_ligands: int, num_receptors: int) -> int:
    """
    估计 Rosetta job 数。

    输入参数:
        - num_sites: int, 后处理后预测 site 数
        - num_ligands: int, 可对接候选 ligand 数
        - num_receptors: int, receptor 来源数量

    输出:
        - count: int, 预计 Rosetta job 数
    """
    return int(num_sites) * int(num_ligands) * int(num_receptors)


def _site_dict(site: InferenceSite) -> dict[str, Any]:
    """返回单个 site 的 JSON 友好摘要。"""
    return {
        "site_id": site.site_id,
        "instance_id": site.instance_id,
        "center_world_xyz": list(site.center_world_xyz),
        "score_mean": site.score_mean,
        "score_max": site.score_max,
        "voxel_count": site.voxel_count,
    }


def _instance_world_xyz(label: np.ndarray, instance_id: int, origin: np.ndarray, voxel_size: np.ndarray) -> np.ndarray:
    """
    提取一个 instance 的世界坐标点云。

    输入参数:
        - label: np.ndarray, (D, H, W), int, instance 标签图
        - instance_id: int, 要提取的 instance 编号
        - origin: np.ndarray, (3,), xyz 世界坐标原点
        - voxel_size: np.ndarray, (3,), xyz 体素大小

    输出:
        - coords: np.ndarray, (N, 3), 当前 instance 的体素中心世界坐标
    """
    zyx = np.argwhere(label == instance_id)
    return np.stack(
        [
            zyx[:, 2] * voxel_size[2] + origin[2],
            zyx[:, 1] * voxel_size[1] + origin[1],
            zyx[:, 0] * voxel_size[0] + origin[0],
        ],
        axis=1,
    )


def _build_merge_candidates(
    sites: list[InferenceSite],
    coords_by_id: dict[int, np.ndarray],
    options: InstancePostprocessOptions,
) -> list[dict[str, Any]]:
    """计算所有两两 instance 合并候选特征。"""
    candidates: list[dict[str, Any]] = []
    for left_index, left in enumerate(sites):
        for right in sites[left_index + 1 :]:
            left_coords = coords_by_id[left.instance_id]
            right_coords = coords_by_id[right.instance_id]
            min_distance = _nearest_point_distance(left_coords, right_coords)
            center_distance = float(np.linalg.norm(np.asarray(left.center_world_xyz) - np.asarray(right.center_world_xyz)))
            bbox_increase = _bbox_span_increase(left_coords, right_coords)
            should_merge = (
                min_distance <= options.merge_min_voxel_distance
                and center_distance <= options.merge_center_distance
                and bbox_increase <= options.merge_max_bbox_span_increase
            )
            candidates.append(
                {
                    "left_site_id": left.site_id,
                    "right_site_id": right.site_id,
                    "left_instance_id": left.instance_id,
                    "right_instance_id": right.instance_id,
                    "min_voxel_distance": min_distance,
                    "center_distance": center_distance,
                    "bbox_span_increase": bbox_increase,
                    "should_merge": should_merge,
                }
            )
    return candidates


def _nearest_point_distance(left: np.ndarray, right: np.ndarray) -> float:
    """分块计算两个点云之间的最近欧氏距离。"""
    if len(left) > len(right):
        left, right = right, left
    best = float("inf")
    chunk_size = 512
    for start in range(0, len(left), chunk_size):
        chunk = left[start : start + chunk_size]
        distances = np.linalg.norm(chunk[:, None, :] - right[None, :, :], axis=2)
        best = min(best, float(distances.min()))
    return best


def _bbox_span_increase(left: np.ndarray, right: np.ndarray) -> float:
    """计算合并后包围盒对角线相对较大单体包围盒的增加量。"""
    left_diag = _bbox_diag(left)
    right_diag = _bbox_diag(right)
    merged_diag = _bbox_diag(np.vstack([left, right]))
    return float(merged_diag - max(left_diag, right_diag))


def _bbox_diag(coords: np.ndarray) -> float:
    """返回点云包围盒对角线长度。"""
    span = coords.max(axis=0) - coords.min(axis=0)
    return float(np.linalg.norm(span))


def _merge_groups(instance_ids: list[int], accepted: list[dict[str, Any]]) -> list[list[int]]:
    """根据已接受的两两合并边构造连通合并组。"""
    parent = {instance_id: instance_id for instance_id in instance_ids}

    def find(value: int) -> int:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for item in accepted:
        union(int(item["left_instance_id"]), int(item["right_instance_id"]))

    grouped: dict[int, list[int]] = {}
    for instance_id in instance_ids:
        grouped.setdefault(find(instance_id), []).append(instance_id)
    return [sorted(values) for values in grouped.values()]


def _apply_merge_groups(
    label: np.ndarray,
    sites: list[InferenceSite],
    coords_by_id: dict[int, np.ndarray],
    groups: list[list[int]],
) -> tuple[np.ndarray, list[InferenceSite], list[dict[str, Any]]]:
    """把合并组应用到标签图和 site 列表。"""
    site_by_id = {site.instance_id: site for site in sites}
    merged_label = label.copy()
    merged_sites: list[InferenceSite] = []
    merges: list[dict[str, Any]] = []
    for group in groups:
        representative = min(group)
        group_sites = [site_by_id[instance_id] for instance_id in group]
        if len(group) == 1:
            merged_sites.append(group_sites[0])
            continue

        for instance_id in group:
            if instance_id != representative:
                merged_label[merged_label == instance_id] = representative

        weights = np.asarray([site.voxel_count for site in group_sites], dtype=float)
        centers = np.asarray([site.center_world_xyz for site in group_sites], dtype=float)
        merged_center = tuple(np.average(centers, axis=0, weights=weights).tolist())
        voxel_count = int(sum(site.voxel_count for site in group_sites))
        score_mean = float(np.average([site.score_mean for site in group_sites], weights=weights))
        score_max = float(max(site.score_max for site in group_sites))
        merged_sites.append(
            InferenceSite(
                instance_id=representative,
                center_world_xyz=merged_center,
                score_mean=score_mean,
                score_max=score_max,
                voxel_count=voxel_count,
            )
        )
        merged_coords = np.vstack([coords_by_id[instance_id] for instance_id in group])
        merges.append(
            {
                "output_site_id": f"site{representative:03d}",
                "merged_instance_ids": group,
                "voxel_count": voxel_count,
                "center_world_xyz": list(merged_center),
                "bbox_diag": _bbox_diag(merged_coords),
            }
        )
    return merged_label, merged_sites, merges
