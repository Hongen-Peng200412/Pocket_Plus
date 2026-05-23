from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
import torch.nn.functional as F
import torch_cluster
from torch import nn


def _sort_by_prob_desc(prob: torch.Tensor) -> torch.Tensor:
    """
    按概率降序返回排序下标, 不保证同概率稳定 tie-break。

    输入参数:
        - prob: torch.Tensor, (M,), 每个候选的概率

    输出:
        - order: torch.Tensor, (M,), 排序后的局部下标
    """
    return torch.argsort(prob, dim=0, descending=True)


def _deduplicate_candidates_by_voxel(
    candidate_voxel_zyx: torch.Tensor,
    candidate_batch_index: torch.Tensor,
    candidate_prob: torch.Tensor,
) -> torch.Tensor:
    """
    对同一 BOX 内同一 voxel 的候选记录做 P 层去重。

    输入参数:
        - candidate_voxel_zyx: torch.Tensor, (sumC, 3), 候选 voxel 坐标, 轴顺序 z/y/x
        - candidate_batch_index: torch.Tensor, (sumC,), 候选所属 BOX 索引
        - candidate_prob: torch.Tensor, (sumC,), 候选概率

    输出:
        - kept_candidate_index: torch.Tensor, (sumC_unique,), 保留的原始 C 行号, 按原始行号升序排列
    """
    # int, 候选总数
    num_candidates = int(candidate_prob.shape[0])
    if num_candidates == 0:
        return torch.empty((0,), device=candidate_prob.device, dtype=torch.long)

    # torch.Tensor, (sumC,), 原始 C 行号
    candidate_index = torch.arange(num_candidates, device=candidate_prob.device, dtype=torch.long)
    # torch.Tensor, (sumC, 4), 当前 batch 内的 voxel key
    key = torch.stack(
        (
            candidate_batch_index.to(dtype=torch.long),
            candidate_voxel_zyx[:, 0].to(dtype=torch.long),
            candidate_voxel_zyx[:, 1].to(dtype=torch.long),
            candidate_voxel_zyx[:, 2].to(dtype=torch.long),
        ),
        dim=1,
    )
    # torch.Tensor, (sumC,), 每条候选对应的 unique key 位置
    inverse = torch.unique(key, dim=0, return_inverse=True)[1]
    # int, unique voxel key 数量
    num_unique = int(inverse.max().item()) + 1
    # torch.Tensor, (sumC_unique,), 每个 key 的最高候选概率
    max_prob_by_key = candidate_prob.new_full((num_unique,), -torch.inf)
    max_prob_by_key.scatter_reduce_(0, inverse, candidate_prob, reduce="amax", include_self=True)
    # torch.Tensor, (sumC,), True 表示该候选达到所属 key 的最高概率
    best_prob_mask = candidate_prob == max_prob_by_key.index_select(0, inverse)
    # torch.Tensor, (sumC,), 非最高概率候选用大行号占位
    candidate_index_or_large = torch.where(
        best_prob_mask,
        candidate_index,
        torch.full_like(candidate_index, num_candidates),
    )
    # torch.Tensor, (sumC_unique,), 每个 key 最高概率候选中最小原始行号
    kept_candidate_index = torch.full((num_unique,), num_candidates, device=candidate_prob.device, dtype=torch.long)
    kept_candidate_index.scatter_reduce_(0, inverse, candidate_index_or_large, reduce="amin", include_self=True)
    return kept_candidate_index.index_select(0, torch.argsort(kept_candidate_index, stable=True))


def build_anchor_coordinates(
    anchor_voxel_zyx: torch.Tensor,
    anchor_batch_index: torch.Tensor,
    box_origin_world: torch.Tensor,
    voxel_size_world: torch.Tensor,
    box_shape_zyx: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """
    将 P anchor 的离散 voxel 坐标转换为 local/world/centered-world 坐标。

    输入参数:
        - anchor_voxel_zyx: torch.Tensor, (sumP, 3), P 来源 voxel 坐标, 轴顺序 z/y/x
        - anchor_batch_index: torch.Tensor, (sumP,), P 所属 BOX 索引
        - box_origin_world: torch.Tensor, (B, 3), BOX 原点世界坐标, 轴顺序 x/y/z
        - voxel_size_world: torch.Tensor, (B, 3), voxel 尺寸, 轴顺序 x/y/z
        - box_shape_zyx: torch.Tensor, (B, 3), BOX 体素形状, 轴顺序 z/y/x

    输出:
        - coords: dict[str, torch.Tensor], P anchor 连续坐标字段字典
            - anchor_coord_local_voxel: torch.Tensor, (sumP, 3), P 中心 local voxel 坐标, 轴顺序 x/y/z
            - anchor_coord_world: torch.Tensor, (sumP, 3), P 世界坐标, 轴顺序 x/y/z
            - anchor_coord_centered_world: torch.Tensor, (sumP, 3), P 相对 BOX 中心的世界坐标, 轴顺序 x/y/z
    """
    # torch.Tensor, (sumP,), voxel z/y/x 整型坐标
    z = anchor_voxel_zyx[:, 0].to(dtype=box_origin_world.dtype)
    y = anchor_voxel_zyx[:, 1].to(dtype=box_origin_world.dtype)
    x = anchor_voxel_zyx[:, 2].to(dtype=box_origin_world.dtype)
    # torch.Tensor, (sumP, 3), corner 语义下的 voxel 中心 local 坐标, 轴顺序 x/y/z
    local_xyz = torch.stack((x + 0.5, y + 0.5, z + 0.5), dim=1)
    # torch.Tensor, (sumP, 3), 每个 P 对应 BOX 的世界原点
    origin_xyz = box_origin_world.index_select(0, anchor_batch_index)
    # torch.Tensor, (sumP, 3), 每个 P 对应 BOX 的 voxel 尺寸
    voxel_size_xyz = voxel_size_world.index_select(0, anchor_batch_index)
    # torch.Tensor, (sumP, 3), P anchor 世界坐标
    world_xyz = origin_xyz + local_xyz * voxel_size_xyz
    # torch.Tensor, (B, 3), BOX 体素形状, 轴顺序 x/y/z
    box_shape_xyz = box_shape_zyx[:, [2, 1, 0]].to(device=box_origin_world.device, dtype=box_origin_world.dtype)
    # torch.Tensor, (sumP, 3), 每个 P 所属 BOX 的中心世界坐标
    box_center_xyz = box_origin_world + 0.5 * box_shape_xyz * voxel_size_world
    # torch.Tensor, (sumP, 3), P 相对 BOX 中心的世界坐标
    centered_world_xyz = world_xyz - box_center_xyz.index_select(0, anchor_batch_index)
    return {
        "anchor_coord_local_voxel": local_xyz,
        "anchor_coord_world": world_xyz,
        "anchor_coord_centered_world": centered_world_xyz,
    }


class SparseAnchorSampler(nn.Module):
    def __init__(
        self,
        candidate_class_ids: Sequence[int],
        max_anchors_per_class: Sequence[int],
        mode: str,
        weight_power: float,
        chunk_size: int,
        nms_radius_voxel: int,
        random_start: bool,
        deduplicate_candidates: bool = False,
    ) -> None:
        """
        从 sparse candidate voxel set C 中采样少量 P anchors。

        输入参数:
            - candidate_class_ids: Sequence[int], (K,), 候选前景类别 ID
            - max_anchors_per_class: Sequence[int], (K,), 每个 BOX/类别最多保留的 P anchor 数
            - mode: str, 采样模式, 取值 weighted_fps/unweighted_fps/topk_nms
            - weight_power: float, weighted_fps 概率权重指数, 建议值 1.0
            - chunk_size: int, weighted_fps 距离更新分块大小, 建议值 8192
            - nms_radius_voxel: int, topk_nms 局部最大池化半径, 建议值 2
            - random_start: bool, torch_cluster.fps 是否随机起点, 建议值 False
            - deduplicate_candidates: bool, 是否在 P 层按同 BOX/voxel 去重, 默认 False

        前向输入:
            - candidate_outputs: dict[str, torch.Tensor], 03 阶段 C 输出字段
                - candidate_voxel_zyx: torch.Tensor, (sumC, 3), 候选 voxel 坐标, 轴顺序 z/y/x
                - candidate_batch_index: torch.Tensor, (sumC,), 候选所属 BOX 索引
                - candidate_class: torch.Tensor, (sumC,), 候选前景类别 ID
                - candidate_prob: torch.Tensor, (sumC,), 候选概率
            - batch: dict[str, Any], collate 后 batch, 提供 BOX 坐标信息
                - box_origin_world: torch.Tensor, (B, 3), BOX 原点世界坐标, 轴顺序 x/y/z
                - voxel_size_world: torch.Tensor, (B, 3), voxel 尺寸, 轴顺序 x/y/z
                - box_shape_zyx: torch.Tensor, (B, 3), BOX 体素形状, 轴顺序 z/y/x

        前向输出:
            - output: dict[str, torch.Tensor], P anchor 坐标、metadata 与计数字段
                - anchor_voxel_zyx: torch.Tensor, (sumP, 3), P 来源 voxel 坐标, 轴顺序 z/y/x
                - anchor_coord_local_voxel: torch.Tensor, (sumP, 3), P 中心 local voxel 坐标, 轴顺序 x/y/z
                - anchor_coord_world: torch.Tensor, (sumP, 3), P 世界坐标, 轴顺序 x/y/z
                - anchor_coord_centered_world: torch.Tensor, (sumP, 3), P 相对 BOX 中心的世界坐标, 轴顺序 x/y/z
                - anchor_batch_index: torch.Tensor, (sumP,), P 所属 BOX 索引
                - anchor_class: torch.Tensor, (sumP,), P 继承的候选前景类别 ID
                - anchor_prob: torch.Tensor, (sumP,), P 继承的候选概率
                - anchor_source_candidate_index: torch.Tensor, (sumP,), P 对应原始 C 行号
                - anchor_counts: torch.Tensor, (B,), 每个 BOX 的 P 总数
                - anchor_counts_by_class: torch.Tensor, (B, K), 每个 BOX/候选类的 P 数量
        """
        super().__init__()
        # tuple[int, ...], (K,), 候选类别 ID
        class_ids = tuple(int(class_id) for class_id in candidate_class_ids)
        # tuple[int, ...], (K,), 每类 P anchor 上限
        max_per_class = tuple(int(max_count) for max_count in max_anchors_per_class)
        if len(class_ids) == 0:
            raise ValueError("candidate_class_ids 不能为空。")
        if len(class_ids) != len(max_per_class):
            raise ValueError("candidate_class_ids 与 max_anchors_per_class 长度必须一致。")
        if any(max_count < 0 for max_count in max_per_class):
            raise ValueError("max_anchors_per_class 每项必须 >= 0。")
        if mode not in {"weighted_fps", "unweighted_fps", "topk_nms"}:
            raise ValueError("mode 只允许 weighted_fps/unweighted_fps/topk_nms。")
        if float(weight_power) < 0.0:
            raise ValueError("weight_power 必须 >= 0。")
        if int(chunk_size) <= 0:
            raise ValueError("chunk_size 必须 > 0。")
        if int(nms_radius_voxel) < 0:
            raise ValueError("nms_radius_voxel 必须 >= 0。")

        self.candidate_class_ids = class_ids
        self.max_anchors_per_class = max_per_class
        self.mode = str(mode)
        self.weight_power = float(weight_power)
        self.chunk_size = int(chunk_size)
        self.nms_radius_voxel = int(nms_radius_voxel)
        self.random_start = bool(random_start)
        self.deduplicate_candidates = bool(deduplicate_candidates)

    # 加权 FPS, 但是不建议用, 看起来需要消耗太多时间
    def _sample_weighted_fps_one_group(
        self,
        candidate_index: torch.Tensor,
        voxel_zyx: torch.Tensor,
        prob: torch.Tensor,
        max_count: int,
    ) -> torch.Tensor:
        """
        对单个 BOX/类别执行 probability-weighted FPS。

        输入参数:
            - candidate_index: torch.Tensor, (M,), 当前组原始 C 行号
            - voxel_zyx: torch.Tensor, (M, 3), 当前组候选 voxel 坐标, 轴顺序 z/y/x
            - prob: torch.Tensor, (M,), 当前组候选概率
            - max_count: int, 当前组最多选择数量

        输出:
            - selected_candidate_index: torch.Tensor, (P_group,), 被选中的原始 C 行号
        """
        # int, 当前 BOX/类别 canonical C 数量
        group_size = int(candidate_index.shape[0])
        if group_size == 0 or max_count <= 0:
            return candidate_index.new_empty((0,))
        if group_size <= max_count:
            return candidate_index.index_select(0, _sort_by_prob_desc(prob))

        # torch.Tensor, (M, 3), voxel 坐标浮点视图
        voxel_float = voxel_zyx.float()
        # torch.Tensor, (M,), 每个候选到已选集合的最小平方距离
        min_dist_to_selected = torch.full((group_size,), float("inf"), device=voxel_zyx.device, dtype=voxel_float.dtype)
        # torch.Tensor, (M,), 概率权重
        prob_weight = prob.float().pow(self.weight_power)
        # torch.Tensor, (P_group,), 已选局部下标
        selected_local = torch.empty((max_count,), device=voxel_zyx.device, dtype=torch.long)
        # torch.Tensor, (M,), bool, 已选候选掩码
        selected_mask = torch.zeros((group_size,), device=voxel_zyx.device, dtype=torch.bool)
        # torch.Tensor, (), 第一枚 anchor 的局部下标
        current_pos = _sort_by_prob_desc(prob)[0]

        for selected_pos in range(max_count):
            selected_local[selected_pos] = current_pos
            selected_mask[current_pos] = True
            # torch.Tensor, (3,), 本轮新增 anchor 的 voxel 坐标
            current_coord = voxel_float[current_pos]
            for start in range(0, group_size, self.chunk_size):
                end = min(start + self.chunk_size, group_size)
                # torch.Tensor, (chunk,), 当前 chunk 到新增 anchor 的平方距离
                dist_chunk = (voxel_float[start:end] - current_coord).pow(2).sum(dim=1)
                min_dist_to_selected[start:end] = torch.minimum(min_dist_to_selected[start:end], dist_chunk)
            # torch.Tensor, (M,), 已选位置置为 -inf, 防止重复选择
            score = min_dist_to_selected * prob_weight
            score = score.masked_fill(selected_mask, -torch.inf)
            current_pos = torch.argmax(score)

        return candidate_index.index_select(0, selected_local)

    def _sample_unweighted_fps_one_group(
        self,
        candidate_index: torch.Tensor,
        voxel_zyx: torch.Tensor,
        prob: torch.Tensor,
        max_count: int,
    ) -> torch.Tensor:
        """
        对单个 BOX/类别调用 torch_cluster.fps 执行无权重 FPS。

        输入参数:
            - candidate_index: torch.Tensor, (M,), 当前组原始 C 行号
            - voxel_zyx: torch.Tensor, (M, 3), 当前组候选 voxel 坐标, 轴顺序 z/y/x
            - prob: torch.Tensor, (M,), 当前组候选概率, 仅用于 M<=cap 时排序
            - max_count: int, 当前组最多选择数量

        输出:
            - selected_candidate_index: torch.Tensor, (P_group,), 被选中的原始 C 行号
        """
        # int, 当前 BOX/类别 canonical C 数量
        group_size = int(candidate_index.shape[0])
        if group_size == 0 or max_count <= 0:
            return candidate_index.new_empty((0,))
        if group_size <= max_count:
            return candidate_index.index_select(0, _sort_by_prob_desc(prob))

        # torch.Tensor, (M, 3), FPS 输入坐标, 轴顺序 x/y/z
        voxel_xyz_float = voxel_zyx[:, [2, 1, 0]].float().contiguous()
        # float, FPS 采样比例
        ratio = min(1.0, float(max_count) / float(group_size))
        # torch.Tensor, (P_raw,), FPS 返回的局部下标
        fps_index = torch_cluster.fps(
            x=voxel_xyz_float,
            batch=None,
            ratio=ratio,
            random_start=self.random_start,
        ).to(device=candidate_index.device, dtype=torch.long)
        return candidate_index.index_select(0, fps_index[:max_count])

    def _sample_topk_nms_one_group(
        self,
        candidate_index: torch.Tensor,
        voxel_zyx: torch.Tensor,
        prob: torch.Tensor,
        max_count: int,
        box_shape_zyx: torch.Tensor,
    ) -> torch.Tensor:
        """
        对单个 BOX/类别执行 dense grid topk-NMS。

        输入参数:
            - candidate_index: torch.Tensor, (M,), 当前组原始 C 行号
            - voxel_zyx: torch.Tensor, (M, 3), 当前组候选 voxel 坐标, 轴顺序 z/y/x
            - prob: torch.Tensor, (M,), 当前组候选概率
            - max_count: int, 当前组最多选择数量
            - box_shape_zyx: torch.Tensor, (3,), 当前 BOX 体素形状, 轴顺序 z/y/x

        输出:
            - selected_candidate_index: torch.Tensor, (P_group,), 被选中的原始 C 行号
        """
        # int, 当前 BOX/类别 canonical C 数量
        group_size = int(candidate_index.shape[0])
        if group_size == 0 or max_count <= 0:
            return candidate_index.new_empty((0,))
        if self.nms_radius_voxel == 0:
            order = _sort_by_prob_desc(prob)
            return candidate_index.index_select(0, order[:max_count])

        # tuple[int, int, int], 当前 BOX 的 D/H/W
        depth, height, width = (int(v) for v in box_shape_zyx.tolist())
        # torch.Tensor, (1, 1, D, H, W), dense 候选分数网格
        score_grid = prob.new_full((1, 1, depth, height, width), -torch.inf)
        score_grid[0, 0, voxel_zyx[:, 0], voxel_zyx[:, 1], voxel_zyx[:, 2]] = prob
        # int, NMS max-pool 半径
        radius = self.nms_radius_voxel
        # torch.Tensor, (1, 1, D, H, W), 局部最大分数
        pooled_score = F.max_pool3d(score_grid, kernel_size=2 * radius + 1, stride=1, padding=radius)
        # torch.Tensor, (M,), True 表示候选为局部最大
        local_max_mask = prob == pooled_score[0, 0, voxel_zyx[:, 0], voxel_zyx[:, 1], voxel_zyx[:, 2]]
        # torch.Tensor, (P_raw,), 局部最大候选的局部下标
        local_max_local_index = local_max_mask.nonzero(as_tuple=False).reshape(-1)
        # torch.Tensor, (P_raw,), 局部最大候选按概率降序排列, 同概率不做额外稳定排序
        order = _sort_by_prob_desc(prob.index_select(0, local_max_local_index))
        # torch.Tensor, (P_group,), 最终选择的局部下标
        selected_local = local_max_local_index.index_select(0, order[:max_count])
        return candidate_index.index_select(0, selected_local)

    def forward(
        self,
        candidate_outputs: dict[str, torch.Tensor],
        batch: dict[str, Any],
    ) -> dict[str, torch.Tensor]:
        """
        从 C 输出中采样 P anchors 并构造坐标与 metadata。

        输入参数:
            - candidate_outputs: dict[str, torch.Tensor], C 输出字段
                - candidate_voxel_zyx: torch.Tensor, (sumC, 3), 候选 voxel 坐标, 轴顺序 z/y/x
                - candidate_batch_index: torch.Tensor, (sumC,), 候选所属 BOX 索引
                - candidate_class: torch.Tensor, (sumC,), 候选前景类别 ID
                - candidate_prob: torch.Tensor, (sumC,), 候选概率
            - batch: dict[str, Any], collate 后 batch, 提供 BOX 坐标信息
                - box_origin_world: torch.Tensor, (B, 3), BOX 原点世界坐标, 轴顺序 x/y/z
                - voxel_size_world: torch.Tensor, (B, 3), voxel 尺寸, 轴顺序 x/y/z
                - box_shape_zyx: torch.Tensor, (B, 3), BOX 体素形状, 轴顺序 z/y/x

        输出:
            - output: dict[str, torch.Tensor], P anchor 坐标、metadata 与计数字段
                - anchor_voxel_zyx: torch.Tensor, (sumP, 3), P 来源 voxel 坐标, 轴顺序 z/y/x
                - anchor_coord_local_voxel: torch.Tensor, (sumP, 3), P 中心 local voxel 坐标, 轴顺序 x/y/z
                - anchor_coord_world: torch.Tensor, (sumP, 3), P 世界坐标, 轴顺序 x/y/z
                - anchor_coord_centered_world: torch.Tensor, (sumP, 3), P 相对 BOX 中心的世界坐标, 轴顺序 x/y/z
                - anchor_batch_index: torch.Tensor, (sumP,), P 所属 BOX 索引
                - anchor_class: torch.Tensor, (sumP,), P 继承的候选前景类别 ID
                - anchor_prob: torch.Tensor, (sumP,), P 继承的候选概率
                - anchor_source_candidate_index: torch.Tensor, (sumP,), P 对应原始 C 行号
                - anchor_counts: torch.Tensor, (B,), 每个 BOX 的 P 总数
                - anchor_counts_by_class: torch.Tensor, (B, K), 每个 BOX/候选类的 P 数量
        """
        # torch.Tensor, (sumC, 3), 候选 voxel 坐标, 轴顺序 z/y/x
        candidate_voxel_zyx = candidate_outputs["candidate_voxel_zyx"].to(dtype=torch.long)
        # torch.Tensor, (sumC,), 候选所属 BOX 索引
        candidate_batch_index = candidate_outputs["candidate_batch_index"].to(dtype=torch.long)
        # torch.Tensor, (sumC,), 候选类别 ID
        candidate_class = candidate_outputs["candidate_class"].to(dtype=torch.long)
        # torch.Tensor, (sumC,), 候选概率
        candidate_prob = candidate_outputs["candidate_prob"]
        # torch.Tensor, (B, 3), BOX 原点世界坐标
        box_origin_world = batch["box_origin_world"]
        # torch.Tensor, (B, 3), voxel 尺寸
        voxel_size_world = batch["voxel_size_world"]
        # torch.Tensor, (B, 3), BOX 体素形状, 轴顺序 z/y/x
        box_shape_zyx = batch["box_shape_zyx"].to(device=candidate_voxel_zyx.device, dtype=torch.long)
        # int, batch 内 BOX 数量
        batch_size = int(box_shape_zyx.shape[0])
        # int, 候选类别数量
        num_classes = len(self.candidate_class_ids)

        if self.deduplicate_candidates:
            # torch.Tensor, (sumC_unique,), P 层去重后保留的原始 C 行号
            kept_index = _deduplicate_candidates_by_voxel(candidate_voxel_zyx, candidate_batch_index, candidate_prob)
        else:
            # torch.Tensor, (sumC_unique,)=(sumC,), 未去重时保留全部原始 C 行号
            kept_index = torch.arange(int(candidate_prob.shape[0]), device=candidate_prob.device, dtype=torch.long)
        # torch.Tensor, (sumC_unique,), P 层输入候选所属 BOX 索引
        kept_batch_index = candidate_batch_index.index_select(0, kept_index)
        # torch.Tensor, (sumC_unique,), P 层输入候选类别 ID
        kept_class = candidate_class.index_select(0, kept_index)

        # torch.Tensor, (K,), 配置候选类别 ID
        class_id_tensor = torch.as_tensor(self.candidate_class_ids, device=kept_class.device, dtype=kept_class.dtype)
        # torch.Tensor, (sumC_unique, K), 候选类别与配置类别的匹配矩阵
        class_match = kept_class[:, None] == class_id_tensor[None, :]
        # torch.Tensor, (sumC_unique,), True 表示该候选类别在配置中启用
        configured_class_mask = class_match.any(dim=1)
        # torch.Tensor, (sumC_unique,), 每条候选所属配置类别的局部下标
        class_pos_by_candidate = class_match.to(dtype=torch.long).argmax(dim=1)
        if bool(configured_class_mask.any()):
            # torch.Tensor, (M_keep,), 先按 BOX 再按配置类别排序的局部下标
            kept_local_order = torch.argsort(
                kept_batch_index[configured_class_mask] * num_classes + class_pos_by_candidate[configured_class_mask],
                stable=True,
            )
            # torch.Tensor, (M_keep,), 已过滤并分组排序后的原始 C 行号
            grouped_index = kept_index[configured_class_mask].index_select(0, kept_local_order)
            # torch.Tensor, (M_keep,), 已过滤并分组排序后的 BOX 索引
            grouped_batch = kept_batch_index[configured_class_mask].index_select(0, kept_local_order)
            # torch.Tensor, (M_keep,), 已过滤并分组排序后的配置类别局部下标
            grouped_class_pos = class_pos_by_candidate[configured_class_mask].index_select(0, kept_local_order)
            # torch.Tensor, (M_keep,), group key: batch_idx * K + class_pos, 充当线性索引
            grouped_key = grouped_batch * num_classes + grouped_class_pos
        else:
            grouped_index = kept_index.new_empty((0,))
            grouped_key = kept_index.new_empty((0,))

        group_start_by_key: dict[int, int] = {}
        group_end_by_key: dict[int, int] = {}
        if int(grouped_key.numel()) > 0:
            # torch.Tensor, (G,), 已排序 group key 的连续段起点
            change_pos = torch.cat(
                (
                    grouped_key.new_tensor([0]),
                    (grouped_key[1:] != grouped_key[:-1]).nonzero(as_tuple=False).reshape(-1) + 1,
                ),
                dim=0,
            )
            # torch.Tensor, (G,), 已排序 group key 的连续段终点
            end_pos = torch.cat((change_pos[1:], grouped_key.new_tensor([int(grouped_key.numel())])), dim=0)
            for start_pos, end_pos_item in zip(change_pos.tolist(), end_pos.tolist()):
                key_int = int(grouped_key[int(start_pos)].item())
                group_start_by_key[key_int] = int(start_pos)
                group_end_by_key[key_int] = int(end_pos_item)

        # list[torch.Tensor], 每个 BOX/类别选中的原始 C 行号
        selected_parts: list[torch.Tensor] = []
        for batch_idx in range(batch_size):
            for class_pos in range(num_classes):
                key_int = batch_idx * num_classes + class_pos
                start_pos = group_start_by_key.get(key_int, 0)
                end_pos = group_end_by_key.get(key_int, 0)
                # torch.Tensor, (M,), 当前 BOX/类别原始 C 行号
                group_index = grouped_index[start_pos:end_pos]
                # torch.Tensor, (M, 3), 当前 BOX/类别 voxel 坐标
                group_voxel = candidate_voxel_zyx.index_select(0, group_index)
                # torch.Tensor, (M,), 当前 BOX/类别概率
                group_prob = candidate_prob.index_select(0, group_index)
                # int, 当前 BOX/类别 P anchor 上限
                max_count = int(self.max_anchors_per_class[class_pos])
                if self.mode == "weighted_fps":
                    selected_group = self._sample_weighted_fps_one_group(group_index, group_voxel, group_prob, max_count)
                elif self.mode == "unweighted_fps":
                    selected_group = self._sample_unweighted_fps_one_group(group_index, group_voxel, group_prob, max_count)
                else:
                    selected_group = self._sample_topk_nms_one_group(
                        group_index,
                        group_voxel,
                        group_prob,
                        max_count,
                        box_shape_zyx[batch_idx],
                    )
                selected_parts.append(selected_group)

        # torch.Tensor, (sumP,), 选中的原始 C 行号
        selected_index = torch.cat(selected_parts, dim=0) if selected_parts else kept_index.new_empty((0,))
        # torch.Tensor, (sumP,), 保持按 BOX/类别采样顺序输出
        anchor_batch_index = candidate_batch_index.index_select(0, selected_index)
        # torch.Tensor, (sumP, 3), P 来源 voxel 坐标
        anchor_voxel_zyx = candidate_voxel_zyx.index_select(0, selected_index)
        # torch.Tensor, (sumP,), P 继承的候选类别 ID
        anchor_class = candidate_class.index_select(0, selected_index)
        # torch.Tensor, (sumP,), P 继承的候选概率
        anchor_prob = candidate_prob.index_select(0, selected_index)
        # torch.Tensor, (B,), 每个 BOX 的 P anchor 数
        anchor_counts = torch.bincount(anchor_batch_index, minlength=batch_size).to(dtype=torch.long)
        # torch.Tensor, (B, K), 每个 BOX/候选类的 P anchor 数
        anchor_counts_by_class = torch.zeros(
            (batch_size, num_classes),
            device=candidate_voxel_zyx.device,
            dtype=torch.long,
        )
        for class_pos, class_id in enumerate(self.candidate_class_ids):
            class_mask = anchor_class == int(class_id)
            if bool(class_mask.any()):
                anchor_counts_by_class[:, class_pos] = torch.bincount(
                    anchor_batch_index[class_mask],
                    minlength=batch_size,
                ).to(dtype=torch.long)

        coords = build_anchor_coordinates(
            anchor_voxel_zyx=anchor_voxel_zyx,
            anchor_batch_index=anchor_batch_index,
            box_origin_world=box_origin_world,
            voxel_size_world=voxel_size_world,
            box_shape_zyx=box_shape_zyx,
        )
        return {
            "anchor_voxel_zyx": anchor_voxel_zyx,
            "anchor_coord_local_voxel": coords["anchor_coord_local_voxel"],
            "anchor_coord_world": coords["anchor_coord_world"],
            "anchor_coord_centered_world": coords["anchor_coord_centered_world"],
            "anchor_batch_index": anchor_batch_index,
            "anchor_class": anchor_class,
            "anchor_prob": anchor_prob,
            "anchor_source_candidate_index": selected_index,
            "anchor_counts": anchor_counts,
            "anchor_counts_by_class": anchor_counts_by_class,
        }
