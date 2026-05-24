from __future__ import annotations

import torch
from torch import nn


class AnchorToCandidateKnnSearch(nn.Module):
    """
    在同 BOX/路由类别约束下，为候选 C 搜索 P anchor 的 KNN 消息边。

    输入参数:
        - mode: str, 邻居构造模式，首版只支持 `knn_message`
        - num_neighbors: int, 每个 C 最多连接的 P 数量，建议值 3
        - same_class_only: bool, 必须为 True，仅连接相同路由类别
        - chunk_size: int, 单次距离计算最多处理的 C 数量，建议值 8192

    前向输入:
        - candidate_coord_centered_world: torch.Tensor, (sumC, 3), C 的 centered-world xyz 坐标
        - candidate_batch_index: torch.Tensor, (sumC,), C 所属 BOX 索引
        - candidate_class: torch.Tensor, (sumC,), C 路由类别 ID
        - anchor_coord_centered_world: torch.Tensor, (sumP, 3), P 的 centered-world xyz 坐标
        - anchor_batch_index: torch.Tensor, (sumP,), P 所属 BOX 索引
        - anchor_class: torch.Tensor, (sumP,), P 继承的路由类别 ID

    前向输出:
        - output: dict[str, torch.Tensor], KNN 边字段. 这里的 K=num_neighbors 而不是有效类别数
            - "candidate_neighbor_index": torch.Tensor, (sumC, K), P 行号，无效位置由 mask 屏蔽
            - "candidate_neighbor_squared_distance": torch.Tensor, (sumC, K), C 到 P 的平方距离
            - "candidate_neighbor_relative_coords": torch.Tensor, (sumC, K, 3), C-P centered-world 相对坐标
            - "candidate_neighbor_valid_mask": torch.Tensor, (sumC, K), 有效邻居掩码
    """

    def __init__(
        self,
        mode: str,
        num_neighbors: int,
        same_class_only: bool,
        chunk_size: int,
    ) -> None:
        super().__init__()
        if str(mode) != "knn_message":
            raise ValueError("mode 只支持 knn_message。")
        if int(num_neighbors) <= 0:
            raise ValueError("num_neighbors 必须 > 0。")
        if not bool(same_class_only):
            raise ValueError("same_class_only 必须为 true；P -> C 首版只允许同路由类别消息。")
        if int(chunk_size) <= 0:
            raise ValueError("chunk_size 必须 > 0。")
        self.mode = str(mode)
        self.num_neighbors = int(num_neighbors)
        self.same_class_only = True
        self.chunk_size = int(chunk_size)

    def forward(
        self,
        candidate_coord_centered_world: torch.Tensor,
        candidate_batch_index: torch.Tensor,
        candidate_class: torch.Tensor,
        anchor_coord_centered_world: torch.Tensor,
        anchor_batch_index: torch.Tensor,
        anchor_class: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        搜索 P -> C 的有效邻居边。

        输入参数:
            - candidate_coord_centered_world: torch.Tensor, (sumC, 3), C 的 centered-world xyz 坐标
            - candidate_batch_index: torch.Tensor, (sumC,), C 所属 BOX 索引
            - candidate_class: torch.Tensor, (sumC,), C 路由类别 ID
            - anchor_coord_centered_world: torch.Tensor, (sumP, 3), P 的 centered-world xyz 坐标
            - anchor_batch_index: torch.Tensor, (sumP,), P 所属 BOX 索引
            - anchor_class: torch.Tensor, (sumP,), P 继承的路由类别 ID

        输出:
            - output: dict[str, torch.Tensor], KNN 边字段. 这里的 K=num_neighbors 而不是有效类别数
                - "candidate_neighbor_index": torch.Tensor, (sumC, K), P 行号，无效位置由 mask 屏蔽
                - "candidate_neighbor_squared_distance": torch.Tensor, (sumC, K), C 到 P 的平方距离
                - "candidate_neighbor_relative_coords": torch.Tensor, (sumC, K, 3), C-P centered-world 相对坐标
                - "candidate_neighbor_valid_mask": torch.Tensor, (sumC, K), 有效邻居掩码
        """
        num_candidates = int(candidate_coord_centered_world.shape[0])
        num_anchors = int(anchor_coord_centered_world.shape[0])
        k = self.num_neighbors
        neighbor_index = torch.zeros((num_candidates, k), device=candidate_batch_index.device, dtype=torch.long)
        squared_distance = candidate_coord_centered_world.new_zeros((num_candidates, k))
        relative_coords = candidate_coord_centered_world.new_zeros((num_candidates, k, 3))
        valid_mask = torch.zeros((num_candidates, k), device=candidate_batch_index.device, dtype=torch.bool)
        if num_candidates == 0 or num_anchors == 0:
            return {
                "candidate_neighbor_index": neighbor_index,
                "candidate_neighbor_squared_distance": squared_distance,
                "candidate_neighbor_relative_coords": relative_coords,
                "candidate_neighbor_valid_mask": valid_mask,
            }

        class_stride = int(torch.maximum(candidate_class.max(), anchor_class.max()).item()) + 1  # 用于构造线性索引
        # torch.Tensor, (sumC,), 同 BOX/路由类别的 C 共享同一个分组 key
        candidate_group_key = candidate_batch_index.to(dtype=torch.long) * class_stride + candidate_class.to(dtype=torch.long)
        # torch.Tensor, (sumP,), 同 BOX/路由类别的 P 共享同一个分组 key
        anchor_group_key = anchor_batch_index.to(dtype=torch.long) * class_stride + anchor_class.to(dtype=torch.long)

        # torch.Tensor, (sumC,), 按分组 key 排序后的 C 行号
        candidate_order = torch.argsort(candidate_group_key, stable=True)
        # torch.Tensor, (sumP,), 按分组 key 排序后的 P 行号
        anchor_order = torch.argsort(anchor_group_key, stable=True)
        sorted_candidate_key = candidate_group_key.index_select(0, candidate_order)
        sorted_anchor_key = anchor_group_key.index_select(0, anchor_order)


        # torch.Tensor[bool], (sumC,), C 分组首行掩码
        candidate_group_start_mask = torch.ones((num_candidates,), device=candidate_batch_index.device, dtype=torch.bool)
        candidate_group_start_mask[1:] = sorted_candidate_key[1:] != sorted_candidate_key[:-1]
        # torch.Tensor, (G_C,), C 分组在 candidate_order 中的起点
        candidate_group_start = candidate_group_start_mask.nonzero(as_tuple=False).reshape(-1)
        # torch.Tensor, (G_C,), C 分组在 candidate_order 中的终点
        candidate_group_end = torch.cat((candidate_group_start[1:], candidate_group_start.new_tensor([num_candidates])))
        # torch.Tensor, (G_C,), C 分组 key
        candidate_group_values = sorted_candidate_key.index_select(0, candidate_group_start)

        # torch.Tensor[bool], (sumP,), P 分组首行掩码
        anchor_group_start_mask = torch.ones((num_anchors,), device=anchor_batch_index.device, dtype=torch.bool)
        anchor_group_start_mask[1:] = sorted_anchor_key[1:] != sorted_anchor_key[:-1]
        # torch.Tensor, (G_P,), P 分组在 anchor_order 中的起点
        anchor_group_start = anchor_group_start_mask.nonzero(as_tuple=False).reshape(-1)
        # torch.Tensor, (G_P,), P 分组在 anchor_order 中的终点
        anchor_group_end = torch.cat((anchor_group_start[1:], anchor_group_start.new_tensor([num_anchors])))
        # torch.Tensor, (G_P,), P 分组 key
        anchor_group_values = sorted_anchor_key.index_select(0, anchor_group_start)


        for group_pos in range(int(candidate_group_start.shape[0])):
            # int, 当前 BOX/路由类别分组 key
            group_key = candidate_group_values[group_pos]
            # int, 当前 C 分组在 P 分组数组中的插入位置
            anchor_group_pos = torch.searchsorted(anchor_group_values, group_key)
            if int(anchor_group_pos.item()) >= int(anchor_group_values.shape[0]):
                continue
            if int(anchor_group_values[anchor_group_pos].item()) != int(group_key.item()):
                continue

            # int, 当前 C 分组在 candidate_order 中的起点
            candidate_start = int(candidate_group_start[group_pos].item())
            # int, 当前 C 分组在 candidate_order 中的终点
            candidate_end = int(candidate_group_end[group_pos].item())
            # int, 当前同 key 的 P 分组在 anchor_order 中的起点
            anchor_start = int(anchor_group_start[anchor_group_pos].item())
            # int, 当前同 key 的 P 分组在 anchor_order 中的终点
            anchor_end = int(anchor_group_end[anchor_group_pos].item())

            # torch.Tensor, (C_group,), 当前 BOX/路由类别下的 C 原始行号
            group_candidate_index = candidate_order[candidate_start:candidate_end]
            # torch.Tensor, (P_group,), 当前 BOX/路由类别下的 P 原始行号
            group_anchor_index = anchor_order[anchor_start:anchor_end]
            # torch.Tensor, (P_group, 3), 当前允许连接的 P centered-world 坐标
            group_anchor_coord = anchor_coord_centered_world.index_select(0, group_anchor_index)
            # int, 当前分组实际可返回的邻居数
            actual_k = min(k, int(group_anchor_index.shape[0]))

            for start in range(0, int(group_candidate_index.shape[0]), self.chunk_size):
                # torch.Tensor, (C_chunk,), 当前 chunk 的 C 原始行号
                candidate_index_chunk = group_candidate_index[start : start + self.chunk_size]
                # torch.Tensor, (C_chunk, 3), 当前 chunk 的 C centered-world 坐标
                candidate_coord_chunk = candidate_coord_centered_world.index_select(0, candidate_index_chunk)
                # torch.Tensor, (C_chunk, P_group), 当前 chunk 到允许 P 的平方距离矩阵
                distance_chunk = torch.cdist(candidate_coord_chunk.float(), group_anchor_coord.float()).pow(2).to(
                    dtype=candidate_coord_centered_world.dtype
                )
                # torch.Tensor, (C_chunk, actual_k), 当前 chunk 最邻近的平方距离
                top_distance, top_local_index = torch.topk(distance_chunk, k=actual_k, dim=1, largest=False)
                # torch.Tensor, (C_chunk, actual_k), 当前 chunk 最邻近的 P 原始行号
                top_anchor_index = group_anchor_index.index_select(0, top_local_index.reshape(-1)).reshape(
                    int(candidate_index_chunk.shape[0]), actual_k
                )
                # torch.Tensor, (C_chunk, actual_k, 3), 当前 chunk 最邻近的 P centered-world 坐标
                top_anchor_coord = anchor_coord_centered_world.index_select(0, top_anchor_index.reshape(-1)).reshape(
                    int(candidate_index_chunk.shape[0]), actual_k, 3
                )
                neighbor_index[candidate_index_chunk, :actual_k] = top_anchor_index
                squared_distance[candidate_index_chunk, :actual_k] = top_distance
                relative_coords[candidate_index_chunk, :actual_k] = candidate_coord_chunk[:, None, :] - top_anchor_coord
                valid_mask[candidate_index_chunk, :actual_k] = True
        return {
            "candidate_neighbor_index": neighbor_index,
            "candidate_neighbor_squared_distance": squared_distance,
            "candidate_neighbor_relative_coords": relative_coords,
            "candidate_neighbor_valid_mask": valid_mask,
        }
