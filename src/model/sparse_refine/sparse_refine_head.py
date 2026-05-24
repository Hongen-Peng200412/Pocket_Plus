from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import nn


class SparseRefineHead(nn.Module):
    def __init__(
        self,
        mode: str,
        detach_voxel_logits: bool,
        edge_weight_activation: str,
        distance_weight: Mapping[str, Any],
        message_dim: int,
        edge_hidden_dim: int,
        hidden_dim: int,
        num_layers: int,
        logit_dim: int,
        C_voxel_backbone_dim: int,
        P_point_backbone_dim: int,
        P_atom_head_dim: int,
        P_voxel_backbone_dim: int,
        inputs: Mapping[str, bool],
        candidate_class_ids: Sequence[int],
        candidate_class_embedding_dim: int,
        zero_init_residual: bool,
    ) -> None:
        """
        将 P anchor 内容沿稀疏邻居边聚合回唯一候选 C，并输出 refined ligand logits。

        输入参数:
            - mode: str, 输出模式，取值 `direct` 或 `residual`
            - detach_voxel_logits: bool, 是否阻断 voxel ligand logits 到原分支的梯度
            - edge_weight_activation: str, edge gate 激活，首版只支持 `sigmoid`
            - distance_weight: Mapping[str, Any], 距离权重配置，包含 mode 与 temperature
            - message_dim: int, P -> C 聚合消息通道数
            - edge_hidden_dim: int, edge gate MLP 隐藏通道数
            - hidden_dim: int, C 输出 MLP 隐藏通道数
            - num_layers: int, C 输出 MLP 线性层数
            - logit_dim: int, refined logits 输出通道数；二分类为 1，多分类为类别数
            - C_voxel_backbone_dim: int, C 位置 `voxel_final` 通道数
            - P_point_backbone_dim: int, P 的 point backbone 特征通道数
            - P_atom_head_dim: int, P 的 atom head 特征通道数
            - P_voxel_backbone_dim: int, P 位置 `voxel_final` 通道数
            - inputs: Mapping[str, bool], 输入源开关字典，控制 logits、P/C 特征、相对坐标与类别 embedding
            - candidate_class_ids: Sequence[int], (K,), 可用路由类别 ID
            - candidate_class_embedding_dim: int, 类别 embedding 通道数；仅在 use_candidate_class_embedding=true 时使用
            - zero_init_residual: bool, residual 输出增量末层是否零初始化

        前向输入:
            - voxel_logits: torch.Tensor | None, (sumC, logit_dim), C 位置原始 ligand logits
            - C_voxel_backbone_feat: torch.Tensor | None, (sumC, C_voxel), C 位置 `voxel_final` 特征
            - P_point_backbone_feat: torch.Tensor | None, (sumP, C_point), P 的 final point backbone 特征
            - P_atom_head_feat: torch.Tensor | None, (sumP, C_pseudo), P 的 atom head `pseudo_feature`
            - P_voxel_backbone_feat: torch.Tensor | None, (sumP, C_voxel), P 位置 `voxel_final` 特征
            - anchor_class: torch.Tensor, (sumP,), P 继承的路由类别 ID
            - candidate_neighbor_index: torch.Tensor, (sumC, K_nn), 每个 C 的 P 邻居行号
            - candidate_neighbor_squared_distance: torch.Tensor, (sumC, K_nn), C 到 P 邻居的平方距离
            - candidate_neighbor_relative_coords: torch.Tensor, (sumC, K_nn, 3), C-P centered-world 相对坐标
            - candidate_neighbor_valid_mask: torch.Tensor, (sumC, K_nn), bool 有效邻居掩码

        前向输出:
            - output: dict[str, torch.Tensor], 包含 C 消息有效掩码与 refined logits
        """
        super().__init__()
        if str(mode) not in {"direct", "residual"}:
            raise ValueError("mode 只允许 direct 或 residual。")
        if str(edge_weight_activation) != "sigmoid":
            raise ValueError("edge_weight_activation 首版只支持 sigmoid。")
        distance_mode = str(distance_weight["mode"])
        if distance_mode != "softmax_negative_squared_distance":
            raise ValueError("distance_weight.mode 只支持 softmax_negative_squared_distance。")
        temperature = float(distance_weight["temperature"])
        if temperature <= 0.0:
            raise ValueError("distance_weight.temperature 必须 > 0。")
        if int(message_dim) <= 0 or int(edge_hidden_dim) <= 0 or int(hidden_dim) <= 0:
            raise ValueError("message_dim、edge_hidden_dim 与 hidden_dim 必须 > 0。")
        if int(num_layers) <= 0 or int(logit_dim) <= 0:
            raise ValueError("num_layers 与 logit_dim 必须 > 0。")

        self.mode = str(mode)
        self.detach_voxel_logits = bool(detach_voxel_logits)
        self.distance_temperature = temperature
        self.message_dim = int(message_dim)
        self.logit_dim = int(logit_dim)
        self.inputs = {str(key): bool(value) for key, value in inputs.items()}
        if not (self.inputs.get("use_P_point_backbone_feat", False) or self.inputs.get("use_P_atom_head_feat", False)):
            raise ValueError("P content 至少启用 use_P_point_backbone_feat 或 use_P_atom_head_feat。")
        if self.mode == "residual" and not self.inputs.get("use_voxel_logits", False):
            raise ValueError("residual 模式必须启用 use_voxel_logits。")

        p_content_dim = (
            int(P_point_backbone_dim) * int(self.inputs.get("use_P_point_backbone_feat", False))
            + int(P_atom_head_dim) * int(self.inputs.get("use_P_atom_head_feat", False))
        )
        self.P_content_mlp = nn.Sequential(
            nn.Linear(p_content_dim, self.message_dim),
            nn.SiLU(),
            nn.Linear(self.message_dim, self.message_dim),
        )

        self.register_buffer("_candidate_class_ids", torch.as_tensor(tuple(int(value) for value in candidate_class_ids), dtype=torch.long))
        self.candidate_class_embedding: nn.Embedding | None = None
        edge_input_dim = (
            int(C_voxel_backbone_dim) * int(self.inputs.get("use_C_voxel_backbone_feat", False))
            + int(P_voxel_backbone_dim) * int(self.inputs.get("use_P_voxel_backbone_feat", False))
            + 3 * int(self.inputs.get("use_relative_coords", False))
        )
        if self.inputs.get("use_candidate_class_embedding", False):
            if int(candidate_class_embedding_dim) <= 0:
                raise ValueError("启用类别 embedding 时 candidate_class_embedding_dim 必须 > 0。")
            self.candidate_class_embedding = nn.Embedding(len(tuple(candidate_class_ids)), int(candidate_class_embedding_dim))
            edge_input_dim += int(candidate_class_embedding_dim)
        self.edge_mlp: nn.Module | None
        if edge_input_dim > 0:
            self.edge_mlp = nn.Sequential(
                nn.Linear(edge_input_dim, int(edge_hidden_dim)),
                nn.SiLU(),
                nn.Linear(int(edge_hidden_dim), 1),
            )
        else:
            self.edge_mlp = None

        final_input_dim = self.message_dim
        final_input_dim += self.logit_dim * int(self.inputs.get("use_voxel_logits", False))
        final_input_dim += int(C_voxel_backbone_dim) * int(self.inputs.get("use_C_voxel_backbone_feat", False))
        self.output_mlp = self._build_output_mlp(final_input_dim, int(hidden_dim), self.logit_dim, int(num_layers))
        if self.mode == "residual" and bool(zero_init_residual):
            final_linear = next(module for module in reversed(self.output_mlp) if isinstance(module, nn.Linear))
            nn.init.zeros_(final_linear.weight)
            nn.init.zeros_(final_linear.bias)

    @staticmethod
    def _build_output_mlp(input_dim: int, hidden_dim: int, output_dim: int, num_layers: int) -> nn.Sequential:
        """
        构造 C logits 输出 MLP。

        输入参数:
            - input_dim: int, 拼接后的 C 特征通道数
            - hidden_dim: int, 隐藏通道数
            - output_dim: int, logits 输出通道数
            - num_layers: int, 线性层数

        输出:
            - mlp: nn.Sequential, `(sumC, input_dim) -> (sumC, output_dim)` 的输出网络
        """
        if num_layers == 1:
            return nn.Sequential(nn.Linear(input_dim, output_dim))
        layers: list[nn.Module] = [nn.Linear(input_dim, hidden_dim), nn.SiLU()]
        for _ in range(num_layers - 2):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.SiLU()])
        layers.append(nn.Linear(hidden_dim, output_dim))
        return nn.Sequential(*layers)

    def _anchor_class_embedding(self, anchor_class: torch.Tensor) -> torch.Tensor:
        """
        将 P 路由类别 ID 映射到局部类别 embedding。

        输入参数:
            - anchor_class: torch.Tensor, (sumP,), P 路由类别 ID

        输出:
            - embedding: torch.Tensor, (sumP, C_class), P 路由类别 embedding
        """
        if self.candidate_class_embedding is None:
            raise RuntimeError("类别 embedding 未启用。")
        class_match = anchor_class[:, None] == self._candidate_class_ids.to(device=anchor_class.device)[None, :]
        if not bool(class_match.any(dim=1).all()):
            raise RuntimeError("anchor_class 含未配置的路由类别。")
        local_index = class_match.to(dtype=torch.long).argmax(dim=1)
        return self.candidate_class_embedding(local_index)

    def forward(
        self,
        voxel_logits: torch.Tensor | None,
        C_voxel_backbone_feat: torch.Tensor | None,
        P_point_backbone_feat: torch.Tensor | None,
        P_atom_head_feat: torch.Tensor | None,
        P_voxel_backbone_feat: torch.Tensor | None,
        anchor_class: torch.Tensor,
        candidate_neighbor_index: torch.Tensor,
        candidate_neighbor_squared_distance: torch.Tensor,
        candidate_neighbor_relative_coords: torch.Tensor,
        candidate_neighbor_valid_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        聚合 P 消息并输出 C 上 refined logits。

        输入参数:
            - voxel_logits: torch.Tensor | None, (sumC, logit_dim), C 位置原始 ligand logits
            - C_voxel_backbone_feat: torch.Tensor | None, (sumC, C_voxel), C 位置 `voxel_final` 特征
            - P_point_backbone_feat: torch.Tensor | None, (sumP, C_point), P 的 final point backbone 特征
            - P_atom_head_feat: torch.Tensor | None, (sumP, C_pseudo), P 的 atom head `pseudo_feature`
            - P_voxel_backbone_feat: torch.Tensor | None, (sumP, C_voxel), P 位置 `voxel_final` 特征
            - anchor_class: torch.Tensor, (sumP,), P 继承的路由类别 ID

            - candidate_neighbor_index: torch.Tensor, (sumC, K_nn), 每个 C 的 P 邻居行号
            - candidate_neighbor_squared_distance: torch.Tensor, (sumC, K_nn), C 到 P 邻居的平方距离
            - candidate_neighbor_relative_coords: torch.Tensor, (sumC, K_nn, 3), C-P centered-world 相对坐标
            - candidate_neighbor_valid_mask: torch.Tensor, (sumC, K_nn), bool 有效邻居掩码

        输出:
            - output: dict[str, torch.Tensor], 包含:
                - "candidate_message_valid_mask": torch.Tensor, (sumC,), 是否至少有一个有效 P 邻居
                - "ligand_refine_logits_C": torch.Tensor, (sumC, logit_dim), refined logits
        """
        # list[torch.Tensor], 组成 P content 的启用特征分块
        p_content_parts: list[torch.Tensor] = []
        if self.inputs.get("use_P_point_backbone_feat", False):
            if P_point_backbone_feat is None:
                raise RuntimeError("启用了 use_P_point_backbone_feat，但输入为空。")
            p_content_parts.append(P_point_backbone_feat)
        if self.inputs.get("use_P_atom_head_feat", False):
            if P_atom_head_feat is None:
                raise RuntimeError("启用了 use_P_atom_head_feat，但输入为空。")
            p_content_parts.append(P_atom_head_feat)
        # torch.Tensor, (sumP, H_msg), 每个 P 的消息内容特征
        p_content = self.P_content_mlp(torch.cat(p_content_parts, dim=1))
        # int, 唯一 C 数量
        num_candidates = int(candidate_neighbor_valid_mask.shape[0])
        # int, 每个 C 预留的邻居槽位数
        num_neighbors = int(candidate_neighbor_valid_mask.shape[1])
        if int(p_content.shape[0]) == 0:
            # torch.Tensor, (sumC, K_nn, H_msg), 空 P 集合下的零消息内容
            neighbor_content = p_content.new_zeros((num_candidates, num_neighbors, self.message_dim))
        else:
            # torch.Tensor, (sumC, K_nn, H_msg), 每条 C->P 边取回的 P 消息内容
            neighbor_content = p_content.index_select(0, candidate_neighbor_index.reshape(-1)).reshape(
                num_candidates, num_neighbors, self.message_dim
            )

        # list[torch.Tensor], 组成 edge gate 输入的启用特征分块
        edge_parts: list[torch.Tensor] = []
        if self.inputs.get("use_C_voxel_backbone_feat", False):
            if C_voxel_backbone_feat is None:
                raise RuntimeError("启用了 use_C_voxel_backbone_feat，但输入为空。")
            edge_parts.append(C_voxel_backbone_feat[:, None, :].expand(-1, num_neighbors, -1))   # (sumC, K_nn, C_voxel)
        if self.inputs.get("use_P_voxel_backbone_feat", False):
            if P_voxel_backbone_feat is None:
                raise RuntimeError("启用了 use_P_voxel_backbone_feat，但输入为空。")
            if int(P_voxel_backbone_feat.shape[0]) == 0:
                edge_parts.append(P_voxel_backbone_feat.new_zeros((num_candidates, num_neighbors, P_voxel_backbone_feat.shape[1])))
            else:
                edge_parts.append(P_voxel_backbone_feat.index_select(0, candidate_neighbor_index.reshape(-1)).reshape(num_candidates, num_neighbors, -1))
        if self.inputs.get("use_relative_coords", False):
            edge_parts.append(candidate_neighbor_relative_coords)
        if self.inputs.get("use_candidate_class_embedding", False):
            # torch.Tensor, (sumP, C_class), 每个 P 的路由类别 embedding
            anchor_embedding = self._anchor_class_embedding(anchor_class)
            if int(anchor_embedding.shape[0]) == 0:
                edge_parts.append(anchor_embedding.new_zeros((num_candidates, num_neighbors, anchor_embedding.shape[1])))
            else:
                edge_parts.append(anchor_embedding.index_select(0, candidate_neighbor_index.reshape(-1)).reshape(num_candidates, num_neighbors, -1))
        if self.edge_mlp is None:
            # torch.Tensor, (sumC, K_nn, 1), 未启用 edge 输入时的单位门控
            learned_gate = candidate_neighbor_squared_distance.new_ones((num_candidates, num_neighbors, 1))
        else:
            # torch.Tensor, (sumC, K_nn, 1), 每条边的可学习门控权重
            learned_gate = torch.sigmoid(self.edge_mlp(torch.cat(edge_parts, dim=2)))

        # torch.Tensor, (sumC, K_nn), 负平方距离除以温度后的距离分数
        distance_score = -candidate_neighbor_squared_distance / self.distance_temperature
        min_value = torch.finfo(distance_score.dtype).min
        # torch.Tensor, (sumC, K_nn), 无效邻居置极小值后的 softmax 输入
        masked_score = distance_score.masked_fill(~candidate_neighbor_valid_mask, min_value)
        # torch.Tensor, (sumC, K_nn, 1), 仅在有效邻居内归一化的距离权重
        distance_weight = torch.softmax(masked_score, dim=1).unsqueeze(-1) * candidate_neighbor_valid_mask.unsqueeze(
            -1
        ).to(dtype=distance_score.dtype)
        # torch.Tensor, (sumC, H_msg), 每个 C 聚合得到的消息增量
        candidate_message_delta = (distance_weight * learned_gate * neighbor_content).sum(dim=1)
        # torch.Tensor, (sumC,), True 表示该 C 至少连接到一个有效 P 邻居
        candidate_message_valid_mask = candidate_neighbor_valid_mask.any(dim=1)

        # list[torch.Tensor], 输出 head 的输入特征分块
        final_parts = [candidate_message_delta]
        base_logits: torch.Tensor | None = None
        if self.inputs.get("use_voxel_logits", False):
            if voxel_logits is None:
                raise RuntimeError("启用了 use_voxel_logits，但输入为空。")
            # torch.Tensor, (sumC, logit_dim), residual base 或 direct head 可选输入 logits
            base_logits = voxel_logits.detach() if self.detach_voxel_logits else voxel_logits
            final_parts.insert(0, base_logits)
        if self.inputs.get("use_C_voxel_backbone_feat", False):
            if C_voxel_backbone_feat is None:
                raise RuntimeError("启用了 use_C_voxel_backbone_feat，但输入为空。")
            final_parts.insert(1 if base_logits is not None else 0, C_voxel_backbone_feat)
        # torch.Tensor, (sumC, logit_dim), 输出 MLP 产生的 refined logits 或 logits 增量
        output_logits = self.output_mlp(torch.cat(final_parts, dim=1))
        if self.mode == "residual":
            if base_logits is None:
                raise RuntimeError("residual 模式要求 voxel_logits 非空。")
            output_logits = base_logits + output_logits
        return {
            "candidate_message_valid_mask": candidate_message_valid_mask,
            "ligand_refine_logits_C": output_logits,
        }
