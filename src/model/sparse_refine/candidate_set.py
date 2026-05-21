from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn


class SparseCandidateSetBuilder(nn.Module):
    """
    从 voxel ligand logits 生成 sparse candidate voxel set C。

    输入参数:
        - candidate_class_ids: Sequence[int], (K,), 候选前景类别 ID; 单通道 sigmoid 只允许 [1]
        - warmup_topc_per_class: Sequence[int], (K,), warmup 阶段每个 BOX/类别固定 topc
        - adaptive_expand_factor: Sequence[float], (K,), adaptive_threshold 阶段相对 best-F1 体素数的扩张倍数
        - max_candidate_voxels_per_class: Sequence[int], (K,), 每个 BOX/类别候选行上限
        - selection_mode: str, 候选选择模式, 取值 adaptive_threshold 或 recorded_threshold

    forward 输入:
        - voxel_logits_ligand: torch.Tensor, (B,1,D,H,W) 或 (B,C,D,H,W), ligand head logits
        - voxel_valid_mask: torch.Tensor, (B,D,H,W) 或 (B,1,D,H,W), 有效体素掩码
        - p_best_by_class: torch.Tensor | None, (K,), adaptive_threshold 使用的 best-F1 阈值
        - p_sampling_by_class: torch.Tensor | None, (K,), recorded_threshold 使用的 sampling 阈值
        - use_fixed_warmup: bool, True 时忽略阈值并使用 warmup topc

    forward 输出:
        - output: dict[str, torch.Tensor], sparse candidate set C 字段字典
            - "candidate_voxel_zyx": torch.Tensor, (sumC, 3), 候选 voxel 离散索引, 轴顺序 z/y/x
            - "candidate_batch_index": torch.Tensor, (sumC,), 每个候选行所属 BOX 的 batch 索引
            - "candidate_class": torch.Tensor, (sumC,), 每个候选体素的前景类别 ID
            - "candidate_prob": torch.Tensor, (sumC,), 每个候选体素对应类别的概率
            - "candidate_logits": torch.Tensor, (sumC, C_logits), 每个候选体素所在 voxel 的 ligand logits
            - "candidate_counts": torch.Tensor, (B,), 每个 BOX 的实际 C 的数目(候选体素数目)
            - "candidate_counts_by_class": torch.Tensor, (B, K), 每个 BOX/候选类 的实际 C 的数目(候选体素数目)
            - "candidate_p_sampling_by_class": torch.Tensor, (B, K), 每个 BOX/候选类 实际使用的候选截断概率
            - "candidate_target_counts_by_class": torch.Tensor, (B, K), 每个 BOX/候选类 原本打算的选取体素数目(不被最大值限制前)
    """

    def __init__(
        self,
        candidate_class_ids: Sequence[int],
        warmup_topc_per_class: Sequence[int],
        adaptive_expand_factor: Sequence[float],
        max_candidate_voxels_per_class: Sequence[int],
        selection_mode: str,
    ) -> None:
        super().__init__()
        # tuple[int, ...], (K,), 候选类别 ID
        class_ids = tuple(int(class_id) for class_id in candidate_class_ids)
        # tuple[int, ...], (K,), warmup 每类 topc
        warmup_topc = tuple(int(topc) for topc in warmup_topc_per_class)
        # tuple[float, ...], (K,), 每类扩张倍数
        expand_factor = tuple(float(factor) for factor in adaptive_expand_factor)
        # tuple[int, ...], (K,), 每类最大候选行数
        max_per_class = tuple(int(max_count) for max_count in max_candidate_voxels_per_class)
        if len(class_ids) == 0:
            raise ValueError("candidate_class_ids 不能为空。")
        if not (len(class_ids) == len(warmup_topc) == len(expand_factor) == len(max_per_class)):
            raise ValueError("candidate_class_ids、warmup_topc_per_class、adaptive_expand_factor、max_candidate_voxels_per_class 长度必须一致。")
        if any(class_id <= 0 for class_id in class_ids):
            raise ValueError("candidate_class_ids 必须全部为前景类别 ID。")
        if any(topc < 0 for topc in warmup_topc):
            raise ValueError("warmup_topc_per_class 每项必须 >= 0。")
        if any(factor <= 0.0 for factor in expand_factor):
            raise ValueError("adaptive_expand_factor 每项必须 > 0。")
        if any(max_count < 0 for max_count in max_per_class):
            raise ValueError("max_candidate_voxels_per_class 每项必须 >= 0。")
        if selection_mode not in {"adaptive_threshold", "recorded_threshold"}:
            raise ValueError("selection_mode 只允许 adaptive_threshold 或 recorded_threshold。")

        self.candidate_class_ids = class_ids
        self.warmup_topc_per_class = warmup_topc
        self.adaptive_expand_factor = expand_factor
        self.max_candidate_voxels_per_class = max_per_class
        self.selection_mode = str(selection_mode)

    def _normalize_valid_mask(self, voxel_valid_mask: torch.Tensor, logits_shape: torch.Size) -> torch.Tensor:
        """
        规范化 voxel_valid_mask 到 `(B,D,H,W)`。

        输入参数:
            - voxel_valid_mask: torch.Tensor, (B,D,H,W) 或 (B,1,D,H,W), 有效体素掩码
            - logits_shape: torch.Size, voxel logits 形状, 用于校验空间维度

        输出:
            - valid_mask: torch.Tensor, (B,D,H,W), bool 有效体素掩码
        """
        if voxel_valid_mask.ndim == 5 and voxel_valid_mask.shape[1] == 1:
            valid_mask = voxel_valid_mask.squeeze(1)
        elif voxel_valid_mask.ndim == 4:
            valid_mask = voxel_valid_mask
        else:
            raise ValueError(f"voxel_valid_mask 期望为 (B,D,H,W) 或 (B,1,D,H,W)，实际 {tuple(voxel_valid_mask.shape)}")
        if tuple(valid_mask.shape) != (int(logits_shape[0]), int(logits_shape[2]), int(logits_shape[3]), int(logits_shape[4])):
            raise ValueError(f"voxel_valid_mask 形状 {tuple(valid_mask.shape)} 与 logits {tuple(logits_shape)} 不匹配。")
        return valid_mask.bool()

    def _prob_by_candidate_class(self, voxel_logits_ligand: torch.Tensor) -> torch.Tensor:
        """
        生成候选类别概率张量。

        输入参数:
            - voxel_logits_ligand: torch.Tensor, (B,1,D,H,W) 或 (B,C,D,H,W), ligand head logits

        输出:
            - prob_by_class: torch.Tensor, (B,K,D,H,W), 与 candidate_class_ids 对齐的概率张量
        """
        if voxel_logits_ligand.ndim != 5:
            raise ValueError(f"voxel_logits_ligand 期望为 (B,C,D,H,W)，实际 {tuple(voxel_logits_ligand.shape)}")
        if voxel_logits_ligand.shape[1] == 1:
            if self.candidate_class_ids != (1,):
                raise ValueError("单通道 sigmoid ligand logits 只允许 candidate_class_ids=[1]。")
            return torch.sigmoid(voxel_logits_ligand[:, :1])
        if max(self.candidate_class_ids) >= int(voxel_logits_ligand.shape[1]):
            raise ValueError(
                f"candidate_class_ids={self.candidate_class_ids} 超出 logits channel 数 {int(voxel_logits_ligand.shape[1])}。"
            )
        # torch.Tensor, (B,C,D,H,W), 多分类 softmax 概率
        prob = torch.softmax(voxel_logits_ligand, dim=1)
        # torch.Tensor, (K,), 候选类别 channel 索引
        class_index = torch.as_tensor(self.candidate_class_ids, device=voxel_logits_ligand.device, dtype=torch.long)
        return prob.index_select(dim=1, index=class_index)

    def _check_threshold_tensor(self, threshold: torch.Tensor | None, threshold_name: str) -> torch.Tensor:
        """
        校验非 fixed topk 模式需要的阈值缓存。

        输入参数:
            - threshold: torch.Tensor | None, (K,), 阈值缓存
            - threshold_name: str, 错误信息中的阈值名称

        输出:
            - threshold_tensor: torch.Tensor, (K,), floating 阈值缓存
        """
        if threshold is None:
            raise RuntimeError(f"{threshold_name} 缺失，非 warmup fixed topk 阶段无法生成 C。")
        threshold_tensor = threshold.detach().float().reshape(-1)
        if int(threshold_tensor.numel()) != len(self.candidate_class_ids):
            raise ValueError(f"{threshold_name} 长度必须等于 candidate_class_ids 数量。")
        if not bool(torch.isfinite(threshold_tensor).all()):
            raise RuntimeError(f"{threshold_name} 包含 NaN/Inf，非 warmup fixed topk 阶段无法生成 C。")
        return threshold_tensor

    @staticmethod
    def _empty_output(
        batch_size: int,
        num_candidate_classes: int,
        logits_channels: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> dict[str, torch.Tensor]:
        """
        构造空 C 输出字典。

        输入参数:
            - batch_size: int, batch 内 BOX 数
            - num_candidate_classes: int, 候选类别数量 K
            - logits_channels: int, ligand logits 通道数
            - device: torch.device, 输出张量设备
            - dtype: torch.dtype, 概率与 logits 输出 dtype

        输出:
            - output: dict[str, torch.Tensor], 空候选输出字段
        """
        return {
            "candidate_voxel_zyx": torch.empty((0, 3), device=device, dtype=torch.long),
            "candidate_batch_index": torch.empty((0,), device=device, dtype=torch.long),
            "candidate_class": torch.empty((0,), device=device, dtype=torch.long),
            "candidate_prob": torch.empty((0,), device=device, dtype=dtype),
            "candidate_logits": torch.empty((0, logits_channels), device=device, dtype=dtype),
            "candidate_counts": torch.zeros((batch_size,), device=device, dtype=torch.long),
            "candidate_counts_by_class": torch.zeros((batch_size, num_candidate_classes), device=device, dtype=torch.long),
            "candidate_p_sampling_by_class": torch.full((batch_size, num_candidate_classes), float("nan"), device=device, dtype=dtype),
            "candidate_target_counts_by_class": torch.zeros((batch_size, num_candidate_classes), device=device, dtype=torch.long),
        }

    def forward(
        self,
        voxel_logits_ligand: torch.Tensor,
        voxel_valid_mask: torch.Tensor,
        p_best_by_class: torch.Tensor | None,
        p_sampling_by_class: torch.Tensor | None,
        use_fixed_warmup: bool,
    ) -> dict[str, torch.Tensor]:
        """
        从 ligand logits 生成当前 batch 的 sparse candidate set C。

        输入参数:
            - voxel_logits_ligand: torch.Tensor, (B,1,D,H,W) 或 (B,C,D,H,W), ligand head logits
            - voxel_valid_mask: torch.Tensor, (B,D,H,W) 或 (B,1,D,H,W), 有效体素掩码
            - p_best_by_class: torch.Tensor | None, (K,), adaptive_threshold 使用的 best-F1 阈值
            - p_sampling_by_class: torch.Tensor | None, (K,), recorded_threshold 使用的 sampling 阈值
            - use_fixed_warmup: bool, True 时忽略阈值并使用 warmup topc

        输出:
            - output: dict[str, torch.Tensor], sparse candidate set C 字段字典
                - "candidate_voxel_zyx": torch.Tensor, (sumC, 3), 候选 voxel 离散索引, 轴顺序 z/y/x
                - "candidate_batch_index": torch.Tensor, (sumC,), 每个候选行所属 BOX 的 batch 索引
                - "candidate_class": torch.Tensor, (sumC,), 每个候选体素的前景类别 ID
                - "candidate_prob": torch.Tensor, (sumC,), 每个候选体素对应类别的概率
                - "candidate_logits": torch.Tensor, (sumC, C_logits), 每个候选体素所在 voxel 的 ligand logits
                - "candidate_counts": torch.Tensor, (B,), 每个 BOX 的实际 C 的数目(候选体素数目)
                - "candidate_counts_by_class": torch.Tensor, (B, K), 每个 BOX/候选类 的实际 C 的数目(候选体素数目)
                - "candidate_p_sampling_by_class": torch.Tensor, (B, K), 每个 BOX/候选类 实际使用的候选截断概率
                - "candidate_target_counts_by_class": torch.Tensor, (B, K), 每个 BOX/候选类 原本打算的选取体素数目(不被最大值限制前)
        """
        with torch.no_grad():
            # torch.Tensor, (B,C,D,H,W), detach 后仅用于候选筛选和诊断输出
            logits = voxel_logits_ligand.detach()
            # torch.Tensor, (B,D,H,W), bool 有效体素掩码
            valid_mask = self._normalize_valid_mask(voxel_valid_mask=voxel_valid_mask, logits_shape=logits.shape)
            # torch.Tensor, (B,K,D,H,W), 与 candidate_class_ids 对齐的候选概率
            prob_by_class = self._prob_by_candidate_class(logits)
            batch_size = int(logits.shape[0])
            logits_channels = int(logits.shape[1])
            num_candidate_classes = len(self.candidate_class_ids)
            output = self._empty_output(
                batch_size=batch_size,
                num_candidate_classes=num_candidate_classes,
                logits_channels=logits_channels,
                device=logits.device,
                dtype=logits.dtype,
            )

            if self.selection_mode == "adaptive_threshold" and not use_fixed_warmup:
                # torch.Tensor, (K,), adaptive_threshold 使用的 best-F1 阈值
                p_best = self._check_threshold_tensor(p_best_by_class, "p_best_by_class").to(device=logits.device)
                p_sampling = None
            elif self.selection_mode == "recorded_threshold" and not use_fixed_warmup:
                p_best = None
                # torch.Tensor, (K,), recorded_threshold 使用的全局 sampling 阈值
                p_sampling = self._check_threshold_tensor(p_sampling_by_class, "p_sampling_by_class").to(device=logits.device)
            else:
                p_best = None
                p_sampling = None

            candidate_voxel_parts: list[torch.Tensor] = []
            candidate_batch_parts: list[torch.Tensor] = []
            candidate_class_parts: list[torch.Tensor] = []
            candidate_prob_parts: list[torch.Tensor] = []
            candidate_logits_parts: list[torch.Tensor] = []

            for batch_idx in range(batch_size):
                # torch.Tensor, (N_valid, 3), 当前 BOX 有效体素 z/y/x 索引
                valid_zyx = valid_mask[batch_idx].nonzero(as_tuple=False)
                if valid_zyx.numel() == 0:
                    continue
                # torch.Tensor, (N_valid,), 展平有效体素线性索引
                valid_flat_index = valid_zyx[:, 0] * logits.shape[3] * logits.shape[4] + valid_zyx[:, 1] * logits.shape[4] + valid_zyx[:, 2]
                # torch.Tensor, (C,D*H*W), 当前 BOX 展平 logits
                logits_flat = logits[batch_idx].reshape(logits_channels, -1)
                for class_pos, class_id in enumerate(self.candidate_class_ids):
                    # torch.Tensor, (N_valid,), 当前 BOX/类别的有效体素概率
                    prob_valid = prob_by_class[batch_idx, class_pos].reshape(-1).index_select(dim=0, index=valid_flat_index)
                    max_count = int(self.max_candidate_voxels_per_class[class_pos])
                    if use_fixed_warmup:
                        target_count = min(int(self.warmup_topc_per_class[class_pos]), max_count, int(prob_valid.numel()))
                        selected_order = torch.topk(prob_valid, k=target_count).indices if target_count > 0 else torch.empty((0,), device=logits.device, dtype=torch.long)
                        output["candidate_target_counts_by_class"][batch_idx, class_pos] = int(self.warmup_topc_per_class[class_pos])
                    elif self.selection_mode == "adaptive_threshold":
                        n_best_box = int((prob_valid > p_best[class_pos]).sum().item())
                        target_before_cap = int(torch.ceil(prob_valid.new_tensor(n_best_box * self.adaptive_expand_factor[class_pos])).item())
                        target_count = min(target_before_cap, max_count, int(prob_valid.numel()))
                        selected_order = torch.topk(prob_valid, k=target_count).indices if target_count > 0 else torch.empty((0,), device=logits.device, dtype=torch.long)
                        output["candidate_target_counts_by_class"][batch_idx, class_pos] = target_before_cap
                    else:
                        # torch.Tensor, (M,), recorded_threshold 下超过全局阈值的局部有效体素位置
                        threshold_selected = (prob_valid > p_sampling[class_pos]).nonzero(as_tuple=False).reshape(-1)
                        target_before_cap = int(threshold_selected.numel())
                        if target_before_cap > max_count:
                            local_top = torch.topk(prob_valid.index_select(dim=0, index=threshold_selected), k=max_count).indices
                            selected_order = threshold_selected.index_select(dim=0, index=local_top)
                        else:
                            selected_order = threshold_selected
                        output["candidate_target_counts_by_class"][batch_idx, class_pos] = target_before_cap
                        output["candidate_p_sampling_by_class"][batch_idx, class_pos] = p_sampling[class_pos]

                    if selected_order.numel() == 0:
                        continue
                    # torch.Tensor, (M,), 选中候选在全体 voxel 展平空间中的线性索引
                    selected_flat_index = valid_flat_index.index_select(dim=0, index=selected_order)
                    # torch.Tensor, (M, 3), 选中候选 voxel z/y/x 索引
                    selected_zyx = valid_zyx.index_select(dim=0, index=selected_order)
                    # torch.Tensor, (M,), 选中候选概率
                    selected_prob = prob_valid.index_select(dim=0, index=selected_order)
                    # torch.Tensor, (M,C), 选中候选原始 logits
                    selected_logits = logits_flat.index_select(dim=1, index=selected_flat_index).transpose(0, 1).contiguous()
                    if use_fixed_warmup or self.selection_mode == "adaptive_threshold":
                        # float, 当前 BOX/类别实际 topk 截断概率
                        cutoff_prob = selected_prob.min() if selected_prob.numel() > 0 else torch.tensor(float("nan"), device=logits.device, dtype=logits.dtype)
                        output["candidate_p_sampling_by_class"][batch_idx, class_pos] = cutoff_prob

                    candidate_voxel_parts.append(selected_zyx.long())
                    candidate_batch_parts.append(torch.full((int(selected_zyx.shape[0]),), batch_idx, device=logits.device, dtype=torch.long))
                    candidate_class_parts.append(torch.full((int(selected_zyx.shape[0]),), int(class_id), device=logits.device, dtype=torch.long))
                    candidate_prob_parts.append(selected_prob.to(dtype=logits.dtype))
                    candidate_logits_parts.append(selected_logits.to(dtype=logits.dtype))
                    output["candidate_counts"][batch_idx] += int(selected_zyx.shape[0])
                    output["candidate_counts_by_class"][batch_idx, class_pos] = int(selected_zyx.shape[0])

            if len(candidate_voxel_parts) == 0:
                return output
            output["candidate_voxel_zyx"] = torch.cat(candidate_voxel_parts, dim=0)
            output["candidate_batch_index"] = torch.cat(candidate_batch_parts, dim=0)
            output["candidate_class"] = torch.cat(candidate_class_parts, dim=0)
            output["candidate_prob"] = torch.cat(candidate_prob_parts, dim=0).detach()
            output["candidate_logits"] = torch.cat(candidate_logits_parts, dim=0).detach()
            return output
