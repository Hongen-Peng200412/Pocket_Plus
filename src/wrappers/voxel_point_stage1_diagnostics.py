from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import torch
from torch import nn

from src.wrappers.voxel_point_stage1_logging import build_metric_key


@dataclass(frozen=True)
class CpcDiagnosticsConfig:
    """
    CPC validation diagnostics 配置。

    输入参数:
        - enabled: bool, 是否启用 diagnostics
        - num_bins: int, score histogram 阈值 bin 数
        - write_local_artifacts: bool, 是否写本地 artifact
        - log_wandb_curves: bool, 是否上传 W&B 曲线
        - wandb_curve_every_n_validation: int, 每隔多少次 validation 上传曲线
        - output_subdir: str, run dir 下 artifact 子目录
    """

    enabled: bool
    num_bins: int
    write_local_artifacts: bool
    log_wandb_curves: bool
    wandb_curve_every_n_validation: int
    output_subdir: str




# ------------------------------------------------------ 对过程中产生的 payload、警告做打包 ------------------------------------------------------
@dataclass(frozen=True)
class CurvePayload:
    """
    曲线或表格 payload。

    输入参数:
        - columns: tuple[str, ...], (M,), 表格列名(一个元素代表一个表格, 如"threshold")
        - rows: tuple[tuple[object, ...], ...], (N,M), 表格行数据
    如：
        CurvePayload(
            columns=("threshold", "precision", "recall", "f1"),
            rows=(
                (0.1, 0.32, 0.91, 0.47),
                (0.2, 0.45, 0.83, 0.58),
                (0.3, 0.56, 0.72, 0.63),
            ),
        )
    """
    columns: tuple[str, ...]
    rows: tuple[tuple[object, ...], ...]

@dataclass(frozen=True)
class DiagnosticsWarning:
    """
    diagnostics 统计 warning。

    输入参数:
        - code: str, 稳定 warning 代码
        - message: str, 人类可读说明
        - scope: str, warning 所属作用域
        - task_class_name: str | None, task class 名; 非 class-specific warning 为 None
    """

    code: str
    message: str
    scope: str
    task_class_name: str | None


# 合并的类
@dataclass(frozen=True)
class CpcDiagnosticsPayload:
    """
    CPC diagnostics epoch 输出。

    输入参数:
        - scalars: dict[str, torch.Tensor], 标量日志 payload
        - curves: dict[str, CurvePayload], W&B 曲线 payload
        - warnings: tuple[DiagnosticsWarning, ...], warning 列表
        - histograms: dict[str, CurvePayload], 本地 histogram artifact payload
    """

    scalars: dict[str, torch.Tensor]
    curves: dict[str, CurvePayload]
    warnings: tuple[DiagnosticsWarning, ...]
    histograms: dict[str, CurvePayload]








# ----------------------------------------------------- 最终总类 -----------------------------------------------------
class CpcValidationDiagnostics(nn.Module):
    """
    维护 Stage1 CPC validation 固定形状统计 buffer。

    输入参数:
        - config: class CpcDiagnosticsConfig, diagnostics 行为配置
        - class_names: Sequence[str], (C,), 就是 ${dataset.class_names}(二分类时是 class_names: [background, foreground])
        - candidate_class_ids: Sequence[int], (K,), 候选前景 task class id, 必须 = adaptive_expand_factor = max_candidate_voxels_per_class(二分类时是 sparse_refine_task: candidate_class_ids: [1]), K <= C - 1 ———— 在目前的代码和配置中总是 = 且去除C的背景后顺序对应
        - adaptive_expand_factor: Sequence[float], (K,), adaptive threshold 扩张倍数
        - max_candidate_voxels_per_class: Sequence[int], (K,), 每 BOX/类候选 C 上限
    """

    def __init__(
        self,
        *,
        config: CpcDiagnosticsConfig,
        class_names: Sequence[str],
        candidate_class_ids: Sequence[int],
        adaptive_expand_factor: Sequence[float],
        max_candidate_voxels_per_class: Sequence[int],
    ) -> None:
        super().__init__()
        if int(config.num_bins) <= 0:
            raise ValueError("CpcDiagnosticsConfig.num_bins 必须 > 0。")
        self.config = config
        # tuple[str, ...], (C,), task class 名
        self.class_names = tuple(str(name) for name in class_names)
        # tuple[int, ...], (K,), 候选前景 task class id
        self.candidate_class_ids = tuple(int(class_id) for class_id in candidate_class_ids)
        # tuple[float, ...], (K,), adaptive_threshold 模式下 best-F1 命中数扩张倍数
        self.adaptive_expand_factor = tuple(float(value) for value in adaptive_expand_factor)
        # tuple[int, ...], (K,), 每 BOX/候选类最多进入 C 的 voxel 数
        self.max_candidate_voxels_per_class = tuple(int(value) for value in max_candidate_voxels_per_class)
        if not (
            len(self.candidate_class_ids)
            == len(self.adaptive_expand_factor)
            == len(self.max_candidate_voxels_per_class)
        ):
            raise ValueError("candidate_class_ids、adaptive_expand_factor、max_candidate_voxels_per_class 长度必须一致。")
        # int, 候选前景类别数 K
        num_candidate_classes = len(self.candidate_class_ids)
        # int, score histogram bin 数 T
        num_bins = int(config.num_bins)
        self._register_stat_buffers(
            num_candidate_classes=num_candidate_classes,
            num_bins=num_bins,
        )

    def _register_stat_buffers(
        self,
        *,
        num_candidate_classes: int,
        num_bins: int,
    ) -> None:
        """
        集中注册 CPC diagnostics 的固定形状统计 buffer。

        输入参数:
            - num_candidate_classes: int, K, candidate 前景类别数
            - num_bins: int, T, score histogram bin 数

        输出:
            - None, 原地注册 persistent=False buffer
        """
        # dict[str, tuple[tuple[int, ...], torch.dtype, str]], buffer 名到形状、dtype、语义说明的映射
        buffer_specs = {
            "threshold_grid": ((num_bins,), torch.float32, "各个 bin 的左边界, 取值范围 [0,1). 永远是等差数列(作为常量)"),

            "uncapped_best_pos_hist": ((num_candidate_classes, num_bins), torch.long, "dense 全空间 valid voxel 中, 每个候选类各概率 bin 里的 GT 正例数"),
            "uncapped_best_neg_hist": ((num_candidate_classes, num_bins), torch.long, "dense 全空间 valid voxel 中, 每个候选类各概率 bin 里的 GT 负例数"),

            "uncapped_sampling_tp": ((num_candidate_classes,), torch.long, "uncapped 状态的 C 在 dense 全空间命中的 GT 正例数"),
            "uncapped_sampling_fp": ((num_candidate_classes,), torch.long, "uncapped 状态的 C 在 dense 全空间误选的 GT 负例数"),
            "uncapped_sampling_fn": ((num_candidate_classes,), torch.long, "uncapped 状态的 C 在 dense 全空间漏掉的 GT 正例数"),
            "uncapped_sampling_num_gt": ((num_candidate_classes,), torch.long, "dense 全空间 valid voxel 中, 每个候选类的 GT 正例总数"),
            "uncapped_sampling_target_count": ((num_candidate_classes,), torch.long, "sampling 策略在 cap 前计划选入 C 的候选 voxel 数, 就是 output[candidate_target_counts_by_class] "),
            "uncapped_sampling_boundary_hist": ((num_candidate_classes, num_bins), torch.long, "对于 BOXs 来说, 它们实际的 sampling cutoff 概率截断阈值落入各概率 bin 的次数"),

            "capped_tp": ((num_candidate_classes,), torch.long, "实际进入候选集 C 的 GT 正例数"),
            "capped_fp": ((num_candidate_classes,), torch.long, "实际进入候选集 C 的 GT 负例数"),
            "capped_fn": ((num_candidate_classes,), torch.long, "dense 全空间 GT 正例中没有进入候选集 C 的数量"),
            "capped_num_C": ((), torch.long, "validation 内实际进入候选集 C 的 voxel 总数"),
            "capped_num_P": ((), torch.long, "validation 内实际生成的 P anchor 总数"),
            "capped_box_count": ((), torch.long, "validation 内参与 capped 统计的 BOX 总数"),
            "capped_routed_prob_hist": ((num_candidate_classes, num_bins), torch.long, "候选集 C 内, 按照路由(routed)类别(而不是 GT 类别)统计的 candidate_prob 概率分布；它纳入全部 C 且不按 GT 正负例拆分，因此二分类时也不等价于 unrefined_pos_hist"),

            "unrefined_pos_hist": ((num_candidate_classes, num_bins), torch.long, "候选集 C 内, refine 前各概率 bin 里的 GT 正例数"),
            "unrefined_neg_hist": ((num_candidate_classes, num_bins), torch.long, "候选集 C 内, refine 前各概率 bin 里的 GT 负例数"),
            "unrefined_dense_gt": ((num_candidate_classes,), torch.long, "计算 unrefined 端到端 recall 时使用的 dense 全空间 GT 正例总数"),

            "refined_pos_hist": ((num_candidate_classes, num_bins), torch.long, "候选集 C 内, refine 后各概率 bin 里的 GT 正例数"),
            "refined_neg_hist": ((num_candidate_classes, num_bins), torch.long, "候选集 C 内, refine 后各概率 bin 里的 GT 负例数"),
            "refined_dense_gt": ((num_candidate_classes,), torch.long, "计算 refined 端到端 recall 时使用的 dense 全空间 GT 正例总数"),
        }
        self._stat_buffer_descriptions: dict[str, str] = {}
        for name, (shape, dtype, description) in buffer_specs.items():
            self._stat_buffer_descriptions[name] = description
            # torch.Tensor, shape, 当前统计 buffer 的初始值
            if name == "threshold_grid":
                buffer = torch.arange(num_bins, dtype=torch.float32) / float(num_bins)
            else:
                buffer = torch.zeros(shape, dtype=dtype)
            self.register_buffer(name, buffer, persistent=False)

    def reset(self) -> None:
        """
        清空所有 diagnostics buffer。

        输出:
            - None, 原地将统计 buffer 置零
        """
        for buffer in self.buffers():
            if buffer.dtype.is_floating_point:
                if buffer.shape == self.threshold_grid.shape and torch.equal(buffer, self.threshold_grid):
                    continue
                buffer.zero_()
            else:
                buffer.zero_()




    # ---------------------------------------- 更新 uncapped_best_* ----------------------------------------
    def update_uncapped_best(
        self,
        *,
        logits: torch.Tensor,
        target: torch.Tensor,
        valid_mask: torch.Tensor,
        allow_cache_update: bool,
    ) -> None:
        """
        更新 dense 全空间 best-F1 histogram。

        输入参数:
            - logits: torch.Tensor, (B,C,D,H,W), dense ligand logits
            - target: torch.Tensor, (B,D,H,W), dense hard-label target
            - valid_mask: torch.Tensor, (B,D,H,W), dense 有效统计掩码
            - allow_cache_update: bool, 当前 validation 是否允许后续写 cache; 本函数只保留调用契约

        输出:
            - None, 原地累积 self.uncapped_best_pos_hist 和 self.uncapped_best_neg_hist
        """
        del allow_cache_update
        if logits.shape[1] == 1:
            # torch.Tensor, (B,1,D,H,W), sigmoid 后的候选前景概率
            prob_by_class = torch.sigmoid(logits[:, :1]).detach().float()
        else:
            # torch.Tensor, (B,C,D,H,W), softmax 后的 task class 概率
            prob = torch.softmax(logits, dim=1).detach().float()
            # torch.Tensor, (K,), long, candidate class 在 dense logits 通道中的 index
            class_index = torch.as_tensor(self.candidate_class_ids, device=logits.device, dtype=torch.long)
            # torch.Tensor, (B,K,D,H,W), 仅保留候选前景类的概率
            prob_by_class = prob.index_select(dim=1, index=class_index)
        # torch.Tensor, (B,D,H,W), bool, dense 有效统计掩码
        valid = valid_mask.bool()
        for class_pos, class_id in enumerate(self.candidate_class_ids):
            # torch.Tensor, (B,D,H,W), 当前候选类概率
            score = prob_by_class[:, class_pos]
            # torch.Tensor, (B,D,H,W), 当前候选类正例掩码
            positive = target.long() == int(class_id)
            self._accumulate_hist_by_class(
                score=score,
                positive=positive,
                valid=valid,
                class_pos=class_pos,
            )

    def _accumulate_hist_by_class(
        self,
        *,
        score: torch.Tensor,
        positive: torch.Tensor,
        valid: torch.Tensor,
        class_pos: int,
    ) -> None:
        """
        按 candidate class 累积 score histogram。

        输入参数:
            - score: torch.Tensor, (B,D,H,W), 当前类别概率
            - positive: torch.Tensor, (B,D,H,W), 当前类别正例掩码
            - valid: torch.Tensor, (B,D,H,W), 有效统计掩码
            - class_pos: int, candidate class 在 "K" 中的位置(0,1,..K-1)
        输出:
            - None, 原地累积 self.uncapped_best_pos_hist 和 self.uncapped_best_neg_hist
        """
        num_bins = int(self.threshold_grid.numel())
        # torch.Tensor, (M,), valid=True 的 voxel score, M 为有效 voxel 数
        score_valid = score[valid].clamp(0.0, 1.0)
        # torch.Tensor, (M,), 每个有效 voxel 所属的概率 bin index
        bin_idx = torch.floor(score_valid * num_bins).long().clamp(max=num_bins - 1)
        # torch.Tensor, (M,), 每个有效 voxel 是否为当前候选类 GT 正例
        positive_valid = positive[valid]
        self.uncapped_best_pos_hist[class_pos] += torch.bincount(bin_idx[positive_valid], minlength=num_bins).to(
            device=self.uncapped_best_pos_hist.device,
            dtype=torch.long,
        )
        self.uncapped_best_neg_hist[class_pos] += torch.bincount(bin_idx[~positive_valid], minlength=num_bins).to(
            device=self.uncapped_best_neg_hist.device,
            dtype=torch.long,
        )



    # ---------------------------------------- 更新 uncapped_sampling_* ----------------------------------------
    def update_uncapped_sampling(
        self,
        *,
        logits: torch.Tensor,
        target: torch.Tensor,
        valid_mask: torch.Tensor,
        candidate_outputs: Mapping[str, torch.Tensor],
        selection_mode: str,
        use_fixed_warmup: bool,
    ) -> None:
        """
        更新真实 sampling 边界在 dense 空间中的覆盖统计。

        输入参数:
            - logits: torch.Tensor, (B,C,D,H,W), dense ligand logits
            - target: torch.Tensor, (B,D,H,W), dense hard-label target
            - valid_mask: torch.Tensor, (B,D,H,W), dense 有效统计掩码
            - candidate_outputs: Mapping[str, torch.Tensor], candidate builder(SparseCandidateSetBuilder) 输出
            - selection_mode: str, 当前正式 selection mode
            - use_fixed_warmup: bool, 当前是否使用 warmup fixed topc

        输出:
            - None, 原地累积 sampling 统计: 
                - uncapped_sampling_tp/fp/fn
                - uncapped_sampling_num_gt, uncapped_sampling_target_count
                - uncapped_sampling_boundary_hist
        """
        del selection_mode, use_fixed_warmup
        required = ("candidate_p_sampling_by_class", "candidate_target_counts_by_class", "candidate_batch_index", "candidate_voxel_zyx")
        if any(name not in candidate_outputs for name in required):
            return
        # torch.Tensor, (B,D,H,W), bool, dense 有效统计掩码
        valid = valid_mask.bool().to(device=logits.device)
        # torch.Tensor, (B,D,H,W), long, dense hard-label target
        target_on_device = target.to(device=logits.device).long()
        # torch.Tensor, (B,K), 每个 BOX/候选类实际 sampling cutoff 概率
        boundary = candidate_outputs["candidate_p_sampling_by_class"].to(device=logits.device).detach().float()
        # torch.Tensor, (B,K), 每个 BOX/候选类 cap 前目标候选数
        target_counts = candidate_outputs["candidate_target_counts_by_class"].to(device=logits.device).detach().long()

        # torch.Tensor, (sumC,), long, builder 最终输出候选所属 BOX index
        candidate_batch_index = candidate_outputs["candidate_batch_index"].to(device=logits.device, dtype=torch.long)
        # torch.Tensor, (sumC,3), long, builder 最终输出候选的 z/y/x index
        candidate_voxel_zyx = candidate_outputs["candidate_voxel_zyx"].to(device=logits.device, dtype=torch.long)
        # torch.Tensor, (sumC,), bool, 最终 C 行在 dense 空间中的有效统计掩码
        valid_C = valid[candidate_batch_index, candidate_voxel_zyx[:, 0], candidate_voxel_zyx[:, 1], candidate_voxel_zyx[:, 2]]
        # torch.Tensor, (sumC,), long, 最终 C 行对应的 dense hard-label target
        target_C = target_on_device[candidate_batch_index, candidate_voxel_zyx[:, 0], candidate_voxel_zyx[:, 1], candidate_voxel_zyx[:, 2]]
        # int, score histogram bin 数 T
        num_bins = int(self.threshold_grid.numel())

        for class_pos, class_id in enumerate(self.candidate_class_ids):
            # torch.Tensor, (B,D,H,W), 当前候选类在 dense 空间中的 GT 正例掩码
            positive_dense = (target_on_device == int(class_id)) & valid
            self.uncapped_sampling_num_gt[class_pos] += positive_dense.sum().to(device=self.uncapped_sampling_num_gt.device, dtype=torch.long)
            self.uncapped_sampling_target_count[class_pos] += target_counts[:, class_pos].sum().to(device=self.uncapped_sampling_target_count.device, dtype=torch.long)
            # torch.Tensor, (B,), bool, 当前候选类存在有限 sampling boundary 的 BOX 掩码
            finite_boundary = torch.isfinite(boundary[:, class_pos])
            if bool(finite_boundary.any()):
                # torch.Tensor, (M_b,), 当前候选类有效 sampling cutoff 概率
                boundary_values = boundary[:, class_pos][finite_boundary].clamp(0.0, 1.0)
                # torch.Tensor, (M_b,), sampling cutoff 对应的概率 bin index
                boundary_bins = torch.floor(boundary_values * num_bins).long().clamp(max=num_bins - 1)
                self.uncapped_sampling_boundary_hist[class_pos] += torch.bincount(boundary_bins, minlength=num_bins).to(
                    device=self.uncapped_sampling_boundary_hist.device,
                    dtype=torch.long,
                )
            for batch_idx in range(int(logits.shape[0])):
                # torch.Tensor, (sumC,), bool, 当前 BOX 对应的 builder 最终 C 行掩码
                sampled_C = (candidate_batch_index == batch_idx) & valid_C
                # torch.Tensor, (sumC,), bool, 当前 BOX 的 C 行中命中当前 GT 类正例的掩码
                positive_C = sampled_C & (target_C == int(class_id))
                # torch.Tensor, (), sampling 命中的 GT 正例数
                tp = positive_C.sum()
                # torch.Tensor, (), sampling 误选的 GT 负例数
                fp = (sampled_C & (target_C != int(class_id))).sum()
                # torch.Tensor, (), sampling 漏掉的 GT 正例数
                fn = positive_dense[batch_idx].sum() - tp
                self.uncapped_sampling_tp[class_pos] += tp.to(device=self.uncapped_sampling_tp.device, dtype=torch.long)
                self.uncapped_sampling_fp[class_pos] += fp.to(device=self.uncapped_sampling_fp.device, dtype=torch.long)
                self.uncapped_sampling_fn[class_pos] += fn.to(device=self.uncapped_sampling_fn.device, dtype=torch.long)




    # ---------------------------------------- 更新 capped_* ----------------------------------------
    def update_capped(
        self,
        *,
        target: torch.Tensor,
        valid_mask: torch.Tensor,
        candidate_outputs: Mapping[str, torch.Tensor],
    ) -> None:
        """
        更新实际 C 覆盖统计。

        输入参数:
            - target: torch.Tensor, (B,D,H,W), dense hard-label target
            - valid_mask: torch.Tensor, (B,D,H,W), dense 有效统计掩码
            - candidate_outputs: Mapping[str, torch.Tensor], candidate builder 输出

        输出:
            - None, 原地累积 C 覆盖计数:
                - capped_tp/fp/fn
                - capped_num_C, capped_num_P
                - capped_box_count
        """
        # torch.Tensor, (sumC,), long, 每个候选 voxel 所属 BOX index
        idx_b = candidate_outputs["candidate_batch_index"].to(device=target.device, dtype=torch.long)
        # torch.Tensor, (sumC,3), long, 每个候选 voxel 的 z/y/x index
        idx_zyx = candidate_outputs["candidate_voxel_zyx"].to(device=target.device, dtype=torch.long)
        # torch.Tensor, (sumC,), bool, C 内有效统计掩码
        valid_C = valid_mask[idx_b, idx_zyx[:, 0], idx_zyx[:, 1], idx_zyx[:, 2]].bool()
        # torch.Tensor, (sumC,), long, C 内 hard-label target
        target_C = target[idx_b, idx_zyx[:, 0], idx_zyx[:, 1], idx_zyx[:, 2]].long()
        for class_pos, class_id in enumerate(self.candidate_class_ids):
            # torch.Tensor, (B,D,H,W), bool, 当前候选类 dense GT 正例掩码
            positive_dense = (target.long() == int(class_id)) & valid_mask.bool()
            # torch.Tensor, (sumC,), bool, 当前候选类 C 内 GT 正例掩码
            positive_C = (target_C == int(class_id)) & valid_C
            self.capped_tp[class_pos] += positive_C.sum().to(device=self.capped_tp.device, dtype=torch.long)
            self.capped_fp[class_pos] += ((target_C != int(class_id)) & valid_C).sum().to(device=self.capped_fp.device, dtype=torch.long)
            self.capped_fn[class_pos] += (positive_dense.sum() - positive_C.sum()).to(device=self.capped_fn.device, dtype=torch.long)
        self.capped_num_C += candidate_outputs["candidate_counts"].sum().to(device=self.capped_num_C.device, dtype=torch.long)
        if "anchor_counts" in candidate_outputs:
            self.capped_num_P += candidate_outputs["anchor_counts"].sum().to(device=self.capped_num_P.device, dtype=torch.long)
        self.capped_box_count += torch.as_tensor(target.shape[0], device=self.capped_box_count.device, dtype=torch.long)
        self._accumulate_capped_routed_prob_hist(candidate_outputs)

    def _accumulate_capped_routed_prob_hist(self, candidate_outputs: Mapping[str, torch.Tensor]) -> None:
        """
        更新 capped_routed_prob_hist: (num_candidate_classes, num_bins), torch.long, "候选集 C 内, 按照路由(routed)类别(而不是 GT 类别)统计的 candidate_prob 概率分布

        输入参数:
            - candidate_outputs: Mapping[str, torch.Tensor], 包含 candidate_class 和 candidate_prob 的候选集输出

        输出:
            - None, 原地累积 self.capped_routed_prob_hist
        """
        if "candidate_class" not in candidate_outputs or "candidate_prob" not in candidate_outputs:
            return
        # torch.Tensor, (sumC,), long, 最终 C 行的 routed candidate class id
        candidate_class = candidate_outputs["candidate_class"].to(device=self.capped_routed_prob_hist.device, dtype=torch.long)
        # torch.Tensor, (sumC,), float, 最终 C 行的 routed candidate 概率
        candidate_prob = candidate_outputs["candidate_prob"].to(device=self.capped_routed_prob_hist.device).detach().float().clamp(0.0, 1.0)
        # int, histogram bin 数 T
        num_bins = int(self.threshold_grid.numel())
        for class_pos, class_id in enumerate(self.candidate_class_ids):
            # torch.Tensor, (sumC,), bool, 当前 routed class 的最终 C 行掩码
            class_mask = candidate_class == int(class_id)
            if not bool(class_mask.any()):
                continue
            # torch.Tensor, (M_C,), 当前 routed class 的概率 bin index
            bin_idx = torch.floor(candidate_prob[class_mask] * num_bins).long().clamp(max=num_bins - 1)
            self.capped_routed_prob_hist[class_pos] += torch.bincount(bin_idx, minlength=num_bins).to(
                device=self.capped_routed_prob_hist.device,
                dtype=torch.long,
            )




    # ---------------------------------------- 更新 unrefined_* 与 refined_* ----------------------------------------
    def update_unrefined(
        self,
        *,
        candidate_outputs: Mapping[str, torch.Tensor],
        target_C: torch.Tensor,
        valid_C: torch.Tensor,
        dense_num_gt: torch.Tensor,
    ) -> None:
        """
        更新 C 内 unrefined dense logit 判别统计。

        输入参数:
            - candidate_outputs: Mapping[str, torch.Tensor], 包含 candidate_logits: (sumC, C_logits), 每个候选体素的 ligand logits
            - target_C: torch.Tensor, (sumC,), C 级 hard-label target
            - valid_C: torch.Tensor, (sumC,), C 级有效掩码
            - dense_num_gt: torch.Tensor, (B,K), 每 BOX/候选类 dense GT 正例数

        输出:
            - None, 原地累积 C 内 histogram 与 dense GT 计数:
                - self.unrefined_pos_hist
                - self.unrefined_neg_hist
                - self.unrefined_dense_gt
        """
        self._accumulate_C_score_hist(
            logits_C=candidate_outputs["candidate_logits"],
            target_C=target_C,
            valid_C=valid_C,
            dense_num_gt=dense_num_gt,
            pos_hist=self.unrefined_pos_hist,
            neg_hist=self.unrefined_neg_hist,
            dense_gt=self.unrefined_dense_gt,
        )

    def update_refined(
        self,
        *,
        refined_logits_C: torch.Tensor,
        candidate_outputs: Mapping[str, torch.Tensor],
        target_C: torch.Tensor,
        valid_C: torch.Tensor,
        dense_num_gt: torch.Tensor,
    ) -> None:
        """
        更新 C 内 refined logit 判别统计。

        输入参数:
            - refined_logits_C: torch.Tensor, (sumC,C_logits), refined C 级 logits
            - candidate_outputs: Mapping[str, torch.Tensor], candidate builder 输出; 当前仅保留接口一致性
            - target_C: torch.Tensor, (sumC,), C 级 hard-label target
            - valid_C: torch.Tensor, (sumC,), C 级有效掩码
            - dense_num_gt: torch.Tensor, (B,K), 每 BOX/候选类 dense GT 正例数

        输出:
            - None, 原地累积 C 内 histogram 与 dense GT 计数:
                - self.refined_pos_hist
                - self.refined_neg_hist
                - self.refined_dense_gt
        """
        del candidate_outputs
        self._accumulate_C_score_hist(
            logits_C=refined_logits_C,
            target_C=target_C,
            valid_C=valid_C,
            dense_num_gt=dense_num_gt,
            pos_hist=self.refined_pos_hist,
            neg_hist=self.refined_neg_hist,
            dense_gt=self.refined_dense_gt,
        )

    def _accumulate_C_score_hist(
        self,
        *,
        logits_C: torch.Tensor,
        target_C: torch.Tensor,
        valid_C: torch.Tensor,
        dense_num_gt: torch.Tensor,
        pos_hist: torch.Tensor,
        neg_hist: torch.Tensor,
        dense_gt: torch.Tensor,
    ) -> None:
        """
        累积 C 级 score histogram————注意这里的 score 实际都是 prob。

        输入参数:
            - logits_C: torch.Tensor, (sumC, C_logits), C 级 logits
            - target_C: torch.Tensor, (sumC,), C 级 hard-label target
            - valid_C: torch.Tensor, (sumC,), C 级有效掩码
            - dense_num_gt: torch.Tensor, (B,K), dense GT 正例数
            - pos_hist: torch.Tensor, (K,T), 正例 histogram buffer, T=num_bins
            - neg_hist: torch.Tensor, (K,T), 负例 histogram buffer
            - dense_gt: torch.Tensor, (K,), dense GT buffer

        输出:
            - None, 原地累积 buffer
        """
        # int, score histogram bin 数 T
        num_bins = int(self.threshold_grid.numel())
        # torch.Tensor, (sumC,), bool, C 内有效统计掩码
        valid = valid_C.bool()
        if logits_C.shape[1] == 1:
            # torch.Tensor, (sumC,1), sigmoid 后的候选前景概率
            prob_by_class = torch.sigmoid(logits_C[:, :1]).detach().float()
        else:
            # torch.Tensor, (sumC,C_logits), softmax 后的 task class 概率
            prob = torch.softmax(logits_C, dim=1).detach().float()
            # torch.Tensor, (K,), long, candidate class 在 C 级 logits 通道中的 index
            class_index = torch.as_tensor(self.candidate_class_ids, device=logits_C.device, dtype=torch.long)
            # torch.Tensor, (sumC,K), 仅保留候选前景类的概率
            prob_by_class = prob.index_select(dim=1, index=class_index)
        for class_pos, class_id in enumerate(self.candidate_class_ids):
            # torch.Tensor, (M_C,), 当前候选类在有效 C 内的概率
            score = prob_by_class[:, class_pos][valid].clamp(0.0, 1.0)
            # torch.Tensor, (M_C,), 当前候选类在有效 C 内的正例掩码
            positive = (target_C.long() == int(class_id))[valid]
            # torch.Tensor, (M_C,), 当前候选类有效 C 概率 bin index
            bin_idx = torch.floor(score * num_bins).long().clamp(max=num_bins - 1)
            pos_hist[class_pos] += torch.bincount(bin_idx[positive], minlength=num_bins).to(device=pos_hist.device, dtype=torch.long)
            neg_hist[class_pos] += torch.bincount(bin_idx[~positive], minlength=num_bins).to(device=neg_hist.device, dtype=torch.long)
            # torch.Tensor, (), 当前 candidate class 在所有 BOX 中的 dense GT 正例数
            class_dense_gt = dense_num_gt[:, class_pos].sum()
            dense_gt[class_pos] += class_dense_gt.to(device=dense_gt.device, dtype=torch.long)







    # ---------------------------------------------------- 计算 --------------------------------------------------
    # 用于 val_uncapped_best. 从 score histogram 计算并记录 best-F1 、best_F1、best_tp、p_sampling 等等
    def _best_f1_scalars_from_hist(
        self,
        *,
        pos_hist: torch.Tensor,
        neg_hist: torch.Tensor,
        class_pos: int,
        scope: str,
        task_class_name: str,
    ) -> tuple[dict[str, torch.Tensor], tuple[DiagnosticsWarning, ...]]:
        """
        用于 val_uncapped_best. 从 score histogram 计算并记录 best-F1 、best_F1、best_tp 等等

        调用时机:
            - validation step 过程中先累积 uncapped_best_pos_hist / uncapped_best_neg_hist
            - epoch 结束时 self.cpc_diagnostics.compute_payload(sync_fn=self._all_reduce_sum), 然后在每个 candidate class 上调用本函数

        输入参数:
            - pos_hist: torch.Tensor, (T,), 正例 score histogram
            - neg_hist: torch.Tensor, (T,), 负例 score histogram
            - class_pos: int, candidate class 在 K 维中的位置
            - scope: str, 当前只使用 global
            - task_class_name: str, task class 名

        输出:
            - scalars: dict[str, torch.Tensor], best 面板标量, 包含 best-F1 、best_F1、best_tp 等等
            - warnings: tuple[DiagnosticsWarning, ...], 无正例等统计 warning
        """
        # torch.Tensor, (T,), tp[i] = pos_hist[i:]之和
        tp = torch.cumsum(pos_hist.flip(0), dim=0).flip(0).to(dtype=torch.float32)
        # torch.Tensor, (T,), fp[i] = neg_hist[i:]之和
        fp = torch.cumsum(neg_hist.flip(0), dim=0).flip(0).to(dtype=torch.float32)
        # torch.Tensor, (), 当前统计空间 GT 正例总数
        num_gt = pos_hist.sum().to(dtype=torch.float32)
        if float(num_gt.item()) <= 0.0:
            nan = pos_hist.new_tensor(float("nan"), dtype=torch.float32)
            key = build_metric_key(
                panel="val_uncapped_best",
                scope=scope,
                metric="best_F1",
                num_classes=len(self.class_names),
                task_class_name=task_class_name,
            )
            warning = DiagnosticsWarning(
                code="no_positive_gt",
                message="当前统计空间没有 GT 正例。",
                scope=scope,
                task_class_name=task_class_name,
            )
            return {key: nan}, (warning,)
        fn = num_gt - tp
        # torch.Tensor, (T,), precision[i] 代表选取第i个bin对应的概率进行截断时, 所得到的 precision
        precision = tp / (tp + fp).clamp_min(1.0)
        recall = tp / num_gt.clamp_min(1.0)
        f1 = 2.0 * precision * recall / (precision + recall).clamp_min(1.0e-12)
        best_idx = int(torch.argmax(f1).item())
        # torch.Tensor, (), best-F1 命中数按 expand factor 放大后的目标候选数
        sampling_target = torch.ceil((tp[best_idx] + fp[best_idx]) * float(self.adaptive_expand_factor[int(class_pos)]))
        all_hist = pos_hist + neg_hist
        # torch.Tensor, (T,), cumulative_all[i] 代表选中第 (num_bins-i) 个概率阈值进行截断时, 产生的候选体素数目
        cumulative_all = torch.cumsum(all_hist.flip(0), dim=0)
        if float(sampling_target.item()) <= 0.0:
            sampling_idx = best_idx
        elif bool(cumulative_all[-1] < sampling_target.to(device=cumulative_all.device, dtype=cumulative_all.dtype)):
            sampling_idx = 0
        else:
            reversed_idx = int((cumulative_all >= sampling_target.to(device=cumulative_all.device, dtype=cumulative_all.dtype)).nonzero(as_tuple=False)[0].item())
            sampling_idx = int(all_hist.numel() - 1 - reversed_idx)
        base_kwargs = {
            "panel": "val_uncapped_best",
            "scope": scope,
            "num_classes": len(self.class_names),
            "task_class_name": task_class_name,
        }
        return {
            build_metric_key(metric="p_best", **base_kwargs): self.threshold_grid[best_idx],
            build_metric_key(metric="p_sampling", **base_kwargs): self.threshold_grid[sampling_idx],
            build_metric_key(metric="best_F1", **base_kwargs): f1[best_idx],
            build_metric_key(metric="best_precision", **base_kwargs): precision[best_idx],
            build_metric_key(metric="best_recall", **base_kwargs): recall[best_idx],
            build_metric_key(metric="best_tp", **base_kwargs): tp[best_idx],
            build_metric_key(metric="best_fp", **base_kwargs): fp[best_idx],
            build_metric_key(metric="best_fn", **base_kwargs): fn[best_idx],
            build_metric_key(metric="num_gt", **base_kwargs): num_gt,
            build_metric_key(metric="numC_p_best_cutoff", **base_kwargs): tp[best_idx] + fp[best_idx],
            build_metric_key(metric="numC_p_sampling_target", **base_kwargs): sampling_target,
        }, ()


    # 用于 val_uncapped_sampling, 计算并记录 sampling_F1、sampling_precision、sampling_tp 等等
    def _sampling_scalars_for_class(
        self,
        *,
        task_class_name: str,
        tp: torch.Tensor,
        fp: torch.Tensor,
        fn: torch.Tensor,
        num_gt: torch.Tensor,
        target_count: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """
        用于 val_uncapped_sampling, 计算并记录 sampling_F1、sampling_precision、sampling_tp 等等

        输入参数:
            - task_class_name: str, task class 名
            - tp/fp/fn: torch.Tensor, (), builder 最终 C 对 dense GT 的覆盖计数
            - num_gt: torch.Tensor, (), dense GT 正例数
            - target_count: torch.Tensor, (), sampling 策略 cap 前目标候选数
            - boundary_hist: torch.Tensor, (T,), sampling 边界概率 histogram

        输出:
            - scalars: dict[str, torch.Tensor], sampling 面板标量, 包含 sampling_F1、sampling_precision、sampling_tp 等等
        """
        # torch.Tensor, (), sampling 统计的 TP 浮点值
        tp_f = tp.to(dtype=torch.float32)
        # torch.Tensor, (), sampling 统计的 FP 浮点值
        fp_f = fp.to(dtype=torch.float32)
        # torch.Tensor, (), sampling 统计的 FN 浮点值
        fn_f = fn.to(dtype=torch.float32)
        # torch.Tensor, (), sampling precision
        precision = tp_f / (tp_f + fp_f).clamp_min(1.0)
        # torch.Tensor, (), sampling recall
        recall = tp_f / (tp_f + fn_f).clamp_min(1.0)
        # torch.Tensor, (), sampling F1
        f1 = 2.0 * precision * recall / (precision + recall).clamp_min(1.0e-12)
        base_kwargs = {
            "panel": "val_uncapped_sampling",
            "scope": "global",
            "num_classes": len(self.class_names),
            "task_class_name": task_class_name,
        }
        scalars = {
            build_metric_key(metric="sampling_F1", **base_kwargs): f1,
            build_metric_key(metric="sampling_precision", **base_kwargs): precision,
            build_metric_key(metric="sampling_recall", **base_kwargs): recall,
            build_metric_key(metric="sampling_tp", **base_kwargs): tp_f,
            build_metric_key(metric="sampling_fp", **base_kwargs): fp_f,
            build_metric_key(metric="sampling_fn", **base_kwargs): fn_f,
            build_metric_key(metric="num_gt", **base_kwargs): num_gt.to(dtype=torch.float32),
            build_metric_key(metric="numC_sampling_target", **base_kwargs): target_count.to(dtype=torch.float32),
        }
        return scalars


    # 用于 val_capped, 计算并记录 F1、precision 等等
    def _capped_scalars_for_class(
        self,
        *,
        task_class_name: str,
        tp: torch.Tensor,
        fp: torch.Tensor,
        fn: torch.Tensor,
        num_C: torch.Tensor,
        num_P: torch.Tensor,
        box_count: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        用于 val_capped, 计算并记录 F1、precision 等等

        输入参数:
            - task_class_name: str, task class 名
            - tp: torch.Tensor, (), 实际进入 C 的 GT 正例数
            - fp: torch.Tensor, (), 实际进入 C 的 GT 负例数
            - fn: torch.Tensor, (), dense GT 正例中未进入 C 的数量
            - num_C: torch.Tensor, (), validation 内 C 总数
            - num_P: torch.Tensor, (), validation 内 P/anchor 总数
            - box_count: torch.Tensor, (), validation 内 BOX 数

        输出:
            - scalars: dict[str, torch.Tensor], val_capped/global 标量
        """
        # torch.Tensor, (), capped 统计的 TP 浮点值
        tp_f = tp.to(dtype=torch.float32)
        # torch.Tensor, (), capped 统计的 FP 浮点值
        fp_f = fp.to(dtype=torch.float32)
        # torch.Tensor, (), capped 统计的 FN 浮点值
        fn_f = fn.to(dtype=torch.float32)
        # torch.Tensor, (), capped precision
        precision = tp_f / (tp_f + fp_f).clamp_min(1.0)
        # torch.Tensor, (), capped recall
        recall = tp_f / (tp_f + fn_f).clamp_min(1.0)
        # torch.Tensor, (), capped F1
        f1 = 2.0 * precision * recall / (precision + recall).clamp_min(1.0e-12)
        base_kwargs = {
            "panel": "val_capped",
            "scope": "global",
            "num_classes": len(self.class_names),
            "task_class_name": task_class_name,
        }
        box_count_f = box_count.to(dtype=torch.float32).clamp_min(1.0)
        return {
            build_metric_key(metric="F1", **base_kwargs): f1,
            build_metric_key(metric="precision", **base_kwargs): precision,
            build_metric_key(metric="recall", **base_kwargs): recall,
            build_metric_key(metric="tp", **base_kwargs): tp_f,
            build_metric_key(metric="fp", **base_kwargs): fp_f,
            build_metric_key(metric="fn", **base_kwargs): fn_f,
            build_metric_key(metric="num_C", **base_kwargs): num_C.to(dtype=torch.float32) / box_count_f,
            build_metric_key(metric="num_P", **base_kwargs): num_P.to(dtype=torch.float32) / box_count_f,
        }


    # 用于 val_unrefined 或 val_refined, 从 C 的 score histogram 计算并记录 local(C内的)版本的: p(截断概率)、F1、precision 等
    # 同时计算对应 val_score 端到端指标; val_score 使用 dense 全空间 GT 作 recall 分母, 并独立选择使端到端 F1 最大的 p
    def _C_panel_scalars_from_hist(
        self,
        *,
        panel: str,
        score_prefix: str,
        pos_hist: torch.Tensor,
        neg_hist: torch.Tensor,
        dense_gt: torch.Tensor,
        task_class_name: str,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """
        - 用于 val_unrefined 或 val_refined, 从 C 的 score histogram 计算并记录 local(C内的)版本的: p(截断概率)、F1、precision 等
        - 同时计算对应的 unrefined_score 或 refined_score; score 使用 dense 全空间 GT 作 recall 分母, 并独立选择使端到端 F1 最大的 p

        输入参数:
            - panel: str, val_unrefined 或 val_refined
            - score_prefix: str, val_score leaf 前缀, unrefined 或 refined
            - pos_hist: torch.Tensor, (T,), C 内正例 score histogram
            - neg_hist: torch.Tensor, (T,), C 内负例 score histogram
            - dense_gt: torch.Tensor, (), dense 全空间 GT 正例数
            - task_class_name: str, task class 名

        输出:
            - local_scalars: dict[str, torch.Tensor], C 内 local 标量
            - score_scalars: dict[str, torch.Tensor], val_score 端到端标量
        """
        # torch.Tensor, (T,), tp[i] = pos_hist[i:]之和
        tp = torch.cumsum(pos_hist.flip(0), dim=0).flip(0).to(dtype=torch.float32)
        # torch.Tensor, (T,), fp[i] = neg_hist[i:]之和
        fp = torch.cumsum(neg_hist.flip(0), dim=0).flip(0).to(dtype=torch.float32)
        # torch.Tensor, (), C 内 GT 正例总数
        num_gt_in_C = pos_hist.sum().to(dtype=torch.float32)
        # torch.Tensor, (), dense 全空间 GT 正例总数
        dense_gt_f = dense_gt.to(dtype=torch.float32)

        if float(num_gt_in_C.item()) <= 0.0:
            # torch.Tensor, (), C 内无正例时的占位 NaN
            nan = pos_hist.new_tensor(float("nan"), dtype=torch.float32)
            # torch.Tensor, (), val_score 端到端 F1; dense 有正例但 C 完全漏检时记为 0.0, dense 无正例时无定义
            score_f1 = pos_hist.new_tensor(0.0, dtype=torch.float32) if float(dense_gt_f.item()) > 0.0 else nan
            # torch.Tensor, (), val_score 端到端 recall; dense 有正例但 C 完全漏检时为 0.0
            score_recall = pos_hist.new_tensor(0.0, dtype=torch.float32) if float(dense_gt_f.item()) > 0.0 else nan
            local_key = build_metric_key(panel=panel, scope="global", metric="F1", num_classes=len(self.class_names), task_class_name=task_class_name)
            score_kwargs = {
                "panel": "val_score",
                "scope": "global",
                "num_classes": len(self.class_names),
                "task_class_name": task_class_name,
            }
            return {local_key: nan}, {
                build_metric_key(metric=f"{score_prefix}_p", **score_kwargs): nan,
                build_metric_key(metric=f"{score_prefix}_F1", **score_kwargs): score_f1,
                build_metric_key(metric=f"{score_prefix}_precision", **score_kwargs): nan,
                build_metric_key(metric=f"{score_prefix}_recall", **score_kwargs): score_recall,
            }

        # torch.Tensor, (T,), C 内 local FN
        local_fn = num_gt_in_C - tp
        # torch.Tensor, (T,), C 内 precision
        precision = tp / (tp + fp).clamp_min(1.0)
        # torch.Tensor, (T,), C 内 local recall
        local_recall = tp / num_gt_in_C.clamp_min(1.0)
        # torch.Tensor, (T,), C 内 local F1
        local_f1 = 2.0 * precision * local_recall / (precision + local_recall).clamp_min(1.0e-12)

        # int, C 内 local F1 最佳阈值 bin index
        local_best_idx = int(torch.argmax(local_f1).item())
        # torch.Tensor, (T,), 用 dense GT 作分母的端到端 recall
        e2e_recall = tp / dense_gt_f.clamp_min(1.0)
        # torch.Tensor, (T,), 用 dense GT 作分母的端到端 F1; p 独立服务于 val_score 自身
        e2e_f1 = 2.0 * precision * e2e_recall / (precision + e2e_recall).clamp_min(1.0e-12)
        # int, 端到端 F1 最佳阈值 bin index; torch.argmax 保持并列时取第一个最大值的既有策略
        score_best_idx = int(torch.argmax(e2e_f1).item())
        local_kwargs = {
            "panel": panel,
            "scope": "global",
            "num_classes": len(self.class_names),
            "task_class_name": task_class_name,
        }
        score_kwargs = {
            "panel": "val_score",
            "scope": "global",
            "num_classes": len(self.class_names),
            "task_class_name": task_class_name,
        }
        return {
            build_metric_key(metric="p", **local_kwargs): self.threshold_grid[local_best_idx],
            build_metric_key(metric="F1", **local_kwargs): local_f1[local_best_idx],
            build_metric_key(metric="precision", **local_kwargs): precision[local_best_idx],
            build_metric_key(metric="recall", **local_kwargs): local_recall[local_best_idx],
            build_metric_key(metric="tp", **local_kwargs): tp[local_best_idx],
            build_metric_key(metric="fp", **local_kwargs): fp[local_best_idx],
            build_metric_key(metric="fn", **local_kwargs): local_fn[local_best_idx],
            build_metric_key(metric="num_gt_in_C", **local_kwargs): num_gt_in_C,
        }, {
            build_metric_key(metric=f"{score_prefix}_p", **score_kwargs): self.threshold_grid[score_best_idx],
            build_metric_key(metric=f"{score_prefix}_F1", **score_kwargs): e2e_f1[score_best_idx],
            build_metric_key(metric=f"{score_prefix}_precision", **score_kwargs): precision[score_best_idx],
            build_metric_key(metric=f"{score_prefix}_recall", **score_kwargs): e2e_recall[score_best_idx],
        }


    # 总计算
    def compute_payload(self, *, sync_fn: Callable[[torch.Tensor], torch.Tensor]) -> CpcDiagnosticsPayload:
        """
        同步并计算当前 epoch diagnostics payload。

        输入参数:
            - sync_fn: Callable[[torch.Tensor], torch.Tensor], DDP all-reduce sum 函数; 所有 rank 必须对称调用

        输出:
            - payload: CpcDiagnosticsPayload, 标量、曲线、warning 与本地表格 payload
        """
        # torch.Tensor, (K,T), 全局 uncapped_best 正例的 score histogram
        uncapped_best_pos_hist = sync_fn(self.uncapped_best_pos_hist.clone())
        # torch.Tensor, (K,T), 全局 uncapped_best 负例的 score histogram
        uncapped_best_neg_hist = sync_fn(self.uncapped_best_neg_hist.clone())
        # torch.Tensor, (K,), 全局 uncapped_sampling TP
        uncapped_sampling_tp = sync_fn(self.uncapped_sampling_tp.clone())
        # torch.Tensor, (K,), 全局 uncapped_sampling FP
        uncapped_sampling_fp = sync_fn(self.uncapped_sampling_fp.clone())
        # torch.Tensor, (K,), 全局 uncapped_sampling FN
        uncapped_sampling_fn = sync_fn(self.uncapped_sampling_fn.clone())
        # torch.Tensor, (K,), 全局 uncapped_sampling dense GT 正例数
        uncapped_sampling_num_gt = sync_fn(self.uncapped_sampling_num_gt.clone())
        # torch.Tensor, (K,), 全局 uncapped_sampling cap 前目标候选数
        uncapped_sampling_target_count = sync_fn(self.uncapped_sampling_target_count.clone())
        # torch.Tensor, (K,T), 全局 uncapped_sampling boundary histogram
        uncapped_sampling_boundary_hist = sync_fn(self.uncapped_sampling_boundary_hist.clone())
        # torch.Tensor, (K,), 全局 capped TP
        capped_tp = sync_fn(self.capped_tp.clone())
        # torch.Tensor, (K,), 全局 capped FP
        capped_fp = sync_fn(self.capped_fp.clone())
        # torch.Tensor, (K,), 全局 capped FN
        capped_fn = sync_fn(self.capped_fn.clone())
        # torch.Tensor, (), 全局 C 总数
        capped_num_C = sync_fn(self.capped_num_C.clone())
        # torch.Tensor, (), 全局 P anchor 总数
        capped_num_P = sync_fn(self.capped_num_P.clone())
        # torch.Tensor, (), 全局 capped 统计 BOX 数
        capped_box_count = sync_fn(self.capped_box_count.clone())
        # torch.Tensor, (K,T), 最终 C routed candidate 概率 histogram
        capped_routed_prob_hist = sync_fn(self.capped_routed_prob_hist.clone())
        # torch.Tensor, (K,T), 全局 unrefined C 内正例 histogram
        unrefined_pos_hist = sync_fn(self.unrefined_pos_hist.clone())
        # torch.Tensor, (K,T), 全局 unrefined C 内负例 histogram
        unrefined_neg_hist = sync_fn(self.unrefined_neg_hist.clone())
        # torch.Tensor, (K,), unrefined 端到端分数使用的 dense GT 正例数
        unrefined_dense_gt = sync_fn(self.unrefined_dense_gt.clone())
        # torch.Tensor, (K,T), 全局 refined C 内正例 histogram
        refined_pos_hist = sync_fn(self.refined_pos_hist.clone())
        # torch.Tensor, (K,T), 全局 refined C 内负例 histogram
        refined_neg_hist = sync_fn(self.refined_neg_hist.clone())
        # torch.Tensor, (K,), refined 端到端分数使用的 dense GT 正例数
        refined_dense_gt = sync_fn(self.refined_dense_gt.clone())
        # dict[str, torch.Tensor], Lightning scalar 日志 payload
        scalars: dict[str, torch.Tensor] = {}
        # list[DiagnosticsWarning], 当前 epoch 统计 warning
        warnings: list[DiagnosticsWarning] = []
        for class_pos, class_id in enumerate(self.candidate_class_ids):
            # str, 当前 candidate class 对应的 task class 名
            class_name = self.class_names[int(class_id)]
            class_scalars, class_warnings = self._best_f1_scalars_from_hist(
                pos_hist=uncapped_best_pos_hist[class_pos],
                neg_hist=uncapped_best_neg_hist[class_pos],
                class_pos=class_pos,
                scope="global",
                task_class_name=class_name,
            )
            scalars.update(class_scalars)
            warnings.extend(class_warnings)
            scalars.update(
                self._sampling_scalars_for_class(
                    task_class_name=class_name,
                    tp=uncapped_sampling_tp[class_pos],
                    fp=uncapped_sampling_fp[class_pos],
                    fn=uncapped_sampling_fn[class_pos],
                    num_gt=uncapped_sampling_num_gt[class_pos],
                    target_count=uncapped_sampling_target_count[class_pos]
                )
            )
            scalars.update(
                self._capped_scalars_for_class(
                    task_class_name=class_name,
                    tp=capped_tp[class_pos],
                    fp=capped_fp[class_pos],
                    fn=capped_fn[class_pos],
                    num_C=capped_num_C,
                    num_P=capped_num_P,
                    box_count=capped_box_count,
                )
            )
            unrefined_local, unrefined_score = self._C_panel_scalars_from_hist(
                panel="val_unrefined",
                score_prefix="unrefined",
                pos_hist=unrefined_pos_hist[class_pos],
                neg_hist=unrefined_neg_hist[class_pos],
                dense_gt=unrefined_dense_gt[class_pos],
                task_class_name=class_name,
            )
            refined_local, refined_score = self._C_panel_scalars_from_hist(
                panel="val_refined",
                score_prefix="refined",
                pos_hist=refined_pos_hist[class_pos],
                neg_hist=refined_neg_hist[class_pos],
                dense_gt=refined_dense_gt[class_pos],
                task_class_name=class_name,
            )
            scalars.update(unrefined_local)
            scalars.update(refined_local)
            scalars.update(unrefined_score)
            scalars.update(refined_score)
        histograms = {
            "uncapped_sampling_boundary_hist": self._histogram_payload(uncapped_sampling_boundary_hist),
            "capped_routed_prob_hist": self._histogram_payload(capped_routed_prob_hist),
        }
        return CpcDiagnosticsPayload(scalars=scalars, curves={}, warnings=tuple(warnings), histograms=histograms)






    # ---------------------------------------------------- 工具函数 --------------------------------------------------
    def _histogram_payload(self, hist: torch.Tensor) -> CurvePayload:
        """
        将按 candidate class 分桶的 histogram 转成 CSV 友好的表格 payload。

        输入参数:
            - hist: torch.Tensor, (K,T), 每个 candidate class 在每个概率 bin 中的计数

        输出:
            - payload: CurvePayload, columns 为 class/bin/count 结构的表格
        """
        # torch.Tensor, (K,T), CPU long histogram
        hist_cpu = hist.detach().cpu().long()
        # torch.Tensor, (T,), CPU float, bin 左边界
        bin_left = self.threshold_grid.detach().cpu().float()
        # float, 单个概率 bin 的宽度
        bin_width = 1.0 / float(int(bin_left.numel()))
        rows: list[tuple[object, ...]] = []
        for class_pos, class_id in enumerate(self.candidate_class_ids):
            # str, 当前 candidate class 的 task class 名
            class_name = self.class_names[int(class_id)]
            for bin_index, left in enumerate(bin_left):
                rows.append(
                    (
                        class_name,
                        int(class_pos),
                        int(class_id),
                        int(bin_index),
                        float(left.item()),
                        min(float(left.item()) + bin_width, 1.0),
                        int(hist_cpu[class_pos, bin_index].item()),
                    )
                )
        return CurvePayload(
            columns=("class_name", "class_pos", "class_id", "bin_index", "bin_left", "bin_right", "count"),
            rows=tuple(rows),
        )
