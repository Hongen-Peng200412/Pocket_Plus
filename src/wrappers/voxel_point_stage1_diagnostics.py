from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

import torch
from torch import nn

from src.wrappers.voxel_point_stage1_logging import build_metric_key


# ------------------------------------------------------ 工具类 --------------------------------------------------------
# 把 source folder 索引化
@dataclass(frozen=True)
class SourceFolderRegistry:
    """
    原始 source folder 名到固定 index 的注册表。

    输入参数:
        - names: tuple[str, ...], (S,), 允许的 source folder 名列表
        - name_to_idx: Mapping[str, int], source folder 名到静态 index 的映射
    """

    names: tuple[str, ...]
    name_to_idx: Mapping[str, int]

    @classmethod
    def from_names(cls, names: Sequence[str]) -> "SourceFolderRegistry":
        """
        从配置中的 source folder 名构造注册表。

        输入参数:
            - names: Sequence[str], (S,), 配置允许的 source folder 名

        输出:
            - registry: SourceFolderRegistry, 固定顺序 source folder 注册表
        """
        source_names = tuple(str(name) for name in names)
        if len(source_names) == 0:
            raise ValueError("source_folder_names 不能为空。")
        if len(set(source_names)) != len(source_names):
            raise ValueError(f"source_folder_names 存在重复项: {source_names}")
        return cls(names=source_names, name_to_idx=MappingProxyType({name: idx for idx, name in enumerate(source_names)}))

    def encode_batch(self, class_names: Sequence[str], *, device: torch.device) -> torch.Tensor:
        """
        将 batch 中每个样本的 source folder 名转化为 class SourceFolderRegistry 的固定 index。

        输入参数:
            - class_names: Sequence[str], (B,), batch metadata 中的原始 source folder 名
            - device: torch.device, 输出 index tensor 所在设备

        输出:
            - source_folder_idx: torch.Tensor, (B,), long, 每个 BOX 对应的 source folder index
        """
        encoded: list[int] = []
        for batch_pos, name in enumerate(class_names):
            source_name = str(name)
            if source_name not in self.name_to_idx:
                raise ValueError(
                    "unknown source folder "
                    f"{source_name!r} at batch position {batch_pos}; "
                    f"allowed source folder names: {self.names}"
                )
            encoded.append(int(self.name_to_idx[source_name]))
        return torch.as_tensor(encoded, device=device, dtype=torch.long)





# ------------------------------------------------------ 对过程中产生的 payload、警告做打包 ------------------------------------------------------
@dataclass(frozen=True)
class CpcDiagnosticsConfig:
    """
    CPC validation diagnostics 配置。

    输入参数:
        - enabled: bool, 是否启用 diagnostics
        - source_folder_breakdown: bool, 是否输出 by_source_folder 指标
        - num_bins: int, score histogram 阈值 bin 数
        - write_local_artifacts: bool, 是否写本地 artifact
        - log_wandb_curves: bool, 是否上传 W&B 曲线
        - wandb_curve_every_n_validation: int, 每隔多少次 validation 上传曲线
        - output_subdir: str, run dir 下 artifact 子目录
    """

    enabled: bool
    source_folder_breakdown: bool
    num_bins: int
    write_local_artifacts: bool
    log_wandb_curves: bool
    wandb_curve_every_n_validation: int
    output_subdir: str

@dataclass(frozen=True)
class CurvePayload:
    """
    曲线或表格 payload。

    输入参数:
        - columns: tuple[str, ...], (M,), 表格列名(一个元素代表一个表格, 如"threshold")
        - rows: tuple[tuple[float, ...], ...], (N,M), 表格行数据
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
    rows: tuple[tuple[float, ...], ...]

@dataclass(frozen=True)
class DiagnosticsWarning:
    """
    diagnostics 统计 warning。

    输入参数:
        - code: str, 稳定 warning 代码
        - message: str, 人类可读说明
        - scope: str, warning 所属作用域
        - source_folder: str | None, source folder 名; global warning 为 None
        - task_class_name: str | None, task class 名; 非 class-specific warning 为 None
    """

    code: str
    message: str
    scope: str
    source_folder: str | None
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
        - local_tables: dict[str, CurvePayload], 本地 artifact 表格 payload
    """

    scalars: dict[str, torch.Tensor]
    curves: dict[str, CurvePayload]
    warnings: tuple[DiagnosticsWarning, ...]
    local_tables: dict[str, CurvePayload]








# ----------------------------------------------------- 最终总类 -----------------------------------------------------
class CpcValidationDiagnostics(nn.Module):
    """
    维护 Stage1 CPC validation 固定形状统计 buffer。

    输入参数:
        - config: class CpcDiagnosticsConfig, diagnostics 行为配置
        - class_names: Sequence[str], (C,), task class 名
        - candidate_class_ids: Sequence[int], (K,), 候选前景 task class id, 必须 = adaptive_expand_factor = max_candidate_voxels_per_class
        - source_folders: class SourceFolderRegistry, source folder 注册表
        - adaptive_expand_factor: Sequence[float], (K,), adaptive threshold 扩张倍数
        - max_candidate_voxels_per_class: Sequence[int], (K,), 每 BOX/类候选 C 上限
    """

    def __init__(
        self,
        *,
        config: CpcDiagnosticsConfig,
        class_names: Sequence[str],
        candidate_class_ids: Sequence[int],
        source_folders: SourceFolderRegistry,
        adaptive_expand_factor: Sequence[float],
        max_candidate_voxels_per_class: Sequence[int],
    ) -> None:
        super().__init__()
        if int(config.num_bins) <= 0:
            raise ValueError("CpcDiagnosticsConfig.num_bins 必须 > 0。")
        self.config = config
        self.class_names = tuple(str(name) for name in class_names)
        self.candidate_class_ids = tuple(int(class_id) for class_id in candidate_class_ids)
        self.source_folders = source_folders
        self.adaptive_expand_factor = tuple(float(value) for value in adaptive_expand_factor)
        self.max_candidate_voxels_per_class = tuple(int(value) for value in max_candidate_voxels_per_class)
        if not (
            len(self.candidate_class_ids)
            == len(self.adaptive_expand_factor)
            == len(self.max_candidate_voxels_per_class)
        ):
            raise ValueError("candidate_class_ids、adaptive_expand_factor、max_candidate_voxels_per_class 长度必须一致。")
        # int, 候选前景类别数 K
        num_candidate_classes = len(self.candidate_class_ids)
        # int, source folder 数 S
        num_source_folders = len(self.source_folders.names)
        # int, score histogram bin 数 T
        num_bins = int(config.num_bins)
        self._register_stat_buffers(
            num_candidate_classes=num_candidate_classes,
            num_source_folders=num_source_folders,
            num_bins=num_bins,
        )

    def _register_stat_buffers(
        self,
        *,
        num_candidate_classes: int,
        num_source_folders: int,
        num_bins: int,
    ) -> None:
        """
        集中注册 CPC diagnostics 的固定形状统计 buffer。

        输入参数:
            - num_candidate_classes: int, K, candidate 前景类别数
            - num_source_folders: int, S, source folder 分组数
            - num_bins: int, T, score histogram bin 数

        输出:
            - None, 原地注册 persistent=False buffer
        """
        # dict[str, tuple[tuple[int, ...], torch.dtype, str]], buffer 名到形状、dtype、语义说明的映射
        buffer_specs = {
            "threshold_grid": ((num_bins,), torch.float32, "score histogram 的 bin lower-edge 阈值网格"),

            "uncapped_best_pos_hist": ((num_candidate_classes, num_bins), torch.long, "dense 全空间 best 面板正例 score histogram"),
            "uncapped_best_neg_hist": ((num_candidate_classes, num_bins), torch.long, "dense 全空间 best 面板负例 score histogram"),
            "uncapped_best_pos_hist_by_source": ((num_source_folders, num_candidate_classes, num_bins), torch.long, "按 source folder 分组的 best 正例 histogram"),
            "uncapped_best_neg_hist_by_source": ((num_source_folders, num_candidate_classes, num_bins), torch.long, "按 source folder 分组的 best 负例 histogram"),

            "capped_tp": ((num_candidate_classes,), torch.long, "实际进入 C 的 GT 正例数"),
            "capped_fp": ((num_candidate_classes,), torch.long, "实际进入 C 的 GT 负例数"),
            "capped_fn": ((num_candidate_classes,), torch.long, "dense GT 正例中未进入 C 的数量"),
            "capped_num_C": ((), torch.long, "validation 内实际 C 总数"),
            "capped_num_P": ((), torch.long, "validation 内实际 P/anchor 总数"),
            "capped_box_count": ((), torch.long, "validation 内参与 capped 统计的 BOX 数"),

            "unrefined_pos_hist": ((num_candidate_classes, num_bins), torch.long, "C 内 dense candidate logits 正例 score histogram"),
            "unrefined_neg_hist": ((num_candidate_classes, num_bins), torch.long, "C 内 dense candidate logits 负例 score histogram"),
            "unrefined_dense_gt": ((num_candidate_classes,), torch.long, "unrefined 端到端 score 使用的 dense GT 正例数"),

            "refined_pos_hist": ((num_candidate_classes, num_bins), torch.long, "C 内 refined logits 正例 score histogram"),
            "refined_neg_hist": ((num_candidate_classes, num_bins), torch.long, "C 内 refined logits 负例 score histogram"),
            "refined_dense_gt": ((num_candidate_classes,), torch.long, "refined 端到端 score 使用的 dense GT 正例数"),
        }
        self._stat_buffer_descriptions: dict[str, str] = {}
        for name, (shape, dtype, description) in buffer_specs.items():
            self._stat_buffer_descriptions[name] = description
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

    def update_uncapped_best(
        self,
        *,
        logits: torch.Tensor,
        target: torch.Tensor,
        valid_mask: torch.Tensor,
        source_folder_idx: torch.Tensor,
        allow_cache_update: bool,
    ) -> None:
        """
        更新 dense 全空间 best-F1 histogram。

        输入参数:
            - logits: torch.Tensor, (B,C,D,H,W), dense ligand logits
            - target: torch.Tensor, (B,D,H,W), dense hard-label target
            - valid_mask: torch.Tensor, (B,D,H,W), dense 有效统计掩码
            - source_folder_idx: torch.Tensor, (B,), source folder index
            - allow_cache_update: bool, 当前 validation 是否允许后续写 cache; 本函数只保留调用契约

        输出:
            - None, 原地累积 histogram
        """
        del allow_cache_update
        if logits.shape[1] == 1:
            # torch.Tensor, (B,1,D,H,W), sigmoid 前景概率
            prob_by_class = torch.sigmoid(logits[:, :1]).detach().float()
        else:
            # torch.Tensor, (B,C,D,H,W), softmax 类别概率
            prob = torch.softmax(logits, dim=1).detach().float()
            class_index = torch.as_tensor(self.candidate_class_ids, device=logits.device, dtype=torch.long)
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
                source_folder_idx=source_folder_idx,
            )

    def _accumulate_hist_by_class(
        self,
        *,
        score: torch.Tensor,
        positive: torch.Tensor,
        valid: torch.Tensor,
        class_pos: int,
        source_folder_idx: torch.Tensor,
    ) -> None:
        """
        按 class 与 source folder 累积 score histogram。

        输入参数:
            - score: torch.Tensor, (B,D,H,W), 当前类别概率
            - positive: torch.Tensor, (B,D,H,W), 当前类别正例掩码
            - valid: torch.Tensor, (B,D,H,W), 有效统计掩码
            - class_pos: int, candidate class 在 K 维中的位置
            - source_folder_idx: torch.Tensor, (B,), source folder index

        输出:
            - None, 原地累积 best histogram
        """
        num_bins = int(self.threshold_grid.numel())
        # torch.Tensor, (B,D,H,W), 有效位置 score
        score_valid = score[valid].clamp(0.0, 1.0)
        # torch.Tensor, (M,), 有效位置 bin index
        bin_idx = torch.floor(score_valid * num_bins).long().clamp(max=num_bins - 1)
        # torch.Tensor, (M,), 有效位置正例标记
        positive_valid = positive[valid]
        self.uncapped_best_pos_hist[class_pos] += torch.bincount(bin_idx[positive_valid], minlength=num_bins).to(
            device=self.uncapped_best_pos_hist.device,
            dtype=torch.long,
        )
        self.uncapped_best_neg_hist[class_pos] += torch.bincount(bin_idx[~positive_valid], minlength=num_bins).to(
            device=self.uncapped_best_neg_hist.device,
            dtype=torch.long,
        )
        for source_idx in range(len(self.source_folders.names)):
            # torch.Tensor, (B,), 当前 source folder 的 BOX 掩码
            box_mask = source_folder_idx == int(source_idx)
            if not bool(box_mask.any()):
                continue
            # torch.Tensor, (B,D,H,W), 当前 source folder 的有效位置掩码
            source_valid = valid & box_mask.view(-1, 1, 1, 1)
            score_source = score[source_valid].clamp(0.0, 1.0)
            bin_source = torch.floor(score_source * num_bins).long().clamp(max=num_bins - 1)
            positive_source = positive[source_valid]
            self.uncapped_best_pos_hist_by_source[source_idx, class_pos] += torch.bincount(
                bin_source[positive_source], minlength=num_bins
            ).to(device=self.uncapped_best_pos_hist_by_source.device, dtype=torch.long)
            self.uncapped_best_neg_hist_by_source[source_idx, class_pos] += torch.bincount(
                bin_source[~positive_source], minlength=num_bins
            ).to(device=self.uncapped_best_neg_hist_by_source.device, dtype=torch.long)

    def update_uncapped_sampling(self, **_: object) -> None:
        """预留 sampling diagnostics 更新接口。"""

    def update_capped(
        self,
        *,
        target: torch.Tensor,
        valid_mask: torch.Tensor,
        candidate_outputs: Mapping[str, torch.Tensor],
        source_folder_idx: torch.Tensor,
    ) -> None:
        """
        更新实际 C 覆盖统计。

        输入参数:
            - target: torch.Tensor, (B,D,H,W), dense hard-label target
            - valid_mask: torch.Tensor, (B,D,H,W), dense 有效统计掩码
            - candidate_outputs: Mapping[str, torch.Tensor], candidate builder 输出
            - source_folder_idx: torch.Tensor, (B,), source folder index; 当前 global 统计不消费

        输出:
            - None, 原地累积 C 覆盖计数
        """
        del source_folder_idx
        idx_b = candidate_outputs["candidate_batch_index"].to(device=target.device, dtype=torch.long)
        idx_zyx = candidate_outputs["candidate_voxel_zyx"].to(device=target.device, dtype=torch.long)
        valid_C = valid_mask[idx_b, idx_zyx[:, 0], idx_zyx[:, 1], idx_zyx[:, 2]].bool()
        target_C = target[idx_b, idx_zyx[:, 0], idx_zyx[:, 1], idx_zyx[:, 2]].long()
        for class_pos, class_id in enumerate(self.candidate_class_ids):
            positive_dense = (target.long() == int(class_id)) & valid_mask.bool()
            positive_C = (target_C == int(class_id)) & valid_C
            self.capped_tp[class_pos] += positive_C.sum().to(device=self.capped_tp.device, dtype=torch.long)
            self.capped_fp[class_pos] += ((target_C != int(class_id)) & valid_C).sum().to(device=self.capped_fp.device, dtype=torch.long)
            self.capped_fn[class_pos] += (positive_dense.sum() - positive_C.sum()).to(device=self.capped_fn.device, dtype=torch.long)
        self.capped_num_C += candidate_outputs["candidate_counts"].sum().to(device=self.capped_num_C.device, dtype=torch.long)
        if "anchor_counts" in candidate_outputs:
            self.capped_num_P += candidate_outputs["anchor_counts"].sum().to(device=self.capped_num_P.device, dtype=torch.long)
        self.capped_box_count += torch.as_tensor(target.shape[0], device=self.capped_box_count.device, dtype=torch.long)

    def update_unrefined(
        self,
        *,
        candidate_outputs: Mapping[str, torch.Tensor],
        target_C: torch.Tensor,
        valid_C: torch.Tensor,
        dense_num_gt: torch.Tensor,
        source_folder_idx: torch.Tensor,
    ) -> None:
        """
        更新 C 内 unrefined dense logit 判别统计。

        输入参数:
            - candidate_outputs: Mapping[str, torch.Tensor], 包含 candidate_logits
            - target_C: torch.Tensor, (sumC,), C 级 hard-label target
            - valid_C: torch.Tensor, (sumC,), C 级有效掩码
            - dense_num_gt: torch.Tensor, (B,), 每 BOX dense GT 正例数
            - source_folder_idx: torch.Tensor, (B,), source folder index; 当前 global 统计不消费

        输出:
            - None, 原地累积 C 内 histogram 与 dense GT 计数
        """
        del source_folder_idx
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
        source_folder_idx: torch.Tensor,
    ) -> None:
        """
        更新 C 内 refined logit 判别统计。

        输入参数:
            - refined_logits_C: torch.Tensor, (sumC,C_logits), refined C 级 logits
            - candidate_outputs: Mapping[str, torch.Tensor], candidate builder 输出; 当前仅保留接口一致性
            - target_C: torch.Tensor, (sumC,), C 级 hard-label target
            - valid_C: torch.Tensor, (sumC,), C 级有效掩码
            - dense_num_gt: torch.Tensor, (B,), 每 BOX dense GT 正例数
            - source_folder_idx: torch.Tensor, (B,), source folder index; 当前 global 统计不消费

        输出:
            - None, 原地累积 C 内 histogram 与 dense GT 计数
        """
        del candidate_outputs, source_folder_idx
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
        累积 C 级 score histogram。

        输入参数:
            - logits_C: torch.Tensor, (sumC,C_logits), C 级 logits
            - target_C: torch.Tensor, (sumC,), C 级 hard-label target
            - valid_C: torch.Tensor, (sumC,), C 级有效掩码
            - dense_num_gt: torch.Tensor, (B,), dense GT 正例数
            - pos_hist: torch.Tensor, (K,T), 正例 histogram buffer
            - neg_hist: torch.Tensor, (K,T), 负例 histogram buffer
            - dense_gt: torch.Tensor, (K,), dense GT buffer

        输出:
            - None, 原地累积 buffer
        """
        num_bins = int(self.threshold_grid.numel())
        valid = valid_C.bool()
        if logits_C.shape[1] == 1:
            prob_by_class = torch.sigmoid(logits_C[:, :1]).detach().float()
        else:
            prob = torch.softmax(logits_C, dim=1).detach().float()
            class_index = torch.as_tensor(self.candidate_class_ids, device=logits_C.device, dtype=torch.long)
            prob_by_class = prob.index_select(dim=1, index=class_index)
        for class_pos, class_id in enumerate(self.candidate_class_ids):
            score = prob_by_class[:, class_pos][valid].clamp(0.0, 1.0)
            positive = (target_C.long() == int(class_id))[valid]
            bin_idx = torch.floor(score * num_bins).long().clamp(max=num_bins - 1)
            pos_hist[class_pos] += torch.bincount(bin_idx[positive], minlength=num_bins).to(device=pos_hist.device, dtype=torch.long)
            neg_hist[class_pos] += torch.bincount(bin_idx[~positive], minlength=num_bins).to(device=neg_hist.device, dtype=torch.long)
            dense_gt[class_pos] += dense_num_gt.sum().to(device=dense_gt.device, dtype=torch.long)

    def compute_payload(self, *, sync_fn: Callable[[torch.Tensor], torch.Tensor]) -> CpcDiagnosticsPayload:
        """
        同步并计算当前 epoch diagnostics payload。

        输入参数:
            - sync_fn: Callable[[torch.Tensor], torch.Tensor], DDP all-reduce sum 函数; 所有 rank 必须对称调用

        输出:
            - payload: CpcDiagnosticsPayload, 标量、曲线、warning 与本地表格 payload
        """
        # torch.Tensor, (K,T), 全局正例 histogram
        pos_hist = sync_fn(self.uncapped_best_pos_hist.clone())
        # torch.Tensor, (K,T), 全局负例 histogram
        neg_hist = sync_fn(self.uncapped_best_neg_hist.clone())
        # torch.Tensor, (S,K,T), source folder 正例 histogram
        pos_hist_by_source = sync_fn(self.uncapped_best_pos_hist_by_source.clone())
        # torch.Tensor, (S,K,T), source folder 负例 histogram
        neg_hist_by_source = sync_fn(self.uncapped_best_neg_hist_by_source.clone())
        capped_tp = sync_fn(self.capped_tp.clone())
        capped_fp = sync_fn(self.capped_fp.clone())
        capped_fn = sync_fn(self.capped_fn.clone())
        capped_num_C = sync_fn(self.capped_num_C.clone())
        capped_num_P = sync_fn(self.capped_num_P.clone())
        capped_box_count = sync_fn(self.capped_box_count.clone())
        unrefined_pos_hist = sync_fn(self.unrefined_pos_hist.clone())
        unrefined_neg_hist = sync_fn(self.unrefined_neg_hist.clone())
        unrefined_dense_gt = sync_fn(self.unrefined_dense_gt.clone())
        refined_pos_hist = sync_fn(self.refined_pos_hist.clone())
        refined_neg_hist = sync_fn(self.refined_neg_hist.clone())
        refined_dense_gt = sync_fn(self.refined_dense_gt.clone())
        scalars: dict[str, torch.Tensor] = {}
        warnings: list[DiagnosticsWarning] = []
        for class_pos, class_id in enumerate(self.candidate_class_ids):
            class_name = self.class_names[int(class_id)]
            class_scalars, class_warnings = self._best_f1_scalars_from_hist(
                pos_hist=pos_hist[class_pos],
                neg_hist=neg_hist[class_pos],
                scope="global",
                source_folder=None,
                task_class_name=class_name,
            )
            scalars.update(class_scalars)
            warnings.extend(class_warnings)
            if self.config.source_folder_breakdown:
                for source_idx, source_name in enumerate(self.source_folders.names):
                    source_scalars, source_warnings = self._best_f1_scalars_from_hist(
                        pos_hist=pos_hist_by_source[source_idx, class_pos],
                        neg_hist=neg_hist_by_source[source_idx, class_pos],
                        scope="by_source_folder",
                        source_folder=source_name,
                        task_class_name=class_name,
                    )
                    scalars.update(source_scalars)
                    warnings.extend(source_warnings)
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
        return CpcDiagnosticsPayload(scalars=scalars, curves={}, warnings=tuple(warnings), local_tables={})

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
        构造 val_capped/global 标量。

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
        tp_f = tp.to(dtype=torch.float32)
        fp_f = fp.to(dtype=torch.float32)
        fn_f = fn.to(dtype=torch.float32)
        precision = tp_f / (tp_f + fp_f).clamp_min(1.0)
        recall = tp_f / (tp_f + fn_f).clamp_min(1.0)
        f1 = 2.0 * precision * recall / (precision + recall).clamp_min(1.0e-12)
        base_kwargs = {
            "panel": "val_capped",
            "subpanel": None,
            "scope": "global",
            "source_folder": None,
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
        从 C 级 histogram 构造 local 面板与 val_score 端到端标量。

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
        tp = torch.cumsum(pos_hist.flip(0), dim=0).flip(0).to(dtype=torch.float32)
        fp = torch.cumsum(neg_hist.flip(0), dim=0).flip(0).to(dtype=torch.float32)
        num_gt_in_C = pos_hist.sum().to(dtype=torch.float32)
        if float(num_gt_in_C.item()) <= 0.0:
            nan = pos_hist.new_tensor(float("nan"), dtype=torch.float32)
            local_key = build_metric_key(panel=panel, subpanel=None, scope="global", source_folder=None, metric="F1", num_classes=len(self.class_names), task_class_name=task_class_name)
            score_key = build_metric_key(panel="val_score", subpanel=None, scope="global", source_folder=None, metric=f"{score_prefix}_F1", num_classes=len(self.class_names), task_class_name=task_class_name)
            return {local_key: nan}, {score_key: nan}
        local_fn = num_gt_in_C - tp
        precision = tp / (tp + fp).clamp_min(1.0)
        local_recall = tp / num_gt_in_C.clamp_min(1.0)
        local_f1 = 2.0 * precision * local_recall / (precision + local_recall).clamp_min(1.0e-12)
        best_idx = int(torch.argmax(local_f1).item())
        dense_gt_f = dense_gt.to(dtype=torch.float32)
        e2e_recall = tp[best_idx] / dense_gt_f.clamp_min(1.0)
        e2e_f1 = 2.0 * precision[best_idx] * e2e_recall / (precision[best_idx] + e2e_recall).clamp_min(1.0e-12)
        local_kwargs = {
            "panel": panel,
            "subpanel": None,
            "scope": "global",
            "source_folder": None,
            "num_classes": len(self.class_names),
            "task_class_name": task_class_name,
        }
        score_kwargs = {
            "panel": "val_score",
            "subpanel": None,
            "scope": "global",
            "source_folder": None,
            "num_classes": len(self.class_names),
            "task_class_name": task_class_name,
        }
        return {
            build_metric_key(metric="p", **local_kwargs): self.threshold_grid[best_idx],
            build_metric_key(metric="F1", **local_kwargs): local_f1[best_idx],
            build_metric_key(metric="precision", **local_kwargs): precision[best_idx],
            build_metric_key(metric="recall", **local_kwargs): local_recall[best_idx],
            build_metric_key(metric="tp", **local_kwargs): tp[best_idx],
            build_metric_key(metric="fp", **local_kwargs): fp[best_idx],
            build_metric_key(metric="fn", **local_kwargs): local_fn[best_idx],
            build_metric_key(metric="num_gt_in_C", **local_kwargs): num_gt_in_C,
        }, {
            build_metric_key(metric=f"{score_prefix}_F1", **score_kwargs): e2e_f1,
            build_metric_key(metric=f"{score_prefix}_precision", **score_kwargs): precision[best_idx],
            build_metric_key(metric=f"{score_prefix}_recall", **score_kwargs): e2e_recall,
        }

    def _best_f1_scalars_from_hist(
        self,
        *,
        pos_hist: torch.Tensor,
        neg_hist: torch.Tensor,
        scope: str,
        source_folder: str | None,
        task_class_name: str,
    ) -> tuple[dict[str, torch.Tensor], tuple[DiagnosticsWarning, ...]]:
        """
        从 score histogram 计算 best-F1 标量。

        输入参数:
            - pos_hist: torch.Tensor, (T,), 正例 score histogram
            - neg_hist: torch.Tensor, (T,), 负例 score histogram
            - scope: str, global 或 by_source_folder
            - source_folder: str | None, source folder 名
            - task_class_name: str, task class 名

        输出:
            - scalars: dict[str, torch.Tensor], best 面板标量
            - warnings: tuple[DiagnosticsWarning, ...], 无正例等统计 warning
        """
        # torch.Tensor, (T,), 从高阈值到低阈值累积的 TP
        tp = torch.cumsum(pos_hist.flip(0), dim=0).flip(0).to(dtype=torch.float32)
        # torch.Tensor, (T,), 从高阈值到低阈值累积的 FP
        fp = torch.cumsum(neg_hist.flip(0), dim=0).flip(0).to(dtype=torch.float32)
        # torch.Tensor, (), 当前统计空间 GT 正例总数
        num_gt = pos_hist.sum().to(dtype=torch.float32)
        if float(num_gt.item()) <= 0.0:
            nan = pos_hist.new_tensor(float("nan"), dtype=torch.float32)
            key = build_metric_key(
                panel="val_uncapped",
                subpanel="best",
                scope=scope,
                source_folder=source_folder,
                metric="best_F1",
                num_classes=len(self.class_names),
                task_class_name=task_class_name,
            )
            warning = DiagnosticsWarning(
                code="no_positive_gt",
                message="当前统计空间没有 GT 正例。",
                scope=scope,
                source_folder=source_folder,
                task_class_name=task_class_name,
            )
            return {key: nan}, (warning,)
        fn = num_gt - tp
        precision = tp / (tp + fp).clamp_min(1.0)
        recall = tp / num_gt.clamp_min(1.0)
        f1 = 2.0 * precision * recall / (precision + recall).clamp_min(1.0e-12)
        best_idx = int(torch.argmax(f1).item())
        base_kwargs = {
            "panel": "val_uncapped",
            "subpanel": "best",
            "scope": scope,
            "source_folder": source_folder,
            "num_classes": len(self.class_names),
            "task_class_name": task_class_name,
        }
        return {
            build_metric_key(metric="p_best", **base_kwargs): self.threshold_grid[best_idx],
            build_metric_key(metric="best_F1", **base_kwargs): f1[best_idx],
            build_metric_key(metric="best_precision", **base_kwargs): precision[best_idx],
            build_metric_key(metric="best_recall", **base_kwargs): recall[best_idx],
            build_metric_key(metric="best_tp", **base_kwargs): tp[best_idx],
            build_metric_key(metric="best_fp", **base_kwargs): fp[best_idx],
            build_metric_key(metric="best_fn", **base_kwargs): fn[best_idx],
            build_metric_key(metric="num_gt", **base_kwargs): num_gt,
            build_metric_key(metric="numC_p_best_cutoff", **base_kwargs): tp[best_idx] + fp[best_idx],
        }, ()
