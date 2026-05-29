from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

import torch
from torch import nn

from src.wrappers.voxel_point_stage1_logging import build_metric_key


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
        将 batch 中的 source folder 名编码为固定 index。

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


class CpcValidationDiagnostics(nn.Module):
    """
    维护 Stage1 CPC validation 固定形状统计 buffer。

    输入参数:
        - config: CpcDiagnosticsConfig, diagnostics 行为配置
        - class_names: Sequence[str], (C,), task class 名
        - candidate_class_ids: Sequence[int], (K,), 候选前景 task class id
        - source_folders: SourceFolderRegistry, source folder 注册表
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
        self.register_buffer("threshold_grid", torch.arange(num_bins, dtype=torch.float32) / float(num_bins), persistent=False)
        self.register_buffer("uncapped_best_pos_hist", torch.zeros((num_candidate_classes, num_bins), dtype=torch.long), persistent=False)
        self.register_buffer("uncapped_best_neg_hist", torch.zeros((num_candidate_classes, num_bins), dtype=torch.long), persistent=False)
        self.register_buffer("uncapped_best_pos_hist_by_source", torch.zeros((num_source_folders, num_candidate_classes, num_bins), dtype=torch.long), persistent=False)
        self.register_buffer("uncapped_best_neg_hist_by_source", torch.zeros((num_source_folders, num_candidate_classes, num_bins), dtype=torch.long), persistent=False)
        self.register_buffer("capped_tp", torch.zeros((num_candidate_classes,), dtype=torch.long), persistent=False)
        self.register_buffer("capped_fp", torch.zeros((num_candidate_classes,), dtype=torch.long), persistent=False)
        self.register_buffer("capped_fn", torch.zeros((num_candidate_classes,), dtype=torch.long), persistent=False)
        self.register_buffer("capped_num_C", torch.zeros((), dtype=torch.long), persistent=False)
        self.register_buffer("capped_num_P", torch.zeros((), dtype=torch.long), persistent=False)
        self.register_buffer("capped_box_count", torch.zeros((), dtype=torch.long), persistent=False)

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

    def update_capped(self, **_: object) -> None:
        """预留 capped diagnostics 更新接口。"""

    def update_unrefined(self, **_: object) -> None:
        """预留 unrefined diagnostics 更新接口。"""

    def update_refined(self, **_: object) -> None:
        """预留 refined diagnostics 更新接口。"""

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
        return CpcDiagnosticsPayload(scalars=scalars, curves={}, warnings=tuple(warnings), local_tables={})

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
