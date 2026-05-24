"""
Stage1 体素-点云联合模型的清理后主流程。

对齐契约（修改时必须全量同步）:
    - 本段、CLAUDE/plans/implement/tri_ligand_sparse_refine/00-master.md、src/model/pseudo_atoms.py、src/model/stage1_atom_head.py 和 tests/model/test_stage1_model.py 必须同步更新。
    - forward 输入 batch 在 _run_embed_head_once 之前必须是 real-only; 伪原子不得进入 embed head。
    - P anchors 只允许在最后一次 recycle 的 _prepare_pseudo_batch 后进入 point backbone; 01 阶段 _prepare_pseudo_batch 返回 real-only batch、None layout、空 pseudo_outputs。
    - mixed layout 若存在, 必须来自 pseudo_atoms.inject_pseudo_atoms, 每个 BOX 内顺序固定为 `[real_i..., pseudo_i...]`, 先真实原子, 然后再是伪原子。

训练时 sparse candidate voxel set C 的契约:
    - C 只在最后一轮 recycle 的 _prepare_pseudo_batch 中生成; 前几轮 recycle 不注入候选体素, 只滚动 voxel/point recycle state。
    - 总开关是 cfg.model.backbone.candidate_set_cfg:
        - null: 关闭 C 生成, 对应 configs/model/sparse_refine/candidate_set/none.yaml。
        - 非 null: 由 VolumePointStage1Model.__init__ 实例化 SparseCandidateSetBuilder, 对应 configs/model/sparse_refine/candidate_set/tri.yaml 或 binary.yaml。
    - 候选类别与每类超参完全由 cfg.model.backbone.candidate_set_cfg 控制。
    - warmup fixed topk 是否生效, 不由 candidate_set_cfg 单独决定, 而是由 wrapper 同步的 runtime 状态决定:
        - src/wrappers/voxel_point_stage1.py::configure_optimizers() 只有在 scheduler.name 为 warmup_plateau 或 warmup_only 时, 才会解析出 candidate_warmup_steps。
        - _sync_sparse_candidate_runtime_to_backbone() 会把 global_step、candidate_warmup_steps、allow_warmup_fixed_topk 同步到本模型。
        - fit/sanity/tuning lifecycle 可允许 scheduler warmup fixed topk；standalone validate/test/predict 不允许。
        - 当前 forward 满足 global_step < candidate_warmup_steps 且 allow_warmup_fixed_topk=True 时, _should_use_candidate_fixed_topk() 返回 True；warmup 外 threshold cache 缺失必须 fail-fast，不做 bootstrap 回退。
    - warmup fixed topk 的真实行为:
        - 对每个 BOX、每个 candidate_class, 先在 voxel_valid_mask 内收集有效体素概率 prob_valid。
        - target_count = min(warmup_topc_per_class[class], max_candidate_voxels_per_class[class], N_valid)。
        - 直接对 prob_valid 做 topk, 取该 BOX/类别概率最高的 target_count 个体素进入 C。
    - adaptive_threshold 的训练主流程分成两个阶段:
        - 验证阶段统计全局 best-F1 阈值: wrapper._update_voxel_ligand_best_f1_stats() 在 voxel_valid_mask 内累加各候选类别的正负样本 histogram。硬标签来自 ligand_dist_map 与 voxel_ligand_loss.hard_label_threshold, 多分类时按 one-vs-rest 统计每个 candidate_class。
        - validation end 刷新阈值缓存: wrapper._compute_log_update_voxel_ligand_best_f1_thresholds() 在 histogram 网格上枚举阈值, 取 F1 最大的 bin 作为 p_best_by_class[class]。同时统计该阈值以上的总体体素数 n_best_total=TP+FP, 再计算 n_sampling_total=ceil(n_best_total * adaptive_expand_factor[class]) 作为扩张后的全局目标规模, 并记录对应 sampling 阈值 p_sampling_by_class[class]。
    - adaptive_threshold 在训练 forward 里生成 C 时, 实际采用的是 per-box best-F1 扩张 topk, 不是直接按全局 sampling 阈值截断:
        - 对当前 BOX/类别先计算 n_best_box = count(prob_valid > p_best_by_class[class])————注意上一段 p_best_by_class 是验证时缓存的, 但是这里和下一个 p_best_by_class 是本box临时计算的结果。
        - target_before_cap = ceil(n_best_box * adaptive_expand_factor[class])。
        - target_count = min(target_before_cap, max_candidate_voxels_per_class[class], N_valid)。
        - 再对该 BOX/类别的 prob_valid 做 topk, 取概率最高的 target_count 个体素进入 C。
        - 这意味着 p_best_by_class 决定“每个 BOX 估计应有多少个 best-F1 体素”, adaptive_expand_factor 决定在这个数量上扩张多少倍, 而真正入选的是该 BOX 内 topk 概率最大的体素。
    - recorded_threshold 是 builder 支持的另一种模式:
        - 它直接使用 wrapper 缓存的 p_sampling_by_class 做按阈值筛选, 仅在超过 max_candidate_voxels_per_class 时再回退为局部 topk 截断。
        - 当前 tri/binary 配置默认不用这一路径, 但 selection_mode 切到 recorded_threshold 后会改为该行为。

forward 输入的 batch 关键字段契约:
    - voxel_grid: torch.Tensor, (B, C_in, D, H, W), floating, voxel backbone 输入密度/特征体。
    - box_shape_zyx: torch.Tensor, (B, 3), int64/long, 每个 BOX 的体素尺寸, 轴顺序 (z, y, x)。
    - voxel_size_world: torch.Tensor, (B, 3), floating, 每个 voxel 的世界坐标尺寸, 轴顺序 (x, y, z)。
    - atom_feat: torch.Tensor, (N_real, F_atom) 或 mixed 路径下 (N_all, F_atom), floating, 点分支输入原子/P anchor 特征。
    - atom_coord_centered_world: torch.Tensor, (N_real, 3) 或 (N_all, 3), floating, 以 BOX 中心为原点的世界坐标, 轴顺序 (x, y, z)。
    - atom_coord_local_voxel: torch.Tensor, (N_real, 3) 或 (N_all, 3), floating, corner 语义连续局部体素坐标, 轴顺序 (x, y, z)。
    - atom_coord_world: torch.Tensor, (N_real, 3) 或 (N_all, 3), floating, 绝对世界坐标, 轴顺序 (x, y, z)。
    - atom_batch_index: torch.Tensor, (N_real,) 或 (N_all,), int64/long, 每个点所属 BOX 索引。
    - atom_offsets: torch.Tensor, (B,), int64/long, 每个 BOX 在展平点序列中的结束偏移。
    - atom_counts: torch.Tensor, (B,), int64/long, 每个 BOX 的点数; wrapper-facing 输出必须恢复为 real-only counts。
    - atom_label: torch.Tensor, (N_real,) 或 (N_all,), int64/long, real atom 监督标签; P anchor 槽位只允许作为占位 0。
    - atom_valid_mask: torch.Tensor, (N_real,) 或 (N_all,), bool, real atom 监督掩码; P anchor 槽位必须为 False。
    - atom_is_in_core_box: torch.Tensor, (N_real,) 或 (N_all,), bool, 点是否在 core box 内。
    - atom_global_indices: torch.Tensor, (N_real,) 或 (N_all,), int64/long, 真实原子全局索引; P anchor 槽位为 -1。
    - real_mask: torch.Tensor, (N_all,), bool, mixed-only 字段, True 表示 real atom。
    - pseudo_mask: torch.Tensor, (N_all,), bool, mixed-only 字段, True 表示 P anchor。

forward 输出契约:
    - atom_logits: torch.Tensor | None, (N_real, atom_logit_dim), floating, wrapper-facing real-only atom logits。
    - atom_target: torch.Tensor | None, (N_real,), long, wrapper-facing real-only atom 标签。
    - atom_valid_mask: torch.Tensor | None, (N_real,), bool, wrapper-facing real-only 监督掩码。
    - atom_counts: torch.Tensor | None, (B,), long, wrapper-facing real-only counts。
    - atom_tokens: torch.Tensor | None, (N_all, C_token), floating, atom head token projection 前输入; mixed 路径保留全点。
    - atom_hidden: torch.Tensor | None, (N_all, C_hidden), floating, atom head shared attention 输出; mixed 路径保留全点。
    - pseudo_feature: torch.Tensor | None, (N_pseudo, C_pseudo), floating, P anchor refined feature; 01 阶段或 real-only 路径为 None。
"""
from __future__ import annotations

from typing import Any, Sequence

import torch
import torch.nn.functional as F
from hydra.utils import instantiate
from torch import nn

from src.model.stage1_atom_head import Stage1AtomHead
from src.model.stage1_embed_head import scatter_to_voxel_grid
from src.model.typed_point import (
    TypedPointConfig,
    apply_type_aware_tensor_module,
    normalize_typed_point_cfg,
    validate_pseudo_mask,
)
from src.model.pseudo_atoms import (
    PseudoAtomLayout,
    extract_pseudo_tensor_from_mixed,
    extract_real_point_output,
    extract_real_tensor_from_mixed,
    inject_pseudo_atoms,
    interleave_real_and_pseudo_tensor,
)
from src.model.sparse_refine.anchor_sampler import build_anchor_coordinates

_PTV3_IMPORT_ERROR: Exception | None = None
try:
    from src.model.PTV3bakcbone.model import resolve_act_layer
except Exception as exc:  # pragma: no cover - 依赖当前本地环境
    resolve_act_layer = None
    _PTV3_IMPORT_ERROR = exc


class VolumePointStage1Model(nn.Module):
    def __init__(
        self,
        voxel_backbone: nn.Module | Any,
        point_backbone: nn.Module | Any,
        point_fusion_map: dict[str, str] | None,
        point_fusion_modes: Sequence[str],
        sampler_modes: Sequence[str],
        fusion_mlp_ratio: float,
        fusion_proj_drop: float,
        atom_head_hidden_dim: int,
        atom_head_num_heads: int,
        atom_head_patch_size: int,
        atom_head_num_layers: int,
        atom_head_serialization_orders: Sequence[str],
        atom_head_shuffle_orders: bool,
        atom_head_qkv_bias: bool,
        atom_head_qk_scale: float | None,
        atom_head_attn_drop: float,
        atom_head_proj_drop: float,
        atom_head_enable_rpe: bool,
        atom_head_enable_flash: bool,
        atom_head_upcast_attention: bool,
        atom_head_upcast_softmax: bool,
        atom_logit_dim: int,
        enable_recycling: bool,
        max_recycles: int,
        randomize_recycles: bool,
        detach_recycle_states: bool,
        act_layer_name: str,
        ffn_type: str,
        atom_head_ffn_type: str,
        atom_head_mlp_ratio: int,
        atom_head_cpe_impl: str,
        atom_head_cpe_kernel_size: int,
        atom_head_cpe_receptive_field: float,
        atom_head_pointconv_max_neighbors: int,
        atom_head_drop_path: float,
        atom_head_pre_norm: bool,
        atom_head_append_coord_mask: bool,
        enable_atom_head: bool = True,
        embed_head: nn.Module | Any | None = None,
        pseudo_atom_cfg: dict | None = None,
        prior_prob: float | None = None,
        prior_probs: Sequence[float] | None = None,
        online_pdb_feature: bool = False,
        online_pdb_feature_reduce: str = "sum",
        online_pdb_feature_dim: int = 49,
        atom_head_pseudo_feature_dim: int | None = None,
        typed_point_cfg: dict[str, Any] | TypedPointConfig | None = None,
        candidate_set_cfg: dict[str, Any] | nn.Module | None = None,
        anchor_sampler_cfg: dict[str, Any] | nn.Module | None = None,
        density_cube_cfg: dict[str, Any] | nn.Module | None = None,
        anchor_to_candidate_cfg: dict[str, Any] | nn.Module | None = None,
        sparse_refine_head_cfg: dict[str, Any] | nn.Module | None = None,
        anchor_class_conditioning_cfg: dict[str, Any] | None = None,
    ) -> None:
        """
        Stage1 体素-点云联合模型, voxel backbone 每轮 recycle, P anchors 只在最后一轮注入。

        输入参数:
            - voxel_backbone: nn.Module | Any, 体素分支模块或 Hydra 配置
            - point_backbone: nn.Module | Any, 点分支模块或 Hydra 配置
            - enable_atom_head: bool, 是否构造 Stage1AtomHead
            - embed_head: nn.Module | Any | None, embed head 模块或 Hydra 配置
            - pseudo_atom_cfg: dict | None, legacy 字段; 新流程只允许 None
            - candidate_set_cfg: dict[str, Any] | nn.Module | None, sparse candidate set builder 配置; None 表示关闭 C 生成
            - anchor_sampler_cfg: dict[str, Any] | nn.Module | None, sparse P anchor sampler 配置; None 表示只生成 C
            - density_cube_cfg: dict[str, Any] | nn.Module | None, density cube encoder 配置; anchor sampler 启用时必填
            - anchor_to_candidate_cfg: dict[str, Any] | nn.Module | None, P -> C 邻居搜索配置; None 表示关闭 refine
            - sparse_refine_head_cfg: dict[str, Any] | nn.Module | None, C refined logits head 配置; None 表示关闭 refine
            - anchor_class_conditioning_cfg: dict[str, Any] | None, P 初始特征类别条件化配置; None 表示关闭

            - prior_prob: float | None, 单通道 sigmoid 正类先验概率
            - prior_probs: Sequence[float] | None, 多通道 softmax 类别先验概率
            - online_pdb_feature: bool, embed head 无 voxel 输出时是否在线 scatter raw atom_feat
            - online_pdb_feature_reduce: str, 在线 scatter 聚合方式
            - online_pdb_feature_dim: int, 在线 scatter 的 raw atom 特征通道数

            - point_fusion_map: dict[str, str] | None, point 变量名到 voxel 特征名的融合映射
            - point_fusion_modes: Sequence[str], 每个 point 变量的融合模式
            - sampler_modes: Sequence[str], 每个 point 变量的 voxel 采样模式
            - fusion_mlp_ratio: float, concat_linear 融合 MLP 隐藏层倍率
            - fusion_proj_drop: float, concat_linear 融合 MLP dropout 概率

            - atom_head_hidden_dim: int, atom head 隐藏通道数
            - atom_head_num_heads: int, atom head attention 头数
            - atom_head_patch_size: int, atom head SerializedAttention patch 点数
            - atom_head_num_layers: int, atom head Block 层数
            - atom_head_serialization_orders: Sequence[str], atom head 序列化顺序列表
            - atom_head_shuffle_orders: bool, atom head forward 时是否打乱序列化顺序
            - atom_head_qkv_bias: bool, atom head QKV 是否使用 bias
            - atom_head_qk_scale: float | None, atom head QK 缩放因子
            - atom_head_attn_drop: float, atom head attention dropout 概率
            - atom_head_proj_drop: float, atom head 输出投影 dropout 概率
            - atom_head_enable_rpe: bool, atom head 是否启用相对位置编码
            - atom_head_enable_flash: bool, atom head 是否启用 flash attention
            - atom_head_upcast_attention: bool, atom head attention 矩阵乘法前是否上转精度
            - atom_head_upcast_softmax: bool, atom head softmax 前是否上转精度
            - atom_logit_dim: int, real atom logits 输出通道数
            - atom_head_pseudo_feature_dim: int | None, P anchor feature 输出通道数; None 表示等于 atom_head_hidden_dim
            - atom_head_ffn_type: str, atom head Block FFN 类型
            - atom_head_mlp_ratio: int, atom head FFN 隐藏层膨胀倍率
            - atom_head_cpe_impl: str, atom head CPE 实现方式, 取值 "none" / "pointconv"
            - atom_head_cpe_kernel_size: int, legacy 字段; pointconv CPE 不消费该值
            - atom_head_cpe_receptive_field: float, atom head pointconv CPE 感受野半径
            - atom_head_pointconv_max_neighbors: int, atom head pointconv CPE 每点最大邻居数
            - atom_head_drop_path: float, atom head Block 随机深度概率
            - atom_head_pre_norm: bool, atom head Block 是否使用 pre-norm
            - atom_head_append_coord_mask: bool, 是否把 centered-world 坐标与 atom_valid_mask 拼入 atom token

            - enable_recycling: bool, 是否启用 recycle
            - max_recycles: int, 最大 recycle 轮数
            - randomize_recycles: bool, 训练态是否随机采样 recycle 轮数
            - detach_recycle_states: bool, recycle 状态跨轮传递时是否 detach
            - act_layer_name: str, 激活函数名称
            - ffn_type: str, point backbone Block FFN 类型

        前向输出:
            - outputs: dict[str, Any], 包含 voxel/point 输出、real-only atom supervised 字段与可选 pseudo_feature
        """
        super().__init__()
        if pseudo_atom_cfg is not None:
            raise ValueError("旧 pseudo_atom_cfg 已删除；P anchors 将由 sparse refine anchor pipeline 提供。")
        if resolve_act_layer is None:
            raise ImportError("VolumePointStage1Model 需要 PTV3 resolve_act_layer。") from _PTV3_IMPORT_ERROR

        # TypedPointConfig, Stage1 全局 typed point 配置
        self.typed_point_cfg = normalize_typed_point_cfg(typed_point_cfg)
        self.embed_head = embed_head if isinstance(embed_head, nn.Module) else instantiate(embed_head) if embed_head is not None else None
        self.online_pdb_feature = bool(online_pdb_feature)
        self.online_pdb_feature_reduce = str(online_pdb_feature_reduce)
        self.online_pdb_feature_dim = int(online_pdb_feature_dim)

        # nn.Module, 体素分支模块
        self.voxel_backbone = voxel_backbone if isinstance(voxel_backbone, nn.Module) else instantiate(voxel_backbone)
        # nn.Module, 点分支模块
        self.point_backbone = point_backbone if isinstance(point_backbone, nn.Module) else instantiate(point_backbone)




        # nn.Module | None, sparse candidate voxel set C 生成器
        self.candidate_set_builder = (
            candidate_set_cfg
            if (candidate_set_cfg is None or isinstance(candidate_set_cfg, nn.Module))
            else instantiate(candidate_set_cfg)
        )
        # nn.Module | None, sparse P anchor sampler
        self.anchor_sampler = (
            anchor_sampler_cfg
            if (anchor_sampler_cfg is None or isinstance(anchor_sampler_cfg, nn.Module))
            else instantiate(anchor_sampler_cfg)
        )
        # nn.Module | None, density cube pseudo feature encoder
        self.density_cube_encoder = (
            density_cube_cfg
            if (density_cube_cfg is None or isinstance(density_cube_cfg, nn.Module))
            else instantiate(density_cube_cfg)
        )
        # nn.Module | None, P -> C KNN 边搜索模块
        self.anchor_to_candidate = (
            anchor_to_candidate_cfg
            if (anchor_to_candidate_cfg is None or isinstance(anchor_to_candidate_cfg, nn.Module))
            else instantiate(anchor_to_candidate_cfg)
        )
        # dict[str, Any] | nn.Module | None, 在通道信息确定后构造的 sparse refine head 配置
        pending_sparse_refine_head_cfg = sparse_refine_head_cfg
        if self.anchor_sampler is not None and self.candidate_set_builder is None:
            raise ValueError("anchor_sampler 启用时必须同时启用 candidate_set_builder。")
        if self.anchor_sampler is not None and self.density_cube_encoder is None:
            raise ValueError("anchor_sampler 启用时必须同时配置 density_cube_encoder。")
        if self.anchor_sampler is None and self.density_cube_encoder is not None:
            raise ValueError("density_cube_encoder 只能在 anchor_sampler 启用时配置。")
        if (self.anchor_to_candidate is None) != (pending_sparse_refine_head_cfg is None):
            raise ValueError("anchor_to_candidate_cfg 与 sparse_refine_head_cfg 必须同时启用或同时关闭。")
        if self.anchor_to_candidate is not None and (self.candidate_set_builder is None or self.anchor_sampler is None):
            raise ValueError("P -> C refine 启用时必须同时启用 candidate_set_builder 与 anchor_sampler。")
        if self.anchor_sampler is not None:
            if hasattr(self.candidate_set_builder, "candidate_class_ids") and hasattr(self.anchor_sampler, "candidate_class_ids"):
                if tuple(int(x) for x in self.candidate_set_builder.candidate_class_ids) != tuple(
                    int(x) for x in self.anchor_sampler.candidate_class_ids
                ):
                    raise ValueError("anchor_sampler.candidate_class_ids 必须与 candidate_set_builder.candidate_class_ids 一致。")
            if not hasattr(self.density_cube_encoder, "out_dim"):
                raise AttributeError("density_cube_encoder 必须暴露 out_dim。")
        self.anchor_class_conditioning_mode = "none"
        self.anchor_class_conditioning_init_std = 0.02
        self.anchor_class_embedding: nn.Embedding | None = None
        self.register_buffer("_anchor_class_ids", torch.empty((0,), dtype=torch.long), persistent=False)
        if anchor_class_conditioning_cfg is not None:
            if self.anchor_sampler is None:
                raise ValueError("anchor_class_conditioning_cfg 启用时必须同时启用 anchor_sampler。")
            mode = str(anchor_class_conditioning_cfg["mode"])
            if mode != "add_embedding":
                raise ValueError("anchor_class_conditioning_cfg.mode 只支持 add_embedding。")
            self.anchor_class_conditioning_mode = mode
            if "init_std" in anchor_class_conditioning_cfg:
                self.anchor_class_conditioning_init_std = float(anchor_class_conditioning_cfg["init_std"])
            if self.anchor_class_conditioning_init_std <= 0.0:
                raise ValueError("anchor_class_conditioning_cfg.init_std 必须 > 0。")
            candidate_class_ids = tuple(int(class_id) for class_id in self.anchor_sampler.candidate_class_ids)
            self._anchor_class_ids = torch.as_tensor(candidate_class_ids, dtype=torch.long)
            self.anchor_class_embedding = nn.Embedding(len(candidate_class_ids), int(self.point_backbone.atom_feature_dim))
            nn.init.normal_(self.anchor_class_embedding.weight, mean=0.0, std=self.anchor_class_conditioning_init_std)
        # torch.Tensor, (K,) 或 (0,), wrapper 同步过来的 best-F1 阈值缓存
        self.register_buffer("_candidate_p_best_by_class", torch.empty((0,), dtype=torch.float32), persistent=False)
        # torch.Tensor | None, (K,), wrapper 同步过来的 sampling 阈值缓存
        self.register_buffer("_candidate_p_sampling_by_class", torch.empty((0,), dtype=torch.float32), persistent=False)
        self._has_candidate_p_best_by_class = False
        self._has_candidate_p_sampling_by_class = False
        self._candidate_warmup_steps = 0
        self._candidate_global_step = 0
        self._candidate_allow_warmup_fixed_topk = False




        self.point_fusion_items = tuple(
            (str(point_name), voxel_name_str)
            for point_name, voxel_name in (point_fusion_map or {}).items()
            if voxel_name is not None and (voxel_name_str := str(voxel_name).strip()) != ""
        )
        self.point_fusion_modes = tuple(str(mode_name).lower() for mode_name in point_fusion_modes)
        self.sampler_modes = tuple(str(mode_name).lower() for mode_name in sampler_modes)
        self.enable_recycling = bool(enable_recycling)
        self.max_recycles = int(max_recycles)
        self.randomize_recycles = bool(randomize_recycles)
        self.detach_recycle_states = bool(detach_recycle_states)
        self.ffn_type = str(ffn_type)
        if not (len(self.point_fusion_items) == len(self.point_fusion_modes) == len(self.sampler_modes)):
            raise ValueError("point_fusion_map、point_fusion_modes 与 sampler_modes 的长度必须一致。")
        unsupported_sampler_modes = [mode_name for mode_name in self.sampler_modes if mode_name not in {"trilinear", "nearest"}]
        if unsupported_sampler_modes:
            raise ValueError(f"Unsupported sampler_modes={unsupported_sampler_modes}, supported=['nearest', 'trilinear']")
        if self.max_recycles <= 0:
            raise ValueError("max_recycles must be > 0")

        self.point_fusion_map = {point_name: voxel_name for point_name, voxel_name in self.point_fusion_items}
        self.sampler_mode_by_point_name = {
            point_name: mode_name
            for (point_name, _), mode_name in zip(self.point_fusion_items, self.sampler_modes)
        }
        self.fusion_mode_by_point_name = {
            point_name: mode_name
            for (point_name, _), mode_name in zip(self.point_fusion_items, self.point_fusion_modes)
        }
        self.voxel_feature_names_to_return = tuple(
            dict.fromkeys(
                tuple(str(feature_name) for feature_name in self.voxel_backbone.return_feature_keys)
                + tuple(voxel_name for _, voxel_name in self.point_fusion_items)
                + (("voxel_final",) if self.anchor_to_candidate is not None else ())
            )
        )
        self.point_feature_names_to_return = tuple(
            dict.fromkeys([point_name for point_name, _ in self.point_fusion_items] + ["point_feat"])
        )

        if self.density_cube_encoder is not None and int(self.density_cube_encoder.out_dim) != int(self.point_backbone.atom_feature_dim):
            raise ValueError("density_cube_encoder.out_dim 必须等于 point_backbone.atom_feature_dim。")

        available_voxel_feature_names = tuple(self.voxel_backbone.feature_channels_by_name.keys())
        available_point_feature_names = tuple(self.point_backbone.feature_channels_by_name.keys())
        for point_name, voxel_name in self.point_fusion_items:
            if point_name not in available_point_feature_names:
                raise KeyError(f"Unknown point fusion name={point_name}, available={available_point_feature_names}")
            if voxel_name not in available_voxel_feature_names:
                raise KeyError(f"Unknown voxel fusion name={voxel_name}, available={available_voxel_feature_names}")

        # type[nn.Module], 统一激活函数类
        act_cls = resolve_act_layer(str(act_layer_name))
        # nn.ModuleDict, point 变量名 -> voxel-to-point concat_linear 融合 MLP
        self.point_fusion_modules = nn.ModuleDict()
        for point_name, voxel_name in self.point_fusion_items:
            fusion_mode = self.fusion_mode_by_point_name[point_name]
            if fusion_mode != "concat_linear":
                raise ValueError(f"Unsupported point fusion mode={fusion_mode}")
            point_channels = int(self.point_backbone.feature_channels_by_name[point_name])
            voxel_channels = int(self.voxel_backbone.feature_channels_by_name[voxel_name])
            fusion_input_dim = point_channels + voxel_channels
            fusion_hidden_dim = max(point_channels, int(round(float(fusion_input_dim) * float(fusion_mlp_ratio))))
            if self.typed_point_cfg.use_separate_fusion:
                # nn.ModuleDict, real/pseudo 两套 voxel-to-point fusion MLP
                self.point_fusion_modules[point_name] = nn.ModuleDict(
                    {
                        "real": self._build_point_fusion_module(
                            fusion_input_dim,
                            fusion_hidden_dim,
                            point_channels,
                            act_cls,
                            float(fusion_proj_drop),
                        ),
                        "pseudo": self._build_point_fusion_module(
                            fusion_input_dim,
                            fusion_hidden_dim,
                            point_channels,
                            act_cls,
                            float(fusion_proj_drop),
                        ),
                    }
                )
            else:
                # nn.Sequential, shared voxel-to-point fusion MLP
                self.point_fusion_modules[point_name] = self._build_point_fusion_module(
                    fusion_input_dim,
                    fusion_hidden_dim,
                    point_channels,
                    act_cls,
                    float(fusion_proj_drop),
                )

        self.enable_atom_head = bool(enable_atom_head)
        if self.enable_atom_head:
            self.atom_head_append_coord_mask = bool(atom_head_append_coord_mask)
            self.atom_head = Stage1AtomHead(
                point_channels=int(self.point_backbone.out_channels),
                hidden_dim=int(atom_head_hidden_dim),
                num_heads=int(atom_head_num_heads),
                patch_size=int(atom_head_patch_size),
                num_layers=int(atom_head_num_layers),
                serialization_orders=atom_head_serialization_orders,
                shuffle_orders=bool(atom_head_shuffle_orders),
                qkv_bias=bool(atom_head_qkv_bias),
                qk_scale=atom_head_qk_scale,
                attn_drop=float(atom_head_attn_drop),
                proj_drop=float(atom_head_proj_drop),
                enable_rpe=bool(atom_head_enable_rpe),
                enable_flash=bool(atom_head_enable_flash),
                upcast_attention=bool(atom_head_upcast_attention),
                upcast_softmax=bool(atom_head_upcast_softmax),
                atom_logit_dim=int(atom_logit_dim),
                pseudo_feature_dim=atom_head_pseudo_feature_dim,
                atom_head_ffn_type=str(atom_head_ffn_type),
                mlp_ratio=int(atom_head_mlp_ratio),
                act_layer=act_cls,
                cpe_impl=str(atom_head_cpe_impl),
                cpe_kernel_size=int(atom_head_cpe_kernel_size),
                cpe_receptive_field=float(atom_head_cpe_receptive_field),
                pointconv_block_max_neighbors=int(atom_head_pointconv_max_neighbors),
                drop_path=float(atom_head_drop_path),
                pre_norm=bool(atom_head_pre_norm),
                append_coord_mask=bool(atom_head_append_coord_mask),
                prior_prob=prior_prob,
                prior_probs=prior_probs,
                typed_point_cfg=self.typed_point_cfg,
            )
        else:
            self.atom_head_append_coord_mask = False
            self.atom_head = None


        self.sparse_refine_head: nn.Module | None = None
        if pending_sparse_refine_head_cfg is not None:
            if self.atom_head is None:
                raise ValueError("sparse_refine_head_cfg 启用时必须启用 atom head。")
            if "voxel_final" not in self.voxel_backbone.feature_channels_by_name:
                raise KeyError("sparse refine 固定要求 voxel backbone 导出 voxel_final。")
            if not hasattr(self.voxel_backbone, "voxel_ligand_logit_dim"):
                raise AttributeError("sparse refine 要求 voxel_backbone 暴露 voxel_ligand_logit_dim。")
            if not hasattr(self.candidate_set_builder, "candidate_class_ids"):
                raise AttributeError("sparse refine 要求 candidate_set_builder 暴露 candidate_class_ids。")
            injected_head_kwargs = {
                "logit_dim": int(self.voxel_backbone.voxel_ligand_logit_dim),
                "C_voxel_backbone_dim": int(self.voxel_backbone.feature_channels_by_name["voxel_final"]),
                "P_point_backbone_dim": int(self.point_backbone.feature_channels_by_name["point_feat"]),
                "P_atom_head_dim": int(self.atom_head.pseudo_feature_dim),
                "P_voxel_backbone_dim": int(self.voxel_backbone.feature_channels_by_name["voxel_final"]),
                "candidate_class_ids": tuple(int(value) for value in self.candidate_set_builder.candidate_class_ids),
            }
            self.sparse_refine_head = (
                pending_sparse_refine_head_cfg
                if isinstance(pending_sparse_refine_head_cfg, nn.Module)
                else instantiate(pending_sparse_refine_head_cfg, **injected_head_kwargs)
            )

    def _build_point_fusion_module(
        self,
        fusion_input_dim: int,
        fusion_hidden_dim: int,
        point_channels: int,
        act_cls: type[nn.Module],
        fusion_proj_drop: float,
    ) -> nn.Module:
        """
        构造 voxel-to-point concat_linear 融合 MLP。

        输入参数:
            - fusion_input_dim: int, 点特征与体素采样特征拼接后的通道数
            - fusion_hidden_dim: int, 融合 MLP 隐藏通道数
            - point_channels: int, 输出点特征通道数
            - act_cls: type[nn.Module], 激活函数类
            - fusion_proj_drop: float, dropout 概率

        输出:
            - module: nn.Module, (N_all, fusion_input_dim) -> (N_all, point_channels) 的融合模块
        """
        return nn.Sequential(
            nn.Linear(fusion_input_dim, fusion_hidden_dim),
            nn.LayerNorm(fusion_hidden_dim),
            act_cls(),
            nn.Dropout(float(fusion_proj_drop)),
            nn.Linear(fusion_hidden_dim, point_channels),
        )


    # ---------------------------------------- 纯粹工具函数 ----------------------------------------
    @staticmethod
    def _voxel_xyz_to_grid_sample_xyz(
        point_coord_local_voxel: torch.Tensor,
        box_shape_zyx: torch.Tensor,
    ) -> torch.Tensor:
        """
        将 BOX 内连续 voxel corner 坐标转换为 grid_sample [-1, 1]归一化坐标。

        输入参数:
            - point_coord_local_voxel: torch.Tensor, (N, 3), 连续体素 corner 坐标, 顺序 x/y/z
            - box_shape_zyx: torch.Tensor, (3,), BOX 体素尺寸, 顺序 z/y/x

        输出:
            - grid_xyz: torch.Tensor, (N, 3), grid_sample 使用的归一化坐标, 顺序 x/y/z
        """
        # torch.Tensor, (3,), BOX 尺寸的 x/y/z 顺序
        box_shape_xyz = box_shape_zyx.to(device=point_coord_local_voxel.device, dtype=point_coord_local_voxel.dtype)[[2, 1, 0]]
        # torch.Tensor, (N, 3), corner 坐标转 center-index 坐标
        point_coord_center_index = point_coord_local_voxel - 0.5
        # torch.Tensor, (3,), align_corners=True 归一化分母
        denom_xyz = torch.clamp(box_shape_xyz - 1.0, min=1.0)
        # torch.Tensor, (N, 3), [-1, 1] 归一化采样坐标
        grid_xyz = (2.0 * point_coord_center_index / denom_xyz) - 1.0
        return grid_xyz

    @staticmethod
    def _centered_world_xyz_to_local_voxel_xyz(
        point_coord_centered_world: torch.Tensor,
        voxel_size_world: torch.Tensor,
        box_shape_zyx: torch.Tensor,
    ) -> torch.Tensor:
        """
        将 centered-world 坐标转换为 BOX 内连续 voxel corner 坐标。

        输入参数:
            - point_coord_centered_world: torch.Tensor, (N, 3), centered-world 坐标: 以BOX中心为原点, 顺序 x/y/z
            - voxel_size_world: torch.Tensor, (3,), voxel 世界尺寸, 顺序 x/y/z
            - box_shape_zyx: torch.Tensor, (3,), BOX 体素尺寸, 顺序 z/y/x

        输出:
            - point_coord_local_voxel: torch.Tensor, (N, 3), 连续体素 corner 坐标, 顺序 x/y/z
        """
        # torch.Tensor, (3,), BOX 尺寸的 x/y/z 顺序
        box_shape_xyz = box_shape_zyx.to(device=point_coord_centered_world.device, dtype=point_coord_centered_world.dtype)[[2, 1, 0]]
        return point_coord_centered_world / voxel_size_world.to(
            device=point_coord_centered_world.device,
            dtype=point_coord_centered_world.dtype,
        ) + (0.5 * box_shape_xyz)

    @staticmethod
    def _gather_voxel_feature_at_zyx(
        voxel_feat: torch.Tensor,
        voxel_zyx: torch.Tensor,
        point_batch_index: torch.Tensor,
    ) -> torch.Tensor:
        """
        按离散 voxel center 坐标直接读取体素特征。

        输入参数:
            - voxel_feat: torch.Tensor, (B, C, D, H, W), 体素特征图
            - voxel_zyx: torch.Tensor, (N, 3), 点来源 voxel 坐标, 轴顺序 z/y/x
            - point_batch_index: torch.Tensor, (N,), 每个点所属 BOX 索引

        输出:
            - point_feat: torch.Tensor, (N, C), 点位置对应的体素特征
        """
        return voxel_feat[
            point_batch_index,
            :,
            voxel_zyx[:, 0],
            voxel_zyx[:, 1],
            voxel_zyx[:, 2],
        ].contiguous()

    def _condition_anchor_pseudo_feat(
        self,
        pseudo_feat: torch.Tensor,
        anchor_class: torch.Tensor,
    ) -> torch.Tensor:
        """
        按 P anchor 来源类别对初始 pseudo feature 做条件化。

        输入参数:
            - pseudo_feat: torch.Tensor, (sumP, F_atom), density cube 输出的 P 初始特征
            - anchor_class: torch.Tensor, (sumP,), P 来源候选类别 ID

        输出:
            - conditioned_feat: torch.Tensor, (sumP, F_atom), 加入类别 embedding 后的 P 初始特征
        """
        if self.anchor_class_conditioning_mode == "none":
            return pseudo_feat
        if self.anchor_class_embedding is None:
            raise RuntimeError("anchor_class_conditioning_mode 启用但 anchor_class_embedding 未构造。")
        # torch.Tensor, (sumP, K), P 来源类别与配置类别的匹配矩阵
        class_match = anchor_class[:, None] == self._anchor_class_ids.to(device=anchor_class.device, dtype=anchor_class.dtype)[None, :]
        # torch.Tensor, (sumP,), True 表示该 P 来源类别存在于 candidate_class_ids 中
        known_mask = class_match.any(dim=1)
        if not bool(known_mask.all()):
            unknown_classes = torch.unique(anchor_class[~known_mask]).detach().cpu().tolist()
            raise RuntimeError(f"anchor_class 包含未配置类别: {unknown_classes}。")
        # torch.Tensor, (sumP,), P 来源类别在 candidate_class_ids 中的局部下标
        local_index = class_match.to(dtype=torch.long).argmax(dim=1)
        return pseudo_feat + self.anchor_class_embedding(local_index.to(device=pseudo_feat.device))

    @staticmethod
    def _counts_from_offsets(atom_offsets: torch.Tensor) -> torch.Tensor:
        """
        从累计 offset 恢复每个 BOX 的点数。

        输入参数:
            - atom_offsets: torch.Tensor, (B,), long, PTV3 风格累计 offset

        输出:
            - atom_counts: torch.Tensor, (B,), long, 每个 BOX 的点数
        """
        if atom_offsets.numel() == 0:
            return atom_offsets.new_zeros((0,), dtype=torch.long)
        atom_counts = atom_offsets.clone()
        atom_counts[1:] = atom_counts[1:] - atom_counts[:-1]
        return atom_counts

    def set_input_channels(self, in_channels: int) -> None:
        """
        将体素输入通道数设置请求透传给 voxel backbone。

        输入参数:
            - in_channels: int, 数据集 voxel_grid 通道数, 不含 embed/online scatter 追加通道

        输出:
            - None, 原地调用 voxel_backbone 与 density_cube_encoder 的 set_input_channels
        """
        raw_in_channels = int(in_channels)
        actual_in_channels = raw_in_channels
        if self.embed_head is not None and self.embed_head.has_voxel_output:
            extra = int(self.embed_head.embed_voxel_out_channels)
            if self.embed_head.add_occupancy_channels:
                extra += 2
            actual_in_channels += extra
        elif self.online_pdb_feature:
            actual_in_channels += self.online_pdb_feature_dim
        if hasattr(self.voxel_backbone, "set_input_channels"):
            self.voxel_backbone.set_input_channels(actual_in_channels)
        if self.density_cube_encoder is not None and hasattr(self.density_cube_encoder, "set_input_channels"):
            self.density_cube_encoder.set_input_channels(raw_in_channels)




    # ---------------------------------------- 体素———>点 的特征融合逻辑 ----------------------------------------
    def _sample_voxel_feature_single_box(
        self,
        voxel_feat_one_box: torch.Tensor,
        point_coord_centered_world_one_box: torch.Tensor,
        voxel_size_world_one_box: torch.Tensor,
        box_shape_zyx_one_box: torch.Tensor,
        fusion_mode: str,
        sampler_mode: str,
    ) -> torch.Tensor:
        """
        对单个 BOX 的点采样一份体素特征。

        输入参数:
            - voxel_feat_one_box: torch.Tensor, (1, C, D, H, W) 或 (C, D, H, W), 单个 BOX 的体素特征图
            - point_coord_centered_world_one_box: torch.Tensor, (N_i, 3), 当前 BOX 点坐标
            - voxel_size_world_one_box: torch.Tensor, (3,), 当前 BOX voxel 世界尺寸
            - box_shape_zyx_one_box: torch.Tensor, (3,), 当前 BOX 体素尺寸, 顺序 z/y/x
            - fusion_mode: str, 当前融合模式, 取值 concat_linear
            - sampler_mode: str, voxel 采样模式, 取值 trilinear / nearest

        输出:
            - sampled_feat_one_box: torch.Tensor, (N_i, C), 采样后的点级体素特征
        """
        if voxel_feat_one_box.ndim == 4:
            voxel_feat_one_box = voxel_feat_one_box.unsqueeze(0)
        point_count = int(point_coord_centered_world_one_box.shape[0])
        if point_count == 0:
            return voxel_feat_one_box.new_empty((0, int(voxel_feat_one_box.shape[1])))
        if fusion_mode != "concat_linear":
            raise ValueError(f"Unsupported point fusion mode={fusion_mode}")
        grid_sample_mode = "bilinear" if sampler_mode == "trilinear" else "nearest"
        point_coord_local_voxel = self._centered_world_xyz_to_local_voxel_xyz(
            point_coord_centered_world=point_coord_centered_world_one_box,
            voxel_size_world=voxel_size_world_one_box,
            box_shape_zyx=box_shape_zyx_one_box,
        )
        # torch.Tensor, (N_i, 3), grid_sample 归一化坐标
        grid_xyz = self._voxel_xyz_to_grid_sample_xyz(point_coord_local_voxel, box_shape_zyx_one_box)
        # torch.Tensor, (1, N_i, 1, 1, 3), 5D grid_sample 采样网格
        grid = grid_xyz.view(1, point_count, 1, 1, 3)
        # torch.Tensor, (1, C, N_i, 1, 1), 单 BOX 采样结果
        sampled = F.grid_sample(
            input=voxel_feat_one_box,
            grid=grid,
            mode=grid_sample_mode,
            padding_mode="zeros",
            align_corners=True,
        )
        return sampled.squeeze(0).squeeze(-1).squeeze(-1).transpose(0, 1).contiguous()

    def _sample_voxel_feature_batch(
        self,
        voxel_feat: torch.Tensor,
        point_coord_centered_world: torch.Tensor,
        point_batch_index: torch.Tensor,
        voxel_size_world: torch.Tensor,
        box_shape_zyx: torch.Tensor,
        fusion_mode: str,
        sampler_mode: str,
    ) -> torch.Tensor:
        """
        对 batch 内所有点采样指定体素特征。

        输入参数:
            - voxel_feat: torch.Tensor, (B, C, D, H, W), 体素特征图
            - point_coord_centered_world: torch.Tensor, (N, 3), 当前点坐标
            - point_batch_index: torch.Tensor, (N,), 当前点所属 BOX 索引
            - voxel_size_world: torch.Tensor, (B, 3), 每个 BOX voxel 世界尺寸
            - box_shape_zyx: torch.Tensor, (B, 3), 每个 BOX 体素尺寸
            - fusion_mode: str, 当前融合模式, 取值 concat_linear
            - sampler_mode: str, voxel 采样模式, 取值 trilinear / nearest

        输出:
            - sampled_feat: torch.Tensor, (N, C), 采样后的点级体素特征
        """
        point_count = int(point_coord_centered_world.shape[0])
        if point_count == 0:
            return voxel_feat.new_empty((0, int(voxel_feat.shape[1])))
        if fusion_mode != "concat_linear":
            raise ValueError(f"Unsupported point fusion mode={fusion_mode}")
        batch_size = int(box_shape_zyx.shape[0])
        # torch.Tensor, (N, C), 保持点顺序的采样结果
        sampled_feat = voxel_feat.new_empty((point_count, int(voxel_feat.shape[1])))
        for box_idx in range(batch_size):
            # torch.Tensor, (N,), bool, 当前 BOX 点掩码
            point_mask = point_batch_index == box_idx
            sampled_feat[point_mask] = self._sample_voxel_feature_single_box(
                voxel_feat_one_box=voxel_feat[box_idx : box_idx + 1],
                point_coord_centered_world_one_box=point_coord_centered_world[point_mask],
                voxel_size_world_one_box=voxel_size_world[box_idx],
                box_shape_zyx_one_box=box_shape_zyx[box_idx],
                fusion_mode=fusion_mode,
                sampler_mode=sampler_mode,
            ).to(dtype=sampled_feat.dtype)
        return sampled_feat

    def _fuse_point_variable(
        self,
        feature_name: str,
        point_like: Any,
        voxel_output_dict: dict[str, Any],
        batch: dict[str, Any],
        *,
        require_pseudo_mask: bool = False,
    ) -> Any:
        """
        对点云分支中名为 feature_name 的变量执行 voxel-to-point 融合。

        输入参数:
            - feature_name: str, 当前 point 变量名
            - point_like: Any, 当前 Point 对象, 包含 feat/coord/batch
            - voxel_output_dict: dict[str, Any], 当前 recycle 的 voxel backbone 输出, 只用到 voxel_output_dict["voxel_features"][voxel_name], 而 voxel_name = self.point_fusion_map[feature_name]
            - batch: dict[str, Any], 与 point_like 同布局的 real 或 mixed batch

        输出:
            - point_like: Any, 更新 feat 后的 Point 对象
        """
        if feature_name not in self.point_fusion_map:
            return point_like
        voxel_name = self.point_fusion_map[feature_name]
        fusion_mode = self.fusion_mode_by_point_name[feature_name]
        sampler_mode = self.sampler_mode_by_point_name[feature_name]
        # torch.Tensor, (N, C_voxel), 当前点集采样得到的体素特征
        sampled_voxel_feat = self._sample_voxel_feature_batch(
            voxel_feat=voxel_output_dict["voxel_features"][voxel_name],
            point_coord_centered_world=point_like.coord,
            point_batch_index=point_like.batch,
            voxel_size_world=batch["voxel_size_world"],
            box_shape_zyx=batch["box_shape_zyx"],
            fusion_mode=fusion_mode,
            sampler_mode=sampler_mode,
        )
        # torch.Tensor, (N, C_point + C_voxel), 融合 MLP 输入特征
        fusion_input = torch.cat([point_like.feat, sampled_voxel_feat], dim=-1)
        # torch.Tensor | None, (N,), bool, 当前 point_like 分辨率的 P anchor 掩码
        pseudo_mask = point_like.get("pseudo_mask", None) if hasattr(point_like, "get") else None
        pseudo_mask = validate_pseudo_mask(
            pseudo_mask,
            int(fusion_input.shape[0]),
            name="VolumePointStage1Model._fuse_point_variable",
        )
        # nn.Module, 当前 point 变量对应的 fusion module
        fusion_module = self.point_fusion_modules[feature_name]
        if self.typed_point_cfg.use_separate_fusion:
            if require_pseudo_mask and pseudo_mask is None:
                raise RuntimeError("typed fusion 的 mixed point_like 必须携带 pseudo_mask。")
            point_like.feat = apply_type_aware_tensor_module(
                fusion_input,
                pseudo_mask,
                fusion_module["real"],
                fusion_module["pseudo"],
            )
        else:
            point_like.feat = fusion_module(fusion_input)
        return point_like




    # ---------------------------------------- 各组件正式 forward ----------------------------------------
    def _run_embed_head_once(self, batch: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any] | None]:
        """
        在 recycle 循环前运行 real-only embed head 并同步裁剪后的 atom 字段。

        输入参数:
            - batch: dict[str, Any], collate 后的 real-only batch

        输出:
            - batch: dict[str, Any], real-only canonical batch, 若 embed head 裁剪则字段已同步
            - embed_output: dict[str, Any] | None, embed head 输出; 未启用时为 None
        """
        if self.embed_head is None:
            return batch, None
        # dict[str, Any], embed head 输出, 包含裁剪后的 real atom 字段与可选 voxel grid
        embed_output = self.embed_head(
            atom_feat=batch["atom_feat"],
            atom_coord_centered_world=batch["atom_coord_centered_world"],
            atom_batch_index=batch["atom_batch_index"],
            atom_offsets=batch["atom_offsets"],
            atom_coord_local_voxel=batch["atom_coord_local_voxel"],
            box_shape_zyx=batch["box_shape_zyx"],
            voxel_size_world=batch["voxel_size_world"],
            atom_is_in_core_box=batch["atom_is_in_core_box"],
        )
        # torch.Tensor, (N_before,), bool, embed head 从原始 real-only 点到裁剪后点的保留掩码
        global_keep_mask = embed_output["global_keep_mask"]
        batch = {**batch}
        batch["atom_feat"] = embed_output["atom_feat"]
        batch["atom_coord_centered_world"] = embed_output["atom_coord_centered_world"]
        batch["atom_batch_index"] = embed_output["atom_batch_index"]
        batch["atom_offsets"] = embed_output["atom_offsets"]
        batch["atom_counts"] = self._counts_from_offsets(batch["atom_offsets"])
        batch["atom_coord_local_voxel"] = embed_output["atom_coord_local_voxel"]
        batch["atom_is_in_core_box"] = embed_output["atom_is_in_core_box"]
        for key in ("atom_label", "atom_valid_mask", "atom_coord_world", "atom_global_indices"):
            if key in batch and batch[key] is not None:
                batch[key] = batch[key][global_keep_mask]
        return batch, embed_output

    def _build_voxel_input(self, batch: dict[str, Any], embed_output: dict[str, Any] | None) -> torch.Tensor:
        """
        构造 voxel backbone 输入张量(不要在这里生成候选集合 C)。

        输入参数:
            - batch: dict[str, Any], real-only canonical batch
            - embed_output: dict[str, Any] | None, embed head 输出; 未启用时为 None

        输出:
            - voxel_input: torch.Tensor, (B, C_in, D, H, W), voxel backbone 输入体素张量
        """
        if embed_output is not None and embed_output.get("voxel_pdb_embed_grid") is not None:
            # torch.Tensor, (B, C_embed, D, H, W), embed head 体素输出
            fused_voxel_grid = embed_output["voxel_pdb_embed_grid"]
            return torch.cat([batch["voxel_grid"], fused_voxel_grid], dim=1)
        if self.online_pdb_feature:
            with torch.no_grad():
                # torch.Tensor, (B, online_pdb_feature_dim, D, H, W), raw atom_feat 在线 scatter 体素网格
                raw_pdb_grid = scatter_to_voxel_grid(
                    point_feat=batch["atom_feat"].detach(),
                    atom_coord_local_voxel=batch["atom_coord_local_voxel"],
                    point_batch=batch["atom_batch_index"],
                    box_shape_zyx=batch["box_shape_zyx"],
                    batch_size=int(batch["box_shape_zyx"].shape[0]),
                    reduce=self.online_pdb_feature_reduce,
                    add_occupancy_channels=False,
                )
            return torch.cat([batch["voxel_grid"], raw_pdb_grid], dim=1)
        return batch["voxel_grid"]

    def _run_voxel_backbone(
        self,
        voxel_input: torch.Tensor,
        voxel_recycle_in: torch.Tensor | None,
    ) -> dict[str, Any]:
        """
        执行一轮 voxel backbone。

        输入参数:
            - voxel_input: torch.Tensor, (B, C_in, D, H, W), voxel backbone 输入
            - voxel_recycle_in: torch.Tensor | None, 上一轮 voxel recycle 状态

        输出:
            - voxel_output_dict: dict[str, Any], 当前轮 voxel 输出字典
        """
        return self.voxel_backbone(
            voxel_grid=voxel_input,
            recycle_in=voxel_recycle_in,
            return_feature_keys=self.voxel_feature_names_to_return,
        )

    def _prepare_pseudo_batch(
        self,
        batch: dict[str, Any],
        voxel_output_dict: dict[str, Any],
    ) -> tuple[dict[str, Any], PseudoAtomLayout | None, dict[str, Any]]:
        """
        在最后一轮 voxel backbone 后准备 sparse candidate C 与 P anchor mixed batch。

        输入参数:
            - batch: dict[str, Any], 当前 real-only canonical batch
            - voxel_output_dict: dict[str, Any], 当前 recycle 的 _run_voxel_backbone() 输出; 03 读取 voxel_logits_ligand 生成 C

        输出:
            - point_batch: dict[str, Any], 03 阶段仍为 real-only batch
            - pseudo_layout: PseudoAtomLayout | None, 03 阶段为 None; 后续阶段描述 real/P mixed 布局
            - pseudo_outputs: dict[str, Any], 03 阶段透传 C 元数据
        """
        if self.candidate_set_builder is None:
            return batch, None, {}
        voxel_logits_ligand = voxel_output_dict.get("voxel_logits_ligand")
        if voxel_logits_ligand is None:
            raise RuntimeError("candidate_set_builder 已启用，但 voxel_logits_ligand 为空；请启用 ligand head 并对齐 voxel_ligand_logit_dim。")
        # bool, scheduler warmup 内使用 fixed per-class topc
        use_fixed_warmup = self._should_use_candidate_fixed_topk()
        # torch.Tensor | None, (K,), best-F1 阈值缓存; buffer 随模型迁移设备
        p_best_by_class = self._candidate_p_best_by_class if self._has_candidate_p_best_by_class else None
        # torch.Tensor | None, (K,), sampling 阈值缓存; buffer 随模型迁移设备
        p_sampling_by_class = self._candidate_p_sampling_by_class if self._has_candidate_p_sampling_by_class else None
        candidate_outputs = self.candidate_set_builder(
            voxel_logits_ligand=voxel_logits_ligand,
            voxel_valid_mask=batch["voxel_valid_mask"],
            p_best_by_class=p_best_by_class,
            p_sampling_by_class=p_sampling_by_class,
            use_fixed_warmup=use_fixed_warmup,
        )
        if self.anchor_sampler is None:
            return batch, None, candidate_outputs

        # dict[str, torch.Tensor], P anchor 坐标、计数与 metadata
        anchor_outputs = self.anchor_sampler(candidate_outputs=candidate_outputs, batch=batch)
        # torch.Tensor, (sumP, F_atom), P anchor 初始点特征
        pseudo_feat = self.density_cube_encoder(
            voxel_grid=batch["voxel_grid"],
            anchor_voxel_zyx=anchor_outputs["anchor_voxel_zyx"],
            anchor_batch_index=anchor_outputs["anchor_batch_index"],
        )
        pseudo_feat = self._condition_anchor_pseudo_feat(pseudo_feat, anchor_outputs["anchor_class"])
        if int(pseudo_feat.shape[-1]) != int(batch["atom_feat"].shape[-1]):
            raise RuntimeError("density cube pseudo_feat 末维必须等于 batch['atom_feat'] 末维。")
        # dict[str, torch.Tensor], inject_pseudo_atoms 输入字段
        pseudo_dict = {
            "pseudo_coord_centered_world": anchor_outputs["anchor_coord_centered_world"],
            "pseudo_coord_local_voxel": anchor_outputs["anchor_coord_local_voxel"],
            "pseudo_coord_world": anchor_outputs["anchor_coord_world"],
            "pseudo_feat": pseudo_feat,
            "pseudo_batch_index": anchor_outputs["anchor_batch_index"],
            "pseudo_counts": anchor_outputs["anchor_counts"],
            "pseudo_anchor_class": anchor_outputs["anchor_class"],
            "pseudo_anchor_voxel_zyx": anchor_outputs["anchor_voxel_zyx"],
            "pseudo_source_candidate_index": anchor_outputs["anchor_source_candidate_index"],
        }
        point_batch, pseudo_layout = inject_pseudo_atoms(batch, pseudo_dict)
        pseudo_outputs = {**candidate_outputs, **anchor_outputs}
        return point_batch, pseudo_layout, pseudo_outputs

    def _run_point_backbone(
        self,
        batch: dict[str, Any],
        voxel_output_dict: dict[str, Any],
        point_recycle_in: torch.Tensor | None,
        pseudo_layout: PseudoAtomLayout | None = None,
    ) -> dict[str, Any]:
        """
        执行一轮 point backbone, mixed 路径下同步扩展 real-only recycle 输入。

        输入参数:
            - batch: dict[str, Any], real-only 或 mixed batch, 与 pseudo_layout 对齐
            - voxel_output_dict: dict[str, Any], 当前 recycle 的 voxel 输出
            - point_recycle_in: torch.Tensor | None, (sumN_real, C_point), 上一轮 real-only point recycle 状态
            - pseudo_layout: PseudoAtomLayout | None, mixed layout; None 表示 real-only 路径

        输出:
            - point_output_dict: dict[str, Any], point backbone 原始输出, mixed 路径下保留 mixed 顺序
        """
        # torch.Tensor | None, (sumN_current, C_point), 当前轮传给 point backbone 的 recycle 状态
        if pseudo_layout is None:
            pseudo_mask = None
        else:
            if "pseudo_mask" not in batch:
                raise RuntimeError("mixed point batch 必须包含 pseudo_mask。")
            # torch.Tensor, (N_all,), bool, mixed batch 的 P anchor 掩码
            pseudo_mask = batch["pseudo_mask"]
        # torch.Tensor | None, (sumN_current, C_point), 当前轮传给 point backbone 的 recycle 状态
        current_point_recycle = (
            interleave_real_and_pseudo_tensor(point_recycle_in, pseudo_layout)
            if pseudo_layout is not None
            else point_recycle_in
        )
        # bool, fusion hook 是否要求当前 point_like 必须携带 pseudo_mask
        require_pseudo_mask_for_fusion = pseudo_layout is not None
        if getattr(self.point_backbone, "backend", None) == "zeros":
            point_output_dict = self.point_backbone.build_zeros_output(
                atom_feat=batch["atom_feat"],
                atom_coord_centered_world=batch["atom_coord_centered_world"],
                atom_batch_index=batch["atom_batch_index"],
                atom_offsets=batch["atom_offsets"],
                return_feature_names=self.point_feature_names_to_return,
                pseudo_mask=pseudo_mask,
            )
        else:
            def point_feature_hook(feature_name: str, point_like: Any) -> Any:
                return self._fuse_point_variable(
                    feature_name=feature_name,
                    point_like=point_like,
                    voxel_output_dict=voxel_output_dict,
                    batch=batch,
                    require_pseudo_mask=require_pseudo_mask_for_fusion,
                )

            point_output_dict = self.point_backbone(
                atom_feat=batch["atom_feat"],
                atom_coord_centered_world=batch["atom_coord_centered_world"],
                atom_batch_index=batch["atom_batch_index"],
                atom_offsets=batch["atom_offsets"],
                recycle_in=current_point_recycle,
                point_feature_hook=point_feature_hook,
                return_feature_names=self.point_feature_names_to_return,
                pseudo_mask=pseudo_mask,
            )
        return point_output_dict

    def _run_atom_head(
        self,
        outputs: dict[str, Any],
        atom_head_batch: dict[str, Any],
        pseudo_layout: PseudoAtomLayout | None,
    ) -> None:
        """
        在最后一轮 point backbone 后运行 Stage1AtomHead, 并把监督字段裁成 real-only。

        输入参数:
            - outputs: dict[str, Any], 最后一轮 backbone 输出汇总, 将会原地写入 atom head 输出
            - atom_head_batch: dict[str, Any], 与 outputs["fused_point_feat"] 同布局的 real 或 mixed batch, 仅用于提供 pseudo_mask 和真实原子的信息(atom label、atom_valid_mask 等), 不提供特征
            - pseudo_layout: PseudoAtomLayout | None, mixed layout; None 表示 real-only 路径

        输出:
            - None, 原地更新 outputs 中 atom_tokens/atom_hidden/atom_logits/pseudo_feature 与 supervised 字段
        """
        if self.atom_head is None:
            outputs["atom_tokens"] = None
            outputs["atom_hidden"] = None
            outputs["atom_logits"] = None
            outputs["pseudo_feature"] = None
            return

        # torch.Tensor | None, (N_all,), bool, mixed 路径下 True 表示 P anchor
        pseudo_mask = atom_head_batch.get("pseudo_mask") if pseudo_layout is not None else None
        atom_head_output = self.atom_head(
            point_feat=outputs["fused_point_feat"],
            point_state=outputs["point_state"],
            atom_coord_centered_world=atom_head_batch["atom_coord_centered_world"],
            atom_valid_mask=atom_head_batch["atom_valid_mask"],
            pseudo_mask=pseudo_mask,
        )
        outputs.update(atom_head_output)

        if pseudo_layout is None:
            outputs["atom_target"] = atom_head_batch.get("atom_label")
            outputs["atom_valid_mask"] = atom_head_batch.get("atom_valid_mask")
            outputs["atom_counts"] = atom_head_batch.get("atom_counts")
            outputs["atom_coord_local_voxel"] = atom_head_batch.get("atom_coord_local_voxel")
            outputs["atom_is_in_core_box"] = atom_head_batch.get("atom_is_in_core_box")
            outputs["atom_global_indices"] = atom_head_batch.get("atom_global_indices")
            return

        outputs["atom_target"] = extract_real_tensor_from_mixed(atom_head_batch.get("atom_label"), pseudo_layout)
        outputs["atom_valid_mask"] = extract_real_tensor_from_mixed(atom_head_batch.get("atom_valid_mask"), pseudo_layout)
        outputs["atom_counts"] = pseudo_layout.real_counts.to(device=outputs["fused_point_feat"].device)
        outputs["atom_coord_local_voxel"] = extract_real_tensor_from_mixed(
            atom_head_batch.get("atom_coord_local_voxel"), pseudo_layout
        )
        outputs["atom_is_in_core_box"] = extract_real_tensor_from_mixed(
            atom_head_batch.get("atom_is_in_core_box"), pseudo_layout
        )
        outputs["atom_global_indices"] = extract_real_tensor_from_mixed(
            atom_head_batch.get("atom_global_indices"), pseudo_layout
        )

    def _run_sparse_refine_head(
        self,
        outputs: dict[str, Any],
        voxel_output_dict: dict[str, Any],
        point_batch: dict[str, Any],
        pseudo_layout: PseudoAtomLayout | None,
    ) -> None:
        """
        在 final atom head 后将 P 消息聚合回唯一候选 C 并输出 refined logits。

        输入参数:
            - outputs: dict[str, Any], final recycle 输出字典，将原地追加 sparse refine 字段
            - voxel_output_dict: dict[str, Any], final voxel backbone 输出
            - point_batch: dict[str, Any], final real/P mixed batch，提供 BOX 坐标字段
            - pseudo_layout: PseudoAtomLayout | None, final mixed 布局

        输出:
            - None, 原地写入消息有效掩码与 `ligand_refine_logits_C`
        """
        if self.sparse_refine_head is None:
            return
        if self.anchor_to_candidate is None or pseudo_layout is None:
            raise RuntimeError("sparse refine 启用时 final recycle 必须存在 P anchor mixed layout。")
        if outputs.get("pseudo_feature") is None:
            raise RuntimeError("sparse refine 必须在 atom head 输出 pseudo_feature 后执行。")
        # torch.Tensor, (sumP, C_point), final mixed point_feat 中属于 P 的 backbone 特征
        P_point_backbone_feat = extract_pseudo_tensor_from_mixed(outputs["fused_point_feat"], pseudo_layout)
        # torch.Tensor, (sumP, C_pseudo), atom head 输出的 P pseudo_feature
        P_atom_head_feat = outputs["pseudo_feature"]
        # torch.Tensor, (B, C_voxel, D, H, W), final voxel backbone 导出的固定 voxel_final 特征图
        voxel_final = voxel_output_dict["voxel_features"]["voxel_final"]
        # dict[str, torch.Tensor], C voxel center 对应的 local/world/centered-world 坐标字典
        candidate_coords = build_anchor_coordinates(
            anchor_voxel_zyx=outputs["candidate_voxel_zyx"],
            anchor_batch_index=outputs["candidate_batch_index"],
            box_origin_world=point_batch["box_origin_world"],
            voxel_size_world=point_batch["voxel_size_world"],
            box_shape_zyx=point_batch["box_shape_zyx"],
        )
        # torch.Tensor, (sumC, 3), C voxel center 的 centered-world 坐标
        candidate_coord_centered_world = candidate_coords["anchor_coord_centered_world"]
        # torch.Tensor, (sumP, C_voxel), P 来源 voxel center 采样得到的 voxel_final 特征
        P_voxel_backbone_feat = self._gather_voxel_feature_at_zyx(
            voxel_feat=voxel_final,
            voxel_zyx=outputs["anchor_voxel_zyx"],
            point_batch_index=outputs["anchor_batch_index"],
        )
        # torch.Tensor, (sumC, C_voxel), C 来源 voxel center 采样得到的 voxel_final 特征
        C_voxel_backbone_feat = self._gather_voxel_feature_at_zyx(
            voxel_feat=voxel_final,
            voxel_zyx=outputs["candidate_voxel_zyx"],
            point_batch_index=outputs["candidate_batch_index"],
        )
        # torch.Tensor, (sumC, C_logits), 默认复用 candidate_set_builder 产出的 detached candidate logits
        voxel_logits_C = outputs["candidate_logits"]
        if not bool(getattr(self.sparse_refine_head, "detach_voxel_logits", True)):
            # torch.Tensor, (B, C_logits, D, H, W), final voxel backbone 输出的带梯度 ligand logits
            source_logits = voxel_output_dict["voxel_logits_ligand"]
            # torch.Tensor, (sumC, 3), 唯一 C 的 voxel z/y/x 坐标
            voxel_zyx = outputs["candidate_voxel_zyx"]
            voxel_logits_C = source_logits[
                outputs["candidate_batch_index"],
                :,
                voxel_zyx[:, 0],
                voxel_zyx[:, 1],
                voxel_zyx[:, 2],
            ]
        # dict[str, torch.Tensor], 每个 C 的 P 邻居索引、距离、相对坐标与有效掩码
        neighbor_outputs = self.anchor_to_candidate(
            candidate_coord_centered_world=candidate_coord_centered_world,
            candidate_batch_index=outputs["candidate_batch_index"],
            candidate_class=outputs["candidate_class"],
            anchor_coord_centered_world=outputs["anchor_coord_centered_world"],
            anchor_batch_index=outputs["anchor_batch_index"],
            anchor_class=outputs["anchor_class"],
        )
        # dict[str, torch.Tensor], 稀疏消息有效掩码与 C 上 refined logits
        refine_outputs = self.sparse_refine_head(
            voxel_logits=voxel_logits_C,
            C_voxel_backbone_feat=C_voxel_backbone_feat,
            P_point_backbone_feat=P_point_backbone_feat,
            P_atom_head_feat=P_atom_head_feat,
            P_voxel_backbone_feat=P_voxel_backbone_feat,
            anchor_class=outputs["anchor_class"],
            **neighbor_outputs,
        )
        outputs.update(neighbor_outputs)
        outputs.update(refine_outputs)

    def forward(self, batch: dict[str, Any]) -> dict[str, Any]:
        """
        执行 Stage1 前向, P anchor 准备点固定在最后一轮 recycle。

        输入参数:
            - batch: dict[str, Any], box_point_collate 输出的 real-only batch

        输出:
            - outputs: dict[str, Any], 最后一轮 voxel/point/atom 输出
        """
        if not self.enable_recycling:
            recycle_steps = 1
        elif self.training and self.randomize_recycles:
            recycle_steps = int(torch.randint(1, self.max_recycles + 1, (1,)).item())
        else:
            recycle_steps = self.max_recycles

        batch, embed_output = self._run_embed_head_once(batch)
        # torch.Tensor, (B, C_in, D, H, W), recycle 循环内复用的 voxel 输入
        voxel_input = self._build_voxel_input(batch, embed_output)
        voxel_recycle_in: torch.Tensor | None = None
        point_recycle_in: torch.Tensor | None = None
        outputs: dict[str, Any] = {}
        last_atom_head_batch: dict[str, Any] = batch
        last_pseudo_layout: PseudoAtomLayout | None = None


        for recycle_idx in range(recycle_steps):
            voxel_output_dict = self._run_voxel_backbone(voxel_input, voxel_recycle_in)
            is_final_recycle = recycle_idx == recycle_steps - 1
            if is_final_recycle:
                point_batch, pseudo_layout, pseudo_outputs = self._prepare_pseudo_batch(batch, voxel_output_dict)
            else:
                point_batch, pseudo_layout, pseudo_outputs = batch, None, {}

            point_output_dict = self._run_point_backbone(
                batch=point_batch,
                voxel_output_dict=voxel_output_dict,
                point_recycle_in=point_recycle_in,
                pseudo_layout=pseudo_layout,
            )

            if not is_final_recycle:
                if pseudo_layout is not None:
                    raise RuntimeError("非最后一轮 recycle 不允许注入 P anchors。")
                voxel_recycle_in = voxel_output_dict["voxel_recycle_out"]
                if voxel_recycle_in is not None and self.detach_recycle_states:
                    voxel_recycle_in = voxel_recycle_in.detach()
                point_recycle_in = point_output_dict["point_recycle_out"]
                if point_recycle_in is not None and self.detach_recycle_states:
                    point_recycle_in = point_recycle_in.detach()
                continue

            if is_final_recycle:
                if pseudo_layout is not None:
                    # dict[str, Any], 仅包含真实原子监督字段的 real-only batch
                    # dict[str, Any], 仅包含真实原子 point backbone 中间输出的结果字典
                    real_batch, _real_point_feat, _real_point_state, real_point_output_dict = extract_real_point_output(
                        mixed_batch=point_batch,
                        fused_point_feat=point_output_dict["point_feat"],
                        point_output_dict=point_output_dict,
                        layout=pseudo_layout,
                    )
                else:
                    real_batch = point_batch
                    real_point_output_dict = point_output_dict
                last_atom_head_batch = point_batch
                last_pseudo_layout = pseudo_layout
                outputs = {
                    "fused_point_feat": point_output_dict["point_feat"],
                    "point_state": point_output_dict["point_state"],
                    "atom_target": real_batch.get("atom_label"),
                    "atom_valid_mask": real_batch.get("atom_valid_mask"),
                    "atom_counts": real_batch.get("atom_counts"),
                    "atom_coord_local_voxel": real_batch.get("atom_coord_local_voxel"),
                    "atom_is_in_core_box": real_batch.get("atom_is_in_core_box"),
                    "atom_global_indices": real_batch.get("atom_global_indices"),
                    "voxel_logits_aux": voxel_output_dict["voxel_logits_aux"],
                    "voxel_logits_ligand": voxel_output_dict.get("voxel_logits_ligand"),
                    "voxel_outputs": voxel_output_dict,
                    "point_outputs": real_point_output_dict,
                    "embed_output": embed_output,
                    "voxel_recycle_out": voxel_output_dict["voxel_recycle_out"],
                    "point_recycle_out": real_point_output_dict["point_recycle_out"],
                    **pseudo_outputs,
                }

        self._run_atom_head(outputs, atom_head_batch=last_atom_head_batch, pseudo_layout=last_pseudo_layout)
        self._run_sparse_refine_head(
            outputs,
            voxel_output_dict=outputs["voxel_outputs"],
            point_batch=last_atom_head_batch,
            pseudo_layout=last_pseudo_layout,
        )
        outputs["recycle_passes_used"] = recycle_steps
        return outputs




    # ============================================================
    # ==================== 工具函数: 关于 Sparse Candidate Builder ====================
    # ============================================================
    
    def get_sparse_candidate_class_ids(self) -> tuple[int, ...] | None:
        """
        返回 sparse candidate builder 配置的候选类别 ID。

        输出:
            - class_ids: tuple[int, ...] | None, builder 未启用时为 None; 启用时为前景候选类别 ID
        """
        if self.candidate_set_builder is None:
            return None
        if not hasattr(self.candidate_set_builder, "candidate_class_ids"):
            raise AttributeError("candidate_set_builder 必须暴露 candidate_class_ids。")
        return tuple(int(class_id) for class_id in self.candidate_set_builder.candidate_class_ids)

    def set_sparse_candidate_thresholds(
        self,
        p_best_by_class: torch.Tensor | None,
        p_sampling_by_class: torch.Tensor | None,
    ) -> None:
        """
        保存 wrapper 同步过来的 candidate threshold cache。

        输入参数:
            - p_best_by_class: torch.Tensor | None, (K,), best-F1 阈值缓存; None 表示尚不可用
            - p_sampling_by_class: torch.Tensor | None, (K,), sampling 阈值缓存; None 表示尚不可用

        输出:
            - None, 原地更新 runtime cache: self._candidate_p_best_by_class / self._candidate_p_sampling_by_class
        """
        target_device = self._candidate_p_best_by_class.device
        if p_best_by_class is None:
            self._candidate_p_best_by_class = torch.empty((0,), device=target_device, dtype=torch.float32)
            self._has_candidate_p_best_by_class = False
        else:
            self._candidate_p_best_by_class = p_best_by_class.detach().to(device=target_device, dtype=torch.float32).reshape(-1)
            self._has_candidate_p_best_by_class = True
        if p_sampling_by_class is None:
            self._candidate_p_sampling_by_class = torch.empty((0,), device=target_device, dtype=torch.float32)
            self._has_candidate_p_sampling_by_class = False
        else:
            self._candidate_p_sampling_by_class = p_sampling_by_class.detach().to(device=target_device, dtype=torch.float32).reshape(-1)
            self._has_candidate_p_sampling_by_class = True

    def set_sparse_candidate_runtime(
        self,
        global_step: int,
        candidate_warmup_steps: int,
        allow_warmup_fixed_topk: bool,
    ) -> None:
        """
        保存 wrapper 同步过来的 candidate runtime 状态。

        输入参数:
            - global_step: int, 当前 optimizer step
            - candidate_warmup_steps: int, scheduler warmup step 数; 仅此阶段允许 fixed topk
            - allow_warmup_fixed_topk: bool, 当前 lifecycle 是否允许 warmup fixed topk; standalone validate/test/predict 为 False

        输出:
            - None, 原地更新 runtime 状态: self._candidate_global_step / self._candidate_warmup_steps / self._candidate_allow_warmup_fixed_topk
        """
        self._candidate_global_step = int(global_step)
        self._candidate_warmup_steps = int(candidate_warmup_steps)
        self._candidate_allow_warmup_fixed_topk = bool(allow_warmup_fixed_topk)

    def _should_use_candidate_fixed_topk(self) -> bool:
        """
        判断当前 forward 是否处于 candidate fixed topk 阶段。

        输出:
            - use_fixed: bool, True 表示 scheduler warmup 内使用固定 per-class topc
        """
        return (
            self._candidate_allow_warmup_fixed_topk
            and self._candidate_warmup_steps > 0
            and self._candidate_global_step < self._candidate_warmup_steps
        )
