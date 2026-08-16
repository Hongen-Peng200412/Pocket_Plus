"""
Stage1 体素-点云联合模型的清理后主流程. 

对齐契约（修改时必须同步）:
    - forward 输入 batch 在 _run_embed_head_once 之前必须是 real-only; P anchors 只允许在最后一次 recycle 的 _prepare_pseudo_batch 后进入 point backbone. 
    - mixed layout 必须来自 pseudo_atoms.inject_pseudo_atoms, 每个 BOX 内顺序固定为 `[real_i..., pseudo_i...]`. 
    - C 候选集只在最后一轮 recycle 生成; SparseCandidateSetBuilder 直接在全 BOX 体素上从 voxel_logits_ligand 选取候选, 不再消费体素有效掩码. 
    - atom 监督的唯一掩码是 atom_is_in_core_box; 体素监督和候选选择默认覆盖整张 BOX. 

关键输入字段:
    - voxel_grid: torch.Tensor, (B,C_in,D,H,W), voxel backbone 输入密度/特征体. 
    - atom_feat: torch.Tensor, (N_real,F_atom) 或 mixed 路径下 (N_all,F_atom), 点分支输入特征. 
    - atom_coord_centered_world: torch.Tensor, (N,3), 以 BOX 中心为原点的连续世界坐标, XYZ 轴序, 单位 Å; 不是 voxel 坐标. 
    - atom_coord_local_voxel: torch.Tensor, (N,3), BOX-local 连续 voxel 坐标, XYZ 轴序, corner 语义; 不是世界坐标或离散索引. 
    - atom_label: torch.Tensor, (N_real,) 或 (N_all,), real atom 监督标签; P anchor 槽位只作为占位. 
    - atom_is_in_core_box: torch.Tensor, (N_real,) 或 (N_all,), bool, real atom 是否参与 atom 监督. 
    - real_mask / pseudo_mask: torch.Tensor, (N_all,), mixed-only 点类型掩码. 

detach、refine 与 logit 残差:
    - detach_real_point_feat_into_atomhead / detach_pseudo_point_feat_into_atomhead 控制进入最终 Stage1AtomHead 的 real/P 点特征是否 detach. 
    - detach_pseudo_point_feat_into_refine 单独控制 refine 消费的 P_final_point_feat; P_after_interaction_feat 在进入 refine 前始终 detach. 
    - detach_voxel_into_refine 统一控制 refine 消费的 base logits、C/P voxel_final 特征是否回传到 voxel backbone. 
    - refine_receptor_from_voxel 为 atom_logits 追加 home 体素 voxel_logits_aux.detach() 残差 base. 
    - sparse_refine_residual_mode / sparse_refine_use_C_voxel_logits 由 detach_residual 配置组注入 sparse_refine_head. 

forward 输出契约:
    - voxel_logits_aux: (B,C_receptor,D,H,W), 受体结合区域体素预测. 
    - voxel_logits_ligand: (B,C_ligand,D,H,W), 配体区域体素预测. 
    - voxel_logits_protein: (B,5,D,H,W) 或 None, 背景/N/CA/C/O 主链类别预测. 
    - voxel_logits_nucleic: (B,7,D,H,W) 或 None, 背景/P/O5'/C5'/C4'/C3'/O3' 主链类别预测. 
    - voxel_logits_distance: (B,1,D,H,W) 或 None, sigmoid 后解释为配体反距离预测. 
    - point_feat_raw: torch.Tensor, (N_all,C_point), point backbone 原始输出. 
    - fused_point_feat: torch.Tensor, (N_all,C_point), 按 into_atomhead detach 路由后的最终分类头输入. 
    - real_feat_before_interaction / real_feat_after_interaction: torch.Tensor | None, (N_real,C_point), real cross-attn 前后特征. 
    - pseudo_feat_before_interaction / pseudo_feat_after_interaction: torch.Tensor | None, (N_pseudo,C_point), P cross-attn 前后特征. 
    - atom_logits: torch.Tensor | None, (N_real,atom_logit_dim), 最终 real atom logits. 
    - pseudo_logits: torch.Tensor | None, (N_pseudo,pseudo_ligand_logit_dim), 最终 P ligand 区域归属 logits. 
    - atom_target / atom_counts / atom_coord_local_voxel / atom_is_in_core_box / atom_global_indices: real-only wrapper-facing 字段. 
    - pseudo_voxel_zyx / pseudo_batch_index: P anchor home 体素与所属 BOX, 供 wrapper 采样 P 监督. 
"""
from __future__ import annotations

import math
from typing import Any, Sequence

import torch
import torch.nn.functional as F
from hydra.utils import instantiate
from torch import nn

from src.model.stage1_atom_head import Stage1AtomHead
from src.model.stage1_embed_head import gauss_scatter_to_voxel_grid, scatter_to_voxel_grid, soft_scatter_to_voxel_grid
from src.model.utils import CubeWeightingParams, FeatureCombine, gather_voxel_cube, gather_voxel_feature_at_zyx
from src.model.typed_point import (
    TypedPointConfig,
    merge_type_aware_tensor_outputs,
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
    """
    组合 voxel、point、候选与 refine 子系统的 Stage1 顶层模型. 

    AdaLigand 的四个 producer 复用本类, 但通过配置关闭不同分支:
        - unet_c1: 只运行单通道 density 与 voxel backbone. 
        - Find_0: 56D density 拼接 core receptor raw50 hard scatter.
        - Find_1: 56D density 拼接 embed head 生成的 50D value + 2D occupancy.
        - Find_2: 56D density与 embed head 生成的 56D Gaussian tune 逐元素相加. 

    完整 :meth:`forward` 还会运行共同的 point 路径并发布 centered 所需特征; 
    :meth:`forward_voxel_probability` 只复现完整图推理所需的 voxel 前半段. 
    输入参数:
        - 初始化参数: 见 `__init__` 的完整参数契约

    前向输入:
        - batch: dict[str,Any], canonical Stage1 batch; 坐标字段同时保留 centered 连续世界 XYZ 与 BOX-local 连续 voxel XYZ

    前向输出:
        - outputs: dict[str,Any], 最后一轮 voxel/point/atom 输出及 centered 生产所需的稳定特征出口
    """

    def __init__(
        self,
        voxel_backbone: nn.Module | Any,
        point_backbone: nn.Module | Any | None,
        point_fusion_map: dict[str, str] | None,
        point_fusion_modes: Sequence[str],
        sampler_modes: Sequence[str],
        fusion_mlp_ratio: float,
        fusion_proj_drop: float,
        atom_head_hidden_dim: int,
        atom_head_interaction_radius: float,
        atom_head_interaction_max_neighbors: int,
        atom_head_interaction_num_heads: int,
        atom_head_interaction_detach_source_feat: bool,
        atom_logit_dim: int,
        enable_recycling: bool,
        max_recycles: int,
        randomize_recycles: bool,
        detach_recycle_states: bool,
        act_layer_name: str,
        ffn_type: str,
        enable_atom_head: bool = True,
        embed_head: nn.Module | Any | None = None,
        pseudo_atom_cfg: dict | None = None,
        prior_prob: float | None = None,
        prior_probs: Sequence[float] | None = None,
        prior_prob_init_enabled: bool = True,
        prior_prob_voxel_receptor: float | None = None,
        prior_prob_point_receptor: float | None = None,
        prior_prob_voxel_ligand: float | None = None,
        prior_prob_point_ligand: float | None = None,
        prior_prob_sparse_refine: float | None = None,
        pseudo_ligand_logit_dim: int = 1,
        online_pdb_feature: bool = False,
        online_pdb_feature_reduce: str = "sum",
        online_pdb_feature_use_soft_splatting: bool = False,
        online_pdb_feature_scatter_kernel: str = "legacy",
        online_pdb_feature_sigma_voxel: float = 0.7,
        online_pdb_feature_add_occupancy: bool = False,
        online_pdb_feature_add_centroid: bool = False,
        online_pdb_feature_dim: int = 50,
        typed_point_cfg: dict[str, Any] | TypedPointConfig | None = None,
        candidate_set_cfg: dict[str, Any] | nn.Module | None = None,
        anchor_sampler_cfg: dict[str, Any] | nn.Module | None = None,
        density_cube_cfg: dict[str, Any] | nn.Module | None = None,
        anchor_to_candidate_cfg: dict[str, Any] | nn.Module | None = None,
        sparse_refine_head_cfg: dict[str, Any] | nn.Module | None = None,
        anchor_class_conditioning_cfg: dict[str, Any] | None = None,
        detach_real_point_feat_into_atomhead: bool = False,
        detach_pseudo_point_feat_into_atomhead: bool = False,
        detach_pseudo_point_feat_into_refine: bool = False,
        detach_voxel_feat_into_real_point: bool = False,
        detach_voxel_feat_into_pseudo_point: bool = False,
        detach_voxel_into_refine: bool = True,
        refine_receptor_from_voxel: bool = False,
        enable_interface_norm: bool = False,
        sparse_refine_residual_mode: str = "residual",
        sparse_refine_use_C_voxel_logits: bool = True,
        real_atom_density_cube: bool = False,
        real_atom_density_share_encoder: bool = True,
        real_atom_density_cube_size: int = 7,
        real_atom_density_combine_mode: str = "film_plus",
        real_atom_density_fusion_mlp_ratio: float = 2.0,
        real_atom_density_fusion_proj_drop: float = 0.2,
        real_density_cube_cfg: dict[str, Any] | nn.Module | None = None,
        sampler_cube_init: Sequence[Any] | None = None,
        fusion_cube_chunk_size: int = 16384,
    ) -> None:
        """
        Stage1 体素-点云联合模型, voxel backbone 每轮 recycle, P anchors 只在最后一轮注入. 

        输入参数:
            - 直传(cfg)
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

            - 重要
                - sparse_refine_residual_mode: str, refine 头输出模式 direct/residual; 延迟实例化时注入覆盖 sparse_refine_head_cfg.mode
                - sparse_refine_use_C_voxel_logits: bool, refine 头是否把 C 原始 logits 拼进输入; 注入覆盖 sparse_refine_head_cfg.inputs.use_C_voxel_logits
                - atom_head_interaction_radius: float, real/P 几何 cross-attn 的邻域半径
                - atom_head_interaction_max_neighbors: int, real/P 几何 cross-attn 每个 query 最多保留的 source 邻居数
                - atom_head_interaction_num_heads: int, real/P 几何 cross-attn 头数
                - atom_head_interaction_detach_source_feat: bool, cross-attn source 特征是否 detach
                - enable_interface_norm: bool, 掌管4项: embed head 输出、伪原子吸收 density cube 作为初始特征、真实原子吸收 density cube 作为condition(然后 FeatureCombine 融合)、point backbone从voxel backbone得到condition(然后 FeatureCombine 融合)
                - sampler_cube_init: Sequence|None, 与 point_fusion_map 等长; weighted_cube 的 hook 填 (a,b,c,d), 其余 None
                - fusion_cube_chunk_size: int, weighted_cube/cube_mean 采样每个向量化块的最大点数

            - detach、refine 与 logit 残差
                - detach_real_point_feat_into_atomhead: bool, 真实原子槽位喂最终 Stage1AtomHead 时是否 detach
                - detach_pseudo_point_feat_into_atomhead: bool, 伪原子(P)槽位喂最终 Stage1AtomHead 时是否 detach
                - detach_pseudo_point_feat_into_refine: bool, 伪原子(P)喂 refine 的 P_final_point_feat 时是否 detach(从 point_feat_raw 取出再 detach); P_after_interaction_feat 始终 detach
                - detach_voxel_feat_into_real_point: bool, _fuse_point_variable 里真实原子接收采样 voxel 特征是否 detach
                - detach_voxel_feat_into_pseudo_point: bool, _fuse_point_variable 里伪原子接收采样 voxel 特征是否 detach
                - detach_voxel_into_refine: bool, 统一覆盖 refine 吃的三处 voxel 信息(base logits、C/P voxel_final 特征)是否 detach
                - refine_receptor_from_voxel: bool, atom_logits 加上该原子 home 体素处 voxel_logits_aux 的 detach 残差 base

            - 真实原子的 density 调制
                - real_atom_density_cube: bool, 真实原子是否加 density cube 特征(combine 零初始化, 开局恒等)
                - real_atom_density_share_encoder: bool, 真实原子 density 是否复用伪原子 density_cube_encoder(False 需 real_density_cube_cfg)
                - real_atom_density_cube_size: int, 真实原子 cube 边长(奇数)
                - real_atom_density_combine_mode: str, embed<->density 融合; concat_mlp/film/film_plus/mini_residue
                - real_atom_density_fusion_mlp_ratio: float, real density 融合 MLP 隐藏层倍率(同 voxel 侧公式, 对 cat 输入维 2*feat 生效); film 不使用
                - real_atom_density_fusion_proj_drop: float, real density 融合 MLP dropout 概率; film 不使用
                - real_density_cube_cfg: dict|nn.Module|None, 独立 real density encoder 配置; 仅 share_encoder=False 时需要

            - 点体素分支融合
                - point_fusion_map: dict[str, str] | None, point 变量名到 voxel 特征名的融合映射
                - point_fusion_modes: Sequence[str], 每个 point 变量的融合模式
                - sampler_modes: Sequence[str], 每个 point 变量的 voxel 采样模式
                - fusion_mlp_ratio: float, concat_mlp / mini_residue / film_plus 融合 MLP 隐藏层倍率
                - fusion_proj_drop: float, concat_mlp / mini_residue / film_plus 融合 MLP dropout 概率; film 不使用

            - atom head
                - atom_head_hidden_dim: int, real/P 轻量分类尾部隐藏通道数
                - atom_logit_dim: int, real atom logits 输出通道数
                - pseudo_ligand_logit_dim: int, P ligand 区域归属 logits 输出通道数
  
            - 其余
                - prior_prob: float | None, legacy 单通道 sigmoid 正类先验概率; 仅 prior_prob_init_enabled=True 且对应头未被命名先验覆盖时使用
                - prior_probs: Sequence[float] | None, 多通道 softmax 类别先验概率; 多通道任务(tri)沿用此字段, 不受命名 sigmoid 先验影响
                - prior_prob_init_enabled: bool, 先验 bias 初始化总开关; False 时所有头跳过先验初始化(各头收到 None)
                - prior_prob_voxel_receptor: float | None, voxel aux(受体区域)头单通道 sigmoid 先验; 注入 voxel_backbone
                - prior_prob_point_receptor: float | None, 点云受体原子头单通道 sigmoid 先验
                - prior_prob_voxel_ligand: float | None, dense voxel ligand 头单通道 sigmoid 先验; 注入 voxel_backbone
                - prior_prob_point_ligand: float | None, P(虚拟原子)ligand 区域归属头单通道 sigmoid 先验
                - prior_prob_sparse_refine: float | None, sparse refine 头单通道 sigmoid 先验; 仅 direct 模式生效(residual 模式零初始化跳过)
                - online_pdb_feature: bool, embed head 无 voxel 输出时是否在线 scatter raw atom_feat
                - online_pdb_feature_reduce: str, 在线 scatter 聚合方式
                - online_pdb_feature_use_soft_splatting: bool, scatter_kernel="legacy" 时, 在线 raw atom_feat scatter 是否使用三线性 soft splatting
                - online_pdb_feature_scatter_kernel: str, 在线 scatter 核; "legacy"=沿用 use_soft_splatting 的三线性/单体素, "gauss27"=3×3×3 各向同性高斯
                - online_pdb_feature_sigma_voxel: float, gauss27 高斯核标准差(单位: 体素); 仅 scatter_kernel="gauss27" 时生效, 推荐值 0.7
                - online_pdb_feature_add_occupancy: bool, gauss27 是否追加 2 维 occupancy 通道; 仅 scatter_kernel="gauss27" 时生效
                - online_pdb_feature_add_centroid: bool, gauss27 是否追加 3 维加权 centroid 通道; 仅 scatter_kernel="gauss27" 时生效
                - online_pdb_feature_dim: int, 在线 scatter 的 raw atom 特征通道数
                - enable_recycling: bool, 是否启用 recycle
                - max_recycles: int, 最大 recycle 轮数
                - randomize_recycles: bool, 训练态是否随机采样 recycle 轮数
                - detach_recycle_states: bool, recycle 状态跨轮传递时是否 detach
                - act_layer_name: str, 激活函数名称
                - ffn_type: str, point backbone Block FFN 类型
            
        前向输出:
            - outputs: dict[str, Any], 包含 voxel/point 输出、real/P 最终分类头输出、real-only atom supervised 字段与可选 sparse refine 输出
        """
        super().__init__()
        if pseudo_atom_cfg is not None:
            raise ValueError("旧 pseudo_atom_cfg 已删除；P anchors 将由 sparse refine anchor pipeline 提供。")
        if resolve_act_layer is None and (point_backbone is not None or embed_head is not None):
            raise ImportError("Find 配置需要 PTV3 resolve_act_layer。") from _PTV3_IMPORT_ERROR

        # -------------------------------------------------------- 平凡初始化 --------------------------------------------------------
        # TypedPointConfig, Stage1 全局 typed point 配置
        # type[nn.Module], 统一激活函数类; density/fusion/atom head 共用同一解析结果
        if resolve_act_layer is None:
            pure_voxel_acts = {"gelu": nn.GELU, "silu": nn.SiLU, "relu": nn.ReLU}
            act_key = str(act_layer_name).lower()
            if act_key not in pure_voxel_acts:
                raise ValueError(f"纯 voxel 配置不支持 act_layer_name={act_layer_name!r}。")
            act_cls = pure_voxel_acts[act_key]
        else:
            act_cls = resolve_act_layer(str(act_layer_name))
        self.typed_point_cfg = normalize_typed_point_cfg(typed_point_cfg)
        self.embed_head = embed_head if isinstance(embed_head, nn.Module) else instantiate(embed_head) if embed_head is not None else None
        # float | None, 注入 voxel_backbone 的 aux/ligand 头单通道 sigmoid 先验; 主开关关时为 None(跳过先验), 多通道任务由 voxel_backbone 配置自带 prior_probs 接管
        _injected_voxel_receptor_prior = prior_prob_voxel_receptor if prior_prob_init_enabled else None
        _injected_voxel_ligand_prior = prior_prob_voxel_ligand if prior_prob_init_enabled else None
        self.voxel_backbone = (
            voxel_backbone
            if isinstance(voxel_backbone, nn.Module)
            else instantiate(
                voxel_backbone,
                prior_prob_voxel_receptor=_injected_voxel_receptor_prior,
                prior_prob_voxel_ligand=_injected_voxel_ligand_prior,
            )
        )
        self.point_backbone = (
            point_backbone
            if (point_backbone is None or isinstance(point_backbone, nn.Module))
            else instantiate(point_backbone)
        )
        if self.point_backbone is None:
            point_dependent = (
                bool(enable_atom_head)
                or bool(point_fusion_map)
                or bool(self.embed_head is not None and self.embed_head.has_point_output)
                or candidate_set_cfg is not None
                or anchor_sampler_cfg is not None
                or density_cube_cfg is not None
                or anchor_to_candidate_cfg is not None
                or sparse_refine_head_cfg is not None
                or bool(real_atom_density_cube)
            )
            if point_dependent:
                raise ValueError("point_backbone=None 只允许不产生点输出的纯 voxel 配置。")

        # 关于 sparse refine head 的 detach
        self.detach_real_point_feat_into_atomhead = bool(detach_real_point_feat_into_atomhead)
        self.detach_pseudo_point_feat_into_atomhead = bool(detach_pseudo_point_feat_into_atomhead)
        self.detach_pseudo_point_feat_into_refine = bool(detach_pseudo_point_feat_into_refine)
        self.detach_voxel_feat_into_real_point = bool(detach_voxel_feat_into_real_point)
        self.detach_voxel_feat_into_pseudo_point = bool(detach_voxel_feat_into_pseudo_point)
        self.detach_voxel_into_refine = bool(detach_voxel_into_refine)

        # ----- prior bias 初始化总开关与各头命名先验(被主开关与各头残差开关压制) -----
        # bool, 先验 bias 初始化总开关; False 时所有头收到 None、跳过先验初始化
        self.prior_prob_init_enabled = bool(prior_prob_init_enabled)
        # float | None, 各头有效单通道 sigmoid 先验; 主开关关时统一为 None, 各头构造时按 None 跳过 bias 先验
        self._eff_prior_voxel_receptor = prior_prob_voxel_receptor if self.prior_prob_init_enabled else None
        self._eff_prior_point_receptor = prior_prob_point_receptor if self.prior_prob_init_enabled else None
        self._eff_prior_voxel_ligand = prior_prob_voxel_ligand if self.prior_prob_init_enabled else None
        self._eff_prior_point_ligand = prior_prob_point_ligand if self.prior_prob_init_enabled else None
        self._eff_prior_sparse_refine = prior_prob_sparse_refine if self.prior_prob_init_enabled else None
        # Sequence[float] | None, 多通道 softmax 先验(tri 等); 主开关关时为 None
        self._eff_prior_probs = prior_probs if self.prior_prob_init_enabled else None
        # float | None, legacy 单通道 sigmoid 先验; 主开关关时为 None, 仅在命名先验缺省时作为兜底
        self._eff_prior_prob = prior_prob if self.prior_prob_init_enabled else None
        # bool, atom head 是否多通道(tri 等); 多通道走 softmax prior_probs, 单通道 sigmoid 走命名 point_receptor(缺省回退 legacy prior_prob)
        _atom_is_multiclass = int(atom_logit_dim) > 1
        # float | None, atom head有效单通道 sigmoid 先验; 多通道时为 None
        self._eff_atom_prior_prob = None if _atom_is_multiclass else (
            self._eff_prior_point_receptor if self._eff_prior_point_receptor is not None else self._eff_prior_prob
        )
        # Sequence[float] | None, atom head有效多通道 softmax 先验; 单通道时为 None
        self._eff_atom_prior_probs = self._eff_prior_probs if _atom_is_multiclass else None
        # int, P ligand 区域归属 logits 通道数
        self.pseudo_ligand_logit_dim = int(pseudo_ligand_logit_dim)

        # refine 与 logit 残差
        self.refine_receptor_from_voxel = bool(refine_receptor_from_voxel)
        # str/bool, refine 头残差模式与 base logit 拼接开关; 归属 detach_residual 组, 在 sparse_refine_head 延迟实例化时注入
        self.sparse_refine_residual_mode = str(sparse_refine_residual_mode)
        self.sparse_refine_use_C_voxel_logits = bool(sparse_refine_use_C_voxel_logits)

        # 其余重要
        self.enable_interface_norm = bool(enable_interface_norm)
        self.online_pdb_feature = bool(online_pdb_feature)
        self.online_pdb_feature_reduce = str(online_pdb_feature_reduce)
        self.online_pdb_feature_use_soft_splatting = bool(online_pdb_feature_use_soft_splatting)
        # str, online raw atom_feat scatter 核; "legacy"=沿用 use_soft_splatting 的三线性/单体素, "gauss27"=3×3×3 各向同性高斯
        self.online_pdb_feature_scatter_kernel = str(online_pdb_feature_scatter_kernel)
        # float, gauss27 核标准差(单位: 体素); 仅 scatter_kernel="gauss27" 时生效
        self.online_pdb_feature_sigma_voxel = float(online_pdb_feature_sigma_voxel)
        # bool, gauss27 是否追加 2 维 occupancy 通道; 仅 scatter_kernel="gauss27" 时生效
        self.online_pdb_feature_add_occupancy = bool(online_pdb_feature_add_occupancy)
        # bool, gauss27 是否追加 3 维加权 centroid 通道; 仅 scatter_kernel="gauss27" 时生效
        self.online_pdb_feature_add_centroid = bool(online_pdb_feature_add_centroid)
        self.online_pdb_feature_dim = int(online_pdb_feature_dim)
        self.enable_recycling = bool(enable_recycling)
        self.max_recycles = int(max_recycles)
        self.randomize_recycles = bool(randomize_recycles)
        self.detach_recycle_states = bool(detach_recycle_states)
        self.ffn_type = str(ffn_type)
        
        # 真实原子 desity cube
        self.real_atom_density_cube = bool(real_atom_density_cube)
        self.real_atom_density_share_encoder = bool(real_atom_density_share_encoder)
        self.real_atom_density_cube_size = int(real_atom_density_cube_size)
        self.real_atom_density_combine_mode = str(real_atom_density_combine_mode)
        self.real_atom_density_fusion_mlp_ratio = float(real_atom_density_fusion_mlp_ratio)
        self.real_atom_density_fusion_proj_drop = float(real_atom_density_fusion_proj_drop)












        # -------------------------------------------------------- C——>P——>C 模块的初始化 --------------------------------------------------------
        # ================================================================================================================================================
        # ================================================================================================================================================
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

        self.anchor_class_conditioning_mode = "none"
        self.anchor_class_conditioning_init_std = 0.02
        self.anchor_class_embedding: nn.Embedding | None = None
        self.register_buffer("_anchor_class_ids", torch.empty((0,), dtype=torch.long), persistent=False)
        if anchor_class_conditioning_cfg is not None:
            if self.anchor_sampler is None:
                raise ValueError("anchor_class_conditioning_cfg 启用时必须同时启用 anchor_sampler。")
            mode = str(anchor_class_conditioning_cfg["mode"]) # 只支持 add_embedding
            self.anchor_class_conditioning_mode = mode
            if "init_std" in anchor_class_conditioning_cfg:
                self.anchor_class_conditioning_init_std = float(anchor_class_conditioning_cfg["init_std"])
            candidate_class_ids = tuple(int(class_id) for class_id in self.anchor_sampler.candidate_class_ids)
            self._anchor_class_ids = torch.as_tensor(candidate_class_ids, dtype=torch.long)
            self.anchor_class_embedding = nn.Embedding(len(candidate_class_ids), int(self.point_backbone.atom_feature_dim))
            nn.init.normal_(self.anchor_class_embedding.weight, mean=0.0, std=self.anchor_class_conditioning_init_std)


        # ----------------- 真实原子 density cube encoder(共享/独立) + combine + 接口 Norm ---------------
        self.real_density_cube_encoder = None
        self.real_density_combine = None
        self.interface_norm_real_density_to_point = None
        if self.real_atom_density_cube:
            # int, 真实原子 density 特征维(= point atom_feature_dim)
            real_density_feat_dim = int(self.point_backbone.atom_feature_dim)
            if self.real_atom_density_share_encoder:
                if self.density_cube_encoder is None:
                    raise ValueError("real_atom_density_share_encoder=True 时需要 density_cube_encoder 也在位(即开启 sparse refine/anchor)")
            else:
                self.real_density_cube_encoder = (
                    real_density_cube_cfg
                    if isinstance(real_density_cube_cfg, nn.Module)
                    else instantiate(real_density_cube_cfg)
                )
                if int(self.real_density_cube_encoder.out_dim) != real_density_feat_dim:
                    raise ValueError("real_density_cube_encoder.out_dim 必须等于 point_backbone.atom_feature_dim。")
            # int, real density 融合 MLP 隐藏维(与 voxel 侧同公式: max(main, round((main+cond)*ratio)), 此处 main==cond==feat)
            real_density_hidden_dim = max(
                real_density_feat_dim,
                int(round(float(2 * real_density_feat_dim) * float(self.real_atom_density_fusion_mlp_ratio))),
            )
            # nn.Module, embed<->density 零初始化恒等融合(concat_mlp/film/film_plus/mini_residue)
            self.real_density_combine = FeatureCombine(
                self.real_atom_density_combine_mode,
                real_density_feat_dim,
                real_density_feat_dim,
                real_density_hidden_dim,
                act_cls,
                proj_drop=float(self.real_atom_density_fusion_proj_drop),
            )
            if self.enable_interface_norm:
                self.interface_norm_real_density_to_point = nn.LayerNorm(real_density_feat_dim)
        
        # -------- warpper 同步对象 -------
        # torch.Tensor, (K,) 或 (0,), wrapper 同步过来的 best-F1 阈值缓存
        self.register_buffer("_candidate_p_best_by_class", torch.empty((0,), dtype=torch.float32), persistent=False)
        # torch.Tensor | None, (K,), wrapper 同步过来的 sampling 阈值缓存
        self.register_buffer("_candidate_p_sampling_by_class", torch.empty((0,), dtype=torch.float32), persistent=False)
        self._has_candidate_p_best_by_class = False
        self._has_candidate_p_sampling_by_class = False
        self._candidate_warmup_steps = 0
        self._candidate_global_step = 0
        self._candidate_allow_warmup_fixed_topk = False

        # --------校验 -------
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
        if self.density_cube_encoder is not None and int(self.density_cube_encoder.out_dim) != int(self.point_backbone.atom_feature_dim):
            raise ValueError("density_cube_encoder.out_dim 必须等于 point_backbone.atom_feature_dim。")













        # -------------------------------------------------------- point backbone 到 voxel backbone 的hook融合--------------------------------------------------------
        # ================================================================================================================================================
        # ================================================================================================================================================

        # ------------------ 构建每个 hook 的映射对应 ------------------
        self.point_fusion_items = tuple(
            (str(point_name), voxel_name_str)
            for point_name, voxel_name in (point_fusion_map or {}).items()
            if voxel_name is not None and (voxel_name_str := str(voxel_name).strip()) != ""
        )
        self.point_fusion_modes = tuple(str(mode_name).lower() for mode_name in point_fusion_modes)
        self.sampler_modes = tuple(str(mode_name).lower() for mode_name in sampler_modes)


        self.point_fusion_map = {point_name: voxel_name for point_name, voxel_name in self.point_fusion_items}
        self.sampler_mode_by_point_name = {
            point_name: mode_name
            for (point_name, _), mode_name in zip(self.point_fusion_items, self.sampler_modes)
        }
        self.fusion_mode_by_point_name = {
            point_name: mode_name
            for (point_name, _), mode_name in zip(self.point_fusion_items, self.point_fusion_modes)
        }
        # list, 与 point_fusion_items 等长的 cube 初值列表; None 表示该 hook 非 weighted_cube
        cube_init_list = list(sampler_cube_init) if sampler_cube_init is not None else [None] * len(self.point_fusion_items)
        if len(cube_init_list) != len(self.point_fusion_items):
            raise ValueError("sampler_cube_init 必须与 point_fusion_map / sampler_modes 等长。")
        # nn.ModuleDict, weighted_cube hook 的 per-hook 可学习正偏置(a/b/c, log 域保存) + log 温度; 其余 sampler 不建
        self.cube_weight_params = nn.ModuleDict()
        for (point_name, _), mode_name, init_tuple in zip(self.point_fusion_items, self.sampler_modes, cube_init_list):
            if mode_name == "weighted_cube":
                # float, home/含原子/其他正偏置与温度初值
                a_init, b_init, c_init, d_init = (float(v) for v in tuple(init_tuple))
                self.cube_weight_params[point_name] = CubeWeightingParams(a_init, b_init, c_init, d_init)

        self.voxel_feature_names_to_return = tuple(
            dict.fromkeys(
                tuple(str(feature_name) for feature_name in self.voxel_backbone.return_feature_keys)
                + tuple(voxel_name for _, voxel_name in self.point_fusion_items)
                + (("voxel_final",) if self.anchor_to_candidate is not None else ())
            )
        )
        self.point_feature_names_to_return = (
            tuple(dict.fromkeys([point_name for point_name, _ in self.point_fusion_items] + ["point_feat"]))
            if self.point_backbone is not None
            else ()
        )


        # ------------------ 构建每个 hook 的模块 ------------------
        # nn.ModuleDict, point 变量名 -> voxel-to-point 融合模块
        self.point_fusion_modules = nn.ModuleDict()
        for point_name, voxel_name in self.point_fusion_items:
            fusion_mode = self.fusion_mode_by_point_name[point_name]
            point_channels = int(self.point_backbone.feature_channels_by_name[point_name])
            voxel_channels = int(self.voxel_backbone.feature_channels_by_name[voxel_name])
            fusion_input_dim = point_channels + voxel_channels
            fusion_hidden_dim = max(point_channels, int(round(float(fusion_input_dim) * float(fusion_mlp_ratio))))
            if self.typed_point_cfg.use_separate_fusion:
                # nn.ModuleDict, real/pseudo 两套 voxel-to-point fusion 模块
                self.point_fusion_modules[point_name] = nn.ModuleDict(
                    {
                        "real": FeatureCombine(
                            fusion_mode,
                            point_channels,
                            voxel_channels,
                            fusion_hidden_dim,
                            act_cls,
                            proj_drop=float(fusion_proj_drop),
                        ),
                        "pseudo": FeatureCombine(
                            fusion_mode,
                            point_channels,
                            voxel_channels,
                            fusion_hidden_dim,
                            act_cls,
                            proj_drop=float(fusion_proj_drop),
                        ),
                    }
                )
            else:
                # FeatureCombine, shared voxel-to-point fusion 模块(main=点特征, cond=采样体素特征)
                self.point_fusion_modules[point_name] = FeatureCombine(
                    fusion_mode,
                    point_channels,
                    voxel_channels,
                    fusion_hidden_dim,
                    act_cls,
                    proj_drop=float(fusion_proj_drop),
                )
        # nn.ModuleDict, enable_interface_norm 下 voxel->point 融合两源的 per-变量 LayerNorm; 关时为空走恒等
        self.interface_norm_fusion = nn.ModuleDict()
        # nn.LayerNorm | None, embed head 输出 -> point 真实原子初始特征接口归一化
        self.interface_norm_embed_to_point: nn.LayerNorm | None = None
        # nn.LayerNorm | None, density cube 输出 -> point 伪原子初始特征接口归一化
        self.interface_norm_density_to_point: nn.LayerNorm | None = None
        if self.enable_interface_norm:
            for point_name, voxel_name in self.point_fusion_items:
                # int, 当前 point 变量通道数与对应 voxel 特征通道数
                norm_point_channels = int(self.point_backbone.feature_channels_by_name[point_name])
                norm_voxel_channels = int(self.voxel_backbone.feature_channels_by_name[voxel_name])
                self.interface_norm_fusion[point_name] = nn.ModuleDict(
                    {
                        "point": nn.LayerNorm(norm_point_channels),
                        "voxel": nn.LayerNorm(norm_voxel_channels),
                    }
                )
            if self.embed_head is not None and self.embed_head.has_point_output:
                self.interface_norm_embed_to_point = nn.LayerNorm(int(self.point_backbone.atom_feature_dim))
            if self.density_cube_encoder is not None:
                self.interface_norm_density_to_point = nn.LayerNorm(int(self.point_backbone.atom_feature_dim))


        # --------------------- 校验 ---------------------
        available_voxel_feature_names = tuple(self.voxel_backbone.feature_channels_by_name.keys())
        available_point_feature_names = (
            tuple(self.point_backbone.feature_channels_by_name.keys())
            if self.point_backbone is not None
            else ()
        )
        for point_name, voxel_name in self.point_fusion_items:
            if point_name not in available_point_feature_names:
                raise KeyError(f"Unknown point fusion name={point_name}, available={available_point_feature_names}")
            if voxel_name not in available_voxel_feature_names:
                raise KeyError(f"Unknown voxel fusion name={voxel_name}, available={available_voxel_feature_names}")

        if not (len(self.point_fusion_items) == len(self.point_fusion_modes) == len(self.sampler_modes)):
            raise ValueError("point_fusion_map、point_fusion_modes 与 sampler_modes 的长度必须一致。")
        supported_sampler_modes = {"trilinear", "nearest", "weighted_cube", "cube_mean"}
        unsupported_sampler_modes = [mode_name for mode_name in self.sampler_modes if mode_name not in supported_sampler_modes]
        if unsupported_sampler_modes:
            raise ValueError(f"Unsupported sampler_modes={unsupported_sampler_modes}, supported={sorted(supported_sampler_modes)}")
        supported_point_fusion_modes = {"concat_mlp", "film", "film_plus", "mini_residue"}
        unsupported_fusion_modes = [
            mode_name for mode_name in self.point_fusion_modes if mode_name not in supported_point_fusion_modes
        ]
        if unsupported_fusion_modes:
            raise ValueError(
                f"Unsupported point_fusion_modes={unsupported_fusion_modes}, "
                f"supported={sorted(supported_point_fusion_modes)}"
            )
        if int(fusion_cube_chunk_size) <= 0:
            raise ValueError("fusion_cube_chunk_size must be > 0")

        self.fusion_cube_chunk_size = int(fusion_cube_chunk_size)













        # -------------------------------------------------------- atom head 和受体原子最终分类头 --------------------------------------------------------
        # ================================================================================================================================================
        # ================================================================================================================================================
        self.enable_atom_head = bool(enable_atom_head)
        if self.enable_atom_head:
            self.atom_head = Stage1AtomHead(
                point_channels=int(self.point_backbone.out_channels),
                hidden_dim=int(atom_head_hidden_dim),
                atom_logit_dim=int(atom_logit_dim),
                pseudo_ligand_logit_dim=self.pseudo_ligand_logit_dim,
                act_layer=act_cls,
                interaction_radius=float(atom_head_interaction_radius),
                interaction_max_neighbors=int(atom_head_interaction_max_neighbors),
                interaction_num_heads=int(atom_head_interaction_num_heads),
                interaction_detach_source_feat=bool(atom_head_interaction_detach_source_feat),
                prior_prob=self._eff_atom_prior_prob,
                prior_probs=self._eff_atom_prior_probs,
                prior_prob_point_ligand=self._eff_prior_point_ligand,
            )
        else:
            self.atom_head = None

        # -------------------------------------------------------- sparse_refine_head --------------------------------------------------------
        # ================================================================================================================================================
        # ================================================================================================================================================
        self.sparse_refine_head: nn.Module | None = None
        if pending_sparse_refine_head_cfg is not None:
            injected_head_kwargs = {
                "logit_dim": int(self.voxel_backbone.voxel_ligand_logit_dim),
                "C_voxel_backbone_dim": int(self.voxel_backbone.feature_channels_by_name["voxel_final"]),
                "P_final_point_dim": int(self.point_backbone.feature_channels_by_name["point_feat"]),
                "P_after_interaction_dim": int(self.point_backbone.feature_channels_by_name["point_feat"]),
                "P_voxel_backbone_dim": int(self.voxel_backbone.feature_channels_by_name["voxel_final"]),
                "candidate_class_ids": tuple(int(value) for value in self.candidate_set_builder.candidate_class_ids),
                "enable_interface_norm": self.enable_interface_norm,
                # str, refine 残差/直连模式; 归属 detach_residual 组, 注入覆盖(已从 sparse_refine_head_cfg.yaml 移除)
                "mode": self.sparse_refine_residual_mode,
                # float | None, refine 头单通道 sigmoid 先验; 仅 direct 模式生效, residual 模式头内零初始化跳过; 主开关关时为 None
                "prior_prob": self._eff_prior_sparse_refine,
            }
            if isinstance(pending_sparse_refine_head_cfg, nn.Module):
                self.sparse_refine_head = pending_sparse_refine_head_cfg
            else:
                # dict[str, bool], 复制 sparse_refine_head_cfg.inputs 其余开关, 再注入 detach_residual 持有的 use_C_voxel_logits
                head_inputs = {str(key): bool(value) for key, value in pending_sparse_refine_head_cfg["inputs"].items()}
                head_inputs["use_C_voxel_logits"] = self.sparse_refine_use_C_voxel_logits
                self.sparse_refine_head = instantiate(
                    pending_sparse_refine_head_cfg, inputs=head_inputs, **injected_head_kwargs
                )

        if self.refine_receptor_from_voxel:
            if int(atom_logit_dim) != int(self.voxel_backbone.voxel_aux_logit_dim):
                raise ValueError("refine_receptor_from_voxel 要求 atom_logit_dim == voxel_aux_logit_dim。")
            if self.atom_head is not None:
                # nn.Linear, real_atom_head 末层; 零初始化后初值 = voxel_aux home 体素 base
                last_linear = next(module for module in reversed(self.atom_head.real_atom_head) if isinstance(module, nn.Linear))
                nn.init.zeros_(last_linear.weight)
                nn.init.zeros_(last_linear.bias)










    # ------------------------------------------------------------------------ 纯粹工具函数 ------------------------------------------------------------------------
    # ================================================================================================================================================
    # ================================================================================================================================================
    @staticmethod
    def _apply_point_feat_detach_routing(
        point_feat: torch.Tensor,
        pseudo_mask: torch.Tensor | None,
        detach_real: bool,
        detach_pseudo: bool,
    ) -> torch.Tensor:
        """
        按真实/伪原子槽位对 point backbone 输出特征做选择性 detach. 

        输入参数:
            - point_feat: torch.Tensor, (N_all, C_point), point backbone 输出特征(mixed 或 real-only)
            - pseudo_mask: torch.Tensor | None, (N_all,), bool, True 表示 P anchor; None 表示 real-only 路径
            - detach_real: bool, 是否 detach 真实原子槽位
            - detach_pseudo: bool, 是否 detach 伪原子槽位

        输出:
            - routed_feat: torch.Tensor, (N_all, C_point), 按开关选择性 detach 后的特征
        """
        if detach_real and detach_pseudo:
            return point_feat.detach()
        if detach_real:
            # torch.Tensor, (N_all,), bool, True 表示真实原子槽位
            real_mask = (
                torch.ones(point_feat.shape[0], dtype=torch.bool, device=point_feat.device)
                if pseudo_mask is None
                else ~pseudo_mask
            )
            return torch.where(real_mask[:, None], point_feat.detach(), point_feat)
        if detach_pseudo and pseudo_mask is not None:
            return torch.where(pseudo_mask[:, None], point_feat.detach(), point_feat)
        return point_feat

    @staticmethod
    def _apply_voxel_point_fusion_by_type(
        point_feat: torch.Tensor,
        voxel_feat: torch.Tensor,
        pseudo_mask: torch.Tensor | None,
        real_module: nn.Module,
        pseudo_module: nn.Module,
    ) -> torch.Tensor:
        """
        按 pseudo_mask 把 point/voxel 两路特征分成 real/pseudo 两组, 分别送入对应 real_module、pseudo_module(实际上都是 FeatureCombine)后按原顺序合并. 

        输入参数:
            - point_feat: torch.Tensor, (N_all, C_point), mixed 或 real-only 点特征
            - voxel_feat: torch.Tensor, (N_all, C_voxel), mixed 或 real-only 体素条件特征
            - pseudo_mask: torch.Tensor | None, (N_all,), True 表示 P anchor; None 表示全部走 real 分支
            - real_module: nn.Module, real 分支 FeatureCombine
            - pseudo_module: nn.Module, pseudo 分支 FeatureCombine

        输出:
            - out: torch.Tensor, (N_all, C_point), 按输入点顺序排列的融合特征
        """
        if int(point_feat.shape[0]) == 0:
            return real_module(point_feat, voxel_feat)
        pseudo_mask = validate_pseudo_mask(
            pseudo_mask,
            int(point_feat.shape[0]),
            name="VolumePointStage1Model._apply_voxel_point_fusion_by_type",
        )
        if pseudo_mask is None or not bool(pseudo_mask.any().item()):
            return real_module(point_feat, voxel_feat)
        if bool(pseudo_mask.all().item()):
            return pseudo_module(point_feat, voxel_feat)

        # torch.Tensor, (N_real, C_point/C_voxel), real 子集; (N_pseudo, ...), pseudo 子集
        real_y = real_module(point_feat[~pseudo_mask], voxel_feat[~pseudo_mask])
        pseudo_y = pseudo_module(point_feat[pseudo_mask], voxel_feat[pseudo_mask])
        return merge_type_aware_tensor_outputs(real_y, pseudo_y, pseudo_mask)

    @staticmethod
    def _gather_voxel_aux_logit_at_atom_home_voxel(
        voxel_logits_aux: torch.Tensor,
        atom_coord_local_voxel: torch.Tensor,
        atom_batch_index: torch.Tensor,
        box_shape_zyx: torch.Tensor,
    ) -> torch.Tensor:
        """
        读取每个真实原子 home 体素处的 voxel aux logits, 作为 refine_receptor 残差 base, 并进行detach. 

        输入参数:
            - voxel_logits_aux: torch.Tensor, (B, C_aux, D, H, W), 体素辅助(receptor)预测 logits
            - atom_coord_local_voxel: torch.Tensor, (N, 3), BOX-local 连续 voxel 坐标 XYZ, corner 语义
            - atom_batch_index: torch.Tensor, (N,), 每个原子所属 BOX 索引
            - box_shape_zyx: torch.Tensor, (B, 3), 每个 BOX 离散 voxel 网格尺寸 ZYX

        输出:
            - base: torch.Tensor, (N, C_aux), home 体素处的 voxel aux logits, 已 detach
        """
        # torch.Tensor[int64], (N,3), BOX-local 连续 voxel XYZ floor 后得到的离散 home voxel-index XYZ
        idx_xyz = torch.floor(atom_coord_local_voxel).to(torch.long)
        # torch.Tensor[int64], (N,3), 离散 home voxel-index 转换后的 ZYX 轴序
        idx_zyx = idx_xyz[:, [2, 1, 0]].clamp(min=0)
        # torch.Tensor[int64], (N,3), 每个 atom 所属 BOX 的离散 voxel 网格尺寸 ZYX
        shape_zyx = box_shape_zyx.to(idx_zyx.device)[atom_batch_index]
        idx_zyx = torch.minimum(idx_zyx, shape_zyx - 1)
        # torch.Tensor, (N, C_aux), home 体素处采样到的 aux logits
        base = voxel_logits_aux[atom_batch_index, :, idx_zyx[:, 0], idx_zyx[:, 1], idx_zyx[:, 2]]
        return base.detach()

    @staticmethod
    def _init_atom_logit_head_prior_bias(
        last_linear: nn.Module,
        logit_dim: int,
        prior_prob: float | None,
        prior_probs: Sequence[float] | None,
    ) -> None:
        """
        用先验概率初始化 atom logit head 末层 Linear 的 bias, 与后置头 prior 初始化保持一致. 

        输入参数:
            - last_linear: nn.Module, atom logit head 末层, 必须是带 bias 的 nn.Linear
            - logit_dim: int, logits 输出通道数
            - prior_prob: float | None, 单通道 sigmoid 正类先验概率; 与 prior_probs 互斥
            - prior_probs: Sequence[float] | None, 多通道 softmax 类别先验概率

        输出:
            - None, 原地更新 last_linear.bias
        """
        if prior_prob is not None and prior_probs is not None:
            raise ValueError("prior_prob 和 prior_probs 不能同时配置。")
        if prior_probs is not None:
            Stage1AtomHead._init_linear_multiclass_prior_bias(last_linear, int(logit_dim), prior_probs)  # 直接复用
        elif prior_prob is not None:
            if int(logit_dim) != 1:
                raise ValueError("多通道前置头请使用 prior_probs，不要使用单通道 prior_prob。")
            # float, sigmoid 正类先验对应的输出 bias
            bias_val = -math.log((1.0 - float(prior_prob)) / float(prior_prob))
            nn.init.constant_(last_linear.bias, bias_val)

    def _condition_anchor_pseudo_feat(
        self,
        pseudo_feat: torch.Tensor,
        anchor_class: torch.Tensor,
    ) -> torch.Tensor:
        """
        按 P anchor 来源类别对初始 pseudo feature 做条件化(conditional embedding): pseudo_feat + self.anchor_class_embedding(local_index)

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
        从累计 offset 算出每个 BOX 的点数. 

        输入参数:
            - atom_offsets: torch.Tensor, (B,), long, PTV3 风格累计 offset(结束索引)

        输出:
            - atom_counts: torch.Tensor, (B,), long, 每个 BOX 的点数
        """
        if atom_offsets.numel() == 0:
            return atom_offsets.new_zeros((0,), dtype=torch.long)
        atom_counts = atom_offsets.clone()
        atom_counts[1:] = atom_counts[1:] - atom_counts[:-1]
        return atom_counts

    @staticmethod
    def _canonicalize_stage1_batch(batch: dict[str, Any]) -> dict[str, Any]:
        """
        把 AdaLigand 外部字段适配为现有模型内部字段. 

        外部统一使用 ``density_input`` 与 ``atom_offsets[B+1]``; 现有 PTV3 内核仍
        使用 ``voxel_grid`` 与 ``B`` 个累计结束偏移. 适配只创建浅拷贝, 不改写
        Dataset/Collator 对外契约. 

        输入参数:
            - batch: dict[str,Any], 外部 Stage1 batch; `density_input` 为 `(B,C,D,H,W)` voxel 网格, `atom_offsets` 为外部 `(B+1,)` ragged 边界

        输出:
            - result: dict[str,Any], 浅拷贝后的内部 batch; `voxel_grid` 与 `atom_offsets` 分别转换为内部别名和 `(B,)` 结束偏移
        """

        result = {**batch}
        if "density_input" in result:
            result["voxel_grid"] = result["density_input"]
        elif "voxel_grid" not in result:
            raise KeyError("Stage1 batch 必须包含 density_input。")
        if "atom_feat" not in result:
            return result
        if "atom_offsets" not in result:
            raise KeyError("Find batch 必须包含 atom_offsets。")
        batch_size = int(result["voxel_grid"].shape[0])
        offsets = result["atom_offsets"].long()
        if int(offsets.numel()) == batch_size + 1:
            if int(offsets[0].item()) != 0:
                raise ValueError("外部 atom_offsets[B+1] 的首项必须为 0。")
            result["atom_offsets"] = offsets[1:]
        elif int(offsets.numel()) == batch_size:
            result["atom_offsets"] = offsets
        else:
            raise ValueError(
                f"atom_offsets 长度必须为 B+1(外部)或 B(PTV3 内部)，实际 B={batch_size}, "
                f"len={int(offsets.numel())}。"
            )
        if "atom_counts" not in result:
            result["atom_counts"] = VolumePointStage1Model._counts_from_offsets(result["atom_offsets"])
        return result

    def set_input_channels(self, in_channels: int) -> None:
        """
        将体素输入通道数设置请求透传给 voxel backbone. 

        输入参数:
            - in_channels: int, 数据集 voxel_grid 通道数, 不含 embed/online scatter 追加通道

        输出:
            - None, 原地调用 voxel_backbone 与 density_cube_encoder 的 set_input_channels
        """
        raw_in_channels = int(in_channels)
        actual_in_channels = raw_in_channels
        if self.embed_head is not None and self.embed_head.has_voxel_output:
            if not bool(getattr(self.embed_head, "voxel_embed_as_tune", False)):
                extra = int(self.embed_head.embed_voxel_out_channels)
                if self.embed_head.add_occupancy_channels:
                    extra += 2
                actual_in_channels += extra
        elif self.online_pdb_feature:
            actual_in_channels += self._online_pdb_voxel_channels()
        # 调用它们内部的方法, 根据 in_channels 重新初始化某些层
        if hasattr(self.voxel_backbone, "set_input_channels"):
            self.voxel_backbone.set_input_channels(actual_in_channels)  
        if self.density_cube_encoder is not None and hasattr(self.density_cube_encoder, "set_input_channels"):
            self.density_cube_encoder.set_input_channels(raw_in_channels)
        if self.real_density_cube_encoder is not None and hasattr(self.real_density_cube_encoder, "set_input_channels"):
            self.real_density_cube_encoder.set_input_channels(raw_in_channels)

    def _online_pdb_voxel_channels(self) -> int:
        """
        计算 online pdb_feature scatter 追加到 voxel backbone 的通道数. 

        输出:
            - channels: int, raw atom feature 通道数加上当前 scatter 核会追加的辅助通道数
        """
        channels = int(self.online_pdb_feature_dim)
        if self.online_pdb_feature_scatter_kernel == "gauss27":
            if self.online_pdb_feature_add_occupancy:
                channels += 2
            if self.online_pdb_feature_add_centroid:
                channels += 3
        return channels












    # ---------------------------------------------------------- 体素———>点 的特征融合逻辑 ----------------------------------------------------------
    # ================================================================================================================================================
    # ================================================================================================================================================

    # ----------------- real atom 的(embed head特征, density box 特征)的调制 -----------------
    @staticmethod
    def _atom_home_voxel_zyx(
        atom_coord_local_voxel: torch.Tensor,
        atom_batch_index: torch.Tensor,
        box_shape_zyx: torch.Tensor,
    ) -> torch.Tensor:
        """
        计算每个原子 home 体素离散下标(z/y/x). 

        输入参数:
            - atom_coord_local_voxel: torch.Tensor, (N,3), BOX-local 连续 voxel 坐标 XYZ, corner 语义
            - atom_batch_index: torch.Tensor, (N,), 每个原子所属 BOX 索引
            - box_shape_zyx: torch.Tensor, (B,3), 每个 BOX 离散 voxel 网格尺寸 ZYX

        输出:
            - idx_zyx: torch.Tensor, (N, 3), long, home 体素离散下标(z/y/x), 已 clamp 到合法范围
        """
        # torch.Tensor, (N, 3), floor 后下标(x/y/z)
        idx_xyz = torch.floor(atom_coord_local_voxel).to(torch.long)
        # torch.Tensor, (N, 3), 转 z/y/x 并下界裁剪
        idx_zyx = idx_xyz[:, [2, 1, 0]].clamp(min=0)
        # torch.Tensor, (N, 3), 每个原子所属 BOX 的体素尺寸(z/y/x)
        shape_zyx = box_shape_zyx.to(idx_zyx.device)[atom_batch_index]
        return torch.minimum(idx_zyx, shape_zyx - 1)

    def _apply_real_atom_density_to_atom_feat(self, batch: dict[str, Any]) -> dict[str, Any]:
        """
        为 real atom 编码属于它的 density box, 再对进行(可选)norm和调制: (embed head特征, density box 特征). 它是零初始化, 开局 = embed head特征. 

        过程:
            - 在 _run_embed_head_once 之后无条件调用(只要 self.real_atom_density_cube), density box 特征和本融合都是跨 recycle 复用的. 
            - 编码 density box(self.real_atom_density_cube 去 forward) ————> Norm(self.interface_norm_real_density_to_point) ————> self.real_density_combine

        输入参数:
            - batch: dict[str, Any], real-only canonical batch(含裁剪后的 atom_coord_local_voxel / atom_batch_index / voxel_grid)

        输出:
            - batch: dict[str, Any], atom_feat 已叠加 density combine 的浅拷贝; self.real_atom_density_cube = None 时原样返回 batch
        """
        if not self.real_atom_density_cube:
            return batch
        batch = {**batch}
        # nn.Module, 共享或独立 density encoder
        encoder = self.density_cube_encoder if self.real_atom_density_share_encoder else self.real_density_cube_encoder
        # torch.Tensor[int64], (N_real,3), 全分辨率 BOX-local 离散 home voxel-index ZYX
        home_zyx = self._atom_home_voxel_zyx(
            batch["atom_coord_local_voxel"], batch["atom_batch_index"], batch["box_shape_zyx"]
        )
        # torch.Tensor, (N_real, F), 真实原子 density cube 特征
        real_density_feat = encoder(
            batch["voxel_grid"], home_zyx, batch["atom_batch_index"], cube_size=self.real_atom_density_cube_size
        )
        if self.interface_norm_real_density_to_point is not None:
            real_density_feat = self.interface_norm_real_density_to_point(real_density_feat)
        # batch["atom_feat"] 此处已是 interface_norm_embed_to_point 之后的 embed 特征(若启用)
        batch["atom_feat"] = self.real_density_combine(batch["atom_feat"], real_density_feat)
        return batch
    

    # ----------------- point backbone 与 voxel backbone 的 hook 融合 -----------------
    # 三线性插值模式
    def _sample_voxel_feature_trilinear(
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
        对 batch 内所有点采样指定体素特征. 

        输入参数:
            - voxel_feat: torch.Tensor, (B, C, D, H, W), 体素特征图
            - point_coord_centered_world: torch.Tensor, (N,3), centered 连续世界坐标 XYZ, 单位 Å
            - point_batch_index: torch.Tensor, (N,), 当前点所属 BOX 索引
            - voxel_size_world: torch.Tensor, (B,3), 世界坐标 XYZ 各轴的 voxel 间距, 单位 Å/voxel
            - box_shape_zyx: torch.Tensor, (B,3), 每个 BOX 离散 voxel 网格尺寸 ZYX
            - fusion_mode: str, 仅兼容无意义
            - sampler_mode: str, voxel 采样模式, 取值 trilinear / nearest

        输出:
            - sampled_feat: torch.Tensor, (N, C), 采样后的点级体素特征
        """
        point_count = int(point_coord_centered_world.shape[0])
        if point_count == 0:
            return voxel_feat.new_empty((0, int(voxel_feat.shape[1])))
        _ = fusion_mode
        # str, grid_sample 插值模式
        grid_sample_mode = "bilinear" if sampler_mode == "trilinear" else "nearest"
        device = voxel_feat.device
        batch_size = int(box_shape_zyx.shape[0])
        point_batch_index = point_batch_index.to(device=device, dtype=torch.long)
        # torch.Tensor, (N, 3), 每点全分辨率 box 尺寸(x/y/z)与 voxel 世界尺寸(x/y/z)
        box_shape_xyz = box_shape_zyx.to(device=device, dtype=point_coord_centered_world.dtype)[point_batch_index][:, [2, 1, 0]]
        voxel_size_xyz = voxel_size_world.to(device=device, dtype=point_coord_centered_world.dtype)[point_batch_index]
        # torch.Tensor, (N,3), 由 centered 连续世界 XYZ 换算得到的 BOX-local 连续 voxel XYZ corner 坐标
        point_local_xyz = point_coord_centered_world / voxel_size_xyz + 0.5 * box_shape_xyz
        # torch.Tensor, (N, 3), grid_sample [-1,1] 归一化坐标(align_corners=True)
        grid_xyz = (2.0 * (point_local_xyz - 0.5) / torch.clamp(box_shape_xyz - 1.0, min=1.0)) - 1.0
        # torch.Tensor, (B,), 每个 BOX 点数; int, 最大点数
        counts = torch.bincount(point_batch_index, minlength=batch_size)
        n_max = int(counts.max())
        # torch.Tensor, (B,), 每个 BOX 中的点在按 box 稳定排序后的起始 rank
        box_start = torch.zeros(batch_size, dtype=torch.long, device=device)
        if batch_size > 1:
            box_start[1:] = torch.cumsum(counts, dim=0)[:-1]
        # torch.Tensor, (N,), 每点在其 BOX 内的 0-based 次序(保持原点顺序)
        sort_idx = torch.argsort(point_batch_index, stable=True)
        # torch.Tensor, (N,), 每个点按照先后出现的顺序, 在当前 BOX 中的次序(0...N_this_box - 1)
        pos_in_box = torch.empty(point_count, dtype=torch.long, device=device)
        pos_in_box[sort_idx] = torch.arange(point_count, device=device) - box_start[point_batch_index[sort_idx]]
        # torch.Tensor, (B, n_max, 3), 点按照"本BOX出现的顺序"所对应的归一化坐标; 空槽位填 0 就好, 在最后一步不会被索引回去
        grid_padded = voxel_feat.new_zeros((batch_size, n_max, 3))
        grid_padded[point_batch_index, pos_in_box] = grid_xyz.to(grid_padded.dtype)
        # torch.Tensor, (B, C, n_max, 1, 1), 单次 grid_sample
        sampled = F.grid_sample(
            input=voxel_feat,
            grid=grid_padded.view(batch_size, n_max, 1, 1, 3),
            mode=grid_sample_mode,
            padding_mode="zeros",
            align_corners=True,
        )
        # torch.Tensor, (B, n_max, C), 整理维度后按 (box, pos) 回收到原点顺序
        sampled = sampled.squeeze(-1).squeeze(-1).permute(0, 2, 1)
        # (N, C), 采样后的点级体素特征
        return sampled[point_batch_index, pos_in_box].contiguous()

    # 3^3 BOX 加权模式
    def _sample_voxel_feature_cube(
        self,
        voxel_feat: torch.Tensor,
        point_like: Any,
        batch: dict[str, Any],
        feature_name: str,
        pseudo_mask: torch.Tensor | None,
        fusion_mode: str,
        sampler_mode: str,
    ) -> torch.Tensor:
        """
        以每个点 home 体素为中心抽取 3^3 邻域并做加权(weighted_cube)或均值(cube_mean)池化采样. 

        输入参数:
            - voxel_feat: torch.Tensor, (B, C, D_l, H_l, W_l), 当前 hook 这一级体素特征(空间尺寸可小于全分辨率)
            - point_like: Any, 当前 Point 对象, `point_like.coord` 为 centered 连续世界坐标 XYZ, `point_like.batch` 为 BOX 索引
            - batch: dict[str, Any], 提供世界坐标 XYZ voxel 间距 `voxel_size_world(B,3)` 与离散网格尺寸 `box_shape_zyx(B,3)`
            - feature_name: str, 当前 point 变量名, 用于取 weighted_cube 的 per-hook 参数
            - pseudo_mask: torch.Tensor | None, (N,), True=P anchor; None=real-only(全部计入占据)
            - fusion_mode: str, 仅兼容无意义
            - sampler_mode: str, weighted_cube / cube_mean

        输出:
            - sampled_feat: torch.Tensor, (N, C), 加权池化采样得到的点级体素特征

        Note:
            - 权重 = softmax_有效邻居( {a|b|c}(按 home/含原子/其他) - dist^2 / d ); cube_mean 取均匀权重. 
            - 含原子的体素由当前 hook 这一级池化后的真实原子(point_like.coord[~pseudo_mask]) scatter 得到. 
        """
        _ = fusion_mode
        # int, 点数与体素通道数
        num_points = int(point_like.coord.shape[0])
        channels = int(voxel_feat.shape[1])
        if num_points == 0:
            return voxel_feat.new_empty((0, channels))
        # torch.Tensor, (N,3), centered 连续世界坐标 XYZ; torch.Tensor, (N,), 所属 BOX 索引
        point_coord = point_like.coord
        point_batch = point_like.batch.to(torch.long)
        device = voxel_feat.device

        # int, 该级 voxel 特征空间尺寸(z/y/x)
        dim_z, dim_y, dim_x = int(voxel_feat.shape[2]), int(voxel_feat.shape[3]), int(voxel_feat.shape[4])
        # torch.Tensor, (1, 3), 该级 grid 尺寸(x/y/z)
        level_shape_xyz = torch.tensor([dim_x, dim_y, dim_z], device=device, dtype=point_coord.dtype)[None, :]
        # torch.Tensor, (3,), long, 该级 grid 尺寸(z/y/x)
        level_shape_zyx = torch.tensor([dim_z, dim_y, dim_x], device=device, dtype=torch.long)
        # torch.Tensor, (N,3), 每点所属 BOX 的离散全分辨率 voxel 网格尺寸 XYZ
        box_shape_xyz = batch["box_shape_zyx"].to(device=device, dtype=point_coord.dtype)[point_batch][:, [2, 1, 0]]
        # torch.Tensor, (N, 3), 每个点 voxel 世界尺寸(x/y/z)
        voxel_size_xyz = batch["voxel_size_world"].to(device=device, dtype=point_coord.dtype)[point_batch]

        # torch.Tensor, (N,3), centered 连续世界 XYZ -> BOX-local 连续 voxel XYZ corner 坐标
        p_full_xyz = point_coord / voxel_size_xyz + 0.5 * box_shape_xyz
        # torch.Tensor, (N,3), 按层级尺寸缩放后的连续 voxel XYZ corner 坐标
        p_level_xyz = p_full_xyz * (level_shape_xyz / box_shape_xyz)
        # torch.Tensor[int64], (N,3), 当前层级离散 home voxel-index XYZ
        home_xyz = torch.floor(p_level_xyz).to(torch.long)
        # torch.Tensor, (N, 3), home 下标转 z/y/x 并裁剪到合法范围
        home_zyx = torch.minimum(torch.clamp(home_xyz[:, [2, 1, 0]], min=0), level_shape_zyx[None, :] - 1)
        # torch.Tensor, (N, 3), 点内小数偏移(x/y/z), 取值 0~1
        frac_zyx = (p_level_xyz - home_xyz.to(point_coord.dtype))[:, [2, 1, 0]]


        # torch.Tensor, (B, D_l, H_l, W_l), bool, 含真实原子的体素占据图
        if pseudo_mask is None:
            real_home_zyx, real_batch = home_zyx, point_batch
        else:
            real_sel = ~pseudo_mask
            real_home_zyx, real_batch = home_zyx[real_sel], point_batch[real_sel]

        # occupancy: (B, D_i=dim_z, H_I=dim_y, W_I=dim_x), 1 代表"这个位置含有原子"
        occupancy = torch.zeros((int(voxel_feat.shape[0]), dim_z, dim_y, dim_x), dtype=point_coord.dtype, device=device)
        if int(real_home_zyx.shape[0]) > 0:
            occupancy[real_batch, real_home_zyx[:, 0], real_home_zyx[:, 1], real_home_zyx[:, 2]] = 1.0
        # torch.Tensor, (3,), 单轴邻居偏移 -1/0/1
        offsets = torch.arange(3, device=device, dtype=point_coord.dtype) - 1.0
        # list[torch.Tensor], 每块 (N_chunk, C), 避免一次性展开 4000*B 个点的 3^3 邻域占用过高显存. 
        sampled_parts: list[torch.Tensor] = []
        chunk_size = min(int(self.fusion_cube_chunk_size), num_points)
        for chunk_start in range(0, num_points, chunk_size):
            chunk_end = min(chunk_start + chunk_size, num_points)
            chunk_points = chunk_end - chunk_start
            # torch.Tensor, (N_chunk, 3), 当前块 home 体素 z/y/x 下标
            home_zyx_chunk = home_zyx[chunk_start:chunk_end]
            # torch.Tensor, (N_chunk,), 当前块每点所属 BOX 索引
            point_batch_chunk = point_batch[chunk_start:chunk_end]
            # torch.Tensor, (N_chunk, 3), 当前块点内小数偏移 z/y/x, 取值 0~1
            frac_zyx_chunk = frac_zyx[chunk_start:chunk_end]
            # torch.Tensor, (N_chunk, C, 3,3,3) 与 (N_chunk, 3,3,3), 邻域特征与越界掩码
            cube, valid_mask = gather_voxel_cube(
                voxel_feat,
                home_zyx_chunk,
                point_batch_chunk,
                cube_size=3,
                zero_fill=False,
            )
            # torch.Tensor, (N_chunk, 1, 3,3,3), 代表邻域有无原子
            occ_cube, _ = gather_voxel_cube(
                occupancy[:, None],
                home_zyx_chunk,
                point_batch_chunk,
                cube_size=3,
                zero_fill=False,
            )
            # torch.Tensor, (N_chunk, 3,3,3), bool, 邻居是否含原子
            occ_neighbors = occ_cube[:, 0] > 0.5
            # category: torch.Tensor, (N_chunk,3,3,3), long, 邻居类别 0=home / 1=含原子 / 2=其他
            category = torch.where(
                occ_neighbors,
                torch.ones_like(occ_neighbors, dtype=torch.long),
                torch.full_like(occ_neighbors, 2, dtype=torch.long),
            )
            category[:, 1, 1, 1] = 0
            # torch.Tensor, (N_chunk,3,3,3), 点到最近3^3个体素中心的距离平方(voxel 单位)
            dist_sq = (
                ((frac_zyx_chunk[:, 0, None, None, None] - 0.5) - offsets[None, :, None, None]) ** 2
                + ((frac_zyx_chunk[:, 1, None, None, None] - 0.5) - offsets[None, None, :, None]) ** 2
                + ((frac_zyx_chunk[:, 2, None, None, None] - 0.5) - offsets[None, None, None, :]) ** 2
            )

            # torch.Tensor, (N_chunk,3,3,3), softmax 前 logit
            if sampler_mode == "weighted_cube":
                logit = self.cube_weight_params[feature_name](category, dist_sq)
            else:
                logit = torch.zeros_like(dist_sq)
            # torch.Tensor, (N_chunk, 3,3,3), 越界邻居 logit 置 -inf, 使其 softmax 权重为 0
            logit = logit.masked_fill(~valid_mask, float("-inf"))
            # torch.Tensor, (N_chunk, 27), 3^3 邻居上的归一化权重(仅有效邻居非 0)
            weights = torch.softmax(logit.reshape(chunk_points, -1), dim=-1).to(cube.dtype)
            # torch.Tensor, (N_chunk, C), 邻域特征按权重加权求和, 得当前块每点采样特征
            sampled_parts.append((cube.reshape(chunk_points, channels, -1) * weights[:, None, :]).sum(dim=-1))
        return torch.cat(sampled_parts, dim=0).to(dtype=voxel_feat.dtype)

    # 钩子
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
        对点云分支中名为 feature_name 的变量执行 voxel-to-point 融合. 

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
        # torch.Tensor | None, (N,), bool, 当前 point_like 分辨率的 P anchor 掩码; 上移到采样之前供类型路由 / cube 占据使用
        pseudo_mask = point_like.get("pseudo_mask", None) if hasattr(point_like, "get") else None
        pseudo_mask = validate_pseudo_mask(
            pseudo_mask,
            int(point_like.coord.shape[0]),
            name="VolumePointStage1Model._fuse_point_variable",
        )
        # torch.Tensor, (N, C_voxel), 当前点集采样得到的体素特征
        if sampler_mode in ("weighted_cube", "cube_mean"):
            sampled_voxel_feat = self._sample_voxel_feature_cube(
                voxel_feat=voxel_output_dict["voxel_features"][voxel_name],
                point_like=point_like,
                batch=batch,
                feature_name=feature_name,
                pseudo_mask=pseudo_mask,
                fusion_mode=fusion_mode,
                sampler_mode=sampler_mode,
            )
        else:
            sampled_voxel_feat = self._sample_voxel_feature_trilinear(
                voxel_feat=voxel_output_dict["voxel_features"][voxel_name],
                point_coord_centered_world=point_like.coord,
                point_batch_index=point_like.batch,
                voxel_size_world=batch["voxel_size_world"],
                box_shape_zyx=batch["box_shape_zyx"],
                fusion_mode=fusion_mode,
                sampler_mode=sampler_mode,
            )
        # (N, C_voxel), voxel->point 采样特征按真实/伪原子类型选择性 detach
        sampled_voxel_feat = self._apply_point_feat_detach_routing(
            sampled_voxel_feat,
            pseudo_mask,
            self.detach_voxel_feat_into_real_point,
            self.detach_voxel_feat_into_pseudo_point,
        )
        # torch.Tensor, (N, C_point), 融合用的点源特征; 接口 Norm 开时先各自 LayerNorm
        point_feat_for_fusion = point_like.feat
        if self.enable_interface_norm:
            # nn.ModuleDict, 当前 point 变量的两源接口 LayerNorm
            fusion_norm = self.interface_norm_fusion[feature_name]
            point_feat_for_fusion = fusion_norm["point"](point_feat_for_fusion)
            sampled_voxel_feat = fusion_norm["voxel"](sampled_voxel_feat)
        # nn.Module, 当前 point 变量对应的 fusion module
        fusion_module = self.point_fusion_modules[feature_name]
        if self.typed_point_cfg.use_separate_fusion:
            if require_pseudo_mask and pseudo_mask is None:
                raise RuntimeError("typed fusion 的 mixed point_like 必须携带 pseudo_mask。")
            point_like.feat = self._apply_voxel_point_fusion_by_type(
                point_feat_for_fusion,
                sampled_voxel_feat,
                pseudo_mask,
                fusion_module["real"],
                fusion_module["pseudo"],
            )
        else:
            point_like.feat = fusion_module(point_feat_for_fusion, sampled_voxel_feat)
        return point_like











    # -------------------------------------------------------------- 各组件正式 forward --------------------------------------------------------------
    # ================================================================================================================================================
    # ================================================================================================================================================
    def _run_embed_head_once(self, batch: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any] | None]:
        """
        在 recycle 循环前运行 real-only embed head 并同步裁剪后的 atom 字段. 

        输入参数:
            - batch: dict[str, Any], collate 后的 real-only batch

        输出:
            - batch: dict[str, Any], real-only canonical batch, 若 embed head 裁剪则字段已同步
            - embed_output: dict[str, Any] | None, embed head 输出; 未启用时为 None
        """
        if self.embed_head is None:
            return batch, None
        # torch.Tensor, (N_real, C_raw), embed head 裁剪前的原始原子特征; online voxel scatter 必须使用 raw 特征. 
        raw_atom_feat = batch["atom_feat"]
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
        batch["_online_pdb_raw_atom_feat"] = raw_atom_feat[global_keep_mask]
        batch["atom_feat"] = embed_output["atom_feat"]
        if self.interface_norm_embed_to_point is not None:
            # embed head 输出 -> point 真实原子初始特征接口归一化
            batch["atom_feat"] = self.interface_norm_embed_to_point(batch["atom_feat"])
        batch["atom_coord_centered_world"] = embed_output["atom_coord_centered_world"]
        batch["atom_batch_index"] = embed_output["atom_batch_index"]
        batch["atom_offsets"] = embed_output["atom_offsets"]
        batch["atom_counts"] = self._counts_from_offsets(batch["atom_offsets"])
        batch["atom_coord_local_voxel"] = embed_output["atom_coord_local_voxel"]
        batch["atom_is_in_core_box"] = embed_output["atom_is_in_core_box"]
        for key in ("atom_label", "atom_coord_world", "atom_global_indices"):
            if key in batch and batch[key] is not None:
                batch[key] = batch[key][global_keep_mask]
        return batch, embed_output

    def _build_voxel_input(self, batch: dict[str, Any], embed_output: dict[str, Any] | None) -> torch.Tensor:
        """
        构造 voxel backbone 输入张量(不要在这里生成候选集合 C). 

        输入参数:
            - batch: dict[str, Any], real-only canonical batch
            - embed_output: dict[str, Any] | None, embed head 输出; 未启用时为 None

        输出:
            - voxel_input: torch.Tensor, (B, C_in, D, H, W), voxel backbone 输入体素张量
        """
        if embed_output is not None and embed_output.get("voxel_pdb_embed_grid") is not None:
            # torch.Tensor, (B, C_embed, D, H, W), embed head 体素输出
            fused_voxel_grid = embed_output["voxel_pdb_embed_grid"]
            if bool(getattr(self.embed_head, "voxel_embed_as_tune", False)):
                return batch["voxel_grid"] + fused_voxel_grid
            return torch.cat([batch["voxel_grid"], fused_voxel_grid], dim=1)
        if self.online_pdb_feature:
            online_atom_feat = batch.get("_online_pdb_raw_atom_feat", batch["atom_feat"])
            with torch.no_grad():
                if self.online_pdb_feature_scatter_kernel == "gauss27":
                    # torch.Tensor, (B, online_pdb_feature_dim+occupancy/centroid, D, H, W), 3×3×3 各向同性高斯 scatter 的 raw atom_feat
                    raw_pdb_grid = gauss_scatter_to_voxel_grid(
                        point_feat=online_atom_feat.detach(),
                        atom_coord_local_voxel=batch["atom_coord_local_voxel"],
                        point_batch=batch["atom_batch_index"],
                        box_shape_zyx=batch["box_shape_zyx"],
                        batch_size=int(batch["box_shape_zyx"].shape[0]),
                        sigma_voxel=self.online_pdb_feature_sigma_voxel,
                        add_occupancy_channels=self.online_pdb_feature_add_occupancy,
                        add_centroid_channels=self.online_pdb_feature_add_centroid,
                    )
                else:
                    # torch.Tensor, (B, online_pdb_feature_dim, D, H, W), legacy 三线性/单体素 scatter 的 raw atom_feat
                    scatter_fn = (
                        soft_scatter_to_voxel_grid
                        if self.online_pdb_feature_use_soft_splatting
                        else scatter_to_voxel_grid
                    )
                    raw_pdb_grid = scatter_fn(
                        point_feat=online_atom_feat.detach(),
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
        执行一轮 voxel backbone. 

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
        在最后一轮 voxel backbone 后准备 sparse candidate C 与 P anchor mixed batch. 

        输入参数:
            - batch: dict[str, Any], 当前 real-only canonical batch
            - voxel_output_dict: dict[str, Any], 当前 recycle 的 _run_voxel_backbone() 输出, 本函数将读取 voxel_logits_ligand 生成 C

        输出:
            - point_batch: dict[str, Any], inject_pseudo_atoms 后的 mixed batch
            - pseudo_layout: PseudoAtomLayout | None, 描述 real/P mixed 布局
            - pseudo_outputs: dict[str, Any], pseudo_outputs = {**candidate_outputs, **anchor_outputs, "pseudo_density_feat"=伪原子特征}
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
        if self.interface_norm_density_to_point is not None:
            # density cube 输出 -> point 伪原子初始特征接口归一化(与 embed->point 真实原子接口对称)
            pseudo_feat = self.interface_norm_density_to_point(pseudo_feat)
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
        # torch.Tensor, (sumP, F_atom), 注入用的同一张 P 初始特征(anchor 顺序), 供 atom head density 残差消费
        pseudo_outputs["pseudo_density_feat"] = pseudo_feat
        return point_batch, pseudo_layout, pseudo_outputs

    def _run_point_backbone(
        self,
        batch: dict[str, Any],
        voxel_output_dict: dict[str, Any],
        point_recycle_in: torch.Tensor | None,
        pseudo_layout: PseudoAtomLayout | None = None,
    ) -> dict[str, Any]:
        """
        执行一轮 point backbone, 它会顺便记录 real-only 形式的 recycle 输入. 

        输入参数:
            - batch: dict[str, Any], real-only 或 mixed batch, 与 pseudo_layout 对齐
            - voxel_output_dict: dict[str, Any], 当前 recycle 的 voxel 输出
            - point_recycle_in: torch.Tensor | None, (sumN_real, C_point), 上一轮 real-only point recycle 状态
            - pseudo_layout: PseudoAtomLayout | None, mixed layout; None 表示 real-only 路径

        输出:
            - point_output_dict: dict[str, Any], point backbone 原始输出, mixed 路径下保留 mixed 顺序
        """
        if self.point_backbone is None:
            raise RuntimeError("纯 voxel 配置不允许调用 _run_point_backbone。")
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
        在最后一轮 point backbone 后运行 Stage1AtomHead, 并把监督字段裁成 real-only. 

        输入参数:
            - outputs: dict[str, Any], 最后一轮 backbone 输出汇总, 将会原地写入 atom head 输出
            - atom_head_batch: dict[str, Any], 与 outputs["fused_point_feat"] 同布局的 real 或 mixed batch, 仅用于提供 pseudo_mask 和真实原子监督字段, 不提供特征
            - pseudo_layout: PseudoAtomLayout | None, mixed layout; None 表示 real-only 路径

        输出:
            - None, 原地更新 outputs 中 before/after 特征、atom_logits、pseudo_logits、
              P 监督采样所需 pseudo_voxel_zyx/pseudo_batch_index, 以及 real-only supervised 字段
        """
        # torch.Tensor | None, (N_real, C_aux), home 体素处 voxel_aux base; 仅 refine_receptor_from_voxel 开启时计算, 恒 detach
        base = None
        if self.refine_receptor_from_voxel:
            if pseudo_layout is not None:
                # torch.Tensor, (N_real, 3), 真实原子连续体素坐标(mixed-real 顺序)
                real_coord_local = extract_real_tensor_from_mixed(atom_head_batch["atom_coord_local_voxel"], pseudo_layout)
                # torch.Tensor, (N_real,), 真实原子所属 BOX 索引(mixed-real 顺序)
                real_batch_index = extract_real_tensor_from_mixed(atom_head_batch["atom_batch_index"], pseudo_layout)
            else:
                real_coord_local = atom_head_batch["atom_coord_local_voxel"]
                real_batch_index = atom_head_batch["atom_batch_index"]
            base = self._gather_voxel_aux_logit_at_atom_home_voxel(
                voxel_logits_aux=outputs["voxel_logits_aux"],
                atom_coord_local_voxel=real_coord_local,
                atom_batch_index=real_batch_index,
                box_shape_zyx=atom_head_batch["box_shape_zyx"],
            )

        # 最终分类头: 产出 before/after 特征与 real/P logits. 
        if self.atom_head is not None:
            # torch.Tensor | None, (N_all,), bool, mixed 路径下 True 表示 P anchor
            pseudo_mask = atom_head_batch.get("pseudo_mask") if pseudo_layout is not None else None
            atom_head_output = self.atom_head(
                point_feat=outputs["fused_point_feat"],  # = point_output_dict["point_feat"]
                point_state=outputs["point_state"],
                atom_coord_centered_world=atom_head_batch["atom_coord_centered_world"],
                pseudo_mask=pseudo_mask,
            )
            outputs.update(atom_head_output)
            if pseudo_layout is None:
                outputs["atom_target"] = atom_head_batch.get("atom_label")
                outputs["atom_counts"] = atom_head_batch.get("atom_counts")
                outputs["atom_coord_local_voxel"] = atom_head_batch.get("atom_coord_local_voxel")
                outputs["atom_is_in_core_box"] = atom_head_batch.get("atom_is_in_core_box")
                outputs["atom_global_indices"] = atom_head_batch.get("atom_global_indices")
            else:
                outputs["atom_target"] = extract_real_tensor_from_mixed(atom_head_batch.get("atom_label"), pseudo_layout)
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
                # torch.Tensor, (N_pseudo, 3), float, P anchor 连续局部体素坐标 (x, y, z); 供 wrapper 采样 P ligand 监督
                pseudo_coord_local_voxel = extract_pseudo_tensor_from_mixed(
                    atom_head_batch.get("atom_coord_local_voxel"), pseudo_layout
                )
                # torch.Tensor, (N_pseudo, 3), long, P anchor home 体素离散索引 (z, y, x); floor 后按 (x,y,z)->(z,y,x) 重排, 与 candidate_voxel_zyx 同序
                pseudo_voxel_xyz = pseudo_coord_local_voxel.floor().to(torch.long)
                outputs["pseudo_voxel_zyx"] = pseudo_voxel_xyz[:, [2, 1, 0]].contiguous()
                # torch.Tensor, (N_pseudo,), long, P anchor 所属 BOX 索引
                outputs["pseudo_batch_index"] = extract_pseudo_tensor_from_mixed(
                    atom_head_batch.get("atom_batch_index"), pseudo_layout
                ).to(torch.long)
        else:
            # atom head 关时仅置空 logits 与 interaction 特征, 真实原子监督字段已由 forward 用 real_batch 写好. 
            outputs["real_feat_before_interaction"] = None
            outputs["real_feat_after_interaction"] = None
            outputs["pseudo_feat_before_interaction"] = None
            outputs["pseudo_feat_after_interaction"] = None
            outputs["atom_logits"] = None
            outputs["pseudo_logits"] = None
        # outputs["atom_logits"] 是最终 real_atom_head 产生的; refine_receptor 时末尾加残差. 
        if outputs.get("atom_logits") is not None and self.refine_receptor_from_voxel:
            outputs["atom_logits"] = outputs["atom_logits"] + base

    def _run_sparse_refine_head(
        self,
        outputs: dict[str, Any],
        voxel_output_dict: dict[str, Any],
        point_batch: dict[str, Any],
        pseudo_layout: PseudoAtomLayout | None,
    ) -> None:
        """
        在 final atom head 后将 P 消息聚合回唯一候选 C 并输出 refined logits. 

        输入参数:
            - outputs: dict[str, Any], final recycle 输出字典, 将原地追加 sparse refine 的结果
            - voxel_output_dict: dict[str, Any], final voxel backbone 输出
            - point_batch: dict[str, Any], final real/P mixed batch, 提供 BOX 坐标字段
            - pseudo_layout: PseudoAtomLayout | None, final mixed 布局

        输出:
            - None, 原地写入:
                - self.sparse_refine_head() 直接输出的 refine_outputs:
                    - "candidate_message_valid_mask": torch.Tensor, (sumC,), 是否至少有一个有效 P 邻居
                    - "ligand_refine_logits_C": torch.Tensor, (sumC, logit_dim), refined logits
                - self.anchor_to_candidate() 也就是 class AnchorToCandidateKnnSearch, 对(C,P)的边构建结果
        """
        if self.sparse_refine_head is None:
            return
        if self.anchor_to_candidate is None or pseudo_layout is None:
            raise RuntimeError("sparse refine 启用时 final recycle 必须存在 P anchor mixed layout。")
        if outputs.get("pseudo_feat_after_interaction") is None:
            raise RuntimeError("sparse refine 必须在 atom head 输出 pseudo_feat_after_interaction 后执行。")

        # torch.Tensor, (sumP, C_point), P 的 final point 特征; 从未 detach 的 point_feat_raw 取出, 再按 into_refine 开关单独 detach(与进 atom head 解耦)
        P_final_point_feat = extract_pseudo_tensor_from_mixed(outputs["point_feat_raw"], pseudo_layout)
        if self.detach_pseudo_point_feat_into_refine:
            P_final_point_feat = P_final_point_feat.detach()
        # torch.Tensor, (sumP, C_point), P 经 cross-attn 后的特征; refine 始终只读 after 路, 不回传 cross-attn
        P_after_interaction_feat = outputs["pseudo_feat_after_interaction"].detach()
        # torch.Tensor, (B, C_voxel, D, H, W), final voxel backbone 导出的固定 voxel_final 特征图
        voxel_final = voxel_output_dict["voxel_features"]["voxel_final"]
        # dict[str, torch.Tensor], C voxel center 对应的 local/world/centered-world 坐标字典
        candidate_coords = build_anchor_coordinates(  # 复用 P 的函数算 C, 没问题
            anchor_voxel_zyx=outputs["candidate_voxel_zyx"],
            anchor_batch_index=outputs["candidate_batch_index"],
            box_origin_world=point_batch["box_origin_world"],
            voxel_size_world=point_batch["voxel_size_world"],
            box_shape_zyx=point_batch["box_shape_zyx"],
        )
        # torch.Tensor, (sumC, 3), C voxel center 的 centered-world 坐标
        candidate_coord_centered_world = candidate_coords["anchor_coord_centered_world"]
        # torch.Tensor, (sumP, C_voxel), P 来源 voxel center 采样得到的 voxel_final 特征
        P_voxel_backbone_feat = gather_voxel_feature_at_zyx(
            voxel_feat=voxel_final,
            voxel_zyx=outputs["anchor_voxel_zyx"],
            point_batch_index=outputs["anchor_batch_index"],
        )
        # torch.Tensor, (sumC, C_voxel), C 来源 voxel center 采样得到的 voxel_final 特征
        C_voxel_backbone_feat = gather_voxel_feature_at_zyx(
            voxel_feat=voxel_final,
            voxel_zyx=outputs["candidate_voxel_zyx"],
            point_batch_index=outputs["candidate_batch_index"],
        )
        # detach_voxel_into_refine 统一控制 refine 吃的三处 voxel 信息是否回传梯度到 voxel backbone
        if self.detach_voxel_into_refine:
            P_voxel_backbone_feat = P_voxel_backbone_feat.detach()
            C_voxel_backbone_feat = C_voxel_backbone_feat.detach()
            # torch.Tensor, (sumC, C_logits), 复用 candidate_set_builder 产出的已 detach candidate logits
            voxel_logits_C = outputs["candidate_logits"]
        else:
            # torch.Tensor, (B, C_logits, D, H, W), final voxel backbone 输出的带梯度 ligand logits
            source_logits = voxel_output_dict["voxel_logits_ligand"]
            # torch.Tensor, (sumC, 3), 唯一 C 的 voxel z/y/x 坐标
            voxel_zyx = outputs["candidate_voxel_zyx"]
            # torch.Tensor, (sumC, C_logits), 从 dense ligand logits 现 gather 的带梯度 C logits
            voxel_logits_C = source_logits[
                outputs["candidate_batch_index"],
                :,
                voxel_zyx[:, 0],
                voxel_zyx[:, 1],
                voxel_zyx[:, 2],
            ]
        # dict[str, torch.Tensor], class AnchorToCandidateKnnSearch 对(C,P)的边构建结果(candidate_neighbor_index 、candidate_neighbor_squared_distance 等)
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
            P_final_point_feat=P_final_point_feat,
            P_after_interaction_feat=P_after_interaction_feat,
            P_voxel_backbone_feat=P_voxel_backbone_feat,
            anchor_class=outputs["anchor_class"],
            **neighbor_outputs,
        )
        outputs.update(neighbor_outputs)
        outputs.update(refine_outputs)

    @staticmethod
    def _publish_stage1_feature_hooks(
        outputs: dict[str, Any],
        voxel_output_dict: dict[str, Any],
        A_feat_L1: torch.Tensor | None,
        A_feat_L2: torch.Tensor | None,
    ) -> None:
        """把 centered 生产所需的真实层出口发布为稳定直键. 

        ``voxel_features`` 原样引用最终 recycle 的命名 V 字典. Find 另外发布: 
        ``A_feat_L1`` 为 point-side embed/interface normalization 后、原子 density
        调制前的 real 表示; ``A_feat_L2`` 为调制后送入 point backbone 的 real
        表示; L3/L4 分别引用 A/P interaction 前后且送入分类头的真实层张量. 
        ``P_feat_L2`` 引用 density/class/interface normalization 后的 P 初始表示, 
        P 的 L3/L4 同样引用 interaction 前后张量. 这里不复制张量, 也不改变训练
        forward、梯度或旧输出键. 
        """

        outputs["voxel_features"] = voxel_output_dict["voxel_features"]
        if A_feat_L1 is not None and A_feat_L2 is not None:
            outputs["A_feat_L1"] = A_feat_L1
            outputs["A_feat_L2"] = A_feat_L2
            outputs["A_feat_L3"] = outputs.get("real_feat_before_interaction")
            outputs["A_feat_L4"] = outputs.get("real_feat_after_interaction")
        if outputs.get("pseudo_density_feat") is not None:
            outputs["P_feat_L2"] = outputs["pseudo_density_feat"]
            outputs["P_feat_L3"] = outputs.get("pseudo_feat_before_interaction")
            outputs["P_feat_L4"] = outputs.get("pseudo_feat_after_interaction")










    def forward_voxel_probability(self, batch: dict[str, Any]) -> torch.Tensor:
        """
        运行三个 producer 与完整 forward 等价的最短 voxel-only 路径. 

        输入参数:
            - batch: dict[str,Any], `Stage1BatchCollator` 输出; Find 含 core+8 Å real atom 表, unet_c1 只需共同 dense voxel 字段

        输出:
            - voxel_logits_ligand: torch.Tensor, (B,1,80,80,80), BOX-local 离散 ZYX voxel 网格上的 ligand logits; 仍为 sigmoid 前值

        该入口固定执行三次 recycle, 并跳过 point blocks、point backbone、候选 C、
        P、A/P heads 与 sparse-refine. 它不调用完整 ``forward``, 也不抽取共享
        ``_forward_voxel_branch``, 从而保持现有训练 forward 的结构边界. 
        """

        if not self.enable_recycling or self.max_recycles != 3:
            raise RuntimeError("AdaLigand forward_voxel_probability 要求 enable_recycling=true 且 max_recycles=3。")
        # dict[str,Any], 统一字段名后的 Stage1 batch; dense 与 ragged 原子字段仍保持原值. 
        canonical = self._canonicalize_stage1_batch(batch)
        # torch.Tensor, (B,C_density,80,80,80), Dataset 构造的 producer 密度通道. 
        density_input = canonical["voxel_grid"]
        if self.embed_head is not None and self.embed_head.has_voxel_output:
            # torch.Tensor, (B,C_receptor,80,80,80), Find_1 的 50D+2D 网格或 Find_2 的 56D tune 网格.
            receptor_grid = self.embed_head.forward_voxel_only(
                atom_feat=canonical["atom_feat"],
                atom_coord_local_voxel=canonical["atom_coord_local_voxel"],
                atom_batch_index=canonical["atom_batch_index"],
                box_shape_zyx=canonical["box_shape_zyx"],
                atom_is_in_core_box=canonical["atom_is_in_core_box"],
            )
            # torch.Tensor, (B,C_voxel,80,80,80), producer-specific voxel backbone 输入. 
            voxel_input = (
                density_input + receptor_grid
                if bool(getattr(self.embed_head, "voxel_embed_as_tune", False))
                else torch.cat([density_input, receptor_grid], dim=1)
            )
        elif self.online_pdb_feature:
            if self.online_pdb_feature_scatter_kernel != "legacy":
                raise RuntimeError("Find_0 voxel-only 路径只允许 legacy hard scatter。")
            if self.online_pdb_feature_use_soft_splatting:
                raise RuntimeError("Find_0 voxel-only 路径禁止 soft splatting。")
            core_keep = canonical["atom_is_in_core_box"].bool()
            # torch.Tensor, (B,50,80,80,80), Find_0 core 原子 50D 特征的 hard-sum 网格.
            raw_grid = scatter_to_voxel_grid(
                point_feat=canonical["atom_feat"][core_keep].detach(),
                atom_coord_local_voxel=canonical["atom_coord_local_voxel"][core_keep],
                point_batch=canonical["atom_batch_index"][core_keep],
                box_shape_zyx=canonical["box_shape_zyx"],
                batch_size=int(canonical["box_shape_zyx"].shape[0]),
                reduce="sum",
                add_occupancy_channels=False,
            )
            # torch.Tensor, (B,C_density+50,80,80,80), Find_0 voxel backbone 输入.
            voxel_input = torch.cat([density_input, raw_grid], dim=1)
        elif self.embed_head is None and self.point_backbone is None:
            # unet_c1 没有原子支路, voxel 输入就是单一实验密度通道. 
            voxel_input = density_input
        else:
            raise RuntimeError("当前模型不是受支持的 Find_0、Find_1、Find_2 或 unet_c1 producer 配置。")

        # torch.Tensor | None, voxel backbone 的跨 recycle 隐状态; 第 1 轮为 None. 
        voxel_recycle_in: torch.Tensor | None = None
        # dict[str,Any] | None, 当前 recycle 的 voxel 输出; 最终从中读取 ligand logits. 
        voxel_output_dict: dict[str, Any] | None = None
        for recycle_index in range(3):
            voxel_output_dict = self._run_voxel_backbone(voxel_input, voxel_recycle_in)
            if recycle_index < 2:
                voxel_recycle_in = voxel_output_dict["voxel_recycle_out"]
                if voxel_recycle_in is not None and self.detach_recycle_states:
                    voxel_recycle_in = voxel_recycle_in.detach()
        if voxel_output_dict is None or voxel_output_dict.get("voxel_logits_ligand") is None:
            raise RuntimeError("voxel backbone 未返回 voxel_logits_ligand。")
        return voxel_output_dict["voxel_logits_ligand"]


    # -------------------------------------------------------------- forward --------------------------------------------------------------
    # ================================================================================================================================================
    # ================================================================================================================================================
    def forward(self, batch: dict[str, Any]) -> dict[str, Any]:
        """
        执行完整 Stage1 前向并发布最后一轮 voxel、point、atom、candidate 与 refine 输出. 

        输入参数:
            - batch: dict[str,Any], `Stage1BatchCollator` 输出的 real-only batch; 坐标字段包括 centered 连续世界 XYZ 与 BOX-local 连续 voxel XYZ

        输出:
            - outputs: dict[str,Any], 最后一轮结构化输出, 包含 `voxel_outputs`、`point_outputs`、real-only supervision 字段, 以及按配置追加的 atom/candidate/refine 字段
        """
        batch = self._canonicalize_stage1_batch(batch)
        if not self.enable_recycling:
            recycle_steps = 1
        elif self.training and self.randomize_recycles:
            recycle_steps = int(torch.randint(1, self.max_recycles + 1, (1,)).item())
        else:
            recycle_steps = self.max_recycles

        # unet_c1 的 Dataset 不返回 atom 表; 直接执行 density -> RAUNet, 既不构造
        # Point, 也不运行 A/P 相关组件. 
        if "atom_feat" not in batch:
            if self.embed_head is not None or self.point_backbone is not None or self.enable_atom_head:
                raise RuntimeError("无 atom 字段只允许 unet_c1 的纯 voxel 配置。")
            voxel_recycle_in: torch.Tensor | None = None
            voxel_output_dict: dict[str, Any] | None = None
            for recycle_idx in range(recycle_steps):
                voxel_output_dict = self._run_voxel_backbone(batch["voxel_grid"], voxel_recycle_in)
                if recycle_idx < recycle_steps - 1:
                    voxel_recycle_in = voxel_output_dict["voxel_recycle_out"]
                    if voxel_recycle_in is not None and self.detach_recycle_states:
                        voxel_recycle_in = voxel_recycle_in.detach()
            if voxel_output_dict is None:
                raise RuntimeError("unet_c1 voxel forward 未执行。")
            outputs = {
                "voxel_logits_aux": voxel_output_dict["voxel_logits_aux"],
                "voxel_logits_ligand": voxel_output_dict.get("voxel_logits_ligand"),
                "voxel_logits_protein": voxel_output_dict.get("voxel_logits_protein"),
                "voxel_logits_nucleic": voxel_output_dict.get("voxel_logits_nucleic"),
                "voxel_logits_distance": voxel_output_dict.get("voxel_logits_distance"),
                "voxel_outputs": voxel_output_dict,
                "embed_output": None,
                "voxel_recycle_out": voxel_output_dict["voxel_recycle_out"],
                "recycle_passes_used": recycle_steps,
            }
            self._publish_stage1_feature_hooks(outputs, voxel_output_dict, None, None)
            return outputs

        batch, embed_output = self._run_embed_head_once(batch)
        # torch.Tensor | None, (N_A,C_A1), embed/interface norm 后、density 调制前的 A. 
        A_feat_L1 = batch["atom_feat"] if embed_output is not None else None
        # torch.Tensor, (B, C_in, D, H, W), recycle 循环内复用的 voxel 输入
        voxel_input = self._build_voxel_input(batch, embed_output)
        # 第 3 点: 真实原子 density cube 叠加到 atom_feat(point 分支初始特征); 放在 voxel_input 之后避免污染 voxel 输入, recycle 前算一次
        batch = self._apply_real_atom_density_to_atom_feat(batch)
        # torch.Tensor | None, (N_A,C_A2), 实际送入 point backbone 的 density-modulated A. 
        A_feat_L2 = batch["atom_feat"] if embed_output is not None else None
        voxel_recycle_in: torch.Tensor | None = None
        point_recycle_in: torch.Tensor | None = None
        outputs: dict[str, Any] = {}
        # 送入 self._run_atom_head, 但仅用于提供 pseudo_mask 和真实原子的监督/坐标信息, 不提供特征
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
                    # real_batch: dict[str, Any], real-only batch, 后面仅使用真实原子监督字段, 不用特征
                    # real_point_output_dict: dict[str, Any], 仅包含真实原子 point backbone 中间输出的结果字典
                    real_batch, _real_point_feat, _real_point_state, real_point_output_dict = extract_real_point_output(
                        mixed_batch=point_batch,
                        fused_point_feat=point_output_dict["point_feat"],
                        point_output_dict=point_output_dict,
                        layout=pseudo_layout,
                    )
                else:
                    # 仅使用真实原子监督字段, 不用特征
                    real_batch = point_batch  
                    real_point_output_dict = point_output_dict
                
                # 仅用于提供 pseudo_mask 和真实原子的监督/坐标信息, 不提供特征
                last_atom_head_batch = point_batch  
                last_pseudo_layout = pseudo_layout
                # torch.Tensor | None, (N_all,), bool, final mixed batch 的 P anchor 掩码; real-only 路径为 None
                final_pseudo_mask = point_batch.get("pseudo_mask")
                outputs = {
                    "point_feat_raw": point_output_dict["point_feat"],
                    "fused_point_feat": self._apply_point_feat_detach_routing(
                        point_output_dict["point_feat"],
                        final_pseudo_mask,
                        self.detach_real_point_feat_into_atomhead,
                        self.detach_pseudo_point_feat_into_atomhead,
                    ),
                    "point_state": point_output_dict["point_state"],
                    "atom_target": real_batch.get("atom_label"),
                    "atom_counts": real_batch.get("atom_counts"),
                    "atom_coord_local_voxel": real_batch.get("atom_coord_local_voxel"),
                    "atom_is_in_core_box": real_batch.get("atom_is_in_core_box"),
                    "atom_global_indices": real_batch.get("atom_global_indices"),
                    "voxel_logits_aux": voxel_output_dict["voxel_logits_aux"],
                    "voxel_logits_ligand": voxel_output_dict.get("voxel_logits_ligand"),
                    "voxel_logits_protein": voxel_output_dict.get("voxel_logits_protein"),
                    "voxel_logits_nucleic": voxel_output_dict.get("voxel_logits_nucleic"),
                    "voxel_logits_distance": voxel_output_dict.get("voxel_logits_distance"),
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
        self._publish_stage1_feature_hooks(
            outputs,
            voxel_output_dict=outputs["voxel_outputs"],
            A_feat_L1=A_feat_L1,
            A_feat_L2=A_feat_L2,
        )
        outputs["recycle_passes_used"] = recycle_steps
        return outputs




    # ============================================================
    # ==================== 工具函数: 关于 Sparse Candidate Builder ====================
    # ============================================================
    
    def get_sparse_candidate_class_ids(self) -> tuple[int, ...] | None:
        """
        返回 sparse candidate builder 配置的候选类别 ID(1~K, K一般=C). 

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
        保存 wrapper 同步过来的 candidate threshold cache. 

        输入参数:
            - p_best_by_class: torch.Tensor | None, (K,), best-F1 阈值缓存; None 表示尚不可用
            - p_sampling_by_class: torch.Tensor | None, (K,), sampling 阈值缓存; None 表示尚不可用

        输出:
            - None, 原地更新 runtime cache: self._candidate_p_best_by_class / self._candidate_p_sampling_by_class(以及 _has...)
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
        保存 wrapper 同步过来的 candidate runtime 状态. 

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
        判断当前 forward 是否处于 candidate fixed topk 阶段. 

        输出:
            - use_fixed: bool, True 表示 scheduler warmup 内使用固定 per-class topc
        """
        return (
            self._candidate_allow_warmup_fixed_topk
            and self._candidate_warmup_steps > 0
            and self._candidate_global_step < self._candidate_warmup_steps
        )
