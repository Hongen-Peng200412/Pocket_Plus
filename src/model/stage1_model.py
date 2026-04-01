from __future__ import annotations

from typing import Any, Sequence

import torch
import torch.nn.functional as F
from hydra.utils import instantiate
from torch import nn

from src.model.pseudo_atoms import PseudoAtomGenerator

_PTV3_HEAD_IMPORT_ERROR: Exception | None = None
try:
    from PTV3bakcbone.model import Point, SerializedAttention, GatedTransition, MLP, resolve_act_layer, Block
except Exception as exc:
    Point = None
    SerializedAttention = None
    GatedTransition = None
    MLP = None
    resolve_act_layer = None
    Block = None
    _PTV3_HEAD_IMPORT_ERROR = exc




class Stage1SerializedAttentionStack(nn.Module):
    def __init__(
        self,
        channels: int,
        num_heads: int,
        patch_size: int,
        num_layers: int,
        serialization_orders: Sequence[str],
        shuffle_orders: bool,
        qkv_bias: bool,
        qk_scale: float | None,
        attn_drop: float,
        proj_drop: float,
        enable_rpe: bool,
        enable_flash: bool,
        upcast_attention: bool,
        upcast_softmax: bool,
        atom_head_ffn_type: str,
        mlp_ratio: int,
        act_layer,
        cpe_impl: str,
        cpe_kernel_size: int,
        cpe_receptive_field: float,
        pointconv_block_max_neighbors: int,
        drop_path: float,
        pre_norm: bool,
    ) -> None:
        """
        由 k 层 Block(cpe_impl 可配) 组成的 atom head 注意力堆叠。

        输入参数:
            - channels: int, token 特征通道数
            - num_heads: int, 多头注意力头数
            - patch_size: int, SerializedAttention 的 patch 大小
            - num_layers: int, 堆叠层数
            - serialization_orders: Sequence[str], 允许使用的序列化顺序列表
            - shuffle_orders: bool, 是否在每次 forward 开始时随机打乱序列化顺序列表
            - qkv_bias: bool, QKV 线性层是否使用 bias
            - qk_scale: float | None, QK 缩放因子
            - attn_drop: float, 注意力 dropout 概率
            - proj_drop: float, 输出投影 dropout 概率
            - enable_rpe: bool, 是否启用相对位置编码
            - enable_flash: bool, 是否启用 flash attention
            - upcast_attention: bool, 是否在注意力计算前上转精度
            - upcast_softmax: bool, 是否在 softmax 前上转精度
            - atom_head_ffn_type: str, FFN 类型 "mlp"/"gated"/"none"
            - mlp_ratio: int, FFN 隐藏层膨胀倍率
            - act_layer: callable, 激活函数类
            - cpe_impl: str, CPE 实现方式 "none"/"sparseconv"/"pointconv"
            - cpe_kernel_size: int, sparseconv CPE 卷积核大小
            - cpe_receptive_field: float, pointconv CPE 世界坐标感受野半径(Å)
            - pointconv_block_max_neighbors: int, pointconv CPE 每个点最大邻居数
            - drop_path: float, 随机深度 drop 概率
            - pre_norm: bool, 是否使用预归一化

        输出:
            - token_feat: torch.Tensor, `(sumN, C)`, 经过 k 层 attention 后的 atom token 特征
        """
        super().__init__()
        if Block is None:
            raise ImportError("Stage1SerializedAttentionStack 需要 PTV3 相关依赖。") from _PTV3_HEAD_IMPORT_ERROR
        self.channels = int(channels)
        self.serialization_orders = tuple(str(order_name) for order_name in serialization_orders)
        self.shuffle_orders = bool(shuffle_orders)

        self.layers = nn.ModuleList(
            [
                Block(
                    channels=self.channels,
                    num_heads=int(num_heads),
                    patch_size=int(patch_size),
                    order_index=int(layer_idx % len(self.serialization_orders)),
                    cpe_impl=str(cpe_impl),
                    qkv_bias=bool(qkv_bias),
                    qk_scale=qk_scale,
                    attn_drop=float(attn_drop),
                    proj_drop=float(proj_drop),
                    enable_rpe=bool(enable_rpe),
                    enable_flash=bool(enable_flash),
                    upcast_attention=bool(upcast_attention),
                    upcast_softmax=bool(upcast_softmax),
                    ffn_type=str(atom_head_ffn_type),
                    mlp_ratio=int(mlp_ratio),
                    act_layer=act_layer,
                    pre_norm=bool(pre_norm),
                    drop_path=float(drop_path),
                    cpe_kernel_size=int(cpe_kernel_size),
                    cpe_receptive_field=float(cpe_receptive_field),
                    pointconv_block_max_neighbors=int(pointconv_block_max_neighbors),
                )
                for layer_idx in range(int(num_layers))
            ]
        )
        self.output_norm = nn.LayerNorm(self.channels)

    def forward(self, point_state: dict[str, Any], token_feat: torch.Tensor) -> torch.Tensor:
        """
            执行完整的 k 层 Block 注意力。

            输入参数:
                - point_state: dict[str, Any], 点分支输出的点状态(coord, batch_index, offsets, grid_size, grid_coord)
                - token_feat: torch.Tensor, `(N, C)`, 进入 atom head 的 token 特征

            输出:
                - output_token_feat: torch.Tensor, `(N, C)`, atom head 输出特征
        """
        if token_feat.shape[0] == 0:
            return token_feat
        if Point is None:
            raise ImportError("Stage1SerializedAttentionStack 需要 PTV3 相关依赖。") from _PTV3_HEAD_IMPORT_ERROR

        # dict[str, Any], 用点状态与当前 token 特征重建的 Point 字典
        point_dict = {
            "feat": token_feat,
            "coord": point_state["coord"],
            "batch": point_state["batch"],
            "offset": point_state["offset"],
            "grid_size": point_state["grid_size"],
        }
        if "grid_coord" in point_state:
            point_dict["grid_coord"] = point_state["grid_coord"]

        # Point, `(sumN, C)` + 点坐标/批次/网格信息, atom head 注意力的直接输入对象
        point = Point(point_dict)
        # Point, 序列化后的点对象; 后续每层注意力将按这里确定的顺序处理 token
        point.serialization(order=self.serialization_orders, shuffle_orders=self.shuffle_orders)

        for layer in self.layers:
            # Point, 当前层 Block 更新后的点对象
            point = layer(point)

        # torch.Tensor, `(sumN, C)`, stack 末尾 LayerNorm 后的 atom token 特征
        return self.output_norm(point.feat)



class VolumePointStage1Model(nn.Module):
    def __init__(
        self,
        voxel_backbone: nn.Module | Any,
        point_backbone: nn.Module | Any,
        point_fusion_map: dict[str, str],
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
        detach_recycle_states: bool,   # True!
        act_layer_name: str,           # "gelu"
        ffn_type: str,                 # "mlp" | "gated", 控制 PTV3 backbone Block 的 FFN 类型(已透传给 point_backbone)
        atom_head_ffn_type: str,       # "mlp" | "gated" | "none", 控制 atom head 内的可选 FFN
        atom_head_mlp_ratio: int,      # 4, atom head FFN 隐藏层膨胀倍率(仅 atom_head_ffn_type != "none" 生效)
        atom_head_cpe_impl: str,       # "none" | "pointconv", atom head Block 的 CPE 实现方式
        atom_head_cpe_kernel_size: int,  # 5, atom head Block 的 sparseconv CPE 卷积核大小
        atom_head_cpe_receptive_field: float,  # 2.0, atom head Block 的 pointconv CPE 感受野半径(Å)
        atom_head_pointconv_max_neighbors: int,  # 16, atom head Block 的 pointconv CPE 最大邻居数
        atom_head_drop_path: float,    # 0.0, atom head Block 的随机深度 drop 概率
        atom_head_pre_norm: bool,      # True, atom head Block 是否使用预归一化
        embed_head: nn.Module | Any | None = None,  # embed head 模块或 Hydra 配置; None 时不启用
        pseudo_atom_cfg: dict | None = None,  # 伪原子配置; None 时不启用
    ) -> None:
        """
            Stage1 总装配模型。

            输入参数:
                - voxel_backbone: nn.Module | Any, 体素分支模块或 Hydra 配置
                - point_backbone: nn.Module | Any, 点分支模块或 Hydra 配置


                - point_fusion_map: dict[str, str], 点分支变量名到体素特征名的映射
                - sampler_modes: Sequence[str], 与 `point_fusion_map` 和 `point_fusion_modes` 等长的 voxel->point 取样策略列表；每个元素对应一次命名点变量融合
                - point_fusion_modes: Sequence[str], 与 `point_fusion_map` 等长的融合策略列表
                
                - fusion_mlp_ratio: float, `concat_linear` 融合模块的隐藏层倍率
                - fusion_proj_drop: float, `concat_linear` 融合模块的 dropout


                - atom_head_hidden_dim: int, atom head 隐藏维度
                - atom_head_num_heads: int, atom head 注意力头数
                - atom_head_patch_size: int, atom head patch size
                - atom_head_num_layers: int, atom head 堆叠层数
                - atom_head_serialization_orders: Sequence[str], atom head 序列化顺序
                - atom_head_shuffle_orders: bool, atom head 是否打乱序列化顺序
                - atom_head_qkv_bias: bool, atom head 的 QKV 线性层是否带 bias
                - atom_head_qk_scale: float | None, atom head 的 QK 缩放因子
                - atom_head_attn_drop: float, atom head 注意力 dropout
                - atom_head_proj_drop: float, atom head 输出投影 dropout
                - atom_head_enable_rpe: bool, atom head 是否启用相对位置编码
                - atom_head_enable_flash: bool, atom head 是否启用 flash attention
                - atom_head_upcast_attention: bool, atom head 是否在注意力前上转精度
                - atom_head_upcast_softmax: bool, atom head 是否在 softmax 前上转精度
                - atom_logit_dim: int, atom 分类头输出维度


                - enable_recycling: bool, 标量, 是否启用 recycle
                - max_recycles: int, 标量, 最大 recycle 次数
                - randomize_recycles: bool, 标量, 训练态是否随机 recycle 次数
                - detach_recycle_states: bool, 标量, recycle 轮间是否截断梯度

                - act_layer_name: str, 标量, 激活函数类名, 建议值 "gelu"
                - ffn_type: str, 标量, PTV3 backbone Block 的 FFN 类型, 可选 "mlp" 或 "gated"
                - atom_head_ffn_type: str, 标量, atom head 内的可选 FFN 类型, 可选 "mlp"、"gated" 或 "none"
                - atom_head_mlp_ratio: int, 标量, atom head FFN 隐藏层膨胀倍率
                - atom_head_cpe_impl: str, 标量, atom head Block 的 CPE 实现方式, 可选 "none" 或 "pointconv"
                - atom_head_cpe_kernel_size: int, 标量, atom head Block 的 sparseconv CPE 卷积核大小
                - atom_head_cpe_receptive_field: float, 标量, atom head Block 的 pointconv CPE 感受野半径(Å)
                - atom_head_pointconv_max_neighbors: int, 标量, atom head Block 的 pointconv CPE 最大邻居数
                - atom_head_drop_path: float, 标量, atom head Block 的随机深度 dropout 概率
                - atom_head_pre_norm: bool, 标量, atom head Block 是否使用预归一化
                - embed_head: nn.Module | Any | None, 无固定形状, embed head 模块或 Hydra 配置

    
                - pseudo_atom_cfg: dict | None, 无固定形状, 伪原子生成器配置
                
            输出:
                - forward() 返回 dict[str, Any]
                    - `"atom_logits"`: torch.Tensor, `(sumN, C_logit)`, atom 分类 logits
                    - `"atom_hidden"`: torch.Tensor, `(sumN, C_head)`, 经过 k 层 SerializedAttention 后、但没经过分类MLP前的 atom 特征
                    - `"atom_tokens"`: torch.Tensor, `(sumN, C_token_in)`, atom head 输入的 token，由点特征、中心化世界坐标与 atom 有效标记拼接而成
                    - `"sampled_point_fusion_feat_dict"`: dict[str, torch.Tensor], 每个点变量名对应的采样体素特征, 仅用于记录
                    - `"voxel_outputs"`: dict[str, Any], 体素backbone(跑完最后一次recycle)输出的dict
                    - `"point_outputs"`: dict[str, Any], 点backbone(跑完最后一次recycle)输出的dict
                    - `"recycle_passes_used"`: int, 本次 forward 实际使用的 recycle 次数
        """
        super().__init__()

        # PseudoAtomGenerator | None, 伪原子生成器; None 时不启用
        if pseudo_atom_cfg is not None:
            self.pseudo_atom_gen = PseudoAtomGenerator(**pseudo_atom_cfg)
        else:
            self.pseudo_atom_gen = None
        # embed head 永远在 recycle 循环外只执行一次
        if embed_head is not None:
            self.embed_head = embed_head if isinstance(embed_head, nn.Module) else instantiate(embed_head)
        else:
            self.embed_head = None

        # nn.Module, 体素分支模块；允许直接传模块实例或 Hydra 配置对象。
        self.voxel_backbone = voxel_backbone if isinstance(voxel_backbone, nn.Module) else instantiate(voxel_backbone)
        # nn.Module, 点分支模块；允许直接传模块实例或 Hydra 配置对象。
        self.point_backbone = point_backbone if isinstance(point_backbone, nn.Module) else instantiate(point_backbone)
        self.point_fusion_items = tuple(
            (str(point_name), voxel_name_str)
            for point_name, voxel_name in point_fusion_map.items()
            if voxel_name is not None and (voxel_name_str := str(voxel_name).strip()) != ""
        )
        # tuple[str, ...]，与 `point_fusion_items` 逐项对应的融合策略列表。
        self.point_fusion_modes = tuple(str(mode_name).lower() for mode_name in point_fusion_modes)
        # tuple[str, ...]，与 `point_fusion_items` 逐项对应的 voxel->point 采样策略列表。
        self.sampler_modes = tuple(str(mode_name).lower() for mode_name in sampler_modes)
        self.enable_recycling = bool(enable_recycling)
        self.max_recycles = int(max_recycles)
        self.randomize_recycles = bool(randomize_recycles)
        self.detach_recycle_states = bool(detach_recycle_states)
        if not (len(self.point_fusion_items) == len(self.point_fusion_modes) == len(self.sampler_modes)):
            raise ValueError("point_fusion_map、point_fusion_modes 与 sampler_modes 的长度必须一致。")
        unsupported_sampler_modes = [mode_name for mode_name in self.sampler_modes if mode_name not in {"trilinear", "nearest"}]
        if unsupported_sampler_modes:
            raise ValueError(f"Unsupported sampler_modes={unsupported_sampler_modes}, supported=['nearest', 'trilinear']")
        if self.max_recycles <= 0:
            raise ValueError("max_recycles must be > 0")
        if self.pseudo_atom_gen is not None:
            self._validate_pseudo_recycle_policy()





        # dict[str, str]，每个需要融合的点变量对应的体素特征名。
        self.point_fusion_map = {point_name: voxel_name for point_name, voxel_name in self.point_fusion_items}
        # dict[str, str]，每个需要融合的点变量对应的采样策略。
        self.sampler_mode_by_point_name = {
            point_name: mode_name
            for (point_name, _), mode_name in zip(self.point_fusion_items, self.sampler_modes)
        }
        # dict[str, str]，每个需要融合的点变量对应的融合策略。
        self.fusion_mode_by_point_name = {
            point_name: mode_name
            for (point_name, _), mode_name in zip(self.point_fusion_items, self.point_fusion_modes)
        }




        # tuple[str, ...]，按“体素分支显式请求返回”与“融合阶段实际需要”取并集后的体素变量名。
        self.voxel_feature_names_to_return = tuple(
            dict.fromkeys(
                tuple(str(feature_name) for feature_name in self.voxel_backbone.return_feature_keys)
                + tuple(voxel_name for _, voxel_name in self.point_fusion_items)
            )
        )
        # tuple[str, ...]，点分支本轮需要导出的命名变量名；至少包含最终的 `point_feat`。
        self.point_feature_names_to_return = tuple(
            dict.fromkeys([point_name for point_name, _ in self.point_fusion_items] + ["point_feat"])
        )




        # 收集体素和点分支的所有变量名, 查看要融合的字典 self.point_fusion_items 的合法性
        available_voxel_feature_names = tuple(self.voxel_backbone.feature_channels_by_name.keys())
        available_point_feature_names = tuple(self.point_backbone.feature_channels_by_name.keys())
        for point_name, voxel_name in self.point_fusion_items:
            if point_name not in available_point_feature_names:
                raise KeyError(f"Unknown point fusion name={point_name}, available={available_point_feature_names}")
            if voxel_name not in available_voxel_feature_names:
                raise KeyError(f"Unknown voxel fusion name={voxel_name}, available={available_voxel_feature_names}")





        # 为融合设定模块————————本代码块目前只支持 concat_linear
        if resolve_act_layer is None:
            raise ImportError("解析激活函数需要 PTV3 相关依赖。") from _PTV3_HEAD_IMPORT_ERROR
        act_cls = resolve_act_layer(str(act_layer_name))
        self.point_fusion_modules = nn.ModuleDict()
        for point_name, voxel_name in self.point_fusion_items:
            fusion_mode = self.fusion_mode_by_point_name[point_name]
            point_channels = int(self.point_backbone.feature_channels_by_name[point_name])
            voxel_channels = int(self.voxel_backbone.feature_channels_by_name[voxel_name])
            if fusion_mode != "concat_linear":
                raise ValueError(f"Unsupported point fusion mode={fusion_mode}")

            # type, 激活函数类: 由 act_layer_name 统一配置
            if resolve_act_layer is None:
                raise ImportError("解析激活函数需要 PTV3 相关依赖。") from _PTV3_HEAD_IMPORT_ERROR
            act_cls = resolve_act_layer(str(act_layer_name))

            fusion_input_dim = point_channels + voxel_channels
            fusion_hidden_dim = max(
                point_channels,
                int(round(float(fusion_input_dim) * float(fusion_mlp_ratio))),
            )
            self.point_fusion_modules[point_name] = nn.Sequential(
                nn.Linear(fusion_input_dim, fusion_hidden_dim),
                nn.LayerNorm(fusion_hidden_dim),
                act_cls(),
                nn.Dropout(float(fusion_proj_drop)),
                nn.Linear(fusion_hidden_dim, point_channels),
            )



        # 为 atom head 设定模块
        atom_token_input_dim = int(self.point_backbone.out_channels) + 3 + 1
        self.atom_token_proj = nn.Sequential(
            nn.Linear(atom_token_input_dim, int(atom_head_hidden_dim)),
            nn.LayerNorm(int(atom_head_hidden_dim)),
            act_cls(),
        )
        self.atom_attention_stack = Stage1SerializedAttentionStack(
            channels=int(atom_head_hidden_dim),
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
            atom_head_ffn_type=str(atom_head_ffn_type),
            mlp_ratio=int(atom_head_mlp_ratio),
            act_layer=act_cls,
            cpe_impl=str(atom_head_cpe_impl),
            cpe_kernel_size=int(atom_head_cpe_kernel_size),
            cpe_receptive_field=float(atom_head_cpe_receptive_field),
            pointconv_block_max_neighbors=int(atom_head_pointconv_max_neighbors),
            drop_path=float(atom_head_drop_path),
            pre_norm=bool(atom_head_pre_norm),
        )
        self.atom_logit_head = nn.Sequential(
            nn.Linear(int(atom_head_hidden_dim), int(atom_head_hidden_dim)),
            act_cls(),
            nn.Linear(int(atom_head_hidden_dim), int(atom_logit_dim)),
        )

        # --- embed head 残差加和融合(仅 embed head 有点云相关输出时创建) ---
        if self.embed_head is not None and self.embed_head.has_point_output:
            # nn.Linear, (embed_head.atom_feature_dim -> embed_point_out_channels), 将 embed head 输出保留的原始 atom_feat(49) 做线性投影残差
            _atom_feat_dim = int(self.embed_head.atom_feature_dim)
            _embed_point_dim = int(self.embed_head.embed_point_out_channels)
            self.embed_point_add_proj = nn.Linear(_atom_feat_dim, _embed_point_dim)
            # nn.Parameter, 标量, 可学习的残差缩放门; 初始化为小值以稳定训练初期
            self.embed_point_gate = nn.Parameter(torch.tensor(0.01))
        else:
            self.embed_point_add_proj = None
            self.embed_point_gate = None



    # -------------------------------------------------------- 工具函数 --------------------------------------------------------
    @staticmethod
    def _voxel_xyz_to_grid_sample_xyz(
        point_coord_local_voxel: torch.Tensor,
        box_shape_zyx: torch.Tensor,
    ) -> torch.Tensor:
        """
        将 BOX 内连续 voxel corner 坐标转换为 `grid_sample` 需要的归一化坐标。

        输入参数:
            - point_coord_local_voxel: torch.Tensor, `(N, 3)`, 当前点在 BOX 内的连续 voxel corner 坐标，顺序为 `(x, y, z)`
            - box_shape_zyx: torch.Tensor, `(3,)`, 当前 BOX 的体素尺寸，顺序为 `(Z, Y, X)`

        输出:
            - grid_xyz: torch.Tensor, `(N, 3)`, 可直接送入 `grid_sample(..., align_corners=True)` 的接近于[-1,1]的归一化坐标(在外壳可能有少许超出)
        """
        # torch.Tensor, `(3,)`，当前 BOX 尺寸的 `(x, y, z)` 版本。
        box_shape_xyz = box_shape_zyx.to(
            device=point_coord_local_voxel.device,
            dtype=point_coord_local_voxel.dtype,
        )[[2, 1, 0]]
        # torch.Tensor, `(N, 3)`，从 corner 语义转成 center-index 语义。
        point_coord_center_index = point_coord_local_voxel - 0.5
        # torch.Tensor, `(3,)`，按 `align_corners=True` 归一化时使用的分母。
        denom_xyz = torch.clamp(box_shape_xyz - 1.0, min=1.0)
        # torch.Tensor, `(N, 3)`，归一化到 `[-1, 1]` 后的采样坐标。
        grid_xyz = (2.0 * point_coord_center_index / denom_xyz) - 1.0

        single_axis_mask = box_shape_xyz <= 1
        if bool(single_axis_mask.any()):
            grid_xyz[:, single_axis_mask] = 0.0
        return grid_xyz

    @staticmethod
    def _centered_world_xyz_to_local_voxel_xyz(
        point_coord_centered_world: torch.Tensor,
        voxel_size_world: torch.Tensor,
        box_shape_zyx: torch.Tensor,
    ) -> torch.Tensor:
        """
        将以 BOX 中心为原点的世界坐标，恢复为 BOX 内连续 voxel corner 坐标。

        输入参数:
            - point_coord_centered_world: torch.Tensor, `(N, 3)`, 当前点坐标，顺序为 `(x, y, z)`，单位 Å
            - voxel_size_world: torch.Tensor, `(3,)`, 当前 BOX 的 voxel size，顺序为 `(x, y, z)`，单位 Å
            - box_shape_zyx: torch.Tensor, `(3,)`, 当前 BOX 的体素尺寸，顺序为 `(Z, Y, X)`

        输出:
            - point_coord_local_voxel: torch.Tensor, `(N, 3)`, 当前点在 BOX 内的连续 voxel corner 坐标
        """
        # torch.Tensor, `(3,)`，当前 BOX 尺寸的 `(x, y, z)` 版本。
        box_shape_xyz = box_shape_zyx.to(
            device=point_coord_centered_world.device,
            dtype=point_coord_centered_world.dtype,
        )[[2, 1, 0]]
        # torch.Tensor, `(N, 3)`，BOX 中心坐标系 -> voxel corner 坐标系。
        return point_coord_centered_world / voxel_size_world.to(
            device=point_coord_centered_world.device,
            dtype=point_coord_centered_world.dtype,
        ) + (0.5 * box_shape_xyz)

    def set_input_channels(self, in_channels: int) -> None:
        """
        将输入通道设置请求透传给体素分支。
        当 embed head 启用时, 自动加上 embed_voxel_out_channels。

        输入参数:
            - in_channels: int, 数据集返回的 voxel_grid 通道数(不含 embed head 贡献)
        """
        # int, 实际输入通道 = 数据通道 + embed head 体素通道(若启用)
        actual_in_channels = int(in_channels)
        if self.embed_head is not None:
            actual_in_channels += int(self.embed_head.embed_voxel_out_channels)
        if hasattr(self.voxel_backbone, "set_input_channels"):
            self.voxel_backbone.set_input_channels(actual_in_channels)








    #  -------------------------------------------------------- 点与体素的融合逻辑 --------------------------------------------------------
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
            对单个 BOX 的点集采样一份体素特征。

            输入参数:
                - voxel_feat_one_box: torch.Tensor, `(1, C, D_f, H_f, W_f)` 或 `(C, D_f, H_f, W_f)`，单个 BOX 的体素特征图
                - point_coord_centered_world_one_box: torch.Tensor, `(N_i, 3)`, 当前点集的中心化世界坐标
                - voxel_size_world_one_box: torch.Tensor, `(3,)`, 当前 BOX 的 voxel size
                - box_shape_zyx_one_box: torch.Tensor, `(3,)`, 当前 BOX 的尺寸
                - fusion_mode: str, 当前命名融合项使用的融合策略；当前实现 `concat_linear`
                - sampler_mode: str, 当前命名融合项使用的采样策略；当前实现 `trilinear` / `nearest`

            输出:
                - sampled_feat_one_box: torch.Tensor, `(N_i, C)`, 当前点集对应的采样体素特征
        """
        if voxel_feat_one_box.ndim == 4:
            voxel_feat_one_box = voxel_feat_one_box.unsqueeze(0)
        point_count = int(point_coord_centered_world_one_box.shape[0])
        if point_count == 0:
            return voxel_feat_one_box.new_empty((0, int(voxel_feat_one_box.shape[1])))
        if fusion_mode != "concat_linear":
            raise ValueError(f"Unsupported point fusion mode={fusion_mode}")
        grid_sample_mode = "bilinear" if sampler_mode == "trilinear" else "nearest"


        # torch.Tensor, `(N_i, 3)`，当前点在 BOX 内的连续 voxel corner 坐标。
        point_coord_local_voxel = self._centered_world_xyz_to_local_voxel_xyz(
            point_coord_centered_world=point_coord_centered_world_one_box,
            voxel_size_world=voxel_size_world_one_box,
            box_shape_zyx=box_shape_zyx_one_box,
        )
        #  `(N_i, 3)`，按原始几何定义换算得到的 `grid_sample` 坐标(可能略微超出[-1,1]), 函数本身不做 clamp
        grid_xyz = self._voxel_xyz_to_grid_sample_xyz(
            point_coord_local_voxel=point_coord_local_voxel,
            box_shape_zyx=box_shape_zyx_one_box,
        )


        # torch.Tensor, `(1, N_i, 1, 1, 3)`，适配 5D `grid_sample` 的采样网格
        grid = grid_xyz.view(1, point_count, 1, 1, 3)
        # torch.Tensor, `(1, C, N_i, 1, 1)`，单个 BOX 内的点采样结果
        sampled = F.grid_sample(
            input=voxel_feat_one_box,
            grid=grid,
            mode=grid_sample_mode,
            padding_mode="zeros",
            align_corners=True,
        )
        # torch.Tensor, `(N_i, C)`，去掉冗余维度后的点特征
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
            对 batch 内所有点采样一份指定命名体素特征。

            输入参数:
                - voxel_feat: torch.Tensor, `(B, C, D_f, H_f, W_f)`, 体素特征图
                - point_coord_centered_world: torch.Tensor, `(N_current, 3)`, 当前点集坐标
                - point_batch_index: torch.Tensor, `(N_current,)`, 当前点集所属 batch 索引
                - voxel_size_world: torch.Tensor, `(B, 3)`, 每个 BOX 的 voxel size
                - box_shape_zyx: torch.Tensor, `(B, 3)`, 每个 BOX 的尺寸
                - fusion_mode: str, 当前命名融合项使用的融合策略；当前实现 `concat_linear`
                - sampler_mode: str, 当前命名融合项使用的采样策略

            输出:
                - sampled_feat: torch.Tensor, `(N_current, C)`, 当前点集对应的采样体素特征
        """
        point_count = int(point_coord_centered_world.shape[0])
        if point_count == 0:
            return voxel_feat.new_empty((0, int(voxel_feat.shape[1])))

        if fusion_mode != "concat_linear":
            raise ValueError(f"Unsupported point fusion mode={fusion_mode}")
        batch_size = int(box_shape_zyx.shape[0])
        # torch.Tensor, `(N_current, C)`, 整个batch的采样体素特征（顺序不变）
        sampled_feat = voxel_feat.new_empty((point_count, int(voxel_feat.shape[1])))

        for box_idx in range(batch_size):
            # torch.Tensor, `(N_current,)`, 当前 BOX 下的点掩码
            point_mask = point_batch_index == box_idx
            # torch.Tensor, `(N_i, 3)`, 当前 BOX 内的点坐标
            point_coord_one_box = point_coord_centered_world[point_mask]
            sampled_feat_one_box = self._sample_voxel_feature_single_box(
                voxel_feat_one_box=voxel_feat[box_idx : box_idx + 1],
                point_coord_centered_world_one_box=point_coord_one_box,
                voxel_size_world_one_box=voxel_size_world[box_idx],
                box_shape_zyx_one_box=box_shape_zyx[box_idx],
                fusion_mode=fusion_mode,
                sampler_mode=sampler_mode,
            )
            sampled_feat[point_mask] = sampled_feat_one_box.to(dtype=sampled_feat.dtype)

        return sampled_feat

    def _fuse_point_variable(
        self,
        feature_name: str,
        point_like: Any,
        voxel_output_dict: dict[str, Any],
        batch: dict[str, Any],
        sampled_point_fusion_feat_dict: dict[str, torch.Tensor],
    ) -> Any:
        """
            对某个命名点变量执行一次即时 voxel->point 融合。

            输入参数:
                - feature_name: str, 当前点变量名(按照字典 self.point_fusion_map 决定的规则选择是否融合、融合的对应体素特征、融合策略)
                - point_like: Any, 当前点对象；`point.feat` 形状为 `(N_current, C_point)`
                - voxel_output_dict: dict[str, Any], 当前轮体素分支输出
                - batch: dict[str, Any], 当前 batch 字典
                - sampled_point_fusion_feat_dict: dict[str, torch.Tensor], 记录采样的体素特征. str为点变量名，值为对应的采样体素特征

            输出:
                - point_like: Any, 融合后的点对象
        """
        if feature_name not in self.point_fusion_map:
            return point_like

        voxel_name = self.point_fusion_map[feature_name]
        fusion_mode = self.fusion_mode_by_point_name[feature_name]
        sampler_mode = self.sampler_mode_by_point_name[feature_name]

        # torch.Tensor, `(N_current, C_voxel)`, 当前点集按该点变量绑定的采样策略采样得到的体素特征。
        sampled_voxel_feat = self._sample_voxel_feature_batch(
            voxel_feat=voxel_output_dict["voxel_features"][voxel_name],
            point_coord_centered_world=point_like.coord,
            point_batch_index=point_like.batch,
            voxel_size_world=batch["voxel_size_world"],
            box_shape_zyx=batch["box_shape_zyx"],
            fusion_mode=fusion_mode,
            sampler_mode=sampler_mode,
        )
        sampled_point_fusion_feat_dict[feature_name] = sampled_voxel_feat

        # torch.Tensor, `(N_current, C_point + C_voxel)`, 点特征与采样体素特征的拼接结果。
        fusion_input = torch.cat([point_like.feat, sampled_voxel_feat], dim=-1)
        point_like.feat = self.point_fusion_modules[feature_name](fusion_input)  # 目前实际调用上个函数 def _sample_voxel_feature_batch
        return point_like



















    #  -------------------------------------------------------- forward --------------------------------------------------------
    def forward(self, batch: dict[str, Any]) -> dict[str, Any]:
        """
            模型的总 forward

            输入:
                - batch: dict[str, Any], `box_point_collate()` 返回的 batch 字典

            输出:
                - output_dict: dict[str, Any], 包含 atom logits、中间点变量融合信息与 recycle 信息
        """
        if not self.enable_recycling:  # int, 未启用 recycle 时固定只执行 1 轮前向。
            recycle_steps = 1
        elif self.training and self.randomize_recycles:  # int, 训练态可随机采样 1~max_recycles 轮
            recycle_steps = int(torch.randint(1, self.max_recycles + 1, (1,)).item())
        else:  # int, 推理态或关闭随机 recycle 时的固定轮数。
            recycle_steps = self.max_recycles

        # `batch` 始终保持当前轮开始时的 real-only canonical 视图。
        voxel_recycle_in: torch.Tensor | None = None
        point_recycle_in: torch.Tensor | None = None
        # torch.Tensor | None, 仅在 recycle_policy="all" 时跨轮保留的伪原子 point recycle 隐状态。
        pseudo_point_recycle_in: torch.Tensor | None = None

        # 当前 real-only 特征空间下缓存的伪原子模板；下一次注入会将维度与 `batch["atom_feat"]` 对齐。
        pseudo_cache: dict[str, Any] | None = None
        # list[bool], 伪原子生命周期 [embed_head, point_backbone, atom_head]
        lifecycle = (
            self.pseudo_atom_gen.lifecycle
            if self.pseudo_atom_gen is not None
            else [False, False, False]
        )


        # -------------------------------- embed head (永远在 recycle 循环外只执行一次) --------------------------------
        # list[tuple[int, int]] | None, embed 阶段 mixed 视图的 `(n_real, n_pseudo)`。
        embed_split_info: list[tuple[int, int]] | None = None
        if lifecycle[0]:
            pseudo_cache = self._prepare_aligned_pseudo_dict(
                batch=batch,
                cached_pseudo_dict=None,
            )
            batch, embed_split_info = self.pseudo_atom_gen.inject(batch, pseudo_cache)
        # dict[str, Any] | None, embed head 输出(含裁剪后的 atom 字段与体素嵌入网格)
        embed_output: dict[str, Any] | None = None
        if self.embed_head is not None:
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
            # ---- filter: 用裁剪后的原子字段替换 batch(不改原 dict, shallow copy) ----
            batch = {**batch}
            # torch.Tensor, (sumN',), bool, 从原始 sumN 到裁剪后长度的全局掩码
            global_keep_mask = embed_output["global_keep_mask"]
            batch["atom_feat"] = embed_output["atom_feat"]
            batch["atom_coord_centered_world"] = embed_output["atom_coord_centered_world"]
            batch["atom_batch_index"] = embed_output["atom_batch_index"]
            batch["atom_offsets"] = embed_output["atom_offsets"]
            batch["atom_counts"] = self._counts_from_offsets(batch["atom_offsets"])
            batch["atom_coord_local_voxel"] = embed_output["atom_coord_local_voxel"]
            batch["atom_is_in_core_box"] = embed_output["atom_is_in_core_box"]

            # 同步裁剪所有 per-atom 监督与辅助字段
            for _key in ("atom_label", "atom_valid_mask", "atom_coord_world", "atom_global_indices"):
                if _key in batch and batch[_key] is not None:
                    batch[_key] = batch[_key][global_keep_mask]
            # 若 embed head 裁剪了原子, 同步更新 split_info 与 atom_counts
            if embed_split_info is not None:
                embed_split_info = self._update_split_info_after_trim(embed_split_info, global_keep_mask)

            # ---- 残差加和: 原始 atom_feat 投影到 embed 表示空间后与 embed_point_feat 相加 ----
            if self.embed_point_add_proj is not None and embed_output.get("embed_point_feat") is not None:
                # torch.Tensor, (sumN', embed_point_out_channels), 原始原子特征投影到 embed 空间
                projected_atom = self.embed_point_add_proj(batch["atom_feat"])
                # torch.Tensor, (sumN', embed_point_out_channels), 投影后的原子特征 + gated embed 特征
                batch["atom_feat"] = projected_atom + self.embed_point_gate * embed_output["embed_point_feat"]

        # embed 阶段之后，统一回到 real-only canonical batch，并缓存当前特征空间下的伪原子模板
        if embed_split_info is not None:
            pseudo_cache = self._capture_pseudo_dict_from_batch(
                batch=batch,
                split_info=embed_split_info,
            )
            batch = self.pseudo_atom_gen.remove(batch, embed_split_info)




        # ----------------------------------------------------------------------------------------------------------------
        final_output_dict: dict[str, Any] = {}
        for _ in range(recycle_steps):
            # ---- 组装体素输入: 数据通道 + embed head 体素通道(若启用) ----
            if embed_output is not None:
                # torch.Tensor, (B, C_data + C_embed, D, H, W), 数据通道与 embed head 输出拼接
                voxel_input = torch.cat([batch["voxel_grid"], embed_output["voxel_pdb_embed_grid"]], dim=1)
            else:
                voxel_input = batch["voxel_grid"]



            # ------------------------------------- voxel backbone -------------------------------------
            # dict[str, Any], 当前轮体素分支原始输出，包含命名体素特征、辅助 logits 与 recycle 输出。
            voxel_output_dict = self.voxel_backbone(
                voxel_grid=voxel_input,
                recycle_in=voxel_recycle_in,
                return_feature_keys=self.voxel_feature_names_to_return,
            )





            # ------------------------------------- point backbone -------------------------------------
            # pseudo_dict 始终表示“本轮注入到当前 mixed 视图里的那份伪原子模板”。
            pseudo_dict: dict[str, Any] | None = pseudo_cache
            point_batch = batch
            point_recycle = point_recycle_in
            # list[tuple[int, int]] | None, 当前 recycle 内 mixed 原子布局 `[real_i, pseudo_i]` 的计数表。
            mixed_split_info: list[tuple[int, int]] | None = None
            if lifecycle[1]:
                pseudo_dict = self._prepare_aligned_pseudo_dict(
                    batch=batch,
                    cached_pseudo_dict=pseudo_cache,
                )
                point_batch, mixed_split_info = self.pseudo_atom_gen.inject(batch, pseudo_dict)
                pseudo_point_recycle = (pseudo_point_recycle_in if self.pseudo_atom_gen.keep_point_recycle_state_across_recycle() else None)
                # 真实原子 recycle 状态与伪原子 recycle 状态在这里重新交错成 mixed 布局。
                point_recycle = self._expand_real_tensor_with_pseudo_slots(
                    real_tensor=point_recycle_in,
                    split_info=mixed_split_info,
                    pseudo_tensor=pseudo_point_recycle,
                )

            # dict[str, torch.Tensor], 记录每个命名点变量实际采样到的体素特征，仅用于可视化与调试。
            sampled_point_fusion_feat_dict: dict[str, torch.Tensor] = {}
            if getattr(self.point_backbone, "backend", None) == "zeros":
                # dict[str, Any], zeros 后端直接提供全零占位点特征；此路径不执行 point forward，也不做 voxel->point 融合。
                point_output_dict = self.point_backbone.build_zeros_output(
                    atom_feat=point_batch["atom_feat"],
                    atom_coord_centered_world=point_batch["atom_coord_centered_world"],
                    atom_batch_index=point_batch["atom_batch_index"],
                    atom_offsets=point_batch["atom_offsets"],
                    return_feature_names=self.point_feature_names_to_return,
                )
            else:
                def point_feature_hook(feature_name: str, point_like: Any) -> Any:
                    return self._fuse_point_variable(
                        feature_name=feature_name,
                        point_like=point_like,
                        voxel_output_dict=voxel_output_dict,
                        batch=point_batch,
                        sampled_point_fusion_feat_dict=sampled_point_fusion_feat_dict,
                    )
                # dict[str, Any], 当前轮点分支原始输出，包含点特征、点状态与中间命名点变量。
                point_output_dict = self.point_backbone(
                    atom_feat=point_batch["atom_feat"],
                    atom_coord_centered_world=point_batch["atom_coord_centered_world"],
                    atom_batch_index=point_batch["atom_batch_index"],
                    atom_offsets=point_batch["atom_offsets"],
                    recycle_in=point_recycle,
                    point_feature_hook=point_feature_hook,
                    return_feature_names=self.point_feature_names_to_return,
                )

            # `point_return` 始终整理成 real-only 视图，用于下一轮 recycle 与对外返回。
            point_return = point_output_dict  # 单纯用来最后的 return
            atom_batch = point_batch   # 这里的atom_*,  表示用于 atom head 的 * (batch)
            atom_point_feat = point_output_dict["point_feat"]
            atom_point_state = point_output_dict["point_state"]
            next_pseudo_point_recycle: torch.Tensor | None = None   # 用于点云分支 recycle 的伪原子特征
            if lifecycle[1] and mixed_split_info is not None:
                if self.pseudo_atom_gen.keep_point_recycle_state_across_recycle():
                    next_pseudo_point_recycle = self.pseudo_atom_gen.extract_pseudo_tensor_from_mixed(
                        mixed_tensor=point_output_dict.get("point_recycle_out"),
                        split_info=mixed_split_info,
                    )
                # point backbone 若带伪原子，保存 real-only 视图
                (
                    _real_mask,
                    real_batch_after_point,
                    real_atom_point_feat,
                    real_atom_point_state,
                    point_return,
                ) = self._build_real_views_from_mixed_point_output(
                    batch=point_batch,
                    fused_point_feat=atom_point_feat,
                    point_output_dict=point_output_dict,
                    split_info=mixed_split_info,
                )
                # 如果伪原子不进入 atom head, 就切回 real-only 视图
                if not lifecycle[2]:
                    batch = real_batch_after_point
                    atom_batch = real_batch_after_point
                    atom_point_feat = real_atom_point_feat
                    atom_point_state = real_atom_point_state
                    mixed_split_info = None
            pseudo_point_recycle_in = next_pseudo_point_recycle
            if pseudo_dict is not None:
                pseudo_cache = pseudo_dict
            # atom head 正式 forward
            atom_tokens = torch.cat(
                [
                    atom_point_feat,
                    atom_batch["atom_coord_centered_world"],
                    atom_batch["atom_valid_mask"].to(dtype=atom_point_feat.dtype).unsqueeze(-1),
                ],
                dim=-1,
            )
            atom_hidden = self.atom_token_proj(atom_tokens)
            atom_hidden = self.atom_attention_stack(
                point_state=atom_point_state,
                token_feat=atom_hidden,
            )
            atom_logits = self.atom_logit_head(atom_hidden)

            # atom head 结束后，只对真实原子保留输出；伪原子只作为消息传递节点，不进入监督。
            if lifecycle[2] and mixed_split_info is not None:
                real_mask = self.pseudo_atom_gen.build_real_mask(mixed_split_info).to(device=atom_logits.device)
                atom_logits = atom_logits[real_mask]
                atom_hidden = atom_hidden[real_mask]
                atom_tokens = atom_tokens[real_mask]
                atom_point_feat = atom_point_feat[real_mask]
                batch = self.pseudo_atom_gen.remove(atom_batch, mixed_split_info)
                atom_batch = batch






            # ------------------------------------- 循环逻辑 -------------------------------------
            # torch.Tensor | None, 下一轮体素 recycle 输入；按配置决定是否截断跨轮梯度。
            voxel_recycle_in = voxel_output_dict["voxel_recycle_out"]
            if voxel_recycle_in is not None and self.detach_recycle_states:
                voxel_recycle_in = voxel_recycle_in.detach()

            # torch.Tensor | None, 下一轮点分支针对真实原子的 recycle 输入。
            # 伪原子的 recycle 隐状态单独存放在 `pseudo_point_recycle_in` 中, 不再与 real-only 视图耦合。
            point_recycle_in = point_return["point_recycle_out"]
            if point_recycle_in is not None and self.detach_recycle_states:
                point_recycle_in = point_recycle_in.detach()
            if pseudo_point_recycle_in is not None and self.detach_recycle_states:
                pseudo_point_recycle_in = pseudo_point_recycle_in.detach()

            final_output_dict = {
                "fused_point_feat": atom_point_feat,
                "atom_tokens": atom_tokens,
                "atom_hidden": atom_hidden,
                "atom_logits": atom_logits,
                "atom_target": atom_batch.get("atom_label", batch.get("atom_label")),
                "atom_valid_mask": atom_batch.get("atom_valid_mask", batch.get("atom_valid_mask")),
                "atom_counts": atom_batch.get("atom_counts", batch.get("atom_counts")),
                "atom_coord_local_voxel": atom_batch.get("atom_coord_local_voxel", batch.get("atom_coord_local_voxel")),
                "atom_is_in_core_box": atom_batch.get("atom_is_in_core_box", batch.get("atom_is_in_core_box")),
                "atom_global_indices": atom_batch.get("atom_global_indices", batch.get("atom_global_indices")),
                
                "sampled_point_fusion_feat_dict": sampled_point_fusion_feat_dict,   # dict[str, torch.Tensor], 记录每个命名点变量实际采样到的体素特征
                "voxel_logits_aux": voxel_output_dict["voxel_logits_aux"],
                "voxel_outputs": voxel_output_dict,
                "point_outputs": point_return,

                # embed head 输出(若启用)
                "embed_output": embed_output,

                # 下2个变量仅作为辅助, 返回的时候不管(所以注释里只写了前8个)
                "voxel_recycle_out": voxel_output_dict["voxel_recycle_out"],        # 体素分支这一轮的体素特征
                "point_recycle_out": point_return["point_recycle_out"],             # 点分支这一轮针对真实原子的点特征
            }

        # int, 本次 forward 实际执行的 recycle 轮数。
        final_output_dict["recycle_passes_used"] = recycle_steps
        return final_output_dict





    # ============================================================ 工具函数 =========================================================

    # ------------------------------------ 伪原子相关逻辑(直接调用 src\model\pseudo_atoms.py 的逻辑) -----------------------------------
    def _align_pseudo_features_to_batch(
        self,
        pseudo_dict: dict[str, Any],
        atom_feat: torch.Tensor,
    ) -> dict[str, Any]:
        """
        如果 `pseudo_feat` 不等于当前的 `batch["atom_feat"]`, 就通过 self.embed_point_add_proj 将 `pseudo_feat` 对齐到当前 `batch["atom_feat"]` 的维度。

        输入参数:
            - pseudo_dict: dict[str, Any], 伪原子字典, 至少包含 `pseudo_feat`
            - atom_feat: torch.Tensor, (sumN, C_target), 当前 batch 的原子特征

        输出:
            - aligned_pseudo_dict: dict[str, Any], 与 `atom_feat` 特征维度对齐后的伪原子字典
        """
        pseudo_feat = pseudo_dict.get("pseudo_feat")
        if pseudo_feat is None:
            return pseudo_dict

        target_dim = int(atom_feat.shape[1])
        pseudo_dim = int(pseudo_feat.shape[1])
        if target_dim == pseudo_dim:
            return pseudo_dict

        if (
            self.embed_point_add_proj is not None
            and pseudo_dim == int(self.embed_point_add_proj.in_features)
            and target_dim == int(self.embed_point_add_proj.out_features)
        ):
            aligned_pseudo_dict = {**pseudo_dict}
            aligned_pseudo_dict["pseudo_feat"] = self.embed_point_add_proj(pseudo_feat)
            return aligned_pseudo_dict

        raise RuntimeError(
            f"Pseudo atom feature dim mismatch before inject: atom_feat={target_dim}, "
            f"pseudo_feat={pseudo_dim}. No projector is available to align them."
        )

    def _prepare_aligned_pseudo_dict(
        self,
        batch: dict[str, Any],
        cached_pseudo_dict: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """
        根据当前 recycle policy 的策略, 依据 batch 调整当前伪原子字典 cached_pseudo_dict，并让伪原子特征与 batch 维度一致。

        输入参数:
            - batch: dict[str, Any], 当前 real-only batch 视图, 可能是 embed 后的 real-only batch
            - cached_pseudo_dict: dict[str, Any] | None, 上一轮或 embed 阶段缓存的伪原子模板

        输出:
            - pseudo_dict: dict[str, Any], 与当前 `batch["atom_feat"]` 维度一致的伪原子字典
        """
        if self.pseudo_atom_gen is None:
            raise RuntimeError("Pseudo-atom generation was requested but pseudo_atom_cfg is disabled.")
        pseudo_dict = self.pseudo_atom_gen.prepare_pseudo_dict_for_recycle(
            batch=batch,
            cached_pseudo_dict=cached_pseudo_dict,
        )
        return self._align_pseudo_features_to_batch(pseudo_dict, batch["atom_feat"])

    def _capture_pseudo_dict_from_batch(
        self,
        batch: dict[str, Any],
        split_info: list[tuple[int, int]],
    ) -> dict[str, Any]:
        """
        从当前 mixed batch 中提取伪原子子字典，并让它与当前 batch 的 atom 维度相同。

        输入参数:
            - batch: dict[str, Any], 当前 mixed batch
            - split_info: list[tuple[int, int]], 长度 = B, 每个 BOX 的 `(n_real, n_pseudo)`

        输出:
            - pseudo_dict: dict[str, Any], 与当前 `batch["atom_feat"]` 对齐的伪原子字典
        """
        if self.pseudo_atom_gen is None:
            raise RuntimeError("Pseudo-atom capture requires pseudo_atom_cfg.")
        pseudo_dict = self.pseudo_atom_gen.extract_pseudo_dict_from_batch(
            batch=batch,
            split_info=split_info,
        )
        return self._align_pseudo_features_to_batch(pseudo_dict, batch["atom_feat"])


    def _expand_real_tensor_with_pseudo_slots(
        self,
        real_tensor: torch.Tensor | None,
        split_info: list[tuple[int, int]],
        pseudo_tensor: torch.Tensor | None = None,
    ) -> torch.Tensor | None:
        """
        按 `[real_i, pseudo_i]` 的交错布局，将 real-only 张量扩展为 mixed 张量。

        输入参数:
            - real_tensor: torch.Tensor | None, `(sumN_real, ...)` 或 `(sumN_real+sumM, ...)`, 当前 real-only 张量
            - split_info: list[tuple[int, int]], 长度 = B, 每个 BOX 的 `(n_real, n_pseudo)`
            - pseudo_tensor: torch.Tensor | None, `(sumM, ...)`, 写入 pseudo 槽位的张量; 为 None 时补零

        输出:
            - mixed_tensor: torch.Tensor | None, `(sumN_real+sumM, ...)`, 交错布局后的 mixed 张量
        """
        if self.pseudo_atom_gen is None:
            raise RuntimeError("Pseudo slot expansion requires pseudo_atom_cfg.")
        return self.pseudo_atom_gen.interleave_real_and_pseudo_tensor(
            real_tensor=real_tensor,
            split_info=split_info,
            pseudo_tensor=pseudo_tensor,
        )










    # -----------------------------------------------------------
    # 从 mixed point 输出中构造 real-only 视图(用于返回或下一阶段)
    # -----------------------------------------------------------
    @staticmethod
    def _real_counts_from_split_info(
        split_info: list[tuple[int, int]],
        device: torch.device,
    ) -> torch.Tensor:
        """
        从 `split_info` 提取每个 BOX 的真实原子数。

        输入参数:
            - split_info: list[tuple[int, int]], 长度 = B, 每个 BOX 的 `(n_real, n_pseudo)`
            - device: torch.device, 返回张量所在设备

        输出:
            - real_counts: torch.Tensor, (B,), long, 每个 BOX 的真实原子数
        """
        return torch.tensor([nr for nr, _ in split_info], dtype=torch.long, device=device)

    @classmethod
    def _filter_point_state_with_mask(
        cls,
        point_state: dict[str, Any],
        keep_mask: torch.Tensor,
        real_counts: torch.Tensor,
    ) -> dict[str, Any]:
        """
        基于真实点掩码 `keep_mask` 裁剪 `point_state` 。

        输入参数:
            - point_state: dict[str, Any], point backbone 输出的状态字典
            - keep_mask: torch.Tensor, (sumN,), bool, 真实点掩码
            - real_counts: torch.Tensor, (B,), long, 每个 BOX 的真实点数

        输出:
            - filtered_state: dict[str, Any], 与 `keep_mask` 对齐后的 point_state
        """
        filtered_state = {**point_state}
        filtered_state["coord"] = point_state["coord"][keep_mask]
        filtered_state["batch"] = point_state["batch"][keep_mask]
        filtered_state["offset"] = torch.cumsum(real_counts.to(device=point_state["coord"].device), dim=0)
        if "grid_coord" in point_state and point_state["grid_coord"] is not None:
            filtered_state["grid_coord"] = point_state["grid_coord"][keep_mask]
        return filtered_state

    def _build_real_views_from_mixed_point_output(
        self,
        batch: dict[str, Any],
        fused_point_feat: torch.Tensor,
        point_output_dict: dict[str, Any],
        split_info: list[tuple[int, int]],
    ) -> tuple[torch.Tensor, dict[str, Any], torch.Tensor, dict[str, Any], dict[str, Any]]:
        """
        从 mixed point 输出中构造 real-only 视图(用于返回或下一阶段)。

        输入参数:
            - batch: dict[str, Any], 已注入伪原子的 mixed batch
            - fused_point_feat: torch.Tensor, (sumN_real+sumM, C_point), point backbone 输出特征
            - point_output_dict: dict[str, Any], point backbone 原始输出(含伪原子)
            - split_info: list[tuple[int, int]], 长度 = B, 每个 BOX 的 `(n_real, n_pseudo)`

        输出:
            - real_mask: torch.Tensor, (sumN_real+sumM,), bool, 真实原子掩码
            - real_batch: dict[str, Any], 仅保留真实原子的 batch 视图
            - real_fused_point_feat: torch.Tensor, (sumN_real, C_point), 仅真实原子的点特征
            - real_point_state: dict[str, Any], 仅真实原子的 point_state 视图
            - point_output_for_return: dict[str, Any], 对外返回的 real-only point 输出字典
        """
        if self.pseudo_atom_gen is None:
            raise RuntimeError("Pseudo-atom real-view extraction requires pseudo_atom_cfg.")
        real_mask = self.pseudo_atom_gen.build_real_mask(split_info).to(device=fused_point_feat.device)
        real_batch = self.pseudo_atom_gen.remove(batch, split_info)
        real_counts = self._real_counts_from_split_info(split_info, device=fused_point_feat.device)
        real_fused_point_feat = fused_point_feat[real_mask]
        real_point_state = self._filter_point_state_with_mask(
            point_output_dict["point_state"],
            keep_mask=real_mask,
            real_counts=real_counts,
        )
        point_output_for_return = {**point_output_dict}
        point_output_for_return["point_feat"] = real_fused_point_feat
        point_output_for_return["point_state"] = real_point_state
        if point_output_dict.get("point_recycle_out") is not None:
            point_output_for_return["point_recycle_out"] = point_output_dict["point_recycle_out"][real_mask]
        feature_dict = point_output_dict.get("point_feature_dict")
        if isinstance(feature_dict, dict):
            trimmed_feature_dict: dict[str, Any] = {}
            for feature_name, feature_value in feature_dict.items():
                if torch.is_tensor(feature_value) and feature_value.ndim >= 1 and feature_value.shape[0] == real_mask.shape[0]:
                    trimmed_feature_dict[feature_name] = feature_value[real_mask]
                else:
                    trimmed_feature_dict[feature_name] = feature_value
            point_output_for_return["point_feature_dict"] = trimmed_feature_dict
        return real_mask, real_batch, real_fused_point_feat, real_point_state, point_output_for_return

    # -----------------------------------------------------------











    # --------------------------------------------- 其它 ---------------------------------------------
    def _validate_pseudo_recycle_policy(self) -> None:
        """
        校验伪原子配置的基础合法性。

        当前 `recycle_policy` 的详细语义由 `PseudoAtomGenerator` 自身定义，
        主模型层只在这里保留最小的启用态校验。
        """
        if self.pseudo_atom_gen is None:
            return
        if not any(bool(flag) for flag in self.pseudo_atom_gen.lifecycle):
            raise ValueError("pseudo_atom_cfg 已启用，但 lifecycle 三个阶段不能全部为 False。")
        if self.pseudo_atom_gen.lifecycle[2] and not self.pseudo_atom_gen.lifecycle[1]:
            raise ValueError(
                "当前实现不支持仅在 atom head 启用伪原子；"
                "若 lifecycle[2] 为 True，则 lifecycle[1] 也必须为 True。"
            )


    @staticmethod
    def _counts_from_offsets(atom_offsets: torch.Tensor) -> torch.Tensor:
        """
        从累计 offset 恢复每个 BOX 的真实原子数。

        输入参数:
            - atom_offsets: torch.Tensor, (B,), long, PTV3 风格的累计原子 offset

        输出:
            - atom_counts: torch.Tensor, (B,), long, 每个 BOX 的真实原子数
        """
        if atom_offsets.numel() == 0:
            return atom_offsets.new_zeros((0,), dtype=torch.long)
        atom_counts = atom_offsets.clone()
        atom_counts[1:] = atom_counts[1:] - atom_counts[:-1]
        return atom_counts


    @staticmethod
    def _update_split_info_after_trim(
        split_info: list[tuple[int, int]],
        global_keep_mask: torch.Tensor,
    ) -> list[tuple[int, int]]:
        """
        当 embed head 的 global_keep_mask 裁剪了原子时, 同步更新 split_info。

        输入参数:
            - split_info: list[tuple[int, int]], 裁剪前每个 BOX 的 (n_real, n_pseudo)
            - global_keep_mask: torch.Tensor, (sumN+sumM,), bool, True 表示保留

        输出:
            - new_split_info: list[tuple[int, int]], 裁剪后每个 BOX 的 (n_real, n_pseudo)
        """
        # torch.Tensor, (sumN+sumM,), bool, CPU 版本
        mask_cpu = global_keep_mask.cpu()
        new_split_info: list[tuple[int, int]] = []
        offset = 0
        for nr, np_ in split_info:
            # bool, 当前 BOX 的 real 部分保留数
            real_kept = int(mask_cpu[offset : offset + nr].sum().item())
            # bool, 当前 BOX 的 pseudo 部分保留数
            pseudo_kept = int(mask_cpu[offset + nr : offset + nr + np_].sum().item())
            new_split_info.append((real_kept, pseudo_kept))
            offset += nr + np_
        return new_split_info
