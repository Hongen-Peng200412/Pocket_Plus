from __future__ import annotations

from typing import Any, Callable, Sequence

import torch
from torch import nn

_PTV3_IMPORT_ERROR: Exception | None = None
try:
    from PTV3bakcbone.model import Point, PointTransformerV3
except Exception as exc:  # pragma: no cover - 依赖当前本地环境
    Point = None
    PointTransformerV3 = None
    _PTV3_IMPORT_ERROR = exc


PointFeatureHook = Callable[[str, Any], Any]


class Stage1PointBackbone(nn.Module):
    """
        Stage1 点分支。

        init参数:
            - backend: str, 点分支主干类型；当前支持 `ptv3` 或 `zeros`
            - atom_feature_dim: int, 原子底层特征维度
            - point_grid_size: float, 点分支内部离散化网格尺寸, 它与体素分支的 voxel size 不是同一个概念
            - input_embed_dim: int, 输入到 PTV3 前的点嵌入维度
            - input_embed_hidden_dim: int, 输入投影 MLP 的隐藏层维度
            - out_channels: int, int, 点云分支最终输出维度, 它 = int(self.enc_channels[-1] if self.cls_mode else self.dec_channels[0]), 必须与 PTV3 最终输出维度保持一致。
            - recycle_feature_dim: int, 点分支 recycle 特征维度, PTV3最终输出将与最原始特征 atom_feature_dim 拼起来做MLP, 再送入PTV3

        其它与PTV3相同.

        输出:
            - forward() 返回 dict[str, Any]
                - `"point_feat"`: torch.Tensor, `(N_final, C_point)`, 点分支最终特征
                - `"point_state"`: dict[str, Any], atom head 复用的点状态
                - `"point_recycle_out"`: torch.Tensor, `(N_final, C_point)`, 点分支 recycle 输出
                - `"point_feature_dict"`: dict[str, torch.Tensor], 被请求导出的命名点特征
    """

    def __init__(
        self,
        backend: str,  # "ptv3"
        atom_feature_dim: int,  # 49
        point_grid_size: float,  # 0.25
        input_embed_dim: int,  # 64
        input_embed_hidden_dim: int,  # 128
        out_channels: int,  # 64
        recycle_feature_dim: int,  # 64
        serialization_orders: Sequence[str],  # ("z", "z-trans", "hilbert", "hilbert-trans")
        shuffle_orders: bool,  # True
        stride: Sequence[int],  # (4, 2, 2, 2)
        embedding_kernel_size: int,  # 7
        embedding_impl: str,  # "pointconv"
        cpe_impl: str,  # "pointconv"
        embedding_receptive_field: float,  # 5.0
        pointconv_embed_max_neighbors: int,  # 64
        pointconv_block_max_neighbors: int,  # 32
        enc_cpe_kernel_size: Sequence[int],  # [5, 5, 5, 5, 5]
        dec_cpe_kernel_size: Sequence[int],  # [5, 5, 5, 5]
        enc_cpe_receptive_field: Sequence[float],  # [2.0, 4.0, 8.0, 12.0, 16.0]
        dec_cpe_receptive_field: Sequence[float],  # [2.0, 4.0, 8.0, 12.0]
        enc_depths: Sequence[int],  # (2, 2, 2, 6, 2)
        enc_channels: Sequence[int],  # (64, 64, 128, 256, 512)
        enc_num_head: Sequence[int],  # (2, 4, 8, 16, 32)
        enc_patch_size: Sequence[int],  # (256, 128, 96, 72, 64)
        dec_depths: Sequence[int],  # (2, 2, 2, 2)
        dec_channels: Sequence[int],  # (64, 64, 128, 256)
        dec_num_head: Sequence[int],  # (4, 4, 8, 16)
        dec_patch_size: Sequence[int],  # (128, 96, 72, 64)
        mlp_ratio: int,  # 4
        qkv_bias: bool,  # True
        qk_scale: float | None,  # None
        attn_drop: float,  # 0.0
        proj_drop: float,  # 0.0
        drop_path: float,  # 0.3
        pre_norm: bool,  # True
        enable_rpe: bool,  # False
        enable_flash: bool,  # True
        upcast_attention: bool,  # False
        upcast_softmax: bool,  # False
        cls_mode: bool,  # False
        pdnorm_bn: bool,  # False
        pdnorm_ln: bool,  # False
        pdnorm_decouple: bool,  # True
        pdnorm_adaptive: bool,  # False
        pdnorm_affine: bool,  # True
        pdnorm_conditions: Sequence[str],  # ("ScanNet", "S3DIS", "Structured3D")
    ) -> None:
        """
            Stage1 点分支。

            init参数:
                - backend: str, 点分支主干类型；当前支持 `ptv3` 或 `zeros`
                - atom_feature_dim: int, 原子底层特征维度
                - point_grid_size: float, 点分支内部离散化网格尺寸
                - input_embed_dim: int, 输入到 PTV3 前的点嵌入维度
                - input_embed_hidden_dim: int, 输入投影 MLP 的隐藏层维度
                - out_channels: int, 点云分支最终输出维度
                - recycle_feature_dim: int, 点分支 recycle 特征维度
                - embedding_impl: str, embedding 实现方式, "sparseconv" 或 "pointconv"
                - cpe_impl: str, Block CPE 实现方式, "sparseconv" 或 "pointconv"
                - embedding_receptive_field: float, embedding 世界坐标感受野(Å)(仅 pointconv)
                - pointconv_embed_max_neighbors: int, embedding 最大邻居数(仅 pointconv)
                - pointconv_block_max_neighbors: int, Block CPE 最大邻居数(仅 pointconv)
                - enc_cpe_kernel_size: Sequence[int], 编码器每层 CPE kernel size(仅 sparseconv)
                - dec_cpe_kernel_size: Sequence[int], 解码器每层 CPE kernel size(仅 sparseconv)
                - enc_cpe_receptive_field: Sequence[float], 编码器每层 CPE 感受野(仅 pointconv)
                - dec_cpe_receptive_field: Sequence[float], 解码器每层 CPE 感受野(仅 pointconv)
                
            其它与PTV3相同.

            输出:
                - forward() 返回 dict[str, Any]
                    - `"point_feat"`: torch.Tensor, `(N_final, C_point)`, 点分支最终特征
                    - `"point_state"`: dict[str, Any], atom head 复用的点状态
                    - `"point_recycle_out"`: torch.Tensor, `(N_final, C_point)`, 它=point_feat(不同于体素分支还需要经过简单投影)
                    - `"point_feature_dict"`: dict[str, torch.Tensor], 导出的命名点特征
        """
        super().__init__()

        self.backend = str(backend)
        self.atom_feature_dim = int(atom_feature_dim)
        self.point_grid_size = float(point_grid_size)
        self.input_embed_dim = int(input_embed_dim)
        self.out_channels = int(out_channels)
        self.recycle_feature_dim = int(recycle_feature_dim)
        self.embedding_kernel_size = int(embedding_kernel_size)
        self.embedding_impl = str(embedding_impl)
        self.cpe_impl = str(cpe_impl)
        self.serialization_orders = tuple(str(order_name) for order_name in serialization_orders)
        self.shuffle_orders = bool(shuffle_orders)
        self.cls_mode = bool(cls_mode)
        self.enc_channels = tuple(int(value) for value in enc_channels)
        self.dec_channels = tuple(int(value) for value in dec_channels)

        # 编码器阶段名称("point_enc0"(无下采样), "point_enc1", "point_enc2", "point_enc3", "point_enc4")
        self.enc_stage_names = tuple(f"point_enc{stage_idx}" for stage_idx in range(len(self.enc_channels)))
        # 解码器阶段名称(point_dec3, point_dec1, ...)
        self.dec_stage_names = tuple(f"point_dec{stage_idx}" for stage_idx in reversed(range(len(self.dec_channels))))

        # 所有可用变量名
        self.available_feature_names = (
            "point_input_feat",
            "point_embed",
            *self.enc_stage_names,
            *self.dec_stage_names,
            "point_feat",
        )

        # feature_channels_by_name 最终形如{可用变量名(str): 这个变量对应的特征维度(int)}, 仅用于检查
        feature_channels_by_name: dict[str, int] = {
            "point_input_feat": self.input_embed_dim,
            "point_embed": self.enc_channels[0],
        }
        feature_channels_by_name.update(
            {
                feature_name: int(channel_count)
                for feature_name, channel_count in zip(self.enc_stage_names, self.enc_channels)
            }
        )
        feature_channels_by_name.update(
            {
                feature_name: int(channel_count)
                for feature_name, channel_count in zip(self.dec_stage_names, reversed(self.dec_channels))
            }
        )
        feature_channels_by_name["point_feat"] = int(self.enc_channels[-1] if self.cls_mode else self.dec_channels[0])
        self.feature_channels_by_name = feature_channels_by_name


        # 检查输出通道是否符合预期值 out_channels
        expected_out_channels = self.feature_channels_by_name["point_feat"]
        if self.out_channels != expected_out_channels:
            raise ValueError(
                f"out_channels={self.out_channels} 与点分支结构不匹配，期望值为 {expected_out_channels}。"
            )



        # 组件
        # nn.Sequential, `(sumN, C_input=49 + C_recycle) -> (sumN, C_input=49)`，输入到点主干前的原子特征投影。
        self.atom_input_proj = nn.Sequential(
            nn.Linear(self.atom_feature_dim + self.recycle_feature_dim, int(input_embed_hidden_dim)),
            nn.LayerNorm(int(input_embed_hidden_dim)),
            nn.GELU(),
            nn.Linear(int(input_embed_hidden_dim), self.input_embed_dim),
        )
        if self.backend == "ptv3":
            if PointTransformerV3 is None:
                raise ImportError("backend='ptv3' 需要 PTV3 相关依赖。") from _PTV3_IMPORT_ERROR

            self.point_encoder = PointTransformerV3(    # NOTE: 原版的PTV3的主模块没有灵活化激活函数而是默认GELU, 未来可能需要加上
                in_channels=self.input_embed_dim,
                order=self.serialization_orders,
                stride=tuple(int(value) for value in stride),
                embedding_kernel_size=self.embedding_kernel_size,
                embedding_impl=self.embedding_impl,
                cpe_impl=self.cpe_impl,
                embedding_receptive_field=float(embedding_receptive_field),
                pointconv_embed_max_neighbors=int(pointconv_embed_max_neighbors),
                pointconv_block_max_neighbors=int(pointconv_block_max_neighbors),
                enc_cpe_kernel_size=tuple(int(value) for value in enc_cpe_kernel_size),
                dec_cpe_kernel_size=tuple(int(value) for value in dec_cpe_kernel_size),
                enc_cpe_receptive_field=tuple(float(value) for value in enc_cpe_receptive_field),
                dec_cpe_receptive_field=tuple(float(value) for value in dec_cpe_receptive_field),
                enc_depths=tuple(int(value) for value in enc_depths),
                enc_channels=self.enc_channels,
                enc_num_head=tuple(int(value) for value in enc_num_head),
                enc_patch_size=tuple(int(value) for value in enc_patch_size),
                dec_depths=tuple(int(value) for value in dec_depths),
                dec_channels=self.dec_channels,
                dec_num_head=tuple(int(value) for value in dec_num_head),
                dec_patch_size=tuple(int(value) for value in dec_patch_size),
                mlp_ratio=int(mlp_ratio),
                qkv_bias=bool(qkv_bias),
                qk_scale=qk_scale,
                attn_drop=float(attn_drop),
                proj_drop=float(proj_drop),
                drop_path=float(drop_path),
                pre_norm=bool(pre_norm),
                shuffle_orders=bool(shuffle_orders),
                enable_rpe=bool(enable_rpe),
                enable_flash=bool(enable_flash),
                upcast_attention=bool(upcast_attention),
                upcast_softmax=bool(upcast_softmax),
                cls_mode=bool(cls_mode),
                pdnorm_bn=bool(pdnorm_bn),
                pdnorm_ln=bool(pdnorm_ln),
                pdnorm_decouple=bool(pdnorm_decouple),
                pdnorm_adaptive=bool(pdnorm_adaptive),
                pdnorm_affine=bool(pdnorm_affine),
                pdnorm_conditions=tuple(str(value) for value in pdnorm_conditions),
            )
        elif self.backend == "zeros":
            self.point_encoder = None
        else:
            raise ValueError(f"Unsupported backend={self.backend}")


    def _normalize_feature_names(self, feature_names: Sequence[str]) -> tuple[str, ...]:
        """
        检查 feature_names, 若没问题("point_{?}"的形式), 也就是属于: "point_enc{stage_idx}", "point_dec{stage_idx}", "point_feat"(最终输出), "point_input_feat"(输入), "point_embed"(输入的简单嵌入), 就做拷贝, 否则报错.

        输入参数:
            - feature_names: Sequence[str], 请求导出的命名点特征列表

        输出:
            - normalized_feature_names: tuple[str, ...], 归一化后的点特征变量名列表
        """
        normalized_feature_names = tuple(str(feature_name) for feature_name in feature_names)
        unknown_feature_names = [
            feature_name
            for feature_name in normalized_feature_names
            if feature_name not in self.available_feature_names
        ]
        if unknown_feature_names:
            raise KeyError(
                f"Unknown point feature names: {unknown_feature_names}, "
                f"available={self.available_feature_names}"
            )
        return normalized_feature_names

    def _export_point_state(self, point_like: Any) -> dict[str, Any]:
        """
        从 Point 对象中导出 atom head 需要复用的点状态(coord, batch, offset, grid_size, 可选grid_coord)

        输入参数:
            - point_like: Any, 当前点状态对象

        输出:
            - point_state: dict[str, Any]
                - `"coord"`: torch.Tensor, `(N, 3)`, 点坐标
                - `"batch"`: torch.Tensor, `(N,)`, 点所属 batch 索引
                - `"offset"`: torch.Tensor, `(B,)`, 点分段结束偏移
                - `"grid_size"`: float, 当前点云 grid size
                - `"grid_coord"`: torch.Tensor, `(N, 3)`, 可选的离散网格坐标
        """
        point_state = {
            "coord": point_like["coord"],
            "batch": point_like["batch"],
            "offset": point_like["offset"],
            "grid_size": point_like["grid_size"],
        }
        if "grid_coord" in point_like:
            point_state["grid_coord"] = point_like["grid_coord"]
        return point_state


    def _apply_feature_hook(
        self,
        feature_name: str,
        point_like: Any,
        point_feature_hook: PointFeatureHook | None,

        feature_dict: dict[str, torch.Tensor],
        return_feature_names: tuple[str, ...],
    ) -> Any:
        """
        在命名点变量产生后，按变量名即时执行外部融合 hook，并按需导出对应点特征。

        输入参数:
            - feature_name: str, 当前点变量名
            - point_like: Any, 当前点状态对象
            - point_feature_hook: Callable | None, 外部融合 hook. 注意, 这里的 hook 实际上是固定了某些变元的函数 _fuse_point_variable, 这个函数在最终forward中被调用: 
                - 那些未固定而在此传入的变元是: 
                    - feature_name(当前的点变量名, 按照是否在主模块的 self.point_fusion_map 中, 直接返回原本的 point 对象 / 当前point与体素特征融合后的point对象);  
                    - point_like(当前的点对象, 包括位置、索引、特征等)
                - 那些在forward中固定的参数是: 
                    - voxel_output_dict(体素分支返回的结果字典);  
                    - batch(本batch所有的样本原信息, 如空间坐标和batch索引);  
                    - 仅用于记录的 sampled_point_fusion_feat_dict

            - feature_dict: dict[str, torch.Tensor], 仅用于记录: 如果 feature name 在 return_feature_names 中, 则新增条目: 键为当前变量名 feature name, 值为它的值
            - return_feature_names: tuple[str, ...], 仅用于记录: 需要记录的变量名列表

        输出:
            - point_like: Any, 可能已融合更新后的点状态对象
        """
        if point_feature_hook is not None:
            point_like = point_feature_hook(feature_name, point_like)
            if hasattr(point_like, "keys") and "sparse_conv_feat" in point_like.keys():
                point_like["sparse_conv_feat"] = point_like["sparse_conv_feat"].replace_feature(point_like.feat)

        if feature_name in return_feature_names:
            # torch.Tensor, `(N_current, C_current)`，当前变量名对应的点特征张量。
            feature_dict[feature_name] = point_like.feat
        return point_like


    def _forward_ptv3(
        self,
        point_input_feat: torch.Tensor,
        atom_coord_centered_world: torch.Tensor,
        atom_batch_index: torch.Tensor,
        atom_offsets: torch.Tensor,
        point_feature_hook: PointFeatureHook | None,
        return_feature_names: tuple[str, ...],
    ) -> dict[str, Any]:
        """
        以显式阶段变量的方式执行 PTV3 前向。

        输入参数:
            - point_input_feat: torch.Tensor, `(sumN, C_input)`, 点分支输入特征, 目前为49维特征向量
            - atom_coord_centered_world: torch.Tensor, `(sumN, 3)`, 点坐标
            - atom_batch_index: torch.Tensor, `(sumN,)`, 点所属 batch
            - atom_offsets: torch.Tensor, `(B,)`, 点分段结束偏移
            - point_feature_hook: Callable | None, 外部融合 hook. 注意, 这里的 hook 实际上是固定了某些变元的函数 _fuse_point_variable, 这个函数在最终forward中被调用: 
                - 那些未固定而在此传入的变元是: 
                    - feature_name(当前的点变量名, 按照是否在主模块的 self.point_fusion_map 中, 直接返回原本的 point 对象 / 当前point与体素特征融合后的point对象), forward 内部将遍历每个大的中间变量 
                    - point_like(当前的点对象, 包括位置、索引、特征等)
                - 那些在forward中固定的参数是: 
                    - voxel_output_dict(体素分支返回的结果字典);  
                    - batch(本batch所有的样本原信息, 如空间坐标和batch索引);  
                    - 仅用于记录的 sampled_point_fusion_feat_dict

            - return_feature_names: tuple[str, ...], 需要记录的点变量名列表

        输出:
            - output_dict: dict[str, Any]
                - `"point_feat"`: torch.Tensor, `(N_final, C_point)`, 最终点特征
                - `"point_state"`: dict[str, Any], 最终点状态
                - `"point_feature_dict"`: dict[str, torch.Tensor], 导出的命名点特征
        """
        point_feature_dict: dict[str, torch.Tensor] = {}
        if Point is None:
            raise ImportError("构造 Point 对象需要 PTV3 相关依赖。") from _PTV3_IMPORT_ERROR

        # 初始阶段
        # 初始化的 Point: `(sumN, C_input)` + 坐标/批次信息，进入 PTV3 前的基础点对象。
        point = Point(
            {
                "feat": point_input_feat,
                "coord": atom_coord_centered_world,
                "batch": atom_batch_index,
                "offset": atom_offsets,
                "grid_size": self.point_grid_size,
            }
        )
        point = self._apply_feature_hook(
            feature_name="point_input_feat",
            point_like=point,
            point_feature_hook=point_feature_hook,
            feature_dict=point_feature_dict,
            return_feature_names=return_feature_names,
        )
        # 序列化与稀疏化
        point.serialization(order=self.serialization_orders, shuffle_orders=self.shuffle_orders)
        point.sparsify()
        # (对原始输入做的) 初始嵌入, 将送入PTV3
        point = self.point_encoder.embedding(point)
        point = self._apply_feature_hook(
            feature_name="point_embed",
            point_like=point,
            point_feature_hook=point_feature_hook,
            feature_dict=point_feature_dict,
            return_feature_names=return_feature_names,
        )



        # 正式的 PTV3 阶段
        # encoder 阶段
        for stage_idx, (_, stage_module) in enumerate(self.point_encoder.enc._modules.items()):
            point = stage_module(point)
            point = self._apply_feature_hook(
                feature_name=self.enc_stage_names[stage_idx],  # 如 "point_enc4", "point_enc3", ...
                point_like=point,
                point_feature_hook=point_feature_hook,
                feature_dict=point_feature_dict,
                return_feature_names=return_feature_names,
            )

        # decoder 阶段
        if not self.cls_mode:
            for stage_name, stage_module in self.point_encoder.dec._modules.items():
                point = stage_module(point)
                point = self._apply_feature_hook(
                    feature_name=f"point_{stage_name}",  # 如 "point_dec3", "point_dec2", ...
                    point_like=point,
                    point_feature_hook=point_feature_hook,
                    feature_dict=point_feature_dict,
                    return_feature_names=return_feature_names,
                )
        # 对最终特征变量 point.feat 通过 hook 处理
        point = self._apply_feature_hook(
            feature_name="point_feat",
            point_like=point,
            point_feature_hook=point_feature_hook,
            feature_dict=point_feature_dict,
            return_feature_names=return_feature_names,
        )
        return {
            "point_feat": point.feat,
            "point_state": self._export_point_state(point_like=point),
            "point_feature_dict": point_feature_dict,
        }

    def build_zeros_output(
        self,
        atom_feat: torch.Tensor,
        atom_coord_centered_world: torch.Tensor,
        atom_batch_index: torch.Tensor,
        atom_offsets: torch.Tensor,
        return_feature_names: Sequence[str] | None = None,
        point_input_feat: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        """
        为 `backend="zeros"` 构造点分支占位输出。
        输入参数:
            - atom_feat: torch.Tensor, `(sumN, F_atom)`, 仅用于继承 device 与 dtype
            - atom_coord_centered_world: torch.Tensor, `(sumN, 3)`, 原子中心化世界坐标
            - atom_batch_index: torch.Tensor, `(sumN,)`, 每个原子所属 batch 索引
            - atom_offsets: torch.Tensor, `(B,)`, batch 视图下的结束偏移
            - return_feature_names: Sequence[str] | None, 需要导出的点变量名列表
            - point_input_feat: torch.Tensor | None, `(sumN, C_input)`, 可选的输入投影特征；仅用于兼容直接调用 point_backbone.forward() 时的调试导出

        输出:
            - output_dict: dict[str, Any]
                - `"point_feat"`: torch.Tensor, `(sumN, C_point)`, 全零点特征占位
                - `"point_state"`: dict[str, Any], atom head 仍需复用的点状态
                - `"point_recycle_out"`: torch.Tensor, `(sumN, C_point)`, recycle 占位输出
                - `"point_feature_dict"`: dict[str, torch.Tensor], 按需导出的中间结果
        """
        requested_feature_names = (
            tuple() if return_feature_names is None else self._normalize_feature_names(return_feature_names)
        )
        # torch.Tensor, `(sumN,)`, 点所属 batch 索引；保持与原始 voxel batch 轴一致
        point_batch_index = atom_batch_index.to(dtype=torch.long)
        # torch.Tensor, `(B,)`, PTV3 风格结束偏移
        point_offsets = atom_offsets.to(device=atom_batch_index.device, dtype=torch.long)
        atom_count = int(atom_feat.shape[0])
        # torch.Tensor, `(sumN, C_point)`, zeros 后端提供给 atom head 的全零占位特征
        point_feat = atom_feat.new_zeros((atom_count, self.out_channels))
        point_state = {
            "coord": atom_coord_centered_world,
            "batch": point_batch_index,
            "offset": point_offsets,
            "grid_size": self.point_grid_size,
        }
        point_feature_dict: dict[str, torch.Tensor] = {}
        if point_input_feat is not None and "point_input_feat" in requested_feature_names:
            point_feature_dict["point_input_feat"] = point_input_feat
        if "point_feat" in requested_feature_names:
            point_feature_dict["point_feat"] = point_feat
        return {
            "point_feat": point_feat,
            "point_state": point_state,
            "point_recycle_out": point_feat,
            "point_feature_dict": point_feature_dict,
        }


    def forward(
        self,
        atom_feat: torch.Tensor,
        atom_coord_centered_world: torch.Tensor,
        atom_batch_index: torch.Tensor,
        atom_offsets: torch.Tensor,
        recycle_in: torch.Tensor | None = None,
        point_feature_hook: PointFeatureHook | None = None,
        return_feature_names: Sequence[str] | None = None,
    ) -> dict[str, Any]:
        """
        执行一次点分支前向。

        输入参数:
            - atom_feat: torch.Tensor, `(sumN, F_atom)`, batch 内全部原子的底层特征
            - atom_coord_centered_world: torch.Tensor, `(sumN, 3)`, 以 BOX 中心为原点的世界坐标
            - atom_batch_index: torch.Tensor, `(sumN,)`, 每个原子所属的原始 batch 索引
            - atom_offsets: torch.Tensor, `(B,)`, 原始 batch 视图下的结束偏移
            - recycle_in: torch.Tensor | None, `(sumN, C_recycle)`, 点分支 recycle 输入
            - point_feature_hook: Callable | None, 外部融合 hook. 注意, 这里的 hook 实际上是固定了某些变元的函数 _fuse_point_variable, 这个函数在最终forward中被调用: 
                - 那些未固定而在此传入的变元是: 
                    - feature_name(当前的点变量名, 按照是否在主模块的 self.point_fusion_map 中, 直接返回原本的 point 对象 / 当前point与体素特征融合后的point对象), forward 内部将遍历每个大的中间变量 
                    - point_like(当前的点对象, 包括位置、索引、特征等)
                - 那些在forward中固定的参数是: 
                    - voxel_output_dict(体素分支返回的结果字典);  
                    - batch(本batch所有的样本原信息, 如空间坐标和batch索引);  
                    - 仅用于记录的 sampled_point_fusion_feat_dict

            - return_feature_names: tuple[str, ...], 需要记录的点变量名列表

        输出:
            - output_dict: dict[str, Any]
                - `"point_feat"`: torch.Tensor, `(N_final, C_point)`, 点分支最终特征
                - `"point_state"`: dict[str, Any], atom head 复用的点状态
                - `"point_recycle_out"`: torch.Tensor, `(N_final, C_point)`, 它=point_feat(不同于体素分支还需要经过简单投影)
                - `"point_feature_dict"`: dict[str, torch.Tensor], 导出的命名点特征
        """
        requested_feature_names = (
            tuple() if return_feature_names is None else self._normalize_feature_names(return_feature_names)
        )
        # torch.Tensor, `(sumN,)`, 点所属 batch 索引，保持与原始 voxel batch 轴一致。
        point_batch_index = atom_batch_index.to(dtype=torch.long)
        # torch.Tensor, `(B,)`, PTV3 风格结束偏移，保持原始 batch 视图不压缩。
        point_offsets = atom_offsets.to(device=atom_batch_index.device, dtype=torch.long)
        atom_count = int(atom_feat.shape[0])
        if recycle_in is None:
            # torch.Tensor, `(sumN, C_recycle)`, 点分支 recycle 输入的零初始化张量。
            point_recycle_in = atom_feat.new_zeros((atom_count, self.recycle_feature_dim))
        else:
            # torch.Tensor, `(sumN, C_recycle)`, 上一轮点分支 recycle 特征。
            point_recycle_in = recycle_in.to(device=atom_feat.device, dtype=atom_feat.dtype)
        # torch.Tensor, `(sumN, C_input=49)`, 输入到点主干前的投影特征。
        point_input_feat = self.atom_input_proj(torch.cat([atom_feat, point_recycle_in], dim=-1))



        if atom_count == 0:
            # torch.Tensor, `(0, C_point)`, 空 batch 下的最终点特征。
            empty_point_feat = atom_feat.new_empty((0, self.out_channels))
            point_state = {
                "coord": atom_coord_centered_world,
                "batch": point_batch_index,
                "offset": point_offsets,
                "grid_size": self.point_grid_size,
            }
            point_feature_dict = {}
            if "point_input_feat" in requested_feature_names:
                point_feature_dict["point_input_feat"] = point_input_feat
            if "point_feat" in requested_feature_names:
                point_feature_dict["point_feat"] = empty_point_feat
            return {
                "point_feat": empty_point_feat,
                "point_state": point_state,
                "point_recycle_out": empty_point_feat,
                "point_feature_dict": point_feature_dict,
            }



        if self.backend == "zeros":
            return self.build_zeros_output(
                atom_feat=atom_feat,
                atom_coord_centered_world=atom_coord_centered_world,
                atom_batch_index=point_batch_index,
                atom_offsets=point_offsets,
                return_feature_names=requested_feature_names,
                point_input_feat=point_input_feat,
            )
        else:
            output_dict = self._forward_ptv3(
                point_input_feat=point_input_feat,
                atom_coord_centered_world=atom_coord_centered_world,
                atom_batch_index=point_batch_index,
                atom_offsets=point_offsets,
                point_feature_hook=point_feature_hook,
                return_feature_names=requested_feature_names,
            )
            point_feat = output_dict["point_feat"]
            point_state = output_dict["point_state"]
            point_feature_dict = output_dict["point_feature_dict"]

        return {
            "point_feat": point_feat,
            "point_state": point_state,
            "point_recycle_out": point_feat,             # 点分支这一轮的点特征
            "point_feature_dict": point_feature_dict,
        }
