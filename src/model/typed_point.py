"""
Typed point 配置与 real/pseudo 分参辅助函数。
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields
from typing import Any

import torch
from torch import nn


@dataclass(frozen=True)
class TypedPointConfig:
    """
    typed point 全局配置。

    输入参数:
        - enabled: bool, 是否启用 typed point 分参的总闸; False 时所有 use_separate_* 均为 False
        - separate_qkv: bool, 是否分离 attention QKV projection
        - separate_attn_proj: bool, 是否分离 attention output projection
        - separate_ffn: bool, 是否分离 Block FFN
        - separate_cpe: bool, 是否分离 CPE 并使用同类子图
        - separate_embedding: bool, 是否分离 PointConvEmbedding 并使用同类子图
        - separate_pooling_proj: bool, 是否分离 SerializedPooling projection path
        - separate_unpooling_proj: bool, 是否分离 SerializedUnpooling projection path
        - separate_fusion: bool, 是否分离 voxel-to-point fusion MLP
        - separate_point_input_proj: bool, 是否分离 point backbone input projection
        - separate_atom_token_proj: bool, 是否分离 atom head token projection
    """

    enabled: bool = True
    separate_qkv: bool = True
    separate_attn_proj: bool = True
    separate_ffn: bool = True
    separate_cpe: bool = True
    separate_embedding: bool = True
    separate_pooling_proj: bool = True
    separate_unpooling_proj: bool = True
    separate_fusion: bool = True
    separate_point_input_proj: bool = True
    separate_atom_token_proj: bool = True

    @property
    def use_separate_qkv(self) -> bool:
        return self.enabled and self.separate_qkv

    @property
    def use_separate_attn_proj(self) -> bool:
        return self.enabled and self.separate_attn_proj

    @property
    def use_separate_ffn(self) -> bool:
        return self.enabled and self.separate_ffn

    @property
    def use_separate_cpe(self) -> bool:
        return self.enabled and self.separate_cpe

    @property
    def use_separate_embedding(self) -> bool:
        return self.enabled and self.separate_embedding

    @property
    def use_separate_pooling_proj(self) -> bool:
        return self.enabled and self.separate_pooling_proj

    @property
    def use_separate_unpooling_proj(self) -> bool:
        return self.enabled and self.separate_unpooling_proj

    @property
    def use_separate_fusion(self) -> bool:
        return self.enabled and self.separate_fusion

    @property
    def use_separate_point_input_proj(self) -> bool:
        return self.enabled and self.separate_point_input_proj

    @property
    def use_separate_atom_token_proj(self) -> bool:
        return self.enabled and self.separate_atom_token_proj


def normalize_typed_point_cfg(cfg: Mapping[str, Any] | TypedPointConfig | None) -> TypedPointConfig:
    """
    将 Hydra/OmegaConf 字典或 dataclass 解析为 TypedPointConfig。

    输入参数:
        - cfg: Mapping[str, Any] | TypedPointConfig | None, typed point 配置; None 表示使用默认开启配置

    输出:
        - typed_cfg: TypedPointConfig, 校验后的 typed point 配置
    """
    if cfg is None:
        return TypedPointConfig()
    if isinstance(cfg, TypedPointConfig):
        return cfg

    # set[str], TypedPointConfig 接受的字段名集合
    valid_keys = {field.name for field in fields(TypedPointConfig)}
    # dict[str, Any], 从 Mapping 拷贝出的配置字段
    cfg_dict = dict(cfg)
    # list[str], 配置中不属于 TypedPointConfig 的字段
    unknown_keys = sorted(set(cfg_dict) - valid_keys)
    if unknown_keys:
        raise ValueError(f"typed_point_cfg 包含未知字段: {unknown_keys}")
    for key, value in cfg_dict.items():
        if not isinstance(value, bool):
            raise ValueError(f"typed_point_cfg.{key} 必须是 bool, 当前为 {type(value).__name__}")
    return TypedPointConfig(**cfg_dict)


def validate_pseudo_mask(
    pseudo_mask: torch.Tensor | None,
    point_count: int,
    *,
    name: str,
) -> torch.Tensor | None:
    """
    校验 real/pseudo 点类型掩码, 不报错则原样返回 pseudo_mask 。

    输入参数:
        - pseudo_mask: torch.Tensor | None, (N_all,), bool, True 表示 P anchor; None 表示未携带 type 标注
        - point_count: int, 当前点数 N_all
        - name: str, 报错信息中的调用点名称

    输出:
        - pseudo_mask: torch.Tensor | None, 校验后的掩码
    """
    if pseudo_mask is None:
        return None
    if pseudo_mask.dtype != torch.bool:
        raise RuntimeError(f"{name}: pseudo_mask 必须是 bool dtype。")
    if pseudo_mask.ndim != 1:
        raise RuntimeError(f"{name}: pseudo_mask 必须是一维张量。")
    if int(pseudo_mask.shape[0]) != int(point_count):
        raise RuntimeError(f"{name}: pseudo_mask 长度必须为 {point_count}, 当前为 {pseudo_mask.shape[0]}。")
    return pseudo_mask


def split_mask_state(
    pseudo_mask: torch.Tensor | None,
    point_count: int,
    *,
    name: str,
) -> tuple[torch.Tensor | None, bool, bool]:
    """
    校验 pseudo_mask 并返回 real/pseudo 子集存在状态。

    输入参数:
        - pseudo_mask: torch.Tensor | None, (N_all,), bool, True 表示 P anchor
        - point_count: int, 当前点数 N_all
        - name: str, 报错信息中的调用点名称

    输出:
        - validated_mask: torch.Tensor | None, 校验后的掩码(无错则 = pseudo_mask)
        - has_real: bool, 当前点集中是否存在 real 点
        - has_pseudo: bool, 当前点集中是否存在 pseudo 点
    """
    validated_mask = validate_pseudo_mask(pseudo_mask, point_count, name=name)
    if validated_mask is None:
        return None, point_count > 0, False
    # bool, 当前点集中是否存在 pseudo 点
    has_pseudo = bool(validated_mask.any().item())
    # bool, 当前点集中是否存在 real 点
    has_real = bool((~validated_mask).any().item())
    return validated_mask, has_real, has_pseudo


def merge_type_aware_tensor_outputs(
    real_y: torch.Tensor,
    pseudo_y: torch.Tensor,
    pseudo_mask: torch.Tensor,
) -> torch.Tensor:
    """
    将 real/pseudo 子集输出按 mixed 顺序恢复，并保留分支计算产生的精度。

    输入参数:
        - real_y: torch.Tensor, (N_real, ...), real 分支输出特征
        - pseudo_y: torch.Tensor, (N_pseudo, ...), pseudo 分支输出特征
        - pseudo_mask: torch.Tensor, (N_real + N_pseudo,), bool, True 表示 P anchor

    输出:
        - y: torch.Tensor, (N_real + N_pseudo, ...), 按原 mixed 顺序排列的输出特征；
          dtype 为两条分支输出 dtype 的提升结果
    """
    # torch.dtype, AMP 下由两类分支输出提升得到的恢复精度
    output_dtype = torch.promote_types(real_y.dtype, pseudo_y.dtype)
    # torch.Tensor, (N_all, ...), mixed 顺序下的 typed 输出
    y = torch.empty(
        (int(pseudo_mask.shape[0]),) + tuple(real_y.shape[1:]),
        dtype=output_dtype,
        device=real_y.device,
    )
    y[~pseudo_mask] = real_y.to(dtype=output_dtype)
    y[pseudo_mask] = pseudo_y.to(dtype=output_dtype)
    return y


def apply_type_aware_tensor_module(
    x: torch.Tensor,
    pseudo_mask: torch.Tensor | None,
    real_module: nn.Module,
    pseudo_module: nn.Module,
) -> torch.Tensor:
    """
    按 pseudo_mask 对输入张量 x 调用 real/pseudo tensor module 并恢复原顺序。

    输入参数:
        - x: torch.Tensor, (N_all, C_in), mixed 或 real-only 输入特征
        - pseudo_mask: torch.Tensor | None, (N_all,), bool, True 表示 P anchor; None 表示全部走 real 分支
        - real_module: nn.Module, real 分支模块
        - pseudo_module: nn.Module, pseudo 分支模块

    输出:
        - y: torch.Tensor, (N_all, C_out), 按输入点顺序排列的输出特征
    """
    if int(x.shape[0]) == 0:
        return real_module(x)
    pseudo_mask = validate_pseudo_mask(
        pseudo_mask,
        int(x.shape[0]),
        name="apply_type_aware_tensor_module",
    )
    if pseudo_mask is None or not bool(pseudo_mask.any().item()):
        return real_module(x)
    if bool(pseudo_mask.all().item()):
        return pseudo_module(x)

    # torch.Tensor, (N_real, C_in), real 子集输入特征
    real_x = x[~pseudo_mask]
    # torch.Tensor, (N_pseudo, C_in), pseudo 子集输入特征
    pseudo_x = x[pseudo_mask]
    # torch.Tensor, (N_real, C_out), real 分支输出特征
    real_y = real_module(real_x)
    # torch.Tensor, (N_pseudo, C_out), pseudo 分支输出特征
    pseudo_y = pseudo_module(pseudo_x)
    return merge_type_aware_tensor_outputs(real_y, pseudo_y, pseudo_mask)


def make_typed_linear_norm_act(
    in_channels: int,
    out_channels: int,
    act_layer: type[nn.Module] | None,
) -> nn.Sequential:
    """
    构造 typed path 专用的 Linear-LayerNorm-Act tensor module。

    输入参数:
        - in_channels: int, 输入通道数
        - out_channels: int, 输出通道数
        - act_layer: type[nn.Module] | None, 激活函数类; None 表示不加激活

    输出:
        - module: nn.Sequential, (N_all, in_channels) -> (N_all, out_channels) 的 tensor module
    """
    # list[nn.Module], typed projection 的模块序列
    modules: list[nn.Module] = [nn.Linear(in_channels, out_channels), nn.LayerNorm(out_channels)]
    if act_layer is not None:
        modules.append(act_layer())
    return nn.Sequential(*modules)
