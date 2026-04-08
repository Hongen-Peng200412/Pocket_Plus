from __future__ import annotations

from typing import Sequence

import torch
from torch import nn

from .raunet import SimpleUnet


class Stage1VoxelBackbone(SimpleUnet):
    def __init__(
        self,
        in_channels: int | None,
        feature_channels: int,
        planes: Sequence[int],
        gradient_checkpoint: bool,
        return_feature_keys: Sequence[str],
        aux_head_hidden_channels: int,
        prior_prob: float | None = None,  # float|None, RetinaNet 式先验正类概率; 不为 None 时初始化 voxel_aux_head[-1].bias
    ) -> None:
        """
        Stage1 体素主干网络。

        输入参数:
            - in_channels: int | None，体素输入通道数；若为 None，表示由上层显式调用 `set_input_channels()` 设定
            - feature_channels: int，最终变量 `final` 的输出通道数
            - planes: Sequence[int]，长度固定为 9 的 RAUNet 通道配置，顺序为 `(enc0, enc1, enc2, enc3, bottleneck, dec3, dec2, dec1, dec0)`
            - gradient_checkpoint: bool，是否在重模块上启用 activation checkpoint
            - return_feature_keys: Sequence[str]，默认返回的命名字典键；命名规则固定为 `voxel_{变量名}` 如 `["voxel_ds_4", "voxel_c4", "voxel_final"]`
            - aux_head_hidden_channels: int，体素辅助监督头的隐藏通道数

        输出:
            - forward() 返回 dict[str, torch.Tensor | dict[str, torch.Tensor]]
                - `"voxel_features"`: dict[str, torch.Tensor]，当前请求导出的命名体素特征
                - `"voxel_logits_aux"`: torch.Tensor，`(B, 1, D, H, W)`，体素辅助监督 logits
                - `"voxel_recycle_out"`: torch.Tensor，`(B, C_recycle, D, H, W)`，voxel_final的简单投影，下一轮 recycle 的输入

        说明:
            - `_forward_single_pass()` 会按局部变量名自动收集可导出的 5D 体素特征，返回键统一写成 `voxel_{变量名}`。因此，若后续想额外返回某个中间变量，只需在 `return_feature_keys` 中加入对应的 `voxel_{变量名}` 字符串即可。
            - `feature_channels_by_name` 只维护当前明确参与融合的常用变量通道信息；它仅用于调试罢了，不需要同步维护这张表。
        """
        super().__init__(
            in_channels=in_channels,
            out_channels=int(feature_channels),
            planes=planes,
            gradient_checkpoint=gradient_checkpoint,
        )

        self.feature_channels = int(feature_channels)
        self.return_feature_keys = tuple(str(key_name) for key_name in return_feature_keys)

        enc0, enc1, enc2, enc3, bottleneck, dec3, dec2, dec1, dec0 = [int(value) for value in planes]
        # dict[str, int]，命名体素变量到通道数的映射；仅用于调试罢了
        self.feature_channels_by_name = {
            "voxel_ds_0": int(enc0 * 4),
            "voxel_ds_1": enc1,
            "voxel_ds_2": enc2,
            "voxel_ds_3": enc3,
            "voxel_ds_4": bottleneck,
            "voxel_c4": bottleneck,
            "voxel_c3": dec3,
            "voxel_c2": dec2,
            "voxel_c1": dec1,
            "voxel_c0": dec0,
            "voxel_final": self.feature_channels,
        }

        # nn.Sequential | None，`(B, C_final, D, H, W) -> (B, 1, D, H, W)`，体素辅助监督头。
        if int(aux_head_hidden_channels) > 0:
            self.voxel_aux_head = nn.Sequential(
                nn.Conv3d(self.feature_channels, int(aux_head_hidden_channels), kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv3d(int(aux_head_hidden_channels), int(aux_head_hidden_channels), kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv3d(int(aux_head_hidden_channels), 1, kernel_size=1),
            )
        else:   # aux_head_hidden_channels <= 0 时不构建(消融模式: 直接用 voxel_final 作为 logit, 节省显存)
            self.voxel_aux_head = None

        # RetinaNet 式偏置初始化: -log((1-π)/π)
        # 有 voxel_aux_head 时: 初始化 voxel_aux_head[4].bias
        #   结构: [0]Conv3d [1]ReLU [2]Conv3d [3]ReLU [4]Conv3d(out=1) ← 目标
        # 无 voxel_aux_head 时 (消融模式, aux_head_hidden_channels<=0):
        #   logit 来自 voxel_final = conv_end 输出 → 初始化 self.conv_end.bias
        if prior_prob is not None:
            import math as _math
            _bias_val = -_math.log((1.0 - float(prior_prob)) / float(prior_prob))
            if self.voxel_aux_head is not None:
                nn.init.constant_(self.voxel_aux_head[4].bias, _bias_val)
            elif self.conv_end.bias is not None:
                nn.init.constant_(self.conv_end.bias, _bias_val)

        # nn.Conv3d，`(B, C_final, D, H, W) -> (B, C_recycle, D, H, W)`，体素 voxel_final 的简单投影。
        self.voxel_recycle_proj = nn.Conv3d(
            in_channels=self.feature_channels,
            out_channels=int(self.shortconvadd.output_channels),
            kernel_size=1,
        )


    def _forward_single_pass(
        self,
        voxel_grid: torch.Tensor,
        recycle_in: torch.Tensor | None,
    ) -> dict[str, torch.Tensor]:
        """
        执行一次体素分支前向。

        输入参数:
            - voxel_grid: torch.Tensor，`(B, C_in, D, H, W)`，当前轮的体素输入网格
            - recycle_in: torch.Tensor | None，`(B, C_recycle, D, H, W)`，上一轮的体素 recycle 特征

        输出:
            - all_feature_dict: dict[str, torch.Tensor]，按局部变量名自动构造的命名字典
        """
        batch_size, _, depth, height, width = voxel_grid.shape
        if recycle_in is None:
            # torch.Tensor，`(B, C_recycle, D, H, W)`，体素 recycle 输入的零初始化张量。
            voxel_recycle = torch.zeros(
                (batch_size, int(self.shortconvadd.output_channels), depth, height, width),
                device=voxel_grid.device,
                dtype=voxel_grid.dtype,
            )
        else:
            # torch.Tensor，`(B, C_recycle, D, H, W)`，上一轮传入的体素 recycle 特征。
            voxel_recycle = recycle_in.to(device=voxel_grid.device, dtype=voxel_grid.dtype)

        # torch.Tensor，`(B, C_fused, D, H, W)`，输入体素与 recycle 特征融合后的结果。
        fused_input = self.shortconvadd(voxel_grid, voxel_recycle)

        # torch.Tensor，`(B, C_ds0, D, H, W)`，编码器第 0 层输出。
        ds_0 = self.shortconv1(fused_input)
        # torch.Tensor，`(B, C_ds1, D/2, H/2, W/2)`，第 1 次下采样输出。
        ds_1 = self._checkpoint_call(self.downsample1, ds_0)
        # torch.Tensor，`(B, C_ds2, D/4, H/4, W/4)`，第 2 次下采样输出。
        ds_2 = self._checkpoint_call(self.downsample2, ds_1)
        # torch.Tensor，`(B, C_ds3, D/8, H/8, W/8)`，第 3 次下采样输出。
        ds_3 = self._checkpoint_call(self.downsample3, ds_2)
        # torch.Tensor，`(B, C_ds4, D/16, H/16, W/16)`，最深层的瓶颈前编码特征。
        ds_4 = self._checkpoint_call(self.downsample4, ds_3)

        # torch.Tensor，`(B, C_c4, D/16, H/16, W/16)`，经过瓶颈注意力后的深层语义特征。
        c4 = self._checkpoint_call(self.A_block, ds_4)

        # torch.Tensor，`(B, C_c3, D/8, H/8, W/8)`，解码第 1 层输出。
        c3 = self._checkpoint_call(self.main1, self.upsample_add(c4, ds_3))
        # torch.Tensor，`(B, C_c2, D/4, H/4, W/4)`，解码第 2 层输出。
        c2 = self._checkpoint_call(self.main2, self._checkpoint_call(self.attn2, c3, ds_2))
        # torch.Tensor，`(B, C_c1, D/2, H/2, W/2)`，解码第 3 层输出。
        c1 = self._checkpoint_call(self.main3, self._checkpoint_call(self.attn3, c2, ds_1))
        # torch.Tensor，`(B, C_c0, D, H, W)`，解码第 4 层输出。
        c0 = self._checkpoint_call(self.main4, self._checkpoint_call(self.attn4, c1, ds_0))

        # torch.Tensor，`(B, C_branch, D, H, W)`，3x3 卷积分支特征。
        f3 = self.conv_end_3(c0)
        # torch.Tensor，`(B, C_branch, D, H, W)`，5x5 卷积分支特征。
        f5 = self.conv_end_5(c0)
        # torch.Tensor，`(B, C_branch, D, H, W)`，7x7 卷积分支特征。
        f7 = self.conv_end_7(c0)
        # torch.Tensor，`(B, 3*C_branch, D, H, W)`，多尺度分支拼接结果。
        fused_multiscale = self.relu1(torch.cat((f3, f5, f7), dim=1))
        # torch.Tensor，`(B, C_final, D, H, W)`，最终高分辨率体素特征。
        final = self.conv_end(fused_multiscale)

        excluded_feature_name_set = {
            "self",
            "voxel_grid",
            "recycle_in",
            "batch_size",
            "depth",
            "height",
            "width",
            "voxel_recycle",
        }
        local_variable_dict = dict(locals())
        # dict[str, torch.Tensor]，按局部变量名自动收集的命名体素特征字典, 前面加了前缀"voxel_"
        all_feature_dict = {
            f"voxel_{variable_name}": variable_value
            for variable_name, variable_value in local_variable_dict.items()
            if variable_name not in excluded_feature_name_set
            and isinstance(variable_value, torch.Tensor)
            and variable_value.ndim == 5
        }
        return all_feature_dict


    def forward_features(
        self,
        voxel_grid: torch.Tensor,
        recycle_in: torch.Tensor | None,
        return_feature_keys: Sequence[str],
    ) -> dict[str, torch.Tensor]:
        """
        返回指定命名体素特征。

        输入参数:
            - voxel_grid: torch.Tensor，`(B, C_in, D, H, W)`，当前轮体素输入网格
            - recycle_in: torch.Tensor | None，`(B, C_recycle, D, H, W)`，当前轮体素 recycle 特征
            - return_feature_keys: Sequence[str]，本次需要返回的命名字典键，例如 `["voxel_ds_4", "voxel_c4", "voxel_final"]`

        输出:
            - selected_feature_dict: dict[str, torch.Tensor]，按请求筛选后的体素特征字典
        """
        all_feature_dict = self._forward_single_pass(voxel_grid=voxel_grid, recycle_in=recycle_in)
        requested_feature_keys = tuple(str(key_name) for key_name in return_feature_keys)

        return {key_name: all_feature_dict[key_name] for key_name in requested_feature_keys}


    def forward(
        self,
        voxel_grid: torch.Tensor,
        recycle_in: torch.Tensor | None = None,
        return_feature_keys: Sequence[str] | None = None,
    ) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        """
        执行一次体素分支前向，并返回命名字典、辅助监督与 recycle 特征。

        输入参数:
            - voxel_grid: torch.Tensor，`(B, C_in, D, H, W)`，当前轮体素输入网格
            - recycle_in: torch.Tensor | None，`(B, C_recycle, D, H, W)`，当前轮体素 recycle 特征
            - return_feature_keys: Sequence[str] | None，本次需要返回的命名字典键, 例如 `["voxel_ds_4", "voxel_c4", "voxel_final"]`；若为 None，则使用init初始化时给定的 `self.return_feature_keys`

        输出:
            - output_dict: dict[str, torch.Tensor | dict[str, torch.Tensor]]
                - `"voxel_features"`: dict[str, torch.Tensor]，当前请求导出的命名体素特征
                - `"voxel_logits_aux"`: torch.Tensor，`(B, 1, D, H, W)`，体素辅助监督 logits
                - `"voxel_recycle_out"`: torch.Tensor，`(B, C_recycle, D, H, W)`，voxel_final的简单投影，下一轮 recycle 的输入
        """
        requested_feature_keys = (
            self.return_feature_keys
            if return_feature_keys is None
            else tuple(str(key_name) for key_name in return_feature_keys)
        )
        all_feature_dict = self._forward_single_pass(voxel_grid=voxel_grid, recycle_in=recycle_in)
        selected_feature_dict = {key_name: all_feature_dict[key_name] for key_name in requested_feature_keys}

        # torch.Tensor，`(B, 1, D, H, W)`，体素辅助监督 logits。
        # voxel_aux_head 为 None 时直接用 voxel_final 作为 logit(需 feature_channels=1)
        if self.voxel_aux_head is not None:
            voxel_logits_aux = self.voxel_aux_head(all_feature_dict["voxel_final"])
        else:
            voxel_logits_aux = all_feature_dict["voxel_final"]
        # torch.Tensor，`(B, C_recycle, D, H, W)`，下一轮体素 recycle 输入。
        voxel_recycle_out = self.voxel_recycle_proj(all_feature_dict["voxel_final"])  # 简单的1x1卷积

        return {
            "voxel_features": selected_feature_dict,
            "voxel_logits_aux": voxel_logits_aux,
            "voxel_recycle_out": voxel_recycle_out,   # voxel_final 经过简单投影
        }
