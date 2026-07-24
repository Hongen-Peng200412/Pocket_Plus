"""运行 Stage1 的三维体素编码器、解码器和独立预测头。

主要入口是 :meth:`Stage1VoxelBackbone.forward`。它接收
``(B, C_in, D, H, W)`` 体素特征，返回指定分辨率的命名特征、受体结合区域与
配体区域预测，以及按配置启用的蛋白主链、核酸主链和配体距离预测。
本模块只计算张量，不写文件；损失与指标由 ``voxel_point_stage1.py`` 解释。
"""

from __future__ import annotations

from typing import Sequence

import torch
from torch import nn

from src.auxiliary_supervision import (
    NUCLEIC_MAINCHAIN_CLASS_NAMES,
    NUCLEIC_MAINCHAIN_PRIORS,
    PROTEIN_MAINCHAIN_CLASS_NAMES,
    PROTEIN_MAINCHAIN_PRIORS,
)
from src.utils.bias_init import init_classification_head_bias

from .raunet import SimpleUnet


class Stage1VoxelBackbone(SimpleUnet):
    """组合 RAUNet 体素主干、循环特征投影和彼此独立的两层 ``1×1×1`` 预测头。

    ``voxel_final`` 是所有预测头共享的最高分辨率特征。结构辅助监督启用时，
    蛋白和核酸输出通道分别严格对应
    ``PROTEIN_MAINCHAIN_CLASS_NAMES`` 与 ``NUCLEIC_MAINCHAIN_CLASS_NAMES``；
    配体距离头输出一个 logit，损失函数对其取 sigmoid 后拟合反距离监督值。
    """

    def __init__(
        self,
        in_channels: int | None,
        feature_channels: int,
        planes: Sequence[int],
        gradient_checkpoint: bool,
        return_feature_keys: Sequence[str],
        aux_head_hidden_channels: int | None = None,
        num_conv3d_aux: int | None = None,
        ligand_head_hidden_channels: int | None = None,
        num_conv3d_ligand: int | None = None,
        enable_multiscale_output: bool = True,
        enable_structure_heads: bool = False,
        prior_prob: float | None = None,  # float|None, legacy 单通道 sigmoid 正类先验概率; 命名先验缺省时作为 aux/ligand 头兜底
        voxel_aux_logit_dim: int = 1,
        voxel_ligand_logit_dim: int = 1,
        prior_probs: Sequence[float] | None = None,
        prior_prob_voxel_receptor: float | None = None,  # float|None, voxel aux(受体区域)头单通道 sigmoid 先验; 由 stage1_model 注入
        prior_prob_voxel_ligand: float | None = None,    # float|None, voxel ligand 头单通道 sigmoid 先验; 由 stage1_model 注入
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
                - `"voxel_logits_ligand"`: torch.Tensor，`(B, 1, D, H, W)`，配体区域 logits
                - `"voxel_logits_protein"`: torch.Tensor | None，蛋白固定类别 logits
                - `"voxel_logits_nucleic"`: torch.Tensor | None，核酸固定类别 logits
                - `"voxel_logits_distance"`: torch.Tensor | None，配体反距离 logit
                - `"voxel_recycle_out"`: torch.Tensor，`(B, C_recycle, D, H, W)`，voxel_final的简单投影，下一轮 recycle 的输入

        说明:
            - `_forward_single_pass()` 显式构造可导出的五维体素特征字典，键统一写成 `voxel_{变量名}`。
            - `feature_channels_by_name` 记录可供融合或诊断代码查询的命名特征通道数，不参与前向数值计算。
        """
        super().__init__(
            in_channels=in_channels,
            out_channels=int(feature_channels),
            planes=planes,
            gradient_checkpoint=gradient_checkpoint,
            enable_multiscale_output=enable_multiscale_output,
        )

        self.feature_channels = int(feature_channels)
        self.voxel_aux_logit_dim = int(voxel_aux_logit_dim)
        self.voxel_ligand_logit_dim = int(voxel_ligand_logit_dim)
        self.voxel_protein_logit_dim = len(PROTEIN_MAINCHAIN_CLASS_NAMES)
        self.voxel_nucleic_logit_dim = len(NUCLEIC_MAINCHAIN_CLASS_NAMES)
        self.return_feature_keys = tuple(str(key_name) for key_name in return_feature_keys)

        enc0, enc1, enc2, enc3, bottleneck, dec3, dec2, dec1, dec0 = [int(value) for value in planes]
        # dict[str, int]，命名体素特征到通道数的映射；不参与前向数值计算。
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

        # 旧实验 YAML 可能仍传入四个宽度/层数字段；新版固定结构不再读取它们。
        del aux_head_hidden_channels, num_conv3d_aux
        del ligand_head_hidden_channels, num_conv3d_ligand
        self.voxel_aux_head = self._build_voxel_head(self.voxel_aux_logit_dim)
        self.voxel_ligand_head = self._build_voxel_head(self.voxel_ligand_logit_dim)
        self.enable_structure_heads = bool(enable_structure_heads)
        self.voxel_protein_head = (
            self._build_voxel_head(self.voxel_protein_logit_dim)
            if self.enable_structure_heads
            else None
        )
        self.voxel_nucleic_head = (
            self._build_voxel_head(self.voxel_nucleic_logit_dim)
            if self.enable_structure_heads
            else None
        )
        self.voxel_distance_head = (
            self._build_voxel_head(1)
            if self.enable_structure_heads
            else None
        )

        if prior_probs is not None:
            init_classification_head_bias(self.voxel_aux_head, list(prior_probs))
            init_classification_head_bias(self.voxel_ligand_head, list(prior_probs))
        else:
            # 单通道 sigmoid 先验: aux(受体)与 ligand 头各自独立; 命名先验缺省时回退 legacy prior_prob; 均为 None 则不初始化
            # float | None, aux/ligand 头各自有效先验
            aux_prior = prior_prob_voxel_receptor if prior_prob_voxel_receptor is not None else prior_prob
            ligand_prior = prior_prob_voxel_ligand if prior_prob_voxel_ligand is not None else prior_prob
            if aux_prior is not None:
                if self.voxel_aux_logit_dim != 1:
                    raise ValueError("多通道 softmax head 请使用 prior_probs，不要使用单通道先验(voxel aux)")
                init_classification_head_bias(self.voxel_aux_head, float(aux_prior))
            if ligand_prior is not None:
                if self.voxel_ligand_logit_dim != 1:
                    raise ValueError("多通道 softmax head 请使用 prior_probs，不要使用单通道先验(voxel ligand)")
                init_classification_head_bias(self.voxel_ligand_head, float(ligand_prior))

        if self.enable_structure_heads:
            init_classification_head_bias(
                self.voxel_protein_head, list(PROTEIN_MAINCHAIN_PRIORS)
            )
            init_classification_head_bias(
                self.voxel_nucleic_head, list(NUCLEIC_MAINCHAIN_PRIORS)
            )
            init_classification_head_bias(self.voxel_distance_head, 1.0 / 11.0)

        # nn.Conv3d，`(B, C_final, D, H, W) -> (B, C_recycle, D, H, W)`，体素 voxel_final 的简单投影。
        self.voxel_recycle_proj = nn.Conv3d(
            in_channels=self.feature_channels,
            out_channels=int(self.shortconvadd.output_channels),
            kernel_size=1,
        )

    def _build_voxel_head(self, output_channels: int) -> nn.Sequential:
        """构造固定的 `1×1×1 卷积 → ReLU → 1×1×1 卷积` 输出头。"""

        output_channels = int(output_channels)
        if output_channels <= 0:
            raise ValueError("体素输出头的输出通道数必须为正。")
        return nn.Sequential(
            nn.Conv3d(self.feature_channels, self.feature_channels, kernel_size=1),
            nn.ReLU(),
            nn.Conv3d(self.feature_channels, output_channels, kernel_size=1),
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

        if self.enable_multiscale_output:
            # 四个张量的空间形状均为 (D, H, W)；f3/f5/f7 分别使用
            # 3×3×3、5×5×5、7×7×7 卷积，再拼接并压回 feature_channels。
            f3 = self.conv_end_3(c0)
            f5 = self.conv_end_5(c0)
            f7 = self.conv_end_7(c0)
            fused_multiscale = self.relu1(torch.cat((f3, f5, f7), dim=1))
            final = self.conv_end(fused_multiscale)
        else:
            final = c0

        # dict[str, torch.Tensor]，命名体素特征字典；显式列出以避免 torch.compile 下 locals() 丢失中间变量
        all_feature_dict = {
            "voxel_fused_input": fused_input,
            "voxel_ds_0": ds_0,
            "voxel_ds_1": ds_1,
            "voxel_ds_2": ds_2,
            "voxel_ds_3": ds_3,
            "voxel_ds_4": ds_4,
            "voxel_c4": c4,
            "voxel_c3": c3,
            "voxel_c2": c2,
            "voxel_c1": c1,
            "voxel_c0": c0,
            "voxel_final": final,
        }
        if self.enable_multiscale_output:
            all_feature_dict.update(
                voxel_f3=f3,
                voxel_f5=f5,
                voxel_f7=f7,
                voxel_fused_multiscale=fused_multiscale,
            )
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

        # torch.Tensor，`(B, C_aux, D, H, W)`，受体结合区域监督 logits。
        voxel_logits_aux = self.voxel_aux_head(all_feature_dict["voxel_final"])
        # (B, C_ligand, D, H, W) 或 None，配体区域预测 logits。
        voxel_logits_ligand = None
        if self.voxel_ligand_head is not None:
            voxel_logits_ligand = self.voxel_ligand_head(all_feature_dict["voxel_final"])
        voxel_logits_protein = (
            self.voxel_protein_head(all_feature_dict["voxel_final"])
            if self.voxel_protein_head is not None
            else None
        )
        # 上述两个多分类张量的类别维依次对应 auxiliary_supervision.py 中的类别名称；
        # 背景类别固定为通道 0。未启用结构辅助监督时二者均为 None。
        voxel_logits_nucleic = (
            self.voxel_nucleic_head(all_feature_dict["voxel_final"])
            if self.voxel_nucleic_head is not None
            else None
        )
        voxel_logits_distance = (
            self.voxel_distance_head(all_feature_dict["voxel_final"])
            if self.voxel_distance_head is not None
            else None
        )
        # (B, 1, D, H, W) 或 None；sigmoid 后解释为 1 / (1 + 最近配体原子距离_Å)。

        # torch.Tensor，`(B, C_recycle, D, H, W)`，下一轮体素 recycle 输入。
        voxel_recycle_out = self.voxel_recycle_proj(all_feature_dict["voxel_final"])  # 简单的1x1卷积

        return {
            "voxel_features": selected_feature_dict,
            "voxel_logits_aux": voxel_logits_aux,
            "voxel_logits_ligand": voxel_logits_ligand,
            "voxel_logits_protein": voxel_logits_protein,
            "voxel_logits_nucleic": voxel_logits_nucleic,
            "voxel_logits_distance": voxel_logits_distance,
            "voxel_recycle_out": voxel_recycle_out,   # voxel_final 经过简单投影
        }
