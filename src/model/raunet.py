"""
=============================================================================
RAUNet (Recycling Attention UNet) 模型文件
=============================================================================
本文件实现了一个用于3D体素数据的U-Net变体模型。该模型的核心特点是：
1. 使用多尺度特征金字塔结构
2. 使用注意力门控机制(Attention Gate)进行特征融合
3. 使用循环迭代(Recycling)机制提高预测精度
4. 融合了3D旋转位置编码(3D RoPE)的自注意力机制

输入输出说明:
- 输入: torch, (B, 13, D, H, W), 表示批量B个样本，每个样本有13通道的3D体素特征
- 输出: torch, (B, 1, D, H, W), 表示批量B个样本的单通道3D预测结果(如口袋检测概率图)
=============================================================================
"""
import torch
import einops
from torch import nn
import numpy as np
from torch.utils.checkpoint import checkpoint as torch_checkpoint
from einops.layers.torch import Rearrange
import contextlib
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from attention_3d_rope import *


class Bottleneck(nn.Module):
    """
    使用1x1卷积，3x3卷积，1x1卷积。
    
    输入参数 (Input Parameters):
        - in_planes: int, 输入特征图的通道数
        - planes: int, 中间层的基础通道数(瓶颈处通道数)
        - stride: int, 默认=1, 卷积步长，stride=2时进行下采样
        - groups: int, 默认=1, 分组卷积的组数
        - activation_class: nn.Module, 默认=nn.ReLU, 激活函数类
        - conv_class: nn.Module, 默认=nn.Conv3d, 卷积层类
        - affine: bool, 默认=False, InstanceNorm3d是否使用可学习的仿射参数
        - checkpoint: bool, 默认=False, 是否使用梯度检查点(节省显存)
        - **kwargs: 其他关键字参数
    
    类属性 (Class Attributes):
        - expansion: int = 4, 通道扩展倍数，输出通道数=planes * expansion
    
    输出 (Output):
        - forward返回: torch, (B, planes*expansion, D', H', W'), 其中D', H', W'由stride决定，stride=2时各维度减半
    """
    expansion = 4

    def __init__(
        self,
        in_planes,
        planes,
        stride=1,
        groups=1,
        activation_class=nn.ReLU,
        conv_class=nn.Conv3d,
        affine=False,
        checkpoint=False,
        **kwargs,
    ):
        super().__init__()
        self.activation_fn = activation_class()
        self.conv1 = conv_class(
            in_planes, planes, kernel_size=1, bias=False, groups=groups
        )
        self.norm1 = nn.InstanceNorm3d(planes, affine=affine)
        self.conv2 = conv_class(
            planes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
            groups=groups,
        )
        self.norm2 = nn.InstanceNorm3d(planes, affine=affine)
        self.conv3 = conv_class(
            planes, self.expansion * planes, kernel_size=1, bias=False, groups=groups
        )
        self.norm3 = nn.InstanceNorm3d(self.expansion * planes, affine=affine)

        # 当stride!=1或通道数不匹配时，需要用1x1卷积调整维度
        self.shortcut_conv = nn.Identity()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut_conv = nn.Conv3d(
                in_planes,
                self.expansion * planes,
                kernel_size=1,
                stride=stride,
                bias=False,
                groups=groups,
            )
        # 根据checkpoint参数选择是否使用梯度检查点
        self.forward = self.forward_checkpoint if checkpoint else self.forward_normal


    def forward_normal(self, x):
        """
        普通前向传播
        """
        out = self.activation_fn(self.norm1(self.conv1(x)))
        out = self.activation_fn(self.norm2(self.conv2(out)))
        out = self.norm3(self.conv3(out))
        out += self.shortcut_conv(x)
        out = self.activation_fn(out)
        return out

    def forward_checkpoint(self, x):
        """
        带梯度检查点的前向传播(节省显存)
        """
        return torch_checkpoint(self.forward_normal, x, preserve_rng_state=False)





class ConvBuildingBlock(nn.Module):
    """
    双层3x3卷积+最后残差, 空间维度不变
    """
    def __init__(self, in_channels:int, out_channels:int, activate_class:nn.Module=nn.ReLU):
        super().__init__()
        self.activate_function = activate_class()
        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm3d(out_channels,affine=True),
            self.activate_function,
            nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm3d(out_channels, affine=True),
        )
        
        self.shortcut_conv = nn.Identity()
        if in_channels != out_channels:
            self.shortcut_conv = nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=1,
                bias=True
            )
    
    def forward(self,x:torch.Tensor):
        return self.activate_function(self.conv1(x) + self.shortcut_conv(x))




class ShortConv(nn.Module):
    """
    简单3x3卷积块, 空间维度不变
    """
    def __init__(self,in_channels:int,out_channels:int):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.norm1 = nn.InstanceNorm3d(out_channels,affine=True)
        self.relu1 = nn.ReLU()
    
    def forward(self,x:torch.Tensor):
        y = self.norm1(self.conv1(x))
        y = self.relu1(y)
        return y




class ShortConvAdd(nn.Module):
    """
    2个特征 x_0, x_1 的融合. x_0 为原始输入特征(如13通道的输入), x_1 为循环传递的特征(如64通道的recycle特征)
    
    输入参数 (Input Parameters):
        - input_channels: int 或 None, 第一个输入(x0)的通道数; None 时使用 LazyConv3d 延迟初始化
        - output_channels: int, 输出通道数，同时也是第二个输入(x1)的通道数
    """
    def __init__(self, input_channels, output_channels: int):
        super().__init__()
        self.output_channels = output_channels
        if input_channels is None:
            # 延迟初始化: 在第一次 forward 时根据输入自动推断 in_channels
            self.conv1 = nn.LazyConv3d(output_channels, kernel_size=3, stride=1, padding=1, bias=False)
        else:
            self.conv1 = nn.Conv3d(input_channels, output_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.norm1 = nn.InstanceNorm3d(output_channels, affine=False)
        self.norm2 = nn.InstanceNorm3d(output_channels, affine=True)
        self.relu1 = nn.ELU()
    
    def forward(self,x0,x1):
        y = self.norm1(self.conv1(x0))
        y = self.relu1( y + self.norm2(x1))
        return y





class Res2NetBlock(nn.Module):
    """
    层级残差模块
    
    输入参数 (Input Parameters):
        - scale: int, 默认=4, 分割尺度数 or 层级数
    """
    def __init__(self, in_channels, out_channels, stride=1, scale=4,activate_class:nn.Module=nn.ReLU):
        super(Res2NetBlock, self).__init__()
        self.scale = scale
        self.conv1 = nn.Sequential(nn.Conv3d(in_channels, out_channels*self.scale, kernel_size=1, stride=1, padding=0, bias=False), nn.InstanceNorm3d(out_channels*self.scale, affine=True))
        self.norm1 = nn.InstanceNorm3d(out_channels*self.scale, affine=True)
        self.conv_list = nn.ModuleList([nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False) for _ in range(self.scale - 1)])
        self.activate_class = activate_class()
        self.conv2 = nn.Sequential(nn.Conv3d(out_channels*self.scale, out_channels, 1, 1, 0, bias=False), nn.InstanceNorm3d(out_channels, affine=True))
        
        self.shortcut_conv = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.shortcut_conv = nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=stride,
                bias=True
            )
    
    def forward(self, x):
        # x_list: tuple of torch tensors, 每个元素形状为(B, out_channels, D, H, W)
        x_list = self.activate_class(self.conv1(x)).chunk(self.scale,dim=1)  # 将扩展后的特征在通道维度上分割成scale份
        # y_list: list of torch tensors, 存储级联卷积的输出
        y_list = []
        for ii,xi in enumerate(x_list):
            if ii == 0:
                y_list.append(xi)
            elif ii == 1:
                y_list.append(self.conv_list[ii-1](xi))
            else:
                y_list.append(self.conv_list[ii-1](xi+y_list[-1]))
        
        y = self.conv2(self.activate_class(self.norm1(  torch.cat(y_list,dim=1)  )))
        y = self.activate_class(y+self.shortcut_conv(x))
        return y
    




class AttentionGate(nn.Module):
    """
    输入参数 (Input Parameters):
        - down_features: int, 下采样路径(跳跃连接)特征的通道数
        - up_features: int, 上采样路径特征的通道数
        - out_features: int, 输出特征的通道数
        - attention_features: int, 默认=64, 注意力计算中间特征的通道数
        - attention_heads: int, 默认=8, 注意力头的数量————规定 up_features = afz * ahz
    
    输出 (Output):
        - forward返回: torch, (B, out_features, D, H, W), 融合后的特征图
    """
    def __init__(self, down_features:int, up_features:int, out_features:int, attention_features:int=64, attention_heads:int=8):
        super(AttentionGate, self).__init__()
        self.dfz = down_features
        self.ufz = up_features
        self.ofz = out_features
        self.afz = attention_features
        self.ahz = attention_heads
        
        # (up_features) -> (attention_features)
        self.conv_q = nn.Sequential(nn.Conv3d(
            in_channels=self.ufz,
            out_channels=self.afz,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        ),nn.InstanceNorm3d(self.afz,affine=True))
        
        # (down_features) -> (attention_features)
        self.conv_k = nn.Sequential(nn.Conv3d(
            in_channels=self.dfz,
            out_channels=self.afz,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        ),nn.InstanceNorm3d(self.afz,affine=True))
        
        # (down_features) -> (up_features), 规定 up_features = afz * ahz
        self.conv_v = nn.Sequential(nn.Conv3d(
            in_channels=self.dfz,
            out_channels=self.ufz,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        ),nn.InstanceNorm3d(self.ufz,affine=True))
        
        # (attention_features) -> (attention_heads), 输出经Sigmoid归一化到[0,1]
        self.gate = nn.Sequential(
            nn.ReLU(),
            nn.Conv3d(
                in_channels=self.afz,
                out_channels=self.ahz,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True
            ),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU()
        # (up_features) -> (out_features)
        self.conv_back = ConvBuildingBlock(self.ufz, self.ofz)
    
    def forward(self,us,ds):
        """
        输入参数 (Input Parameters):
            - us: torch, (B, up_features, D_us, H_us, W_us), 上采样路径的特征(较粗分辨率)
            - ds: torch, (B, down_features, D, H, W), 跳跃连接的特征(较细分辨率)
        
        输出 (Output):
            - torch, (B, out_features, D, H, W), 注意力加权融合后的特征
        """
        ds_shape = ds.shape
        D, H, W = ds_shape[2:]
        # 将上采样特征插值到与跳跃连接相同的分辨率
        upsampled = nn.functional.interpolate(input=us, size=(D, H, W), mode='trilinear', align_corners=True)
        query = self.conv_q(upsampled)
        key = self.conv_k(ds)
        value = self.conv_v(ds)
        
        # value: torch, (B, afz, ahz, D, H, W)
        # 将value重排为多头形式，其中 up_features = afz * ahz
        value = einops.rearrange(value, "N (afz ahz) d h w -> N afz ahz d h w", ahz=self.ahz)
        
        # gate: torch, (B, ahz, D, H, W), 注意力权重，范围[0,1]
        # 通过query和key的加法融合计算得到
        gate = self.gate(query+key)  # N ahz d h w
        
        # out: torch, (B, afz, ahz, D, H, W)
        # 对value应用注意力权重(广播乘法)
        out = value*gate[:,None]
        
        # out: torch, (B, up_features, D, H, W)
        # 将多头输出重排回原始形式
        out = einops.rearrange(out,"N afz ahz d h w -> N (afz ahz) d h w", ahz=self.ahz)
        
        # 将注意力加权的输出与上采样特征相加，经ReLU激活后通过输出卷积块
        return self.conv_back(self.relu(out+upsampled))


















class SimpleUnet(nn.Module):
    """
    forward输入:
        - run_iters: int, 默认=3, 循环迭代次数
    
    输出 (Output):
        - forward返回: torch, (B, 1, D, H, W), 预测结果(如口袋检测概率图)
    """
    def __init__(
        self,
        in_channels: int = None,
        out_channels: int = 1,
        planes: list = (64, 256, 256, 256, 256, 256, 128, 64, 64),
        gradient_checkpoint: bool = False,
    ):
        super(SimpleUnet, self).__init__()
        # planes 映射映射固定为：(enc0, enc1, enc2, enc3, bottleneck, dec3, dec2, dec1, dec0)
        enc0, enc1, enc2, enc3, bottleneck, dec3, dec2, dec1, dec0 = planes
        self._planes_for_reinit = planes

        # 如果 in_channels 为 None，则由 wrapper 在 forward 时自动识别
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.gradient_checkpoint = bool(gradient_checkpoint)
        
        # 如果 self.in_channels 是 None，这里的第一个卷积层将在 forward 中或包装器中被延迟初始化或重新创建
        self.shortconvadd = ShortConvAdd(input_channels=in_channels, output_channels=enc0)   # 用于原始输入 or 循环融合
        self.shortconv1 = ShortConv(enc0, enc0 * 4)   # 经过上面的shortcovadd之后进行初始的 "特征提取"
        
        # ===== 编码器：4次下采样 =====
        # downsample1: Bottleneck, 第1次下采样，(enc0*4) -> (enc1)
        self.downsample1 = Bottleneck(enc0 * 4, enc1 // 4, stride=2, affine=True)
        # downsample2: Bottleneck, 第2次下采样，(enc1) -> (enc2)
        self.downsample2 = Bottleneck(enc1, enc2 // 4, stride=2, affine=True)
        # downsample3: Bottleneck, 第3次下采样，(enc2) -> (enc3)
        self.downsample3 = Bottleneck(enc2, enc3 // 4, stride=2, affine=True)
        # downsample4: Bottleneck, 第4次下采样，(enc3) -> (bottleneck)
        self.downsample4 = Bottleneck(enc3, bottleneck // 4, stride=2, affine=True)
        
        # ===== 瓶颈层：3D旋转位置编码自注意力 =====
        # A_block: nn.Sequential, 4层3D RoPE自注意力
        self.A_block = nn.Sequential(*[AttentionWith3DRoPE(bottleneck, 8, bottleneck // 8 * 6) for _ in range(4)])
        

        # ===== 解码器：4次上采样 =====
        # main1: nn.Sequential, 4层Res2NetBlock，处理1/8分辨率特征, (bottleneck) -> (dec3), scale=3
        self.main1 = self.main_layer(dec3, 3, 4)
        
        # attn2: AttentionGate, 融合1/4分辨率的跳跃连接特征, (dec3, enc2) -> (dec2)
        self.attn2 = AttentionGate(enc2, dec3, dec2)
        # main2: nn.Sequential, 4层Res2NetBlock
        self.main2 = self.main_layer(dec2, 4, 4)
        
        # attn3: AttentionGate, 融合1/2分辨率的跳跃连接特征, (dec2, enc1) -> (dec1)
        self.attn3 = AttentionGate(enc1, dec2, dec1)
        # main3: nn.Sequential, 4层Res2NetBlock
        self.main3 = self.main_layer(dec1, 4, 4)
        
        # attn4: AttentionGate, 融合原始分辨率的跳跃连接特征, (dec1, enc0*4) -> (dec0)
        self.attn4 = AttentionGate(enc0 * 4, dec1, dec0)
        # main4: nn.Sequential, 4层Res2NetBlock
        self.main4 = self.main_layer(dec0, 4, 4)
        
        # ===== 多尺度输出层 =====
        mid_ch = max(32, dec0 // 2)
        # conv_end_3: nn.Conv3d, 3x3卷积分支
        self.conv_end_3 = nn.Conv3d(dec0, mid_ch, kernel_size=3, stride=1, padding=1)
        # conv_end_5: nn.Conv3d, 5x5卷积分支
        self.conv_end_5 = nn.Conv3d(dec0, mid_ch, kernel_size=5, stride=1, padding=2)
        # conv_end_7: nn.Conv3d, 7x7卷积分支
        self.conv_end_7 = nn.Conv3d(dec0, mid_ch, kernel_size=7, stride=1, padding=3)
        # relu1: nn.ReLU, 激活函数
        self.relu1 = nn.ReLU()
        # conv_end: nn.Conv3d, 最终输出卷积
        self.conv_end = nn.Conv3d(in_channels=mid_ch * 3, out_channels=out_channels, padding=1, kernel_size=3)

    def set_input_channels(self, in_channels: int):  # 将会在包装器 src\wrappers\volume_segmentation.py 中被调用
        """
        动态设置输入通道并重新初始化输入层。
        """
        self.in_channels = in_channels
        # 重新初始化 shortconvadd
        current_out = self.shortconvadd.norm2.num_features  # 探测当前的输出通道
        device = next(self.parameters()).device              # 先记录当前模型所在设备
        self.shortconvadd = ShortConvAdd(input_channels=in_channels, output_channels=current_out)
        self.shortconvadd.to(device)                        # 再将新层移动到正确设备

    def _checkpoint_call(self, fn, *args):
        """
        Activation checkpoint wrapper for heavy blocks.
        """
        if not (self.gradient_checkpoint and self.training and torch.is_grad_enabled()):
            return fn(*args)
        try:
            return torch_checkpoint(fn, *args, use_reentrant=False, preserve_rng_state=False)
        except TypeError:
            # Backward compatibility for older torch versions without use_reentrant.
            return torch_checkpoint(fn, *args, preserve_rng_state=False)
    
    def main_layer(self,input_channels,expansion,num_layers):
        """
        创建主处理层(由多个Res2NetBlock组成)
        
        输入参数 (Input Parameters):
        - input_channels: int, 输入/输出通道数
        - expansion: int, Res2NetBlock的scale参数
        - num_layers: int, Res2NetBlock的数量
        
        输出 (Output):
        - nn.Sequential, 包含num_layers个Res2NetBlock的序列
        """
        # layer: list, 存储Res2NetBlock实例
        layer=[]
        for i in range(num_layers):
            layer.append(Res2NetBlock(input_channels,input_channels,scale=expansion))
        return nn.Sequential(*layer)
    
    def upsample_add(self,f, g):
        """
        上采样并相加(用于跳跃连接)
        
        输入参数 (Input Parameters):
        - f: torch, (B, C, D_f, H_f, W_f), 需要上采样的特征(较粗分辨率)
        - g: torch, (B, C, D, H, W), 目标特征(较细分辨率)
        
        输出 (Output):
        - torch, (B, C, D, H, W), 上采样后与g相加的结果
        """
        g_shape = g.shape
        D, H, W = g_shape[2:]
        upsampled = nn.functional.interpolate(input=f, size=(D, H, W), mode='trilinear', align_corners=True)
        return (g + upsampled)
    
    #multi_scale_conv
    def forward(self,voxel_grid,run_iters:int=3):
        """
        输入参数 (Input Parameters):
        - voxel_grid: torch, (B, 13, D, H, W), 输入体素特征
        - run_iters: int, 默认=3, 循环迭代次数
        
        输出 (Output):
        - f: torch, (B, 1, D, H, W), 预测结果
          1: 输出通道数(如口袋存在概率)
        
        注意：
        - 只有最后一次迭代会计算梯度，前面的迭代在no_grad模式下运行, 这种设计可以在保持循环机制优势的同时减少显存消耗
        """
        B, C, D, H, W = voxel_grid.shape
        
        # voxel_recycle: torch, (B, 64, D, H, W), 循环特征的初始化. 初始为全零，后续迭代中会被更新为上一次的输出特征
        voxel_recycle = torch.zeros((B, 64, D, H, W), device=voxel_grid.device, dtype=voxel_grid.dtype)

        for run_iter in range(run_iters):
            not_last_iter = (run_iter != (run_iters - 1))
            
            # 只有最后一次迭代计算梯度，其他迭代在no_grad模式下运行
            with torch.no_grad() if not_last_iter else contextlib.nullcontext():
                # fused_input: torch, (B, 64, D, H, W), 融合输入特征和循环特征
                fused_input = self.shortconvadd(voxel_grid,voxel_recycle)
                
                # ===== 编码器路径 =====
                # ds_0: torch, (B, 256, D, H, W), 初始特征提取
                ds_0 = self.shortconv1(fused_input)
                # ds_1: torch, (B, 256, D/2, H/2, W/2), 第1次下采样
                ds_1 = self._checkpoint_call(self.downsample1, ds_0)
                # ds_2: torch, (B, 256, D/4, H/4, W/4), 第2次下采样
                ds_2 = self._checkpoint_call(self.downsample2, ds_1)
                # ds_3: torch, (B, 256, D/8, H/8, W/8), 第3次下采样
                ds_3 = self._checkpoint_call(self.downsample3, ds_2)
                # ds_4: torch, (B, 256, D/16, H/16, W/16), 第4次下采样(瓶颈层)
                ds_4 = self._checkpoint_call(self.downsample4, ds_3)
                
                # ===== 瓶颈层：3D RoPE自注意力 =====
                # c4: torch, (B, 256, D/16, H/16, W/16), 经自注意力处理的瓶颈特征
                c4 = self._checkpoint_call(self.A_block, ds_4)
                
                # ===== 解码器路径 =====
                # c3: torch, (B, 256, D/8, H/8, W/8), 第1次上采样+跳跃连接
                c3 = self._checkpoint_call(self.main1, self.upsample_add(c4, ds_3))
                # c2: torch, (B, 128, D/4, H/4, W/4), 第2次上采样+注意力门控
                c2 = self._checkpoint_call(self.main2, self._checkpoint_call(self.attn2, c3, ds_2))
                # c1: torch, (B, 64, D/2, H/2, W/2), 第3次上采样+注意力门控
                c1 = self._checkpoint_call(self.main3, self._checkpoint_call(self.attn3, c2, ds_1))
                # c0: torch, (B, 64, D, H, W), 第4次上采样+注意力门控
                c0 = self._checkpoint_call(self.main4, self._checkpoint_call(self.attn4, c1, ds_0))
                
                # ===== 多尺度输出 =====
                # f3: torch, (B, 32, D, H, W), 3x3卷积分支
                f3 = self.conv_end_3(c0)
                # f5: torch, (B, 32, D, H, W), 5x5卷积分支
                f5 = self.conv_end_5(c0)
                # f7: torch, (B, 32, D, H, W), 7x7卷积分支
                f7 = self.conv_end_7(c0)
                
                # fused_multiscale: torch, (B, 96, D, H, W), 拼接三个尺度的特征
                fused_multiscale = self.relu1(torch.cat((f3, f5, f7), dim=1))
                # final_feature: torch, (B, 1, D, H, W), 最终1x1卷积输出预测结果
                final_feature = self.conv_end(fused_multiscale)
                
                # voxel_recycle: torch, (B, 64, D, H, W), detach()断开梯度
                voxel_recycle = c0.detach()
        return final_feature





