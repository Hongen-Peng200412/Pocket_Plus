import torch
import einops
from torch import nn
import random
import numpy as np
from torch.utils.checkpoint import checkpoint as torch_checkpoint
from einops.layers.torch import Rearrange
import contextlib

def get_lattice_meshgrid_np(shape_size, no_shift=False):
    """ 返回 (D, H, W, 3) 表示每个体素的 (x,y,z) 坐标 ; (D,H,W) = shape_size
     - no_shift=False时， (d,h,w,i)返回的是对应网格中心的坐标  """
    # 生成一个三维格点网格 (lattice meshgrid)
    # 参数 shape_size: (D, H, W)
    # no_shift=False 时每个维度坐标从 0.5 到 shape-0.5（将坐标置于体素中心）
    # no_shift=True 时坐标从 0 到 shape-1（整数格点）
    linspace = [np.linspace(
        0.5 if not no_shift else 0,
        shape - (0.5 if not no_shift else 1),
        shape,
    ) for shape in shape_size]
    # np.meshgrid -> shape (D, H, W, 3) 表示每个体素的 (x,y,z) 坐标
    mesh = np.stack(
        np.meshgrid(linspace[0], linspace[1], linspace[2], indexing="ij"),
        axis=-1,
    )
    return mesh


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, channels, freq_inv=100):
        """
        Sinusoidal positional encoding（正弦位置编码）
            - channels: 在为偶数(推荐)的情况下，等于forward中输入的网格张量新增的最后一维长度(在一维情况下，就=attention_features), 在调用时 channels=attention_features // 3, 内部会向上凑成偶数以便可以拼接 sin 和 cos。
            - freq_inv: 控制频率基底（frequency base），默认 100。

        Forward 输入:
            - tensor 的最后一维必须是坐标维（coord_dim），调用时 coord_dim=3 对应 (x部分,y部分,z部分)。
            - 允许形状： (..., coord_dim) = (D, H, W, 3) 由上一个函数输入
        Forward 输出:
            - emb_x: 形状 (..., coord_dim, channels=attention_features/3)， (B=b, L=l, coord_dim=i, channels=c) 表示第 B=b 个样本,展平后的第L=l个位置, 这个位置的第i个坐标的第c个特征(channel)对应的 sin 或 cos编码值。最后一维 channels=attention_features/3 中 sin 和 cos 的排列是交替的
            它的第(..., coorf_dim=i, channels=j)表示    “第...个‘位置’，在第i个坐标维度上对应  [第j个频率 or feature的第j个维度] 的sin或cos编码值(sin\cos内部交替且按照特征的维度序数递增)”。

        相关调用：
            self.pos_encoding = SinusoidalPositionalEncoding(channels=attention_features // 3)
            pos_vector = einops.rearrange(pos_grid, "B H D W C -> B (H D W) C") * 1.5  # pos_grid为 B H D W C=3为网格坐标; C=3表示xyz; 1.5为缩放因子; 
            pos_emb = self.pos_encoding(pos_vector)   # flatten前: (B L 3 channels/3)
            pos_emb = pos_emb.flatten(-2)   # flatten(-2)后: (B L channels), 这仍然有对齐关系
                        flatten前 pos_emd 有意义：它的第(..., coorf_dim=i, channels=j)表示    “第...个‘位置’，在第i个坐标维度上对应  [第j个频率 or feature的第j个维度] 的sin或cos编码值(sin\cos内部交替且按照特征的维度序数递增)”(同上)
                        flatten后 pos_emb 有意义： pos_emb[...(前面表示batch、位置)选定它们, :] = 该样本下关于位置xyz的旋转位置编码[ x:(sin,cos,sin,cos,sin,cos), y:(sin,cos,sin,cos,sin,cos), z:(sin,cos,sin,cos,sin,cos) ]   内部的sin cos 频率(按照特征的维度序数)递增
        """
        super().__init__()
        # 保存用户传入的原始通道数（事实上=原始希望的 embedding 大小）
        self.org_channels = channels

        # 为了能把 sin & cos 直接 concat，确保内部 channels 为偶数
        channels = int(np.ceil(channels / 2) * 2)
        self.channels = channels

        # inv_freq shape: (channels // 2,),   inv_freq[j] = 1.0 / (freq_inv ** (2*j/channels)), j=0,1,2,...,channels//2-1
        inv_freq = 1.0 / (freq_inv ** (torch.arange(0, channels, 2).float() / channels))
        # 注册为 buffer（与模型一起移动到 device，但不作为可训练参数）
        # self.inv_freq shape: (F,)， 之后 F = channels // 2
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, tensor):
        # -------------------------
        # Step A: 外积outer product
        # -------------------------
        # einsum 表达式 "...i,j->...ij" 的含义：
        #   - 把 tensor(=pos,网格坐标) 最后轴记为 i（i 表示 coord_dim, 这里为3，表示0/1/2 or x/y/z这三个维度）
        #   - 把 inv_freq 的轴记为 j（j 表示 frequency index）
        #   - 输出保留前缀 "..."（任意维度），再接上 i 和 j 轴：(..., i, j)
        #
        # 结果解释（shape）：
        #   - self.inv_freq shape = (F,)  (F = channels // 2)
        #   - sin_inp_x shape -> (N, L, coord_dim, F)
        #
        # 等价写法（不使用 einsum）：
        #   inv = self.inv_freq.to(tensor.device)          # shape (F,)
        #   sin_inp_x = tensor.unsqueeze(-1) * inv         # broadcast -> (..., coord_dim, F)
        # 
        # 数值示例：
        #   - 假设 channels=16 -> F = 8
        #   - 假设 tensor shape = (N=2, L=64, coord_dim=3, )
        #   - 则 sin_inp_x shape = (N=2, L=64, coord_dim=3, F=8, )
        sin_inp_x = torch.einsum("...i, j -> ...ij", tensor, self.inv_freq.to(tensor.device))[...,None]   # 增加维度作为辅助,使得下面sin cos交替


        # -------------------------
        # Step B: compute sin and cos, then concat
        # -------------------------
        # sin_inp_x.sin() shape: (..., coord_dim, F)
        # sin_inp_x.cos() shape: (..., coord_dim, F)
        # emb_x = cat(sin, cos) along last dim -> (..., coord_dim, 2*F)
        # 因为我们在 __init__ 中把 F = channels // 2 ，所以如果默认channels是偶数(推荐)：
        # emb_x shape = (..., coord_dim, channels), cons sin交替且频率随特征位置递增： cos 0*distance, sin 1*distance, cos 2*distance, sin 3*distance, ...
        #
        # 数值示例（延续上例）：
        #   - sin_inp_x shape = (N=2,L=64,coord_dim=3,F=8)
        #   - sin(...) & cos(...) each -> (N=2, L=64, coord_dim=3, F=8)
        #   - emb_x after concat -> (N=2, L=64, coord_dim=3, channels=16)
        emb_x = torch.cat((sin_inp_x.sin(), sin_inp_x.cos()), dim=-1).flatten(-2)
        return emb_x




def ThreeD_Rope(q, k, pos_emb, edge_index=None):
    """
    3D RoPE 核心变换（3D Rotary Positional Embedding）
    输入说明：
      - q: 张量，形状 (..., A, F)特别(B, L, A, F), B = batch 大小，L = 展平后的位置数（H*D*W），A = 注意力头数，F = 三个坐标总的特征维度（afz或head_dim或attention_features）。事实上,我们可以让最后一维的按照(x方向的特征=afz/3, y方向的特征=afz/3, z方向的特征=afz/3)的顺序排列，这个顺序可以保证后续的对齐。
      - k: 张量，形状同 q
      - pos_emb: 位置编码张量，形状 (..., F)（典型：(B, L, F)），最后一维的数目应能与 q/k 的最后一维的数目 F 相同(注意力是Bahan内积注意力,要用内积的)。它是通过 SinusoidalPositionalEncoding() 生成后传入的（用的是flatten后的pos_emb):

            self.pos_encoding = SinusoidalPositionalEncoding(channels=attention_features // 3)
            pos_vector = einops.rearrange(pos_grid, "B H D W C -> B (H D W) C") * 1.5  # pos_grid为 B H D W C=3为网格坐标; C=3表示xyz; 1.5为缩放因子; 
            pos_emb = self.pos_encoding(pos_vector)   # flatten前: (B L 3 channels/3)
            pos_emb = pos_emb.flatten(-2)   # flatten(-2)后: (B L channels), 这仍然有对齐关系
                flatten前 pos_emd 有意义：它的第(..., coorf_dim=i, channels=j)表示    “第...个‘位置’，在第i个坐标维度上对应  [第j个频率 or feature的第j个维度] 的sin或cos编码值(sin\cos内部交替且按照特征的维度序数递增)”
                flatten后 pos_emb 有意义： pos_emb[...(前面表示batch、位置)选定它们, :] = 该样本下关于位置xyz的旋转位置编码[ x:(sin,cos,sin,cos,sin,cos), y:(sin,cos,sin,cos,sin,cos), z:(sin,cos,sin,cos,sin,cos) ]   内部的sin cos 频率(按照特征的维度序数)递增
            
             ！！！  注意：q 的形状 (..., A, F), 这个F=(该样本该位置在x上的特征, 该样本该位置在y上的特征, 该样本该位置在上的特征), 等价于原始 q(..., A, 3, F/3).flatten(-2), 这刚好跟展平后的 pos_emb 的最后一维 F 形状对齐: 因为同样 pos_emd = pos_emb.flatten(-2)  ！！！

      - edge_index: 本项目没用到。它是可选的索引（用于图/稀疏情形），用于从 pos_emb 中索引对应 k 的位置编码。
    返回：
      - q_new, k_new：和 q/k 形状相同，均为 (..., A, F)

    说明：
        直观上的公式（每对 (2i,2i+1)）为：
          q_new_even = q_even * cos的编码值 - q_odd * sin的编码值
          q_new_odd  = q_even * sin的编码值 + q_odd * cos的编码值
        这等价于把 (q_even, q_odd) 看作复数 q_even + i*q_odd，与复数 cos + i*sin 相乘。
    """

    # --------------------------
    # 步骤1：准备 cos/sin 系数
    # --------------------------
    # pos_emb[..., 1::2]：从 pos_emb 的最后一维取奇数索引元素（索引 1,3,5,...）,如果 pos_emb.shape = (B, L, F)，则 pos_emb[..., 1::2].shape = (B, L, F/2)。它们他们刚好对齐了(也就是说c0 c1 ...这些频率确实是递增的)
    # repeat_interleave(2, dim=-1)：将每个元素沿最后一维重复两次，把长度扩展回 F, 于是 cos_pos.shape = (B, L, F)，元素排列为 [c0,c0,c1,c1,...]
    cos_pos = pos_emb[..., 1::2].repeat_interleave(2, dim=-1)  # 形状: (..., F=afz)

    # pos_emb[..., ::2]：取偶数索引元素（0,2,4,...），形状 (B, L, F/2)
    # repeat_interleave -> sin_pos.shape = (B, L, F)，元素排列为 [s0,s0,s1,s1,...]
    sin_pos = pos_emb[..., ::2].repeat_interleave(2, dim=-1)  # 形状: (..., F)




    # --------------------------
    # 步骤2：对 q/k 应用 RoPE
    # --------------------------
    if edge_index is None:
        # ！！！  注意：q 的形状 (..., A, F), 这个F=(该样本该位置在x上的特征, 该样本该位置在y上的特征, 该样本该位置在上的特征), 等价于原始 q(..., A, 3, F/3).flatten(-2), 这刚好跟展平后的 pos_emb 的最后一维 F 形状对齐: 因为同样 pos_emd = pos_emb.flatten(-2)  ！！！
        # cos_pos[..., None, :] 将 cos_pos 从 (..., F) 变为 (..., 1, F)，以便在 head 维度上广播到 (..., A, F)
        # 旋转项（rotate term）通过把偶数/奇数通道对配对构造：
        #   q[..., ::2] -> 偶数通道，形状 (..., A, F/2)
        #   q[..., 1::2] -> 奇数通道，形状 (..., A, F/2)
        # torch.stack([-q_odd, q_even], dim=-1) -> 形状 (..., A, F/2, 2)
        # 注意对齐！！！按dim=-1做stack后，会产生一个新的张量, 这个张量的最后一维度就是照着-q[..., 1::2], q[..., ::2]交替排列的(先取前者第一个,再取后者第一个...), 保证元素顺序为 [-odd1, even0, -odd3, even5, ...]——————而不会是上面的[-odd1, -odd3, -odd5...., even0, even2, ...]。前者可以跟 sin_pos 对齐而后者不可以(当当然不会出现)
        q_new = (
            q * cos_pos[..., None, :]  # q 与 cos 逐元素相乘（广播到各个 head）
            + torch.stack([-q[..., 1::2], q[..., ::2]], dim=-1).reshape(q.shape) * sin_pos[..., None, :]
        )
        k_new = (
            k * cos_pos[..., None, :]  # 对 k 做相同处理
            + torch.stack([-k[..., 1::2], k[..., ::2]], dim=-1).reshape(k.shape) * sin_pos[..., None, :]
        )



    else:
        # --------------------------
        # 稀疏 / 图情形：对 k 使用 edge_index 从 pos_emb 中取对应的位置编码
        # --------------------------
        # q_new 的计算与上面相同（对 q 使用 q 位置上对应的 pos_emb）
        q_new = (
            q * cos_pos[..., None, :]
            + torch.stack([-q[..., 1::2], q[..., ::2]], dim=-1).reshape(q.shape) * sin_pos[..., None, :]
        )

        # 对 k，我们先用 edge_index 从 cos_pos/sin_pos 中取出对应项，然后再广播到 head 维：
        # 例子：
        #   pos_emb: (B, L, F)
        #   edge_index: (E,) 或其他索引形状
        #   pos_emb[edge_index]: (B, E, F) 或 (E, F)（取决于 pos_emb 是否含 batch 维）
        # 然后 cos_pos[edge_index] / sin_pos[edge_index] 形状为 (B, E, F)，再通过 [..., None, :] 广播为 (B, E, 1, F)
        k_new = (
            k * cos_pos[edge_index][..., None, :]
            + torch.stack([-k[..., 1::2], k[..., ::2]], dim=-1).reshape(k.shape) * sin_pos[edge_index][..., None, :]
        )

        # 注意：使用 edge_index 时，务必保证 pos_emb[edge_index] 的输出 shape 与 k 的空间轴一致，否则会发生 broadcasting/对齐错误。调试时请打印 pos_emb.shape、edge_index.shape、k.shape。

    # 返回应用位置旋转后的 q, k（形状不变）
    return q_new, k_new





class Transition(nn.Module):
    """
    小型的通道混合模块（类似 Transformer 的 MLP/FFN，但带了两个线性的乘性交互）
    结构：
      - w1: Linear(ifz -> ifz * n), ifz 为 input_features
      - w2: Linear(ifz -> ifz * n)
      - elementwise: silu(w1(x)) * w2(x)
      - w3: Linear(ifz * n -> ifz)
      - residual + norm
    作用：增强通道间非线性交互（channel mixing），并做残差归一化（residual + norm）。
    """
    def __init__(self, in_features:int, norm:nn.Module, n:int=3):
        super().__init__()
        self.norm = norm(in_features)
        self.w1 = nn.Linear(in_features, in_features * n, bias=False)
        self.w2 = nn.Linear(in_features, in_features * n, bias=False)
        self.w3 = nn.Linear(in_features * n, in_features, bias=False)
        self.short = nn.Identity()

    def forward(self, x):
        # y = w3( silu(w1(x)) * w2(x) )
        # 注意这里是逐元素相乘（gating-like interaction），比普通的两层 FFN 多了乘性交互。
        y = self.w3(nn.functional.silu(self.w1(x)) * self.w2(x))
        # 残差 + 归一化
        y = self.norm(y + self.short(x))
        return y






class AttentionWith3DRoPE(nn.Module):
    """ 多头注意力（multi-head attention）在 3D-volume 上的实现，带 3D RoPE（rotary PE）
    Args:
        - in_features: 输入通道数（ifz）
        - attention_heads: 注意力头数（ahz）
        - attention_features: 但它是包含三个维度的信息，每个维度的 attention_features = attention_features//3

    Forward Args:
        - x: (B, C, H, D, W) 输入张量，其中 C=ifz=input_features，把每个体素处的通道数看作'feature'
    
    Note/Process:
        - 这里是自注意力, 将x通过三个参数独立的线性层（self.q, self.k ,self.v）生成初始的 q, k, v, 然后q, k经过三维旋转位置编码, 进行更新
        -  q, k, v(包括旋转位置编码后的) 的形状均为 (B L A=ahz afz)，其中 L=H*D*W (经过"展平"或token化)，ahz 表示注意力头数，
        afz 表示每个注意力头的特征维度=attention_features, 这个特征实际上由三个维度xyz的特征拼接而成, 每个维度的特征维数为 attention_features//3 且按 x方向的特征, y方向的特征, z方向的特征依次排列。分别对每个维度的特征应用一维的旋转位置编码，频率取决于: 这个特征分量在本维度特征的位置 (0~attention_features//3)、 这个体素[token]在这一维度的坐标。 前者为偶数时sin内部系数=前两者相乘 * 默认频率(100), 前者为奇数时cos内部系数=前两者相乘 * 默认频率(100)。
        -  注意力分数计算：对 Q/K 应用 3D RoPE（rotary PE）后，再计算点积注意力分数（按 head 内积）, 之后对(k,v)-pairs中每个k对所有v做softmax归一化。
        - 注意力结果为 out=(B, L=H*D*W, A=ahz, I=attention_features)表示某个体素处(体素位置按照L索引)三个维度总的feature
        - 最终将 注意力结果 out 经过线性层, 与原本的 x_vec(输入x一步reshape得到)相加， + LayerNorm + 小型MLP（Transition）, 得到与x形状相同的输出 (N, C, H, D, W)
    """
    def __init__(self, in_features:int, attention_heads:int, attention_features:int):
        super(AttentionWith3DRoPE, self).__init__()
        self.ifz = in_features          # 输入通道数（input feature dim）
        self.ahz = attention_heads      # number of heads
        self.afz = attention_features   # 这里的attention_features当然是q,k的features，但它是包含三个维度的信息，每个维度的 attention_features = attention_features//3  ————这就顺便解释了下文 SinusoidalPositionalEncoding(channels=attention_features // 3) 的合理性
        # attention_scale 用于缩放点积，常用 sqrt(d_k)
        self.attention_scale = np.sqrt(self.afz)

        # Q,K,V 的投影：先线性投影到 (ahz*afz)，再 rearrange 为 (ahz, afz)
        # q: B, L, ahz, afz
        self.q = nn.Sequential(
            nn.Linear(self.ifz, self.ahz * self.afz),
            Rearrange("B L (ahz afz) -> B L ahz afz", ahz=self.ahz, afz=self.afz)
        )
        self.k = nn.Sequential(
            nn.Linear(self.ifz, self.ahz * self.afz, bias=False),
            Rearrange("B L (ahz afz) -> B L ahz afz", ahz=self.ahz, afz=self.afz)
        )
        self.v = nn.Sequential(
            nn.Linear(self.ifz, self.ahz * self.afz, bias=False),
            Rearrange("B L (ahz afz) -> B L ahz afz", ahz=self.ahz, afz=self.afz)
        )

        # back: 把每个 head 的输出拼回原始通道并线性映射回 ifz
        self.back = nn.Sequential(
            Rearrange("B L ahz afz -> B L (ahz afz)", ahz=self.ahz, afz=self.afz),
            nn.Linear(self.ahz * self.afz, self.ifz, bias=False)
        )

        # 位置编码模块：注意这里传入的是 attention_features // 3（把 3 个坐标拆分到每个维度集）
        # 理论上我们希望 pos_encoding 的最终维度能匹配或被广播到每个 head 的维度 afz。
        self.pos_encoding = SinusoidalPositionalEncoding(channels=attention_features // 3)

        # LayerNorm 与 Transition（通道混合 MLP）
        self.norm1 = nn.LayerNorm(in_features)
        self.transition1 = Transition(in_features, nn.LayerNorm)

    def forward(self, x):
        # x: (B, C, H, D, W)
        B, C, H, D, W = x.shape
        # 1) [None]开头加一维, repeat之后 生成坐标位置网格 (B, H, D, W, 3) -> 展平为 (B, L, 3)，L = H*D*W
        pos_grid = torch.from_numpy(get_lattice_meshgrid_np((H, D, W), no_shift=True)).float().to(x.device)[None].repeat(B, 1, 1, 1, 1)
        pos_vector = einops.rearrange(pos_grid, "B H D W C -> B (H D W) C") * 1.5  # pos 一开始为 B H D W C=3为网格坐标; 这里C=3表示xyz而不是x的channel, 1.5为缩放因子; 
        # 2) 位置编码 -> 每个位置得到一个向量 (B, L, P)
        # 展平 (Flattening): 将张量中从 start_dim 到 end_dim（包含这两个维度）之间的所有维度合并成一个单一的维度，放到最后
        # 对于pos_emb形状: flatten前 (B L 3 attention_features//3)(注意pos_encoding的输出) -> (flatten后)  (B L attention_features)
                    # flatten后 pos_emb 有意义： pos_emb[...(前面表示batch+位置)选定它们, :] = 该样本下关于位置xyz的旋转位置编码[ x:(sin,cos,sin,cos,sin,cos), y:(sin,cos,sin,cos,sin,cos), z:(sin,cos,sin,cos,sin,cos) ]   内部的sin cos 频率(按照特征的维度序数)递增
        pos_emb = self.pos_encoding(pos_vector)
        pos_emb = pos_emb.flatten(-2)
        

        # 3) 把 x 展平为序列形式（token sequence）
        x_vec = einops.rearrange(x, "B C H D W -> B (H D W) C")  # -> 这里的C就是x本身的channel

        # 4) 线性投影得到 Q,K,V，并 reshape 为 (B, L=H*D*W, heads=A=ahz, head_dim=afz=attention_features依次排列xyz三个维度的特征)
        query = self.q(x_vec)   # (B, L, ahz=A, afz)
        key = self.k(x_vec)     # (B, L, ahz, afz)
        value = self.v(x_vec)   # (B, L, ahz, afz)

        # 5) 对 Q/K 应用 3D RoPE（旋转位置编码）
        #    ThreeD_Rope 会基于 pos_emb 生成 cos_pos, sin_pos 并对 q/k 的每对相邻维度进行变换
        query, key = ThreeD_Rope(query, key, pos_emb)

        # 6) 计算点积注意力分数（按 head 内积）,einsum 'blai,bkai->blka':
        #   维数分别是(Batch, L=H*D*W表示query数目, A=Heads数目, I=Attention_features) ; (Batch, K=H*D*W表示key数目, A=Heads数目, I=Attention_features)。 它计算 query(l) 与 key(k) 的点积，保留 head 轴 a 不被求和
        #    结果： (B, L_query, L_key, A) —— 这里 L_query = L_key = L = H*D*W 表示query数目, key数目
        attention_scores = torch.einsum('blai,bkai->blka', query, key) / self.attention_scale

        # 7) 对 key 位置做 softmax 归一化（dim=-2 指倒数第二个维度，对应 L_key）
        attention_weights = attention_scores.softmax(dim=-2)  # 归一化方向：keys

        # 8) 用注意力权重对 value (Batch, L=H*D*W表示query数目, A=Heads数目, I=Attention_features) 加上权重 attention_weights (Batch, L=H*D*W表示query数目, K=H*D*W表示key数目, A=Heads数目)求和 -> 对I求和, out  (B, L=H*D*W, A, I=attention_features 这描述了xyz3个坐标的特征)
        out = torch.einsum('blka,bkai->blai', attention_weights, value)

        # 9) back projection：把多头输出拼回原始通道数并做残差 + LayerNorm
        out = self.norm1(x_vec + self.back(out))

        # 10) 经过 Transition（通道混合 MLP + residual + norm）
        out = self.transition1(out)

        # 11) reshape 回 B, C, H, D, W = x.shape
        return einops.rearrange(out, "B (H D W) C -> B C H D W", B=B, C=C, H=H, D=D, W=W)
