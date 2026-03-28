"""
针对本项目可能改进的点:
1. 前几次池化时保持C_alpha原子, 或者可以更进一步让某次池化后只保持C_alpha原子, 这样就能与残基级别的特征(ESM2)做灵活的融合
2. 加入别的序列化方法, 比如按照氨基酸顺序排列
"""
import sys  # sys, 系统相关功能
from typing import List, Optional, Tuple  # typing, 类型提示
from functools import partial  # functools, 偏函数
from addict import Dict  # addict, 字典增强库
import math  # math, 数学运算
import torch  # torch, PyTorch深度学习框架
import torch.nn as nn  # torch.nn, 神经网络模块
import spconv.pytorch as spconv  # spconv, 稀疏卷积库
import torch_scatter  # torch_scatter, 张量分散操作
import torch_cluster  # torch_cluster, 点云邻域搜索
from timm.models.layers import DropPath  # timm, 随机深度模块
from collections import OrderedDict  # collections, 有序字典

# 尝试导入flash_attn加速库,如果未安装则设为None
try:
    import flash_attn
except ImportError:
    flash_attn = None
    print("环境中没有 flash_attn ！！！")
# 从序列化模块导入编码函数
from .serialization import encode

@torch.inference_mode()
def offset2bincount(offset):
    """
    将offset转换为bincount(每个batch的点数统计)。
    
    输入参数:
        - offset: torch.Tensor, (batch_size,), offset数组,表示每个batch的结束索引
    
    输出:
        - bincount: torch.Tensor, (batch_size,), 每个batch的点数
    """
    # torch.Tensor, 计算offset的差值,得到每个batch的点数
    return torch.diff(
        offset, prepend=torch.tensor([0], device=offset.device, dtype=torch.long)   # 在前面加上0, 做差
    )


@torch.inference_mode()
def offset2batch(offset):
    """
    将offset(每个batch的结束索引)转换为batch索引数组
    
    输入参数:
        - offset: torch.Tensor, (batch_size,), offset数组
    
    输出:
        - batch: torch.Tensor, (N,), 每个点所属的batch索引
    """
    # torch.Tensor, 每个batch的点数
    bincount = offset2bincount(offset)
    # torch.Tensor, 生成batch索引数组,每个点标记其所属的batch
    return torch.arange(
        len(bincount), device=offset.device, dtype=torch.long
    ).repeat_interleave(bincount)


@torch.inference_mode()
def batch2offset(batch):
    """
    将batch索引数组转换为offset数组(每个batch的结束索引), 左闭右开即 [offset[i], offset[i+1]) 代表第i个batch的点云索引
    
    输入参数:
        - batch: torch.Tensor, (N,), batch索引数组
    
    输出:
        - offset: torch.Tensor, (batch_size,), offset数组
    """
    # torch.Tensor, 计算累加和得到offset
    return torch.cumsum(batch.bincount(), dim=0).long()


class Point(Dict):
    """
        Point Structure of Pointcept
        
        点云数据结构,继承自Dict,用于存储批处理的点云数据及其属性。
        
        必需属性:
            - "coord": torch.Tensor, (N, 3), 点云的原始坐标
            - "grid_coord": torch.Tensor, (N, 3), 网格化后的离散坐标(与GridSampling相关)
        
        可选属性:
            - "offset": torch.Tensor, (batch_size,), offset数组(每个batch的结束索引),若不存在则初始化为batch_size=1
            - "batch": torch.Tensor, (N,), 单调递增的batch索引数组,若不存在则初始化为batch_size=1
            - "feat": torch.Tensor, (N, C), 点云特征,模型的默认输入
            - "grid_size": float, 网格大小(与GridSampling相关)
        
        序列化相关属性:
            - "serialized_depth": int, 序列化深度, 2**depth*grid_size描述点云的最大范围
            - "serialized_code": torch.Tensor, (k, N), 序列化编码列表; k 表示序列化顺序（Serialization Orders）的数量
            - "serialized_order": torch.Tensor, (k, N), 由编码确定的序列化顺序列表：代表一个映射, order[i][j]代表第i种序列化顺序下, 第j个点经过排序后所处的位置
            - "serialized_inverse": torch.Tensor, (k, N), 由编码确定的逆映射列表：代表一个映射, inverse[i][j]代表第i种序列化顺序下, 第j个点在排序前所处的位置
        
        稀疏卷积相关属性:
            - "sparse_shape": list, 稀疏卷积张量的空间形状
            - "sparse_conv_feat": spconv.SparseConvTensor, 由Point信息初始化的稀疏卷积张量
    """
    def __init__(self, *args, **kwargs):
        """
        Point Structure of Pointcept
        
        点云数据结构,继承自Dict,用于存储批处理的点云数据及其属性。
        
        必需属性:
            - "coord": torch.Tensor, (N, 3), 点云的原始坐标
            - "grid_coord": torch.Tensor, (N, 3), 网格化后的离散坐标(与GridSampling相关)
        
        可选属性:
            - "offset": torch.Tensor, (batch_size,), offset数组,若不存在则初始化为batch_size=1
            - "batch": torch.Tensor, (N,), 单调递增的batch索引数组,若不存在则初始化为batch_size=1
            - "feat": torch.Tensor, (N, C), 点云特征,模型的默认输入
            - "grid_size": float, 网格大小(与GridSampling相关)
        
        序列化相关属性:
            - "serialized_depth": int, 序列化深度, 2**depth*grid_size描述点云的最大范围
            - "serialized_code": torch.Tensor, (k, N), 序列化编码列表; k 表示序列化顺序（Serialization Orders）的数量
            - "serialized_order": torch.Tensor, (k, N), 由编码确定的序列化顺序列表：代表一个映射, order[i][j]代表第i种序列化顺序下, 第j个点经过排序后所处的位置
            - "serialized_inverse": torch.Tensor, (k, N), 由编码确定的逆映射列表：代表一个映射, inverse[i][j]代表第i种序列化顺序下, 第j个点在排序前所处的位置
        
        稀疏卷积相关属性:
            - "sparse_shape": list, 稀疏卷积张量的空间形状
            - "sparse_conv_feat": spconv.SparseConvTensor, 由Point信息初始化的稀疏卷积张量
        """
        super().__init__(*args, **kwargs)
        # 如果"batch"不存在但"offset"存在,则根据offset生成batch
        if "batch" not in self.keys() and "offset" in self.keys():
            self["batch"] = offset2batch(self.offset)
        # 如果"offset"不存在但"batch"存在,则根据batch生成offset
        elif "offset" not in self.keys() and "batch" in self.keys():
            self["offset"] = batch2offset(self.batch)

    def serialization(self, order="z", depth=None, shuffle_orders=False):
        """
        点云序列化,将3D点云转换为1D序列以便处理。
        
        输入参数:
            - order: str 或 list[str], 序列化顺序,默认"z",可选"z","z-trans","hilbert","hilbert-trans"
            - depth: int 或 None, 序列化深度,若为None则自适应计算
            - shuffle_orders: bool, 是否打乱多个序列化顺序,默认False
        """
        assert "batch" in self.keys()
        # 如果grid_coord不存在,则根据coord和grid_size计算网格坐标
        if "grid_coord" not in self.keys():
            assert {"grid_size", "coord"}.issubset(self.keys())
            # torch.Tensor, (N, 3), 点云坐标按 grid size 做正向归一化([0]返回坐标的最小值)
            self["grid_coord"] = torch.div(
                self.coord - self.coord.min(0)[0], self.grid_size, rounding_mode="trunc"
            ).int()

        # 如果depth未指定,则自适应计算序列化深度
        if depth is None:
            # int, 自适应测量序列化立方体的深度(边长 = 2^depth)
            depth = int(self.grid_coord.max()).bit_length()
        # int, 保存序列化深度
        self["serialized_depth"] = depth
        # 序列化编码的最大位长度为63(int64)
        assert depth * 3 + len(self.offset).bit_length() <= 63
        # 其中将深度限制为16(48位)用于点位置编码。
        assert depth <= 16
        # list[torch.Tensor], 对每个指定的顺序进行编码
        code = [
            encode(self.grid_coord, self.batch, depth, order=order_) for order_ in order
        ]
        # torch.Tensor, (k, n), k 为序列化顺序数, n 为点云数量
        code = torch.stack(code)
        # torch.Tensor, (k, n), 这代表一个映射, order[i][j]代表第i种序列化顺序下, 第j个点经过排序后所处的位置
        order = torch.argsort(code)
        # torch.Tensor, (k, n), 这代表一个映射, inverse[i][j]代表第i种序列化顺序下, 第j个点在排序前所处的位置
        inverse = torch.zeros_like(order).scatter_(   # 相当于令 inverse[i][order[i][j]] = src[i][j] = j
            dim=1,
            index=order,
            src=torch.arange(0, code.shape[1], device=order.device).repeat(   # (k, n), 且 src[i][j] = j
                code.shape[0], 1
            ),
        )

        # 如果需要打乱顺序,则随机排列多个序列化顺序
        if shuffle_orders:
            # torch.Tensor, 随机排列索引
            perm = torch.randperm(code.shape[0])
            # torch.Tensor, 按随机索引排列编码
            code = code[perm]
            # torch.Tensor, 按随机索引排列顺序
            order = order[perm]
            # torch.Tensor, 按随机索引排列逆映射
            inverse = inverse[perm]

        # 保存序列化结果
        self["serialized_code"] = code
        self["serialized_order"] = order
        self["serialized_inverse"] = inverse

    def sparsify(self, pad=96):
        """
        点云稀疏化,为稀疏卷积准备spconv.SparseConvTensor。
        
        输入参数:
            - pad: int, 稀疏形状的填充值,默认96
        
        依赖:
            ["grid_coord" 或 "coord" + "grid_size"; "batch"; "feat"]
        
        功能:
            1. 如果不存在grid_coord, 则根据coord和grid_size计算
            2. 计算稀疏形状
            3. 创建SparseConvTensor对象
        
        输出:
            在self中添加"sparse_shape"和"sparse_conv_feat"属性
        """
        # 检查必需的属性是否存在
        assert {"feat", "batch"}.issubset(self.keys())
        # 如果grid_coord不存在,则根据coord和grid_size计算网格坐标
        if "grid_coord" not in self.keys():
            assert {"grid_size", "coord"}.issubset(self.keys())
            # torch.Tensor, (N, 3), 计算网格坐标
            self["grid_coord"] = torch.div(
                self.coord - self.coord.min(0)[0], self.grid_size, rounding_mode="trunc"
            ).int()
        # 计算稀疏形状
        if "sparse_shape" in self.keys():
            # list, 如果已存在则直接使用
            sparse_shape = self.sparse_shape
        else:
            # list, 否则根据grid_coord的最大值和pad计算
            sparse_shape = torch.add(
                torch.max(self.grid_coord, dim=0).values, pad
            ).tolist()
        # spconv.SparseConvTensor, 创建稀疏卷积张量
        sparse_conv_feat = spconv.SparseConvTensor(
            features=self.feat,  # torch.Tensor, (N, C), 特征
            indices=torch.cat(
                [self.batch.unsqueeze(-1).int(), self.grid_coord.int()], dim=1  # torch.Tensor, (N, 4), 索引 [batch, x, y, z]
            ).contiguous(),
            spatial_shape=sparse_shape,  # list, 空间形状
            batch_size=self.batch[-1].tolist() + 1,  # int, batch大小
        )
        # 保存稀疏形状和稀疏卷积特征
        self["sparse_shape"] = sparse_shape
        self["sparse_conv_feat"] = sparse_conv_feat


class PointModule(nn.Module):
    """
    点云模块基类,所有子类都会在PointSequential中接受Point对象作为输入。
    
    功能:
        作为占位符,所有继承自该类的模块都会在PointSequential中处理Point对象
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class PointSequential(PointModule):
    """
    序列容器,模块将按照传入构造函数的顺序添加, 也可以传入有序字典来添加模块。
    
    功能:
        类似于nn.Sequential,但专门用于处理Point对象, 支持PointModule、SpConv模块和普通PyTorch模块
    """

    def __init__(self, *args, **kwargs):
        """
        初始化PointSequential。
        
        输入参数:
            - *args: 位置参数,可以是模块列表或OrderedDict
            - **kwargs: 关键字参数,模块名称和模块对象的键值对
        
        功能:
            将模块按顺序添加到容器中
        """
        super().__init__()
        # 如果传入的是OrderedDict,则按键值对添加模块
        if len(args) == 1 and isinstance(args[0], OrderedDict):
            for key, module in args[0].items():
                self.add_module(key, module)
        # 否则按索引添加模块
        else:
            for idx, module in enumerate(args):
                self.add_module(str(idx), module)
        # 添加关键字参数中的模块
        for name, module in kwargs.items():
            if sys.version_info < (3, 6):
                raise ValueError("kwargs only supported in py36+")
            if name in self._modules:
                raise ValueError("name exists.")
            self.add_module(name, module)

    def __getitem__(self, idx):
        """
        根据索引获取模块。
        
        输入参数:
            - idx: int, 模块索引,支持负索引
        
        输出:
            - module: nn.Module, 对应索引的模块
        """
        if not (-len(self) <= idx < len(self)):
            raise IndexError("index {} is out of range".format(idx))
        # 处理负索引
        if idx < 0:
            idx += len(self)
        # 迭代到指定索引
        it = iter(self._modules.values())
        for i in range(idx):
            next(it)
        return next(it)

    def __len__(self):
        """
        返回模块数量。
        
        输出:
            - int, 模块数量
        """
        return len(self._modules)

    def add(self, module, name=None):
        """
        添加模块到容器。
        
        输入参数:
            - module: nn.Module, 要添加的模块
            - name: str 或 None, 模块名称,若为None则自动生成
        
        功能:
            将模块添加到容器末尾
        """
        if name is None:
            name = str(len(self._modules))
            if name in self._modules:
                raise KeyError("name exists")
        self.add_module(name, module)

    def forward(self, input):
        """
        前向传播,依次执行所有模块。
        
        输入参数:
            - input: Point 或 spconv.SparseConvTensor 或 torch.Tensor, 输入数据
        
        输出:
            - output: Point 或 spconv.SparseConvTensor 或 torch.Tensor, 输出数据
        
        功能:
            根据模块类型分别处理:
            1. PointModule: 直接传入Point对象
            2. SpConv模块: 处理sparse_conv_feat
            3. 普通PyTorch模块: 处理feat或features
        """
        for k, module in self._modules.items():
            # Point module: 直接传入Point对象
            if isinstance(module, PointModule):
                input = module(input)
            # Spconv module: 处理稀疏卷积特征
            elif spconv.modules.is_spconv_module(module):
                if isinstance(input, Point):
                    # spconv.SparseConvTensor, 对sparse_conv_feat进行稀疏卷积
                    input.sparse_conv_feat = module(input.sparse_conv_feat)
                    # torch.Tensor, 更新特征
                    input.feat = input.sparse_conv_feat.features
                else:
                    input = module(input)
            # PyTorch module: 处理普通特征
            else:
                if isinstance(input, Point):
                    # torch.Tensor, 对feat进行前向传播
                    input.feat = module(input.feat)
                    # 如果存在sparse_conv_feat,则更新其特征
                    if "sparse_conv_feat" in input.keys():
                        input.sparse_conv_feat = input.sparse_conv_feat.replace_feature(
                            input.feat
                        )
                elif isinstance(input, spconv.SparseConvTensor):
                    # 如果索引不为空,则更新特征
                    if input.indices.shape[0] != 0:
                        input = input.replace_feature(module(input.features))
                else:
                    input = module(input)
        return input


class PDNorm(PointModule):
    """
    Point-wise Decoupled Normalization, 点级解耦归一化层。
    
    功能:
        支持多条件解耦归一化和自适应归一化
    """
    def __init__(
        self,
        num_features,
        norm_layer,
        context_channels=256,
        conditions=("ScanNet", "S3DIS", "Structured3D"),
        decouple=True,
        adaptive=False,
    ):
        """
        初始化PDNorm。
        
        输入参数:
            - num_features: int, 特征维度
            - norm_layer: callable, 归一化层类
            - context_channels: int, 上下文通道数,默认256
            - conditions: tuple[str], 条件列表,默认("ScanNet", "S3DIS", "Structured3D")
            - decouple: bool, 是否解耦归一化(每个condition都有独立的归一化层), 默认True
            - adaptive: bool, 是否自适应归一化, 默认False。 NOTE: 如果使用, 那么forward的输入point必须包含context属性point.context, 用来传入 self.modulation 
        """
        super().__init__()
        # tuple[str], 条件列表
        self.conditions = conditions
        # bool, 是否解耦归一化
        self.decouple = decouple
        # bool, 是否自适应归一化
        self.adaptive = adaptive
        # 如果解耦,则为每个条件创建独立的归一化层
        if self.decouple:
            # nn.ModuleList, 每个条件一个归一化层
            self.norm = nn.ModuleList([norm_layer(num_features) for _ in conditions])
        else:
            # nn.Module, 共享的归一化层
            self.norm = norm_layer
        # 如果自适应,则创建调制网络
        if self.adaptive:
            # nn.Sequential, 调制网络,生成scale和shift
            self.modulation = nn.Sequential(
                nn.SiLU(), nn.Linear(context_channels, 2 * num_features, bias=True)
            )

    def forward(self, point):
        """
        前向传播,执行归一化。
        
        输入参数:
            - point: Point, 点云数据
        
        输出:
            - point: Point, 归一化后的点云数据
        
        功能:
            1. 根据条件选择归一化层
            2. 执行归一化
            3. 如果自适应,则应用scale和shift
        """
        # 检查必需的属性
        assert {"feat", "condition"}.issubset(point.keys())
        # 获取条件
        if isinstance(point.condition, str):
            # str, 单个条件
            condition = point.condition
        else:
            # str, 取第一个条件
            condition = point.condition[0]
        # 如果解耦,则根据条件选择对应的归一化层
        if self.decouple:
            assert condition in self.conditions
            # nn.Module, 对应条件的归一化层
            norm = self.norm[self.conditions.index(condition)]
        else:
            # nn.Module, 共享的归一化层
            norm = self.norm
        # torch.Tensor, (N, C), 执行归一化
        point.feat = norm(point.feat)
        # 如果自适应,则应用调制
        if self.adaptive:
            assert "context" in point.keys()
            # torch.Tensor, (N, C), shift和scale
            shift, scale = self.modulation(point.context).chunk(2, dim=1)  # .chunk: 将张量沿指定维度分割成多个块，返回一个张量元组
            # torch.Tensor, (N, C), 应用自适应归一化: feat * (1 + scale) + shift
            point.feat = point.feat * (1.0 + scale) + shift
        return point


class RPE(torch.nn.Module):
    """
    相对位置编码
    """
    
    def __init__(self, patch_size, num_heads):
        """
            初始化RPE。
            
            输入参数:
                - patch_size: int, patch大小
                - num_heads: int, 注意力头数
        """
        super().__init__()
        # int, patch大小
        self.patch_size = patch_size
        # int, 注意力头数
        self.num_heads = num_heads
        # int, 位置边界,启发式选取(假设patch的分布是正方体, 那么给它四倍大立方体对应的边长)  # FIXME: pos_bnd的目前默认选法不适合本项目
        self.pos_bnd = int((4*patch_size)**(1 / 3)   * 2)
        # int, 相对位置编码的数量
        self.rpe_num = 2 * self.pos_bnd + 1
        # torch.nn.Parameter, (3 * rpe_num, num_heads), 相对位置编码表(3表示3个维度, rpe_num表示编码范围)
        self.rpe_table = torch.nn.Parameter(torch.zeros(3 * self.rpe_num, num_heads))
        # (3 * rpe_num, num_heads)
        torch.nn.init.trunc_normal_(self.rpe_table, std=0.02)

    def forward(self, coord):
        """
        前向传播,计算相对位置编码。
        
        输入参数:
            - coord: torch.Tensor, (N, K, K, 3), 每个 patch 内所有点对的相对位置坐标; N为这个batch中patch的数量, K为每个patch的点数/大小, 3为坐标维度(x, y, z)
        
        输出:
            - out: torch.Tensor, (N, H, K, K), 相对位置编码; H为注意力头数
        """
        # torch.Tensor, (N, K, K, 3), 计算索引
        idx = (
            coord.clamp(-self.pos_bnd, self.pos_bnd) 
            + self.pos_bnd  # 转为正索引

            + torch.arange(3, device=coord.device) * self.rpe_num
        )
        # torch.Tensor, (N*K*K*3, num_heads), 从rpe_table中选择对应的编码
        out = self.rpe_table.index_select(dim=0, index=idx.reshape(-1))
        # torch.Tensor, (N, K, K, 3, num_heads) --> (N, K, K, num_heads)重塑形状并对坐标维度求和
        out = out.view(idx.shape + (-1,)).sum(3)
        # torch.Tensor, (N, H, K, K), 调整维度顺序
        out = out.permute(0, 3, 1, 2)  # (N, K, K, H) -> (N, H, K, K)
        return out


class MLP(nn.Module):
    """
    标准的两层MLP
    """
    
    def __init__(
        self,
        in_channels,
        hidden_channels=None,
        out_channels=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        """
        初始化MLP。
        
        输入参数:
            - in_channels: int, 输入通道数
            - hidden_channels: int 或 None, 隐藏层通道数,若为None则等于in_channels
            - out_channels: int 或 None, 输出通道数,若为None则等于in_channels
            - act_layer: callable, 激活函数类,默认nn.GELU
            - drop: float, dropout率,默认0.0
        """
        super().__init__()
        # int, 输出通道数
        out_channels = out_channels or in_channels
        # int, 隐藏层通道数
        hidden_channels = hidden_channels or in_channels
        # nn.Linear, 第一层全连接
        self.fc1 = nn.Linear(in_channels, hidden_channels)
        # nn.Module, 激活函数
        self.act = act_layer()
        # nn.Linear, 第二层全连接
        self.fc2 = nn.Linear(hidden_channels, out_channels)
        # nn.Dropout, dropout层
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        """
        输入参数:
            - x: torch.Tensor, (N, in_channels), 输入特征
        输出:
            - x: torch.Tensor, (N, out_channels), 输出特征
        """
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


# ----------------------------------------------- 激活函数解析 -----------------------------------------------
# dict[str, type], 支持的激活函数名到 nn.Module 类的映射
_ACT_LAYER_MAP = {
    "gelu": nn.GELU,
    "silu": nn.SiLU,
    "relu": nn.ReLU,
    "leakyrelu": nn.LeakyReLU,
}

def resolve_act_layer(act_layer_name: str):
    """
    将激活函数名称字符串解析为对应的 nn.Module 类。

    输入参数:
        - act_layer_name: str, 激活函数名称, 支持 "gelu", "silu", "relu", "leakyrelu"

    输出:
        - act_cls: type, 对应的 nn.Module 类(如 nn.GELU)
    """
    key = str(act_layer_name).lower()
    if key not in _ACT_LAYER_MAP:
        raise ValueError(f"不支持的 act_layer_name='{act_layer_name}', 可选: {list(_ACT_LAYER_MAP)}")
    return _ACT_LAYER_MAP[key]


class GatedTransition(nn.Module):
    """
    门控通道混合模块（SwiGLU 风格 FFN），不含内部残差: y = w3( act(w1(x)) * w2(x) ) → Dropout
    残差和归一化由外层 Block.forward 负责。

    输入参数:
        - in_channels: int, 输入输出通道数
        - mlp_ratio: int, 隐藏层膨胀倍率
        - act_layer: callable, 门控激活函数类
        - drop: float, 输出 dropout 概率

    前向输入:
        - x: torch.Tensor, (N, in_channels)

    前向输出:
        - y: torch.Tensor, (N, in_channels)
    """
    def __init__(self, in_channels: int, mlp_ratio: int, act_layer, drop: float):
        super().__init__()
        # int, 隐藏通道数 = in_channels * mlp_ratio
        hidden = int(in_channels) * int(mlp_ratio)
        # nn.Linear, (in_channels -> hidden), 门控分支 1
        self.w1 = nn.Linear(in_channels, hidden, bias=False)
        # nn.Linear, (in_channels -> hidden), 门控分支 2
        self.w2 = nn.Linear(in_channels, hidden, bias=False)
        # nn.Linear, (hidden -> in_channels), 输出投影
        self.w3 = nn.Linear(hidden, in_channels, bias=False)
        # nn.Module, 门控激活函数
        self.act = act_layer()
        # nn.Dropout, 输出 dropout
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        """
        输入参数:
            - x: torch.Tensor, (N, in_channels), 输入特征
        输出:
            - y: torch.Tensor, (N, in_channels), 门控混合后的特征(不含残差)
        """
        # torch.Tensor, (N, in_channels), 门控混合: act(w1(x)) * w2(x) → w3 → drop
        y = self.w3(self.act(self.w1(x)) * self.w2(x))
        return self.drop(y)


# ----------------------------------------------- 5大核心模块 -----------------------------------------------
class SerializedAttention(PointModule):
    def __init__(
        self,
        channels,
        num_heads,
        patch_size,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        order_index=0,
        enable_rpe=False,
        enable_flash=True,
        upcast_attention=True,
        upcast_softmax=True,
    ):
        """
            SerializedAttention, 序列化注意力机制.
            
            输入参数:
                - channels: int, 特征通道数(输入输出相同)
                - num_heads: int, 注意力头数
                - patch_size: int, patch大小
                - qkv_bias: bool, QKV线性层是否使用偏置,默认True
                - qk_scale: float 或 None, QK缩放因子,若为None则自动计算
                - attn_drop: float, 注意力dropout率,默认0.0
                - proj_drop: float, 投影dropout率,默认0.0
                - order_index: int, 使用的序列化顺序索引,默认0
                - enable_rpe: bool, 是否启用相对位置编码,默认False
                - enable_flash: bool, 是否启用Flash Attention,默认True
                - upcast_attention: bool, 是否在注意力计算时上转为float,默认True
                - upcast_softmax: bool, 是否在softmax时上转为float,默认True
        """

        super().__init__()
        # 检查通道数是否能被头数整除
        assert channels % num_heads == 0
        # int, 特征通道数
        self.channels = channels
        # int, 注意力头数
        self.num_heads = num_heads
        # float, QK缩放因子
        self.scale = qk_scale or (channels // num_heads) ** -0.5
        # int, 使用的序列化顺序索引
        self.order_index = order_index
        # bool, 是否在注意力计算时上转为float
        self.upcast_attention = upcast_attention
        # bool, 是否在softmax时上转为float
        self.upcast_softmax = upcast_softmax
        # bool, 是否启用相对位置编码
        self.enable_rpe = enable_rpe
        # bool, 是否启用Flash Attention
        self.enable_flash = enable_flash
        # 如果启用Flash Attention
        if enable_flash:
            # 检查配置兼容性
            assert (
                enable_rpe is False
            ), "Set enable_rpe to False when enable Flash Attention"
            assert (
                upcast_attention is False
            ), "Set upcast_attention to False when enable Flash Attention"
            assert (
                upcast_softmax is False
            ), "Set upcast_softmax to False when enable Flash Attention"
            assert flash_attn is not None, "Make sure flash_attn is installed."
            # int, patch大小
            self.patch_size = patch_size
            # float, 注意力dropout率
            self.attn_drop = attn_drop
        # 如果不启用Flash Attention
        else:
            # 当禁用flash attention时,我们仍然不想使用mask, 因此,patch size将自动设置为patch_size_max和点数的最小值
            # int, 最大patch大小
            self.patch_size_max = patch_size
            # int, patch大小(将在forward中动态设置)
            self.patch_size = 0
            # torch.nn.Dropout, 注意力dropout层
            self.attn_drop = torch.nn.Dropout(attn_drop)

        # torch.nn.Linear, QKV投影层
        self.qkv = torch.nn.Linear(channels, channels * 3, bias=qkv_bias)
        # torch.nn.Linear, 输出投影层
        self.proj = torch.nn.Linear(channels, channels)
        # torch.nn.Dropout, 输出dropout层
        self.proj_drop = torch.nn.Dropout(proj_drop)
        # torch.nn.Softmax, softmax层
        self.softmax = torch.nn.Softmax(dim=-1)
        # RPE或None, 相对位置编码模块
        self.rpe = RPE(patch_size, num_heads) if self.enable_rpe else None

    @torch.no_grad()
    def get_rel_pos(self, point, order):
        """
        计算 patch 内所有点对之间的相对位置坐标. patch的划分是按照序列化order的顺序
        
        输入参数:
            - point: Point (自定义类), 类似于字典的对象, 包含点云及其相关属性(如 grid_coord)
            - order: torch.Tensor, (N_pad,), 表示 patch 所在的序列化顺序索引, N_pad 为填充后的总点数
        
        输出:
            - rel_pos: torch.Tensor, (N_patch, K, K, 3), N_patch 为 patch 的总数, K 为 patch_size, 3 为 (x, y, z) 相对坐标
        """
        # int, 表示每个 patch 的大小 (点数)
        K = self.patch_size
        # str, 用于在 point 对象中缓存相对位置的键名，包含当前的序列索引
        rel_pos_key = f"rel_pos_{self.order_index}"
        # 如果当前序列索引对应的相对位置尚未计算并缓存，则进行计算
        if rel_pos_key not in point.keys():
            # torch.Tensor, (N_pad, 3), 对应序列化顺序下的网格坐标；grid_coord 原本是 (N, 3)
            grid_coord = point.grid_coord[order]
            # torch.Tensor, (N_patch, K, 3), 将扁平的序列重塑为 patch 结构
            grid_coord = grid_coord.reshape(-1, K, 3)
            # torch.Tensor, (N_patch, K, K, 3), 利用广播机制计算 patch 内每对点之间的三维相对位移
            # grid_coord.unsqueeze(2) 形状为 (N_patch, K, 1, 3)
            # grid_coord.unsqueeze(1) 形状为 (N_patch, 1, K, 3)
            point[rel_pos_key] = grid_coord.unsqueeze(2) - grid_coord.unsqueeze(1)
        # torch.Tensor, (N_patch, K, K, 3), 返回缓存或新计算的相对位置张量
        return point[rel_pos_key]

    @torch.no_grad()
    def get_padding_and_inverse(self, point):
        """
        获取填充和逆映射索引, 用于支持非定长序列的 Flash Attention. 填充机制为: 对于小于patch_size的batch不填充; 对于较大batch的每个样本i, 在它原本索引的后方加上前一个patch的索引的后 N_i_pad-N_i 个样本, 使得新点数 N_i_pad 恰好是 patch_size 的整数倍
        
        输入参数:
            - point: Point (自定义类)
        
        输出:
            - pad: torch.Tensor, (N_pad,), 对于填充后的点云来说, 目前(填充后)点云的第i个元素, 是填充前点云的第pad[i]个元素; 取值为0~N-1
            - unpad: torch.Tensor, (N,), 对于填充前的点云来说, 目前(填充前)点云的第i个元素, 是填充后点云的第unpad[i]个元素;  | (或等价地)对于填充后的点云来说, 代表所有"原本就是真实点"在填充后点云中的索引; 取值为 0~N_pad-1
            - cu_seqlens_key: torch.Tensor, (num_patches + 1,), 累积序列长度, 表示 Flash Attention 中每个 patch 的边界索引
        """
        # str, 存储填充索引的键名; 包含 patch_size 以避免不同 stage 之间缓存冲突
        pad_key = f"pad_{self.patch_size}"
        # str, 存储逆填充索引的键名
        unpad_key = f"unpad_{self.patch_size}"
        # str, 存储 Flash Attention 累积长度的键名
        cu_seqlens_key = f"cu_seqlens_{self.patch_size}"
        # 如果这些处理索引尚未计算, 则进行计算并缓存
        if (
            pad_key not in point.keys()
            or unpad_key not in point.keys()
            or cu_seqlens_key not in point.keys()
        ):
            # torch.Tensor, (batch_size,), 记录每个样本结束位置的偏移量
            offset = point.offset
            # torch.Tensor, (batch_size,), 记录每个样本包含的实际点数
            bincount = offset2bincount(offset)
            # torch.Tensor, (batch_size,), 填充后的点数(向上取整到patch_size的倍数)
            bincount_pad = (
                torch.div(
                    bincount + self.patch_size - 1,
                    self.patch_size,
                    rounding_mode="trunc",
                )
                * self.patch_size
            )
            # torch.Tensor, (batch_size,), 布尔掩码, 仅在点数不少于 patch_size 时才进行填充/对齐处理
            mask_pad = bincount > self.patch_size
            # torch.Tensor, (batch_size,), 结合掩码确定的每个样本最终处理的点数
            bincount_pad = ~mask_pad * bincount + mask_pad * bincount_pad
            # torch.Tensor, (batch_size + 1,), 填充offset(前面补1个0)
            _offset = nn.functional.pad(offset, (1, 0))
            # torch.Tensor, (batch_size + 1,), 填充对齐后的累积偏移量，首位补0
            _offset_pad = nn.functional.pad(torch.cumsum(bincount_pad, dim=0), (1, 0))

            # torch.Tensor, (N_pad,)
            pad = torch.arange(_offset_pad[-1], device=offset.device)
            # torch.Tensor, (N,)
            unpad = torch.arange(_offset[-1], device=offset.device)

            # cu_seqlens将会是一维张量, 形如 (num_patches + 1,), 代表本batch内, 所有patch的第一个点的索引
            cu_seqlens = []
            for i in range(len(offset)):   # 处理第 i 个样本
                # 第 i 个样本的原始有效点在填充后点云中的索引
                unpad[_offset[i] : _offset[i + 1]] += _offset_pad[i] - _offset[i]
                # 如果第 i 个样本的点数不是 patch_size 的倍数, 则需处理最后一个 patch
                if bincount[i] != bincount_pad[i]:
                    # 对 pad 数组末尾进行特定处理, 使得最后一个不足 patch_size 的部分通过重复前面点的方式对齐
                    pad[       # 第 i 个样本最后一个patch需要被填充的索引, 注意拿取的上一个patch的索引仍然必定在本样本[_offset_pad[i] : _offset_pad[i + 1])内
                        _offset_pad[i + 1]  - self.patch_size + (bincount[i] % self.patch_size) : 
                        _offset_pad[i + 1]
                    ] = pad[   # 再往前一个patch
                        _offset_pad[i + 1] - 2 * self.patch_size + (bincount[i] % self.patch_size) : 
                        _offset_pad[i + 1] - self.patch_size
                    ]
                # 把本样本(第i个样本)的索引转化为原始索引
                pad[_offset_pad[i] : _offset_pad[i + 1]] -= _offset_pad[i] - _offset[i]
                cu_seqlens.append(
                    torch.arange(
                        _offset_pad[i],
                        _offset_pad[i + 1],
                        step=self.patch_size,
                        dtype=torch.int32,
                        device=offset.device,
                    )
                )
            # 将生成的 pad 索引存入 point 对象
            point[pad_key] = pad
            point[unpad_key] = unpad
            point[cu_seqlens_key] = nn.functional.pad(
                torch.concat(cu_seqlens), (0, 1), value=_offset_pad[-1]   # 最右边补1个_offset_pad[-1]
            )

        return point[pad_key], point[unpad_key], point[cu_seqlens_key]

    def forward(self, point):
        """
        前向传播, 执行基于序列化(Serialized)的自注意力计算。
        
        输入参数:
            - point: Point (自定义类), 包含点云特征 feat (N, C) 及各种几何顺序/偏移信息
        
        输出:
            - point: Point (自定义类), 更新了 feat 属性的点云数据
        """
        # 如果不启用专用的 Flash Attention, 则根据当前数据量动态计算 patch 大小以防 OOM
        if not self.enable_flash:
            # int, 选取 patch_size_max 与场景中最小 batch 点数中的较小者作为实际 patch 大小
            self.patch_size = min(
                offset2bincount(point.offset).min().tolist(), self.patch_size_max
            )

        # int, 预定义的注意力头数
        H = self.num_heads
        # int, 当前使用的 patch 大小 (每个组内的点数)
        K = self.patch_size
        # int, 特征总通道数
        C = self.channels
        # pad: (N_pad,), unpad: (N,), cu_seqlens: (num_patches + 1,)
        pad, unpad, cu_seqlens = self.get_padding_and_inverse(point)

        # point.serialized_order[self.order_index]:  (N, ), ..[j]代表第j个点经过序列化后所处的位置
        # pad: (N_pad,), 对于填充后的点云来说, 第i个位置是填充前点云的第pad[i]个位置
        # order: (N_pad,), 应用序列化+填充后, 每个位置所对应的最原始点云索引
        order = point.serialized_order[self.order_index][pad]
        # point.serialized_inverse[self.order_index]: (N, ), ..[j]代表第j个点在序列化前所处的位置
        # unpad: (N,), 对于填充前的点云来说, 第i个元素是填充后点云的第unpad[i]个元素
        # inverse: (N,), 代表原始的第i个点在经过 序列化+填充 之后在填充后点云中的索引是 inverse[i]
        inverse = unpad[point.serialized_inverse[self.order_index]]
        # qkv: (N_pad, 3 * C), 3*C 包含了 Q, K, V 三个部分
        qkv = self.qkv(point.feat)[order]

        # ------------------------------------------------------------
        # 情况一: 普通注意力计算 (手动实现, 支持 RPE 和高精度模式)
        # ------------------------------------------------------------
        if not self.enable_flash:
            # q, k, v 分别为 torch.Tensor, 形状均为 (N_patch, H, K, C // H)
            # reshape为 (N_patch, K,   3, H, C//H) --> (3, N_patch, H, K, C//H) --> 拆分为 (N_patch, H, K, C//H), K=patch_size
            q, k, v = (
                qkv.reshape(-1, K, 3, H, C // H).permute(2, 0, 3, 1, 4).unbind(dim=0)
            )
            # 如果配置上转, 则将 Q, K 转为 float32 以在此后的矩阵乘法中减少精度损失
            if self.upcast_attention:
                # torch.Tensor, (N_patch, H, K, C // H), float32 类型
                q = q.float()
                # torch.Tensor, (N_patch, H, K, C // H), float32 类型
                k = k.float()
            
            # torch.Tensor, (N_patch, H, K, K), 计算点对之间的注意力分数 (scaled dot-product)
            attn = (q * self.scale) @ k.transpose(-2, -1)
            
            # 如果启用了相对位置编码 (Relative Position Encoding)
            if self.enable_rpe:
                # self.get_rel_pos 获取形状为 (N_patch, K, K, 3) 的相对坐标, self.rpe 生成形状为 (N_patch, H, K, K) 的编码并叠加到分数上
                attn = attn + self.rpe(self.get_rel_pos(point, order))
            
            # 如果配置上转, Softmax 计算使用 float32 保证数值稳定性
            if self.upcast_softmax:
                # torch.Tensor, (N_patch, H, K, K), float32 类型
                attn = attn.float()
            
            # torch.Tensor, (N_patch, H, K, K), 在最后一个维度上执行 Softmax 归一化
            attn = self.softmax(attn)
            # torch.Tensor, (N_patch, H, K, K), 应用注意力 dropout 层并将类型转回原 QKV 精度 (如 FP16/BF16)
            attn = self.attn_drop(attn).to(qkv.dtype)
            
            # (N_patch, H, K, K) @ (N_patch, H, K, C//H) --> (N_patch, H, K, C//H) --> 仅仅换维度位置(N_patch, K, H, C//H) --> (N_pad, C)
            # @ 操作：PyTorch 会自动将前面所有的维度 (N_patch, H) 视作独立的 Batch，然后对最后两个维度执行标准的 2D 矩阵乘法： [K, K] @ [K, C // H] -> [K, C // H]
            feat = (attn @ v).transpose(1, 2).reshape(-1, C)
        
        # ------------------------------------------------------------
        # 情况二: Flash Attention 加速计算 (高吞吐量)
        # ------------------------------------------------------------
        else:
            # torch.Tensor, (N_pad, C), 使用变长 Flash Attention API 计算注意力输出
            feat = flash_attn.flash_attn_varlen_qkvpacked_func(
                qkv.half().reshape(-1, 3, H, C // H),  # 将 qkv 转为 half 精度并重塑为要求的输入格式
                cu_seqlens,                            # torch.Tensor, (num_patches + 1,), 序列边界信息
                max_seqlen=self.patch_size,            # int, patch 的最大长度限制
                dropout_p=self.attn_drop if self.training else 0,  # float, dropout 概率
                softmax_scale=self.scale,              # float, scale 缩放因子
            ).reshape(-1, C)
            # torch.Tensor, (N_pad, C), 将半精度结果还原回原始数据类型控制精度
            feat = feat.to(qkv.dtype)
        
        # inverse: (N,), 代表原始的第i个点在经过 序列化+填充 之后在填充后点云中的索引是 inverse[i]
        # (N_pad, C) --> (N, C)
        feat = feat[inverse]


        # ------------------------------------------------------------
        # 输出后处理
        # ------------------------------------------------------------
        # torch.Tensor, (N, C), 经过最终的线性投影层进行特征融合
        feat = self.proj(feat)
        # torch.Tensor, (N, C), 应用输出 dropout
        feat = self.proj_drop(feat)
        
        # 将计算得到的特征赋值回 point 对象的 feat 属性
        point.feat = feat
        # 返回更新后的 point 对象
        return point



class PointConvCPE(PointModule):
    # int, 内部固定瓶颈缩放比
    _BOTTLENECK_RATIO: int = 4

    def __init__(
        self,
        channels: int,
        receptive_field: float,
        max_neighbors: int,
        cache_key: str | None,
        norm_layer=None,
    ):
        """
            基于连续世界坐标的点云卷积条件位置编码(CPE)。

            在每个 encoder/decoder stage 的 Block 内，替代稀疏卷积 CPE。
            按世界坐标建 radius 邻域图，对邻居特征做低秩 MLP-weighted 聚合，再经 Linear + Norm 得到位置增量。

            输入参数:
                - channels: int, 标量, 特征通道数(输入输出相同)
                - receptive_field: float, 标量, 世界坐标系下的感受野半径(Å)
                - max_neighbors: int, 标量, 每个点的最大邻居数
                - bottleneck_ratio: int, 标量, 瓶颈缩放比(内部固定, 不暴露配置)
                - cache_key: str 或 None, 邻域图缓存键(同 stage 内复用)
                - norm_layer: callable 或 None, 归一化层类

            前向输入:
                - point: Point, 点云数据(包含 feat, coord, batch)

            前向输出:
                - point: Point, 更新了 feat 的点云数据(输出为纯 delta, 残差在 Block.forward 外层做)
        """
        super().__init__()
        # int, 特征通道数
        self.channels = int(channels)
        # float, 世界坐标感受野半径(Å)
        self.receptive_field = float(receptive_field)
        # int, 每个点的最大邻居数
        self.max_neighbors = int(max_neighbors)
        # str 或 None, 邻域图缓存键
        self.cache_key = cache_key

        # int, 瓶颈通道数: C_r = max(C // 4, 16)
        c_r = max(self.channels // self._BOTTLENECK_RATIO, 16)
        self.c_r = c_r

        # nn.Linear, (C -> C_r), 邻居特征低秩投影
        self.w_v = nn.Linear(self.channels, c_r)
        # nn.Sequential, (3 -> C_r -> C_r), 相对坐标 -> 权重 MLP
        self.mlp_w = nn.Sequential(
            nn.Linear(3, c_r),
            nn.GELU(),
            nn.Linear(c_r, c_r),
        )
        # nn.Linear, (C_r -> C), 投影回原维度
        self.w_o = nn.Linear(c_r, self.channels)
        # nn.Linear, (C -> C), 保留的 post linear
        self.linear_post = nn.Linear(self.channels, self.channels)
        # norm_layer, 归一化层
        self.norm = norm_layer(self.channels) if norm_layer is not None else nn.Identity()

    def _get_or_build_graph(self, point):
        """
        获取或构建邻域图。优先从 point 对象的缓存中取，否则新建并缓存。

        输入参数:
            - point: Point, 点云数据(需含 coord, batch)

        输出:
            - edge_index: torch.Tensor, (2, E), 邻域边索引, [0]为目标点(就是0呢), [1]为源点
        """
        graph_cache_attr = f"_pointconv_graph_{self.cache_key}" if self.cache_key else None
        if graph_cache_attr is not None and graph_cache_attr in point.keys():
            # 缓存命中
            return point[graph_cache_attr]

        # torch.Tensor, (2, E), 新建 radius graph
        edge_index = torch_cluster.radius_graph(
            x=point.coord,
            r=self.receptive_field,
            batch=point.batch,
            max_num_neighbors=self.max_neighbors,
            loop=False,
        )
        if graph_cache_attr is not None:
            point[graph_cache_attr] = edge_index
        return edge_index

    def forward(self, point):
        """
        前向传播。输出为纯 delta(不含残差), 残差在 Block.forward 外层做。

        输入参数:
            - point: Point, 点云数据

        输出:
            - point: Point, 更新了 feat 的点云数据
        """
        # torch.Tensor, (2, E), 邻域边索引
        edge_index = self._get_or_build_graph(point)
        # torch.Tensor, (E,), 目标点索引
        idx_i = edge_index[0]
        # torch.Tensor, (E,), 源点索引
        idx_j = edge_index[1]

        # torch.Tensor, (E, 3), 归一化相对坐标
        rel_coord = (point.coord[idx_j] - point.coord[idx_i]) / self.receptive_field
        # torch.Tensor, (E, C_r), 相对坐标权重
        w_ij = self.mlp_w(rel_coord)
        # torch.Tensor, (E, C_r), 邻居特征低秩投影
        v_j = self.w_v(point.feat[idx_j])
        # torch.Tensor, (E, C_r), 加权消息
        m_ij = w_ij * v_j
        # torch.Tensor, (N, C_r), 聚合邻居消息
        agg_i = torch_scatter.scatter_mean(m_ij, idx_i, dim=0, dim_size=point.feat.shape[0])
        # torch.Tensor, (N, C), 投影回原维度
        delta_i = self.w_o(agg_i)
        # torch.Tensor, (N, C), post linear + norm
        delta_i = self.norm(self.linear_post(delta_i))
        # 更新点特征(纯 delta)
        point.feat = delta_i
        return point



class Block(PointModule):
    """
    Transformer Block,Transformer块。
    
    功能:
        包含条件位置编码(CPE)、序列化注意力、MLP和残差连接的标准Transformer块
    """
    
    def __init__(
        self,
        channels,
        num_heads,
        patch_size=48,
        cpe_kernel_size=5,
        cpe_impl="sparseconv",
        cpe_receptive_field=2.0,
        pointconv_block_max_neighbors=16,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
        act_layer=nn.GELU,
        pre_norm=True,
        order_index=0,
        cpe_indice_key=None,
        enable_rpe=False,
        enable_flash=True,
        upcast_attention=True,
        upcast_softmax=True,
        ffn_type="mlp",
    ):
        """
            初始化Block。
            
            输入参数:
                - channels: int, 特征通道数
                - num_heads: int, 注意力头数
                - patch_size: int, patch大小,默认48
                - cpe_kernel_size: int, Block 内 CPE 稀疏卷积的卷积核大小(仅 sparseconv 模式)
                - cpe_impl: str, CPE 实现方式, "sparseconv" / "pointconv" / "none"(不使用CPE)
                - cpe_receptive_field: float, 世界坐标感受野半径(Å)(仅 pointconv 模式)
                - pointconv_block_max_neighbors: int, 点云卷积 CPE 最大邻居数(仅 pointconv 模式)
                - mlp_ratio: float, MLP隐藏层通道数比例,默认4.0
                - qkv_bias: bool, QKV线性层是否使用偏置,默认True
                - qk_scale: float 或 None, QK缩放因子,若为None则自动计算
                - attn_drop: float, 注意力dropout率,默认0.0
                - proj_drop: float, 投影dropout率,默认0.0
                - drop_path: float, 随机深度drop率,默认0.0
                - norm_layer: callable, 归一化层类,默认nn.LayerNorm
                - act_layer: callable, 激活函数类,默认nn.GELU
                - pre_norm: bool, 是否使用预归一化,默认True
                - order_index: int, 使用的序列化顺序索引,默认0
                - cpe_indice_key: str 或 None, CPE的索引键(sparseconv 做缓存; pointconv 做邻域图缓存)
                - enable_rpe: bool, 是否启用相对位置编码,默认False
                - enable_flash: bool, 是否启用Flash Attention,默认True
                - upcast_attention: bool, 是否在注意力计算时上转为float,默认True
                - upcast_softmax: bool, 是否在softmax时上转为float,默认True
                - ffn_type: str, FFN 类型, "mlp"(经典两层MLP) / "gated"(GatedTransition 门控FFN) / "none"(无FFN), 默认 "mlp"
        """
        super().__init__()
        # int, 特征通道数
        self.channels = channels
        # str, CPE 实现方式
        self.cpe_impl = str(cpe_impl)
        # int, Block 内 CPE 稀疏卷积核大小; 原模型默认 kernel_size=3, 当前项目默认改为 5。
        self.cpe_kernel_size = int(cpe_kernel_size)
        # bool, 是否使用预归一化
        self.pre_norm = pre_norm

        # 条件位置编码(CPE): 根据 cpe_impl 选择实现
        if self.cpe_impl == "none":
            # None, 不使用 CPE(用于 embed head / atom head 等无稀疏卷积场景)
            self.cpe = None
        elif self.cpe_impl == "sparseconv":
            # PointSequential, 稀疏卷积 CPE: SubMConv3d -> Linear -> Norm
            self.cpe = PointSequential(
                spconv.SubMConv3d(
                    channels,
                    channels,
                    kernel_size=self.cpe_kernel_size,
                    bias=True,
                    indice_key=cpe_indice_key,
                ),
                nn.Linear(channels, channels),
                norm_layer(channels),
            )
        elif self.cpe_impl == "pointconv":
            # PointConvCPE, 点云卷积 CPE
            self.cpe = PointConvCPE(
                channels=channels,
                receptive_field=float(cpe_receptive_field),
                max_neighbors=int(pointconv_block_max_neighbors),
                cache_key=cpe_indice_key,
                norm_layer=norm_layer,
            )
        else:
            raise ValueError(f"Block: cpe_impl 必须是 'sparseconv'/'pointconv'/'none', 当前为 '{self.cpe_impl}'")

        # PointSequential, 第一个归一化层
        self.norm1 = PointSequential(norm_layer(channels))
        # SerializedAttention, 序列化注意力层
        self.attn = SerializedAttention(
            channels=channels,
            patch_size=patch_size,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            order_index=order_index,
            enable_rpe=enable_rpe,
            enable_flash=enable_flash,
            upcast_attention=upcast_attention,
            upcast_softmax=upcast_softmax,
        )
        # FFN 层: "mlp"(经典两层MLP)、"gated"(GatedTransition 门控FFN)、"none"(无FFN)
        self.ffn_type = str(ffn_type).lower()
        if self.ffn_type == "none":
            self.norm2 = None
            self.mlp = None
        elif self.ffn_type == "gated":
            self.norm2 = PointSequential(norm_layer(channels))
            self.mlp = PointSequential(
                GatedTransition(
                    in_channels=channels,
                    mlp_ratio=int(mlp_ratio),
                    act_layer=act_layer,
                    drop=proj_drop,
                )
            )
        else:
            # 默认 "mlp"
            self.norm2 = PointSequential(norm_layer(channels))
            self.mlp = PointSequential(
                MLP(
                    in_channels=channels,
                    hidden_channels=int(channels * mlp_ratio),
                    out_channels=channels,
                    act_layer=act_layer,
                    drop=proj_drop,
                )
            )
        # PointSequential, 随机深度层
        self.drop_path = PointSequential(
            DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        )

    def forward(self, point: Point):
        """
        前向传播,执行Transformer块。
        
        输入参数:
            - point: Point, 点云数据
        
        输出:
            - point: Point, 处理后的点云数据
        
        运算过程:
            1. CPE + 残差连接(若 cpe_impl != "none")
            2. 归一化 + 注意力 + 残差连接
            3. 归一化 + MLP + 残差连接(若 ffn_type != "none")
        """
        # --- CPE + 残差(若启用) ---
        if self.cpe is not None:
            # torch.Tensor, (N, C), 保存快捷连接
            shortcut = point.feat
            # Point, 应用条件位置编码(CPE)
            point = self.cpe(point)
            # torch.Tensor, (N, C), CPE残差连接
            point.feat = shortcut + point.feat

        # --- Attention + 残差 ---
        # torch.Tensor, (N, C), 更新快捷连接
        shortcut = point.feat
        if self.pre_norm:
            point = self.norm1(point)
        # Point, 应用注意力(带"随机深度"：本次运算有一定概率退化为恒等映射)
        point = self.drop_path(self.attn(point))
        # torch.Tensor, (N, C), 注意力残差连接
        point.feat = shortcut + point.feat
        if not self.pre_norm:
            point = self.norm1(point)

        # --- FFN + 残差(若启用) ---
        if self.mlp is not None:
            # torch.Tensor, (N, C), 更新快捷连接
            shortcut = point.feat
            if self.pre_norm:
                point = self.norm2(point)
            # Point, 应用MLP(带随机深度)
            point = self.drop_path(self.mlp(point))
            # torch.Tensor, (N, C), MLP残差连接
            point.feat = shortcut + point.feat
            if not self.pre_norm:
                point = self.norm2(point)

        # 更新稀疏卷积特征(仅在有 sparse_conv_feat 时)
        if hasattr(point, "keys") and "sparse_conv_feat" in point.keys():
            point.sparse_conv_feat = point.sparse_conv_feat.replace_feature(point.feat)
        return point



class SerializedPooling(PointModule):
    """
    Serialized Pooling,序列化池化层。
    
    功能:
        基于序列化编码的下采样池化,支持多种归约方式和可追踪性
    """
    
    def __init__(
        self,
        in_channels,
        out_channels,
        stride=2,
        norm_layer=None,
        act_layer=None,
        reduce="max",
        shuffle_orders=True,
        traceable=True,  # record parent and cluster
    ):
        """
        初始化SerializedPooling。
        
        输入参数:
            - in_channels: int, 输入通道数
            - out_channels: int, 输出通道数
            - stride: int, 池化步长,必须是2的幂次(2, 4, 8),默认2
            - norm_layer: callable 或 None, 归一化层类
            - act_layer: callable 或 None, 激活函数类
            - reduce: str, 点特征的归约方式,可选"sum","mean","min","max",默认"max"
            - shuffle_orders: bool, 是否打乱多个序列化顺序,默认True
            - traceable: bool, 是否记录父节点和簇信息,默认True
        """
        super().__init__()
        assert stride == 2 ** (math.ceil(stride) - 1).bit_length()  # 2, 4, 8
        assert reduce in ["sum", "mean", "min", "max"]
        self.in_channels = in_channels
        self.out_channels = out_channels
        # int, 池化步长
        self.stride = stride
        # str, 归约方式
        self.reduce = reduce
        # bool, 是否打乱多个序列化顺序
        self.shuffle_orders = shuffle_orders
        # bool, 是否记录父节点和簇信息
        self.traceable = traceable
        # nn.Linear, 特征投影层
        self.proj = nn.Linear(in_channels, out_channels)
        # PointSequential, 归一化层(如果指定)
        if norm_layer is not None:
            self.norm = PointSequential(norm_layer(out_channels))
        # PointSequential, 激活函数(如果指定)
        if act_layer is not None:
            self.act = PointSequential(act_layer())

    def forward(self, point: Point):
        """
        前向传播,执行序列化池化。
        
        输入参数:
            - point: Point, 点云数据
        
        输出:
            - point: Point, 池化后的点云数据. 重新构造池化后新点云的 code, order, inverse
        
        功能:
            1. 根据序列化编码进行聚类, 对每个簇进行归约, 更新点云类Point的序列化信息
            2. 如果可追踪,则记录父节点和簇信息
        """
        # int, 计算池化深度(以2为底的对数)
        pooling_depth = (math.ceil(self.stride) - 1).bit_length()
        # 如果池化深度超过序列化深度,则不进行池化
        if pooling_depth > point.serialized_depth:
            pooling_depth = 0
        # 检查序列化信息是否存在
        assert {
            "serialized_code",
            "serialized_order",
            "serialized_inverse",
            "serialized_depth",
        }.issubset(point.keys()), "Run point.serialization() point cloud before SerializedPooling"

        # torch.Tensor, (k, N), k表示不同的编码规则; 等价于所有的xyz坐标 // 2^{pooling_depth}
        code = point.serialized_code >> pooling_depth * 3  # 右移编码以进行池化, 注意池化规则仅仅按照第0种编码规则
        # torch.Tensor, (N',), 去重后的唯一编码值本身 (代表池化产生的簇的空间身份码/莫顿坐标), 取值范围同 code[0]
        # torch.Tensor, (N,), cluster[i]表示原始点云的第i个点的簇索引, 取值0~N'-1
        # torch.Tensor, (N',), counts[i]表示第i个簇的点数, 取值1~N
        code_, cluster, counts = torch.unique(
            # torch.Tensor, (N,), 输入需要去重的张量(这里是第0种规则下所有输入点的空间编码值)
            code[0],
            # bool, 控制返回值是否按升序排列
            sorted=True,
            # bool, 是否返回原输入元素在新去重结果中的索引(即生成 cluster 变量，告诉你“旧点”变成了哪个“新点”)
            return_inverse=True,
            # bool, 是否返回每个去重后元素出现的次数(即生成 counts 变量，告诉你“新点”合并了几个“旧点”)
            return_counts=True,
        )
        # torch.Tensor, (N,), 按照原始点云所属的簇id大小进行排序所返回的 原始点云的索引
        _, indices = torch.sort(cluster)
        # torch.Tensor, (N' + 1,), [idx_ptr[i], idx_ptr[i+1])的长度 = 第i个簇的点的个数(i=0,...,N'-1)
        idx_ptr = torch.cat([counts.new_zeros(1), torch.cumsum(counts, dim=0)])
        # torch.Tensor, (N',), 每个簇的第一个点的 原始点云索引
        head_indices = indices[idx_ptr[:-1]]

        # 一个簇的第一个元素就是下采样后剩下的点
        # torch.Tensor, (k, N'), 下采样后的序列化编码
        code = code[:, head_indices]
        order = torch.argsort(code)
        # (k, N'), inverse[i][index[i][j]] = src[i][j] = j
        inverse = torch.zeros_like(order).scatter_(
            dim=1,
            index=order,
            src=torch.arange(0, code.shape[1], device=order.device).repeat(code.shape[0], 1),   # src[i][j] = j
        )

        # 如果需要打乱多个序列化顺序
        if self.shuffle_orders:
            perm = torch.randperm(code.shape[0])
            code = code[perm]
            order = order[perm]
            inverse = inverse[perm]

        # 收集信息
        # Dict, 创建新的点云字典
        point_dict = Dict(
            # torch.Tensor, (N', out_channels), 池化后的特征
            feat=torch_scatter.segment_csr(
                self.proj(point.feat)[indices],  # 排好序的输入特征 (N, C)
                idx_ptr,                         # 每一段的起止指针 (N'+1, )
                reduce=self.reduce               # 怎么打包？"max" 或 "mean" 等
            ),
            # torch.Tensor, (N', 3), 池化后的坐标(均值)
            coord=torch_scatter.segment_csr(point.coord[indices], idx_ptr, reduce="mean"),
            # torch.Tensor, (N', 3), (向下整除)// 2^n, 池化后的网格坐标
            grid_coord=point.grid_coord[head_indices] >> pooling_depth,
            # torch.Tensor, (k, N'), 下采样后的编码
            serialized_code=code,
            # torch.Tensor, (k, N'), 下采样后的顺序
            serialized_order=order,
            # torch.Tensor, (k, N'), 下采样后的逆映射
            serialized_inverse=inverse,
            # int, 下采样后的序列化深度
            serialized_depth=point.serialized_depth - pooling_depth,
            # torch.Tensor, (N',), 下采样后的batch索引
            batch=point.batch[head_indices],
        )

        # 如果存在条件,则传递
        if "condition" in point.keys():
            point_dict["condition"] = point.condition
        # 如果存在上下文,则传递
        if "context" in point.keys():
            point_dict["context"] = point.context

        # 如果可追踪,则记录父节点和簇信息
        if self.traceable:
            # torch.Tensor, (N,), 每个点所属的簇索引
            point_dict["pooling_inverse"] = cluster
            # Point, 父节点(池化前的点云)
            point_dict["pooling_parent"] = point
        # Point, 创建新的点云对象
        point = Point(point_dict)
        # 如果存在归一化层,则应用
        if self.norm is not None:
            point = self.norm(point)
        # 如果存在激活函数,则应用
        if self.act is not None:
            point = self.act(point)
        # 稀疏化点云
        point.sparsify()
        return point



class SerializedUnpooling(PointModule):
    """
    Serialized Unpooling,序列化上采样层。
    """
    
    def __init__(
        self,
        in_channels,
        skip_channels,
        out_channels,
        norm_layer=None,
        act_layer=None,
        traceable=False,  # record parent and cluster
    ):
        """
        初始化SerializedUnpooling。
        
        输入参数:
            - in_channels: int, 输入通道数(来自解码器)
            - skip_channels: int, 跳跃连接中, 编码器输出的特征通道数
            - out_channels: int, 输出通道数
            - norm_layer: callable 或 None, 归一化层类
            - act_layer: callable 或 None, 激活函数类
            - traceable: bool, 是否记录父节点和簇信息,默认False
        """
        super().__init__()
        # PointSequential, 输入特征投影
        self.proj = PointSequential(nn.Linear(in_channels, out_channels))
        # PointSequential, 跳跃连接特征投影
        self.proj_skip = PointSequential(nn.Linear(skip_channels, out_channels))

        if norm_layer is not None:
            self.proj.add(norm_layer(out_channels))
            self.proj_skip.add(norm_layer(out_channels))

        if act_layer is not None:
            self.proj.add(act_layer())
            self.proj_skip.add(act_layer())

        self.traceable = traceable

    def forward(self, point):
        """
        前向传播,执行序列化上采样。
        
        输入参数:
            - point: Point, 池化后的点云数据(包含pooling_parent和pooling_inverse)
        
        输出:
            - parent: Point, 上采样后的点云数据(与原始分辨率一致)
        
        功能:
            1. 从point中获取父节点和逆映射
            2. 投影输入特征和跳跃连接特征
            3. 通过逆映射将特征上采样并与跳跃连接相加
        """
        assert "pooling_parent" in point.keys()
        assert "pooling_inverse" in point.keys()
        # Point, 获取父节点(池化前的点云)
        parent = point.pop("pooling_parent")
        # torch.Tensor, (N,), 获取逆映射(每个点所属的簇索引), 构造见上一个 SerializedPooling
        inverse = point.pop("pooling_inverse")
        # Point, 投影输入特征
        point = self.proj(point)
        # Point, 投影跳跃连接特征
        parent = self.proj_skip(parent)
        # torch.Tensor, (N, out_channels), 上采样特征并与跳跃连接相加
        parent.feat = parent.feat + point.feat[inverse]

        # 如果可追踪,则记录上采样父节点
        if self.traceable:
            parent["unpooling_parent"] = point
        return parent
# ----------------------------------------------- 5大核心模块 -----------------------------------------------









# 仿照上面 class PointConvCPE(PointModule)
class PointConvEmbedding(PointModule):
    def __init__(
        self,
        in_channels: int,
        embed_channels: int,
        receptive_field: float,
        max_neighbors: int,
        norm_layer=None,
        act_layer=None,
    ):
        """
            基于连续世界坐标的点云卷积 Embedding 层(和 Embedding 对立)
            以 atom_coord_centered_world 为准，按世界坐标建 radius 邻域图，对邻域特征做 MLP-weighted 聚合，替代稀疏卷积 stem。

            输入参数:
                - in_channels: int, 标量, 输入特征维度
                - embed_channels: int, 标量, 输出嵌入维度
                - receptive_field: float, 标量, 世界坐标系下的感受野半径(Å)
                - max_neighbors: int, 标量, 每个点的最大邻居数
                - norm_layer: callable 或 None
                - act_layer: callable 或 None

            前向输入:
                - point: Point, 点云数据(包含 feat, coord, batch)

            前向输出:
                - point: Point, 嵌入后的点云数据
        """
        super().__init__()
        self.in_channels = int(in_channels)
        self.embed_channels = int(embed_channels)
        self.receptive_field = float(receptive_field)
        self.max_neighbors = int(max_neighbors)

        # nn.Linear, (in_channels -> embed_channels), 邻居特征投影
        self.w_v = nn.Linear(self.in_channels, self.embed_channels)
        # nn.Sequential, (3 -> embed_channels -> embed_channels), 相对坐标 -> 权重 MLP
        self.mlp_w = nn.Sequential(
            nn.Linear(3, self.embed_channels),
            nn.GELU(),
            nn.Linear(self.embed_channels, self.embed_channels),
        )
        self.norm = norm_layer(self.embed_channels) if norm_layer is not None else None
        self.act = act_layer() if act_layer is not None else None

    def forward(self, point):
        """
        前向传播，执行点云卷积嵌入。

        输入参数:
            - point: Point, 点云数据

        输出:
            - point: Point, 嵌入后的点云数据
        """
        # torch.Tensor, (2, E), 邻域边索引
        edge_index = torch_cluster.radius_graph(
            x=point.coord,
            r=self.receptive_field,
            batch=point.batch,
            max_num_neighbors=self.max_neighbors,
            loop=False,
        )
        # torch.Tensor, (E,), 目标点索引 / 源点索引
        idx_i = edge_index[0]
        idx_j = edge_index[1]

        # torch.Tensor, (E, 3), 归一化相对坐标
        rel_coord = (point.coord[idx_j] - point.coord[idx_i]) / self.receptive_field
        # torch.Tensor, (E, embed_channels), 相对坐标权重
        w_ij = self.mlp_w(rel_coord)
        # torch.Tensor, (E, embed_channels), 邻居特征投影
        v_j = self.w_v(point.feat[idx_j])
        # torch.Tensor, (E, embed_channels), 加权消息
        m_ij = w_ij * v_j
        # torch.Tensor, (N, embed_channels), 聚合
        agg_i = torch_scatter.scatter_mean(m_ij, idx_i, dim=0, dim_size=point.feat.shape[0])

        # 更新特征
        point.feat = agg_i
        if self.norm is not None:
            point.feat = self.norm(point.feat)
        if self.act is not None:
            point.feat = self.act(point.feat)
        # 保持 point.feat 与 sparse_conv_feat 一致, 以兼容 pointconv embedding + sparseconv CPE 的混合路径
        if "sparse_conv_feat" in point.keys():
            point.sparse_conv_feat = point.sparse_conv_feat.replace_feature(point.feat)
        return point


class Embedding(PointModule):
    def __init__(
        self,
        in_channels,
        embed_channels,
        embedding_kernel_size=7,
        embedding_impl="sparseconv",
        embedding_receptive_field=5.0,
        pointconv_embed_max_neighbors=32,
        norm_layer=None,
        act_layer=None,
    ):
        """
            对最原始特征的嵌入层(内部 PointConvEmbedding 和 稀疏卷积嵌入 二选一)
            
            输入参数:
                - in_channels: int, 输入通道数
                - embed_channels: int, 嵌入通道数
                - embedding_kernel_size: int, embedding 稀疏卷积核大小(仅 sparseconv 模式)
                - embedding_impl: str, 实现方式, "sparseconv" 或 "pointconv"
                - embedding_receptive_field: float, 世界坐标感受野半径(Å)(仅 pointconv 模式)
                - pointconv_embed_max_neighbors: int, 点云卷积最大邻居数(仅 pointconv 模式)
                - norm_layer: callable 或 None, 归一化层类
                - act_layer: callable 或 None, 激活函数类
        """
        super().__init__()
        # int, 输入通道数
        self.in_channels = in_channels
        # int, 嵌入通道数
        self.embed_channels = embed_channels
        # str, embedding 实现方式
        self.embedding_impl = str(embedding_impl)
        # int, embedding 稀疏卷积核大小
        self.embedding_kernel_size = int(embedding_kernel_size)

        if self.embedding_impl == "sparseconv":
            # PointSequential, 稀疏卷积 stem
            self.stem = PointSequential(
                conv=spconv.SubMConv3d(
                    in_channels,
                    embed_channels,
                    kernel_size=self.embedding_kernel_size,
                    padding=1,
                    bias=False,
                    indice_key="stem",
                )
            )
            if norm_layer is not None:
                self.stem.add(norm_layer(embed_channels), name="norm")
            if act_layer is not None:
                self.stem.add(act_layer(), name="act")
            self._pointconv_embed = None
        elif self.embedding_impl == "pointconv":
            # PointConvEmbedding, 点云卷积 embedding
            self._pointconv_embed = PointConvEmbedding(
                in_channels=in_channels,
                embed_channels=embed_channels,
                receptive_field=float(embedding_receptive_field),
                max_neighbors=int(pointconv_embed_max_neighbors),
                norm_layer=norm_layer,
                act_layer=act_layer,
            )
            self.stem = None
        else:
            raise ValueError(f"Embedding: embedding_impl 必须是 'sparseconv' 或 'pointconv', 当前为 '{self.embedding_impl}'")


    def forward(self, point: Point):
        """
            前向传播,执行特征嵌入。
            
            输入参数:
                - point: Point, 点云数据
            
            输出:
                - point: Point, 嵌入后的点云数据
        """
        if self.embedding_impl == "sparseconv":
            # Point, 应用稀疏卷积 stem
            point = self.stem(point)
        else:
            # Point, 应用点云卷积 embedding
            point = self._pointconv_embed(point)
        return point


class PointTransformerV3(PointModule):
    """
        Point Transformer V3 (PTv3) 主干网络。

        功能:
            基于序列化注意力机制的3D点云Transformer骨干网络,采用U-Net式编码器-解码器结构。
            支持多种空间填充曲线(Z-order, Hilbert)对点云进行序列化, 而在序列化后的patch内执行高效的注意力计算。
            编码器通过SerializedPooling逐层下采样并提取多尺度特征, 解码器通过SerializedUnpooling逐层上采样并融合跳跃连接恢复分辨率。

        架构:
            输入 -> Embedding -> 编码器(多阶段: Block + SerializedPooling) -> 解码器(多阶段: SerializedUnpooling + Block) -> 输出

        默认配置:
            - 5个编码器阶段, 通道数: 32 -> 64 -> 128 -> 256 -> 512
            - 4个解码器阶段, 通道数: 256 -> 128 -> 64 -> 64
            - 4种序列化顺序: z, z-trans, hilbert, hilbert-trans
    """

    def __init__(
        self,
        in_channels=49,
        order=("z", "z-trans", "hilbert", "hilbert-trans"),
        stride=(4, 2, 2, 2),                      
        embedding_kernel_size=7,
        embedding_impl="sparseconv",
        cpe_impl="sparseconv",

        embedding_receptive_field=5.0,
        pointconv_embed_max_neighbors=64,
        pointconv_block_max_neighbors=32,
        enc_cpe_kernel_size=(5, 5, 5, 5, 5),
        dec_cpe_kernel_size=(5, 5, 5, 5, 5),
        enc_cpe_receptive_field=(2.0, 4.0, 8.0, 12.0, 16.0),
        dec_cpe_receptive_field=(2.0, 4.0, 8.0, 12.0, 16.0),  # 注意先池化/反池化, 之后才CPE、算attn

        enc_depths=(2, 2, 2, 6, 2),
        enc_channels=(64, 64, 128, 256, 512),
        enc_num_head=(2, 4, 8, 16, 32),
        enc_patch_size=(256, 128, 96, 72, 64),
        dec_depths=(2, 2, 2, 2, 2),         # see me: s最后一项为 dec4(最低分辨率、不做 unpooling); 若设为 0 则回退到旧版无 dec4 行为
        dec_channels=(64, 64, 128, 256, 512),
        dec_num_head=(4, 4, 8, 16, 32),
        dec_patch_size=(128, 96, 72, 64, 64),

        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.3,
        pre_norm=True,
        shuffle_orders=True,
        enable_rpe=False,
        enable_flash=True,                         # NOTE: 任务侧可在 Stage1PointBackbone 里切换 flash/rpe；当前骨架仅保留可配置开关，不直接内嵌特定任务的消融结论。  建议: 如果path_size=128时显存速度差别不大, 需要尝试禁止 flash, 打开相对位置编码。若有时间, 把注意力按照AF2的风格优化一下(降低中间特征到32; 后面加transition)
        upcast_attention=False,
        upcast_softmax=False,
        cls_mode=False,
        pdnorm_bn=False,
        pdnorm_ln=False,
        pdnorm_decouple=True,
        pdnorm_adaptive=False,
        pdnorm_affine=True,
        pdnorm_conditions=("ScanNet", "S3DIS", "Structured3D"),
        act_layer_name="gelu",
        ffn_type="mlp",
    ):
        """
            初始化 PointTransformerV3。

            输入参数:
                - 基础参数:
                    - in_channels: int, 输入点云特征通道数,默认6
                    - order: tuple[str] 或 str, 序列化顺序,可选"z"(Z-order)、"z-trans"(转置Z-order)、"hilbert"(Hilbert曲线)、"hilbert-trans"(转置Hilbert曲线)
                    - stride: tuple[int], 每个编码器阶段(除第一阶段外)的下采样步长,长度=num_stages-1,默认(2,2,2,2)
                    - enable_flash: bool, 是否启用Flash Attention加速,默认True
                    - shuffle_orders: bool, 训练时是否随机打乱多种序列化顺序的使用,默认True

                - embedding 参数:
                    - embedding_impl: str, embedding 实现方式, "sparseconv" 或 "pointconv"
                    - embedding_kernel_size: int, embedding 稀疏卷积核大小(仅 sparseconv 模式)
                    - embedding_receptive_field: float, embedding 世界坐标感受野(Å)(仅 pointconv 模式)
                    - pointconv_embed_max_neighbors: int, embedding 点云卷积最大邻居数(仅 pointconv 模式)
                    
                - cpe 参数:
                    - cpe_impl: str, Block CPE 实现方式, "sparseconv" 或 "pointconv"
                    - enc_cpe_kernel_size: tuple[int], 编码器每层 CPE 离散 kernel size(仅 sparseconv 模式), len=num_stages
                    - dec_cpe_kernel_size: tuple[int], 解码器每层 CPE 离散 kernel size(仅 sparseconv 模式), len=num_stages-1
                    - enc_cpe_receptive_field: tuple[float], 编码器每层 CPE 世界坐标感受野(Å)(仅 pointconv 模式), len=num_stages
                    - dec_cpe_receptive_field: tuple[float], 解码器每层 CPE 世界坐标感受野(Å)(仅 pointconv 模式), len=num_stages-1
                    - pointconv_block_max_neighbors: int, Block CPE 点云卷积最大邻居数(仅 pointconv 模式)

                - enc/dec 参数：
                    - enc_depths: tuple[int], 每个编码器阶段的Transformer Block数量,长度=num_stages,默认(2,2,2,6,2)
                    - enc_channels: tuple[int], 每个编码器的输出特征通道数,长度=num_stages,默认(32,64,128,256,512)
                    - enc_num_head: tuple[int], 每个编码器阶段的注意力头数,长度=num_stages,默认(2,4,8,16,32)
                    - enc_patch_size: tuple[int], 每个编码器阶段的patch大小(注意力窗口),长度=num_stages,默认(1024,1024,1024,1024,1024)

                    - dec_depths: tuple[int], 每个解码器阶段的Transformer Block数量,长度=num_stages-1,默认(2,2,2,2)
                    - dec_channels: tuple[int], 每个解码器的输出特征通道数,长度=num_stages-1,默认(64,64,128,256)
                    - dec_num_head: tuple[int], 每个解码器阶段的注意力头数,长度=num_stages-1,默认(4,4,8,16)
                    - dec_patch_size: tuple[int], 每个解码器阶段的patch大小,长度=num_stages-1,默认(1024,1024,1024,1024)

                - 其余参数：
                    - mlp_ratio: int, MLP隐藏层通道数相对于输入通道数的倍率,默认4
                    - qkv_bias: bool, QKV线性变换是否使用偏置项,默认True
                    - qk_scale: float 或 None, QK点积的缩放因子,若None则自动为1/sqrt(head_dim),默认None
                    - attn_drop: float, 注意力权重的dropout率,默认0.0
                    - proj_drop: float, 输出投影的dropout率,默认0.0
                    - drop_path: float, 随机深度(Stochastic Depth)的最大丢弃概率,线性递增,默认0.3
                    - pre_norm: bool, 是否使用Pre-Norm(归一化在注意力/MLP之前),默认True

                    - enable_rpe: bool, 是否启用相对位置编码(RPE),默认False, 只有在关闭 flash attention 时才有效
                    - upcast_attention: bool, 注意力计算时是否上转为float32,默认False
                    - upcast_softmax: bool, softmax计算时是否上转为float32,默认False
                    - cls_mode: bool, 是否为分类模式(仅编码器,无解码器),默认False

                    - pdnorm_bn: bool, BatchNorm是否使用PDNorm(Prompt-Driven Normalization),默认False
                    - pdnorm_ln: bool, LayerNorm是否使用PDNorm,默认False
                    - pdnorm_decouple: bool, PDNorm是否使用解耦模式,默认True
                    - pdnorm_adaptive: bool, PDNorm是否使用自适应模式,默认False
                    - pdnorm_affine: bool, PDNorm是否使用仿射变换参数,默认True
                    - pdnorm_conditions: tuple[str], PDNorm的数据集条件列表,默认("ScanNet","S3DIS","Structured3D")
        """
        super().__init__()
        # int, 编码器阶段总数(等于enc_depths的长度)
        self.num_stages = len(enc_depths)
        # list[str], 序列化顺序列表(如果输入是单个字符串则包装为列表)
        self.order = [order] if isinstance(order, str) else order
        # bool, 是否为分类模式(仅编码器,无解码器)
        self.cls_mode = cls_mode
        # bool, 是否在训练时随机打乱序列化顺序
        self.shuffle_orders = shuffle_orders
        # str, embedding / CPE 实现方式
        self.embedding_impl = str(embedding_impl)
        self.cpe_impl = str(cpe_impl)

        # 断言: 确保各个超参数的长度与阶段数匹配
        assert self.num_stages == len(stride) + 1
        assert self.num_stages == len(enc_depths)
        assert self.num_stages == len(enc_channels)
        assert self.num_stages == len(enc_num_head)
        assert self.num_stages == len(enc_patch_size)
        assert self.num_stages == len(enc_cpe_kernel_size), (
            f"enc_cpe_kernel_size 长度({len(enc_cpe_kernel_size)})必须等于 num_stages({self.num_stages})"
        )
        assert self.num_stages == len(enc_cpe_receptive_field), (
            f"enc_cpe_receptive_field 长度({len(enc_cpe_receptive_field)})必须等于 num_stages({self.num_stages})"
        )
        assert self.cls_mode or self.num_stages == len(dec_depths)
        assert self.cls_mode or self.num_stages == len(dec_channels)
        assert self.cls_mode or self.num_stages == len(dec_num_head)
        assert self.cls_mode or self.num_stages == len(dec_patch_size)
        assert self.cls_mode or self.num_stages == len(dec_cpe_kernel_size), (
            f"dec_cpe_kernel_size 长度({len(dec_cpe_kernel_size)})必须等于 num_stages({self.num_stages})"
        )
        assert self.cls_mode or self.num_stages == len(dec_cpe_receptive_field), (
            f"dec_cpe_receptive_field 长度({len(dec_cpe_receptive_field)})必须等于 num_stages({self.num_stages})"
        )

        # 参数校验
        if self.cpe_impl == "sparseconv":
            for s, ks in enumerate(enc_cpe_kernel_size):
                ks_int = int(ks)
                assert ks_int > 0 and ks_int % 2 == 1, (
                    f"sparseconv 模式下 enc_cpe_kernel_size[{s}]={ks} 必须为正奇数"
                )
            for s, ks in enumerate(dec_cpe_kernel_size):
                ks_int = int(ks)
                assert ks_int > 0 and ks_int % 2 == 1, (
                    f"sparseconv 模式下 dec_cpe_kernel_size[{s}]={ks} 必须为正奇数"
                )
        elif self.cpe_impl == "pointconv":
            for s, rf in enumerate(enc_cpe_receptive_field):
                assert float(rf) > 0, (
                    f"pointconv 模式下 enc_cpe_receptive_field[{s}]={rf} 必须为正数"
                )
            for s, rf in enumerate(dec_cpe_receptive_field):
                assert float(rf) > 0, (
                    f"pointconv 模式下 dec_cpe_receptive_field[{s}]={rf} 必须为正数"
                )

        # 归一化层
        if pdnorm_bn:
            # 使用PDNorm的BatchNorm
            bn_layer = partial(
                PDNorm,
                norm_layer=partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01, affine=pdnorm_affine),
                conditions=pdnorm_conditions,
                decouple=pdnorm_decouple,
                adaptive=pdnorm_adaptive,
            )
        else:
            # 使用标准BatchNorm
            bn_layer = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
        if pdnorm_ln:
            # 使用PDNorm的LayerNorm
            ln_layer = partial(
                PDNorm,
                norm_layer=partial(nn.LayerNorm, elementwise_affine=pdnorm_affine),
                conditions=pdnorm_conditions,
                decouple=pdnorm_decouple,
                adaptive=pdnorm_adaptive,
            )
        else:
            # 使用标准LayerNorm
            ln_layer = nn.LayerNorm
        # 激活函数层: 通过 act_layer_name 统一配置
        act_layer = resolve_act_layer(act_layer_name)

        # Embedding, 嵌入层
        self.embedding = Embedding(
            in_channels=in_channels,
            embed_channels=enc_channels[0],
            embedding_kernel_size=int(embedding_kernel_size),
            embedding_impl=self.embedding_impl,
            embedding_receptive_field=float(embedding_receptive_field),
            pointconv_embed_max_neighbors=int(pointconv_embed_max_neighbors),
            norm_layer=bn_layer,
            act_layer=act_layer,
        )


        # -------------------------------------------------- 编码器 -------------------------------------------------
        # list[float], 编码器随机深度drop路径
        enc_drop_path = [x.item() for x in torch.linspace(start=0, end=drop_path, steps=sum(enc_depths))]
        # PointSequential, 编码器容器
        self.enc = PointSequential()
        # 遍历每个阶段
        for s in range(self.num_stages):
            # list[float], 当前阶段的drop路径
            enc_drop_path_ = enc_drop_path[sum(enc_depths[:s]) : sum(enc_depths[: s + 1])]
            # PointSequential, 当前阶段容器
            enc = PointSequential()
            # 如果不是第一个阶段,则添加下采样层
            if s > 0:
                enc.add(
                    SerializedPooling(
                        in_channels=enc_channels[s - 1],
                        out_channels=enc_channels[s],
                        stride=stride[s - 1],
                        norm_layer=bn_layer,
                        act_layer=act_layer,
                    ),
                    name="down",
                )
            # 添加当前阶段的Transformer块
            for i in range(enc_depths[s]):
                enc.add(
                    Block(
                        channels=enc_channels[s],
                        num_heads=enc_num_head[s],
                        patch_size=enc_patch_size[s],
                        cpe_kernel_size=int(enc_cpe_kernel_size[s]),
                        cpe_impl=self.cpe_impl,
                        cpe_receptive_field=float(enc_cpe_receptive_field[s]),
                        pointconv_block_max_neighbors=int(pointconv_block_max_neighbors),
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        qk_scale=qk_scale,
                        attn_drop=attn_drop,
                        proj_drop=proj_drop,
                        drop_path=enc_drop_path_[i],
                        norm_layer=ln_layer,
                        act_layer=act_layer,
                        pre_norm=pre_norm,
                        order_index=i % len(self.order),
                        cpe_indice_key=f"stage{s}",
                        enable_rpe=enable_rpe,
                        enable_flash=enable_flash,
                        upcast_attention=upcast_attention,
                        upcast_softmax=upcast_softmax,
                        ffn_type=ffn_type,
                    ),
                    name=f"block{i}",
                )
            # 如果当前阶段不为空,则添加到编码器
            if len(enc) != 0:
                self.enc.add(module=enc, name=f"enc{s}")



        # -------------------------------------------------- 解码器 -------------------------------------------------
        if not self.cls_mode:
            # dec_depths 长度 = num_stages; 最后一项 dec_depths[-1] 对应 dec4(最低分辨率层, 不做 unpooling)。
            # 当 dec_depths[-1] == 0 时, 该阶段初始化为恒等映射(nn.Identity), 等价于无 dec4 行为。
            # list[float], 解码器随机深度drop路径
            dec_drop_path = [
                x.item() for x in torch.linspace(0, drop_path, sum(dec_depths))
            ]
            # PointSequential, 解码器容器
            self.dec = PointSequential()
            # list[int], 解码器输出通道数
            dec_channels = list(dec_channels)
            # 反向遍历解码器阶段(从低分辨率到高分辨率)
            for s in reversed(range(self.num_stages)):
                if int(dec_depths[s]) == 0:
                    # depth=0: 该阶段为恒等映射, 不执行任何计算
                    self.dec.add(module=nn.Identity(), name=f"dec{s}")
                    continue
                # list[float], 当前阶段的drop路径
                dec_drop_path_ = dec_drop_path[
                    sum(dec_depths[:s]) : sum(dec_depths[: s + 1])
                ]
                # 反转drop路径(因为是从低到高分辨率)
                dec_drop_path_.reverse()
                # PointSequential, 当前阶段容器
                dec = PointSequential()

                if s == self.num_stages - 1:
                    # dec4: 最低分辨率阶段, 不做 unpooling, 只做通道投影 + Block
                    if enc_channels[-1] != dec_channels[s]:
                        dec.add(nn.Linear(enc_channels[-1], dec_channels[s]), name="proj")
                        dec.add(ln_layer(dec_channels[s]), name="proj_norm")
                else:
                    # dec3–dec0: 正常 unpooling。in_channels 取上一层(更低分辨率)的输出通道。
                    up_in_channels = dec_channels[s + 1]
                    dec.add(
                        SerializedUnpooling(
                            in_channels=up_in_channels,
                            skip_channels=enc_channels[s],
                            out_channels=dec_channels[s],
                            norm_layer=bn_layer,
                            act_layer=act_layer,
                        ),
                        name="up",
                    )
                # 添加当前阶段的Transformer块
                for i in range(dec_depths[s]):
                    dec.add(
                        Block(
                            channels=dec_channels[s],
                            num_heads=dec_num_head[s],
                            patch_size=dec_patch_size[s],
                            cpe_kernel_size=int(dec_cpe_kernel_size[s]),
                            cpe_impl=self.cpe_impl,
                            cpe_receptive_field=float(dec_cpe_receptive_field[s]),
                            pointconv_block_max_neighbors=int(pointconv_block_max_neighbors),
                            mlp_ratio=mlp_ratio,
                            qkv_bias=qkv_bias,
                            qk_scale=qk_scale,
                            attn_drop=attn_drop,
                            proj_drop=proj_drop,
                            drop_path=dec_drop_path_[i],
                            norm_layer=ln_layer,
                            act_layer=act_layer,
                            pre_norm=pre_norm,
                            order_index=i % len(self.order),
                            cpe_indice_key=f"stage{s}",
                            enable_rpe=enable_rpe,
                            enable_flash=enable_flash,
                            upcast_attention=upcast_attention,
                            upcast_softmax=upcast_softmax,
                            ffn_type=ffn_type,
                        ),
                        name=f"block{i}",
                    )
                # 将当前阶段添加到解码器
                self.dec.add(module=dec, name=f"dec{s}")


    def forward(self, data_dict):
        """
        前向传播,执行完整的PTv3编码-解码流程。

        输入参数:
            - data_dict: dict, 批量点云的属性字典,必须包含以下字段:
                1. "feat": torch.Tensor, (N, in_channels), 点云特征
                2. "grid_coord": torch.Tensor, (N, 3), 网格采样(体素化)后的离散坐标; 或者提供 "coord" (连续坐标) + "grid_size" (网格大小) 由内部自动计算
                3. "offset": torch.Tensor, (B,) 或 "batch": torch.Tensor, (N,), 表示批量中各点云的分界偏移量或每个点所属的batch索引

        输出:
            - point: Point, 处理后的点云数据对象,包含:
                - point.feat: torch.Tensor, (N, C), 最终特征(C取决于cls_mode: 分类模式下C=enc_channels[-1]; 分割模式下C=dec_channels[0])
                - 以及其他点云属性(coord, grid_coord, batch, serialized_*等)
        """
        # Point, 将输入字典包装为Point对象(包含feat, coord, grid_coord, batch/offset等)
        point = Point(data_dict)
        # 执行序列化: 根据空间填充曲线对点云排序,生成serialized_code/order/inverse/depth
        point.serialization(order=self.order, shuffle_orders=self.shuffle_orders)
        # 稀疏化: 构建SparseTensor(稀疏卷积所需的数据结构)
        point.sparsify()

        # Point, 通过嵌入层将原始特征(in_channels维)映射到enc_channels[0]维
        point = self.embedding(point)
        # Point, 通过编码器逐层提取特征(逐阶段下采样: N -> N/s1 -> N/s1/s2 -> ...)
        point = self.enc(point)
        # 如果不是分类模式,则通过解码器逐层上采样恢复分辨率并融合跳跃连接
        if not self.cls_mode:
            # Point, 通过解码器恢复到原始分辨率,特征维度为dec_channels[0]
            point = self.dec(point)
        return point






class PTV3BackboneAdapter(nn.Module):
    """
    PTV3 Backbone 适配器：将 PointTransformerV3 的编码器输出整理为 Mask3D 风格的多尺度特征。

    # 输入参数:
        - ptv3: PointTransformerV3, 已初始化好的 PTV3 主干网络
        - level_channels_low_to_high: list[int], 多尺度通道数(低分辨率 -> 高分辨率)
        - return_dec: bool, 是否在编码后继续执行解码(默认 False)

    # 输出:
        - pcd_features: Point, 最高分辨率点云特征(与输入点云一一对应)
        - aux: list[Point], 多尺度点云特征列表(低分辨率 -> 高分辨率)
    """

    def __init__(
        self,
        ptv3: PointTransformerV3,
        level_channels_low_to_high: List[int],
        return_dec: bool = False,
    ):
        super().__init__()
        self.ptv3 = ptv3
        # 低->高尺度通道数，用于 Mask3D 侧对齐
        self.level_channels = list(level_channels_low_to_high)
        # 高分辨率输出通道数(用于 Mask3D 的 mask_features_head)
        self.out_channels = self.level_channels[-1]
        self.return_dec = bool(return_dec)

        # 复用 PTV3 的序列化配置
        self._order = getattr(ptv3, "order", ["z"])
        self._shuffle_orders = getattr(ptv3, "shuffle_orders", True)

    def forward(self, data_dict: dict):
        """
        执行 PTV3 编码并收集多尺度特征。

        # 输入参数:
            - data_dict: dict, PTV3 标准输入字典(需包含 feat/coord 或 grid_coord/offset 或 batch)

        # 输出:
            - pcd_features: Point, 最高分辨率点云特征
            - aux: list[Point], 多尺度点云特征(低 -> 高)
        """
        # 1) 构建 Point 并完成序列化与稀疏化
        point = Point(data_dict)
        point.serialization(order=self._order, shuffle_orders=self._shuffle_orders)
        point.sparsify()

        # 2) Embedding
        point = self.ptv3.embedding(point)

        # 3) 编码器逐层输出(高 -> 低)
        feats_high_to_low: List[Point] = []
        for stage in self.ptv3.enc:
            point = stage(point)
            feats_high_to_low.append(point)

        # 4) 可选：走解码器(通常不需要)
        if self.return_dec and hasattr(self.ptv3, "dec"):
            point = self.ptv3.dec(point)
            # 这里不覆盖多尺度，仅在需要时返回 dec 输出(可按需扩展)

        # 5) 组织输出：pcd_features 取最高分辨率
        pcd_features = feats_high_to_low[0]
        aux_low_to_high = list(reversed(feats_high_to_low))

        return pcd_features, aux_low_to_high

