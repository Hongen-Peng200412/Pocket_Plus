# src/model/utils.py
# 通用、与具体 head 无关的小构建器。
# 边界: 这里只放可被多处复用的"零初始化恒等融合 / 调制 / 采样权重"构建, 不放业务逻辑。
from __future__ import annotations

import math

import torch
from torch import nn

# ------------------------------------------------ 用于: 调制模块(main,cond) ------------------------------------------------
# 用于: 
# - real atom 的 (embed head 特征, density cube 特征)
# - 融合hook: (point backbone, voxel backbone) 
def build_zero_init_residual_mlp(
    in_dim: int,
    out_dim: int,
    hidden_dim: int,
    act_layer: type[nn.Module],
    proj_drop: float = 0.0,
) -> nn.Sequential:
    """
    构造末层零初始化的两层 MLP, 用于"开局恒等"的残差 / 调制融合。

    输入参数:
        - in_dim: int, 输入通道数
        - out_dim: int, 输出通道数(残差用法下通常等于被残差对象的通道数; film_plus 用法下为 2*主特征通道)
        - hidden_dim: int, 隐藏层通道数
        - act_layer: type[nn.Module], 激活函数类(如 nn.SiLU)
        - proj_drop: float, 隐藏激活后 dropout 概率; 0.0 时为恒等

    输出:
        - mlp: nn.Sequential, Linear(in,hidden)->act->Dropout(proj_drop)->Linear(hidden,out); 末层 weight/bias 置零
    """
    # nn.Sequential, 两层 MLP(act 后接 dropout); 末层零初始化 => 开局输出全 0 => 残差/调制恒等
    mlp = nn.Sequential(
        nn.Linear(int(in_dim), int(hidden_dim)),
        act_layer(),
        nn.Dropout(float(proj_drop)),
        nn.Linear(int(hidden_dim), int(out_dim)),
    )
    nn.init.zeros_(mlp[-1].weight)
    nn.init.zeros_(mlp[-1].bias)
    return mlp


class MiniResidueCombine(nn.Module):
    """
    mini_residue 融合: 主特征加 cond 经零初始化 MLP 的残差(开局恒等返回主特征)。

    前向: main + mlp(cond), mlp 末层零初始化 => 开局残差为 0。

    输入参数:
        - main_dim: int, 主特征通道数(也是输出通道数)
        - cond_dim: int, 条件特征通道数
        - hidden_dim: int, 残差 MLP 隐藏层通道数
        - act_layer: type[nn.Module], 隐藏层激活函数类
        - proj_drop: float, 残差 MLP 隐藏激活后 dropout 概率
    """

    def __init__(
        self,
        main_dim: int,
        cond_dim: int,
        hidden_dim: int,
        act_layer: type[nn.Module],
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        # nn.Sequential, cond -> 主特征残差; 末层零初始化
        self.mlp = build_zero_init_residual_mlp(int(cond_dim), int(main_dim), int(hidden_dim), act_layer, proj_drop)

    def forward(self, main: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        输入参数:
            - main: torch.Tensor, (N, main_dim), 主特征
            - cond: torch.Tensor, (N, cond_dim), 条件特征

        输出:
            - out: torch.Tensor, (N, main_dim), 融合后特征(开局恒等于 main)
        """
        return main + self.mlp(cond)


class ConcatMLPCombine(nn.Module):
    """
    concat_mlp 融合: 主特征加 [main, cond] 拼接经零初始化 MLP 的残差(开局恒等返回主特征)。

    前向: main + mlp(cat([main, cond])), mlp 末层零初始化 => 开局残差为 0。

    输入参数:
        - main_dim: int, 主特征通道数(也是输出通道数)
        - cond_dim: int, 条件特征通道数
        - hidden_dim: int, 残差 MLP 隐藏层通道数
        - act_layer: type[nn.Module], 隐藏层激活函数类
        - proj_drop: float, 残差 MLP 隐藏激活后 dropout 概率
    """

    def __init__(
        self,
        main_dim: int,
        cond_dim: int,
        hidden_dim: int,
        act_layer: type[nn.Module],
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        # nn.Sequential, [main, cond] -> 主特征残差; 末层零初始化
        self.mlp = build_zero_init_residual_mlp(
            int(main_dim) + int(cond_dim), int(main_dim), int(hidden_dim), act_layer, proj_drop
        )

    def forward(self, main: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        输入参数:
            - main: torch.Tensor, (N, main_dim), 主特征
            - cond: torch.Tensor, (N, cond_dim), 条件特征

        输出:
            - out: torch.Tensor, (N, main_dim), 融合后特征(开局恒等于 main)
        """
        return main + self.mlp(torch.cat([main, cond], dim=-1))


class FiLMCombine(nn.Module):
    """
    film 融合: 用 cond 经单层零初始化 Linear 生成 gamma/beta 调制主特征。

    前向: main * (1 + gamma) + beta, 其中 [gamma, beta] = Linear(cond)。
    Linear 零初始化 => 开局 gamma=beta=0 => 输出恒等于 main。

    输入参数:
        - main_dim: int, 主特征通道数(gamma/beta 各 main_dim)
        - cond_dim: int, 条件特征通道数
    """

    def __init__(self, main_dim: int, cond_dim: int) -> None:
        super().__init__()
        self.main_dim = int(main_dim)
        # nn.Linear, cond -> [gamma, beta]; 零初始化保证开局恒等
        self.to_gamma_beta = nn.Linear(int(cond_dim), 2 * int(main_dim))
        nn.init.zeros_(self.to_gamma_beta.weight)
        nn.init.zeros_(self.to_gamma_beta.bias)

    def forward(self, main: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        输入参数:
            - main: torch.Tensor, (N, main_dim), 主特征
            - cond: torch.Tensor, (N, cond_dim), 调制条件特征

        输出:
            - out: torch.Tensor, (N, main_dim), 调制后特征(开局恒等于 main)
        """
        # torch.Tensor, (N, 2*main_dim), 调制参数
        gamma_beta = self.to_gamma_beta(cond)
        # torch.Tensor, (N, main_dim), 缩放项 gamma 与 偏移项 beta
        gamma = gamma_beta[:, : self.main_dim]
        beta = gamma_beta[:, self.main_dim :]
        return main * (1.0 + gamma) + beta


class FiLMPlusCombine(nn.Module):
    """
    film_plus 融合: 用 cond 经零初始化两层 MLP 生成 gamma/beta 调制主特征。

    前向: main * (1 + gamma) + beta, 其中 [gamma, beta] = mlp(cond)。
    mlp 末层零初始化 => 开局 gamma=beta=0 => 输出恒等于 main。

    输入参数:
        - main_dim: int, 主特征通道数(gamma/beta 各 main_dim)
        - cond_dim: int, 条件特征通道数
        - hidden_dim: int, 生成器 MLP 隐藏层通道数
        - act_layer: type[nn.Module], 隐藏层激活函数类
        - proj_drop: float, 生成器 MLP 隐藏激活后 dropout 概率
    """

    def __init__(
        self,
        main_dim: int,
        cond_dim: int,
        hidden_dim: int,
        act_layer: type[nn.Module],
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.main_dim = int(main_dim)
        # nn.Sequential, cond -> [gamma, beta]; 末层零初始化
        self.generator = build_zero_init_residual_mlp(
            int(cond_dim), 2 * int(main_dim), int(hidden_dim), act_layer, proj_drop
        )

    def forward(self, main: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        输入参数:
            - main: torch.Tensor, (N, main_dim), 主特征
            - cond: torch.Tensor, (N, cond_dim), 调制条件特征

        输出:
            - out: torch.Tensor, (N, main_dim), 调制后特征(开局恒等于 main)
        """
        # torch.Tensor, (N, 2*main_dim), 调制参数
        gamma_beta = self.generator(cond)
        # torch.Tensor, (N, main_dim), 缩放项 gamma 与 偏移项 beta
        gamma = gamma_beta[:, : self.main_dim]
        beta = gamma_beta[:, self.main_dim :]
        return main * (1.0 + gamma) + beta


class FeatureCombine(nn.Module):
    """
    主特征 <- 条件特征 的统一融合派发器(四种 mode 共享同一组小类, 均开局恒等返回主特征)。

    被 voxel->point 融合与 real 原子 embed<->density 融合复用:
        - voxel->point: main=点特征, cond=采样后体素特征
        - real density: main=embed 特征, cond=density cube 特征

    输入参数:
        - mode: str, 融合方式 concat_mlp / film / film_plus / mini_residue
        - main_dim: int, 主特征通道数(也是输出通道数)
        - cond_dim: int, 条件特征通道数
        - hidden_dim: int, concat_mlp/mini_residue/film_plus 隐藏层通道数; film 不使用
        - act_layer: type[nn.Module], 隐藏层激活函数类; film 不使用
        - proj_drop: float, concat_mlp/mini_residue/film_plus 隐藏激活后 dropout; film 不使用
    """

    def __init__(
        self,
        mode: str,
        main_dim: int,
        cond_dim: int,
        hidden_dim: int,
        act_layer: type[nn.Module],
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.mode = str(mode).lower()
        main_dim = int(main_dim)
        cond_dim = int(cond_dim)
        hidden_dim = int(hidden_dim)
        # nn.Module, 按 mode 选定的具体融合小类; 统一 forward(main, cond) 接口
        if self.mode == "concat_mlp":
            self.combine: nn.Module = ConcatMLPCombine(main_dim, cond_dim, hidden_dim, act_layer, proj_drop)
        elif self.mode == "mini_residue":
            self.combine = MiniResidueCombine(main_dim, cond_dim, hidden_dim, act_layer, proj_drop)
        elif self.mode == "film":
            self.combine = FiLMCombine(main_dim, cond_dim)
        elif self.mode == "film_plus":
            self.combine = FiLMPlusCombine(main_dim, cond_dim, hidden_dim, act_layer, proj_drop)
        else:
            raise ValueError(f"未知 combine mode={mode}, 取值 concat_mlp/film/film_plus/mini_residue。")

    def forward(self, main: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        输入参数:
            - main: torch.Tensor, (N, main_dim), 主特征
            - cond: torch.Tensor, (N, cond_dim), 条件特征

        输出:
            - out: torch.Tensor, (N, main_dim), 融合后主特征(开局恒等于 main)
        """
        return self.combine(main, cond)



# ------------------------------------------------ 用于: point backbone 中 fusion hook 的(3^3 BOX)特征抽取 ------------------------------------------------
class CubeWeightingParams(nn.Module):
    """
    weighted_cube 采样器的 per-hook 可学习权重参数容器。

    把 3^3 邻域每个体素的"类别正偏置 + 距离温度项"组成 softmax 前的 logit:
        logit_i = exp(log_cat_bias(i)) - dist_i^2 / exp(log_d)
    其中 a/b/c 是 home/含原子/其他 三类正偏置, 内部以 log 参数保存。

    输入参数:
        - a_init: float, home 体素类别正偏置初值, 必须 > 0
        - b_init: float, 含原子体素类别正偏置初值, 必须 > 0
        - c_init: float, 其他体素类别正偏置初值, 必须 > 0
        - d_init: float, 温度初值(>0), 内部存 log_d=log(d)
    """

    def __init__(self, a_init: float, b_init: float, c_init: float, d_init: float) -> None:
        super().__init__()
        if min(float(a_init), float(b_init), float(c_init), float(d_init)) <= 0.0:
            raise ValueError("a/b/c/d_init 必须 > 0。")
        # nn.Parameter, (), home/含原子/其他 三类正偏置的 log 参数。
        self.a_log = nn.Parameter(torch.tensor(math.log(float(a_init))))
        self.b_log = nn.Parameter(torch.tensor(math.log(float(b_init))))
        self.c_log = nn.Parameter(torch.tensor(math.log(float(c_init))))
        # nn.Parameter, (), 温度的 log; exp 保正
        self.log_d = nn.Parameter(torch.tensor(math.log(float(d_init))))

    def forward(self, category: torch.Tensor, dist_sq: torch.Tensor) -> torch.Tensor:
        """
        输入参数:
            - category: torch.Tensor, (..., K), long, 每个邻居类别 0=home / 1=含原子 / 2=其他
            - dist_sq: torch.Tensor, (..., K), 每个邻居到连续点位的距离平方(voxel 单位)

        输出:
            - logit: torch.Tensor, (..., K), softmax 前的 logit(未含越界 -inf 掩码)
        """
        # torch.Tensor, (3,), 三类正偏置的 log 参数, 顺序 home/含原子/其他
        cat_bias_log = torch.stack([self.a_log, self.b_log, self.c_log])
        # torch.Tensor, (..., K), 按类别 gather 得到的正偏置 log 参数
        bias_log = cat_bias_log[category]
        bias = torch.exp(bias_log)
        # torch.Tensor, (), 温度(>0)
        temperature = torch.exp(self.log_d)
        return bias - dist_sq / temperature


def gather_voxel_cube(
    grid: torch.Tensor,
    center_zyx: torch.Tensor,
    batch_index: torch.Tensor,
    cube_size: int,
    zero_fill: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    以 center_zyx 为中心抽取每个点的 cube_size^3 体素邻域(无 padding, 用 clamp + 越界掩码)。
    与 zero-pad 抽取数值等价: 越界邻居 zero_fill=True 时置 0; 越界与否由 valid_mask 标记。
    仅需中心单体素(cube_size=1)时用更轻的 gather_voxel_feature_at_zyx, 避免 offsets/clamp/permute 开销。

    输入参数:
        - grid: torch.Tensor, (B, C, D, H, W), 体素特征 / 密度图
        - center_zyx: torch.Tensor, (N, 3), 每个点 home 体素坐标, 轴顺序 z/y/x
        - batch_index: torch.Tensor, (N,), 每个点所属 BOX 索引
        - cube_size: int, cube 边长(>=1 奇数)
        - zero_fill: bool, True 把越界邻居特征置 0(等价 zero-pad); False 保留 clamp 取到的值(由调用方用 mask 处理)

    输出:
        - cube: torch.Tensor, (N, C, k, k, k), 邻域特征, k=cube_size, 轴顺序 z/y/x
        - valid_mask: torch.Tensor, (N, k, k, k), bool, True 表示该邻居在 grid 边界内
    """
    if int(cube_size) < 1 or int(cube_size) % 2 == 0:
        raise ValueError("cube_size 必须为 >=1 的奇数。")
    if grid.ndim != 5:
        raise ValueError(f"grid 期望为 (B,C,D,H,W), 实际 {tuple(grid.shape)}。")
    # int, cube 边长与半径
    k = int(cube_size)
    radius = k // 2
    # int, grid 三个空间维尺寸
    dim_z, dim_y, dim_x = int(grid.shape[2]), int(grid.shape[3]), int(grid.shape[4])
    center_zyx = center_zyx.to(device=grid.device, dtype=torch.long)
    batch_index = batch_index.to(device=grid.device, dtype=torch.long)
    # torch.Tensor, (k,), 单轴邻居相对中心的偏移 -radius..radius
    offsets = torch.arange(k, device=grid.device, dtype=torch.long) - radius
    # torch.Tensor, (N, k), 三轴未裁剪的绝对邻居索引
    z_raw = center_zyx[:, 0, None] + offsets[None, :]
    y_raw = center_zyx[:, 1, None] + offsets[None, :]
    x_raw = center_zyx[:, 2, None] + offsets[None, :]
    # torch.Tensor, (N, k), 各轴是否在边界内
    z_ok = (z_raw >= 0) & (z_raw < dim_z)
    y_ok = (y_raw >= 0) & (y_raw < dim_y)
    x_ok = (x_raw >= 0) & (x_raw < dim_x)
    # torch.Tensor, (N, k, k, k), 三轴广播得到的越界掩码
    valid_mask = z_ok[:, :, None, None] & y_ok[:, None, :, None] & x_ok[:, None, None, :]
    # torch.Tensor, (N, k, 1, 1) / (N, 1, k, 1) / (N, 1, 1, k), clamp 后用于安全 gather 的高级索引
    z_index = z_raw.clamp(0, dim_z - 1)[:, :, None, None]
    y_index = y_raw.clamp(0, dim_y - 1)[:, None, :, None]
    x_index = x_raw.clamp(0, dim_x - 1)[:, None, None, :]
    # torch.Tensor, (N, 1, 1, 1), 每个点所属 BOX
    b_index = batch_index[:, None, None, None]
    # torch.Tensor, (N, k, k, k, C), 混合高级索引(b,z,y,x 被切片 : 分隔 => 广播维在前)
    cube_channels_last = grid[b_index, :, z_index, y_index, x_index]
    # torch.Tensor, (N, C, k, k, k), 还原通道维顺序
    cube = cube_channels_last.permute(0, 4, 1, 2, 3).contiguous()
    if zero_fill:
        cube = cube * valid_mask[:, None, :, :, :].to(cube.dtype)
    return cube, valid_mask


def gather_voxel_feature_at_zyx(
    voxel_feat: torch.Tensor,
    voxel_zyx: torch.Tensor,
    point_batch_index: torch.Tensor,
) -> torch.Tensor:
    """
    按离散 voxel center 坐标直接读取体素特征(cube_size=1 的精简特例; 多体素邻域见 gather_voxel_cube)。

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
