"""
=============================================================================
冻结子树 eval 工具 (Frozen Submodule Eval)
=============================================================================
仅设 requires_grad=False 只能停掉梯度更新; BatchNorm 的 running_mean/var 等 buffer 仍会在
train 模式前向里按 momentum 漂移, Dropout 也仍随机. 本模块提供把「完全冻结子树」切到 eval 的
通用工具, 使被冻结部分的前向固定. 

适用范围:
  - 任何 nn.Module(LightningModule 自身、普通 backbone 等). 
  - 典型用法: 在每次 model.train() 之后调用一次(Lightning 每个 train epoch 会把全部子模块
    设回 training=True, 需重新维持冻结子树的 eval). 
"""

from __future__ import annotations

from torch import nn


def set_fully_frozen_submodules_eval(root: nn.Module) -> tuple[int, int]:
    """
    把「自身递归参数全部 requires_grad=False」的子模块切到 eval. 

    输入参数:
        - root: nn.Module, 待扫描的根模块(通常是 LightningModule 自身)

    输出:
        - num_frozen_subtrees: int, 被切到 eval 的「极大」冻结子树数(直接父模块非完全冻结的那些根), 仅用于日志/自检
        - num_frozen_bn: int, 其中带 running 统计 buffer 的归一化层数(本会漂移、现已被固定的 BN 数)

    说明:
        - 半冻结(部分参数仍可训练)的模块保持 train; 仅在「极大」冻结子树根上调用 .eval(). 
        - 对子树根 .eval() 会递归覆盖整棵, 因此子树内 affine=False(无 param 有 buffer)的 BN 也被固定. 
        - 无完全冻结模块时为 no-op. 
    """
    # list[tuple[str, nn.Module]], (限定名, 模块) 全表; 复用以避免重复遍历
    named_modules = list(root.named_modules())
    # dict[str, bool], 模块限定名 -> 是否「确有参数且参数全部冻结」
    fully_frozen: dict[str, bool] = {}
    for name, module in named_modules:
        # list[nn.Parameter], 该模块递归持有的全部参数; 空列表表示纯无参模块(激活 / 部分 Dropout 等)
        params = list(module.parameters(recurse=True))
        fully_frozen[name] = bool(params) and all(not p.requires_grad for p in params)

    num_frozen_subtrees = 0
    num_frozen_bn = 0
    for name, module in named_modules:
        if not fully_frozen[name]:
            continue
        # str, 直接父模块限定名; 父也完全冻结 => 当前不是「极大」根, 留给父统一处理, 避免重复计数
        parent_name = name.rsplit(".", 1)[0] if "." in name else ""
        if fully_frozen.get(parent_name, False):
            continue
        module.eval()  # 递归把整棵冻结子树切到 eval
        num_frozen_subtrees += 1
        # 统计该子树内所有带 running 统计的归一化层(BatchNorm 系; 含 affine=False), 即本会漂移、现已固定的 BN
        for submodule in module.modules():
            if getattr(submodule, "track_running_stats", False) and getattr(submodule, "running_mean", None) is not None:
                num_frozen_bn += 1
    return num_frozen_subtrees, num_frozen_bn
