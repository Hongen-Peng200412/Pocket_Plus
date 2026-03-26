import torch
from .z_order import xyz2key as z_order_encode_
from .z_order import key2xyz as z_order_decode_
from .hilbert import encode as hilbert_encode_
from .hilbert import decode as hilbert_decode_


@torch.inference_mode()
def encode(grid_coord, batch=None, depth=16, order="z"):
    """
    点云序列化编码主接口。根据指定的排序方式（Z-Order 或 Hilbert）将三维网格坐标编码为一维整数。

    输入参数:
        - grid_coord: torch.Tensor, (N, 3), 整数类型的点云网格坐标 (x, y, z)
        - batch: torch.Tensor | None, (N,), 每个点所属的 batch 索引。若提供, 则拼接在编码高位
        - depth: int, 编码深度, 决定了坐标的分辨率, 通常为 16 (支持 2^16 的范围)
        - order: str, 排序方式, 支持 "z", "z-trans", "hilbert", "hilbert-trans"

    输出:
        - code: torch.Tensor, (N,), 编码后的一维长整型张量 (int64)
    """
    # str, 验证排序方式是否在支持列表中
    assert order in {"z", "z-trans", "hilbert", "hilbert-trans"}
    if order == "z":
        # torch.Tensor, (N,), Z-Order 编码结果
        code = z_order_encode(grid_coord, depth=depth)
    elif order == "z-trans":
        # torch.Tensor, (N,), 交换 x,y 后的 Z-Order 编码结果
        code = z_order_encode(grid_coord[:, [1, 0, 2]], depth=depth)
    elif order == "hilbert":
        # torch.Tensor, (N,), Hilbert 编码结果
        code = hilbert_encode(grid_coord, depth=depth)
    elif order == "hilbert-trans":
        # torch.Tensor, (N,), 交换 x,y 后的 Hilbert 编码结果
        code = hilbert_encode(grid_coord[:, [1, 0, 2]], depth=depth)
    else:
        raise NotImplementedError
    if batch is not None:
        # torch.Tensor, (N,), 转化为长整型以进行位运算
        batch = batch.long()
        # torch.Tensor, (N,), 最终编码: [batch_index (高位) | serialized_code (低位)]
        code = batch << depth * 3 | code
    return code


@torch.inference_mode()
def decode(code, depth=16, order="z"):
    """
    点云序列化解码主接口。将一维编码还原为三维网格坐标及 batch 索引。

    输入参数:
        - code: torch.Tensor, (N,), 包含 batch 信息的一维编码张量
        - depth: int, 编码深度, 必须与 encode 时一致
        - order: str, 排序方式, 支持 "z" 或 "hilbert"

    输出:
        - grid_coord: torch.Tensor, (N, 3), 还原后的网格坐标 (x, y, z)
        - batch: torch.Tensor, (N,), 还原后的 batch 索引
    """
    # str, 验证解码排序方式
    assert order in {"z", "hilbert"}
    # torch.Tensor, (N,), 通过右移提取高位的 batch 信息
    batch = code >> depth * 3
    # torch.Tensor, (N,), 通过掩码提取低位的序列化编码
    code = code & ((1 << depth * 3) - 1)
    if order == "z":
        # torch.Tensor, (N, 3), Z-Order 解码后的网格坐标
        grid_coord = z_order_decode(code, depth=depth)
    elif order == "hilbert":
        # torch.Tensor, (N, 3), Hilbert 解码后的网格坐标
        grid_coord = hilbert_decode(code, depth=depth)
    else:
        raise NotImplementedError
    return grid_coord, batch


def z_order_encode(grid_coord: torch.Tensor, depth: int = 16):
    """
    Z-Order (莫顿编码) 封装函数。

    输入参数:
        - grid_coord: torch.Tensor, (N, 3), 网格坐标 (x, y, z)
        - depth: int, 编码深度 (默认为 16)

    输出:
        - code: torch.Tensor, (N,), Z-Order 编码值
    """
    x, y, z = grid_coord[:, 0].long(), grid_coord[:, 1].long(), grid_coord[:, 2].long()
    # we block the support to batch, maintain batched code in Point class
    code = z_order_encode_(x, y, z, b=None, depth=depth)
    return code


def z_order_decode(code: torch.Tensor, depth):
    """
    Z-Order (莫顿编码) 解码封装函数。

    输入参数:
        - code: torch.Tensor, (N,), Z-Order 编码值
        - depth: int, 编码深度

    输出:
        - grid_coord: torch.Tensor, (N, 3), 还原后的 (x, y, z) 坐标
    """
    x, y, z = z_order_decode_(code, depth=depth)
    grid_coord = torch.stack([x, y, z], dim=-1)  # (N,  3)
    return grid_coord


def hilbert_encode(grid_coord: torch.Tensor, depth: int = 16):
    """
    Hilbert 曲线编码封装函数。

    输入参数:
        - grid_coord: torch.Tensor, (N, 3), 网格坐标
        - depth: int, 编码深度 (决定位宽)

    输出:
        - code: torch.Tensor, (N,), Hilbert 序列化编码值
    """
    return hilbert_encode_(grid_coord, num_dims=3, num_bits=depth)


def hilbert_decode(code: torch.Tensor, depth: int = 16):
    """
    Hilbert 曲线解码封装函数。

    输入参数:
        - code: torch.Tensor, (N,), Hilbert 编码值
        - depth: int, 编码深度

    输出:
        - grid_coord: torch.Tensor, (N, 3), 还原后的网格坐标
    """
    return hilbert_decode_(code, num_dims=3, num_bits=depth)

