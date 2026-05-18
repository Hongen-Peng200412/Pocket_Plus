# --------------------------------------------------------
"""假设坐标的二进制表示为：

$x = x_2 x_1 x_0$
$y = y_2 y_1 y_0$
$z = z_2 z_1 z_0$
莫顿码会将它们交织在一起，顺序通常是 $X$ 在高位，$Y$ 居中，$Z$ 在低位（每个位轮回一次）： 结果二进制序列为：$x_2 y_2 z_2 x_1 y_1 z_1  x_0 y_0 z_0$"""
# --------------------------------------------------------

import torch
from typing import Optional, Union


class KeyLUT:
    """
    Morton 编码/解码查询表 (Look-Up Table)。
    用于通过查表法快速转换坐标与莫顿码，避开费时的循环计算。
    """
    def __init__(self):
        # torch.Tensor, (256,), 0-255 的长整型序列, 用于生成 8 位深度查询表
        r256 = torch.arange(256, dtype=torch.int64)
        # torch.Tensor, (512,), 0-511 的长整型序列, 用于生成解码查询表
        r512 = torch.arange(512, dtype=torch.int64)
        # torch.Tensor, (256,), 全 0 序列
        zero = torch.zeros(256, dtype=torch.int64)
        # torch.device, 初始化在 CPU 上
        device = torch.device("cpu")

        # dict, 存储编码表: {device: (x_table, y_table, z_table)}
        self._encode = {
            device: (
                self.xyz2key(r256, zero, zero, 8),
                self.xyz2key(zero, r256, zero, 8),
                self.xyz2key(zero, zero, r256, 8),
            )
        }
        # dict, 存储解码表: {device: (x_table, y_table, z_table)}
        self._decode = {device: self.key2xyz(r512, 9)}

    def encode_lut(self, device=torch.device("cpu")):
        """
        获取指定设备上的编码查询表。
        输出:
            - lut: tuple of torch.Tensor, 包含 3 个 (256,) 的表格, 分别对应轴坐标位偏移后的键值
        """
        if device not in self._encode:
            cpu = torch.device("cpu")
            self._encode[device] = tuple(e.to(device) for e in self._encode[cpu])
        return self._encode[device]

    def decode_lut(self, device=torch.device("cpu")):
        """
        获取指定设备上的解码查询表。
        输出:
            - lut: tuple of torch.Tensor, 解码后的 (x, y, z) 坐标分量表
        """
        if device not in self._decode:
            # torch.device, CPU 引用
            cpu = torch.device("cpu")
            self._decode[device] = tuple(e.to(device) for e in self._decode[cpu])
        return self._decode[device]

    def xyz2key(self, x, y, z, depth):
        """
        计算坐标对应的莫顿键值 (循环法，仅用于初始化 LUT)。

        输入参数:
            - x: torch.Tensor, x 轴分量
            - y: torch.Tensor, y 轴分量
            - z: torch.Tensor, z 轴分量
            - depth: int, 编码深度 (位数)

        输出:
            - key: torch.Tensor, 混合后的一维莫顿键
        """
        # torch.Tensor, 初始化结果张量, 与输入形状一致
        key = torch.zeros_like(x)
        for i in range(depth):
            # int, 当前位的位掩码
            mask = 1 << i  # 把二进制数字的第i位向左移动2i + *
            key = (
                key                            # |: 按位或运算
                | ((x & mask) << (2 * i + 2))  # &: 逐位做与运算
                | ((y & mask) << (2 * i + 1))
                | ((z & mask) << (2 * i + 0))
            )
        return key

    def key2xyz(self, key, depth):
        """
        将莫顿键还原为坐标 (循环法，仅用于初始化 LUT)。

        输入参数:
            - key: torch.Tensor, 莫顿键
            - depth: int, 涉及的层数

        输出:
            - x, y, z: 分离后的坐标分量
        """
        # torch.Tensor, 初始化各轴结果
        x = torch.zeros_like(key)
        y = torch.zeros_like(key)
        z = torch.zeros_like(key)
        for i in range(depth):
            # 分别提取对应的交织位并移位回原坐标位置
            x = x | ((key & (1 << (3 * i + 2))) >> (2 * i + 2))  
            y = y | ((key & (1 << (3 * i + 1))) >> (2 * i + 1))
            z = z | ((key & (1 << (3 * i + 0))) >> (2 * i + 0))
        return x, y, z


# 全局查询表单例
_key_lut = KeyLUT()


def xyz2key(
    x: torch.Tensor,
    y: torch.Tensor,
    z: torch.Tensor,
    b: Optional[Union[torch.Tensor, int]] = None,
    depth: int = 16,
):
    """
    查表计算莫顿编码。支持 batch 索引的高位编码。

    输入参数:
        - x: torch.Tensor, (N,), x 轴网格坐标
        - y: torch.Tensor, (N,), y 轴网格坐标
        - z: torch.Tensor, (N,), z 轴网格坐标
        - b: torch.Tensor | int | None, batch 索引, 若提供则置于 48 位以上
        - depth: int, 编码深度 (建议 <= 16)

    输出:
        - key: torch.Tensor, (N,), 对应的编码结果 (int64)
    """
    # EX, EY, EZ: 分别为 x, y, z 对应的三条 8 位编码 LUT
    EX, EY, EZ = _key_lut.encode_lut(x.device)
    x, y, z = x.long(), y.long(), z.long()

    # int, 指向低 8 位的掩码
    mask = 255 if depth > 8 else (1 << depth) - 1
    # 查表合并低 8 位对应的莫顿编码
    key = EX[x & mask] | EY[y & mask] | EZ[z & mask]
    if depth > 8:
        # 提取 8-16 位的信息并再次查表合并
        mask = (1 << (depth - 8)) - 1
        # key16: 高 8 位对应的编码
        key16 = EX[(x >> 8) & mask] | EY[(y >> 8) & mask] | EZ[(z >> 8) & mask]
        key = key16 << 24 | key  # 左移 24 位 (因为每位 xyz 占 3 位, 8位占 24位)

    if b is not None:
        b = b.long()
        # 将 batch ID 放入高 48 位
        key = b << 48 | key

    return key


def key2xyz(key: torch.Tensor, depth: int = 16):
    """
    查表解码莫顿编码。

    输入参数:
        - key: torch.Tensor, (N,), 莫顿编码
        - depth: int, 编码深度

    输出:
        - x, y, z: torch.Tensor, (N,), 还原后的三维网格坐标
        - b: torch.Tensor, (N,), 还原后的 batch 索引
    """
    # DX, DY, DZ: (512,) 形状的解码 LUT
    DX, DY, DZ = _key_lut.decode_lut(key.device)
    # 初始化结果张量
    x, y, z = torch.zeros_like(key), torch.zeros_like(key), torch.zeros_like(key)

    # 提取高 48 位的 batch 信息
    b = key >> 48
    # 清除 batch 位, 仅保留 48 位以内的莫顿码
    key = key & ((1 << 48) - 1)

    # int, 需要循环解码的次数 (每 9 位对应一组 xyz 坐标分量)
    n = (depth + 2) // 3
    for i in range(n):
        # int, 提取当前 9 位窗口的数据作为查表索引
        k = key>>(i * 9)   & 511
        # 查表并移位回对应坐标位置
        x = x | (DX[k] << (i * 3))
        y = y | (DY[k] << (i * 3))
        z = z | (DZ[k] << (i * 3))

    return x, y, z, b

