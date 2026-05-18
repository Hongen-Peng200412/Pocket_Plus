# 这个脚本不看了
"""
Hilbert编码的形式化定义

1. 基本概念:
   - 设 H_d(k) 表示 d 维空间中阶数为 k 的 Hilbert 曲线
   - 阶数 k 表示将每条边等分为 2^k 段,形成 2^(dk) 个网格单元
   - Hilbert曲线是连续的、空间填充的曲线,将 d 维网格单元映射到一维整数索引

2. 递归构造 (Skilling算法):
   - 基础情况: k=0 时,H_d(0) 为单点,编码为 0
   - 递归步骤: 将 d 维超立方体划分为 2^d 个子超立方体
     * 每个子超立方体递归构造阶数为 k-1 的 Hilbert 曲线
     * 通过适当的旋转和反射,将子曲线连接成完整的 Hilbert 曲线

3. 格雷码转换:
   - 二进制到格雷码: G = B ⊕ (B >> 1)
     其中 ⊕ 表示按位异或,>> 表示右移
   - 格雷码到二进制: B = G ⊕ (G >> 1) ⊕ (G >> 2) ⊕ ... ⊕ (G >> (n-1))

4. 编码函数 encode(x₁, x₂, ..., x_d):
   输入: d 维整数坐标 (x₁, x₂, ..., x_d), 每个坐标 ∈ [0, 2^k-1]
   输出: 一维 Hilbert 编码 H ∈ [0, 2^(dk)-1]
   
   算法步骤:
   a) 将每个坐标表示为 k 位二进制: x_i = ∑_{j=0}^{k-1} b_{i,j}·2^j
   b) 构造格雷码矩阵 G_{i,j} = b_{i,j} ⊕ b_{i,j+1}
   c) 通过位交换和异或操作,将格雷码矩阵转换为 Hilbert 编码
   d) 将所有位打包为 64 位整数

5. 解码函数 decode(H):
   输入: 一维 Hilbert 编码 H ∈ [0, 2^(dk)-1]
   输出: d 维整数坐标 (x₁, x₂, ..., x_d)
   
   算法步骤:
   a) 将 H 展开为 dk 位二进制序列
   b) 转换为格雷码表示
   c) 通过逆向位交换和异或操作,恢复坐标的二进制表示
   d) 将二进制转换为整数坐标

6. 局部保持性:
   对于任意两个点 p, q ∈ [0, 2^k-1]^d, 若它们在 d 维空间中的欧氏距离为 d(p,q),
   则其 Hilbert 编码差值 |H(p) - H(q)| 与 d(p,q) 成正比。
   这保证了空间邻近的点在编码后仍然保持邻近性。

7. 时间复杂度:
   - 编码: O(d·k)
   - 解码: O(d·k)
   其中 d 为维度,k 为每维度的比特数

8. 应用场景:
   - 多维数据索引
   - 空间数据库查询优化
   - 负载均衡
   - 图像处理
"""
import torch


def right_shift(binary, k=1, axis=-1):
    """
    对二进制张量沿指定轴进行右移。

    # 输入参数:
        - binary: torch.Tensor, 输入的二进制张量 (通常包含 0 或 1)
        - k: int, 右移的位数
        - axis: int, 移位的维度轴

    # 输出:
        - shifted: torch.Tensor, 移位后的张量, 高位补 0, 低位截断
    """
    # If we're shifting the whole thing, just return zeros.
    if binary.shape[axis] <= k:
        return torch.zeros_like(binary)

    # Determine the slicing pattern to eliminate just the last one.
    # list, 用于多维切片的 slice 对象列表
    slicing = [slice(None)] * len(binary.shape)
    slicing[axis] = slice(None, -k)
    # torch.Tensor, 截断低位后的张量
    # torch.Tensor, 在高位填充 k 个 0
    shifted = torch.nn.functional.pad(
        binary[tuple(slicing)], (k, 0), mode="constant", value=0
    )

    return shifted


def binary2gray(binary, axis=-1):
    """
    将二进制编码转换为格雷码 (Gray Code)。使用经典公式: G = B ^ (B >> 1)。

    # 输入参数:
        - binary: torch.Tensor, 二进制张量
        - axis: int, 计算轴

    # 输出:
        - gray: torch.Tensor, 转换后的格雷码张量
    """
    # torch.Tensor, 沿指定轴右移一位后的张量
    shifted = right_shift(binary, axis=axis)

    # Do the X ^ (X >> 1) trick.
    # torch.Tensor, 逐位异或
    gray = torch.logical_xor(binary, shifted)

    return gray


def gray2binary(gray, axis=-1):
    """
    将格雷码转换回二进制编码。使用循环位移取异或的方法。

    # 输入参数:
        - gray: torch.Tensor, 格雷码张量
        - axis: int, 解码轴

    # 输出:
        - binary: torch.Tensor, 解码后的二进制张量
    """

    # Loop the log2(bits) number of times necessary, with shift and xor.
    # int, 初始位移量, 为轴长度的最高幂次
    shift = 2 ** (torch.Tensor([gray.shape[axis]]).log2().ceil().int() - 1)
    while shift > 0:
        # torch.Tensor, 累积异或
        gray = torch.logical_xor(gray, right_shift(gray, shift))
        # int, 位移量减半
        shift = torch.div(shift, 2, rounding_mode="floor")
    return gray


def encode(locs, num_dims, num_bits):
    """
    将高维空间坐标编码为希尔伯特整数 (Hilbert Integer)。
    采用 John Skilling 的向量化实现。

    # 输入参数:
        - locs: torch.Tensor, (..., num_dims), 整数坐标张量
        - num_dims: int, 空间维度 (例如 3)
        - num_bits: int, 每个维度分配的比特数

    # 输出:
        - hh_uint64: torch.Tensor, (..., ), 希尔伯特一维编码 (int64)
    """

    # Keep around the original shape for later.
    # torch.Size, 记录输入形状以便后续还原
    orig_shape = locs.shape
    # torch.Tensor, (8,), 用于位打包的掩码
    bitpack_mask = 1 << torch.arange(0, 8).to(locs.device)
    # torch.Tensor, (8,), 翻转后的掩码
    bitpack_mask_rev = bitpack_mask.flip(-1)

    if orig_shape[-1] != num_dims:
        raise ValueError(
            """
      The shape of locs was surprising in that the last dimension was of size
      %d, but num_dims=%d.  These need to be equal.
      """
            % (orig_shape[-1], num_dims)
        )

    if num_dims * num_bits > 63:
        raise ValueError(
            """
      num_dims=%d and num_bits=%d for %d bits total, which can't be encoded
      into a int64.  Are you sure you need that many points on your Hilbert
      curve?
      """
            % (num_dims, num_bits, num_dims * num_bits)
        )

    # Treat the location integers as 64-bit unsigned and then split them up into
    # a sequence of uint8s.  Preserve the association by dimension.
    # torch.Tensor, (N, num_dims, 8), 将坐标分解为 8 个字节
    locs_uint8 = locs.long().view(torch.uint8).reshape((-1, num_dims, 8)).flip(-1)

    # Now turn these into bits and truncate to num_bits.
    # torch.Tensor, (N, num_dims, num_bits), 提取出每个坐标分量的二进制位
    gray = (
        locs_uint8.unsqueeze(-1)
        .bitwise_and(bitpack_mask_rev)
        .ne(0)
        .byte()
        .flatten(-2, -1)[..., -num_bits:]
    )

    # Run the decoding process the other way.
    # Iterate forwards through the bits.
    for bit in range(0, num_bits):
        # Iterate forwards through the dimensions.
        for dim in range(0, num_dims):
            # Identify which ones have this bit active.
            # torch.Tensor, (N,), 当前维度在当前位的状态掩码
            mask = gray[:, dim, bit]

            # Where this bit is on, invert the 0 dimension for lower bits.
            # 对更高索引 (更低权重) 的位执行逻辑取反 (异或)
            gray[:, 0, bit + 1 :] = torch.logical_xor(
                gray[:, 0, bit + 1 :], mask[:, None]
            )

            # Where the bit is off, exchange the lower bits with the 0 dimension.
            # torch.Tensor, (N, num_bits - bit - 1), 记录需要交换的位掩码
            to_flip = torch.logical_and(
                torch.logical_not(mask[:, None]).repeat(1, gray.shape[2] - bit - 1),
                torch.logical_xor(gray[:, 0, bit + 1 :], gray[:, dim, bit + 1 :]),
            )
            # 通过逻辑异或实现位交换
            gray[:, dim, bit + 1 :] = torch.logical_xor(
                gray[:, dim, bit + 1 :], to_flip
            )
            gray[:, 0, bit + 1 :] = torch.logical_xor(gray[:, 0, bit + 1 :], to_flip)

    # Now flatten out.
    # torch.Tensor, (N, num_bits * num_dims), 将所有维度的位混合
    gray = gray.swapaxes(1, 2).reshape((-1, num_bits * num_dims))

    # Convert Gray back to binary.
    # torch.Tensor, (N, num_bits * num_dims), 格雷码转回二进制
    hh_bin = gray2binary(gray)

    # Pad back out to 64 bits.
    # int, 距离 64 位整型的剩余偏移
    extra_dims = 64 - num_bits * num_dims
    # torch.Tensor, (N, 64), 补齐至 64 位
    padded = torch.nn.functional.pad(hh_bin, (extra_dims, 0), "constant", 0)

    # Convert binary values into uint8s.
    # torch.Tensor, (N, 8), 重新打包回 8 字节
    hh_uint8 = (
        (padded.flip(-1).reshape((-1, 8, 8)) * bitpack_mask)
        .sum(2)
        .squeeze()
        .type(torch.uint8)
    )

    # Convert uint8s into uint64s.
    # torch.Tensor, (N,), 转换为最终的长整型结果
    hh_uint64 = hh_uint8.view(torch.int64).squeeze()

    return hh_uint64


def decode(hilberts, num_dims, num_bits):
    """
    将希尔伯特整数解码为高维网格坐标。

    # 输入参数:
        - hilberts: torch.Tensor, 一维或多维编码张量
        - num_dims: int, 目标维度
        - num_bits: int, 每个维度的比特精度

    # 输出:
        - flat_locs: torch.Tensor, (..., num_dims), 还原后的网格坐标
    """

    if num_dims * num_bits > 64:
        raise ValueError(
            """
      num_dims=%d and num_bits=%d for %d bits total, which can't be encoded
      into a uint64.  Are you sure you need that many points on your Hilbert
      curve?
      """
            % (num_dims, num_bits)
        )

    # Handle the case where we got handed a naked integer.
    # 确保输入至少是一维张量
    hilberts = torch.atleast_1d(hilberts)

    # Keep around the shape for later.
    # torch.Size, 原始输入形状
    orig_shape = hilberts.shape
    # torch.Tensor, (8,), 位掩码
    bitpack_mask = 2 ** torch.arange(0, 8).to(hilberts.device)
    # torch.Tensor, (8,), 翻转掩码
    bitpack_mask_rev = bitpack_mask.flip(-1)

    # Treat each of the hilberts as a s equence of eight uint8.
    # This treats all of the inputs as uint64 and makes things uniform.
    # torch.Tensor, (N, 8), 解压成 8 字节
    hh_uint8 = (
        hilberts.ravel().type(torch.int64).view(torch.uint8).reshape((-1, 8)).flip(-1)
    )

    # Turn these lists of uints into lists of bits and then truncate to the size
    # we actually need for using Skilling's procedure.
    # torch.Tensor, (N, num_dims * num_bits), 提取所有位
    hh_bits = (
        hh_uint8.unsqueeze(-1)
        .bitwise_and(bitpack_mask_rev)
        .ne(0)
        .byte()
        .flatten(-2, -1)[:, -num_dims * num_bits :]
    )

    # Take the sequence of bits and Gray-code it.
    # torch.Tensor, 解码前所需的格雷变换
    gray = binary2gray(hh_bits)

    # There has got to be a better way to do this.
    # I could index them differently, but the eventual packbits likes it this way.
    # torch.Tensor, (N, num_dims, num_bits), 重新排列维度
    gray = gray.reshape((-1, num_bits, num_dims)).swapaxes(1, 2)

    # Iterate backwards through the bits.
    # 逆向遍历比特和维度
    for bit in range(num_bits - 1, -1, -1):
        # Iterate backwards through the dimensions.
        for dim in range(num_dims - 1, -1, -1):
            # Identify which ones have this bit active.
            # torch.Tensor, (N,), 当前掩码
            mask = gray[:, dim, bit]

            # Where this bit is on, invert the 0 dimension for lower bits.
            gray[:, 0, bit + 1 :] = torch.logical_xor(
                gray[:, 0, bit + 1 :], mask[:, None]
            )

            # Where the bit is off, exchange the lower bits with the 0 dimension.
            # torch.Tensor, 状态标记
            to_flip = torch.logical_and(
                torch.logical_not(mask[:, None]),
                torch.logical_xor(gray[:, 0, bit + 1 :], gray[:, dim, bit + 1 :]),
            )
            # 进行位交换还原
            gray[:, dim, bit + 1 :] = torch.logical_xor(
                gray[:, dim, bit + 1 :], to_flip
            )
            gray[:, 0, bit + 1 :] = torch.logical_xor(gray[:, 0, bit + 1 :], to_flip)

    # Pad back out to 64 bits.
    # int, 补齐偏移
    extra_dims = 64 - num_bits
    # torch.Tensor, 准备重新打包
    padded = torch.nn.functional.pad(gray, (extra_dims, 0), "constant", 0)

    # Now chop these up into blocks of 8.
    # torch.Tensor, (N, num_dims, 8, 8), 还原为字节块形状
    locs_chopped = padded.flip(-1).reshape((-1, num_dims, 8, 8))

    # Take those blocks and turn them unto uint8s.
    # torch.Tensor, (N, num_dims, 8), 合并为字节流
    locs_uint8 = (locs_chopped * bitpack_mask).sum(3).squeeze().type(torch.uint8)

    # Finally, treat these as uint64s.
    # torch.Tensor, (N, num_dims), 还原为坐标长整型
    flat_locs = locs_uint8.view(torch.int64)

    # Return them in the expected shape.
    return flat_locs.reshape((*orig_shape, num_dims))