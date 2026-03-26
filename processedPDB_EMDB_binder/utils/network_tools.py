import time
import numpy as np
import numpy as np
import random


class RandomCrop(object):
    def __init__(self, output_size: int, ispadding: bool = True):
        """ 只能用在3D
         - output_size: int, 输出尺寸。
         - 注意！！！：o输出的形状就是 (output_size, output_size, output_size) """
        assert isinstance(output_size, int)
        self.output_size = (output_size, output_size, output_size)
        # ispadding 标志（是否进行填充）
        self.ispadding = ispadding

    # 定义 __call__ "魔术方法"，这使得类的实例可以像函数一样被调用
    # 例如: cropped_img, cropped_mask = crop_transform(image, mask)
    def __call__(self, *x):
        # *x 会将所有传入的参数（例如 image, mask）打包成一个元组 x
        # y 用于存储裁剪后的结果
        y = []
        
        # 获取第一个输入数组 x[0] 的形状 (Depth, Height, Width)
        # 这里假设所有传入的数组 x[0], x[1], ... 都具有相同的形状
        d, h, w = x[0].shape
        # 从 self 中获取目标输出尺寸 (Output_Depth, Output_Height, Output_Width)
        od, oh, ow = self.output_size

        # 检查是否需要执行填充
        if self.ispadding:
            x = list(x)
            # --- 计算深度(D)方向的填充 ---
            # k1: 需要填充的总深度。如果输入d >= od，则 k1=0
            k1 = max(od - d, 0)
            # pad1: 基础填充量（分到一侧的量）
            pad1 = k1 // 2
            # pads1: (pad_before, pad_after) 元组。如果 k1 是奇数，则在后面(after)多填充1个
            pads1 = (pad1, pad1) if k1 % 2 == 0 else (pad1, pad1 + 1)
            
            # --- 计算高度(H)方向的填充 ---
            k2 = max(oh - h, 0)
            pad2 = k2 // 2
            pads2 = (pad2, pad2) if k2 % 2 == 0 else (pad2, pad2 + 1)
            
            # --- 计算宽度(W)方向的填充 ---
            k3 = max(ow - w, 0)
            pad3 = k3 // 2
            pads3 = (pad3, pad3) if k3 % 2 == 0 else (pad3, pad3 + 1)

            # 遍历列表 x 中的所有数组（例如 image 和 mask）
            for i in range(len(x)):
                # 使用 np.pad 对当前数组 x[i] 进行填充
                # (pads1, pads2, pads3) 指定了三个维度的 (before, after) 填充量
                # mode='constant' 表示使用默认值 0.0 进行填充
                x[i] = np.pad(x[i], (pads1, pads2, pads3), mode='constant')

        # 在填充后，重新获取第一个数组的（可能已经增大了的）形状
        d, h, w = x[0].shape

        # --- 计算随机裁剪的起始坐标 ---
        # sd: 随机起始深度。范围是从 0 到 (d - od)
        sd = random.randint(0, d - od)
        # sh: 随机起始高度。范围是从 0 到 (h - oh)
        sh = random.randint(0, h - oh)
        # sw: 随机起始宽度。范围是从 0 到 (w - ow)
        sw = random.randint(0, w - ow)

        # 再次遍历 x 中的所有（可能已填充的）数组
        for ix in x:
            # 应用完全相同的裁剪坐标 (sd, sh, sw)
            # 使用 NumPy 切片语法提取子块
            cropped_array = ix[sd:sd + od, sh:sh + oh, sw:sw + ow]
            # 将裁剪后的数组添加到结果列表 y 中
            y.append(cropped_array)
        # 将结果列表 y 转换回元组
        y = tuple(y)
        
        # 返回包含所有裁剪后数组的元组
        return y


class RandomCrop_4D(object):
    def __init__(self, output_size: int, ispadding: bool = True):
        """ 只能用在4D
         - output_size: int, 输出尺寸。
         - 注意！！！：o输出的形状就是 (output_size, output_size, output_size) """
        assert isinstance(output_size, int)
        self.output_size = (output_size, output_size, output_size)
        # ispadding 标志（是否进行填充）
        self.ispadding = ispadding

    # 定义 __call__ "魔术方法"，这使得类的实例可以像函数一样被调用
    # 例如: cropped_img, cropped_mask = crop_transform(image, mask)
    def __call__(self, *x):
        # *x 会将所有传入的参数（例如 image, mask）打包成一个元组 x
        # y 用于存储裁剪后的结果
        y = []
        
        # 获取第一个输入数组 x[0] 的形状 (Depth, Height, Width)
        # 这里假设所有传入的数组 x[0], x[1], ... 都具有相同的形状
        c, d, h, w = x[0].shape
        # 从 self 中获取目标输出尺寸 (Output_Depth, Output_Height, Output_Width)
        od, oh, ow = self.output_size

        # 检查是否需要执行填充
        if self.ispadding:
            x = list(x)
            # --- 计算深度(D)方向的填充 ---
            # k1: 需要填充的总深度。如果输入d >= od，则 k1=0
            k1 = max(od - d, 0)
            # pad1: 基础填充量（分到一侧的量）
            pad1 = k1 // 2
            # pads1: (pad_before, pad_after) 元组。如果 k1 是奇数，则在后面(after)多填充1个
            pads1 = (pad1, pad1) if k1 % 2 == 0 else (pad1, pad1 + 1)
            
            # --- 计算高度(H)方向的填充 ---
            k2 = max(oh - h, 0)
            pad2 = k2 // 2
            pads2 = (pad2, pad2) if k2 % 2 == 0 else (pad2, pad2 + 1)
            
            # --- 计算宽度(W)方向的填充 ---
            k3 = max(ow - w, 0)
            pad3 = k3 // 2
            pads3 = (pad3, pad3) if k3 % 2 == 0 else (pad3, pad3 + 1)

            # 遍历列表 x 中的所有数组（例如 image 和 mask）
            for i in range(len(x)):
                # 使用 np.pad 对当前数组 x[i] 进行填充
                # (pads1, pads2, pads3) 指定了三个维度的 (before, after) 填充量
                # mode='constant' 表示使用默认值 0.0 进行填充
                x[i] = np.pad(x[i], ((0, 0), pads1, pads2, pads3), mode='constant')

        # 在填充后，重新获取第一个数组的（可能已经增大了的）形状
        c, d, h, w = x[0].shape

        # --- 计算随机裁剪的起始坐标 ---
        # sd: 随机起始深度。范围是从 0 到 (d - od)
        sd = random.randint(0, d - od)
        # sh: 随机起始高度。范围是从 0 到 (h - oh)
        sh = random.randint(0, h - oh)
        # sw: 随机起始宽度。范围是从 0 到 (w - ow)
        sw = random.randint(0, w - ow)

        # 再次遍历 x 中的所有（可能已填充的）数组
        for ix in x:
            # 应用完全相同的裁剪坐标 (sd, sh, sw)
            # 使用 NumPy 切片语法提取子块
            cropped_array = ix[:, sd:sd + od, sh:sh + oh, sw:sw + ow]
            # 将裁剪后的数组添加到结果列表 y 中
            y.append(cropped_array)
        # 将结果列表 y 转换回元组
        y = tuple(y)
        
        # 返回包含所有裁剪后数组的元组
        return y


class RandomCrop_plus(object):
    def __init__(self, output_size: int, ispadding: bool = True):
        """ 
        Args:
         - output_size: int, 输出尺寸。
         - 注意！！！：输出的形状就是 (output_size, output_size, output_size)
         
        Forward args:
         - *x 里面可以是3D或4D图的混合, 可以是tensor或者ndarray.但最后三维必须是空间维 D H W 用于裁剪
            """
        assert isinstance(output_size, int)
        self.output_size = (output_size, output_size, output_size)
        # ispadding 标志（是否进行填充）
        self.ispadding = ispadding

    # 定义 __call__ "魔术方法"，这使得类的实例可以像函数一样被调用
    # 例如: cropped_img, cropped_mask = crop_transform(image, mask)
    def __call__(self, *x):
        # *x 会将所有传入的参数（例如 image, mask）打包成一个元组 x
        # y 用于存储裁剪后的结果
        y = []
        
        # 获取第一个输入数组 x[0] 的形状 (Depth, Height, Width)
        # 这里假设所有传入的数组 x[0], x[1], ... 都具有相同的形状
        d, h, w = x[0].shape[-3:]
        # 从 self 中获取目标输出尺寸 (Output_Depth, Output_Height, Output_Width)
        od, oh, ow = self.output_size

        # 检查是否需要执行填充
        if self.ispadding:
            x = list(x)
            # --- 计算深度(D)方向的填充 ---
            # k1: 需要填充的总深度。如果输入d >= od，则 k1=0
            k1 = max(od - d, 0)
            # pad1: 基础填充量（分到一侧的量）
            pad1 = k1 // 2
            # pads1: (pad_before, pad_after) 元组。如果 k1 是奇数，则在后面(after)多填充1个
            pads1 = (pad1, pad1) if k1 % 2 == 0 else (pad1, pad1 + 1)
            
            # --- 计算高度(H)方向的填充 ---
            k2 = max(oh - h, 0)
            pad2 = k2 // 2
            pads2 = (pad2, pad2) if k2 % 2 == 0 else (pad2, pad2 + 1)
            
            # --- 计算宽度(W)方向的填充 ---
            k3 = max(ow - w, 0)
            pad3 = k3 // 2
            pads3 = (pad3, pad3) if k3 % 2 == 0 else (pad3, pad3 + 1)

            # 遍历列表 x 中的所有数组（例如 image 和 mask）
            for i in range(len(x)):
                # 使用 np.pad 对当前数组 x[i] 进行填充
                # (pads1, pads2, pads3) 指定了三个维度的 (before, after) 填充量， mode='constant' 表示使用默认值 0.0 进行填充
                if len(x[i].shape) == 3:
                    x[i] = np.pad(x[i], (pads1, pads2, pads3), mode='constant')
                elif len(x[i].shape) == 4:
                    x[i] = np.pad(x[i], ((0, 0), pads1, pads2, pads3), mode='constant')

        # 在填充后，重新获取第一个数组的（可能已经增大了的）形状
        d, h, w = x[0].shape[-3:]

        # --- 计算随机裁剪的起始坐标 ---
        # sd: 随机起始深度。范围是从 0 到 (d - od)
        sd = random.randint(0, d - od)
        # sh: 随机起始高度。范围是从 0 到 (h - oh)
        sh = random.randint(0, h - oh)
        # sw: 随机起始宽度。范围是从 0 到 (w - ow)
        sw = random.randint(0, w - ow)

        # 再次遍历 x 中的所有（可能已填充的）数组
        for ix in x:
            # 应用完全相同的裁剪坐标 (sd, sh, sw), 使用 NumPy 切片语法提取子块
            cropped_array = ix[..., sd:sd + od, sh:sh + oh, sw:sw + ow]
            # 将裁剪后的数组添加到结果列表 y 中
            y.append(cropped_array)
        y = tuple(y)
        
        return y





def image_segmentation(image):
    Z,Y,X = image.shape
    stride = 50
    kernel = 64
    pad_temp = (kernel-stride)//2
    segmentation=[]
    tz = int(np.ceil(Z/stride));ty = int(np.ceil(Y/stride));tx = int(np.ceil(X/stride));
    dz = (-Z)%stride;dy = (-Y)%stride;dx = (-X)%stride;
    step = (tz,ty,tx)
    mask = np.ones(image.shape)
    image_new = np.pad(image,((dz//2,dz//2),(dy//2,dy//2),(dx//2,dx//2)),mode='constant')
    mask = np.pad(mask, ((dz // 2, dz // 2), (dy // 2, dy // 2), (dx // 2, dx // 2)), mode='constant')
    image_new = np.pad(image_new,((pad_temp,pad_temp),(pad_temp,pad_temp),(pad_temp,pad_temp)),mode='constant')
    mask = np.pad(mask, ((pad_temp, pad_temp), (pad_temp, pad_temp), (pad_temp, pad_temp)), mode='constant')
    for ii in range(tz):
        for jj in range(ty):
            for kk in range(tx):
                segmentation.append((image_new[ii*stride:ii*stride+kernel,jj*stride:jj*stride+kernel,kk*stride:kk*stride+kernel]
                                    ,mask[ii*stride:ii*stride+kernel,jj*stride:jj*stride+kernel,kk*stride:kk*stride+kernel]))
    return segmentation,step
def image_reconstruction(segmentation,img_shape,step):
    tz,ty,tx = step
    Z,Y,X = img_shape
    stride = 50
    kernel = 64
    pad_temp = (kernel - stride) // 2
    ideal_shape = (tz*stride,ty*stride,tx*stride)
    CZ,CY,CX = np.array(ideal_shape)//2
    ideal_reconsturction = np.zeros(ideal_shape)
    img_number = 0
    for ii in range(tz):
        for jj in range(ty):
            for kk in range(tx):
                img_temp = segmentation[img_number]
                ideal_reconsturction[ii*stride:ii*stride+stride,jj*stride:jj*stride+stride,
                kk*stride:kk*stride+stride] = img_temp[pad_temp:pad_temp+stride,pad_temp:pad_temp+stride
                                              ,pad_temp:pad_temp+stride]
                img_number = img_number + 1
    reconsturction = ideal_reconsturction[CZ-Z//2:CZ+Z//2,CY-Y//2:CY+Y//2,CX-X//2:CX+X//2]
    return reconsturction

def map2atom(index,voxel_size,global_origin):
    ii,jj,kk=index
    index_new = np.array([kk,jj,ii])
    atom_coordinate = index_new * voxel_size + global_origin
    return atom_coordinate
def mass_center(mass,coordinate):
    new = np.dot(mass,coordinate)/np.sum(mass)
    return new
def range_intersection(i,m,arg):
    if m>0:
        range1 = np.array(range(i,i+m+1))
        if (i+1) <= arg:
            return range1[np.where(range1<=(arg-1))][-1]
        else:
            return (arg-1)
    elif m<0:
        range1 = np.array(range(i+m,i+1))
        if i >= 0:
            return range1[np.where(range1 >= 0)][0]
        else:
            return 0
    else:
        return i

def Find_trace(data,target,L,voxel_size,global_origin):
    atom=[]
    atom_in = 0
    data[:2,:,:] = 0
    data[-2:,:,] = 0
    data[:,:2,:] = 0
    data[:,-2:,:] = 0
    data[:,:,:2] = 0
    data[:,:,-2:] = 0
    for i in range(L):
        max_index = np.unravel_index(np.argmax(data),data.shape)
        ii,jj,kk = max_index
        atom_in = atom_in + target[ii,jj,kk]
        mass = np.array([data[ii,jj,kk],data[ii-1,jj,kk],data[ii+1,jj,kk],data[ii,jj-1,kk],data[ii,jj+1,kk],
        data[ii,jj,kk-1],data[ii,jj,kk+1]])
        coordinate = np.array([[ii+0.5,jj+0.5,kk+0.5],[ii-0.5,jj+0.5,kk+0.5],[ii+1.5,jj+0.5,kk+0.5],
                               [ii+0.5,jj-0.5,kk+0.5],[ii+0.5,jj+1.5,kk+0.5],[ii+0.5,jj+0.5,kk-0.5],
                               [ii+0.5,jj+0.5,kk+1.5]])
        new_i,new_j,new_k = mass_center(mass,coordinate)
        atom.append(map2atom((new_i,new_j,new_k),voxel_size=voxel_size,global_origin=global_origin))
        data[ii-1:ii+2,jj-1:jj+2,kk]=0
        data[ii-1:ii+2,jj,kk-1]=0
        data[ii,jj-1:jj+2,kk-1]=0
        data[ii - 1:ii + 2, jj, kk + 1] = 0
        data[ii, jj - 1:jj + 2, kk + 1] = 0
        data[ii,jj,kk+2]=0
        data[ii, jj, kk-2] = 0
        data[ii-2,jj,kk]=0
        data[ii + 2, jj, kk] = 0
        data[ii, jj-2, kk] = 0
        data[ii, jj + 2, kk] = 0
    atom = np.array(atom)
    precision = atom_in / L
    return atom,precision








# ------------------------------- 推理用到的 ------------------------------

def _compute_window_starts(dim: int, windows_size: int, stride: int) -> list:
    """
    Compute sliding-window start indices, ensuring the last window hits the boundary without duplicates.
    """
    if stride <= 0:
        raise ValueError(f"stride must be positive, got {stride}")
    if dim <= windows_size:
        return [0]
    starts = list(range(0, dim - windows_size + 1, stride))
    last = dim - windows_size
    if starts[-1] != last:
        starts.append(last)
    return starts


def map_segmentation(image, windows_size: int = 64, stride: int = 50):
    """
    Split a volume into sliding-window blocks and return their start coordinates.

    Returns:
        - blocks: list, each block has shape (..., D, H, W)
        - coords: list[tuple], each item is (z0, y0, x0)
    """
    Z, Y, X = image.shape[-3:]
    blocks = []
    coords = []
    z_starts = _compute_window_starts(Z, windows_size, stride)
    y_starts = _compute_window_starts(Y, windows_size, stride)
    x_starts = _compute_window_starts(X, windows_size, stride)
    for z0 in z_starts:
        for y0 in y_starts:
            for x0 in x_starts:
                block = image[..., z0:z0 + windows_size,
                              y0:y0 + windows_size,
                              x0:x0 + windows_size]
                blocks.append(block)
                coords.append((z0, y0, x0))
    return blocks, coords




def make_gaussian_weight(windows_size: int, sigma_scale: float = 0.125, dtype=np.float32):
    """
    Create a 3D Gaussian weight template for sliding-window fusion.

    Args:
        - windows_size: int, cubic window edge length
        - sigma_scale: float, sigma = sigma_scale * windows_size
        - dtype: np.dtype, output dtype

    Returns:
        - weight: np.ndarray, shape (D, H, W), max-normalized to 1
    """
    if windows_size <= 0:
        raise ValueError(f"windows_size must be positive, got {windows_size}")
    sigma = max(windows_size * float(sigma_scale), 1e-6)
    center = (windows_size - 1) / 2.0
    ax = np.arange(windows_size, dtype=np.float32) - center
    g1 = np.exp(-0.5 * (ax / sigma) ** 2)
    g3 = g1[:, None, None] * g1[None, :, None] * g1[None, None, :]
    max_val = float(g3.max())
    if max_val > 0:
        g3 = g3 / max_val
    return g3.astype(dtype, copy=False)




def _to_numpy(arr):
    if isinstance(arr, np.ndarray):
        return arr
    try:
        import torch
        if isinstance(arr, torch.Tensor):
            return arr.detach().cpu().numpy()
    except Exception:
        pass
    return np.asarray(arr)



def map_reconstruction(
    blocks,
    image_shape,
    coords,
    windows_size=64,
    core_offset=2,
    blocks_weight=None,
):
    """
    提取预测块重建出整体的图像 (滑窗推理重构), 要重建的图像可以是三维或四维。

    Args:
        - blocks: numpy.ndarray, (N, D, H, W) 或 (N, C, D, H, W), 表示包含的 N 个网络预测出的块的集合。
        - image_shape: tuple, (D, H, W) 或 (C, D, H, W), 想要重构出的完整 3D 目标区域的通道数与尺寸形状。
        - coords: list[tuple], (N, 3), 表示各 patch 的坐标起始位置。
        - windows_size: int, 分割切块时所采用的窗格边缘长度，默认为 64。
        - core_offset: int, 默认 2，每个 block 各个有效数据表面因为感受野截断效应而被抛弃减去的大小。
        - blocks_weight: numpy.ndarray 或 None, (D,H,W) 或 (N,C,D,H,W), 各预测图对应位置融合权值。默认为 None。

    Return:
        - reconstruction: numpy.ndarray, (C, D, H, W)
    """
    # blocks_np: numpy.ndarray, 形状 (N, C, D, H, W) 或 (N, D, H, W)
    blocks_np = _to_numpy(blocks)
    # parsed_weight: numpy.ndarray 或 None, 形状 (D, H, W) 或 (N, C, D, H, W)
    parsed_weight = _to_numpy(blocks_weight) if blocks_weight is not None else None
    # C: 通道数, Z, Y, X: 空间维度。如果 image_shape 是 (D, H, W)，则 C 默认为 1。
    # 注意：image_shape 通常为 tuple，不支持 .ndim 属性，故使用 len() 判断。
    C, Z, Y, X = image_shape if len(image_shape) == 4 else (1, *image_shape)

    if blocks_np.ndim == 4:
        # (N, D, H, W) -> (N, 1, D, H, W)
        blocks_np = np.expand_dims(blocks_np, axis=1)
    if len(coords) != blocks_np.shape[0]:
        raise ValueError(f'coords length ({len(coords)}) != blocks count ({blocks_np.shape[0]})')
    # weight_mode: str, 记录权重配置策略是共用还是单块特供
    weight_mode = None
    # numpy.ndarray 或 None, 形状 (1, 1, D, H, W) 或 (N, C, D, H, W), 被整理好的可做广播操作的重构权重系数
    weight_block = None
    if parsed_weight is not None:
        if parsed_weight.ndim == 3:
            # numpy.ndarray, (D, H, W) -> (1, 1, D, H, W)
            weight_block = np.expand_dims(parsed_weight, axis=(0, 1))
            weight_mode = 'shared'
        elif parsed_weight.ndim == 5:
            weight_block = parsed_weight
            weight_mode = 'per_block'
        else:
            raise ValueError('blocks_weight must be (D, H, W) or (N, C, D, H, W)')
    reconstruction = np.zeros((C, Z, Y, X), dtype=np.float32)
    counts = np.zeros((C, Z, Y, X), dtype=np.float32)
    image_number = 0
    for (raw_z0, raw_y0, raw_x0) in coords:
        # block的未修剪末端索引
        raw_z1 = min(raw_z0 + windows_size, Z)
        raw_y1 = min(raw_y0 + windows_size, Y)
        raw_x1 = min(raw_x0 + windows_size, X)

        block_item = blocks_np[image_number]
        # 修剪后的待拼位置
        z0 = raw_z0 + core_offset if raw_z0 > 0 else raw_z0
        z1 = raw_z1 - core_offset if raw_z1 < Z else raw_z1
        y0 = raw_y0 + core_offset if raw_y0 > 0 else raw_y0
        y1 = raw_y1 - core_offset if raw_y1 < Y else raw_y1
        x0 = raw_x0 + core_offset if raw_x0 > 0 else raw_x0
        x1 = raw_x1 - core_offset if raw_x1 < X else raw_x1
        # 最终跨度
        tz = z1 - z0
        ty = y1 - y0
        tx = x1 - x0
        if tz <= 0 or ty <= 0 or tx <= 0:
            image_number += 1
            continue

        # BOX内部起始
        bz0 = core_offset if raw_z0 > 0 else 0
        by0 = core_offset if raw_y0 > 0 else 0
        bx0 = core_offset if raw_x0 > 0 else 0
        # BOX内部终止
        bz1 = bz0 + tz
        by1 = by0 + ty
        bx1 = bx0 + tx
        # 裁剪后有效BOX区域
        cropped = block_item[:, bz0:bz1, by0:by1, bx0:bx1]

        if parsed_weight is None:
            reconstruction[:, z0:z1, y0:y1, x0:x1] += cropped
            counts[:, z0:z1, y0:y1, x0:x1] += 1
        else:
            if weight_mode == 'shared':
                weight_cropped = weight_block[0, :, bz0:bz1, by0:by1, bx0:bx1]
            else:
                weight_cropped = weight_block[image_number, :, bz0:bz1, by0:by1, bx0:bx1]
            
            reconstruction[:, z0:z1, y0:y1, x0:x1] += cropped * weight_cropped
            counts[:, z0:z1, y0:y1, x0:x1] += weight_cropped

        image_number += 1

    # numpy.ndarray, 形状 (C, Z, Y, X)
    reconstruction = reconstruction / (counts + 1e-6)
    return reconstruction

# ------------------------------- 推理用到的 ------------------------------



def random_rotation90(*x):
    """
    对输入的 3D 数组应用随机的 90 度旋转。旋转操作选择两个随机空间轴并执行 90、180、270 或 360 度的旋转。
    
    参数:
        *x: 一个或多个 3D 数组。

    返回:
        tuple: 包含旋转后的 3D 数组的元组。
    """
    y = []  # 用于存储旋转后的数组
    
    # 随机选择两个不同的空间轴（0, 1, 2 分别代表 D, H, W 轴）
    axis1, axis2 = np.random.choice([0, 1, 2], size=2, replace=False)
    # 随机选择旋转的次数（k: 0 - 不旋转, 1 - 旋转 90°，2 - 旋转 180°，3 - 旋转 270°）
    k = random.randint(0, 3)
    # 遍历所有输入数组，执行旋转
    for ix in x:
        rotated_view = np.rot90(ix, k, (axis1, axis2))
        # 对每个数组在选定的两个空间轴上进行旋转
        y.append(rotated_view.copy())
    # 将旋转后的数组放入元组返回
    y = tuple(y)
    return y



def random_rotation90_4D(*x):
    """
    对传入的 3D (D, H, W) 或 4D (C, D, H, W) 数组应用相同的随机空间旋转。
    
    该函数随机选择两个空间轴，并在这些轴上对数组进行 90 度旋转，旋转角度为 90、180、270 或 360 度。
    对 4D 数组，旋转操作应用到每个通道 (C 维度)，而对 3D 数组则直接应用旋转。
    
    参数:
        - *x: 一个或多个 3D (D, H, W) 或 4D (C, D, H, W) 数组。
        - 4D 数组 (C, D, H, W)时, 通道维度 (C 维度)一定不能被旋转

    返回:
        tuple: 包含旋转后的 3D 或 4D 数组的元组。
    """
    y = []  # 用于存储旋转后的数组
    
    # 从 3 个空间轴 (D, H, W) 中随机选择两个轴进行旋转
    axis1, axis2 = np.random.choice([0, 1, 2], size=2, replace=False)
    #  随机选择旋转次数（k: 0 - 不旋转, 1 - 旋转 90°，2 - 旋转 180°，3 - 旋转 270°）
    k = random.randint(0, 3)

    # 遍历所有传入的数组
    for ix in x:
        if ix.ndim == 4:
            # 4D 数组 (C, D, H, W)：通道维度 (C 维度)一定不能被旋转
            # 对于 4D 数组，空间轴 D, H, W 对应于 (1, 2, 3)，因此需要将旋转轴加 1
            rot_axes = (axis1 + 1, axis2 + 1)
            rotated_view = np.rot90(ix, k, rot_axes)
            y.append(rotated_view.copy())  # 对整个 4D 数组进行旋转
            
        elif ix.ndim == 3:
            # 3D 数组 (D, H, W)：直接对 (D, H, W) 轴进行旋转
            rot_axes = (axis1, axis2)
            rotated_view = np.rot90(ix, k, rot_axes)
            y.append(rotated_view.copy())  # 对 3D 数组进行旋转
            
        else:
            # 如果输入的数据不是 3D 或 4D 数组，则不做旋转，原样返回
            y.append(ix)

    # 返回旋转后的结果元组
    return tuple(y)




def random_rotation90_plus(*x):
    """
    对输入的 数组(3D,4D都可以)应用随机的 90 度旋转。旋转操作选择两个随机空间轴并执行 90、180、270 或 360 度的旋转。
    
    参数:
        *x: 一个或多个数组, 它们都要在3D以上， 对最后三维旋转， 最后三个维度必须是 D H W

    返回:
        tuple: 包含旋转后数组的元组。
    """
    y = []  # 用于存储旋转后的数组
    
    # 随机选择两个不同的空间轴（三维时， 0, 1, 2 分别代表 D, H, W 轴）
    axis1, axis2 = np.random.choice([0, 1, 2], size=2, replace=False)
    # 随机选择旋转的次数（k: 0 - 不旋转, 1 - 旋转 90°，2 - 旋转 180°，3 - 旋转 270°）
    k = random.randint(0, 3)
    # 遍历所有输入数组，执行旋转
    for ix in x:
        ix_dim = len(ix.shape)
        rotated_view = np.rot90(ix, k, (axis1 + (ix_dim-3), axis2 + (ix_dim-3)))
        # 对每个数组在选定的两个空间轴上进行旋转
        y.append(rotated_view.copy())
    # 将旋转后的数组放入元组返回
    y = tuple(y)
    return y





def make_mask(target):
    pad = 2
    data = np.pad(np.copy(target),((pad,pad),(pad,pad),(pad,pad)),mode='constant')
    mask_index = np.where(data==1)
    mask_pairs = zip(mask_index[0],mask_index[1],mask_index[2])
    for ii,jj,kk in mask_pairs:
        data[ii - 1:ii + 2, jj - 1:jj + 2, kk] = 1
        data[ii - 1:ii + 2, jj, kk - 1] = 1
        data[ii, jj - 1:jj + 2, kk - 1] = 1
        data[ii - 1:ii + 2, jj, kk + 1] = 1
        data[ii, jj - 1:jj + 2, kk + 1] = 1
        data[ii, jj, kk + 2] = 1
        data[ii, jj, kk - 2] = 1
        data[ii - 2, jj, kk] = 1
        data[ii + 2, jj, kk] = 1
        data[ii, jj - 2, kk] = 1
        data[ii, jj + 2, kk] = 1
    data = data[2:-2,2:-2,2:-2]
    return data

def test_segmentation(image):
    Z,Y,X = image.shape
    stride = 50
    kernel = 64
    blocks = []
    for ii in range(0, Z + stride, stride):
        for jj in range(0, Y + stride, stride):
            for kk in range(0, X + stride, stride):
                block = image[min(ii, Z - kernel):min(ii + kernel, Z),
                        min(jj, Y - kernel):min(jj + kernel, Y)
                , min(kk, X - kernel):min(kk + kernel, X)]
                blocks.append(block)
    return blocks
def test_reconstruction(blocks,img_shape):
    Z,Y,X = img_shape
    stride = 50
    kernel = 64
    pad_temp = (kernel-stride)//2
    reconstruction = np.zeros(img_shape)
    counts = np.zeros(img_shape)
    image_number = 0
    for ii in range(0, Z + stride, stride):
        for jj in range(0, Y + stride, stride):
            for kk in range(0, X + stride, stride):
                ii = ii + pad_temp
                jj = jj + pad_temp
                kk = kk + pad_temp
                IZ = Z - pad_temp
                IY = Y - pad_temp
                IX = X - pad_temp
                reconstruction[min(ii, IZ - stride):min(ii + stride, IZ),
                min(jj, IY - stride):min(jj + stride, IY)
                , min(kk, IX - stride):min(kk + stride, IX)] = reconstruction[min(ii, IZ - stride):min(
                    ii + stride, IZ), min(jj, IY - stride):min(jj + stride, IY)
                                                                         , min(kk, IX - stride):min(
                    kk + stride, IX)] + blocks[image_number][pad_temp:pad_temp+stride,pad_temp:pad_temp+stride,pad_temp:pad_temp+stride]
                counts[min(ii, IZ - stride):min(ii + stride, IZ),
                min(jj, IY - stride):min(jj + stride, IY)
                , min(kk, IX - stride):min(kk + stride, IX)] = counts[min(ii, IZ - stride):min(
                    ii + stride, IZ), min(jj, IY - stride):min(jj + stride, IY)
                                                                         , min(kk, IX - stride):min(
                    kk + stride, IX)] + 1
                image_number = image_number + 1
    counts = np.where(counts==0,1e-8,counts)
    reconstruction = reconstruction / counts
    return reconstruction





import numpy as np
import os
import tempfile

def atomic_np_save(filename, arr, do_not_replace=True):
    """
    原子地保存 numpy .npy 文件，用法为：atomic_np_save(target_file, data_array)
    """
    if do_not_replace and os.path.exists(filename):
        return
    target_dir = os.path.dirname(filename)
    if not target_dir:
        target_dir = '.'
    os.makedirs(target_dir, exist_ok=True)

    # 使用后缀便于调试
    fd = None
    tmp_filename = None
    try:
        # 使用 mkstemp 得到文件描述符，避免 Windows 上 NamedTemporaryFile 的删除/重命名问题
        fd, tmp_filename = tempfile.mkstemp(suffix='.npy', dir=target_dir)
        # 将 fd 转为文件对象以便 np.save
        with os.fdopen(fd, 'wb') as f:
            fd = None  # 表示 fd 已经由 fdopen 管理，不需要再次关闭
            np.save(f, arr)
            f.flush()
            os.fsync(f.fileno())
        # 此时临时文件已关闭，可以安全替换
        os.replace(tmp_filename, filename)
    except Exception:
        # 若出错，尝试关闭并删除临时文件（注意 fd 可能还未被 fdopen 管理）
        try:
            if fd is not None:
                os.close(fd)
        except Exception:
            pass
        if tmp_filename and os.path.exists(tmp_filename):
            try:
                os.remove(tmp_filename)
            except Exception:
                # 不要阻塞原始异常，记录或忽略
                pass
        raise


# # --- 示例使用 ---
# if __name__ == '__main__':
#     # 1. 准备一个数组数据
#     data_array = np.random.rand(4, 5)
#     target_file = 'atomic_test_array.npy'
#     # 2. 调用原子保存函数
#     atomic_np_save(target_file, data_array)

#     # 3. 验证保存结果
#     try:
#         loaded_array = np.load(target_file)
#         print("\n验证加载的数据:")
#         print(loaded_array)
#         # 确认数据正确
#         assert np.array_equal(data_array, loaded_array)
#         print("数据验证成功!")
#     finally:
#         # 4. 清理创建的文件
#         if os.path.exists(target_file):
#             os.remove(target_file)
#             print(f"\n清理文件: {target_file}")








def atomic_np_savez(filename, do_not_replace=True, **kwargs):
    """
    原子地保存 numpy .npz 文件,用法为：
    atomic_np_savez(target_file, array_a=data_a, array_b=data_b)
    读取时loaded_data = np.load(target_file) 包含了loaded_data['array_a']和loaded_data['array_b']这两个变量
    """
    if do_not_replace and os.path.exists(filename):
        return
    target_dir = os.path.dirname(filename)
    if not target_dir:
        target_dir = '.'
    os.makedirs(target_dir, exist_ok=True)

    fd = None
    tmp_filename = None
    try:
        fd, tmp_filename = tempfile.mkstemp(suffix='.npz', dir=target_dir)
        with os.fdopen(fd, 'wb') as f:
            fd = None
            # np.savez 接受 file-like object
            np.savez(f, **kwargs)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_filename, filename)
    except Exception:
        try:
            if fd is not None:
                os.close(fd)
        except Exception:
            pass
        if tmp_filename and os.path.exists(tmp_filename):
            try:
                os.remove(tmp_filename)
            except Exception:
                pass
        raise


# if __name__ == '__main__':
#     # 1. 准备一些数据
#     data_a = np.array([1, 2, 3])
#     data_b = np.arange(10).reshape(2, 5)

#     target_file = 'atomic_test_data.npz'

#     # 2. 调用原子保存函数
#     atomic_np_savez(target_file, array_a=data_a, array_b=data_b)

#     # 3. 验证保存结果
#     try:
#         loaded_data = np.load(target_file)
#         print("\n验证加载的数据:")
#         print("array_a:", loaded_data['array_a'])
#         print("array_b:", loaded_data['array_b'])
        
#         # 确认数据正确
#         assert np.array_equal(data_a, loaded_data['array_a'])
#         assert np.array_equal(data_b, loaded_data['array_b'])
#         print("数据验证成功!")
        
#     finally:
#         # 4. 清理创建的文件
#         if os.path.exists(target_file):
#             os.remove(target_file)
#             print(f"\n清理文件: {target_file}")
    


import mrcfile
import mrcfile as mrc
def save_map_perfect_copy(file_path, data, original_map_path, new_origin_xyz):
    """
    将给定的数据写为新的 MRC 文件，同时尽量保留原 MRC 的头信息（voxel_size, cella, map axes 等），并设置新的 origin。
    常见调用方式：
    save_map_perfect_copy(out_path_1, grid_self, original_map_path, origin)
    这里  origin由：   grid,voxel_size,global_origin = load_map_and_origin(original_map_path) 得到。
    grid_self是自己要可视化的坐标网格，它与原始密度图 original_map_path 必须是匹配的(由它生成的)

    Inputs:
      - file_path (str): 新 MRC 文件保存路径（若存在将被覆盖）。
      - data (np.ndarray): 要写入的密度数据，假定轴顺序为 (z, y, x)。
      - original_map_path (str): 原始 MRC 文件路径，用于读取并复制头信息（metadata）。
      - new_origin_xyz (array-like, length 3): 要写入的新 origin，按 (x, y, z)。
    Outputs:
      - None（函数在磁盘上创建/覆盖 file_path）

    Key steps / 关键步骤:
      1. 打开原始 MRC，读取 voxel_size、cella 与 mapc/mapr/maps（轴映射信息），以便新文件与原文件保持一致性。
      2. 新建 MRC 并写入 data（cast 为 float32）。
      3. 将 header.origin 设置为 new_origin_xyz（x,y,z），并把 nxstart/nystart/nzstart 设为 0（表示从数据开头写）。
      4. 恢复原始 voxel_size、cella 与 mapc/mapr/maps，并更新头部统计信息。

    Caveats / 注意事项:
      - 函数假定 data 的形状与期望的头信息兼容；若不兼容，需要先 reshape/transpose 或调整头信息（例如 mapc/mapr/maps）。
      - 将 nxstart/nystart/nzstart 设为 0 是简化处理；若你需要保留原始 nxstart/nystart/nzstart，请读取并恢复它们。
      - 强制把数据转为 float32 可能改变原始数据类型与精度；如需保留 dtype，可读取 original_mrc.data.dtype 并据此处理。
    """
    # 读取原始文件的关键头信息以便复制
    with mrcfile.open(original_map_path, permissive=True) as original_mrc:
        original_voxel_size = original_mrc.voxel_size.copy()
        original_cella = original_mrc.header.cella.copy()
        original_map_axes = (original_mrc.header.mapc, original_mrc.header.mapr, original_mrc.header.maps)
    # 创建新 MRC 并写入数据与头信息
    with mrcfile.new(file_path, overwrite=True) as mrc:
        # 写入数据，转换为 float32
        mrc.set_data(data.astype(np.float32))
        # 设置新的 origin（header 中保存为 x,y,z）
        mrc.header.origin.x, mrc.header.origin.y, mrc.header.origin.z = new_origin_xyz
        # 将起始索引设置为 0（表示数据从文件起始处写入）
        mrc.header.nxstart, mrc.header.nystart, mrc.header.nzstart = 0, 0, 0
        # 恢复原始 voxel size 与 cella
        mrc.voxel_size = original_voxel_size
        mrc.header.cella = original_cella
        # 恢复原始轴映射信息，确保其他软件读取时轴含义一致
        mrc.header.mapc, mrc.header.mapr, mrc.header.maps = original_map_axes
        # 更新头部统计（min/max/mean 等）以保持一致性
        mrc.update_header_stats()
def clean_temp_file(temp_file_path):  # str | list[str]
    """
    删除文件夹或文件夹列表 temp_file_path 里面的所有临时文件。
    这里认为临时文件是开头为'temp'的文件, 文件名形如 tmpXXXXXX.*
    """
    if isinstance(temp_file_path, str):
        temp_file_path = [temp_file_path]

    total_removed = 0
    for folder in temp_file_path:
        if not os.path.isdir(folder):
            print(f"[clean_temp_file] 跳过不存在的目录: {folder}")
            continue
        for fname in os.listdir(folder):
            # tempfile.mkstemp 生成的文件名以 'tmp' 开头, 后缀为 .npy 或 .npz
            if fname.startswith('tmp'):
                full_path = os.path.join(folder, fname)
                try:
                    os.remove(full_path) 
                    total_removed += 1
                except Exception as e:
                    print(f"[clean_temp_file] 删除失败: {full_path}, 原因: {e}")
    print(f"[clean_temp_file] 共删除 {total_removed} 个临时文件")



def apply_sharding(item_list, part_id, total_parts):
    """
    将 item_list 按 part_id / total_parts 切片, 返回当前分片.
    """
    if total_parts <= 1:
        return item_list
    n = len(item_list)
    shard_size = n // total_parts
    remainder = n % total_parts
    if part_id < remainder:
        start = part_id * (shard_size + 1)
        end = start + shard_size + 1
    else:
        start = remainder * (shard_size + 1) + (part_id - remainder) * shard_size
        end = start + shard_size
    return item_list[start:end]