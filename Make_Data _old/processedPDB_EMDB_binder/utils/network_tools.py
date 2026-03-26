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
def map_segmentation(image, windows_size: int = 64, stride: int = 50):
    Z, Y, X = image.shape[-3:]
    blocks = []
    coords = []
    for ii in range(0, Z + stride, stride):
        for jj in range(0, Y + stride, stride):
            for kk in range(0, X + stride, stride):
                z0 = max(0, min(ii, Z - windows_size))
                y0 = max(0, min(jj, Y - windows_size))
                x0 = max(0, min(kk, X - windows_size))
                
                block = image[..., z0:min(ii + windows_size, Z), 
                              y0:min(jj + windows_size, Y),
                              x0:min(kk + windows_size, X)]
                blocks.append(block)
                coords.append((z0, y0, x0))
    return blocks, coords



def map_reconstruction(blocks, image_shape, coords=None, windows_size=64, stride=50):
    """
    修正版：支持 image_shape=(Z,Y,X) 或 (C,Z,Y,X)。
    关键修正：
      - 明确把 start clamp 到 >=0，end clamp 到 <=dim，避免负 start 导致的 zero-length 切片
      - 支持 blocks=(N, D,H,W) 或 (N, C, D,H,W)
      - 对 blocks 数量与预期循环次数不匹配时抛出错误（帮助定位 segmentation/stride/win_size 不一致）
    """
    blocks = np.asarray(blocks)

    # 如果 image_shape 有 4 个元素（C,Z,Y,X），逐类处理（递归）
    if len(image_shape) == 4:
        C, Z, Y, X = image_shape
        recon = np.zeros((C, Z, Y, X), dtype=np.float32)

        # 若 blocks 是 (N, C, D, H, W)，直接按通道拆分
        if blocks.ndim == 5 and blocks.shape[1] == C:
            for c in range(C):
                recon[c] = map_reconstruction(blocks[:, c, ...], (Z, Y, X), windows_size=windows_size, stride=stride)
            return recon
        else:
            # 还是尝试按 blocks[:, c, ...] 的方式调用（会在递归里抛错以便调试）
            for c in range(C):
                recon[c] = map_reconstruction(blocks[:, c, ...], (Z, Y, X), windows_size=windows_size, stride=stride)
            return recon

    # 现在 image_shape 应为 (Z,Y,X)
    Z, Y, X = image_shape
    reconstruction = np.zeros((Z, Y, X), dtype=np.float32)
    counts = np.zeros((Z, Y, X), dtype=np.float32)

    # 计算循环的“起点集合”数量（仅用于诊断），这里保持和你原来循环结构等价
    total_expected = 0
    ii_list = list(range(0, Z + stride, stride))
    jj_list = list(range(0, Y + stride, stride))
    kk_list = list(range(0, X + stride, stride))
    total_expected = len(ii_list) * len(jj_list) * len(kk_list)

    image_number = 0
    core_offset = 2  # 你原来的中心裁剪 offset
    # 三重循环（保持原来循环形式，但在索引时做严格裁剪）
    for ii in ii_list:
        for jj in jj_list:
            for kk in kk_list:
                # 原来的表达式可能产生负 start，例如 X - windows_size + 2 可能为负
                # 我们先计算未经截断的 start/end，再把它 clamp 到 [0, dim]
                raw_z0 = min(ii + core_offset, Z - windows_size + core_offset)
                raw_z1 = min(ii + windows_size - core_offset, Z - core_offset)
                raw_y0 = min(jj + core_offset, Y - windows_size + core_offset)
                raw_y1 = min(jj + windows_size - core_offset, Y - core_offset)
                raw_x0 = min(kk + core_offset, X - windows_size + core_offset)
                raw_x1 = min(kk + windows_size - core_offset, X - core_offset)

                # clamp 到有效范围，避免负 start 引发 Python 负索引问题
                z0 = max(0, int(raw_z0))
                z1 = max(0, int(raw_z1))
                y0 = max(0, int(raw_y0))
                y1 = max(0, int(raw_y1))
                x0 = max(0, int(raw_x0))
                x1 = max(0, int(raw_x1))

                tz = z1 - z0
                ty = y1 - y0
                tx = x1 - x0

                # 如果 target 区域为零，仍然要消费 blocks 的索引以保证顺序一致
                if tz <= 0 or ty <= 0 or tx <= 0:
                    image_number += 1
                    continue

                # 检查 blocks 是否足够
                if image_number >= getattr(blocks, "shape", (0,))[0]:
                    raise IndexError(
                        f"[map_reconstruction] blocks too short: need index {image_number}, but blocks.shape = {getattr(blocks, 'shape', None)}"
                    )

                # 取 block 并确保是 (D,H,W)
                block = blocks[image_number]
                if block.ndim == 4 and block.shape[0] == 1:
                    block = block[0]

                # center crop（若 block 太小就不裁剪）
                if block.shape[0] > 4 and block.shape[1] > 4 and block.shape[2] > 4:
                    cropped = block[core_offset:-core_offset, core_offset:-core_offset, core_offset:-core_offset]
                else:
                    cropped = block

                bz, by, bx = cropped.shape

                # 实际重叠尺寸（按 target 可放下的大小与 cropped 的大小取最小）
                ov_z = min(bz, tz)
                ov_y = min(by, ty)
                ov_x = min(bx, tx)

                if ov_z <= 0 or ov_y <= 0 or ov_x <= 0:
                    image_number += 1
                    continue

                # 注意：此处我们把 cropped 的前 ov_* 区域放到目标位置的左/前对齐区域
                # 目标切片为 [z0:z0+ov_z, y0:y0+ov_y, x0:x0+ov_x]
                reconstruction[z0:z0+ov_z, y0:y0+ov_y, x0:x0+ov_x] += cropped[:ov_z, :ov_y, :ov_x]
                counts[z0:z0+ov_z, y0:y0+ov_y, x0:x0+ov_x] += 1
                image_number += 1

    # 诊断提示：如果消费数目和循环迭代数不符，打印信息（可能是 segmentation 与 reconstruction 的起点不一致）
    if image_number != total_expected:
        print(f"[map_reconstruction] consumed patches {image_number}, loop iterations {total_expected}")

    # 平均化
    reconstruction = reconstruction / (counts + 1e-6)
    return reconstruction



    # ==========================
    # 使用示例（示意，不在函数内运行）：
    # 假设有一个体积 grid.shape = (120,120,120)，class_num = 4
    # blocks 的形状可能是 (N, 4, D, H, W) 或 (N, D, H, W)
    # 调用时：
    # recon = map_reconstruction(blocks, (4, 120, 120, 120), windows_size=64, stride=50)
    # 结果 recon.shape == (4, 120, 120, 120)
    # ==========================






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
    原子地保存 numpy .npy 文件，用法为：atomic_np_save(target_file, data_array)。
    如果 do_not_replace, 那么会在原文件已经存在时不替换.
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
    原子地保存 numpy .npz 文件,如果 do_not_replace, 那么会在原文件已经存在时不替换.
    用法为：
    atomic_np_savez(filename, array_a=data_a, array_b=data_b)
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
