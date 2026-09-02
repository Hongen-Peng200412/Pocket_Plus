# -*- coding: utf-8 -*-
"""用无 padding 80³ 滑窗生成一个 PDB 的完整图配体概率.

主要入口 :class:`FullMapInferenceSession` 连续处理多个 PDB, 复用 CPU 请求
物化线程, 并在当前 PDB 的窗口提交完毕后预取下一 PDB 的几何与首批输入.
兼容入口 :func:`infer_full_map` 只处理一个 PDB. 两个入口都返回
:class:`FullMapResult`, 不直接创建目录或落盘文件.

窗口按完整图 ZYX 起点的字典序前向. 重叠体素使用规范化三维 Gaussian 权重
以 float32 固定顺序融合, 因而跨 PDB 预取不会改变窗口边界或概率累加顺序.
"""

from __future__ import annotations

from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
import time
from typing import Any, Sequence

import numpy as np
import torch


@dataclass(frozen=True)
class FullMapResult:
    """保存一个 PDB 的完整图概率, 世界几何和执行计时.

    形状符号:
        - D: 完整图 Z 轴体素数.
        - H: 完整图 Y 轴体素数.
        - W: 完整图 X 轴体素数.

    字段:
        - probability_map: float32, `(D, H, W)`, 完整图每个 ZYX 体素的融合配体概率.
        - origin_xyz: float32, `(3,)`, 完整图索引 `(z=0, y=0, x=0)` 对应的世界 XYZ 角点坐标, 单位 Å.
        - voxel_size_xyz: float32, `(3,)`, 完整图 X, Y, Z 三轴的体素尺寸, 单位 Å/voxel.
        - window_count: int, 实际执行的无 padding 80³ 窗口数.
        - wall_seconds: float, 当前 PDB 完整图调用的墙钟秒数.
        - materialize_wait_seconds: float, GPU 调用线程等待 CPU 滑窗物化 Future 的累计秒数.
        - fusion_wait_seconds: float, GPU 调用线程等待有序 CPU 融合 Future 的累计秒数.
    """

    probability_map: np.ndarray
    origin_xyz: np.ndarray
    voxel_size_xyz: np.ndarray
    window_count: int
    wall_seconds: float
    materialize_wait_seconds: float
    fusion_wait_seconds: float


@dataclass(frozen=True)
class _MaterializedFullMapBatch:
    """保存一批已物化滑窗和逐窗完整图起点.

    形状符号:
        - B_window: 当前模型批次中的滑窗数, 最后一批可以小于 `window_batch_size`.
        - C_density: 当前 producer 的 Stage1 密度输入通道数.
        - N_A: Find producer 当前批次拼接后的受体原子总数.

    字段:
        - batch: ``dict[str, Any]``, `Stage1BatchCollator` 生成的模型批次. 下列身份字段和带 B_window 轴的张量与 `starts_zyx` 逐窗口对齐; Find 拼接原子字段通过 `atom_offsets` 和 `atom_batch_index` 映射回窗口.
            - batch.pdb_id: 长度 B_window 的字符串列表, 每项均为当前 PDB 标识.
            - batch.request_role: 长度 B_window 的字符串列表, 每项均为 `sliding`.
            - batch.occurrence_id: 长度 B_window 的列表, 滑窗请求的每项均为 None.
            - batch.candidate_index: 长度 B_window 的列表, 滑窗请求的每项均为 None.
            - batch.box_start_zyx: int32 张量, ``(B_window, 3)``, 每个 80³ 窗口的完整图 ZYX 起点, 单位 voxel.
            - batch.box_shape_zyx: int64 张量, ``(B_window, 3)``, 每个窗口沿 ZYX 三轴的体素数, 固定为 ``(80, 80, 80)``.
            - batch.box_origin_world: float32 张量, ``(B_window, 3)``, 每个窗口零索引体素网格角点对应的世界 XYZ 坐标, 单位 Å.
            - batch.voxel_size_world: float32 张量, ``(B_window, 3)``, 每个窗口沿世界 XYZ 三轴的体素间距, 单位 Å/voxel.
            - batch.density_input: float32 张量, ``(B_window, C_density, 80, 80, 80)``, 通道顺序由 `Stage1Dataset.resolved_density_channels` 冻结, 空间维按局部 ZYX 排列.
            - batch.hardmask: bool 张量, ``(B_window, 80, 80, 80)``, True 表示受体占据体素, 空间维按局部 ZYX 排列.
            - batch.atom_counts: Find 专用 int64 张量, ``(B_window,)``, 每个窗口的受体原子数, 并同步切分拼接后的原子字段.
            - batch.atom_offsets: Find 专用 int64 张量, ``(B_window + 1,)``, 同步切分全部拼接原子字段的第一维, 首值为 0, 末值为 N_A.
            - batch.atom_batch_index: Find 专用 int64 张量, ``(N_A,)``, 每个拼接受体原子所属的窗口编号, 数值位于 ``[0, B_window)``.
            - batch.atom_global_indices: Find 专用 int64 张量, ``(N_A,)``, 每个受体原子在当前 PDB `receptor_tokens.npz` 原子轴上的编号.
            - batch.atom_feat: Find 专用 float32 张量, ``(N_A, 49)``, 每个受体原子的基础特征.
            - batch.atom_is_backbone: Find 专用 bool 张量, ``(N_A,)``, True 表示对应受体原子属于主链.
            - batch.atom_coord_world: Find 专用 float32 张量, ``(N_A, 3)``, 受体原子世界 XYZ 坐标, 单位 Å.
            - batch.atom_coord_local_voxel: Find 专用 float32 张量, ``(N_A, 3)``, 受体原子相对各自窗口起点的 XYZ 体素坐标, 单位 voxel.
            - batch.atom_coord_centered_world: Find 专用 float32 张量, ``(N_A, 3)``, 受体原子相对各自窗口中心的世界 XYZ 位移, 单位 Å.
            - batch.atom_is_in_core_box: Find 专用 bool 张量, ``(N_A,)``, True 表示对应受体原子位于其 80³ 窗口的核心范围内.
        - starts_zyx: 长度 `B_window` 的 ZYX 整数起点元组; 第 i 项定位 `batch` 中第 i 个 80³ 滑窗在完整图内的位置.
    """

    batch: dict[str, Any]
    starts_zyx: tuple[tuple[int, int, int], ...]


@dataclass(frozen=True)
class _PreparedFullMapPdb:
    """保存一个 PDB 的固定几何, 窗口分批和首批物化任务.

    形状符号:
        - D, H, W: 完整图 Z, Y, X 三轴体素数.
        - N_batch: 当前 PDB 的窗口批次数.

    字段:
        - full_shape_zyx: 三个整数 `(D, H, W)`, 完整图 ZYX 形状.
        - voxel_size_xyz: float32, `(3,)`, 世界 X, Y, Z 三轴体素尺寸, 单位 Å/voxel.
        - origin_xyz: float32, `(3,)`, 完整图零索引体素的世界 XYZ 角点坐标, 单位 Å.
        - start_batches: 长度 `N_batch` 的元组; 每项是一个模型批次的完整图 ZYX 起点元组, 所有起点保持全局字典序.
        - window_count: int, 当前 PDB 的滑窗总数.
        - first_batch: `Future[_MaterializedFullMapBatch]`, `start_batches[0]` 对应的异步物化结果.
    """

    full_shape_zyx: tuple[int, int, int]
    voxel_size_xyz: np.ndarray
    origin_xyz: np.ndarray
    start_batches: tuple[tuple[tuple[int, int, int], ...], ...]
    window_count: int
    first_batch: Future[_MaterializedFullMapBatch]


def window_starts_zyx(
    full_shape_zyx: Sequence[int],
    window_shape_zyx: Sequence[int],
    stride_zyx: Sequence[int],
) -> tuple[tuple[int, int, int], ...]:
    """按 Z, Y, X 字典序生成覆盖完整图边界的无 padding 窗口起点.

    形状符号:
        - D, H, W: 完整图 Z, Y, X 三轴体素数.
        - D_window, H_window, W_window: 单个窗口 Z, Y, X 三轴体素数.
        - N_window: 三轴起点数量乘积, 即返回的窗口总数.

    输入参数:
        - full_shape_zyx: 三个整数 `(D, H, W)`, 完整图 ZYX 形状.
        - window_shape_zyx: 三个整数 `(D_window, H_window, W_window)`, 无 padding 滑窗形状.
        - stride_zyx: 三个正整数, 相邻窗口在 Z, Y, X 三轴的步长, 单位 voxel.

    返回值:
        - starts: 长度 `N_window` 的 ZYX 整数起点元组; 每项定位一个窗口在完整图中的半开区间起点.

    首项为 `(0, 0, 0)`. 每轴末项强制等于 `full_length - window_length`.
    """

    # 三个整数, 分别保存完整图形状, 窗口形状和 ZYX 步长; 输入容器类型在此统一为元组.
    full_shape = tuple(int(value) for value in full_shape_zyx)
    window_shape = tuple(int(value) for value in window_shape_zyx)
    stride = tuple(int(value) for value in stride_zyx)
    # 长度 3 的列表; 第 i 项是对应 Z, Y 或 X 轴上按升序排列的全部合法窗口起点.
    axes: list[tuple[int, ...]] = []
    for length, window, step in zip(full_shape, window_shape, stride):
        if length < window or step <= 0:
            raise ValueError(
                f"完整图轴长度必须不小于窗口且 stride 为正: {(length, window, step)}."
            )
        # 当前轴的整数起点列表; 每个起点都保证长度为 `window` 的半开区间不超出完整图.
        values = list(range(0, length - window + 1, step))
        if values[-1] != length - window:
            values.append(length - window)
        axes.append(tuple(values))
    # 长度 `N_window` 的 ZYX 起点元组; 三重笛卡尔积保留 Z 外层, Y 中层, X 内层的字典序.
    return tuple((z, y, x) for z in axes[0] for y in axes[1] for x in axes[2])


def gaussian_window_weight(
    window_shape_zyx: Sequence[int],
    sigma: float,
) -> np.ndarray:
    """计算规范化 ZYX 坐标上的 float32 三维 Gaussian 融合权重.

    形状符号:
        - D_window, H_window, W_window: 当前滑窗 Z, Y, X 三轴体素数.

    输入参数:
        - window_shape_zyx: 三个整数 `(D_window, H_window, W_window)`, 滑窗 ZYX 形状.
        - sigma: float, 规范化坐标中的标准差; 每轴从 -1 到 1 均匀取样, 因此正式值 0.5 不表示 40 个体素.

    返回值:
        - weight: float32, `(D_window, H_window, W_window)`, 与滑窗 ZYX 体素逐项对齐的正融合权重.
    """

    # 三个整数, 当前滑窗的 ZYX 形状.
    shape = tuple(int(value) for value in window_shape_zyx)
    # 长度 3 的列表; 每项是对应 Z, Y 或 X 轴从 -1 到 1 的 float32 规范化坐标.
    axes = [np.linspace(-1.0, 1.0, length, dtype=np.float32) for length in shape]
    # 三个 `(D_window, H_window, W_window)` 网格, 分别保存每个体素的规范化 Z, Y, X 坐标.
    z, y, x = np.meshgrid(*axes, indexing="ij")
    # `(D_window, H_window, W_window)`, 每个滑窗体素到规范化窗口中心的平方半径.
    squared_radius = z * z + y * y + x * x
    # float32, `(D_window, H_window, W_window)`, 未做总和归一化的 Gaussian 正权重; 后续以 `weighted_sum / weight_sum` 归一化重叠体素.
    return np.exp(-squared_radius / np.float32(2.0 * float(sigma) ** 2)).astype(
        np.float32, copy=False
    )


# ================================================================================================


class FullMapInferenceSession:
    """复用完整图滑窗物化线程并预取下一 PDB 的首批输入.

    形状符号:
        - B_window: 当前模型批次的滑窗数.
        - D, H, W: 当前 PDB 完整图 Z, Y, X 三轴体素数.

    :meth:`infer` 是主要入口. 当前 PDB 的全部窗口 batch 提交后,
    `next_pdb_id` 的完整图几何与首个 batch 排在同一物化队列末尾, 因而会与
    当前 PDB 的 GPU 前向和概率发布重叠, 但不会越过尚未提交的当前窗口.

    初始化参数:
        - dataset: Stage1Dataset, `full_map_context(pdb_id)` 返回完整图形状和世界几何, `materialize_request(request)` 物化一个 80³ 滑窗.
        - collator: callable, 把长度 `B_window` 的物化样本拼成 Stage1 模型批次, 并保持输入顺序.
        - wrapper: Stage1 模型包装器, `forward_voxel_probability(batch)` 返回 `(B_window, 1, 80, 80, 80)` 配体体素 logits.
        - device: str, 模型批次和 wrapper 前向使用的设备, 如 `cuda:0` 或 `cpu`.
        - stride_zyx: 三个正整数, 完整图 Z, Y, X 三轴滑窗步长, 单位 voxel.
        - sigma: float, 规范化 Gaussian 融合权重的标准差.
        - window_batch_size: int, 一次模型前向的滑窗数.
        - window_workers: int, 跨 PDB 复用的 CPU 请求物化线程数.
        - prefetch_batches: int, 当前 PDB 在 GPU 前方保留的有序批次数.
        - precision: str, CUDA autocast 精度.
        - pending_fusion_batches: int, GPU 后方保留的待融合批次数.

    副作用:
        - 构造时创建一个 CPU 请求物化线程池, 上下文退出时等待已提交任务并关闭线程池.
        - `infer(pdb_id, next_pdb_id)` 可以把 `next_pdb_id` 的几何和首批物化 Future 写入会话缓存, 但不为该 PDB 执行模型前向.
    """

    def __init__(
        self,
        dataset: Any,
        collator: Any,
        wrapper: Any,
        device: str,
        stride_zyx: Sequence[int],
        sigma: float,
        window_batch_size: int,
        window_workers: int,
        prefetch_batches: int,
        precision: str,
        pending_fusion_batches: int,
    ) -> None:
        """保存完整图推理依赖和有界队列参数, 并创建共享物化线程池.

        输入参数:
            - dataset: Stage1Dataset, 提供完整图几何和 80³ 滑窗样本物化.
            - collator: callable, 把一批保持窗口顺序的样本拼成 Stage1 模型字段映射.
            - wrapper: Stage1 模型包装器, 通过 `forward_voxel_probability` 输出逐窗口配体 logits.
            - device: 字符串, Stage1 模型前向使用的 PyTorch 设备, 如 `cuda:0`.
            - stride_zyx: 三个正整数, 相邻完整图窗口沿 Z/Y/X 三轴的步长, 单位 voxel.
            - sigma: float, 每个 80³ 窗口在规范化坐标 ``[-1, 1]`` 上的 Gaussian 融合标准差.
            - window_batch_size: int, 单次 Stage1 前向最多包含的滑窗数.
            - window_workers: int, 跨 PDB 共享的 CPU 样本物化线程数.
            - prefetch_batches: int, GPU 前方最多保留的有序滑窗批次数.
            - precision: 字符串, CUDA autocast 精度, `bf16` 或 `float16` 启用混合精度.
            - pending_fusion_batches: int, GPU 后方最多保留的待 D2H 或待融合批次数.

        副作用:
            - 创建最多使用 `window_workers` 个 CPU 线程的物化线程池, 生命周期由 :meth:`__exit__` 收口.
            - 不读取 PDB 数据, 不执行模型前向, 不创建输出目录或文件.
        """

        # Stage1Dataset 依赖, 为每个 PDB 提供完整图几何并物化 80³ 滑窗样本.
        self.dataset = dataset
        # callable 依赖, 把按窗口排序的样本序列拼成模型批次并保持第一维顺序.
        self.collator = collator
        # Stage1 wrapper 依赖, 在 batch_device 上输出每个滑窗的配体体素 logits.
        self.wrapper = wrapper
        # 三个整数, 完整图 Z, Y, X 三轴滑窗步长, 单位 voxel.
        self.stride_zyx = tuple(int(value) for value in stride_zyx)
        # float 标量, 规范化三维 Gaussian 融合权重的标准差.
        self.sigma = float(sigma)
        # 正整数, 单次 Stage1 前向最多包含的完整图滑窗数.
        self.window_batch_size = int(window_batch_size)
        # 正整数, GPU 前方最多保留的有序滑窗批次数.
        self.prefetch_batches = int(prefetch_batches)
        # 字符串, CUDA autocast 使用的精度名称; `bf16` 和 `float16` 以外的值关闭 autocast.
        self.precision = str(precision)
        # 正整数, GPU 后方最多保留的待 D2H 或待有序融合批次数.
        self.pending_fusion_batches = int(pending_fusion_batches)
        # 当前模型批次的显式 PyTorch 设备; CUDA 路径还据此启用锁页内存和异步传输.
        self.batch_device = torch.device(device)
        # 跨 PDB 共享且最多使用 window_workers 个线程的 CPU 线程池; 任务依次是几何准备和单个滑窗批次物化.
        self.materializer = ThreadPoolExecutor(
            max_workers=int(window_workers),
            thread_name_prefix="stage1-window",
        )
        # PDB 标识到 `_PreparedFullMapPdb` Future 的映射; 同一待推理 PDB 的几何和首批输入最多提交一次.
        self.prepared_pdbs: dict[str, Future[_PreparedFullMapPdb]] = {}

    def __enter__(self) -> FullMapInferenceSession:
        """进入跨 PDB 完整图推理上下文并返回当前会话.

        返回值:
            - session: :class:`FullMapInferenceSession`, 与当前实例相同, 可连续调用 :meth:`infer` 并复用物化线程池.

        本方法不创建额外线程, 不读取 PDB 数据, 也不执行模型前向.
        """

        return self

    def __exit__(self, *_: object) -> None:
        """退出完整图推理上下文并关闭跨 PDB 共享物化线程池.

        输入参数:
            - *_: 上下文管理器传入的异常类型, 异常实例和 traceback; 本方法不改变异常传播语义.

        副作用:
            - 等待已经提交的 PDB 准备和滑窗物化任务结束, 取消尚未开始的 Future, 然后关闭 `materializer`.
        """

        self.materializer.shutdown(wait=True, cancel_futures=True)

    def _materialize_batch(
        self,
        pdb_id: str,
        batch_starts: Sequence[tuple[int, int, int]],
    ) -> _MaterializedFullMapBatch:
        """物化一批有序滑窗, 并在 CUDA 模式下锁定模型张量的 CPU 内存.

        形状符号:
            - B_window: `batch_starts` 中的滑窗数.
            - C_density: 当前 producer 的 Stage1 密度输入通道数.

        输入参数:
            - pdb_id: 字符串, 当前批次全部滑窗所属的 PDB 标识, 如 `1abc`.
            - batch_starts: 长度 B_window 的完整图 ZYX 整数起点序列, 单位 voxel; 第 i 项决定返回 `batch` 第 i 个窗口的位置.

        返回值:
            - materialized: :class:`_MaterializedFullMapBatch`, 保存 collator 模型批次和不变的窗口起点顺序.
            - materialized.batch: ``dict[str, Any]``, `Stage1BatchCollator` 定义的字段映射; `density_input` 形状为 ``(B_window, C_density, 80, 80, 80)``, 其他字段见 :class:`_MaterializedFullMapBatch`.
            - materialized.starts_zyx: 长度 B_window 的 ZYX 整数起点元组, 与 `batch` 的窗口轴逐项对齐.

        副作用:
            - 对每个起点调用一次 `dataset.materialize_request`; CUDA 模式下把返回批次中的 CPU 张量转换为锁页张量, 供主线程异步 H2D.
            - 不访问 GPU, 不执行模型前向, 不写文件.
        """
        from src.datasets.stage1_requests import ResolvedStage1Crop

        # 长度 B_window 的 ResolvedStage1Crop 列表, 第 i 项以 batch_starts[i] 为完整图 ZYX 起点并固定 role=`sliding`, 不请求训练目标.
        requests = [
            ResolvedStage1Crop(
                pdb_id=pdb_id,
                box_start_zyx=start,
                require_targets=False,
                role="sliding",
            )
            for start in batch_starts
        ]
        # 长度 B_window 的 Stage1 单窗口字段映射列表, 第 i 项由 requests[i] 物化并保持完整图窗口顺序.
        samples = [self.dataset.materialize_request(request) for request in requests]
        # dict[str, Any], Stage1 模型字段映射; density_input 为 (B_window, C_density, 80, 80, 80), 身份字段和固定形状张量的 B_window 轴与窗口起点直接对齐, Find 拼接原子字段通过 atom_offsets 和 atom_batch_index 映射回窗口.
        batch = self.collator(samples)
        if self.batch_device.type == "cuda":
            # 身份字段和固定形状张量的 B_window 轴与 starts_zyx 直接对齐; Find 拼接原子字段仍通过 atom_offsets 和 atom_batch_index 映射. CPU 张量转换为锁页内存以支持主线程异步 H2D, Python 列表等非张量字段保持原值.
            batch = {
                name: value.pin_memory() if torch.is_tensor(value) else value
                for name, value in batch.items()
            }
        return _MaterializedFullMapBatch(batch=batch, starts_zyx=tuple(batch_starts))

    def _prepare_pdb(self, pdb_id: str) -> _PreparedFullMapPdb:
        """读取一个 PDB 的完整图几何, 冻结窗口顺序并提交首个 batch.

        形状符号:
            - D, H, W: 当前 PDB 完整图沿 ZYX 三轴的体素数.
            - N_window: 以 80³ 无 padding 滑窗覆盖完整图所需的窗口总数.
            - N_batch: 把 N_window 个窗口按 `window_batch_size` 顺序分组后的批次数.
            - B_first: 首个窗口批次的窗口数, 满足 ``1 <= B_first <= window_batch_size``.

        输入参数:
            - pdb_id: 字符串, 待准备完整图推理的 PDB 标识, 如 `1abc`.

        返回值:
            - prepared_pdb: :class:`_PreparedFullMapPdb`, 保存当前 PDB 的固定几何, 全部窗口批次和首批物化 Future.
            - prepared_pdb.full_shape_zyx: 三个整数 ``(D, H, W)``, 完整图沿 ZYX 三轴的体素数.
            - prepared_pdb.voxel_size_xyz: float32, ``(3,)``, 完整图沿世界 XYZ 三轴的体素间距, 单位 Å/voxel.
            - prepared_pdb.origin_xyz: float32, ``(3,)``, 完整图零索引体素网格角点对应的世界 XYZ 坐标, 单位 Å.
            - prepared_pdb.start_batches: 长度 N_batch 的嵌套元组, 第 j 项含第 j 个模型批次的完整图 ZYX 整数起点, 展平后保持全局字典序.
            - prepared_pdb.window_count: int, 当前 PDB 的窗口总数 N_window.
            - prepared_pdb.first_batch: ``Future[_MaterializedFullMapBatch]``, 长度 B_first 的 `start_batches[0]` 对应物化任务.

        副作用:
            - 在线程池工作线程中调用一次 `dataset.full_map_context`, 并向同一 `materializer` 提交首个滑窗批次的物化任务.
            - 不执行 wrapper 前向, 不构造完整图概率数组, 不写文件.
        """

        # `full_shape` 为三个整数 (D, H, W), 表示完整图 ZYX 体素数; `voxel_size` 为 (3,) 世界 XYZ 体素间距, 单位 Å/voxel; `origin` 为 (3,) 零索引体素网格角点的世界 XYZ 坐标, 单位 Å.
        full_shape, voxel_size, origin, _ = self.dataset.full_map_context(pdb_id)
        # 长度 N_window 的完整图 ZYX 整数起点元组, 单位 voxel; 起点按 Z 外层, Y 中层, X 内层的字典序排列, 每轴末窗贴住完整图边界.
        starts = window_starts_zyx(full_shape, (80, 80, 80), self.stride_zyx)
        # 长度 N_batch 的嵌套元组, 每项含不超过 window_batch_size 个连续窗口起点; 顺序展平后与 starts 完全一致.
        start_batches = tuple(
            starts[offset : offset + self.window_batch_size]
            for offset in range(0, len(starts), self.window_batch_size)
        )
        return _PreparedFullMapPdb(
            full_shape_zyx=tuple(int(value) for value in full_shape),
            voxel_size_xyz=np.asarray(voxel_size, dtype=np.float32),
            origin_xyz=np.asarray(origin, dtype=np.float32),
            start_batches=start_batches,
            window_count=len(starts),
            first_batch=self.materializer.submit(
                self._materialize_batch,
                pdb_id,
                start_batches[0],
            ),
        )

    def _schedule_pdb(self, pdb_id: str) -> Future[_PreparedFullMapPdb]:
        """保证指定 PDB 的几何和首批物化最多提交一次.

        输入参数:
            - pdb_id: 字符串, 当前或下一个待推理 PDB 标识, 如 `1abc`; 字符串值作为会话缓存键.

        返回值:
            - prepared_future: ``Future[_PreparedFullMapPdb]``, 完成后得到 :meth:`_prepare_pdb` 定义的完整图几何, 有序窗口批次和首批物化 Future; 同一缓存键的重复调用返回同一对象.

        副作用:
            - 首次遇到该 PDB 标识时向共享 `materializer` 提交一次 :meth:`_prepare_pdb`, 并把 Future 写入 `prepared_pdbs`.
            - 重复调用不新增任务; 当前 PDB 进入 :meth:`infer` 后才从缓存移除.
        """

        # 字符串, 当前 PDB 在 prepared_pdbs 中的规范缓存键; 不改变调用者提供的大小写或内容.
        identity = str(pdb_id)
        if identity not in self.prepared_pdbs:
            self.prepared_pdbs[identity] = self.materializer.submit(
                self._prepare_pdb,
                identity,
            )
        return self.prepared_pdbs[identity]

    def infer(self, pdb_id: str, next_pdb_id: str | None) -> FullMapResult:
        """用固定窗口顺序生成一个 PDB 的完整图配体概率.

        形状符号:
            - B_window: 当前模型批次的滑窗数.
            - D, H, W: 当前 PDB 完整图 Z, Y, X 三轴体素数.
            - N_window: 覆盖当前完整图并实际执行模型前向的 80³ 窗口总数.
            - N_batch: N_window 个窗口按 `window_batch_size` 顺序组成的模型批次数.

        输入参数:
            - pdb_id: 字符串, 当前需要生成完整图配体概率的 PDB 标识, 如 `1abc`.
            - next_pdb_id: 字符串或 None, 当前 probability 清单中下一个实际需要推理的 PDB 标识; 清单末项传入 None.

        返回值:
            - result: :class:`FullMapResult`, 当前 PDB 的完整图概率, 世界几何和执行计时.
            - result.probability_map: float32, ``(D, H, W)``, 所有 80³ 窗口按固定顺序 Gaussian 融合后的配体概率, 三维按完整图 ZYX 排列.
            - result.origin_xyz: float32, ``(3,)``, 完整图零索引体素网格角点对应的世界 XYZ 坐标, 单位 Å.
            - result.voxel_size_xyz: float32, ``(3,)``, 完整图沿世界 XYZ 三轴的体素间距, 单位 Å/voxel.
            - result.window_count: int, 当前 PDB 实际执行 Stage1 前向的窗口数 N_window.
            - result.wall_seconds: float, 从进入本方法到完成完整图融合的墙钟秒数.
            - result.materialize_wait_seconds: float, GPU 调用线程等待当前 PDB 滑窗物化 Future 的累计秒数.
            - result.fusion_wait_seconds: float, GPU 调用线程等待有序 D2H 和 Gaussian 融合 Future 的累计秒数.

        执行阶段:
            - CPU 物化线程按窗口字典序准备有界批次, GPU 调用线程按同一顺序执行 wrapper 前向.
            - CUDA 路径把每批 float32 概率异步复制到锁页 CPU 张量, 单线程融合器按提交顺序累加 Gaussian 加权概率.
            - 当前 PDB 的全部窗口物化任务提交后, `next_pdb_id` 的几何和首批输入进入共享物化队列, 但下一 PDB 的模型前向仍由下一次 `infer` 调用执行.

        失败语义:
            - 任一完整图体素没有获得正 Gaussian 累计权重时抛出 RuntimeError, 不返回未完整覆盖的概率图.

        副作用:
            - 从 `prepared_pdbs` 取出并移除当前 PDB 的准备 Future, 向共享物化线程池提交其余窗口批次, 并创建当前 PDB 专用的单线程融合器.
            - `next_pdb_id` 非 None 时可能向 `prepared_pdbs` 写入下一 PDB 的准备 Future.
            - 在 `batch_device` 上执行 Stage1 前向, 但不创建目录, 不写概率图文件, 不修改 Dataset 源数据.
        """

        # float 标量, 当前 PDB 完整图推理的高精度单调时钟起点, 用于计算 wall_seconds.
        started_at = time.perf_counter()
        # 两个非负 float 秒数标量, 分别累计主线程等待窗口物化 Future 和有序融合 Future 的时间.
        materialize_wait_seconds = 0.0
        fusion_wait_seconds = 0.0
        # 当前 PDB 的固定几何, 有序窗口批次和首批物化 Future; 预取命中时 `.result()` 可以直接返回.
        prepared_pdb = self._schedule_pdb(pdb_id).result()
        # 当前 PDB 准备 Future 已被本次 infer 消费, 从缓存移除以避免会话长期持有完成对象.
        self.prepared_pdbs.pop(str(pdb_id), None)
        # float32, (80, 80, 80), 在规范化坐标上计算的 Gaussian 正权重, 与每个窗口的局部 ZYX 体素逐项对齐; 权重和由后续 weight_sum 逐体素归一化.
        weight = gaussian_window_weight((80, 80, 80), self.sigma)
        # 两个 float32 `(D, H, W)` 完整图数组; 分别累计窗口概率乘权重和窗口权重, 体素轴按 ZYX 排列.
        weighted_sum = np.zeros(prepared_pdb.full_shape_zyx, dtype=np.float32)
        weight_sum = np.zeros(prepared_pdb.full_shape_zyx, dtype=np.float32)

        def fuse_batch(
            host_probability: torch.Tensor,
            ready_event: torch.cuda.Event | None,
            batch_starts: tuple[tuple[int, int, int], ...],
        ) -> None:
            """等待一批 D2H 完成, 再按窗口顺序原位累加概率与权重.

            形状符号:
                - B_window: 当前待融合批次的滑窗数.
                - D, H, W: 外层 `weighted_sum` 和 `weight_sum` 沿完整图 ZYX 三轴的体素数.

            输入参数:
                - host_probability: float32 CPU 张量, ``(B_window, 80, 80, 80)``, 当前模型批次每个滑窗的 sigmoid 配体概率, 后三维按局部 ZYX 排列.
                - ready_event: CUDA event 或 None, CUDA 路径标记 `host_probability` 的异步 D2H 已完成, CPU 前向传入 None.
                - batch_starts: 长度 B_window 的完整图 ZYX 整数起点元组, 单位 voxel, 与 `host_probability` 第一维逐窗口对齐.

            闭包数组:
                - weight: float32, ``(80, 80, 80)``, 每个窗口共同使用的规范化坐标 Gaussian 正权重, 三维按局部 ZYX 排列.
                - weighted_sum: float32, ``(D, H, W)``, 已融合窗口的 ``probability * weight`` 累计和, 三维按完整图 ZYX 排列.
                - weight_sum: float32, ``(D, H, W)``, 已融合窗口的 Gaussian 权重累计和, 与 `weighted_sum` 逐体素对齐.

            副作用:
                - `ready_event` 非 None 时先阻塞当前融合线程, 直到异步 D2H 完成.
                - 按 `batch_starts` 顺序原位更新 `weighted_sum` 和 `weight_sum`, 不返回新数组, 不写文件.
            """

            if ready_event is not None:
                ready_event.synchronize()
            # float32, (B_window, 80, 80, 80), 与 batch_starts 逐窗口对齐的 CPU NumPy 概率视图; 后三维按局部 ZYX 排列.
            probabilities = host_probability.numpy()
            for probability, (z, y, x) in zip(probabilities, batch_starts):
                # 三个 slice, 分别定位当前 80³ 窗口在完整图 Z/Y/X 轴上的半开区间, 每个区间长度为 80.
                slices = (
                    slice(z, z + 80),
                    slice(y, y + 80),
                    slice(x, x + 80),
                )
                # 当前 float32 (80, 80, 80) 窗口概率逐体素乘 Gaussian 权重后累加到 weighted_sum, 同一切片的 weight 同步累加到 weight_sum. [80, 80, 80] * [80, 80, 80] -> [80, 80, 80]
                weighted_sum[slices] += probability * weight
                weight_sum[slices] += weight

        # 按窗口批次顺序保存 `_MaterializedFullMapBatch` Future; 首项对应 `start_batches[0]`.
        prepared: deque[Future[_MaterializedFullMapBatch]] = deque(
            [prepared_pdb.first_batch]
        )
        # 按 GPU 前向顺序保存融合 Future; 每项负责一批异步 D2H 等待和完整图原位累加.
        pending_fusions: deque[Future[None]] = deque()
        # 下一个尚未提交物化任务的窗口批次编号; `start_batches[0]` 已在 PDB 准备阶段提交.
        next_batch = 1
        while next_batch < min(
            self.prefetch_batches,
            len(prepared_pdb.start_batches),
        ):
            prepared.append(
                self.materializer.submit(
                    self._materialize_batch,
                    pdb_id,
                    prepared_pdb.start_batches[next_batch],
                )
            )
            next_batch += 1
        # True 表示 `next_pdb_id` 的几何和首批输入已经提交, 防止在当前窗口循环中重复预取.
        next_pdb_scheduled = False

        # 当前 PDB 专用的单线程融合器; Future 提交顺序就是 float32 Gaussian 累加顺序.
        fusion_executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="stage1-fusion",
        )
        try:
            with torch.inference_mode():
                while prepared:
                    wait_started_at = time.perf_counter()
                    # 当前最早窗口批次的物化结果; 身份字段和固定形状张量的 B_window 轴与 starts_zyx 直接对齐, Find 拼接原子字段通过 atom_offsets 和 atom_batch_index 映射回窗口.
                    materialized = prepared.popleft().result()
                    materialize_wait_seconds += time.perf_counter() - wait_started_at
                    if next_batch < len(prepared_pdb.start_batches):
                        prepared.append(
                            self.materializer.submit(
                                self._materialize_batch,
                                pdb_id,
                                prepared_pdb.start_batches[next_batch],
                            )
                        )
                        next_batch += 1
                    if (
                        next_pdb_id is not None
                        and not next_pdb_scheduled
                        and next_batch == len(prepared_pdb.start_batches)
                    ):
                        self._schedule_pdb(next_pdb_id)
                        next_pdb_scheduled = True

                    # 当前窗口批次的 Stage1 模型字段映射; 带批次轴的张量第一维为 `B_window`.
                    cpu_batch = materialized.batch
                    if self.batch_device.type == "cuda":
                        # 字段与 `cpu_batch` 相同; CPU 锁页张量异步复制到 `batch_device`, 非张量字段保持原值.
                        model_batch = {
                            key: (
                                value.to(self.batch_device, non_blocking=True)
                                if torch.is_tensor(value)
                                else value
                            )
                            for key, value in cpu_batch.items()
                        }
                        # CUDA autocast 计算类型; 只有 `precision` 为 `bf16` 或 `float16` 时启用.
                        autocast_dtype = (
                            torch.bfloat16
                            if self.precision == "bf16"
                            else torch.float16
                        )
                        with torch.autocast(
                            device_type="cuda",
                            dtype=autocast_dtype,
                            enabled=self.precision in {"bf16", "float16"},
                        ):
                            # `(B_window, 1, 80, 80, 80)`, 当前滑窗批次每个体素的配体 logit.
                            logits = self.wrapper.forward_voxel_probability(model_batch)
                        # float32, `(B_window, 80, 80, 80)`, 删除单通道轴后的配体概率. [B_window, 1, 80, 80, 80] -> [B_window, 80, 80, 80]
                        probability = torch.sigmoid(logits[:, 0]).to(torch.float32)
                        while len(pending_fusions) >= self.pending_fusion_batches:
                            wait_started_at = time.perf_counter()
                            pending_fusions.popleft().result()
                            fusion_wait_seconds += time.perf_counter() - wait_started_at
                        # 锁页 float32 CPU 张量, `(B_window, 80, 80, 80)`, 接收当前 CUDA 概率的异步 D2H 复制.
                        host_probability = torch.empty(
                            probability.shape,
                            dtype=torch.float32,
                            device="cpu",
                            pin_memory=True,
                        )
                        host_probability.copy_(probability, non_blocking=True)
                        # 当前 CUDA stream 的完成事件; 融合线程等待它以保证 `host_probability` 已可读.
                        event = torch.cuda.Event()
                        event.record(torch.cuda.current_stream(self.batch_device))
                        pending_fusions.append(
                            fusion_executor.submit(
                                fuse_batch,
                                host_probability,
                                event,
                                materialized.starts_zyx,
                            )
                        )
                    else:
                        # 字段与 `cpu_batch` 相同; 张量显式放到 CPU `batch_device`, 非张量字段保持原值.
                        model_batch = {
                            key: (
                                value.to(self.batch_device)
                                if torch.is_tensor(value)
                                else value
                            )
                            for key, value in cpu_batch.items()
                        }
                        # `(B_window, 1, 80, 80, 80)`, CPU 前向得到的配体体素 logit.
                        logits = self.wrapper.forward_voxel_probability(model_batch)
                        # float32, `(B_window, 80, 80, 80)`, 删除单通道轴后的 CPU 配体概率.
                        host_probability = torch.sigmoid(logits[:, 0]).to(
                            device="cpu",
                            dtype=torch.float32,
                        )
                        pending_fusions.append(
                            fusion_executor.submit(
                                fuse_batch,
                                host_probability,
                                None,
                                materialized.starts_zyx,
                            )
                        )
            while pending_fusions:
                wait_started_at = time.perf_counter()
                pending_fusions.popleft().result()
                fusion_wait_seconds += time.perf_counter() - wait_started_at
        finally:
            fusion_executor.shutdown(wait=True, cancel_futures=True)

        # 验证每个完整图 ZYX 体素都至少收到一个窗口的正 Gaussian 权重, 否则不能执行逐体素归一化.
        if np.any(weight_sum <= 0.0):
            raise RuntimeError(f"{pdb_id}: 滑窗没有覆盖完整图的全部体素.")
        # float32, `(D, H, W)`, 每个完整图 ZYX 体素的加权概率和除以累计 Gaussian 权重.
        probability_map = np.divide(
            weighted_sum,
            weight_sum,
            out=np.zeros_like(weighted_sum),
            where=weight_sum > 0.0,
        )
        return FullMapResult(
            probability_map=probability_map,
            origin_xyz=prepared_pdb.origin_xyz,
            voxel_size_xyz=prepared_pdb.voxel_size_xyz,
            window_count=prepared_pdb.window_count,
            wall_seconds=time.perf_counter() - started_at,
            materialize_wait_seconds=materialize_wait_seconds,
            fusion_wait_seconds=fusion_wait_seconds,
        )


def infer_full_map(
    dataset: Any,
    collator: Any,
    wrapper: Any,
    pdb_id: str,
    device: str,
    stride_zyx: Sequence[int],
    sigma: float,
    window_batch_size: int,
    window_workers: int,
    prefetch_batches: int,
    precision: str,
    pending_fusion_batches: int,
) -> FullMapResult:
    """生成单个 PDB 的完整图配体概率, 并在返回前关闭物化线程池.

    形状符号:
        - B_window: 当前模型批次中的滑窗数.
        - D, H, W: 当前 PDB 完整图 Z, Y, X 三轴体素数.

    输入参数:
        - dataset: Stage1Dataset, `full_map_context(pdb_id)` 返回完整图几何, `materialize_request(request)` 返回一个 80³ 滑窗样本.
        - collator: callable, 把长度 `B_window` 的有序滑窗样本拼成 Stage1 模型批次.
        - wrapper: Stage1 模型包装器, `forward_voxel_probability(batch)` 返回 `(B_window, 1, 80, 80, 80)` 配体 logits.
        - pdb_id: str, 当前完整图所属 PDB 标识.
        - device: str, 模型前向设备, 例如 `cuda:0` 或 `cpu`.
        - stride_zyx: Sequence[int], 完整图 ZYX 三轴无 padding 滑窗步长.
        - sigma: float, 规范化 Gaussian 融合权重的标准差.
        - window_batch_size: int, 一次模型前向的滑窗数.
        - window_workers: int, CPU 请求物化线程数.
        - prefetch_batches: int, GPU 前方最多保留的有序 batch 数.
        - precision: str, CUDA autocast 精度; `bf16`, `float16` 或关闭混合精度的其他值.
        - pending_fusion_batches: int, GPU 后方最多保留的待融合 batch 数.

    返回值:
        - result: `FullMapResult`, `probability_map` 是 float32 `(D, H, W)` 完整图配体概率, 并带世界几何, 窗口数量和执行计时.

    该入口保留单 PDB 调用兼容性. 多 PDB probability 阶段使用
    :class:`FullMapInferenceSession`, 以复用线程和预取下一 PDB 的首批输入.
    """

    with FullMapInferenceSession(
        dataset=dataset,
        collator=collator,
        wrapper=wrapper,
        device=device,
        stride_zyx=stride_zyx,
        sigma=sigma,
        window_batch_size=window_batch_size,
        window_workers=window_workers,
        prefetch_batches=prefetch_batches,
        precision=precision,
        pending_fusion_batches=pending_fusion_batches,
    ) as session:
        return session.infer(pdb_id, None)
