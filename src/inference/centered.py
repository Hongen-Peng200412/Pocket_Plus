# -*- coding: utf-8 -*-
"""对 F-alpha blob 候选执行 80³ centered 前向并构造归档字段.

读者应先看 :func:`infer_centered_boxes`. 该入口按来源 blob 顺序编排 Dataset 物化, Stage1 GPU 前向, 异步 D2H 和 CPU 字段整理. :func:`pack_centered_entries` 再把逐候选字段映射压成共享 offsets 的 NumPy 数组.

本模块不直接写文件. :func:`infer_centered_boxes` 返回 centered NPZ 字段映射的 ``Future`` 和性能统计, 实际文件由 ``pipeline.run_centered_stage`` 写入 ``<output_root>/<producer>/<split>/<pdb_id>/centered/F<alpha>_centered.npz``. 本模块不拟合阈值, 不决定最终候选集合, 也不计算评估指标.
"""

from __future__ import annotations

from collections import deque
from concurrent.futures import Executor, Future, ThreadPoolExecutor
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.spatial import cKDTree
import torch


def pack_centered_entries(
    entries: Sequence[Mapping[str, np.ndarray]],
    producer: str,
) -> dict[str, np.ndarray]:
    """把同一 PDB 的逐候选字段映射压成无 object dtype 的归档数组.

    形状符号:
        - N_entry: 当前 PDB 中执行 centered 前向的来源 blob 数量, 可为零.
        - L_voxel: 全部 centered BOX 内来源体素的合计数量.
        - L_aux: 全部 centered BOX 内受体占据体素的合计数量.
        - N_A: Find producer 在全部 centered BOX 中保留的 A 原子合计数量.
        - N_P: Find producer 在全部 centered BOX 中保留的 P 锚点合计数量.
        - C_voxel: `voxel_final` 的 V 特征通道数.
        - C_A1, C_A2, C_A3: `A_feat_L1`, `A_feat_L2`, `A_feat_L3` 的特征通道数.
        - C_P2, C_P3: `P_feat_L2`, `P_feat_L3` 的特征通道数.

    输入参数:
        - entries: 长度 N_entry 的候选字段映射序列, 顺序与 alpha blobs 文件中的 `source_blob_index` 选择顺序一致; 每个映射包含下述同名输出字段在单个候选上的切片, 但不包含 offsets 和 `centered_box_index`.
        - producer: 字符串, Stage1 producer 名称; 以 `Find_` 开头时要求并归档 A/P 字段, 其余 producer 只归档共同字段.

    返回字段中的候选级数组:
        - centered_box_index: int32, ``(N_entry,)``, 当前 centered 文件内从 0 连续编号的候选编号.
        - source_blob_index: int32, ``(N_entry,)``, 每个 centered 候选对应的来源 blob 编号; 数值索引同一 F-alpha blobs 文件中候选级数组的第一维.
        - box_start_zyx: int32, ``(N_entry, 3)``, 每个 80³ BOX 在完整密度图中的整数起点; 最后一维按 ZYX 排列, 单位 voxel.
        - box_shape_zyx: uint8, ``(N_entry, 3)``, 每个 centered BOX 的 ZYX 体素数, 固定为 ``(80, 80, 80)``.
        - box_origin_world: float32, ``(N_entry, 3)``, 每个 BOX 起点对应的世界坐标; 最后一维按 XYZ 排列, 单位 Å.
        - voxel_size_world: float32, ``(N_entry, 3)``, 每个 BOX 沿世界 XYZ 三轴的体素间距, 单位 Å/voxel.
        - source_probability_mean: float32, ``(N_entry,)``, 每个完整来源 blob 的 Stage1 平均配体概率.
        - source_threshold_value: float32, ``(N_entry,)``, 生成每个来源 blob 时使用的语义概率阈值.
        - source_blob_fits_centered_box: bool, ``(N_entry,)``, True 表示完整来源 blob 可被一个合法 80³ BOX 容纳, False 表示归档中的稀疏体素仅覆盖实际落入 BOX 的部分.
        - source_voxel_count: int32, ``(N_entry,)``, 每个完整来源 blob 在裁切前的体素数; centered 选择阶段的体素数门槛使用该字段.

    返回字段中的稀疏 V/auxiliary 数组:
        - voxel_offsets: int64, ``(N_entry + 1,)``, 同步切分 `voxel_index_local_zyx`, `source_probability`, `centered_probability` 和 `voxel_final` 的第一维; 第 i 个半开区间对应第 i 个 centered 候选, 首值为 0, 末值为 L_voxel.
        - voxel_index_local_zyx: int16, ``(L_voxel, 3)``, 实际位于各自 80³ BOX 内的来源体素索引; 最后一维按局部 ZYX 排列, 原点为该 BOX 起点, 单位 voxel.
        - source_probability: float32, ``(L_voxel,)``, 完整图前向在每个归档来源体素上的配体概率, 与 `voxel_index_local_zyx` 第一维逐体素对齐.
        - centered_probability: float32, ``(L_voxel,)``, centered 前向在每个归档来源体素上的配体概率, 与 `voxel_index_local_zyx` 第一维逐体素对齐.
        - voxel_final: float16, ``(L_voxel, C_voxel)``, centered 前向在每个归档来源体素上的 V 学习特征, 第一维与 `voxel_index_local_zyx` 逐体素对齐.
        - voxel_aux_offsets: int64, ``(N_entry + 1,)``, 同步切分 `voxel_aux_index_local_zyx` 和 `voxel_aux_probability` 的第一维; 第 i 个半开区间对应第 i 个 centered 候选, 首值为 0, 末值为 L_aux.
        - voxel_aux_index_local_zyx: int16, ``(L_aux, 3)``, Dataset `hardmask=True` 的受体占据体素索引; 最后一维按局部 ZYX 排列, 原点为该 BOX 起点, 单位 voxel.
        - voxel_aux_probability: float32, ``(L_aux,)``, centered 前向在每个受体占据体素上的辅助受体概率, 与 `voxel_aux_index_local_zyx` 第一维逐体素对齐.

    返回字段中的 Find A/P 数组:
        - A_offsets: int64, ``(N_entry + 1,)``, 同步切分全部 `A_*` 数组的第一维; 第 i 个半开区间对应第 i 个 centered 候选, 首值为 0, 末值为 N_A.
        - A_global_index: int64, ``(N_A,)``, 每个保留 A 原子在当前 PDB `receptor_tokens.npz` 原子轴上的编号.
        - A_coord_local_xyz: float32, ``(N_A, 3)``, A 原子相对各自 BOX 起点的 XYZ 体素坐标, 单位 voxel.
        - A_coord_centered_world: float32, ``(N_A, 3)``, A 原子相对各自 BOX 局部坐标 ``(40, 40, 40)`` 的世界 XYZ 位移, 单位 Å.
        - A_probability: float32, ``(N_A,)``, centered 前向输出的 A 原子概率, 与 `A_global_index` 逐原子对齐.
        - A_feat_L0: float32, ``(N_A, 50)``, 每个 A 原子的 49 维 Dataset 基础特征和 1 维主链标志.
        - A_feat_L1: float16, ``(N_A, C_A1)``, centered 前向输出的第一层 A 学习特征.
        - A_feat_L2: float16, ``(N_A, C_A2)``, centered 前向输出的第二层 A 学习特征.
        - A_feat_L3: float16, ``(N_A, C_A3)``, centered 前向输出的第三层 A 学习特征.
        - P_offsets: int64, ``(N_entry + 1,)``, 同步切分全部 `P_*` 数组的第一维; 第 i 个半开区间对应第 i 个 centered 候选, 首值为 0, 末值为 N_P.
        - P_coord_local_xyz: float32, ``(N_P, 3)``, P 锚点相对各自 BOX 起点的 XYZ 体素坐标, 单位 voxel.
        - P_probability: float32, ``(N_P,)``, centered 前向输出的 P 锚点概率, 与 `P_coord_local_xyz` 第一维逐点对齐.
        - P_feat_L2: float16, ``(N_P, C_P2)``, centered 前向输出的第二层 P 学习特征.
        - P_feat_L3: float16, ``(N_P, C_P3)``, centered 前向输出的第三层 P 学习特征.

    返回字段中的稠密 48³ 数组:
        - v_centroid_local_zyx: float32, ``(N_entry, 3)``, 每个 BOX 内归档来源体素的整数索引质心; 最后一维按局部 ZYX 排列, 单位 voxel.
        - crop_start_local_zyx: int16, ``(N_entry, 3)``, 每个 48³ 裁块相对其 80³ BOX 的整数起点; 最后一维按 ZYX 排列, 单位 voxel.
        - crop_center_offset_zyx: float32, ``(N_entry, 3)``, 来源体素索引质心减去 48³ 裁块中心 ``crop_start_local_zyx + 23.5``; 最后一维按 ZYX 排列, 单位 voxel.
        - crop_clipped_axis_mask: bool, ``(N_entry, 3)``, True 表示对应 Z/Y/X 轴上的理想 48³ 起点被 80³ BOX 边界截断, False 表示该轴未截断.
        - experimental_density_48: float32, ``(N_entry, 48, 48, 48)``, 当前 PDB 实验密度在各个 48³ 裁块内的数值, 后三维按 ZYX 排列.
        - simulated_density_48: float32, ``(N_entry, 48, 48, 48)``, 当前 PDB 模拟密度在各个 48³ 裁块内的数值, 后三维按 ZYX 排列.
        - source_probability_48: float32, ``(N_entry, 48, 48, 48)``, 完整图前向概率在各个 48³ 裁块内的数值, 后三维按 ZYX 排列.

    所有 offsets 使用半开区间并保持 `entries` 的候选顺序. N_entry 为零时返回形状合法的空数组; 无法从空输入获知的学习特征通道数设为 0. 返回映射不包含 object dtype.
    """

    # bool 标量, True 表示当前 producer 的每个候选还必须提供并归档 A 原子表和 P 锚点表.
    is_find = str(producer).startswith("Find_")

    # dict[str, type], 候选级字段名到正式归档 dtype 的映射; stack 后每个数组的第一维均为 N_entry.
    metadata_dtypes = {
        "source_blob_index": np.int32,
        "box_start_zyx": np.int32,
        "box_shape_zyx": np.uint8,
        "box_origin_world": np.float32,
        "voxel_size_world": np.float32,
        "source_probability_mean": np.float32,
        "source_threshold_value": np.float32,
        "source_blob_fits_centered_box": np.bool_,
        "source_voxel_count": np.int32,
    }
    # dict[str, np.ndarray], 将由 pipeline 原样写入单个 PDB centered NPZ 的字段映射.
    arrays: dict[str, np.ndarray] = {}
    for field, dtype in metadata_dtypes.items():
        if entries:
            # (N_entry, *field_shape), 沿新增的候选轴堆叠当前候选级字段, 不改变单个候选字段的内部维度.
            arrays[field] = np.stack(
                [np.asarray(entry[field], dtype=dtype) for entry in entries],
                axis=0,
            )
        elif field in {
            "box_start_zyx",
            "box_shape_zyx",
            "box_origin_world",
            "voxel_size_world",
        }:
            # (0, 3), 在无候选时保留 ZYX 或 XYZ 三分量字段的二维契约.
            arrays[field] = np.empty((0, 3), dtype=dtype)
        else:
            # (0,), 在无候选时保留标量候选字段的一维契约.
            arrays[field] = np.empty((0,), dtype=dtype)
    # int32, (N_entry,), centered NPZ 内从 0 连续编号的候选编号; 数值索引所有候选级数组的第一维.
    arrays["centered_box_index"] = np.arange(len(entries), dtype=np.int32)

    # tuple, 每项依次给出 offsets 字段名和共享该 offsets 的稀疏值表规格; 值表规格包含字段名, 归档 dtype 和无候选时的尾部形状.
    ragged_groups = (
        (
            "voxel_offsets",
            (
                ("voxel_index_local_zyx", np.int16, (3,)),
                ("source_probability", np.float32, ()),
                ("centered_probability", np.float32, ()),
            ),
        ),
        (
            "voxel_aux_offsets",
            (
                ("voxel_aux_index_local_zyx", np.int16, (3,)),
                ("voxel_aux_probability", np.float32, ()),
            ),
        ),
    )
    if is_find:
        # Find producer 的 A 原子表和 P 锚点表各自使用一组 offsets, 并保持每个候选内的模型输出顺序.
        ragged_groups += (
            (
                "A_offsets",
                (
                    ("A_global_index", np.int64, ()),
                    ("A_coord_local_xyz", np.float32, (3,)),
                    ("A_coord_centered_world", np.float32, (3,)),
                    ("A_probability", np.float32, ()),
                    ("A_feat_L0", np.float32, (50,)),
                    ("A_feat_L1", np.float16, None),
                    ("A_feat_L2", np.float16, None),
                    ("A_feat_L3", np.float16, None),
                ),
            ),
            (
                "P_offsets",
                (
                    ("P_coord_local_xyz", np.float32, (3,)),
                    ("P_probability", np.float32, ()),
                    ("P_feat_L2", np.float16, None),
                    ("P_feat_L3", np.float16, None),
                ),
            ),
        )

    for offsets_name, fields in ragged_groups:
        # str, 当前稀疏字段组的基准值表名; 该值表在每个候选中的实体数必须与同组其他值表相同.
        first_field = fields[0][0]
        # int64, (N_entry,), 每个 centered 候选在当前稀疏字段组中的体素数, A 原子数或 P 锚点数.
        counts = np.asarray(
            [np.asarray(entry[first_field]).shape[0] for entry in entries],
            dtype=np.int64,
        )
        # int64, (N_entry + 1,), 同步切分当前组全部值表的第一维; 第 i 个半开区间对应第 i 个 centered 候选, 首值为 0, 末值为当前组实体总数.
        arrays[offsets_name] = np.concatenate(
            (np.zeros(1, dtype=np.int64), np.cumsum(counts, dtype=np.int64))
        )
        for field, dtype, empty_tail in fields:
            if entries:
                # (L_group, *field_shape), 按 centered 候选顺序拼接当前稀疏值表; L_group 是当前组在全部候选中的实体总数.
                arrays[field] = np.concatenate(
                    [np.asarray(entry[field], dtype=dtype) for entry in entries],
                    axis=0,
                )
            else:
                # tuple[int, ...], 无候选时当前值表除第一维外的固定形状; None 表示未知学习特征宽度并按 0 处理.
                tail = (0,) if empty_tail is None else empty_tail
                # (0, *field_shape), 当前稀疏值表的空数组; 第一维仍可由对应 offsets 的唯一值 0 正确切分.
                arrays[field] = np.empty((0, *tail), dtype=dtype)

    # float16, (L_voxel, C_voxel), 按候选顺序拼接的 V 学习特征; 第一维由 voxel_offsets 切分并与 voxel_index_local_zyx 逐体素对齐.
    arrays["voxel_final"] = (
        np.concatenate(
            [np.asarray(entry["voxel_final"], dtype=np.float16) for entry in entries],
            axis=0,
        )
        if entries
        else np.empty((0, 0), dtype=np.float16)
    )
    # tuple[str, ...], 每个候选都必须提供的 48³ 几何和稠密裁块字段名; 这些字段不使用 offsets.
    dense_fields = (
        "v_centroid_local_zyx",
        "crop_start_local_zyx",
        "crop_center_offset_zyx",
        "crop_clipped_axis_mask",
        "experimental_density_48",
        "simulated_density_48",
        "source_probability_48",
    )
    for field in dense_fields:
        if entries:
            # (N_entry, *field_shape), 沿新增候选轴堆叠当前 48³ 几何或稠密裁块字段.
            arrays[field] = np.stack([entry[field] for entry in entries], axis=0)
        elif field in {
            "v_centroid_local_zyx",
            "crop_start_local_zyx",
            "crop_center_offset_zyx",
            "crop_clipped_axis_mask",
        }:
            # type, 当前空几何数组的归档 dtype; 只有边界截断掩码使用 bool.
            dtype = np.bool_ if field == "crop_clipped_axis_mask" else np.float32
            if field == "crop_start_local_zyx":
                dtype = np.int16
            # (0, 3), 无候选时保留 ZYX 三分量几何字段的二维契约.
            arrays[field] = np.empty((0, 3), dtype=dtype)
        else:
            # float32, (0, 48, 48, 48), 无候选时保留 ZYX 稠密裁块的四维契约.
            arrays[field] = np.empty((0, 48, 48, 48), dtype=np.float32)
    return arrays


# ================================================================================================


def infer_centered_boxes(
    dataset: Any,
    collator: Any,
    wrapper: Any,
    pdb_id: str,
    producer: str,
    blobs: Mapping[str, np.ndarray],
    full_probability: np.ndarray,
    origin_xyz: np.ndarray,
    voxel_size_xyz: np.ndarray,
    device: str,
    precision: str,
    centered_batch_size: int,
    centered_workers: int,
    prefetch_batches: int,
    pending_cpu_batches: int,
    forward_min_voxels: int,
    packer: Executor,
) -> tuple[Future[dict[str, np.ndarray]], dict[str, float | int]]:
    """对达到来源体素数门槛的 blob 候选执行 80³ centered Stage1 前向.

    形状符号:
        - N_blob: 当前 PDB 在输入 F-alpha blobs 文件中的来源 blob 数量.
        - L_source: 当前 PDB 全部来源 blob 的稀疏体素合计数量.
        - N_entry: 满足 `forward_min_voxels` 并实际执行 centered 前向的来源 blob 数量.
        - B_entry: 当前 centered 模型批次中的候选数量, 满足 ``1 <= B_entry <= centered_batch_size``.
        - D, H, W: 当前 PDB 完整密度图沿 ZYX 三轴的体素数.
        - K_full: 单个完整来源 blob 的体素数.
        - K_box: 单个来源 blob 中实际落入所选 80³ BOX 的体素数, 满足 ``K_box <= K_full``.
        - K_aux: 单个 80³ BOX 中 Dataset `hardmask=True` 的受体占据体素数.
        - C_voxel: Stage1 `voxel_final` 的 V 特征通道数.

    输入参数:
        - dataset: Stage1Dataset, 根据 ``ResolvedStage1Crop`` 物化当前 PDB 的 80³ 模型输入; `dataset.root` 还定位 ``density/<pdb_id>/exp.npy`` 和 ``sim.npy``.
        - collator: `dataset.collate_fn`, 把保持候选顺序的 B_entry 个样本拼成 Stage1 模型批次.
        - wrapper: Stage1 模型包装器, 接收 collator 批次并返回体素 logits, V 特征以及 Find producer 的 A/P 字段.
        - pdb_id: 字符串, 当前 PDB 的小写标识, 如 `1abc`.
        - producer: 字符串, 当前 Stage1 producer 名称; `Find_*` 额外归档 A 原子表和 P 锚点表, `unet_*` 只归档共同字段.
        - blobs.voxel_offsets: int64, ``(N_blob + 1,)``, 同步切分 `voxel_index_global_zyx` 和 `source_probability` 的第一维; 第 i 个半开区间对应第 i 个来源 blob, 首值为 0, 末值为 L_source.
        - blobs.voxel_count: int32, ``(N_blob,)``, 每个完整来源 blob 的体素数, 与 blobs 文件的候选轴对齐.
        - blobs.voxel_index_global_zyx: int32, ``(L_source, 3)``, 完整来源 blob 的稀疏体素索引; 最后一维按完整图 ZYX 排列, 单位 voxel.
        - blobs.source_probability: float32, ``(L_source,)``, 完整图前向在每个来源体素上的配体概率, 与 `voxel_index_global_zyx` 第一维逐体素对齐.
        - blobs.source_probability_mean: float32, ``(N_blob,)``, 每个完整来源 blob 的 Stage1 平均配体概率.
        - blobs.fits_centered_box: bool, ``(N_blob,)``, True 表示完整来源 blob 可被一个合法 80³ BOX 容纳, False 表示仍执行前向但只归档实际落入 BOX 的稀疏体素.
        - blobs.centered_box_start_zyx: int32, ``(N_blob, 3)``, 可完整容纳时的 80³ BOX 完整图 ZYX 起点, 单位 voxel; `fits_centered_box=False` 时三个分量均为 -1 并在本函数中重算.
        - blobs.source_threshold_value: float32, ``(1,)``, 生成当前 F-alpha 来源 blobs 时使用的语义概率阈值.
        - full_probability: float32, ``(D, H, W)``, 当前 PDB 的完整图配体概率, 三维按 ZYX 排列.
        - origin_xyz: float32, ``(3,)``, 完整密度图索引 ``(0, 0, 0)`` 对应的世界 XYZ 坐标, 单位 Å.
        - voxel_size_xyz: float32, ``(3,)``, 完整密度图沿世界 XYZ 三轴的体素间距, 单位 Å/voxel.
        - device: 字符串, Stage1 前向设备, 如 `cuda:0`.
        - precision: 字符串, CUDA autocast 精度; `bf16` 或 `float16` 开启混合精度, 其他值关闭 autocast.
        - centered_batch_size: int, 单次 Stage1 前向最多包含的 centered 候选数.
        - centered_workers: int, 并行执行 `dataset.materialize_request` 的 CPU 线程数.
        - prefetch_batches: int, 已提交 CPU 物化但尚未开始 GPU 前向的最大批次数.
        - pending_cpu_batches: int, 已完成 GPU 前向但尚未完成 CPU 字段整理的最大批次数.
        - forward_min_voxels: int, 进入 centered 前向的完整来源 blob 最小体素数, 包含端点; 该门槛不设置最终 `selected`.
        - packer: 跨 PDB 共享的 CPU Executor, 异步执行 :func:`pack_centered_entries`, 使当前 PDB 的 concatenate 可与下一 PDB 的 GPU 前向重叠.

    返回值:
        - packed: ``Future[dict[str, np.ndarray]]``, 完成后得到 :func:`pack_centered_entries` 定义的全部 centered NPZ 字段; 候选顺序与达到 `forward_min_voxels` 的来源 blob 顺序一致.
        - performance: ``dict[str, float | int]``, 当前 PDB 的 centered 执行统计, 由 pipeline 合并到 ``status/<role>/performance.json``.
            - performance.wall_seconds: float, 从进入本函数到提交 packer 前的 centered 总墙钟秒数.
            - performance.materialize_wait_seconds: float, 当前线程等待 CPU 请求物化完成的累计秒数.
            - performance.cpu_arrange_wait_seconds: float, 当前线程等待异步 D2H 后字段整理完成的累计秒数.
            - performance.batch_count: int, 实际执行 Stage1 centered 前向的批次数.
            - performance.entry_count: int, 实际执行 Stage1 centered 前向的来源 blob 数量 N_entry.

    科学边界:
        - `fits_centered_box=False` 的来源 blob 使用完整 K_full 个体素的质心确定合法 80³ BOX, 仍与普通 blob 一样执行完整 Stage1 前向.
        - centered NPZ 的 `source_voxel_count` 保留 K_full, 但稀疏概率和 V 特征只归档实际落入 BOX 的 K_box 个来源体素.
        - BOX 起点和体素索引使用 ZYX 顺序, A/P 局部坐标和世界坐标使用 XYZ 顺序.

    副作用:
        - 只读 mmap 当前 PDB 的实验密度和模拟密度 NPY, 启动并关闭请求物化与字段整理线程池, 并在 `device` 上执行模型前向.
        - 本函数不写 centered NPZ, 不修改输入 blobs, 不拟合 centered 选择参数.
    """
    from src.datasets.stage1_requests import ResolvedStage1Crop

    started_at = time.perf_counter()
    materialize_wait_seconds = 0.0
    cpu_arrange_wait_seconds = 0.0
    # int64, (N_blob,), 每个完整来源 blob 的体素数; 第一维与 blobs 候选级数组对齐.
    blob_count = np.asarray(blobs["voxel_count"], dtype=np.int64)
    # bool, (N_blob,), True 表示对应来源 blob 的全部体素可被一个合法 80³ BOX 容纳, 并遮盖 blobs 候选轴.
    fits = np.asarray(blobs["fits_centered_box"], dtype=np.bool_)
    # int32, (N_entry,), 体素数达到 forward_min_voxels 的来源 blob 编号; 数值索引 blobs 候选级数组的第一维, 不按 fits 删除候选.
    eligible = np.flatnonzero(blob_count >= int(forward_min_voxels)).astype(np.int32)
    # int32, (N_blob, 3), 每个来源 blob 将使用的 80³ BOX 完整图整数起点; 最后一维按 ZYX 排列, 单位 voxel.
    box_starts = np.asarray(blobs["centered_box_start_zyx"], dtype=np.int32).copy()
    # int64, (3,), 完整概率图沿 ZYX 三轴的体素数, 用于把重算 BOX 起点逐轴限制到 [0, full_shape - 80].
    full_shape = np.asarray(full_probability.shape, dtype=np.int64)
    # 仅重算达到体素数门槛且无法完整容纳的来源 blob, 已可完整容纳的 BOX 起点保持 blobs 文件中的权威值.
    for source_index in eligible[~fits[eligible]].tolist():
        # Python 整数, 当前来源 blob 在两个稀疏值表中的半开区间端点.
        voxel_begin = int(blobs["voxel_offsets"][source_index])
        voxel_end = int(blobs["voxel_offsets"][source_index + 1])
        # float64, (K_full, 3), 当前完整来源 blob 的完整图整数体素索引; 最后一维按 ZYX 排列, 单位 voxel.
        source_global_zyx = np.asarray(
            blobs["voxel_index_global_zyx"][voxel_begin:voxel_end],
            dtype=np.float64,
        )
        # int64, (3,), 让完整来源体素中心的平均位置尽量对齐 80³ BOX 中心的理想 ZYX 起点, 单位 voxel.
        requested_start = np.rint(
            source_global_zyx.mean(axis=0) + 0.5 - 40.0
        ).astype(np.int64)
        # [3] -> [3], 每个 Z/Y/X 分量独立裁到合法起点区间, 保证后续 80³ 切片完全位于完整图内.
        box_starts[source_index] = np.clip(
            requested_start,
            0,
            full_shape - 80,
        ).astype(np.int32)
    # tuple[np.ndarray, ...], 按 eligible 顺序分组的来源 blob 编号; 每个 int32 数组形状为 (B_entry,), 数值索引 blobs 候选轴.
    batches = tuple(
        eligible[offset : offset + int(centered_batch_size)]
        for offset in range(0, eligible.shape[0], int(centered_batch_size))
    )
    # bool 标量, True 表示模型输出和 centered 归档还包含 A 原子表与 P 锚点表.
    is_find = producer.startswith("Find_")

    # Path, 当前 PDB 完整密度目录 `<dataset.root>/density/<pdb_id>`, 内含 exp.npy 和 sim.npy.
    density_root = Path(dataset.root) / "density" / str(pdb_id).lower()
    # float 数组 mmap, (D, H, W), 当前 PDB 完整实验密度; 三维按 ZYX 排列, 仅在构造 48³ 裁块时读取.
    experimental = np.load(
        density_root / "exp.npy", mmap_mode="r", allow_pickle=False
    )[0]
    # float 数组 mmap, (D, H, W), 当前 PDB 完整模拟密度; 三维按 ZYX 排列, 仅在构造 48³ 裁块时读取.
    simulated = np.load(
        density_root / "sim.npy", mmap_mode="r", allow_pickle=False
    )[0]
    # -----------------------------------------------------------------------------------------------------------
    def arrange_batch(
        host_output: Mapping[str, Any],
        cpu_batch: Mapping[str, Any],
        source_indices: np.ndarray,
        ready_event: torch.cuda.Event | None,
    ) -> list[dict[str, np.ndarray]]:
        """等待当前批次的 D2H 完成并构造逐候选 centered 字段映射.

        形状符号:
            - B_entry: 当前 centered 模型批次的候选数量.
            - N_A_output: 当前批次模型输出的 A 原子合计数量.
            - N_A_input: 当前批次 Dataset 输入的 A 原子合计数量.
            - N_P_batch: 当前批次模型输出的 P 锚点合计数量.
            - K_full: 单个完整来源 blob 的体素数.
            - K_box: 单个来源 blob 中实际落入当前 80³ BOX 的体素数.
            - K_aux: 单个 80³ BOX 中 Dataset `hardmask=True` 的受体占据体素数.
            - N_A: 单个候选经过核心 BOX 和 10 Å 距离门槛后保留的 A 原子数.
            - N_P: 单个候选的 P 锚点数.
            - C_voxel, C_A1, C_A2, C_A3, C_P2, C_P3: 对应 V, A 和 P 学习特征的通道数.

        输入参数:
            - host_output.voxel_logits_ligand: CPU 张量, ``(B_entry, 1, 80, 80, 80)``, centered 前向的配体 logits, 后三维按 ZYX 排列.
            - host_output.voxel_logits_aux: CPU 张量, ``(B_entry, 1, 80, 80, 80)``, centered 前向的辅助受体 logits, 后三维按 ZYX 排列.
            - host_output.voxel_final: CPU 张量, ``(B_entry, C_voxel, 80, 80, 80)``, centered 前向的 V 学习特征, 后三维按 ZYX 排列.
            - host_output.atom_counts: Find 专用 int64 CPU 张量, ``(B_entry,)``, 每个候选的模型输出 A 原子数, 并同步切分全部模型输出 `A_*` 字段.
            - host_output.atom_global_indices: Find 专用 int64 CPU 张量, ``(N_A_output,)``, 每个模型输出 A 原子在当前 PDB `receptor_tokens.npz` 原子轴上的编号.
            - host_output.atom_coord_local_voxel: Find 专用 CPU 张量, ``(N_A_output, 3)``, 模型输出 A 原子相对各自 BOX 起点的 XYZ 体素坐标, 单位 voxel.
            - host_output.atom_logits: Find 专用 CPU 张量, ``(N_A_output, 1)``, 模型输出 A 原子 logits, 与 `atom_global_indices` 逐原子对齐.
            - host_output.A_feat_L1: Find 专用 CPU 张量, ``(N_A_output, C_A1)``, 模型输出第一层 A 学习特征.
            - host_output.A_feat_L2: Find 专用 CPU 张量, ``(N_A_output, C_A2)``, 模型输出第二层 A 学习特征.
            - host_output.A_feat_L3: Find 专用 CPU 张量, ``(N_A_output, C_A3)``, 模型输出第三层 A 学习特征.
            - host_output.anchor_batch_index: Find 专用 int64 CPU 张量, ``(N_P_batch,)``, 每个 P 锚点所属的 centered 批次编号, 数值位于 ``[0, B_entry)``.
            - host_output.anchor_coord_local_voxel: Find 专用 CPU 张量, ``(N_P_batch, 3)``, P 锚点相对所属 BOX 起点的 XYZ 体素坐标, 单位 voxel.
            - host_output.pseudo_logits: Find 专用 CPU 张量, ``(N_P_batch, 1)``, P 锚点 logits, 与 `anchor_batch_index` 逐点对齐.
            - host_output.P_feat_L2: Find 专用 CPU 张量, ``(N_P_batch, C_P2)``, 模型输出第二层 P 学习特征.
            - host_output.P_feat_L3: Find 专用 CPU 张量, ``(N_P_batch, C_P3)``, 模型输出第三层 P 学习特征.
            - cpu_batch.hardmask: bool CPU 张量, ``(B_entry, 80, 80, 80)``, True 表示当前 BOX 中由 Dataset 标记的受体占据体素, False 表示非受体体素.
            - cpu_batch.atom_counts: Find 专用 int64 CPU 张量, ``(B_entry,)``, 每个候选的 Dataset 输入 A 原子数, 并同步切分三个 Dataset 输入原子字段.
            - cpu_batch.atom_global_indices: Find 专用 int64 CPU 张量, ``(N_A_input,)``, Dataset 输入 A 原子在当前 PDB `receptor_tokens.npz` 原子轴上的编号.
            - cpu_batch.atom_feat: Find 专用 float32 CPU 张量, ``(N_A_input, 49)``, Dataset 输入 A 原子的 49 维基础特征.
            - cpu_batch.atom_is_backbone: Find 专用 bool CPU 张量, ``(N_A_input,)``, True 表示对应 Dataset 输入 A 原子属于主链, False 表示非主链.
            - source_indices: int32 数组, ``(B_entry,)``, 当前批次的来源 blob 编号, 数值索引 `blobs` 候选级数组的第一维.
            - ready_event: CUDA event 或 None, 非 None 时标记当前批次全部异步 D2H 完成; CPU 前向传入 None.

        返回值:
            - candidate_entries: 长度 B_entry 的 ``list[dict[str, np.ndarray]]``, 顺序与 `source_indices` 一致, 每个映射是 :func:`pack_centered_entries` 的单候选输入.
            - candidate_entries[*].source_blob_index: int32 标量, 当前候选在来源 blobs 文件候选轴上的编号.
            - candidate_entries[*].box_start_zyx: int32, ``(3,)``, 80³ BOX 的完整图 ZYX 起点, 单位 voxel.
            - candidate_entries[*].box_shape_zyx: uint8, ``(3,)``, 80³ BOX 沿 ZYX 三轴的体素数, 固定为 ``(80, 80, 80)``.
            - candidate_entries[*].box_origin_world: float32, ``(3,)``, BOX 起点对应的世界 XYZ 坐标, 单位 Å.
            - candidate_entries[*].voxel_size_world: float32, ``(3,)``, 世界 XYZ 三轴的体素间距, 单位 Å/voxel.
            - candidate_entries[*].source_probability_mean: float32 标量, 完整来源 blob 的平均配体概率.
            - candidate_entries[*].source_threshold_value: float32 标量, 生成来源 blob 时使用的语义概率阈值.
            - candidate_entries[*].source_blob_fits_centered_box: bool 标量, 完整来源 blob 是否可被一个合法 80³ BOX 容纳.
            - candidate_entries[*].source_voxel_count: int32 标量, 完整来源 blob 在裁切前的体素数 K_full.
            - candidate_entries[*].voxel_index_local_zyx: int16, ``(K_box, 3)``, 实际落入 BOX 的来源体素局部 ZYX 索引, 单位 voxel.
            - candidate_entries[*].source_probability: float32, ``(K_box,)``, 完整图前向在归档来源体素上的配体概率.
            - candidate_entries[*].centered_probability: float32, ``(K_box,)``, centered 前向在归档来源体素上的配体概率.
            - candidate_entries[*].voxel_final: float16, ``(K_box, C_voxel)``, centered 前向在归档来源体素上的 V 学习特征.
            - candidate_entries[*].voxel_aux_index_local_zyx: int16, ``(K_aux, 3)``, Dataset 受体占据体素的局部 ZYX 索引, 单位 voxel.
            - candidate_entries[*].voxel_aux_probability: float32, ``(K_aux,)``, centered 前向在受体占据体素上的辅助受体概率.
            - candidate_entries[*].v_centroid_local_zyx: float32, ``(3,)``, K_box 个归档来源体素的局部 ZYX 整数索引质心, 单位 voxel.
            - candidate_entries[*].crop_start_local_zyx: int16, ``(3,)``, 48³ 裁块相对 80³ BOX 的局部 ZYX 起点, 单位 voxel.
            - candidate_entries[*].crop_center_offset_zyx: float32, ``(3,)``, 来源体素索引质心相对 48³ 裁块中心的 ZYX 位移, 单位 voxel.
            - candidate_entries[*].crop_clipped_axis_mask: bool, ``(3,)``, True 表示对应 Z/Y/X 轴的理想 48³ 起点被 80³ BOX 边界截断.
            - candidate_entries[*].experimental_density_48: float32, ``(48, 48, 48)``, 当前 48³ 裁块的实验密度, 三维按 ZYX 排列.
            - candidate_entries[*].simulated_density_48: float32, ``(48, 48, 48)``, 当前 48³ 裁块的模拟密度, 三维按 ZYX 排列.
            - candidate_entries[*].source_probability_48: float32, ``(48, 48, 48)``, 当前 48³ 裁块的完整图前向概率, 三维按 ZYX 排列.
            - candidate_entries[*].A_global_index: Find 专用 int64, ``(N_A,)``, 保留 A 原子在当前 PDB `receptor_tokens.npz` 原子轴上的编号.
            - candidate_entries[*].A_coord_local_xyz: Find 专用 float32, ``(N_A, 3)``, 保留 A 原子的 BOX 局部 XYZ 体素坐标, 单位 voxel.
            - candidate_entries[*].A_coord_centered_world: Find 专用 float32, ``(N_A, 3)``, 保留 A 原子相对 BOX 局部坐标 ``(40, 40, 40)`` 的世界 XYZ 位移, 单位 Å.
            - candidate_entries[*].A_probability: Find 专用 float32, ``(N_A,)``, 保留 A 原子的 centered 概率.
            - candidate_entries[*].A_feat_L0: Find 专用 float32, ``(N_A, 50)``, 保留 A 原子的 49 维 Dataset 基础特征和 1 维主链标志.
            - candidate_entries[*].A_feat_L1: Find 专用数组, ``(N_A, C_A1)``, 保留 A 原子的第一层学习特征.
            - candidate_entries[*].A_feat_L2: Find 专用数组, ``(N_A, C_A2)``, 保留 A 原子的第二层学习特征.
            - candidate_entries[*].A_feat_L3: Find 专用数组, ``(N_A, C_A3)``, 保留 A 原子的第三层学习特征.
            - candidate_entries[*].P_coord_local_xyz: Find 专用 float32, ``(N_P, 3)``, 当前候选 P 锚点的 BOX 局部 XYZ 体素坐标, 单位 voxel.
            - candidate_entries[*].P_probability: Find 专用 float32, ``(N_P,)``, 当前候选 P 锚点的 centered 概率.
            - candidate_entries[*].P_feat_L2: Find 专用数组, ``(N_P, C_P2)``, 当前候选 P 锚点的第二层学习特征.
            - candidate_entries[*].P_feat_L3: Find 专用数组, ``(N_P, C_P3)``, 当前候选 P 锚点的第三层学习特征.

        科学边界:
            - 完整来源 blob 使用 K_full 个体素记录 `source_voxel_count`, 但概率和 V 特征只归档实际落入 BOX 的 K_box 个体素.
            - Find A 原子必须同时位于局部 XYZ 范围 ``[0, 80)`` 且到最近归档来源体素中心的世界距离不超过 10 Å.
            - Find P 锚点只按 `anchor_batch_index` 归入所属候选, 不在本函数中增加概率或距离门槛.

        副作用:
            - `ready_event` 非 None 时阻塞当前 CPU 整理线程, 直到当前批次的异步 D2H 完成.
            - 本函数不写文件, 不访问 GPU, 不修改 `blobs` 或 `cpu_batch`.
        """
        if ready_event is not None:
            ready_event.synchronize()

        # dict[str, np.ndarray], 模型输出名到 CPU NumPy 数组的映射; 每个字段保持 host_output 中的形状, BF16 字段先提升为 float32.
        host_arrays: dict[str, np.ndarray] = {}
        for name, value in host_output.items():
            # CPU 张量, (*field_shape), 当前模型输出字段的 D2H 副本; detach 只移除 autograd 关系, 不改变形状.
            tensor = value.detach()
            if tensor.dtype == torch.bfloat16:
                # [*field_shape] -> [*field_shape], BF16 转为 NumPy 支持的 float32, 数值仍对应同一模型输出实体.
                tensor = tensor.to(torch.float32)
            host_arrays[name] = tensor.numpy()

        # float32, (B_entry, 1, 80, 80, 80), centered 配体 logits, 后三维按局部 ZYX 排列.
        ligand_logits = np.asarray(
            host_arrays["voxel_logits_ligand"], dtype=np.float32
        )
        # float32, (B_entry, 80, 80, 80), 对单通道 logits 做 sigmoid 后的 centered 配体概率. [B_entry, 1, 80, 80, 80] -> [B_entry, 80, 80, 80]
        ligand_probability = 1.0 / (1.0 + np.exp(-ligand_logits[:, 0]))
        # float32, (B_entry, 80, 80, 80), 对单通道辅助 logits 做 sigmoid 后的受体辅助概率, 后三维按局部 ZYX 排列.
        auxiliary = 1.0 / (
            1.0
            + np.exp(
                -np.asarray(host_arrays["voxel_logits_aux"], dtype=np.float32)[:, 0]
            )
        )
        # (B_entry, C_voxel, 80, 80, 80), centered 前向的 V 学习特征; 后三维按局部 ZYX 排列.
        voxel_final = host_arrays["voxel_final"]
        # int64, (B_entry,), Find 前向对每个 centered 候选输出的 A 原子数; 非 Find producer 使用形状 (0,) 的空数组.
        atom_counts = (
            np.asarray(host_arrays["atom_counts"], dtype=np.int64).reshape(-1)
            if is_find
            else np.empty((0,), dtype=np.int64)
        )
        # int64, (B_entry + 1,), 同步切分模型输出 `atom_global_indices`, `atom_coord_local_voxel`, `atom_logits` 和 `A_feat_L1:L3`; 第 i 个半开区间属于第 i 个批次候选, 首值为 0, 末值为 N_A_output.
        atom_offsets = np.concatenate(
            (np.zeros(1, dtype=np.int64), np.cumsum(atom_counts, dtype=np.int64))
        )
        # int64, (B_entry,), Dataset 为每个 centered 候选物化的输入 A 原子数; 非 Find producer 使用形状 (0,) 的空数组.
        input_atom_counts = (
            np.asarray(cpu_batch["atom_counts"], dtype=np.int64).reshape(-1)
            if is_find
            else np.empty((0,), dtype=np.int64)
        )
        # int64, (B_entry + 1,), 同步切分 Dataset 输入 `atom_global_indices`, `atom_feat` 和 `atom_is_backbone`; 第 i 个半开区间属于第 i 个批次候选, 首值为 0, 末值为 N_A_input.
        input_atom_offsets = np.concatenate(
            (np.zeros(1, dtype=np.int64), np.cumsum(input_atom_counts, dtype=np.int64))
        )
        # list[dict[str, np.ndarray]], 按 source_indices 顺序收集的单候选 centered 字段映射, 每项对应一个实际执行前向的 80³ BOX.
        candidate_entries: list[dict[str, np.ndarray]] = []
        for batch_index, source_index_value in enumerate(source_indices.tolist()):
            # Python 整数, 当前 centered 候选对应的来源 blob 编号; 数值索引 blobs 候选级数组的第一维.
            source_index = int(source_index_value)
            # Python 整数, 当前完整来源 blob 在 `voxel_index_global_zyx` 和 `source_probability` 第一维上的半开区间端点.
            voxel_begin = int(blobs["voxel_offsets"][source_index])
            voxel_end = int(blobs["voxel_offsets"][source_index + 1])
            # int32, (K_full, 3), 当前完整来源 blob 的完整图整数体素索引; 最后一维按 ZYX 排列, 单位 voxel.
            full_global_zyx = np.asarray(
                blobs["voxel_index_global_zyx"][voxel_begin:voxel_end],
                dtype=np.int32,
            )
            # float32, (K_full,), 完整图前向在当前完整来源 blob 体素上的配体概率, 与 full_global_zyx 第一维逐体素对齐.
            full_source_probability = np.asarray(
                blobs["source_probability"][voxel_begin:voxel_end],
                dtype=np.float32,
            )
            # int32, (3,), 当前 80³ BOX 在完整图中的整数起点; 三个分量按 ZYX 排列, 单位 voxel.
            box_start = np.asarray(
                box_starts[source_index],
                dtype=np.int32,
            )
            # bool, (K_full,), True 表示对应 full_global_zyx 体素位于当前 80³ BOX 的三个半开轴区间内, 并遮盖完整来源体素轴.
            source_in_box = np.all(
                (full_global_zyx >= box_start[None, :])
                & (full_global_zyx < box_start[None, :] + 80),
                axis=1,
            )
            # int32, (K_box, 3), 当前来源 blob 中实际落入 80³ BOX 的完整图整数体素索引; 最后一维按 ZYX 排列.
            global_zyx = full_global_zyx[source_in_box]
            # float32, (K_box,), 完整图前向在 K_box 个归档来源体素上的配体概率, 与 global_zyx 第一维逐体素对齐.
            source_probability = full_source_probability[source_in_box]
            # int32, (K_box, 3), 归档来源体素相对当前 BOX 起点的局部整数索引; 最后一维按 ZYX 排列, 数值位于 [0, 80).
            local_zyx = global_zyx - box_start[None, :]
            # float32, (3,), 当前 BOX 完整图起点从 ZYX 重排为 XYZ 后的整数体素索引, 用于换算世界坐标.
            box_start_xyz = box_start[[2, 1, 0]].astype(np.float32)
            # dict[str, np.ndarray], 当前 centered 候选的共同归档字段; 稀疏 V 字段第一维长度为 K_box.
            candidate_entry: dict[str, np.ndarray] = {
                # int32 标量, 当前候选对应的来源 blob 编号, 数值索引 blobs 候选轴.
                "source_blob_index": np.asarray(source_index, dtype=np.int32),
                # int32, (3,), 当前 80³ BOX 的完整图 ZYX 起点, 单位 voxel.
                "box_start_zyx": box_start.astype(np.int32),
                # uint8, (3,), 当前 centered BOX 沿 ZYX 三轴的固定体素数.
                "box_shape_zyx": np.asarray((80, 80, 80), dtype=np.uint8),
                # float32, (3,), 当前 BOX 起点对应的世界 XYZ 坐标, 单位 Å. [3] + [3] * [3] -> [3]
                "box_origin_world": np.asarray(origin_xyz, dtype=np.float32)
                + box_start_xyz * np.asarray(voxel_size_xyz, dtype=np.float32),
                # float32, (3,), 完整密度图沿世界 XYZ 三轴的体素间距, 单位 Å/voxel.
                "voxel_size_world": np.asarray(voxel_size_xyz, dtype=np.float32),
                # float32 标量, 当前完整来源 blob 的平均配体概率.
                "source_probability_mean": np.asarray(
                    blobs["source_probability_mean"][source_index],
                    dtype=np.float32,
                ),
                # float32 标量, 生成当前 F-alpha 来源 blobs 时使用的语义概率阈值.
                "source_threshold_value": np.asarray(
                    blobs["source_threshold_value"][0],
                    dtype=np.float32,
                ),
                # bool 标量, True 表示 K_box 等于 K_full 且完整来源 blob 可由当前合法 80³ BOX 容纳.
                "source_blob_fits_centered_box": np.asarray(
                    fits[source_index],
                    dtype=np.bool_,
                ),
                # int32 标量, 当前完整来源 blob 在裁切前的体素数 K_full, 不因 K_box 较小而改变.
                "source_voxel_count": np.asarray(
                    blob_count[source_index],
                    dtype=np.int32,
                ),
                # int16, (K_box, 3), 归档来源体素的 BOX 局部 ZYX 整数索引, 单位 voxel.
                "voxel_index_local_zyx": local_zyx.astype(np.int16),
                # float32, (K_box,), 完整图前向在归档来源体素上的配体概率.
                "source_probability": source_probability,
                # float32, (K_box,), centered 前向在同一归档来源体素上的配体概率. [80, 80, 80] -> [K_box]
                "centered_probability": ligand_probability[batch_index][
                    tuple(local_zyx.T)
                ].astype(np.float32),
            }
            # float16, (K_box, C_voxel), centered V 特征在归档来源体素上的取值; NumPy 高级索引把 K_box 体素轴放到通道轴之前. [C_voxel, 80, 80, 80] -> [K_box, C_voxel]
            candidate_entry["voxel_final"] = voxel_final[
                batch_index,
                :,
                local_zyx[:, 0],
                local_zyx[:, 1],
                local_zyx[:, 2],
            ].astype(np.float16)
            # bool, (80, 80, 80), Dataset 为当前 BOX 生成的受体占据掩码; True 表示受体占据体素, 三维按局部 ZYX 排列.
            hardmask = np.asarray(cpu_batch["hardmask"][batch_index], dtype=np.bool_)
            # int16, (K_aux, 3), hardmask=True 的受体占据体素局部整数索引; 最后一维按 ZYX 排列, 单位 voxel. [80, 80, 80] -> [K_aux, 3]
            aux_index = np.argwhere(hardmask).astype(np.int16)
            candidate_entry["voxel_aux_index_local_zyx"] = aux_index
            # float32, (K_aux,), centered 辅助前向在受体占据体素上的概率, 与 aux_index 第一维逐体素对齐. [80, 80, 80] -> [K_aux]
            candidate_entry["voxel_aux_probability"] = auxiliary[batch_index][
                tuple(aux_index.T)
            ].astype(np.float32)

            # float64, (3,), K_box 个归档来源体素的局部整数索引质心; 三个分量按 ZYX 排列, 单位 voxel. [K_box, 3] -> [3]
            centroid = local_zyx.astype(np.float64).mean(axis=0)
            # int64, (3,), 让来源体素中心的平均位置尽量对齐 48³ 裁块中心的理想局部 ZYX 起点, 单位 voxel.
            requested_crop = np.rint(centroid + 0.5 - 24.0).astype(np.int64)
            # int64, (3,), 48³ 裁块相对 80³ BOX 的合法局部 ZYX 起点; 每个分量独立限制到 [0, 32], 单位 voxel. [3] -> [3]
            crop_start = np.clip(requested_crop, 0, 32)
            # int64, (3,), 48³ 裁块在完整密度图中的 ZYX 起点, 单位 voxel. [3] + [3] -> [3]
            full_crop_start = box_start.astype(np.int64) + crop_start
            # tuple[slice, slice, slice], 依次切分完整图 Z/Y/X 三轴的长度 48 半开区间.
            crop_slices = tuple(
                slice(int(value), int(value) + 48) for value in full_crop_start
            )
            # float32, (3,), 归档来源体素的局部 ZYX 整数索引质心, 单位 voxel.
            candidate_entry["v_centroid_local_zyx"] = centroid.astype(np.float32)
            # int16, (3,), 48³ 裁块相对 80³ BOX 的局部 ZYX 起点, 单位 voxel.
            candidate_entry["crop_start_local_zyx"] = crop_start.astype(np.int16)
            # float32, (3,), 来源体素索引质心减去 48³ 裁块中心 `crop_start + 23.5`, 三个分量按 ZYX 排列, 单位 voxel. [3] - [3] -> [3]
            candidate_entry["crop_center_offset_zyx"] = (
                centroid - (crop_start.astype(np.float64) + 23.5)
            ).astype(np.float32)
            # bool, (3,), True 表示对应 Z/Y/X 轴的理想 48³ 起点被 80³ BOX 边界截断, 并遮盖三个空间轴.
            candidate_entry["crop_clipped_axis_mask"] = (
                crop_start != requested_crop
            ).astype(np.bool_)
            # float32, (48, 48, 48), 当前裁块的实验密度, 三维按完整图 ZYX 排列. [D, H, W] -> [48, 48, 48]
            candidate_entry["experimental_density_48"] = np.asarray(
                experimental[crop_slices],
                dtype=np.float32,
            )
            # float32, (48, 48, 48), 当前裁块的模拟密度, 三维按完整图 ZYX 排列. [D, H, W] -> [48, 48, 48]
            candidate_entry["simulated_density_48"] = np.asarray(
                simulated[crop_slices],
                dtype=np.float32,
            )
            # float32, (48, 48, 48), 当前裁块的完整图配体概率, 三维按完整图 ZYX 排列. [D, H, W] -> [48, 48, 48]
            candidate_entry["source_probability_48"] = np.asarray(
                full_probability[crop_slices],
                dtype=np.float32,
            )
            if not is_find:
                candidate_entries.append(candidate_entry)
                continue
            # slice, 当前候选在模型输出 `atom_global_indices`, `atom_coord_local_voxel`, `atom_logits` 和 `A_feat_L1:L3` 第一维上的半开区间.
            atom_slice = slice(
                int(atom_offsets[batch_index]),
                int(atom_offsets[batch_index + 1]),
            )
            # int64, (N_A_output_entry,), 当前候选模型输出 A 原子在 `receptor_tokens.npz` 原子轴上的编号.
            output_global = np.asarray(
                host_arrays["atom_global_indices"],
                dtype=np.int64,
            )[atom_slice]
            # slice, 当前候选在 Dataset 输入 `atom_global_indices`, `atom_feat` 和 `atom_is_backbone` 第一维上的半开区间.
            input_slice = slice(
                int(input_atom_offsets[batch_index]),
                int(input_atom_offsets[batch_index + 1]),
            )
            # int64, (N_A_input_entry,), 当前候选 Dataset 输入 A 原子在 `receptor_tokens.npz` 原子轴上的编号.
            input_global = np.asarray(
                cpu_batch["atom_global_indices"],
                dtype=np.int64,
            )[input_slice]
            # float32, (N_A_input_entry, 50), 按 Dataset 输入原子顺序拼接的 49 维基础特征和 1 维主链 0/1 标志. [N_A_input_entry, 49] + [N_A_input_entry, 1] -> [N_A_input_entry, 50]
            input_l0 = np.concatenate(
                (
                    np.asarray(cpu_batch["atom_feat"], dtype=np.float32)[input_slice],
                    np.asarray(
                        cpu_batch["atom_is_backbone"],
                        dtype=np.float32,
                    )[input_slice, None],
                ),
                axis=1,
            )
            # dict[int, int], `receptor_tokens.npz` 全局原子编号到当前候选 Dataset 输入紧凑编号的映射.
            input_index_by_global_atom = {
                int(value): index for index, value in enumerate(input_global.tolist())
            }
            # int64, (N_A_output_entry,), 每个模型输出 A 原子对应的 Dataset 输入紧凑编号, 数值索引 input_l0 的第一维.
            output_to_input_index = np.asarray(
                [input_index_by_global_atom[int(value)] for value in output_global],
                dtype=np.int64,
            )
            # float32, (N_A_output_entry, 50), 按模型输出 A 原子顺序重排的 Dataset 基础特征和主链标志. [N_A_input_entry, 50] -> [N_A_output_entry, 50]
            l0 = input_l0[output_to_input_index]

            # float32, (N_A_output_entry, 3), 模型输出 A 原子相对当前 BOX 起点的 XYZ 体素坐标, 单位 voxel.
            atom_local = np.asarray(
                host_arrays["atom_coord_local_voxel"], dtype=np.float32
            )[atom_slice]
            # float32, (N_A_output_entry, 3), 模型输出 A 原子相对 BOX 局部坐标 (40, 40, 40) 的世界 XYZ 位移, 单位 Å. [N_A_output_entry, 3] * [1, 3] -> [N_A_output_entry, 3]
            atom_centered = (atom_local - np.float32(40.0)) * np.asarray(
                voxel_size_xyz, dtype=np.float32
            )[None, :]

            # float32, (K_box, 3), 归档来源体素中心相对 BOX 起点的世界 XYZ 坐标, 单位 Å; `+ 0.5` 从整数体素索引移动到体素中心. [K_box, 3] * [1, 3] -> [K_box, 3]
            source_centers_xyz = (
                local_zyx[:, [2, 1, 0]].astype(np.float32) + np.float32(0.5)
            ) * np.asarray(voxel_size_xyz, dtype=np.float32)[None, :]
            # float64, (N_A_output_entry,), 每个模型输出 A 原子到最近归档来源体素中心的欧氏距离, 单位 Å. [N_A_output_entry, 3] 对 [K_box, 3] 最近邻查询 -> [N_A_output_entry]
            distance, _ = cKDTree(source_centers_xyz).query(
                atom_local * np.asarray(voxel_size_xyz, dtype=np.float32)[None, :],
                k=1,
            )
            # bool, (N_A_output_entry,), True 表示对应模型输出 A 原子的三个局部 XYZ 分量均位于当前 80³ BOX 的半开范围 [0, 80), 并遮盖模型输出 A 原子轴.
            in_core = np.all((atom_local >= 0.0) & (atom_local < 80.0), axis=1)
            # bool, (N_A_output_entry,), True 表示对应 A 原子同时位于核心 BOX 内且到最近归档来源体素中心的距离不超过 10 Å, 并同步遮盖全部 atom_fields 的第一维.
            keep = in_core & (distance <= 10.0)
            # dict[str, np.ndarray], 与 N_A_output_entry 模型输出 A 原子轴对齐的编号, 坐标, 概率和 L0:L3 特征.
            atom_fields = {
                "A_global_index": output_global,
                "A_coord_local_xyz": atom_local,
                "A_coord_centered_world": atom_centered,
                "A_probability": 1.0
                / (
                    1.0
                    + np.exp(
                        -np.asarray(host_arrays["atom_logits"], dtype=np.float32)[
                            atom_slice, 0
                        ]
                    )
                ),
                "A_feat_L0": l0,
                "A_feat_L1": host_arrays["A_feat_L1"][atom_slice],
                "A_feat_L2": host_arrays["A_feat_L2"][atom_slice],
                "A_feat_L3": host_arrays["A_feat_L3"][atom_slice],
            }
            # 每个 A 字段从 N_A_output_entry 同步筛到 N_A, 保持模型输出原子顺序和逐原子对齐关系.
            candidate_entry.update(
                {name: value[keep] for name, value in atom_fields.items()}
            )

            # bool, (N_P_batch,), True 表示对应 P 锚点属于当前 batch_index 候选, 并同步遮盖全部模型输出 P 字段的第一维.
            candidate_p_mask = (
                np.asarray(host_arrays["anchor_batch_index"], dtype=np.int64)
                == batch_index
            )
            # float32, (N_P, 3), 当前候选 P 锚点相对 BOX 起点的 XYZ 体素坐标, 单位 voxel. [N_P_batch, 3] -> [N_P, 3]
            candidate_entry["P_coord_local_xyz"] = np.asarray(
                host_arrays["anchor_coord_local_voxel"], dtype=np.float32
            )[candidate_p_mask]
            # float32, (N_P,), 当前候选 P 锚点 logits 的 sigmoid 概率, 与 P_coord_local_xyz 第一维逐点对齐. [N_P_batch, 1] -> [N_P]
            candidate_entry["P_probability"] = 1.0 / (
                1.0
                + np.exp(
                    -np.asarray(host_arrays["pseudo_logits"], dtype=np.float32)[
                        candidate_p_mask, 0
                    ]
                )
            )
            # (N_P, C_P2), 当前候选 P 锚点的第二层学习特征, 与 P_probability 逐点对齐. [N_P_batch, C_P2] -> [N_P, C_P2]
            candidate_entry["P_feat_L2"] = host_arrays["P_feat_L2"][
                candidate_p_mask
            ]
            # (N_P, C_P3), 当前候选 P 锚点的第三层学习特征, 与 P_probability 逐点对齐. [N_P_batch, C_P3] -> [N_P, C_P3]
            candidate_entry["P_feat_L3"] = host_arrays["P_feat_L3"][
                candidate_p_mask
            ]
            candidate_entries.append(candidate_entry)
        return candidate_entries
    # 请求物化, GPU 前向和 CPU 字段整理形成三个相邻阶段; 两个队列都保持提交顺序.
    materializer = ThreadPoolExecutor(
        max_workers=int(centered_workers),
        thread_name_prefix="stage1-centered",
    )
    arranger = ThreadPoolExecutor(max_workers=1, thread_name_prefix="stage1-pack")
    # prepared: 按 batch 顺序保存来源 blob 下标和逐请求物化 Future; batch 内顺序保持不变.
    prepared: deque[
        tuple[np.ndarray, tuple[Future[dict[str, Any]], ...]]
    ] = deque()
    # pending: 按提交顺序保存等待 D2H 或 CPU 字段整理的候选列表 Future.
    pending: deque[Future[list[dict[str, np.ndarray]]]] = deque()
    # completed_entries: 已按来源 blob 顺序完成整理的 centered 候选字段映射.
    completed_entries: list[dict[str, np.ndarray]] = []
    next_batch = 0
    try:
        # target_device: 当前模型前向与 autocast 使用的显式 PyTorch 设备.
        target_device = torch.device(device)
        with torch.inference_mode():
            while next_batch < len(batches) or prepared:
                while (
                    next_batch < len(batches)
                    and len(prepared) < int(prefetch_batches)
                ):
                    # source_indices: int32, (B_entry,), 当前预取 batch 的来源 blob 轴下标.
                    source_indices = batches[next_batch]
                    # requests: 长度 B_entry 的 80³ centered 请求, 顺序与 source_indices 一致.
                    requests = tuple(
                        ResolvedStage1Crop(
                            pdb_id=pdb_id,
                            box_start_zyx=tuple(
                                int(value) for value in box_starts[index]
                            ),
                            require_targets=False,
                            role="centered",
                        )
                        for index in source_indices
                    )
                    # sample_futures: 与 requests 逐项对齐; 单请求物化可占用不同 CPU worker.
                    sample_futures = tuple(
                        materializer.submit(dataset.materialize_request, request)
                        for request in requests
                    )
                    prepared.append((source_indices, sample_futures))
                    next_batch += 1
                wait_started_at = time.perf_counter()
                source_indices, sample_futures = prepared.popleft()
                # cpu_batch: collator 按来源候选顺序拼装的 Stage1 模型字段映射.
                cpu_batch = collator([future.result() for future in sample_futures])
                if target_device.type == "cuda":
                    cpu_batch = {
                        name: value.pin_memory() if torch.is_tensor(value) else value
                        for name, value in cpu_batch.items()
                    }
                materialize_wait_seconds += time.perf_counter() - wait_started_at
                # CPU batch 按当前显式设备搬运; CUDA 张量利用锁页内存执行非阻塞复制.
                # model_batch: 与 cpu_batch 同字段的张量映射, 张量已搬到 target_device.
                model_batch = {
                    name: (
                        value.to(target_device, non_blocking=True)
                        if torch.is_tensor(value) and target_device.type == "cuda"
                        else (
                            value.to(target_device) if torch.is_tensor(value) else value
                        )
                    )
                    for name, value in cpu_batch.items()
                }
                # autocast_dtype: CUDA 混合精度使用的计算 dtype.
                autocast_dtype = (
                    torch.bfloat16 if precision == "bf16" else torch.float16
                )
                with torch.autocast(
                    device_type=target_device.type,
                    dtype=autocast_dtype,
                    enabled=target_device.type == "cuda"
                    and precision in {"bf16", "float16"},
                ):
                    # forward: 模型完整前向字段映射, batch 轴长度为 B_entry.
                    forward = wrapper(model_batch)
                    # output: 当前 producer 归档所需的模型字段子集.
                    output = {
                        "voxel_logits_ligand": forward["voxel_logits_ligand"],
                        "voxel_logits_aux": forward["voxel_logits_aux"],
                        "voxel_final": forward["voxel_features"]["voxel_final"],
                    }
                    if is_find:
                        for name in (
                            "atom_counts",
                            "atom_global_indices",
                            "atom_coord_local_voxel",
                            "atom_logits",
                            "A_feat_L1",
                            "A_feat_L2",
                            "A_feat_L3",
                            "anchor_batch_index",
                            "anchor_coord_local_voxel",
                            "pseudo_logits",
                            "P_feat_L2",
                            "P_feat_L3",
                        ):
                            output[name] = forward[name]
                # GPU 输出先异步复制到锁页 CPU 张量, 整理线程再等待同一 stream 的 event.
                # host_output: 模型输出名到 CPU 张量的映射, 字段形状与 output 保持一致.
                host_output: dict[str, torch.Tensor] = {}
                for name, value in output.items():
                    if value.device.type == "cpu":
                        host_output[name] = value
                    else:
                        # host: 与当前 GPU 输出同形状和 dtype 的锁页 CPU 张量.
                        host = torch.empty(
                            value.shape,
                            dtype=value.dtype,
                            device="cpu",
                            pin_memory=True,
                        )
                        host.copy_(value, non_blocking=True)
                        host_output[name] = host
                # event: 当前 CUDA stream 完成全部异步 D2H 后触发的批次就绪事件; CPU 推理为 None.
                event = torch.cuda.Event() if target_device.type == "cuda" else None
                if event is not None:
                    event.record(torch.cuda.current_stream(target_device))
                while len(pending) >= int(pending_cpu_batches):
                    wait_started_at = time.perf_counter()
                    completed_entries.extend(pending.popleft().result())
                    cpu_arrange_wait_seconds += time.perf_counter() - wait_started_at
                pending.append(
                    arranger.submit(
                        arrange_batch,
                        host_output,
                        cpu_batch,
                        source_indices,
                        event,
                    )
                )
        while pending:
            wait_started_at = time.perf_counter()
            completed_entries.extend(pending.popleft().result())
            cpu_arrange_wait_seconds += time.perf_counter() - wait_started_at
    finally:
        materializer.shutdown(wait=True, cancel_futures=True)
        arranger.shutdown(wait=True, cancel_futures=True)

    # concatenate 与 NPZ 字段组装交给跨 PDB 共享的线程池, 使其能和下一 PDB 的 GPU 前向重叠.
    # packed: 把 completed_entries 压成正式 NPZ 字段映射的异步 Future.
    packed = packer.submit(
        pack_centered_entries,
        completed_entries,
        producer,
    )
    return (
        packed,
        {
            "wall_seconds": time.perf_counter() - started_at,
            "materialize_wait_seconds": materialize_wait_seconds,
            "cpu_arrange_wait_seconds": cpu_arrange_wait_seconds,
            "batch_count": len(batches),
            "entry_count": len(completed_entries),
        },
    )
