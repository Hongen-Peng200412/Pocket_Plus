# -*- coding: utf-8 -*-
"""重新运行候选 80³ BOX 并生成通用 F-alpha centered 归档.

主要入口 :func:`infer_centered_boxes` 直接编排 Dataset 物化, GPU 前向, 异步
D2H 和 CPU 归档整理. :func:`pack_centered_entries` 把逐候选字典转换为共享
offsets 的正式 NPZ 数组. 本模块不选择阈值, 也不计算最终评估指标.
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
    """把同一 PDB 的逐候选载荷压成无 object dtype 的数组.

    输入 ``entries`` 按 centered 候选顺序排列. 每项包含编号, 几何, 来源 blob 的局部 ZYX 体素, 来源/重算概率, `voxel_final`, 辅助受体体素, V-centered 48³ 几何, 实验密度裁块, 模拟密度裁块和完整图概率裁块.
    `producer` 以 `Find_` 开头时, 同一候选还包含 A/P 表.

    候选级字段:
        - centered_box_index: int32 ``(N_entry,)``, 连续 centered 编号.
        - source_blob_index: int32 ``(N_entry,)``, 指向同 alpha blobs 文件 `blob_index` 第一维.
        - box_start_zyx: int32 ``(N_entry, 3)``, 完整图 ZYX 起点.
        - box_shape_zyx: uint8 ``(N_entry, 3)``, 固定为 80³.
        - box_origin_world: float32 ``(N_entry, 3)``, BOX 角点世界 XYZ 坐标, 单位 Å.
        - voxel_size_world: float32 ``(N_entry, 3)``, 世界 XYZ 体素尺寸, 单位 Å/voxel.
        - source_probability_mean: float32 ``(N_entry,)``, 来源 blob 平均概率.
        - source_threshold_value: float32 ``(N_entry,)``, 来源语义阈值.

    稀疏体素字段:
        - voxel_offsets: int64 ``(N_entry+1,)``, 以半开区间同步切分 `voxel_index_local_zyx`, `source_probability`, `centered_probability` 与 `voxel_final`; 首值为 0, 末值为 L_voxel.
        - voxel_index_local_zyx: int16 ``(L_voxel, 3)``, 80³ BOX 内 ZYX 索引.
        - source_probability: float32 ``(L_voxel,)``, 完整图来源概率.
        - centered_probability: float32 ``(L_voxel,)``, centered 重算概率.
        - voxel_final: float16 ``(L_voxel, C_voxel)``, 来源体素的 V 学习特征.
        - voxel_aux_offsets: int64 ``(N_entry+1,)``, 以半开区间同步切分 `voxel_aux_index_local_zyx` 与 `voxel_aux_probability`; 首值为 0, 末值为 L_aux.
        - voxel_aux_index_local_zyx: int16 ``(L_aux, 3)``, auxiliary 受体体素的局部 ZYX 索引.
        - voxel_aux_probability: float32 ``(L_aux,)``, 独立辅助受体概率.

    Find A/P 字段:
        - A_offsets: int64 ``(N_entry+1,)``, 以半开区间同步切分 `A_global_index`, `A_coord_local_xyz`, `A_coord_centered_world`, `A_probability`, `A_feat_L0`, `A_feat_L1`, `A_feat_L2` 与 `A_feat_L3`; 首值为 0, 末值为 N_A.
        - A_global_index: int64 ``(N_A,)``, 指向 `receptor_tokens.npz` 第一维的原子编号.
        - A_coord_local_xyz: float32 ``(N_A, 3)``, BOX 局部 XYZ 体素坐标.
        - A_coord_centered_world: float32 ``(N_A, 3)``, A 原子相对 80³ BOX 世界中心的 XYZ 位移, 单位 Å.
        - A_probability: float32 ``(N_A,)``, A 原子概率.
        - A_feat_L0: float32 ``(N_A, 50)``, 49 维 token 与主链标志.
        - A_feat_L1: float16 ``(N_A, C_A1)``, 第一层 A 学习特征.
        - A_feat_L2: float16 ``(N_A, C_A2)``, 第二层 A 学习特征.
        - A_feat_L3: float16 ``(N_A, C_A3)``, 第三层 A 学习特征.
        - P_offsets: int64 ``(N_entry+1,)``, 以半开区间同步切分 `P_coord_local_xyz`, `P_probability`, `P_feat_L2` 与 `P_feat_L3`; 首值为 0, 末值为 N_P.
        - P_coord_local_xyz: float32 ``(N_P, 3)``, P 点局部 XYZ 体素坐标.
        - P_probability: float32 ``(N_P,)``, P 点概率.
        - P_feat_L2: float16 ``(N_P, C_P2)``, 第二层 P 学习特征.
        - P_feat_L3: float16 ``(N_P, C_P3)``, 第三层 P 学习特征.

    稠密 48³ 字段:
        - v_centroid_local_zyx: float32 ``(N_entry, 3)``, V 来源体素质心.
        - crop_start_local_zyx: int16 ``(N_entry, 3)``, 48³ 裁块局部 ZYX 起点.
        - crop_center_offset_zyx: float32 ``(N_entry, 3)``, 来源 blob 质心相对 48³ 裁块中心的 ZYX 偏移, 单位 voxel.
        - crop_clipped_axis_mask: bool ``(N_entry, 3)``, True 表示对应轴的起点受 80³ 边界限制, False 表示未受限.
        - experimental_density_48: float32 ``(N_entry, 48, 48, 48)``, 实验密度裁块, 后三轴按 ZYX 排列.
        - simulated_density_48: float32 ``(N_entry, 48, 48, 48)``, 模拟密度裁块, 后三轴按 ZYX 排列.
        - source_probability_48: float32 ``(N_entry, 48, 48, 48)``, 完整图概率裁块, 后三轴按 ZYX 排列.

    所有 offsets 都是半开边界并保持候选顺序. 完全没有候选时, 未知学习
    特征宽度用零表示, 且不产生 object dtype.
    """

    # is_find: True 表示归档除共同字段外还包含 A/P 表.
    is_find = str(producer).startswith("Find_")

    # metadata_dtypes: 候选级字段到正式落盘 dtype 的映射, 每个字段首轴均为 N_entry.
    metadata_dtypes = {
        "source_blob_index": np.int32,
        "box_start_zyx": np.int32,
        "box_shape_zyx": np.uint8,
        "box_origin_world": np.float32,
        "voxel_size_world": np.float32,
        "source_probability_mean": np.float32,
        "source_threshold_value": np.float32,
    }
    # arrays: 最终写入 centered NPZ 的字段映射, 不包含 object dtype.
    arrays: dict[str, np.ndarray] = {}
    for field, dtype in metadata_dtypes.items():
        if entries:
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
            arrays[field] = np.empty((0, 3), dtype=dtype)
        else:
            arrays[field] = np.empty((0,), dtype=dtype)
    # int32, (N_entry,), 从 0 连续编号的 centered 候选轴.
    arrays["centered_box_index"] = np.arange(len(entries), dtype=np.int32)

    # ragged_groups: offsets 字段及其同步切分的值表, 每个描述项包含正式 dtype 和空数组尾部形状.
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
        # first_field: 当前 ragged 组的基准值表, 其逐候选长度同时适用于组内全部字段.
        first_field = fields[0][0]
        # int64, (N_entry,), 当前 ragged 组内每个候选的实体数量.
        counts = np.asarray(
            [np.asarray(entry[first_field]).shape[0] for entry in entries],
            dtype=np.int64,
        )
        # int64, (N_entry + 1,), 以半开区间同步切分当前 ragged 组的全部值表.
        arrays[offsets_name] = np.concatenate(
            (np.zeros(1, dtype=np.int64), np.cumsum(counts, dtype=np.int64))
        )
        for field, dtype, empty_tail in fields:
            if entries:
                arrays[field] = np.concatenate(
                    [np.asarray(entry[field], dtype=dtype) for entry in entries],
                    axis=0,
                )
            else:
                tail = (0,) if empty_tail is None else empty_tail
                arrays[field] = np.empty((0, *tail), dtype=dtype)

    # float16, (L_voxel, C_voxel), 与 voxel_offsets 切分的来源体素逐项对齐的 V 学习特征.
    arrays["voxel_final"] = (
        np.concatenate(
            [np.asarray(entry["voxel_final"], dtype=np.float16) for entry in entries],
            axis=0,
        )
        if entries
        else np.empty((0, 0), dtype=np.float16)
    )
    # dense_fields: 所有候选固定保存 V-centered 几何, 实验密度, 模拟密度与完整图概率.
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
            arrays[field] = np.stack([entry[field] for entry in entries], axis=0)
        elif field in {
            "v_centroid_local_zyx",
            "crop_start_local_zyx",
            "crop_center_offset_zyx",
            "crop_clipped_axis_mask",
        }:
            dtype = np.bool_ if field == "crop_clipped_axis_mask" else np.float32
            if field == "crop_start_local_zyx":
                dtype = np.int16
            arrays[field] = np.empty((0, 3), dtype=dtype)
        else:
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
    """对符合 BOX 包络和体素数要求的候选执行 centered 推理.

    输入参数:
        - dataset: Stage1Dataset, 提供当前 PDB 的 80³ 请求物化和完整密度 NPY 根目录.
        - collator: `dataset.collate_fn`, 把同一 PDB 的请求列表拼成 Stage1 模型批次.
        - wrapper: Stage1 模型包装器, 每个候选都调用完整前向.
        - pdb_id: 字符串, 当前小写 PDB 标识.
        - producer: 字符串, `unet_*` 或一个 `Find_*` producer; Find 在共同字段外额外保存 A/P 表.
        - blobs.voxel_offsets: int64 `(N_blob + 1,)`, 同步切分 `voxel_index_global_zyx` 与 `source_probability`.
        - blobs.voxel_count: int32 `(N_blob,)`, 每个来源 blob 的体素数.
        - blobs.voxel_index_global_zyx: int32 `(L_voxel, 3)`, 完整图 ZYX 来源体素索引.
        - blobs.source_probability: float32 `(L_voxel,)`, 与来源体素逐项对齐的完整图概率.
        - blobs.source_probability_mean: float32 `(N_blob,)`, 每个来源 blob 的平均概率.
        - blobs.fits_centered_box: bool `(N_blob,)`, 是否存在可容纳来源 blob 的合法 80³ BOX.
        - blobs.centered_box_start_zyx: int32 `(N_blob, 3)`, 完整图 ZYX BOX 起点.
        - blobs.source_threshold_value: float32 `(1,)`, 当前角色的语义概率阈值.
        - full_probability: float32 `(D, H, W)` 完整图 ZYX 概率; 所有 producer 保存 `source_probability_48`.
        - origin_xyz: float32 `(3,)`, 完整图世界 XYZ 角点坐标, 单位 Å.
        - voxel_size_xyz: float32 `(3,)`, 世界 XYZ 体素尺寸, 单位 Å/voxel.
        - device: 字符串, 模型前向设备.
        - precision: 字符串, GPU autocast 精度; `bf16` 或 `float16` 开启混合精度.
        - centered_batch_size: int, 一次模型前向的候选数量.
        - centered_workers: int, CPU 请求物化线程数.
        - prefetch_batches: int, 尚未送入模型的最大预取 batch 数.
        - pending_cpu_batches: int, 已完成模型前向但尚未完成 CPU 整理的最大 batch 数.
        - forward_min_voxels: int, 进入 centered 前向的来源 blob 最小体素数, 包含端点; 不决定最终 `selected`.
        - packer: 跨 PDB 共享的 CPU Executor, 用于异步执行 :func:`pack_centered_entries`.

    输入的 ``blobs`` 使用 ``voxel_offsets`` 和完整图 ZYX 坐标值表. 所有 producer 都执行完整前向并保存 `voxel_final`, auxiliary, V-centered 几何, 实验密度裁块, 模拟密度裁块与完整图概率裁块;
    Find producer 另外保存 A/P 表. alpha 与前向字段集合相互独立.
    CPU 线程提前物化下一批请求. GPU 只由当前线程访问, 所需输出异步复制到页锁定 CPU 张量后, 单独 CPU 线程按提交顺序整理候选载荷. 该并行边界不改变候选顺序或单个候选内部的实体顺序.

    返回值: 
    第一项是由 ``packer`` 执行 :func:`pack_centered_entries` 的 Future; 它允许最终 concatenate 与下一个 PDB 的 GPU 前向重叠. 
    第二项是只供 ``status/<role>/performance.json`` 使用的映射:

        - wall_seconds: float, 当前 PDB centered 推理总墙钟秒数.
        - materialize_wait_seconds: float, 等待 CPU 请求物化的累计秒数.
        - cpu_arrange_wait_seconds: float, 等待 CPU 整理完成的累计秒数.
        - batch_count: int, 实际执行的 centered batch 数.
        - entry_count: int, 实际进入 centered 推理的候选数.

    正式字段中的 BOX 起点为完整图 ZYX 索引; 原子局部坐标与世界坐标使用 XYZ. BF16 host tensor 在进入 NumPy 前转成 float32, 随后由字段契约决定是否落盘为 float16.
    """
    from src.datasets.stage1_requests import ResolvedStage1Crop

    started_at = time.perf_counter()
    materialize_wait_seconds = 0.0
    cpu_arrange_wait_seconds = 0.0
    # int64, (N_blob,), 每个来源 blob 的体素数量.
    blob_count = np.asarray(blobs["voxel_count"], dtype=np.int64)
    # bool, (N_blob,), True 表示该来源 blob 能完整放入合法 80³ BOX.
    fits = np.asarray(blobs["fits_centered_box"], dtype=np.bool_)
    # int32, (N_entry,), 同时满足 BOX 包络和前向体素数门槛的来源 blob 轴下标.
    eligible = np.flatnonzero(fits & (blob_count >= int(forward_min_voxels))).astype(np.int32)
    # batches: 保持 eligible 顺序的候选下标分批, 每项为 int32 (B_entry,).
    batches = tuple(
        eligible[offset : offset + int(centered_batch_size)]
        for offset in range(0, eligible.shape[0], int(centered_batch_size))
    )
    # is_find: True 表示本次前向除共同字段外还需整理 A/P 表.
    is_find = producer.startswith("Find_")

    # density_root: 当前 PDB 完整实验密度和模拟密度 NPY 所在目录.
    density_root = Path(dataset.root) / "density" / str(pdb_id).lower()
    # experimental: 所有 producer 共用的只读 mmap, (D, H, W), 完整图实验密度, 三轴按 ZYX 排列.
    experimental = np.load(
        density_root / "exp.npy", mmap_mode="r", allow_pickle=False
    )[0]
    # simulated: 所有 producer 共用的只读 mmap, (D, H, W), 完整图模拟密度, 三轴按 ZYX 排列.
    simulated = np.load(
        density_root / "sim.npy", mmap_mode="r", allow_pickle=False
    )[0]


    # -----------------------------------------------------------------------------------------------------------
    def materialize_batch(
        source_indices: np.ndarray,
    ) -> tuple[dict[str, Any], np.ndarray]:
        """在线程池中物化一批来源 blob 对应的 80³ centered 请求.

        输入参数:
            - source_indices: int32 `(B_entry,)`, 数值索引 `blobs` 的候选轴, 顺序也是返回 batch 的候选顺序.

        返回值:
            - batch: `dataset.collate_fn` 拼装的 Stage1 模型批次; 字段契约由 `src/datasets/README.md` 定义, 本回调不改字段, CUDA 推理时张量已锁页.
            - batch.hardmask: bool `(B_entry, 80, 80, 80)`, 所有 producer 都存在; True 是受体占据体素, 用于确定 auxiliary 位置.
            - batch.atom_counts: Find 专用 int64 `(B_entry,)`, 同步切分 `atom_global_indices`, `atom_feat` 与 `atom_is_backbone`.
            - batch.atom_global_indices: Find 专用 int64 `(N_A_input,)`, 指向完整 receptor token 原子轴.
            - batch.atom_feat: Find 专用 float32 `(N_A_input, 49)`, Dataset 输入 A 原子基础特征.
            - batch.atom_is_backbone: Find 专用 bool `(N_A_input,)`, True 表示主链原子, False 表示非主链原子.
            - source_indices: int32 `(B_entry,)`, 原样返回的来源 blob 编号, 供 CPU 整理阶段恢复候选身份.
        """
        # requests: 长度 B_entry 的 80³ centered 请求, 顺序与 source_indices 完全一致.
        requests = [
            ResolvedStage1Crop(
                pdb_id=pdb_id,
                box_start_zyx=tuple(
                    int(value) for value in blobs["centered_box_start_zyx"][index]
                ),
                require_targets=False,
                role="centered",
            )
            for index in source_indices
        ]
        # batch: collator 拼装的 Stage1 张量映射, batch 轴长度为 B_entry.
        batch = collator([dataset.materialize_request(request) for request in requests])
        if torch.device(device).type == "cuda":
            batch = {
                name: value.pin_memory() if torch.is_tensor(value) else value
                for name, value in batch.items()
            }
        return batch, source_indices


    # -----------------------------------------------------------------------------------------------------------
    def arrange_batch(
        host_output: Mapping[str, Any],
        cpu_batch: Mapping[str, Any],
        source_indices: np.ndarray,
        ready_event: torch.cuda.Event | None,
    ) -> list[dict[str, np.ndarray]]:
        """等待一批 D2H 完成, 再按来源 blob 顺序构造 centered 候选字段.

        输入参数:
            - host_output.voxel_logits_ligand: `(B_entry, 1, 80, 80, 80)`, centered 配体 logits.
            - host_output.voxel_logits_aux: `(B_entry, 1, 80, 80, 80)`, 所有 producer 的受体辅助 logits.
            - host_output.voxel_final: `(B_entry, C_voxel, 80, 80, 80)`, 所有 producer 必存的 V 学习特征.
            - host_output.atom_counts: Find producer 专用 int64 `(B_entry,)` 模型输出 A 原子数.
            - host_output.atom_global_indices: Find producer 专用 int64 `(N_A_output,)` 全局原子编号.
            - host_output.atom_coord_local_voxel: Find producer 专用 `(N_A_output, 3)` BOX 局部 XYZ 体素坐标.
            - host_output.atom_logits: Find producer 专用 `(N_A_output, 1)` A 原子 logits.
            - host_output.A_feat_L1: Find producer 专用 `(N_A_output, C_A1)` 第一层 A 特征.
            - host_output.A_feat_L2: Find producer 专用 `(N_A_output, C_A2)` 第二层 A 特征.
            - host_output.A_feat_L3: Find producer 专用 `(N_A_output, C_A3)` 第三层 A 特征.
            - host_output.anchor_batch_index: Find producer 专用 int64 `(N_P,)` P 点所属 batch 编号.
            - host_output.anchor_coord_local_voxel: Find producer 专用 `(N_P, 3)` P 点 BOX 局部 XYZ 体素坐标.
            - host_output.pseudo_logits: Find producer 专用 `(N_P, 1)` P 点 logits.
            - host_output.P_feat_L2: Find producer 专用 `(N_P, C_P2)` 第二层 P 特征.
            - host_output.P_feat_L3: Find producer 专用 `(N_P, C_P3)` 第三层 P 特征.
            - cpu_batch.hardmask: bool `(B_entry, 80, 80, 80)`, True 是受体占据体素, False 是非受体体素.
            - cpu_batch.atom_counts: Find 专用 int64 `(B_entry,)`, 同步切分 `atom_global_indices`, `atom_feat` 与 `atom_is_backbone`.
            - cpu_batch.atom_global_indices: Find 专用 int64 `(N_A_input,)`, Dataset 输入全局原子编号.
            - cpu_batch.atom_feat: Find 专用 float32 `(N_A_input, 49)`, Dataset 输入 A 原子基础特征.
            - cpu_batch.atom_is_backbone: Find 专用 bool `(N_A_input,)`, True 表示主链原子, False 表示非主链原子.
            - source_indices: int32 `(B_entry,)`, 数值索引 `blobs` 的候选轴.
            - ready_event: CUDA event 或 None; 非 None 时先等待当前批次的异步 D2H 完成.

        返回值:
            - entries: 长度 B_entry 的候选字段映射列表, 顺序与 `source_indices` 一致; 每项字段精确遵循 :func:`pack_centered_entries` 的输入契约.
        """
        if ready_event is not None:
            ready_event.synchronize()

        # 每个模型输出字段转为 CPU NumPy 数组; BF16 先提升为 float32, 避免 NumPy 不支持 BF16.
        # host_arrays: 模型输出名到 CPU NumPy 数组的映射, 字段形状保持 host_output 不变.
        host_arrays: dict[str, np.ndarray] = {}
        for name, value in host_output.items():
            tensor = value.detach()
            if tensor.dtype == torch.bfloat16:
                tensor = tensor.to(torch.float32)
            host_arrays[name] = tensor.numpy()

        # float32, (B_entry, 80, 80, 80), 每个 centered BOX 的配体概率.
        ligand = np.asarray(host_arrays["voxel_logits_ligand"], dtype=np.float32)
        ligand = 1.0 / (1.0 + np.exp(-ligand[:, 0]))
        # float32, (B_entry, 80, 80, 80), 每个 centered BOX 的受体辅助概率.
        auxiliary = 1.0 / (
            1.0
            + np.exp(
                -np.asarray(host_arrays["voxel_logits_aux"], dtype=np.float32)[:, 0]
            )
        )
        # voxel_final: (B_entry, C_voxel, 80, 80, 80), 所有 producer 的 V 学习特征.
        voxel_final = host_arrays["voxel_final"]
        # int64, (B_entry,), Find 模型输出中每个候选的 A 原子数; 非 Find 为空数组.
        atom_counts = (
            np.asarray(host_arrays["atom_counts"], dtype=np.int64).reshape(-1)
            if is_find
            else np.empty((0,), dtype=np.int64)
        )
        # atom_offsets: int64, (B_entry + 1,), 同步切分 atom_global_indices, atom_coord_local_voxel, atom_logits, A_feat_L1, A_feat_L2 与 A_feat_L3; 首值为 0, 末值为输出 A 原子总数.
        atom_offsets = np.concatenate(
            (np.zeros(1, dtype=np.int64), np.cumsum(atom_counts, dtype=np.int64))
        )
        # int64, (B_entry,), Dataset 输入中每个候选的 A 原子数; 非 Find 为空数组.
        input_atom_counts = (
            np.asarray(cpu_batch["atom_counts"], dtype=np.int64).reshape(-1)
            if is_find
            else np.empty((0,), dtype=np.int64)
        )
        # input_atom_offsets: int64, (B_entry + 1,), 同步切分 atom_global_indices, atom_feat 与 atom_is_backbone; 首值为 0, 末值为输入 A 原子总数.
        input_atom_offsets = np.concatenate(
            (np.zeros(1, dtype=np.int64), np.cumsum(input_atom_counts, dtype=np.int64))
        )
        # entries: 按 source_indices 顺序收集的候选字段映射, 每项对应一个 centered BOX.
        entries: list[dict[str, np.ndarray]] = []
        for batch_index, source_index_value in enumerate(source_indices.tolist()):
            # source_index: 当前候选在来源 blobs 候选轴中的下标.
            source_index = int(source_index_value)
            # voxel_begin, voxel_end: 当前来源 blob 在稀疏体素值表中的半开区间端点.
            voxel_begin = int(blobs["voxel_offsets"][source_index])
            voxel_end = int(blobs["voxel_offsets"][source_index + 1])
            # int32, (K_source, 3), 当前来源 blob 在完整图中的 ZYX 体素索引.
            global_zyx = np.asarray(
                blobs["voxel_index_global_zyx"][voxel_begin:voxel_end],
                dtype=np.int32,
            )
            # float32, (K_source,), 与 global_zyx 逐体素对齐的完整图来源概率.
            source_probability = np.asarray(
                blobs["source_probability"][voxel_begin:voxel_end],
                dtype=np.float32,
            )
            # int32, (3,), 当前 80³ BOX 在完整图中的 ZYX 起点.
            box_start = np.asarray(
                blobs["centered_box_start_zyx"][source_index],
                dtype=np.int32,
            )
            # int32, (K_source, 3), 从完整图 ZYX 转成当前 80³ BOX 的局部 ZYX 索引.
            local_zyx = global_zyx - box_start[None, :]
            # float32, (3,), 同一 BOX 起点按 XYZ 顺序重排后的体素索引.
            box_start_xyz = box_start[[2, 1, 0]].astype(np.float32)
            # entry: 当前候选的通用字段映射, 体素值表长度为 K_source.
            entry: dict[str, np.ndarray] = {
                "source_blob_index": np.asarray(source_index, dtype=np.int32),
                "box_start_zyx": box_start.astype(np.int32),
                "box_shape_zyx": np.asarray((80, 80, 80), dtype=np.uint8),
                "box_origin_world": np.asarray(origin_xyz, dtype=np.float32) + box_start_xyz * np.asarray(voxel_size_xyz, dtype=np.float32),
                "voxel_size_world": np.asarray(voxel_size_xyz, dtype=np.float32),
                "source_probability_mean": np.asarray(
                    blobs["source_probability_mean"][source_index],
                    dtype=np.float32,
                ),
                "source_threshold_value": np.asarray(
                    blobs["source_threshold_value"][0],
                    dtype=np.float32,
                ),
                "voxel_index_local_zyx": local_zyx.astype(np.int16),
                "source_probability": source_probability,
                "centered_probability": ligand[batch_index][tuple(local_zyx.T)].astype(np.float32),
            }
            # float16, (K_source, C_voxel), NumPy 高级索引把来源体素轴放在通道轴之前.
            entry["voxel_final"] = voxel_final[
                batch_index,
                :,
                local_zyx[:, 0],
                local_zyx[:, 1],
                local_zyx[:, 2],
            ].astype(np.float16)
            # bool, (80, 80, 80), 当前 BOX 的受体占据掩码, 三轴按 ZYX 排列.
            hardmask = np.asarray(cpu_batch["hardmask"][batch_index], dtype=np.bool_)
            # int16, (K_aux, 3), hardmask 为 True 的 BOX 局部 ZYX 体素索引.
            aux_index = np.argwhere(hardmask).astype(np.int16)
            entry["voxel_aux_index_local_zyx"] = aux_index
            entry["voxel_aux_probability"] = auxiliary[batch_index][tuple(aux_index.T)].astype(np.float32)

            # float64, (3,), 来源 blob 在当前 80³ BOX 中的 ZYX 体素质心.
            centroid = local_zyx.astype(np.float64).mean(axis=0)
            requested_crop = np.rint(centroid + 0.5 - 24.0).astype(np.int64)
            # int64, (3,), 48³ 起点在当前 80³ BOX 内逐轴限制到 [0, 32].
            crop_start = np.clip(requested_crop, 0, 32)
            # int64, (3,), 48³ 裁块在完整图中的 ZYX 起点.
            full_crop_start = box_start.astype(np.int64) + crop_start
            # crop_slices: 三个长度为 48 的完整图 ZYX 半开切片.
            crop_slices = tuple(slice(int(value), int(value) + 48) for value in full_crop_start)
            entry["v_centroid_local_zyx"] = centroid.astype(np.float32)
            entry["crop_start_local_zyx"] = crop_start.astype(np.int16)
            entry["crop_center_offset_zyx"] = (centroid - (crop_start.astype(np.float64) + 23.5)).astype(np.float32)
            entry["crop_clipped_axis_mask"] = (crop_start != requested_crop).astype(np.bool_)
            entry["experimental_density_48"] = np.asarray(experimental[crop_slices], dtype=np.float32)
            entry["simulated_density_48"] = np.asarray(simulated[crop_slices], dtype=np.float32)
            entry["source_probability_48"] = np.asarray(full_probability[crop_slices], dtype=np.float32)
            if not is_find:
                entries.append(entry)
                continue


            # atom_slice: 当前候选在模型输出 A 原子值表中的半开区间.
            atom_slice = slice(int(atom_offsets[batch_index]), int(atom_offsets[batch_index + 1]))
            # int64, (N_A_output_entry,), 当前模型输出 A 原子的完整受体原子编号.
            output_global = np.asarray(host_arrays["atom_global_indices"], dtype=np.int64)[atom_slice]
            # input_slice: 当前候选在 Dataset 输入 A 原子值表中的半开区间.
            input_slice = slice(int(input_atom_offsets[batch_index]), int(input_atom_offsets[batch_index + 1]))
            # int64, (N_A_input_entry,), 当前 Dataset 输入 A 原子的完整受体原子编号.
            input_global = np.asarray(cpu_batch["atom_global_indices"], dtype=np.int64)[input_slice]
            # float32, (N_A_input_entry, 50), 49 维基础特征与 1 维主链标识拼接后的 A_feat_L0.
            input_l0 = np.concatenate(
                (
                    np.asarray(cpu_batch["atom_feat"], dtype=np.float32)[input_slice],
                    np.asarray(cpu_batch["atom_is_backbone"], dtype=np.float32)[input_slice, None],
                ),
                axis=1,
            )
            # 全局原子编号把模型输出 A 原子映射回 Dataset 输入的 49+1 维基础特征.
            input_row = {int(value): row for row, value in enumerate(input_global.tolist())}
            # float32, (N_A_output_entry, 50), 按模型输出原子顺序重排的 A_feat_L0.
            l0 = input_l0[
                np.asarray(
                    [input_row[int(value)] for value in output_global],
                    dtype=np.int64,
                )
            ]

            # float32, (N_A_output_entry, 3), 模型输出 A 原子的 BOX 局部 XYZ 体素坐标.
            atom_local = np.asarray(
                host_arrays["atom_coord_local_voxel"], dtype=np.float32
            )[atom_slice]
            # float32, (N_A, 3), 以 BOX 中心为原点的世界 XYZ 坐标, 单位 Å.
            atom_centered = (atom_local - np.float32(40.0)) * np.asarray(
                voxel_size_xyz, dtype=np.float32
            )[None, :]

            # float32, (K_source, 3), 来源体素中心相对 BOX 角点的世界 XYZ 坐标, 单位 Å.
            source_centers_xyz = (
                local_zyx[:, [2, 1, 0]].astype(np.float32) + np.float32(0.5)
            ) * np.asarray(voxel_size_xyz, dtype=np.float32)[None, :]
            # A 表只保留核心 BOX 内且到来源 blob 最近体素中心不超过 10 Å 的原子.
            # float64, (N_A_output_entry,), 每个输出 A 原子到来源 blob 最近体素中心的距离, 单位 Å.
            distance, _ = cKDTree(source_centers_xyz).query(
                atom_local * np.asarray(voxel_size_xyz, dtype=np.float32)[None, :],
                k=1,
            )
            # bool, (N_A_output_entry,), True 表示 A 原子 XYZ 体素坐标位于当前 80³ BOX 的核心范围 [0, 80) 内.
            in_core = np.all((atom_local >= 0.0) & (atom_local < 80.0), axis=1)
            # bool, (N_A_batch_entry,), 同步筛选核心 BOX 内且距离不超过 10 Å 的 A 字段.
            keep = in_core & (distance <= 10.0)
            # atom_fields: 与输出 A 原子轴对齐的坐标, 概率和 L0-L3 特征, 随后由 keep 同步筛选.
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
            entry.update({name: value[keep] for name, value in atom_fields.items()})

            # bool, (N_P_batch,), 同步筛选属于当前候选的 P 坐标, 概率和 L2/L3 特征.
            pseudo_rows = (
                np.asarray(host_arrays["anchor_batch_index"], dtype=np.int64)
                == batch_index
            )
            # float32, (N_P_entry, 3), 当前候选 P 点的 BOX 局部 XYZ 体素坐标.
            entry["P_coord_local_xyz"] = np.asarray(
                host_arrays["anchor_coord_local_voxel"], dtype=np.float32
            )[pseudo_rows]
            # float32, (N_P_entry,), 与 P_coord_local_xyz 逐点对齐的模型概率.
            entry["P_probability"] = 1.0 / (
                1.0
                + np.exp(
                    -np.asarray(host_arrays["pseudo_logits"], dtype=np.float32)[
                        pseudo_rows, 0
                    ]
                )
            )
            entry["P_feat_L2"] = host_arrays["P_feat_L2"][pseudo_rows]
            entry["P_feat_L3"] = host_arrays["P_feat_L3"][pseudo_rows]
            entries.append(entry)
        return entries


    # 请求物化, GPU 前向和 CPU 字段整理形成三个相邻阶段; 两个队列都保持提交顺序.
    materializer = ThreadPoolExecutor(
        max_workers=int(centered_workers),
        thread_name_prefix="stage1-centered",
    )
    arranger = ThreadPoolExecutor(max_workers=1, thread_name_prefix="stage1-pack")
    # prepared: 按提交顺序保存尚未送入模型的 CPU batch 与来源 blob 下标 Future.
    prepared: deque[Future[tuple[dict[str, Any], np.ndarray]]] = deque()
    # pending: 按提交顺序保存等待 D2H 或 CPU 字段整理的候选列表 Future.
    pending: deque[Future[list[dict[str, np.ndarray]]]] = deque()
    # completed_entries: 已按来源 blob 顺序完成整理的 centered 候选字段映射.
    completed_entries: list[dict[str, np.ndarray]] = []
    next_batch = 0
    try:
        while next_batch < min(int(prefetch_batches), len(batches)):
            prepared.append(materializer.submit(materialize_batch, batches[next_batch]))
            next_batch += 1
        # target_device: 当前模型前向与 autocast 使用的显式 PyTorch 设备.
        target_device = torch.device(device)
        with torch.inference_mode():
            while prepared:
                wait_started_at = time.perf_counter()
                cpu_batch, source_indices = prepared.popleft().result()
                materialize_wait_seconds += time.perf_counter() - wait_started_at
                if next_batch < len(batches):
                    prepared.append(
                        materializer.submit(materialize_batch, batches[next_batch])
                    )
                    next_batch += 1
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
