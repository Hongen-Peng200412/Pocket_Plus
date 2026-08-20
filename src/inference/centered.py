# -*- coding: utf-8 -*-
"""重新运行候选 80³ BOX 并生成 F1 basic 或 F3 centered 归档.

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
    save_auxiliary: bool,
    save_voxel_final: bool,
    save_dense48: bool,
    save_atom_point_tables: bool,
) -> dict[str, np.ndarray]:
    """把同一 PDB 的逐候选载荷压成无 object dtype 的数组.

    输入 ``entries`` 按 centered 条目顺序排列. 每项包含标量身份与几何, 来源
    blob 的局部 ZYX 体素, 来源/重算概率, 并按显式保存开关携带 auxiliary,
    V 特征, 48³ 密度及 Find A/P 表.

    候选级字段:
        - centered_box_index: int32 ``(N_entry,)``, 连续 centered 身份.
        - source_blob_index: int32 ``(N_entry,)``, 来源 blob 身份.
        - box_start_zyx: int32 ``(N_entry, 3)``, 完整图 ZYX 起点.
        - box_shape_zyx: uint8 ``(N_entry, 3)``, 固定为 80³.
        - box_origin_world: float32 ``(N_entry, 3)``, BOX 角点世界 XYZ.
        - voxel_size_world: float32 ``(N_entry, 3)``, 世界 XYZ 体素尺寸.
        - source_probability_mean: float32 ``(N_entry,)``, 来源 blob 平均概率.
        - source_threshold_value: float32 ``(N_entry,)``, 来源语义阈值.
        - score: float32 ``(N_entry,)``, 冻结候选分数.
        - selected: bool ``(N_entry,)``, 是否达到冻结分数与最小体素数.

    稀疏体素字段:
        - voxel_offsets: int64 ``(N_entry+1,)``, 同时切分 `voxel_index_local_zyx`, `source_probability`, `centered_probability` 和可选 `voxel_final`.
        - voxel_index_local_zyx: int16 ``(L_voxel, 3)``, 80³ BOX 内 ZYX 索引.
        - source_probability: float32 ``(L_voxel,)``, 完整图来源概率.
        - centered_probability: float32 ``(L_voxel,)``, centered 重算概率.
        - voxel_final: 可选 float16 ``(L_voxel, C_voxel)``, 来源体素的 V 学习特征.
        - voxel_aux_offsets: 可选 int64 ``(N_entry+1,)``, 切分 auxiliary 两个值表.
        - voxel_aux_index_local_zyx: 可选 int16 ``(L_aux, 3)``, auxiliary 受体体素的局部 ZYX 索引.
        - voxel_aux_probability: 可选 float32 ``(L_aux,)``, 独立辅助受体概率.

    Find A/P 字段:
        - A_offsets: int64 ``(N_entry+1,)``, 切分全部 A 值表.
        - A_global_index: int64 ``(N_A,)``, receptor token 全局原子身份.
        - A_coord_local_xyz: float32 ``(N_A, 3)``, BOX 局部 XYZ 体素坐标.
        - A_coord_centered_world: float32 ``(N_A, 3)``, centered 世界 XYZ 坐标.
        - A_probability: float32 ``(N_A,)``, A 原子概率.
        - A_feat_L0: float32 ``(N_A, 50)``, 49 维 token 与主链标志.
        - A_feat_L1: float16 ``(N_A, C_A1)``, 第一层 A 学习特征.
        - A_feat_L2: float16 ``(N_A, C_A2)``, 第二层 A 学习特征.
        - A_feat_L3: float16 ``(N_A, C_A3)``, 第三层 A 学习特征.
        - P_offsets: int64 ``(N_entry+1,)``, 切分全部 P 值表.
        - P_coord_local_xyz: float32 ``(N_P, 3)``, P 点局部 XYZ 体素坐标.
        - P_probability: float32 ``(N_P,)``, P 点概率.
        - P_feat_L2: float16 ``(N_P, C_P2)``, 第二层 P 学习特征.
        - P_feat_L3: float16 ``(N_P, C_P3)``, 第三层 P 学习特征.

    稠密 48³ 字段:
        - v_centroid_local_zyx: float32 ``(N_entry, 3)``, V 来源体素质心.
        - crop_start_local_zyx: int16 ``(N_entry, 3)``, 48³ 裁块局部 ZYX 起点.
        - crop_center_offset_zyx: float32 ``(N_entry, 3)``, 裁块中心相对质心偏移.
        - crop_clipped_axis_mask: bool ``(N_entry, 3)``, 起点被边界裁切的轴.
        - experimental_density_48: float32 ``(N_entry, 48, 48, 48)``, 实验密度裁块.
        - simulated_density_48: float32 ``(N_entry, 48, 48, 48)``, 模拟密度裁块.
        - source_probability_48: float32 ``(N_entry, 48, 48, 48)``, 完整图概率裁块.

    所有 offsets 都是半开边界并保持候选顺序. 完全没有候选时, 未知学习
    特征宽度用零表示, 且不产生 object dtype.
    """

    metadata_dtypes = {
        "source_blob_index": np.int32,
        "box_start_zyx": np.int32,
        "box_shape_zyx": np.uint8,
        "box_origin_world": np.float32,
        "voxel_size_world": np.float32,
        "source_probability_mean": np.float32,
        "source_threshold_value": np.float32,
        "score": np.float32,
        "selected": np.bool_,
    }
    arrays: dict[str, np.ndarray] = {}
    for field, dtype in metadata_dtypes.items():
        if entries:
            arrays[field] = np.stack(
                [np.asarray(entry[field], dtype=dtype) for entry in entries],
                axis=0,
            )
        elif field in {"box_start_zyx", "box_shape_zyx", "box_origin_world", "voxel_size_world"}:
            arrays[field] = np.empty((0, 3), dtype=dtype)
        else:
            arrays[field] = np.empty((0,), dtype=dtype)
    arrays["centered_box_index"] = np.arange(len(entries), dtype=np.int32)

    ragged_groups = (
        (
            "voxel_offsets",
            (
                ("voxel_index_local_zyx", np.int16, (3,)),
                ("source_probability", np.float32, ()),
                ("centered_probability", np.float32, ()),
            ),
        ),
    )
    if save_auxiliary:
        ragged_groups += (
            (
                "voxel_aux_offsets",
                (
                    ("voxel_aux_index_local_zyx", np.int16, (3,)),
                    ("voxel_aux_probability", np.float32, ()),
                ),
            ),
        )
    if save_atom_point_tables:
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
        first_field = fields[0][0]
        counts = np.asarray(
            [np.asarray(entry[first_field]).shape[0] for entry in entries],
            dtype=np.int64,
        )
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

    if save_voxel_final:
        arrays["voxel_final"] = (
            np.concatenate(
                [np.asarray(entry["voxel_final"], dtype=np.float16) for entry in entries],
                axis=0,
            )
            if entries
            else np.empty((0, 0), dtype=np.float16)
        )
    if save_dense48:
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
    min_voxels: int,
    centered_forward: str,
    save_voxel_final: bool,
    save_dense48: bool,
    packer: Executor,
) -> tuple[Future[dict[str, np.ndarray]], dict[str, float | int]]:
    """对符合 BOX 包络和体素数要求的候选执行 centered 推理.

    输入的 ``blobs`` 使用 ``voxel_offsets`` 和完整图 ZYX 坐标值表. F1 basic
    以 ``centered_forward='voxel_only'`` 调用最短概率路径; F3 centered 以
    ``centered_forward='full'`` 取得 V 特征, Find producer 另外取得 A/P 表.
    ``save_voxel_final`` 与 ``save_dense48`` 必须由命令或配置显式传入.

    CPU 线程提前物化下一批请求. GPU 只由当前线程访问, 所需输出异步复制到
    页锁定 CPU 张量后, 单独 CPU 线程按提交顺序整理候选载荷. 该并行边界不
    改变候选顺序或单个候选内部的实体顺序.

    返回值第一项是由 ``packer`` 执行 :func:`pack_centered_entries` 的 Future;
    它允许最终 concatenate 与下一个 PDB 的 GPU 前向重叠. 第二项是只供
    ``status/<role>/performance.json`` 使用的映射, 精确包含
    `wall_seconds: float`, `materialize_wait_seconds: float`,
    `cpu_arrange_wait_seconds: float`, `batch_count: int` 和 `entry_count: int`.
    正式字段中的 BOX 起点为完整图 ZYX 索引;
    原子局部坐标与世界坐标使用 XYZ. BF16 host tensor 在进入 NumPy 前转成
    float32, 随后由字段契约决定是否落盘为 float16.
    """

    from src.datasets.stage1_requests import ResolvedStage1Crop

    started_at = time.perf_counter()
    materialize_wait_seconds = 0.0
    cpu_arrange_wait_seconds = 0.0
    blob_count = np.asarray(blobs["voxel_count"], dtype=np.int64)
    fits = np.asarray(blobs["fits_centered_box"], dtype=np.bool_)
    eligible = np.flatnonzero(fits & (blob_count >= int(min_voxels))).astype(np.int32)
    batches = tuple(
        eligible[offset : offset + int(centered_batch_size)]
        for offset in range(0, eligible.shape[0], int(centered_batch_size))
    )
    save_atom_point_tables = producer.startswith("Find_") and centered_forward == "full"

    density_root = Path(dataset.root) / "density" / str(pdb_id).lower()
    experimental = (
        np.load(density_root / "exp.npy", mmap_mode="r", allow_pickle=False)[0]
        if save_dense48
        else None
    )
    simulated = (
        np.load(density_root / "sim.npy", mmap_mode="r", allow_pickle=False)[0]
        if save_dense48
        else None
    )

    def materialize_batch(
        source_indices: np.ndarray,
    ) -> tuple[dict[str, Any], np.ndarray]:
        requests = [
            ResolvedStage1Crop(
                pdb_id=pdb_id,
                box_start_zyx=tuple(int(value) for value in blobs["centered_box_start_zyx"][index]),
                require_targets=False,
                role="centered",
            )
            for index in source_indices
        ]
        batch = collator(
            [dataset.materialize_request(request) for request in requests]
        )
        if torch.device(device).type == "cuda":
            batch = {
                name: value.pin_memory() if torch.is_tensor(value) else value
                for name, value in batch.items()
            }
        return batch, source_indices

    def copy_to_host(value: Any) -> Any:
        if torch.is_tensor(value):
            if value.device.type == "cpu":
                return value
            host = torch.empty(
                value.shape,
                dtype=value.dtype,
                device="cpu",
                pin_memory=True,
            )
            host.copy_(value, non_blocking=True)
            return host
        if isinstance(value, dict):
            return {name: copy_to_host(item) for name, item in value.items()}
        return value

    def arrange_batch(
        host_output: Mapping[str, Any],
        cpu_batch: Mapping[str, Any],
        source_indices: np.ndarray,
        ready_event: torch.cuda.Event | None,
    ) -> list[dict[str, np.ndarray]]:
        if ready_event is not None:
            ready_event.synchronize()

        def array(value: Any, dtype: Any | None = None) -> np.ndarray:
            if torch.is_tensor(value):
                value = value.detach()
                if value.dtype == torch.bfloat16:
                    value = value.to(torch.float32)
                value = value.numpy()
            return np.asarray(value, dtype=dtype)

        ligand = array(host_output["voxel_logits_ligand"], np.float32)
        ligand = 1.0 / (1.0 + np.exp(-ligand[:, 0]))
        auxiliary = (
            1.0 / (1.0 + np.exp(-array(host_output["voxel_logits_aux"], np.float32)[:, 0]))
            if centered_forward == "full"
            else None
        )
        voxel_final = (
            array(host_output["voxel_final"])
            if save_voxel_final
            else None
        )
        atom_counts = (
            array(host_output["atom_counts"], np.int64).reshape(-1)
            if save_atom_point_tables
            else np.empty((0,), dtype=np.int64)
        )
        atom_offsets = np.concatenate(
            (np.zeros(1, dtype=np.int64), np.cumsum(atom_counts, dtype=np.int64))
        )
        input_atom_counts = (
            array(cpu_batch["atom_counts"], np.int64).reshape(-1)
            if save_atom_point_tables
            else np.empty((0,), dtype=np.int64)
        )
        input_atom_offsets = np.concatenate(
            (np.zeros(1, dtype=np.int64), np.cumsum(input_atom_counts, dtype=np.int64))
        )
        entries: list[dict[str, np.ndarray]] = []
        for batch_index, source_index_value in enumerate(source_indices.tolist()):
            source_index = int(source_index_value)
            voxel_begin = int(blobs["voxel_offsets"][source_index])
            voxel_end = int(blobs["voxel_offsets"][source_index + 1])
            global_zyx = np.asarray(
                blobs["voxel_index_global_zyx"][voxel_begin:voxel_end],
                dtype=np.int32,
            )
            source_probability = np.asarray(
                blobs["source_probability"][voxel_begin:voxel_end],
                dtype=np.float32,
            )
            box_start = np.asarray(
                blobs["centered_box_start_zyx"][source_index],
                dtype=np.int32,
            )
            # int32 (K_source, 3), 从完整图 ZYX 转成当前 80³ BOX 的局部 ZYX 索引.
            local_zyx = global_zyx - box_start[None, :]
            box_start_xyz = box_start[[2, 1, 0]].astype(np.float32)
            entry: dict[str, np.ndarray] = {
                "source_blob_index": np.asarray(source_index, dtype=np.int32),
                "box_start_zyx": box_start.astype(np.int32),
                "box_shape_zyx": np.asarray((80, 80, 80), dtype=np.uint8),
                "box_origin_world": np.asarray(origin_xyz, dtype=np.float32)
                + box_start_xyz * np.asarray(voxel_size_xyz, dtype=np.float32),
                "voxel_size_world": np.asarray(voxel_size_xyz, dtype=np.float32),
                "source_probability_mean": np.asarray(
                    blobs["source_probability_mean"][source_index],
                    dtype=np.float32,
                ),
                "source_threshold_value": np.asarray(
                    blobs["source_threshold_value"][0],
                    dtype=np.float32,
                ),
                "score": np.asarray(blobs["source_probability_mean"][source_index], dtype=np.float32),
                "selected": np.asarray(False, dtype=np.bool_),
                "voxel_index_local_zyx": local_zyx.astype(np.int16),
                "source_probability": source_probability,
                "centered_probability": ligand[batch_index][tuple(local_zyx.T)].astype(np.float32),
            }
            if auxiliary is not None:
                hardmask = array(cpu_batch["hardmask"][batch_index], np.bool_)
                aux_index = np.argwhere(hardmask).astype(np.int16)
                entry["voxel_aux_index_local_zyx"] = aux_index
                entry["voxel_aux_probability"] = auxiliary[batch_index][
                    tuple(aux_index.T)
                ].astype(np.float32)
            if save_voxel_final and voxel_final is not None:
                entry["voxel_final"] = voxel_final[
                    batch_index,
                    :,
                    local_zyx[:, 0],
                    local_zyx[:, 1],
                    local_zyx[:, 2],
                ].T.astype(np.float16)

            if save_dense48 and experimental is not None and simulated is not None:
                centroid = local_zyx.astype(np.float64).mean(axis=0)
                requested_crop = np.rint(centroid + 0.5 - 24.0).astype(np.int64)
                # int64 (3,), 48³ 起点在当前 80³ BOX 内逐轴限制到 [0, 32].
                crop_start = np.clip(requested_crop, 0, 32)
                full_crop_start = box_start.astype(np.int64) + crop_start
                crop_slices = tuple(slice(int(value), int(value) + 48) for value in full_crop_start)
                entry["v_centroid_local_zyx"] = centroid.astype(np.float32)
                entry["crop_start_local_zyx"] = crop_start.astype(np.int16)
                entry["crop_center_offset_zyx"] = (
                    centroid - (crop_start.astype(np.float64) + 23.5)
                ).astype(np.float32)
                entry["crop_clipped_axis_mask"] = (crop_start != requested_crop).astype(np.bool_)
                entry["experimental_density_48"] = np.asarray(experimental[crop_slices], dtype=np.float32)
                entry["simulated_density_48"] = np.asarray(simulated[crop_slices], dtype=np.float32)
                entry["source_probability_48"] = np.asarray(full_probability[crop_slices], dtype=np.float32)

            if save_atom_point_tables:
                atom_slice = slice(int(atom_offsets[batch_index]), int(atom_offsets[batch_index + 1]))
                input_slice = slice(
                    int(input_atom_offsets[batch_index]),
                    int(input_atom_offsets[batch_index + 1]),
                )
                output_global = array(host_output["atom_global_indices"], np.int64)[atom_slice]
                input_global = array(cpu_batch["atom_global_indices"], np.int64)[input_slice]
                input_l0 = np.concatenate(
                    (
                        array(cpu_batch["atom_feat"], np.float32)[input_slice],
                        array(cpu_batch["atom_is_backbone"], np.float32)[input_slice, None],
                    ),
                    axis=1,
                )
                # 全局原子 identity 把模型输出 A 行映射回输入的 49+1 维基础特征.
                input_row = {int(value): row for row, value in enumerate(input_global.tolist())}
                l0 = input_l0[
                    np.asarray([input_row[int(value)] for value in output_global], dtype=np.int64)
                ]
                atom_local = array(host_output["atom_coord_local_voxel"], np.float32)[atom_slice]
                # float32 (N_A, 3), 以 BOX 中心为原点的世界 XYZ 坐标, 单位 Å.
                atom_centered = (
                    atom_local - np.float32(40.0)
                ) * np.asarray(voxel_size_xyz, dtype=np.float32)[None, :]
                source_centers_xyz = (
                    local_zyx[:, [2, 1, 0]].astype(np.float32) + np.float32(0.5)
                ) * np.asarray(voxel_size_xyz, dtype=np.float32)[None, :]
                # A 表只保留核心 BOX 内且到来源 blob 最近体素中心不超过 10 Å 的行.
                distance, _ = cKDTree(source_centers_xyz).query(
                    atom_local * np.asarray(voxel_size_xyz, dtype=np.float32)[None, :],
                    k=1,
                    distance_upper_bound=10.0,
                )
                in_core = np.all((atom_local >= 0.0) & (atom_local < 80.0), axis=1)
                keep = in_core & np.isfinite(distance)
                atom_fields = {
                    "A_global_index": output_global,
                    "A_coord_local_xyz": atom_local,
                    "A_coord_centered_world": atom_centered,
                    "A_probability": 1.0 / (
                        1.0 + np.exp(-array(host_output["atom_logits"], np.float32)[atom_slice, 0])
                    ),
                    "A_feat_L0": l0,
                    "A_feat_L1": array(host_output["A_feat_L1"])[atom_slice],
                    "A_feat_L2": array(host_output["A_feat_L2"])[atom_slice],
                    "A_feat_L3": array(host_output["A_feat_L3"])[atom_slice],
                }
                entry.update({name: value[keep] for name, value in atom_fields.items()})
                pseudo_rows = array(host_output["anchor_batch_index"], np.int64) == batch_index
                entry["P_coord_local_xyz"] = array(
                    host_output["anchor_coord_local_voxel"], np.float32
                )[pseudo_rows]
                entry["P_probability"] = 1.0 / (
                    1.0
                    + np.exp(
                        -array(host_output["pseudo_logits"], np.float32)[pseudo_rows, 0]
                    )
                )
                entry["P_feat_L2"] = array(host_output["P_feat_L2"])[pseudo_rows]
                entry["P_feat_L3"] = array(host_output["P_feat_L3"])[pseudo_rows]
            entries.append(entry)
        return entries

    materializer = ThreadPoolExecutor(
        max_workers=int(centered_workers),
        thread_name_prefix="stage1-centered",
    )
    arranger = ThreadPoolExecutor(max_workers=1, thread_name_prefix="stage1-pack")
    prepared: deque[Future[tuple[dict[str, Any], np.ndarray]]] = deque()
    pending: deque[Future[list[dict[str, np.ndarray]]]] = deque()
    completed_entries: list[dict[str, np.ndarray]] = []
    next_batch = 0
    try:
        while next_batch < min(int(prefetch_batches), len(batches)):
            prepared.append(materializer.submit(materialize_batch, batches[next_batch]))
            next_batch += 1
        target_device = torch.device(device)
        with torch.inference_mode():
            while prepared:
                wait_started_at = time.perf_counter()
                cpu_batch, source_indices = prepared.popleft().result()
                materialize_wait_seconds += time.perf_counter() - wait_started_at
                if next_batch < len(batches):
                    prepared.append(materializer.submit(materialize_batch, batches[next_batch]))
                    next_batch += 1
                model_batch = {
                    name: (
                        value.to(target_device, non_blocking=True)
                        if torch.is_tensor(value) and target_device.type == "cuda"
                        else value.to(target_device) if torch.is_tensor(value) else value
                    )
                    for name, value in cpu_batch.items()
                }
                autocast_dtype = torch.bfloat16 if precision == "bf16" else torch.float16
                with torch.autocast(
                    device_type=target_device.type,
                    dtype=autocast_dtype,
                    enabled=target_device.type == "cuda" and precision in {"bf16", "float16"},
                ):
                    if centered_forward == "voxel_only":
                        output = {"voxel_logits_ligand": wrapper.forward_voxel_probability(model_batch)}
                    else:
                        forward = wrapper(model_batch)
                        output = {
                            "voxel_logits_ligand": forward["voxel_logits_ligand"],
                            "voxel_logits_aux": forward["voxel_logits_aux"],
                        }
                        if save_voxel_final:
                            output["voxel_final"] = forward["voxel_features"]["voxel_final"]
                        if save_atom_point_tables:
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
                host_output = copy_to_host(output)
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

    packed = packer.submit(
        pack_centered_entries,
        completed_entries,
        centered_forward == "full",
        save_voxel_final,
        save_dense48,
        save_atom_point_tables,
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
