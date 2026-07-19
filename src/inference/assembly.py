# -*- coding: utf-8 -*-
"""AdaLigand A—G 产物到 Stage1 正式推理 runner 的最小装配层。"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from src.datasets.box_geometry import build_hardmask_from_world_coordinates
from src.datasets.stage1_dataset import Stage1Dataset
from src.datasets.stage1_requests import ResolvedStage1Crop

from .centered import CenteredRequest
from .checkpoint import load_stage1_wrapper, resolve_checkpoint_config_path
from .full_map import window_starts_zyx
from .runner import FullMapTaskInput, ProductionTask


_MASK_KEY = re.compile(r"^mask_(\d+)$")


def load_pdb_id_list(path: str | Path) -> tuple[str, ...]:
    """读取固定 PDB 清单并保持文件内顺序。

    参数:
        path: JSON、JSONL 或纯文本路径。JSON 可为字符串列表、对象列表，或含
            ``pdb_ids`` 列表的对象；JSONL 每行可为字符串或含 ``pdb_id`` 的对象。

    返回:
        小写、非空且无重复的 PDB ID 元组。重复项会直接报错，避免一次固定清单
        在分片前后产生不清楚的去重语义。
    """
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"PDB 固定清单不存在: {source}")
    suffix = source.suffix.lower()
    if suffix == ".json":
        value = json.loads(source.read_text(encoding="utf-8"))
        if isinstance(value, dict):
            value = value.get("pdb_ids")
        if not isinstance(value, list):
            raise TypeError("JSON PDB 清单必须是列表或包含 pdb_ids 列表的对象")
        rows = value
    elif suffix == ".jsonl":
        rows = [
            json.loads(line)
            for line in source.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    else:
        rows = [line.strip() for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]

    pdb_ids: list[str] = []
    for row in rows:
        if isinstance(row, Mapping):
            if "pdb_id" not in row:
                raise KeyError("PDB 清单对象行缺少 pdb_id")
            row = row["pdb_id"]
        pdb_id = str(row).strip().lower()
        if not pdb_id:
            raise ValueError("PDB 清单不能包含空身份")
        pdb_ids.append(pdb_id)
    duplicates = sorted(
        pdb_id for pdb_id, count in Counter(pdb_ids).items() if count > 1
    )
    if duplicates:
        raise ValueError(f"PDB 固定清单含重复身份: {duplicates[:10]}")
    if not pdb_ids:
        raise ValueError("PDB 固定清单不能为空")
    return tuple(pdb_ids)


def shard_pdb_ids(
    pdb_ids: Sequence[str],
    shard_index: int,
    shard_count: int,
) -> tuple[str, ...]:
    """按固定清单行号取模生成稳定、互斥且完备的 worker 分片。"""
    count = int(shard_count)
    index = int(shard_index)
    if count <= 0 or index < 0 or index >= count:
        raise ValueError("shard_count 必须为正，shard_index 必须位于 [0, shard_count)")
    return tuple(str(value) for value in pdb_ids[index::count])


def build_production_tasks(
    stage1_model_name: str,
    split: str,
    pdb_list_path: str | Path,
    shard_index: int,
    shard_count: int,
) -> tuple[ProductionTask, ...]:
    """从同一固定清单直接构造 runner 任务，不现场追求“最新”样本。"""
    pdb_ids = shard_pdb_ids(
        load_pdb_id_list(pdb_list_path),
        shard_index=shard_index,
        shard_count=shard_count,
    )
    return tuple(
        ProductionTask(stage1_model_name, str(split), pdb_id) for pdb_id in pdb_ids
    )


class AGOccurrenceVoxelLoader:
    """读取 Stage E3 schema-v3 occurrence 稀疏 ZYX mask。"""

    def __init__(self, data_root: str | Path) -> None:
        self.data_root = Path(data_root)

    def __call__(
        self,
        pdb_id: str,
        full_shape_zyx: tuple[int, int, int],
    ) -> dict[int, np.ndarray]:
        """返回 ``occurrence_id -> 全图 C-order linear voxel index``。"""
        path = self.data_root / "density" / str(pdb_id).lower() / "ligand_area.npz"
        with np.load(path, allow_pickle=False) as data:
            if "schema_version" not in data or int(np.asarray(data["schema_version"]).item()) != 3:
                raise ValueError(f"{path}: occurrence loader 只接受 Stage E3 schema_version=3")
            shape = tuple(int(value) for value in np.asarray(data["grid_shape_zyx"]).tolist())
            expected_shape = tuple(int(value) for value in full_shape_zyx)
            if shape != expected_shape:
                raise ValueError(f"{path}: grid_shape_zyx={shape} 与 probability={expected_shape} 不一致")
            union = np.asarray(data["union_mask"], dtype=np.bool_)
            if union.shape != (1, *shape):
                raise ValueError(f"{path}: union_mask 必须为 (1,Z,Y,X)")
            mask_keys = sorted(
                (int(match.group(1)), key)
                for key in data.files
                if (match := _MASK_KEY.fullmatch(key)) is not None
            )
            occurrences: dict[int, np.ndarray] = {}
            reconstructed_union = np.zeros(shape, dtype=np.bool_)
            for occurrence_id, key in mask_keys:
                sparse = np.asarray(data[key])
                if sparse.dtype.kind not in "iu" or sparse.ndim != 2 or sparse.shape[1] != 3:
                    raise ValueError(f"{path}/{key}: mask 必须是整数 [K,3] ZYX")
                sparse = sparse.astype(np.int64, copy=False)
                if sparse.size and (
                    bool(np.any(sparse < 0))
                    or bool(np.any(sparse >= np.asarray(shape, dtype=np.int64)))
                ):
                    raise ValueError(f"{path}/{key}: mask voxel 越过完整图")
                if sparse.shape[0] > 1:
                    order = np.lexsort((sparse[:, 2], sparse[:, 1], sparse[:, 0]))
                    if not np.array_equal(order, np.arange(sparse.shape[0])):
                        raise ValueError(f"{path}/{key}: mask 必须按 ZYX 字典序排序")
                    if bool(np.any(np.all(sparse[1:] == sparse[:-1], axis=1))):
                        raise ValueError(f"{path}/{key}: mask voxel 必须唯一")
                linear = np.ravel_multi_index(sparse.T, shape).astype(np.int64)
                occurrences[int(occurrence_id)] = linear
                if sparse.size:
                    reconstructed_union[tuple(sparse.T)] = True
            if not np.array_equal(reconstructed_union, union[0]):
                raise ValueError(f"{path}: union_mask 不等于全部 mask_{{cid}} 的并集")
        return occurrences

    def for_task(
        self,
        task: ProductionTask,
        full_shape_zyx: tuple[int, int, int],
    ) -> dict[int, np.ndarray]:
        """适配 ``make_component_role_producer`` 的 task-first 签名。"""
        return self(task.pdb_id, full_shape_zyx)


class CachedStage1WrapperProvider:
    """从唯一 checkpoint/config 严格恢复并在当前 worker 内复用完整 wrapper。"""

    def __init__(
        self,
        stage1_model_name: str,
        checkpoint_path: str | Path,
        resolved_config_path: str | Path | None,
        device: str,
        wrapper_loader: Callable[..., Any] | None = None,
    ) -> None:
        self.stage1_model_name = str(stage1_model_name)
        self.checkpoint_path = Path(checkpoint_path)
        self.resolved_config_path = (
            None if resolved_config_path is None else Path(resolved_config_path)
        )
        self.device = str(device)
        self.wrapper_loader = wrapper_loader
        self._wrapper: Any | None = None

    def __call__(self, task: ProductionTask) -> Any:
        """返回与 task producer 一致、已移到显式 device 的 eval wrapper。"""
        if task.stage1_model_name != self.stage1_model_name:
            raise ValueError("wrapper provider 不能跨 stage1_model_name 复用")
        if self._wrapper is None:
            loader = self.wrapper_loader or load_stage1_wrapper
            wrapper = loader(
                checkpoint_path=self.checkpoint_path,
                resolved_config_path=self.resolved_config_path,
                map_location="cpu",
            )
            if hasattr(wrapper, "to"):
                wrapper = wrapper.to(self.device)
            if hasattr(wrapper, "eval"):
                wrapper.eval()
            self._wrapper = wrapper
        return self._wrapper


class _TaskDatasetMaterializer:
    """为一个 PDB 复用同一 Stage1Dataset cache 并物化动态推理请求。"""

    def __init__(
        self,
        data_root: Path,
        task: ProductionTask,
        density_channel_config: Mapping[str, Any],
        atom_buffer_radius: float,
        cache_max_bytes: int,
        device: str,
    ) -> None:
        seed_request = ResolvedStage1Crop(
            pdb_id=task.pdb_id,
            box_start_zyx=(0, 0, 0),
            require_targets=False,
            role="centered",
        )
        self.dataset = Stage1Dataset(
            all_data_path=str(data_root),
            split_file=(seed_request,),
            mode="centered",
            stage1_model_name=task.stage1_model_name,
            box_pool_root=None,
            density_channel_config=density_channel_config,
            atom_buffer_radius=float(atom_buffer_radius),
            cache_max_bytes=int(cache_max_bytes),
            enable_random_rotation=False,
        )
        self.task = task
        self.device = str(device)
        self.collator = self.dataset.collate_fn

    def window_batch(
        self,
        starts_zyx: Sequence[tuple[int, int, int]],
    ) -> dict[str, Any]:
        """把一批滑窗起点交给统一 Dataset/materializer 与正式 collator。"""
        samples = [
            self.dataset.materialize_request(
                ResolvedStage1Crop(
                    pdb_id=self.task.pdb_id,
                    box_start_zyx=tuple(start),
                    require_targets=False,
                    role="sliding",
                )
            )
            for start in starts_zyx
        ]
        return _move_batch_to_device(self.collator(samples), self.device)

    def full_map_context(
        self,
    ) -> tuple[tuple[int, int, int], np.ndarray, np.ndarray, np.ndarray]:
        """复用 Dataset cache 返回完整图几何与 receptor 坐标。"""
        return self.dataset.full_map_context(self.task.pdb_id)

    def centered_batch(self, request: CenteredRequest) -> dict[str, Any]:
        """把一个 formal centered 请求物化成 batch-size-one 模型输入。"""
        if (
            request.stage1_model_name != self.task.stage1_model_name
            or request.split != self.task.split
            or request.pdb_id != self.task.pdb_id
        ):
            raise ValueError("CenteredRequest 与当前 task 身份不一致")
        sample = self.dataset.materialize_request(
            ResolvedStage1Crop(
                pdb_id=request.pdb_id,
                box_start_zyx=request.box_start_zyx,
                require_targets=False,
                role="centered",
            )
        )
        return _move_batch_to_device(self.collator([sample]), self.device)


class Stage1RuntimeAssembly:
    """把 A—G root、resolved config 与 checkpoint 装成全部 inference providers。"""

    def __init__(
        self,
        data_root: str | Path,
        stage1_model_name: str,
        checkpoint_path: str | Path,
        resolved_config_path: str | Path | None,
        device: str,
        window_batch_size: int,
        cache_max_bytes: int = 536_870_912,
        wrapper_loader: Callable[..., Any] | None = None,
    ) -> None:
        self.data_root = Path(data_root)
        self.stage1_model_name = str(stage1_model_name)
        self.device = str(device)
        self.window_batch_size = int(window_batch_size)
        self.cache_max_bytes = int(cache_max_bytes)
        if self.window_batch_size <= 0:
            raise ValueError("window_batch_size 必须为正")
        config_path = resolve_checkpoint_config_path(
            checkpoint_path, resolved_config_path
        )
        self.resolved_config_path = config_path
        self.density_channel_config, self.atom_buffer_radius = _load_dataset_contract(
            config_path, self.stage1_model_name
        )
        self.wrapper_provider = CachedStage1WrapperProvider(
            stage1_model_name=self.stage1_model_name,
            checkpoint_path=checkpoint_path,
            resolved_config_path=config_path,
            device=self.device,
            wrapper_loader=wrapper_loader,
        )
        self.occurrence_loader = AGOccurrenceVoxelLoader(self.data_root)

    def full_map_input(self, task: ProductionTask) -> FullMapTaskInput:
        """构造一张 A—G 完整图的滑窗 Dataset、Find hardmask 与 wrapper。"""
        self._check_task(task)
        materializer = self._materializer(task)
        full_shape, voxel_size, origin, receptor_coords = (
            materializer.full_map_context()
        )
        # 先验证正式 80³/stride40 确实可覆盖，错误在申请大数组前暴露。
        window_starts_zyx(full_shape, (80, 80, 80), (40, 40, 40))
        if task.stage1_model_name.startswith("Find"):
            receptor_hardmask = build_hardmask_from_world_coordinates(
                atom_coords_world=receptor_coords,
                box_origin_world=origin,
                voxel_size_world=voxel_size,
                box_shape_zyx=np.asarray(full_shape, dtype=np.int64),
            ).astype(np.bool_, copy=False)
        else:
            receptor_hardmask = None
        return FullMapTaskInput(
            model=self.wrapper_provider(task),
            full_shape_zyx=full_shape,
            window_batch_builder=materializer.window_batch,
            receptor_hardmask_full=receptor_hardmask,
            window_batch_size=self.window_batch_size,
            origin_xyz=tuple(float(value) for value in origin),
            voxel_size_xyz=tuple(float(value) for value in voxel_size),
        )

    def centered_batch_builder(
        self,
        task: ProductionTask,
    ) -> Callable[[CenteredRequest], Mapping[str, object]]:
        """返回复用单 PDB Dataset cache 的 centered batch builder。"""
        self._check_task(task)
        return self._materializer(task).centered_batch

    def occurrence_voxels(
        self,
        task: ProductionTask,
        full_shape_zyx: tuple[int, int, int],
    ) -> Mapping[int, np.ndarray]:
        """读取 task 对应的 occurrence 稀疏体素。"""
        self._check_task(task)
        return self.occurrence_loader(task.pdb_id, full_shape_zyx)

    def _materializer(self, task: ProductionTask) -> _TaskDatasetMaterializer:
        """构造只在当前 role callback 生命周期内存在的 per-PDB materializer。"""
        return _TaskDatasetMaterializer(
            data_root=self.data_root,
            task=task,
            density_channel_config=self.density_channel_config,
            atom_buffer_radius=self.atom_buffer_radius,
            cache_max_bytes=self.cache_max_bytes,
            device=self.device,
        )

    def _check_task(self, task: ProductionTask) -> None:
        """阻止一个 runtime assembly 跨 producer 使用。"""
        if task.stage1_model_name != self.stage1_model_name:
            raise ValueError("runtime assembly 不能跨 stage1_model_name 使用")


def _load_dataset_contract(
    config_path: Path,
    stage1_model_name: str,
) -> tuple[dict[str, object], float]:
    """从 resolved config 冷读 Dataset 输入通道与 8 Å buffer 契约。"""
    from omegaconf import OmegaConf

    cfg = OmegaConf.load(config_path)
    OmegaConf.resolve(cfg)
    if "dataset" not in cfg:
        raise KeyError(f"{config_path}: resolved config 缺少 dataset")
    dataset_cfg = cfg.dataset
    configured_name = str(dataset_cfg.stage1_model_name)
    if configured_name != stage1_model_name:
        raise ValueError(
            f"resolved config producer={configured_name!r} 与 CLI={stage1_model_name!r} 不一致"
        )
    if "density_channel_config" not in dataset_cfg:
        raise KeyError("resolved config.dataset 缺少 density_channel_config")
    density = OmegaConf.to_container(
        dataset_cfg.density_channel_config,
        resolve=True,
    )
    if not isinstance(density, dict):
        raise TypeError("dataset.density_channel_config 必须解析为 mapping")
    atom_buffer_radius = float(dataset_cfg.get("atom_buffer_radius", 8.0))
    if atom_buffer_radius != 8.0:
        raise ValueError("AdaLigand Stage1 推理的 atom_buffer_radius 固定为 8.0 Å")
    return density, atom_buffer_radius


def _move_batch_to_device(batch: Mapping[str, Any], device: str) -> dict[str, Any]:
    """只移动 tensor 字段；PDB 身份和 ragged 元数据保持 Python 值。"""
    try:
        import torch
    except ImportError:
        return dict(batch)
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


__all__ = [
    "AGOccurrenceVoxelLoader",
    "CachedStage1WrapperProvider",
    "Stage1RuntimeAssembly",
    "build_production_tasks",
    "load_pdb_id_list",
    "shard_pdb_ids",
]
