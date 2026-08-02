# -*- coding: utf-8 -*-
"""把 AdaLigand Stage A 至 Stage G 产物、冻结配置和 checkpoint 装配成 Stage1 推理输入。

主要入口:
    - `load_pdb_id_list`、`shard_pdb_ids`、`build_production_tasks`: 冷读冻结 PDB 清单，并按原始行号取模生成稳定、互斥、完备的 worker 任务。
    - `AGOccurrenceVoxelLoader`: 读取 Stage E3 schema-v3 `ligand_area.npz`，恢复 occurrence 的完整图 C-order voxel 索引。
    - `Stage1RuntimeAssembly`: 严格恢复完整模型包装器，复用单 PDB `Stage1Dataset` 缓存，并提供完整图滑窗、居中批次和 occurrence 输入。

本模块不重新实现 Dataset、模型 forward、概率融合、组件构造或 centered 产物算法。坐标约定为：网格形状和离散索引使用 ZYX，物理原点、体素尺寸和原子坐标使用世界 XYZ，长度单位 Å。
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .centered import CenteredRequest
from .checkpoint import load_stage1_wrapper, resolve_checkpoint_config_path
from .full_map import window_starts_zyx
from .runner import FullMapTaskInput, ProductionTask


# Stage E3 `ligand_area.npz` 的 occurrence 稀疏 mask 字段名；捕获组是十进制 occurrence_id。
_MASK_KEY = re.compile(r"^mask_(\d+)$")



# ================================================= 工具函数 =================================================
def load_pdb_id_list(path: str | Path) -> tuple[str, ...]:
    """
    读取固定 PDB 清单并保持文件内顺序。

    输入参数:
        - path: str | Path, JSON、JSONL 或纯文本固定清单路径；JSON 顶层可为列表或含 `pdb_ids` 列表的对象，列表元素和 JSONL 行可为字符串或含 `pdb_id` 的对象。

    输出:
        - pdb_ids: tuple[str, ...], 小写、非空、无重复的 PDB identity；严格保持源文件顺序，不按名称重排。
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

    # list[str], 按源文件顺序规范化为小写的 PDB identity；对象行只读取显式 `pdb_id`。
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
    """
    按固定清单行号取模生成稳定、互斥且完备的 worker 分片。

    输入参数:
        - pdb_ids: Sequence[str], 固定且无重复的 PDB 清单，顺序决定分片归属。
        - shard_index: int, 当前 worker 的 0-based 分片编号，范围 `[0, shard_count)`。
        - shard_count: int, 分片总数，必须为正。

    输出:
        - shard_pdb_ids: tuple[str, ...], 原清单中满足 `row_index % shard_count == shard_index` 的 PDB identity；保持原相对行序。
    """
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
    """
    从同一固定清单直接构造 runner 任务，不现场追求“最新”样本。

    输入参数:
        - stage1_model_name: str, 当前 producer 正式名
        - split: str, 当前数据划分
        - pdb_list_path: str | Path, 固定 PDB 清单路径
        - shard_index: int, 当前 worker 分片编号
        - shard_count: int, 分片总数

    输出:
        - tasks: tuple[ProductionTask, ...], 与当前分片 PDB 行序一致的 producer/split/PDB 任务；不探测输出目录或样本新旧状态。
    """
    pdb_ids = shard_pdb_ids(
        load_pdb_id_list(pdb_list_path),
        shard_index=shard_index,
        shard_count=shard_count,
    )
    return tuple(ProductionTask(stage1_model_name, str(split), pdb_id) for pdb_id in pdb_ids)

def _load_dataset_contract(
    config_path: Path,
    stage1_model_name: str,
) -> tuple[dict[str, object], float]:
    """从 resolved config 读取 Dataset 输入通道和固定 8 Å atom buffer 契约。
    返回 tuple 依次为：
        - density_channel_config: dict[str, object]，完整保留 resolved dataset density channel 的顺序、开关和参数。
        - atom_buffer_radius: float，Find 在 core BOX 外选择 receptor 原子的世界坐标半径，正式值为 8.0 Å。
    """
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
    density = OmegaConf.to_container(dataset_cfg.density_channel_config, resolve=True)
    if not isinstance(density, dict):
        raise TypeError("dataset.density_channel_config 必须解析为 mapping")
    atom_buffer_radius = float(dataset_cfg.get("atom_buffer_radius", 8.0))
    if atom_buffer_radius != 8.0:
        raise ValueError("AdaLigand Stage1 推理的 atom_buffer_radius 固定为 8.0 Å")
    return density, atom_buffer_radius

def _move_batch_to_device(batch: Mapping[str, Any], device: str) -> dict[str, Any]:
    """把 batch 移到device上(PDB identity 和 ragged 元数据保持 Python 值)。
    输入的 batch 是 Stage1Dataset.collate_fn 生成的混合 mapping；返回 mapping 保留所有非 tensor 字段的原值，只对 tensor 调用 non-blocking device copy。
    """
    try:
        import torch
    except ImportError:
        return dict(batch)
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }




# ==================== 加载 GT-occurrence ====================
class AGOccurrenceVoxelLoader:
    """
    读取 Stage E3 schema-v3 occurrence 稀疏 ZYX mask。

    输入参数:
        - data_root: str | Path, A-G 正式数据根目录

    调用输出:
        - occurrences: dict[int, np.ndarray], occurrence_id 到 `(K_occ,)` int64 完整 ZYX 网格 C-order 离散线性 voxel 索引的映射；键按数值升序。
    """
    def __init__(self, data_root: str | Path) -> None:
        """
        读取 Stage E3 schema-v3 occurrence 稀疏 ZYX mask。

        输入参数:
            - data_root: str | Path, A-G 正式数据根目录

        调用输出:
            - occurrences: dict[int, np.ndarray], occurrence_id 到 `(K_occ,)` int64 完整 ZYX 网格 C-order 离散线性 voxel 索引的映射；键按数值升序。
        """
        self.data_root = Path(data_root)

    def __call__(
        self,
        pdb_id: str,
        full_shape_zyx: tuple[int, int, int],
    ) -> dict[int, np.ndarray]:
        """
        返回 occurrence 到完整图 C-order 离散线性 voxel indices 的映射。

        输入参数:
            - pdb_id: str, 当前 PDB identity
            - full_shape_zyx: tuple[int, int, int], probability 完整图的 ZYX voxel 形状 `(D, H, W)`；必须与文件 `grid_shape_zyx` 完全一致。

        输出:
            - occurrences: dict[int, np.ndarray], occurrence_id 到 `(K_occ,)` int64 完整图 C-order 离散线性 voxel 索引的映射。

        读取文件:
            - `<data_root>/density/{pdb_id}/ligand_area.npz`: Stage E3 schema-v3 NPZ，必须包含 `schema_version/grid_shape_zyx/union_mask` 和零个或多个 `mask_{occurrence_id}`。
        """
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

            # list[tuple[int, str]], 按 occurrence_id 数值升序排列的严格 `mask_<整数>` 字段。
            mask_keys = sorted(
                (int(match.group(1)), key)
                for key in data.files
                if (match := _MASK_KEY.fullmatch(key)) is not None
            )
            # dict[int, np.ndarray], occurrence_id 到唯一、升序的完整图 C-order voxel 索引；插入顺序即正式 occurrence 顺序。
            occurrences: dict[int, np.ndarray] = {}
            # bool, (D, H, W), 由全部 occurrence 稀疏 mask 重新构造的并集，用于逐 voxel 核对文件 `union_mask[0]`。
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

                # int64, (K_occ,), 稀疏 ZYX voxel 行在完整图中的 C-order 离散线性索引。
                linear = np.ravel_multi_index(sparse.T, shape).astype(np.int64)
                occurrences[int(occurrence_id)] = linear
                if sparse.size:
                    reconstructed_union[tuple(sparse.T)] = True
            if not np.array_equal(reconstructed_union, union[0]):
                raise ValueError(f"{path}: union_mask 不等于全部 mask_{{cid}} 的并集")
        return occurrences



# ==================== 给路径加载 warpper ====================
class CachedStage1WrapperProvider:
    """
    从唯一 checkpoint/config 严格恢复并在当前 worker 内复用完整 wrapper。

    输入参数:
        - stage1_model_name: str, 当前提供器绑定的模型来源正式名
        - checkpoint_path: str | Path, 完整 Stage1 wrapper checkpoint
        - resolved_config_path: str | Path | None, 与 checkpoint 对应的已解析配置；None 时按 checkpoint 快照的固定相邻规则解析。
        - device: str, 模型包装器最终所在设备，例如 `cpu` 或 `cuda:0`。
        - wrapper_loader: Callable | None, 可选显式加载器；None 时使用严格的 `load_stage1_wrapper`。
        - allow_current_workspace_code: bool, checkpoint 缺少完整代码快照时是否显式允许使用当前工作区代码；正式可复现运行应保持 False。

    __call__ 输入参数:
        - task: ProductionTask, 当前任务 identity；producer 必须与 provider 绑定值一致

    __call__ 输出:
        - wrapper: Any, 严格恢复、移动到目标 device 并处于 eval 的完整 wrapper；同一 provider 后续任务复用同一对象。
    """
    def __init__(
        self,
        stage1_model_name: str,
        checkpoint_path: str | Path,
        resolved_config_path: str | Path | None,
        device: str,
        wrapper_loader: Callable[..., Any] | None = None,
        allow_current_workspace_code: bool = False,
    ) -> None:
        """
        从唯一 checkpoint/config 严格恢复并在当前 worker 内复用完整 wrapper。

        输入参数:
            - stage1_model_name: str, 当前提供器绑定的模型来源正式名
            - checkpoint_path: str | Path, 完整 Stage1 wrapper checkpoint
            - resolved_config_path: str | Path | None, 与 checkpoint 对应的已解析配置；None 时按 checkpoint 快照的固定相邻规则解析。
            - device: str, 模型包装器最终所在设备，例如 `cpu` 或 `cuda:0`。
            - wrapper_loader: Callable | None, 可选显式加载器；None 时使用严格的 `load_stage1_wrapper`。
            - allow_current_workspace_code: bool, checkpoint 缺少完整代码快照时是否显式允许使用当前工作区代码；正式可复现运行应保持 False。

        __call__ 输入参数:
            - task: ProductionTask, 当前任务 identity；producer 必须与 provider 绑定值一致

        __call__ 输出:
            - wrapper: Any, 严格恢复、移动到目标 device 并处于 eval 的完整 wrapper；同一 provider 后续任务复用同一对象。
        """
        self.stage1_model_name = str(stage1_model_name)
        self.checkpoint_path = Path(checkpoint_path)
        self.resolved_config_path = (None if resolved_config_path is None else Path(resolved_config_path))
        self.device = str(device)
        self.wrapper_loader = wrapper_loader
        self.allow_current_workspace_code = bool(allow_current_workspace_code)
        self._wrapper: Any | None = None

    def __call__(self, task: ProductionTask) -> Any:
        """
        返回与 task producer 一致、已移到显式 device 的 eval wrapper。

        输入参数:
            - task: ProductionTask, 当前任务 identity；producer 必须与 provider 绑定值一致

        输出:
            - wrapper: Any, 严格恢复、移动到目标 device 并处于 eval 的完整 wrapper；同一 provider 后续任务复用同一对象。
        """
        if task.stage1_model_name != self.stage1_model_name:
            raise ValueError("wrapper provider 不能跨 stage1_model_name 复用")
        if self._wrapper is None:
            loader = self.wrapper_loader or load_stage1_wrapper
            wrapper = loader(
                checkpoint_path=self.checkpoint_path,
                resolved_config_path=self.resolved_config_path,
                map_location="cpu",
                allow_current_workspace_code=self.allow_current_workspace_code,
            )
            if hasattr(wrapper, "to"):
                wrapper = wrapper.to(self.device)
            if hasattr(wrapper, "eval"):
                wrapper.eval()
            self._wrapper = wrapper
        return self._wrapper




# ==================== 给定请求组装 batch(windows_batch / centered_batch) ====================
class _TaskDatasetMaterializer:
    """
    给定请求组装 batch(windows_batch / centered_batch)。

    输入参数:
        - data_root: Path, A-G 正式数据根目录
        - task: ProductionTask, 当前 producer/split/PDB identity
        - density_channel_config: Mapping[str, Any], 从 resolved config 冷读的 producer density channel 配方。
        - atom_buffer_radius: float, Find 在 core BOX 外纳入原子的世界坐标 buffer 半径，正式值为 8 Å。
        - cache_max_bytes: int, 当前 Dataset 的 worker-local PDB 资产缓存字节上限。
        - device: str, collated tensor batch 的目标设备。

    内部方法:
        - window_batch: 将一批滑窗起点交给统一 Dataset/materializer 与正式 collator 生成 batch。
        - full_map_context: self.dataset.full_map_context(self.task.pdb_id): 复用 Dataset cache 返回完整图几何与 receptor 坐标。
        - centered_batch: 将一批居中图像交给统一 Dataset/materializer 与正式 collator 生成 batch。
    """
    def __init__(
        self,
        data_root: Path,
        task: ProductionTask,
        density_channel_config: Mapping[str, Any],
        atom_buffer_radius: float,
        cache_max_bytes: int,
        device: str,
    ) -> None:
        """
        给定请求组装 batch(windows_batch / centered_batch)。

        输入参数:
            - data_root: Path, A-G 正式数据根目录
            - task: ProductionTask, 当前 producer/split/PDB identity
            - density_channel_config: Mapping[str, Any], 从 resolved config 冷读的 producer density channel 配方。
            - atom_buffer_radius: float, Find 在 core BOX 外纳入原子的世界坐标 buffer 半径，正式值为 8 Å。
            - cache_max_bytes: int, 当前 Dataset 的 worker-local PDB 资产缓存字节上限。
            - device: str, collated tensor batch 的目标设备。

        内部方法:
            - window_batch: 将一批滑窗起点交给统一 Dataset/materializer 与正式 collator 生成 batch。
            - full_map_context: self.dataset.full_map_context(self.task.pdb_id): 复用 Dataset cache 返回完整图几何与 receptor 坐标。
            - centered_batch: 将一批居中图像交给统一 Dataset/materializer 与正式 collator 生成 batch。
        """
        from src.datasets.stage1_dataset import Stage1Dataset
        from src.datasets.stage1_requests import ResolvedStage1Crop

        # ResolvedStage1Crop, 仅用于让 Dataset 绑定当前 PDB 的合法无目标请求, 真实滑窗和居中请求由后续方法动态物化。
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
        """
        把一批滑窗起点交给统一 Dataset/materializer 与正式 collator 生成 batch。

        输入参数:
            - starts_zyx: Sequence[tuple[int, int, int]], 长度 B_window；每项是完整图离散 ZYX voxel-index 窗口起点。

        输出:
            - batch: dict[str, Any], 第一维批量大小 `B == len(requests)` 的目标设备 Stage1 输入。dense 密度、hardmask、BOX 几何按 B 堆叠；Find 原子表按第一维拼接，并由 `atom_counts: int64 (B,)`、`atom_offsets: int64 (B+1,)` 和 `atom_batch_index: int64 (N_A_total,)` 保存 BOX 归属；字段契约与训练 `Stage1Collator` 完全一致。
        """
        from src.datasets.stage1_requests import ResolvedStage1Crop

        # list[dict[str, Any]]，长度 B_window；每项由正式 Dataset 以 `role=sliding` 物化一个不含补零区域的真实 BOX。
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
        """
        self.dataset.full_map_context(self.task.pdb_id): 复用 Dataset cache 返回完整图几何与 receptor 坐标。

        输出:
            - shape_zyx: tuple[int, int, int], 完整图的 ZYX voxel 形状 `(D, H, W)`。
            - voxel_size_xyz: float array, (3,), 世界 XYZ 三轴的体素尺寸，单位 Å/voxel。
            - origin_xyz: float array, (3,), 完整图 voxel-grid 起点的世界 XYZ 坐标，单位 Å。
            - receptor_coord_xyz: float array, (N_receptor, 3), 当前 PDB 全部 receptor 原子的世界 XYZ 坐标，单位 Å。
        """
        return self.dataset.full_map_context(self.task.pdb_id)

    def centered_batch(self, requests: Sequence[CenteredRequest]) -> dict[str, Any]:
        """
        把一组正式 centered 请求物化成一次完整模型输入 生成 batch。

        输入参数:
            - requests: Sequence[CenteredRequest], 同一 producer、split、PDB 和 centered role 的有序请求；每项给出完整图离散 ZYX voxel-index BOX 角点起点及来源 tree/node/threshold 身份。

        输出:
            - batch: dict[str, Any], 第一维批量大小 `B == len(requests)` 的目标设备 Stage1 输入。dense 密度、hardmask、BOX 几何按 B 堆叠；Find 原子表按第一维拼接，并由 `atom_counts: int64 (B,)`、`atom_offsets: int64 (B+1,)` 和 `atom_batch_index: int64 (N_A_total,)` 保存 BOX 归属；字段契约与训练 `Stage1Collator` 完全一致。
        """
        from src.datasets.stage1_requests import ResolvedStage1Crop

        samples = [
            self.dataset.materialize_request(
                ResolvedStage1Crop(
                    pdb_id=request.pdb_id,
                    box_start_zyx=request.box_start_zyx,
                    require_targets=False,
                    role="centered",
                )
            )
            for request in requests
        ]
        return _move_batch_to_device(self.collator(samples), self.device)





# ======================================== 总打包 ========================================
class Stage1RuntimeAssembly:
    """
    为一个推理 worker（一个独立的推理进程）组装 Stage-1 所需的模型、数据集和稀疏 occurrence（A-G `ligand_area.npz` 中的一个稀疏 ligand 区域）读取器。
    这里的“组装”只建立可调用的输入提供器，不执行模型 forward，也不生成 centered/完整图归档文件。`ProductionTask` 是一个三元任务身份 `(stage1_model_name, split, pdb_id)`；所有公开方法都要求调用任务与本对象绑定的 `stage1_model_name` 一致。

    输入参数:
        - data_root: str | Path，AdaLigand Stage A-G 正式数据根目录；Stage1Dataset 和 `density/{pdb_id}/ligand_area.npz` occurrence 文件都从这里读取。
        - stage1_model_name: str，当前推理所绑定的 Stage-1 producer 正式名称；它同时用于匹配 resolved config、恢复 checkpoint 和校验后续 ProductionTask。
        - checkpoint_path: str | Path，完整 Stage-1 模型包装器 checkpoint 路径；包装器包含模型权重及恢复 forward 所需的 checkpoint 快照信息。
        - resolved_config_path: str | Path | None，已经解析完变量和默认值的配置文件路径；传入 None 时由 checkpoint 的固定相邻路径规则解析，配置中必须包含与 stage1_model_name 一致的 Dataset 输入契约。
        - device: str，模型和批次拼装器（collator）输出的 tensor batch 的目标设备，例如 `cpu`、`cuda` 或 `cuda:0`；PDB identity、计数和其他 Python 元数据不搬到该设备。
        - window_batch_size: int，一次完整图滑窗模型调用包含的窗口数；只影响 full-map forward 的批大小，不影响 centered BOX 批大小。
        - centered_batch_size: int，一次 centered 模型调用包含的 80³ BOX 数；只影响 centered forward 的批大小，正式默认值为 12。
        - cache_max_bytes: int，单个推理进程的 Stage1Dataset 资产缓存允许占用的最大字节数，默认值为 536870912000（500 GiB）；该值是上限，不会预先分配内存。
        - wrapper_loader: Callable | None，可选的模型包装器加载函数；为 None 时使用 `load_stage1_wrapper`，该函数接收 checkpoint/config 路径并返回可调用的完整 wrapper。
        - allow_current_workspace_code: bool，checkpoint 缺少完整代码快照时是否允许加载当前工作区代码；正式可复现运行应保持 False。

    构造后的内部组件:
        - resolved_config_path: Path，最终采用的 resolved config；从中读取 density_channel_config 和固定的 8 Å atom_buffer_radius。
        - density_channel_config: dict[str, object]，Dataset 读取密度通道的完整配置映射；键和值的顺序由 resolved config 决定。
        - atom_buffer_radius: float，Find 数据集在 core BOX（坐标位于 80³ BOX 范围内）外纳入 receptor 原子的世界坐标半径，单位 Å，当前契约固定为 8.0。
        - wrapper_provider: CachedStage1WrapperProvider，按需从 checkpoint 恢复一个处于 eval 模式且位于 device 的完整 Stage-1 wrapper（接收 Dataset 批次并执行模型 forward 的可调用对象）；同一 assembly 生命周期内复用该 wrapper，不跨 stage1_model_name 复用。
        - occurrence_loader: AGOccurrenceVoxelLoader，读取 Stage E3 schema-v3 `ligand_area.npz`，返回 `dict[int, np.ndarray]`；每个键是 occurrence_id，每个值是该 occurrence 在完整图 ZYX 网格中的 int64 `(K_occ,)` C-order 线性 voxel 索引。
        - _active_materializer_task: ProductionTask | None，当前缓存 Dataset 所绑定的三元任务身份；初始值为 None。
        - _active_materializer: _TaskDatasetMaterializer | None，当前任务的进程私有 Stage1Dataset、collator 和 batch builder；初始值为 None。

    公开方法:
        - full_map_input(task): 返回 FullMapTaskInput，包含完整图 shape、世界几何、滑窗 batch builder、Find 专用完整图 receptor hardmask、wrapper 和 window_batch_size。
        - centered_batch_builder(task): 返回以 `Sequence[CenteredRequest]` 为输入、batch 为输出的 centered batch builder；多个 centered role 在同一 task 内复用同一个 Dataset materializer。
        - occurrence_voxels(task, full_shape_zyx): 返回当前 PDB 的 occurrence_id 到完整图 C-order voxel 索引映射，并核对文件 shape 与完整图 shape 一致。

    缓存边界:
        - wrapper_provider 的缓存按 assembly 生命周期存在；第一次请求任务时加载 wrapper，后续同 producer 任务直接复用，不重复读取 checkpoint。
        - _active_materializer 只缓存最近一个完全相同的 ProductionTask；切换 split 或 pdb_id 时创建新的 Dataset materializer，并丢弃旧 materializer 的引用。
        - occurrence_loader 不缓存模型或 Dataset；每次 occurrence_voxels 调用按指定 pdb_id 和 full_shape_zyx 读取并核对对应的 occurrence 文件。
    """

    def __init__(
        self,
        data_root: str | Path,
        stage1_model_name: str,
        checkpoint_path: str | Path,
        resolved_config_path: str | Path | None,
        device: str,
        window_batch_size: int,
        centered_batch_size: int = 12,
        cache_max_bytes: int = 536_870_912_000,
        wrapper_loader: Callable[..., Any] | None = None,
        allow_current_workspace_code: bool = False,
    ) -> None:
        """
        为一个推理 worker（一个独立的推理进程）组装 Stage-1 所需的模型、数据集和稀疏 occurrence（A-G `ligand_area.npz` 中的一个稀疏 ligand 区域）读取器。
        这里的“组装”只建立可调用的输入提供器，不执行模型 forward，也不生成 centered/完整图归档文件。`ProductionTask` 是一个三元任务身份 `(stage1_model_name, split, pdb_id)`；所有公开方法都要求调用任务与本对象绑定的 `stage1_model_name` 一致。

        输入参数:
            - data_root: str | Path，AdaLigand Stage A-G 正式数据根目录；Stage1Dataset 和 `density/{pdb_id}/ligand_area.npz` occurrence 文件都从这里读取。
            - stage1_model_name: str，当前推理所绑定的 Stage-1 producer 正式名称；它同时用于匹配 resolved config、恢复 checkpoint 和校验后续 ProductionTask。
            - checkpoint_path: str | Path，完整 Stage-1 模型包装器 checkpoint 路径；包装器包含模型权重及恢复 forward 所需的 checkpoint 快照信息。
            - resolved_config_path: str | Path | None，已经解析完变量和默认值的配置文件路径；传入 None 时由 checkpoint 的固定相邻路径规则解析，配置中必须包含与 stage1_model_name 一致的 Dataset 输入契约。
            - device: str，模型和批次拼装器（collator）输出的 tensor batch 的目标设备，例如 `cpu`、`cuda` 或 `cuda:0`；PDB identity、计数和其他 Python 元数据不搬到该设备。
            - window_batch_size: int，一次完整图滑窗模型调用包含的窗口数；只影响 full-map forward 的批大小，不影响 centered BOX 批大小。
            - centered_batch_size: int，一次 centered 模型调用包含的 80³ BOX 数；只影响 centered forward 的批大小，正式默认值为 12。
            - cache_max_bytes: int，单个推理进程的 Stage1Dataset 资产缓存允许占用的最大字节数，默认值为 536870912000（500 GiB）；该值是上限，不会预先分配内存。
            - wrapper_loader: Callable | None，可选的模型包装器加载函数；为 None 时使用 `load_stage1_wrapper`，该函数接收 checkpoint/config 路径并返回可调用的完整 wrapper。
            - allow_current_workspace_code: bool，checkpoint 缺少完整代码快照时是否允许加载当前工作区代码；正式可复现运行应保持 False。

        构造后的内部组件:
            - resolved_config_path: Path，最终采用的 resolved config；从中读取 density_channel_config 和固定的 8 Å atom_buffer_radius。
            - density_channel_config: dict[str, object]，Dataset 读取密度通道的完整配置映射；键和值的顺序由 resolved config 决定。
            - atom_buffer_radius: float，Find 数据集在 core BOX（坐标位于 80³ BOX 范围内）外纳入 receptor 原子的世界坐标半径，单位 Å，当前契约固定为 8.0。
            - wrapper_provider: CachedStage1WrapperProvider，按需从 checkpoint 恢复一个处于 eval 模式且位于 device 的完整 Stage-1 wrapper（接收 Dataset 批次并执行模型 forward 的可调用对象）；同一 assembly 生命周期内复用该 wrapper，不跨 stage1_model_name 复用。
            - occurrence_loader: AGOccurrenceVoxelLoader，读取 Stage E3 schema-v3 `ligand_area.npz`，返回 `dict[int, np.ndarray]`；每个键是 occurrence_id，每个值是该 occurrence 在完整图 ZYX 网格中的 int64 `(K_occ,)` C-order 线性 voxel 索引。
            - _active_materializer_task: ProductionTask | None，当前缓存 Dataset 所绑定的三元任务身份；初始值为 None。
            - _active_materializer: _TaskDatasetMaterializer | None，当前任务的进程私有 Stage1Dataset、collator 和 batch builder；初始值为 None。

        公开方法:
            - full_map_input(task): 返回 FullMapTaskInput，包含完整图 shape、世界几何、滑窗 batch builder、Find 专用完整图 receptor hardmask、wrapper 和 window_batch_size。
            - centered_batch_builder(task): 返回以 `Sequence[CenteredRequest]` 为输入、batch 为输出的 centered batch builder；多个 centered role 在同一 task 内复用同一个 Dataset materializer。
            - occurrence_voxels(task, full_shape_zyx): 返回当前 PDB 的 occurrence_id 到完整图 C-order voxel 索引映射，并核对文件 shape 与完整图 shape 一致。

        缓存边界:
            - wrapper_provider 的缓存按 assembly 生命周期存在；第一次请求任务时加载 wrapper，后续同 producer 任务直接复用，不重复读取 checkpoint。
            - _active_materializer 只缓存最近一个完全相同的 ProductionTask；切换 split 或 pdb_id 时创建新的 Dataset materializer，并丢弃旧 materializer 的引用。
            - occurrence_loader 不缓存模型或 Dataset；每次 occurrence_voxels 调用按指定 pdb_id 和 full_shape_zyx 读取并核对对应的 occurrence 文件。
        """
        self.data_root = Path(data_root)
        self.stage1_model_name = str(stage1_model_name)
        self.device = str(device)
        self.window_batch_size = int(window_batch_size)
        self.centered_batch_size = int(centered_batch_size)
        self.cache_max_bytes = int(cache_max_bytes)
        if self.window_batch_size <= 0:
            raise ValueError("window_batch_size 必须为正")
        # Path, 与 checkpoint 快照绑定的唯一 resolved config；后续 Dataset 契约只从该文件冷读。
        config_path = resolve_checkpoint_config_path(checkpoint_path, resolved_config_path)
        self.resolved_config_path = config_path
        # `dict[str, object]` 与 float，推理 Dataset 必须复用的 density channel 配方和固定 8 Å 原子 buffer。
        self.density_channel_config, self.atom_buffer_radius = _load_dataset_contract(config_path, self.stage1_model_name)
        self.wrapper_provider = CachedStage1WrapperProvider(
            stage1_model_name=self.stage1_model_name,
            checkpoint_path=checkpoint_path,
            resolved_config_path=config_path,
            device=self.device,
            wrapper_loader=wrapper_loader,
            allow_current_workspace_code=allow_current_workspace_code,
        )
        self.occurrence_loader = AGOccurrenceVoxelLoader(self.data_root)
        self._active_materializer_task: ProductionTask | None = None
        self._active_materializer: _TaskDatasetMaterializer | None = None

    def full_map_input(self, task: ProductionTask) -> FullMapTaskInput:
        """
        构造一张 A-G 完整图的滑窗 Dataset、Find hardmask 与 wrapper。

        输入参数:
            - task: ProductionTask, 当前 producer/split/PDB identity

        输出:
            - inputs: FullMapTaskInput, 包含完整 wrapper、完整图 ZYX 形状、滑窗 batch builder、可选完整图 receptor hardmask、窗口 batch 数、世界 XYZ 原点与世界 XYZ 体素尺寸。
        """
        self._check_task(task)
        wrapper = self.wrapper_provider(task)
        materializer = self._materializer(task)
        # tuple 与三个数组，依次为完整图 ZYX 形状、世界 XYZ 体素尺寸、世界 XYZ 原点和 receptor 原子世界 XYZ 坐标。
        full_shape, voxel_size, origin, receptor_coords = (materializer.full_map_context())
        # 只读验证正式 80³/stride40 能以真实无 padding 窗口覆盖完整图，使形状错误在申请融合大数组前暴露。
        window_starts_zyx(full_shape, (80, 80, 80), (40, 40, 40))
        if task.stage1_model_name.startswith("Find"):
            from src.datasets.box_geometry import build_hardmask_from_world_coordinates
            # bool，(D, H, W)，完整图 ZYX voxel 网格上的 receptor home voxel；Find 完整图融合后在 True 位置把 ligand 概率清零。
            receptor_hardmask = build_hardmask_from_world_coordinates(
                atom_coords_world=receptor_coords,
                box_origin_world=origin,
                voxel_size_world=voxel_size,
                box_shape_zyx=np.asarray(full_shape, dtype=np.int64),
            ).astype(np.bool_, copy=False)
        else:
            receptor_hardmask = None
        return FullMapTaskInput(
            model=wrapper,
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
    ) -> Callable[[Sequence[CenteredRequest]], Mapping[str, object]]:
        """
        返回以 `Sequence[CenteredRequest]` 为输入、batch 为输出的 centered batch builder。

        输入参数:
            - task: ProductionTask, 当前 producer/split/PDB identity

        输出:
            - builder: Callable[[Sequence[CenteredRequest]], Mapping[str, object]], 当前由 centered 领域 producer 按 `centered_batch_size` 调用；它复用同一 PDB 的 Stage1Dataset cache。
        """
        self._check_task(task)
        self.wrapper_provider(task)
        return self._materializer(task).centered_batch   # 以 `Sequence[CenteredRequest]` 为输入、batch 为输出的 centered batch builder

    def occurrence_voxels(
        self,
        task: ProductionTask,
        full_shape_zyx: tuple[int, int, int],
    ) -> Mapping[int, np.ndarray]:
        """
        读取 task 对应的 occurrence 稀疏体素。

        输入参数:
            - task: ProductionTask, 当前 producer/split/PDB identity
            - full_shape_zyx: tuple[int, int, int], 完整图的 ZYX voxel 形状 `(D, H, W)`。

        输出:
            - occurrences: Mapping[int, np.ndarray], occurrence_id 到 `(K_occ,)` int64 完整图 C-order 离散线性 voxel 索引的映射。
        """
        self._check_task(task)
        return self.occurrence_loader(task.pdb_id, full_shape_zyx)

    def _materializer(self, task: ProductionTask) -> _TaskDatasetMaterializer:
        """
        在同一 PDB 的连续 roles 之间复用 materializer，切换 task 时释放旧引用。

        输入参数:
            - task: ProductionTask, 当前 producer/split/PDB identity

        输出:
            - materializer: _TaskDatasetMaterializer, 与 task 完全绑定的 Dataset/materializer
        """
        if task != self._active_materializer_task:
            self._active_materializer_task = task
            self._active_materializer = _TaskDatasetMaterializer(
                data_root=self.data_root,
                task=task,
                density_channel_config=self.density_channel_config,
                atom_buffer_radius=self.atom_buffer_radius,
                cache_max_bytes=self.cache_max_bytes,
                device=self.device,
            )
        assert self._active_materializer is not None
        return self._active_materializer

    def _check_task(self, task: ProductionTask) -> None:
        """
        if task.stage1_model_name != self.stage1_model_name:
            raise ValueError("runtime assembly 不能跨 stage1_model_name 使用")
        阻止一个 runtime assembly 跨 producer 使用。

        输入参数:
            - task: ProductionTask, 要交给当前 assembly 的任务

        输出:
            - None: producer identity 一致时返回，否则直接报错
        """
        if task.stage1_model_name != self.stage1_model_name:
            raise ValueError("runtime assembly 不能跨 stage1_model_name 使用")


__all__ = [
    "AGOccurrenceVoxelLoader",
    "CachedStage1WrapperProvider",
    "Stage1RuntimeAssembly",
    "build_production_tasks",
    "load_pdb_id_list",
    "shard_pdb_ids",
]
