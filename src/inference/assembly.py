# -*- coding: utf-8 -*-
"""把 AdaLigand A—G 产物、冻结配置和 checkpoint 装配成 Stage1 推理 providers。

主要入口:
    - `load_pdb_id_list`、`shard_pdb_ids`、`build_production_tasks`: 冷读冻结 PDB 清单，并按原始行号取模生成稳定、互斥、完备的 worker 任务。
    - `AGOccurrenceVoxelLoader`: 读取 Stage E3 schema-v3 `ligand_area.npz`，恢复 occurrence 的完整图 C-order voxel 索引。
    - `Stage1RuntimeAssembly`: 严格恢复完整 wrapper，复用单 PDB `Stage1Dataset` cache，并提供完整图滑窗、centered batch 和 occurrence providers。

本模块不重新实现 Dataset、模型 forward、概率融合、组件构造或 centered 产物算法。坐标约定为：网格形状和离散索引使用 ZYX，物理原点、体素尺寸和原子坐标使用世界 XYZ，长度单位 Å。
"""

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


# Stage E3 `ligand_area.npz` 的 occurrence 稀疏 mask 字段名；捕获组是十进制 occurrence_id。
_MASK_KEY = re.compile(r"^mask_(\d+)$")


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
    return tuple(
        ProductionTask(stage1_model_name, str(split), pdb_id) for pdb_id in pdb_ids
    )


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
        绑定 A-G 正式数据根目录。

        输入参数:
            - data_root: str | Path, 包含 `density/{pdb_id}/ligand_area.npz` 的根目录

        输出:
            - None: 原地保存规范化 Path
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
            - occurrences: dict[int, np.ndarray], occurrence_id 到 `(K_occ,)` int64 完整图 C-order 离散线性 voxel 索引的映射；每个数组按 ZYX 字典序对应的 C-order 升序排列且唯一。

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

    def for_task(
        self,
        task: ProductionTask,
        full_shape_zyx: tuple[int, int, int],
    ) -> dict[int, np.ndarray]:
        """
        适配 `make_component_role_producer` 的 task-first 签名。

        输入参数:
            - task: ProductionTask, 当前 producer/split/PDB identity
            - full_shape_zyx: tuple[int,int,int], (3,), probability 完整图 ZYX voxel-grid shape

        输出:
            - occurrences: dict[int,np.ndarray], occurrence_id 到完整图 C-order 离散线性 voxel indices 的映射
        """
        return self(task.pdb_id, full_shape_zyx)


class CachedStage1WrapperProvider:
    """
    从唯一 checkpoint/config 严格恢复并在当前 worker 内复用完整 wrapper。

    输入参数:
        - stage1_model_name: str, 当前 provider 绑定的 producer 正式名
        - checkpoint_path: str | Path, 完整 Stage1 wrapper checkpoint
        - resolved_config_path: str | Path | None, 与 checkpoint 对应的 resolved config；None 时按 checkpoint 快照的固定相邻规则解析。
        - device: str, wrapper 最终所在设备，例如 `cpu` 或 `cuda:0`。
        - wrapper_loader: Callable | None, 可选显式加载器；None 时使用严格的 `load_stage1_wrapper`。
    """

    def __init__(
        self,
        stage1_model_name: str,
        checkpoint_path: str | Path,
        resolved_config_path: str | Path | None,
        device: str,
        wrapper_loader: Callable[..., Any] | None = None,
    ) -> None:
        """
        保存恢复契约并初始化空 wrapper cache。

        输入参数:
            - stage1_model_name: str, 当前 producer 正式名
            - checkpoint_path: str | Path, 完整 wrapper checkpoint
            - resolved_config_path: str | Path | None, resolved config 路径
            - device: str, wrapper 最终设备
            - wrapper_loader: Callable | None, 可选自定义加载器

        输出:
            - None: 原地保存路径与设备，`_wrapper` 初始为 None
        """
        self.stage1_model_name = str(stage1_model_name)
        self.checkpoint_path = Path(checkpoint_path)
        self.resolved_config_path = (
            None if resolved_config_path is None else Path(resolved_config_path)
        )
        self.device = str(device)
        self.wrapper_loader = wrapper_loader
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
            )
            if hasattr(wrapper, "to"):
                wrapper = wrapper.to(self.device)
            if hasattr(wrapper, "eval"):
                wrapper.eval()
            self._wrapper = wrapper
        return self._wrapper


class _TaskDatasetMaterializer:
    """
    为一个 PDB 复用同一 Stage1Dataset cache 并物化动态推理请求。

    输入参数:
        - data_root: Path, A-G 正式数据根目录
        - task: ProductionTask, 当前 producer/split/PDB identity
        - density_channel_config: Mapping[str, Any], 从 resolved config 冷读的 producer density channel 配方。
        - atom_buffer_radius: float, Find 在 core BOX 外纳入原子的世界坐标 buffer 半径，正式值为 8 Å。
        - cache_max_bytes: int, 当前 Dataset 的 worker-local PDB 资产缓存字节上限。
        - device: str, collated tensor batch 的目标设备。
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
        构造绑定单一 task 的无旋转 Stage1Dataset 与 collator。

        输入参数:
            - data_root: Path, A-G 正式数据根目录
            - task: ProductionTask, 当前 producer/split/PDB identity
            - density_channel_config: Mapping[str,Any], producer density channel recipe
            - atom_buffer_radius: float, 世界坐标原子 buffer 半径
            - cache_max_bytes: int, Dataset 资产缓存上限
            - device: str, 输出 batch 设备

        输出:
            - None: 原地建立 Dataset、collator、task 与 device
        """
        # ResolvedStage1Crop, 仅用于让 Dataset 绑定当前 PDB 的合法无目标请求；真实滑窗和 centered 请求由后续方法动态物化。
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
        把一批滑窗起点交给统一 Dataset/materializer 与正式 collator。

        输入参数:
            - starts_zyx: Sequence[tuple[int, int, int]], 长度 B_window；每项是完整图离散 ZYX voxel-index 窗口起点。

        输出:
            - batch: dict[str, Any], fixed-grid `(B_window, ...)` 字段与可选 ragged 原子字段组成的目标设备模型输入；具体字段由统一 `Stage1Dataset.collate_fn` 定义。
        """
        # list[dict[str, Any]], 长度 B_window；每项由正式 Dataset 以 `role=sliding` 物化一个真实无 padding BOX。
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
        复用 Dataset cache 返回完整图几何与 receptor 坐标。

        输出:
            - shape_zyx: tuple[int, int, int], 完整图的 ZYX voxel 形状 `(D, H, W)`。
            - voxel_size_xyz: float array, (3,), 世界 XYZ 三轴的体素尺寸，单位 Å/voxel。
            - origin_xyz: float array, (3,), 完整图 voxel-grid 起点的世界 XYZ 坐标，单位 Å。
            - receptor_coord_xyz: float array, (N_receptor, 3), 当前 PDB 全部 receptor 原子的世界 XYZ 坐标，单位 Å。
        """
        return self.dataset.full_map_context(self.task.pdb_id)

    def centered_batch(self, request: CenteredRequest) -> dict[str, Any]:
        """
        把一个正式 centered 请求物化成 batch-size-one 模型输入。

        输入参数:
            - request: CenteredRequest, identity 与完整图离散 ZYX voxel-index BOX corner 起点已冻结的请求

        输出:
            - batch: dict[str, Any], batch size 1 的目标设备 Stage1 模型输入；字段契约与训练 `Stage1Dataset.collate_fn` 完全一致。
        """
        if (
            request.stage1_model_name != self.task.stage1_model_name
            or request.split != self.task.split
            or request.pdb_id != self.task.pdb_id
        ):
            raise ValueError("CenteredRequest 与当前 task 身份不一致")
        # dict[str, Any], 当前冻结 BOX 起点对应的单请求 Dataset 样本；`require_targets=False` 表示推理不加载训练标签。
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
    """
    把 A-G root、resolved config 与 checkpoint 装成全部 inference providers。

    输入参数:
        - data_root: str | Path, A-G 正式数据根目录
        - stage1_model_name: str, 当前 assembly 绑定的 producer 正式名
        - checkpoint_path: str | Path, 完整 Stage1 wrapper checkpoint
        - resolved_config_path: str | Path | None, checkpoint 对应的 resolved config
        - device: str, wrapper 与 tensor batch 的目标设备
        - window_batch_size: int, 单次完整图 voxel-only forward 的窗口数量 B_window。
        - cache_max_bytes: int, 单 PDB Dataset 资产缓存上限，默认 512 MiB。
        - wrapper_loader: Callable | None, 可选显式 wrapper 加载器；测试可注入，正式运行使用默认严格加载器。

    缓存边界:
        - `wrapper_provider`: 整个 worker 生命周期复用同一 eval wrapper。
        - `_active_materializer`: 只复用最近一个完全相同的 producer/split/PDB 任务；切换任务时释放旧 Dataset 引用。
    """

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
        """
        冷读 Dataset 契约并建立 wrapper、occurrence 与 materializer providers。

        输入参数:
            - data_root: str | Path, A-G 正式数据根目录
            - stage1_model_name: str, 当前 producer 正式名
            - checkpoint_path: str | Path, 完整 wrapper checkpoint
            - resolved_config_path: str | Path | None, resolved config 路径
            - device: str, 运行设备
            - window_batch_size: int, 单次窗口 forward 数
            - cache_max_bytes: int, Dataset 资产缓存上限
            - wrapper_loader: Callable | None, 可选自定义加载器

        输出:
            - None: 原地建立全部 providers；materializer cache 初始为空
        """
        self.data_root = Path(data_root)
        self.stage1_model_name = str(stage1_model_name)
        self.device = str(device)
        self.window_batch_size = int(window_batch_size)
        self.cache_max_bytes = int(cache_max_bytes)
        if self.window_batch_size <= 0:
            raise ValueError("window_batch_size 必须为正")
        # Path, 与 checkpoint 快照绑定的唯一 resolved config；后续 Dataset 契约只从该文件冷读。
        config_path = resolve_checkpoint_config_path(
            checkpoint_path, resolved_config_path
        )
        self.resolved_config_path = config_path
        # `dict[str, object]` 与 float，推理 Dataset 必须复用的 density channel 配方和固定 8 Å 原子 buffer。
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
        materializer = self._materializer(task)
        # tuple 与三个数组，依次为完整图 ZYX 形状、世界 XYZ 体素尺寸、世界 XYZ 原点和 receptor 原子世界 XYZ 坐标。
        full_shape, voxel_size, origin, receptor_coords = (
            materializer.full_map_context()
        )
        # 只读验证正式 80³/stride40 能以真实无 padding 窗口覆盖完整图，使形状错误在申请融合大数组前暴露。
        window_starts_zyx(full_shape, (80, 80, 80), (40, 40, 40))
        if task.stage1_model_name.startswith("Find"):
            # bool, (D, H, W), 完整 ZYX voxel 网格上的 receptor home voxels；Find 完整融合后在 True 位置清零概率。
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
        """
        返回复用单 PDB Dataset cache 的 centered batch builder。

        输入参数:
            - task: ProductionTask, 当前 producer/split/PDB identity

        输出:
            - builder: Callable[[CenteredRequest], Mapping[str, object]], 把一个冻结的完整图 ZYX BOX 起点物化为 batch-size-one 模型输入。
        """
        self._check_task(task)
        return self._materializer(task).centered_batch

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
        阻止一个 runtime assembly 跨 producer 使用。

        输入参数:
            - task: ProductionTask, 要交给当前 assembly 的任务

        输出:
            - None: producer identity 一致时返回，否则直接报错
        """
        if task.stage1_model_name != self.stage1_model_name:
            raise ValueError("runtime assembly 不能跨 stage1_model_name 使用")


def _load_dataset_contract(
    config_path: Path,
    stage1_model_name: str,
) -> tuple[dict[str, object], float]:
    """
    从 resolved config 冷读 Dataset 输入通道与 8 Å buffer 契约。

    输入参数:
        - config_path: Path, checkpoint 对应的 resolved config
        - stage1_model_name: str, CLI 指定的 producer 正式名

    输出:
            - density_channel_config: dict[str, object], Dataset 构造当前 producer density channels 的完整 resolved 配置。
            - atom_buffer_radius: float, Find 在世界坐标中选择 core BOX 外原子的固定半径，严格为 8.0 Å。
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
    # 标准 Python 容器，完整保留 resolved `dataset.density_channel_config` 的通道顺序、开关与参数。
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
    """
    只移动 tensor 字段，PDB identity 和 ragged 元数据保持 Python 值。

    输入参数:
        - batch: Mapping[str, Any], `Stage1Dataset.collate_fn` 输出；可能同时包含 tensor、PDB identity 和 ragged Python 元数据。
        - device: str, tensor 字段的目标设备。

    输出:
        - moved_batch: dict[str, Any], tensor 字段以 non-blocking 方式移动到 device，其余字段保持原对象和值。
    """
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
