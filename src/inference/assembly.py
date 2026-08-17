# -*- coding: utf-8 -*-
"""把 Stage A—G 正式资产、冻结配置和 checkpoint 装配为 Stage1 推理输入。

公开入口:
    - :func:`load_pdb_id_list`、:func:`shard_pdb_ids` 和 :func:`build_production_tasks`：读取冻结 PDB 清单，按清单位置索引取模生成互斥、完备的 worker 任务。
    - :class:`AGOccurrenceVoxelLoader`：读取 Stage E3 schema 3 的 ``ligand_area.npz``，将 occurrence 稀疏 ZYX 坐标恢复为完整图 C-order 线性索引。
    - :class:`Stage1RuntimeAssembly`：恢复 checkpoint wrapper，复用一个 PDB 的 ``Stage1Dataset`` cache，提供 full-map 滑窗、centered batch 和 occurrence 输入。

边界:
    - 本模块不重写 Dataset、模型 forward、概率融合、组件构造或 centered 产物算法；它只组装已有入口并搬运 batch。
    - 网格形状和离散索引使用 ZYX；物理原点、体素尺寸和原子坐标使用世界 XYZ；长度单位为 Å。
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch

from .centered import CenteredRequest
from .checkpoint import load_stage1_wrapper, resolve_checkpoint_config_path
from .full_map import window_starts_zyx
from .runner import FullMapTaskInput, ProductionTask


# Stage E3 ``ligand_area.npz`` 的 occurrence 稀疏 mask 字段名；捕获组是十进制 occurrence identity。
_MASK_KEY = re.compile(r"^mask_(\d+)$")



# 工具函数：清单读取、任务分片、配置契约和 batch 设备搬运。
def load_pdb_id_list(path: str | Path) -> tuple[str, ...]:
    """读取冻结 PDB 清单并保持源文件顺序。

    输入参数:
        - path: str | Path；JSON、JSONL 或纯文本清单；JSON 顶层可为字符串列表或包含 ``pdb_ids`` 列表的对象，JSONL 行和列表元素可为字符串或含 ``pdb_id`` 的对象。

    返回值:
        - pdb_ids: tuple[str, ...]；去空白、转小写、非空且无重复的 PDB identity，保持源文件顺序，不按名称重排。

    失败语义:
        - 文件不存在、JSON 顶层类型错误、对象缺少 ``pdb_id``、identity 为空、清单重复或清单为空时抛出异常。
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

    # list[str]；按源文件顺序保存规范化后的小写 PDB identity；对象行只读取显式 ``pdb_id``。
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
    """按冻结清单位置索引取模生成稳定、互斥且完备的 worker 分片。

    输入参数:
        - pdb_ids: Sequence[str]；已固定且无重复的 PDB identity 序列；顺序决定分片归属。
        - shard_index: int；当前 worker 的零基分片编号，必须满足 ``0 <= shard_index < shard_count``。
        - shard_count: int；分片总数，必须为正整数。

    返回值:
        - shard_pdb_ids: tuple[str, ...]；满足 ``row_index % shard_count == shard_index`` 的 identity，保持原相对顺序。
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
    """从同一冻结清单构造 runner 任务，不在运行时寻找“最新”样本。

    输入参数:
        - stage1_model_name: str；当前 Stage1 producer 正式名称。
        - split: str；当前数据划分名称。
        - pdb_list_path: str | Path；冻结 PDB 清单路径。
        - shard_index: int；当前 worker 的零基分片编号。
        - shard_count: int；分片总数。

    返回值:
        - tasks: tuple[ProductionTask, ...]；按当前分片 PDB 行序构造的 producer/split/PDB 任务；不探测输出目录，也不比较样本新旧。
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
    """从 resolved config 冷读 Dataset 通道和固定原子缓冲契约。

    输入参数:
        - config_path: Path；与 checkpoint 绑定的已解析配置文件。
        - stage1_model_name: str；CLI 绑定的 producer identity，必须与配置的 ``dataset.stage1_model_name`` 一致。

    返回值:
        - density_channel_config: dict[str, object]；完整保留 resolved dataset 密度通道顺序、开关和参数。
        - atom_buffer_radius: float；Find core BOX 外纳入受体原子的世界坐标半径，固定为 ``8.0 Å``。

    失败语义:
        - 缺少 dataset 或 density_channel_config、通道配置不是 mapping、producer 不一致或缓冲半径不是 8.0 时抛出异常。
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
    """把 collated batch 的 tensor 字段搬到目标设备。

    输入参数:
        - batch: Mapping[str, Any]；``Stage1Dataset.collate_fn`` 生成的混合 mapping；身份 list、计数语义和其他非 tensor 值保持 Python 对象。
        - device: str；目标设备，例如 ``cpu``、``cuda`` 或 ``cuda:0``。

    返回值:
        - device_batch: dict[str, Any]；保留所有键；tensor 字段使用 ``non_blocking=True`` 搬运，非 tensor 字段原样引用。
    """
    try:
        import torch
    except ImportError:
        return dict(batch)
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }




# occurrence 读取：将 Stage E3 稀疏 ZYX mask 转为完整图线性索引。
class AGOccurrenceVoxelLoader:
    """读取 Stage E3 schema 3 occurrence 稀疏 mask 并恢复线性索引。

    构造参数:
        - data_root: str | Path；A-G 正式数据根目录；后续从 ``density/<pdb_id>`` 读取 occurrence NPZ 和 union NPY。

    调用返回:
        - occurrences: dict[int, np.ndarray]；occurrence identity 到 int64 ``(K_occ,)`` 完整图 C-order 线性 voxel index 的映射，键按数值升序插入。
    """
    def __init__(self, data_root: str | Path) -> None:
        """绑定 A-G 正式根目录，不在构造阶段读取 occurrence 文件。

        输入参数:
            - data_root: str | Path；后续 occurrence 调用读取的正式数据根目录。
        """
        self.data_root = Path(data_root)

    def __call__(
        self,
        pdb_id: str,
        full_shape_zyx: tuple[int, int, int],
    ) -> dict[int, np.ndarray]:
        """返回 occurrence 到完整图 C-order 线性 voxel index 的映射。

        输入参数:
            - pdb_id: str；当前 PDB identity；读取目录前转为小写。
            - full_shape_zyx: tuple[int, int, int]；完整图的 ``(D, H, W)`` ZYX 形状，必须与 NPZ ``grid_shape_zyx`` 和 union NPY 一致。

        返回值:
        - occurrences: dict[int, np.ndarray]；每个 occurrence 的 int64 ``(K_occ,)`` C-order 线性索引；每个稀疏 mask 的坐标按 ZYX 字典序检查且不重复。

        文件契约:
            - ``ligand_area.npz`` 必须是 schema 3，``mask_<id>`` 字段必须是整数 ``(K_occ, 3)`` ZYX index。
            - ``union_mask.npy`` 必须是 bool ``(1, D, H, W)``，且逐体素等于全部 occurrence mask 的并集。
        """
        path = self.data_root / "density" / str(pdb_id).lower() / "ligand_area.npz"
        with np.load(path, allow_pickle=False) as data:
            if "schema_version" not in data or int(np.asarray(data["schema_version"]).item()) != 3:
                raise ValueError(f"{path}: occurrence loader 只接受 Stage E3 schema_version=3")
            shape = tuple(int(value) for value in np.asarray(data["grid_shape_zyx"]).tolist())
            expected_shape = tuple(int(value) for value in full_shape_zyx)
            if shape != expected_shape:
                raise ValueError(f"{path}: grid_shape_zyx={shape} 与 probability={expected_shape} 不一致")
            union = np.load(path.with_name("union_mask.npy"), mmap_mode="r", allow_pickle=False)
            if union.dtype != np.bool_ or union.shape != (1, *shape):
                raise ValueError(f"{path.with_name('union_mask.npy')}: 必须为 (1,Z,Y,X)")

            # list[tuple[int, str]]；按 occurrence identity 数值升序排列的严格 ``mask_<整数>`` 字段。
            mask_keys = sorted(
                (int(match.group(1)), key)
                for key in data.files
                if (match := _MASK_KEY.fullmatch(key)) is not None
            )
            # dict[int, np.ndarray]；occurrence identity 到升序完整图 C-order 线性索引；插入顺序即当前读取到的 occurrence 字段顺序。
            occurrences: dict[int, np.ndarray] = {}
            # np.ndarray bool (D, H, W)；由全部 occurrence 稀疏 mask 重建的并集，用于逐体素核对 union_mask[0]。
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

                # np.ndarray int64 (K_occ,)；稀疏 ZYX voxel 行在完整图中的 C-order 线性索引。
                linear = np.ravel_multi_index(sparse.T, shape).astype(np.int64)
                occurrences[int(occurrence_id)] = linear
                if sparse.size:
                    reconstructed_union[tuple(sparse.T)] = True
            if not np.array_equal(reconstructed_union, union[0]):
                raise ValueError(f"{path}: union_mask 不等于全部 mask_{{cid}} 的并集")
        return occurrences



# wrapper 读取：绑定 checkpoint/config，并在 worker 生命周期内复用 eval wrapper。
class CachedStage1WrapperProvider:
    """严格恢复并在当前推理 worker 内缓存一个完整 Stage1 wrapper。

    构造参数:
        - stage1_model_name: str；provider 绑定的 producer identity。
        - checkpoint_path: str | Path；完整 Stage1 wrapper checkpoint。
        - resolved_config_path: str | Path | None；与 checkpoint 绑定的 resolved config；为 ``None`` 时由 checkpoint 快照规则解析。
        - device: str；wrapper 最终所在设备，例如 ``cpu`` 或 ``cuda:0``。
        - wrapper_loader: Callable | None；显式 wrapper 加载器；为 ``None`` 时调用严格的 ``load_stage1_wrapper``。
        - allow_current_workspace_code: bool；checkpoint 缺少代码快照时是否允许当前工作区代码；正式可复现运行应保持假。

    调用契约:
        - 输入 ``task``：ProductionTask；其 producer 必须等于 provider 绑定值。
        - 返回 ``wrapper``：已恢复、移动到目标设备并处于 eval 的完整 wrapper；同一 provider 生命周期只加载一次。
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
        """绑定 checkpoint、配置、producer 和目标设备，延迟到首次调用时恢复 wrapper。

        输入参数:
            - stage1_model_name: str；provider 绑定的 producer identity。
            - checkpoint_path: str | Path；完整 wrapper checkpoint。
            - resolved_config_path: str | Path | None；与 checkpoint 对应的 resolved config，或由 checkpoint 快照解析。
            - device: str；wrapper 的目标设备。
            - wrapper_loader: Callable | None；可替换的加载器；为空使用正式加载入口。
            - allow_current_workspace_code: bool；是否允许缺少代码快照时使用当前工作区。

        状态变化:
            - 保存路径、设备和加载策略；``_wrapper`` 初始为 ``None``，不在构造时读取 checkpoint。
        """
        self.stage1_model_name = str(stage1_model_name)
        self.checkpoint_path = Path(checkpoint_path)
        self.resolved_config_path = (None if resolved_config_path is None else Path(resolved_config_path))
        self.device = str(device)
        self.wrapper_loader = wrapper_loader
        self.allow_current_workspace_code = bool(allow_current_workspace_code)
        self._wrapper: Any | None = None

    def __call__(self, task: ProductionTask) -> Any:
        """按任务 producer 校验并返回缓存的 eval wrapper。

        输入参数:
            - task: ProductionTask；producer 必须等于 provider 绑定的 ``stage1_model_name``。

        返回值:
            - wrapper: Any；首次调用时由 checkpoint/config 恢复、搬到目标设备并调用 ``eval``；后续调用返回同一对象。

        失败语义:
            - 任务 producer 不一致时抛出 ``ValueError``；checkpoint 恢复失败由加载器原样抛出。
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




# 请求物化：通过正式 Stage1Dataset 和 collator 组装 full-map、centered batch。
class _TaskDatasetMaterializer:
    """为同一 producer/split/PDB 复用 Dataset 并组装 full-map、centered batch。

    构造参数:
        - data_root: Path；A-G 正式数据根目录。
        - task: ProductionTask；当前 producer、split 和 PDB identity。
        - density_channel_config: Mapping[str, Any]；从 resolved config 冷读的 producer 密度通道配置。
        - atom_buffer_radius: float；Find 核心 BOX 外受体原子缓冲半径，正式值为 ``8 Å``。
        - cache_max_bytes: int；当前进程 Dataset 的 PDB 资产缓存字节上限。
        - device: str；collated tensor batch 的目标设备。

    生命周期:
        - 构造一个不启用随机旋转的 centered-mode ``Stage1Dataset``；full-map 和 centered 请求均通过该 Dataset 的 ``materialize_request`` 与同一个 collator 展开。
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
        """创建当前 task 绑定的 Dataset、collator 和设备信息。

        输入参数:
            - data_root: Path；A-G 正式数据根目录。
            - task: ProductionTask；producer/split/PDB identity。
            - density_channel_config: Mapping[str, Any]；resolved config 中的 producer 通道配置。
            - atom_buffer_radius: float；Find 的 8 Å 核心 BOX 外原子缓冲半径。
            - cache_max_bytes: int；Dataset worker-local 资产缓存上限。
            - device: str；输出 batch tensor 的目标设备。

        状态变化:
            - 用一个合法的零起点 seed request 绑定 PDB；真实滑窗和 centered 起点在公开方法中动态传给共享 Dataset，不写入补零样本。
        """
        from src.datasets.stage1_dataset import Stage1Dataset
        from src.datasets.stage1_requests import ResolvedStage1Crop

        # ResolvedStage1Crop；仅用于让 Dataset 绑定当前 PDB 的无目标请求，真实滑窗和 centered 起点由公开方法动态物化。
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
        """把一批完整图滑窗起点物化为目标设备上的 Stage1 batch。

        输入参数:
            - starts_zyx: Sequence[tuple[int, int, int]]；长度为 B_window；每项是完整图离散 ZYX 真实 BOX corner index。

        返回值:
            - batch: dict[str, Any]；第一维 B 等于起点数；密度、hardmask、几何按 B 堆叠，Find 原子表沿原子轴拼接并由 ``atom_counts``、``atom_offsets`` 和 ``atom_batch_index`` 保存归属。

        失败语义:
            - 起点越界、完整图不足 80³ 或 Dataset 资产契约失败时直接抛出异常，不进行 padding。
        """
        from src.datasets.stage1_requests import ResolvedStage1Crop

        # list[dict[str, Any]]；长度 B_window；每项由正式 Dataset 以 role=sliding 物化一个不含补零区域的真实 BOX。
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
        """复用 Dataset cache 返回 full-map 几何和完整受体坐标。

        返回值:
            - shape_zyx: tuple[int, int, int]；exp 完整图的 ``(D, H, W)`` ZYX 形状。
            - voxel_size_xyz: np.ndarray float32 ``(3,)``；世界 XYZ 体素尺寸，单位为 Å。
            - origin_xyz: np.ndarray float32 ``(3,)``；完整图 voxel-grid corner 的世界 XYZ 原点，单位为 Å。
            - receptor_coord_xyz: np.ndarray float32 ``(N_receptor, 3)``；完整受体世界 XYZ 坐标，单位为 Å。
        """
        return self.dataset.full_map_context(self.task.pdb_id)

    def centered_batch(self, requests: Sequence[CenteredRequest]) -> dict[str, Any]:
        """把一组 centered 请求物化为一次目标设备 Stage1 batch。

        输入参数:
            - requests: Sequence[CenteredRequest]；同一 producer、split、PDB 的有序 centered 请求；每项提供完整图 ZYX BOX 起点及 tree/node/threshold 来源身份。

        返回值:
            - batch: dict[str, Any]；第一维 B 等于请求数；dense 字段按 B 堆叠，Find 原子字段按原子轴拼接，契约与训练 ``Stage1BatchCollator`` 一致。

        兼容字段:
            - 推理不需要逐原子 label，但旧版 Find 伪原子注入读取其 bool dtype；若 collator 产出 Find 原子表而无 ``atom_label``，本方法补一个全 False tensor，不表示真实监督。
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
        batch = self.collator(samples)
        if "atom_global_indices" in batch and "atom_label" not in batch:
            # 兼容旧版 Find 伪原子注入读取的 bool 字段；推理只需要 dtype 和形状，不把它当作真实监督。
            batch["atom_label"] = batch["atom_global_indices"].new_zeros(
                batch["atom_global_indices"].shape,
                dtype=torch.bool,
            )
        return _move_batch_to_device(batch, self.device)





# runtime assembly：把 wrapper、Dataset materializer 和 occurrence loader 绑定到一个 worker。
class Stage1RuntimeAssembly:
    """组装一个推理 worker 所需的 Stage1 wrapper、Dataset materializer 和 occurrence loader。

    该类只建立可调用的输入提供器，不执行模型 forward，也不生成 centered 或 full-map 归档文件。``ProductionTask`` 的身份是 ``(stage1_model_name, split, pdb_id)``；所有公开方法都会拒绝跨 producer 任务。

    构造参数:
        - data_root: str | Path；A-G 正式数据根目录；Dataset 和 occurrence 文件都从此处读取。
        - stage1_model_name: str；当前推理绑定的 Stage1 producer identity。
        - checkpoint_path: str | Path；完整 Stage1 wrapper checkpoint。
        - resolved_config_path: str | Path | None；与 checkpoint 对应的 resolved config；为空时按 checkpoint 快照规则解析。
        - device: str；wrapper 和输出 tensor batch 的目标设备；PDB identity、计数和其他 Python 元数据不搬运。
        - window_batch_size: int；一次 full-map forward 的窗口数，必须为正。
        - centered_batch_size: int；centered forward 的 80³ BOX 数，保存为 assembly 配置供上层使用。
        - cache_max_bytes: int；单进程 Dataset 资产缓存字节上限，不预分配内存。
        - wrapper_loader: Callable | None；可替换的 wrapper 加载器；为空使用正式 checkpoint loader。
        - allow_current_workspace_code: bool；checkpoint 缺少代码快照时是否允许当前工作区代码。

    生命周期状态:
        - ``resolved_config_path``、``density_channel_config`` 和 ``atom_buffer_radius`` 从同一 resolved config 冷读。
        - ``wrapper_provider`` 按需恢复并缓存一个 eval wrapper；``_active_materializer`` 只缓存最近一个完全相同的 ProductionTask。
        - ``occurrence_loader`` 不缓存模型或 Dataset；每次调用按指定 PDB 和完整图形状核对 occurrence 文件。

    公开输出:
        - ``full_map_input(task)``：FullMapTaskInput，含完整图几何、滑窗 batch builder、Find hardmask、wrapper 和窗口批大小。
        - ``centered_batch_builder(task)``：接收 centered 请求序列并返回 batch 的 builder。
        - ``occurrence_voxels(task, full_shape_zyx)``：返回 occurrence identity 到完整图线性 voxel index 的映射。
    """

    def __init__(
        self,
        data_root: str | Path,
        stage1_model_name: str,
        checkpoint_path: str | Path,
        resolved_config_path: str | Path | None,
        device: str,
        window_batch_size: int,
        centered_batch_size: int = 10,
        cache_max_bytes: int = 536_870_912_000,
        wrapper_loader: Callable[..., Any] | None = None,
        allow_current_workspace_code: bool = False,
    ) -> None:
        """绑定推理输入的路径、producer、设备和批大小，并延迟恢复 wrapper。

        输入参数:
            - data_root: str | Path；A-G 正式数据根目录。
            - stage1_model_name: str；当前推理绑定的 producer identity。
            - checkpoint_path: str | Path；完整 Stage1 wrapper checkpoint。
            - resolved_config_path: str | Path | None；已解析配置，或由 checkpoint 快照解析。
            - device: str；wrapper 和 batch tensor 的目标设备。
            - window_batch_size: int；full-map 窗口批大小，必须为正。
            - centered_batch_size: int；centered BOX 批大小。
            - cache_max_bytes: int；Dataset 资产 cache 的字节上限。
            - wrapper_loader: Callable | None；可选 wrapper 加载器。
            - allow_current_workspace_code: bool；是否允许当前工作区代码回退。

        状态变化:
            - 解析并保存 checkpoint 绑定的 resolved config、密度通道和固定 8 Å atom buffer；创建 wrapper provider 和 occurrence loader，但 wrapper 仍延迟到首次任务调用。
        """
        self.data_root = Path(data_root)
        self.stage1_model_name = str(stage1_model_name)
        self.device = str(device)
        self.window_batch_size = int(window_batch_size)
        self.centered_batch_size = int(centered_batch_size)
        self.cache_max_bytes = int(cache_max_bytes)
        if self.window_batch_size <= 0:
            raise ValueError("window_batch_size 必须为正")
        # Path；与 checkpoint 快照绑定的唯一 resolved config，后续 Dataset 契约只从该文件冷读。
        config_path = resolve_checkpoint_config_path(checkpoint_path, resolved_config_path)
        self.resolved_config_path = config_path
        # tuple[dict[str, object], float]；推理 Dataset 必须复用的密度通道配方和固定 8 Å 原子缓冲半径。
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
        """构造一个 PDB 的 full-map 滑窗输入。

        输入参数:
            - task: ProductionTask；当前 producer、split 和 PDB identity。

        返回值:
            - inputs: FullMapTaskInput；包含 eval wrapper、完整图 ZYX 形状、滑窗 batch builder、Find 专用完整图受体 hardmask、窗口批大小和世界 XYZ 几何。

        预检:
            - 使用真实 80³、stride 40 的滑窗生成器验证完整图可被无 padding 窗口覆盖；Find 额外构造完整图受体 home-voxel hardmask。
        """
        self._check_task(task)
        wrapper = self.wrapper_provider(task)
        materializer = self._materializer(task)
        # tuple[shape_zyx, voxel_size_xyz, origin_xyz, receptor_coords]；几何轴分别是完整图 ZYX 和世界 XYZ。
        full_shape, voxel_size, origin, receptor_coords = (materializer.full_map_context())
        # 只读验证真实 80³/stride40 窗口覆盖完整图，使形状错误在申请 full-map 融合数组前暴露。
        window_starts_zyx(full_shape, (80, 80, 80), (40, 40, 40))
        if task.stage1_model_name.startswith("Find"):
            from src.datasets.box_geometry import build_hardmask_from_world_coordinates
            # np.ndarray bool (D, H, W)；完整图 ZYX 网格上的受体 home voxel，Find 融合后在 True 位置清零配体概率。
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
        """返回当前 task 的 centered batch builder。

        输入参数:
            - task: ProductionTask；当前 producer、split 和 PDB identity。

        返回值:
            - builder: Callable[[Sequence[CenteredRequest]], Mapping[str, object]]；上层按 ``centered_batch_size`` 分批调用；同一 PDB 复用当前 Dataset cache。
        """
        self._check_task(task)
        self.wrapper_provider(task)
        return self._materializer(task).centered_batch  # 接收 centered 请求序列并返回 collated batch。

    def occurrence_voxels(
        self,
        task: ProductionTask,
        full_shape_zyx: tuple[int, int, int],
    ) -> Mapping[int, np.ndarray]:
        """读取当前 task 的 occurrence 稀疏体素线性索引。

        输入参数:
            - task: ProductionTask；当前 producer、split 和 PDB identity。
            - full_shape_zyx: tuple[int, int, int]；完整图的 ``(D, H, W)`` ZYX 形状。

        返回值:
            - occurrences: Mapping[int, np.ndarray]；occurrence identity 到 int64 ``(K_occ,)`` 完整图 C-order 线性 voxel index 的映射。
        """
        self._check_task(task)
        return self.occurrence_loader(task.pdb_id, full_shape_zyx)

    def _materializer(self, task: ProductionTask) -> _TaskDatasetMaterializer:
        """取得与 task 完全绑定的 Dataset materializer。

        输入参数:
            - task: ProductionTask；producer、split 或 PDB 任一变化都会切换绑定。

        返回值:
            - materializer: _TaskDatasetMaterializer；当前 task 的 Dataset、cache 和 collator。

        生命周期:
            - 相同 task 复用现有 materializer；切换 task 时只替换引用，旧 materializer 由 Python 生命周期回收。
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
        """阻止一个 runtime assembly 跨 producer 使用。

        输入参数:
            - task: ProductionTask；要交给当前 assembly 的任务。

        返回值:
            - None；producer identity 一致时返回。

        失败语义:
            - ``task.stage1_model_name`` 与 assembly 绑定值不一致时抛出 ``ValueError``。
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
