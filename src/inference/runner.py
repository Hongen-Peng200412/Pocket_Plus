"""以单个 PDB 和产物角色为粒度、无需数据库的可续跑 Stage1 生产编排。

主要入口:
    - `make_probability_role_producer`: 生成完整图概率产物的回调。
    - `make_component_role_producer`: 生成组件森林、组件谱系组和交集产物的回调。
    - `make_f1_clg_centered_role_producers`: 生成 F1 与 CLG 居中特征的两个回调。
    - `make_selected_refined_role_producer`: 根据 Selector 结果重跑精修居中特征。
    - `Stage1ProductionRunner`: 在 PDB 互斥租约内按依赖顺序补齐缺失产物。

编排器只读取任务身份、产物依赖顺序和文件状态；各产物的科学计算由注入的生成
回调完成。一次 PDB 租约内仅补齐缺失角色，已有 `_COMPLETE` 的角色保持不变；
`_BLOB_EXCEED` 和被其他进程持有的 `_RUNNING` 都会形成可统计的确定结果。
"""

from __future__ import annotations

import json
import os
import socket
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np

from src.artifacts.io import load_npz_strict
from src.artifacts.paths import OUTPUT_ROLES, Stage1ArtifactPaths
from src.artifacts.states import PdbRunningLease, is_role_complete, mark_blob_exceed
from src.component_lineage.clg import CLGEnumerationConfig, enumerate_clgs
from src.component_lineage.forest import (
    build_component_forest,
    count_f1_eligible,
    publish_component_artifacts,
)
from src.component_lineage.overlap import build_candidate_occurrence_overlap
from src.component_lineage.structures import ComponentForest, clgs_from_arrays
from .centered import (
    CenteredGeometry,
    CenteredRequest,
    produce_clg_centered_entries,
    produce_f1_centered_entries,
    produce_threshold_centered_entries,
    produce_selected_refined_entries,
    publish_centered_entries,
)
from .full_map import infer_full_map, publish_full_map
from .li_centered import build_li_nodes, publish_li_centered_entries


@dataclass(frozen=True)
class ProductionTask:
    """
    表示可重分配、可重入的一个 producer/split/PDB 任务。

    输入参数:
        - stage1_model_name: str, `unet_c1/find1/find2` 之一，标识产物的 Stage1
          模型来源
        - split: str, `validation/calibration/train` 之一，标识数据划分
        - pdb_id: str, 当前结构的稳定 PDB 标识
    """
    stage1_model_name: str
    split: str
    pdb_id: str


@dataclass(frozen=True)
class FullMapTaskInput:
    """
    一个 PDB 的完整图 probability role 所需运行时输入。

    输入参数:
        - model: object, 已严格恢复参数、处于推理模式且实现 `forward_voxel_probability` 的完整模型包装器
        - full_shape_zyx: tuple[int, int, int], `(D, H, W)`，完整图离散 voxel 网格的 ZYX 轴尺寸
        - window_batch_builder: Callable, 输入一批完整图离散 ZYX voxel-index BOX 起点，返回统一 Dataset/Collator 批次
        - receptor_hardmask_full: np.ndarray | None, `(D, H, W)`，完整图离散 ZYX voxel 网格上的 receptor 布尔掩码；Find 模型必填，unet_c1 为 None
        - window_batch_size: int, 单次仅计算 voxel 概率的模型调用所含滑窗数
        - origin_xyz: tuple[float, float, float], `(3,)`，完整图索引原点对应的连续世界坐标 XYZ，单位 Å
        - voxel_size_xyz: tuple[float, float, float], `(3,)`，世界坐标 XYZ 各轴的 voxel 间距，单位 Å/voxel
    """
    model: object
    full_shape_zyx: tuple[int, int, int]
    window_batch_builder: Callable[
        [Sequence[tuple[int, int, int]]], Mapping[str, object]
    ]
    receptor_hardmask_full: np.ndarray | None
    window_batch_size: int
    origin_xyz: tuple[float, float, float]
    voxel_size_xyz: tuple[float, float, float]


@dataclass(frozen=True)
class RunRecord:
    """
    保存编排器对一个 PDB 的状态摘要。

    输入参数:
        - task: ProductionTask, 当前任务身份
        - status: str, `completed/skipped_complete/skipped_running/blob_exceed` 之一，表示本次调用的最终状态
        - completed_roles: tuple[str, ...], 本次进程实际新发布完成的产物角色；跳过时为空，处理途中发生体素数超限时可保留已完成前缀
    """
    task: ProductionTask
    status: str
    completed_roles: tuple[str, ...]


class BlobExceeded(RuntimeError):
    """
    通知 runner 当前 PDB 的 `N_F1_eligible` 超过正式上限。

    输入参数:
        - n_f1_eligible: int, `t_F1` 层 eligible component 数
        - limit: int, 正式上限，当前为 200
    """
    def __init__(self, n_f1_eligible: int, limit: int) -> None:
        """
        保存超限计数并构造供 runner 记录 `_BLOB_EXCEED` 的异常消息。

        输入参数:
            - n_f1_eligible: int, `t_F1` 层 eligible component 数
            - limit: int, eligible component 正式上限
        """
        self.n_f1_eligible = int(n_f1_eligible)
        self.limit = int(limit)
        super().__init__(f"N_F1_eligible={self.n_f1_eligible} > limit={self.limit}")


# 产物生成回调在给定路径下原子发布自身数据，完成校验后最后写 `_COMPLETE`。
RoleProducer = Callable[[ProductionTask, Stage1ArtifactPaths], None]
# occurrence 提供器返回 `occurrence_id -> 完整图 C-order 线性 voxel 索引`。
OccurrenceVoxelProvider = Callable[
    [ProductionTask, tuple[int, int, int]], Mapping[int, np.ndarray]
]
# 两个提供器分别复用已恢复的完整 Stage1 wrapper，并把居中请求转换为正式批次。
CenteredWrapperProvider = Callable[[ProductionTask], object]
CenteredBatchBuilderProvider = Callable[
    [ProductionTask], Callable[[Sequence[CenteredRequest]], Mapping[str, object]]
]
# Selector 结果路径提供器允许把 selection 产物放在 Stage1 PDB 目录之外。
SelectionPathProvider = Callable[[ProductionTask, Stage1ArtifactPaths], Path]



# 完整图输入提供器把任务身份解析为一次 probability 生产所需的全部冻结输入。
FullMapInputProvider = Callable[[ProductionTask], FullMapTaskInput]


def _load_centered_start_resolver() -> Callable[..., tuple[int, int, int]]:
    """在 checkpoint 源码激活后加载训练同源的居中 BOX 起点解析函数。"""
    from src.datasets.stage1_requests import centered_start_from_centroid_zyx

    return centered_start_from_centroid_zyx


def make_probability_role_producer(
    input_provider: FullMapInputProvider,
) -> RoleProducer:
    """
    构造完整图概率产物的正式编排回调。

    输入参数:
        - input_provider: FullMapInputProvider，按任务提供模型包装器、滑窗批次构造器、完整图几何与当前模型来源所需的 receptor 掩码

    输出:
        - producer: RoleProducer，固定执行 `80³` 窗口、`(40, 40, 40)` ZYX 步长和 `sigma=0.5` 的 float32 高斯融合，再原子发布 `probability.npz`、`geometry.json` 与 probability `_COMPLETE`
    """
    def produce(task: ProductionTask, paths: Stage1ArtifactPaths) -> None:
        """
        生成并发布一个 PDB 的完整图 probability role。

        输入参数:
            - task: ProductionTask, 当前 producer/split/PDB 任务身份
            - paths: Stage1ArtifactPaths, 当前任务的 probability payload 与完成标记路径
        """
        inputs = input_provider(task)
        result = infer_full_map(
            model=inputs.model,
            full_shape_zyx=inputs.full_shape_zyx,
            window_batch_builder=inputs.window_batch_builder,
            stage1_model_name=task.stage1_model_name,
            receptor_hardmask_full=inputs.receptor_hardmask_full,
            window_batch_size=inputs.window_batch_size,
            window_shape_zyx=(80, 80, 80),
            stride_zyx=(40, 40, 40),
            gaussian_sigma=0.5,
        )
        publish_full_map(
            paths=paths,
            result=result,
            origin_xyz=inputs.origin_xyz,
            voxel_size_xyz=inputs.voxel_size_xyz,
            window_shape_zyx=(80, 80, 80),
            stride_zyx=(40, 40, 40),
            gaussian_sigma=0.5,
        )

    return produce


@dataclass(frozen=True)
class ComponentRuntimeContract:
    """
    表示从当前模型来源的 `thresholds.json` 冷读得到的组件运行参数。

    属性:
        - denominator: int，整数阈值分母
        - threshold_grid_indices_descending: tuple[int, ...]，七个 alpha 所对应的整数。阈值下标 `j` 去重后降序排列；若有 k 个下标碰撞，实际森林自然只有 `7-k` 个阈值层
        - f1_threshold_grid_index: int，`alpha=1` 对应的整数阈值下标 `j`
        - min_voxels: int，候选组件的最小体素数
        - max_voxels: int，校准阶段冻结的候选组件最大体素数
    """
    denominator: int
    threshold_grid_indices_descending: tuple[int, ...]
    f1_threshold_grid_index: int
    alpha_threshold_grid_indices: tuple[tuple[float, int], ...]
    min_voxels: int
    max_voxels: int

    def threshold_grid_index_for_alpha(self, alpha: float) -> int:
        """返回 calibration 中与给定 alpha 对应的唯一阈值网格编号。"""

        matches = [
            grid_index
            for value, grid_index in self.alpha_threshold_grid_indices
            if np.isclose(value, float(alpha), rtol=0.0, atol=1e-12)
        ]
        if len(matches) != 1:
            raise ValueError(f"alpha={alpha!r} 不在冻结 alpha_values 中")
        return int(matches[0])


def load_component_runtime_contract(
    paths: Stage1ArtifactPaths,
) -> ComponentRuntimeContract:
    """
    严格读取当前模型来源的阈值表，并把重复整数下标折叠为实际森林层。

    输入参数:
        - paths: Stage1ArtifactPaths，任一当前模型来源的路径对象；函数只读取同一模型来源的 calibration 根目录，不读取当前 PDB 目录

    输出:
        - contract: ComponentRuntimeContract，供该模型来源全部数据划分共用的冻结参数

    异常:
        - calibration `_COMPLETE` 缺失、字段缺失、模型来源不一致、阈值数组未对齐、`t_alpha != j/denominator` 或 connectivity 不是 26 时抛出异常
    """
    if not paths.calibration_complete_path.is_file():
        raise RuntimeError(f"calibration 阈值尚未完整发布: {paths.calibration_complete_path}")
    with paths.thresholds_json.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    required = {
        "stage1_model_name",
        "denominator",
        "alpha_values",
        "alpha_threshold_grid_index",
        "t_alpha",
        "t_F1",
        "min_voxels",
        "max_voxels",
        "connectivity",
    }
    missing = sorted(required.difference(payload))
    if missing:
        raise KeyError(f"thresholds.json 缺少字段: {missing}")
    if str(payload["stage1_model_name"]) != paths.stage1_model_name:
        raise ValueError("thresholds.json 的 stage1_model_name 与产物路径不一致")
    # int, 固定阈值整数分母，正式值为 32768
    denominator = int(payload["denominator"])
    # np.ndarray[float64]，(N_alpha,)，校准阶段冻结的 alpha 顺序。
    alpha_values = np.asarray(payload["alpha_values"], dtype=np.float64)
    # np.ndarray[int64]，(N_alpha,)，每个 alpha 对应的阈值网格下标 j。
    grid_indices = np.asarray(payload["alpha_threshold_grid_index"], dtype=np.int64)
    # np.ndarray[float64]，(N_alpha,)，冷读阈值；必须逐项等于 j/denominator。
    threshold_values = np.asarray(payload["t_alpha"], dtype=np.float64)

    if denominator <= 0 or alpha_values.shape != grid_indices.shape or grid_indices.shape != threshold_values.shape:
        raise ValueError("thresholds.json 的 alpha/j/t_alpha 长度或 denominator 不合法")
    if alpha_values.ndim != 1 or alpha_values.size == 0:
        raise ValueError("thresholds.json 的 alpha_values 必须是一维非空序列")
    if bool(np.any(grid_indices < 0)) or bool(np.any(grid_indices > denominator)):
        raise ValueError("alpha_threshold_grid_index 越过 [0,denominator]")
    expected_values = grid_indices.astype(np.float64) / float(denominator)
    if not bool(np.allclose(threshold_values, expected_values, rtol=0.0, atol=1e-7)):
        raise ValueError("t_alpha 必须逐项等于 j/denominator")
    alpha_one_rows = np.flatnonzero(
        np.isclose(alpha_values, 1.0, rtol=0.0, atol=1e-12)
    )
    if alpha_one_rows.size != 1:
        raise ValueError("thresholds.json 必须恰好包含一个 alpha=1")

    f1_grid_index = int(grid_indices[int(alpha_one_rows[0])])
    if not np.isclose(
        float(payload["t_F1"]),
        float(f1_grid_index) / float(denominator),
        rtol=0.0,
        atol=1e-7,
    ):
        raise ValueError("t_F1 与 alpha=1 的 j 不一致")
    if int(payload["connectivity"]) != 26:
        raise ValueError("Stage1 components 只允许 26-connectivity")
    min_voxels = int(payload["min_voxels"])
    max_voxels = int(payload["max_voxels"])
    if min_voxels <= 0 or max_voxels < min_voxels:
        raise ValueError("thresholds.json 的 min/max_voxels 不合法")
    return ComponentRuntimeContract(
        denominator=denominator,
        threshold_grid_indices_descending=tuple(          # 去重
            sorted({int(value) for value in grid_indices}, reverse=True)
        ),
        f1_threshold_grid_index=f1_grid_index,
        alpha_threshold_grid_indices=tuple(
            (float(alpha), int(grid_index))
            for alpha, grid_index in zip(alpha_values, grid_indices, strict=True)
        ),
        min_voxels=min_voxels,
        max_voxels=max_voxels,
    )


def make_component_role_producer(
    occurrence_voxel_provider: OccurrenceVoxelProvider,
    clg_config: CLGEnumerationConfig,
    f1_eligible_limit: int = 200,
    continue_on_blob_exceed: bool = False,
) -> RoleProducer:
    """
    构造可直接交给 `Stage1ProductionRunner` 的正式 components producer。

    输入参数:
        - occurrence_voxel_provider: OccurrenceVoxelProvider，按任务与完整图尺寸返回 `occurrence_id -> 完整图 C-order 线性 voxel 索引`
        - clg_config: CLGEnumerationConfig，正式 depth1 配置的拆分事件上限、合并事件上限和单个 CLG 节点数上限分别为 `(1, 1, 32)`
        - f1_eligible_limit: int，`_BLOB_EXCEED` 上限，正式值为 200

    输出:
        - producer: RoleProducer，读取正式 probability/thresholds，则构造森林，检查上限，枚举组件谱系组及 occurrence 交集，最后原子发布 components 产物(用 publish_component_artifacts )
    """
    if int(f1_eligible_limit) <= 0:
        raise ValueError("f1_eligible_limit 必须为正")

    def produce(task: ProductionTask, paths: Stage1ArtifactPaths) -> None:
        """
        从完整图概率构造并发布 forest、CLG 与 occurrence overlap。

        输入参数:
            - task: ProductionTask, 当前 producer/split/PDB 任务身份
            - paths: Stage1ArtifactPaths, 当前任务的 probability、components 与 calibration 路径
        """
        if not is_role_complete(paths, "probability"):
            raise RuntimeError(f"components 前置 probability 尚未完成: {task}")

        # np.ndarray[float32], (D,H,W), 已发布并完成 producer 后处理的完整图概率
        probability = np.asarray(
            load_npz_strict(paths.probability_npz)["probability_map"],
            dtype=np.float32,
        )
        # ComponentRuntimeContract, 当前 producer calibration 冻结的实际阈值层与体积界限
        contract = load_component_runtime_contract(paths)
        # ComponentForest + summary, 从完整图概率构造的只读多阈值谱系
        forest, summary = build_component_forest(
            probability_map=probability,
            threshold_grid_indices=contract.threshold_grid_indices_descending,
            denominator=contract.denominator,
            min_voxels=contract.min_voxels,
            max_voxels=contract.max_voxels,
            resolve_box_start=_load_centered_start_resolver(),
        )
        # int, t_F1 层满足体积与 bbox 约束的 component 数
        n_f1_eligible = count_f1_eligible(
            forest, contract.f1_threshold_grid_index
        )
        if n_f1_eligible > int(f1_eligible_limit):
            if not continue_on_blob_exceed:
                raise BlobExceeded(n_f1_eligible, int(f1_eligible_limit))
            mark_blob_exceed(paths, n_f1_eligible, int(f1_eligible_limit))
        # CLGEnumerationResult, 当前 depth 配置下成功 CLG 与 cap 统计
        clg_result = enumerate_clgs(
            forest=forest,
            f1_threshold_grid_index=contract.f1_threshold_grid_index,
            config=clg_config,
        )
        # Mapping[int,np.ndarray], occurrence_id -> 完整图 C-order linear voxel indices
        occurrences = occurrence_voxel_provider(
            task, tuple(int(value) for value in probability.shape)
        )
        # dict[str,np.ndarray], candidate 与 GT occurrence 的稀疏正交集基础事实
        overlap = build_candidate_occurrence_overlap(
            clgs=clg_result.clgs,
            occurrence_voxel_indices=occurrences,
        )
        publish_component_artifacts(
            paths=paths,
            forest=forest,
            clg_result=clg_result,
            overlap_arrays=overlap,
            forest_summary=summary,
        )

    return produce


def _load_centered_context(
    task: ProductionTask,
    paths: Stage1ArtifactPaths,
    wrapper_provider: CenteredWrapperProvider,
    batch_builder_provider: CenteredBatchBuilderProvider,
) -> tuple[
    ComponentForest,
    CenteredGeometry,
    ComponentRuntimeContract,
    object,
    Callable[[Sequence[CenteredRequest]], Mapping[str, object]],
]:
    """读取 F1/CLG centered producer 共用的 forest、几何和运行时依赖。

    输出 tuple 依次为：
        - `forest`: ComponentForest，从 components/forest.npz 恢复的完整图 component 谱系。
        - `geometry`: CenteredGeometry，完整图 ZYX shape、世界 XYZ 原点、voxel size 和固定 80³ BOX shape。
        - `contract`: ComponentRuntimeContract，calibration 冻结的阈值网格和体积范围。
        - `wrapper`: object，当前 task 对应、已 strict 恢复并处于 eval 的完整 Stage1 wrapper。
        - `batch_builder`: Callable，接收同一 producer/split/PDB/role 的有序 CenteredRequest 切片，返回训练同源 Collator 生成的目标设备 batch。
    """

    if not is_role_complete(paths, "components"):
        raise RuntimeError(f"centered 前置 components 尚未完成: {task}")
    forest = ComponentForest.from_arrays(load_npz_strict(paths.forest_npz))
    with paths.probability_geometry_json.open("r", encoding="utf-8") as handle:
        geometry_payload = json.load(handle)
    geometry = CenteredGeometry(
        full_shape_zyx=tuple(int(value) for value in geometry_payload["full_shape_zyx"]),
        origin_xyz=np.asarray(geometry_payload["origin_xyz"], dtype=np.float32),
        voxel_size_xyz=np.asarray(geometry_payload["voxel_size_xyz"], dtype=np.float32),
    )
    contract = load_component_runtime_contract(paths)
    return (
        forest,
        geometry,
        contract,
        wrapper_provider(task),
        batch_builder_provider(task),
    )


def make_f1_clg_centered_role_producers(
    wrapper_provider: CenteredWrapperProvider,
    batch_builder_provider: CenteredBatchBuilderProvider,
    centered_batch_size: int = 10,
) -> dict[str, RoleProducer]:
    """
    构造 F1/CLG 两个居中 role 的正式 runner callbacks。

    输入参数:
        - wrapper_provider: Callable，按 ProductionTask 返回已 strict 恢复且 eval 的同一
          producer 完整 wrapper；调用方可在 worker 内缓存，不能换用裸 backbone
        - batch_builder_provider: Callable，正式由 `Stage1RuntimeAssembly.centered_batch_builder` 提供；按 task 返回接收同一 producer/split/PDB/role 有序 `CenteredRequest` 序列的 Stage1Dataset/Collator batch builder。
        - centered_batch_size: int, 单次完整 wrapper forward 的 BOX 数；尾批允许更短，CLI 正式默认 10，拆分后 entry 顺序不变。

    输出:
        - role_producers: dict[str, RoleProducer]，含 `F1_centered` 与
          `CLG_centered` 两个键，可直接并入 `Stage1ProductionRunner.role_producers`
    """
    def produce_f1(task: ProductionTask, paths: Stage1ArtifactPaths) -> None:
        """
        为 `t_F1` 层 eligible component 生成并发布 F1 居中条目。

        输入参数:
            - task: ProductionTask, 当前 producer/split/PDB 任务身份
            - paths: Stage1ArtifactPaths, 当前任务的 forest 输入与 `F1_centered` 输出路径
        """
        forest, geometry, contract, wrapper, batch_builder = _load_centered_context(
            task,
            paths,
            wrapper_provider,
            batch_builder_provider,
        )
        entries = produce_f1_centered_entries(
            nodes=forest.nodes,
            f1_threshold_grid_index=contract.f1_threshold_grid_index,
            geometry=geometry,
            stage1_model_name=task.stage1_model_name,
            split=task.split,
            pdb_id=task.pdb_id,
            resolve_box_start=_load_centered_start_resolver(),
            wrapper=wrapper,
            batch_builder=batch_builder,
            centered_batch_size=centered_batch_size,
        )
        publish_centered_entries(
            paths,
            "F1_centered",
            entries,
        )

    def produce_clg(task: ProductionTask, paths: Stage1ArtifactPaths) -> None:
        """
        为已枚举 CLG 生成并发布其 candidate 居中条目。

        输入参数:
            - task: ProductionTask, 当前 producer/split/PDB 任务身份
            - paths: Stage1ArtifactPaths, 当前任务的 forest/CLG 输入与 `CLG_centered` 输出路径
        """
        forest, geometry, _contract, wrapper, batch_builder = _load_centered_context(
            task,
            paths,
            wrapper_provider,
            batch_builder_provider,
        )
        clgs = clgs_from_arrays(load_npz_strict(paths.clg_npz), forest)
        entries = produce_clg_centered_entries(
            clgs=clgs,
            geometry=geometry,
            stage1_model_name=task.stage1_model_name,
            split=task.split,
            pdb_id=task.pdb_id,
            resolve_box_start=_load_centered_start_resolver(),
            wrapper=wrapper,
            batch_builder=batch_builder,
            centered_batch_size=centered_batch_size,
        )
        publish_centered_entries(
            paths,
            "CLG_centered",
            entries,
        )

    return {
        "F1_centered": produce_f1,
        "CLG_centered": produce_clg,
    }


def make_falpha_centered_role_producer(
    wrapper_provider: CenteredWrapperProvider,
    batch_builder_provider: CenteredBatchBuilderProvider,
    centered_role: str,
    alpha: float,
    centered_batch_size: int = 10,
) -> RoleProducer:
    """构造只补充一个冻结 F_alpha 阈值层 centered 产物的回调。"""

    def produce(task: ProductionTask, paths: Stage1ArtifactPaths) -> None:
        forest, geometry, contract, wrapper, batch_builder = _load_centered_context(
            task,
            paths,
            wrapper_provider,
            batch_builder_provider,
        )
        entries = produce_threshold_centered_entries(
            nodes=forest.nodes,
            threshold_grid_index=contract.threshold_grid_index_for_alpha(alpha),
            centered_role=centered_role,
            geometry=geometry,
            stage1_model_name=task.stage1_model_name,
            split=task.split,
            pdb_id=task.pdb_id,
            resolve_box_start=_load_centered_start_resolver(),
            wrapper=wrapper,
            batch_builder=batch_builder,
            centered_batch_size=centered_batch_size,
        )
        publish_centered_entries(paths, centered_role, entries)

    return produce


def make_li_centered_role_producer(
    probability_output_root: str | Path,
    wrapper_provider: CenteredWrapperProvider,
    batch_builder_provider: CenteredBatchBuilderProvider,
    denominator: int,
    min_voxels: int,
    max_voxels: int,
    eligible_limit: int,
    continue_on_blob_exceed: bool,
    centered_batch_size: int = 10,
) -> RoleProducer:
    """构造读取既有概率图、仅向独立根目录发布 Li-centered 的回调。"""

    source_root = Path(probability_output_root)

    def produce(task: ProductionTask, paths: Stage1ArtifactPaths) -> None:
        source_paths = Stage1ArtifactPaths(
            source_root,
            task.stage1_model_name,
            task.split,
            task.pdb_id,
        )
        if not is_role_complete(source_paths, "probability"):
            raise RuntimeError(f"Li-centered 前置 probability 尚未完成: {task}")
        probability = np.asarray(
            load_npz_strict(source_paths.probability_npz)["probability_map"],
            dtype=np.float32,
        )
        with source_paths.probability_geometry_json.open("r", encoding="utf-8") as handle:
            geometry_payload = json.load(handle)
        geometry = CenteredGeometry(
            full_shape_zyx=tuple(int(value) for value in geometry_payload["full_shape_zyx"]),
            origin_xyz=np.asarray(geometry_payload["origin_xyz"], dtype=np.float32),
            voxel_size_xyz=np.asarray(
                geometry_payload["voxel_size_xyz"], dtype=np.float32
            ),
        )
        nodes, raw_threshold, grid_index, applied_threshold = build_li_nodes(
            probability_map=probability,
            denominator=denominator,
            min_voxels=min_voxels,
            max_voxels=max_voxels,
            resolve_box_start=_load_centered_start_resolver(),
        )
        if len(nodes) > int(eligible_limit):
            mark_blob_exceed(paths, len(nodes), int(eligible_limit))
            if not continue_on_blob_exceed:
                raise BlobExceeded(len(nodes), int(eligible_limit))
        entries = produce_threshold_centered_entries(
            nodes=nodes,
            threshold_grid_index=grid_index,
            centered_role="Li_centered",
            geometry=geometry,
            stage1_model_name=task.stage1_model_name,
            split=task.split,
            pdb_id=task.pdb_id,
            resolve_box_start=_load_centered_start_resolver(),
            wrapper=wrapper_provider(task),
            batch_builder=batch_builder_provider(task),
            centered_batch_size=centered_batch_size,
        )
        for local_id, (entry, node) in enumerate(zip(entries, nodes, strict=True)):
            entry["source_tree_id"] = 0
            entry["source_node_id"] = local_id
            entry["source_probability_mean"] = np.float32(node.probability_mean)
        publish_li_centered_entries(
            paths=paths,
            entries=entries,
            raw_threshold=raw_threshold,
            grid_index=grid_index,
            applied_threshold=applied_threshold,
            denominator=denominator,
        )

    return produce


# Selected_Refined_Centered: 暂时不看
def make_selected_refined_role_producer(
    wrapper_provider: CenteredWrapperProvider,
    batch_builder_provider: CenteredBatchBuilderProvider,
    centered_batch_size: int = 10,
    selection_path_provider: SelectionPathProvider | None = None,
) -> RoleProducer:
    """
    构造 Selector selection→forest nodes→居中重跑→发布的正式 role producer。

    输入参数:
        - wrapper_provider: Callable, 按 task 返回完整、strict 恢复且处于 eval 的 Stage1 wrapper
        - batch_builder_provider: Callable, 正式由 `Stage1RuntimeAssembly.centered_batch_builder` 提供；按 task 返回接收有序 `CenteredRequest` 序列的 Stage1Dataset/Collator batch builder。
        - centered_batch_size: int, 单次完整 wrapper forward 的 BOX 数；尾批允许更短，CLI 正式默认 10，拆分后 Selected 来源顺序不变。
        - selection_path_provider: Callable | None, 可选路径解析器; None 读取当前 PDB 正式目录中的 `selector/selection.npz`

    输出:
        - producer: RoleProducer, 可注册为 `Selected_Refined_Centered`；只消费 selection 恢复的原森林节点，并在精修产物中保留来源 tree/node/threshold 身份
    """

    def produce(task: ProductionTask, paths: Stage1ArtifactPaths) -> None:
        """
        恢复 Selector 选中节点并发布其重新运行得到的居中 V/P/A 条目。

        输入参数:
            - task: ProductionTask, 当前 producer/split/PDB 任务身份
            - paths: Stage1ArtifactPaths, 当前任务的 selection、forest、geometry 与 Selected 输出路径
        """
        if not is_role_complete(paths, "components"):
            raise RuntimeError(f"Selected 前置 components 尚未完成: {task}")
        if selection_path_provider is None:
            selection_path = paths.pdb_root / "selector" / "selection.npz"
        else:
            selection_path = Path(selection_path_provider(task, paths))
        if not selection_path.is_file():
            raise FileNotFoundError(f"Selector selection.npz 不存在: {selection_path}")

        from src.selector.inference import load_selected_nodes_for_pdb

        selected_nodes = load_selected_nodes_for_pdb(
            selection_path=selection_path,
            forest_path=paths.forest_npz,
            clg_path=paths.clg_npz,
        )
        with paths.probability_geometry_json.open("r", encoding="utf-8") as handle:
            geometry_payload = json.load(handle)
        geometry = CenteredGeometry(
            full_shape_zyx=tuple(
                int(value) for value in geometry_payload["full_shape_zyx"]
            ),
            origin_xyz=np.asarray(geometry_payload["origin_xyz"], dtype=np.float32),
            voxel_size_xyz=np.asarray(
                geometry_payload["voxel_size_xyz"], dtype=np.float32
            ),
        )
        wrapper = wrapper_provider(task)
        batch_builder = batch_builder_provider(task)
        entries = produce_selected_refined_entries(
            selected_nodes=selected_nodes,
            geometry=geometry,
            stage1_model_name=task.stage1_model_name,
            split=task.split,
            pdb_id=task.pdb_id,
            resolve_box_start=_load_centered_start_resolver(),
            wrapper=wrapper,
            batch_builder=batch_builder,
            centered_batch_size=centered_batch_size,
        )
        publish_centered_entries(
            paths,
            "Selected_Refined_Centered",
            entries,
        )

    return produce


class Stage1ProductionRunner:
    """
    顺序补齐一个 PDB 缺失的产物角色，并以 `_RUNNING` 防止多进程重复生产。

    输入参数:
        - output_root: str, `stage1_outputs` 根目录
        - role_producers: Mapping[str, RoleProducer], 每个回调必须原子发布自身数据，
          完成基本校验后最后写对应产物角色的 `_COMPLETE`
        - owner_token: str, 当前进程唯一身份；建议包含调度任务、主机和进程身份
    """

    def __init__(
        self,
        output_root: str,
        role_producers: Mapping[str, RoleProducer],
        owner_token: str,
        continue_on_blob_exceed: bool = False,
    ) -> None:
        """
        校验 role producer 集合并保存当前 worker 的生产上下文。

        输入参数:
            - output_root: str, `stage1_outputs` 根目录
            - role_producers: Mapping[str,RoleProducer], role 名到原子发布 callback 的映射
            - owner_token: str, 当前 worker 唯一身份
        """
        unknown = set(role_producers) - set(OUTPUT_ROLES)
        if unknown:
            raise ValueError(f"role_producers 含未知 role: {sorted(unknown)}")
        self.output_root = output_root
        self.role_producers = dict(role_producers)
        self.owner_token = str(owner_token)
        self.continue_on_blob_exceed = bool(continue_on_blob_exceed)

    @classmethod
    def for_current_process(
        cls,
        output_root: str,
        role_producers: Mapping[str, RoleProducer],
        continue_on_blob_exceed: bool = False,
    ) -> "Stage1ProductionRunner":
        """
        以 hostname/pid 组成当前本地 worker 的 owner_token。

        输入参数:
            - output_root: str, `stage1_outputs` 根目录
            - role_producers: Mapping[str,RoleProducer], role 名到原子发布 callback 的映射

        输出:
            - runner: Stage1ProductionRunner, 绑定当前 hostname/pid 身份的生产 runner
        """
        return cls(
            output_root=output_root,
            role_producers=role_producers,
            owner_token=f"{socket.gethostname()}:{os.getpid()}",
            continue_on_blob_exceed=continue_on_blob_exceed,
        )

    def run_task(
        self,
        task: ProductionTask,
        requested_roles: Sequence[str],
    ) -> RunRecord:
        """
        获取一个 PDB 的互斥租约，并在租约上下文内顺序补齐请求的缺失角色。

        输入参数:
            - task: ProductionTask, 当前 producer/split/PDB 身份
            - requested_roles: Sequence[str], 已按依赖先后排列的目标产物角色

        输出:
            - record: RunRecord, 其他进程持有租约时立即返回 `skipped_running`；已有 `_BLOB_EXCEED` 返回 `blob_exceed`，全部角色已有完成标记时返回
              `skipped_complete`，本次进入生产流程则返回 `completed` 或 `blob_exceed`
        """
        # tuple[str, ...]，当前入口已按依赖先后排列的目标产物角色。
        roles = tuple(str(role) for role in requested_roles)
        if any(role not in self.role_producers for role in roles):
            missing = [role for role in roles if role not in self.role_producers]
            raise KeyError(f"requested role 没有 producer callback: {missing}")
        # Stage1ArtifactPaths，当前模型来源/数据划分/PDB 的全部数据和状态路径。
        paths = Stage1ArtifactPaths(
            output_root=self.output_root,
            stage1_model_name=task.stage1_model_name,
            split=task.split,
            pdb_id=task.pdb_id,
        )
        if paths.blob_exceed_path.is_file() and not self.continue_on_blob_exceed:
            return RunRecord(task, "blob_exceed", ())
        if all(is_role_complete(paths, role) for role in roles):
            return RunRecord(task, "skipped_complete", ())

        # PdbRunningLease | None，PDB 级互斥租约；None 表示正由其他进程处理。
        lease = PdbRunningLease.acquire(paths, self.owner_token)
        if lease is None:
            return RunRecord(task, "skipped_running", ())
        # list[str]，只记录当前进程本次新发布完成的产物角色。
        completed: list[str] = []
        with lease:
            if paths.blob_exceed_path.is_file() and not self.continue_on_blob_exceed:
                return RunRecord(task, "blob_exceed", ())
            for role in roles:
                if is_role_complete(paths, role):
                    continue
                try:
                    self.role_producers[role](task, paths)
                except BlobExceeded as error:
                    mark_blob_exceed(paths, error.n_f1_eligible, error.limit)
                    return RunRecord(task, "blob_exceed", tuple(completed))
                if not is_role_complete(paths, role):
                    raise RuntimeError(
                        f"role producer 返回后未发布 {role} 的 `_COMPLETE`: {task}"
                    )
                completed.append(role)
        return RunRecord(task, "completed", tuple(completed))

    def run_calibration_probability(
        self,
        calibration_tasks: Sequence[ProductionTask],
    ) -> tuple[RunRecord, ...]:
        """
        阶段一只为 calibration 100 生成完整图 probability。

        输入参数:
            - calibration_tasks: Sequence[ProductionTask], 当前 worker 被分配的 calibration PDB

        输出:
            - records: tuple[RunRecord,...], 与输入固定顺序一致的续跑结果
        """
        return tuple(self.run_task(task, ("probability",)) for task in calibration_tasks)

    def run_validation_calibration_train(
        self,
        validation_tasks: Sequence[ProductionTask],
        calibration_tasks: Sequence[ProductionTask],
        train_tasks: Sequence[ProductionTask],
    ) -> tuple[RunRecord, ...]:
        """
        阈值冻结后先消费本 worker 的 validation 与 calibration 份额，再消费 train。

        输入参数:
            - validation_tasks: Sequence[ProductionTask], 从 probability 开始的 validation 份额
            - calibration_tasks: Sequence[ProductionTask], 已有 probability、从 components 开始的份额
            - train_tasks: Sequence[ProductionTask], 从 probability 开始的 train 份额

        输出:
            - records: tuple[RunRecord,...], 可用于吞吐、跳过与失败统计
        """
        return (
            *self.run_val_produce_prob_f1_clg(validation_tasks),
            *self.run_cal_produce_f1_clg(calibration_tasks),
            *self.run_train_produce_prob_f1_clg(train_tasks),
        )

    def run_cal_produce_f1_clg(
        self,
        tasks: Sequence[ProductionTask],
    ) -> tuple[RunRecord, ...]:
        """
        执行阈值冻结后的 `cal-produce-F1-CLG` 明确入口。

        calibration 的 probability 已由阶段一发布，因此本入口只依次补齐 components、F1_centered 与 CLG_centered。

        输入参数:
            - tasks: Sequence[ProductionTask], 已完成 probability 的 calibration 任务

        输出:
            - records: tuple[RunRecord,...], 与输入固定顺序一致的续跑结果
        """
        return self._run_calibration_f1(tasks, include_clg=True)

    def run_cal_produce_f1(
        self,
        tasks: Sequence[ProductionTask],
    ) -> tuple[RunRecord, ...]:
        """为已有 calibration probability 只补齐 components 与 F1-centered。"""
        return self._run_calibration_f1(tasks, include_clg=False)

    def _run_calibration_f1(
        self,
        tasks: Sequence[ProductionTask],
        include_clg: bool,
    ) -> tuple[RunRecord, ...]:
        """执行 calibration 共用前置检查，并按需包含 CLG-centered。"""
        task_tuple = tuple(tasks)
        self._require_calibration_frozen(task_tuple)
        for task in task_tuple:
            paths = Stage1ArtifactPaths(
                output_root=self.output_root,
                stage1_model_name=task.stage1_model_name,
                split=task.split,
                pdb_id=task.pdb_id,
            )
            if not is_role_complete(paths, "probability"):
                raise RuntimeError(
                    f"cal-produce-F1 要求既有 probability `_COMPLETE`: {task}"
                )
        roles = ("components", "F1_centered")
        if include_clg:
            roles = (*roles, "CLG_centered")
        return tuple(self.run_task(task, roles) for task in task_tuple)

    def run_val_produce_prob_f1(
        self,
        tasks: Sequence[ProductionTask],
    ) -> tuple[RunRecord, ...]:
        """为 validation 依次补齐 probability、components 与 F1-centered。"""
        return self._run_probability_f1(tasks, include_clg=False)

    def run_val_produce_prob_f1_clg(
        self,
        tasks: Sequence[ProductionTask],
    ) -> tuple[RunRecord, ...]:
        """
        执行 `val-produce-Prob-F1-CLG`，从 probability 连续补齐四个 role。

        输入参数:
            - tasks: Sequence[ProductionTask], validation 任务序列

        输出:
            - records: tuple[RunRecord,...], 与输入固定顺序一致的续跑结果
        """
        return self._run_probability_f1(tasks, include_clg=True)

    def run_train_produce_prob_f1(
        self,
        tasks: Sequence[ProductionTask],
    ) -> tuple[RunRecord, ...]:
        """为 train 依次补齐 probability、components 与 F1-centered。"""
        return self._run_probability_f1(tasks, include_clg=False)

    def run_train_produce_prob_f1_clg(
        self,
        tasks: Sequence[ProductionTask],
    ) -> tuple[RunRecord, ...]:
        """
        执行 `train-produce-Prob-F1-CLG`，从 probability 连续补齐四个 role。

        输入参数:
            - tasks: Sequence[ProductionTask], train 任务序列

        输出:
            - records: tuple[RunRecord,...], 与输入固定顺序一致的续跑结果
        """
        return self._run_probability_f1(tasks, include_clg=True)

    def _run_probability_f1(
        self,
        tasks: Sequence[ProductionTask],
        include_clg: bool,
    ) -> tuple[RunRecord, ...]:
        """
        为 validation/train 共用正式 role 顺序与阈值前置检查。

        输入参数:
            - tasks: Sequence[ProductionTask], 同一阶段待补齐四个 role 的任务序列

        输出:
            - records: tuple[RunRecord,...], 与输入固定顺序一致的续跑结果
        """
        task_tuple = tuple(tasks)
        self._require_calibration_frozen(task_tuple)
        roles = ("probability", "components", "F1_centered")
        if include_clg:
            roles = (*roles, "CLG_centered")
        return tuple(self.run_task(task, roles) for task in task_tuple)

    def _require_calibration_frozen(
        self,
        tasks: Sequence[ProductionTask],
    ) -> None:
        """
        要求每个待消费 producer 的 calibration `_COMPLETE` 已正式存在。

        输入参数:
            - tasks: Sequence[ProductionTask], 待检查 producer 来源的任务序列
        """
        for stage1_model_name in sorted({task.stage1_model_name for task in tasks}):
            marker = Stage1ArtifactPaths(
                output_root=self.output_root,
                stage1_model_name=stage1_model_name,
                split="calibration",
                pdb_id="_threshold_precondition",
            ).calibration_complete_path
            if not marker.is_file():
                raise RuntimeError(
                    f"{stage1_model_name} 尚未冻结 calibration 阈值: {marker}"
                )

    def run_selected(
        self,
        tasks: Sequence[ProductionTask],
    ) -> tuple[RunRecord, ...]:
        """
        在正式 selector variant 冻结后独立补齐 Selected role。

        输入参数:
            - tasks: Sequence[ProductionTask], 已具备 Selector selection 的任务序列

        输出:
            - records: tuple[RunRecord,...], 与输入固定顺序一致的 Selected role 续跑结果
        """
        return tuple(
            self.run_task(task, ("Selected_Refined_Centered",)) for task in tasks
        )
