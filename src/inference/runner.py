"""无需数据库的 per-PDB/role 可续跑 Stage1 生产编排。"""

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
from src.datasets.stage1_requests import centered_start_from_centroid_zyx

from .centered import (
    CenteredGeometry,
    CenteredRequest,
    make_model_centered_callback,
    produce_clg_centered_entries,
    produce_f1_centered_entries,
    produce_selected_refined_entries,
    publish_centered_entries,
    resolve_component_centered_start,
)
from .full_map import infer_full_map, publish_full_map


@dataclass(frozen=True)
class ProductionTask:
    """
    表示可重分配、可重入的一个 producer/split/PDB 任务。

    输入参数:
        - stage1_model_name: str, producer 正式名
        - split: str, validation、calibration、train 等划分
        - pdb_id: str, 当前 PDB 身份
    """

    stage1_model_name: str
    split: str
    pdb_id: str


@dataclass(frozen=True)
class RunRecord:
    """
    保存 runner 对一个 PDB 的无副作用状态摘要。

    输入参数:
        - task: ProductionTask, 当前任务身份
        - status: str, `completed/skipped_complete/skipped_running/blob_exceed`
        - completed_roles: tuple[str,...], 本次 worker 新完成的 role
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
        self.n_f1_eligible = int(n_f1_eligible)
        self.limit = int(limit)
        super().__init__(f"N_F1_eligible={self.n_f1_eligible} > limit={self.limit}")


RoleProducer = Callable[[ProductionTask, Stage1ArtifactPaths], None]
OccurrenceVoxelProvider = Callable[
    [ProductionTask, tuple[int, int, int]], Mapping[int, np.ndarray]
]
CenteredWrapperProvider = Callable[[ProductionTask], object]
CenteredBatchBuilderProvider = Callable[
    [ProductionTask], Callable[[CenteredRequest], Mapping[str, object]]
]
SelectionPathProvider = Callable[[ProductionTask, Stage1ArtifactPaths], Path]


@dataclass(frozen=True)
class FullMapTaskInput:
    """
    一个 PDB 的完整图 probability role 所需运行时输入。

    属性:
        - model: object，已恢复、eval 且实现 `forward_voxel_probability` 的完整 wrapper
        - full_shape_zyx: tuple[int,int,int]，完整图 shape
        - window_batch_builder: Callable，按一批 ZYX 起点构造统一 Dataset/Collator batch
        - receptor_hardmask_full: np.ndarray | None，Find 必填，unet_c1 为 None
        - window_batch_size: int，单次 voxel-only forward 窗口数
        - origin_xyz: tuple[float,float,float]，完整图物理原点
        - voxel_size_xyz: tuple[float,float,float]，XYZ Å/voxel
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


FullMapInputProvider = Callable[[ProductionTask], FullMapTaskInput]


def make_probability_role_producer(
    input_provider: FullMapInputProvider,
) -> RoleProducer:
    """
    构造完整图 probability 的正式 runner callback。

    输入参数:
        - input_provider: Callable，按 task 提供 wrapper、滑窗 batch builder、完整图几何
          与 producer-specific hardmask

    输出:
        - producer: RoleProducer，固定执行 80³/stride40/sigma0.5 的 float32 融合，
          再原子发布 probability/geometry 与 `_COMPLETE`
    """

    def produce(task: ProductionTask, paths: Stage1ArtifactPaths) -> None:
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
    表示从 producer `thresholds.json` 冷读得到的组件运行参数。

    属性:
        - denominator: int，整数阈值分母
        - threshold_grid_indices_descending: tuple[int,...]，七个 alpha 的实际 j 去重后降序；
          若有 k 个碰撞，自然只有 7-k 个 runtime 层
        - f1_threshold_grid_index: int，alpha=1 对应 j
        - min_voxels: int，candidate 最小体素数
        - max_voxels: int，冻结的 candidate 最大体素数
    """

    denominator: int
    threshold_grid_indices_descending: tuple[int, ...]
    f1_threshold_grid_index: int
    min_voxels: int
    max_voxels: int


def load_component_runtime_contract(
    paths: Stage1ArtifactPaths,
) -> ComponentRuntimeContract:
    """
    严格读取当前 producer 的阈值表，并把重复 j 自然折叠为实际 forest 层。

    输入参数:
        - paths: Stage1ArtifactPaths，任一当前 producer 路径对象；只使用 producer
          calibration 根目录

    输出:
        - contract: ComponentRuntimeContract，供所有 split 共用的冻结参数
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
    denominator = int(payload["denominator"])
    alpha_values = np.asarray(payload["alpha_values"], dtype=np.float64)
    grid_indices = np.asarray(payload["alpha_threshold_grid_index"], dtype=np.int64)
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
        threshold_grid_indices_descending=tuple(
            sorted({int(value) for value in grid_indices}, reverse=True)
        ),
        f1_threshold_grid_index=f1_grid_index,
        min_voxels=min_voxels,
        max_voxels=max_voxels,
    )


def make_component_role_producer(
    occurrence_voxel_provider: OccurrenceVoxelProvider,
    clg_config: CLGEnumerationConfig,
    f1_eligible_limit: int = 200,
) -> RoleProducer:
    """
    构造可直接交给 `Stage1ProductionRunner` 的正式 components producer。

    输入参数:
        - occurrence_voxel_provider: Callable，按 task 与完整图 shape 返回
          `occurrence_id -> 全图 C-order linear voxel index`
        - clg_config: CLGEnumerationConfig，正式 depth1 为 (1,1,32)
        - f1_eligible_limit: int，`_BLOB_EXCEED` 上限，正式值为 200

    输出:
        - producer: RoleProducer，读取正式 probability/thresholds，使用共享 centroid
          resolver 建 forest，检查上限，枚举 CLG/overlap 并原子发布 components
    """
    if int(f1_eligible_limit) <= 0:
        raise ValueError("f1_eligible_limit 必须为正")

    def produce(task: ProductionTask, paths: Stage1ArtifactPaths) -> None:
        if not is_role_complete(paths, "probability"):
            raise RuntimeError(f"components 前置 probability 尚未完成: {task}")
        probability = np.asarray(
            load_npz_strict(paths.probability_npz)["probability_map"],
            dtype=np.float32,
        )
        contract = load_component_runtime_contract(paths)
        forest, summary = build_component_forest(
            probability_map=probability,
            threshold_grid_indices=contract.threshold_grid_indices_descending,
            denominator=contract.denominator,
            min_voxels=contract.min_voxels,
            max_voxels=contract.max_voxels,
            resolve_box_start=centered_start_from_centroid_zyx,
        )
        n_f1_eligible = count_f1_eligible(
            forest, contract.f1_threshold_grid_index
        )
        if n_f1_eligible > int(f1_eligible_limit):
            raise BlobExceeded(n_f1_eligible, int(f1_eligible_limit))
        clg_result = enumerate_clgs(
            forest=forest,
            f1_threshold_grid_index=contract.f1_threshold_grid_index,
            config=clg_config,
        )
        occurrences = occurrence_voxel_provider(
            task, tuple(int(value) for value in probability.shape)
        )
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


def make_f1_clg_centered_role_producers(
    wrapper_provider: CenteredWrapperProvider,
    batch_builder_provider: CenteredBatchBuilderProvider,
    output_adapter: Callable[[object, Mapping[str, object]], Mapping[str, object]]
    | None = None,
) -> dict[str, RoleProducer]:
    """
    构造 F1/CLG 两个居中 role 的正式 runner callbacks。

    输入参数:
        - wrapper_provider: Callable，按 ProductionTask 返回已 strict 恢复且 eval 的同一
          producer 完整 wrapper；调用方可在 worker 内缓存，不能换用裸 backbone
        - batch_builder_provider: Callable，按 task 返回接收 CenteredRequest 的统一
          Dataset/Collator batch builder
        - output_adapter: Callable | None，可选测试/特殊适配器；正式为 None，使用模型
          具名 V/P/A 输出的 `adapt_stage1_centered_output`

    输出:
        - role_producers: dict，键为 `F1_centered` 与 `CLG_centered`，可直接并入
          `Stage1ProductionRunner.role_producers`
    """

    def load_context(
        task: ProductionTask,
        paths: Stage1ArtifactPaths,
    ) -> tuple[
        ComponentForest,
        CenteredGeometry,
        ComponentRuntimeContract,
        Callable[[CenteredRequest], Mapping[str, object]],
    ]:
        if not is_role_complete(paths, "components"):
            raise RuntimeError(f"centered 前置 components 尚未完成: {task}")
        forest = ComponentForest.from_arrays(load_npz_strict(paths.forest_npz))
        with paths.probability_geometry_json.open("r", encoding="utf-8") as handle:
            geometry_payload = json.load(handle)
        geometry = CenteredGeometry(
            full_shape_zyx=tuple(int(value) for value in geometry_payload["full_shape_zyx"]),
            origin_xyz=np.asarray(geometry_payload["origin_xyz"], dtype=np.float32),
            voxel_size_xyz=np.asarray(
                geometry_payload["voxel_size_xyz"], dtype=np.float32
            ),
        )
        contract = load_component_runtime_contract(paths)
        callback = make_model_centered_callback(
            wrapper=wrapper_provider(task),
            batch_builder=batch_builder_provider(task),
            output_adapter=output_adapter,
        )
        return forest, geometry, contract, callback

    def produce_f1(task: ProductionTask, paths: Stage1ArtifactPaths) -> None:
        forest, geometry, contract, callback = load_context(task, paths)
        entries = produce_f1_centered_entries(
            nodes=forest.nodes,
            f1_threshold_grid_index=contract.f1_threshold_grid_index,
            geometry=geometry,
            stage1_model_name=task.stage1_model_name,
            split=task.split,
            pdb_id=task.pdb_id,
            resolve_box_start=resolve_component_centered_start,
            full_forward=callback,
        )
        publish_centered_entries(paths, "F1_centered", entries)

    def produce_clg(task: ProductionTask, paths: Stage1ArtifactPaths) -> None:
        forest, geometry, _contract, callback = load_context(task, paths)
        clgs = clgs_from_arrays(load_npz_strict(paths.clg_npz), forest)
        entries = produce_clg_centered_entries(
            clgs=clgs,
            geometry=geometry,
            stage1_model_name=task.stage1_model_name,
            split=task.split,
            pdb_id=task.pdb_id,
            resolve_box_start=resolve_component_centered_start,
            full_forward=callback,
        )
        publish_centered_entries(paths, "CLG_centered", entries)

    return {
        "F1_centered": produce_f1,
        "CLG_centered": produce_clg,
    }


def make_selected_refined_role_producer(
    wrapper_provider: CenteredWrapperProvider,
    batch_builder_provider: CenteredBatchBuilderProvider,
    selection_path_provider: SelectionPathProvider | None = None,
    output_adapter: Callable[[object, Mapping[str, object]], Mapping[str, object]]
    | None = None,
) -> RoleProducer:
    """构造 Selector selection→forest nodes→居中重跑→发布的正式 role producer。

    参数:
        wrapper_provider: 按 task 返回完整、strict 恢复且处于 eval 的 Stage1 wrapper。
        batch_builder_provider: 按 task 返回统一 Stage1Dataset/collator 的居中 batch builder。
        selection_path_provider: 可选路径解析器。缺省读取当前 PDB 正式目录中的
            ``selector/selection.npz``；外置 Selector run 可显式注入其它固定根。
        output_adapter: 仅供测试或显式兼容使用；正式缺省读取 wrapper 具名 V/P/A 输出。

    返回:
        可直接注册为 ``Selected_Refined_Centered`` 的 RoleProducer。它只消费
        selection 中恢复出的原 forest nodes，并保留 source tree/node/threshold 身份。
    """

    def produce(task: ProductionTask, paths: Stage1ArtifactPaths) -> None:
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
        callback = make_model_centered_callback(
            wrapper=wrapper_provider(task),
            batch_builder=batch_builder_provider(task),
            output_adapter=output_adapter,
        )
        entries = produce_selected_refined_entries(
            selected_nodes=selected_nodes,
            geometry=geometry,
            stage1_model_name=task.stage1_model_name,
            split=task.split,
            pdb_id=task.pdb_id,
            resolve_box_start=resolve_component_centered_start,
            full_forward=callback,
        )
        publish_centered_entries(
            paths,
            "Selected_Refined_Centered",
            entries,
        )

    return produce


class Stage1ProductionRunner:
    """
    顺序补齐 PDB 缺失 role，并以 `_RUNNING` 防止多 worker 重复生产。

    输入参数:
        - output_root: str, `stage1_outputs` 根目录
        - role_producers: Mapping[str,RoleProducer], 每个 callback 必须原子发布其 payload，
          基本校验成功后最后写自己的 role `_COMPLETE`
        - owner_token: str, 当前 worker 唯一身份；建议包含调度任务与进程身份
    """

    def __init__(
        self,
        output_root: str,
        role_producers: Mapping[str, RoleProducer],
        owner_token: str,
    ) -> None:
        unknown = set(role_producers) - set(OUTPUT_ROLES)
        if unknown:
            raise ValueError(f"role_producers 含未知 role: {sorted(unknown)}")
        self.output_root = output_root
        self.role_producers = dict(role_producers)
        self.owner_token = str(owner_token)

    @classmethod
    def for_current_process(
        cls,
        output_root: str,
        role_producers: Mapping[str, RoleProducer],
    ) -> "Stage1ProductionRunner":
        """以 hostname/pid 组成当前本地 worker 的 owner_token。"""
        return cls(
            output_root=output_root,
            role_producers=role_producers,
            owner_token=f"{socket.gethostname()}:{os.getpid()}",
        )

    def run_task(
        self,
        task: ProductionTask,
        requested_roles: Sequence[str],
    ) -> RunRecord:
        """
        抢占一个 PDB 并在同一 `try/finally` 中顺序补齐请求的缺失 role。

        输入参数:
            - task: ProductionTask, 当前 producer/split/PDB 身份
            - requested_roles: Sequence[str], 依赖顺序排列的目标 role

        输出:
            - record: RunRecord, 其它 worker 持锁时立即 `skipped_running`；
              `_BLOB_EXCEED` 与全部完成均不重复运行
    """
        roles = tuple(str(role) for role in requested_roles)
        if any(role not in self.role_producers for role in roles):
            missing = [role for role in roles if role not in self.role_producers]
            raise KeyError(f"requested role 没有 producer callback: {missing}")
        paths = Stage1ArtifactPaths(
            output_root=self.output_root,
            stage1_model_name=task.stage1_model_name,
            split=task.split,
            pdb_id=task.pdb_id,
        )
        if paths.blob_exceed_path.is_file():
            return RunRecord(task, "blob_exceed", ())
        if all(is_role_complete(paths, role) for role in roles):
            return RunRecord(task, "skipped_complete", ())

        lease = PdbRunningLease.acquire(paths, self.owner_token)
        if lease is None:
            return RunRecord(task, "skipped_running", ())
        completed: list[str] = []
        with lease:
            if paths.blob_exceed_path.is_file():
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

        calibration 的 probability 已由阶段一发布，因此本入口只依次补齐
        components、F1_centered 与 CLG_centered。
        """
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
                    f"cal-produce-F1-CLG 要求既有 probability `_COMPLETE`: {task}"
                )
        roles = ("components", "F1_centered", "CLG_centered")
        return tuple(self.run_task(task, roles) for task in task_tuple)

    def run_val_produce_prob_f1_clg(
        self,
        tasks: Sequence[ProductionTask],
    ) -> tuple[RunRecord, ...]:
        """执行 `val-produce-Prob-F1-CLG`，从 probability 连续补齐四个 role。"""
        return self._run_probability_f1_clg(tasks)

    def run_train_produce_prob_f1_clg(
        self,
        tasks: Sequence[ProductionTask],
    ) -> tuple[RunRecord, ...]:
        """执行 `train-produce-Prob-F1-CLG`，从 probability 连续补齐四个 role。"""
        return self._run_probability_f1_clg(tasks)

    def _run_probability_f1_clg(
        self,
        tasks: Sequence[ProductionTask],
    ) -> tuple[RunRecord, ...]:
        """为 validation/train 共用正式 role 顺序与阈值前置检查。"""
        task_tuple = tuple(tasks)
        self._require_calibration_frozen(task_tuple)
        roles = (
            "probability",
            "components",
            "F1_centered",
            "CLG_centered",
        )
        return tuple(self.run_task(task, roles) for task in task_tuple)

    def _require_calibration_frozen(
        self,
        tasks: Sequence[ProductionTask],
    ) -> None:
        """要求每个待消费 producer 的 calibration `_COMPLETE` 已正式存在。"""
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
        """在正式 selector variant 冻结后独立补齐 Selected role。"""
        return tuple(
            self.run_task(task, ("Selected_Refined_Centered",)) for task in tasks
        )
