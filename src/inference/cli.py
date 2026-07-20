# -*- coding: utf-8 -*-
"""AdaLigand Stage1 完整图、阈值校准、居中特征与最终精修产物的命令行入口。

主要入口:
    - `build_parser`: 声明六个互斥子命令及其显式参数。
    - `main`: 把命令行参数转换为运行时对象和固定 PDB 任务，再调用阶段编排器。

本模块只负责选择明确阶段、装配回调函数和输出任务摘要。数据读取与模型恢复由
`Stage1RuntimeAssembly` 负责，数值计算由 evaluation、component_lineage 和
centered 模块负责，续跑状态由 `Stage1ProductionRunner` 负责。命令不会猜测
checkpoint、配置文件或冻结阈值，也不会跨阶段自动补做未声明的工作。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from src.component_lineage.clg import CLGEnumerationConfig
from src.stage1_producers import STAGE1_MODEL_NAMES

from .assembly import (
    AGOccurrenceVoxelLoader,
    Stage1RuntimeAssembly,
    build_production_tasks,
    load_pdb_id_list,
)
from .runner import (
    RunRecord,
    Stage1ProductionRunner,
    make_component_role_producer,
    make_f1_clg_centered_role_producers,
    make_probability_role_producer,
    make_selected_refined_role_producer,
)


# int，冻结阈值命令使用的 candidate 体素数上限；该默认值来自正式 occurrence
# 体素数分布的 Q95 放大结果，仍允许调用方通过 `--max-voxels` 显式覆盖。
DEFAULT_MAX_VOXELS = 2046


def _add_identity_arguments(parser: argparse.ArgumentParser, split: str) -> None:
    """
    向子命令加入模型来源、固定 PDB 清单、稳定分片和产物根目录参数。

    输入参数:
        - parser: argparse.ArgumentParser, 当前待扩展的子命令解析器
        - split: str, 写入解析结果的固定数据划分名；调用方不能再由命令行覆盖

    副作用:
        - 原地修改 `parser`，加入 `producer/pdb_list/shard_index/shard_count/output_root`
          参数；分片规则随后由 `build_production_tasks` 按清单位置取模实现。
    """
    parser.set_defaults(split=split)
    parser.add_argument("--producer", required=True, choices=STAGE1_MODEL_NAMES)
    parser.add_argument("--pdb-list", required=True)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--output-root", required=True)


def _add_runtime_arguments(parser: argparse.ArgumentParser) -> None:
    """
    向子命令加入 Stage1 原始数据、模型恢复、运行设备和受控批量参数。

    输入参数:
        - parser: argparse.ArgumentParser, 当前待扩展的子命令解析器

    副作用:
        - 原地修改 `parser`，加入 `data_root/checkpoint/config/device` 以及模型输入缓存
          以及完整图滑窗、centered 完整 forward 的批量大小参数。
    """
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default=None)
    parser.add_argument(
        "--allow-current-workspace-code",
        action="store_true",
        help="checkpoint 缺少 src_snapshot/src 时显式允许使用当前工作区代码",
    )
    parser.add_argument("--device", required=True)
    parser.add_argument("--window-batch-size", type=int, default=1)
    parser.add_argument("--centered-batch-size", type=int, default=12)
    parser.add_argument("--cache-max-bytes", type=int, default=536_870_912)


def _add_component_arguments(parser: argparse.ArgumentParser) -> None:
    """
    向子命令加入 depth1 CLG 的显式可覆盖工程参数。

    输入参数:
        - parser: argparse.ArgumentParser, 当前待扩展的子命令解析器

    副作用:
        - 原地修改 `parser`，加入拆分事件、合并事件、单个 CLG 节点数和
          `t_F1` 层 eligible component 数量上限。
    """
    parser.add_argument("--max-split-events", type=int, default=1)
    parser.add_argument("--max-merge-events", type=int, default=1)
    parser.add_argument("--max-nodes-per-clg", type=int, default=32)
    parser.add_argument("--f1-eligible-limit", type=int, default=200)


def build_parser() -> argparse.ArgumentParser:
    """
    构造阈值冻结前后五个生产入口和 Selected 独立补跑入口。

    输出:
        - parser: argparse.ArgumentParser, 含六个子命令及其参数契约的根解析器；
          解析结果始终含唯一 `command`
    """
    parser = argparse.ArgumentParser(
        description="AdaLigand Stage1 可续跑完整图与居中推理"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    cal_probability = subparsers.add_parser(
        "cal-probability",
        help="阶段一：只生成 calibration 完整图 probability",
    )
    _add_identity_arguments(cal_probability, "calibration")
    _add_runtime_arguments(cal_probability)

    freeze = subparsers.add_parser(
        "freeze-thresholds",
        help="消费完整 calibration 清单并冻结 32769-bin 阈值与 fitted 指标",
    )
    freeze.add_argument("--producer", required=True, choices=STAGE1_MODEL_NAMES)
    freeze.add_argument("--pdb-list", required=True)
    freeze.add_argument("--data-root", required=True)
    freeze.add_argument("--output-root", required=True)
    freeze.add_argument("--min-voxels", type=int, default=32)
    freeze.add_argument(
        "--max-voxels",
        type=int,
        default=DEFAULT_MAX_VOXELS,
        help="正式默认 2046，来自 673364 occurrence 的 Q95=682×3.0",
    )
    freeze.add_argument("--denominator", type=int, default=32768)

    for command, split, help_text in (
        (
            "cal-produce-f1-clg",
            "calibration",
            "阶段二：为已有 calibration probability 补齐 components/F1/CLG",
        ),
        (
            "val-produce-prob-f1-clg",
            "validation",
            "为固定 validation 分片连续补齐 probability/components/F1/CLG",
        ),
        (
            "train-produce-prob-f1-clg",
            "train",
            "为固定 train 分片连续补齐 probability/components/F1/CLG",
        ),
    ):
        stage = subparsers.add_parser(command, help=help_text)
        _add_identity_arguments(stage, split)
        _add_runtime_arguments(stage)
        _add_component_arguments(stage)

    selected = subparsers.add_parser(
        "selected-refined",
        help="把 Selector selection 恢复为 source nodes 并重跑发布 Selected role",
    )
    selected.add_argument(
        "--split", required=True, choices=("calibration", "validation", "train")
    )
    selected.add_argument("--producer", required=True, choices=STAGE1_MODEL_NAMES)
    selected.add_argument("--pdb-list", required=True)
    selected.add_argument("--shard-index", type=int, default=0)
    selected.add_argument("--shard-count", type=int, default=1)
    selected.add_argument("--output-root", required=True)
    selected.add_argument(
        "--selection-root",
        default=None,
        help="外置时固定布局为 ROOT/producer/split/pdb_id/selection.npz；缺省读取 PDB 正式目录/selector/selection.npz",
    )
    _add_runtime_arguments(selected)
    return parser


def _build_runtime(arguments: argparse.Namespace) -> Stage1RuntimeAssembly:
    """
    从命令行参数构造单个 Stage1 模型来源的延迟运行时装配器。

    输入参数:
        - arguments: argparse.Namespace, 当前子命令解析出的数据路径、checkpoint、
          可选配置、运行设备、滑窗批量大小和缓存上限

    输出:
        - runtime: Stage1RuntimeAssembly, 延迟恢复模型并提供完整图输入、occurrence
          体素和居中批次构造器的运行时装配器
    """
    return Stage1RuntimeAssembly(
        data_root=arguments.data_root,
        stage1_model_name=arguments.producer,
        checkpoint_path=arguments.checkpoint,
        resolved_config_path=arguments.config,
        device=arguments.device,
        window_batch_size=arguments.window_batch_size,
        centered_batch_size=arguments.centered_batch_size,
        cache_max_bytes=arguments.cache_max_bytes,
        allow_current_workspace_code=arguments.allow_current_workspace_code,
    )


def _tasks(arguments: argparse.Namespace):
    """
    按固定清单位置取模规则构造当前进程的任务。

    输入参数:
        - arguments: argparse.Namespace, 含 producer、split、PDB 清单路径与分片参数的解析结果

    输出:
        - tasks: tuple[ProductionTask,...], 满足
          `清单位置 % shard_count == shard_index` 的模型来源/数据划分/PDB 任务；
          结果保持原清单顺序
    """
    return build_production_tasks(
        stage1_model_name=arguments.producer,
        split=arguments.split,
        pdb_list_path=arguments.pdb_list,
        shard_index=arguments.shard_index,
        shard_count=arguments.shard_count,
    )


def _clg_config(arguments: argparse.Namespace) -> CLGEnumerationConfig:
    """
    把显式命令行值转换为组件谱系组枚举契约。

    输入参数:
        - arguments: argparse.Namespace, 含拆分事件、合并事件和单个 CLG 节点数上限
          的解析结果

    输出:
        - config: CLGEnumerationConfig, 当前命令使用的确定性组件谱系组枚举限制
    """
    return CLGEnumerationConfig(
        max_split_events=arguments.max_split_events,
        max_merge_events=arguments.max_merge_events,
        max_nodes_per_CLG=arguments.max_nodes_per_clg,
    )


def _standard_role_producers(
    runtime: Stage1RuntimeAssembly,
    arguments: argparse.Namespace,
    include_probability: bool,
) -> dict[str, Any]:
    """
    装配完整图概率、组件谱系与两类居中特征产物的生成回调。

    输入参数:
        - runtime: Stage1RuntimeAssembly, 当前 producer 的数据、wrapper 与 batch builder 提供器
        - arguments: argparse.Namespace, 含 CLG 与 blob 上限的解析结果
        - include_probability: bool, 是否把完整图概率生成回调纳入当前命令；校准集
          第二阶段为 False，validation/train 连续生产为 True

    输出:
        - producers: dict[str, RoleProducer], 产物角色名到生成回调的映射。
        - `components`: 构造组件森林、组件谱系组及 occurrence 交集。
        - `F1_centered`: 为 `t_F1` 层 eligible component 生成居中特征。
        - `CLG_centered`: 为组件谱系组的候选集合生成居中特征。
        - `probability`: 仅在 `include_probability=True` 时存在，生成完整图概率。
    """
    # dict[str, RoleProducer]，当前命令按依赖装配的产物角色回调表。
    producers: dict[str, Any] = {}
    if include_probability:
        producers["probability"] = make_probability_role_producer(
            runtime.full_map_input
        )
    producers["components"] = make_component_role_producer(
        occurrence_voxel_provider=runtime.occurrence_voxels,
        clg_config=_clg_config(arguments),
        f1_eligible_limit=arguments.f1_eligible_limit,
    )
    producers.update(
        make_f1_clg_centered_role_producers(
            wrapper_provider=runtime.wrapper_provider,
            batch_builder_provider=runtime.centered_batch_builder,
            centered_batch_size=runtime.centered_batch_size,
        )
    )
    return producers


def _records_payload(
    command: str,
    records: Sequence[RunRecord],
) -> dict[str, object]:
    """
    生成机器可读、可直接收集的 worker 结果摘要。

    输入参数:
        - command: str, 当前执行的 CLI 子命令名
        - records: Sequence[RunRecord], 当前 worker 按任务顺序得到的生产结果

    输出:
        - payload: dict[str, object], 当前进程的机器可读结果摘要。
        - `command`: str, 当前子命令名。
        - `task_count`: int, 结果记录数。
        - `status_counts`: dict[str, int], 各
          `completed/skipped_complete/skipped_running/blob_exceed` 状态的记录数。
        - `records`: list[dict[str, object]], 与输入同序的逐任务记录；每项含模型
          来源、数据划分、PDB 标识、结束状态和本次新完成的产物角色。
    """
    # dict[str, int]，当前进程按最终状态累计的任务数。
    status_counts: dict[str, int] = {}
    rows = []
    for record in records:
        status_counts[record.status] = status_counts.get(record.status, 0) + 1
        rows.append(
            {
                "producer": record.task.stage1_model_name,
                "split": record.task.split,
                "pdb_id": record.task.pdb_id,
                "status": record.status,
                "completed_roles": list(record.completed_roles),
            }
        )
    return {
        "command": str(command),
        "task_count": len(records),
        "status_counts": status_counts,
        "records": rows,
    }


def main(argv: Sequence[str] | None = None) -> int:
    """
    执行一个明确阶段并把当前进程的结果摘要写到标准输出。

    输入参数:
        - argv: Sequence[str] | None, 显式命令参数; None 时由 argparse 读取当前进程参数

    输出:
        - exit_code: int, 当前阶段的全部任务已得到确定结果时为 0；参数、输入、
          模型恢复或生产异常不在本函数中吞掉，由进程以非零状态退出
    """
    arguments = build_parser().parse_args(argv)

    # 阈值冻结只读取已发布的 calibration 完整图，不需要恢复 Stage1 模型。
    if arguments.command == "freeze-thresholds":
        from src.evaluation.calibration import (
            calibrate_published_full_maps_and_freeze_thresholds,
        )

        pdb_ids = load_pdb_id_list(arguments.pdb_list)
        result, metrics = calibrate_published_full_maps_and_freeze_thresholds(
            output_root=arguments.output_root,
            stage1_model_name=arguments.producer,
            calibration_pdb_ids=pdb_ids,
            occurrence_voxel_loader=AGOccurrenceVoxelLoader(arguments.data_root),
            min_voxels=arguments.min_voxels,
            max_voxels=arguments.max_voxels,
            denominator=arguments.denominator,
            split="calibration",
        )
        print(
            json.dumps(
                {
                    "command": arguments.command,
                    "producer": arguments.producer,
                    "pdb_count": len(pdb_ids),
                    "denominator": result.denominator,
                    "t_F1": result.t_F1,
                    "max_voxels": arguments.max_voxels,
                    "metrics": metrics,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    # 其余五个命令均需要模型或数据访问；运行时对象按需加载并在当前进程复用。
    runtime = _build_runtime(arguments)
    tasks = _tasks(arguments)
    if arguments.command == "cal-probability":
        runner = Stage1ProductionRunner.for_current_process(
            output_root=arguments.output_root,
            role_producers={
                "probability": make_probability_role_producer(
                    runtime.full_map_input
                )
            },
        )
        records = runner.run_calibration_probability(tasks)
    elif arguments.command in {
        "cal-produce-f1-clg",
        "val-produce-prob-f1-clg",
        "train-produce-prob-f1-clg",
    }:
        include_probability = arguments.command != "cal-produce-f1-clg"
        runner = Stage1ProductionRunner.for_current_process(
            output_root=arguments.output_root,
            role_producers=_standard_role_producers(
                runtime, arguments, include_probability=include_probability
            ),
        )
        if arguments.command == "cal-produce-f1-clg":
            records = runner.run_cal_produce_f1_clg(tasks)
        elif arguments.command == "val-produce-prob-f1-clg":
            records = runner.run_val_produce_prob_f1_clg(tasks)
        else:
            records = runner.run_train_produce_prob_f1_clg(tasks)
    elif arguments.command == "selected-refined":
        selection_root = (
            None if arguments.selection_root is None else Path(arguments.selection_root)
        )

        def resolve_selection(task, paths):
            """
            解析当前 PDB 的默认同目录或显式外置 Selector 选择结果路径。

            输入参数:
                - task: ProductionTask, 当前模型来源/数据划分/PDB 任务身份
                - paths: Stage1ArtifactPaths, 当前任务的正式 Stage1 产物路径集合

            输出:
                - selection_path: Path, 当前 PDB 对应的 `selection.npz` 路径；外置
                  目录固定为
                  `selection_root/producer/split/pdb_id/selection.npz`
            """
            if selection_root is None:
                return paths.pdb_root / "selector" / "selection.npz"
            return (
                selection_root
                / task.stage1_model_name
                / task.split
                / task.pdb_id
                / "selection.npz"
            )

        producer = make_selected_refined_role_producer(
            wrapper_provider=runtime.wrapper_provider,
            batch_builder_provider=runtime.centered_batch_builder,
            centered_batch_size=runtime.centered_batch_size,
            selection_path_provider=resolve_selection,
        )
        runner = Stage1ProductionRunner.for_current_process(
            output_root=arguments.output_root,
            role_producers={"Selected_Refined_Centered": producer},
        )
        records = runner.run_selected(tasks)
    else:
        raise AssertionError(f"未处理命令: {arguments.command}")

    print(
        json.dumps(
            _records_payload(arguments.command, records),
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["DEFAULT_MAX_VOXELS", "build_parser", "main"]
