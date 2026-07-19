# -*- coding: utf-8 -*-
"""AdaLigand Stage1 全图、校准、居中与 Selected 生产命令。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from src.artifacts.paths import STAGE1_MODEL_NAMES
from src.component_lineage.clg import CLGEnumerationConfig
from src.evaluation.calibration import (
    calibrate_published_full_maps_and_freeze_thresholds,
)

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


DEFAULT_MAX_VOXELS = 1023


def _add_identity_arguments(parser: argparse.ArgumentParser, split: str) -> None:
    """加入 producer、固定 PDB 清单与稳定分片参数。"""
    parser.set_defaults(split=split)
    parser.add_argument("--producer", required=True, choices=STAGE1_MODEL_NAMES)
    parser.add_argument("--pdb-list", required=True)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--output-root", required=True)


def _add_runtime_arguments(parser: argparse.ArgumentParser) -> None:
    """加入 A—G、checkpoint/config、device 与受控窗口 batch 参数。"""
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default=None)
    parser.add_argument("--device", required=True)
    parser.add_argument("--window-batch-size", type=int, default=1)
    parser.add_argument("--cache-max-bytes", type=int, default=536_870_912)


def _add_component_arguments(parser: argparse.ArgumentParser) -> None:
    """加入 depth1 CLG 的显式可覆盖工程参数。"""
    parser.add_argument("--max-split-events", type=int, default=1)
    parser.add_argument("--max-merge-events", type=int, default=1)
    parser.add_argument("--max-nodes-per-clg", type=int, default=32)
    parser.add_argument("--f1-eligible-limit", type=int, default=200)


def build_parser() -> argparse.ArgumentParser:
    """构造五阶段生产命令与 Selected 独立补跑命令。"""
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
        help="正式默认 1023，来自 673364 occurrence 的 Q95=682×1.5",
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
    """从 CLI 参数构造单 producer 的完整 runtime assembly。"""
    return Stage1RuntimeAssembly(
        data_root=arguments.data_root,
        stage1_model_name=arguments.producer,
        checkpoint_path=arguments.checkpoint,
        resolved_config_path=arguments.config,
        device=arguments.device,
        window_batch_size=arguments.window_batch_size,
        cache_max_bytes=arguments.cache_max_bytes,
    )


def _tasks(arguments: argparse.Namespace):
    """按固定清单与行号取模规则构造当前 worker 的任务。"""
    return build_production_tasks(
        stage1_model_name=arguments.producer,
        split=arguments.split,
        pdb_list_path=arguments.pdb_list,
        shard_index=arguments.shard_index,
        shard_count=arguments.shard_count,
    )


def _clg_config(arguments: argparse.Namespace) -> CLGEnumerationConfig:
    """把显式 CLI 值转成树枚举契约。"""
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
    """装配 probability、components 与两个 centered role 的具体 producers。"""
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
        )
    )
    return producers


def _records_payload(
    command: str,
    records: Sequence[RunRecord],
) -> dict[str, object]:
    """生成机器可读、可直接收集的 worker 结果摘要。"""
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
    """执行一个明确阶段；失败直接非零退出，不猜默认 checkpoint 或阈值。"""
    arguments = build_parser().parse_args(argv)

    if arguments.command == "freeze-thresholds":
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
            """解析默认同目录或显式外置 Selector 产物。"""
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
