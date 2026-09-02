# -*- coding: utf-8 -*-
"""解析 Stage1 V3 五阶段命令并装配正式入口.

主要入口 :func:`main` 提供 `probability`, `blobs`, `centered`, `tune` 和
`evaluate`. 本模块读取 YAML, JSON PDB 清单与选择参数, 并仅在 GPU 阶段恢复
wrapper 和构造 Stage1Dataset. 科学计算与产物发布位于 `pipeline.py`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random

from omegaconf import OmegaConf
import torch

from .checkpoint import load_stage1_wrapper
from .pipeline import (
    run_blobs_stage,
    run_centered_stage,
    run_evaluate_stage,
    run_probability_stage,
    run_tune_stage,
)


# ================================================================================================


def main() -> None:
    """解析一个 Stage1 阶段命令, 构造所需依赖并调用对应正式入口.

    所有命令显式接收 `producer`, JSON PDB 清单, `split` 与 `output_root`.
    清单顶层可以是字符串列表, 也可以是含 `pdb_ids` 字符串列表的对象.
    probability 和正常 centered 额外接收 checkpoint, resolved config 与模型
    代码来源. 可分片阶段先以固定随机种子 3407 打乱完整清单, 再按
    `[shard_index::shard_count]` 选择 0-based 分片. tune, evaluate 与语义拟合
    始终消费完整清单; tune 显式接收参数搜索前固定的来源体素数门槛;
    centered 可显式把 `_BLOB_EXCEED` 改为提示后继续; evaluate 显式接收结果名,
    并在选择参数与全部候选之间二选一. CLI 不计算文件摘要, 不比较 producer
    与训练配置.
    """

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", required=True)
    common.add_argument("--producer", required=True)
    common.add_argument("--pdb-json", required=True)
    common.add_argument("--split", required=True)
    common.add_argument("--output-root", required=True)

    sharded = argparse.ArgumentParser(add_help=False)
    sharded.add_argument("--shard-count", type=int)
    sharded.add_argument("--shard-index", type=int)

    gpu = argparse.ArgumentParser(add_help=False)
    gpu.add_argument("--checkpoint", required=True)
    gpu.add_argument("--resolved-config", required=True)
    gpu.add_argument(
        "--model-code-source",
        choices=("current_workspace", "training_snapshot"),
        required=True,
    )

    parser = argparse.ArgumentParser(description="AdaLigand Stage1 V3 inference")
    commands = parser.add_subparsers(dest="command", required=True)

    probability_parser = commands.add_parser(
        "probability", parents=(common, sharded, gpu)
    )
    probability_parser.add_argument("--overwrite", action="store_true")

    blobs_parser = commands.add_parser("blobs", parents=(common, sharded))
    blobs_parser.add_argument("--alpha", type=float)
    blobs_parser.add_argument("--data-root")
    threshold_source = blobs_parser.add_mutually_exclusive_group(required=True)
    threshold_source.add_argument("--fit-semantic", action="store_true")
    threshold_source.add_argument("--semantic-threshold", type=float)
    threshold_source.add_argument("--semantic-parameters")
    blobs_parser.add_argument("--overwrite", action="store_true")

    centered_parser = commands.add_parser("centered", parents=(common, sharded))
    centered_parser.add_argument("--alpha", type=float)
    centered_parser.add_argument("--forward-min-voxels", type=int)
    centered_parser.add_argument("--selection-parameters")
    centered_parser.add_argument(
        "--continue-on-blob-exceed",
        action="store_true",
    )
    centered_mode = centered_parser.add_mutually_exclusive_group()
    centered_mode.add_argument("--overwrite", action="store_true")
    centered_mode.add_argument("--score-only", action="store_true")
    centered_parser.add_argument("--checkpoint")
    centered_parser.add_argument("--resolved-config")
    centered_parser.add_argument(
        "--model-code-source",
        choices=("current_workspace", "training_snapshot"),
    )

    tune_parser = commands.add_parser("tune", parents=(common,))
    tune_parser.add_argument("--alpha", type=float)
    tune_parser.add_argument("--objective-beta", type=float)
    tune_parser.add_argument(
        "--score-mode", choices=("basic", "gaussian"), required=True
    )
    tune_parser.add_argument("--prefiltered-min-voxel", type=int, required=True)
    tune_parser.add_argument("--data-root", required=True)

    evaluate_parser = commands.add_parser("evaluate", parents=(common,))
    evaluate_parser.add_argument("--alpha", type=float)
    evaluate_parser.add_argument(
        "--artifact", choices=("blobs", "centered"), required=True
    )
    evaluate_parser.add_argument("--evaluation-name", required=True)
    evaluation_selection = evaluate_parser.add_mutually_exclusive_group(required=True)
    evaluation_selection.add_argument("--selection-parameters")
    evaluation_selection.add_argument("--all-candidates", action="store_true")
    evaluate_parser.add_argument("--data-root", required=True)

    arguments = parser.parse_args()
    config = OmegaConf.load(Path(arguments.config))
    OmegaConf.resolve(config)
    alpha = float(
        config.alpha if getattr(arguments, "alpha", None) is None else arguments.alpha
    )
    pdb_payload = json.loads(Path(arguments.pdb_json).read_text(encoding="utf-8"))
    # 字符串序列, 兼容直接列表和 held-out 清单的顶层 `pdb_ids` 字段, 后续顺序原样定义推理 PDB 轴.
    pdb_values = pdb_payload["pdb_ids"] if isinstance(pdb_payload, dict) else pdb_payload
    pdb_ids = tuple(str(value).strip().lower() for value in pdb_values)
    if len(pdb_ids) != len(set(pdb_ids)):
        raise ValueError("PDB JSON 中的标识必须唯一.")
    shard_count = getattr(arguments, "shard_count", None)
    shard_index = getattr(arguments, "shard_index", None)
    if (shard_count is None) != (shard_index is None):
        raise ValueError("--shard-count 与 --shard-index 必须同时提供.")
    if shard_count is not None:
        shuffled = list(pdb_ids)
        random.Random(3407).shuffle(shuffled)
        pdb_ids = tuple(shuffled[int(shard_index) :: int(shard_count)])
    if (
        arguments.command == "blobs"
        and arguments.fit_semantic
        and shard_count is not None
    ):
        raise ValueError("--fit-semantic 必须使用完整 PDB JSON, 不接受分片.")
    if (
        arguments.command == "blobs"
        and arguments.fit_semantic
        and arguments.data_root is None
    ):
        raise ValueError("--fit-semantic 必须提供 --data-root.")

    selection = None
    if getattr(arguments, "selection_parameters", None) is not None:
        selection = json.loads(
            Path(arguments.selection_parameters).read_text(encoding="utf-8")
        )

    dataset = collator = wrapper = None
    needs_model = arguments.command == "probability" or (
        arguments.command == "centered" and not arguments.score_only
    )
    if needs_model:
        if arguments.command == "centered" and (
            arguments.checkpoint is None
            or arguments.resolved_config is None
            or arguments.model_code_source is None
            or arguments.forward_min_voxels is None
        ):
            raise ValueError(
                "正常 centered 必须提供 checkpoint, resolved config, model code source 和 forward min voxels."
            )
        wrapper, training_config = load_stage1_wrapper(
            checkpoint_path=Path(arguments.checkpoint),
            resolved_config_path=Path(arguments.resolved_config),
            map_location="cpu",
            model_code_source=str(arguments.model_code_source),
        )
        from src.datasets.stage1_requests import ResolvedStage1Crop
        import src.datasets.stage1_dataset as current_stage1_dataset

        dataset_arguments = OmegaConf.to_container(
            training_config.dataset, resolve=True
        )
        dataset_arguments.pop("_target_")
        dataset_arguments.update(
            split_file=[
                ResolvedStage1Crop(
                    pdb_id=pdb_id,
                    box_start_zyx=(0, 0, 0),
                    require_targets=False,
                    role="sliding",
                )
                for pdb_id in pdb_ids
            ],
            mode="full_map",
            box_pool_root=None,
            enable_random_rotation=False,
        )
        dataset = current_stage1_dataset.Stage1Dataset(**dataset_arguments)
        collator = dataset.collate_fn
        wrapper.to(torch.device(str(config.device)))

    if arguments.command == "probability":
        run_probability_stage(
            config,
            dataset,
            collator,
            wrapper,
            str(config.device),
            arguments.producer,
            arguments.split,
            pdb_ids,
            Path(arguments.output_root),
            bool(arguments.overwrite),
        )
    elif arguments.command == "blobs":
        semantic_threshold = arguments.semantic_threshold
        if arguments.semantic_parameters is not None:
            semantic_parameters = json.loads(
                Path(arguments.semantic_parameters).read_text(encoding="utf-8")
            )
            semantic_threshold = float(semantic_parameters["threshold_value"])
        run_blobs_stage(
            config,
            arguments.data_root,
            arguments.producer,
            arguments.split,
            pdb_ids,
            Path(arguments.output_root),
            alpha,
            semantic_threshold,
            bool(arguments.fit_semantic),
            bool(arguments.overwrite),
        )
    elif arguments.command == "centered":
        if arguments.score_only and selection is None:
            raise ValueError("--score-only 必须提供 --selection-parameters.")
        run_centered_stage(
            config,
            dataset,
            collator,
            wrapper,
            str(config.device),
            arguments.producer,
            arguments.split,
            pdb_ids,
            Path(arguments.output_root),
            alpha,
            arguments.forward_min_voxels,
            selection,
            bool(arguments.overwrite),
            bool(arguments.score_only),
            bool(arguments.continue_on_blob_exceed),
        )
    elif arguments.command == "tune":
        run_tune_stage(
            config,
            Path(arguments.data_root),
            arguments.producer,
            arguments.split,
            pdb_ids,
            Path(arguments.output_root),
            alpha,
            arguments.score_mode,
            float(
                config.objective_beta
                if arguments.objective_beta is None
                else arguments.objective_beta
            ),
            int(arguments.prefiltered_min_voxel),
        )
    else:
        run_evaluate_stage(
            config,
            Path(arguments.data_root),
            arguments.producer,
            arguments.split,
            pdb_ids,
            Path(arguments.output_root),
            alpha,
            arguments.artifact,
            arguments.evaluation_name,
            None if arguments.all_candidates else selection,
        )


if __name__ == "__main__":
    main()
