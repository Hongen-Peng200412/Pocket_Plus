from __future__ import annotations

"""
检查 phenix array 预生成结果是否足够进入 baseline 评估。

检查内容:
    1. 读取每个 shard summary, 汇总 generated / skipped / failures。
    2. 重建完整 (system, sample_name) 任务表。
    3. 检查每个目标 phenix_diff_aligned.mrc 是否存在。

用法:
    python src/inference/baseline/check_phenix_diff_maps.py \
        --phenix_output_root /home/penghongen/My_Project/EVAL_OUT/phenix_/phenix_diff_maps \
        --num_shards 30 \
        --systems stardard strict
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[3]
project_root = str(PROJECT_ROOT)
if project_root in sys.path:
    sys.path.remove(project_root)
sys.path.insert(0, project_root)

from src.inference.baseline.build_baseline_cache import derive_phenix_map_path
from src.inference.baseline.generate_phenix_diff_maps import (
    DEFAULT_PHENIX_OUTPUT_ROOT,
    DEFAULT_TEST_JSON,
    DEFAULT_VAL_JSON,
    SYSTEMS,
    _load_samples,
    build_phenix_tasks,
)


def _load_shard_summaries(phenix_output_root: str, num_shards: int) -> tuple[list[dict[str, Any]], list[str]]:
    """
    读取固定数量的 shard summary。

    输入参数:
        - phenix_output_root: str, phenix 差图根目录
        - num_shards: int, 预期 shard 数量

    输出:
        - result: tuple[list[dict[str, Any]], list[str]], 包含:
            - summaries: list[dict[str, Any]], 成功读取的 shard summary
            - missing_summary_paths: list[str], 不存在或无法读取的 summary 路径
    """
    # list[dict[str, Any]], 成功读取的 shard 摘要
    summaries: list[dict[str, Any]] = []
    # list[str], 缺失或损坏的 summary 路径
    missing_summary_paths: list[str] = []
    for shard_index in range(int(num_shards)):
        summary_path = (
            Path(phenix_output_root)
            / "_shard_summaries"
            / f"generate_summary_shard_{shard_index:04d}_of_{int(num_shards):04d}.json"
        )
        try:
            with open(summary_path, "r", encoding="utf-8") as handle:
                summaries.append(json.load(handle))
        except Exception as exc:
            missing_summary_paths.append(f"{summary_path}: {exc}")
    return summaries, missing_summary_paths


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="检查 phenix array 预生成差图是否完整。")
    parser.add_argument("--phenix_output_root", default=DEFAULT_PHENIX_OUTPUT_ROOT, help="phenix 对齐差图根目录。")
    parser.add_argument("--num_shards", type=int, default=30, help="预期 shard summary 数量。")
    parser.add_argument("--systems", nargs="*", default=SYSTEMS, help="要检查的系统, 默认 stardard strict。")
    parser.add_argument("--val_json", default=DEFAULT_VAL_JSON, help="protein_40 样本列表 JSON。")
    parser.add_argument("--test_json", default=DEFAULT_TEST_JSON, help="protein_110 样本列表 JSON。")
    parser.add_argument("--extra_test_json", action="append", default=[], help="追加样本列表 JSON, 支持 label=/path/to/json 或直接路径。")
    args = parser.parse_args(argv)

    # list[str], 追加样本列表 JSON 路径; label=path 只取 path
    extra_jsons = [str(value).split("=", 1)[-1] for value in args.extra_test_json]
    # list[dict[str, Any]], 合并 val/test/extra 后重建完整任务表
    all_samples = _load_samples(str(args.val_json)) + _load_samples(str(args.test_json))
    for extra_json in extra_jsons:
        all_samples.extend(_load_samples(extra_json))
    tasks = build_phenix_tasks(all_samples, [str(v) for v in args.systems])

    # list[dict[str, Any]] / list[str], shard 摘要与缺失摘要
    summaries, missing_summary_paths = _load_shard_summaries(str(args.phenix_output_root), int(args.num_shards))
    # list[dict[str, Any]], 所有 shard 中记录的失败样本
    failures: list[dict[str, Any]] = []
    # int, 各类产物计数
    generated_count = 0
    skipped_count = 0
    for summary in summaries:
        failures.extend(summary.get("failures", []))
        generated_count += len(summary.get("generated", []))
        skipped_count += len(summary.get("skipped", []))

    # list[dict[str, str]], 目标 MRC 缺失记录
    missing_maps: list[dict[str, str]] = []
    for task in tasks:
        system = str(task["system"])
        sample_name = str(task["sample"]["sample_name"])
        out_path = derive_phenix_map_path(str(args.phenix_output_root), system, sample_name)
        if not os.path.exists(out_path):
            missing_maps.append({"system": system, "sample_name": sample_name, "out_path": out_path})

    print("[check_phenix_diff_maps] summary")
    print(f"  phenix_output_root: {args.phenix_output_root}")
    print(f"  expected_tasks: {len(tasks)}")
    print(f"  shard_summaries_loaded: {len(summaries)} / {int(args.num_shards)}")
    print(f"  generated_count: {generated_count}")
    print(f"  skipped_count: {skipped_count}")
    print(f"  failures_count: {len(failures)}")
    print(f"  missing_maps_count: {len(missing_maps)}")

    if missing_summary_paths:
        print("[check_phenix_diff_maps][missing_summaries]")
        for item in missing_summary_paths[:50]:
            print(f"  {item}")
        if len(missing_summary_paths) > 50:
            print(f"  ... 还有 {len(missing_summary_paths) - 50} 条")

    if failures:
        print("[check_phenix_diff_maps][failures]")
        for item in failures[:50]:
            print(f"  system={item.get('system')} sample={item.get('sample_name')} error={item.get('error')}")
        if len(failures) > 50:
            print(f"  ... 还有 {len(failures) - 50} 条")

    if missing_maps:
        print("[check_phenix_diff_maps][missing_maps]")
        for item in missing_maps[:50]:
            print(f"  system={item['system']} sample={item['sample_name']} path={item['out_path']}")
        if len(missing_maps) > 50:
            print(f"  ... 还有 {len(missing_maps) - 50} 条")

    if missing_summary_paths or failures or missing_maps:
        raise SystemExit(1)

    print("[check_phenix_diff_maps] ok: phenix maps are complete.")


if __name__ == "__main__":
    main()
