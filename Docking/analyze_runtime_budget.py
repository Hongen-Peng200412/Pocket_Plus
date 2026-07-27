from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


def main() -> None:
    """
    汇总已完成样本的逐 Rosetta job 实测时间, 并估算当前 oracle 数据集成本. 

    输入参数:
        - CLI 参数, 包含 pipeline run 根目录、easy20 CSV、输出目录与当前 `nstruct`

    输出:
        - None; 在输出目录写入单位成本 JSON、样本成本 CSV、逐 job CSV 与中文报告
    """
    parser = argparse.ArgumentParser(description="统计 docking 单 job 耗时并估算 oracle 实验成本")
    parser.add_argument("--run-root", type=Path, required=True, help="pipeline run 根目录")
    parser.add_argument("--sample-csv", type=Path, required=True, help="待估算样本集合 CSV")
    parser.add_argument("--output-dir", type=Path, required=True, help="统计输出目录")
    parser.add_argument("--nstruct", type=int, required=True, help="单个 Rosetta job 的 decoy 数")
    args = parser.parse_args()

    sample_inputs = read_sample_inputs(args.sample_csv)
    completed_samples = read_completed_samples(args.run_root)
    job_rows = read_available_job_rows(args.run_root, sample_inputs, completed_samples, args.nstruct)
    completed_rows = [row for row in job_rows if row["sample_complete"]]
    successful_rows = [row for row in completed_rows if row["success"]]
    failed_rows = [row for row in completed_rows if not row["success"]]
    unit_cost = build_unit_cost(successful_rows, failed_rows, args.nstruct)
    sample_rows = build_sample_rows(sample_inputs, completed_samples, job_rows, unit_cost)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "runtime_jobs.csv", job_rows)
    write_csv(args.output_dir / "runtime_budget_samples.csv", sample_rows)
    (args.output_dir / "runtime_budget_summary.json").write_text(
        json.dumps(unit_cost, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (args.output_dir / "runtime_budget_report.md").write_text(
        render_report(args.run_root.name, unit_cost, sample_rows),
        encoding="utf-8",
    )
    print(json.dumps(unit_cost, ensure_ascii=False, indent=2))


def read_sample_inputs(path: Path) -> list[dict[str, Any]]:
    """
    读取 prescan 样本表, 并提取实验成本估算所需字段. 

    输入参数:
        - path: Path, 含 `sample_id`、`num_dockable_ligands` 与 `nk` 的 CSV

    输出:
        - samples: list[dict[str, Any]], 每项包含样本 ID、真实 ligand 数、预测 `nk` 与当前 oracle 预期 job 数
    """
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8", newline="") as handle:
        for item in csv.DictReader(handle):
            n = int(item["num_dockable_ligands"])
            rows.append(
                {
                    "sample_id": item["sample_id"],
                    "n": n,
                    "prescan_nk": int(item["nk"]),
                    "num_conservative_sites": int(item["num_conservative_sites"]),
                    "expected_oracle_jobs": 20 * n * (n + 1),
                }
            )
    return rows


def read_completed_samples(run_root: Path) -> set[str]:
    """
    读取已有样本级 summary, 确定哪些样本已完整结束. 

    输入参数:
        - run_root: Path, 当前 pipeline run 根目录

    输出:
        - sample_ids: set[str], 已写出样本级 summary 的样本 ID 集合
    """
    return {path.parents[1].name for path in (run_root / "samples").glob("*/audit/summary.json")}


def read_available_job_rows(
    run_root: Path,
    sample_inputs: list[dict[str, Any]],
    completed_samples: set[str],
    nstruct: int,
) -> list[dict[str, Any]]:
    """
    读取全部已落盘的 Rosetta `results.json`, 形成逐 job 实测耗时表. 

    输入参数:
        - run_root: Path, 当前 pipeline run 根目录
        - sample_inputs: list[dict[str, Any]], prescan 样本输入字段
        - completed_samples: set[str], 已完整结束的样本 ID 集合
        - nstruct: int, 当前每 job 的 decoy 数

    输出:
        - rows: list[dict[str, Any]], 每项对应一个实际执行过的 Rosetta 子进程
    """
    sample_meta = {row["sample_id"]: row for row in sample_inputs}
    rows: list[dict[str, Any]] = []
    for path in sorted((run_root / "samples").glob("*/variants/*/*/audit/results.json")):
        sample_id = path.parents[4].name
        task = path.parents[2].name
        variant_id = path.parents[1].name
        for result in json.loads(path.read_text(encoding="utf-8")):
            job = result["job"]
            rows.append(
                {
                    "sample_id": sample_id,
                    "n": sample_meta.get(sample_id, {}).get("n", ""),
                    "prescan_nk": sample_meta.get(sample_id, {}).get("prescan_nk", ""),
                    "sample_complete": sample_id in completed_samples,
                    "task": task,
                    "variant_id": variant_id,
                    "ligand_identity": job["ligand"]["label"],
                    "receptor_source": job["receptor"]["name"],
                    "nstruct": nstruct,
                    "runtime_seconds": float(result["seconds"]),
                    "success": bool(result["success"]),
                    "returncode": int(result["returncode"]),
                }
            )
    return rows


def build_unit_cost(
    successful_rows: list[dict[str, Any]],
    failed_rows: list[dict[str, Any]],
    nstruct: int,
) -> dict[str, Any]:
    """
    构造供后续 AI 直接使用的单位成本统计. 

    输入参数:
        - successful_rows: list[dict[str, Any]], 已完整样本中的成功 Rosetta jobs
        - failed_rows: list[dict[str, Any]], 已完整样本中的失败 Rosetta jobs
        - nstruct: int, 当前每 job 的 decoy 数

    输出:
        - summary: dict[str, Any], 含成功 job 分位耗时、失败资源消耗与默认预算口径
    """
    successful_seconds = sorted(float(row["runtime_seconds"]) for row in successful_rows)
    failed_seconds = [float(row["runtime_seconds"]) for row in failed_rows]
    return {
        "unit_definition": "一次 Rosetta docking job；当前脚本为单活跃 CPU 核串行执行",
        "nstruct": nstruct,
        "completed_sample_count": len({row["sample_id"] for row in successful_rows + failed_rows}),
        "successful_job_count": len(successful_seconds),
        "failed_job_count": len(failed_seconds),
        "successful_job_runtime_seconds": {
            "median": nearest_rank(successful_seconds, 0.50),
            "p75_default_budget": nearest_rank(successful_seconds, 0.75),
            "p90_conservative_budget": nearest_rank(successful_seconds, 0.90),
            "sum": sum(successful_seconds),
        },
        "failed_job_consumed_seconds": sum(failed_seconds),
        "total_consumed_seconds": sum(successful_seconds) + sum(failed_seconds),
        "budget_rule": "新任务默认使用成功 job 的 P75 秒/job；含明显长尾时同时报告 P90 上界。",
    }


def build_sample_rows(
    sample_inputs: list[dict[str, Any]],
    completed_samples: set[str],
    job_rows: list[dict[str, Any]],
    unit_cost: dict[str, Any],
) -> list[dict[str, Any]]:
    """
    按样本聚合实测成本与下一轮可用的时间预算. 

    输入参数:
        - sample_inputs: list[dict[str, Any]], easy20 样本定义
        - completed_samples: set[str], 已完整结束的样本 ID 集合
        - job_rows: list[dict[str, Any]], 已落盘逐 job 实测记录
        - unit_cost: dict[str, Any], 单位成本统计

    输出:
        - rows: list[dict[str, Any]], 每样本一行的成本统计与 P75/P90 预算
    """
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in job_rows:
        grouped.setdefault(str(row["sample_id"]), []).append(row)
    p75 = float(unit_cost["successful_job_runtime_seconds"]["p75_default_budget"])
    p90 = float(unit_cost["successful_job_runtime_seconds"]["p90_conservative_budget"])
    rows: list[dict[str, Any]] = []
    for item in sample_inputs:
        sample_jobs = grouped.get(item["sample_id"], [])
        elapsed = sum(float(row["runtime_seconds"]) for row in sample_jobs)
        expected_jobs = int(item["expected_oracle_jobs"])
        rows.append(
            {
                **item,
                "sample_complete": item["sample_id"] in completed_samples,
                "observed_jobs": len(sample_jobs),
                "observed_success_jobs": sum(bool(row["success"]) for row in sample_jobs),
                "observed_elapsed_seconds": elapsed,
                "observed_seconds_per_job": elapsed / len(sample_jobs) if sample_jobs else "",
                "remaining_jobs_if_resumed": max(expected_jobs - len(sample_jobs), 0),
                "p75_estimated_full_seconds": expected_jobs * p75,
                "p90_estimated_full_seconds": expected_jobs * p90,
                "p75_estimated_remaining_seconds": max(expected_jobs - len(sample_jobs), 0) * p75,
                "p90_estimated_remaining_seconds": max(expected_jobs - len(sample_jobs), 0) * p90,
                "actual_to_p75_ratio": elapsed / (expected_jobs * p75)
                if item["sample_id"] in completed_samples and p75
                else "",
            }
        )
    return rows


def nearest_rank(values: list[float], quantile: float) -> float:
    """
    以向上取整的 nearest-rank 口径计算保守分位数. 

    输入参数:
        - values: list[float], 已排序的耗时序列
        - quantile: float, 目标分位, 取值范围 `(0, 1]`

    输出:
        - value: float, 分位数对应的耗时秒数; 空序列返回 0
    """
    if not values:
        return 0.0
    index = max(math.ceil(quantile * len(values)) - 1, 0)
    return values[index]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """
    将结构化行写为 UTF-8 CSV. 

    输入参数:
        - path: Path, 输出文件路径
        - rows: list[dict[str, Any]], 同字段顺序的数据行

    输出:
        - None
    """
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def render_report(run_id: str, unit_cost: dict[str, Any], sample_rows: list[dict[str, Any]]) -> str:
    """
    渲染面向用户与后续 AI 的时间预算速查报告. 

    输入参数:
        - run_id: str, pipeline run ID
        - unit_cost: dict[str, Any], 单位成本统计
        - sample_rows: list[dict[str, Any]], 样本级成本统计

    输出:
        - report: str, Markdown 格式中文报告
    """
    stats = unit_cost["successful_job_runtime_seconds"]
    lines = [
        "# Docking 时间预算速查表",
        "",
        f"- run_id: `{run_id}`",
        f"- 单位: `nstruct={unit_cost['nstruct']}` 时的一次 Rosetta docking job，在当前单活跃 CPU 核串行执行策略下的实测 wall time。",
        f"- 基线样本数: `{unit_cost['completed_sample_count']}` 个完整样本。",
        f"- 成功 job: `{unit_cost['successful_job_count']}`；失败 job: `{unit_cost['failed_job_count']}`。",
        "",
        "## 单 Job 默认预算",
        "",
        "| 口径 | 秒/job | 分钟/job | 用途 |",
        "| --- | ---: | ---: | --- |",
        f"| Median | {stats['median']:.1f} | {stats['median'] / 60:.2f} | 回顾典型耗时 |",
        f"| P75 | {stats['p75_default_budget']:.1f} | {stats['p75_default_budget'] / 60:.2f} | 新任务默认估算 |",
        f"| P90 | {stats['p90_conservative_budget']:.1f} | {stats['p90_conservative_budget'] / 60:.2f} | 长尾保守上界 |",
        "",
        f"失败 job 已实际消耗 `{unit_cost['failed_job_consumed_seconds'] / 3600:.2f}` CPU-core 小时，不混入成功 job 默认预算。",
        "",
        "## 当前 Oracle 成本公式",
        "",
        "本轮 oracle 使用两个 receptor、一个真实中心 variant、九个偏移中心 variant，并同时运行 identity 与 Hungarian。",
        "令 `n` 为真实 dockable ligand 数，则完整样本将运行 `20*n*(n+1)` 个 Rosetta jobs。",
        "`prescan_nk` 用于真实预测位点流程的复杂度参考，不是本轮真实中心 oracle 的直接 job 数公式。",
        "",
        "## Easy20 样本预算",
        "",
        "| sample_id | n | prescan_nk | 完整 jobs | 已观察 jobs | 完整 | 实测小时 | P75 预算小时 | P90 预算小时 |",
        "| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: |",
    ]
    for row in sample_rows:
        lines.append(
            "| {sample_id} | {n} | {prescan_nk} | {expected_oracle_jobs} | {observed_jobs} | "
            "{complete} | {observed:.2f} | {p75:.2f} | {p90:.2f} |".format(
                sample_id=row["sample_id"],
                n=row["n"],
                prescan_nk=row["prescan_nk"],
                expected_oracle_jobs=row["expected_oracle_jobs"],
                observed_jobs=row["observed_jobs"],
                complete="是" if row["sample_complete"] else "否",
                observed=float(row["observed_elapsed_seconds"]) / 3600,
                p75=float(row["p75_estimated_full_seconds"]) / 3600,
                p90=float(row["p90_estimated_full_seconds"]) / 3600,
            )
        )
    lines.extend(
        [
            "",
            "提交新实验时，先计算将生成的 Rosetta job 数，再乘以 P75 得到默认 wall-time 预算；",
            "对明显大样本或长尾实验，同时乘以 P90 报告保守上界。",
            "",
        ]
    )
    return "\n".join(lines)


if __name__ == "__main__":
    main()
