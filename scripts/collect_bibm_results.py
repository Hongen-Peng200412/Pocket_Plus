from __future__ import annotations

import argparse
import datetime as dt
import json
import re
from pathlib import Path
from typing import Any


DATASET_ORDER = ("protein", "nucleic")

MAIN_METRICS = (
    ("Dice", "avg_voxel_dice"),
    ("PR-AUC", "pr_auc_macro"),
    ("60%cov-F1", "global_instance_f1_loose_cov06"),
    ("60%1to1-F1", "global_instance_f1_cov06"),
    ("60%top3", "top3_success_ratio_cov06"),
    ("60%top4", "top4_success_ratio_cov06"),
    ("60%top5", "top5_success_ratio_cov06"),
    ("30%cov-F1", "global_instance_f1_loose_cov03"),
    ("30%1to1-F1", "global_instance_f1_cov03"),
    ("30%top3", "top3_success_ratio_cov03"),
    ("30%top4", "top4_success_ratio_cov03"),
    ("30%top5", "top5_success_ratio_cov03"),
)

SUPPLEMENT_METRICS = (
    ("Voxel Precision", "avg_voxel_precision"),
    ("Voxel Recall", "avg_voxel_recall"),
    ("Voxel F1", "avg_voxel_f1"),
    ("Voxel IoU", "avg_voxel_iou"),
    ("Avg Num Candidates", "avg_num_candidates"),
    ("60%cov-Precision", "global_instance_precision_loose_cov06"),
    ("60%cov-Recall", "global_instance_recall_loose_cov06"),
    ("60%1to1-Precision", "global_instance_precision_cov06"),
    ("60%1to1-Recall", "global_instance_recall_cov06"),
    ("30%cov-Precision", "global_instance_precision_loose_cov03"),
    ("30%cov-Recall", "global_instance_recall_loose_cov03"),
    ("30%1to1-Precision", "global_instance_precision_cov03"),
    ("30%1to1-Recall", "global_instance_recall_cov03"),
    ("样本数", "num_samples"),
    ("有指标样本数", "num_with_metrics"),
    ("无指标样本数", "num_without_metrics"),
    ("期望样本数", "num_expected_samples"),
    ("缺失期望样本数", "num_missing_expected"),
)


def read_json(path: Path) -> Any:
    """
    读取 UTF-8 JSON 文件. 

    输入参数:
        - path: Path, JSON 文件路径

    输出:
        - data: Any, JSON 反序列化后的对象
    """
    with path.open("r", encoding="utf-8-sig") as file_obj:
        return json.load(file_obj)


def write_json(path: Path, data: Any) -> None:
    """
    写出 UTF-8 JSON 文件. 

    输入参数:
        - path: Path, 输出路径
        - data: Any, 可 JSON 序列化对象

    输出:
        - None
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file_obj:
        json.dump(data, file_obj, ensure_ascii=False, indent=2, sort_keys=True)


def safe_div(numerator: float, denominator: float) -> float | None:
    """
    计算除法, 分母为 0 时返回 None. 

    输入参数:
        - numerator: float, 分子
        - denominator: float, 分母

    输出:
        - value: float | None, 除法结果或 None
    """
    if float(denominator) == 0.0:
        return None
    return float(numerator) / float(denominator)


def mean(values: list[float]) -> float | None:
    """
    计算非空数值列表均值. 

    输入参数:
        - values: list[float], 待平均的数值

    输出:
        - value: float | None, 均值; 空列表返回 None
    """
    if not values:
        return None
    return sum(values) / len(values)


def normalize_sample_name(value: Any) -> str:
    """
    规范化样本名. 

    输入参数:
        - value: Any, 原始样本名

    输出:
        - sample_name: str, 小写样本名
    """
    return str(value).strip().lower()


def metrics_from_row(row: dict[str, Any]) -> dict[str, Any] | None:
    """
    从逐样本行中取出指标字典. 

    输入参数:
        - row: dict[str, Any], 逐样本结果; DL/Phenix 通常把指标放在 metrics 字段

    输出:
        - metrics: dict[str, Any] | None, 单样本指标; 缺失时返回 None
    """
    metrics = row.get("metrics")
    if isinstance(metrics, dict):
        return metrics
    if "voxel_dice" in row:
        return row
    return None


def aggregate_rows(rows: list[dict[str, Any]], expected_samples: set[str]) -> dict[str, Any]:
    """
    按 Pocket Plus 数据集级语义重聚合逐样本指标. 

    输入参数:
        - rows: list[dict[str, Any]], 当前模型逐样本结果; 每项需要 sample_name 和 metrics
        - expected_samples: set[str], 本次聚合期望纳入的样本名集合

    输出:
        - summary: dict[str, Any], 数据集级汇总指标与样本计数
    """
    rows_by_name: dict[str, dict[str, Any]] = {}
    for row in rows:
        sample_name = normalize_sample_name(row.get("sample_name", ""))
        if sample_name and sample_name not in rows_by_name:
            rows_by_name[sample_name] = row

    selected_rows = [rows_by_name[name] for name in sorted(expected_samples) if name in rows_by_name]
    missing_expected = sorted(expected_samples.difference(rows_by_name.keys()))
    metric_rows = [metrics for row in selected_rows if (metrics := metrics_from_row(row)) is not None]
    without_metrics = [
        normalize_sample_name(row.get("sample_name", ""))
        for row in selected_rows
        if metrics_from_row(row) is None
    ]

    summary: dict[str, Any] = {
        "num_expected_samples": len(expected_samples),
        "num_samples": len(selected_rows),
        "num_with_metrics": len(metric_rows),
        "num_without_metrics": len(without_metrics),
        "num_missing_expected": len(missing_expected),
        "missing_expected_samples": missing_expected,
        "without_metrics_samples": sorted(without_metrics),
    }
    if not metric_rows:
        return summary

    for key in ("num_candidates", "voxel_precision", "voxel_recall", "voxel_f1", "voxel_iou", "voxel_dice"):
        values = [float(item[key]) for item in metric_rows if item.get(key) is not None]
        summary[f"avg_{key}"] = mean(values)

    pr_auc_values = [float(item["pr_auc"]) for item in metric_rows if item.get("pr_auc") is not None]
    summary["pr_auc_num_valid"] = len(pr_auc_values)
    summary["pr_auc_macro"] = mean(pr_auc_values)

    sum_num_pred = sum(int(item.get("num_pred_instances", 0)) for item in metric_rows)
    sum_num_gt = sum(int(item.get("num_gt_instances", 0)) for item in metric_rows)
    summary["sum_num_pred_instances"] = sum_num_pred
    summary["sum_num_gt_instances"] = sum_num_gt

    for tag in ("03", "06"):
        sum_tp = sum(int(item.get(f"tp_cov{tag}", 0)) for item in metric_rows)
        precision = safe_div(sum_tp, sum_num_pred)
        recall = safe_div(sum_tp, sum_num_gt)
        summary[f"global_instance_precision_cov{tag}"] = precision
        summary[f"global_instance_recall_cov{tag}"] = recall
        summary[f"global_instance_f1_cov{tag}"] = f1_from_pr(precision, recall)

        sum_tp_pred_loose = sum(int(item.get(f"tp_pred_loose_cov{tag}", 0)) for item in metric_rows)
        sum_tp_gt_loose = sum(int(item.get(f"tp_gt_loose_cov{tag}", 0)) for item in metric_rows)
        precision_loose = safe_div(sum_tp_pred_loose, sum_num_pred)
        recall_loose = safe_div(sum_tp_gt_loose, sum_num_gt)
        summary[f"global_instance_precision_loose_cov{tag}"] = precision_loose
        summary[f"global_instance_recall_loose_cov{tag}"] = recall_loose
        summary[f"global_instance_f1_loose_cov{tag}"] = f1_from_pr(precision_loose, recall_loose)

        for topk in (3, 4, 5):
            key = f"top{topk}_success_cov{tag}"
            summary[f"top{topk}_success_ratio_cov{tag}"] = safe_div(
                sum(int(item.get(key, 0)) for item in metric_rows),
                len(metric_rows),
            )

    return summary


def f1_from_pr(precision: float | None, recall: float | None) -> float | None:
    """
    由 precision 和 recall 计算 F1. 

    输入参数:
        - precision: float | None, 精确率
        - recall: float | None, 召回率

    输出:
        - f1: float | None, F1; 输入缺失或 P+R 为 0 时返回 None
    """
    if precision is None or recall is None or precision + recall == 0.0:
        return None
    return 2.0 * precision * recall / (precision + recall)


def aggregate_entry(entry: dict[str, Any], expected_samples: set[str]) -> dict[str, Any]:
    """
    聚合单个模型条目并保留展示元信息. 

    输入参数:
        - entry: dict[str, Any], 快照中的模型条目
        - expected_samples: set[str], 本次聚合期望纳入的样本名集合

    输出:
        - result: dict[str, Any], 含 entry 元信息和 summary
    """
    summary = aggregate_rows(list(entry.get("rows", [])), expected_samples)
    return {
        "id": entry["id"],
        "name": entry["name"],
        "kind": entry.get("kind"),
        "mode": entry.get("mode"),
        "source_path": entry.get("source_path"),
        "source_exists": bool(entry.get("source_exists", True)),
        "summary": summary,
    }


def build_dataset_tables(dataset: dict[str, Any]) -> dict[str, Any]:
    """
    为一个数据集构造全量表和 Emap2lig 成功子集表. 

    输入参数:
        - dataset: dict[str, Any], 快照中的单数据集结构

    输出:
        - tables: dict[str, Any], 包含 main 和 comparison 两组聚合条目
    """
    all_samples = set(dataset["sample_meta"].keys())
    emap_rows = list(dataset["emap_rows"])
    emap_success = {
        normalize_sample_name(row["sample_name"])
        for row in emap_rows
        if row.get("status") == "ok"
    }
    base_entries = list(dataset["entries"])

    main_entries = [aggregate_entry(entry, all_samples) for entry in base_entries]
    main_entries.append(
        aggregate_entry(
            {
                "id": "emap2lig_original",
                "name": "Emap2lig 原始结果",
                "kind": "Emap2lig",
                "mode": "all",
                "source_path": dataset["emap_source_path"],
                "rows": emap_rows,
            },
            all_samples,
        )
    )
    main_entries.append(
        aggregate_entry(
            {
                "id": "emap2lig_success_only",
                "name": "Emap2lig 去除失败样本后",
                "kind": "Emap2lig",
                "mode": "success-only",
                "source_path": dataset["emap_source_path"],
                "rows": emap_rows,
            },
            emap_success,
        )
    )

    comparison_entries = [aggregate_entry(entry, emap_success) for entry in base_entries]
    comparison_entries.append(
        aggregate_entry(
            {
                "id": "emap2lig_success_only",
                "name": "Emap2lig 去除失败样本后",
                "kind": "Emap2lig",
                "mode": "success-only",
                "source_path": dataset["emap_source_path"],
                "rows": emap_rows,
            },
            emap_success,
        )
    )
    return {
        "emap_success_samples": sorted(emap_success),
        "main": main_entries,
        "comparison": comparison_entries,
    }


def format_value(value: Any, key: str) -> str:
    """
    把指标值格式化为 Markdown 单元格文本. 

    输入参数:
        - value: Any, 原始指标值
        - key: str, 指标键名

    输出:
        - text: str, 格式化文本
    """
    if value is None:
        return "NA"
    if key.startswith("num_") or key in {"sum_num_pred_instances", "sum_num_gt_instances"}:
        return str(int(value))
    if key == "avg_num_candidates":
        return f"{float(value):.2f}"
    return f"{float(value):.4f}"


def markdown_cell(value: Any) -> str:
    """
    清理 Markdown 表格单元格文本. 

    输入参数:
        - value: Any, 原始值

    输出:
        - text: str, 可放入 Markdown 表格的文本
    """
    text = "" if value is None else str(value)
    text = text.replace("\r", " ").replace("\n", " ")
    return text.replace("|", "\\|")


def render_table(headers: list[str], rows: list[list[Any]]) -> str:
    """
    渲染 GitHub 风格 Markdown 表格. 

    输入参数:
        - headers: list[str], 表头
        - rows: list[list[Any]], 表格行

    输出:
        - markdown: str, Markdown 表格文本
    """
    lines = [
        "| " + " | ".join(markdown_cell(item) for item in headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(markdown_cell(item) for item in row) + " |")
    return "\n".join(lines)


def render_metric_table(entries: list[dict[str, Any]], metrics: tuple[tuple[str, str], ...]) -> str:
    """
    渲染模型指标表. 

    输入参数:
        - entries: list[dict[str, Any]], 已聚合模型条目
        - metrics: tuple[tuple[str, str], ...], 展示列名和 summary 键名

    输出:
        - markdown: str, Markdown 表格
    """
    headers = ["模型"] + [label for label, _ in metrics]
    rows = []
    for entry in entries:
        summary = entry["summary"]
        rows.append([entry["name"]] + [format_value(summary.get(key), key) for _, key in metrics])
    return render_table(headers, rows)


def render_failure_table(dataset: dict[str, Any]) -> str:
    """
    渲染 Emap2lig 失败样本清单. 

    输入参数:
        - dataset: dict[str, Any], 快照中的单数据集结构

    输出:
        - markdown: str, Markdown 表格
    """
    rows = []
    sample_meta = dataset["sample_meta"]
    for row in dataset["emap_rows"]:
        if row.get("status") == "ok":
            continue
        sample_name = normalize_sample_name(row.get("sample_name", ""))
        meta = sample_meta.get(sample_name, {})
        rows.append(
            [
                sample_name,
                meta.get("pdb_id", sample_name.upper()),
                meta.get("emdb_id") or "NA",
                row.get("status") or "NA",
                row.get("error") or "NA",
            ]
        )
    return render_table(["sample_name", "PDB ID", "EMDB ID", "status", "error"], rows)


def render_entry_notes(snapshot: dict[str, Any]) -> str:
    """
    渲染每个条目的来源说明. 

    输入参数:
        - snapshot: dict[str, Any], 结果快照

    输出:
        - markdown: str, 条目说明文本
    """
    seen: set[tuple[str, str]] = set()
    lines = [
        "## 条目说明",
        "",
        "- `unet_raw` 展示名对应服务器旧目录 `unet_c1_*`。",
        "- `unet_diff` 展示名对应服务器旧目录 `unet_c2_*`。",
        "- `unet_full` 展示名对应服务器旧目录 `unet_base_*`; `unet000` 只作为历史运行产物名，不作为额外模型展示。",
        "- `emb_unet` 保持服务器目录名和展示名一致。",
        "- `standard` 在服务器目录和配置里拼作 `stardard`; 本文件展示为 `standard`。",
        "- `cov` 指标使用 `global_instance_*_loose_covXX`; `1to1` 指标使用 Hungarian one-to-one 的 `global_instance_*_covXX`。",
        "- `Emap2lig 原始结果` 使用全量测试集，失败样本按 empty prediction metrics 纳入统计；`Emap2lig 去除失败样本后` 仅统计 `status == ok` 的样本。",
        "",
    ]
    for dataset_key in DATASET_ORDER:
        dataset = snapshot["datasets"][dataset_key]
        lines.append(f"### {dataset['title']}来源")
        lines.append("")
        for entry in dataset["entries"]:
            dedup_key = (entry["name"], str(entry.get("source_path")))
            if dedup_key in seen:
                continue
            seen.add(dedup_key)
            status = "存在" if entry.get("source_exists", True) else "缺失"
            lines.append(f"- {entry['name']}: {entry.get('source_path')} ({status})")
        lines.append(f"- Emap2lig: {dataset.get('emap_source_path')}")
        lines.append("")
    return "\n".join(lines).rstrip()


def render_markdown(snapshot: dict[str, Any], tables: dict[str, Any]) -> str:
    """
    渲染最终结果 Markdown. 

    输入参数:
        - snapshot: dict[str, Any], 结果快照
        - tables: dict[str, Any], 聚合后的表格数据

    输出:
        - markdown: str, 完整 Markdown 文本
    """
    generated_at = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    source_time = snapshot.get("generated_at", "NA")
    lines = [
        "# BIBM 结果汇总",
        "",
        "## 来源与口径",
        "",
        f"- 生成时间: {generated_at}",
        f"- 快照时间: {source_time}",
        "- 数值均为比例值，不乘 100%。普通指标保留 4 位小数，`Avg Num Candidates` 保留 2 位小数，缺失写 `NA`。",
        "- 主表与对比表均同时报告 30%/60% coverage 的 F1 与 top3/top4/top5。",
        "",
    ]

    for dataset_key in DATASET_ORDER:
        dataset = snapshot["datasets"][dataset_key]
        dataset_tables = tables[dataset_key]
        lines.extend(
            [
                f"## {dataset['title']}上的结果",
                "",
                render_metric_table(dataset_tables["main"], MAIN_METRICS),
                "",
                f"## {dataset['title']}上的结果—补充细则",
                "",
                render_metric_table(dataset_tables["main"], SUPPLEMENT_METRICS),
                "",
            ]
        )

    for dataset_key in DATASET_ORDER:
        dataset = snapshot["datasets"][dataset_key]
        dataset_tables = tables[dataset_key]
        lines.extend(
            [
                f"## 专门用于与 Emap2lig 对比：{dataset['title']}上的结果",
                "",
                render_metric_table(dataset_tables["comparison"], MAIN_METRICS),
                "",
                f"## 专门用于与 Emap2lig 对比：{dataset['title']}上的结果—补充细则",
                "",
                render_metric_table(dataset_tables["comparison"], SUPPLEMENT_METRICS),
                "",
            ]
        )

    for dataset_key in DATASET_ORDER:
        dataset = snapshot["datasets"][dataset_key]
        lines.extend(
            [
                f"## Emap2lig {dataset['title']}失败样本清单",
                "",
                render_failure_table(dataset),
                "",
            ]
        )

    lines.append(render_entry_notes(snapshot))
    lines.append("")
    return "\n".join(lines)


def build_tables(snapshot: dict[str, Any]) -> dict[str, Any]:
    """
    从快照构造全部聚合表. 

    输入参数:
        - snapshot: dict[str, Any], 远端逐样本结果快照

    输出:
        - tables: dict[str, Any], 数据集名到聚合表的映射
    """
    return {
        dataset_key: build_dataset_tables(snapshot["datasets"][dataset_key])
        for dataset_key in DATASET_ORDER
    }


def parse_args() -> argparse.Namespace:
    """
    解析命令行参数. 

    输出:
        - args: argparse.Namespace, 命令行参数集合
    """
    parser = argparse.ArgumentParser(description="整合 BIBM 多模型评估结果并写出 Markdown 表格。")
    parser.add_argument("--snapshot-json", required=True, help="远端逐样本结果快照 JSON。")
    parser.add_argument("--output-md", required=True, help="最终 Markdown 输出路径。")
    parser.add_argument("--audit-json", required=True, help="聚合审计 JSON 输出路径。")
    return parser.parse_args()


def main() -> None:
    """
    命令行入口. 

    输出:
        - None
    """
    args = parse_args()
    snapshot_path = Path(args.snapshot_json)
    output_md_path = Path(args.output_md)
    audit_json_path = Path(args.audit_json)

    snapshot = read_json(snapshot_path)
    tables = build_tables(snapshot)
    markdown = render_markdown(snapshot, tables)

    output_md_path.parent.mkdir(parents=True, exist_ok=True)
    output_md_path.write_text(markdown, encoding="utf-8")
    write_json(audit_json_path, {"snapshot_path": str(snapshot_path), "tables": tables})
    print(f"[collect_bibm_results] wrote {output_md_path}")
    print(f"[collect_bibm_results] wrote {audit_json_path}")


if __name__ == "__main__":
    main()
