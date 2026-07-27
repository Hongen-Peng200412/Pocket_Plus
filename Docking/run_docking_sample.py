from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from docking_pipeline.config import MatchingOptions, RosettaOptions, ServerPaths
from docking_pipeline.runner import run_sample_smoke


def main() -> None:
    """
    运行单样本低成本 docking smoke pipeline. 
    输入参数:
        - CLI `--pdb-id`: str, 待处理样本 PDB ID
        - CLI `--summary-json`: Path, 可选, 输出摘要 JSON 路径

    输出:
        - None; 结果写入服务器允许目录, 并可选写入摘要 JSON
    """
    parser = argparse.ArgumentParser(description="运行 Pocket Plus 分子对接 smoke pipeline")
    parser.add_argument("--pdb-id", required=True, help="样本 PDB ID, 例如 7zdf")
    parser.add_argument("--summary-json", type=Path, help="可选摘要 JSON 输出路径")
    args = parser.parse_args()

    summary = run_sample_smoke(
        pdb_id=args.pdb_id,
        paths=ServerPaths.default(),
        rosetta_options=RosettaOptions.smoke(),
        matching_options=MatchingOptions.current(),
    )
    serializable = _to_jsonable(summary)
    print(json.dumps(serializable, ensure_ascii=False, indent=2))
    if args.summary_json:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(json.dumps(serializable, ensure_ascii=False, indent=2), encoding="utf-8")


def _to_jsonable(value):
    """把 dataclass 和 Path 等对象转换成 JSON 友好结构. """
    if hasattr(value, "__dataclass_fields__"):
        return _to_jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


if __name__ == "__main__":
    main()

