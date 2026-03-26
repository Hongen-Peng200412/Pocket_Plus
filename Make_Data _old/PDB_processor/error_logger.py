"""
================================================================================
统一预处理系统 - 错误日志工具 / Unified Preprocessing System - Error Logging
================================================================================

记录处理错误到 JSON 文件，便于后续分析和调试。
Log processing errors to JSON files for analysis and debugging.
================================================================================
"""

import os
import json
from pathlib import Path
from datetime import datetime
from typing import Optional


def return_error_info(
    file_path: str,
    line: int,
    error_type: str,
    error_detail: str,
    output_dir: str,
    sample_id: Optional[str] = None
) -> None:
    """
    记录错误信息到 JSON 文件
    Log error information to a JSON file
    
    输入参数 / Input:
        - file_path: str, 发生错误的文件路径
        - line: int, 发生错误的行号 (如果不适用则为 -1)
        - error_type: str, 错误类型 (如 'MISSING_BACKBONE', 'NO_LIGAND', 'PARSE_ERROR')
        - error_detail: str, 错误详细描述
        - output_dir: str, 输出目录 (错误日志将存储在 {output_dir}/error_logs/)
        - sample_id: str, 样本ID (可选，如果不提供则从文件名推断)
    
    输出 / Output:
        - None, 直接写入文件
    
    错误类型定义 / Error Types:
        - MISSING_BACKBONE: 骨架原子缺失 (N, CA, C 或 C4', C1', N1/N9)
        - NO_LIGAND: 未检测到配体
        - NO_BINDING_SITE: 无结合位点 (所有原子到配体距离 > 阈值)
        - PARSE_ERROR: PDB/CIF 解析错误
        - INVALID_RESIDUE: 无效残基类型
        - INVALID_ELEMENT: 无效元素类型
        - FILE_NOT_FOUND: 文件不存在
        - EMPTY_STRUCTURE: 结构为空 (无有效原子)
    """
    # 推断 sample_id / Infer sample_id
    if sample_id is None:
        try:
            # str, 从文件名提取 sample_id
            sample_id = Path(file_path).stem
        except Exception:
            sample_id = "unknown"
    # 创建错误日志目录 / Create error log directory
    # Path, 错误日志目录路径
    error_log_dir = Path(output_dir) / "error_logs"
    try:
        error_log_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        print(f"[Error] Failed to create error log directory: {e}")
        return
    # Path, 错误日志文件路径
    error_log_path = error_log_dir / f"error_{sample_id}.json"
    # dict, 错误条目
    error_entry = {
        "sample_id": sample_id,
        "file": str(file_path),
        "line": line,
        "error_type": error_type,
        "error_detail": error_detail,
        "timestamp": datetime.now().isoformat()
    }
    
    
    try:
        # list[dict], 现有日志列表
        existing_logs = []
        if error_log_path.exists():
            try:
                with open(error_log_path, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    if content:
                        existing_logs = json.loads(content)
            except (json.JSONDecodeError, IOError):
                # 文件损坏或为空，重新开始
                existing_logs = []
        
        # 添加新条目
        existing_logs.append(error_entry)
        
        # 写入文件
        with open(error_log_path, 'w', encoding='utf-8') as f:
            json.dump(existing_logs, f, indent=2, ensure_ascii=False)
            
    except Exception as e:
        # 兜底日志，防止日志记录本身导致程序崩溃
        print(f"[Error] Failed to write error log for {sample_id}: {e}")
        print(f"  Error type: {error_type}")
        print(f"  Error detail: {error_detail}")


# ============================================================================
# 预定义错误类型常量 / Predefined Error Type Constants
# ============================================================================

class ErrorType:
    """错误类型常量类"""
    MISSING_BACKBONE = "MISSING_BACKBONE"       # 骨架原子缺失
    NO_LIGAND = "NO_LIGAND"                     # 未检测到配体
    NO_BINDING_SITE = "NO_BINDING_SITE"         # 无结合位点
    PARSE_ERROR = "PARSE_ERROR"                 # 解析错误
    INVALID_RESIDUE = "INVALID_RESIDUE"         # 无效残基
    INVALID_ELEMENT = "INVALID_ELEMENT"         # 无效元素
    FILE_NOT_FOUND = "FILE_NOT_FOUND"           # 文件不存在
    EMPTY_STRUCTURE = "EMPTY_STRUCTURE"         # 结构为空
    DUPLICATE_ATOM = "DUPLICATE_ATOM"           # 重复原子
    INCOMPLETE_RESIDUE = "INCOMPLETE_RESIDUE"   # 残基不完整
    INCOMPLETE_BACKBONE = "INCOMPLETE_BACKBONE" # 缺少代表性骨架原子 (CA 或 C4')
