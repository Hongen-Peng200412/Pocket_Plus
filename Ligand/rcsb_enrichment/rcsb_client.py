from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import requests

from . import config
from .io_utils import make_output_dirs


@dataclass
class DownloadResult:
    """
    单个下载动作的结果. 

    输入字段:
        - path: Path, 目标文件路径
        - url: str, 下载 URL
        - attempts: int, 实际请求次数
        - error_message: str, 失败信息; 成功时为空字符串
    """

    path: Path
    url: str
    attempts: int
    error_message: str = ""


@dataclass
class RCSBClient:
    """
    RCSB 下载客户端. 

    输入参数:
        - output_root: Path, 输出根目录
        - force_download: bool, 是否覆盖已有下载文件
        - max_retries: int, 失败最大重试次数
        - retry_sleep: float, 每次重试前 sleep 秒数
    """

    output_root: Path
    force_download: bool
    max_retries: int
    retry_sleep: float
    session: requests.Session = field(default_factory=requests.Session)

    def __post_init__(self) -> None:
        make_output_dirs(self.output_root)

    def _download_text(self, url: str, path: Path) -> DownloadResult:
        """
        下载文本文件并用临时文件原子替换目标文件. 

        输入参数:
            - url: str, 下载 URL
            - path: Path, 目标文件路径

        输出:
            - result: DownloadResult, 包含目标路径、URL、尝试次数和错误信息
        """

        if path.exists() and path.stat().st_size > 0 and not self.force_download:
            return DownloadResult(path=path, url=url, attempts=0)

        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_name(f"{path.name}.tmp.{os.getpid()}")
        last_error = ""
        attempts = 0
        for attempt in range(1, self.max_retries + 1):
            attempts = attempt
            try:
                response = self.session.get(url, timeout=90)
                if response.status_code == 200 and response.text.strip():
                    tmp_path.write_text(response.text, encoding="utf-8")
                    tmp_path.replace(path)
                    return DownloadResult(path=path, url=url, attempts=attempt)
                last_error = f"HTTP {response.status_code}: {response.text[:200]}"
            except Exception as exc:
                last_error = repr(exc)
            time.sleep(self.retry_sleep)

        if tmp_path.exists():
            tmp_path.unlink()
        return DownloadResult(path=path, url=url, attempts=attempts, error_message=last_error)

    def download_full_cif(self, pdb_id_upper: str) -> DownloadResult:
        """
        下载或复用 RCSB full CIF. 

        输入参数:
            - pdb_id_upper: str, 大写 PDB ID, 如 '5GMK'

        输出:
            - result: DownloadResult, full CIF 下载结果
        """

        url = config.RCSB_FULL_CIF_URL.format(pdb_id=pdb_id_upper)
        path = self.output_root / "rcsb_full_cif" / f"{pdb_id_upper}.cif"
        return self._download_text(url, path)

    def download_chemcomp(self, ccd_id: str) -> DownloadResult:
        """
        下载或复用 RCSB chemcomp JSON. 

        输入参数:
            - ccd_id: str, CCD ID

        输出:
            - result: DownloadResult, chemcomp JSON 下载结果
        """

        url = config.RCSB_CHEMCOMP_URL.format(ccd_id=ccd_id)
        path = self.output_root / "rcsb_chemcomp_cache" / f"{ccd_id}.json"
        return self._download_text(url, path)

    def download_native_mol2(self, pdb_id: str, label_asym_id: str, pdb_seq_num: str, ccd_id: str, candidate_id: int) -> DownloadResult:
        """
        下载或复用 RCSB native ligand mol2. 

        输入参数:
            - pdb_id: str, 小写 PDB ID
            - label_asym_id: str, RCSB label asym ID
            - pdb_seq_num: str, RCSB 页面 residue number
            - ccd_id: str, CCD ID
            - candidate_id: int, Make_Data candidate ID

        输出:
            - result: DownloadResult, native mol2 下载结果
        """

        filename = f"{candidate_id}_{ccd_id}_{label_asym_id}_{pdb_seq_num}.mol2"
        url = config.RCSB_LIGAND_MOL2_URL.format(
            pdb_id=pdb_id,
            pdb_seq_num=pdb_seq_num,
            label_asym_id=label_asym_id,
            filename=filename,
        )
        path = self.output_root / "rcsb_ligand_mol2" / pdb_id / filename
        return self._download_text(url, path)
