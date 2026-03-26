"""
可通用代码——————实验管理与文件处理器
该模块负责管理深度学习实验的运行目录、配置备份、日志迁移以及异常清理。
"""

import glob
import logging
import os
import shutil
import time
from pathlib import Path
from typing import Optional, Tuple

from omegaconf import DictConfig, OmegaConf


class ExperimentManager:
    """
    实验管理器类，主要功能包括：
    1. 自动创建实验运行目录（包含时间戳和核心参数标签）。
    2. 保存配置快照（Snapshot），方便后续复现。
    3. 支持 SLURM 集群环境下的日志自动迁移。
    4. 支持异常或短期运行（可能是调试）的自动清理。
    """

    def __init__(
        self,
        config: DictConfig,
        project_root: str,
        feedback_root: str,
        hard_params_keys: list = None,
    ):
        """
        初始化实验管理器。
        Args:
            config: Hydra 解析后的完整配置对象。
            project_root: 项目根路径。
            feedback_root: 实验结果（日志、反馈）的总存储路径。
            hard_params_keys: 需要在目录名中体现的核心参数键值列表（如模型名、数据集名）。
        """
        self.config = config
        self.project_root = Path(project_root)
        self.feedback_root = Path(feedback_root)
        self.start_time = time.time()
        self.hard_params_keys = hard_params_keys or ["model", "dataset", "loss"]            # 默认的"核心参数键"，若未提供则使用 ["model", "dataset", "loss"], 用于在 def _resolve_run_dir 创建反馈文件夹
        self.rank = int(os.environ.get("SLURM_PROCID", os.environ.get("LOCAL_RANK", "0")))
        self.is_rank_zero = self.rank == 0
        self.run_stamp, self.run_stamp_source = self._resolve_run_stamp()
        
        self.run_dir = self._resolve_run_dir()     # 解析并生成本次运行的具体目录
        if self.is_rank_zero:
            os.makedirs(self.run_dir, exist_ok=True)
            print(f"[ExperimentManager] Run stamp: {self.run_stamp} (source: {self.run_stamp_source})")
            print(f"[ExperimentManager] 实验目录已创建: {self.run_dir}")
            self._save_config_snapshot()
            self._migrate_slurm_logs_to_run_dir()
            self._relocate_hydra_logging()

    @staticmethod
    def _sanitize_path_component(value: str) -> str:
        text = str(value).strip()
        if not text:
            return "Unnamed"
        return text.replace("\\", "-").replace("/", "-").replace(" ", "_")

    def _resolve_run_stamp(self) -> Tuple[str, str]:
        shared_stamp = os.environ.get("POCKET_RUN_STAMP", "").strip()
        if shared_stamp:
            return self._sanitize_path_component(shared_stamp), "POCKET_RUN_STAMP"

        job_id = os.environ.get("SLURM_JOB_ID", "").strip()
        step_id = os.environ.get("SLURM_STEP_ID", "").strip()
        if job_id and step_id and step_id.lower() != "batch":
            return f"job{job_id}_step{self._sanitize_path_component(step_id)}", "SLURM_JOB_ID+SLURM_STEP_ID"
        if job_id:
            return f"job{job_id}", "SLURM_JOB_ID"

        local_stamp = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime(self.start_time))
        return local_stamp, "localtime"

    def _resolve_run_dir(self) -> Path:
        """
        根据 hard_params_keys 寻找相应的配置信息, 如果config[key]有name这个属性, 那就把name的值加入到hard_names列表中作为日志文件名的一部分。
        路径格式：{feedback_root}/logs/{hard_params}/{tag}____{timestamp}
        """
        hard_names = []
        for key in self.hard_params_keys:
            if key in self.config and "name" in self.config[key]:
                hard_names.append(str(self.config[key].name))
            else:
                hard_names.append("Unknown")
                print(f"[ExperimentManager] 警告: 配置中缺失关键键值 '{key}' 的 'name' 属性。")

        self.hard_param_str = "-".join(hard_names)
        tag = self._sanitize_path_component(self.config.get("tag", "NoTag"))
        self.run_name = f"{tag}____{self.run_stamp}"
        return self.feedback_root / "logs" / self.hard_param_str / self.run_name

    def _save_config_snapshot(self):
        """
        将本次实验的配置保存为 YAML 文件"config.yaml", 同时额外保存一份简略版的训练配置"train.yaml".
        """
        full_cfg_path = self.run_dir / "config.yaml"
        OmegaConf.save(self.config, full_cfg_path)

        if "train" in self.config:
            train_cfg_path = self.run_dir / "train.yaml"
            OmegaConf.save(self.config.train, train_cfg_path)

    def _migrate_slurm_logs_to_run_dir(self):
        """
        (SLURM 专用) 将原本保存在临时目录的 SLURM 标准输出和错误日志迁移到正式的实验目录下。
        """
        if not self.config.get("migrate_slurm_logs", False):
            return
        job_id = os.environ.get("SLURM_JOB_ID")
        if not job_id:
            return
        # 预设的 SLURM 日志临时存放区
        temp_slurm_dir = self.feedback_root / "logs" / "_temp_slurm"
        if not temp_slurm_dir.exists():
            return

        # 查找匹配 Job ID 的所有日志文件
        log_files = glob.glob(str(temp_slurm_dir / f"*_{job_id}.out"))
        log_files += glob.glob(str(temp_slurm_dir / f"*_{job_id}.err"))
        for log_f in log_files:
            try:
                dest = self.run_dir / Path(log_f).name
                shutil.move(log_f, dest)
                print(f"[ExperimentManager] 已迁移 SLURM 日志: {log_f} -> {dest}")
            except Exception as e:
                print(f"[ExperimentManager] 迁移 SLURM 日志失败 {log_f}: {e}")

    def _relocate_hydra_logging(self):
        """
        重定向 Hydra 和 Python 标准日志到实验目录下。
        Hydra 默认会在当前工作目录生成 .hydra 文件夹和日志，本方法将其移动到我们自定义的 run_dir。
        """
        cwd = Path.cwd()
        hydra_dir = cwd / ".hydra"
        
        # 1. 移动 .hydra 配置文件夹
        if hydra_dir.exists():
            try:
                target = self.run_dir / ".hydra"
                if target.exists():
                    shutil.rmtree(target)
                shutil.move(str(hydra_dir), str(target))
                print(f"[ExperimentManager] 已移动 .hydra 文件夹到 {target}")
            except Exception as e:
                print(f"[ExperimentManager] 移动 .hydra 文件夹失败: {e}")

        # 2. 定位并重定向当前所有的 FileHandler
        target_log_file = self.run_dir / "train.log"
        loggers = [logging.getLogger()] + [
            logging.getLogger(name) for name in logging.root.manager.loggerDict
        ]

        for logger in loggers:
            for handler in list(logger.handlers):
                if not isinstance(handler, logging.FileHandler):
                    continue
                
                # 获取 handler 正在写入的文件路径
                current_log_path = Path(handler.baseFilename).resolve()
                # 仅处理位于当前工作目录下的日志文件
                if not (current_log_path.parent == cwd or cwd in current_log_path.parents):
                    continue

                try:
                    handler.close()
                    logger.removeHandler(handler)

                    # 如果旧日志文件存在且目标路径尚无文件，则执行搬迁
                    if current_log_path.exists() and not target_log_file.exists():
                        shutil.move(str(current_log_path), str(target_log_file))

                    # 创建新的指向 run_dir 的 FileHandler
                    new_handler = logging.FileHandler(
                        str(target_log_file), mode="a", encoding=handler.encoding
                    )
                    new_handler.setFormatter(handler.formatter)
                    new_handler.setLevel(handler.level)
                    logger.addHandler(new_handler)
                except Exception as e:
                    print(f"[ExperimentManager] 重定向日志处理器失败: {e}")

    def check_and_cleanup(self, error: Optional[Exception] = None):
        """
        在实验结束时（或出错时）调用。
        如果运行时间过短且未设置保留短时运行，则自动删除相关目录。
        这有助于清理调试产生的垃圾文件夹。

        Args:
            error: 如果是因为抛出异常而退出，传入异常对象。
        """
        if not self.is_rank_zero:
            return

        duration = time.time() - self.start_time
        # 获取配置中的最小保留时长（秒），默认 1 小时
        min_duration = int(self.config.get("min_duration", 3600))
        # 是否强制保留短时运行（用于调试环境）
        keep_short_runs = bool(self.config.get("keep_short_runs", False))

        # 场景 1: 发生错误退出
        if error is not None:
            print(f"[ExperimentManager] 检测到异常退出: {error}")
            if duration < min_duration and self.run_dir.exists():
                print(
                    f"[ExperimentManager] 实验时长 ({duration:.2f}s) 短于设定的保留阈值 ({min_duration}s)，且发生了错误。"
                    "正在自动清理实验目录以节省空间。"
                )
                try:
                    shutil.rmtree(self.run_dir)
                    print(f"[ExperimentManager] 已删除目录: {self.run_dir}")
                except Exception as e:
                    print(f"[ExperimentManager] 删除目录失败 {self.run_dir}: {e}")
            return

        # 场景 2: 正常退出但时间太短
        if duration < min_duration and not keep_short_runs:
            print(
                f"[ExperimentManager] 实验时长 ({duration:.2f}s) < 最小保留时长 ({min_duration}s)。"
                "清理目录（可能是由于手动终止或调试）。"
            )
            try:
                if self.run_dir.exists():
                    shutil.rmtree(self.run_dir)
                    print(f"[ExperimentManager] 已删除目录: {self.run_dir}")
            except Exception as e:
                print(f"[ExperimentManager] 删除目录失败 {self.run_dir}: {e}")
        else:
            print(f"[ExperimentManager] 实验结束，结果已保存。总时长: {duration:.2f}s")
