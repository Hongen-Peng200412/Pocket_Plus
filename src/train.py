"""
可通用代码——————训练代码
"""

import sys
import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:256,expandable_segments:True")

import rootutils
from pathlib import Path

# Setup Root, 设置根目录; rootutils, (module), 用于自动查找项目根目录的工具库
ROOT = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
Pocket_Plus_ROOT = Path(__file__).resolve().parent.parent  # Pocket_Plus/
if str(Pocket_Plus_ROOT) not in sys.path:
    sys.path.insert(0, str(Pocket_Plus_ROOT))

from src.utils.slurm_utils import (
    fix_gloo_socket_ifname as _fix_gloo_socket_ifname,
    log_distributed_launch_state as _log_distributed_launch_state,
)

# 统一使用 slurm_utils 中的网卡推导逻辑，避免本地旧实现与 sbatch helper 出现分叉。
_fix_gloo_socket_ifname()

import torch
import torch.multiprocessing
torch.set_float32_matmul_precision("high")
torch.multiprocessing.set_sharing_strategy('file_system')
import hydra
from omegaconf import DictConfig, OmegaConf
import lightning as pl
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor, RichProgressBar


from src.utils.wandb_utils import _setup_wandb_mode




from lightning.pytorch.loggers import WandbLogger
import wandb
# Setup Root and PYTHONPATH already moved to the top of the file
FEEDBACK_ROOT = Path(                # NOTE: 返回结果的存放目录由这里更改
    os.environ.get("EXPERIMENT_FEEDBACK_ROOT", str(ROOT / "feedback_plus"))
)
from src.utils.experiment_manager import ExperimentManager
from src.utils.fault_tolerant_dataset import maybe_wrap_dataset
# 模型、包装器和回调类现在将根据配置动态导入, 无需手动导入


def _get_config_name() -> str:
    """
    从命令行参数中解析 --config 参数，用于指定 Hydra 配置文件名。
    用法: python src/train.py --config baseline  或  python src/train.py --config=baseline
    默认值: "default"
    """
    import argparse
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", type=str, default="base",
                        help="Hydra config file name (without .yaml)")
    args, _ = parser.parse_known_args()
    # 从 sys.argv 中移除 --config 及其值，防止 Hydra 解析时报错
    cleaned = []
    skip_next = False
    for i, arg in enumerate(sys.argv):
        if skip_next:
            skip_next = False
            continue
        if arg == "--config":
            skip_next = True
            continue
        if arg.startswith("--config="):
            continue
        cleaned.append(arg)
    sys.argv = cleaned
    return args.config

_CONFIG_NAME = _get_config_name()


def _has_uninitialized_parameters(module: torch.nn.Module) -> bool:
    """Return True when any parameter is still lazy/uninitialized."""
    from torch.nn.parameter import UninitializedParameter

    for p in module.parameters():
        if isinstance(p, UninitializedParameter):
            return True
    return False


def _extract_input_tensor(sample):
    """Extract input tensor from dataset sample."""
    if torch.is_tensor(sample):
        return sample
    if isinstance(sample, dict):
        if "voxel_grid" in sample and torch.is_tensor(sample["voxel_grid"]):
            return sample["voxel_grid"]
        raise TypeError("Unsupported dict sample format; expected key 'voxel_grid' with Tensor value.")
    if isinstance(sample, (list, tuple)) and len(sample) > 0 and torch.is_tensor(sample[0]):
        return sample[0]
    raise TypeError(
        "Unsupported sample format; expected Tensor, dict['voxel_grid'], or tuple/list with Tensor at index 0."
    )


def _initialize_lazy_modules_before_ddp(model: torch.nn.Module, datamodule: pl.LightningDataModule, verbose: bool = True) -> None:
    """
    Materialize lazy parameters before Lightning wraps model with DDP.
    """
    if not _has_uninitialized_parameters(model):
        return

    datamodule.setup(stage="fit")
    if not hasattr(datamodule, "train_ds"):
        raise RuntimeError("Datamodule setup did not create train_ds; cannot initialize lazy modules.")

    sample = datamodule.train_ds[0]
    x = _extract_input_tensor(sample)
    if x.dim() == 4:
        in_channels = int(x.shape[0])  # C,D,H,W from dataset sample
    elif x.dim() == 5:
        in_channels = int(x.shape[1])  # B,C,D,H,W from pre-batched sample
    else:
        raise ValueError(f"Unexpected input tensor shape for lazy init: {tuple(x.shape)}")

    backbone = getattr(model, "backbone", None)
    maybe_compiled_backbone = getattr(backbone, "_orig_mod", backbone)
    if not hasattr(maybe_compiled_backbone, "set_input_channels"):
        raise RuntimeError(
            "Model has uninitialized parameters, but no set_input_channels() hook is available."
        )

    maybe_compiled_backbone.set_input_channels(in_channels)
    if verbose:
        print(f"[Train] Lazy backbone initialized before DDP: in_channels={in_channels}")

    if _has_uninitialized_parameters(model):
        raise RuntimeError(
            "Model still has uninitialized parameters after pre-DDP initialization."
        )




# Hydra 将所有子配置合并成一个大的 cfg 对象传入 main 函数: main(cfg)
def _get_eager_backbone(model: torch.nn.Module) -> torch.nn.Module | None:
    """Return the underlying backbone, unwrapping torch.compile if needed."""
    backbone = getattr(model, "backbone", None)
    if backbone is None:
        return None
    return getattr(backbone, "_orig_mod", backbone)


def _prepare_model_for_batch_size_tuning(model: torch.nn.Module, verbose: bool = True) -> dict[str, object]:
    """Make batch-size probing follow a worst-case recycle path."""
    state: dict[str, object] = {}
    backbone = _get_eager_backbone(model)
    if backbone is None:
        return state

    if hasattr(backbone, "randomize_recycles"):
        state["randomize_recycles"] = getattr(backbone, "randomize_recycles")
        if getattr(backbone, "randomize_recycles"):
            setattr(backbone, "randomize_recycles", False)
            if verbose:
                print("[Train] Batch size tuning: force deterministic max recycle passes for worst-case memory probing")
    return state


def _restore_model_after_batch_size_tuning(model: torch.nn.Module, state: dict[str, object]) -> None:
    """Restore model attributes mutated for batch-size tuning."""
    backbone = _get_eager_backbone(model)
    if backbone is None:
        return
    for attr_name, attr_value in state.items():
        setattr(backbone, attr_name, attr_value)


@hydra.main(version_base="1.3", config_path="../configs", config_name=_CONFIG_NAME)


def main(cfg: DictConfig):
    """
    主训练入口点。
    Args:
        - cfg: DictConfig, (Dict-like), Hydra 解析后的完整配置对象，包含 model, dataset, train 等所有参数
    Returns:
        - None
    """
    
    if os.environ.get("RANK") is not None:
        print(
            f"[Train] Dist Env: RANK={os.environ.get('RANK')}, "
            f"LOCAL_RANK={os.environ.get('LOCAL_RANK')}, "
            f"WORLD_SIZE={os.environ.get('WORLD_SIZE')}"
        )

    # 1. -------------- 初始化实验管理器, 创建目录并立即保存配置 --------------
    # ExperimentManager, (Object), 自定义的实验管理器实例，负责目录创建、配置备份和清理
    exp_manager = ExperimentManager(
        config=cfg,
        project_root=str(ROOT),
        feedback_root=str(FEEDBACK_ROOT),
        experiment_group=cfg.experiment_group,
    )
    # 更新 Pl Lightning 默认根目录为我们的新运行目录
    # str, (Path String), 实验反馈的保存路径, 格式: {feedback_root}/logs/{experiment_group}/{tag}____{timestamp}
    run_dir = str(exp_manager.run_dir)
    if exp_manager.is_rank_zero:
        os.makedirs(os.path.join(run_dir, "checkpoints"), exist_ok=True)
    
    

    # 2. -------------- 动态确定 Num Workers --------------
    req_workers = cfg.train.num_workers
    # 获取可用的CPU核心数
    try:
        avail_cores = len(os.sched_getaffinity(0))
    except AttributeError:
        avail_cores = os.cpu_count() or 1
    # 最终使用的Num Workers = min(requested, available)
    final_workers = min(req_workers, avail_cores)
    if exp_manager.is_rank_zero:
        print(f"[Train] Worker Config: Requested={req_workers}, Available={avail_cores}, Using={final_workers}")
    try:
        cfg.train.num_workers = final_workers
    except Exception as e:
        print(f"[Train] Could not update cfg.train.num_workers directly: {e}. Passing explicitly if possible.")



    # 3. -------------- 实例化 DataModule --------------
    if exp_manager.is_rank_zero:
        print(f"[Train] Instantiating DataModule with: {cfg.dataset.name}")
    class UnifiedDataModule(pl.LightningDataModule):
        """
        训练的统一数据模块。
        """
        def __init__(self, dataset_cfg, train_cfg):
            """
            Args:
                - dataset_cfg: DictConfig, (Dict-like), 数据集相关的配置 (configs/dataset/...)
                - train_cfg: DictConfig, (Dict-like), 训练相关的配置 (configs/train/...)
            """
            super().__init__() 
            # DictConfig, (Dict-like), 保存数据集配置
            self.dataset_cfg = dataset_cfg
            # DictConfig, (Dict-like), 保存训练配置
            self.train_cfg = train_cfg
            
        @property
        def batch_size(self):
            return self.train_cfg.batch_size
            
        @batch_size.setter
        def batch_size(self, value):
            self.train_cfg.batch_size = value
            
        def setup(self, stage=None):  
            """
            初始化训练dataset (self.train_ds) 和验证dataset (self.val_ds)
            Args:
                - stage: str, (Optional), 当前阶段 ('fit', 'validate', 'test', 'predict')
            #NOTE: 训练时初始化Dataset的逻辑见这里(半通用), 它部分地依赖于特定dataset的参数, 依赖关系如下:
                - 表示"所有数据的根目录"的变量名只能是 all_data_path (若训练/验证相统一) ;  all_data_path_train, all_data_path_val (若训练/验证相分离).
                - 表示"训练/验证集划分文件"的变量名只能是 split_file, 且在配置文件中必须分别用变量名 split_train, split_val 来指定训练集和验证集的划分文件.
                - 在Dataset的初始化时, 必须含有变量名"mode", 它必须可以接受"train"与"val"(虽然可以通过内部转义扩大范围), 表示当前数据集用于训练或验证.
            """
            if stage == "fit" or stage is None:
                # 以下的两行代码允许了训练数据集和验证数据集的根目录可以不同, 它们通过"all_data_path_train"和"all_data_path_val"指定
                _train_data_path = self.dataset_cfg.get(   # 先找 "all_data_path_train" 再找 "all_data_path", 最后None
                    "all_data_path_train",
                    self.dataset_cfg.get("all_data_path", None)
                )
                _val_data_path = self.dataset_cfg.get(   # 先找 "all_data_path_val" 再找 "all_data_path", 最后None
                    "all_data_path_val",
                    self.dataset_cfg.get("all_data_path", None)
                )

                # 通过 Hydra 实例化训练数据集
                self.train_ds = hydra.utils.instantiate(
                    self.dataset_cfg,
                    split_file=self.dataset_cfg.split_train,
                    all_data_path=_train_data_path,
                    mode="train"
                )
                # 通过 Hydra 实例化验证数据集
                self.val_ds = hydra.utils.instantiate(
                    self.dataset_cfg,
                    split_file=self.dataset_cfg.split_val,
                    all_data_path=_val_data_path,
                    mode="val"
                )

                # --- 容错包装: 坏样本自动替换, 防止 DDP 卡死 ---
                _ft_cfg = self.train_cfg.get("data_fault_tolerance", None)
                if _ft_cfg is not None:
                    self.train_ds = maybe_wrap_dataset(self.train_ds, "train", _ft_cfg)
                    self.val_ds = maybe_wrap_dataset(self.val_ds, "val", _ft_cfg)

        def _get_dataloader(self, ds, shuffle=False):
            """
            统一 DataLoader 的创建逻辑。
            支持根据样本类型自动选择 torch_geometric 或 torch.utils.data.DataLoader
            """
            backend = self.train_cfg.dataloader_backend
            
            # 探测第一个样本
            use_pyg = False
            if backend.lower() == "pyg":
                use_pyg = True
            elif backend.lower() == "torch":
                use_pyg = False
            else: # "auto"
                try:
                    import torch_geometric
                    sample = ds[0]
                    from torch_geometric.data import Data, Batch
                    if isinstance(sample, (Data, Batch)):
                        use_pyg = True
                except ImportError:
                    use_pyg = False
            
            if use_pyg:
                import torch_geometric.loader
                loader_class = torch_geometric.loader.DataLoader
            else:
                loader_class = torch.utils.data.DataLoader

            collate_fn = None if use_pyg else getattr(ds, "collate_fn", None)
                
            return loader_class(
                ds,
                batch_size=self.train_cfg.batch_size,
                shuffle=shuffle,
                num_workers=self.train_cfg.num_workers,
                pin_memory=self.train_cfg.get("pin_memory", True) if sys.platform != "win32" else False,
                collate_fn=collate_fn,
            )

        def train_dataloader(self):
            """
            Returns:
                - DataLoader, (torch.utils.data.DataLoader), 训练数据加载器
            """
            return self._get_dataloader(self.train_ds, shuffle=True)
            
        def val_dataloader(self):
            """
            Returns:
                - DataLoader, (torch.utils.data.DataLoader), 验证数据加载器
            """
            return self._get_dataloader(self.val_ds, shuffle=False)
    
    dm = UnifiedDataModule(cfg.dataset, cfg.train)



    # 4. -------------- 实例化模型 --------------
    if exp_manager.is_rank_zero:
        print(f"[Train] 实例化(Instantiate)模型的名字: {cfg.model.name}")
        print(f"[Train] 实例化(Instantiate)模型的路径: {cfg.model._target_}")
    global_batch_size = cfg.train.get("global_batch_size", None)
    batch_size_tuning_enabled = global_batch_size is not None and global_batch_size > 0
    compile_requested = bool(cfg.model.get("compile", False))
    compile_deferred = bool(compile_requested and batch_size_tuning_enabled)
    if compile_deferred and exp_manager.is_rank_zero:
        print("[Train] 检测到自动 Batch Size 探测已启用; 暂缓 torch.compile, 待 tuner 完成后再编译 backbone")

    model = hydra.utils.instantiate(
        cfg.model,
        optimizer=cfg.train.optimizer,
        scheduler=cfg.train.scheduler,
        compile=False if compile_deferred else compile_requested,
    )
    _initialize_lazy_modules_before_ddp(model, dm, verbose=exp_manager.is_rank_zero)
    _fix_gloo_socket_ifname()
    _log_distributed_launch_state("LazyInit完成")



    # 5. -------------- 日志记录器 --------------
    prefer_online = not cfg.offline
    wandb_mode = _setup_wandb_mode(prefer_online=prefer_online, verbose=exp_manager.is_rank_zero)
    is_offline = (wandb_mode == "offline")
    
    logger = WandbLogger(
        project=cfg.project_name,
        name=f"{cfg.model.name}-{cfg.dataset.name}-{cfg.tag}", 
        save_dir=run_dir,
        offline=is_offline,
        log_model=False
    )

    # ---- WandB 在线模式真实连通性兜底 ----
    # WandbLogger 构造时不会调用 wandb.init(), 而是延迟到 trainer.fit() 内部
    # 首次访问 logger.experiment 时才触发。如果此时超时, 会直接崩溃且无法回退。
    # 因此在此处主动提前触发 wandb.init(), 捕获任何异常后自动回退为离线模式。
    # 注意: 由于此时 Lightning 尚未初始化分布式环境，只有真实 Rank 0 才能访问 logger.experiment，否则会导致多进程并发写 WandB 发生死锁。
    if not is_offline and exp_manager.is_rank_zero:
        try:
            _ = logger.experiment          # 触发 wandb.init()
            print("[Train] [OK] WandB 在线初始化成功 (wandb.init() succeeded)")
        except Exception as _wandb_err:
            print(f"[Train] [X] WandB 在线初始化失败: {_wandb_err}")
            print("[Train]      自动回退到离线模式 (Falling back to offline mode)...")
            # 尝试清理失败的 wandb run
            try:
                wandb.finish()
            except Exception:
                pass
            os.environ["WANDB_MODE"] = "offline"
            is_offline = True
            
    # 如果 Rank 0 决定降级为离线模式，在此重建 logger。
    # 其他进程的 logger 可能仍是在线模式对象，但在其他进程仅是 Dummy，不影响训练。
    if is_offline and exp_manager.is_rank_zero:
        logger = WandbLogger(
            project=cfg.project_name,
            name=f"{cfg.model.name}-{cfg.dataset.name}-{cfg.tag}",
            save_dir=run_dir,
            offline=True,
            log_model=False
        )
        print(f"[Train]   WandB 已启用离线模式 (Offline Mode)。")
        print(f"[Train]   训练将在本地生成离线日志: {run_dir}/wandb/")
        print("[Train]   训练结束后，请使用 'wandb sync' 或配套的同步脚本上传到云端。")



    # 6. -------------- 模型检查点回调(Callback), 用于保存最佳模型 --------------
    # monitor 和 monitor_mode 统一从 cfg.model 读取，与 Wrapper 中 self.log() 和 configure_optimizers() 使用的 monitor_metric 保持一致
    _monitor = cfg.model.monitor_metric
    _monitor_mode = cfg.model.monitor_mode
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(run_dir, "checkpoints"),    # str, save path
        filename="TOP_epoch_{epoch:02d}_score_{" + _monitor + ":.4f}", # str, 带有 TOP_ 前缀的文件名
        auto_insert_metric_name=False,                   # bool, 关闭指标自动拼接，防止基于 '/' 创建意外的子文件夹
        monitor=_monitor,
        mode=_monitor_mode,                              # str, 'min' or 'max'
        save_top_k=cfg.output.save_top_k,                # int, number of models to save
        save_last=True,                                  # bool, save last epoch
    )
    # 建立最初的回调列表
    callbacks = [checkpoint_callback]

    # ------ 额外开启周期性保存 ------
    # 从配置中获取 save_every_n_epochs (例如: 10)
    save_every_n_epochs = cfg.output.get("save_every_n_epochs", None)
    if save_every_n_epochs is not None and save_every_n_epochs > 0:
        periodic_checkpoint = ModelCheckpoint(
            dirpath=os.path.join(run_dir, "checkpoints"),
            filename="PERIODIC_epoch_{epoch:02d}",       # str, 带有 PERIODIC_ 前缀的文件名
            auto_insert_metric_name=False,               # bool, 对于无监控指标的保存机制，关闭指标自动拼接
            every_n_epochs=save_every_n_epochs,          # int, 每隔这么多个 epoch 保存一次
            save_top_k=-1,                               # int, 为 -1 时永久保留每个由此机制产出的周期性模型
            save_last=False,                             # bool, 上面的 checkpoint_callback 已经接手保存 last，避免冲突
        )
        callbacks.append(periodic_checkpoint)

    # LearningRateMonitor, (Callback), 学习率监控
    lr_monitor = LearningRateMonitor(logging_interval="step")
    # RichProgressBar, (Callback), 终端进度条
    rich_bar = RichProgressBar()
    
    # 将进度条与学习率监控加入 list
    callbacks.extend([lr_monitor, rich_bar])




    # '训练时测试' 这个功能暂时关闭
    # # 7. -------------- 周期性测试回调(支持多个测试集、多种回调机制) --------------(本条目暂略)
    # # dict, (Dict[str, DataLoader]), 存储测试集 DataLoader 的字典
    # test_dataloaders = {}
    # # 从 cfg.dataset.split_test 获取测试集配置
    # if "split_test" in cfg.dataset and cfg.dataset.split_test is not None:

    #     # 这要求 cfg.dataset.split_test是字典或列表(都兼容), 比如 {'test_set_name1': 'path/to/test1.json', 'test_set_name2': 'path/to/test2.json', ...}
    #     splits = OmegaConf.to_container(cfg.dataset.split_test, resolve=True)
    #     if isinstance(splits, (list, tuple)):
    #         splits = {f"test_{i}": path for i, path in enumerate(splits)}
    #     for name, split_path in splits.items():    
    #         if os.path.exists(split_path):
    #             if exp_manager.is_rank_zero:
    #                 print(f"[Train] Loading Test Split for Callback.训练时测试,测试集名字是: {name}、划分路径是 {split_path}")
    #             # 通过 Hydra 实例化测试数据集 (路径由 dataset.all_data_path_test 显式指定)
    #             ds = hydra.utils.instantiate(
    #                 cfg.dataset,
    #                 split_file=split_path,
    #                 all_data_path=cfg.dataset.all_data_path_test,
    #                 mode="test"
    #             )
    #             dl = dm._get_dataloader(ds, shuffle=False)
    #             test_dataloaders[name] = dl
    #         else:
    #             if exp_manager.is_rank_zero:
    #                 print(f"[Train] 警告：测试集{name}或者划分文件未找到：{split_path}。请检查路径。")
    # else:
    #     if exp_manager.is_rank_zero:
    #         print("[Train] No 'split_test' found in dataset config. Periodic testing disabled.")
        

    # # 暂且不实现 periodic_test.py，但在此留下接口并说明。
    # # 回调类(训练时测试)
    # # 实例化类是 src.callbacks.periodic_test.PeriodicTestCallback ,只需传入通用参数 dataloaders_dict=test_dataloaders (如前所述的字典) 和 interval (间隔的epoch)
    # if len(test_dataloaders) > 0:
    #     # 此回调的作用是在训练过程中（特定epoch间隔）调用测试流程，以监控全集或其它测试集的表现。
    #     # 未来的实现应大部分调用推断程序 (src/infer.py) 中的代码逻辑（例如导入相关推断函数），
    #     # 从而避免在训练代码中重新写一遍推断逻辑并确保二者一致性。
    #     #
    #     # 下面为原本的调用接口，当前已注释：

    #     # callback_class = hydra.utils.get_class("src.callbacks.periodic_test.PeriodicTestCallback")
    #     # periodic_eval_cb = callback_class(
    #     #     dataloaders_dict=test_dataloaders,
    #     #     interval=cfg.train.get("test_interval", 50)
    #     # )
    #     # callbacks.append(periodic_eval_cb)
    #     if exp_manager.is_rank_zero:
    #         print("[Train] TODO: callbacks.periodic_test 暂未实现。正在按要求显式跳过周期测试回调。")


    # # 可视化回调(暂略)
    # # 配置中支持 cfg.output.visualization 这个条目, 开关为 cfg.output.visualization.enabled
    # if cfg.output.get("visualization", {}).get("enabled", False):
    #     from src.callbacks.visualization_callback import VisualizationCallback
    #     vis_callback = VisualizationCallback(
    #         run_dir=run_dir,
    #         vis_config=cfg.output.visualization,
    #         dataset_config=cfg.dataset,
    #         model_path=cfg.model._target_
    #     )
    #     callbacks.append(vis_callback)
    #     if exp_manager.is_rank_zero:
    #         print(f"[Train] VisualizationCallback enabled, interval_plot={cfg.output.visualization.get('interval_plot', 300)}")







    # 8. -------------- 训练器 --------------
    use_distributed = (cfg.train.devices > 1 or cfg.train.nnodes > 1)
    find_unused = bool(cfg.train.get("ddp_find_unused_parameters", False))
    strategy = "auto"
    if use_distributed:
        from lightning.pytorch.strategies import DDPStrategy
        strategy = DDPStrategy(
            find_unused_parameters=find_unused,
            process_group_backend="nccl"       # 强制使用 NCCL 后端, 避免 Gloo 辅助进程组导致的网络接口问题
        )

    trainer = pl.Trainer(
        default_root_dir=run_dir, # str, save path
        accelerator=cfg.train.accelerator,                # str, hardware accelerator
        devices=cfg.train.devices,                        # int, number of GPUs
        num_nodes=cfg.train.nnodes,                       # int, number of nodes
        strategy=strategy,                                # str, distributed strategy
        max_epochs=cfg.train.max_epochs,                  # int, max epochs
        logger=logger,                                    # Logger
        callbacks=callbacks,                              # List[Callback]
        precision=cfg.train.precision,                    # str, mixed precision setting
        gradient_clip_val=cfg.train.gradient_clip_val,    # float, gradient clipping
        accumulate_grad_batches=cfg.train.get("accumulate_grad_batches", 1), # int, 默认 1
        check_val_every_n_epoch=cfg.train.check_val_every_n_epoch, # int
        log_every_n_steps=cfg.train.get("log_every_n_steps", 10), # int, 控制wandb记录日志的频率
        num_sanity_val_steps=2 # int, 用于检查bug
    )
    



    # -------------- 8.5 自动推导全局 Batch Size 与 Accumulate Steps --------------
    if global_batch_size is not None and global_batch_size > 0:
        if exp_manager.is_rank_zero:
            print(f"[Train] 开始自动探测显存，寻找最佳 Batch Size (目标 Global Batch = {global_batch_size})...")
        
        from lightning.pytorch.tuner import Tuner
        tuner = Tuner(trainer)
        init_val = cfg.train.get("tuning_init_batch_size", 2)
        tuning_state = _prepare_model_for_batch_size_tuning(model, verbose=exp_manager.is_rank_zero)
        try:
            # Search maximum batch size that fits in memory, with exact integer precision
            tuner.scale_batch_size(
                model,
                datamodule=dm,
                mode="binsearch",
                init_val=init_val,
                max_trials=25
            )
        finally:
            _restore_model_after_batch_size_tuning(model, tuning_state)

        # Tuner only samples a few batches; leave headroom for heavier real batches.
        found_bs = int(dm.batch_size)
        safety_factor = float(cfg.train.get("batch_size_tuning_safety_factor", 0.8))
        safe_bs = max(1, int(found_bs * safety_factor))
        if safe_bs > found_bs:
            safe_bs = found_bs
        if found_bs > 1 and safe_bs == found_bs and safety_factor < 1.0:
            safe_bs = found_bs - 1
        dm.batch_size = safe_bs
        try:
            cfg.train.batch_size = safe_bs
        except Exception:
            pass

        # Calculate actual accumulate_grad_batches based on the safe batch size
        num_devices = trainer.world_size
        accumulate_steps = max(1, global_batch_size // (safe_bs * num_devices))
        
        # Dynamically modify trainer's runtime property
        trainer.accumulate_grad_batches = accumulate_steps

        if exp_manager.is_rank_zero:
            print(f"[Train] => Target Global Batch Size: {global_batch_size}")
            print(f"[Train] => Initial per-device guess: {init_val}")
            print(f"[Train] => Tuned per-device batch size: {found_bs}")
            print(f"[Train] => Safe per-device batch size: {safe_bs} (safety_factor={safety_factor:.2f})")
            print(f"[Train] => Accumulate Steps: {accumulate_steps}")
            print(f"[Train] => world_size: {num_devices}")
            print(f"[Train] => Effective Global Batch Size: {safe_bs * num_devices * accumulate_steps}")

        if False and exp_manager.is_rank_zero:
            print(f"[Train] => 目标 Global Batch Size: {global_batch_size}")
            print(f"[Train] => 初始猜测单卡 Batch Size: {init_val}")
            print(f"[Train] => 实际探测最佳单卡 Batch Size: {found_bs}")
            print(f"[Train] => 计算得 Accumulate Steps: {accumulate_steps}")
            print(f"[Train] => 总并行硬件数 (world_size): {num_devices}")
            print(f"[Train] => 实际等效 Global Batch Size: {found_bs * num_devices * accumulate_steps}")

    if compile_deferred:
        if exp_manager.is_rank_zero:
            print("[Train] 自动 Batch Size 探测完成, 开始执行 torch.compile(backbone)...")
        model.backbone = torch.compile(model.backbone)




    # 9. -------------- 开始训练 --------------
    if exp_manager.is_rank_zero:
        print("============================================================")
        print(f" Starting Training")
        print(f" Model: {cfg.model.name}")
        print(f" Dataset: {cfg.dataset.name}")
        print(f" PostProcess: {cfg.post_process.name}")
        print(f" Output Dir: {run_dir}")
        print("============================================================")
    try:
        # trainer.fit 触发整个 Lightning 生命周期：
        # 1. dm.setup("fit") → 创建 train/val 数据集
        # 2. model.configure_optimizers()（Wrapper L194–L246） → 从 self.hparams.optimizer/scheduler（来自 cfg.train）创建优化器和调度器，调度器的 monitor/interval/frequency 来自 self.hparams（来自 cfg.model）
        # 3. 每个 batch：model.training_step() →  _extract_batch → forward → _compute_loss → log("train/loss")
        # 4. 每个 epoch 末：model.validation_step() → 计算 PR-AUC → log("val/score") → ModelCheckpoint 检查是否保存
        _fix_gloo_socket_ifname()
        _log_distributed_launch_state("即将调用trainer.fit")
        trainer.fit(model, datamodule=dm)
    except Exception as e:
        print(f"[Train] Critical Exception occurred(严重异常): {e}")
        exp_manager.check_and_cleanup(error=e)
        raise e
    finally:
        # 如果没有传递 error，check_and_cleanup 视为正常退出 (会检查是否太短)
        exp_manager.check_and_cleanup()

if __name__ == "__main__":
    main()
