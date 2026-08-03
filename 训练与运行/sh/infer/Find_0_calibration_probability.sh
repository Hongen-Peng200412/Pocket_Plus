#!/usr/bin/env bash

# 本文件只定义“得到一张 GPU 后怎样处理一个 calibration 分片”，不直接申请 GPU。
# 正式申请由项目根目录的 `训练与运行/submit_task.sh` 完成，例如：
#
# bash /home/penghongen/My_Project/Pocket_Plus/训练与运行/submit_task.sh \
#   --sh /home/penghongen/My_Project/Pocket_Plus/训练与运行/sh/infer/Find_0_calibration_probability.sh \
#   --resource a100 \
#   --gpus 1 \
#   --cpus 8 \
#   --array '0-1' \
#   --after_hold \
#   --job-name find0_cal_prob
#
# 上述命令只调用一次 sbatch，但建立两个数组元素：编号 0 和 1 各申请一张 GPU。
# 每个数组元素申请 1 张 GPU 和 8 个 CPU 核；提交时将 `<gpu_resource>` 替换为
# 当时可用的资源名（例如 a800、a100、h100 或 h200）。Slurm 为每个数组元素
# 设置不同的 SLURM_ARRAY_TASK_ID, 本脚本把该编号直接作为全局分片编号。
#
# 正式模式会在每次真正执行前冻结项目 release，并由 allocation_runner.sh 管理四锁：
# `pre_lock` 只在提交时增加 --pre_hold 才出现；本示例的 --after_hold 会在执行结束后创建 `try_lock`；
# `kill_lock` 只终止当前命令而保留 allocation；删除 `after_lock` 才结束 Job 并释放资源。
# 本脚本不创建、删除或解释这些锁，只负责一次 run_cmd 中的推理工作。
# 人工核对两个分片的产物后，分别删除两个数组元素的 after_lock，才会结束 Job 并释放两张卡。
# 若提交时省略 --after_hold，每个数组元素完成一次执行后会自动结束并释放自己的 GPU。

# 任一命令失败立即退出；读取未定义变量时报错；管道中任一命令失败都令整个管道失败。
# Python 推理失败会成为本次 run_cmd 的非零退出码；只有提交时使用 --after_hold 才会随后进入 try_lock。
set -euo pipefail

# 完整模式下，本脚本位于本次 release 的 `训练与运行/sh/infer/`；simple 模式下则位于当前项目。
# 两种模式都通过脚本自身位置向上三级得到实际执行的 Pocket_Plus 项目根目录。
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd -P)"

# 允许服务器通过环境变量替换 Conda 安装位置或环境名；没有覆盖时使用项目正式 Linux 环境。
CONDA_BASE="${CONDA_BASE:-${HOME}/anaconda3}"
CONDA_ENV_NAME="${POCKET_CONDA_ENV:-Pocket_Plus_centos7_cu121_allgpu}"

# Conda 激活脚本可能读取尚未定义的 shell 变量，因此只在激活期间暂时关闭 nounset。
set +u
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_NAME}"
set -u

# 让 Python 优先导入本次 release 中的 src；无缓冲输出使 allocation 日志及时显示当前 PDB 和异常。
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

# 限制常见数值库各自只建立一个 CPU 线程，避免同一节点上的多个数组元素重复扩张线程池。
# `--cpus 8` 是 Slurm 分给整个数组元素的 CPU 配额；这里的 1 是每个底层数值库线程池的上限。
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# 减少 PyTorch CUDA 缓存分配器的大块碎片；服务器可在提交环境中显式覆盖该值。
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:256}"

# 模型来源保持指向已经完成的 Find_0 CPC1 运行，不复制或改写 checkpoint。
run_root="/home/penghongen/My_Project/feedback_plus/logs/AdaLigand_Stage1-Find_0-CPC1/Find_0-CPC1____job321743_Find_0_CPC1_lr5e5_p2_val30_chunk2x_2gpu_m8_w1"
checkpoint="${run_root}/checkpoints/TOP_epoch_00_score_0.2843.ckpt"  # 现存 checkpoint 中配体区域 PR-AUC 最高的一份。
config="${run_root}/config.yaml"                                     # 该训练运行真正落盘且可解析的完整配置。

# inference_root 保存所有模型共用的 PDB 清单；formal_root 只属于当前 Find_0 checkpoint 的正式推理。
data_root="/storage/penghongen/AdaLigand/Ori_Data"                    # A—G 正式密度、受体和标签产物根目录。
inference_root="/storage/penghongen/AdaLigand_stage1_inference"
formal_root="${inference_root}/Find_0-CPC1-ligand_PRAUC_0.675477"

# prepare_inputs 从已经过滤的 calibration 集合提取有序、无重复的 100 个 PDB 名称。
# 该公共清单由所有 Stage1 模型复用；清单顺序是分片归属的一部分，正式运行期间不能更换。
pdb_list="${inference_root}/calibration_pdb_ids.json"

# 每个 PDB 的完整图概率写入 `${output_root}/Find_0/calibration/<pdb_id>/`：
# `probability/probability_map.npz` 保存 float32 (D,H,W) 的 ZYX 完整图概率及世界坐标几何；
# `probability/geometry.json` 保存滑窗与融合参数；两个文件成功发布后才写 `status/probability/_COMPLETE`。
output_root="${formal_root}/artifacts"

# Python 使用 `pdb_ids[shard_index::global_shard_count]` 分片，也就是按清单位置取模：
# 分片 0 处理清单位置 0、2、4……，分片 1 处理 1、3、5……。
# 两个分片互不重复，合起来恰好覆盖 100 个 PDB；每个分片处理 50 个 PDB。
global_shard_count=2                                                   # calibration 概率图、F1 和 CLG 始终使用相同的两份归属。

# Slurm 数组运行时编号由 `--array` 决定；不通过数组直接执行本脚本时默认编号为 0，
# 因而只会处理第 0 分片，不会自动继续处理第 1 分片。
shard_index="${SLURM_ARRAY_TASK_ID:-0}"                               # 0-based，必须满足 0 <= shard_index < global_shard_count。

# 每次模型前向同时处理 8 个 80³ 滑窗。它影响单个 PDB 的速度和显存，不改变 PDB 分片。
window_batch_size=6                                                    # 当前 smoke 验证的保守批量；正式提交时按可用显存调整。

# 单进程 Dataset 缓存最多保留约 100 GiB 的已读取内容，但不会启动时一次性分配 100 GiB。
# 这不是 Slurm 的 `--mem` 申请；多个数组元素落在同一节点时，各自拥有独立缓存上限。
cache_max_bytes=107374182400                                           # 100 × 1024³ bytes。

# 只检查推理命令实际读取的三个文件；manifest.json 仅供人查看，不作为运行门槛。
[[ -f "${checkpoint}" && -f "${config}" && -f "${pdb_list}" ]] || { echo "正式输入文件不完整" >&2; exit 2; }

# 从本次 release 的项目根目录运行统一推理入口，确保相对导入与 checkpoint 快照激活路径稳定。
# 参数含义：
# - `--producer`: 产物路径中的模型来源名，也是 checkpoint 装配时的模型契约。
# - `--pdb-list`: 有序冻结清单；CLI 会拒绝空 identity 和重复 identity。
# - `--shard-index`: 当前数组元素只选择属于该编号的 PDB。
# - `--shard-count`: 所有数组元素使用相同总数，才能互斥且完整覆盖清单。
# - `--data-root`: 按选中的 PDB identity 读取 A—G 完整图输入。
# - `--checkpoint`: 严格恢复 Find_0 模型参数，并使用 checkpoint 中的源码快照定义。
# - `--config`: 恢复该训练运行的数据通道、模型和推理装配配置。
# - `--device cuda:0`: 每个数组元素只获一张可见 GPU；Slurm 重映射后它在进程内就是 cuda:0。
# - `--window-batch-size`: 一个模型前向中的 80³ BOX 数量，不是 PDB 数量。
# - `--cache-max-bytes`: 当前进程的 Dataset 缓存上限，不是共享缓存或 Slurm 内存申请。
# - `--output-root`: 各分片写同一根目录，但因 PDB 归属互斥而不会写同一 PDB。
cd "${PROJECT_ROOT}"
python -u -m src.inference.cli cal-probability \
    --producer Find_0 \
    --pdb-list "${pdb_list}" \
    --shard-index "${shard_index}" \
    --shard-count "${global_shard_count}" \
    --data-root "${data_root}" \
    --checkpoint "${checkpoint}" \
    --config "${config}" \
    --device cuda:0 \
    --window-batch-size "${window_batch_size}" \
    --cache-max-bytes "${cache_max_bytes}" \
    --output-root "${output_root}"
