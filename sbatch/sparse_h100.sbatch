#!/bin/bash
#SBATCH -o /home/penghongen/My_Project/feedback_plus/logs/_temp_slurm/%x_%j.out
#SBATCH -e /home/penghongen/My_Project/feedback_plus/logs/_temp_slurm/%x_%j.err
#SBATCH -p h100
#SBATCH --qos=h100g2
#SBATCH --job-name=sparse

# ==========================================================================================
# [USER MANUAL / 用户手册]
# 本脚本用于训练 Pocket_Plus 项目 (RAUNet + BinaryFocalLoss) 的 baseline 配置
# This script trains the Pocket_Plus project with baseline config (RAUNet backbone)
#
# 基于 run_unified.sbatch，保留三锁机制
# Based on run_unified.sbatch, preserves three-lock mechanism
#
# Examples / 示例:
#   1. Standard High-Performance: 4 Nodes, 4 GPUs (1 per node)
#      #SBATCH --nodes=4
#      #SBATCH --gres=gpu:a100:1
#
#   2. High-Density: 1 Node, 4 GPUs
#      #SBATCH --nodes=1
#      #SBATCH --gres=gpu:a100:4
# ==========================================================================================

# --- [Resources / 资源申请] ---
#SBATCH --nodes=1                                      # Num Nodes (节点数)
#SBATCH --gres=gpu:h100:1                              # Num GPUs per node (每节点 GPU 数)
#SBATCH --ntasks-per-node=1                            # 双卡时需与 GPUS_PER_NODE / Lightning devices 保持一致
#SBATCH --cpus-per-task=24                             # 16 CPUs per GPU (总计 32 CPUs per node)

# SLURM中 sbatch 会将脚本拷贝到 /var/spool/slurmd 临时目录导致 BASH_SOURCE[0] 原路径失效
# 直接使用项目中的绝对路径来 source
export Pocket_Plus_DIST_HELPER_PATH="/home/penghongen/My_Project/Pocket_Plus/sbatch/_slurm_net_common.sh"
source "$Pocket_Plus_DIST_HELPER_PATH"

# ========================================== 1. 环境准备区 ==========================================
Pocket_Plus_dist_init_env || { echo "[Dist] 初始化通信环境失败（也可能是 Slurm step 启动失败，请查看 stderr）"; exit 1; }

echo "Job $SLURM_JOB_ID allocated on nodes: $SLURM_JOB_NODELIST"
echo "Master Node: $MASTER_NODE"
echo "Master Address: $MASTER_ADDR, Port: $MASTER_PORT, Dist IF: $SLURM_NET_DIST_IFNAME"

export HYDRA_FULL_ERROR=1
export OMP_NUM_THREADS=1
export NCCL_DEBUG=INFO
export TORCH_DISTRIBUTED_DEBUG=${TORCH_DISTRIBUTED_DEBUG:-OFF}
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

source /home/penghongen/anaconda3/bin/activate Pocket_Plus

# ========================================== Path Configuration / 路径配置 ==========================================
# PROJECT_ROOT 指向 .project-root 所在的 My_Project/ (即 rootutils 找到的 ROOT)
# train.py 内部会额外将 Pocket_Plus/ 加入 sys.path
export PROJECT_ROOT="/home/penghongen/My_Project"
cd $PROJECT_ROOT || { echo "Error: Could not cd to $PROJECT_ROOT"; exit 1; }

echo "[Path] Switched to Project Root: $(pwd)"
export PYTHONPATH=$PROJECT_ROOT:$PROJECT_ROOT/Pocket_Plus:$PYTHONPATH

# ========================================== Auto-Detection / 自动检测区 ==========================================
GPUS_PER_NODE=$(Pocket_Plus_dist_detect_gpus_per_node) || {
    echo "[Auto-Detect] Failed to detect GPUs per node from SLURM env or nvidia-smi"
    exit 1
}
export GPUS_PER_NODE

echo "================================================================================="
echo "[Auto-Detect] Detected Resources per Node:"
echo "   -> GPUS_PER_NODE = $GPUS_PER_NODE"
echo "================================================================================="

# ========================================== 2. 锁机制 ==========================================
# 创建前向锁 (可选，手动删除后才开始训练)
LOCK_PRE="/home/penghongen/pre_lock_${SLURM_JOB_ID}"
# touch $LOCK_PRE
echo "[Status] pre_Lock created: $LOCK_PRE"
echo "[Wait] Waiting for user to delete $LOCK_PRE to start..."

# 创建后向锁
LOCK_AFTER="/home/penghongen/after_lock_${SLURM_JOB_ID}"
touch $LOCK_AFTER
echo "[Status] after_Lock created: $LOCK_AFTER"

# 定义重试锁
LOCK_TRY="/home/penghongen/try_lock_${SLURM_JOB_ID}"

# ========================================== 3. "门神" 锁机制 (Pre-Run) ==========================================
# 取消注释以下代码以启用前向锁等待
# Uncomment to enable pre-lock waiting

while [ -f "$LOCK_PRE" ];
do
    sleep 20
done
echo "[Start] Starting Distributed Training..."

# ========================================== 4. 容错运行区 ==========================================
while true; do
    Pocket_Plus_dist_refresh_run_stamp
    echo "[Run] Shared run stamp: $Pocket_Plus_RUN_STAMP"

    echo "================================================================================="
    echo "[Attempt] Starting Pocket_Plus baseline Training (RAUNet)..."
    echo "================================================================================="
    echo "[Preflight] Checking distributed network..."

    # NOTE: 在这里改文件与配置：Pocket_Plus/src/train.py --config=half_3_h100
    # train.py 通过 --config 参数选择配置文件 (非标准 Hydra 参数，由 _get_config_name() 解析)
    # 为了稳定 TCPStore 初始化连接，加入一个节点间的错峰等待
    if Pocket_Plus_dist_preflight && srun --label --kill-on-bad-exit=1 --ntasks-per-node=$GPUS_PER_NODE --export=ALL bash -c "
        sleep \$((SLURM_NODEID * 5));
        python Pocket_Plus/src/train.py \
            --config=sparse_h100 \
            train.nnodes=$SLURM_JOB_NUM_NODES \
            train.devices=$GPUS_PER_NODE;
    "; then
        # ------------- Success / 成功 -------------
        echo "================================================================================="
        echo "[Success] Training finished successfully."
        echo "================================================================================="
        rm -f $LOCK_AFTER
        rm -f $LOCK_TRY
        break

    else
        # ------------- Failure / 失败 -------------
        echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
        echo "[Error] Training FAILED! Check error logs above."
        echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"

        touch $LOCK_TRY

        echo "[Action Required] Job is PAUSED. "
        echo "   -> To RETRY: Delete file '$LOCK_TRY'"
        echo "   -> To CANCEL: Delete file '$LOCK_AFTER'"

        while [ -f "$LOCK_TRY" ] && [ -f "$LOCK_AFTER" ]; do
            sleep 20
        done

        if [ ! -f "$LOCK_AFTER" ]; then
            echo "[Stop] '$LOCK_AFTER' was deleted by user. Exiting loop."
            break
        else
            echo "[Retry] '$LOCK_TRY' was deleted by user. Restarting task immediately..."
            continue
        fi
    fi

done

# ========================================== 5. "不死鸟" 兜底机制 ==========================================
while [ -f "$LOCK_AFTER" ];
do
    sleep 20
done

echo "[Exit] Job finished."
