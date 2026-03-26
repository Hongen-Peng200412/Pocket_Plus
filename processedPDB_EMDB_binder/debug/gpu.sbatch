#!/bin/bash
#SBATCH -o /home/penghongen/Out/NEW_final.txt    # %j 会自动替换为 JOBID
#SBATCH -e /home/penghongen/Error/NEW_final.txt
#SBATCH -p a100
#SBATCH --qos=a100g2                                # 依然使用高优先级的QoS
#SBATCH -J FT

# --- [排队核心] 若拆分为双节点 ---
#SBATCH -N 1                                            #  申请 1 个节点
#SBATCH --gres=gpu:a100:1                               #  每个节点 1 张卡 (容易排队!)
#SBATCH --ntasks-per-node=1                             #  每个节点启动 1 个 task (由 torchrun 接管)
#SBATCH --cpus-per-task=16                              #  每个task16核 (注意资源分配逻辑为：Node->task->卡[CPU]，每个task2个卡，一个卡16核，总共16x1=16核)

# ======================================================= 1. 环境准备区 ========================================================
echo "Job $SLURM_JOB_ID allocated on 1 Node with 16 Cores."

# [CPU 满载优化] 
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

source /home/penghongen/anaconda3/bin/activate vnegnn
# ======================================================= 1. 环境准备区结束 =====================================================


# ======================================================== 2. 锁文件定义 ========================================================
LOCK_PRE="/home/penghongen/pre_lock_cpu_${SLURM_JOB_ID}"
# touch $LOCK_PRE
echo "[Status] pre_Lock created: $LOCK_PRE"

LOCK_AFTER="/home/penghongen/after_lock_cpu_${SLURM_JOB_ID}"
touch $LOCK_AFTER
echo "[Status] after_Lock created: $LOCK_AFTER"
# ======================================================== 2. 锁文件定义结束 =====================================================


# ======================================================== 3. “门神” 锁机制 =====================================================
echo "[Wait] Waiting for user to delete $LOCK_PRE to start..."
while [ -f "$LOCK_PRE" ]; do
    sleep 60
done
echo "[Start] Lock released! Starting Calculation..."
# ======================================================== 3. “门神” 锁机制结束 =================================================


# ============================================================== 4. 运行区 =======================================================
if srun python /home/penghongen/My_Project/Pocket/check.py --n_jobs 16; then
    
    echo "[Success] CPU Task finished successfully."
    rm -f $LOCK_AFTER

else
    echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
    echo "[Error] Task FAILED! Check error logs above."
    echo "[Hold] Job is HELD. Delete file to end: $LOCK_AFTER"
    echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
fi
# ============================================================== 4. 运行区结束 ===================================================


# ============================================================= 5. “不死鸟” 兜底 =================================================
while [ -f "$LOCK_AFTER" ]; do
    sleep 60
done
echo "[Exit] Job finished."