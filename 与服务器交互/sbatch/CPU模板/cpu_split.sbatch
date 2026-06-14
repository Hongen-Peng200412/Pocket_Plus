#!/bin/bash
#SBATCH -o /home/penghongen/Stage1/Out/CPU_96_NonExclusive.txt
#SBATCH -e /home/penghongen/Stage1/Error/CPU_96_NonExclusive.txt
#SBATCH -p cpu                      # <--- 确认分区名
#SBATCH --qos=Cpu96                 # <--- 依然遵守96核配额限制
#SBATCH -J CPU_NoEx

# --- [资源申请] ---
#SBATCH -N 1                        # 单节点
#SBATCH --ntasks=1                  # 1 个主进程
#SBATCH --cpus-per-task=96          # 申请 96 个核

# ======================================================= 1. 环境准备区 ========================================================
echo "Job $SLURM_JOB_ID allocated on 1 Node with 96 Cores."

# [CPU 满载优化] 
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

source /home/penghongen/anaconda3/bin/activate py310_tensorboard
# ======================================================= 1. 环境准备区结束 =====================================================


# ======================================================== 2. 锁文件定义 ========================================================
LOCK_PRE="pre_lock_cpu_${SLURM_JOB_ID}"
touch $LOCK_PRE
echo "[Status] pre_Lock created: $LOCK_PRE"

LOCK_AFTER="after_lock_cpu_${SLURM_JOB_ID}"
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
# 请将下方路径替换为你真实的脚本路径
if srun python /home/penghongen/Stage1/NEW_final.py; then
    
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