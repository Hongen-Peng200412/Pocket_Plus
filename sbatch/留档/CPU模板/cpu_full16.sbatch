#!/bin/bash
#SBATCH -o /home/penghongen/Stage1/Out/CPU_96_Locked.txt
#SBATCH -e /home/penghongen/Stage1/Error/CPU_96_Locked.txt
#SBATCH -p cpu                      # 分区名
#SBATCH --qos=Cpu96               
#SBATCH -J Data

# --- [资源申请：96核] ---
#SBATCH -N 1                        # 单节点 (CPU运算不建议跨节点)
#SBATCH --ntasks=1                  # 1 个主进程
#SBATCH --cpus-per-task=16          # 申请满 16 个核



# ======================================================= 1. 环境准备区 ========================================================
echo "Job $SLURM_JOB_ID allocated on 1 Node with 96 Cores."

# [CPU 满载优化] 让 NumPy / Pandas / PyTorch 知道有 96 个核可用
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16
export OPENBLAS_NUM_THREADS=16
export VECLIB_MAXIMUM_THREADS=16
export NUMEXPR_NUM_THREADS=16

source /home/penghongen/anaconda3/bin/activate py310_tensorboard
# ======================================================= 1. 环境准备区结束 =====================================================






# ======================================================== 2. 锁文件定义 ========================================================
# 创建前向锁 (门神)
LOCK_PRE="pre_lock_cpu_${SLURM_JOB_ID}"
touch $LOCK_PRE
echo "[Status] pre_Lock created: $LOCK_PRE"
echo "[Wait] Waiting for user to delete $LOCK_PRE to start..."

# 创建后向锁 (不死鸟)
LOCK_AFTER="after_lock_cpu_${SLURM_JOB_ID}"
touch $LOCK_AFTER
echo "[Status] after_Lock created: $LOCK_AFTER"
# ======================================================== 2. 锁文件定义结束 =====================================================





# ======================================================== 3. “门神” 锁机制 (Pre-Run) ===========================================
# 循环检查，直到用户手动删除 pre_lock
while [ -f "$LOCK_PRE" ]; do
    sleep 60
done

echo "[Start] Lock released! Starting CPU High-Performance Calculation..."
# ======================================================== 3. “门神” 锁机制结束 =================================================






# ============================================================== 4. 容错运行区 ===================================================
# 说明：这里使用 srun 运行你的 Python 脚本
# 注意：请将下面的 /path/to/YOUR_SCRIPT.py 替换为你真实的脚本路径

if srun python /home/penghongen/Stage1/NEW_final.py; then  # <--- 修改这里为你的 CPU 脚本路径
    
    echo "[Success] CPU Task finished successfully."
    # 成功后删除后向锁，允许脚本正常结束
    rm -f $LOCK_AFTER

else
    echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
    echo "[Error] Task FAILED! Check error logs above."
    echo "[Hold] Job is HELD. Script will NOT exit."
    echo "       You can check logs, modify code, or debug."
    echo "       To end this job, delete file: $LOCK_AFTER"
    echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
fi
# ============================================================== 4. 容错运行区结束 ===============================================






# ============================================================= 5. “不死鸟” 兜底机制 (Post-Run) ==================================
# 如果上面的任务失败了，after_lock 还存在，脚本就会在这里死循环挂机
while [ -f "$LOCK_AFTER" ]; do
    sleep 60
done
# ========================================================== 5. “不死鸟” 兜底机制结束 ============================================

echo "[Exit] Job finished."