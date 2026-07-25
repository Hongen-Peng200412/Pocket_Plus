#!/bin/bash
#SBATCH -o /home/penghongen/Stage1/Out/NEW_final.txt    # %j 会自动替换为 JOBID
#SBATCH -e /home/penghongen/Stage1/Error/NEW_final.txt
#SBATCH -p a100
#SBATCH --qos=a100g6                                    # 依然使用高优先级的QoS
#SBATCH -J FT

# --- [排队核心] 若拆分为双节点 ---
#SBATCH -N 2                                            #  申请 2 个节点
#SBATCH --gres=gpu:a100:2                               #  每个节点 2 张卡 (容易排队!)
#SBATCH --ntasks-per-node=1                             #  每个节点启动 1 个 task (由 torchrun 接管)
#SBATCH --cpus-per-task=32                              #  每个task32核 (注意资源分配逻辑为：Node->task->卡[CPU]，每个task2个卡，一个卡16核，总共16x4=64核)





# 改变卡的分布、调整运行的.py文件见 “4.容错运行区”
# ======================================================= 1. 环境准备区（要动的最多是source） ================================================
# 获取第一个节点的主机名作为 Master Address (多节点通讯必须)
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
# 基于 JOBID 生成唯一端口，避免冲突
export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))

echo "Job $SLURM_JOB_ID allocated on nodes: $SLURM_JOB_NODELIST"
echo "Master Address: $MASTER_ADDR, Port: $MASTER_PORT"

# [Gemini 修复] 设置线程数，避免大量的警告导致日志难以阅读或CPU过载
# export OMP_NUM_THREADS=1
# [CPU 满载优化] 让 NumPy / Pandas / PyTorch 知道有 16 个核可用
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16
export OPENBLAS_NUM_THREADS=16
export VECLIB_MAXIMUM_THREADS=16
export NUMEXPR_NUM_THREADS=16

# [Gemini 修复] 关键网络配置！解决 Socket Timeout 和 IndexError
# 排除回环地址(lo)和docker地址，强制使用物理网口进行通讯
export NCCL_SOCKET_IFNAME=^lo,docker0
export GLOO_SOCKET_IFNAME=^lo,docker0
# 开启 INFO 级别的日志，如果再次失败可以看到具体的 IP 绑定情况
export NCCL_DEBUG=INFO

source /home/penghongen/anaconda3/bin/activate py310_tensorboard
# ======================================================= 1. 环境准备区（要动的最多是source） ================================================








# ========================================== 2. 看看你安装哪种锁(如果不需要锁可以注释掉这里，但是更推荐注释下面的3、5就好了) ===================================
# ========================================== 说明：如果你不需要前向锁，那么就注释掉“3门神锁”的全部和这里的“创建前向锁” =========================================
# ==========================================  说明：如果你不需要后向锁，那么就注释掉“5、不死鸟”和这里的“创建后向锁” ========================================== 

# 创建前向锁
LOCK_PRE="pre_lock_${SLURM_JOB_ID}"
touch $LOCK_PRE
echo "[Status] pre_Lock created: $LOCK_PRE"
echo "[Wait] Waiting for user to delete $LOCK_PRE to start..."

# 创建后向锁
LOCK_AFTER="after_lock_${SLURM_JOB_ID}"
touch $LOCK_AFTER
echo "[Status] after_Lock created: $LOCK_AFTER"
# =================================================== 2. 看看你安装哪种锁 ==========================================================










# ======================================================== 3. “门神” 锁机制 (Pre-Run) ==================================================================
# 循环检查，直到用户手动删除 pre_lock
while [ -f "$LOCK_PRE" ];
do
    sleep 60
done

echo "[Start] Lock released! Starting Distributed Training..."
# ======================================================== 3. “门神” 锁机制结束 (Pre-Run) ===============================================================







# ============================================================== 4. 容错运行区 ============================================================================
# [Gemini 终极修复]
# 1. 后端更换：--rdzv_backend=static
#    既然我们在 Slurm 上，节点和 Rank 都是确定的，使用 static 后端最稳定，
#    它不进行复杂的动态发现，直接根据 IP 和 Rank 建立连接，彻底避开 IndexError。
# 2. 增加错峰等待：让非主节点 (NodeID != 0) 等待 10 秒，
#    确保主节点 (gnode02) 的端口已经打开并在监听了，防止从节点连接过早导致崩溃。
# ----------------------------------- 调节参数看这里(--nnodes=2, --nproc_per_node=2 以及 /home/penghongen/Stage1/NEW111.py) ---------------------------------
# --nnodes=2: 告诉 torchrun 有两个物理节点
# --nproc_per_node=2: 每个物理节点启动 2 个进程 (对应每节点2卡)
# /home/penghongen/Stage1/NEW111.py: 训练脚本路径

if srun --export=ALL bash -c "
    # 调试信息
    echo \"[Debug] Host: \$(hostname) | NodeID: \$SLURM_NODEID | Master: $MASTER_ADDR:$MASTER_PORT\"
    
    # [错峰启动] 如果不是主节点 (NodeID 0)，则等待 10 秒，让主节点先启动 TCPStore
    if [ \$SLURM_NODEID -ne 0 ]; then
        echo \"[Info] Worker Node \$SLURM_NODEID is sleeping 10s waiting for Master...\"
        sleep 10
    fi

    python -m torch.distributed.run \
    --nnodes=2 \
    --nproc_per_node=2 \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=static \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    --node_rank=\$SLURM_NODEID \
    /home/penghongen/Stage1/NEW_final.py
    ";
then
    
    echo "[Success] Training finished successfully."
    rm -f $LOCK_AFTER

else
    echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
    echo "[Error] Training FAILED! Check error logs above."
    echo "[Hold] Job is HELD. Script will NOT exit."
    echo "       You can check logs, modify code, or debug."
    echo "       To end this job, delete file: $LOCK_AFTER"
    echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
fi
# ============================================================== 4. 容错运行区结束 ============================================================================












# ============================================================= 5. “不死鸟” 兜底机制 (Post-Run) ===============================================================
# 如果上面的任务失败了，after_lock 还存在，脚本就会在这里死循环挂机
# 这样你就有机会看到报错，而不是因为任务退出而丢了日志或资源
while [ -f "$LOCK_AFTER" ];
do
    sleep 60
done
# ========================================================== 5. “不死鸟” 兜底机制 (Post-Run)结束 ================================================================



echo "[Exit] Job finished."