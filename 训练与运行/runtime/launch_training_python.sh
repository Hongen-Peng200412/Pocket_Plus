#!/usr/bin/env bash
set -euo pipefail

# 统一启动训练 Python. 单节点保持原来的直接调用; 跨节点时, 每个 Slurm 节点
# 启动一个 torchrun agent, 再由该 agent 为本节点的每张 GPU 创建一个训练进程.

python_bin="${TASK_PYTHON_BIN:-python}"

if [[ "${TASK_DDP_ENABLED:-0}" != "1" ]]; then
    exec "${python_bin}" -u "$@"
fi

node_count="${TASK_NNODES:?缺少 TASK_NNODES}"
gpus_per_node="${TASK_GPUS:?缺少 TASK_GPUS}"
node_rank="${TASK_DDP_NODE_RANK:?缺少 TASK_DDP_NODE_RANK}"
master_addr="${TASK_DDP_MASTER_ADDR:?缺少 TASK_DDP_MASTER_ADDR}"
master_port="${TASK_DDP_MASTER_PORT:?缺少 TASK_DDP_MASTER_PORT}"

exec "${python_bin}" -u -m torch.distributed.run \
    --nnodes="${node_count}" \
    --nproc-per-node="${gpus_per_node}" \
    --rdzv-backend=static \
    --node-rank="${node_rank}" \
    --master-addr="${master_addr}" \
    --master-port="${master_port}" \
    "$@"
