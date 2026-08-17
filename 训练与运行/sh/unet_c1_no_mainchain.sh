#!/usr/bin/env bash

# 双卡 unet_c1 入口：保留结构预测头和配体距离监督，只关闭蛋白与核酸主链损失。
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
export UNET_C1_VARIANT=no_mainchain
export TASK_GPUS="${TASK_GPUS:-2}"
exec "${SCRIPT_DIR}/unet_c1.sh" "$@"
