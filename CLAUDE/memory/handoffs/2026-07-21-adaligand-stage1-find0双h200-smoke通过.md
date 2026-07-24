# Handoff: AdaLigand Stage1 Find_0 双 H200 smoke 通过

Date: 2026-07-21

## Current State

正式 Job 为 unet_c1 `321107`（1×H100）、Find_1 `321540`（2×H100）和 Find_0 `321743`（2×H200）。Find_2 已撤回。

Find_0 使用 micro8/accum3/GBS48 完成五步 smoke，`exit_code=0`；双 H200 峰值 97.346%/99.387%，无 OOM。正式 CPC1 已创建并进入冷初始化，上次检查尚无 `wandb_status` 或 summary，但 Job/锁正常且明确错误匹配 0。

Find_1 W&B `qqmuqyxk` 至少到 step149；unet 和 Find_1 明确错误匹配均为 0。

## Decisions

- 不因 Find_0 smoke 的 99.387% 峰值停止正式训练；百分比仅记录，真实 OOM 才恢复。
- heartbeat 暂时保持每 15 分钟，且不再申请 H200。Find_0 online 并推进若干 step 后降回每 3 小时。

## Next Actions

1. 检查 321743 正式 CPC1 W&B online、resolved config 和首 optimizer step。
2. 使用文件 mtime 选择三个 producer 最新 W&B summary，避免按路径字典序误选旧 run。
3. 稳定后更新 ExecPlan/handoff并降低 heartbeat 频率。
4. 持续验收五段 validation30、LR-drop4/1、BEST 和 CPC2 model-only 谱系。

## Files To Reopen

- `talk/Excx_执行stage1端到端训练.md`
- `tmp/stage1_formal_find_chain.sh`
- `tmp/stage1_formal_train_stage.sh`

