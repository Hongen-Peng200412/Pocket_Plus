# Handoff: Stage1 V3 推理可读性重做与双线重建

Date: 2026-08-20

## Current State

Stage1 V3 推理的可读性重做、本地验证、三类独立审查和 Git 双线重建均已完成。`Learn/CUMULATIVE` 与 `Learn/stage1-inference-v3` 指向同一个本地学习端点；真实实现端点保留在 `codex/stage1-inference-v3-readability`。正式 checkpoint smoke、真实 F1/F3 生产和服务器 GPU 利用率基准尚未执行。

## Completed

- 主代理逐名检查 `src/inference/` 的 48 个类、方法、顶层函数和局部回调，并在 `talk/refactor/stage1_v3_inference.md` 记录保留理由、函数顺序、嵌套必要性和字段契约。
- 正式代码删除 checkpoint、配置和代码摘要，只保留 checkpoint 规范化路径；不同科学配置由显式 `output_root` 目录区分。
- `save_dense48=false` 时，centered 生产只读取概率产物中的 `origin_xyz` 与 `voxel_size_xyz`，不再解压完整概率图。
- 布局/Git/函数审查、注释与文档审查、科学逻辑审查均完成第 3/3 轮全面核查，之后只复核已报告问题，最终全部批准。
- 重建后的学习端点再次通过 Black、`compileall`、28 项回归、三个 CLI 帮助入口、OmegaConf 解析、Bash 语法和 `git diff --check`。

## Decisions

- 不建立 SHA、`_valid*`、运行身份对象或 calibration 目录相等限制。
- `--calibration` 可以显式指向另一输出目录；代码只比较 calibration JSON、同目录 `_COMPLETE` 和当前命令的 checkpoint 路径。
- 保留 9 个必要局部回调，只用于线程池事务、惰性读取或递归增广；不新增 Dataset、collator 或 wrapper 包装层。
- 学习历史从共同基点重新构造原有 8 个主题提交，不在旧端点后追加可读性修复提交。

## Open Questions

- 使用哪个最终 checkpoint 执行首个真实 calibration smoke。
- 真实推理输出使用哪个显式版本目录名，以区分 F3/F2 等科学配置。

## Next Actions

1. 用户确定 checkpoint、producer、PDB 短清单和输出版本目录。
2. 先在本地或服务器执行真实 calibration smoke，核对五类产物与完成标记。
3. 得到服务器运行授权后，执行串行与流水线成对 GPU 利用率基准。
4. smoke 验收后再安排 validation、train 和 Matcher 实战。

## Files To Reopen

- `talk/refactor/stage1_v3_inference.md`
- `src/inference/README.md`
- `configs/inference/stage1_v3.yaml`
- `训练与运行/sh/infer/README.md`
- `C:/Users/15919/Desktop/AdaLigand/文档/规划文档/BOX-level数据契约.md`
- `C:/Users/15919/Desktop/AdaLigand/文档/exec_plan/Stage1_V3推理重写实施记录.md`
