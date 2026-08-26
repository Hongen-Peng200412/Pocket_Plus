# Stage1 Li 阈值与候选比例实验计划

## 目标

在独立代码工作树、独立 Git 分支和独立服务器结果根中，复用 `unet_c1` 已有
calibration/validation probability，比较逐 PDB Li 阈值加 basic-style 比例选择
的正式评估结果。参数只在 calibration 拟合，再冻结用于 validation；本实验不
改写 Stage1 正式 BOX 契约，也不决定后续是否推广到其他 producer。

## 已冻结边界

- calibration 处理 100 个 PDB，validation 处理 200 个 PDB。
- 只执行 CPU Li blobs、basic_ratio 调参和 evaluate；不执行模型前向或 centered。
- F1 和 F2 都运行；分别使用 objective_beta 1 和 2。
- validation 的每个 PDB 仍计算自己的无监督 Li 阈值，但 F1/F2 basic_ratio 参数
  必须完整复用 calibration JSON，不在 validation 重新调参。
- Li 数值规则、1/32768 向上量化、26 连通区域和阈值包含端点规则不再变更。
- 比例总体、half-up 最近整数、精确变化点扫描、较小比例并列获胜和最终
  `min_voxels` 后过滤规则不再变更。
- 既有概率网格、macro 三项目标、评估指标、并行方式及其余科学契约保持不变。
- 服务器结果只写入名称以 `--Li` 结尾的目标根。
- 本轮代码改动只保持 unstaged；没有用户后续明确允许时，不执行 commit。

## 实施阶段

1. 从 `Learn/CUMULATIVE` 建立独立工作树和实验分支，确认主工作树既有改动不受影响。
2. 在 `src/inference` 增加 Li 数值函数、比例选择函数和精确比例搜索；让 evaluate
   读取显式候选角色，使 `Li_blobs` 无需伪装成 F-alpha 角色。
3. 在 `ops/stage1_li_ratio_trial` 增加一次性 calibration 编排和正式 CPU 任务脚本。
4. 用合成不均衡候选数案例证明精确比例状态、两阶段相同 macro 目标和串并行一致。
5. 用小型端到端案例证明正常目录、Li 字段、F1/F2 选择 JSON 和 micro/macro/PRAUC 评估。
6. 把隔离工作树非删除式同步到独立远端任务根，使用通用 Slurm 入口提交 16 CPU、64 GiB calibration 作业。
7. calibration 验收后同步 validation 入口，提交同资源 validation 作业，并冻结复用 calibration 选择参数。
8. 以 10 至 30 分钟自然睡眠监视，完成后核对两个数据划分的 PDB 数、产物数、选择参数和评估汇总。
9. 回填执行记录和 Claude handoff；保留全部代码为 unstaged，等待用户决定是否提交或放弃。

## 验收条件

- calibration 100 个和 validation 200 个 PDB 都有真实复制的 probability 与
  `Li_blobs/_COMPLETE`。
- 每个 `Li_blobs.npz` 同时保存原始、网格编号和实际 Li 阈值。
- F1 与 F2 选择 JSON 只保存 `score_ratio_threshold`，不保存 `score_threshold`。
- 精确比例扫描和最终 `min_voxels` 搜索使用同一 PDB 等权 macro 三项目标。
- workers=1 与 workers>1 得到逐字段相同的 basic_ratio 选择结果。
- F1 与 F2 evaluate 分别完成 calibration 100 个和 validation 200 个 PDB，并
  同时发布 micro、macro 与 PRAUC。
- validation 执行前后两份 calibration tuning JSON 逐字节不变。
- 服务器目标目录与正常 Stage1 推理结构同构，不含旧 blobs、centered 或旧调参文件。
- 执行记录含代码来源、数据来源、同步方式、提交命令、Job 编号、运行结果和关键指标。
- Git 状态仍为 unstaged，且没有新 commit。

## 当前状态

- [x] 独立工作树和实验分支已经建立。
- [x] Li、basic_ratio 和实验编排的第一版实现已经完成。
- [x] 定向 CPU 测试已经通过。
- [x] 逐文件两遍自查与完整本地回归。
- [x] 隔离服务器同步与 calibration CPU 作业。
- [x] calibration 服务器产物验收。
- [x] validation 入口回归、同步、CPU 作业与产物验收。
- [x] 两个数据划分的执行记录。
- [x] Li 与三种经典方案的两个数据划分联合分析。
- [x] 隔离实验 handoff。

计划进度只在阶段状态改变时更新；逐条命令和运行结果写入 `EXECUTION.md`。
