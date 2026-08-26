# F2 语义 blobs 的 beta=1 basic 增量实验计划

## 目标

在原 macro 结果目录中增加一套“F2 语义 blobs + beta=1 basic 调参”结果，并用
calibration 拟合出的冻结参数评估 calibration 与 validation。该结果将与 Li
阈值方案、F1 经典方案和 F2 beta=2 经典方案共同进入最终比较。

## 不变内容

- 不重算 probability、F2 semantic threshold 或 F2 blobs。
- 不修改、移动或覆盖 `F2_basic.json` 及既有评估结果。
- 不改变概率网格、候选连通规则、参数搜索顺序、端点包含规则、并列规则、并行
  方式、I/O 布局或正式 Stage1 Python 接口。
- 不使用 Li 实验的 `score_ratio_threshold`。
- 本轮本地改动全部保持 unstaged，未经用户后续明确允许不提交。

## 实施顺序

1. 核对原 macro 根的 100/200 个 F2 blobs、输入清单与既有调参文件。
2. 增加只调用正式 Stage1 CLI 的独立 CPU 编排脚本和配套文档。
3. 完成脚本语法、自查与正式推理 CPU 回归。
4. 非删除式同步到独立远端任务根，提交 16 CPU、64 GiB 作业。
5. 核对 `F2_basic_beta1.json`、两个数据划分的逐 PDB 评估与汇总指标。
6. 与 Li 及另外两套经典结果做两个数据划分上的联合分析。
7. 回填执行记录和主工作树 handoff；不整理或提交 Git 历史。

## 验收条件

- 新选择 JSON 的 `alpha=2.0`、`objective_beta=1.0`、`score_mode=basic`，并保存
  绝对 `score_threshold` 与 `min_voxels`。
- 原 `F2_basic.json` 仍表达 `objective_beta=2.0`。
- calibration 与 validation 分别发布 100/200 个
  `f2_blobs_basic_beta1_selected.npz`，JSONL 行数也分别为 100/200。
- 两份 `.metrics.json` 同时含 micro、macro 与 PRAUC。
- 服务器新增正式文件全部位于原 macro 根，临时调参目录已经退出。
- 执行记录包含来源、提交命令、Job 编号、运行结果和关键指标。

## 当前状态

- [x] 双线端点、主工作树既有修改和原 macro 产物只读核对。
- [x] 实验契约与直接编排方式确定。
- [x] 编排脚本、文档、自查和测试。
- [x] Job 356956 正式运行与产物验收。
- [x] 两个数据划分的 Li 与三种经典方案联合分析。
- [x] 主工作树执行记录和 handoff。
