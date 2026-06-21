# Docking 阶段需警惕金属离子预测 instance

Type: gotcha
Date: 2026-05-19
Tags: docking, inference, ligand-instance, metal-ion

## Context

Pocket Plus 前置神经网络有可能在训练时把金属离子也当作正类（比如，/home/penghongen/My_Project/feedback_plus/infer_out/ligand_base2_new/ 就把金属离子当作正类了）；但下游分子对接阶段不需要、也不能对接独立金属离子。

## Memory

推理产物中的 predicted instance 不能全部信任。某些 instance 可能对应真实金属离子，从训练/推理角度是正确正类，但在 docking 流程里应视为不可对接对象，若直接进入 site-ligand assignment 会表现为 docking 假阳性。

因此 docking pipeline 在读取推理 instance 后，应额外支持后处理过滤与合并，例如：

- 按最小体素数过滤过小 instance。
- 根据最近体素距离、中心距离、合并前后拓扑形状或统计信息决定是否合并相邻 instance。
- 尽量把这类 instance 后处理逻辑单独成文件，方便未来复制或迁移回推理管线。

## When To Use

在实现或解释 docking 批处理、site-ligand 匹配、虚拟节点惩罚、样本级统计、instance 召回/假阳性分析时使用。不要把网络预测的所有 instance 直接解释为可对接 ligand 位点。

## Related Files

- `Docking/readme.md`
- `Docking/docking_pipeline/`
- `src/inference/`
