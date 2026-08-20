# Stage1 Dataset 测试

本目录验证 `src/datasets/stage1_dataset.py` 的 V3 请求物化契约。当前测试文件 `test_stage1_dataset.py` 按以下顺序覆盖：PDB 清单读取、固定请求展开、完整图 mmap 裁块、受体主链 50 维特征、同步空间变换、线程共享 LRU 状态和训练/推理模式边界。

测试使用临时目录构造最小 NPY/NPZ 资产，不读取服务器数据，也不启动训练。科学字段的核心断言包括：

- 密度输入保持 `float32 (C,80,80,80)`，BOX 起点采用完整图 ZYX 索引；
- `atom_feat` 为 `float32 (N_atom,49)`，`atom_is_backbone` 为 `bool (N_atom,)`；
- 推理侧可在模型内部把二者拼成 50 维 `A_feat_L0`；
- 同一 Dataset 实例的 LRU 命中可被物化线程共享，首次并发 miss 允许重复打开 mmap，但缓存顺序和字节计数必须一致；
- 实际 80³ 裁块拒绝 NaN/Inf，距离裁块另外拒绝负数。
