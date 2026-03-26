本项目的初衷是，给定真实的冷冻电镜密度图 + 受体结构(来自于去掉配体的PDB结构, 或AF2等结构预测软件预测/建模出来的无配体结构)，预测出结合口袋的位置。

1. 数据处理：相关脚本分布在 Pocket++\Make_Data 与 Pocket++\processedPDB_EMDB_binder , 先后运行它们.
2. 训练: 直接运行 Pocket++\src\train.py . 

注意, 当前已经完成了数据处理+训练
3. 对于推断or评估: 见 src\inference .

我已经运行了推断程序, 当前我取得了较高的recall(0.6+), 但precision并不高(0.3).