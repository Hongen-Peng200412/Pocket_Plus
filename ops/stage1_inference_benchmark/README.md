# Stage1 V3 GPU 利用率基准

本目录只包含一次真实推理命令的外部采样工具。采样器本身不改写被测子进程的产物；被测推理命令仍执行正常写盘行为。采样器在子进程运行期间读取 `nvidia-smi`，结束后读取 `probability/geometry.json` 和 `status/<role>/performance.json` 的计数与队列等待时间。科学 NPZ 不保存性能字段。

Windows RTX 和 Linux 服务器都可使用：

```powershell
python ops/stage1_inference_benchmark/benchmark_gpu_utilization.py `
  --output-dir C:/temp/stage1_gpu_benchmark `
  --sample-interval-seconds 0.5 `
  --artifact-root C:/temp/stage1_output `
  --producer unet_c1 `
  --split calibration `
  --gpu-index 0 `
  --run-mode pipeline `
  -- python -m src.inference.cli probability ...
```

`samples.csv` 的列为：

| 列 | 单位与缺失行为 |
| --- | --- |
| `unix_time` | Unix 秒；一次 `nvidia-smi` 查询失败时不写该次记录 |
| `gpu_index` | 显式 `--gpu-index`；其他可见 GPU 的记录被过滤 |
| `utilization_percent` | GPU 利用率百分比 |
| `memory_used_mib` | 已用显存 MiB |

`summary.json` 的字段为：

| 字段 | 类型与单位 | 含义与缺失行为 |
| --- | --- | --- |
| `command` | 字符串列表 | 本次工具启动的完整子进程参数 |
| `run_mode` | 字符串 | `serial` 或 `pipeline` |
| `gpu_index` | 整数 | 显式采样的 GPU 编号 |
| `return_code` | 整数 | 子进程退出码 |
| `wall_seconds` | 浮点数，秒 | 子进程总墙钟时间 |
| `sample_interval_seconds` | 浮点数，秒 | 两次 GPU 查询之间的目标间隔 |
| `gpu_sample_count` | 整数 | 有效 GPU 采样数 |
| `gpu_active_ratio` | 浮点数 | `utilization_percent > 0` 的有效采样比例；没有有效采样时为 0.0 |
| `gpu_utilization_mean_percent` | 浮点数，百分比 | 有效样本的平均 GPU 利用率；没有有效采样时为 0.0 |
| `gpu_utilization_p50_percent` | 浮点数，百分比 | 有效样本的 GPU 利用率中位数；没有有效采样时为 0.0 |
| `gpu_utilization_p95_percent` | 浮点数，百分比 | 有效样本的 GPU 利用率 95 分位数；没有有效采样时为 0.0 |
| `window_count` | 整数 | `probability/geometry.json` 汇总的完整图窗口数 |
| `window_per_second` | 浮点数，窗口/秒 | 窗口数除以完整图墙钟时间；分母为零时为 0.0 |
| `full_map_materialize_wait_seconds` | 浮点数，秒 | 完整图等待 CPU 请求物化的累计时间 |
| `full_map_fusion_wait_seconds` | 浮点数，秒 | 完整图等待 CPU Gaussian 融合的累计时间 |
| `centered_entry_count` | 整数 | centered 性能 JSON 汇总的候选数 |
| `centered_entry_per_second` | 浮点数，候选/秒 | centered 候选数除以 centered 墙钟时间；分母为零时为 0.0 |
| `centered_materialize_wait_seconds` | 浮点数，秒 | centered 等待 CPU 请求物化的累计时间 |
| `centered_cpu_arrange_wait_seconds` | 浮点数，秒 | centered 等待 CPU 字段整理的累计时间 |

GPU active ratio 是观测量，不设伪造的通过阈值。

串行与流水线必须使用相同 checkpoint、PDB 清单和科学配置分别运行。第二次运行通过 `--comparison-summary <第一次的 summary.json>` 引用另一模式，输出新增 `comparison`：

| `comparison` 子字段 | 类型 | 含义与缺失行为 |
| --- | --- | --- |
| `summary_path` | 字符串 | 被引用的对照 `summary.json` 路径；未传参数时整个 `comparison` 不存在 |
| `run_mode` | 字符串 | 对照运行的 `serial` 或 `pipeline` 模式 |
| `wall_speed_ratio` | 浮点数或 `null` | 对照墙钟时间除以当前墙钟时间；当前墙钟时间为零时写 `null` |
| `window_throughput_ratio` | 浮点数或 `null` | 当前窗口吞吐除以对照窗口吞吐；对照值为零时写 `null` |
| `centered_throughput_ratio` | 浮点数或 `null` | 当前 centered 吞吐除以对照 centered 吞吐；对照值为零时写 `null` |

串行基线通过把完整图与 centered 的预取、待处理队列和发布线程配置为 1 获得；只改变流水并发参数，不改变 stride、阈值、模型或科学字段。完整图和 centered 已拆成独立命令，因此两类基准分别运行：`probability` 命令测完整图流水，`centered` 命令测 centered 流水。两类结果不能合并成同一个阶段吞吐。

基准必须使用真实 checkpoint、真实 PDB 清单和目标配置。短清单适合比较实现变化，正式清单适合记录最终实战吞吐；两者的结果不能混作同一基线。
