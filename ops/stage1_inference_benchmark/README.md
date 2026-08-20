# Stage1 V3 GPU 利用率基准

本目录只包含一次真实推理命令的外部采样工具。它不改变模型、配置或产物，只在子进程运行期间读取 `nvidia-smi`，结束后读取 `probability/geometry.json` 和 `status/<role>/performance.json` 的计数与队列等待时间。科学 NPZ 不保存性能字段。

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
  -- python -m src.inference.cli calibrate ...
```

`samples.csv` 的列为：

| 列 | 单位与缺失行为 |
| --- | --- |
| `unix_time` | Unix 秒；一次 `nvidia-smi` 查询失败时不写该次记录 |
| `gpu_index` | 显式 `--gpu-index`；其他可见 GPU 的记录被过滤 |
| `utilization_percent` | GPU 利用率百分比 |
| `memory_used_mib` | 已用显存 MiB |

`summary.json` 的键为：`command`、`run_mode`、`gpu_index`、`return_code`、`wall_seconds`、`sample_interval_seconds`、`gpu_sample_count`、`gpu_active_ratio`、`gpu_utilization_mean_percent`、`gpu_utilization_p50_percent`、`gpu_utilization_p95_percent`、`window_count`、`window_per_second`、`full_map_materialize_wait_seconds`、`full_map_fusion_wait_seconds`、`centered_entry_count`、`centered_entry_per_second`、`centered_materialize_wait_seconds` 和 `centered_cpu_arrange_wait_seconds`。没有有效采样或吞吐分母为零时，对应统计为 `0.0`。GPU active ratio 定义为 `utilization.gpu > 0` 的有效采样比例；它是观测量，不设伪造的通过阈值。

串行与流水线必须使用相同 checkpoint、PDB 清单和科学配置分别运行。第二次运行通过 `--comparison-summary <第一次的 summary.json>` 引用另一模式；输出新增 `comparison`，其中 `summary_path` 和 `run_mode` 标识对照，`wall_speed_ratio`、`window_throughput_ratio` 和 `centered_throughput_ratio` 给出当前运行相对对照的比值，分母为零时写 `null`。串行基线通过把完整图与 centered 的预取、待处理队列和发布线程配置为 1 获得；只改变流水并发参数，不改变 stride、阈值、模型或科学字段。

基准必须使用真实 checkpoint、真实 PDB 清单和目标配置。短清单适合比较实现变化，正式清单适合记录最终实战吞吐；两者的结果不能混作同一基线。
