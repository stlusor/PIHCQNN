# PIHCQNN GitHub 上传包

本目录汇集原始论文代码和第10版返修稿实际采用的补充实验代码，配套保存最终结果、参数、随机种子、流场数据及采样索引。

上传 GitHub 时，解压 ZIP，把本目录内的文件和子目录放在仓库根目录，保留现有层级。英文 `README.md` 可直接作为仓库首页。

| 目录或文件 | 内容 |
|---|---|
| `original_code/` | 原始 `PIHCQNN.rar` 和逐字节保留的13份源码 |
| `supplementary/reviewer1_experiments/` | 现代PINN对照、数据重上传次数与频率诊断 |
| `supplementary/reviewer2_experiments/` | 热方程及Poisson五种子重复结果 |
| `supplementary/reviewer11_experiments/` | 弹性正问题、参数敏感性、梯度、DEM、数值解法及三参数反演 |
| `supplementary/reviewer_supplement_20261007/` | 固定边界权重实验、梯度审计、流场经典模型对照 |
| `supplementary/flow_reproduction_20261007/` | 流场实现、参考数据、193个观测点的索引、最终模型及预测 |
| `MANUSCRIPT_MAP.csv` | 代码、数据与稿件表格/章节的对应关系 |
| `supplementary/publication_results/` | 由最终实验记录导出的汇总表 |
| `validation/` | 整理时的核查记录 |

补充实验环境为 Python 3.11.11、PyTorch 2.5.1+cu121、PennyLane 0.45.1，GPU 为 RTX 4070 12 GB。固定权重扫描及流场经典模型采用 CPU，具体环境写在对应说明和配置中。

在仓库根目录可执行：

```console
python -m pip install -r requirements.txt
python verify_release.py
python supplementary/run_experiments.py --list
python supplementary/export_results.py
python supplementary/verify_flow_release.py --checkpoints
```

运行指定补充实验，例如：

```console
python supplementary/run_experiments.py --suite modern_poisson --seed 0 --execute
python supplementary/run_experiments.py --suite fixed_weights --seed 0 --execute
python supplementary/run_flow_final.py --model all --execute
```

新运行结果写入 `supplementary/rerun_results/`。英文模块说明给出了各组实验的具体参数和入口。原始代码中的旧版PennyLane设备、CUDA编号和输出路径见 `original_code/README.md`，用于运行的工作副本可按机器配置调整。

原稿结果与新增实验分别对应稿件中的原始研究和返修补充对照。流场保存的权重、固定采样记录和预测文件对应新增统一协议实验。原始压缩包保留完整原貌；正文原始流场结果和Fig. 10保持其历史实验身份。
