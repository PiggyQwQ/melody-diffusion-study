# training：离散扩散模型的训练、重建与生成

本目录是训练组的正式实现：均匀替换离散扩散（T = 100，51 类符号，64 格窗口）、x_0 预测网络、反向采样、批量重建、一次预测消融、无条件生成与轨迹导出。方法说明见 [METHOD.md](METHOD.md)，9 次正式训练的汇总见 [runs.md](runs.md)，与评测组交换的文件格式见 [INTERFACE.md](INTERFACE.md)。

实现以组内参考实现 `reference_diffusion.py`（随训练指南 V2 分发，不在本仓库中）为基础重构。转移矩阵、闭式加噪、网络结构和反向后验 `reverse_probs` 沿用参考实现的定义；载入参考实现的检查点后，两者的 logits 和反向概率逐位相同。解码和 MIDI、MusicXML 导出直接调用仓库 `scripts/encode_folk_windows.py` 的 `decode` 与 `scripts/folk_window_samples.py` 的 `export_midi`、`write_musicxml`。

## 文件

| 路径 | 内容 |
| --- | --- |
| `diffusion.py` | 常数、数据读取、Q_t 与 Q̄_t、闭式与单步加噪、`reverse_probs`、网络、反向链、一次预测、解码与乐谱导出 |
| `train.py` | 训练；写出 `config.json`、`train_log.csv`、`summary.json`、`best.pt`、`last.pt` |
| `sample.py` | `make-xt`（临时 xt）、`reconstruct`、`generate`、`trajectory` |
| `tests/` | pytest：转移矩阵、反向后验、两种加噪抽样的分布一致性、数据读取、可复现性、忽略 `clean_tokens` |
| `scripts/` | 服务器脚本：测试、冒烟、GPU 检查、任务启动、第 3 步训练、第 4 步产出、批量重建 |
| `checkpoints/` | 9 个正式检查点 |
| `outputs/` | 训练记录、重建、生成、轨迹与乐谱（见下文） |

## 环境

服务器上使用 Python 3.14.7 的虚拟环境，PyTorch 2.9.1（CUDA 12.8 版本的 wheel，正式训练在 CPU 上运行）、NumPy 2.5.5、Mido 1.3.3、pytest 9.2.0：

```bash
python3 -m venv ~/homework/melody-study/.venv
~/homework/melody-study/.venv/bin/pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128
~/homework/melody-study/.venv/bin/pip install mido==1.3.3 numpy pytest
```

代码本身只需要 Python ≥ 3.11、PyTorch 和 Mido（导出 MIDI 时）。所有命令从仓库根目录运行。

## 复现命令

测试（CPU）：

```bash
python -m pytest -q training/tests
```

训练（以 H2 种子 0 和 H3 流行种子 0 为例；`--device cpu --threads 4` 与正式运行一致）：

```bash
OMP_NUM_THREADS=4 python training/train.py --device cpu --threads 4 --dataset acfs_full \
  --layers 2 --width 128 --steps 20000 --validate-every 1000 --seed 0 --run-dir RUNS/h2-acfs_full-s0
OMP_NUM_THREADS=4 python training/train.py --device cpu --threads 4 --dataset cantopop \
  --layers 2 --width 128 --steps 2000 --schedule-steps 10000 --validate-every 250 --seed 0 \
  --run-dir RUNS/h3-cantopop-s0
```

其余超参数用 `train.py` 的默认值（批大小 64、学习率 3×10⁻⁴、预热 500 步、余弦衰减到 0、权重衰减 0.01、梯度裁剪 1.0、dropout 0.1）。服务器上 9 次训练由 `training/scripts/step3.sh` 一次启动。

重建、一次预测消融、生成与轨迹：

```bash
python training/sample.py --device cpu make-xt --output xt-provisional-validation.jsonl
python training/sample.py --device cpu reconstruct --checkpoint training/checkpoints/h2-acfs_full-s0.pt \
  --xt XT.jsonl --output recon__h2-acfs_full-s0.jsonl
python training/sample.py --device cpu generate --checkpoint training/checkpoints/h2-acfs_full-s0.pt \
  --output generate__h2-acfs_full-s0.jsonl
python training/sample.py --device cpu trajectory --checkpoint training/checkpoints/h2-acfs_full-s0.pt \
  --output-dir trajectory
```

评测组的正式 xt 文件用 `bash training/scripts/reconstruct_all.sh <xt.jsonl> <输出目录>` 处理，9 个模型各输出一个文件。`reconstruct` 的默认值：每份 xt 扩散采样 3 次、一次预测抽样 3 次、argmax 1 次，`--batch-size 500`，`--seed 20261121`。同一检查点、输入、批大小和种子在 CPU 上结果逐位相同。

所有输出路径必须是新的；脚本不会覆盖已有文件。

## 输出文件

| 文件 | 内容 |
| --- | --- |
| `outputs/step2/` | 过拟合检查、四种网络比较、H3 步数探针的 `config.json`、`train_log.csv`、`summary.json`，说明见其中的 `README.md` |
| `outputs/runs/<运行>/` | 9 次正式训练和 2×128 过拟合记录的 `config.json`（完整配置、提交哈希、数据与词表 SHA-256、窗口数、参数量、PyTorch/CUDA 版本、设备与线程数）、`train_log.csv`（步数、训练损失、验证损失、学习率、累计耗时、累计验证耗时、分 t 验证损失）、`summary.json`（选中的步数、验证损失、总耗时） |
| `outputs/xt/xt-provisional-validation.jsonl` | 临时 xt：ACFS 验证集 100 窗口 × t = 10/25/50/75/90 × 2 次加噪，共 1,000 行，带 `clean_tokens`；只用于跑通流程 |
| `outputs/recon-provisional/recon__<模型>.jsonl` | 某个 xt 文件在一个模型上的重建与一次预测消融，每个实例 7 行（diffusion ×3、oneshot_argmax ×1、oneshot_sample ×3）；同名 `.meta.json` 记录输入 SHA-256、批大小和处理顺序 |
| `outputs/generate/generate__h2-acfs_full-s<种子>.jsonl` | 每个 H2 模型 100 段无条件生成，含 x_100、解码结果和训练集最近邻（普通汉明距离与 ±12 半音移调后的最小汉明距离及对应 `window_id`） |
| `outputs/trajectory/trajectories.json` | 3 个验证窗口的前向逐步轨迹（x_0…x_100）和分别从 t = 50、t = 100 反向到 0 的轨迹 |
| `outputs/trajectory/<窗口>/<链>_t<步>.mid`、`.musicxml` | 上述轨迹在 t = 0、10、25、50、75、90、100 的状态导出的 MIDI（90 BPM，钢琴）和 MusicXML |

字段定义见 [INTERFACE.md](INTERFACE.md)。
