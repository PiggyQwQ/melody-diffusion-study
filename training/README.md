# training

64 格旋律窗口的离散扩散模型：训练代码、9 个已训练的检查点，以及两个评测用命令：`make_xt.py`（生成受扰输入）和 `reconstruct.py`（用模型重建）。实现以组内参考实现 `reference_diffusion.py` 为基础重构；解码和 MIDI、MusicXML 导出调用仓库 `scripts/encode_folk_windows.py` 与 `scripts/folk_window_samples.py`。

## 环境

需要 64 位 Python 3.11–3.14。在仓库根目录执行（以 Windows PowerShell 为例；Linux 上把 `.\.venv\Scripts\python.exe` 换成 `.venv/bin/python`）：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r training\requirements.txt
```

`requirements.txt` 固定 `torch==2.9.1+cpu`、`numpy==2.3.5`、`mido==1.3.3`、`pytest==9.2.0`，并加入 PyTorch 的 CPU 软件源。如果 PyTorch 下载失败，可以先单独安装 CPU 版，再装其余依赖：

```powershell
.\.venv\Scripts\python.exe -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r training\requirements.txt
```

以下命令都在仓库根目录用该环境的 Python（下文写作 `python`）运行。

## make_xt.py：生成受扰输入

```
python training/make_xt.py --split {validation,test} (--window-ids 文件 | --all)
                           [--times 10,25,50,75,90] [--noise-seeds 0,1,2,3,4] --out 文件.jsonl
```

| 参数 | 含义 |
| --- | --- |
| `--split` | 读取 `data/acfs-v261103/split-encode-v1/<split>.jsonl` |
| `--window-ids 文件` | 文本文件，每行一个 `window_id`；实例按文件中的顺序排列 |
| `--all` | 使用该集合的全部窗口，按文件顺序 |
| `--times` | 逗号分隔的扩散步 t，取值 0–100（默认 `10,25,50,75,90`） |
| `--noise-seeds` | 逗号分隔的非负整数（默认 `0,1,2,3,4`） |
| `--out` | 输出 `.jsonl`，不得已存在；同时在旁边写同名的 `.meta.json` |

每个实例是一个窗口加噪一次：每格以概率 t/100 从 51 类中均匀重抽，否则保留原符号，与训练时一致。`case_id` 为 `{window_id}|t{t}|n{noise_seed}`；实例的随机数只由 `case_id` 的 SHA-256 前 8 字节决定，与文件中有哪些其他窗口无关。每行字段：`case_id, window_id, song_id, group_id, split, t, rho, noise_seed, noisy_tokens, clean_tokens`。

`python training/scripts/select_windows.py --split S --count N --seed K --out 文件` 按固定种子随机选 N 个窗口，按文件顺序写出 window_id 列表。

## reconstruct.py：用模型重建

```
python training/reconstruct.py --xt 文件(.jsonl|.jsonl.gz) --out 目录 [--runs all|h2|h3|运行名,...]
                               [--samples 3] [--device auto|cpu|cuda] [--threads 4] [--batch-size 500]
```

| 参数 | 含义 |
| --- | --- |
| `--xt` | 输入 xt 文件；只读取 `case_id, window_id, song_id, group_id, split, t, rho, noise_seed, noisy_tokens`，忽略 `clean_tokens` |
| `--out` | 输出目录；已存在的同名输出文件不会被覆盖，程序报错退出 |
| `--runs` | `all`（默认，9 个模型）、`h2`（`h2-acfs_full-s0..s2`）、`h3`（`h3-acfs_matched-s0..s2` 与 `h3-cantopop-s0..s2`），或逗号分隔的运行名 |
| `--samples` | `diffusion` 和 `oneshot_sample` 每个实例的采样次数（默认 3） |
| `--device` | `auto`（有 CUDA 用 GPU，否则 CPU）、`cpu` 或 `cuda` |
| `--threads` | CPU 线程数（默认 4） |
| `--batch-size` | 每批实例数（默认 500）；调小可减少内存占用 |

每个模型输出 `目录/recon__<运行名>.jsonl.gz` 和 `目录/recon__<运行名>.meta.json`。每个实例 7 行：`diffusion` 3 行、`oneshot_argmax` 1 行、`oneshot_sample` 3 行；字段为 `case_id, window_id, t, method, model_dataset, train_seed, checkpoint_sha256, sample_index, sample_seed, raw_generated_tokens, decoded_tokens, leading_hold_repaired, has_sounding_note`。gzip 文件的 mtime 为 0、不含文件名。读取方法：`gzip.open(path, "rt", encoding="utf-8")`。

检查点位于 `training/checkpoints/<运行名>.pt`；`training/checkpoints/manifest.json` 列出每个检查点的数据集、种子、步数、验证交叉熵和 SHA-256，`reconstruct.py` 使用前会核对 SHA-256。

## 测试

```powershell
python -m pytest -q training/tests
```

测试只读取训练集和验证集文件。

## 其他代码

`train.py` 训练模型；`sample.py` 提供 `generate`（无条件生成）、`trajectory`（轨迹）和 `nn-reference`（最近邻参照）；`diffusion.py` 是共用的加噪过程、网络和采样器；`training/scripts/` 是服务器上训练和产出时使用的脚本。
