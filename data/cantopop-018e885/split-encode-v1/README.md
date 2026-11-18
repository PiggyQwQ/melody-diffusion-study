# Cantopop 划分与编码 V1

由 Jason Y H Lee 的 *A Corpus of Cantonese Popular Music, 2000–2020* 衍生。使用作者仓库提交 `018e88540b917c40dc3f7248ed98908208544b54` 的 Humdrum 转录谱。此版提取单声部、按谱面拍号筛选、切片、去重、划分和编码，并保留曲名及演唱者供溯源。衍生数据采用 CC BY 4.0。[作者项目](https://cantopop.humdrum.org/) · [固定版本](https://github.com/jasonleeubc/Cantopop-corpus/tree/018e88540b917c40dc3f7248ed98908208544b54) · [许可](LICENSE)

| 集合 | 曲目 | 窗口 | 文件 |
| --- | ---: | ---: | --- |
| 训练 | 25 | 1,323 | [train.jsonl](train.jsonl) |
| 验证 | 3 | 181 | [validation.jsonl](validation.jsonl) |
| 测试 | 3 | 180 | [test.jsonl](test.jsonl) |
| 合计 | 31 | 1,684 | |

每行对应一个连续 8 小节的窗口。`tokens` 为 64 个整数，`REST=0` 开始休止，`HOLD=1` 延长已有发声或休止，ID 2–50 对应 MIDI 38–86。`window_id`、`song_id`、`volume`、`group_id`、`split` 和 `start_bar` 与 ACFS 使用相同字段。`start_bar` 为从 1 开始的源谱小节顺序索引，包含弱起。[共享词表](vocab.json) · [曲目清单](songs.csv) · [窗口索引](windows.csv)

2/4 依据作者转录谱的拍号和小节线。同一旋律在其他转录中可能采用不同的小节分法，涉及原录音重音或拍号的比较需要另行核验。

筛选和分区统计见 [screen_audit.json](screen_audit.json) 和 [summary.json](summary.json)。[discarded_windows.csv](discarded_windows.csv) 保存精确重复窗口的剔除记录。[SHA256SUMS](SHA256SUMS) 可核对数据文件。

## 复现

从仓库根目录执行以下命令。示例使用 Python 3.11 或更新版本、Verovio 6.4.1 和 Mido 1.3.3。将作者 ZIP 和已生成的 ACFS 数据放在仓库外，按实际位置调整路径。ACFS 目录需包含三个整数 JSONL、词表及 `near_repeats_review.csv`。一并生成的匹配民歌训练集由已有的 ACFS 本地数据构建。[ACFS 生成说明](../../acfs-v261103/split-encode-v1/README.md)

```powershell
$deps = Join-Path $env:TEMP 'acfs-split-encode-deps'
python -m pip install --target $deps verovio==6.4.1 mido==1.3.3
$env:PYTHONPATH = $deps
python -X utf8 .\scripts\prepare_cantopop.py `
  ..\datasets\cantopop-018e885.zip `
  ..\datasets\acfs-v261103\split-encode-v1 `
  ..\datasets\cantopop-018e885\split-encode-v1 --seed 20261118
```
