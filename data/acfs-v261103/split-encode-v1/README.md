# ACFS v261103 划分与编码 V1

数据源为《中国民间歌曲集成》的 ACFS `v261103`，由上游 `v261104` 发行版提供。筛选版本为 `acfs-screen-v1`，划分种子为 `20261118`。[上游发行版](https://github.com/m-july/Anthology-of-Chinese-Folk-Songs/releases/tag/v261104)

| 集合 | 曲目数 | 64 格窗口数 |
| --- | ---: | ---: |
| `train` | 3,393 | 32,656 |
| `validation` | 428 | 4,074 |
| `test` | 426 | 4,065 |
| 合计 | 4,247 | 40,795 |

- [song_splits.csv](song_splits.csv) 记录每首曲目的卷、曲目组、分区和窗口数。
- [window_index.csv](window_index.csv) 记录三个集合中所有窗口的位置，顺序与各自的整数文件一致。`start_bar` 从 1 开始，按 MusicXML 中的小节顺序计数。
- [vocab.json](vocab.json) 记录词表。`REST=0`、`HOLD=1`，MIDI 38–86 映射为 ID 2–50。
- [volume_split.csv](volume_split.csv) 和 [summary.json](summary.json) 记录各卷分区、训练音域、验证结果与输入校验值。跨集合精确移调匹配为 0，近重复筛查另有 12 对跨集合候选待复核。
- [encoded_SHA256SUMS](encoded_SHA256SUMS) 可核对本地生成的三个整数 JSONL 文件。

## 生成模型输入

从上游发行版下载元数据 ZIP 和 11 个省份乐谱 ZIP，将它们放进克隆仓库的同级目录 `acfs-v261103`，保持 ZIP 原样。以下 PowerShell 命令在仓库根目录执行，使用 Python 3.11 或更新版本。Windows 上请把 Verovio 安装在纯 ASCII 路径下。

```powershell
$archives = Join-Path (Split-Path -Parent (Get-Location).Path) 'acfs-v261103'
$screen = Join-Path $archives 'screen-v1'
$encoded = Join-Path $archives 'split-encode-v1'
$deps = Join-Path $env:TEMP 'acfs-split-encode-deps'
python -m pip install --target $deps verovio==6.4.1 mido==1.3.3
$env:PYTHONPATH = $deps
python -X utf8 .\scripts\screen_folk_scores.py $archives $screen
python -X utf8 .\scripts\encode_folk_windows.py $screen $encoded --seed 20261118

Get-Content .\data\acfs-v261103\split-encode-v1\encoded_SHA256SUMS | ForEach-Object {
    $expected, $name = $_ -split '\s+', 2
    $actual = (Get-FileHash -Algorithm SHA256 -LiteralPath (Join-Path $encoded $name)).Hash
    if ($actual -ne $expected) { throw "SHA-256 mismatch for $name" }
}
```

模型读取生成目录下的 `train.jsonl`、`validation.jsonl` 或 `test.jsonl`。每行的 `tokens` 为长度 64 的整数数组，`window_id` 可在窗口索引中定位。复跑时可选择新的编码输出目录。
