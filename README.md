# melody-diffusion-study

## ACFS 八小节数据

完整数据在 [ACFS 划分与编码 V1](data/acfs-v261103/split-encode-v1)。模型可直接读取 [训练集](data/acfs-v261103/split-encode-v1/train.jsonl)、[验证集](data/acfs-v261103/split-encode-v1/validation.jsonl)和[测试集](data/acfs-v261103/split-encode-v1/test.jsonl)，分别有 32,656、4,074 和 4,065 个窗口。另附[整数序列校验和](data/acfs-v261103/encoded_SHA256SUMS)、[曲目划分索引](data/acfs-v261103/song_splits.csv)和[窗口索引](data/acfs-v261103/window_index.csv)。

`scripts/encode_folk_windows.py` 读取 ACFS `screen-v1` 的审计结果与候选起点。程序按曲目分组划分，确定训练词表，验证抽查样例，再输出三个集合的整数序列。`screen-v1` 的同级目录需包含已校验的乐谱包与元数据 ZIP，供抽样核对原谱。

以下命令从课程根目录运行。使用 Python 3.11 或更新版本。Windows 上请将 Verovio 安装在纯 ASCII 路径下，例如当前用户的临时目录。

```powershell
$deps = Join-Path $env:TEMP 'acfs-split-encode-deps'
python -m pip install --target $deps verovio==6.4.1 mido==1.3.3
$env:PYTHONPATH = $deps
python -X utf8 .\melody-diffusion-study\scripts\encode_folk_windows.py `
  .\local-worklog\datasets\acfs-v261103\screen-v1 `
  .\local-worklog\datasets\acfs-v261103\split-encode-v1 --seed 20261118
python -m unittest discover -s .\melody-diffusion-study\scripts -p test_encode_folk_windows.py
```

复跑时使用一个新的输出目录。模型读取 `train.jsonl`、`validation.jsonl` 或 `test.jsonl`，逐行解析 `tokens` 为长度 64 的整数序列，并从 `vocab.json` 读取类别数 `size`。`group_id` 用于曲目组级统计，`start_bar` 从 1 起按原谱顺序计数。`songs.csv` 保留曲目分组、原谱位置和简谱参照。`summary.json` 记录输入校验和、分区统计、音域和验证结果。浏览样例时打开 `samples/index.html`。

## Cantopop 八小节数据

完整数据在 [Cantopop 划分与编码 V1](data/cantopop-018e885/split-encode-v1)。模型可直接读取 [训练集](data/cantopop-018e885/split-encode-v1/train.jsonl)、[验证集](data/cantopop-018e885/split-encode-v1/validation.jsonl)和[测试集](data/cantopop-018e885/split-encode-v1/test.jsonl)，分别有 1,323、181 和 180 个窗口。H3 实验另有 [匹配的 ACFS 训练集](data/cantopop-018e885/split-encode-v1/acfs_matched_train.jsonl)，含 1,323 个窗口。数据与 ACFS 使用相同的字段、64 格长度和整数 ID。[校验和](data/cantopop-018e885/SHA256SUMS) · [上游许可](data/cantopop-018e885/LICENSE)
