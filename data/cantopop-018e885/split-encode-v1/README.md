# Cantopop 划分与编码 V1

2026 年 11 月 18 日。该版保留作者转录谱的 2/4 小节线，用于 64 格的粤语流行旋律实验准备。已确认同一作品不同转录可能标出不同拍号，现阶段以**转录谱标注的 2/4**解释该数据。[拍号复核](meter_review.md)

## 数据规模

| 集合 | 曲目 | 曲目组 | 窗口 | 整数符号 |
| --- | ---: | ---: | ---: | ---: |
| 训练 | 25 | 25 | 1,323 | 84,672 |
| 验证 | 3 | 3 | 181 | 11,584 |
| 测试 | 3 | 3 | 180 | 11,520 |
| 合计 | 31 | 31 | 1,684 | 107,776 |

输入是 Jason Y H Lee 的 Cantopop Corpus，提交 `018e88540b917c40dc3f7248ed98908208544b54`，共 105 首，作者仓库标注 CC BY 4.0。[作者仓库](https://github.com/jasonleeubc/Cantopop-corpus/tree/018e88540b917c40dc3f7248ed98908208544b54) · [本地 ZIP](../../pop-candidates-2026-11-18/cantopop-018e885.zip)

逐谱提取单一 `**kern` 旋律列，沿有明确边线的原谱小节读取。连续 8 小节标为 2/4 的窗口共 3,294 个，其中 2,514 个满足每小节 8 格及精确量化，1,697 个满足有发声、延音配对及边界完整。这 1,697 个窗口都落在 MIDI 38–86。正式解析识别了写在音名后的附点，按 Verovio 和 Humlib 的读法计算时值，同时剔除一处重复的延音结束标记。相较初筛净增 23 个窗口。[筛选统计](screen_audit.json) · [Humlib 时值实现](https://github.com/craigsapp/humlib/blob/master/src/Convert-rhythm.cpp#L61-L70)

跨曲目精确碰撞来自单音或三次起音后长休止的片段。剔除 13 个这类窗口，其中 3 个也与民歌训练集同一窗口移调后完全相同。剔除后流行集合之间和民歌、流行两库之间的精确移调重复均为 0。近重复筛查未产生候选。划分使用固定种子 `20261118`，同一曲目的全部保留窗口在同一集合。[剔除清单](discarded_windows.csv) · [跨库审核](acfs_overlap_review.csv)

## 模型输入

- `train.jsonl`、`validation.jsonl`、`test.jsonl` 与 ACFS 使用相同格式。逐行读取 `tokens`，每行 64 个整数。`REST=0` 进入休止，`HOLD=1` 延长当前状态，音高类 ID 2–50 对应 MIDI 38–86。连续同音的独立起音各有一个音高 ID，配对延音的后续位置为 `HOLD`。词表与 ACFS 逐字节相同。[共享词表](vocab.json)
- `songs.csv` 保存 105 首的来源和筛选结果。`windows.csv` 可由 `window_id` 找到 ZIP 内源谱、物理小节序号、原谱打印小节号、源文件行号及稳定的谱面调性标注。物理序号从 1 开始，弱起计入序号。1,655 个保留窗口有单一谱面调性标注。[曲目清单](songs.csv) · [窗口索引](windows.csv)
- `samples/index.html` 提供 20 首各 1 段的五线谱、MusicXML 和 MIDI。样例跨三个集合，10 段含跨小节延音，13 段含相邻同音重奏。MIDI 是整数数据解码后以固定 90 BPM 演奏的回放。[查看回放样例](samples/index.html)

## H3 匹配训练子集

`acfs_matched_train.jsonl` 来自原 ACFS `train.jsonl`。从不同民歌曲目组各取一首，按流行训练曲目逐首配对，使每对曲目的窗口数相等。筛选民歌候选时排除了跨库匹配涉及的训练组，也排除了 ACFS 原近重复表中涉及跨集合候选的 12 个训练组。匹配用每段的起音数、平均起音 MIDI、音高跨度和休止格数，按流行训练侧的标准差缩放。[逐曲配对](training_pairs.json)

| 指标 | Cantopop 训练 | ACFS 匹配训练 |
| --- | ---: | ---: |
| 曲目组 | 25 | 25 |
| 窗口 | 1,323 | 1,323 |
| 整数符号 | 84,672 | 84,672 |
| 每窗平均起音 | 22.58 | 22.38 |
| 平均起音 MIDI | 60.16 | 62.05 |
| 每窗平均音高跨度（半音） | 11.65 | 11.68 |
| 每窗平均休止格 | 14.32 | 6.81 |

H3 的训练输入对应 `train.jsonl` 和 `acfs_matched_train.jsonl`。正式评测沿用现有 ACFS 留出集，对两模型提供同一受扰输入。两侧的音高跨度与起音数已接近，绝对音区和休止仍存在差异。训练先验文件附有两侧的 12 音级频数、相邻发声音程和最佳五音模板的描述性覆盖率。按曲目从干净输入拟合一个模板，先最大化起音覆盖，再用时值覆盖处理并列。此统计按训练窗口加权，包含重叠产生的重复观察。正式五声性评测仍用预先核定的目标集合。[训练先验统计](training_priors.json) · [运行汇总](summary.json)

## 复跑与验证

从课程根目录运行。实测 Python 3.13.11、Verovio 6.4.1、Mido 1.3.3。Windows 上请把 Verovio 装在纯 ASCII 路径下。输出目录须使用新路径。

```powershell
$deps = Join-Path $env:TEMP 'acfs-split-encode-deps'
python -m pip install --target $deps verovio==6.4.1 mido==1.3.3
$env:PYTHONPATH = $deps
python -X utf8 .\melody-diffusion-study\scripts\prepare_cantopop.py `
  .\local-worklog\datasets\pop-candidates-2026-11-18\cantopop-018e885.zip `
  .\local-worklog\datasets\acfs-v261103\split-encode-v1 `
  .\local-worklog\datasets\cantopop-018e885\split-encode-v1 --seed 20261118
python -m unittest discover -s .\melody-diffusion-study\scripts -p 'test_*.py'
```

全量 1,684 个窗口逐一核对整数回读的音高、起音、休止及时值。另用 Verovio 直接导入原 `.krn` 独立核对 54 个候选片段，包括所有 24 个涉及后置附点的候选窗口。26 项测试通过。从相同输入和种子在新目录复跑，77 个程序生成文件的 SHA-256 逐一相同。独立核验了两库精确重复、曲目组分区、窗口索引完整性，以及 1,323 个匹配民歌窗口与原训练记录的对应关系。
