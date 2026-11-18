# ACFS 划分与编码 V1

2026 年 11 月 18 日。输入为 ACFS `v261103` 的 `screen-v1`，沿用已经审定的 40,795 个窗口。随机种子为 `20261118`。

## 数据规模

| 集合 | 曲目 | 曲目组 | 64 格窗口 | 整数符号 | 观察到的 MIDI 音域 | 超范围窗口 |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
| 训练 `train` | 3,393 | 3,369 | 32,656 | 2,089,984 | 38–86 | 0 |
| 验证 `validation` | 428 | 427 | 4,074 | 260,736 | 48–86 | 0 |
| 测试 `test` | 426 | 423 | 4,065 | 260,160 | 48–86 | 0 |
| 合计 | 4,247 | 4,219 | 40,795 | 2,610,880 | 38–86 | 0 |

三组曲目数占比约为 79.89%、10.08%、10.03%。按卷分别计算，曲目比例距目标最多 0.91 个百分点。窗口允许重叠，同一首曲的窗口始终归属同一集合。

## 分组与重复检查

- 先用相同的 `source_split_group` 和相同的 `clean_melody_hash` 建组。八小节的发声音用「起音格、相对首个发声音的 MIDI 半音差、时值」比对。配对延音已合并，保留同音重新起奏及休止占据的格子。曲目之间命中的 79 种精确窗口指纹全部归入同一集合。
- 近重复扫描用第 1–2、3–4、5–6 和 7–8 小节建立候选索引，每侧允许从一个二小节块略去一个起音。索引至少匹配 4 个带时值的起音和 2 种音高。随后用固定半音位移对整段比较，按 `2 × 相同起音数 / 两段起音数之和` 计算 Dice。相同起音要求格位、移调后音高和时值全等。进入复核表需达到 Dice 0.80、匹配至少 10 个起音和 3 种音高。该方法可能漏掉局部变化遍布小节、起音稀疏或节奏整体错位的异版。
- 异版合并使用同一半音位移及同一个小节偏移下的证据。考虑 Dice 至少 0.90 且至少匹配 12 个起音的窗口。标准化曲名相同的曲目满足以下任一条件即可合并：匹配窗口在两曲各覆盖至少 60% 的候选小节，或至少一个窗口的 Dice 达到 0.95 且匹配 20 个起音。曲名不同的曲目要求两曲各覆盖至少 80% 的候选小节和 60% 的原谱小节。曲名标准化统一字符宽度、去除空白和末尾的括号编号。共记录 7 条异版证据，具体曲目、起始小节和匹配数据见 `group_edges.csv`。
- 按所有分组约束取连通分量。以固定种子打破排序并列，贪心控制各卷及总体的曲目数和窗口数。发布前从整数文件重新解码并重算指纹，跨集合精确匹配为 0。近重复复核表有 55 对曲目，其中 12 对跨集合，已排在表首。`review` 表示需要进一步判断。

## 模型读取

每行 JSON 对应一个窗口。`window_id` 由曲目 ID 和起始小节构成，例如 `beijing:107@20`。`song_id`、`volume`、`group_id`、`split` 和 `start_bar` 可追踪来源。`start_bar` 从 1 开始按 MusicXML 小节顺序计数。用 `songs.csv` 的 `musicxml` 列定位谱文件，`pitch_reference` 留存原简谱音高参照。

```python
import json
from pathlib import Path

dataset = Path("local-worklog/datasets/acfs-v261103/split-encode-v1")
vocab = json.loads((dataset / "vocab.json").read_text(encoding="utf-8"))
with (dataset / "train.jsonl").open(encoding="utf-8") as stream:
    for line in stream:
        row = json.loads(line)
        tokens = row["tokens"]
        assert len(tokens) == vocab["window_slots"] == 64
        assert all(0 <= token < vocab["size"] for token in tokens)
```

每格是一个十六分音符。`REST=0` 开始一个休止事件，`HOLD=1` 延长上一格的发声或休止状态。相邻但独立记写的休止分别以 `REST` 开始。音高 ID 为 `MIDI − 38 + 2`，按训练窗口的实际最小和最大 MIDI 值取连续半音，共 51 类。配对延音的续段全写为 `HOLD`，相邻同音重奏写新的音高 ID。首格为音高 ID 或 `REST`。音高采用 MusicXML 的 MIDI 坐标，简谱参照作为元数据留档。

超范围时整窗排除，保持原音高、词表和曲目划分。`out_of_range.csv` 逐窗列出超范围 MIDI 值及起音次数，`summary.json` 按集合分别统计。验证集和测试集本次各有 0 个超范围起音，40,795 个候选窗口全部保留。

## 文件与验证

| 文件 | 用途 |
| --- | --- |
| `train.jsonl`、`validation.jsonl`、`test.jsonl` | 模型输入和窗口级追踪字段 |
| [vocab.json](vocab.json) | MIDI 与整数 ID 映射 |
| [songs.csv](songs.csv)、[volume_split.csv](volume_split.csv) | 曲目清单、各卷的实际划分 |
| [group_edges.csv](group_edges.csv)、[exact_repeats.csv](exact_repeats.csv) | 分组依据和精确移调匹配的位置 |
| [near_repeats_review.csv](near_repeats_review.csv) | 每对近重复曲目最强的匹配和位置 |
| [summary.json](summary.json)、[out_of_range.csv](out_of_range.csv) | 输入 SHA-256、分区和音域统计、排除明细 |
| [samples/index.html](samples/index.html) | 20 首真实曲目的五线谱、MusicXML 和 MIDI |
| [samples/round_trip.json](samples/round_trip.json) | 原小节事件、整数序列和解码事件的对应关系 |

20 首样例覆盖 14 卷和三个集合。16 首样例含休止，7 首以休止开头，7 首含相邻休止，12 首含相邻同音重奏。逐首对照原 MusicXML 文件，核对原事件、解码的起音位置、音高、时值和休止。生成的 MusicXML 与 MIDI 均通过回读，MIDI 使用固定 90 BPM 和钢琴音色。

40,795 个真实窗口均无跨小节配对延音。单独的 [合成延音测试](samples/synthetic_tie/round_trip.json) 验证了延音跨过小节边线后继续保持、同音立即重奏时产生新起音，另附 [五线谱](samples/synthetic_tie/cross_bar_tie.svg) 和 [MIDI](samples/synthetic_tie/cross_bar_tie.mid)。

复现命令、依赖版本和检查结果见 [工作日志](../../../WORKLOG.md)。
