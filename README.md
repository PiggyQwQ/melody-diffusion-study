# 扩散模型实验室

本项目研究离散扩散模型给旋律加噪、再逐步恢复时，原来的音高、节奏和五声性保留多少。实验包括前向加噪的理论校验、民歌旋律重建，以及民歌与粤语流行训练语料对重建结果的影响。

## 数据

每个样本是连续八小节的单声部旋律。每小节按两拍、每拍四格编码，一共 64 个十六分格。同一首曲目的片段放在同一个集合。

| 数据 | 曲目数 | 训练窗口 | 验证窗口 | 测试窗口 |
| --- | ---: | ---: | ---: | ---: |
| 中国民歌 | 4,247 | [32,656](data/acfs-v261103/split-encode-v1/train.jsonl) | [4,074](data/acfs-v261103/split-encode-v1/validation.jsonl) | [4,065](data/acfs-v261103/split-encode-v1/test.jsonl) |
| 粤语流行 | 31 | [1,323](data/cantopop-018e885/split-encode-v1/train.jsonl) | [181](data/cantopop-018e885/split-encode-v1/validation.jsonl) | [180](data/cantopop-018e885/split-encode-v1/test.jsonl) |

逐行读入 JSONL 的 `tokens` 即可得到模型输入。0 表示进入休止，1 表示延续当前音高或休止，2–50 表示新起奏，依次对应 MIDI 音高 38–86。相邻两次同音起奏各用一次音高编号。[共享词表](data/acfs-v261103/split-encode-v1/vocab.json)

- **中国民歌**来自 *Anthology of Chinese Folk Songs*（ACFS），底本为《中国民间歌曲集成》。按曲目组划分，考虑同曲异版和移调后相同的片段，记录跨集合近重复复核结果。数据目录附曲目与窗口索引、审计和回放样例。上游发行说明列有下载和校验入口，底本权利按其声明核对。[数据目录](data/acfs-v261103/split-encode-v1) · [整数序列校验和](data/acfs-v261103/encoded_SHA256SUMS) · [上游发行说明](https://github.com/m-july/Anthology-of-Chinese-Folk-Songs/releases/tag/v261104)
- **粤语流行**来自 Jason Y H Lee 的 *Cantopop Corpus*。这里从上游转录中提取单声部，筛选可精确量化的八小节窗口，再分组和编码。数据目录附拍号复核、跨库重复检查和回放样例。[数据目录](data/cantopop-018e885/split-encode-v1) · [校验和](data/cantopop-018e885/SHA256SUMS) · [上游转录](https://github.com/jasonleeubc/Cantopop-corpus/tree/018e88540b917c40dc3f7248ed98908208544b54) · [CC BY 4.0 许可](data/cantopop-018e885/LICENSE)
- **匹配民歌训练集**从中国民歌的训练集合选取 1,323 个窗口，与粤语流行训练集等规模。其底本仍来自 ACFS。[匹配训练集](data/cantopop-018e885/split-encode-v1/acfs_matched_train.jsonl)

`scripts/` 包含筛选、划分、重复检查、整数编码和抽样回放的实现及测试。数据目录中的处理记录沿用实验时的输入目录结构，原始谱包和转录可通过上述上游入口获取。[数据处理脚本](scripts)

## 训练与评测

正式训练包括完整民歌、等规模民歌和粤语流行三套训练材料，每套各训练三个随机种子。比较训练语料时，两种模型处理同一份加噪后的民歌，便于比较在相同输入下的输出差异。[训练与推理命令](https://github.com/PiggyQwQ/melody-diffusion-study/blob/25a99389b538cb9494bdfd2f47df4eb914bff9e4/training/README.md) · [九份权重及校验清单](https://github.com/PiggyQwQ/melody-diffusion-study/blob/25a99389b538cb9494bdfd2f47df4eb914bff9e4/training/checkpoints/manifest.json)

评测说明涵盖前向加噪、民歌恢复、训练语料对比与生成检验，并提供汇总表、指标说明和统计图。[评测方法和结果](evaluation/README.md)

## 离线演示

网页可以试听原旋律、加噪和模型处理后的版本，查看逐步运行的轨迹，比较两种训练语料的模型，并浏览民歌和粤语流行曲库。界面的 `n/64` 比较每格实际发声的音高或休止，正式评测使用各自定义的指标。粤语曲目作为输入的比较属于网页补充示例。

演示源码和 Windows 桌面封装均在 `visualization/`。构建完整演示需要保存的案例、加噪输入、模型输出和逐步轨迹。网页构建和桌面打包的命令及文件要求在使用说明中逐项列出。[演示构建与 Windows 运行说明](visualization/README.md)

## 许可

本仓库项目代码采用 MIT 许可。曲库与第三方组件按各自来源说明和许可使用。[项目许可](LICENSE) · [VexFlow 许可](visualization/vendor/VexFlow-LICENSE.txt)
