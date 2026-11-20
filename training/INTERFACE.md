# 训练组与评测组的文件接口

本文件说明评测组提供的受扰输入（xt 文件）格式，以及训练组返回的重建、一次预测消融和无条件生成结果格式。所有文件为 UTF-8 JSONL，每行一个 JSON 对象。整数符号沿用 `data/*/split-encode-v1/vocab.json`：`0` 为 REST，`1` 为 HOLD，`2…50` 为起音，MIDI 音高 = ID + 36；每段 64 格。

## 1. 输入：xt 文件

每行一个受扰实例。

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `case_id` | 字符串 | 文件内唯一 |
| `window_id` | 字符串 | 例如 `beijing:114@19` |
| `song_id` | 字符串 | |
| `group_id` | 字符串 | |
| `split` | 字符串 | `validation` 或 `test` 等 |
| `t` | 整数 0–100 | 扩散步；`rho = t/100` |
| `rho` | 数 | `t/100` |
| `noise_seed` | 整数 | 生成该 `noisy_tokens` 所用的种子，仅留档 |
| `noisy_tokens` | 64 个整数 | 受扰序列 x_t，取值 0–50 |
| `clean_tokens` | 64 个整数，可选 | 供评测打分；**训练组代码读取时直接丢弃** |

读取规则（`training/sample.py` 的 `read_xt`）：

- 只复制上表前九个字段，其余字段（包括 `clean_tokens`）一律不进入内存中的实例，模型只看到 `noisy_tokens` 和 `t`。测试 `test_sampling_is_reproducible_and_ignores_clean_tokens` 把 `clean_tokens` 全部改写后，输出逐字节不变。
- 缺字段、`case_id` 重复、`t` 不在 0–100、符号不是 64 个 0–50 的整数时报错退出。
- `noisy_tokens` 首格可以是 HOLD，原样输入模型。
- `t = 0` 的实例不经过网络，三种方法都原样返回输入。

## 2. 输出：重建与一次预测消融

文件名由命令行指定，每行一次采样。同一行的字段：

| 字段 | 说明 |
| --- | --- |
| `case_id`、`window_id`、`t` | 原样取自输入 |
| `method` | `diffusion`：从 x_t 逐步反向采样到 0；`oneshot_argmax`：同一网络在 x_t 上一次输出 f(x_0 \| x_t, t)，逐格取最大；`oneshot_sample`：同一分布逐格独立抽样 |
| `model_dataset` | `acfs_full`、`acfs_matched` 或 `cantopop` |
| `train_seed` | 训练种子 |
| `checkpoint_sha256` | 所用检查点文件的 SHA-256 |
| `sample_index` | `diffusion`、`oneshot_sample` 为 0、1、2；`oneshot_argmax` 是确定性的，只输出 0 |
| `sample_seed` | 该次采样的随机数种子；`oneshot_argmax` 为 `null` |
| `raw_generated_tokens` | 采样得到的 64 个整数，未作任何改动 |
| `decoded_tokens` | 与 raw 相同，唯一例外是首格 HOLD 改为 REST |
| `leading_hold_repaired` | 是否发生了上述首格改写 |
| `has_sounding_note` | 解码后是否至少有一个发声音符；`false` 应在评测中记为无效 |

除首格 HOLD→REST 外，不对采样结果作任何修正（不删除五声音集合外的音，不改节奏）。

行的顺序：按输入文件的行序；同一实例内先按 `method`（diffusion、oneshot_argmax、oneshot_sample），再按 `sample_index`。

### 同名 `.meta.json`

与输出同名、扩展名为 `.meta.json`，记录：输入文件路径和 SHA-256、实例数、检查点路径、SHA-256 和步数、方法与每个方法的采样次数、`base_seed` 与每次采样种子的派生规则、`batch_size`、处理顺序规则以及每一批的 `case_id` 列表、设备与 PyTorch 版本、耗时。

处理顺序：实例按 `t` 从大到小稳定排序（同 `t` 保持文件行序），每 `batch_size` 个为一批。每个 (method, sample_index) 用一个设备随机数生成器，种子为 `stable_seed(base_seed, method, sample_index)`，按上述批次顺序依次消耗。同一检查点、同一输入、同一 `batch_size` 和 `base_seed` 在同一设备类型上可复现。

## 3. 输出：无条件生成

每行一段生成旋律。除第 2 节的字段外（`window_id` 为 `null`，`t = 100`，`method = diffusion`，`sample_index` 为 0–99），另有：

| 字段 | 说明 |
| --- | --- |
| `initial_tokens` | 起点 x_100，64 格各自从 51 类均匀独立抽样 |
| `nn_hamming` | 与该模型训练集全部窗口的最小汉明距离（逐格比较 `decoded_tokens`，0–64） |
| `nn_window_id` | 取得该最小值的训练窗口；并列时取训练文件中最靠前的一个 |
| `nn_transposed_hamming` | 把生成段的所有起音整体移调 −12…+12 个半音后的最小汉明距离；移出 MIDI 38–86 的音视为不匹配 |
| `nn_transposed_window_id` | 对应的训练窗口；并列时先取最靠前的窗口，同一窗口再取最小的移调量 |
| `nn_transposed_shift` | 取得最小值的移调半音数（施加在生成段上） |

`.meta.json` 记录检查点、种子派生规则、参照训练文件及其 SHA-256 和窗口数、距离定义。

## 4. 轨迹文件

`trajectories.json` 记录从 ACFS 验证集按固定种子选出的 3 个窗口（规则写在文件的 `selection_rule` 字段）：

- `forward_states`：下标 s 处是 x_s（s = 0…100），按单步转移 Q_s 逐步前推得到，同一条轨迹；
- `reverse_from_050`、`reverse_from_100`：分别从前推轨迹在 t = 50 和 t = 100 的状态出发，用模型反向采样到 0，`states` 从起点步列到 0；
- `exports`：t = 0、10、25、50、75、90、100 处状态导出的 MIDI 与 MusicXML 路径（导出时首格 HOLD 写成 REST，记在 `leading_hold_repaired`）。

## 5. 临时 xt

正式 xt 到达前，训练组用 `sample.py make-xt` 从 ACFS `validation.jsonl` 生成临时文件，文件名带 `provisional`：随机种子 20261120 选 100 个窗口，t = 10、25、50、75、90，每档 2 次加噪，共 1,000 个实例，附 `clean_tokens`。每个实例的 `noise_seed = stable_seed("provisional-xt", window_id, t, 重复序号)`，用闭式 Q̄_t 一次加噪。它只用于跑通流程，不用于任何结论。

`stable_seed(*parts)`：把各部分转成字符串后用 `|` 连接，取 UTF-8 编码的 SHA-256 前 4 字节（大端）并保留低 31 位。
