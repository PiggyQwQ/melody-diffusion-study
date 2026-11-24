# 结果字段说明

## 实验和方法标识

| 标识 | 含义 |
| --- | --- |
| H1、H2、H3 | 依次为前向加噪理论校验、民歌重建、匹配训练语料对照。表名和方法键保留对应小写前缀 |
| `acfs_full`、`acfs_matched`、`cantopop` | 完整民歌训练集、与粤语流行规模匹配的民歌训练集、粤语流行训练集 |
| `s0`、`s1`、`s2` | 训练随机种子 0、1、2。如 `h2-acfs_full-s0__diffusion` 表示种子 0 的完整民歌模型采用多步扩散 |
| `diffusion`、图中 M | 多步反向扩散采样 |
| `oneshot_argmax`、`oneshot_sample` | 同网络单步预测，各格取后验概率最大的符号或按预测分布抽样 |
| `b0_noisy`、图中 B0 | 直接解码受扰输入 |
| `b1_<训练集>_argmax`、`b1_<训练集>_sample`、图中 B1 | 单格贝叶斯基线，先验为训练集符号频率，各格取后验概率最大的符号或按后验抽样 |
| `b2_<训练集>_ffbs`、`b2_<训练集>_viterbi`、图中 B2 | 一阶马尔可夫基线，分别用前向滤波与后向采样（FFBS）或 Viterbi 最可能路径解码。平滑参数用验证集选定 |
| `clean_ref` | 干净原曲的参照统计 |
| `shuffle_clean`、`shuffle_noisy` | 保持各自输入的起音位置和时值，打乱音高次序。两者分别以原曲和受扰序列为输入，用于诊断指标对旋律次序的灵敏度 |

`method_key` 是方法标识。单种子表中的标识含训练集和种子，汇总表把同类方法的种子结果合并。`t` 是离散加噪时间步。`rho` 是累计混合量，在本实验中等于 `t / 100`。图中的 `(a)` 至 `(d)` 为同张图中的子图编号。

## 前向校验

| 字段或图中记号 | 含义 |
| --- | --- |
| `channel` | `pitch_only_strict_p0_eq_1` 为全五声音级民歌窗口，`pitch_only_frozen_test` 为全部冻结测试窗口，`pitch_only_low_pent_control` 为初始覆盖率小于 0.30 的粤语流行对照窗口 |
| `p0`（p₀）、`n_windows`、`n_onsets` | 初始五声音级比例、窗口数、纳入的原始起音数 |
| `vocab_k`（K）、`p_pitch_tokens`（P）、`m_pent_tokens`（m） | 词表符号数 51、起音音高符号数 49、属于目标五声音高的符号数 21 |
| `a0_all`、`b0_all` | 干净窗口所有格中为目标五声音高符号的比例、为任意起音音高符号的比例 |
| `n_clean_onsets`、`a0_clean_onsets` | 固定原曲起音位置总数、其中属于目标五声音高的比例 |
| `mismatch` | 随机替换后符号与原符号不同的格数占比 |
| `a_all`、`a_clean_onsets`（图中 aₜ） | 固定分母的五声音高比例。分母分别为全部格、原曲起音所在格 |
| `b_all` | 所有格中当前为起音音高符号的比例 |
| `cond_pooled`（图中 c(ρ)） | 合并所有窗口的格，五声起音数除以当前全部起音数 |
| `win_mean_raw` | 先各窗计算当前起音中的五声比例，再取窗口均值 |
| `decoded_f_on`、`decoded_f_dur`、`valid_rate` | 解码后按起音、按时值的五声覆盖比例和含发声音符的有效比例 |
| `theo_*`、`theory`、`sim_*`、`abs_err` | 理论值、模拟统计量、理论与模拟均值的绝对误差 |
| `sim_std_across_runs`、`*_q025`、`*_q975`、`theo_single_run_se` | 模拟重复间标准差，重复结果的 2.5% 和 97.5% 分位数，以及单次模拟均值的理论标准误 |

`h1_summary.json` 的 `vocab_constants` 给出词表规模和理论极限。`times` 是模拟档位，`n_runs` 是每档模拟次数，`max_abs_errors` 给出各通道跨档最大绝对误差。`seconds` 是当次运行用时。

## 重建指标

| 字段 | 含义 |
| --- | --- |
| `valid`、`leading_hold_repaired` | 解码有发声音符的指示值、首格延音被修正为休止的指示值 |
| `slot_acc_all` | 全部 64 格的编码符号准确率 |
| `slot_acc_pitch`、`slot_acc_rest`、`slot_acc_hold` | 分别仅在原曲起音、休止、延音格上的符号准确率 |
| `slot_acc_corrupted`、`slot_acc_uncorrupted` | 分别仅在加噪后符号实际改变、未实际改变的位置上的准确率 |
| `onset_pitch_hit`、`onset_pc_hit` | 原曲起音位置的发声音高命中率。前者比对 MIDI 音高，后者比对不计八度的音级，输出休止均计错 |
| `sounding_slot_pitch_acc` | 解码并展开延音后，仅在原曲发声格上的音高命中率 |
| `onset_precision`、`onset_recall`、`onset_f1` | 起音位置的精确率、召回率、调和平均 |
| `onset_count_ratio` | 重建起音数除以原曲起音数 |
| `rest_ratio`、`rest_ratio_clean`、`rest_ratio_abs_err` | 解码后输出休止格占比、原曲休止格占比、两者的绝对差 |
| `mean_note_dur` | 每个音符的平均时值，以格为单位 |
| `f_on`、`f_dur`（图中 Fₒₙ、F𝒹ᵤᵣ） | 固定五声音级集合的覆盖率。起音计数与发声时长加权两种口径均以有声部分为分母 |
| `f_on_clean`、`f_dur_clean` | 干净原曲的两种固定集合覆盖率 |
| `f_on_alt_song_s`、`f_dur_alt_song_s` | 使用该曲在加噪前根据其干净窗口确定的最佳五音集合，计算重建覆盖率 |
| `f_on_best_pent`、`f_dur_best_pent` | 在固定五音模板的 12 种整体移调中，分别取输出起音覆盖、时值覆盖的最大值 |
| `black_key_frac` | 黑键音级 C♯、D♯、F♯、G♯、A♯ 的起音占比 |
| `pc_js_on`、`pc_js_dur` | 输出与原曲的 12 音级 JS 距离，分别按起音计数、按时值加权 |
| `iv_js`、`contour_dist_js` | 输出与原曲的相邻音符音程分布、升降平轮廓分布的 JS 距离 |
| `contour_dir_acc` | 在相邻的两个原曲起音位置比较发声音高升降平，输出两处均发声且方向一致时计对 |
| `semitone_frac`、`stepwise_le3_frac`、`large_leap_gt7_frac` | 相邻音符绝对音程分别等于 1、不超过 3、超过 7 个半音的比例 |
| `mean_abs_interval`、`pitch_span` | 相邻音符平均绝对音程、最高与最低音的音高差，均以半音为单位 |
| `pc_js_to_train_*` | 输出起音音级分布与对应训练集分布的 JS 距离 |
| `pc_js_drift_folk_minus_pop` | 到粤语流行训练集的音级 JS 距离减去到匹配民歌训练集的距离，正值表示更接近民歌 |

固定五音集合在图中记为 `S_C = {C, D, E, G, A}`。MIDI 音高用整数表示相差几格半音，音级按 MIDI 音高对 12 取余。JS 距离为以 2 为对数底的 Jensen–Shannon 散度的平方根，范围 0 到 1。同一窗没有某指标所需的发声或相邻音符时，相应指标为空值。配对比较仅使用双方都有值的组。

## 汇总与配对统计

- `*_mean` 和 `*_sd` 为曲目组均值与标准差。`*_ci_lo`、`*_ci_hi` 为 2,000 次自助重采样的 95% 百分位置信区间，重采样种子为 `20261125`。`n_groups` 为曲目组数，`*_n_valid_groups` 为该指标含有效数据的组数。
- 配对表的 `method_a`、`method_b` 是顺序固定的第一和第二方法。`metric` 是指标名，`mean_a`、`mean_b` 是均值。`diff_mean` 是共同曲目组逐对相减的均值，`n_paired` 是配对组数。`ci_lo`、`ci_hi` 是差值的 95% 置信区间。`p_value` 是将零作为基准的双侧自助检验概率，`frac_a_gt_b` 是第一方法指标值大于第二方法的配对比例。
- `chance_corrected_gain` 为两方法均值之差除以 `1 − 第二方法均值`，仅用于相应准确率或覆盖指标。第二方法均值达到 0.98 时空置。
- `comparison` 中以 `folk_minus_pop_` 开头的行表示同一种方法下匹配民歌减粤语流行，后缀为多步扩散、单格抽样、马尔可夫后向采样或同网络单步抽样。`did_diffusion_minus_b1_drift` 表示扩散模型的语料差减去单格频率抽样基线的语料差，等价于先求每种方法中的民歌减粤语流行，再对两种方法作差。
- `corpus_distribution_js.csv` 先汇总语料的计数直方图再计算距离。列中的 `pc` 是音级，`iv` 是相邻音符的有符号音程，`bigram` 是连续两个音程的组合，`contour` 是升降平轮廓。`vs_clean` 以干净测试窗口为参照，`vs_train_*` 以相应训练集为参照。`pooled_f_on`、`pooled_black_key_frac` 是合并全部起音计数后的占比。
- `training_priors_comparison.json` 中 `n_windows`、`mean_onsets_per_window` 为训练窗数、每窗起音数。`f_on_c_pent`、`f_dur_c_pent` 为固定 C 宫五声音级集合的训练语料覆盖。训练分布数组 `pc_on_dist`、`pc_dur_dist` 按 C、C♯ 到 B 的半音顺序排列。`iv_dist` 按小于 −12、−12 到 +12、大于 +12 半音归箱。`contour_dist` 按下降、同音、上升排列。两项 `js_acfs_matched_vs_cantopop_*` 分别比较匹配民歌和粤语流行的起音音级分布与相邻音程分布。

## 生成及最近邻检验

- `generation_summary.csv` 的 `group` 分为完整民歌训练窗口、冻结民歌测试窗口、总计 300 段生成样本和三个各 100 段的训练种子样本。`n_samples` 是该组窗口或样本数，`valid_rate`、`leading_hold_repair_rate` 是有效比例、首格修正比例。`unique_exact_count`、`unique_exact_rate` 统计完全不同的符号序列数量与比例。
- `pairwise_hamming_*` 汇总组内成对比较的不同符号格数。组内超过 500 个窗口时，从中抽取 300 个不同窗口进行计算。`pc_entropy_bits` 和 `iv_entropy_bits` 是以比特为单位的起音音级、相邻音程香农熵。`unique_interval_bigrams` 是连续两个音程的不同组合数。`n_onsets` 和 `unique_pitches` 是单窗起音数和不同音高数。其余均值、标准差及 JS 距离依照前述符号和参照定义。
- `nn_plain_median`、`nn_transposed_median` 为各组到训练集的普通、允许整体移调的最近邻汉明距离中位数。`nn_le_4_either` 为任一度量距离不超过 4 格的样本个数。
- `memorization_report.json` 汇总生成与未见验证窗口的普通和移调距离分布。`n`、`min`、`max` 为样本数、最小值、最大值，`p05` 等字段表示对应百分位点。`count_eq_0` 为最近距离等于零的数量，`count_le_*` 为距离不超过阈值的数量。`tests` 给出 Mann–Whitney 秩检验、Kolmogorov–Smirnov（KS）两样本检验及阈值比例的 Fisher 精确检验，带 `_p` 的字段为检验概率。`distributions` 汇总整个民歌训练集、冻结测试集、生成样本的起音音级和相邻音程分布，数组顺序与训练分布相同。
