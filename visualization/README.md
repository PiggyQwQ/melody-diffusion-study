# 扩散模型实验室

构建完成后，保留整个输出目录，用 Chrome 或 Edge 打开其中的 `index.html`。网页使用本地数据、刻谱组件和合成音色，可断网运行。顶部的「全屏演示」可切换投影显示。

- **模型效果**可选 12 个民歌案例。原旋律固定显示，滑杆切换加噪程度并更新两路结果和重合度。
- **运行过程**可选三条实验轨迹。拖动时间线观察任一步，或按「播放过程」连续看变化。各状态可以单独试听。
- **模型对比**可在中国民歌与粤语流行之间切换，每种输入有三首可选。曲线显示加噪后和两份模型输出各自与原旋律的重合度。
- **旋律库**提供全部 4,247 首中国民歌和 31 首粤语流行的合格八小节窗口，可按曲名或来源搜索，每页显示 8 首。试听中翻页会保留当前片段。

模型效果和民歌输入对比读取正式实验存档。粤语输入对比来自同一对正式权重对粤语留出片段的补充运行。加噪程度显示累计混合比例，五个实验档位分别对应前向扩散第 10、25、50、75、90 步。网页把延音展开到每格的实际音高，然后逐格与原旋律比较，双方同为休止也计入重合，显示为重合格数除以 64。试听采用统一的每分钟 90 拍，卷帘和五线谱均显示播放位置。

## 构建网页

从克隆后的仓库根目录运行。需要 Python 3.11 或更新版本。构建脚本从仓库的 `data/` 读取两套编码数据，同时需要下列实验导出。实验逐样本导出与生成的网页数据分别保存在本地。

| 构建选项 | 所需文件或目录 |
| --- | --- |
| `--case-catalog` | 案例清单 `case_catalog.json`，包含 12 个民歌重建案例和 3 个民歌输入的模型对比案例 |
| `--frozen-inputs` | 冻结受扰输入 `xt-test.jsonl` |
| `--reconstructions` | 重建目录，包含 `recon__h2-acfs_full-s0.jsonl.gz`、`recon__h3-acfs_matched-s0.jsonl.gz`、`recon__h3-cantopop-s0.jsonl.gz` 及后两份文件对应的 `.meta.json` |
| `--trajectories` | 逐步加噪和反向采样轨迹 `trajectories.json` |
| `--pop-comparison` | 补充对比目录，包含 `xt-test-pop.jsonl`、`inputs.meta.json` 和两个模型各自的 `.jsonl.gz` 与 `.meta.json` |

文件名中的 `h2` 表示民歌重建实验，`h3` 表示匹配民歌模型与粤语流行模型的比较，`s0` 表示训练随机种子为 0。将下面的路径替换为本地的实际位置。

```sh
python visualization/build.py --case-catalog ../experiment-inputs/case_catalog.json --frozen-inputs ../experiment-inputs/xt-test.jsonl --reconstructions ../experiment-inputs/reconstructions --trajectories ../experiment-inputs/trajectories.json --pop-comparison ../experiment-inputs/pop-comparison
```

网页默认输出到 `dist/melody-lab/`，也可用 `--output` 指定位置。默认使用系统字体。若拥有两款本地字体的相应使用许可，可安装 `fonttools` 后同时传入 `--title-font` 和 `--body-font`，构建脚本会将需要的字形保存为 WOFF。该选项适合使用独立的输出目录。

如需重新生成粤语输入对比，先准备正式训练代码及其推理依赖。脚本会校验两份权重与正式评测的记录一致。[训练代码与依赖](https://github.com/PiggyQwQ/melody-diffusion-study/tree/25a99389b538cb9494bdfd2f47df4eb914bff9e4/training)

```sh
python visualization/prepare_pop_comparison.py --training ../model-training/training --reconstructions ../experiment-inputs/reconstructions --output ../experiment-inputs/pop-comparison
```

## 构建 Windows 便携包

在 Windows x64 上先构建网页，再运行以下命令。构建使用 Windows 的 .NET Framework C# 编译器，首次运行会从 NuGet 下载并校验 WebView2 SDK。网页输出位置不同时，用 `--site` 指定。

```sh
python visualization/desktop/build_exe.py
```

默认生成 `dist/melody-lab-windows-x64.zip`。解压整个压缩包，保留程序旁的 `site/` 和 DLL，然后双击 `MelodyLab.exe`。运行需要 Windows x64、.NET Framework 4.8 和 Microsoft Edge WebView2 Runtime。[微软 WebView2 下载页](https://developer.microsoft.com/microsoft-edge/webview2/#download-section)

五线谱采用 VexFlow 4.2.5。代码和随库嵌入的字形分别附有许可。[VexFlow 许可](vendor/VexFlow-LICENSE.txt) · [刻谱与文字字形许可](vendor/VexFlow-Font-LICENSES.txt)
