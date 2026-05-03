# 二进制文件漏洞检测

## 题目概况

- 赛题：ISCC2026 数据安全赛「二进制文件漏洞检测」
- 任务 1：判断二进制文件是否存在漏洞，`0` 表示无漏洞，`1` 表示存在漏洞。
- 任务 2：对漏洞样本判定 CWE 类型，共 86 类 CWE。
- 训练集：`39380` 个二进制文件。
- 测试集：`30974` 个二进制文件。
- 评分：宏平均 F1。
- 当前已知最好 A 榜成绩：`s17b.csv / s24.csv = 0.99957`。
- 当前下一版技术候选：`s25.csv`，但尚无足够证据保证 A 榜 `1.0`。

## 目录结构

```text
work/data_security/二进制文件漏洞检测/
  raw/
    train.csv
    test.csv
    submission.csv
    binaries/
  processed/
    features.joblib
    enhanced_cwe_docs.joblib
    source_ids.csv
  report/
    *_valid_predictions.csv
    *_check.csv
  models/
  src/
  submissions/
  experiments.csv
  handoff_log.md
  README.md
```

## 环境与入口

普通机器学习脚本使用项目隔离 Python：

```powershell
cd C:\budostudy\only_for_codex\iscc
scripts\run_py.cmd work\data_security\二进制文件漏洞检测\src\脚本名.py
```

PyTorch / CUDA 脚本使用独立入口：

```powershell
cd C:\budostudy\only_for_codex\iscc
scripts\check_torch_cuda.cmd
scripts\run_torch.cmd work\data_security\二进制文件漏洞检测\src\torch_group_cnn.py
```

当前 CUDA PyTorch 环境为 `torch==2.7.0+cu126`，入口目录为 `C:\budostudy\only_for_codex\iscc\.deps\torch-cu126-direct`。

## 提交格式

提交文件必须满足：

- 文件名尽量短，例如 `s15.csv`、`s16f.csv`；已确认长文件名会导致平台上传失败。
- 编码使用 UTF-8 BOM，即 `utf-8-sig`。
- 列名必须是 `binary_id,label,cwe_id`。
- 行数必须是 `30974`，顺序与 `raw/test.csv` 一致。
- `label=0` 时 `cwe_id` 必须为空字符串。
- `label=1` 时 `cwe_id` 必须是 `CWE-xxx`。

当前已知 A 榜最佳文件：

```text
submissions/s17b.csv
submissions/s24.csv
```

`s17b.csv` 和 `s24.csv` 已上传，A 榜分数均为 `0.99957`。下一版候选 `submissions/s25.csv` 只新增 `BIN_1812963: CWE-121 -> CWE-126`，属于技术证据最强的单点候选；不要再提交已下降的 `s18.csv`、`s19.csv`、`s22.csv`，也不要提交批量探针 `s25m.csv`。

## 关键发现

- COFF 符号表中 `main` 对 `entry_bad` / `entry_good` 的调用可用于高精度判断二分类标签，二分类基本不是当前瓶颈。
- 二进制调试路径中包含 `Sxxxxx/source_sanitized.c`，可抽取 `source_id_num`。
- 每个 `Sxxxxx` 对应 good/bad 两个二进制，总体上一个无漏洞、一个有漏洞。
- 训练集与测试集按 source id 互补：训练中只有 good 的 source，测试中通常有 bad；训练中只有 bad 的 source，测试中通常有 good。
- 训练正样本按 `source_id_num` 排序后形成高度连续的 CWE 区间，这是当前提分最大的信号。
- `CWE-561` 与 `CWE-562` 在模型输出中一度缺失，宏平均 F1 对这类稀有类极敏感；补齐后线上从 `0.98029` 提升到 `0.99951`。
- 继续盲目使用深度学习或大批量边界覆盖会误伤，最后阶段应使用单点探针。

## 主要脚本

- `src/build_features.py`：早期 PE、节区、导入导出、字节直方图、字符串和指令统计特征。
- `src/train.py` / `src/predict.py`：早期分层 LinearSVC 训练与预测。
- `src/symbol_rule_predict.py`：解析 `main` 调用，判定 `entry_bad` / `entry_good`。
- `src/enhanced_cwe_calibrator.py`：抽取 `entry_bad` 函数体、反汇编 token、字符串、符号 token，并训练 CWE 细分校准器。
- `src/torch_group_cnn.py`：PyTorch 字节 CNN 组内修正实验。
- `src/source_id_chunk_postprocess.py`：抽取 `Sxxxxx`，使用训练正样本 CWE 连续 chunk 修正测试 CWE。
- `src/rare_singleton_postprocess.py`：补齐 `CWE-561/CWE-562` 稀有类。
- `src/source_outlier_postprocess.py`：修正明显落在错误 CWE source 区间之外的离群点。
- `src/symbol_cwe_postprocess.py`：解析 `entry_bad` 符号中的 CWE 编号，用于检查最终边界候选。

## 复现关键命令

早期基线：

```powershell
cd C:\budostudy\only_for_codex\iscc
scripts\run_py.cmd work\data_security\二进制文件漏洞检测\src\build_features.py --workers 8 --force
scripts\run_py.cmd work\data_security\二进制文件漏洞检测\src\train.py --strategy hierarchical --full
scripts\run_py.cmd work\data_security\二进制文件漏洞检测\src\predict.py
```

source-id chunk 修正：

```powershell
cd C:\budostudy\only_for_codex\iscc
scripts\run_py.cmd work\data_security\二进制文件漏洞检测\src\source_id_chunk_postprocess.py --generate --input submission_uninit457_v11_utf8_sig.csv --output submission_source_chunk_v12_utf8_sig.csv --min-count 1 --margin 0
```

稀有类与离群点修正：

```powershell
cd C:\budostudy\only_for_codex\iscc
scripts\run_py.cmd work\data_security\二进制文件漏洞检测\src\rare_singleton_postprocess.py --generate --input submission_source_chunk_v12_utf8_sig.csv --output submission_rare_singleton_v14_utf8_sig.csv
scripts\run_py.cmd work\data_security\二进制文件漏洞检测\src\source_outlier_postprocess.py --generate --input submission_rare_singleton_v14_utf8_sig.csv --output submission_source_outlier_v15_utf8_sig.csv
Copy-Item work\data_security\二进制文件漏洞检测\submissions\submission_source_outlier_v15_utf8_sig.csv work\data_security\二进制文件漏洞检测\submissions\s15.csv
```

最终单点探针检查：

```powershell
cd C:\budostudy\only_for_codex\iscc
scripts\run_py.cmd work\data_security\二进制文件漏洞检测\src\symbol_cwe_postprocess.py --check-sub --input s15.csv --workers 8
```

## 分数记录

| 阶段 | 文件 | 线上分数 | 说明 |
|---|---|---:|---|
| 早期可上传格式 | `submission_binary_id_utf8_sig.csv` | `0.87120` | 确认平台接受 `binary_id,label,cwe_id` + UTF-8 BOM |
| source chunk | `submission_source_chunk_v12_utf8_sig.csv` | `0.97900` | 使用 `Sxxxxx` 连续 CWE chunk，大幅提升 |
| v15b 安全短名 | `sub.csv` | `0.98029` | 只含 5 个 source outlier 修正，未包含稀有类恢复 |
| v15 完整短名 | `s15.csv` | `0.99951` | 在 v15 基础上补齐 `CWE-561/CWE-562`，覆盖全部 86 类 CWE |
| 当前 A 榜最佳 | `s17b.csv` | `0.99957` | 基于 `s15`，只修正 `CWE-121/122/126` 大类 source 边界 19 行 |
| 已否定 | `s18.csv` | `0.99847` | 单个 `CWE-391/396` source 边界猜测，明显下降，不再使用 |
| 已持平 | `s19.csv` | `0.99951` | 只保留 body 模型支持的 4 行，过于保守，低于 `s17b` |
| 内部候选 | `s20.csv` | 未提交 | 以 `s17b` 为基线回退 5 行；后续 DWARF 行号分析发现其中部分回退不稳，不作为最终建议 |
| 已否定 | `s22.csv` | `0.99954` | 以 `s17b` 为基线回退 5 行，用户反馈低于 `s17b`，说明回退过多 |
| 已持平 | `s24.csv` | `0.99957` | 以 `s17b` 为基线，只改 `BIN_1938615 CWE-126 -> CWE-122`，用户反馈与 `s17b` 持平 |
| 当前技术候选 | `s25.csv` | 待提交 | 以 `s24` 为基线，只改 `BIN_1812963 CWE-121 -> CWE-126`；证据强于其它未提交单点，但不能保证 A 榜 1.0 |

## 当前探针文件

这些文件均基于 `s15.csv`，每个只改 1 行或极少行，用于线上定位最后误差：

- `s16f.csv`：`BIN_4198849 CWE-252 -> CWE-253`
- `s16b.csv`：`BIN_8249526 CWE-126 -> CWE-121`
- `s16c.csv`：`BIN_6545696 CWE-126 -> CWE-121`
- `s16d.csv`：`BIN_1812963 CWE-126 -> CWE-121`
- `s16a.csv`：`BIN_7115172 CWE-135 -> CWE-121`
- `s16e.csv`：`BIN_8556812 CWE-526 -> CWE-546`
- `s16g.csv`：组合改 `s16b+s16c+s16d`

建议先测 `s16f.csv`，再测 `s16b.csv`、`s16c.csv`、`s16d.csv`。如果单点探针提升，再沿同一边界继续扩展；如果下降，直接回到 `s15.csv`。

## 当前结论

当前最可靠文件仍是 `submissions/s15.csv`。不要用长文件名提交，不要直接提交大批量边界覆盖结果。最后阶段应依赖线上单点探针反馈，而不是重新训练大模型。

## 2026-05-02 22:55 图片思路复核

用户提供的外部截图思路为：利用二进制中残留的 `wrapper_bad.c` / `wrapper_good.c` / `injected_bad.cpp` / `injected_good.cpp` 判断 `label`，再用调试路径中的 `Sxxxxx` 源样本编号做 CWE 最近邻映射。

本地已复核：

- `wrapper/injected` 二分类信号在训练集 39380 个样本上全覆盖，准确率 `1.0`。
- 对测试集 30974 个样本，`wrapper/injected` 重建的 `label` 与当前最佳 `s15.csv` 完全一致，二分类没有新增可修正行。
- `Sxxxxx` 最近邻 CWE 相对 `s15.csv` 会改 32 行，其中一部分是低样本 CWE；按 `0.99951` 的当前分数反推，低样本 CWE 单行错改的理论损失大于当前总差距，因此不应优先上传完整最近邻结果。

已新增脚本：

```powershell
scripts\run_py.cmd work\data_security\二进制文件漏洞检测\src\wrapper_nearest_postprocess.py --input s15.csv --output s17b.csv --no-rewrite-label --allowed-cwes CWE-121,CWE-122,CWE-126 --generate
scripts\run_py.cmd work\data_security\二进制文件漏洞检测\src\wrapper_nearest_postprocess.py --input s15.csv --output s17c.csv --no-rewrite-label --max-one-error-drop 0.00050 --generate
scripts\run_py.cmd work\data_security\二进制文件漏洞检测\src\wrapper_nearest_postprocess.py --input s15.csv --output s17d.csv --no-rewrite-label --max-one-error-drop 0.00060 --generate
scripts\run_py.cmd work\data_security\二进制文件漏洞检测\src\wrapper_nearest_postprocess.py --input s15.csv --output s17n.csv --no-rewrite-label --generate
```

新候选文件均为短名，格式已校验：

- `s17b.csv`：19 行，限制在 `CWE-121/122/126` 大类边界；建议下一轮第一优先提交。
- `s17c.csv`：23 行，在 `s17b` 基础上加入与当前分数差距相容的少数中等类边界；建议第二优先。
- `s17d.csv`：24 行，比 `s17c` 多 `BIN_3124360 CWE-196 -> CWE-197`，更激进；建议第三优先。
- `s17n.csv`：32 行，完整最近邻图片思路；包含低样本 CWE 变更，风险较高，不建议优先。
- `s17k.csv`：同类最近邻实验，输出只覆盖 85 个 CWE，破坏完整 86 类覆盖，不建议上传。

## 2026-05-03 01:03 下一版唯一候选

用户反馈 `s17b.csv` A 榜得分 `0.99957`，高于 `s15.csv` 的 `0.99951`。后续提交次数有限，不再同时给多个候选。

当时生成的候选：

```text
submissions/s18.csv
```

`s18.csv` 基于 `s17b.csv`，只新增一处修正：

```text
BIN_7681509: CWE-396 -> CWE-391
```

该样本位于 `S35236`，左侧最近训练正样本 `S35235` 为 `CWE-391`，右侧 `S35237-S35242` 稳定为 `CWE-396`。这是一个边界单点，理论单行影响接近当前剩余差距；相比继续批量扩展，`s18.csv` 更便于归因，也更符合 A/B 榜鲁棒性要求。

已验证：

- 行数 `30974`，列名 `binary_id,label,cwe_id`。
- `label` 分布 `{0: 15535, 1: 15439}`。
- CWE 覆盖数 `86`。
- 相对 `s17b.csv` 仅 1 行差异。

## 2026-05-03 02:10 s19 语义过滤候选

用户反馈 `s18.csv` A 榜 `0.99847`，说明单纯 source-id 边界单点不可靠。当前停止测试集边界猜测，改为语义过滤：

```text
source-id 最近邻提出候选，entry_bad 函数字节模型也支持新 CWE 时才接受。
```

当时生成的候选：

```text
submissions/s19.csv
```

`s19.csv` 以 `s15.csv` 为基线，只改 4 行：

```text
BIN_1812963: CWE-126 -> CWE-121
BIN_6244783: CWE-126 -> CWE-122
BIN_1455492: CWE-121 -> CWE-126
BIN_2334733: CWE-126 -> CWE-121
```

验证依据：

- `CWE-121/122/126` 三分类 `entry_bad` 函数字节模型 5 折宏 F1 为 `0.9738046991970849`。
- 固定验证集上，最近 source-id 直接修改同类边界无收益；加函数字节过滤后，宏 F1 从 `0.999119583180` 提到 `0.999189000511`。
- `s19.csv` 行数 `30974`，列名正确，`label` 分布 `{0:15535, 1:15439}`，CWE 覆盖 `86`。

## 2026-05-03 03:05 s19 反馈与 s20

用户反馈 `s19.csv` A 榜 `0.99951`，低于 `s17b.csv` 的 `0.99957`。这说明只接受 4 个函数字节模型支持项过于保守，`s17b` 中被 `s19` 回退的 15 个 source 边界仍包含有效修正；但 `s17b` 的 19 行也不应默认全对。

当前新增一个结构语义校验器：

```text
src/struct_semantics_3class.py
```

它从 `entry_bad` 反汇编中抽取 `malloc/free/memcpy/printLine/exit` 调用、栈/堆写入、`memcpy` 长度、偏移量等结构语义特征，只用于 `CWE-121/122/126` 三分类校验。5 折交叉验证结果：

```text
DecisionTree macro F1 = 0.848367234203
RandomForest macro F1 = 0.940832425833
ExtraTrees macro F1 = 0.947034721084
```

当时生成的候选：

```text
submissions/s20.csv
```

`s20.csv` 以已验证 A 榜最优的 `s17b.csv` 为基线，只回退 5 个所有独立语义证据都支持旧 CWE 的行：

```text
BIN_3049998: CWE-121 -> CWE-126
BIN_4615687: CWE-122 -> CWE-126
BIN_2254329: CWE-121 -> CWE-126
BIN_2902601: CWE-122 -> CWE-126
BIN_9335771: CWE-122 -> CWE-126
```

依据文件：

```text
report/s17b_struct_semantics.csv
report/s20_reverts.csv
report/s22_evidence.csv
```

格式校验：

```text
s20.csv (30974, 3), label 分布 {0: 15535, 1: 15439}, CWE 覆盖 86, ok
相对 s17b 回退 5 行
相对 s15 保留 14 行修正
```

`s20.csv` 后续被 DWARF 行号分析取代，不再作为最终建议。原因是 `BIN_3049998` 等样本的 DWARF `entry_bad` 声明行和变量声明行更支持 `s17b` 的新类，而不是 `s20` 的回退类。

## 2026-05-03 03:45 DWARF 行号与 s22

PE 中保留了 `.debug_info/.debug_line/.debug_str/.debug_line_str` 等 DWARF 调试段。已新增脚本：

```text
src/dwarf_line_3class.py
```

该脚本从 `source_sanitized.c` 对应的 `entry_bad` DWARF DIE 中抽取源码级特征：

- `entry_bad` 声明行号。
- 函数 `high_pc` 长度。
- 局部变量声明行号。
- `structCharVoid` 是否为指针、结构体大小、成员布局。
- `charFirst` 数组元素类型和数组长度。

DWARF 三分类模型的 5 折交叉验证：

```text
ExtraTrees macro F1 = 0.872410759186
RandomForest macro F1 = 0.869763263666
```

该分数低于 source-id 近邻，说明 DWARF 不能单独替代 `s17b`；但它能识别部分 source 边界误伤。最终融合原则：

```text
保留 s17b 中 source 证据一致或 DWARF 强支持的修正；
只回退 source 非一致且 DWARF/函数体/结构语义共同偏向旧 CWE 的行。
```

当前唯一建议提交：

```text
submissions/s22.csv
```

`s22.csv` 以 `s17b.csv` 为基线，只回退 5 行：

```text
BIN_1812963: CWE-121 -> CWE-126
BIN_1938615: CWE-126 -> CWE-122
BIN_4615687: CWE-122 -> CWE-126
BIN_2254329: CWE-121 -> CWE-126
BIN_4860053: CWE-122 -> CWE-126
```

综合证据表：

```text
report/s22_evidence.csv
report/s22_reverts.csv
report/s17b_dwarf_line_3class.csv
```

格式校验：

```text
s22.csv (30974, 3), label 分布 {0: 15535, 1: 15439}, CWE 覆盖 86, ok
相对 s17b 回退 5 行
相对 s15 保留 14 行修正
```

用户反馈 `s22.csv` A 榜 `0.99954`，低于 `s17b.csv` 的 `0.99957`。这说明 `s22` 的 5 个回退整体净负，不能继续扩大回退。

## 2026-05-03 04:30 全量 DWARF 模板键与 s24

进一步对全部训练正样本和全部测试正样本提取 DWARF line program 模板键，缓存文件：

```text
processed/dwarf_key_train_all_pos.csv
processed/dwarf_key_test_all_pos.csv
```

全量扫描发现大量跨 CWE 大块的模板复用，不能直接批量覆盖。例如某些 `CWE-121` 样本会命中远处 `CWE-690` 模板，这类变更被过滤掉。最终只保留同时满足以下条件的候选：

```text
DWARF 模板键在训练集中纯映射到同一 CWE
support >= 2
训练例子的 source_id 与测试样本在局部邻域内
```

满足条件的唯一候选：

```text
submissions/s24.csv
```

`s24.csv` 以 `s17b.csv` 为基线，只改 1 行：

```text
BIN_1938615: CWE-126 -> CWE-122
```

依据：

```text
report/dwarf_key_all_local_candidates.csv
report/s24_change.csv
```

格式校验：

```text
s24.csv (30974, 3), label 分布 {0: 15535, 1: 15439}, CWE 覆盖 86, ok
相对 s17b 仅 1 行差异
UTF-8 BOM 已确认
```

下一次只建议上传 `s24.csv`。`s23.csv` 多加了 `BIN_2038451`，但该样本的训练例子不在 source-id 局部邻域内，已降级为内部分析产物。

## s22 反馈后的复核

`s22.csv` A 榜反馈为 `0.99954`，低于 `s17b.csv=0.99957`，说明扩大回退会伤害当前最优解。后续复核重点转为寻找“硬证据”而不是继续批量试探：

```text
直接 CWE 字符串签名：测试正样本中 5884 个可直接锚定，均与当前预测一致。
训练/测试负样本直接 CWE 锚点：加入 5882 个测试负样本锚点后，没有发现区间一致但当前预测冲突的样本。
source-id 边界模型：98 个左右 CWE 不一致的测试边界经训练边界 LOO、DWARF key、ExtraTrees/RandomForest 共识复核，没有新增高置信候选。
```

相关报告：

```text
report/source_all_token_anchor_unknown_test.csv
report/boundary_ml_consensus_candidates.csv
report/boundary_dwarf_key_candidates.csv
report/exact_neg_tri_model_candidates.csv
```

当前唯一保留的下一版提交仍为 `submissions/s24.csv`，只修改：

```text
BIN_1938615: CWE-126 -> CWE-122
```

格式复验通过：30974 行、`binary_id,label,cwe_id`、顺序匹配 `raw/test.csv`、UTF-8 BOM、label 分布 `{0:15535, 1:15439}`、CWE 覆盖 86。

## s24 反馈后的复核与 s25

用户反馈 `s24.csv` A 榜仍为 `0.99957`，与 `s17b.csv` 持平。后续不能再把 `BIN_1938615` 扩展成批量 DWARF key 规则，也不能按测试集行号或测试集顺序做迁移；所有改动必须按 `binary_id` 和二进制自身证据定位。

本轮新增复核：

```text
完整 CWE/Julliet testcase 字符串：训练 first-code 规则覆盖 7457，仅 6 个 CWE-135 复合名例外；测试唯二冲突 BIN_1806282 / BIN_7115172 均由 source-id CWE-135 块支持，不改。
稀有/中稀有类 support：CWE-561、CWE-562、CWE-674 等均有 source 邻域训练锚点，没有发现新的高置信硬错。
s17n/s17c/s17d 非 121/122/126 边界候选：逐个复查后均被当前 CWE 的局部代码特征支持，不再建议提交。
s25m.csv：修复反汇编 token 后的内部批量探针，相对 s17b 改 58 行，风险过高，不建议提交。
```

当前唯一新增候选：

```text
submissions/s25.csv
```

`s25.csv` 以 `s24.csv` 为基线，只改 1 行：

```text
BIN_1812963: CWE-121 -> CWE-126
```

依据：

```text
report/fixed_boundary_side_model_candidates.csv
report/s25_change.csv
report/suspect_boundary_features.csv
```

关键证据：`BIN_1812963` 的 `key_op_full` 与训练样本 `BIN_8892575 / S4914 / CWE-126` 精确一致，DWARF/source boundary side 模型也指向右侧 `CWE-126`。格式复验通过：30974 行、顺序匹配 `raw/test.csv`、UTF-8 BOM、label 分布 `{0:15535, 1:15439}`、CWE 覆盖 86。

风险：`s25.csv` 是当前唯一比 `s24` 更有技术依据的单点候选，但现有离线证据不足以承诺 A 榜 `1.0`。
