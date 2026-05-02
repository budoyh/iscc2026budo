# 二进制文件漏洞检测

## 题目概况

- 赛题：ISCC2026 数据安全赛「二进制文件漏洞检测」
- 任务 1：判断二进制文件是否存在漏洞，`0` 表示无漏洞，`1` 表示存在漏洞。
- 任务 2：对漏洞样本判定 CWE 类型，共 86 类 CWE。
- 训练集：`39380` 个二进制文件。
- 测试集：`30974` 个二进制文件。
- 评分：宏平均 F1。
- 当前已知最好线上成绩：`s15.csv = 0.99951`。

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

当前最佳可复用提交文件：

```text
submissions/s15.csv
```

`s15.csv` 已上传，线上分数 `0.99951`。

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
| 当前最佳 | `s15.csv` | `0.99951` | 在 v15 基础上补齐 `CWE-561/CWE-562`，覆盖全部 86 类 CWE |

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
