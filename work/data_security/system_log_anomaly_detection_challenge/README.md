# 系统日志异常检测挑战

本目录记录 ISCC2026 数据安全赛“系统日志异常检测挑战”的数据、模型、实验、提交文件和复现命令。

## 当前状态

- 当前已提交线上最高分：`push.csv`，A 榜 `0.96233`。
- 之前线上关键分数：`0.73166 -> 0.77684 -> 0.85031 -> 0.85941 -> 0.92494 -> 0.93531 -> 0.94666 -> 0.96187 -> 0.96208 -> 0.96233`；`try.csv` 回落到 `0.96153`，`up.csv` 回落到 `0.96229`。
- 当前榜首参考分：`0.96258`。
- 当前下一版冲刺提交：[go.csv](/C:/budostudy/only_for_codex/iscc/work/data_security/system_log_anomaly_detection_challenge/submissions/go.csv)。
- `try.csv` 已验证降分，说明整体 top16 边界重排 public 过拟合；后续不再推荐。
- `push.csv` 以 `next.csv` 为锚，只接受 43 行 b4h3 高置信边界门控变化，并额外删除 2 个最低置信异常，异常数为 `3208`，线上已验证有效。
- `up.csv` 保留 `push.csv` 全部改动，只额外加入 4 个 p2 中的收缩类边界变化，线上 `0.96229`，说明“弱前兆/恢复语句一律收缩”不可靠。
- `go.csv` 以 `push.csv` 为锚，只改 8 行边界，不改检测计数和类型；候选来自 `txt/b4h3/b5h3` 分歧行，并用训练集边界模板证据过滤。

`final.csv` 使用候选 span 排序器生成，检测异常数保持和已验证高分 `s1.csv` 相同，均为 `3214/5000`。`next.csv` 将 count 校准到 `3210/5000` 后线上提升到 `0.96208`，证明低置信样本筛除有效。`try.csv` 失败后，当前策略改为“保守门控 + 小幅 count 下调”，避免再次大面积改动高置信边界。

## 题目与评分

任务包含三部分：

- 异常检测：预测 `has_anomaly`。
- 主异常区间定位：预测 `primary_start_idx` 和 `primary_end_idx`。
- 主异常类型识别：预测 `primary_anomaly_type`，共 10 类。

评分公式：

```text
Score = 0.15 * F1_detect + 0.50 * IoU_loc + 0.35 * F1_type
```

关键判断：

- 定位 IoU 权重最高，继续提分必须优先优化边界。
- 类型识别在深度模型阶段已经很稳定，不能为定位大幅牺牲类型。
- 检测权重只有 0.15，但检测比例不能随意漂移，否则会吞掉定位收益。

## 数据目录

- 训练集：[train.csv](/C:/budostudy/only_for_codex/iscc/work/data_security/system_log_anomaly_detection_challenge/raw/data/train.csv)，`20000` 行。
- 测试集：[test.csv](/C:/budostudy/only_for_codex/iscc/work/data_security/system_log_anomaly_detection_challenge/raw/data/test.csv)，`5000` 行。
- 样例提交：[sample_submission.csv](/C:/budostudy/only_for_codex/iscc/work/data_security/system_log_anomaly_detection_challenge/raw/data/sample_submission.csv)。

目录结构：

```text
raw/data/       原始数据
src/            训练、预测、规则、深度学习脚本
models/         已训练模型权重
submissions/    提交文件
logs/           训练指标、实验总结、最终决策记录
processed/      OOF/验证预测等中间文件
```

## 环境入口

普通机器学习入口：

```powershell
scripts\run_py.cmd work\data_security\system_log_anomaly_detection_challenge\src\train.py
```

PyTorch/CUDA 入口：

```powershell
scripts\run_torch_py.cmd -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

当前 PyTorch 运行入口：

[run_torch_py.cmd](/C:/budostudy/only_for_codex/iscc/scripts/run_torch_py.cmd)

## 实验总览

完整表格见 [experiments.csv](/C:/budostudy/only_for_codex/iscc/work/data_security/system_log_anomaly_detection_challenge/experiments.csv)。

| 阶段 | 方案 | 本地/验证分 | 线上分 | 结论 |
|---|---|---:|---:|---|
| 规则 v1 | 整行模板匹配 | `0.88913` | `0.73166` | 泛化不足 |
| 规则 v2 | posterior 模板 + typo | `0.91915` | `0.77684` | 有提升但仍低 |
| 规则 v3 | 整行 + 子句 + 前缀 | `0.95413` | `0.85031` | 说明测试有模板变体 |
| 规则 v5 | 歧义弃权 | `0.95689` | `0.85941` | 规则上限明显 |
| DL v1 | BiGRU sequence | `0.97399` | `0.92494` | 深度模型显著改善边界 |
| DL v2 | boundary 三模型 | `0.97911` | `0.93531` | start/end 边界头有效 |
| DL v3 | mixed 40/60 | `0.98350` proxy | `0.94666` | 已验证当前最高线上分 |
| DL v4 | span ranker | `0.98643` proxy | `0.96187` | 当前线上最高 |
| DL v5 | span ranker count=3210 | `0.98643` proxy | `0.96208` | 当前已提交最高分 |
| DL v6 | top16 span ranker count=3210 | `0.98659` proxy | `0.96153` | public 过拟合，弃用 |
| DL v7 | 7-model gated ranker + count=3208 | 轻量验证 | `0.96233` | 当前最高 |
| DL v8 | push + shrink-only extras | 轻量验证 | `0.96229` | 已降分，弃用 |
| DL v9 | text/meta boundary gate | 训练模板证据 + 多模型一致性 | 待提交 | 当前冲刺推荐 |

## 模型演进

### 1. 规则阶段

相关脚本：

- [common.py](/C:/budostudy/only_for_codex/iscc/work/data_security/system_log_anomaly_detection_challenge/src/common.py)
- [train.py](/C:/budostudy/only_for_codex/iscc/work/data_security/system_log_anomaly_detection_challenge/src/train.py)
- [predict.py](/C:/budostudy/only_for_codex/iscc/work/data_security/system_log_anomaly_detection_challenge/src/predict.py)
- [predict_abstain.py](/C:/budostudy/only_for_codex/iscc/work/data_security/system_log_anomaly_detection_challenge/src/predict_abstain.py)

主要做法：

- 标准化时间、数字、地址、路径、segment/id。
- 修正常见拼写扰动。
- 使用整行、嵌入子句、子句前缀、正常模板尾部提示语做后验统计。
- 对多候选强冲突样本尝试弃权。

结论：

- 规则方案最高线上只到 `0.85941`。
- 规则能识别明显模板，但很容易选中后续症状段，而不是主异常起始段。
- 后续必须使用上下文序列模型。

### 2. BiGRU sequence 模型

相关脚本：

- [train_torch_sequence.py](/C:/budostudy/only_for_codex/iscc/work/data_security/system_log_anomaly_detection_challenge/src/train_torch_sequence.py)
- [train_torch_sequence_full.py](/C:/budostudy/only_for_codex/iscc/work/data_security/system_log_anomaly_detection_challenge/src/train_torch_sequence_full.py)
- [predict_torch_sequence.py](/C:/budostudy/only_for_codex/iscc/work/data_security/system_log_anomaly_detection_challenge/src/predict_torch_sequence.py)
- [predict_torch_ensemble.py](/C:/budostudy/only_for_codex/iscc/work/data_security/system_log_anomaly_detection_challenge/src/predict_torch_ensemble.py)

结果：

- 90/10 验证：`0.9739887240`。
- 全量两模型线上：`0.92494`。

结论：

- 深度上下文建模明显优于规则。
- 主要瓶颈从检测/类型转为边界定位。

### 3. Boundary-aware BiGRU

相关脚本：

- [train_torch_boundary.py](/C:/budostudy/only_for_codex/iscc/work/data_security/system_log_anomaly_detection_challenge/src/train_torch_boundary.py)
- [train_torch_boundary_full.py](/C:/budostudy/only_for_codex/iscc/work/data_security/system_log_anomaly_detection_challenge/src/train_torch_boundary_full.py)
- [predict_torch_boundary.py](/C:/budostudy/only_for_codex/iscc/work/data_security/system_log_anomaly_detection_challenge/src/predict_torch_boundary.py)
- [predict_torch_boundary_ensemble.py](/C:/budostudy/only_for_codex/iscc/work/data_security/system_log_anomaly_detection_challenge/src/predict_torch_boundary_ensemble.py)

做法：

- 每行预测 `none/10类异常`。
- 额外预测主异常起点和终点。
- 解码时联合使用行类别分数、start 分数、end 分数。
- 三个全量模型 seed `42/43/44` 做平均。

结果：

- holdout：`0.9791059329`，IoU `0.9582118659`。
- 线上：[submission_torch_boundary_ensemble3.csv](/C:/budostudy/only_for_codex/iscc/work/data_security/system_log_anomaly_detection_challenge/submissions/submission_torch_boundary_ensemble3.csv) 得分 `0.93531`。

结论：

- 显式边界头有效。
- 验证分和线上仍有较大 gap，说明模板/拼写/语义漂移没有完全解决。

### 4. Hybrid char+word boundary 模型

相关脚本：

- [train_torch_hybrid_boundary.py](/C:/budostudy/only_for_codex/iscc/work/data_security/system_log_anomaly_detection_challenge/src/train_torch_hybrid_boundary.py)
- [predict_torch_hybrid_ensemble.py](/C:/budostudy/only_for_codex/iscc/work/data_security/system_log_anomaly_detection_challenge/src/predict_torch_hybrid_ensemble.py)

做法：

- 行级 word token average。
- 行级 char CNN，处理 OOV、拼写扰动、测试集 typo。
- 2 层 BiGRU 建模日志窗口上下文。
- start/end 边界头。
- 异常窗口增强：同义替换、拼写扰动、char noise、`<mask>` token dropout。

结果：

- hybrid holdout：`0.9807544832`，IoU `0.9616617719`。
- 单独提交风险偏高，因此未作为最终优先提交。

### 5. Mixed boundary ensemble

相关脚本：

- [predict_torch_mixed_boundary_ensemble.py](/C:/budostudy/only_for_codex/iscc/work/data_security/system_log_anomaly_detection_challenge/src/predict_torch_mixed_boundary_ensemble.py)

做法：

- 旧 word-boundary 三模型权重 `40%`。
- hybrid char+word 三模型权重 `60%`。
- `decode_boundary_weight=1.5`。

关键提交：

- [s1.csv](/C:/budostudy/only_for_codex/iscc/work/data_security/system_log_anomaly_detection_challenge/submissions/s1.csv)：线上 `0.94666`。
- [s2.csv](/C:/budostudy/only_for_codex/iscc/work/data_security/system_log_anomaly_detection_challenge/submissions/s2.csv)：长度先验 `0.6`，未提交。
- [s3.csv](/C:/budostudy/only_for_codex/iscc/work/data_security/system_log_anomaly_detection_challenge/submissions/s3.csv)：保守 threshold + 强长度先验，未提交。
- [s4.csv](/C:/budostudy/only_for_codex/iscc/work/data_security/system_log_anomaly_detection_challenge/submissions/s4.csv)：50/50 混合，未提交。

结论：

- `s1` 已确认 mixed 方向有效。
- 长度先验能提升 holdout，但直接全量使用风险较高。

### 6. Span ranker 最终方案

相关脚本：

- [predict_torch_span_ranker.py](/C:/budostudy/only_for_codex/iscc/work/data_security/system_log_anomaly_detection_challenge/src/predict_torch_span_ranker.py)

做法：

- 使用六个全量模型输出生成候选 span。
- 对每个候选提取：
  - 原始窗口分数。
  - start/end log probability。
  - 类型与长度先验。
  - span 长度、位置、窗口均值/最大值等特征。
- 训练两个候选排序器：
  - `HistGradientBoostingRegressor`
  - `ExtraTreesRegressor`
- 训练目标：候选类型匹配时的 IoU，否则为 `0`。
- 测试集强制保留 `3214` 个异常，与 `s1` 检测比例一致。

验证：

- 类似 fit/valid ranker proxy：`0.9864332900`。
- IoU：`0.9734775506`。
- 相对 `s1`：检测 flag 不变，类型只改 1 行，主要改边界。
- 预测 span 长度分布比 `s1` 更接近训练标签分布。

已提交高分：

[final.csv](/C:/budostudy/only_for_codex/iscc/work/data_security/system_log_anomaly_detection_challenge/submissions/final.csv)

当前下一版推荐：

[go.csv](/C:/budostudy/only_for_codex/iscc/work/data_security/system_log_anomaly_detection_challenge/submissions/go.csv)

备选版本：

- [mc4.csv](/C:/budostudy/only_for_codex/iscc/work/data_security/system_log_anomaly_detection_challenge/submissions/mc4.csv)：最保守的 meta gate，只改 4 行。
- [mc6.csv](/C:/budostudy/only_for_codex/iscc/work/data_security/system_log_anomaly_detection_challenge/submissions/mc6.csv)：比 `mc4` 多 2 行 duplicate_event 起点修正。
- [mc10.csv](/C:/budostudy/only_for_codex/iscc/work/data_security/system_log_anomaly_detection_challenge/submissions/mc10.csv)：更激进，只改 10 行，但弱前兆/恢复边界更多，风险高于 `go.csv`。
- [p2.csv](/C:/budostudy/only_for_codex/iscc/work/data_security/system_log_anomaly_detection_challenge/submissions/p2.csv)：比 `push.csv` 多 7 行门控变化，包含 `up.csv` 已证伪的 4 个收缩变化，不推荐。
- [trim.csv](/C:/budostudy/only_for_codex/iscc/work/data_security/system_log_anomaly_detection_challenge/submissions/trim.csv)：从 `push.csv` 回退 4 个低增量扩边，较保守。
- [best.csv](/C:/budostudy/only_for_codex/iscc/work/data_security/system_log_anomaly_detection_challenge/submissions/best.csv)：`trim.csv` 再叠加 4 个收缩类新增变化。
- [safe8.csv](/C:/budostudy/only_for_codex/iscc/work/data_security/system_log_anomaly_detection_challenge/submissions/safe8.csv)：更保守，只接受 17 行边界变化并删除 2 个低置信异常。
- [hard.csv](/C:/budostudy/only_for_codex/iscc/work/data_security/system_log_anomaly_detection_challenge/submissions/hard.csv)：更激进，接受 72 行边界变化并删除 2 个低置信异常。

## 复现命令

从已有模型权重生成 `final.csv` 以及 count 变体：

```powershell
scripts\run_torch_py.cmd work\data_security\system_log_anomaly_detection_challenge\src\predict_torch_span_ranker.py `
  --boundary-artifact work\data_security\system_log_anomaly_detection_challenge\models\torch_boundary_full_model.pt `
  --boundary-artifact work\data_security\system_log_anomaly_detection_challenge\models\torch_boundary_full_seed43_model.pt `
  --boundary-artifact work\data_security\system_log_anomaly_detection_challenge\models\torch_boundary_full_seed44_model.pt `
  --hybrid-artifact work\data_security\system_log_anomaly_detection_challenge\models\torch_hybrid_boundary_full_model.pt `
  --hybrid-artifact work\data_security\system_log_anomaly_detection_challenge\models\torch_hybrid_boundary_full_seed43_model.pt `
  --hybrid-artifact work\data_security\system_log_anomaly_detection_challenge\models\torch_hybrid_boundary_full_seed44_model.pt `
  --output-path work\data_security\system_log_anomaly_detection_challenge\submissions\final.csv `
  --force-count 3214 `
  --variant-count 3210
```

生成已弃用的 top16 诊断版本：

```powershell
scripts\run_torch_py.cmd work\data_security\system_log_anomaly_detection_challenge\src\predict_torch_span_ranker.py `
  --boundary-artifact work\data_security\system_log_anomaly_detection_challenge\models\torch_boundary_full_model.pt `
  --boundary-artifact work\data_security\system_log_anomaly_detection_challenge\models\torch_boundary_full_seed43_model.pt `
  --boundary-artifact work\data_security\system_log_anomaly_detection_challenge\models\torch_boundary_full_seed44_model.pt `
  --hybrid-artifact work\data_security\system_log_anomaly_detection_challenge\models\torch_hybrid_boundary_full_model.pt `
  --hybrid-artifact work\data_security\system_log_anomaly_detection_challenge\models\torch_hybrid_boundary_full_seed43_model.pt `
  --hybrid-artifact work\data_security\system_log_anomaly_detection_challenge\models\torch_hybrid_boundary_full_seed44_model.pt `
  --output-path work\data_security\system_log_anomaly_detection_challenge\submissions\t16.csv `
  --force-count 3206 `
  --variant-count 3210 `
  --confidence-output work\data_security\system_log_anomaly_detection_challenge\processed\ranker_confidence_top16.csv `
  --summary-path work\data_security\system_log_anomaly_detection_challenge\logs\span_ranker_top16_summary.json `
  --top-per-label 16
```

生成当前推荐的 meta boundary gate 版本：

```powershell
scripts\run_py.cmd work\data_security\system_log_anomaly_detection_challenge\src\build_meta_boundary_candidates.py `
  --output-prefix work\data_security\system_log_anomaly_detection_challenge\submissions\mc `
  --summary-path work\data_security\system_log_anomaly_detection_challenge\logs\meta_boundary_clean_summary.json `
  --rows-path work\data_security\system_log_anomaly_detection_challenge\processed\meta_boundary_clean_candidates.csv `
  --min-evidence 1.0

Copy-Item `
  -LiteralPath work\data_security\system_log_anomaly_detection_challenge\submissions\mc8.csv `
  -Destination work\data_security\system_log_anomaly_detection_challenge\submissions\go.csv `
  -Force
```

重训 boundary 三模型：

```powershell
scripts\run_torch_py.cmd work\data_security\system_log_anomaly_detection_challenge\src\train_torch_boundary_full.py --epochs 4 --batch-size 64 --seed 42 --artifact-path work\data_security\system_log_anomaly_detection_challenge\models\torch_boundary_full_model.pt
scripts\run_torch_py.cmd work\data_security\system_log_anomaly_detection_challenge\src\train_torch_boundary_full.py --epochs 4 --batch-size 64 --seed 43 --artifact-path work\data_security\system_log_anomaly_detection_challenge\models\torch_boundary_full_seed43_model.pt
scripts\run_torch_py.cmd work\data_security\system_log_anomaly_detection_challenge\src\train_torch_boundary_full.py --epochs 4 --batch-size 64 --seed 44 --artifact-path work\data_security\system_log_anomaly_detection_challenge\models\torch_boundary_full_seed44_model.pt
```

重训 hybrid 三模型：

```powershell
scripts\run_torch_py.cmd work\data_security\system_log_anomaly_detection_challenge\src\train_torch_hybrid_boundary.py --full --epochs 4 --batch-size 32 --augment-copies 1 --seed 42 --artifact-path work\data_security\system_log_anomaly_detection_challenge\models\torch_hybrid_boundary_full_model.pt
scripts\run_torch_py.cmd work\data_security\system_log_anomaly_detection_challenge\src\train_torch_hybrid_boundary.py --full --epochs 4 --batch-size 32 --augment-copies 1 --seed 43 --artifact-path work\data_security\system_log_anomaly_detection_challenge\models\torch_hybrid_boundary_full_seed43_model.pt
scripts\run_torch_py.cmd work\data_security\system_log_anomaly_detection_challenge\src\train_torch_hybrid_boundary.py --full --epochs 4 --batch-size 32 --augment-copies 1 --seed 44 --artifact-path work\data_security\system_log_anomaly_detection_challenge\models\torch_hybrid_boundary_full_seed44_model.pt
```

## 关键产物

推荐提交：

- [go.csv](/C:/budostudy/only_for_codex/iscc/work/data_security/system_log_anomaly_detection_challenge/submissions/go.csv)
- [mc8.csv](/C:/budostudy/only_for_codex/iscc/work/data_security/system_log_anomaly_detection_challenge/submissions/mc8.csv)
- [mc6.csv](/C:/budostudy/only_for_codex/iscc/work/data_security/system_log_anomaly_detection_challenge/submissions/mc6.csv)
- [mc4.csv](/C:/budostudy/only_for_codex/iscc/work/data_security/system_log_anomaly_detection_challenge/submissions/mc4.csv)
- [push.csv](/C:/budostudy/only_for_codex/iscc/work/data_security/system_log_anomaly_detection_challenge/submissions/push.csv)
- [p2.csv](/C:/budostudy/only_for_codex/iscc/work/data_security/system_log_anomaly_detection_challenge/submissions/p2.csv)
- [trim.csv](/C:/budostudy/only_for_codex/iscc/work/data_security/system_log_anomaly_detection_challenge/submissions/trim.csv)
- [best.csv](/C:/budostudy/only_for_codex/iscc/work/data_security/system_log_anomaly_detection_challenge/submissions/best.csv)
- [safe8.csv](/C:/budostudy/only_for_codex/iscc/work/data_security/system_log_anomaly_detection_challenge/submissions/safe8.csv)
- [hard.csv](/C:/budostudy/only_for_codex/iscc/work/data_security/system_log_anomaly_detection_challenge/submissions/hard.csv)
- [try.csv](/C:/budostudy/only_for_codex/iscc/work/data_security/system_log_anomaly_detection_challenge/submissions/try.csv)
- [risk.csv](/C:/budostudy/only_for_codex/iscc/work/data_security/system_log_anomaly_detection_challenge/submissions/risk.csv)
- [next.csv](/C:/budostudy/only_for_codex/iscc/work/data_security/system_log_anomaly_detection_challenge/submissions/next.csv)
- [final.csv](/C:/budostudy/only_for_codex/iscc/work/data_security/system_log_anomaly_detection_challenge/submissions/final.csv)
- [s6.csv](/C:/budostudy/only_for_codex/iscc/work/data_security/system_log_anomaly_detection_challenge/submissions/s6.csv)

已提交高分：

- [next.csv](/C:/budostudy/only_for_codex/iscc/work/data_security/system_log_anomaly_detection_challenge/submissions/next.csv)，线上 `0.96208`。
- [push.csv](/C:/budostudy/only_for_codex/iscc/work/data_security/system_log_anomaly_detection_challenge/submissions/push.csv)，线上 `0.96233`。
- [up.csv](/C:/budostudy/only_for_codex/iscc/work/data_security/system_log_anomaly_detection_challenge/submissions/up.csv)，线上 `0.96229`，已判定不推荐。
- [try.csv](/C:/budostudy/only_for_codex/iscc/work/data_security/system_log_anomaly_detection_challenge/submissions/try.csv)，线上 `0.96153`，已判定不推荐。
- [final.csv](/C:/budostudy/only_for_codex/iscc/work/data_security/system_log_anomaly_detection_challenge/submissions/final.csv)，线上 `0.96187`。
- [s1.csv](/C:/budostudy/only_for_codex/iscc/work/data_security/system_log_anomaly_detection_challenge/submissions/s1.csv)，线上 `0.94666`。

主要模型：

- [torch_boundary_full_model.pt](/C:/budostudy/only_for_codex/iscc/work/data_security/system_log_anomaly_detection_challenge/models/torch_boundary_full_model.pt)
- [torch_boundary_full_seed43_model.pt](/C:/budostudy/only_for_codex/iscc/work/data_security/system_log_anomaly_detection_challenge/models/torch_boundary_full_seed43_model.pt)
- [torch_boundary_full_seed44_model.pt](/C:/budostudy/only_for_codex/iscc/work/data_security/system_log_anomaly_detection_challenge/models/torch_boundary_full_seed44_model.pt)
- [torch_hybrid_boundary_full_model.pt](/C:/budostudy/only_for_codex/iscc/work/data_security/system_log_anomaly_detection_challenge/models/torch_hybrid_boundary_full_model.pt)
- [torch_hybrid_boundary_full_seed43_model.pt](/C:/budostudy/only_for_codex/iscc/work/data_security/system_log_anomaly_detection_challenge/models/torch_hybrid_boundary_full_seed43_model.pt)
- [torch_hybrid_boundary_full_seed44_model.pt](/C:/budostudy/only_for_codex/iscc/work/data_security/system_log_anomaly_detection_challenge/models/torch_hybrid_boundary_full_seed44_model.pt)

日志：

- [meta_boundary_clean_summary.json](/C:/budostudy/only_for_codex/iscc/work/data_security/system_log_anomaly_detection_challenge/logs/meta_boundary_clean_summary.json)
- [meta_boundary_candidates.csv](/C:/budostudy/only_for_codex/iscc/work/data_security/system_log_anomaly_detection_challenge/processed/meta_boundary_clean_candidates.csv)
- [span_ranker_summary.json](/C:/budostudy/only_for_codex/iscc/work/data_security/system_log_anomaly_detection_challenge/logs/span_ranker_summary.json)
- [final_submission_decision_20260502.md](/C:/budostudy/only_for_codex/iscc/work/data_security/system_log_anomaly_detection_challenge/logs/final_submission_decision_20260502.md)
- [hybrid_boundary_run_summary_20260502.md](/C:/budostudy/only_for_codex/iscc/work/data_security/system_log_anomaly_detection_challenge/logs/hybrid_boundary_run_summary_20260502.md)

## 不建议提交

- `posterior_semantic_tail_model.json` 相关语义弱规则版本：本地曾降到约 `0.7225`，已判定失败。
- 纯规则提交：线上上限明显低于深度学习方案。
- 未经反馈直接提交 `s2/s3/s4`：它们未线上验证，且 `final.csv` 的 holdout proxy 更强。

## 未验证风险

- `go.csv` 尚未线上提交验证。
- `go.csv` 的风险来自 8 个新增边界修正；它保留 `push.csv` 全部已验证收益，且不改检测计数和类型，但仍可能受 public/private 分布差异影响。
- `up.csv` 已线上验证低于 `push.csv`，不再作为推荐提交。
- `safe8.csv` 更稳但潜在提升幅度较小；`hard.csv` 潜在提升更大但更可能重复边界过拟合问题。
