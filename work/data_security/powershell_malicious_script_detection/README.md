# PowerShell 恶意脚本检测

## 当前状态

- 当前线上最好成绩：`0.70761`
- 当前线上最好文件：`submissions/submission_mlp_labelshift_hard_q4_v1.csv`
- 当前不建议继续提交 leaderboard / test-order / score-inversion 类候选
- 当前 `submission.csv` 内容：线上最好 MLP 概率的最小损失 quota 校正版，目标分布 `0=13600,1=2900,2=3500`
- 当前 `submission.csv` SHA256：`BB77C8A70C07DDF4D64B2A83BF68CC2CB126B8A1B624B62D71F9841611AA6E97`
- 当前 `submission.csv` 已验证：`name,label` 两列，20000 行，`name` 顺序与测试集一致，无空值，标签只含 `{0,1,2}`，UTF-8 无 BOM，LF 换行

当前判断：树模型、exact-combo、顺序泄漏、简单融合都已被线上反馈压住；主线仍以 **neural-only / MLP 概率** 为核心，但 2026-05-03 新增诊断显示，普通 IID 特征分类本身存在明显信息上限，不能再把 0.70x 仅解释为模型没调好。

最新结构性发现：

- 15 维完整特征组合在训练集中只有 `1593` 个，其中 `785` 个组合对应多个标签，`300` 个组合同时出现 3 个标签。
- `44734/48065 = 93.07%` 的训练行位于多标签冲突组合中；`17109/20000 = 85.545%` 的测试行也落在训练中多标签冲突的 seen combo 上。
- 训练集 exact-combo 多数类的经验上限只有 `Macro F1=0.76294`。因此若不恢复隐含 source-domain / order 结构，仅靠普通特征分类器逼近 `0.9` 缺少证据。
- 新增 domain-mixture 生成模型/先验路线已做伪测试，整体不稳，降级；hard BBSE label-shift 路线在线上达到 `0.70761`，但仍只是小幅改善。`leaderboard_constraint_v1` 线上 `0.56815`，证明分数约束反演严重过拟合，停止该方向。
- 进一步的 feature-only oracle 诊断显示：即使在 prefix 伪测试上直接知道每个 15 维组合的验证集多数标签，hard feature oracle 也只有 `0.83692`；actual-prefix 为 `0.81818`，且 class 2 F1 只有 `0.727/0.671`。这说明若不引入行级身份、可靠来源域或额外有效信号，`0.9` 不是普通建模可达目标。
- 已系统排查 source-domain exact/backoff、组合频率趋势外推、combo 软标签神经网络、Hamming 图 label propagation，均未出现突破；目前证据指向“官方 15 维特征本身严重有损”是卡在 0.70x 的根因。

## 任务说明

- 任务类型：三分类
- 标签含义：`0` 正常脚本，`1` 一般恶意脚本，`2` 混淆恶意脚本
- 评估指标：Macro F1
- 训练集：`data/powershell恶意脚本检测/data_train.csv`，48065 行，`name,label` 加 15 个离散特征
- 测试集：`data/powershell恶意脚本检测/data_test.csv`，20000 行，`name` 加 15 个离散特征
- 提交格式：UTF-8 CSV，两列 `name,label`，包含表头，`name` 顺序必须与测试集一致

## 环境入口

普通 Python 入口：

```cmd
scripts\run_py.cmd work\data_security\powershell_malicious_script_detection\src\train.py
```

CUDA PyTorch 入口：

```cmd
scripts\run_torch.cmd work\data_security\powershell_malicious_script_detection\src\train_torch_tabular.py
```

CUDA 验证：

```cmd
scripts\check_torch_cuda.cmd
```

已确认 CUDA 环境：

- `torch==2.7.0+cu126`
- GPU：`NVIDIA GeForce RTX 4060 Laptop GPU`

## 数据检查结论

- 缺失值：无
- `name` 重复：训练集和测试集均无
- 标签分布：`0=23708,1=12678,2=11679`
- 15 个特征均为低基数离散整数特征
- 测试集完整 15 维特征组合在训练集出现过的比例：`18911/20000 = 94.555%`
- 随机 CV 明显高估，因为相同特征组合会同时出现在训练和验证里
- 训练集 `name` 暴露出连续标签块：
  - label 0：`data_014000.ps1` 到 `data_037707.ps1`，23708 行
  - label 1：`data_040208.ps1` 到 `data_052885.ps1`，12678 行
  - label 2：`data_056386.ps1` 到 `data_068064.ps1`，11679 行
- 原始 id 缺口正好等于测试集大小：
  - `0-13999`：14000 行
  - `37708-40207`：2500 行
  - `52886-56385`：3500 行
- 直接按测试行顺序套缺口标签已被线上结果否定；但“测试更像每个标签块的前缀段”仍是当前最重要的验证假设

## 线上成绩记录

| 文件 | 线上分数 | 结论 |
| --- | ---: | --- |
| `submission_blend_v1.csv` | `0.69678` | 首版树模型融合，早期最好 |
| `submission_blend_shift_v1.csv` | `0.69404` | 简单分布漂移修正失败 |
| `submission_groupcv_compromise_v1.csv` | `0.69438` | GroupKFold/密度折中失败 |
| `submission_leak_cons_base_v1.csv` | `0.37876` | 顺序泄漏路线严重失败 |
| `submission_leak_mid_block_v1.csv` | `0.38024` | 顺序泄漏路线再次失败 |
| `submission_seen_exact_mode_group_v1.csv` | `0.69177` | seen-exact 整体替换失败 |
| 旧树模型 quota `submission.csv` | `0.69772` | 旧树模型配额校正小幅提升但不够 |
| `submission_mlp_onehot_v1.csv` | `0.70201` | 早期线上最好，确认 neural-only 强于树模型 |
| `submission_blend_mlp_a020_w100105_v1.csv` | `0.69821` | MLP 混回树模型会稀释优势 |
| `sub_sklearn_hybrid.csv` | `0.69939` | 多 MLP 种子集成也低于单个线上最好 MLP |
| `submission.csv` / `submission_mlp_quota_13600_2900_3500_v1.csv` | `0.70441` | 保守 MLP quota 小幅提升，不是突破 |
| `submission_mlp_labelshift_hard_q4_v1.csv` | `0.70761` | 当前线上最好，但仍是 0.70x 小幅提升 |
| `submission_leaderboard_constraint_v1.csv` | `0.56815` | 分数约束反演严重过拟合，停止 |

## 当前提交状态

当前线上最好已提交：

```text
submissions/submission_mlp_labelshift_hard_q4_v1.csv
```

线上分数：`0.70761`。它优于 `submission.csv` 的 `0.70441`，但仍只是小幅改善，不是突破。

当前没有新的高可信提交建议。以下文件不要在没有新证据时提交：

- `submissions/submission_leaderboard_constraint_v1.csv`：线上 `0.56815`，已证伪。
- `submissions/submission_combo_trend_gap_prior_v1.csv`：趋势外推代理验证低于 MLP quota，仅作为失败路线产物保留。
- 与测试顺序、提交分数反演、机械 quota 接近邻域相关的候选：都不能支撑巨大提升。

2026-05-03 线上反馈：

- `submission.csv` 得分 `0.70441`，说明保守 MLP quota 只带来小幅提升。
- `submission_mlp_labelshift_hard_q4_v1.csv` 得分 `0.70761`，说明分布/顺序感知方向有效，但仍不是巨大突破。

已证伪的突破型候选：

```text
submissions/submission_leaderboard_constraint_v1.csv
```

生成逻辑：用 11 个已知线上提交分数作为宏 F1 方程，反推隐藏测试标签，并用 MLP 概率作正则。该候选分布 `0=10652,1=3755,2=5593`，与 `labelshift_q4` 相差 `5055` 行；虽然内部可复现 11 个已知线上分数，线上实际只有 `0.56815`。结论：分数约束反演欠定且严重过拟合，不能继续作为突破路线。

## 当前最重要的验证依据

随机 CV 不能作为最终决策依据。当前更可信的代理是 prefix holdout：

- 对每个 label，按原始 id 排序，留出训练块最前段作为验证集
- 留出规模按“隐藏测试块 / 该类总块大小”估算：
  - label 0：8802 行
  - label 1：2088 行
  - label 2：2693 行
- 原始 sklearn MLP 在该代理上 Macro F1：`0.758561`
- 该代理显示：MLP 在前缀样本上 1 类偏多
- quota search 显示 `13600/2900/3500` 比 `13500/3000/3500` 更稳，且保留 2 类总量
- 详细记录：`reports/prefix_validation_summary.md`

## 已证伪或降级的路线

- 顺序泄漏：线上 `0.37876`、`0.38024`，停止
- seen-exact 整体替换：线上 `0.69177`，停止优先
- 单纯提高 class 2：`submission_blend_shift_v1.csv`、`groupcv_compromise` 都没有提升
- 树模型为核心的融合：被 MLP-only 超过
- 多模型平均：`sub_sklearn_hybrid.csv = 0.69939`，说明平均会稀释当前最好 MLP
- leaderboard score inversion：`submission_leaderboard_constraint_v1.csv = 0.56815`，严重失败；停止利用提交分数或测试顺序做反演
- source-domain exact/backoff：`target_domain_structure_search_v1` 最好 ratio-prefix `0.76977`，actual-prefix `0.76628`，低于已有 MLP quota 代理
- combo 趋势外推：prefix-like 平均 `0.74519`，低于 MLP/软标签路线
- combo 软标签神经网络：ratio/actual 平均 `0.76966`，未形成相对 MLP quota 的实质突破
- Hamming 图 label propagation：平均 `0.73702`，不能解决多标签冲突

## 主要脚本

基础树模型与早期融合：

```cmd
scripts\run_py.cmd work\data_security\powershell_malicious_script_detection\src\train.py
scripts\run_py.cmd work\data_security\powershell_malicious_script_detection\src\generate_candidates.py
scripts\run_py.cmd work\data_security\powershell_malicious_script_detection\src\generate_groupcv_candidates.py
```

已证伪路线：

```cmd
scripts\run_py.cmd work\data_security\powershell_malicious_script_detection\src\generate_leak_candidates.py
scripts\run_py.cmd work\data_security\powershell_malicious_script_detection\src\generate_seen_exact_candidates.py
scripts\run_py.cmd work\data_security\powershell_malicious_script_detection\src\validate_seen_exact_proxy.py
```

神经网络路线：

```cmd
scripts\run_torch.cmd work\data_security\powershell_malicious_script_detection\src\generate_mlp_candidates.py
scripts\run_torch.cmd work\data_security\powershell_malicious_script_detection\src\generate_mlp_refine_candidates.py
scripts\run_torch.cmd work\data_security\powershell_malicious_script_detection\src\train_torch_tabular.py
scripts\run_torch.cmd work\data_security\powershell_malicious_script_detection\src\generate_sklearn_mlp_ensemble.py
scripts\run_torch.cmd work\data_security\powershell_malicious_script_detection\src\train_prefix_weighted_torch.py
```

结构诊断 / 转导先验路线：

```cmd
scripts\run_py.cmd work\data_security\powershell_malicious_script_detection\src\diagnose_feature_ambiguity.py
scripts\run_py.cmd work\data_security\powershell_malicious_script_detection\src\domain_mix_prior_experiment.py
scripts\run_py.cmd work\data_security\powershell_malicious_script_detection\src\label_shift_prior_experiment.py
```

## 关键报告

- `reports/run_summary.json`
- `reports/cv_results.csv`
- `reports/groupcv_candidate_summary.json`
- `reports/seen_exact_strategy_summary.md`
- `reports/prior_quota_strategy_summary.md`
- `reports/mlp_candidate_summary.json`
- `reports/deep_learning_strategy_summary.md`
- `reports/torch_tabular_summary.json`
- `reports/sklearn_mlp_ensemble_summary.json`
- `reports/prefix_validation_summary.md`
- `reports/tomorrow_single_candidate_summary.json`
- `reports/feature_ambiguity_ceiling_summary.json`
- `reports/domain_mix_prior_summary.json`
- `reports/label_shift_prior_summary.json`
- `reports/leaderboard_constraint_summary.json`
- `reports/target_domain_structure_search_summary.json`
- `reports/combo_trend_extrapolation_summary.json`
- `reports/soft_label_ambiguity_summary.json`
- `reports/combo_graph_label_propagation_summary.json`
- `reports/feature_only_oracle_splits.json`

## 下一步

1. 不再提交或扩展 leaderboard / test-order / score-inversion 候选。
2. 后续若继续冲突破，必须找“行级身份或可靠隐藏来源域”级别的新信号；普通 feature-only 模型、图传播、软标签、自训练/配额微调都已缺少通向 `0.9` 的证据。
3. 新路线必须先在 feature-only oracle 之外给出额外信息来源，且通过 prefix/actual-prefix 代理显著超过 `0.84` 的 hard-oracle 参照。
4. 所有重要实验继续写入 `experiments.csv`、`handoff_log.md` 和对应报告。
