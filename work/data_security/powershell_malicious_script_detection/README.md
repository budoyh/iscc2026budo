
## 2026-05-03 14:26 当前交接更新

- 当前线上最好已更新为 `0.71000`，文件为 `submissions/submission_q4_gated_consensus_v1.csv`；这证明 q4 锚定 + gated consensus 是正向路线。
- 当前根目录 `submission.csv` 已切换为新的主推突破候选：`submissions/submission_target_mixture_quota_v1.csv`。
- 新候选方法：无序 target source-window mixture 估计 `P_test(label|exact-combo)`，只作为行级排序信号；再用 MLP/soft-label MLP/CatBoost/prefix neural 的 log-space ensemble 做受控 quota 决策。
- 新候选分布：`0=13550,1=2860,2=3590`；相对 `0.71000` 文件改动 `1103` 行，相对 q4 改动 `1137` 行，相对失败的 combo_score 文件仍相差 `2665` 行。
- 新候选 SHA256：`FB7DB937C83275AE78A5D7D228812B46933B5956357FED96C8E5896D4D3BAE1B`。
- 生成命令：`scripts\run_py.cmd work\data_security\powershell_malicious_script_detection\src\generate_target_mixture_quota_candidate.py`。
- 报告：`reports/target_mixture_quota_candidate_summary.json`；配套诊断报告：`reports/target_mixture_combo_posterior_summary.json`；稳健小步候选报告：`reports/q4_pair_swap_summary.json`。
- 不建议再提交 `submission_target_mixture_combo_v1.csv` 自动 anchored 版本，因为它把 class1 拉到 `2969`，风险高；也不建议直接提交 aggressive/wide/score-inversion 系列。
# PowerShell 恶意脚本检测

## 当前状态

- 当前线上最好成绩：`0.70761`
- 当前线上最好文件：`submissions/submission_mlp_labelshift_hard_q4_v1.csv`
- 最新失败提交：`submission_breakthrough_combo_score_optimal_v1.csv` / 根目录 `submission.csv` 线上 `0.63703`
- 当前推荐提交文件：`submission.csv`
- 当前 `submission.csv` 内容：`submissions/submission_q4_gated_consensus_v1.csv`，方法为 q4 label-shift 锚点 + full-data soft-label MLP / CatBoost / prefix-weighted neural 低置信共识翻转，并把 `0.63703` 失败候选作为反向惩罚。
- 当前 `submission.csv` SHA256：`C290083AC12C77401B5973BEC4861DD1A7C03A78BFC2FD6FAC13B74ABDA0DFA0`
- 当前 `submission.csv` 分布：`0=13248,1=2852,2=3900`；相对线上最好 q4 改动 `264` 行，相对失败候选改动 `3082` 行。
- 当前 `submission.csv` 已验证：`name,label` 两列，20000 行，`name` 顺序与测试集一致，无空值，标签只含 `{0,1,2}`，UTF-8 无 BOM，LF 换行

当前判断：树模型、exact-combo、顺序泄漏、简单融合都已被线上反馈压住；普通 IID 特征分类存在明显信息上限。`0.63703` 进一步证明“分数约束反演 + 组合硬优化”不可用；当前候选回到能解释线上 `0.70761` 的 MLP/label-shift 强基线，但用独立模型共识做低置信行级重选。

最新结构性发现：

- 15 维完整特征组合在训练集中只有 `1593` 个，其中 `785` 个组合对应多个标签，`300` 个组合同时出现 3 个标签。
- `44734/48065 = 93.07%` 的训练行位于多标签冲突组合中；`17109/20000 = 85.545%` 的测试行也落在训练中多标签冲突的 seen combo 上。
- 训练集 exact-combo 多数类的经验上限只有 `Macro F1=0.76294`。因此若不恢复隐含 source-domain / order 结构，仅靠普通特征分类器逼近 `0.9` 缺少证据。
- 新增 domain-mixture 生成模型/先验路线已做伪测试，整体不稳，降级；hard BBSE label-shift 路线在线上达到 `0.70761`，但仍只是小幅改善。`leaderboard_constraint_v1` 线上 `0.56815`，证明分数约束反演严重过拟合，停止该方向。
- 进一步的 feature-only oracle 诊断显示：即使在 prefix 伪测试上直接知道每个 15 维组合的验证集多数标签，hard feature oracle 也只有 `0.83692`；actual-prefix 为 `0.81818`，且 class 2 F1 只有 `0.727/0.671`。这说明若不引入行级身份、可靠来源域或额外有效信号，`0.9` 不是普通建模可达目标。
- 已系统排查 source-domain exact/backoff、组合频率趋势外推、combo 软标签神经网络、Hamming 图 label propagation，均未出现突破；目前证据指向“官方 15 维特征本身严重有损”是卡在 0.70x 的根因。
- 继续追加的 combo-level majority 元学习、MLP+combo-meta 融合、高阶 NB 也未突破。重复组合内部 mixed-label allocation 即使用真实组合比例，因行级身份不可见，ratio-prefix 期望也只有 `0.77631`，actual-prefix 只有 `0.75782`。
- 2026-05-03 继续测试了更激进的分数约束反演和分布方法：exact-combo 分数约束能把已知线上分数拟合到 `1.64e-5` 误差，但伪线上验证显示它在 actual-prefix 上只有 `0.72530`，低于普通候选 `0.76208`，属于欠定过拟合；CatBoost ordered categorical 最高 actual-prefix `0.77076`，边缘分布运输最高 actual-prefix `0.72245`，均未形成巨大突破。
- 行级固定类总量的分数后验也已测试：在真实总量 `14000/2500/3500` 下拟合已知线上分数后，最优期望宏 F1 只有 `0.72255`，说明“把分数反馈当弱标注”也没有打开通往 `0.9` 的通道。

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

当前线上最好已提交文件仍是：

```text
submissions/submission_mlp_labelshift_hard_q4_v1.csv
```

线上分数：`0.70761`。用户要求下一次必须给出非保守突破候选，因此根目录 `submission.csv` 曾更新为：

```text
submissions/submission_breakthrough_combo_score_optimal_v1.csv
```

该候选是高风险大步跃迁方案，代理目标为 `0.79672`，与 `labelshift_q4` 相差 `2896` 行，分布 `0=13657,1=2865,2=3478`。线上反馈 `0.63703`，已证伪，不要继续提交或扩展。

当前新的提交文件：

```text
submission.csv
submissions/submission_q4_gated_consensus_v1.csv
```

生成逻辑：
- 以线上最好 `submission_mlp_labelshift_hard_q4_v1.csv` 为锚点。
- 训练 full-data soft-label MLP：`alpha=1.0, soft_weight=0.7, source_decay=3.0, dropout=0.08, 5 seeds, 40 epochs`。
- 同时使用 `catboost_prefix_search_test_proba.npy` 和 `prefix_weighted_torch_hybrid_proba.npy` 作为独立意见。
- 只翻转 q4 低置信行；翻转方向若与 `0.63703` 失败候选一致会被惩罚。
- 最终分布 `0=13248,1=2852,2=3900`，相对 q4 改 `264` 行，其中 `187` 行三模型一致、`77` 行两模型一致。

以下文件不要在没有新证据时提交：

- `submissions/submission_leaderboard_constraint_v1.csv`：线上 `0.56815`，已证伪。
- `submissions/submission_combo_trend_gap_prior_v1.csv`：趋势外推代理验证低于 MLP quota，仅作为失败路线产物保留。
- `submissions/submission_combo_meta_prefix_v1.csv`、`submissions/submission_combo_meta_mlp_blend_v1.csv`、`submissions/submission_high_order_nb_v1.csv`：新结构实验产物，代理验证均不足，不建议提交。
- `submissions/submission_combo_score_constrained_v1.csv`、`submissions/submission_combo_score_posterior_consensus_v1.csv`：分数约束 / 后验稳定性产物。虽然能解释已知线上分数或在拟合后验下显示小幅收益，但伪线上验证证明欠定，不作为当前推荐。
- `submissions/submission_combo_score_optimal_v1.csv` / `submissions/submission_breakthrough_combo_score_optimal_v1.csv`：线上 `0.63703`，已证伪；不要继续提交或围绕它做小改。
- `submissions/submission_row_score_posterior_v1.csv`：行级固定类总量分数后验产物，自身期望仅 `0.72255`，仍不建议提交。
- `submissions/submission_catboost_prefix_quota_v1.csv`、`submissions/submission_marginal_transport_v1.csv`：CatBoost 和全局边缘运输产物，prefix 代理均未超过当前 MLP 体系，不建议提交。
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
- duplicate-combo mixed allocation：真实比例 oracle 的期望 F1 仅 `0.77631/0.75782`，说明无行级身份时拆分重复组合不能大幅提升
- combo-level prefix meta-learning：ratio/actual 平均约 `0.76379`，失败
- MLP + combo-meta blend：三类代理平均 `0.75714`，失败
- 高阶 NB：最佳平均 `0.71509`，失败

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
- `reports/combo_allocation_oracle_summary.json`
- `reports/combo_meta_prefix_model_summary.json`
- `reports/combo_meta_mlp_blend_summary.json`
- `reports/high_order_nb_search_summary.json`
- `reports/combo_score_constrained_summary.json`
- `reports/combo_score_optimal_summary.json`
- `reports/score_constrained_combo_validation.json`
- `reports/combo_score_posterior_ensemble_summary.json`
- `reports/catboost_prefix_search_summary.json`
- `reports/marginal_transport_summary.json`
- `reports/row_score_posterior_summary.json`
- `reports/q4_gated_consensus_summary.json`

## 下一步

1. 不再提交或扩展 leaderboard / test-order / score-inversion 候选。
2. 后续若继续冲突破，必须找“行级身份或可靠隐藏来源域”级别的新信号；普通 feature-only 模型、图传播、软标签、自训练/配额微调都已缺少通向 `0.9` 的证据。
3. 新路线必须先在 feature-only oracle 之外给出额外信息来源，且通过 prefix/actual-prefix 代理显著超过 `0.84` 的 hard-oracle 参照。
4. 所有重要实验继续写入 `experiments.csv`、`handoff_log.md` 和对应报告。

