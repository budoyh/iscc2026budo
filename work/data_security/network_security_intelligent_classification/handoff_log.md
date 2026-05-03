# 数据安全赛工作日志与交接

## 当前状态

- 题目：网络安全智能分类挑战
- 任务：12 类网络安全事件分类，标签为 `class_0` 到 `class_11`
- 指标：宏平均 F1
- 当前已知最高线上分：`c04_c03_drop_pattern_50.csv = 0.71704`
- 当前榜首参考：`0.71968`
- 当前主推未提交候选：暂无；`c04` 已提交且明显提升，需要继续扩大 pattern-nuisance。
- 当前备选未提交候选：`upload_ready_breakthrough4\c05_c03_drop_pattern_40_safe.csv`
- 每日提交次数有限，后续每轮只给 1 个主提交和最多 1 个备选。

## 环境边界

普通机器学习运行入口：

```powershell
C:\budostudy\only_for_codex\iscc\scripts\run_py.cmd
```

PyTorch/CUDA 运行入口：

```powershell
C:\budostudy\only_for_codex\iscc\scripts\run_torch.cmd
```

CUDA 已验证：

```text
torch: 2.7.0+cu126
cuda available: True
cuda version: 12.6
device: NVIDIA GeForce RTX 4060 Laptop GPU
```

不要使用系统 Python、Anaconda 全局环境或 `.venv`。

## 数据检查结论

- `train_data.csv`：`53477` 行，50 个数值特征，包含 `id,label`。
- `test_data.csv`：`19440` 行，50 个数值特征，包含 `id`。
- 训练/测试均无缺失值。
- `id` 唯一且递增，但只用于提交，不进入模型。
- 训练标签顺序随机，测试预测顺序也无明显分段结构。
- 训练/测试协变量漂移强，domain classifier AUC 约 `0.91`。
- 随机 CV 对线上分数高估严重，需要结合线上反馈和测试域代理验证选模型。

## 线上反馈时间线

| 阶段 | 文件/方案 | 线上分数 | 结论 |
|---|---|---:|---|
| 初版 | `lgbm_adv_weighted_f5_s42` / `r00` | `0.68887` | 对抗权重单模不足 |
| r 系列 | `r01_blend_raw_std_xgb_coral.csv` | `0.68154` | CORAL/split-standard 融合方向偏负 |
| r 系列 | `r02_blend_raw_xgb.csv` | `0.69083` | LGBM+XGB 基线有效 |
| r 系列 | `r03_blend_raw_xgb_soft_uniform.csv` | `0.69291` | soft uniform prior 有效 |
| n 系列 | `n01_raw_xgb_uniform_a1p5.csv` | `0.69266` | uniform alpha 过强不涨 |
| n 系列 | `n13_target_class1_11_high.csv` | `0.69186` | 手工类别偏置无效 |
| d 系列 | `d02_raw_xgb_ft_pseudo_uniform_a1p0.csv` | `0.69616` | 深度模型 + 伪标签 + uniform 有效 |
| m 系列 | `m01_proxy_iter2_uniform_a1.csv` | `0.69848` | 迭代伪标签有效 |
| b 系列 | `b01_target_aug_high.csv` | `0.70584` | 目标域增强有效 |
| c 系列 | `c01_b01_iter_target_aug_high.csv` | `0.71035` | 当前已知最高，b01 教师迭代有效 |
| c 系列 | `c03_c01_target_b01_proxy_reblend.csv` | `0.71059` | 当前已知最高；proxy 有方向但只小涨 |

## 实验记录摘要

完整记录见 `experiments.csv`。关键 run 如下：

| run_id | 方法 | 本地/代理分数 | 备注 |
|---|---|---:|---|
| `lgbm_base_f5_s42` | LightGBM | `0.9296851169` | 基础单模 |
| `xgb_base_f5_s42` | XGBoost | `0.9295233752` | 基础单模 |
| `lgbm_adv_weighted_f5_s42` | LightGBM + domain weights | `0.9287215352` | 线上 `0.68887` |
| `lgbm_base_split_standard_f5_s42` | split standard | `0.9298942313` | 单独不差，但线上方向不佳 |
| `lgbm_base_coral_f5_s42` | CORAL | `0.9201414219` | 弱，仅少量融合试过 |
| `mlp_standard_f5_s42` | sklearn MLP | `0.8620916269` | 弱 |
| `torch_resnet_raw_rank_dann015_cw_f5_s42` | PyTorch ResNet + DANN | `0.8927968364` | 单模弱 |
| `torch_ft_raw_dann010_cw_f5_s42` | FT-Transformer + DANN | `0.9011105898` | 融合多样性有效 |
| `torch_ft_raw_dann030_cw_f5_s43` | 第二个 FT-Transformer | `0.9012668211` | 融合多样性有效 |
| `pseudo_lgbm_ftblend_t098_w035_f5_s42` | 高置信伪标签 LGBM | `0.9306613353` | 9438 个伪标签 |
| `pseudo_lgbm_iter2_twoft_uniform100_t097_w045_f5_s42` | 二代伪标签 LGBM | `0.9308982992` | 10119 个伪标签 |
| `catboost_raw_bal_d6_lr005_f5_s42` | CatBoost | `0.9102305212` | 弱，未进入最终 |
| `target_aug_lgbm_m01_pow2_aug045_t097_w035_f5_s42` | 目标域矩增强 | `0.9313765419` | 9740 个伪标签 |
| `target_aug_lgbm_m01_pow1p5_aug060_t095_w045_f5_s42` | 更激进目标域矩增强 | `0.9316003006` | 11035 个伪标签 |
| `b01_target_aug_high` | 目标域增强融合 | `0.9328080314` | A 榜 `0.70584` |
| `b02_target_aug_mid` | 目标域增强融合 | `0.9326931022` | 当前备选 |
| `pseudo_lgbm_iter3_b01_uniform100_t095_w050_f5_s42` | 三代伪标签 LGBM | `0.9312698260` | 11928 个伪标签 |
| `target_aug_lgbm_b01_pow1p2_aug080_t093_w055_f5_s42` | b01 教师目标域增强 | `0.9323161136` | 12785 个伪标签 |
| `target_aug_lgbm_b01_pow0p8_aug100_t090_w065_f5_s42` | 更激进目标域增强 | `0.9318755648` | 13670 个伪标签，偏弱 |
| `c01_b01_iter_target_aug_high` | b01 迭代目标增强融合 | `0.9330872501` | A 榜 `0.71035`，当前已知最高 |
| `c02_b01_iter_target_aug_safe` | b01 迭代目标增强融合 | `0.9330908438` | 旧备选，diff_vs_b01=136 |
| `local_cluster_diag_c01_k3_pow1p2_aug075_t093_w055_n700_f5_s42` | 子簇级目标增强 | `0.9318447231` | 单模弱，target-like proxy 弱，不单独提交 |
| `target_aug_domain_c01_pow1p2_aug080_t093_w055_g050_n700_f5_s42` | domain-focus 目标增强 | `0.9321018054` | 单模未超过 c01，不单独提交 |
| `lgbm_stable_drop10_smdstd_n700_f5_s42` | 删除最漂移特征 | `0.9073602925` | 证明漂移特征同时承载强类别信号，删除不可行 |
| `c03_c01_target_b01_proxy_reblend` | c01 与目标增强单模代理重融合 | `0.9332779796` | 新主推；target-like proxy top20/top30 明显高于 c01，diff_vs_c01=130 |

## 2026-05-03 突破诊断

- 新增 `src\analyze_breakthrough.py`：domain classifier AUC `0.908819`，随机 OOF 仍不可直接信；top20/top30 目标相似训练子集更能区分 b01/c01。
- 普通无监督 KMeans 不适合直接簇到类别匹配：`k=96` 时训练簇多数类 F1 仅 `0.355`，簇纯度低。
- 三个颠覆方向被本地否掉：子簇目标增强单模、domain-focus 单模、删除漂移特征稳定子集，均未超过 c01 的目标相似代理。
- EM/BBSE/test prior 估计没有超过 c01 的 uniform prior；c01 的 uniform alpha=1.0 仍是当前最稳的概率校准。
- 代理重融合显示 `0.6*c01_raw + 0.4*target_aug_lgbm_b01_pow1p2` 后再做 uniform prior 最强：top20 `0.925217`、top30 `0.924993`，高于 c01 的 top20 `0.923573`、top30 `0.923755`。

## 2026-05-03 继续突破尝试

- `c03` 已提交，A 榜 `0.71059`，仅比 `c01=0.71035` 高 `0.00024`，说明旧 proxy 已明显过乐观。
- subagent 独立审查指出 `pattern_*` 是强域弱类风险源：域 AUC 高但单组标签 F1 很弱，且多个 pattern 特征高度冗余。
- 训练 drop-pattern 目标增强模型：单模 OOF 不高，但与 c03 融合显著提升 proxy，说明它主要提供域指纹抑制的互补边界。
- 当前主推 `c04_c03_drop_pattern_50.csv`：`0.5*c03 + 0.5*target_aug_drop_pattern_c03...`，OOF `0.9348153887`，target-like proxy top20/top30 `0.926775/0.926335`，高于 c03 的 `0.925217/0.924993`，相对 c03 改 `569` 行。
- 当前备选 `c05_c03_drop_pattern_40_safe.csv`：`0.6*c03 + 0.4*drop-pattern`，OOF `0.9348213724`，proxy `0.926250/0.925940`，相对 c03 改 `448` 行。
- 已排除：class-center transport 首版 OOF 伤分；PCA 低维去冗余 OOF 仅 `0.84`；分位数增强单模弱；多视图稳定伪标签几乎筛不掉高置信样本，说明各视图共享同一错误边界。

## 方法结论

有效方向：

- LGBM+XGB 是稳定基线。
- soft uniform prior 已被线上反馈验证有效。
- FT-Transformer 单模弱，但作为概率融合多样性有效。
- 高置信测试伪标签有效，二代伪标签继续小幅提升。
- 类别条件目标域矩增强是当前最有潜力的突破方向。

无效或低优先级方向：

- CORAL/split-standard 大权重融合线上下降。
- 手工 class_1/class_11 偏置线上下降。
- 更强 uniform alpha 没带来提升。
- CatBoost 单模和小权重融合无收益。
- kNN/标签传播过弱。
- 测试 kNN 图平滑代理验证下降。
- `id` 或行顺序没有可利用规律。
- 普通 KMeans/子簇直接匹配纯度太低，不能作为直接标签分配突破口。
- 简单删除最漂移特征会严重伤类别信号，不建议继续做粗暴特征剔除。

## 当前推荐提交

只建议先提交：

```text
upload_ready_breakthrough4\c04_c03_drop_pattern_50.csv
```

如需保守备选，再提交：

```text
upload_ready_breakthrough4\c05_c03_drop_pattern_40_safe.csv
```

不要同时提交多个旧目录候选；`d02/m01/b01/c01/c03` 已有线上反馈，旧文件继续提交价值较低。

## 文件格式校验

`c04_c03_drop_pattern_50.csv` 和 `c05_c03_drop_pattern_40_safe.csv` 均已校验通过：

- 行数：`19440`
- 列名：`id,label`
- ID 顺序：与 `sample_submission.csv` 完全一致
- 空值：`0`
- 重复 ID：`0`
- 标签合法：全部属于 `class_0` 到 `class_11`
- 文件头示例：

- `c04` 相对 `c03` 改动：`569` 行；标签计数：`class_0=1796, class_1=846, class_10=1546, class_11=1216, class_2=2018, class_3=1935, class_4=1523, class_5=1813, class_6=1747, class_7=1545, class_8=1856, class_9=1599`
- `c05` 相对 `c03` 改动：`448` 行；标签计数：`class_0=1795, class_1=849, class_10=1551, class_11=1213, class_2=2016, class_3=1933, class_4=1524, class_5=1815, class_6=1747, class_7=1537, class_8=1864, class_9=1596`

不要上传同名 `.json` 报告文件。

## 关键命令记录

PyTorch CUDA 检查：

```powershell
cd C:\budostudy\only_for_codex\iscc
scripts\check_torch_cuda.cmd
```

训练 FT-Transformer：

```powershell
cd C:\budostudy\only_for_codex\iscc
scripts\run_torch.cmd work\data_security\network_security_intelligent_classification\src\train_torch_tabular.py --model fttransformer --features raw --domain-weight 0.10 --class-weight --seed 42 --run-id torch_ft_raw_dann010_cw_f5_s42
```

训练二代伪标签模型：

```powershell
cd C:\budostudy\only_for_codex\iscc\work\data_security\network_security_intelligent_classification

C:\budostudy\only_for_codex\iscc\scripts\run_py.cmd src\train_pseudo_lgbm.py --source-runs lgbm_base_f5_s42 xgb_base_f5_s42 torch_ft_raw_dann010_cw_f5_s42 torch_ft_raw_dann030_cw_f5_s43 pseudo_lgbm_ftblend_uniform100_t098_w035_f5_s42 --source-weights 0.43 0.14 0.06 0.06 0.31 --soft-prior uniform --soft-alpha 1.0 --threshold 0.97 --pseudo-weight 0.45 --n-estimators 1400 --run-id pseudo_lgbm_iter2_twoft_uniform100_t097_w045_f5_s42
```

训练目标域增强模型：

```powershell
cd C:\budostudy\only_for_codex\iscc\work\data_security\network_security_intelligent_classification

C:\budostudy\only_for_codex\iscc\scripts\run_py.cmd src\train_target_aug_lgbm.py --source-runs lgbm_base_f5_s42 xgb_base_f5_s42 torch_ft_raw_dann010_cw_f5_s42 torch_ft_raw_dann030_cw_f5_s43 pseudo_lgbm_ftblend_t098_w035_f5_s42 pseudo_lgbm_ftblend_uniform100_t098_w035_f5_s42 pseudo_lgbm_iter2_twoft_uniform100_t097_w045_f5_s42 --source-weights 0.449625 0.113030 0.111684 0.050832 0.019178 0.004121 0.251531 --soft-alpha 1.0 --target-power 2.0 --aug-weight 0.45 --pseudo-threshold 0.97 --pseudo-weight 0.35 --n-estimators 1400 --run-id target_aug_lgbm_m01_pow2_aug045_t097_w035_f5_s42
```

生成 `b01`：

```powershell
cd C:\budostudy\only_for_codex\iscc\work\data_security\network_security_intelligent_classification

C:\budostudy\only_for_codex\iscc\scripts\run_py.cmd src\write_blend_submission.py --run-ids lgbm_base_f5_s42 xgb_base_f5_s42 torch_ft_raw_dann010_cw_f5_s42 torch_ft_raw_dann030_cw_f5_s43 pseudo_lgbm_ftblend_t098_w035_f5_s42 pseudo_lgbm_ftblend_uniform100_t098_w035_f5_s42 pseudo_lgbm_iter2_twoft_uniform100_t097_w045_f5_s42 target_aug_lgbm_m01_pow2_aug045_t097_w035_f5_s42 target_aug_lgbm_m01_pow1p5_aug060_t095_w045_f5_s42 --weights 0.277615 0.069789 0.068958 0.031386 0.011841 0.002544 0.155305 0.075962 0.306600 --soft-prior uniform --soft-alpha 1.0 --output-dir upload_ready_breakthrough --output-name b01_target_aug_high
```

## 下一步计划

收到 `c04/c05` 线上分数后：

1. 如果 `c04` 明显提升，继续扩大 pattern-nuisance 路线：训练 XGB/drop-pattern、drop-pattern 目标增强多 seed，并探索 pattern 随机置换增强。
2. 如果 `c04` 小涨但不大，提交 `c05` 判断 50% 权重是否过激。
3. 如果 `c04` 下降，说明 pattern 特征虽有域指纹但线上仍需要它，退回 c03，并转向相邻类/分组标签后处理或深度自监督。
4. 所有新线上分数必须同步写入 `experiments.csv` 和本文档。

## 注意事项

- 每轮最多给一个主提交和一个备选，避免浪费每日次数。
- 本地 OOF 高不代表线上高，必须结合线上反馈。
- 不要上传 `.json`。
- 不要从 `submissions` 目录随意挑旧文件上传，优先使用当前明确推荐的 `upload_ready_breakthrough3`。

## 2026-05-03 c04 反馈后继续突破

- 用户反馈 `c04_c03_drop_pattern_50.csv = 0.71704`，这是当前最大线上涨幅，证明 `pattern_*` 去指纹是真方向。
- 两个 subagent 独立结论一致：没有全局 label remap、ID、duplicate 泄漏证据；`pattern_*` 是强域弱类信号，`behavior_template_*` 是强模板坐标但已被 c04 大体吸收。
- 高纯 template hard override 被 OOF 否决：规则覆盖 OOF `4562` 行、规则准确率 `0.9879`，但与 c04 冲突的 `55` 行里规则准确率仅 `0.018`、c04 正确率 `0.945`，因此不提交硬覆盖。
- `pattern-replace` 单模 OOF `0.9321004057`，只提供很小融合信号；`pattern-shuffle` 单模 OOF `0.9313960634`，但与 c04 融合在 0.10 权重附近正向。
- 已生成 `upload_ready_breakthrough5\c08_c04_patternshuffle_10.csv`，OOF `0.9351027719`，格式校验通过。
- 模板簇 soft prior 比硬覆盖有效：在 c08 上用 template k384、alpha=0.5、smooth=30 做 cluster-conditional prior，OOF 升至 `0.9353962398`，测试仅改 `89` 行；已生成当前主候选 `upload_ready_breakthrough5\c09_c08_template_prior_k384_a050.csv`。
- 激进备选为 `upload_ready_breakthrough5\c06_c03_drop_pattern_55.csv`，是 c04 线上验证后的 drop-pattern 权重外推；OOF `0.9344882520`，风险高于 c09。
- 已排除：c04 教师版 drop-pattern 自蒸馏（OOF `0.9305887279`）、drop pattern + tail/high spike（OOF `0.9305451791` 且融合弱）、pattern PCA 去首主成分（OOF `0.9308088066`，target-like top20 好但 full/top50 伤分）。

## 2026-05-03 seed43 pattern-shuffle 更新

- seed43 的 n700 pattern-shuffle 单模 OOF `0.9322716891`，比 seed42 n700 更高；与 c04 融合时 0.25 权重最稳。
- 已生成 `upload_ready_breakthrough5\c10_c04_patternshuffle_s43_25.csv`，即 `0.75*c04 + 0.25*seed43 pattern-shuffle`，OOF `0.9354188187`，格式校验通过。
- 在 c10 上继续做 template k384 alpha=0.5 soft prior，OOF 升至 `0.9358588308`，测试改 `95` 行；已生成当前主候选 `upload_ready_breakthrough5\c11_c10_template_prior_k384_best.csv`。
- 当前只建议提交 `c11`。若担心 template prior 线上过拟合，备选提交 `c10`，不要再提交 c06/c07/c08/c09。

## 2026-05-03 c11 线上反馈

- 用户反馈 `c11_c10_template_prior_k384_best.csv = 0.71839`，较 `c04=0.71704` 继续正向，但仍低于最新最高分 `0.72347`。
- 结论：seed43 pattern-shuffle 和 template soft prior 都是线上有效机制，但增幅仍是小步；下一轮应扩大这两个机制，而不是回到无证据的硬覆盖或粗暴特征删除。

## 2026-05-03 c11 反馈后扩展 template prior

- 以 c10 为 base 重新扫 template cluster prior，k384/alpha0.5 的 c11 不是最优；k768、smooth=15、alpha=1.2 在 OOF 达到 `0.9365995004`，改动 OOF 行准确率 `0.536`，base 对这些行仅 `0.388`。
- 已生成 `upload_ready_breakthrough6\c12_c10_template_prior_k768_a120_s15.csv`，格式校验通过，测试改 `274` 行；这是当前主提交。
- 更激进的 k768/smooth15/alpha1.4 OOF `0.9366079590` 略高，但测试改 `316` 行且改动精度略低，暂不生成提交，避免过激。
- seed44 pattern-shuffle 单模 OOF `0.9316427979`，融合弱于 seed43，不纳入下一提交。

## 2026-05-03 c12 线上反馈

- 用户反馈 `c12_c10_template_prior_k768_a120_s15.csv = 0.72041`，相对 `c11=0.71839` 再涨 `0.00202`，说明更强 template prior 线上有效。
- 最新最高分约 `0.72347`，当前差距约 `0.00306`。下一步继续围绕 template cluster soft prior 做 k/smooth/alpha/多视图扩展，同时检查是否存在更高覆盖的局部模板结构。

## 2026-05-03 c12 后继续突破：pair gate 与独立局部专家

- 没有继续盲目加 template alpha，而是把 c12 的 412 个 OOF 改动拆成 base->prior flow，发现只有部分 flow 真正正向。
- `c13_c12_positive_flow_filter.csv`：只保留 OOF 正收益流，OOF `0.9366681171`，测试改 `185` 行；`c14/c15` 分别测试 alpha1.4/1.0，均弱于 c13。
- `c16_pairwise_gate_template_top10_t050.csv`：对 10 个正收益 flow 训练二分类 gate，OOF `0.9369204286`，测试改 `99` 行；改动行 OOF 新标签准确率 `0.729`，base 在同批行仅 `0.201`。
- `c20/c21`：对每个 flow 单独搜索 gate 阈值。宽网格 `c21_pairwise_gate_perpair_thresholds_wide.csv` 最强，OOF `0.9370715092`，测试改 `109` 行。
- subagent 独立分析指出 c16 的瓶颈不是 gate 精度，而是 c12 候选太窄，特别是 `class_3 -> class_8`。据此新增 `src/independent_pair_proposal.py`，训练 full-OOF 二分类专家，在所有验证行上预测，避免只在真实 pair 类上 OOF 造成乐观偏差。
- `c22_c21_independent_3to8_best.csv`：在 c21 上独立补 `class_3 -> class_8`，阈值 `0.90`，OOF `0.9372681548`，额外测试改 `64` 行；改动行 OOF 新准确率 `0.680`，原预测准确率 `0.240`。这是候选生成机制的实质突破，不再依赖 c12 是否先提出该行。
- `c23` 三分类 `0/10/11` 专家未过滤时虽有 OOF `0.9373345232`，但包含 `11->0` 等 OOF 负向 flow，不推荐。`c24_c22_triad_positive_flows.csv` 仅保留 `10->11`、`11->10` 正向 flow，OOF `0.9374293579`，只比 c22 多改 2 个测试行，可作为稳健备选。
- 独立 pair 扫描进一步发现 `8->1`、`6->5`、`11->10` 在 c22 后仍有正收益。`c25_c21_chain_independent_pairs.csv` 链式叠加 `3->8@0.90 + 8->1@0.90 + 6->5@0.95 + 11->10@0.80`，OOF `0.9374502087`，当前本地最高；格式校验通过，测试标签计数为 `class_0=1794, class_1=899, class_10=1560, class_11=1203, class_2=2012, class_3=1858, class_4=1520, class_5=1823, class_6=1739, class_7=1540, class_8=1895, class_9=1597`。
- 当前推荐：主提交 `upload_ready_breakthrough7/c25_c21_chain_independent_pairs.csv`；如果担心 `8->1` 覆盖 50 行偏激，保守备选 `upload_ready_breakthrough7/c24_c22_triad_positive_flows.csv`。`c17/c18/c19` 的 pattern6 prior 均弱或负向，后续低优先级。
