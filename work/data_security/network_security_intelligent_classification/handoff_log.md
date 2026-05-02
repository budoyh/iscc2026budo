# 数据安全赛工作日志与交接

## 当前状态

- 题目：网络安全智能分类挑战
- 任务：12 类网络安全事件分类，标签为 `class_0` 到 `class_11`
- 指标：宏平均 F1
- 当前已知最高线上分：`c03_c01_target_b01_proxy_reblend.csv = 0.71059`
- 当前榜首参考：`0.71968`
- 当前主推未提交候选：暂无；`c03` 已提交且仅小幅提升，需要重新找机制。
- 当前备选未提交候选：暂无强备选。
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
upload_ready_breakthrough3\c03_c01_target_b01_proxy_reblend.csv
```

该文件相对 `c01` 改动 `130` 行，属于较保守但代理验证显著更好的新候选。不要同时提交多个旧目录候选；`d02/m01/b01/c01` 已有线上反馈，旧文件继续提交价值较低。

## 文件格式校验

`c03_c01_target_b01_proxy_reblend.csv` 已校验通过：

- 行数：`19440`
- 列名：`id,label`
- ID 顺序：与 `sample_submission.csv` 完全一致
- 空值：`0`
- 重复 ID：`0`
- 标签合法：全部属于 `class_0` 到 `class_11`
- 文件头示例：

- 相对 `c01` 改动：`130` 行
- 标签计数：`class_0=1789, class_1=871, class_10=1556, class_11=1205, class_2=2026, class_3=1931, class_4=1517, class_5=1818, class_6=1753, class_7=1532, class_8=1877, class_9=1565`

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

收到 `c03` 线上分数后：

1. 如果 `c03` 高于 `0.71035`，继续使用 target-like proxy 优化目标增强家族权重，并考虑训练完整树数的子簇增强模型作多样性源。
2. 如果 `c03` 持平或小降，说明代理验证仍有噪声，回到 `c01` 作为教师，不再扩大 `target_b01_pow1p2` 权重。
3. 如果 `c03` 明显下降，停止这条代理重融合路线，优先研究更强的验证体系或自监督深度模型。
4. 所有新线上分数必须同步写入 `experiments.csv` 和本文档。

## 注意事项

- 每轮最多给一个主提交和一个备选，避免浪费每日次数。
- 本地 OOF 高不代表线上高，必须结合线上反馈。
- 不要上传 `.json`。
- 不要从 `submissions` 目录随意挑旧文件上传，优先使用当前明确推荐的 `upload_ready_breakthrough3`。
