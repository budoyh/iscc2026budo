# 网络安全智能分类挑战

## 项目概况

本目录用于 ISCC2026 数据安全赛题「网络安全智能分类挑战」。任务是根据 50 个匿名化网络流量统计特征，将测试集样本预测为 `class_0` 到 `class_11` 共 12 类安全事件。官方评价指标为宏平均 F1。提交文件必须是 UTF-8 CSV，且只能包含两列：`id,label`。

当前工作目录：

```text
C:\budostudy\only_for_codex\iscc\work\data_security\network_security_intelligent_classification
```

本项目不使用全局 Python 环境。普通机器学习脚本统一通过：

```powershell
C:\budostudy\only_for_codex\iscc\scripts\run_py.cmd
```

PyTorch/CUDA 脚本统一通过：

```powershell
C:\budostudy\only_for_codex\iscc\scripts\run_torch.cmd
```

## 数据与提交

- 训练集：`raw\data\train_data.csv`
- 测试集：`raw\data\test_data.csv`
- 样例提交：`raw\data\sample_submission.csv`
- 训练样本数：`53477`
- 测试样本数：`19440`
- 特征数：`50`
- 标签列：`label`
- 提交列：`id,label`
- 外部数据：未使用

数据检查结论：

- 训练集和测试集均无缺失值。
- `id` 在训练集和测试集中均唯一且递增，但只作为提交标识，不进入模型。
- 训练标签顺序近似随机，不存在可利用的连续分段规律。
- 训练/测试存在明显协变量漂移，LightGBM domain classifier AUC 约 `0.91`。
- 普通随机 CV 明显高估线上成绩，因此不能只按 OOF 分数选提交。

## 目录结构

```text
raw\data\                         原始 CSV 和样例提交
src\                              训练、融合、后处理和提交生成脚本
models\                           每个 run_id 的 oof_proba.npy、test_proba.npy、metadata.json
submissions\                      生成过的提交文件镜像
upload_ready_recommended\         早期 r 系列可上传文件
upload_ready_next\                n 系列校准/定向偏置文件
upload_ready_deep\                d 系列深度学习/伪标签文件
upload_ready_main\                m 系列主线融合文件
upload_ready_breakthrough\        b 系列目标域增强文件
upload_ready_breakthrough2\       c01/c02 迭代目标域增强文件
upload_ready_breakthrough3\       c03 代理验证重融合文件
experiments.csv                   实验、线上反馈和提交记录
handoff_log.md                    工作日志和交接说明
```

## 关键脚本

- `src\train.py`：LightGBM/XGBoost 基线、对抗权重、特征变换。
- `src\predict.py`：按 run_id 融合概率并生成基础提交。
- `src\make_variants.py`：生成 soft prior、quota prior 等校准候选。
- `src\train_torch_tabular.py`：PyTorch ResNet/FT-Transformer，支持 DANN 域对抗。
- `src\train_pseudo_lgbm.py`：高置信测试伪标签 LightGBM。
- `src\train_catboost.py`：CatBoost 多样性检查。
- `src\train_knn.py`：kNN/标签传播探索，效果较弱。
- `src\train_mlp.py`：sklearn MLP 基线，效果较弱。
- `src\train_target_aug_lgbm.py`：类别条件目标域矩增强 LightGBM。
- `src\analyze_breakthrough.py`：目标域代理验证、簇结构和提交计数诊断。
- `src\train_local_target_aug_lgbm.py`：类条件全协方差/子簇级目标增强探索。
- `src\train_stable_subset_lgbm.py`：删除高漂移特征的稳定子集探索。
- `src\search_prior_adjust.py`、`src\search_proxy_blend.py`：transductive prior 与 target-like proxy 融合搜索。
- `src\write_blend_submission.py`：严格格式校验并输出最终 CSV。

## 已知线上反馈

用户已反馈的 A 榜分数：

| 文件/方案 | A 榜分数 | 结论 |
|---|---:|---|
| `r00` / `lgbm_adv_weighted_f5_s42` | `0.68887` | 随机 CV 高估，单纯对抗权重不足 |
| `r01_blend_raw_std_xgb_coral.csv` | `0.68154` | CORAL/split-standard 融合方向偏负 |
| `r02_blend_raw_xgb.csv` | `0.69083` | 基础 LGBM+XGB 有效但不够 |
| `r03_blend_raw_xgb_soft_uniform.csv` | `0.69291` | soft uniform prior 是正向信号 |
| `n01_raw_xgb_uniform_a1p5.csv` | `0.69266` | uniform 过强未继续提升 |
| `n13_target_class1_11_high.csv` | `0.69186` | 手工定向 class_1/class_11 偏置无效 |
| `d02_raw_xgb_ft_pseudo_uniform_a1p0.csv` | `0.69616` | 深度模型 + 伪标签 + uniform 校准有效 |
| `m01_proxy_iter2_uniform_a1.csv` | `0.69848` | 迭代伪标签和双 FT 有效 |
| `b01_target_aug_high.csv` | `0.70584` | 目标域增强方向有效 |
| `c01_b01_iter_target_aug_high.csv` | `0.71035` | 当前已知最高，b01 教师迭代有效 |

当前榜首参考分为 `0.71968`。差距主要来自训练/测试漂移和本地 OOF 与线上评测分布不一致。

## 当前推荐提交

最新生成但尚未收到线上反馈的突破候选在：

```text
upload_ready_breakthrough3\
```

建议顺序：

1. `c03_c01_target_b01_proxy_reblend.csv`

`c01_b01_iter_target_aug_high.csv` 当前已知线上 A 榜为 `0.71035`。`c03_c01_target_b01_proxy_reblend.csv` 在 c01 原始融合概率基础上提高 `target_aug_lgbm_b01_pow1p2...` 权重，并按 target-like adversarial proxy 选择；相对 `c01` 改动 `130` 行，本地 OOF 为 `0.9332779796`。

该文件已通过格式校验：

- 行数：`19440`
- 列名：`id,label`
- `id` 顺序：与 `sample_submission.csv` 一致
- 空值：`0`
- 重复 ID：`0`
- 标签范围：`class_0` 到 `class_11`
- 编码/换行：UTF-8，CRLF，无索引列

## 方法演进

### 1. GBDT 基线

已训练：

- `lgbm_base_f5_s42`：OOF `0.9296851169`
- `xgb_base_f5_s42`：OOF `0.9295233752`
- `lgbm_adv_weighted_f5_s42`：OOF `0.9287215352`，A 榜 `0.68887`
- `lgbm_base_split_standard_f5_s42`：OOF `0.9298942313`
- `lgbm_base_coral_f5_s42`：OOF `0.9201414219`

结论：OOF 不低，但和线上分数偏离很大。CORAL 单模弱，强行 quota prior 在代理验证中伤分。

### 2. soft prior 校准

`r03` 使用 LGBM+XGB 概率融合后，对测试概率做 uniform prior soft adjustment。它从 `r02=0.69083` 提升到 `r03=0.69291`，证明测试集类别先验与原始模型输出均值存在偏差。

后续 `n01/n13` 说明：

- 更强 uniform alpha 不一定更好。
- 人工定向增加 `class_1/class_11` 没有效果。

### 3. 深度学习与伪标签

PyTorch CUDA 环境已验证：

```text
torch==2.7.0+cu126
cuda available: True
GPU: NVIDIA GeForce RTX 4060 Laptop GPU
```

已训练：

- `torch_resnet_raw_rank_dann015_cw_f5_s42`：OOF `0.8927968364`
- `torch_ft_raw_dann010_cw_f5_s42`：OOF `0.9011105898`
- `torch_ft_raw_dann030_cw_f5_s43`：OOF `0.9012668211`

深度单模不如 GBDT，但作为融合多样性有效。基于 GBDT+FT 的高置信测试伪标签训练：

- `pseudo_lgbm_ftblend_t098_w035_f5_s42`：OOF `0.9306613353`，伪标签数 `9438`
- `pseudo_lgbm_ftblend_uniform100_t098_w035_f5_s42`：OOF `0.9304804945`，伪标签数 `9260`
- `pseudo_lgbm_iter2_twoft_uniform100_t097_w045_f5_s42`：OOF `0.9308982992`，伪标签数 `10119`

融合后：

- `d02_raw_xgb_ft_pseudo_uniform_a1p0.csv`：A 榜 `0.69616`
- `m01_proxy_iter2_uniform_a1.csv`：A 榜 `0.69848`
- `b01_target_aug_high.csv`：A 榜 `0.70584`

结论：深度模型不是单独突破点，但配合伪标签和校准能稳定涨分。

### 4. 目标域增强

为解决训练/测试漂移，新增类别条件目标域矩增强：

1. 用当前最好融合模型的测试概率估计每个类别在测试集的特征均值和方差。
2. 对训练样本按真实类别做仿射变换，使其更接近目标域。
3. 将增强样本和高置信测试伪标签一起加入 LightGBM 训练。

已训练：

- `target_aug_lgbm_m01_pow2_aug045_t097_w035_f5_s42`：OOF `0.9313765419`，伪标签数 `9740`
- `target_aug_lgbm_m01_pow1p5_aug060_t095_w045_f5_s42`：OOF `0.9316003006`，伪标签数 `11035`
- `target_aug_lgbm_b01_pow1p2_aug080_t093_w055_f5_s42`：OOF `0.9323161136`，伪标签数 `12785`
- `target_aug_lgbm_b01_pow0p8_aug100_t090_w065_f5_s42`：OOF `0.9318755648`，伪标签数 `13670`

融合后：

- `b01_target_aug_high.csv`：OOF `0.9328080314`，A 榜 `0.70584`
- `b02_target_aug_mid.csv`：OOF `0.9326931022`
- `c01_b01_iter_target_aug_high.csv`：OOF `0.9330872501`
- `c02_b01_iter_target_aug_safe.csv`：OOF `0.9330908438`

结论：这是目前最像“突破性方法”的路线，改变的是训练分布而不是只调融合权重。

## 已排除或低优先级路线

- `id` 或行顺序：训练标签顺序随机，测试预测 run 长度也近似随机，没有可利用分段。
- CatBoost：`catboost_raw_bal_d6_lr005_f5_s42` OOF `0.9102305212`，单模弱，小权重融合无代理收益。
- sklearn MLP：OOF `0.8620916269`，只适合作为极小多样性源，当前未使用。
- kNN/标签传播：OOF 约 `0.62-0.65`，过弱。
- 测试 kNN 图平滑：代理验证下降，说明邻域平均会抹掉有效边界。
- 强制 quota prior：代理验证伤分，不推荐。
- 手工类别偏置：`n13` 线上下降，不推荐。

## 复现命令

普通 GBDT 基线：

```powershell
cd C:\budostudy\only_for_codex\iscc\work\data_security\network_security_intelligent_classification

C:\budostudy\only_for_codex\iscc\scripts\run_py.cmd src\train.py --strategy lgbm_base --folds 5 --seed 42
C:\budostudy\only_for_codex\iscc\scripts\run_py.cmd src\train.py --strategy xgb_base --folds 5 --seed 42
C:\budostudy\only_for_codex\iscc\scripts\run_py.cmd src\train.py --strategy lgbm_adv_weighted --folds 5 --seed 42
```

PyTorch CUDA 检查：

```powershell
cd C:\budostudy\only_for_codex\iscc
scripts\check_torch_cuda.cmd
```

FT-Transformer 示例：

```powershell
cd C:\budostudy\only_for_codex\iscc
scripts\run_torch.cmd work\data_security\network_security_intelligent_classification\src\train_torch_tabular.py --model fttransformer --features raw --domain-weight 0.10 --class-weight --seed 42 --run-id torch_ft_raw_dann010_cw_f5_s42
```

伪标签 LGBM 示例：

```powershell
cd C:\budostudy\only_for_codex\iscc\work\data_security\network_security_intelligent_classification

C:\budostudy\only_for_codex\iscc\scripts\run_py.cmd src\train_pseudo_lgbm.py --source-runs lgbm_base_f5_s42 xgb_base_f5_s42 torch_ft_raw_dann010_cw_f5_s42 --source-weights 0.72 0.18 0.10 --soft-prior uniform --soft-alpha 1.0 --threshold 0.98 --pseudo-weight 0.35 --n-estimators 1200 --run-id pseudo_lgbm_ftblend_uniform100_t098_w035_f5_s42
```

目标域增强示例：

```powershell
cd C:\budostudy\only_for_codex\iscc\work\data_security\network_security_intelligent_classification

C:\budostudy\only_for_codex\iscc\scripts\run_py.cmd src\train_target_aug_lgbm.py --source-runs lgbm_base_f5_s42 xgb_base_f5_s42 torch_ft_raw_dann010_cw_f5_s42 torch_ft_raw_dann030_cw_f5_s43 pseudo_lgbm_ftblend_t098_w035_f5_s42 pseudo_lgbm_ftblend_uniform100_t098_w035_f5_s42 pseudo_lgbm_iter2_twoft_uniform100_t097_w045_f5_s42 --source-weights 0.449625 0.113030 0.111684 0.050832 0.019178 0.004121 0.251531 --soft-alpha 1.0 --target-power 2.0 --aug-weight 0.45 --pseudo-threshold 0.97 --pseudo-weight 0.35 --n-estimators 1400 --run-id target_aug_lgbm_m01_pow2_aug045_t097_w035_f5_s42
```

最终提交生成示例：

```powershell
cd C:\budostudy\only_for_codex\iscc\work\data_security\network_security_intelligent_classification

C:\budostudy\only_for_codex\iscc\scripts\run_py.cmd src\write_blend_submission.py --run-ids lgbm_base_f5_s42 xgb_base_f5_s42 torch_ft_raw_dann010_cw_f5_s42 torch_ft_raw_dann030_cw_f5_s43 pseudo_lgbm_ftblend_t098_w035_f5_s42 pseudo_lgbm_ftblend_uniform100_t098_w035_f5_s42 pseudo_lgbm_iter2_twoft_uniform100_t097_w045_f5_s42 target_aug_lgbm_m01_pow2_aug045_t097_w035_f5_s42 target_aug_lgbm_m01_pow1p5_aug060_t095_w045_f5_s42 --weights 0.277615 0.069789 0.068958 0.031386 0.011841 0.002544 0.155305 0.075962 0.306600 --soft-prior uniform --soft-alpha 1.0 --output-dir upload_ready_breakthrough --output-name b01_target_aug_high
```

## 提交注意事项

- 只上传 `upload_ready_*` 目录中的 `.csv` 文件，不要上传 `.json` 报告。
- 每次只优先提交一个主文件，避免浪费每日提交次数。
- 当前优先级：先提交 `upload_ready_breakthrough2\c01_b01_iter_target_aug_high.csv`。
- 提交后立刻把线上分数补到 `experiments.csv` 和 `handoff_log.md`。
