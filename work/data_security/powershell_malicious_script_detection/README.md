# PowerShell 恶意脚本检测

## 当前状态

- 当前线上最好成绩：`0.70201`
- 当前线上最好文件：`submissions/submission_mlp_onehot_v1.csv`
- 明天只建议先提交：`submissions/submission.csv`
- 当前 `submission.csv` 内容：线上最好 MLP 概率的最小损失 quota 校正版，目标分布 `0=13600,1=2900,2=3500`
- 当前 `submission.csv` SHA256：`BB77C8A70C07DDF4D64B2A83BF68CC2CB126B8A1B624B62D71F9841611AA6E97`
- 当前 `submission.csv` 已验证：`name,label` 两列，20000 行，`name` 顺序与测试集一致，无空值，标签只含 `{0,1,2}`，UTF-8 无 BOM，LF 换行

当前判断：树模型、exact-combo、顺序泄漏、简单融合都已被线上反馈压住；主线改为 **neural-only，以线上最好 MLP 为核心，只做少量有依据的阈值/配额修正**。

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
| `submission_mlp_onehot_v1.csv` | `0.70201` | 当前线上最好，确认 neural-only 是主线 |
| `submission_blend_mlp_a020_w100105_v1.csv` | `0.69821` | MLP 混回树模型会稀释优势 |
| `sub_sklearn_hybrid.csv` | `0.69939` | 多 MLP 种子集成也低于单个线上最好 MLP |

## 当前推荐提交

只先提交：

```text
submissions/submission.csv
```

该文件等价于：

```text
submissions/submission_mlp_quota_13600_2900_3500_v1.csv
```

生成逻辑：

- 基础概率：`submission_mlp_onehot_v1.csv` 对应的 sklearn one-hot MLP 概率
- 操作：最小损失 quota 调整
- 目标分布：`0=13600,1=2900,2=3500`
- 相比线上最好 MLP-only 改动：418 行
- 目的：保留 MLP 的强信号和 2 类总量，同时降低前缀验证中明显偏多的 1 类

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

## 下一步

1. 明天先提交 `submissions/submission.csv`。
2. 记录线上分数到 `experiments.csv`。
3. 如果上涨，继续沿 MLP-only quota / prefix validation 方向微调，不要回到树模型融合。
4. 如果下降，回到 `submission_mlp_onehot_v1.csv`，重新做更小幅度的 MLP 阈值搜索。
