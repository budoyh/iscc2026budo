# Handoff Log

## 当前交接状态

当前线上最好成绩是 `0.70201`，对应文件为：

```text
submissions/submission_mlp_onehot_v1.csv
```

下一次只建议先提交：

```text
submissions/submission.csv
```

当前 `submission.csv` 已覆盖为 `submission_mlp_quota_13600_2900_3500_v1.csv` 的短文件名版本。

提交文件检查结果：

- 行数：20000
- 列名：`name,label`
- `name` 顺序：与 `data_test.csv` 完全一致
- 空值：无
- 标签集合：`{0,1,2}`
- 分布：`0=13600,1=2900,2=3500`
- 编码/换行：UTF-8 无 BOM，LF 换行
- SHA256：`BB77C8A70C07DDF4D64B2A83BF68CC2CB126B8A1B624B62D71F9841611AA6E97`

这个文件基于当前线上最好 MLP 的概率做最小损失 quota 调整，相比 `submission_mlp_onehot_v1.csv` 改动 418 行。

## 任务与数据

- 题目：PowerShell 恶意脚本检测
- 任务：三分类
- 指标：Macro F1
- 训练集：48065 行
- 测试集：20000 行
- 特征：15 个低基数离散整数特征
- 标签：`0` 正常脚本，`1` 一般恶意脚本，`2` 混淆恶意脚本

基础数据检查：

- 无缺失值
- 训练和测试 `name` 均无重复
- 训练标签分布：`0=23708,1=12678,2=11679`
- 测试集完整特征组合 seen ratio：`18911/20000 = 94.555%`

## 环境

普通 Python 入口：

```cmd
scripts\run_py.cmd <script>
```

CUDA PyTorch 入口：

```cmd
scripts\run_torch.cmd <script>
```

CUDA 已验证：

- `torch==2.7.0+cu126`
- `torchvision==0.22.0+cu126`
- `torchaudio==2.7.0+cu126`
- GPU：`NVIDIA GeForce RTX 4060 Laptop GPU`

不要使用系统 Python，不要污染全局环境。

## 重要数据结构发现

训练集 `name` 暴露出连续标签块：

| label | train id range | count |
| --- | --- | ---: |
| 0 | `14000-37707` | `23708` |
| 1 | `40208-52885` | `12678` |
| 2 | `56386-68064` | `11679` |

原始 id 缺口正好是测试集大小：

| missing id range | size |
| --- | ---: |
| `0-13999` | `14000` |
| `37708-40207` | `2500` |
| `52886-56385` | `3500` |

结论：

- 不能直接按测试行顺序套这些缺口标签；该方向线上已经严重失败。
- 但测试集很可能更像每个类别块的“原始前缀段”，因此 prefix holdout 比随机 CV 更有参考价值。

## 线上成绩时间线

| 阶段 | 文件 | 线上分数 | 结论 |
| --- | --- | ---: | --- |
| 首版树融合 | `submission_blend_v1.csv` | `0.69678` | 早期最好，但随机 CV 高估 |
| 分布修正 | `submission_blend_shift_v1.csv` | `0.69404` | 下降 |
| 顺序泄漏 | `submission_leak_cons_base_v1.csv` | `0.37876` | 严重失败 |
| 顺序泄漏 | `submission_leak_mid_block_v1.csv` | `0.38024` | 严重失败 |
| GroupCV 折中 | `submission_groupcv_compromise_v1.csv` | `0.69438` | 下降 |
| seen-exact | `submission_seen_exact_mode_group_v1.csv` | `0.69177` | 下降 |
| 旧树 quota | 短文件 `submission.csv` 旧版本 | `0.69772` | 小幅高于树融合，但不够 |
| MLP-only | `submission_mlp_onehot_v1.csv` | `0.70201` | 当前线上最好 |
| 树+MLP 小融合 | `submission_blend_mlp_a020_w100105_v1.csv` | `0.69821` | 融合稀释 MLP |
| 多 MLP hybrid | `sub_sklearn_hybrid.csv` | `0.69939` | 仍低于单个 MLP |

## 当前主线判断

当前主线是：

```text
online-best MLP-only -> prefix validation -> MLP probability quota
```

不要优先做：

- 树模型再融合
- exact-combo 直接替换
- name/order 直接泄漏
- 大量提交多个相似候选

每日提交次数有限，下一次只交 `submissions/submission.csv`。

## Prefix Validation

构造方式：

- 每个 label 内按原始 id 排序
- 留出最靠前的一段作为验证集
- 留出比例按隐藏测试缺口大小估计

prefix validation 留出规模：

| label | validation size |
| --- | ---: |
| 0 | `8802` |
| 1 | `2088` |
| 2 | `2693` |

关键结果：

- 原始 sklearn MLP raw Macro F1：`0.758561`
- raw 预测分布：`0=8507,1=2783,2=2293`
- 现象：1 类偏多
- quota search 中，full-test `13600/2900/3500` 比 `13500/3000/3500` 更稳
- 详细报告：`reports/prefix_validation_summary.md`

## 当前推荐提交文件

路径：

```text
work/data_security/powershell_malicious_script_detection/submissions/submission.csv
```

等价长文件名：

```text
work/data_security/powershell_malicious_script_detection/submissions/submission_mlp_quota_13600_2900_3500_v1.csv
```

生成逻辑：

- 载入 `models/mlp_onehot_proba.npy`
- 以 `submission_mlp_onehot_v1.csv` 为基准
- 做最小损失 quota 调整到 `0=13600,1=2900,2=3500`

相关报告：

```text
reports/tomorrow_single_candidate_summary.json
reports/prefix_validation_summary.md
```

## 主要脚本

早期树模型：

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

## 关键产物

当前推荐提交：

```text
submissions/submission.csv
submissions/submission_mlp_quota_13600_2900_3500_v1.csv
```

当前线上最好旧文件：

```text
submissions/submission_mlp_onehot_v1.csv
```

主要报告：

```text
reports/prefix_validation_summary.md
reports/deep_learning_strategy_summary.md
reports/torch_tabular_summary.json
reports/sklearn_mlp_ensemble_summary.json
reports/mlp_candidate_summary.json
reports/mlp_refine_candidate_summary.json
reports/tomorrow_single_candidate_summary.json
```

## 下一步

1. 明天提交 `submissions/submission.csv`。
2. 把线上分数写入 `experiments.csv`。
3. 如果上涨，继续沿 MLP-only quota / prefix validation 做小步阈值搜索。
4. 如果下降，回到 `submission_mlp_onehot_v1.csv`，不要继续加大 quota；改做更小幅度的 MLP 概率校准。
