# 数据安全赛工作日志

题目：系统日志异常检测挑战。

目标：对 `5000` 条测试日志预测异常检测、主异常区间、主异常类型。最终评分为：

```text
0.15 * F1_detect + 0.50 * IoU_loc + 0.35 * F1_type
```

当前结论：`final.csv` 已提交并取得 `0.96187`，距离榜首 `0.96258` 只差 `0.00071`。下一版推荐提交 [next.csv](/C:/budostudy/only_for_codex/iscc/work/data_security/system_log_anomaly_detection_challenge/submissions/next.csv)，它只比 `final.csv` 少 4 个最低置信异常。

## 线上反馈记录

| 文件/方案 | 线上分数 | 反馈结论 |
|---|---:|---|
| `submission_posterior_rule.csv` | `0.77684` | 规则模板泛化不足 |
| `submission_clause_prefix_rule.csv` | `0.85031` | 子句/前缀有效，但仍受限 |
| `tail_clause_abstain_many.csv` | `0.85941` | 歧义弃权有效但上限低 |
| `submission_torch_sequence_ensemble.csv` | `0.92494` | 深度上下文模型显著优于规则 |
| `submission_torch_boundary_ensemble3.csv` | `0.93531` | start/end 边界头有效 |
| `s1.csv` | `0.94666` | 旧 boundary + hybrid 混合方向被线上验证 |
| `final.csv` | `0.96187` | span ranker 方向被线上验证，当前第二名附近 |
| `next.csv` | 待提交 | 当前下一版推荐，count 从 3214 调到 3210 |

当前榜首参考：`0.96258`，当前差距：`0.00071`。

## 工作阶段记录

### 阶段 1：规则模板与后验统计

时间：2026-05-01 晚到 2026-05-02 凌晨。

完成内容：

- 解析训练集标签与提交格式。
- 建立行级归一化：时间、数字、地址、路径、segment/id。
- 增加 typo 修正。
- 从整行模板扩展到嵌入 `INFO/WARN/ERROR` 子句。
- 加入子句前缀和正常模板尾部异常提示语。
- 尝试歧义样本弃权。

核心文件：

- [common.py](/C:/budostudy/only_for_codex/iscc/work/data_security/system_log_anomaly_detection_challenge/src/common.py)
- [train.py](/C:/budostudy/only_for_codex/iscc/work/data_security/system_log_anomaly_detection_challenge/src/train.py)
- [predict.py](/C:/budostudy/only_for_codex/iscc/work/data_security/system_log_anomaly_detection_challenge/src/predict.py)
- [predict_abstain.py](/C:/budostudy/only_for_codex/iscc/work/data_security/system_log_anomaly_detection_challenge/src/predict_abstain.py)

线上结果：

- `0.73166`
- `0.77684`
- `0.85031`
- `0.85941`

判断：

- 规则能持续提升，但定位主异常起止位置仍不稳定。
- 错误主要来自选择后续症状段，而不是主异常真正开始段。
- 必须转向序列模型。

### 阶段 2：PyTorch sequence BiGRU

时间：2026-05-02 凌晨。

完成内容：

- 配置并验证 PyTorch CUDA 环境。
- 新增 `scripts\run_torch_py.cmd`。
- 建立行序列模型：每行 token embedding 平均，窗口级 2 层 BiGRU。
- 用 3-10 行滑动窗口解码主异常区间。

核心文件：

- [train_torch_sequence.py](/C:/budostudy/only_for_codex/iscc/work/data_security/system_log_anomaly_detection_challenge/src/train_torch_sequence.py)
- [train_torch_sequence_full.py](/C:/budostudy/only_for_codex/iscc/work/data_security/system_log_anomaly_detection_challenge/src/train_torch_sequence_full.py)
- [predict_torch_sequence.py](/C:/budostudy/only_for_codex/iscc/work/data_security/system_log_anomaly_detection_challenge/src/predict_torch_sequence.py)
- [predict_torch_ensemble.py](/C:/budostudy/only_for_codex/iscc/work/data_security/system_log_anomaly_detection_challenge/src/predict_torch_ensemble.py)

结果：

- 90/10 holdout：`0.9739887240`。
- 线上：`0.92494`。

判断：

- 深度学习是正确方向。
- 线上仍有 gap，下一步要显式学习起点和终点。

### 阶段 3：Boundary-aware BiGRU

时间：2026-05-02 上午。

完成内容：

- 在 sequence 模型基础上加入 start/end 边界头。
- 行分类 loss + 起点 loss + 终点 loss 联合训练。
- 全量训练 seed `42/43/44`。
- 三模型平均 line/start/end log probability。

核心文件：

- [train_torch_boundary.py](/C:/budostudy/only_for_codex/iscc/work/data_security/system_log_anomaly_detection_challenge/src/train_torch_boundary.py)
- [train_torch_boundary_full.py](/C:/budostudy/only_for_codex/iscc/work/data_security/system_log_anomaly_detection_challenge/src/train_torch_boundary_full.py)
- [predict_torch_boundary.py](/C:/budostudy/only_for_codex/iscc/work/data_security/system_log_anomaly_detection_challenge/src/predict_torch_boundary.py)
- [predict_torch_boundary_ensemble.py](/C:/budostudy/only_for_codex/iscc/work/data_security/system_log_anomaly_detection_challenge/src/predict_torch_boundary_ensemble.py)

结果：

- holdout：`0.9791059329`。
- 线上：`0.93531`。

判断：

- 显式边界头有效。
- 线上仍低于榜首，原因更像测试集拼写扰动、模板漂移、边界泛化不足。

### 阶段 4：Hybrid char+word boundary 与数据增强

时间：2026-05-02 中午前后。

用户反馈：`0.93531` 仍低于榜首 `0.96258`，要求突破。

完成内容：

- 统计测试 OOV token，发现大量 typo 和变体。
- 新增 fuzzy OOV 脚本，但旧模型只改变 2 行，单独收益有限。
- 新增 hybrid 模型：
  - word token average。
  - line-level char CNN。
  - 2 层 BiGRU。
  - start/end 边界头。
  - 异常窗口增强。

核心文件：

- [predict_torch_boundary_ensemble_fuzzy.py](/C:/budostudy/only_for_codex/iscc/work/data_security/system_log_anomaly_detection_challenge/src/predict_torch_boundary_ensemble_fuzzy.py)
- [train_torch_hybrid_boundary.py](/C:/budostudy/only_for_codex/iscc/work/data_security/system_log_anomaly_detection_challenge/src/train_torch_hybrid_boundary.py)
- [predict_torch_hybrid_ensemble.py](/C:/budostudy/only_for_codex/iscc/work/data_security/system_log_anomaly_detection_challenge/src/predict_torch_hybrid_ensemble.py)

结果：

- hybrid holdout：`0.9807544832`。
- hybrid IoU：`0.9616617719`。

判断：

- hybrid 确实提升 holdout。
- 单独提交 hybrid 变化过大，因此采用与旧 boundary 混合。

### 阶段 5：Mixed ensemble 与短文件名

时间：2026-05-02 中午。

完成内容：

- 新增旧 boundary + hybrid 混合预测。
- 测试多个权重。
- holdout 上 `boundary 40% + hybrid 60%` 更优。
- 加入 type-specific span length prior。
- 生成短文件名，避免上传失败。

核心文件：

- [predict_torch_mixed_boundary_ensemble.py](/C:/budostudy/only_for_codex/iscc/work/data_security/system_log_anomaly_detection_challenge/src/predict_torch_mixed_boundary_ensemble.py)

短文件映射：

- [s1.csv](/C:/budostudy/only_for_codex/iscc/work/data_security/system_log_anomaly_detection_challenge/submissions/s1.csv)：mixed 40/60，线上 `0.94666`。
- [s2.csv](/C:/budostudy/only_for_codex/iscc/work/data_security/system_log_anomaly_detection_challenge/submissions/s2.csv)：mixed 40/60 + length prior 0.6，未提交。
- [s3.csv](/C:/budostudy/only_for_codex/iscc/work/data_security/system_log_anomaly_detection_challenge/submissions/s3.csv)：mixed 40/60 + threshold 2 + length prior 1.0，未提交。
- [s4.csv](/C:/budostudy/only_for_codex/iscc/work/data_security/system_log_anomaly_detection_challenge/submissions/s4.csv)：mixed 50/50，未提交。

线上结果：

- `s1.csv`：`0.94666`。

判断：

- mixed 方向已被线上确认有效。
- 最后一次提交不应盲交 s2/s3，而应进一步优化定位边界。

### 阶段 6：选择式长度先验

时间：2026-05-02 下午。

完成内容：

- 分析 length prior 在 holdout 中的收益和伤害。
- 发现有效模式主要是同类型、边界轻微收缩到更常见长度，例如 `6->5`、`7->6`、`8->7`。
- 生成 [s5.csv](/C:/budostudy/only_for_codex/iscc/work/data_security/system_log_anomaly_detection_challenge/submissions/s5.csv)。

结果：

- `s5.csv` 相对 `s1.csv` 检测不变、类型不变、边界改 440 行。
- 仍不如后续 span ranker 有突破性。

判断：

- 手写选择规则可以提高 holdout，但提升幅度有限。
- 更强方案是训练候选 span ranker。

### 阶段 7：Span ranker 最终突破方案

时间：2026-05-02 下午到傍晚。

完成内容：

- 新增候选 span 排序器。
- 使用六模型 mixed 输出生成候选。
- 对每个候选提取 19 个特征。
- 训练 `HistGradientBoostingRegressor` 和 `ExtraTreesRegressor`。
- 目标为候选 IoU。
- 测试集强制异常数量为 `3214`，保持与已验证高分 `s1` 一致。

核心文件：

- [predict_torch_span_ranker.py](/C:/budostudy/only_for_codex/iscc/work/data_security/system_log_anomaly_detection_challenge/src/predict_torch_span_ranker.py)

验证：

- 类似 fit/valid ranker proxy：`0.9864332900`。
- IoU：`0.9734775506`。
- 相比 `s1`，`final.csv` 检测 flag 不变，类型只改 1 行，边界改 1411 行。
- `final.csv` 的 span 长度分布更接近训练标签分布。

已提交文件：

- [final.csv](/C:/budostudy/only_for_codex/iscc/work/data_security/system_log_anomaly_detection_challenge/submissions/final.csv)
- [s6.csv](/C:/budostudy/only_for_codex/iscc/work/data_security/system_log_anomaly_detection_challenge/submissions/s6.csv)

判断：

- 这是目前已验证最强方案，线上 `0.96187`。
- 它集中优化最高权重的定位 IoU，同时控制检测和类型不漂移。

### 阶段 8：Ranker count 校准

时间：2026-05-02 晚。

用户反馈：`final.csv` 得分 `0.96187`，当前第二，榜首 `0.96258`。

完成内容：

- 将 `final.csv` 线上分数写入实验记录。
- 修改 [predict_torch_span_ranker.py](/C:/budostudy/only_for_codex/iscc/work/data_security/system_log_anomaly_detection_challenge/src/predict_torch_span_ranker.py)，支持一次训练输出多个 `force-count` 版本。
- 生成 count 变体：`3198/3204/3208/3210/3212/3216/3220/3226/3232`。
- 保存每行 ranker 置信度：[ranker_confidence_next.csv](/C:/budostudy/only_for_codex/iscc/work/data_security/system_log_anomaly_detection_challenge/processed/ranker_confidence_next.csv)。
- 选择 [next.csv](/C:/budostudy/only_for_codex/iscc/work/data_security/system_log_anomaly_detection_challenge/submissions/next.csv) 作为下一版。

选择依据：

- holdout 中 span ranker 最优 count 比完整异常数少 4。
- `next.csv` 相对 `final.csv` 也只少 4 个最低置信异常。
- 这 4 行 ranker 置信度约在 `0.55-0.65`，明显低于已保留异常的中位置信度。
- `next.csv` 与 `final.csv` 只差检测 flag，不改已保留异常的类型和边界，风险集中且可解释。

## 当前提交建议

如果现在还有提交机会：

优先提交 [next.csv](/C:/budostudy/only_for_codex/iscc/work/data_security/system_log_anomaly_detection_challenge/submissions/next.csv)。

不要提交：

- `s2.csv`
- `s3.csv`
- `s4.csv`
- `s5.csv`
- 其他 count 距离更大的 ranker 变体，除非 `next.csv` 反馈显示 count 方向正确。

原因：

- `final.csv` 已线上验证为 `0.96187`，但仍低于榜首 `0.00071`。
- `next.csv` 是围绕 `final.csv` 的最小改动，只去掉 4 个最低置信异常。
- 直接换回 `s2/s3/s5` 会改动更多边界，风险比 count 校准更大。

## 复现最终提交

使用已有模型权重重新生成 `final.csv` 和 `next.csv`：

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

## 已验证事项

- `final.csv` 行数：`5000`，线上 `0.96187`。
- `next.csv` 行数：`5000`，格式校验 OK。
- `s1/s5/s6/final` 均使用短文件名，避免上传文件名过长。
- `predict_torch_span_ranker.py` 语法检查通过。
- 关键实验已记录到 [experiments.csv](/C:/budostudy/only_for_codex/iscc/work/data_security/system_log_anomaly_detection_challenge/experiments.csv)。

## 未验证事项

- `next.csv` 尚未线上提交，因此真实 A 榜分数未知。
- 如果去掉的 4 行里有真实异常，`next.csv` 会略降；如果它们是误报或低 IoU 样本，可能超过当前榜首。
- 如果 `next.csv` 线上低于 `final.csv`，应回退 `final.csv` 作为当前可靠最高分。

## 后续如果拿到新线上分数

如果 `next.csv` 高于 `final.csv`：

- 继续围绕 ranker 做 count threshold 微调，例如 `force-count=3208/3212`。
- 考虑保存 ranker confidence，按置信度做更精细的检测阈值。
- 考虑加入更多候选特征，例如类型长度联合先验、窗口内 start/end 曲线形状。

如果 `next.csv` 低于 `final.csv`：

- 保留 `final.csv` 作为当前最优。
- 不再大幅调整 count，优先做边界候选 rerank 的特征增强或 ranker ensemble。
