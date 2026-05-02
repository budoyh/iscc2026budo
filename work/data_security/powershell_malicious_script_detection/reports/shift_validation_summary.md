# Shift Validation Summary

## Why

随机 5 折 CV 给出 `0.755`，但 `submission_blend_v1.csv` 线上只有 `0.69678`。为贴近测试分布，使用测试集特征组合频率构造 `density-top` 代理验证：把训练集中最像测试集的 30% 和 40% 样本分别作为验证集。

## Proxy Results

候选方案与 `Macro F1`：

- `current_like = 0.3*exact + 0.7*core, weights [1.0,1.15,1.0]`
  - `density-top 30%`: `0.687641`
  - `density-top 40%`: `0.686556`
- `core_w07_115 = core, weights [1.0,0.7,1.15]`
  - `density-top 30%`: `0.711763`
  - `density-top 40%`: `0.693318`
- `blend01_w07_115 = 0.1*exact + 0.9*core, weights [1.0,0.7,1.15]`
  - `density-top 30%`: `0.711957`
  - `density-top 40%`: `0.693540`
- `blend01_w07_120 = 0.1*exact + 0.9*core, weights [1.0,0.7,1.20]`
  - `density-top 30%`: `0.711763`
  - `density-top 40%`: `0.693444`
- `blend01_w07_130 = 0.1*exact + 0.9*core, weights [1.0,0.7,1.30]`
  - `density-top 30%`: `0.711981`
  - `density-top 40%`: `0.692230`

其中 `core = mean(lgbm,xgb,hist,extra_trees)`。

## Decision

- 首选：`submission_blend_shift_v1.csv`
- 备选：`submission_core_shift_v1.csv`
- 更激进的 `2 类` 版本：`submission_blend_shift_v2.csv`、`submission_blend_shift_v3.csv`
