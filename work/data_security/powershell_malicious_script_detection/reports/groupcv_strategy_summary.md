# GroupCV Strategy Summary

## Why The Previous Validation Failed

- 随机 5 折会把同一个精确特征组合同时放进训练和验证。
- 本题只有 1593 个唯一特征组合，但训练集有 48065 行，重复极多。
- 这会让模型“记住组合”而不是学习组合外泛化，导致本地分数显著高估。

## GroupKFold Results

按“精确特征组合”分组做 5 折验证：

- `rf`: `0.725864`
- `lgbm`: `0.725312`
- `xgb`: `0.723388`
- `hist`: `0.722846`
- `et`: `0.714733`
- `catboost`: `0.716208`

最佳融合：

- `0.4*rf + 0.1*lgbm + 0.5*xgb`
- 类别权重：`[1.0, 0.95, 1.05]`
- GroupKFold Macro F1：`0.7385682761`

兼顾 `density-top` 代理验证的折中融合：

- `0.7*(0.4*rf + 0.1*lgbm + 0.5*xgb) + 0.3*mean(lgbm,xgb,hist,et)`
- 类别权重：`[1.0, 0.95, 1.125]`
- GroupKFold Macro F1：`0.7371982002`
- `density-top 30%`: `0.7134493986`
- `density-top 40%`: `0.6939290519`

## Recommendation Order

1. `submission_groupcv_compromise_v1.csv`
2. `submission_groupcv_blend_v1.csv`
3. `submission_groupcv_equal_v1.csv`
