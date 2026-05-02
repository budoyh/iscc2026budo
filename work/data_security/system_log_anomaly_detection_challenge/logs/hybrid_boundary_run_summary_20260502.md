# 2026-05-02 Hybrid Boundary Run Summary

## Why This Run

线上 `submission_torch_boundary_ensemble3.csv` 得分为 `0.93531`，明显低于当前榜首 `0.96258`。旧方案的随机 holdout 已到 `0.9791059329`，但线上仍有较大 gap，说明主要问题不是训练集内拟合，而是测试集模板漂移、拼写扰动和边界泛化。

## Changes

- 新增 `src/predict_torch_boundary_ensemble_fuzzy.py`：对旧 boundary ensemble 增加 OOV 近邻拼写纠错。该版本只改变 2 行测试预测，单独收益预期有限。
- 新增 `src/train_torch_hybrid_boundary.py`：使用 word token average + line-level char CNN + 2-layer BiGRU + start/end boundary heads。
- 新增异常窗口增强：同义词替换、常见拼写扰动、轻量 char noise、`<mask>` token dropout。
- 新增 `src/predict_torch_hybrid_ensemble.py`：三 seed hybrid ensemble 预测。
- 新增 `src/predict_torch_mixed_boundary_ensemble.py`：旧 boundary 模型族与 hybrid 模型族加权混合。
- `src/predict_torch_mixed_boundary_ensemble.py` 后续加入 type-specific span length prior，用训练集每类异常长度分布约束 decoder，直接优化 IoU。

## Validation

- Hybrid holdout best: `0.9807544832`
- Hybrid holdout IoU: `0.9616617719`
- Mixed holdout proxy best: `0.9835013191`
- Mixed holdout proxy IoU: `0.9671554437`
- Mixed + length prior holdout proxy best: `0.9839092054`
- Mixed + length prior holdout proxy IoU: `0.9679712163`
- 旧 boundary holdout 对照: `0.9791059329`, IoU `0.9582118659`

## Generated Submissions

- `submissions/submission_torch_mixed_boundary_40_60_bw1p5.csv`
  - 稳健推荐优先提交。
  - 旧 boundary 40%, hybrid 60%, `threshold=-4.0`, `decode_boundary_weight=1.5`。
  - 测试异常数 `3214/5000`。
  - 相对线上 `0.93531` 基线改动 `603` 行，检测 flag 只改 `1` 行。

- `submissions/submission_torch_mixed_boundary_40_60_bw1p5_thr2.csv`
  - 保守备选。
  - 测试异常数 `3212/5000`。
  - 与非 `thr2` 版本仅差 `2` 个检测行。

- `submissions/submission_torch_mixed_boundary_40_60_bw1p5_lenprior0p6.csv`
  - 带长度先验的中风险版本。
  - 测试异常数 `3213/5000`。
  - 相对基线改动 `858` 行。

- `submissions/submission_torch_mixed_boundary_40_60_bw1p5_lenprior1.csv`
  - 带长度先验的高风险版本。
  - 测试异常数 `3213/5000`。
  - 相对基线改动 `990` 行。

- `submissions/submission_torch_mixed_boundary_40_60_bw1p5_thr2_lenprior1.csv`
  - holdout proxy 最高版本。
  - 测试异常数 `3211/5000`。
  - 因长度先验强，改动大，适合作为第二或第三次试探提交。

- `submissions/submission_torch_mixed_boundary_50_50_bw1p5.csv`
  - 风险较低的混合备选。
  - 相对基线改动 `524` 行。

- `submissions/submission_torch_hybrid_boundary_ensemble3.csv`
  - 纯 hybrid 三模型版本。
  - 相对基线改动 `793` 行，风险更高。

## Recommendation

优先提交 `submission_torch_mixed_boundary_40_60_bw1p5.csv`，这是稳健混合版本。如果它提升，继续提交 `submission_torch_mixed_boundary_40_60_bw1p5_lenprior0p6.csv`；如果仍提升，再试 holdout proxy 最高的 `submission_torch_mixed_boundary_40_60_bw1p5_thr2_lenprior1.csv`。如果第一版下降，回退到 `submission_torch_boundary_ensemble3.csv` 或只试旧模型的 `submission_torch_boundary_ensemble3_bw1p5_lenprior0p4.csv`。
