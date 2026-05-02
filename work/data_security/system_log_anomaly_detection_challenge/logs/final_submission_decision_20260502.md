# Final Submission Decision 2026-05-02

## Public Anchor

- `s1.csv` public score: `0.94666`
- Previous best before `s1`: `0.93531`
- Current public top: `0.96258`

`s1` confirmed that the mixed boundary + hybrid family is better than the old boundary ensemble.

## Breakthrough Attempt

Added `src/predict_torch_span_ranker.py`.

Method:

- Use six full-data models: 3 word-boundary + 3 hybrid char/word-boundary.
- Generate candidate spans for each row and anomaly type.
- Train two regressors on full train candidates:
  - `HistGradientBoostingRegressor`
  - `ExtraTreesRegressor`
- Target is candidate IoU when type matches, otherwise `0`.
- Force test anomaly count to `3214`, the same as `s1`, to avoid detection-ratio drift.

Holdout proxy from the analogous fit/valid ranker:

- Best fixed decoder family before ranker: around `0.98315` to `0.98393`
- Span ranker: `0.9864332900`
- Ranker IoU: `0.9734775506`

## Final File

- `submissions/final.csv`
- Same content as `submissions/s6.csv`
- Validation: OK
- Rows: `5000`
- Predicted anomaly count: `3214`

Compared with submitted `s1.csv`:

- Detection flag changes: `0`
- Type changes: `1`
- Boundary changes: `1411`

Reasoning:

- The ranker changes mainly the localization target, which has the highest weight (`0.50`).
- It keeps detection count unchanged and almost all types unchanged.
- Predicted span length distribution is much closer to the training label distribution than `s1`, reducing the overlong-span problem.

Submit `final.csv` for the last available submission.
