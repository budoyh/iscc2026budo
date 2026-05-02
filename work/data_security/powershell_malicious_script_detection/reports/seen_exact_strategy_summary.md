# Seen-Exact Strategy Summary

## Rationale

- `18911 / 20000` test rows have an exact 15-feature combination already present in train.
- The previous `submission_groupcv_compromise_v1.csv` online score was `0.69438`, lower than `submission_blend_v1.csv` at `0.69678`.
- The failed shift/order attempts show that broad prior hacking and name-order leakage are unreliable.
- New strategy: for seen exact combinations, use the train combo label posterior directly; for the remaining `1089` unseen rows, use a fitted tree-model fallback.

## Generated Candidates

All files passed format validation: columns `name,label`, 20000 rows, test-order `name`, no missing values, labels only in `{0,1,2}`.

| file | recipe | prediction counts |
| --- | --- | --- |
| `submission_seen_exact_mode_group_v1.csv` | seen exact posterior, no smoothing, no class weights; unseen fallback = group blend | `0=13223, 1=3377, 2=3400` |
| `submission_seen_exact_s0_w120080090_group_v1.csv` | no smoothing, weights `[1.2,0.8,0.9]`; unseen fallback = group blend | `0=13655, 1=3091, 2=3254` |
| `submission_seen_exact_s1_w120080090_group_v1.csv` | smoothing `1`, weights `[1.2,0.8,0.9]`; unseen fallback = group blend | `0=13809, 1=3085, 2=3106` |
| `submission_seen_exact_s5_w115080090_group_v1.csv` | smoothing `5`, weights `[1.15,0.8,0.9]`; unseen fallback = group blend | `0=13973, 1=3066, 2=2961` |
| `submission_seen_exact_s5_w120080095_group_v1.csv` | smoothing `5`, weights `[1.2,0.8,0.95]`; unseen fallback = group blend | `0=13974, 1=3062, 2=2964` |
| `submission_seen_exact_s10_w120080105_group_v1.csv` | smoothing `10`, weights `[1.2,0.8,1.05]`; unseen fallback = group blend | `0=13959, 1=3030, 2=3011` |
| `submission_seen_exact_s5_w120080095_oldcore_v1.csv` | smoothing `5`, weights `[1.2,0.8,0.95]`; unseen fallback = mean old core models | `0=13960, 1=3059, 2=2981` |
| `submission_seen_exact_s10_w120080105_oldcore_v1.csv` | smoothing `10`, weights `[1.2,0.8,1.05]`; unseen fallback = mean old core models | `0=13947, 1=3027, 2=3026` |

## Local Proxy

`reports/seen_exact_proxy_validation.csv` uses 10 stratified shuffle pseudo-tests with 20000 validation rows.

Reproduce command:

```cmd
scripts\run_py.cmd work\data_security\powershell_malicious_script_detection\src\validate_seen_exact_proxy.py
```

Best proxy result:

- `mode_w111`: Macro F1 mean `0.745633`, std `0.002119`.
- Weighted/smoothed variants were lower in this proxy.

This proxy overestimates true online performance because its seen-combo ratio is `0.989155`, higher than the real test ratio `0.94555`; use it for ranking, not as an online-score estimate.

## Submit Priority

1. `submission_seen_exact_mode_group_v1.csv`
2. `submission_seen_exact_s0_w120080090_group_v1.csv`
3. `submission_seen_exact_s1_w120080090_group_v1.csv`

If the first file improves online score, explore small exact-mode threshold edits around the changed rows. If it drops, revert toward `submission_blend_v1.csv` and test `submission_groupcv_blend_v1.csv` before more aggressive smoothing.
