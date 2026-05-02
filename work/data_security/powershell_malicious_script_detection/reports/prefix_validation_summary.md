# Prefix Validation Summary

## Motivation

The public online scores now show that the pure one-hot MLP is the best current model:

- `submission_mlp_onehot_v1.csv`: `0.70201`
- `sub_sklearn_hybrid.csv`: `0.69939`
- Old tree quota `submission.csv`: `0.69772`

The original IDs reveal that the test rows are likely hidden prefixes of each class block. Random CV is therefore not the right model-selection proxy.

## Prefix Holdout

For each label, rows were sorted by original `name` id and the first class-specific prefix was held out:

| class | hidden test size | train size | prefix validation size |
| --- | ---: | ---: | ---: |
| 0 | `14000` | `23708` | `8802` |
| 1 | `2500` | `12678` | `2088` |
| 2 | `3500` | `11679` | `2693` |

The original sklearn MLP on this proxy:

- raw Macro F1: `0.758561`
- raw prediction counts: `0=8507,1=2783,2=2293`

This indicates class 1 is over-predicted on prefix-like rows.

## Threshold / Quota Search

Best simple class-weight search on prefix validation:

- weights around `[1.25,0.75,0.975]`
- proxy Macro F1 around `0.7707`
- full-test counts around `0=13972,1=2933,2=3095`

Quota search on the same prefix validation gave a less aggressive and more stable point:

- full-test target `0=13600,1=2900,2=3500`
- prefix proxy Macro F1 `0.788034`
- keeps the online-best MLP's class 2 count essentially unchanged

## Tomorrow Candidate

The single recommended next submission is:

- `submissions/submission.csv`
- same content as `submission_mlp_quota_13600_2900_3500_v1.csv`
- counts: `0=13600,1=2900,2=3500`
- diff vs online-best MLP: `418`
- SHA256: `BB77C8A70C07DDF4D64B2A83BF68CC2CB126B8A1B624B62D71F9841611AA6E97`

This candidate stays neural-only and modifies only the online-best MLP probabilities by least-loss quota adjustment.
