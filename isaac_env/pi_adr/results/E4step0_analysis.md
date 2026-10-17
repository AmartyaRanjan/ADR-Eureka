# E4 step-0 diagnostic (pre-registered, project_docs/18_E4_step0_preregistration.md)

Consistency check (rotations reproduce earlier values within 1 %): PASSED


## rotations / 20 s (mean ± sd over seeds; per-seed in brackets)

| ADR | L0 | L1 | L2 |
|---|---|---|---|
| 0 | 8.42 ± 2.22 [10.88, 6.57, 7.80] | 8.68 ± 2.76 [7.53, 11.83, 6.68] | 8.00 ± 2.11 [5.68, 9.80, 8.52] |
| 5 | 8.21 ± 2.28 [10.73, 6.29, 7.61] | 8.54 ± 2.74 [7.41, 11.66, 6.55] | 7.87 ± 2.10 [5.56, 9.67, 8.39] |
| 10 | 5.60 ± 1.61 [7.42, 4.39, 4.98] | 5.46 ± 1.73 [4.95, 7.38, 4.04] | 6.31 ± 1.99 [4.02, 7.36, 7.56] |
| 15 | 5.50 ± 1.65 [7.36, 4.21, 4.92] | 5.37 ± 1.71 [4.79, 7.30, 4.02] | 6.16 ± 1.96 [3.90, 7.18, 7.40] |
| 20 | 1.45 ± 0.79 [2.26, 0.69, 1.41] | 0.99 ± 0.31 [0.94, 1.32, 0.70] | 3.49 ± 1.88 [1.71, 3.29, 5.46] |
| 25 | 0.56 ± 0.31 [0.92, 0.37, 0.38] | 0.75 ± 0.12 [0.86, 0.79, 0.62] | 1.53 ± 1.20 [0.60, 1.11, 2.89] |

## falls / 20 s (mean ± sd over seeds; per-seed in brackets)

| ADR | L0 | L1 | L2 |
|---|---|---|---|
| 0 | 0.00 ± 0.00 [0.00, 0.00, 0.00] | 0.00 ± 0.00 [0.00, 0.00, 0.00] | 0.00 ± 0.01 [0.01, 0.00, 0.00] |
| 5 | 0.01 ± 0.01 [0.00, 0.00, 0.02] | 0.00 ± 0.00 [0.01, 0.00, 0.01] | 0.00 ± 0.00 [0.01, 0.00, 0.00] |
| 10 | 0.28 ± 0.16 [0.26, 0.45, 0.12] | 0.31 ± 0.07 [0.38, 0.27, 0.26] | 0.02 ± 0.01 [0.02, 0.02, 0.00] |
| 15 | 0.26 ± 0.16 [0.24, 0.43, 0.12] | 0.31 ± 0.07 [0.38, 0.26, 0.27] | 0.03 ± 0.01 [0.04, 0.03, 0.01] |
| 20 | 1.27 ± 0.32 [1.44, 1.46, 0.90] | 1.78 ± 0.40 [2.25, 1.56, 1.54] | 0.23 ± 0.21 [0.47, 0.16, 0.06] |
| 25 | 3.23 ± 1.44 [4.87, 2.16, 2.65] | 6.87 ± 1.82 [8.57, 7.09, 4.94] | 1.07 ± 0.95 [2.13, 0.76, 0.30] |

## Pre-declared criterion: quality degrades while the quantity gate has not fired

- L0: per-seed ADR levels meeting the gap rule (rot ≥ 80 % of ADR-0, falls ≥ 2× and +0.5): [[], [], []] → seeds with a gap: 0/3 → condition gap: False
- L1: per-seed ADR levels meeting the gap rule (rot ≥ 80 % of ADR-0, falls ≥ 2× and +0.5): [[], [], []] → seeds with a gap: 0/3 → condition gap: False
- L2: per-seed ADR levels meeting the gap rule (rot ≥ 80 % of ADR-0, falls ≥ 2× and +0.5): [[], [], []] → seeds with a gap: 0/3 → condition gap: False

**Conditions with a quality-before-quantity gap: 0/3 → diagnostic DOES NOT SUPPORT building E4.**
