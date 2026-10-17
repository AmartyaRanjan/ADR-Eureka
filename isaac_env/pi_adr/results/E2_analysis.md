# E2 analysis (pre-registered in project_docs/17_E2_preregistration.md)

## Rotations / 20 s vs frozen ADR increments (mean ± sd over 3 seeds; per-seed in brackets)

| ADR | L0 | L1 | L2 |
|---|---|---|---|
| 0 | 8.42 ± 2.22 [10.9, 6.6, 7.8] | 8.68 ± 2.76 [7.5, 11.8, 6.7] | 8.00 ± 2.11 [5.7, 9.8, 8.5] |
| 5 | 8.21 ± 2.28 [10.7, 6.3, 7.6] | 8.54 ± 2.74 [7.4, 11.7, 6.5] | 7.87 ± 2.10 [5.6, 9.7, 8.4] |
| 10 | 5.60 ± 1.61 [7.4, 4.4, 5.0] | 5.46 ± 1.73 [4.9, 7.4, 4.0] | 6.31 ± 1.99 [4.0, 7.4, 7.6] |
| 15 | 5.50 ± 1.65 [7.4, 4.2, 4.9] | 5.37 ± 1.71 [4.8, 7.3, 4.0] | 6.16 ± 1.96 [3.9, 7.2, 7.4] |
| 20 | 1.45 ± 0.79 [2.3, 0.7, 1.4] | 0.99 ± 0.31 [0.9, 1.3, 0.7] | 3.49 ± 1.88 [1.7, 3.3, 5.5] |
| 25 | 0.56 ± 0.31 [0.9, 0.4, 0.4] | 0.75 ± 0.12 [0.9, 0.8, 0.6] | 1.53 ± 1.20 [0.6, 1.1, 2.9] |

## Primary: normalised AUC (0–25)

- L0: [6.735, 3.811, 4.602]  mean 5.050 ± 1.512
- L1: [4.457, 6.795, 3.792]  mean 5.015 ± 1.577
- L2: [3.667, 6.591, 6.901]  mean 5.720 ± 1.785

Ordering predicted AUC(L2) ≥ AUC(L1) ≥ AUC(L0): L2 5.720, L1 5.015, L0 5.050 → hypothesis NOT consistent
- AUC L1 − L0: -0.035, exact perm p = 0.900 (min attainable 0.10)
- AUC L2 − L1: +0.705, exact perm p = 0.900 (min attainable 0.10)
- AUC L2 − L0: +0.670, exact perm p = 0.800 (min attainable 0.10)

Pairings in the hypothesised direction: 2/3

## Secondary: ADR 10 and 15, per-seed consistency

- ADR 10: L1 [4.95, 7.38, 4.04] vs L0 [7.42, 4.39, 4.98]; complete separation (L1 above L0): False; perm p = 0.700
- ADR 10: L2 [4.02, 7.36, 7.56] vs L1 [4.95, 7.38, 4.04]; complete separation (L2 above L1): False; perm p = 0.800
- ADR 10: L2 [4.02, 7.36, 7.56] vs L0 [7.42, 4.39, 4.98]; complete separation (L2 above L0): False; perm p = 0.900
- ADR 15: L1 [4.79, 7.3, 4.02] vs L0 [7.36, 4.21, 4.92]; complete separation (L1 above L0): False; perm p = 0.700
- ADR 15: L2 [3.9, 7.18, 7.4] vs L1 [4.79, 7.3, 4.02]; complete separation (L2 above L1): False; perm p = 0.900
- ADR 15: L2 [3.9, 7.18, 7.4] vs L0 [7.36, 4.21, 4.92]; complete separation (L2 above L0): False; perm p = 0.900
