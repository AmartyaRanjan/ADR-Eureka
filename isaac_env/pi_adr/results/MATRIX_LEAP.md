# LEAP Hand matrix (500 PPO iters, 4096 envs, ADR frozen at eval; metric = rotations / 20 s / hand)

| cond | description | n | nominal (ADR=0) | full DR (ADR=25) | per-seed nominal / fullDR |
|---|---|---|---|---|---|
| L0 | native ADR off | 3 | 8.42 ± 2.22 | 0.56 ± 0.31 | 10.9/0.9, 6.6/0.4, 7.8/0.4 |
| L1 | native ADR (upstream) | 3 | 8.68 ± 2.76 | 0.75 ± 0.12 | 7.5/0.9, 11.8/0.8, 6.7/0.6 |
| L2 | ADR warm-started at 8/25 increments (not RAPP) | 3 | 8.00 ± 2.11 | 1.53 ± 1.20 | 5.7/0.6, 9.8/1.1, 8.5/2.9 |
