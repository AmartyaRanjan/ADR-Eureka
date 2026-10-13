# LEAP Hand matrix (500 PPO iters, 4096 envs, ADR frozen at eval; metric = rotations / 20 s / hand)

| cond | description | n | nominal (ADR=0) | full DR (ADR=25) | per-seed nominal / fullDR |
|---|---|---|---|---|---|
| L0 | native ADR off | 1 | 10.88 | 0.92 | 10.9/0.9 |
| L1 | native ADR (upstream) | 1 | 7.53 | 0.86 | 7.5/0.9 |
| L2 | ADR warm-started at 8/25 increments (not RAPP) | 0 | — | — |  |
