# E5 analysis (pre-registered, project_docs/19_E5_sample_efficiency_preregistration.md)

A = human (vanilla) reward, B = Eureka reward; seeds 42,123,7,1,2; nominal physics


**mean err (mm) vs training iteration (mean ± sd, n = 5)**

| iteration | A human | B Eureka |
|---|---|---|
| 100 | 39.81 ± 11.42 | 57.16 ± 28.33 |
| 200 | 43.36 ± 14.61 | 38.70 ± 7.70 |
| 300 | 37.63 ± 9.86 | 51.60 ± 21.69 |
| 400 | 42.74 ± 10.06 | 42.37 ± 10.88 |
| 600 | 41.26 ± 15.02 | 34.94 ± 7.65 |
| 800 | 37.75 ± 3.95 | 37.69 ± 9.39 |

**worst-10% (mm) vs training iteration (mean ± sd, n = 5)**

| iteration | A human | B Eureka |
|---|---|---|
| 100 | 163.83 ± 118.18 | 326.15 ± 274.34 |
| 200 | 199.29 ± 137.61 | 149.20 ± 79.42 |
| 300 | 159.09 ± 80.18 | 283.06 ± 207.72 |
| 400 | 211.44 ± 90.25 | 191.51 ± 113.79 |
| 600 | 184.65 ± 149.34 | 111.22 ± 61.00 |
| 800 | 149.38 ± 31.63 | 138.71 ± 74.10 |

**2cm % vs training iteration (mean ± sd, n = 5)**

| iteration | A human | B Eureka |
|---|---|---|
| 100 | 21.35 ± 4.60 | 18.37 ± 5.04 |
| 200 | 24.21 ± 4.58 | 24.32 ± 10.02 |
| 300 | 32.86 ± 7.72 | 27.25 ± 11.38 |
| 400 | 35.36 ± 12.88 | 28.07 ± 9.68 |
| 600 | 27.92 ± 4.15 | 25.68 ± 5.44 |
| 800 | 28.99 ± 4.20 | 26.87 ± 6.26 |

**5cm % vs training iteration (mean ± sd, n = 5)**

| iteration | A human | B Eureka |
|---|---|---|
| 100 | 98.33 ± 0.98 | 96.53 ± 3.16 |
| 200 | 97.92 ± 1.44 | 98.59 ± 0.65 |
| 300 | 97.90 ± 1.21 | 97.40 ± 2.34 |
| 400 | 97.76 ± 0.92 | 98.40 ± 1.16 |
| 600 | 97.82 ± 1.60 | 98.98 ± 0.82 |
| 800 | 98.48 ± 0.44 | 98.98 ± 0.76 |

## Primary: learning-curve area of mean error (lower = better; hypothesis: B < A)

- A: [47.6, 32.8, 44.9, 40.8, 37.7] mean 40.75 ± 5.87
- B: [44.1, 43.8, 37.7, 46.6, 35.0] mean 41.43 ± 4.87
- diff (B − A) = +0.68 mm; exact permutation p = 0.8413; direction as hypothesised (B smaller): False

**PRIMARY CRITERION (B smaller AND p < 0.05): False**

## Secondary

- S1 mean error at 100 iterations: A 39.8 ± 11.4 vs B 57.2 ± 28.3; p = 0.2857
- S1 mean error at 200 iterations: A 43.4 ± 14.6 vs B 38.7 ± 7.7; p = 0.5556
- S2 first checkpoint with mean error ≤ 45 mm: A [100, 100, 100, 100, 200] vs B [100, 100, 200, 200, 200] (None = never reached by 800)
- S2 first checkpoint with 2 cm success ≥ 20 %: A [100, 100, 100, 200, 100] vs B [200, 100, 400, 100, 200] (None = never reached by 800)
- S3/S4 curve area of worst-10% (mm): A 182.30 ± 55.97 vs B 177.68 ± 43.80; p = 0.8968
- S3/S4 curve area of 2cm %: A 29.38 ± 4.13 vs B 25.87 ± 5.20; p = 0.2778
