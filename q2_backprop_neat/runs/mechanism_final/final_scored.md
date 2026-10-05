# Mechanism final test (seeds 20–29) — scored

Cells not ok: none

| arm | xor | spiral | spiral3 |
|---|--:|--:|--:|
| lam_default | 0.968 ± 0.008 (n=10) | 0.879 ± 0.047 (n=10) | 0.807 ± 0.043 (n=10) |
| bald_default | 0.956 ± 0.036 (n=10) | 0.813 ± 0.032 (n=10) | 0.655 ± 0.050 (n=10) |
| lam_norep | 0.969 ± 0.009 (n=10) | 0.889 ± 0.059 (n=10) | 0.828 ± 0.060 (n=10) |
| bald_norep | 0.955 ± 0.083 (n=10) | 0.794 ± 0.025 (n=10) | 0.701 ± 0.057 (n=10) |

```
lam_default: median child loss -0.132 (n=16006); per run spiral3_s20 -0.120, spiral3_s21 -0.132, spiral3_s22 -0.052, spiral_s20 -0.184, spiral_s21 -0.182, spiral_s22 -0.141
lam_norep: median child loss -0.107 (n=16672); per run spiral3_s20 -0.095, spiral3_s21 -0.102, spiral3_s22 -0.073, spiral_s20 -0.145, spiral_s21 -0.121, spiral_s22 -0.125
F1 change in median child loss +0.025 (predicted >= +0.03): contradicted
F2 spiral   lam change +0.011, bald change -0.019, interaction +0.030 (spread 0.043): inconclusive
F3 spiral   lam_default - bald_default +0.065 (spread 0.040): supported
F2 spiral3  lam change +0.021, bald change +0.047, interaction -0.026 (spread 0.053): inconclusive
F3 spiral3  lam_default - bald_default +0.153 (spread 0.047): supported
```
