# Small-network track — scored

Cells not ok: none; champions over 6 connections: none

## All cells (seeds 0–9)

| arm | xor | spiral | spiral3 |
|---|--:|--:|--:|
| lam_std (<= 6) | 0.967 ± 0.019 (6.0 conns, n=10) | 0.826 ± 0.043 (5.5 conns, n=10) | 0.779 ± 0.030 (6.0 conns, n=10) |
| bald_std (<= 6) | 0.956 ± 0.020 (6.0 conns, n=10) | 0.826 ± 0.071 (5.4 conns, n=10) | 0.711 ± 0.033 (5.7 conns, n=10) |
| lam_sine (<= 6) | 0.969 ± 0.004 (5.7 conns, n=10) | 0.906 ± 0.034 (6.0 conns, n=10) | 0.863 ± 0.049 (6.0 conns, n=10) |
| bald_sine (<= 6) | 0.976 ± 0.006 (5.9 conns, n=10) | 0.849 ± 0.048 (5.8 conns, n=10) | 0.816 ± 0.056 (5.9 conns, n=10) |
| A_lamarckian (unconstrained) | 0.967 ± 0.007 (12.3 conns) | 0.903 ± 0.069 (10.5 conns) | 0.825 ± 0.055 (10.2 conns) |
| B_baldwinian (unconstrained) | 0.964 ± 0.018 (8.7 conns) | 0.869 ± 0.057 (9.1 conns) | 0.740 ± 0.060 (8.4 conns) |

```
S1a lam  xor   <=6 - unconstrained +0.000 (spread 0.015): supported
S1b lam  spiral   <=6 - unconstrained -0.077 (spread 0.057): supported
S1b lam  spiral3  <=6 - unconstrained -0.046 (spread 0.045): supported
S1a bald xor   <=6 - unconstrained -0.008 (spread 0.019): supported
S1b bald spiral   <=6 - unconstrained -0.043 (spread 0.064): inconclusive
S1b bald spiral3  <=6 - unconstrained -0.029 (spread 0.048): inconclusive
S3  lam  xor      sine - std +0.001 (spread 0.014): supported
S2  lam  spiral   sine - std +0.080 (spread 0.039): supported
S2  lam  spiral3  sine - std +0.084 (spread 0.041): supported
S3  bald xor      sine - std +0.020 (spread 0.015): contradicted
S2  bald spiral   sine - std +0.023 (spread 0.061): inconclusive
S2  bald spiral3  sine - std +0.105 (spread 0.046): supported
S4  spiral   lam_std - bald_std +0.000 (spread 0.059): does not hold
S4  spiral3  lam_std - bald_std +0.069 (spread 0.032): holds
```

## Excluding the three cells run during development

| arm | xor | spiral | spiral3 |
|---|--:|--:|--:|
| lam_std (<= 6) | 0.967 ± 0.019 (6.0 conns, n=10) | 0.830 ± 0.044 (5.6 conns, n=9) | 0.782 ± 0.031 (6.0 conns, n=9) |
| bald_std (<= 6) | 0.956 ± 0.020 (6.0 conns, n=10) | 0.826 ± 0.071 (5.4 conns, n=10) | 0.711 ± 0.033 (5.7 conns, n=10) |
| lam_sine (<= 6) | 0.969 ± 0.004 (5.7 conns, n=10) | 0.906 ± 0.034 (6.0 conns, n=10) | 0.863 ± 0.049 (6.0 conns, n=10) |
| bald_sine (<= 6) | 0.976 ± 0.006 (5.9 conns, n=10) | 0.849 ± 0.051 (5.8 conns, n=9) | 0.816 ± 0.056 (5.9 conns, n=10) |
| A_lamarckian (unconstrained) | 0.967 ± 0.007 (12.3 conns) | 0.903 ± 0.069 (10.5 conns) | 0.825 ± 0.055 (10.2 conns) |
| B_baldwinian (unconstrained) | 0.964 ± 0.018 (8.7 conns) | 0.869 ± 0.057 (9.1 conns) | 0.740 ± 0.060 (8.4 conns) |

```
S1a lam  xor   <=6 - unconstrained +0.000 (spread 0.015): supported
S1b lam  spiral   <=6 - unconstrained -0.073 (spread 0.058): supported
S1b lam  spiral3  <=6 - unconstrained -0.043 (spread 0.045): inconclusive
S1a bald xor   <=6 - unconstrained -0.008 (spread 0.019): supported
S1b bald spiral   <=6 - unconstrained -0.043 (spread 0.064): inconclusive
S1b bald spiral3  <=6 - unconstrained -0.029 (spread 0.048): inconclusive
S3  lam  xor      sine - std +0.001 (spread 0.014): supported
S2  lam  spiral   sine - std +0.076 (spread 0.039): supported
S2  lam  spiral3  sine - std +0.081 (spread 0.041): supported
S3  bald xor      sine - std +0.020 (spread 0.015): contradicted
S2  bald spiral   sine - std +0.023 (spread 0.062): inconclusive
S2  bald spiral3  sine - std +0.105 (spread 0.046): supported
S4  spiral   lam_std - bald_std +0.004 (spread 0.059): holds
S4  spiral3  lam_std - bald_std +0.072 (spread 0.032): holds
```

