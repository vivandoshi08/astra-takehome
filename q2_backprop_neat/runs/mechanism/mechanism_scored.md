# Mechanism experiments — scored

Cells not ok: none

| arm | xor | spiral | spiral3 | max species (median per run: spiral / spiral3) |
|---|--:|--:|--:|--:|
| bald_d10 | 0.973 ± 0.009 (n=10) | 0.815 ± 0.035 (n=10) | 0.677 ± 0.061 (n=10) | 3 / 3 |
| bald_d30 | 0.975 ± 0.009 (n=10) | 0.897 ± 0.038 (n=10) | 0.695 ± 0.031 (n=10) | 3 / 4 |
| bald_d100 | 0.964 ± 0.018 (n=10) | 0.869 ± 0.057 (n=10) | 0.740 ± 0.060 (n=10) | 3 / 3 |
| bald_d300 | 0.912 ± 0.114 (n=10) | 0.833 ± 0.067 (n=10) | 0.727 ± 0.061 (n=10) | 3 / 3 |
| lam_d10 | 0.967 ± 0.007 (n=10) | 0.903 ± 0.069 (n=10) | 0.825 ± 0.055 (n=10) | 11 / 8.5 |
| lam_topo | 0.967 ± 0.009 (n=10) | 0.867 ± 0.052 (n=10) | 0.786 ± 0.041 (n=10) | 1 / 1 |
| bald_topo | 0.970 ± 0.019 (n=10) | 0.820 ± 0.055 (n=10) | 0.677 ± 0.046 (n=10) | 1 / 1 |

```
M1 spiral   bald_d300 - bald_d10 +0.018 (spread 0.054): inconclusive
   spiral   reversals beyond spread along depth: none
M2 spiral   gap to lam_d10 by depth: d10 +0.087, d30 +0.006, d100 +0.034, d300 +0.069 -> supported
M1 spiral3  bald_d300 - bald_d10 +0.049 (spread 0.061): inconclusive
   spiral3  reversals beyond spread along depth: none
M2 spiral3  gap to lam_d10 by depth: d10 +0.148, d30 +0.131, d100 +0.085, d300 +0.099 -> partial
M3 spiral   lam_topo max species per run [2, 1, 1, 1, 1, 1, 2, 2, 3, 1]; crashed 0; median 1 -> supported
M4 spiral   lam_topo - bald_topo +0.047 (spread 0.054): inconclusive; ratio to weight-aware gap +0.087: 0.53x
M3 spiral3  lam_topo max species per run [1, 1, 1, 1, 1, 1, 1, 2, 2, 1]; crashed 0; median 1 -> supported
M4 spiral3  lam_topo - bald_topo +0.109 (spread 0.044): supported; ratio to weight-aware gap +0.148: 0.74x
M5 lamarckian  child pre - parent post: median -0.129 over 15979 children (predicted > -0.05): contradicted; median gain gens 20-29 +0.036
M5 baldwinian  child pre - parent post: median -0.179 over 16609 children (predicted < -0.10): supported; median gain gens 20-29 +0.125
consistency with the allocation grid: identical
```
