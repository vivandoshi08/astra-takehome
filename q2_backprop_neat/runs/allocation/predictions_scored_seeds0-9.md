# Allocation predictions — scored (seeds 0-9)

From `runs/allocation/results.csv`; held-out accuracy, mean ± sd over seeds 0-9.

| dataset | strategy | A | B | C | D | E |
|---|---|--:|--:|--:|--:|--:|
| xor | lamarckian | 0.967 ± 0.007 (n=10) | 0.967 ± 0.016 (n=10) | 0.963 ± 0.017 (n=10) | 0.966 ± 0.013 (n=9) | 0.968 ± 0.004 (n=10) |
| xor | baldwinian | 0.973 ± 0.009 (n=10) | 0.964 ± 0.018 (n=10) | 0.928 ± 0.082 (n=10) | 0.976 ± 0.006 (n=10) | 0.966 ± 0.022 (n=10) |
| spiral | lamarckian | 0.903 ± 0.069 (n=10) | 0.877 ± 0.068 (n=9) | 0.808 ± 0.038 (n=10) | 0.903 ± 0.050 (n=7) | 0.877 ± 0.061 (n=10) |
| spiral | baldwinian | 0.815 ± 0.035 (n=10) | 0.869 ± 0.057 (n=10) | 0.845 ± 0.024 (n=10) | 0.879 ± 0.059 (n=10) | 0.851 ± 0.037 (n=10) |
| spiral3 | lamarckian | 0.825 ± 0.055 (n=10) | 0.809 ± 0.061 (n=10) | 0.736 ± 0.024 (n=10) | 0.833 ± 0.040 (n=8) | 0.793 ± 0.045 (n=10) |
| spiral3 | baldwinian | 0.677 ± 0.061 (n=10) | 0.740 ± 0.060 (n=10) | 0.695 ± 0.033 (n=10) | 0.683 ± 0.027 (n=10) | 0.673 ± 0.064 (n=10) |

```
P1  spiral   baldwinian  B-A +0.053 (spread 0.048): contradicted
P1b spiral   (B-A) lamarckian -0.026 vs baldwinian +0.053: gap -0.079 (spread 0.069): contradicted
P2  spiral   lamarckian  C-A -0.095 (spread 0.056): supported
P2  spiral   baldwinian  C-A +0.029 (spread 0.030): inconclusive
P3  spiral   baldwinian  D-A +0.064 (spread 0.049): contradicted
P1  spiral3  baldwinian  B-A +0.063 (spread 0.060): contradicted
P1b spiral3  (B-A) lamarckian -0.017 vs baldwinian +0.063: gap -0.079 (spread 0.060): contradicted
P2  spiral3  lamarckian  C-A -0.089 (spread 0.043): supported
P2  spiral3  baldwinian  C-A +0.018 (spread 0.049): inconclusive
P3  spiral3  baldwinian  D-A +0.005 (spread 0.047): inconclusive
xor lamarckian  B-A +0.000 (spread 0.013): within spread
xor lamarckian  C-A -0.005 (spread 0.013): within spread
xor lamarckian  D-A -0.001 (spread 0.010): within spread
xor lamarckian  E-A +0.001 (spread 0.006): within spread
xor baldwinian  B-A -0.009 (spread 0.014): within spread
xor baldwinian  C-A -0.045 (spread 0.058): within spread
xor baldwinian  D-A +0.003 (spread 0.008): within spread
xor baldwinian  E-A -0.007 (spread 0.017): within spread
E   spiral   lamarckian  E-A -0.025 (spread 0.065) [no prediction]
E   spiral3  lamarckian  E-A -0.033 (spread 0.050) [no prediction]
E   spiral   baldwinian  E-A +0.036 (spread 0.036) [no prediction]
E   spiral3  baldwinian  E-A -0.004 (spread 0.062) [no prediction]
```
