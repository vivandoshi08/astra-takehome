# Allocation predictions — scored (seeds 0-2)

From `runs/allocation/results.csv`; held-out accuracy, mean ± sd over seeds 0-2.

| dataset | strategy | A | B | C | D | E |
|---|---|--:|--:|--:|--:|--:|
| xor | lamarckian | 0.971 ± 0.010 (n=3) | 0.951 ± 0.021 (n=3) | 0.960 ± 0.012 (n=3) | 0.957 ± 0.014 (n=2) | 0.967 ± 0.007 (n=3) |
| xor | baldwinian | 0.969 ± 0.010 (n=3) | 0.962 ± 0.025 (n=3) | 0.876 ± 0.148 (n=3) | 0.976 ± 0.010 (n=3) | 0.978 ± 0.015 (n=3) |
| spiral | lamarckian | 0.856 ± 0.096 (n=3) | 0.840 ± 0.041 (n=3) | 0.804 ± 0.020 (n=3) | 0.900 ± 0.019 (n=2) | 0.936 ± 0.004 (n=3) |
| spiral | baldwinian | 0.791 ± 0.017 (n=3) | 0.818 ± 0.023 (n=3) | 0.860 ± 0.024 (n=3) | 0.927 ± 0.037 (n=3) | 0.838 ± 0.032 (n=3) |
| spiral3 | lamarckian | 0.793 ± 0.029 (n=3) | 0.800 ± 0.107 (n=3) | 0.749 ± 0.034 (n=3) | 0.867 ± nan (n=1) | 0.776 ± 0.020 (n=3) |
| spiral3 | baldwinian | 0.733 ± 0.020 (n=3) | 0.749 ± 0.091 (n=3) | 0.664 ± 0.032 (n=3) | 0.680 ± 0.029 (n=3) | 0.664 ± 0.052 (n=3) |

```
P1  spiral   baldwinian  B-A +0.027 (spread 0.020): contradicted
P1b spiral   (B-A) lamarckian -0.016 vs baldwinian +0.027: gap -0.042 (spread 0.074): inconclusive
P2  spiral   lamarckian  C-A -0.051 (spread 0.070): inconclusive
P2  spiral   baldwinian  C-A +0.069 (spread 0.021): contradicted
P3  spiral   baldwinian  D-A +0.136 (spread 0.029): contradicted
P1  spiral3  baldwinian  B-A +0.016 (spread 0.066): inconclusive
P1b spiral3  (B-A) lamarckian +0.007 vs baldwinian +0.016: gap -0.009 (spread 0.079): inconclusive
P2  spiral3  lamarckian  C-A -0.044 (spread 0.032): supported
P2  spiral3  baldwinian  C-A -0.069 (spread 0.026): supported
P3  spiral3  baldwinian  D-A -0.053 (spread 0.025): supported
xor lamarckian  B-A -0.020 (spread 0.017): beyond spread
xor lamarckian  C-A -0.011 (spread 0.011): beyond spread
xor lamarckian  D-A -0.014 (spread 0.012): beyond spread
xor lamarckian  E-A -0.004 (spread 0.009): within spread
xor baldwinian  B-A -0.007 (spread 0.019): within spread
xor baldwinian  C-A -0.093 (spread 0.105): within spread
xor baldwinian  D-A +0.007 (spread 0.010): within spread
xor baldwinian  E-A +0.009 (spread 0.013): within spread
E   spiral   lamarckian  E-A +0.080 (spread 0.068) [no prediction]
E   spiral3  lamarckian  E-A -0.018 (spread 0.025) [no prediction]
E   spiral   baldwinian  E-A +0.047 (spread 0.025) [no prediction]
E   spiral3  baldwinian  E-A -0.069 (spread 0.039) [no prediction]
```
