# Complexity frontier — scored

Cells not ok: none

| strategy | penalty | xor held-out | xor conns | spiral held-out | spiral conns | spiral3 held-out | spiral3 conns |
|---|---|--:|--:|--:|--:|--:|--:|
| lam | linear λ=0 | 0.967 ± 0.007 | 12.3 ± 1.6 | 0.903 ± 0.069 | 10.5 ± 3.2 | 0.825 ± 0.055 | 10.2 ± 3.3 |
| lam | linear λ=0.001 | 0.968 ± 0.012 | 7.3 ± 1.8 | 0.910 ± 0.058 | 9.4 ± 2.5 | 0.780 ± 0.060 | 8.3 ± 3.4 |
| lam | linear λ=0.003 | 0.969 ± 0.006 | 7.2 ± 1.3 | 0.861 ± 0.052 | 7.9 ± 2.5 | 0.779 ± 0.044 | 6.8 ± 2.6 |
| lam | linear λ=0.01 | 0.975 ± 0.008 | 6.6 ± 0.7 | 0.855 ± 0.079 | 5.8 ± 1.7 | 0.726 ± 0.064 | 4.2 ± 1.5 |
| lam | linear λ=0.03 | 0.928 ± 0.008 | 5.0 ± 0.0 | 0.685 ± 0.042 | 1.4 ± 0.8 | 0.641 ± 0.022 | 1.2 ± 0.4 |
| lam | mult λ=0.1 | 0.967 ± 0.008 | 9.1 ± 1.8 | 0.831 ± 0.060 | 6.6 ± 2.3 | 0.700 ± 0.053 | 4.3 ± 1.9 |
| lam | mult λ=0.3 | 0.967 ± 0.014 | 8.9 ± 1.7 | 0.759 ± 0.108 | 4.0 ± 3.7 | 0.641 ± 0.022 | 1.2 ± 0.4 |
| lam | mult λ=1 | 0.487 ± 0.000 | 0.0 ± 0.0 | 0.480 ± 0.000 | 0.0 ± 0.0 | 0.433 ± 0.000 | 0.0 ± 0.0 |
| bald | linear λ=0 | 0.964 ± 0.018 | 8.7 ± 1.6 | 0.869 ± 0.057 | 9.1 ± 2.3 | 0.740 ± 0.060 | 8.4 ± 2.4 |
| bald | linear λ=0.001 | 0.973 ± 0.010 | 7.5 ± 1.4 | 0.826 ± 0.074 | 6.7 ± 1.8 | 0.693 ± 0.065 | 5.5 ± 2.4 |
| bald | linear λ=0.003 | 0.967 ± 0.017 | 6.9 ± 1.0 | 0.781 ± 0.078 | 4.9 ± 1.4 | 0.676 ± 0.060 | 4.4 ± 2.5 |
| bald | linear λ=0.01 | 0.855 ± 0.164 | 4.7 ± 1.9 | 0.729 ± 0.051 | 3.4 ± 0.5 | 0.641 ± 0.013 | 1.2 ± 0.4 |
| bald | linear λ=0.03 | 0.585 ± 0.072 | 1.8 ± 0.4 | 0.668 ± 0.013 | 1.0 ± 0.0 | 0.633 ± 0.023 | 1.3 ± 0.5 |
| bald | mult λ=0.1 | 0.663 ± 0.175 | 3.3 ± 3.1 | 0.729 ± 0.062 | 4.4 ± 1.5 | 0.624 ± 0.021 | 1.6 ± 0.5 |
| bald | mult λ=0.3 | 0.567 ± 0.076 | 1.2 ± 1.0 | 0.667 ± 0.011 | 1.3 ± 0.7 | 0.643 ± 0.013 | 1.1 ± 0.3 |
| bald | mult λ=1 | 0.487 ± 0.000 | 0.0 ± 0.0 | 0.480 ± 0.000 | 0.0 ± 0.0 | 0.433 ± 0.000 | 0.0 ± 0.0 |

```
FP1 lam  xor      conns λ0.03 − λ0 -7.3 (spread 1.1)
FP1 lam  spiral   conns λ0.03 − λ0 -9.1 (spread 2.3)
FP1 lam  spiral3  conns λ0.03 − λ0 -9.0 (spread 2.3)
FP1 bald xor      conns λ0.03 − λ0 -6.9 (spread 1.1)
FP1 bald spiral   conns λ0.03 − λ0 -8.1 (spread 1.6)
FP1 bald spiral3  conns λ0.03 − λ0 -7.1 (spread 1.7)
FP1 supported (6/6 cells)
FP2 increases in conns beyond spread along λ: none -> supported
FP3 lam  xor      held-out λ0.01 − λ0 +0.007 (spread 0.008): as predicted
FP3 lam  spiral   held-out λ0.01 − λ0 -0.048 (spread 0.074): not as predicted
FP3 lam  spiral3  held-out λ0.01 − λ0 -0.099 (spread 0.060): as predicted
FP3 bald xor      held-out λ0.01 − λ0 -0.109 (spread 0.117): as predicted
FP3 bald spiral   held-out λ0.01 − λ0 -0.140 (spread 0.054): as predicted
FP3 bald spiral3  held-out λ0.01 − λ0 -0.099 (spread 0.043): as predicted
FP3 partly supported (5/6 cells as predicted)
FP4 penalised arms beating λ=0 beyond spread: none -> supported
FP5 lam  conns ratio mult λ1 / λ0: xor 0.00, spiral3 0.00
FP5 bald conns ratio mult λ1 / λ0: xor 0.00, spiral3 0.00
FP5 contradicted
FP6 Lamarckian >= Baldwinian in 10/10 (spiral, spiral3) x linear λ cells -> supported
λ=0 reproduces the allocation grid's cells: True
```

Figure: `figures/frontier.png`
