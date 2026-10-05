# Mechanism round 2 — scored

```
Test A (Lamarckian, default config, runs/mechanism_logs):
  clones  median loss -0.125 (n=2999)
  crosses median loss -0.129 (n=12980)
  clones with unchanged node/connection counts -0.089 (n=852)
  d = clones - crosses = +0.004 -> H3 supported (clones lose about as much)
  per-run median loss (all children): spiral3_s0 -0.129, spiral3_s1 -0.054, spiral3_s2 -0.118, spiral_s0 -0.141, spiral_s1 -0.157, spiral_s2 -0.184
  (Baldwinian reference: clones -0.182, crosses -0.179)

Test B (low parameter mutation):
| arm | xor | spiral | spiral3 |
|---|--:|--:|--:|
| lam_lowmut | 0.967 ± 0.007 (n=10) | 0.908 ± 0.077 (n=10) | 0.819 ± 0.038 (n=10) |
| bald_lowmut | 0.920 ± 0.109 (n=10) | 0.821 ± 0.048 (n=10) | 0.645 ± 0.050 (n=10) |
| lam_d10 | 0.967 ± 0.007 (n=10) | 0.903 ± 0.069 (n=10) | 0.825 ± 0.055 (n=10) |
| bald_d10 | 0.973 ± 0.009 (n=10) | 0.815 ± 0.035 (n=10) | 0.677 ± 0.061 (n=10) |

B1 lamarckian low-mutation children median loss -0.125 (n=16278; predicted > -0.08): contradicted
B3 baldwinian low-mutation children median loss -0.207 (n=16870; predicted < -0.10): supported
B2 spiral   lam change +0.005, bald change +0.006, interaction -0.001 (spread 0.060): inconclusive
B2 spiral3  lam change -0.007, bald change -0.032, interaction +0.025 (spread 0.052): inconclusive
```
