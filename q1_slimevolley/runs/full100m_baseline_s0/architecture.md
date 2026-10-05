# Champion architecture: `full100m_baseline_s0` (genome 21946)

| | count |
|---|---|
| inputs | 14 |
| nodes (stored) | 15 = 3 output + 12 hidden |
| nodes (effective, on an input->output path) | 5 |
| enabled connections (stored) | 16 (18 disabled) |
| enabled connections (effective) | 5 |
| activation functions | {'sigmoid': 15} |
| aggregation | {'sum': 15} |
| network type | feed-forward |

Inputs used (effective): x, y, bx-x
Inputs unused: vx, vy, bx, by, bvx, bvy, ox, oy, ovx, ovy, by-y

Node evaluation (neat-python): out = sigmoid(5 * (bias + response * sum(w * in))); responses here: [0.2].

| node | bias | inputs (weight), strongest first | on a path to an output? |
|---|---|---|---|
| forward | +0.34 | h4035 (+1.44) | yes |
| backward | -1.23 | h3358 (+0.12) | yes |
| jump | +0.10 | x (+1.59), bx-x (+1.17) | yes |
| h2285 | -0.44 | — | no (dead end) |
| h2831 | +0.81 | bvx (+1.52) | no (dead end) |
| h3087 | -0.24 | — | no (dead end) |
| h3150 | +0.03 | h4343 (-3.00), h4388 (-2.15) | no (dead end) |
| h3358 | +0.10 | — | yes |
| h4035 | -0.51 | y (+0.66) | yes |
| h4068 | -1.10 | h4359 (+0.54) | no (dead end) |
| h4196 | -0.01 | bx (-2.05), h4068 (-0.18), ovx (-0.07) | no (dead end) |
| h4297 | +1.10 | by (+1.59) | no (dead end) |
| h4343 | -0.66 | bx (-0.46) | no (dead end) |
| h4359 | +0.02 | by-y (+0.74) | no (dead end) |
| h4388 | -0.53 | bx-x (+0.59) | no (dead end) |
