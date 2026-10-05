# Champion architecture: `full100m_direction_s1` (genome 7639)

| | count |
|---|---|
| inputs | 14 |
| nodes (stored) | 4 = 2 output + 2 hidden |
| nodes (effective, on an input->output path) | 4 |
| enabled connections (stored) | 19 (13 disabled) |
| enabled connections (effective) | 19 |
| activation functions | {'sigmoid': 4} |
| aggregation | {'sum': 4} |
| network type | feed-forward |

Inputs used (effective): x, y, vx, vy, by, bvx, bvy, oy, ovx, ovy, bx-x, by-y
Inputs unused: bx, ox

Node evaluation (neat-python): out = sigmoid(5 * (bias + response * sum(w * in))); responses here: [1.0].

| node | bias | inputs (weight), strongest first | on a path to an output? |
|---|---|---|---|
| direction | -0.47 | bx-x (-11.46), by-y (-1.02), bvx (-0.65), ovx (-0.32) | yes |
| jump | +1.18 | bvy (-5.22), x (-1.98), h1045 (-1.40), vy (+0.84), h590 (-0.21), ovx (+0.08), vx (-0.07) | yes |
| h590 | -1.42 | oy (+1.91), bvy (+0.49), by (-0.32), ovy (-0.32), ovx (+0.16) | yes |
| h1045 | -1.71 | x (+2.52), y (+0.59), bvy (-0.15) | yes |
