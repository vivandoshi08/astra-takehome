# Champion architecture: `full100m_direction_s0` (genome 5487)

| | count |
|---|---|
| inputs | 14 |
| nodes (stored) | 6 = 2 output + 4 hidden |
| nodes (effective, on an input->output path) | 6 |
| enabled connections (stored) | 10 (16 disabled) |
| enabled connections (effective) | 10 |
| activation functions | {'sigmoid': 6} |
| aggregation | {'sum': 6} |
| network type | feed-forward |

Inputs used (effective): vy, bvx, bvy, ovx, bx-x
Inputs unused: x, y, vx, bx, by, ox, oy, ovy, by-y

Node evaluation (neat-python): out = sigmoid(5 * (bias + response * sum(w * in))); responses here: [1.0].

| node | bias | inputs (weight), strongest first | on a path to an output? |
|---|---|---|---|
| direction | -0.47 | bx-x (-8.14), bvx (-0.42) | yes |
| jump | +0.54 | h982 (+2.22), vy (-1.30), ovx (+0.75), bvx (+0.39) | yes |
| h581 | +1.18 | bvy (+1.20) | yes |
| h965 | -1.56 | h581 (+0.70) | yes |
| h982 | -0.34 | h1051 (+1.82) | yes |
| h1051 | +0.00 | h965 (+0.86) | yes |
