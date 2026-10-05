# Champion architecture: `full100m_direction_s2` (genome 7033)

| | count |
|---|---|
| inputs | 14 |
| nodes (stored) | 4 = 2 output + 2 hidden |
| nodes (effective, on an input->output path) | 4 |
| enabled connections (stored) | 16 (11 disabled) |
| enabled connections (effective) | 16 |
| activation functions | {'sigmoid': 4} |
| aggregation | {'sum': 4} |
| network type | feed-forward |

Inputs used (effective): x, y, vx, bx, by, bvx, bvy, ox, ovy, bx-x, by-y
Inputs unused: vy, oy, ovx

Node evaluation (neat-python): out = sigmoid(5 * (bias + response * sum(w * in))); responses here: [1.0].

| node | bias | inputs (weight), strongest first | on a path to an output? |
|---|---|---|---|
| direction | -1.44 | bx-x (-11.36), x (+4.43), bx (-3.46), ox (-1.62), h841 (+0.89), bvx (-0.45) | yes |
| jump | -0.65 | by-y (+7.43), ovy (+3.23), bvy (+1.96), bx (-1.89), bx-x (-1.02), h1297 (-0.91), x (+0.79), y (-0.76) | yes |
| h841 | +1.34 | by (-2.55) | yes |
| h1297 | +1.09 | vx (-0.35) | yes |
