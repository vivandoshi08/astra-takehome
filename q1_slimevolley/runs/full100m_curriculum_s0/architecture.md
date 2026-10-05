# Champion architecture: `full100m_curriculum_s0` (genome 9993)

| | count |
|---|---|
| inputs | 14 |
| nodes (stored) | 8 = 3 output + 5 hidden |
| nodes (effective, on an input->output path) | 6 |
| enabled connections (stored) | 11 (34 disabled) |
| enabled connections (effective) | 10 |
| activation functions | {'sigmoid': 8} |
| aggregation | {'sum': 8} |
| network type | feed-forward |

Inputs used (effective): bvx, oy, ovx, ovy, bx-x, by-y
Inputs unused: x, y, vx, vy, bx, by, bvy, ox

Node evaluation (neat-python): out = sigmoid(5 * (bias + response * sum(w * in))); responses here: [0.2].

| node | bias | inputs (weight), strongest first | on a path to an output? |
|---|---|---|---|
| forward | +0.03 | bvx (-4.13), oy (+0.39) | yes |
| backward | +0.07 | bx-x (+11.24), h857 (+3.42) | yes |
| jump | +0.32 | h1938 (-0.97), ovx (-0.30), h1056 (+0.17) | yes |
| h150 | -2.30 | bx (-4.38) | no (dead end) |
| h857 | -1.34 | bvx (+3.40) | yes |
| h890 | +0.21 | — | no (dead end) |
| h1056 | -0.55 | ovy (-0.47) | yes |
| h1938 | +0.00 | by-y (+1.76) | yes |
