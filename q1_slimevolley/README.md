# Slime volleyball environment

Single-file slime-volleyball environment, vendored for this exam so you don't have to
fight dependencies. Adapted from [`hardmaru/slimevolleygym`](https://github.com/hardmaru/slimevolleygym),
Apache-2.0, original copyright David Ha (2020).

## Requirements

Python 3.10+, `gymnasium`, `numpy`. No GPU. Rendering is lazy-loaded, so headless use
pulls in nothing extra.

## Use

```python
import slimevolley
env = slimevolley.SlimeVolleyEnv()
obs = env.reset()
```

The observation is a 12-dimensional vector. The environment ships with a built-in
baseline opponent (a 120-parameter network from the original 2015 demo) — that is what
you are evolving against.

Throughput is roughly 78,000 steps/sec headless on a single laptop core, and it
parallelises cleanly across cores, so the full step budget for this question runs in
about 20 minutes. If you are seeing anything close to that, you are fine on compute.
