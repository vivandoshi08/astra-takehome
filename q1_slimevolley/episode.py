"""The part you implement.

`evolve.py` handles the plumbing — population setup, parallel evaluation, step
accounting, writing results. This file is the piece it can't do for you.
"""

from __future__ import annotations


def play_episode(net, env) -> float:
    """Play one episode against the built-in opponent and return total reward.

    Two things to work out:

    1. **How a network output becomes an action.** ``net.activate(observation)``
       returns one value per output node. ``env.action_space`` tells you what
       the environment expects. The mapping between them is your call.

    2. **How an episode runs.** Reset the environment, drive it with your
       network until the episode is over, accumulating reward along the way.

    Return the episode's total reward. Keep the signature — `evolve.py` calls
    this, and so will anything else you write.
    """
    obs = env.reset()
    total = 0.0
    done = False
    while not done:
        obs, reward, done, _ = env.step(action_from_outputs(net.activate(obs)))
        total += reward
    return total


def action_from_outputs(out) -> list[int]:
    """Three outputs -> MultiBinary(3) [forward, backward, jump].

    Thresholded at 0.5, the midpoint of the default sigmoid output. Must stay in
    sync with `replay.action_from()`.
    """
    return [1 if o > 0.5 else 0 for o in out]
