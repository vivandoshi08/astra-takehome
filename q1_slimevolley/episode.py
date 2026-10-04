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
    raise NotImplementedError("implement play_episode()")
