"""The part you implement.

`evolve.py` handles the plumbing — population setup, parallel evaluation, step
accounting, writing results. This file is the piece it can't do for you.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class EpisodeStats:
    """Everything one episode tells us, for the logs (see LOGS.md)."""

    score: float          # total reward: points_for - points_against
    steps: int
    points_for: int
    points_against: int
    act_fwd: int          # steps with each button pressed (may co-occur)
    act_back: int
    act_jump: int
    act_idle: int         # steps with no button pressed

    def to_dict(self) -> dict:
        return asdict(self)


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
    return run_episode(net, env).score


def run_episode(net, env) -> EpisodeStats:
    """`play_episode`, keeping per-episode detail for the logs."""
    obs = env.reset()
    # env.reset() resets the game but not the built-in opponent, which is an
    # RNN: its hidden state would carry over from the previous episode, so the
    # same env seed would not replay the same game. Reset it explicitly.
    env.policy.reset()
    s = EpisodeStats(0.0, 0, 0, 0, 0, 0, 0, 0)
    done = False
    while not done:
        action = action_from_outputs(net.activate(obs))
        obs, reward, done, _ = env.step(action)
        s.score += reward
        s.steps += 1
        s.points_for += reward > 0
        s.points_against += reward < 0
        s.act_fwd += action[0]
        s.act_back += action[1]
        s.act_jump += action[2]
        s.act_idle += not any(action)
    return s


def action_from_outputs(out) -> list[int]:
    """Three outputs -> MultiBinary(3) [forward, backward, jump], each on if > 0.5.

    neat_config.ini keeps every node sigmoid (activation_mutate_rate = 0), so
    0.5 is the midpoint of every output. Must stay in sync with
    `replay.action_from()`, which imports this.
    """
    return [1 if o > 0.5 else 0 for o in out]
