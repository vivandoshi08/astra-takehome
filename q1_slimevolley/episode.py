"""The part you implement.

`evolve.py` handles the plumbing — population setup, parallel evaluation, step
accounting, writing results. This file is the piece it can't do for you.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import slimevolley

# --- Ball-contact counting -------------------------------------------------
# The env doesn't report contacts, and they are the earliest sign of learning
# (an agent has to touch the ball long before it wins points). We wrap
# Particle.bounce: the original runs unchanged, we only count when the "other
# ball" is an agent (dir +1 = our agent on the right, -1 = the opponent).
# Physics is untouched — see the trajectory-equality check in the commit log.
_CONTACTS = {1: 0, -1: 0}
_bounce = slimevolley.Particle.bounce


def _counting_bounce(self, p):
    side = getattr(p, "dir", 0)
    if side in _CONTACTS:
        _CONTACTS[side] += 1
    return _bounce(self, p)


slimevolley.Particle.bounce = _counting_bounce


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
    act_both: int         # steps with forward AND backward (they cancel: no move)
    out_sat: int          # output values < 0.02 or > 0.98 (3 per step)
    contacts: int         # times the ball bounced off our agent
    opp_contacts: int     # times it bounced off the opponent
    returns: int          # our contacts after which the ball reached the opponent's side

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
    s = EpisodeStats(0.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    pending_return = False  # we touched the ball; has it crossed the net yet?
    done = False
    while not done:
        out = net.activate(net_inputs(net, obs))
        action = action_from_outputs(out)
        before = _CONTACTS[1], _CONTACTS[-1]
        obs, reward, done, _ = env.step(action)
        touched = _CONTACTS[1] - before[0]
        s.contacts += touched
        s.opp_contacts += _CONTACTS[-1] - before[1]
        if touched:
            pending_return = True
        elif pending_return and (env.game.ball.x < 0 or reward):
            # crossed to the opponent's half (world x < 0), or the point ended
            # with the ball on their side (reward > 0)
            s.returns += env.game.ball.x < 0 or reward > 0
            pending_return = False
        s.out_sat += sum(o < 0.02 or o > 0.98 for o in out)
        s.score += reward
        s.steps += 1
        s.points_for += reward > 0
        s.points_against += reward < 0
        s.act_fwd += action[0]
        s.act_back += action[1]
        s.act_jump += action[2]
        s.act_idle += not any(action)
        s.act_both += action[0] and action[1]
    return s


def net_inputs(net, obs):
    """The network's input vector: the 12 observations, plus — for 14-input
    networks (configs/relative.ini) — ball position relative to our agent.

    Observations are in our own frame, scaled /10: [x, y, vx, vy, bx, by, bvx,
    bvy, ox, oy, ovx, ovy]. 'Get under the ball' needs output ~ f(bx - x);
    from raw inputs that takes two weights tuned to cancel, from bx - x it
    takes one, which a single mutation can find. The input count is read from
    the network, so 12-input champions keep working unchanged.
    """
    if len(net.input_nodes) == 14:
        return [*obs, obs[4] - obs[0], obs[5] - obs[1]]
    return obs


def action_from_outputs(out) -> list[int]:
    """Three outputs -> MultiBinary(3) [forward, backward, jump], each on if > 0.5.

    neat_config.ini keeps every node sigmoid (activation_mutate_rate = 0), so
    0.5 is the midpoint of every output. Must stay in sync with
    `replay.action_from()`, which imports this.
    """
    return [1 if o > 0.5 else 0 for o in out]
