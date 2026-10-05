"""Watch a champion play. Writes an animated GIF you can scrub through.

    python replay.py --winner runs/run/champion.pkl --out champion.gif
    python replay.py --winner runs/run/champion.pkl --seed 1000042

Defaults: seed 1_000_000 (the first held-out seed; seeds below 1_000_000 are
training seeds) and 3000 max steps (the episode limit used in training, so the
score shown is the full-episode score, not a truncated one). The NEAT config is
read from `config.ini` next to `--winner` when present (the exact copy the run
used), else from `--config`.

The environment's own `env.render()` does not work: `slimevolley.py` is
vendored from a project that predates gymnasium and its renderer imports
`gym.envs.classic_control.rendering`, which no longer exists. Rather than
have you fight that, this draws frames directly from the game state.

Behavioural analysis is a Q1 deliverable, and this is here so you can do it
by looking rather than only by summary statistics. Read it, change it, or
replace it — it is a convenience, not part of the specification.

ONE THING TO CHECK: `action_from()` below turns a network's outputs into an
action the same way the shipped comments suggest — three outputs, thresholded
at 0.5. If your `play_episode()` maps outputs to actions differently, edit
`action_from()` to match, or the replay will not show the policy you trained.
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import neat
from PIL import Image, ImageDraw, ImageFont

import slimevolley as sv
from episode import _CONTACTS, net_inputs, policy_action, run_episode
from evolve import load_config, make_net

HERE = Path(__file__).parent

W, H = 640, 400
X_MIN, X_MAX = -sv.REF_W / 2, sv.REF_W / 2
Y_MIN, Y_MAX = 0.0, sv.REF_W / 2

SKY = (222, 232, 245)
GROUND = (120, 160, 110)
FENCE = (70, 70, 80)
LEFT = (200, 80, 70)       # the built-in opponent
RIGHT = (60, 110, 190)     # you
BALL = (245, 205, 70)
INK = (40, 40, 50)

try:
    FONT = ImageFont.load_default(size=15)
except TypeError:            # Pillow < 10.1 has no size argument
    FONT = ImageFont.load_default()


def action_from(net, obs) -> list[int]:
    """Network outputs -> a MultiBinary(3) action. Edit to match your own."""
    return policy_action(net, net.activate(net_inputs(net, obs)))


def _px(wx: float, wy: float) -> tuple[float, float]:
    x = (wx - X_MIN) / (X_MAX - X_MIN) * W
    y = H - (wy - Y_MIN) / (Y_MAX - Y_MIN) * H
    return x, y


def _scale(v: float) -> float:
    return v / (X_MAX - X_MIN) * W


def draw_frame(game, step: int, score: float, hud: str = "") -> Image.Image:
    img = Image.new("RGB", (W, H), SKY)
    d = ImageDraw.Draw(img)

    _, gy = _px(X_MIN, sv.REF_U)
    d.rectangle([0, gy, W, H], fill=GROUND)

    f = game.fence
    fx0, fy0 = _px(f.x - f.w / 2, f.y + f.h / 2)
    fx1, fy1 = _px(f.x + f.w / 2, f.y - f.h / 2)
    d.rectangle([fx0, fy0, fx1, fy1], fill=FENCE)

    for agent, colour in ((game.agent_left, LEFT), (game.agent_right, RIGHT)):
        cx, cy = _px(agent.x, agent.y)
        r = _scale(agent.r)
        d.pieslice([cx - r, cy - r, cx + r, cy + r], 180, 360, fill=colour)
        ex, ey = _px(agent.x + agent.dir * agent.r * 0.45, agent.y + agent.r * 0.55)
        er = max(2.0, _scale(0.22))
        d.ellipse([ex - er, ey - er, ex + er, ey + er], fill=(255, 255, 255))
        d.ellipse([ex - er / 2, ey - er / 2, ex + er / 2, ey + er / 2], fill=INK)

    bx, by = _px(game.ball.x, game.ball.y)
    br = max(3.0, _scale(game.ball.r * 1.6))
    d.ellipse([bx - br, by - br, bx + br, by + br], fill=BALL, outline=INK)

    d.text((12, 10), f"built-in AI   lives {game.agent_left.life}",
           fill=LEFT, font=FONT)
    right = f"lives {game.agent_right.life}   you"
    d.text((W - 12 - d.textlength(right, font=FONT), 10), right,
           fill=RIGHT, font=FONT)
    mid = f"step {step}      score {score:+.0f}"
    d.text(((W - d.textlength(mid, font=FONT)) / 2, 10), mid, fill=INK, font=FONT)
    if hud:
        d.text(((W - d.textlength(hud, font=FONT)) / 2, 30), hud, fill=INK, font=FONT)
    return img


def record(net, out: str, seed: int = 1_000_000, max_steps: int = 3000,
           every: int = 3, fps: int = 30, sheet: str | None = None,
           sheet_frames: int = 24) -> dict:
    """Play one episode, writing frames to `out` (and a contact sheet PNG).

    The HUD shows our contacts / returns so far and the buttons pressed this
    step, counted exactly as episode.run_episode counts them for the logs.
    Returns the episode totals so they can be checked against run_episode.
    """
    env = sv.SlimeVolleyEnv()
    env.seed(seed)
    obs = env.reset()
    # env.reset() does not reset the built-in opponent's RNN state; without
    # this the same seed would not replay the game evaluate.py scored.
    env.policy.reset()
    if hasattr(net, "reset"):
        net.reset()
    frames, score = [], 0.0
    contacts = returns = 0
    pending = False
    for step in range(max_steps):
        action = action_from(net, obs)
        before = _CONTACTS[1]
        result = env.step(action)
        obs, reward, done = result[0], result[1], result[2]
        score += reward
        touched = _CONTACTS[1] - before
        contacts += touched
        if touched:
            pending = True
        elif pending and (env.game.ball.x < 0 or reward):
            returns += env.game.ball.x < 0 or reward > 0
            pending = False
        if step % every == 0 or done:
            buttons = "".join(c if a else "." for c, a in zip("FBJ", action))
            hud = f"contacts {contacts}   returns {returns}   buttons {buttons}"
            frames.append(draw_frame(env.game, step, score, hud))
        if done:
            break
    frames[0].save(out, save_all=True, append_images=frames[1:],
                   duration=int(1000 / fps), loop=0)
    if sheet:
        write_sheet(frames, sheet, sheet_frames)
    return {"score": score, "steps": step + 1, "contacts": contacts, "returns": returns}


def write_sheet(frames: list, path: str, n: int, cols: int = 4) -> None:
    """n evenly spaced frames tiled into one PNG, for a quick look without a GIF viewer."""
    pick = [frames[round(i * (len(frames) - 1) / max(1, n - 1))] for i in range(n)]
    tw, th = W // 2, H // 2
    rows = (len(pick) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * tw, rows * th), (255, 255, 255))
    for i, f in enumerate(pick):
        sheet.paste(f.resize((tw, th)), ((i % cols) * tw, (i // cols) * th))
    sheet.save(path)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--winner", default=str(HERE / "runs" / "run" / "champion.pkl"))
    ap.add_argument("--config", default=str(HERE / "neat_config.ini"))
    ap.add_argument("--seed", type=int, default=1_000_000,
                    help="episode seed (>= 1_000_000 are held-out, never trained on)")
    ap.add_argument("--max-steps", type=int, default=3000)
    ap.add_argument("--every", type=int, default=3, help="keep 1 frame in N")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--out", default="champion.gif")
    ap.add_argument("--sheet", help="also write a PNG contact sheet of evenly spaced frames")
    args = ap.parse_args()

    run_config = Path(args.winner).parent / "config.ini"
    config_path = run_config if run_config.exists() else Path(args.config)
    print(f"config: {config_path}")
    config = load_config(config_path)
    with open(args.winner, "rb") as fh:
        blob = pickle.load(fh)
    genome = blob["genome"] if isinstance(blob, dict) else blob

    shown = record(make_net(genome, config), args.out, seed=args.seed, max_steps=args.max_steps,
                   every=args.every, fps=args.fps, sheet=args.sheet)
    print(f"seed {args.seed}: score {shown['score']:+.1f} over {shown['steps']} steps, "
          f"contacts {shown['contacts']}, returns {shown['returns']} -> {args.out}")
    # Cross-check: the replay must show the same episode the logs/evaluate score.
    env = sv.SlimeVolleyEnv()
    env.seed(args.seed)
    net = make_net(genome, config)
    if hasattr(net, "reset"):
        net.reset()
    logged = run_episode(net, env).to_dict()
    same = all(shown[k] == logged[k] for k in shown)
    print(f"matches run_episode (what the logs record): {same}"
          + ("" if same else f"  replay {shown} vs logged { {k: logged[k] for k in shown} }"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
