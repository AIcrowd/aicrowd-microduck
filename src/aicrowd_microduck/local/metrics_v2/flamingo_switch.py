# VENDORED from microduck-evaluator/src/mdeval/server/metrics_v2/flamingo_switch.py by scripts/sync_from_evaluator.py.
# Do not edit by hand: change the evaluator and re-run the sync.
"""``flamingo_support_v2``: the score of ``flamingo_switch_v2`` ("commanded support").

Defined in docs/tasks/flamingo_switch.md §3 (formula) and §7.2 (inputs, readouts). Per step, a foot's lift is
``clip((z - 1 cm) / 2 cm, 0, 1)``, measured from the floor. On "one foot" steps (flag 1) the step earns
``lift_swing - lift_stance``; on "both feet" steps (flag 0) it costs ``max(lift_L, lift_R)``. The first 0.1 s after
every flip and the spawn stand before the first lift command are unscored. An episode's score is the mean over its
scored flag-1 steps plus the mean over its scored flag-0 steps, in [-2, 1]: standing still 0, perfect 1, wrong foot
-1. The rollout score is ``max(0, mean over envs)``.

Inputs, all over the fixed horizon ``tr.max_steps`` (500 at 10 s):

- ``flag`` and ``side`` come from the schedule saved at reset, ``tr.schedule("twist")`` = ``[flag, obs_side, 0]``,
  with ``side`` latched as in v1. Never the recorded commands: after an env's done step those belong to its
  auto-reset episode, and the trace stops 0.5 s after the last fall.
- Foot heights come from the recorded ``qpos`` (``kinematics_for``), NaN past each env's done step.
- A step after the done step is dead: ``q = 0``, and it stays in its denominator. So a fall costs the time it
  leaves, and the score never depends on when recording stopped.

A trace without a saved schedule (recorded before ``cmd_schedule`` existed) is scored on the recorded commands
over the recorded steps and flagged ``flamingo_v2_short_trace = 1`` (§7.2 fallback).
"""
from __future__ import annotations

import numpy as np

from aicrowd_microduck.local.metrics import _f, alive_horizon, latched_side, metric, pad_horizon
from aicrowd_microduck.local.recorder import RolloutTrace

FLAMINGO_V2_LIFT_LO_M = 0.01      # a foot at or below this height scores as down
FLAMINGO_V2_LIFT_HI_M = 0.03      # a foot at or above this height scores as fully up
FLAMINGO_V2_ALLOWANCE_S = 0.1     # unscored after every flag flip (5 steps at 50 Hz): the sensorimotor delay


def flamingo_v2_lift(z: np.ndarray) -> np.ndarray:
    """Foot height above the floor -> lift in [0, 1]: 0 up to 1 cm, 1 from 3 cm. NaN (no data) reads as 0."""
    return np.clip((np.nan_to_num(z) - FLAMINGO_V2_LIFT_LO_M) / (FLAMINGO_V2_LIFT_HI_M - FLAMINGO_V2_LIFT_LO_M), 0.0, 1.0)


def flamingo_v2_episode(flag: np.ndarray, side: np.ndarray, z_left: np.ndarray, z_right: np.ndarray,
                        alive: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """All inputs ``[T, K]``; ``side`` latched (+1 = right foot is stance). Returns per-env ``(S, su, sd)``:
    the episode score in [-2, 1] and its flag-1 and flag-0 halves. The §7.2 code."""
    up = flag > 0.5
    lL, lR = flamingo_v2_lift(z_left), flamingo_v2_lift(z_right)
    right = side > 0
    q = np.where(up, np.where(right, lL - lR, lR - lL), -np.maximum(lL, lR))
    q = np.where(alive, q, 0.0)                                       # dead steps: 0, still counted
    change = np.zeros_like(up)
    change[0] = up[0]
    change[1:] = up[1:] != up[:-1]
    T = up.shape[0]
    d = round(FLAMINGO_V2_ALLOWANCE_S / dt)
    last = np.maximum.accumulate(np.where(change, np.arange(T)[:, None], -10 ** 9), axis=0)
    keep = (np.arange(T)[:, None] - last) >= d
    started = np.maximum.accumulate(up, axis=0)                       # scoring starts at the first lift command
    up_s, dn_s = up & keep, ~up & started & keep
    su = np.where(up_s, q, 0).sum(0) / np.maximum(up_s.sum(0), 1)
    sd = np.where(dn_s, q, 0).sum(0) / np.maximum(dn_s.sum(0), 1)
    return su + sd, su, sd


@metric("flamingo_support_v2")
def flamingo_support_v2(tr: RolloutTrace) -> dict:
    """Commanded support over the fixed horizon (docs/tasks/flamingo_switch.md §3, §7.2)."""
    from aicrowd_microduck.local.kinematics import kinematics_for

    dt = float(tr.meta.get("control_dt", 0.02))
    foot_z = kinematics_for(tr).foot_pos[..., 2].astype(np.float64)   # [T, K, 2] (left, right), NaN after done
    if "twist" in tr.cmd_schedule:
        sched = tr.schedule("twist")                                  # [max_steps, K, 3] = [flag, obs_side, 0]
        flag, side_obs = sched[..., 0], sched[..., 1]
        alive = alive_horizon(tr)                                     # [max_steps, K]
        z = pad_horizon(tr, foot_z)                                   # [max_steps, K, 2], NaN when not alive
        short = 0
    else:                                                             # §7.2 fallback: the recorded stream
        cmd = tr.commands["twist"]
        flag, side_obs = cmd[..., 0], cmd[..., 1]
        alive = tr.alive_mask()
        z = foot_z
        short = 1
    S, su, sd = flamingo_v2_episode(flag, latched_side(side_obs), z[..., 0], z[..., 1], alive, dt)
    d = tr.done_step
    falls = int(((d >= 0) & (d + 1 < tr.max_steps)).sum())            # ended before the horizon
    return {"flamingo_v2_score": max(0.0, _f(S.mean())),             # rollout clip at 0
            "flamingo_v2_score_raw": _f(S.mean()),
            "flamingo_v2_up": _f(su.mean()), "flamingo_v2_down": _f(sd.mean()),
            "flamingo_v2_score_per_env": S.tolist(),
            "flamingo_v2_falls": falls,
            "flamingo_v2_short_trace": short}
