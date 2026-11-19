# VENDORED from microduck-evaluator/src/mdeval/server/metrics_v2/command_progress.py by scripts/sync_from_evaluator.py.
# Do not edit by hand: change the evaluator and re-run the sync.
"""``command_progress``: the score of ``rough_terrain_v2`` and ``jelly_walk_v2``.

Defined once, in docs/tasks/rough_terrain.md §5.2 (score §3); jelly_walk.md §5 uses it unchanged. The code below is
that block, reading the horizon through ``tr.max_steps`` and the schedule through ``tr.schedule``. Both families list
``command_progress`` in their yamls with ``primary_metric: command_progress_score``; neither family module defines it.
"""
from __future__ import annotations

import numpy as np

from aicrowd_microduck.local.metrics import _quat_rotate_inverse, metric
from aicrowd_microduck.local.recorder import RolloutTrace

_CP_WIN, _CP_VREF, _CP_WREF, _CP_EPS = 50, 0.4, 1.0, 1e-6   # 1 s windows at 50 Hz; 0.4 m/s; 1.0 rad/s


def _cp_tent(x):                     # 1 at the target, linear down to 0 at twice the target
    return np.where(x <= 1.0, x, np.maximum(0.0, 2.0 - x))


@metric("command_progress")
def command_progress(tr: RolloutTrace) -> dict:
    """Commanded progress (docs/tasks/rough_terrain.md §3). Statue 0, perfect tracking 1, opposite -1."""
    H = tr.max_steps                                         # 500: the fixed 10 s horizon
    n = H // _CP_WIN                                         # 10 windows
    # Commands: the episode's schedule saved at reset, never the recorded commands: after a fall those are the
    # next episode's, and the trace stops 0.5 s after the last fall (README 2026-10-02).
    cmd = tr.schedule("twist")[: n * _CP_WIN].astype(np.float64)                    # [H,K,3]
    T = min(tr.T, n * _CP_WIN)                               # velocities exist only where the trace does; any
    v = np.zeros(cmd.shape[:2] + (2,))                       # window past its end is dead for every env
    wz = np.zeros(cmd.shape[:2])
    v[:T] = _quat_rotate_inverse(tr.qpos[:T, :, 3:7], tr.qvel[:T, :, 0:3])[..., :2]   # body-frame planar velocity
    wz[:T] = tr.qvel[:T, :, 5]

    def wmean(x):                                            # [H,...] -> [n,...] window means
        return x.reshape(n, _CP_WIN, *x.shape[1:]).mean(axis=1)

    c, vw, ww = wmean(cmd), wmean(v), wmean(wz)
    cn = np.linalg.norm(c[..., :2], axis=-1)                 # |c|
    ce = np.maximum(cn, _CP_EPS)
    chat = c[..., :2] / ce[..., None]
    p = (vw * chat).sum(-1) / ce                             # progress along the command
    perp = np.abs(vw[..., 0] * chat[..., 1] - vw[..., 1] * chat[..., 0])   # |v_perp|
    q_lin = np.where(p > 0, _cp_tent(p) * np.clip(1.0 - perp / ce, 0.0, 1.0), p)
    wc = c[..., 2]
    q_ang = _cp_tent(ww / np.where(np.abs(wc) < _CP_EPS, _CP_EPS, wc))      # no drift term
    a, b = cn / _CP_VREF, np.abs(wc) / _CP_WREF              # window weights

    d = tr.done_step                                         # 0-based last step; H-1 = time_out; -1 never
    ended_early = (d >= 0) & (d + 1 < H)                     # fell_over, nan_state, out_of_terrain_bounds
    first_dead = np.where(ended_early, d // _CP_WIN, n)      # the window holding the end scores 0, and all later
    alive_w = np.arange(n)[:, None] < first_dead[None, :]
    S = (alive_w * (a * q_lin + b * q_ang)).sum(0) / np.maximum((a + b).sum(0), 1e-9)
    return {"command_progress_score": max(0.0, float(S.mean())),   # rollout clip at 0
            "command_progress_mean_unclipped": float(S.mean()),
            "command_progress_per_env": S.tolist(),
            "command_progress_truncated": int(tr.T < H)}   # informational: the score no longer depends on it
