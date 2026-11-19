# VENDORED from microduck-evaluator/src/mdeval/server/metrics_v2/roller_descent.py by scripts/sync_from_evaluator.py.
# Do not edit by hand: change the evaluator and re-run the sync.
"""``descent_score``: the score of ``roller_descent_v2`` (docs/tasks/roller_descent.md §3 and §5 "Evaluator").

Each episode spawns rolling (entry speed ``qvel0``) just past the top edge of a ramp that descends along world +x.
It earns credit only if, over the first 3.5 s (``W``, spawn to the deployed handback at phase 0.7), it

- stays **upright** (alive on every step of W; falls are ``fell_over``, ``nan_state`` or ``fell_into_void``),
- stays **in lane** (never more than 0.5 m sideways from the spawn line),
- is **crouched** through the hold (``H`` = 0.75-2.25 s: trunk <= 0.085 m above the ground straight below it on
  >= 90% of the steps), and
- keeps the speed the slope gives: ``speed`` = 0 at a ratio of 0.3 of the lossless rolling speed, full at 0.7.

``S[k] = upright * crouched * in_lane * speed``; the rollout score is the plain mean of S over its envs (no band
weights: the family weights live in ``tasks/competition_v2.yaml``).

**Horizon fix** (roller_descent.md §5): every window is laid on the horizon ``tr.max_steps``, not on the trace. A
batch that falls before 3.5 s ends its trace early (0.5 s after the last fall); the steps past the trace end count
as fallen (``alive_horizon``), so ``upright`` is False and the score is a finite 0, never NaN.

"Ground" is a downward ray onto the terrain (geom group 0) of the rollout's own compiled model (``tr.model_path``,
``assets/model_<task>_<seed>.mjb``), with the robot parked far below the course.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from aicrowd_microduck.local.metrics import _f, alive_horizon, metric, qpos_horizon, qvel_horizon
from aicrowd_microduck.local.recorder import RolloutTrace

K_ROLL, G = 0.293, 9.81            # 0.737 / (0.737 + 4 * 1e-4 / 0.015**2), wheel armature 1e-4 (T1)
H_CUT, H_SHARE = 0.085, 0.9        # crouched: trunk <= 0.085 m above ground on >= 90% of the hold
R_ZERO, R_FULL = 0.3, 0.7          # speed credit 0 at ratio 0.3, full at 0.7
LANE = 0.5                         # max sideways offset from the spawn line, m
W_END, H_START, H_END = 3.5, 0.75, 2.25   # scored window and hold window, s since spawn
_RAY_TOP = 5.0                     # rays start this high (world z) and point straight down


def _ground_under(model_path: str | Path, xy: np.ndarray) -> np.ndarray:
    """Terrain height straight below each xy (``[..., 2]`` -> ``[...]``); NaN where the ray misses or xy is not
    finite. The robot (free joint first in qpos) is parked 100 m below the course so it can never be hit."""
    import mujoco

    m = mujoco.MjModel.from_binary_path(str(model_path))
    d = mujoco.MjData(m)
    d.qpos[:3] = (0.0, 0.0, -100.0)
    mujoco.mj_forward(m, d)
    grp = np.array([1, 0, 0, 0, 0, 0], dtype=np.uint8)            # geom group 0 = terrain
    gid = np.array([-1], dtype=np.int32)
    down = np.array([0.0, 0.0, -1.0])
    out = np.full(xy.shape[:-1], np.nan)
    for idx in np.ndindex(*xy.shape[:-1]):
        x, y = xy[idx]
        if not (np.isfinite(x) and np.isfinite(y)):
            continue
        dist = mujoco.mj_ray(m, d, np.array([x, y, _RAY_TOP]), down, grp, 1, -1, gid)
        if dist >= 0:
            out[idx] = _RAY_TOP - dist
    return out


@metric("descent_score")
def descent_score(tr: RolloutTrace) -> dict:
    """Roll down the ramp crouched, keep the speed the slope gives, stay in lane (roller_descent.md §3)."""
    if tr.qvel0 is None or tr.qpos0 is None:
        raise ValueError("descent_score needs the reset state: the trace has no qpos0/qvel0 "
                         "(recorded by an evaluator older than the v2 trace fields)")
    if tr.model_path is None or not Path(tr.model_path).exists():
        raise FileNotFoundError("descent_score needs the rollout's compiled model for the ground height: expected "
                                f"<run>/assets/model_<task>_<seed>.mjb (got {tr.model_path!r})")
    n_t = tr.max_steps                                            # the horizon (225 at 4.5 s); never tr.T
    dt = float(tr.meta["control_dt"])
    t = (np.arange(n_t) + 1) * dt                                 # qpos[t] is recorded after step t
    W = t <= W_END + 1e-9
    H = (t >= H_START - 1e-9) & (t <= H_END + 1e-9)
    alive = alive_horizon(tr)                                     # [n_t, K]; past the trace end counts as fallen
    upright = alive[W].all(0)                                     # [K]

    qpos, qvel = qpos_horizon(tr), qvel_horizon(tr)               # NaN wherever not alive
    xy, z = qpos[..., :2], qpos[..., 2]
    xy0 = np.asarray(tr.qpos0, dtype=np.float64)[:, :2]
    g = _ground_under(tr.model_path, np.concatenate([xy0[None], xy[H]], 0))   # [1 + |H|, K]; only H needs ground
    g0, g_hold = g[0], g[1:]
    h_hold = z[H] - g_hold                                        # trunk height above the ground, hold window
    v_hold = qvel[H][..., 0]                                      # speed along the course: world +x on every level
    v0 = np.asarray(tr.qvel0, dtype=np.float64)[:, 0]             # entry speed along +x (0.3 cos θ; U(0.4,1.5) flat)
    lat = np.abs(xy[..., 1] - xy0[None, :, 1])                    # sideways offset from the spawn line
    drop = np.maximum(g0[None] - g_hold, 0.0)
    v_ref = np.sqrt(v0[None] ** 2 + 2 * K_ROLL * G * drop)        # lossless rolling speed after the same drop
    with np.errstate(invalid="ignore", divide="ignore"):
        crouched = upright & ((h_hold <= H_CUT).mean(0) >= H_SHARE)
        in_lane = upright & (np.nanmax(np.where(W[:, None], lat, 0.0), 0) <= LANE)
        ratio = np.where(upright, v_hold.mean(0) / np.maximum(v_ref.mean(0), 1e-6), 0.0)
        h_mean = np.where(upright, np.nanmean(np.where(upright[None], h_hold, 0.0), 0), np.nan)
    ratio = np.where(np.isfinite(ratio), ratio, 0.0)              # a ray miss earns no speed credit, never NaN
    speed = np.clip((ratio - R_ZERO) / (R_FULL - R_ZERO), 0.0, 1.0)
    S = np.where(upright & crouched & in_lane, speed, 0.0)
    lat_max = np.nanmax(np.where(W[:, None] & alive, lat, 0.0), 0)
    return {"descent_score": _f(S.mean()), "descent_score_per_env": S.tolist(),
            "descent_upright_frac": _f(upright.mean()), "descent_crouched_frac": _f(crouched.mean()),
            "descent_in_lane_frac": _f(in_lane.mean()),
            "descent_speed_ratio_mean": _f(ratio[upright].mean()) if upright.any() else 0.0,
            "descent_upright_per_env": upright.astype(int).tolist(),
            "descent_crouched_per_env": crouched.astype(int).tolist(),
            "descent_in_lane_per_env": in_lane.astype(int).tolist(),
            "descent_speed_ratio_per_env": ratio.tolist(),
            "descent_h_hold_mean_per_env": [None if not np.isfinite(x) else float(x) for x in h_mean],
            "descent_lat_max_W_per_env": lat_max.tolist(),
            "descent_v0_per_env": v0.tolist()}
