# VENDORED from microduck-evaluator/src/mdeval/server/metrics_v2/trampoline_height.py by scripts/sync_from_evaluator.py.
# Do not edit by hand: change the evaluator and re-run the sync.
"""``trampoline_height``: the score of ``trampoline_height_v2`` (levels L1-L6).

docs/tasks/trampoline_height.md §3 (v2 score) and §5 "Implementation spec (v2)", 2. Evaluator::

    per step t, env k:  h[t,k] = trunk z (free joint qpos[2]) - 0.1088 m        (HOME rest on the K300 mat)
    episode k:          H[k]   = max(0, max of h[t,k] over 50 <= t <= done step)   (1 s settle)
    rollout:            mean of H[k] over its 8 envs / S, S = 0.1 m            -> trunk_rise_score (not capped at 1)

Rows are the trace's post-step rows: row 50 is the state after the 51st control step. The window ends at each env's
done step (fell_over, nan_state, off_mat or the time-out), whose row holds the terminal state. Rows after it belong
to the env's auto-reset episode and the trace stops 0.5 s after the last fall (``record_after_done_s``); neither can
reach the score, because the window is ``alive_horizon`` and the reading is a max, with no division by the trace
length. A crash scores its peak ("crashes count", M53); a non-finite height (a diverged state) is not a height and is
skipped, so it cannot turn the score into NaN.
"""
from __future__ import annotations

import numpy as np

from aicrowd_microduck.local.metrics import _f, alive_horizon, metric, qpos_horizon
from aicrowd_microduck.local.recorder import RolloutTrace

TRAMPOLINE_V2_REST_Z = 0.1088   # m, HOME trunk z standing on the K300 mat (§2); the plane is at z = 0
TRAMPOLINE_V2_SETTLE_S = 1.0    # s skipped at the start: the soles-at-mat spawn puts the trunk ~9 mm above rest
TRAMPOLINE_V2_SCALE = 0.1       # m, S (§3): a fixed unit so the family sits near the other tasks' 0-1 range


@metric("trampoline_height")
def trampoline_height(tr: RolloutTrace) -> dict:
    """Trunk rise after a 1 s settle, per env, over the episode up to its done step (docs/tasks/trampoline_height.md §3)."""
    dt = float(tr.meta.get("control_dt", 0.02))
    settle = int(round(TRAMPOLINE_V2_SETTLE_S / dt))          # 50 rows at 50 Hz
    z = qpos_horizon(tr)[..., 2]                              # [H, K] trunk z; NaN after the done step / trace end
    win = alive_horizon(tr) & np.isfinite(z)                  # rows settle..done_step of each env
    win[:settle] = False
    zw = np.where(win, z, -np.inf)
    zmax = zw.max(axis=0)                                     # -inf if the env ended before row 50
    rise = np.clip(zmax - TRAMPOLINE_V2_REST_Z, 0.0, None)    # [K] trunk rise, m
    peak = np.where(win.any(axis=0), zw.argmax(axis=0), -1)
    return {"trunk_rise_score": _f(rise.mean() / TRAMPOLINE_V2_SCALE),     # primary; uncapped
            "trunk_rise_m_mean": _f(rise.mean()), "trunk_rise_m_max": _f(rise.max()),
            "trunk_rise_m_per_env": rise.tolist(),
            "trunk_rise_peak_step_per_env": peak.astype(int).tolist(),      # trace row of the max, -1 if none
            "ended_before_settle": int((~win.any(axis=0)).sum())}
