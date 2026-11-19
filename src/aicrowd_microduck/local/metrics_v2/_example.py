# VENDORED from microduck-evaluator/src/mdeval/server/metrics_v2/_example.py by scripts/sync_from_evaluator.py.
# Do not edit by hand: change the evaluator and re-run the sync.
"""A worked v2 metric. The leading underscore keeps it out of discovery, so it never registers in production;
``tests/test_v2_shared.py`` imports it explicitly to keep it honest.

It scores the fraction of the horizon each episode spends upright, and reports how much of the horizon the
schedule commanded the robot to move. Copy the shape, not the score.
"""
from __future__ import annotations

import numpy as np

from aicrowd_microduck.local.metrics import _f, alive_horizon, metric, qpos_horizon
from aicrowd_microduck.local.recorder import RolloutTrace

UPRIGHT_Z_M = 0.08          # trunk height above which the robot counts as upright (illustrative)


@metric("example_v2")
def example_v2(tr: RolloutTrace) -> dict:
    H = tr.max_steps                                   # the horizon, e.g. 500 for 10 s: the denominator
    alive = alive_horizon(tr)                          # [H, K]: True up to and including each env's done step
    qpos = qpos_horizon(tr)                            # [H, K, nq]: NaN wherever alive is False
    twist = tr.schedule("twist")                       # [H, K, 3]: every step's command, recorded or not
    upright = np.where(alive, qpos[..., 2] > UPRIGHT_Z_M, False)   # np.where, not *: NaN * 0 is NaN
    score = upright.sum(axis=0) / H                    # time after a fall counts as failed
    moving = np.linalg.norm(twist[..., :2], axis=-1) > 0.05
    return {"example_v2_score": _f(score.mean()), "example_v2_per_env": score.tolist(),
            "example_v2_commanded_moving_fraction": _f(moving.mean())}
