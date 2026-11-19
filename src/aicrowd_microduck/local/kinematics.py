# VENDORED from microduck-evaluator/src/mdeval/server/kinematics.py by scripts/sync_from_evaluator.py.
# Do not edit by hand: change the evaluator and re-run the sync.
"""Offline kinematics from a recorded trace: foot contacts and site positions re-derived from ``qpos``.

Metrics that need "is the foot on the ground" or "how high is the swing foot" do not read the simulator's
sensors at record time; they re-run MuJoCo's kinematics + collision detection over the recorded ``qpos``
with the rollout's compiled model (``assets/model_<task>_<seed>.mjb``, saved by the env process). Keyed by
seed as well as task because a procedurally generated environment compiles differently per seed. That
keeps the trace format env-agnostic, works offline (``mdeval metrics``, ``mdeval rescore``) and on any
platform: collision detection on identical geometry and identical joint positions gives the same contact
set, so these metrics are as reproducible as the qpos they come from.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from aicrowd_microduck.local.recorder import RolloutTrace

ROBOT_PREFIX = "robot/"
FOOT_GEOMS = ("left_foot_collision", "right_foot_collision")      # order: left, right
FOOT_SITES = ("left_foot", "right_foot")


class Kinematics:
    """Lazily computed, cached on the trace as ``trace._kin``.

    ``foot_contact``: bool [T, K, 2] — each foot collision geom touches something that is not the robot.
    ``foot_pos``: float [T, K, 2, 3] — world position of the foot sites (left, right).
    Steps after an env's terminal step are False / NaN.
    """

    def __init__(self, model_path: str | Path, trace: RolloutTrace):
        import mujoco

        self.model = mujoco.MjModel.from_binary_path(str(model_path))
        m = self.model
        self.trace = trace
        names = {mujoco.mjtObj.mjOBJ_GEOM: [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i) or "" for i in range(m.ngeom)],
                 mujoco.mjtObj.mjOBJ_SITE: [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_SITE, i) or "" for i in range(m.nsite)],
                 mujoco.mjtObj.mjOBJ_BODY: [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i) or "" for i in range(m.nbody)]}
        robot_bodies = np.array([n.startswith(ROBOT_PREFIX) for n in names[mujoco.mjtObj.mjOBJ_BODY]])
        self.robot_geom = robot_bodies[m.geom_bodyid]                       # [ngeom] bool
        self.foot_geom_ids = np.array([self._find(names[mujoco.mjtObj.mjOBJ_GEOM], g) for g in FOOT_GEOMS])
        self.foot_site_ids = np.array([self._find(names[mujoco.mjtObj.mjOBJ_SITE], s) for s in FOOT_SITES])
        self._foot_contact: np.ndarray | None = None
        self._foot_pos: np.ndarray | None = None

    @staticmethod
    def _find(names: list[str], short: str) -> int:
        for cand in (ROBOT_PREFIX + short, short):
            if cand in names:
                return names.index(cand)
        raise KeyError(f"model has no element named {short!r} (or {ROBOT_PREFIX + short!r})")

    def _compute(self) -> None:
        import mujoco

        m, tr = self.model, self.trace
        d = mujoco.MjData(m)
        T, K = tr.T, tr.K
        alive = tr.alive_mask()
        contact = np.zeros((T, K, 2), dtype=bool)
        pos = np.full((T, K, 2, 3), np.nan, dtype=np.float32)
        geom_is_foot = np.full(m.ngeom, -1, dtype=np.int64)
        geom_is_foot[self.foot_geom_ids] = np.arange(2)
        for k in range(K):
            last = int(np.argmax(~alive[:, k])) - 1 if not alive[:, k].all() else T - 1
            for t in range(last + 1):
                d.qpos[:] = tr.qpos[t, k]
                mujoco.mj_kinematics(m, d)
                mujoco.mj_collision(m, d)
                pos[t, k] = d.site_xpos[self.foot_site_ids]
                for c in range(d.ncon):
                    g1, g2 = int(d.contact.geom1[c]), int(d.contact.geom2[c])
                    for a, b in ((g1, g2), (g2, g1)):
                        f = geom_is_foot[a]
                        if f >= 0 and not self.robot_geom[b]:
                            contact[t, k, f] = True
        self._foot_contact, self._foot_pos = contact, pos

    @property
    def foot_contact(self) -> np.ndarray:
        if self._foot_contact is None:
            self._compute()
        assert self._foot_contact is not None
        return self._foot_contact

    @property
    def foot_pos(self) -> np.ndarray:
        if self._foot_pos is None:
            self._compute()
        assert self._foot_pos is not None
        return self._foot_pos


def kinematics_for(trace: RolloutTrace) -> Kinematics:
    kin = getattr(trace, "_kin", None)
    if kin is None:
        if trace.model_path is None or not Path(trace.model_path).exists():
            raise FileNotFoundError(
                "this metric needs the task's compiled model: expected manifest 'model_mjb' to resolve to "
                f"<run>/assets/model_<task>_<seed>.mjb (got {trace.model_path!r}); pass model_path to RolloutTrace.load")
        kin = Kinematics(trace.model_path, trace)
        trace._kin = kin  # type: ignore[attr-defined]
    return kin
