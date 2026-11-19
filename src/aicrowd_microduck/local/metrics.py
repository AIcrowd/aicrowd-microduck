# VENDORED from microduck-evaluator/src/mdeval/server/metrics.py by scripts/sync_from_evaluator.py.
# Do not edit by hand: change the evaluator and re-run the sync.
"""Metrics computed from a :class:`RolloutTrace` — pure numpy, so they can be recomputed offline.

Each metric function takes the trace and returns a flat ``{name: value}`` dict where values are
floats (rollout aggregates over the K envs) or lists (per-env). Register new metrics with
``@metric("name")`` and list them in the task YAML. ``primary_metric`` in the task YAML must be
one of the flat keys produced here.

v2 metrics live one module per family in ``server/metrics_v2/`` (imported at the end of this module; see
its ``__init__``) and use the horizon helpers below: they score over ``max_steps``, never the trace length.
"""
from __future__ import annotations

from typing import Callable

import numpy as np

from aicrowd_microduck.local.recorder import RolloutTrace

_REGISTRY: dict[str, Callable[[RolloutTrace], dict]] = {}


def metric(name: str):
    """Register ``fn`` under ``name``. Names are global across this module and every metrics_v2 plugin:
    registering a different function under a taken name raises, so one family cannot silently replace another's."""
    def deco(fn):
        prev = _REGISTRY.get(name)
        if prev is not None and (prev.__module__, prev.__qualname__) != (fn.__module__, fn.__qualname__):
            raise ValueError(f"metric {name!r} is already registered by {prev.__module__}.{prev.__qualname__}")
        _REGISTRY[name] = fn
        return fn
    return deco


def load_plugins(package: str = "aicrowd_microduck.local.metrics_v2") -> list[str]:
    """Import every module of ``package`` whose name does not start with ``_`` (sorted; subpackages skipped), so
    their ``@metric`` functions register. Returns the imported module names. Import errors propagate: a broken
    metric module must stop the server at start, not score zero later."""
    import importlib
    import pkgutil

    pkg = importlib.import_module(package)
    names = []
    for info in sorted(pkgutil.iter_modules(pkg.__path__), key=lambda i: i.name):
        if info.name.startswith("_") or info.ispkg:
            continue
        importlib.import_module(f"{package}.{info.name}")
        names.append(f"{package}.{info.name}")
    return names


def available() -> list[str]:
    return sorted(_REGISTRY)


def compute(trace: RolloutTrace, names: list[str]) -> dict:
    out: dict = {}
    for n in names:
        if n not in _REGISTRY:
            raise KeyError(f"unknown metric {n!r}; available: {available()}")
        out.update(_REGISTRY[n](trace))
    return out


def _f(x) -> float:
    return float(np.asarray(x, dtype=np.float64))


# ------------------------------------------------------------------- horizon helpers (v2)
# A trace stops ``record_after_done_s`` after the last env's done step, or at the horizon, and an env's rows after
# its done step are its auto-reset episode. v2 scores cover the whole horizon ``tr.max_steps`` and count time after
# a fall as failed, so they work on [max_steps, K, ...] arrays built by these helpers.

def alive_horizon(tr: RolloutTrace) -> np.ndarray:
    """[max_steps, K] bool: True up to and including each env's done step; False after it and past the trace end."""
    H = tr.max_steps
    t = np.arange(H)[:, None]
    last = np.where(tr.done_step < 0, tr.T - 1, tr.done_step)[None, :]
    return (t <= last) & (t < tr.T)


def pad_horizon(tr: RolloutTrace, x: np.ndarray, fill=np.nan, mask_dead: bool = True) -> np.ndarray:
    """``x`` [T, K, ...] -> [max_steps, K, ...], ``fill`` past the trace end and, with ``mask_dead`` (default),
    wherever ``alive_horizon`` is False, so the auto-reset episode never leaks into a score and the result does not
    depend on when recording stopped. A float array keeps its dtype; pass a non-NaN ``fill`` for bool or int."""
    x = np.asarray(x)
    H = tr.max_steps
    nan_fill = isinstance(fill, float) and np.isnan(fill)
    dtype = np.float64 if nan_fill and not np.issubdtype(x.dtype, np.floating) else x.dtype
    out = np.full((H,) + x.shape[1:], fill, dtype=dtype)
    n = min(H, x.shape[0])
    out[:n] = x[:n]
    if mask_dead:
        dead = ~alive_horizon(tr)
        out[dead] = fill
    return out


def qpos_horizon(tr: RolloutTrace) -> np.ndarray:
    """[max_steps, K, nq] recorded qpos, NaN wherever ``alive_horizon`` is False."""
    return pad_horizon(tr, tr.qpos)


def qvel_horizon(tr: RolloutTrace) -> np.ndarray:
    """[max_steps, K, nv] recorded qvel, NaN wherever ``alive_horizon`` is False."""
    return pad_horizon(tr, tr.qvel)


def _quat_rotate_inverse(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """q: [..., 4] (w, x, y, z); v: [..., 3]. Rotate v by q^-1 (world -> body)."""
    w = q[..., :1]
    xyz = q[..., 1:]
    t = np.cross(xyz, v) * 2.0
    return v - w * t + np.cross(xyz, t)


# --------------------------------------------------------------------------- generic
@metric("episode_length")
def episode_length(tr: RolloutTrace) -> dict:
    T = tr.T
    steps = np.where(tr.done_step < 0, T, tr.done_step + 1)
    return {"episode_length_steps": steps.tolist(), "episode_length_mean": _f(steps.mean()),
            "episode_length_fraction": _f(steps.mean() / max(1, tr.meta.get("max_steps", T)))}


@metric("survival")
def survival(tr: RolloutTrace) -> dict:
    """An env 'survived' if it never hit a non-timeout termination."""
    T = tr.T
    max_steps = int(tr.meta.get("max_steps", T))
    fatal = np.zeros(tr.K, dtype=bool)
    alive = tr.alive_mask()
    for name, flags in tr.term_flags.items():
        if name == "time_out":
            continue
        fatal |= (flags & alive).any(axis=0)
    survived = ~fatal & ((tr.done_step < 0) | (tr.done_step + 1 >= max_steps))
    return {"survived": survived.astype(int).tolist(), "survival_fraction": _f(survived.mean())}


@metric("termination_reasons")
def termination_reasons(tr: RolloutTrace) -> dict:
    out = {}
    for name, flags in tr.term_flags.items():
        alive = tr.alive_mask()
        out[f"terminated_by/{name}"] = _f((flags & alive).any(axis=0).mean())
    return out


@metric("reward_terms")
def reward_terms(tr: RolloutTrace) -> dict:
    names = tr.meta.get("reward_terms") or [f"term{i}" for i in range(tr.reward_terms.shape[-1])]
    alive = tr.alive_mask()[..., None]
    # mjlab's per-step term values are weight*raw (no dt); scale by control_dt so the term sums
    # add up to the dt-scaled episode return the env reports (reward_total).
    dt = float(tr.meta.get("control_dt", 0.02))
    sums = (tr.reward_terms * alive).sum(axis=0) * dt     # [K, n]
    total = (tr.rewards * tr.alive_mask()).sum(axis=0)    # [K]
    out = {"reward_total_mean": _f(total.mean()), "reward_total_per_env": total.tolist()}
    for i, n in enumerate(names):
        out[f"reward/{n}"] = _f(sums[:, i].mean())
    return out


@metric("trunk_height")
def trunk_height(tr: RolloutTrace) -> dict:
    z = tr.qpos[..., 2]                 # [T, K] free-joint z (relative to world; terrain may offset)
    z0 = tr.qpos0[:, 2] if tr.qpos0 is not None else z[0]
    rel = z - z0[None, :]
    alive = tr.alive_mask()
    n = np.maximum(alive.sum(axis=0), 1)
    return {"trunk_z_mean": _f(((z * alive).sum(axis=0) / n).mean()),
            "trunk_z_rel_min": _f((np.where(alive, rel, np.inf)).min(axis=0).mean()),
            "trunk_z_final_rel": _f(np.array([rel[min(max(d, 0), tr.T - 1), k] if d >= 0 else rel[-1, k]
                                              for k, d in enumerate(tr.done_step)]).mean())}


# ------------------------------------------------------------------ velocity tasks
@metric("velocity_tracking")
def velocity_tracking(tr: RolloutTrace) -> dict:
    """Body-frame planar velocity + yaw-rate error vs the twist command, over alive steps.

    Uses the free-joint state: v_world = qvel[0:3], omega_body = qvel[3:6], quat = qpos[3:7].
    velocity_tracking_score = mean exp(-err_lin^2 / 0.1) * exp(-err_ang^2 / 0.5), the shape of
    the env's own tracking rewards, weighted by survival fraction of steps.
    """
    if "twist" not in tr.commands:
        return {"velocity_tracking_score": 0.0}
    cmd = tr.commands["twist"]                       # [T, K, 3]
    q = tr.qpos[..., 3:7]
    v_w = tr.qvel[..., 0:3]
    v_b = _quat_rotate_inverse(q, v_w)
    wz = tr.qvel[..., 5]
    alive = tr.alive_mask()
    err_lin = np.linalg.norm(v_b[..., :2] - cmd[..., :2], axis=-1)
    err_ang = np.abs(wz - cmd[..., 2])
    n = np.maximum(alive.sum(axis=0), 1)
    lin = (err_lin * alive).sum(axis=0) / n
    ang = (err_ang * alive).sum(axis=0) / n
    per_step = np.exp(-(err_lin ** 2) / 0.1) * np.exp(-(err_ang ** 2) / 0.5)
    score_env = (per_step * alive).sum(axis=0) / tr.T    # dead steps count as zero
    return {"lin_vel_error_mean": _f(lin.mean()), "ang_vel_error_mean": _f(ang.mean()),
            "velocity_tracking_score": _f(score_env.mean()), "velocity_tracking_score_per_env": score_env.tolist(),
            "distance_travelled_mean": _f(np.linalg.norm(
                tr.qpos[np.clip(np.where(tr.done_step < 0, tr.T - 1, tr.done_step), 0, tr.T - 1), np.arange(tr.K), :2]
                - (tr.qpos0[:, :2] if tr.qpos0 is not None else tr.qpos[0, :, :2]), axis=-1).mean())}


# ---------------------------------------------------------------- roller crouch task
@metric("crouch_profile")
def crouch_profile(tr: RolloutTrace) -> dict:
    """Roller crouch-glide: phase comes from cmd twist = [cos 2pi phi, sin 2pi phi, 0].

    Trapezoid target: descend phi in [0, .10], hold [.10, .50], rise [.50, .60], stand [.60, 1].
    crouch_score = survival * (hold-phase height drop achieved) * (stood back up at the end).
    Height drop is normalised by 0.05 m (HOME ~0.12-0.14 m trunk, crouch ~0.07 m); standing back up is
    credited linearly down to a 6 cm deviation from the start height.
    """
    if "twist" not in tr.commands:
        return {"crouch_score": 0.0}
    c = tr.commands["twist"]
    phase = (np.arctan2(c[..., 1], c[..., 0]) / (2 * np.pi)) % 1.0    # [T, K]
    z = tr.qpos[..., 2]
    z0 = tr.qpos0[:, 2] if tr.qpos0 is not None else z[0]
    drop = z0[None, :] - z
    alive = tr.alive_mask()
    hold = (phase >= 0.15) & (phase <= 0.45) & alive
    stand = (phase >= 0.70) & alive
    n_hold = np.maximum(hold.sum(axis=0), 1)
    n_stand = np.maximum(stand.sum(axis=0), 1)
    hold_drop = (drop * hold).sum(axis=0) / n_hold                      # [K] mean drop while holding
    stand_drop = (np.abs(drop) * stand).sum(axis=0) / n_stand           # [K] |deviation| when standing again
    survived = np.array(survival(tr)["survived"], dtype=float)
    crouch_frac = np.clip(hold_drop / 0.05, 0.0, 1.0)
    stand_frac = np.clip(1.0 - stand_drop / 0.06, 0.0, 1.0)   # 6 cm off the start height = no credit for standing
    score = survived * crouch_frac * np.where(stand.sum(axis=0) > 0, stand_frac, 0.0)
    return {"hold_phase_height_drop_mean": _f(hold_drop.mean()), "stand_phase_height_dev_mean": _f(stand_drop.mean()),
            "crouch_score": _f(score.mean()), "crouch_score_per_env": score.tolist(),
            "phase_cycles_completed_mean": _f(((phase[1:] < phase[:-1]) & alive[1:]).sum(axis=0).mean())}


# ------------------------------------------------------------- the env's own metric terms
@metric("env_metrics")
def env_metrics(tr: RolloutTrace) -> dict:
    """Per-step values of the env's mjlab ``MetricsManager`` terms (e.g. the trampoline env's
    ``takeoff_rate``): mean over alive steps and episode sum, averaged over envs."""
    names = tr.meta.get("metric_terms") or []
    if tr.env_metrics is None or not names or tr.env_metrics.shape[-1] != len(names):
        return {}
    alive = tr.alive_mask()[..., None]
    n = np.maximum(alive.sum(axis=0), 1)
    total = (tr.env_metrics * alive).sum(axis=0)            # [K, M]
    mean = total / n
    out = {}
    for i, name in enumerate(names):
        out[f"env_metric/{name}_mean"] = _f(mean[:, i].mean())
        out[f"env_metric/{name}_sum"] = _f(total[:, i].mean())
    return out


# ---------------------------------------------------------------- trampoline jump task
TRAMPOLINE_SETTLE_S = 1.0       # takeoffs before this are the spawn transient, not bouncing


def count_takeoffs(contact_any: np.ndarray, vz: np.ndarray, alive: np.ndarray, min_upward_speed: float = 0.15) -> np.ndarray:
    """[T, K] bool takeoff events: both feet leave the surface while the trunk moves up faster than
    ``min_upward_speed`` (the training env's ``bounce_takeoff`` definition; a topple does not count).
    The step before the first is taken as "in contact"."""
    prev = np.ones_like(contact_any)
    prev[1:] = contact_any[:-1]
    return prev & ~contact_any & (vz > min_upward_speed) & alive


@metric("trampoline_bounce")
def trampoline_bounce(tr: RolloutTrace) -> dict:
    """Bounce count (primary), airborne fraction and apex rise, from re-derived foot contacts (kinematics.py).

    ``bounce_count`` is the number of takeoffs per episode after a 1 s settle window — physics rate-limits it
    (you must land before taking off again), so one violent launch scores 1 and sustained bouncing scores many;
    a fall ends the episode and therefore the count. ``apex_rise_m`` = highest trunk z minus the median trunk z
    while in contact (the surface's rest height under load, which the spawn height is not).
    """
    from aicrowd_microduck.local.kinematics import kinematics_for

    kin = kinematics_for(tr)
    alive = tr.alive_mask()
    contact = kin.foot_contact.any(axis=-1)                    # [T, K]
    takeoffs = count_takeoffs(contact, tr.qvel[..., 2], alive)
    # the spawn drop is thrown back by the mat once (~0.3 s in): a policy that just stands would score 1 —
    # takeoffs inside the settle window do not count (the env's reward counts them; the competition does not)
    dt = float(tr.meta.get("control_dt", 0.02))
    settle = int(round(TRAMPOLINE_SETTLE_S / dt))
    n_take_raw = takeoffs.sum(axis=0)
    n_take = takeoffs[settle:].sum(axis=0)
    n_alive = np.maximum(alive.sum(axis=0), 1)
    airborne_frac = ((~contact) & alive).sum(axis=0) / n_alive
    z = tr.qpos[..., 2]
    rest = np.array([np.median(z[contact[:, k] & alive[:, k], k]) if (contact[:, k] & alive[:, k]).any() else z[0, k]
                     for k in range(tr.K)])
    apex = np.where(alive, z, -np.inf).max(axis=0) - rest
    return {"bounce_count_mean": _f(n_take.mean()), "bounce_count_per_env": n_take.astype(int).tolist(),
            "bounce_count_raw_mean": _f(n_take_raw.mean()),
            "airborne_fraction": _f(airborne_frac.mean()), "apex_rise_m_mean": _f(apex.mean()), "apex_rise_m_max": _f(apex.max()),
            "takeoffs_per_10s": _f((n_take / np.maximum(n_alive - settle, 1) / dt * 10.0).mean())}


# ----------------------------------------------------------------- flamingo switch task
FLAMINGO_GRACE_S = 1.5          # the env's RAMP_S: a stand<->one-foot transition takes this long
FLAMINGO_CLEARANCE_M = 0.03     # swing foot at least this high above the stance foot = "lifted"


def latched_side(side_obs: np.ndarray) -> np.ndarray:
    """[T, K] observed side slot (+1 / -1 / 0 while standing) -> [T, K] in {-1, +1}: the last non-zero side,
    leading zeros back-filled with the first non-zero side, +1 if the episode never names one."""
    s = np.where(side_obs > 0.5, 1.0, np.where(side_obs < -0.5, -1.0, 0.0))
    for t in range(1, s.shape[0]):
        s[t] = np.where(s[t] == 0.0, s[t - 1], s[t])
    for k in range(s.shape[1]):
        nz = np.flatnonzero(s[:, k])
        if len(nz) == 0:
            s[:, k] = 1.0
        elif nz[0] > 0:
            s[:nz[0], k] = s[nz[0], k]
    return s


def flamingo_success(flag: np.ndarray, side: np.ndarray, left_contact: np.ndarray, right_contact: np.ndarray,
                     left_z: np.ndarray, right_z: np.ndarray, alive: np.ndarray, grace_steps: int,
                     clearance: float = FLAMINGO_CLEARANCE_M) -> dict[str, np.ndarray]:
    """Per-step ([T, K]) verdicts. flag > 0.5 asks for single support on ``side`` (+1 = right foot down):
    stance foot in contact, swing foot off the ground and ``clearance`` above it; flag = 0 asks for both feet
    down. Steps within ``grace_steps`` after a flag change (and after a flag=1 spawn) are not evaluated."""
    up = flag > 0.5
    change = np.zeros_like(up)
    change[0] = up[0]
    change[1:] = up[1:] != up[:-1]
    T = up.shape[0]
    last_change = np.where(change, np.arange(T)[:, None], -10 ** 9)
    last_change = np.maximum.accumulate(last_change, axis=0)
    in_grace = (np.arange(T)[:, None] - last_change) < grace_steps
    right_stance = side > 0
    stance_c = np.where(right_stance, right_contact, left_contact)
    swing_c = np.where(right_stance, left_contact, right_contact)
    swing_clear = np.where(right_stance, left_z - right_z, right_z - left_z)
    single = stance_c & ~swing_c & (swing_clear >= clearance)
    double = left_contact & right_contact
    success = np.where(up, single, double) & alive
    return {"success": success, "evaluated": ~in_grace, "in_grace": in_grace, "up": up, "change": change,
            "single": single & alive, "double": double & alive, "swing_clear": swing_clear}


@metric("flamingo_switch")
def flamingo_switch(tr: RolloutTrace) -> dict:
    """Flamingo cycle: twist command = [flag, side, 0]. Score = fraction of evaluated steps (outside the 1.5 s
    transition grace after each flag change) on which the commanded support is achieved — one foot lifted
    >= 3 cm with the other planted when flag = 1, both feet planted when flag = 0. Steps after a fall count
    as failures. Foot contacts and heights are re-derived from qpos (kinematics.py)."""
    from aicrowd_microduck.local.kinematics import kinematics_for

    if "twist" not in tr.commands:
        return {"flamingo_score": 0.0}
    kin = kinematics_for(tr)
    cmd = tr.commands["twist"]
    alive = tr.alive_mask()
    dt = float(tr.meta.get("control_dt", 0.02))
    fc, fp = kin.foot_contact, kin.foot_pos
    v = flamingo_success(cmd[..., 0], latched_side(cmd[..., 1]), fc[..., 0], fc[..., 1], fp[..., 0, 2], fp[..., 1, 2],
                         alive, grace_steps=int(round(FLAMINGO_GRACE_S / dt)))
    ev = v["evaluated"]
    n_ev = np.maximum(ev.sum(axis=0), 1)
    score = (v["success"] & ev).sum(axis=0) / n_ev                              # dead steps = failures
    up_ev, down_ev = ev & v["up"], ev & ~v["up"]
    single_frac = (v["success"] & up_ev).sum(axis=0) / np.maximum(up_ev.sum(axis=0), 1)
    double_frac = (v["success"] & down_ev).sum(axis=0) / np.maximum(down_ev.sum(axis=0), 1)
    lifted = v["single"] & v["up"]
    clear = np.where(lifted, v["swing_clear"], np.nan)
    return {"flamingo_score": _f(score.mean()), "flamingo_score_per_env": score.tolist(),
            "single_support_fraction": _f(single_frac.mean()), "double_support_fraction": _f(double_frac.mean()),
            "n_switches_mean": _f(v["change"].sum(axis=0).mean()),
            "commanded_one_foot_fraction": _f(up_ev.sum(axis=0).mean() / n_ev.mean()),
            "swing_clearance_m_mean": _f(np.nanmean(clear)) if np.isfinite(clear).any() else 0.0}


# ------------------------------------------------------------------ v2 plugins (one module per family)
load_plugins()
