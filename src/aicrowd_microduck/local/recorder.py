# VENDORED from microduck-evaluator/src/mdeval/server/recorder.py by scripts/sync_from_evaluator.py.
# Do not edit by hand: change the evaluator and re-run the sync.
"""Rollout trace: everything recorded per step, saved as one ``trace.npz`` + ``manifest.json``.

The trace is the reproducibility record. ``qpos`` alone re-renders the video anywhere;
``actions`` + the manifest (seed, env cfg digest, image digest) re-simulate bit-exactly on
the same image/CPU architecture.

Rows are per control step for the whole batch, so envs that finished earlier keep getting rows: an
env's rows after its ``done_step`` belong to its auto-reset episode (mjlab resets inside ``step()``).
Every reader stops at or masks by ``done_step`` (``alive_mask``; ``metrics.alive_horizon`` over the
horizon). The trace ends ``TaskSpec.record_after_done_s`` after the last env's done step (v2: 0.5 s)
or at the horizon, so its length ``T`` is not a score denominator: v2 metrics use ``max_steps``.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from aicrowd_microduck.local.env_runner import StepRecord


@dataclass
class RolloutTrace:
    meta: dict[str, Any]
    obs: np.ndarray            # [T, K, obs_dim]
    actions: np.ndarray        # [T, K, act_dim]
    rewards: np.ndarray        # [T, K]
    reward_terms: np.ndarray   # [T, K, n_terms]
    qpos: np.ndarray           # [T, K, nq]
    qvel: np.ndarray           # [T, K, nv]
    done_step: np.ndarray      # [K] step index at which env finished (0-based, inclusive), -1 never
    commands: dict[str, np.ndarray] = field(default_factory=dict)      # name -> [T, K, d]
    term_flags: dict[str, np.ndarray] = field(default_factory=dict)    # name -> [T, K] bool
    step_timing: np.ndarray | None = None  # [T, 3]: t_recv, t_done, policy_wall_s (server clock, unix s)
    qpos0: np.ndarray | None = None        # [K, nq] state at reset (before any step)
    env_metrics: np.ndarray | None = None  # [T, K, n_metrics] the env's own metric terms (names: meta["metric_terms"])
    model_path: Path | None = None         # compiled MjModel (.mjb) of the task, for kinematics-based metrics; not saved in the npz
    qvel0: np.ndarray | None = None        # [K, nv] velocity at reset (before any step)
    # command name -> [K, H, dims] float32: each env's whole first-episode command schedule, read from the
    # env's scheduled command terms right after reset (H = the env's max episode length = max_steps).
    # Empty for envs without scheduled commands (v1) and for traces recorded before it existed.
    cmd_schedule: dict[str, np.ndarray] = field(default_factory=dict)

    @property
    def T(self) -> int:
        return int(self.obs.shape[0])

    @property
    def K(self) -> int:
        return int(self.obs.shape[1])

    @property
    def max_steps(self) -> int:
        """The episode horizon in control steps (manifest ``max_steps``). Score denominators use this, not ``T``."""
        return int(self.meta.get("max_steps", self.T))

    def schedule(self, name: str) -> np.ndarray:
        """The saved schedule of command ``name``, horizon-major: ``[max_steps, K, dims]`` (a view)."""
        if name not in self.cmd_schedule:
            raise KeyError(f"command {name!r} has no saved schedule (cmd_schedule has {sorted(self.cmd_schedule)}); "
                           "only scheduled (v2) command terms are saved, and only by evaluators that record cmd_schedule")
        s = self.cmd_schedule[name]
        H = self.max_steps
        if s.shape[1] < H:
            raise ValueError(f"schedule {name!r} covers {s.shape[1]} steps, shorter than the horizon {H}")
        return np.transpose(s, (1, 0, 2))[:H]

    def alive_mask(self) -> np.ndarray:
        """[T, K] True while the env is still running (inclusive of its terminal step)."""
        t = np.arange(self.T)[:, None]
        ds = np.where(self.done_step < 0, self.T - 1, self.done_step)[None, :]
        return t <= ds

    def save(self, directory: str | Path) -> None:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        arrays = dict(obs=self.obs, actions=self.actions, rewards=self.rewards, reward_terms=self.reward_terms,
                      qpos=self.qpos, qvel=self.qvel, done_step=self.done_step)
        if self.step_timing is not None:
            arrays["step_timing"] = self.step_timing
        if self.qpos0 is not None:
            arrays["qpos0"] = self.qpos0
        if self.qvel0 is not None:
            arrays["qvel0"] = self.qvel0
        for k, v in self.cmd_schedule.items():
            arrays[f"sched__{k}"] = v
        if self.env_metrics is not None:
            arrays["env_metrics"] = self.env_metrics
        for k, v in self.commands.items():
            arrays[f"cmd__{k}"] = v
        for k, v in self.term_flags.items():
            arrays[f"term__{k}"] = v
        np.savez_compressed(d / "trace.npz", **arrays)
        with open(d / "manifest.json", "w") as f:
            json.dump(self.meta, f, indent=1, default=str)

    @classmethod
    def load(cls, directory: str | Path, model_path: str | Path | None = None) -> "RolloutTrace":
        d = Path(directory)
        z = np.load(d / "trace.npz")
        meta = json.load(open(d / "manifest.json"))
        commands = {k[len("cmd__"):]: z[k] for k in z.files if k.startswith("cmd__")}
        term_flags = {k[len("term__"):]: z[k] for k in z.files if k.startswith("term__")}
        cmd_schedule = {k[len("sched__"):]: z[k] for k in z.files if k.startswith("sched__")}
        if model_path is None and meta.get("model_mjb"):
            cand = d.parent.parent / meta["model_mjb"]      # <run>/rollouts/<rid> -> <run>/assets/model_<task>_<seed>.mjb
            model_path = cand if cand.exists() else None
        return cls(meta=meta, obs=z["obs"], actions=z["actions"], rewards=z["rewards"], reward_terms=z["reward_terms"],
                   qpos=z["qpos"], qvel=z["qvel"], done_step=z["done_step"], commands=commands, term_flags=term_flags,
                   step_timing=z["step_timing"] if "step_timing" in z.files else None,
                   qpos0=z["qpos0"] if "qpos0" in z.files else None,
                   env_metrics=z["env_metrics"] if "env_metrics" in z.files else None,
                   model_path=Path(model_path) if model_path else None,
                   qvel0=z["qvel0"] if "qvel0" in z.files else None, cmd_schedule=cmd_schedule)


class RolloutRecorder:
    def __init__(self, meta: dict[str, Any], qpos0: np.ndarray | None = None, model_path: str | Path | None = None,
                 qvel0: np.ndarray | None = None, cmd_schedule: dict[str, np.ndarray] | None = None):
        self.meta = dict(meta)
        self.meta.setdefault("started_at", time.time())
        self.records: list[StepRecord] = []
        self.timing: list[tuple[float, float, float]] = []
        self.qpos0 = qpos0
        self.qvel0 = qvel0
        self.cmd_schedule = {k: np.asarray(v, dtype=np.float32) for k, v in (cmd_schedule or {}).items()}
        self.model_path = Path(model_path) if model_path else None

    def add(self, rec: StepRecord, t_recv: float, t_done: float, policy_wall_s: float) -> None:
        self.records.append(rec)
        self.timing.append((t_recv, t_done, policy_wall_s))

    def finish(self, done_step: np.ndarray, extra_meta: dict[str, Any] | None = None) -> RolloutTrace:
        rs = self.records
        if not rs:
            raise ValueError("no steps recorded")
        cmd_names = list(rs[0].commands.keys())
        term_names = list(rs[0].term_flags.keys())
        meta = dict(self.meta)
        meta["finished_at"] = time.time()
        meta["n_steps"] = len(rs)
        if extra_meta:
            meta.update(extra_meta)
        return RolloutTrace(
            meta=meta,
            obs=np.stack([r.obs for r in rs]), actions=np.stack([r.actions for r in rs]),
            rewards=np.stack([r.rewards for r in rs]), reward_terms=np.stack([r.reward_terms for r in rs]),
            qpos=np.stack([r.qpos for r in rs]), qvel=np.stack([r.qvel for r in rs]),
            done_step=np.asarray(done_step, dtype=np.int64),
            commands={n: np.stack([r.commands[n] for r in rs]) for n in cmd_names},
            term_flags={n: np.stack([r.term_flags[n] for r in rs]) for n in term_names},
            step_timing=np.asarray(self.timing, dtype=np.float64),
            qpos0=self.qpos0,
            env_metrics=np.stack([r.env_metrics for r in rs]) if rs[0].env_metrics is not None else None,
            model_path=self.model_path,
            qvel0=self.qvel0,
            cmd_schedule=dict(self.cmd_schedule),
        )
