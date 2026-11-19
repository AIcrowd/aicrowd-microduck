# VENDORED from microduck-evaluator/src/mdeval/server/env_runner.py by scripts/sync_from_evaluator.py.
# Do not edit by hand: change the evaluator and re-run the sync.
"""Thin wrapper around an mjlab ``ManagerBasedRlEnv`` running on CPU for one rollout batch.

One instance = one task, ``num_envs`` parallel episodes, one seed. It exposes numpy in/out,
tracks per-env "done" stickily (the env auto-resets finished episodes; we keep stepping them
but ignore everything after the first done), and hands out the raw state needed for
recording (qpos, qvel, commands, reward terms, termination flags), plus the state at reset:
``qpos0``/``qvel0`` and every command term's full episode schedule (``cmd_schedule``).

The rollout ends (``all_done``) once every env is ``TaskSpec.post_done_steps`` past its done step,
or at the horizon (``recording_done``). After its done step an env is in its auto-reset episode.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from aicrowd_microduck.local.tasks import TaskSpec, apply_eval_overrides, describe_cfg

log = logging.getLogger(__name__)


def _ensure_tasks_registered(extra_module: str | None = None) -> None:
    """mjlab discovers task plugins through the ``mjlab.tasks`` entry-point group only on some code
    paths (``list-envs``); importing the microduck task package registers everything explicitly.
    ``extra_module`` (``TaskSpec.env_module``) lets a task YAML name any importable module that registers
    its env id — the drop-in path for envs that live outside the plugin package."""
    import importlib
    from importlib import metadata

    for ep in metadata.entry_points(group="mjlab.tasks"):
        try:
            importlib.import_module(ep.value.split(":")[0])
        except Exception as e:  # noqa: BLE001
            log.warning("could not import task plugin %s: %r", ep.value, e)
    for mod in ("mjlab.tasks", "mjlab_microduck.tasks"):
        try:
            importlib.import_module(mod)
        except Exception as e:  # noqa: BLE001
            log.debug("optional task module %s not importable: %r", mod, e)
    if extra_module:
        importlib.import_module(extra_module)      # a typo here must fail the build loudly


def recording_done(done_step: np.ndarray, step_count: int, max_steps: int, post_done_steps: int) -> bool:
    """True once every env has been recorded ``post_done_steps`` steps past its done step, or at the horizon.

    ``done_step`` holds 1-based step counts (-1 = still running). With ``post_done_steps`` 0 this is the v1
    rule: the rollout ends the step the last env finishes. v2 tasks keep 0.5 s (25 steps) after the last fall.
    """
    if step_count >= max_steps:
        return True
    if (done_step < 0).any():
        return False
    return step_count >= int(done_step.max()) + post_done_steps


@dataclass
class StepRecord:
    step: int
    obs: np.ndarray          # [K, obs_dim] observation the policy acted on
    actions: np.ndarray      # [K, act_dim] raw actions
    rewards: np.ndarray      # [K]
    reward_terms: np.ndarray  # [K, n_terms] per-step term values (weighted, per mjlab)
    qpos: np.ndarray         # [K, nq] after the step
    qvel: np.ndarray         # [K, nv]
    commands: dict[str, np.ndarray]  # command name -> [K, dim] (pre-step)
    newly_done: np.ndarray   # [K] bool: finished on this step
    term_flags: dict[str, np.ndarray]  # term name -> [K] bool at this step
    env_metrics: np.ndarray | None = None  # [K, n_metrics] the env's own MetricsManager terms this step
    t_wall: float = field(default_factory=time.time)


class EnvRunner:
    def __init__(self, spec: TaskSpec, seed: int, num_envs: int | None = None, device: str = "cpu"):
        import torch
        from mjlab.envs import ManagerBasedRlEnv
        from mjlab.tasks.registry import load_env_cfg

        _ensure_tasks_registered(spec.env_module)

        self.spec = spec
        self.seed = seed
        self.num_envs = num_envs or spec.num_envs
        cfg = load_env_cfg(spec.env_id, play=False)
        cfg = apply_eval_overrides(cfg, spec, self.num_envs, seed)
        self.cfg_description = describe_cfg(cfg)
        t0 = time.perf_counter()
        torch.manual_seed(seed)
        self.env = ManagerBasedRlEnv(cfg=cfg, device=device)
        self.build_s = time.perf_counter() - t0
        self._torch = torch
        assert abs(self.env.step_dt - spec.control_dt) < 1e-9, (self.env.step_dt, spec.control_dt)
        self.max_steps = spec.max_steps
        self.post_done_steps = spec.post_done_steps
        self.obs_dim = int(self.env.observation_manager.group_obs_dim["actor"][0])
        self.act_dim = int(self.env.action_manager.total_action_dim)
        self.reward_term_names = list(self.env.reward_manager.active_terms)
        self.term_names = list(self.env.termination_manager.active_terms)
        self.command_names = list(getattr(self.env.command_manager, "_terms", {}).keys())
        # the env's own metric terms (mjlab MetricsManager; e.g. the trampoline's takeoff_rate) are recorded per step
        self.metric_term_names = list(getattr(self.env.metrics_manager, "active_terms", []) or [])
        self.mj_model = self.env.sim.mj_model
        self.done = np.zeros(self.num_envs, dtype=bool)
        self.done_step = np.full(self.num_envs, -1, dtype=np.int64)
        self.step_count = 0
        self.last_obs: np.ndarray | None = None
        self.qvel0: np.ndarray | None = None
        self.cmd_schedule: dict[str, np.ndarray] = {}
        # mjlab auto-resets finished envs INSIDE step(), so the state read after step() would be the
        # post-reset pose (review F4). Wrap _reset_idx to snapshot qpos/qvel of the envs about to be
        # reset; step() splices those terminal states into the record.
        self._terminal: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        orig_reset_idx = self.env._reset_idx

        def _reset_idx_capturing(env_ids=None):
            if env_ids is not None and self.step_count > 0:
                ids = env_ids.detach().cpu().numpy().tolist() if hasattr(env_ids, "detach") else list(env_ids)
                qpos = self.qpos(); qvel = self.qvel()
                for i in ids:
                    self._terminal[int(i)] = (qpos[int(i)].copy(), qvel[int(i)].copy())
            return orig_reset_idx(env_ids)

        self.env._reset_idx = _reset_idx_capturing
        log.info("env built task=%s env=%s K=%d seed=%d build_s=%.2f obs_dim=%d act_dim=%d",
                 spec.task_id, spec.env_id, self.num_envs, seed, self.build_s, self.obs_dim, self.act_dim)

    # ------------------------------------------------------------------ state access
    def _np(self, t) -> np.ndarray:
        return t.detach().cpu().numpy().copy()

    def qpos(self) -> np.ndarray:
        return self._np(self.env.sim.data.qpos)

    def qvel(self) -> np.ndarray:
        return self._np(self.env.sim.data.qvel)

    def env_origins(self) -> np.ndarray | None:
        """``[K, 3]`` each env's origin in the world (``env.scene.env_origins``: the grid, or the terrain's
        sub-terrain origins). Positions are world coordinates; a task may measure from the origin (the
        trampoline's 1.5 m edge). Recorded for the render only; ``None`` if the scene does not expose it."""
        try:
            return self._np(self.env.scene.env_origins).astype(np.float64)
        except Exception:  # noqa: BLE001 -- provenance for rendering, never a reason to fail a build
            return None

    def commands(self) -> dict[str, np.ndarray]:
        return {name: self._np(self.env.command_manager.get_command(name)) for name in self.command_names}

    def command_schedules(self) -> dict[str, np.ndarray]:
        """Each scheduled command term's whole current episode, ``[K, max_episode_steps, dims]`` float32.

        v2 command terms draw every value from a per-env stream fixed at reset and expose it through
        ``full_schedule()`` (a copy); terms without it (every v1 env) are skipped. Read once, right after
        ``env.reset(seed=...)``, so a score knows each window's command without recording past a fall."""
        out: dict[str, np.ndarray] = {}
        for name in self.command_names:
            fs = getattr(self.env.command_manager.get_term(name), "full_schedule", None)
            if callable(fs):
                out[name] = self._np(fs()).astype(np.float32)
        return out

    def term_flags(self) -> dict[str, np.ndarray]:
        return {n: self._np(self.env.termination_manager.get_term(n)).astype(bool) for n in self.term_names}

    # ---------------------------------------------------------------------- control
    def reset(self) -> np.ndarray:
        obs, _ = self.env.reset(seed=self.seed)
        self.last_obs = self._np(obs["actor"]).astype(np.float32)
        self.qvel0 = self.qvel()
        self.cmd_schedule = self.command_schedules()
        self.done[:] = False
        self.done_step[:] = -1
        self.step_count = 0
        return self.last_obs

    def step(self, actions: np.ndarray) -> StepRecord:
        if actions.shape != (self.num_envs, self.act_dim):
            raise ValueError(f"actions shape {actions.shape} != {(self.num_envs, self.act_dim)}")
        actions = np.asarray(actions, dtype=np.float32)
        if not np.all(np.isfinite(actions)):
            raise FloatingPointError("non-finite action")
        obs_before = self.last_obs
        commands = self.commands()
        self._terminal.clear()
        self.step_count += 1                      # so the capture hook knows this is not the initial reset
        obs, rew, term, trunc, _ = self.env.step(self._torch.from_numpy(actions))
        done_now = (self._np(term) | self._np(trunc)).astype(bool)
        term_flags = self.term_flags()
        reward_terms = self._np(self.env.reward_manager._step_reward)  # [K, n_terms]
        env_metrics = (self._np(self.env.metrics_manager._step_values).astype(np.float32) if self.metric_term_names
                       else np.zeros((self.num_envs, 0), dtype=np.float32))
        newly = done_now & ~self.done
        if self.step_count >= self.max_steps:
            newly |= ~self.done
        self.done_step[newly] = self.step_count
        self.done |= newly
        self.last_obs = self._np(obs["actor"]).astype(np.float32)
        qpos, qvel = self.qpos(), self.qvel()
        for i, (tq, tv) in self._terminal.items():      # terminal state, not the post-reset pose
            qpos[i] = tq
            qvel[i] = tv
        return StepRecord(
            step=self.step_count - 1, obs=obs_before, actions=actions, rewards=self._np(rew).astype(np.float32),
            reward_terms=reward_terms.astype(np.float32), qpos=qpos, qvel=qvel, commands=commands,
            newly_done=newly, term_flags=term_flags, env_metrics=env_metrics,
        )

    @property
    def all_done(self) -> bool:
        return recording_done(self.done_step, self.step_count, self.max_steps, self.post_done_steps)

    def close(self) -> None:
        try:
            self.env.close()
        except Exception:  # noqa: BLE001
            pass

    def describe(self) -> dict[str, Any]:
        return {
            "task_id": self.spec.task_id, "env_id": self.spec.env_id, "seed": self.seed,
            "num_envs": self.num_envs, "max_steps": self.max_steps, "control_dt": self.env.step_dt,
            "physics_dt": self.env.physics_dt, "obs_dim": self.obs_dim, "act_dim": self.act_dim,
            "reward_terms": self.reward_term_names, "termination_terms": self.term_names, "metric_terms": self.metric_term_names,
            "commands": self.command_names, "nq": int(self.mj_model.nq), "nv": int(self.mj_model.nv),
            "build_s": self.build_s, "env_cfg": self.cfg_description,
        }
