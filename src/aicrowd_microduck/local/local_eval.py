# VENDORED from microduck-evaluator/src/mdeval/server/local_eval.py by scripts/sync_from_evaluator.py.
# Do not edit by hand: change the evaluator and re-run the sync.
"""Single-process local scorer: the grader's simulation, policy runner and metrics without its plumbing.

``mdeval evaluate`` reproduces the cloud grader end to end: a sim server and policy workers talking
over TLS, leases, a seccomp sandbox around onnxruntime. That machinery protects the grader from
untrusted graphs. A participant scoring their own policy needs none of it, so this module runs the
same pieces in-process:

    EnvRunner (env build, reset, step)  ->  OnnxPolicy (graph per task, LSTM state lifecycle)
    RolloutRecorder -> RolloutTrace     ->  metrics.compute  ->  scoring (task, family, session)

Every one of those is the grader's own code, imported, not reimplemented. What is left out: the
per-step deadline is measured and reported but never enforced (a laptop is not the grader's
machine), there is no retry of our own failures, and nothing is rendered.

Each rollout runs in a fresh spawned process (one simulator per process, as on the grader's
``env_proc``): mjlab, torch and warp draw from process-global RNGs, so a fresh process is what makes
a rollout's result independent of what ran before it and of ``jobs``, the number run concurrently.

**Seeds.** Competition tasks use ``seed_mode: secret``: each rollout's seed is an HMAC of a secret
and the task id. The grader's secret is not public, so a local run uses ``MDEVAL_SEED_SECRET`` or a
fixed public default. Same code and same seeds give the same score; local and grader scores differ
only by which episodes were drawn.

This module is vendored into the public ``aicrowd-microduck`` CLI (``aicrowd-microduck evaluate``) by
that repo's ``scripts/sync_from_evaluator.py``. Keep its imports to ``mdeval.common`` and
``mdeval.server`` modules that are vendored too (tasks, scoring, submission, onnx_policy,
env_runner, recorder, metrics), and nothing transport-, cloud- or render-specific.
"""
from __future__ import annotations

import json
import logging
import os
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path
from typing import Any, Callable

import numpy as np

from aicrowd_microduck.local.scoring import family_scores, session_score, task_score
from aicrowd_microduck.local.tasks import CompetitionSpec, TaskSpec, _dataclass_from, load_competition

log = logging.getLogger(__name__)

LOCAL_REPORT_SCHEMA = "local-1"
#: Used when neither the caller nor ``MDEVAL_SEED_SECRET`` names a secret. Public on purpose: a local
#: score is reproducible by anyone, and it is not the grader's secret, so it does not reveal the
#: grader's episodes.
DEFAULT_LOCAL_SEED_SECRET = "microduck-local"

# Rollout outcomes. The participant-fault codes are the grader's (protocol.RolloutErrorCode), so
# scoring.task_score counts them as 0 exactly as the grader does; anything else is "ours" and is
# left out of the mean, as on the grader.
SUCCESS = "SUCCESS"
POLICY_EXCEPTION = "POLICY_EXCEPTION"
ACTION_SHAPE_MISMATCH = "ACTION_SHAPE_MISMATCH"
ACTION_NOT_FINITE = "ACTION_NOT_FINITE"
SIM_ERROR = "SIM_ERROR"
LOCAL_WORKER_CRASHED = "LOCAL_WORKER_CRASHED"


# ------------------------------------------------------------------------------ selection
def select_tasks(comp: CompetitionSpec, only: list[str] | None) -> list[TaskSpec]:
    """The competition's tasks, or those named by ``only`` (task ids or family names), in competition order."""
    if not only:
        return list(comp.tasks)
    wanted = set(only)
    known = {t.task_id for t in comp.tasks} | set(comp.families())
    unknown = sorted(wanted - known)
    if unknown:
        raise ValueError(f"unknown task or family {unknown}; families: {sorted(comp.families())}")
    return [t for t in comp.tasks if t.task_id in wanted or (t.family and t.family in wanted)]


def plan_rollouts(tasks: list[TaskSpec], seed_secret: str, num_rollouts: int | None = None) -> list[dict[str, Any]]:
    """One entry per rollout, ids and seeds exactly as the grader's session assigns them.

    ``num_rollouts`` caps each task's rollouts (the first n; their seeds are unchanged), for a quick run.
    """
    plan = []
    for task in tasks:
        seeds = task.rollout_seeds(seed_secret)
        if num_rollouts is not None:
            seeds = seeds[: max(1, int(num_rollouts))]
        for i, seed in enumerate(seeds):
            plan.append({"rollout_id": f"{task.task_id}-r{i:02d}", "task_id": task.task_id, "index": i,
                         "seed": int(seed), "task": task.to_dict()})
    return plan


# ------------------------------------------------------------------------------- rollout
def _worker_init() -> None:  # pragma: no cover -- runs in the pool process
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    from aicrowd_microduck.local.env_runner import _ensure_tasks_registered
    _ensure_tasks_registered()


def _failed(entry: dict[str, Any], code: str, message: str, tb: str = "", **extra: Any) -> dict[str, Any]:
    return {**{k: v for k, v in entry.items() if k != "task"}, "state": "failed", "error_code": code,
            "error_message": message[:500], "traceback": tb[-4000:], "metrics": {}, **extra}


def run_rollout(entry: dict[str, Any], submission_dir: str, manifest: dict[str, Any], out_dir: str,
                save_trace: bool = True) -> dict[str, Any]:
    """Simulate one rollout and score it. Runs in a pool process; returns a JSON-able result entry."""
    import mujoco

    from aicrowd_microduck.local.onnx_policy import OnnxPolicy
    from aicrowd_microduck.local import metrics as M
    from aicrowd_microduck.local.env_runner import EnvRunner
    from aicrowd_microduck.local.recorder import RolloutRecorder

    spec: TaskSpec = _dataclass_from(TaskSpec, entry["task"])
    seed = int(entry["seed"])
    out = Path(out_dir)
    t_start = time.time()

    # The policy first: a graph that will not load is the submission's failure, not the simulator's.
    try:
        policy = OnnxPolicy(submission_dir, manifest)
        policy.setup()
        policy.reset(spec.task_id, spec.num_envs, spec.family)
    except Exception as e:  # noqa: BLE001
        return _failed(entry, POLICY_EXCEPTION, f"policy setup failed: {e!r}", traceback.format_exc())

    try:
        runner = EnvRunner(spec, seed, num_envs=spec.num_envs)
        obs = runner.reset()
    except Exception as e:  # noqa: BLE001
        return _failed(entry, SIM_ERROR, f"env build failed: {e!r}", traceback.format_exc())

    mjb = out / "assets" / f"model_{spec.task_id}_{seed}.mjb"
    if not mjb.exists():
        mjb.parent.mkdir(parents=True, exist_ok=True)
        tmp = f"{mjb}.{os.getpid()}.tmp"
        mujoco.mj_saveModel(runner.mj_model, tmp, None)
        os.replace(tmp, mjb)
    desc = runner.describe()
    recorder = RolloutRecorder(meta={
        "session_id": "local", "rollout_id": entry["rollout_id"], "agent_idx": 0, "n_agents": 1,
        "worker_id": "local", "attempt": 1, "task_spec": spec.to_dict(), "task_spec_digest": spec.digest(),
        **{k: v for k, v in desc.items() if k != "env_cfg"},
        "env_cfg_digest": desc.get("env_cfg", {}).get("_digest"),
        "model_mjb": f"assets/model_{spec.task_id}_{seed}.mjb",
    }, qpos0=runner.qpos(), model_path=mjb, qvel0=runner.qvel0, cmd_schedule=runner.cmd_schedule)

    K, act_dim = spec.num_envs, runner.act_dim
    done_mask: np.ndarray | None = None
    policy_wall = 0.0
    slow_steps = 0
    try:
        while True:
            t0 = time.perf_counter()
            try:
                action = np.asarray(policy.act(obs, done=done_mask), dtype=np.float32)
            except Exception as e:  # noqa: BLE001
                return _failed(entry, POLICY_EXCEPTION, f"policy raised at step {runner.step_count}: {e!r}",
                               traceback.format_exc())
            wall = time.perf_counter() - t0
            policy_wall += wall
            slow_steps += int(wall > spec.step_deadline_s)
            if action.shape != (K, act_dim):
                return _failed(entry, ACTION_SHAPE_MISMATCH, f"policy returned {action.shape}, expected {(K, act_dim)}")
            if not np.all(np.isfinite(action)):
                return _failed(entry, ACTION_NOT_FINITE, f"non-finite action at step {runner.step_count}")
            t_recv = time.time()
            rec = runner.step(action)
            recorder.add(rec, t_recv, time.time(), wall)
            if runner.all_done:
                break
            obs = runner.last_obs
            # the sticky finished mask, as the grader's server sends it: re-zeroes LSTM state of ended episodes
            done_mask = runner.done.astype(np.uint8)
    except Exception as e:  # noqa: BLE001
        return _failed(entry, SIM_ERROR, f"simulation failed at step {runner.step_count}: {e!r}", traceback.format_exc())
    finally:
        runner.close()

    done_step = runner.done_step
    trace = recorder.finish(np.where(done_step > 0, done_step - 1, -1),
                            extra_meta={"n_step_timeouts": 0, "policy_wall_s_total": policy_wall})
    if save_trace:
        trace.save(out / "rollouts" / entry["rollout_id"])
    try:
        mets = M.compute(trace, spec.metrics)
    except Exception as e:  # noqa: BLE001
        mets = {"_metrics_error": repr(e)[:300]}
    mets["primary"] = mets.get(spec.primary_metric)
    return {**{k: v for k, v in entry.items() if k != "task"}, "state": "completed", "error_code": SUCCESS,
            "metrics": mets, "n_steps": int(runner.step_count), "policy_wall_s_total": policy_wall,
            "n_slow_steps": slow_steps,
            # the grader holds the previous action on a missed deadline and fails the rollout after
            # max_step_timeouts misses; on this machine that would have happened if this is True
            "would_time_out_on_grader": slow_steps > spec.max_step_timeouts,
            "wall_s": time.time() - t_start}


# -------------------------------------------------------------------------------- report
def build_report(comp: CompetitionSpec, tasks: list[TaskSpec], results: list[dict[str, Any]],
                 **extra: Any) -> dict[str, Any]:
    """The grader's report shape, scored with the grader's own functions (``common/scoring.py``)."""
    tasks_out = []
    for task in tasks:
        rs = sorted((r for r in results if r["task_id"] == task.task_id), key=lambda r: r["index"])
        ts = task_score(rs)
        completed = [r for r in rs if r["state"] == "completed"]
        keys = sorted({k for r in completed for k, v in r["metrics"].items()
                       if isinstance(v, (int, float)) and not isinstance(v, bool)})
        agg = {}
        for k in keys:
            vals = [r["metrics"][k] for r in completed if isinstance(r["metrics"].get(k), (int, float))]
            if vals:
                agg[k] = sum(vals) / len(vals)
        tasks_out.append({
            "task_id": task.task_id, "family": task.family, "env_id": task.env_id, "description": task.description,
            "weight": comp.weight(task.task_id), "primary_metric": task.primary_metric, "score": ts.score,
            "n_rollouts": len(rs), "n_completed": ts.n_completed, "n_failed": ts.n_failed,
            "n_unscored": ts.n_unscored, "scorable": ts.scorable, "num_envs": task.num_envs,
            "max_steps": task.max_steps, "episode_length_s": task.episode_length_s,
            "metrics_mean": agg, "rollouts": rs,
        })
    score, unscored = session_score(tasks_out)
    failed = [r for r in results if r["state"] != "completed"]
    return {
        "schema_version": LOCAL_REPORT_SCHEMA,
        "competition": {"name": comp.name, "digest": comp.digest(), "source_path": comp.source_path},
        "state": "completed" if not failed else "completed_with_failures",
        "score": score, "families": family_scores(tasks_out), "unscored_tasks": unscored, "tasks": tasks_out,
        **extra,
    }


# ------------------------------------------------------------------------------ evaluate
def evaluate_local(submission_dir: str | Path, competition: str | Path | CompetitionSpec, out_dir: str | Path, *,
                   only: list[str] | None = None, jobs: int = 1, seed_secret: str | None = None,
                   num_rollouts: int | None = None, save_traces: bool = True,
                   on_result: Callable[[dict[str, Any], int, int], None] | None = None) -> dict[str, Any]:
    """Validate the submission with the grader's checks, simulate every selected rollout, score, write report.json.

    Raises ``SubmissionRejected`` before any simulation if the grader would reject the submission.
    ``on_result(result, n_done, n_total)`` is called as each rollout finishes.
    """
    from aicrowd_microduck.submission import validate_submission

    comp = competition if isinstance(competition, CompetitionSpec) else load_competition(competition)
    tasks = select_tasks(comp, only)
    sub = Path(submission_dir).resolve()
    # the full competition's coverage and action-scale checks, as the grader runs them before scoring
    verdict = validate_submission(sub, tasks=comp.tasks)
    manifest = verdict["manifest"]

    source = "argument" if seed_secret else ("MDEVAL_SEED_SECRET" if os.environ.get("MDEVAL_SEED_SECRET") else "default")
    seed_secret = seed_secret or os.environ.get("MDEVAL_SEED_SECRET") or DEFAULT_LOCAL_SEED_SECRET
    plan = plan_rollouts(tasks, seed_secret, num_rollouts)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    import multiprocessing as mp

    results: list[dict[str, Any]] = []
    t0 = time.time()
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=max(1, int(jobs)), mp_context=ctx, initializer=_worker_init,
                             max_tasks_per_child=1) as pool:
        futs = {pool.submit(run_rollout, e, str(sub), manifest, str(out), save_traces): e for e in plan}
        for fut in as_completed(futs):
            entry = futs[fut]
            try:
                res = fut.result()
            except BrokenProcessPool as e:
                res = _failed(entry, LOCAL_WORKER_CRASHED, f"rollout process died: {e!r}")
            except Exception as e:  # noqa: BLE001
                res = _failed(entry, LOCAL_WORKER_CRASHED, f"rollout process failed: {e!r}", traceback.format_exc())
            results.append(res)
            if on_result:
                on_result(res, len(results), len(plan))

    partial = bool(only) or num_rollouts is not None
    report = build_report(comp, tasks, results, submission_dir=str(sub), manifest=manifest,
                          graphs=verdict["graphs"], seed_secret_source=source,
                          seed_secret_is_default=seed_secret == DEFAULT_LOCAL_SEED_SECRET,
                          partial=partial, selected=only or None, num_rollouts_override=num_rollouts,
                          jobs=jobs, wall_s=time.time() - t0, created_at=t0)
    with open(out / "report.json", "w") as f:
        json.dump(report, f, indent=1, default=str)
    return report
