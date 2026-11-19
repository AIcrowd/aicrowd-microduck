# VENDORED from microduck-evaluator/src/mdeval/common/scoring.py by scripts/sync_from_evaluator.py.
# Do not edit by hand: change the evaluator and re-run the sync.
"""Score aggregation shared by the session report, the shard merge and ``mdeval rescore``.

A task's score is the mean primary metric over its **scorable** rollouts; the session score is the
weighted mean over the tasks that have one. v2 adds a score per family: the weighted mean of its
level tasks' scores, ``sum(w * level score) / sum(w)``. Pure functions of the report's task entries.

Scorable is the whole subtlety, and it is why :func:`task_score` exists rather than one line in each
caller. A rollout that failed because the *submission* failed scores 0 -- that is the honest number
for a policy that crashed or ran out of step budget. A rollout that failed because *we* failed (a
Spot reclaim took its shard, the sim went non-finite) is excluded: scoring it 0 charges a
participant for our outage, and at 24 shards per submission that is not a rare event. The two were
distinguished by the scheduler, which retries only the second kind, but not by the scorer -- and the
expression that got it wrong had been copied into three files.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from aicrowd_microduck.local.protocol import is_participant_fault


@dataclass(frozen=True)
class TaskScore:
    """A task's score and what it was computed from."""

    score: float
    n_rollouts: int
    n_completed: int
    #: Failed and attributable to the submission. Counted in the mean, as 0.
    n_failed: int
    #: Failed but not the submission's doing, or never finished. Left out of the mean entirely.
    n_unscored: int

    @property
    def n_scored(self) -> int:
        return self.n_completed + self.n_failed

    @property
    def scorable(self) -> bool:
        """False when nothing counted. Then the task has **no score**, which is not a score of 0 --
        a caller that treats it as 0 reintroduces the bug this class exists to prevent."""
        return self.n_scored > 0


def task_score(rollouts: list[dict[str, Any]]) -> TaskScore:
    """Score one task from its rollout entries."""
    values: list[float] = []
    n_completed = n_failed = n_unscored = 0
    for r in rollouts:
        if r.get("state") == "completed":
            n_completed += 1
            values.append(float(r.get("metrics", {}).get("primary") or 0.0))
        elif r.get("state") == "failed" and is_participant_fault(r.get("error_code")):
            n_failed += 1
            values.append(0.0)
        else:
            # Our failure, an unknown code, or a rollout that never reached a terminal state.
            n_unscored += 1
    return TaskScore(score=(sum(values) / len(values)) if values else 0.0,
                     n_rollouts=len(rollouts), n_completed=n_completed,
                     n_failed=n_failed, n_unscored=n_unscored)


def session_score(tasks: list[dict[str, Any]]) -> tuple[float, list[str]]:
    """The weighted mean over tasks that have a score, plus the task ids that have none.

    An unscorable task is left out of both the numerator and the denominator. Including it as 0
    would be the same error one level up: the submission would be marked down for a level it was
    never actually run on.
    """
    total = wsum = 0.0
    unscored: list[str] = []
    for t in tasks:
        if t.get("scorable") is False:
            unscored.append(t["task_id"])
            continue
        w = float(t.get("weight", 0.0))
        total += w * float(t.get("score") or 0.0)
        wsum += w
    return ((total / wsum) if wsum else 0.0), unscored


def task_family(task: dict[str, Any]) -> str | None:
    """A report task entry's family: the ``family`` field, else the one in its spec (reports written by an
    evaluator that did not copy it up), else None (v1 tasks)."""
    return task.get("family") or (task.get("spec") or {}).get("family") or None


def family_scores(tasks: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """family -> {score, weight, task_ids}, in order of first appearance. ``score`` is the weight-averaged task
    score of the family's tasks (0 when their weights sum to 0, like the session score); ``weight`` is that sum.
    Tasks without a family are left out, so a v1 report has no families."""
    acc: dict[str, dict[str, Any]] = {}
    for t in tasks:
        fam = task_family(t)
        if not fam:
            continue
        a = acc.setdefault(fam, {"total": 0.0, "weight": 0.0, "task_ids": [], "unscored_task_ids": []})
        a["task_ids"].append(t["task_id"])
        # A level with no score is left out of its family's mean for the same reason it is left out
        # of the session's: it was not run, so it is not a zero.
        if t.get("scorable") is False:
            a["unscored_task_ids"].append(t["task_id"])
            continue
        w = float(t.get("weight", 0.0))
        a["total"] += w * float(t.get("score") or 0.0)
        a["weight"] += w
    return {fam: {"score": (a["total"] / a["weight"]) if a["weight"] else 0.0, "weight": a["weight"],
                  "task_ids": a["task_ids"], "unscored_task_ids": a["unscored_task_ids"]}
            for fam, a in acc.items()}
