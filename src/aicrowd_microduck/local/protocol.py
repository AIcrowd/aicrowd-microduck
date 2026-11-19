# VENDORED from microduck-evaluator/src/mdeval/common/protocol.py (rollout error codes only)
# by scripts/sync_from_evaluator.py. Do not edit by hand.
from __future__ import annotations

from enum import Enum

from aicrowd_microduck.submission import SubmissionErrorCode  # noqa: F401


class RolloutErrorCode(str, Enum):
    SUCCESS = "SUCCESS"
    POLICY_EXCEPTION = "POLICY_EXCEPTION"         # act()/reset() raised
    STEP_TIMEOUT = "STEP_TIMEOUT"                 # too many per-step deadline misses
    ACTION_SHAPE_MISMATCH = "ACTION_SHAPE_MISMATCH"
    ACTION_NOT_FINITE = "ACTION_NOT_FINITE"
    POLICY_CRASHED = "POLICY_CRASHED"             # child process died
    LEASE_LOST = "LEASE_LOST"
    SIM_ERROR = "SIM_ERROR"                       # server-side env failure (NaN physics etc.)


#: Rollout failures the submission caused. These two properties follow from that and nothing else:
#: the rollout scores 0 for its task, and it is not retried (it would fail the same way again).
#: Everything not in here is ours -- retried while attempts remain, and never scored against the
#: participant. Defined here rather than next to either user because the scheduler and the scorer
#: have to agree on it: they disagreed once, and an evaluator failure became a participant's zero.
PARTICIPANT_FAULT_ROLLOUT = frozenset({
    RolloutErrorCode.POLICY_EXCEPTION, RolloutErrorCode.STEP_TIMEOUT,
    RolloutErrorCode.ACTION_SHAPE_MISMATCH, RolloutErrorCode.ACTION_NOT_FINITE,
    RolloutErrorCode.POLICY_CRASHED,
})


def is_participant_fault(error_code: "str | RolloutErrorCode | None") -> bool:
    """Whether a failed rollout's failure is the submission's doing.

    Anything we cannot attribute is **ours**, deliberately: an unrecognised or absent code means we
    do not know whose fault it was, and a participant must not lose score on our uncertainty.

    Today that covers ``LEASE_LOST`` (the shard went away, e.g. a Spot reclaim) and ``SIM_ERROR``
    (a server-side env failure such as NaN physics). ``SIM_ERROR`` is the arguable one, since a
    policy might in principle provoke it. It counts as ours because joint targets are bounded by
    the task's action scale and the joint limits, so an exploding sim is far likelier our bug than
    their policy -- and a submission that does provoke it systematically surfaces as a task with no
    score, which an organizer sees, rather than as a silent zero.
    """
    if error_code is None:
        return False
    code = error_code.value if isinstance(error_code, RolloutErrorCode) else str(error_code)
    try:
        return RolloutErrorCode(code) in PARTICIPANT_FAULT_ROLLOUT
    except ValueError:          # a code this build does not know: not attributable to them
        return False
