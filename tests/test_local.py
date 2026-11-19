"""The vendored local scorer: importable without the simulator, competition intact, `evaluate` explains what is missing.

Simulation itself is not exercised here (CI has no simulator); the evaluator's
tests/test_local_eval_parity.py runs the same code against the grader.
"""

from __future__ import annotations

import pytest

import aicrowd_microduck.cli as cli
from aicrowd_microduck.cli import main

from test_cli import FAMILIES, _submission  # noqa: E402  (tests/ is on sys.path under pytest)


def test_scorer_modules_import_without_the_simulator():
    from aicrowd_microduck.local import local_eval, metrics, scoring, tasks  # noqa: F401

    # every v2 family's metric registered through the vendored plugin loader
    for name in ("command_progress", "descent_score", "flamingo_support_v2", "trampoline_height"):
        assert name in metrics.available(), metrics.available()


def test_vendored_competition_is_the_live_one():
    from aicrowd_microduck.local.tasks import load_competition

    comp = load_competition(cli._local_competition_path())
    assert len(comp.tasks) == 30
    assert sorted(comp.families()) == sorted(FAMILIES)
    assert abs(sum(comp.weights.values()) - 1.0) < 1e-9
    # the reduced snapshot validate uses agrees with the full specs evaluate uses
    _, snap = cli.competition_tasks()
    assert [(t.task_id, t.family, t.action_scale) for t in snap] == \
           [(t.task_id, t.family, t.action_scale) for t in comp.tasks]


def test_select_tasks_by_family_and_id():
    from aicrowd_microduck.local.local_eval import plan_rollouts, select_tasks
    from aicrowd_microduck.local.tasks import load_competition

    comp = load_competition(cli._local_competition_path())
    sel = select_tasks(comp, ["jelly_walk_v2", "roller_descent_v2_r5"])
    assert [t.task_id for t in sel] == ["roller_descent_v2_r5"] + [f"jelly_walk_v2_e{i}" for i in range(1, 5)]
    with pytest.raises(ValueError):
        select_tasks(comp, ["no_such_family"])
    # seeds are the grader's derivation (HMAC of task id and index), capped without being reshuffled
    full = plan_rollouts(sel[:1], "s")
    capped = plan_rollouts(sel[:1], "s", num_rollouts=1)
    assert capped == full[:1]
    assert plan_rollouts(sel[:1], "other")[0]["seed"] != full[0]["seed"]


def test_evaluate_without_simulator_says_how_to_get_one(tmp_path, monkeypatch, capsys):
    sub = _submission(tmp_path)
    monkeypatch.setattr(cli, "_check_simulator", lambda env_id: "mjlab is not installed")
    assert main(["evaluate", str(sub), "-o", str(tmp_path / "run")]) == 2
    err = capsys.readouterr().err
    assert "cannot simulate" in err and "starter kit" in err


def test_evaluate_rejects_unknown_family(tmp_path, capsys):
    sub = _submission(tmp_path)
    assert main(["evaluate", str(sub), "--only", "nope", "-o", str(tmp_path / "run")]) == 2
    assert "unknown task or family" in capsys.readouterr().err
