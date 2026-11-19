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
    monkeypatch.setattr(cli, "_check_simulator", lambda env_ids: "mjlab is not installed")
    assert main(["evaluate", str(sub), "-o", str(tmp_path / "run")]) == 2
    err = capsys.readouterr().err
    assert "cannot simulate" in err and "starter kit" in err


def test_evaluate_rejects_unknown_family(tmp_path, capsys):
    sub = _submission(tmp_path)
    assert main(["evaluate", str(sub), "--only", "nope", "-o", str(tmp_path / "run")]) == 2
    assert "unknown task or family" in capsys.readouterr().err


def test_sim_extra_pins_the_grader_stack():
    """The `sim` extra must install exactly the grader's simulator and inference builds."""
    from pathlib import Path

    import toml  # a runtime dependency; tomllib is 3.11+ and CI runs 3.10

    pins = {}
    for req in toml.loads((Path(__file__).parents[1] / "pyproject.toml").read_text())["project"][
            "optional-dependencies"]["sim"]:
        spec = req.split(";")[0].strip()
        name, version = spec.split("==")
        pins[name] = version
    assert pins == cli.grader_stack()


def test_stack_mismatch_names_each_package():
    stack = dict(cli.grader_stack())
    assert cli.stack_mismatches(stack) == []
    stack["mujoco"] = "4.0.1"
    assert cli.stack_mismatches(stack) == [f"mujoco 4.0.1 (grader {cli.grader_stack()['mujoco']})"]


def test_evaluate_json_reports_early_failures(tmp_path, capsys):
    import json

    sub = _submission(tmp_path)
    assert main(["evaluate", str(sub), "--only", "nope", "--json", "-o", str(tmp_path / "run")]) == 2
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is False and "unknown task or family" in payload["error"]


def test_every_pinned_package_is_checked_at_run_time():
    """stack_mismatches(stack_versions()) can only be clean if stack_versions() reports every pinned package."""
    from aicrowd_microduck.local.local_eval import STACK_PACKAGES, stack_versions

    assert set(cli.grader_stack()) <= set(STACK_PACKAGES)
    assert set(cli.grader_stack()) <= set(stack_versions())


def test_validate_names_unknown_manifest_keys(tmp_path, capsys):
    sub = _submission(tmp_path, polices={"x": "y"})          # the classic typo of "policies"
    assert main(["validate", str(sub)]) == 0
    assert "polices" in capsys.readouterr().err
