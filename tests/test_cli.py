"""End-to-end CLI tests: real ONNX graphs through the vendored validator, AIcrowd mocked at the HTTP layer."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import httpx
import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper

import aicrowd_microduck.aicrowd_client as client_mod
import aicrowd_microduck.cli as cli
from aicrowd_microduck.cli import main

FAMILIES = ["rough_terrain_v2", "roller_descent_v2", "jelly_walk_v2", "flamingo_switch_v2",
            "trampoline_height_v2"]


def _graph(path: Path, *, constant: bool = False) -> None:
    """obs float32[1, 61] -> actions float32[1, 14]; a zero weight makes the output constant."""
    rng = np.random.default_rng(0)
    w = np.zeros((61, 14), np.float32) if constant else rng.normal(0, 0.1, (61, 14)).astype(np.float32)
    b = np.full((14,), 0.1, np.float32)
    g = helper.make_graph(
        [helper.make_node("MatMul", ["obs", "W"], ["h"]), helper.make_node("Add", ["h", "B"], ["actions"])],
        "policy",
        [helper.make_tensor_value_info("obs", TensorProto.FLOAT, [1, 61])],
        [helper.make_tensor_value_info("actions", TensorProto.FLOAT, [1, 14])],
        initializer=[numpy_helper.from_array(w, "W"), numpy_helper.from_array(b, "B")],
    )
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)])
    m.ir_version = 8
    onnx.save(m, path)


def _submission(root: Path, *, policies=None, constant: bool = False, **manifest) -> Path:
    d = root / "my_policy"
    d.mkdir()
    _graph(d / "walk.onnx", constant=constant)
    body = {"schema_version": 1, "model_api": 1, "action_scale": 1.0,
            "policies": policies if policies is not None else {f: "walk.onnx" for f in FAMILIES}}
    body.update(manifest)
    (d / "manifest.json").write_text(json.dumps(body))
    return d


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    monkeypatch.setattr("aicrowd_microduck.aicrowd_config._app_dir", lambda: tmp_path / "cfg")
    monkeypatch.delenv("AICROWD_API_KEY", raising=False)
    monkeypatch.setattr(cli, "POLL_INTERVAL_S", 0.0)


class FakeAIcrowd:
    """Just enough of the Rails API for one submit, recording every request."""

    def __init__(self, *, allowed: bool = True, statuses=("submitted", "graded")):
        self.allowed = allowed
        self.statuses = list(statuses)
        self.calls: list[tuple[str, str]] = []
        self.uploaded: dict = {}
        self.created: dict = {}

    def __call__(self, req: httpx.Request) -> httpx.Response:
        self.calls.append((req.method, req.url.path))
        path = req.url.path
        if path == "/api/v1/api_user":
            return httpx.Response(200, json={"id": 7, "username": "duck"})
        if path.endswith("/eligibility"):
            if self.allowed:
                return httpx.Response(200, json={"submissions_allowed": True})
            return httpx.Response(200, json={"submissions_allowed": False, "denied_reason": "round_closed",
                                             "message": "Round 1 has ended.", "rules_accepted": True})
        if path == "/api/v1/submissions" and req.method == "GET":
            return httpx.Response(200, json={"success": True, "data": {
                "url": "https://s3.example/bucket", "fields": {"key": "subs/abc/${filename}", "policy": "p"}}})
        if req.url.host == "s3.example":
            self.uploaded = {"auth": req.headers.get("authorization"), "body": req.content}
            return httpx.Response(204)
        if path == "/api/v1/submissions" and req.method == "POST":
            self.created = json.loads(req.content)
            return httpx.Response(200, json={"success": True, "data": {"submission_id": 4242}})
        if path == "/api/v1/submissions/4242":
            st = self.statuses.pop(0) if len(self.statuses) > 1 else self.statuses[0]
            return httpx.Response(200, json={"id": 4242, "grading_status_cd": st, "score": 0.61,
                                             "grading_message": "<p>Scored 30/30 rollouts.</p>"})
        return httpx.Response(404, json={"message": f"unexpected {req.method} {path}"})


@pytest.fixture
def fake(monkeypatch):
    server = FakeAIcrowd()
    real = client_mod.AIcrowdClient

    def factory(*, api_key, **kw):
        return real(api_key=api_key, http=httpx.Client(transport=httpx.MockTransport(server)))

    monkeypatch.setattr(client_mod, "AIcrowdClient", factory)
    monkeypatch.setenv("AICROWD_API_KEY", "K")
    return server


# --------------------------------------------------------------------------------- validate
def test_validate_accepts_a_family_routed_submission(tmp_path, capsys):
    d = _submission(tmp_path)
    assert main(["validate", str(d), "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["ok"] and out["competition"] == "microduck-eval-v2"
    assert len(out["tasks"]) == 30 and set(out["tasks"].values()) == {"walk.onnx"}


def test_validate_rejects_a_graph_that_ignores_the_observation(tmp_path, capsys):
    d = _submission(tmp_path, constant=True)
    assert main(["validate", str(d), "--json"]) == 1
    assert json.loads(capsys.readouterr().out)["error_code"] == "ONNX_OUTPUT_INVALID"


def test_validate_rejects_a_submission_that_misses_a_family(tmp_path, capsys):
    d = _submission(tmp_path, policies={f: "walk.onnx" for f in FAMILIES[:-1]})
    assert main(["validate", str(d), "--json"]) == 1
    out = json.loads(capsys.readouterr().out)
    assert out["error_code"] == "POLICY_FILE_MISSING" and "trampoline_height_v2_l1" in out["error"]


def test_validate_rejects_a_different_action_scale(tmp_path, capsys):
    d = _submission(tmp_path, action_scale=0.5)
    assert main(["validate", str(d), "--json"]) == 1
    assert json.loads(capsys.readouterr().out)["error_code"] == "ACTION_SCALE_MISMATCH"


def test_validate_rejects_a_symlink_in_an_archive(tmp_path, capsys):
    z = tmp_path / "evil.zip"
    with zipfile.ZipFile(z, "w") as zf:
        info = zipfile.ZipInfo("manifest.json")
        info.external_attr = 0o120777 << 16
        zf.writestr(info, "/etc/passwd")
    assert main(["validate", str(z), "--json"]) == 1
    assert json.loads(capsys.readouterr().out)["error_code"] == "UNSAFE_ARCHIVE_MEMBER"


def test_validate_rejects_an_archive_over_the_gateway_cap(tmp_path, capsys, monkeypatch):
    d = _submission(tmp_path)
    monkeypatch.setattr(cli, "MAX_ARCHIVE_BYTES", 10)
    assert main(["validate", str(d), "--json"]) == 1
    assert json.loads(capsys.readouterr().out)["error_code"] == "SUBMISSION_TOO_LARGE"


def test_validate_accepts_an_archive_wrapped_in_a_folder(tmp_path):
    d = _submission(tmp_path)
    z = tmp_path / "wrapped.zip"
    with zipfile.ZipFile(z, "w") as zf:
        for f in d.iterdir():
            zf.write(f, arcname=f"my_policy/{f.name}")
    assert main(["validate", str(z)]) == 0


def test_validate_explains_a_finder_zip(tmp_path, capsys):
    """Finder/ditto zips put __MACOSX/ beside the wrapper folder, so the evaluator cannot unwrap it."""
    d = _submission(tmp_path)
    z = tmp_path / "finder.zip"
    with zipfile.ZipFile(z, "w") as zf:
        for f in d.iterdir():
            zf.write(f, arcname=f"my_policy/{f.name}")
        zf.writestr("__MACOSX/my_policy/._manifest.json", "x")
    assert main(["validate", str(z), "--json"]) == 1
    out = json.loads(capsys.readouterr().out)
    assert out["error_code"] == "MANIFEST_INVALID" and "__MACOSX" in out["hint"]


# ------------------------------------------------------------------------------------- pack
def test_pack_puts_files_at_the_root_and_skips_hidden(tmp_path):
    d = _submission(tmp_path)
    (d / ".DS_Store").write_text("x")
    (d / "__pycache__").mkdir()
    (d / "__pycache__" / "a.pyc").write_text("x")
    out = tmp_path / "out.zip"
    assert main(["pack", str(d), "-o", str(out)]) == 0
    assert sorted(zipfile.ZipFile(out).namelist()) == ["manifest.json", "walk.onnx"]


def test_pack_refuses_a_directory_without_a_manifest(tmp_path):
    (tmp_path / "empty").mkdir()
    assert main(["pack", str(tmp_path / "empty")]) == 2


# ----------------------------------------------------------------------------------- submit
def test_submit_uploads_creates_and_watches(tmp_path, fake, capsys):
    d = _submission(tmp_path)
    assert main(["submit", str(d), "--watch", "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["submission_id"] == 4242 and out["status"]["grading_status_cd"] == "graded"
    # the token never goes to S3, and the archive that went up is the packed directory
    assert fake.uploaded["auth"] is None and b"manifest.json" in fake.uploaded["body"]
    assert fake.created == {"challenge_id": "microduck-sim2real-challenge-2026",
                            "submission": {"description": "Submitted via aicrowd-microduck"},
                            "submission_files": [{"submission_file_s3_key": "subs/abc/my_policy.zip"}]}
    # never api.aicrowd.com
    assert all(not p.startswith("/challenges/?") for _, p in fake.calls)


def test_submit_dry_run_uploads_nothing(tmp_path, fake):
    d = _submission(tmp_path)
    assert main(["submit", str(d), "--dry-run"]) == 0
    assert fake.uploaded == {} and fake.created == {}
    assert ("GET", "/api/v1/submissions") in fake.calls  # the presign gate (quota, terms) was asked


def test_submit_stops_before_upload_when_ineligible(tmp_path, fake, capsys):
    fake.allowed = False
    d = _submission(tmp_path)
    assert main(["submit", str(d), "--json"]) == 1
    out = json.loads(capsys.readouterr().out)
    assert out["error"] == "Round 1 has ended." and out["error_code"] == "round_closed"
    assert fake.uploaded == {} and fake.created == {}


def test_submit_refuses_an_invalid_submission_without_calling_aicrowd(tmp_path, fake):
    d = _submission(tmp_path, constant=True)
    assert main(["submit", str(d)]) == 1
    assert fake.calls == []


def test_submit_without_a_key_says_how_to_log_in(tmp_path, capsys):
    d = _submission(tmp_path)
    assert main(["submit", str(d)]) == 2
    assert "aicrowd-microduck login" in capsys.readouterr().err


def test_submit_reports_a_failed_grading(tmp_path, fake):
    fake.statuses = ["failed"]
    d = _submission(tmp_path)
    assert main(["submit", str(d), "--watch"]) == 1


# ----------------------------------------------------------------------------- login, status
def test_login_saves_a_key_aicrowd_cli_can_read(tmp_path, fake):
    assert main(["login", "--api-key", "K"]) == 0
    assert 'aicrowd_api_key = "K"' in (tmp_path / "cfg" / "config.toml").read_text()


def test_status_strips_html_from_the_grading_message(fake, capsys):
    fake.statuses = ["graded"]
    assert main(["status", "4242"]) == 0
    assert "Graded: score 0.61 (Scored 30/30 rollouts.)" in capsys.readouterr().err


def test_watch_gives_up_at_its_timeout(fake, capsys):
    fake.statuses = ["submitted"]
    assert main(["status", "4242", "--watch", "--watch-timeout", "0"]) == 0
    assert "Still grading" in capsys.readouterr().err
