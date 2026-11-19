"""Copy what the CLI needs from the evaluator: the submission validator, the worker's unpacking
rules, the task list, and the local scorer behind `aicrowd-microduck evaluate`.

    python scripts/sync_from_evaluator.py [--evaluator ../microduck-evaluator] [--check]

The validator and the worker's ingestion module are vendored nearly verbatim (only their mdeval
imports are replaced), so a submission this CLI passes is one the worker unpacks and accepts. The
competition's tasks are reduced to the three fields the coverage check reads. Run it whenever either
changes in the evaluator; `--check` exits 1 if the vendored copies have drifted (tests/test_parity.py
runs that when the evaluator checkout is present).

The local scorer (`aicrowd_microduck/local/`) is the evaluator's own simulation, policy runner,
metrics and scoring code (`server/local_eval.py` and what it imports), copied with every
`mdeval.common` / `mdeval.server` import rewritten to the vendored module, plus the competition's
task YAMLs verbatim. The evaluator's tests/test_local_eval_parity.py proves that code scores exactly
as the grader does.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / "src" / "aicrowd_microduck"
COMPETITION = "tasks/competition_v2.yaml"  # what the live gateway runs (infrastructure/variables.tf)

_DOC_HEAD = '"""The submission format: a manifest plus ONNX graphs. No participant Python runs anywhere.'
_DOC_NOTE = """

VENDORED from the evaluator (microduck-evaluator, src/mdeval/common/submission.py) so that
`aicrowd-microduck validate` gives the grader's own verdict without installing the evaluator.
Only the import of SubmissionErrorCode changed; regenerate with scripts/sync_from_evaluator.py
rather than editing by hand."""
_IMPORT = "from mdeval.common.protocol import SubmissionErrorCode\n"


def vendored_submission(evaluator: Path) -> str:
    src = (evaluator / "src/mdeval/common/submission.py").read_text()
    proto = (evaluator / "src/mdeval/common/protocol.py").read_text()
    if not src.startswith(_DOC_HEAD) or _IMPORT not in src:
        raise SystemExit("evaluator submission.py changed shape; update scripts/sync_from_evaluator.py")
    # lift the enum out of protocol.py as-is, so a new rejection code arrives with the next sync
    start = proto.index("class SubmissionErrorCode(str, Enum):")
    end = proto.index("\n\n\n", start)
    enum_src = "from enum import Enum\n\n\n" + proto[start:end] + "\n"
    return src.replace(_DOC_HEAD, _DOC_HEAD + _DOC_NOTE, 1).replace(_IMPORT, enum_src, 1)


_INGEST_DOC = '"""Fetch and safely unpack a submission (tar.gz / zip / directory), then validate it.'
_INGEST_IMPORTS = ("from mdeval.common.protocol import SubmissionErrorCode\n"
                   "from mdeval.common.submission import PolicyManifest, SubmissionRejected, load_manifest\n")


def vendored_ingestion(evaluator: Path) -> str:
    src = (evaluator / "src/mdeval/worker/ingestion.py").read_text()
    if not src.startswith(_INGEST_DOC) or _INGEST_IMPORTS not in src:
        raise SystemExit("evaluator ingestion.py changed shape; update scripts/sync_from_evaluator.py")
    note = ("\n\nVENDORED from the evaluator (microduck-evaluator, src/mdeval/worker/ingestion.py): the worker's\n"
            "own unpacking rules. Only the mdeval imports changed; regenerate with scripts/sync_from_evaluator.py.")
    return (src.replace(_INGEST_DOC, _INGEST_DOC + note, 1)
               .replace(_INGEST_IMPORTS, "from .submission import PolicyManifest, SubmissionErrorCode, "
                                         "SubmissionRejected, load_manifest\n", 1))


def task_snapshot(evaluator: Path) -> str:
    import yaml

    comp_path = evaluator / COMPETITION
    comp = yaml.safe_load(comp_path.read_text())
    tasks = []
    for entry in comp["tasks"]:
        t = yaml.safe_load((comp_path.parent / entry).read_text()) if isinstance(entry, str) else entry
        tasks.append({"task_id": t["task_id"], "family": t.get("family"),
                      "action_scale": float(t.get("action_scale", 1.0))})
    snap = {"competition": comp["name"], "source": COMPETITION, "tasks": tasks}
    return json.dumps(snap, indent=1) + "\n"


# ------------------------------------------------------------------------------ local scorer
LOCAL = PKG / "local"
# evaluator module -> vendored module name under aicrowd_microduck.local
LOCAL_MODULES = {
    "src/mdeval/common/tasks.py": "tasks",
    "src/mdeval/common/scoring.py": "scoring",
    "src/mdeval/common/onnx_policy.py": "onnx_policy",
    "src/mdeval/server/env_runner.py": "env_runner",
    "src/mdeval/server/recorder.py": "recorder",
    "src/mdeval/server/metrics.py": "metrics",
    "src/mdeval/server/kinematics.py": "kinematics",
    "src/mdeval/server/local_eval.py": "local_eval",
}
_MODULE_RE = re.compile(r"\bmdeval\.(common|server)\.(\w+)")
_FROM_PKG_RE = re.compile(r"\bfrom mdeval\.(common|server) import\b")


def _target(sub: str, name: str) -> str:
    if (sub, name) == ("common", "submission"):
        return "aicrowd_microduck.submission"
    return f"aicrowd_microduck.local.{name}"


def rewrite_imports(src: str, origin: str) -> str:
    out = _MODULE_RE.sub(lambda m: _target(m.group(1), m.group(2)), src)
    out = _FROM_PKG_RE.sub("from aicrowd_microduck.local import", out)
    left = [ln for ln in out.splitlines() if re.search(r"\b(from|import) mdeval\b", ln)]
    if left:
        raise SystemExit(f"{origin}: imports left unrewritten: {left}")
    note = (f"# VENDORED from microduck-evaluator/{origin} by scripts/sync_from_evaluator.py.\n"
            "# Do not edit by hand: change the evaluator and re-run the sync.\n")
    return note + out


def local_protocol(evaluator: Path) -> str:
    """The rollout error codes and the participant-fault rule, lifted from protocol.py as-is (the scorer
    must count a failure as the participant's exactly when the grader does)."""
    proto = (evaluator / "src/mdeval/common/protocol.py").read_text()
    start = proto.index("class RolloutErrorCode(str, Enum):")
    end = proto.index("class SubmissionErrorCode(str, Enum):", start)
    return ("# VENDORED from microduck-evaluator/src/mdeval/common/protocol.py (rollout error codes only)\n"
            "# by scripts/sync_from_evaluator.py. Do not edit by hand.\n"
            "from __future__ import annotations\n\nfrom enum import Enum\n\n"
            "from aicrowd_microduck.submission import SubmissionErrorCode  # noqa: F401\n\n\n"
            + proto[start:end].rstrip() + "\n")


_LOCAL_INIT = '''"""The evaluator's local scorer, vendored: `aicrowd-microduck evaluate` runs the grader's own
simulation, policy runner, metrics and scoring in-process. Needs the `sim` extra and the Microduck
environments (the starter kit installs both). Generated by scripts/sync_from_evaluator.py."""
'''


def local_files(evaluator: Path) -> dict[Path, str]:
    files: dict[Path, str] = {LOCAL / "__init__.py": _LOCAL_INIT, LOCAL / "protocol.py": local_protocol(evaluator)}
    for origin, name in LOCAL_MODULES.items():
        files[LOCAL / f"{name}.py"] = rewrite_imports((evaluator / origin).read_text(), origin)
    for f in sorted((evaluator / "src/mdeval/server/metrics_v2").glob("*.py")):
        origin = str(f.relative_to(evaluator))
        files[LOCAL / "metrics_v2" / f.name] = rewrite_imports(f.read_text(), origin)
    comp = evaluator / COMPETITION
    files[LOCAL / "competition" / comp.name] = comp.read_text()
    for entry in yaml_entries(comp):
        files[LOCAL / "competition" / entry] = (comp.parent / entry).read_text()
    return files


def yaml_entries(comp_path: Path) -> list[str]:
    import yaml

    return [e for e in yaml.safe_load(comp_path.read_text())["tasks"] if isinstance(e, str)]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--evaluator", type=Path, default=ROOT.parent / "microduck-evaluator")
    ap.add_argument("--check", action="store_true", help="report drift instead of writing")
    args = ap.parse_args()

    wanted = {PKG / "submission.py": vendored_submission(args.evaluator),
              PKG / "ingestion.py": vendored_ingestion(args.evaluator),
              PKG / "tasks.json": task_snapshot(args.evaluator),
              **local_files(args.evaluator)}
    # a vendored local/ file the evaluator no longer has is drift too (a deleted metric module)
    stale = sorted(p for p in LOCAL.rglob("*") if p.is_file() and p not in wanted
                   and "__pycache__" not in p.parts)
    drifted = [p for p, text in wanted.items() if not p.exists() or p.read_text() != text]
    if args.check:
        for p in drifted:
            print(f"out of date: {p.relative_to(ROOT)}", file=sys.stderr)
        for p in stale:
            print(f"no longer in the evaluator: {p.relative_to(ROOT)}", file=sys.stderr)
        return 1 if drifted or stale else 0
    for p in drifted:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(wanted[p])
        print(f"wrote {p.relative_to(ROOT)}")
    for p in stale:
        p.unlink()
        print(f"removed {p.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
