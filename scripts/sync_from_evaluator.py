"""Copy what the CLI needs from the evaluator: the submission validator, the worker's unpacking
rules and the task list.

    python scripts/sync_from_evaluator.py [--evaluator ../microduck-evaluator] [--check]

The validator and the worker's ingestion module are vendored nearly verbatim (only their mdeval
imports are replaced), so a submission this CLI passes is one the worker unpacks and accepts. The
competition's tasks are reduced to the three fields the coverage check reads. Run it whenever either
changes in the evaluator; `--check` exits 1 if the vendored copies have drifted (tests/test_parity.py
runs that when the evaluator checkout is present).
"""

from __future__ import annotations

import argparse
import json
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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--evaluator", type=Path, default=ROOT.parent / "microduck-evaluator")
    ap.add_argument("--check", action="store_true", help="report drift instead of writing")
    args = ap.parse_args()

    wanted = {PKG / "submission.py": vendored_submission(args.evaluator),
              PKG / "ingestion.py": vendored_ingestion(args.evaluator),
              PKG / "tasks.json": task_snapshot(args.evaluator)}
    drifted = [p for p, text in wanted.items() if not p.exists() or p.read_text() != text]
    if args.check:
        for p in drifted:
            print(f"out of date: {p.relative_to(ROOT)}", file=sys.stderr)
        return 1 if drifted else 0
    for p in drifted:
        p.write_text(wanted[p])
        print(f"wrote {p.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
