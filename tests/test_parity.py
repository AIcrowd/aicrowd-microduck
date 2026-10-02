"""The vendored validator, ingestion and task list must match the evaluator they were copied from."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
EVALUATOR = Path(os.environ.get("MDEVAL_REPO", ROOT.parent / "microduck-evaluator"))


@pytest.mark.skipif(not (EVALUATOR / "src/mdeval/common/submission.py").exists(),
                    reason=f"no evaluator checkout at {EVALUATOR} (set MDEVAL_REPO)")
def test_vendored_copies_match_the_evaluator():
    r = subprocess.run([sys.executable, str(ROOT / "scripts/sync_from_evaluator.py"), "--check",
                        "--evaluator", str(EVALUATOR)], capture_output=True, text=True)
    assert r.returncode == 0, f"{r.stderr}\nrun: python scripts/sync_from_evaluator.py"
