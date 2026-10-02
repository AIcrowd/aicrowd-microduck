"""`aicrowd-microduck`: validate, pack and submit Microduck policies to AIcrowd.

Submitting talks to the AIcrowd Rails API directly (identity, eligibility, presign, S3 upload,
create), the same path `whest submit` uses. It does not go through aicrowd-cli or api.aicrowd.com.
Once AIcrowd has the submission, the evaluator picks it up and reports the score back to it.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
import time
import zipfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional

from . import __version__

DEFAULT_CHALLENGE = "microduck-sim2real-challenge-2026"
# The evaluator's gateway refuses to register anything larger (MDEVAL_MAX_SUBMISSION_BYTES); the
# worker's own 500 MB cap applies to the unpacked contents and is checked by the vendored ingestion.
MAX_ARCHIVE_BYTES = 256 * 1024 * 1024
POLL_INTERVAL_S = 10.0
TERMINAL = {"graded", "failed"}


# ------------------------------------------------------------------------------------- output
class Say:
    """Human output on stderr, so stdout stays clean for --json."""

    def __init__(self, quiet: bool = False) -> None:
        self.quiet = quiet

    def _p(self, prefix: str, msg: str) -> None:
        if not self.quiet:
            print(f"{prefix}{msg}", file=sys.stderr)

    def step(self, msg: str) -> None:
        self._p("• ", msg)

    def ok(self, msg: str) -> None:
        self._p("✓ ", msg)

    def warn(self, msg: str) -> None:
        self._p("✗ ", msg)

    def hint(self, msg: str) -> None:
        self._p("  ", msg)


def _emit_json(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, indent=1, default=str))


def _plain(html: Optional[str]) -> str:
    """AIcrowd stores grading_message as HTML; show it as text."""
    return re.sub(r"<[^>]+>", "", html or "").strip()


def submission_url(challenge: str, sub_id: Any) -> str:
    return f"https://www.aicrowd.com/challenges/{challenge}/submissions/{sub_id}"


# ------------------------------------------------------------------------------- submissions
def competition_tasks() -> tuple[str, list[SimpleNamespace]]:
    """The live competition's tasks (snapshot made by scripts/sync_from_evaluator.py)."""
    snap = json.loads((Path(__file__).parent / "tasks.json").read_text())
    return snap["competition"], [SimpleNamespace(**t) for t in snap["tasks"]]


def pack(submission_dir: Path, out: Path) -> Path:
    """Zip a submission directory with its files at the archive root.

    Same layout as `mdeval pack --zip`: manifest.json must be at the root, because a wrapping folder
    is the most common way a submission fails to load. Hidden files, __pycache__ and __MACOSX are
    left out.
    """
    files = sorted(
        p for p in submission_dir.rglob("*")
        if p.is_file() and not any(part.startswith(".") or part in ("__pycache__", "__MACOSX")
                                   for part in p.relative_to(submission_dir).parts)
    )
    if not (submission_dir / "manifest.json").is_file():
        raise ValueError(f"{submission_dir} has no manifest.json at its top level")
    out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in files:
            zf.write(f, arcname=str(f.relative_to(submission_dir)))
    return out


def check_archive(archive: Path) -> dict[str, Any]:
    """Run the evaluator's own acceptance checks on an archive.

    Unpacks with the worker's ingestion rules and validates against every task of the live
    competition, so a pass here is a submission the evaluator will accept. Raises
    SubmissionRejected with the evaluator's error code otherwise.
    """
    from .ingestion import extract
    from .submission import SubmissionErrorCode, SubmissionRejected, validate_submission

    size = archive.stat().st_size
    if size > MAX_ARCHIVE_BYTES:
        raise SubmissionRejected(
            SubmissionErrorCode.SUBMISSION_TOO_LARGE,
            f"archive is {size} bytes; the evaluator accepts at most {MAX_ARCHIVE_BYTES}",
        )
    competition, tasks = competition_tasks()
    with tempfile.TemporaryDirectory(prefix="aicrowd-microduck-") as tmp:
        sub_dir = extract(archive, Path(tmp) / "submission")
        report = validate_submission(sub_dir, tasks=tasks)
    report["competition"] = competition
    report["archive"] = {"path": str(archive), "bytes": size}
    return report


def _describe_report(report: dict[str, Any], say: Say) -> None:
    for file, g in report["graphs"].items():
        kind = f"LSTM {g['layers']}x{g['hidden']}" if g["recurrent"] else "feed-forward"
        batch = "dynamic batch" if g["dynamic_batch"] else "batch 1"
        say.hint(f"{file}: {kind}, {batch}")
    used = sorted(set((report.get("tasks") or {}).values()))
    say.hint(f"covers all {len(report.get('tasks') or {})} tasks of {report['competition']} "
             f"with {len(used)} graph{'s' if len(used) != 1 else ''}")
    if report.get("unused_task_ids"):
        say.hint(f"note: the manifest names task(s) this competition does not run: "
                 f"{report['unused_task_ids']}")


class _Prepared:
    """A path given on the command line, as an archive ready to upload (packed if it was a directory)."""

    def __init__(self, path: Path, tmp: Optional[tempfile.TemporaryDirectory]) -> None:
        self.path = path
        self._tmp = tmp

    def cleanup(self) -> None:
        if self._tmp is not None:
            self._tmp.cleanup()


def _prepare(path: Path) -> _Prepared:
    if path.is_dir():
        tmp = tempfile.TemporaryDirectory(prefix="aicrowd-microduck-pack-")
        name = path.resolve().name or "submission"
        return _Prepared(pack(path, Path(tmp.name) / f"{name}.zip"), tmp)
    if path.is_file():
        return _Prepared(path, None)
    raise FileNotFoundError(f"no such file or directory: {path}")


def _macosx_hint(archive: Path) -> Optional[str]:
    """A zip made by Finder or `ditto` holds a __MACOSX/ folder beside the wrapping folder. The
    evaluator only unwraps a single top-level folder, so it then cannot find manifest.json."""
    try:
        with zipfile.ZipFile(archive) as zf:
            if any(n.startswith("__MACOSX/") for n in zf.namelist()):
                return ("the archive has a __MACOSX/ folder (macOS Finder adds it). Submit the directory "
                        "itself, or re-pack it with `aicrowd-microduck pack`.")
    except (zipfile.BadZipFile, OSError):
        pass
    return None


def _validate_or_report(path: Path, say: Say, json_output: bool) -> tuple[Optional[_Prepared], Optional[dict], int]:
    """Prepare and check `path`. Returns (prepared, report, exit code); exit code 0 means it passed."""
    from .submission import SubmissionRejected

    try:
        prepared = _prepare(path)
    except (FileNotFoundError, ValueError) as e:
        if json_output:
            _emit_json({"ok": False, "error": str(e)})
        say.warn(str(e))
        return None, None, 2
    try:
        report = check_archive(prepared.path)
    except SubmissionRejected as e:
        hint = _macosx_hint(prepared.path) if e.code.value == "MANIFEST_INVALID" else None
        prepared.cleanup()
        if json_output:
            _emit_json({"ok": False, "error": e.message, "error_code": e.code.value, "hint": hint})
        say.warn(f"REJECTED [{e.code.value}] {e.message}")
        if hint:
            say.hint(hint)
        return None, None, 1
    return prepared, report, 0


# ----------------------------------------------------------------------------------- commands
def cmd_login(args, say: Say) -> int:
    import getpass

    from . import aicrowd_config as cfg
    from .aicrowd_client import AIcrowdClient, describe_error

    api_key = args.api_key or os.environ.get("AICROWD_API_KEY")
    if not api_key:
        if args.json:
            _emit_json({"ok": False, "error": "no api key provided"})
            return 2
        print("Copy your API key from https://www.aicrowd.com/participants/me", file=sys.stderr)
        api_key = getpass.getpass("AIcrowd API key: ").strip()
    if not api_key:
        say.warn("No API key entered; aborting.")
        return 2
    try:
        ident = AIcrowdClient(api_key=api_key).whoami()
    except Exception as e:  # noqa: BLE001 - AIcrowdAPIError or a transport error
        info = describe_error(e)
        if args.json:
            _emit_json({"ok": False, "error": info["message"], "error_code": info["code"]})
        say.warn(info["message"])
        say.hint(info["hint"] or "Copy your key from your AIcrowd profile page and try again.")
        return 1
    path = cfg.save_api_key(api_key)
    if args.json:
        _emit_json({"ok": True, "username": ident.get("username"), "id": ident.get("id"),
                    "config_path": str(path)})
    say.ok(f"Logged in as {ident.get('username') or ident.get('id')}")
    say.hint(f"Key saved to {path} (shared with aicrowd-cli)")
    return 0


def cmd_validate(args, say: Say) -> int:
    prepared, report, code = _validate_or_report(Path(args.path), say, args.json)
    if code:
        return code
    prepared.cleanup()
    if args.json:
        _emit_json({"ok": True, **report})
    say.ok("Valid: the evaluator will accept this submission")
    _describe_report(report, say)
    return 0


def cmd_pack(args, say: Say) -> int:
    d = Path(args.dir)
    out = Path(args.out) if args.out else d.resolve().with_suffix(".zip")
    try:
        pack(d, out)
    except ValueError as e:
        say.warn(str(e))
        return 2
    say.ok(f"Packed {out} ({out.stat().st_size} bytes)")
    print(out)
    return 0


def cmd_submit(args, say: Say) -> int:
    from . import aicrowd_config as cfg
    from .aicrowd_client import AIcrowdClient, describe_error, extract_submission_id

    try:
        api_key = cfg.resolve_api_key(args.api_key)
    except cfg.NotLoggedIn as e:
        if args.json:
            _emit_json({"ok": False, "error": str(e)})
        say.warn(str(e))
        return 2

    say.step(f"Checking {args.path}")
    if args.no_validate:
        try:
            prepared, report = _prepare(Path(args.path)), None
        except (FileNotFoundError, ValueError) as e:
            say.warn(str(e))
            return 2
        say.hint("skipping local validation (--no-validate)")
    else:
        prepared, report, code = _validate_or_report(Path(args.path), say, args.json)
        if code:
            say.hint("Fix it and try again, or see `aicrowd-microduck validate --help`.")
            return code
        say.ok("Valid")
        _describe_report(report, say)

    try:
        client = AIcrowdClient(api_key=api_key)
        try:
            ident = client.whoami()
            say.ok(f"Authenticated as {ident.get('username') or ident.get('id')}")
            elig = client.check_eligibility(challenge_slug=args.challenge)
            if not elig.get("submissions_allowed", True):
                msg = elig.get("message") or f"AIcrowd will not accept submissions to '{args.challenge}' right now."
                if args.json:
                    _emit_json({"ok": False, "error": msg, "error_code": elig.get("denied_reason"),
                                "rules_url": elig.get("rules_url")})
                say.warn(msg)
                rules_url = elig.get("rules_url")
                on_terms = not elig.get("rules_accepted", True) or not elig.get("participation_terms_accepted", True)
                if on_terms and rules_url and rules_url not in msg:
                    say.hint(f"Accept the challenge rules at {rules_url}")
                return 1
            if args.dry_run:
                client.get_upload_details(challenge_slug=args.challenge)  # proves the presign would work
                if args.json:
                    _emit_json({"ok": True, "dry_run": True, "archive": str(prepared.path),
                                "bytes": prepared.path.stat().st_size, "challenge": args.challenge})
                say.ok(f"Dry run: AIcrowd would accept {prepared.path.name} "
                       f"({prepared.path.stat().st_size} bytes) for {args.challenge}; nothing uploaded")
                return 0
            upload = client.get_upload_details(challenge_slug=args.challenge)
            say.step(f"Uploading {prepared.path.name} ({prepared.path.stat().st_size} bytes)")
            s3_key = client.upload_to_s3(upload=upload, file_path=str(prepared.path))
            sub = client.create_submission(challenge_slug=args.challenge, s3_key=s3_key,
                                           description=args.description)
        except Exception as e:  # noqa: BLE001
            info = describe_error(e)
            if args.json:
                _emit_json({"ok": False, "error": info["message"], "error_code": info["code"],
                            "status": info["status"]})
            say.warn(info["message"])
            if info["hint"]:
                say.hint(info["hint"])
            return 1
    finally:
        prepared.cleanup()

    sub_id = extract_submission_id(sub)
    say.ok(f"Submitted: submission id {sub_id}")
    say.hint(f"Track it at {submission_url(args.challenge, sub_id)}")
    final: dict[str, Any] = {"grading_status_cd": "submitted"}
    if args.watch and sub_id is not None:
        final = watch(client, sub_id, challenge=args.challenge, timeout=args.watch_timeout, say=say)
    else:
        say.hint(f"Check the score with `aicrowd-microduck status {sub_id} --watch`")
    if args.json:
        _emit_json({"ok": True, "submission_id": sub_id, "status": final})
    return 1 if str(final.get("grading_status_cd")) == "failed" else 0


def cmd_status(args, say: Say) -> int:
    from . import aicrowd_config as cfg
    from .aicrowd_client import AIcrowdClient, describe_error

    try:
        client = AIcrowdClient(api_key=cfg.resolve_api_key(args.api_key))
        st = client.get_submission_status(args.submission_id)
    except Exception as e:  # noqa: BLE001 - NotLoggedIn, AIcrowdAPIError or transport
        info = describe_error(e)
        if args.json:
            _emit_json({"ok": False, "error": info["message"], "error_code": info["code"]})
        say.warn(info["message"])
        if info["hint"]:
            say.hint(info["hint"])
        return 1
    if args.watch and str(st.get("grading_status_cd")) not in TERMINAL:
        st = watch(client, args.submission_id, challenge=args.challenge, timeout=args.watch_timeout,
                   say=say, initial=st)
    else:
        _describe_status(st, say)
    if args.json:
        _emit_json({"ok": True, "submission_id": args.submission_id, "status": st})
    return 1 if str(st.get("grading_status_cd")) == "failed" else 0


def _describe_status(st: dict[str, Any], say: Say) -> None:
    status = str(st.get("grading_status_cd"))
    msg = _plain(st.get("grading_message"))
    if status == "graded":
        say.ok(f"Graded: score {st.get('score')}" + (f" ({msg})" if msg else ""))
    elif status == "failed":
        say.warn(f"Failed: {msg or 'no message from the evaluator'}")
    else:
        say.step(f"Status: {status}" + (f" ({msg})" if msg else ""))


def watch(client, sub_id, *, challenge: str, timeout: float, say: Say,
          initial: Optional[dict] = None) -> dict[str, Any]:
    """Poll until graded/failed or `timeout` seconds pass. Never raises; returns the last status seen."""
    from .aicrowd_client import AIcrowdAPIError, AIcrowdTransientError

    st = initial or {}
    last_shown = None
    deadline = time.monotonic() + timeout
    say.step("Waiting for the evaluator (a full evaluation takes several minutes)")
    while True:
        if st:
            shown = (st.get("grading_status_cd"), _plain(st.get("grading_message")))
            if shown != last_shown:
                _describe_status(st, say)
                last_shown = shown
            if str(st.get("grading_status_cd")) in TERMINAL:
                return st
        if time.monotonic() >= deadline:
            say.hint(f"Still grading after {int(timeout)}s; track it at {submission_url(challenge, sub_id)}")
            return st
        time.sleep(POLL_INTERVAL_S)
        try:
            st = client.get_submission_status(int(sub_id))
        except AIcrowdTransientError:
            continue
        except AIcrowdAPIError as e:
            say.hint(f"Could not read the status ({e.summary}); see {submission_url(challenge, sub_id)}")
            return st


# --------------------------------------------------------------------------------------- main
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="aicrowd-microduck",
        description="Validate, pack and submit Microduck policies to the AIcrowd Microduck challenge.",
    )
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    def common(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--json", action="store_true", help="machine-readable output on stdout")

    sp = sub.add_parser("login", help="store your AIcrowd API key (shared with aicrowd-cli)")
    sp.add_argument("--api-key", help="AIcrowd API key; prompted for if omitted")
    common(sp)
    sp.set_defaults(func=cmd_login)

    sp = sub.add_parser("validate", help="run the evaluator's acceptance checks locally",
                        description="Unpack and check a submission exactly as the evaluator's worker does: "
                                    "manifest, ONNX tensor contract, a smoke inference, and coverage of every "
                                    "task in the live competition.")
    sp.add_argument("path", help="submission directory, or a .zip / .tar.gz of one")
    common(sp)
    sp.set_defaults(func=cmd_validate)

    sp = sub.add_parser("pack", help="zip a submission directory (files at the archive root)")
    sp.add_argument("dir", help="submission directory containing manifest.json")
    sp.add_argument("-o", "--out", help="output path (default <dir>.zip)")
    sp.set_defaults(func=cmd_pack)

    sp = sub.add_parser("submit", help="validate, then submit to AIcrowd",
                        description="Validate locally, then submit through the AIcrowd API. A directory is "
                                    "zipped first. Auth comes from `aicrowd-microduck login`, "
                                    "AICROWD_API_KEY, or the aicrowd-cli config.")
    sp.add_argument("path", help="submission directory, or a .zip / .tar.gz of one")
    sp.add_argument("--challenge", default=DEFAULT_CHALLENGE, help=f"challenge slug (default {DEFAULT_CHALLENGE})")
    sp.add_argument("--description", default="Submitted via aicrowd-microduck",
                    help="label shown on the AIcrowd submission")
    sp.add_argument("--watch", action="store_true", help="wait for the score")
    sp.add_argument("--watch-timeout", type=float, default=1800.0, metavar="SECONDS",
                    help="stop waiting after this long (default 1800)")
    sp.add_argument("--dry-run", action="store_true",
                    help="validate, authenticate and check eligibility, but upload nothing")
    sp.add_argument("--no-validate", action="store_true", help=argparse.SUPPRESS)
    sp.add_argument("--api-key", help=argparse.SUPPRESS)
    common(sp)
    sp.set_defaults(func=cmd_submit)

    sp = sub.add_parser("status", help="show a submission's grading status and score")
    sp.add_argument("submission_id", type=int)
    sp.add_argument("--watch", action="store_true", help="wait until graded or failed")
    sp.add_argument("--watch-timeout", type=float, default=1800.0, metavar="SECONDS")
    sp.add_argument("--challenge", default=DEFAULT_CHALLENGE, help=argparse.SUPPRESS)
    sp.add_argument("--api-key", help=argparse.SUPPRESS)
    common(sp)
    sp.set_defaults(func=cmd_status)
    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    say = Say(quiet=getattr(args, "json", False))
    try:
        return int(args.func(args, say))
    except KeyboardInterrupt:
        say.warn("interrupted")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
