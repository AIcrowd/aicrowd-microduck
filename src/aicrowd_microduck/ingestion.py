"""Fetch and safely unpack a submission (tar.gz / zip / directory), then validate it.

VENDORED from the evaluator (microduck-evaluator, src/mdeval/worker/ingestion.py): the worker's
own unpacking rules. Only the mdeval imports changed; regenerate with scripts/sync_from_evaluator.py.

Guards: raw size cap, member count cap, total uncompressed size cap (computed before extracting),
regular files and directories only (no symlinks, devices, FIFOs), resolved-path containment.
Every exception here is caught by the caller and reported as a SubmissionErrorCode — this runs in
the trusted worker before any graph is loaded for real.

The submission format itself (manifest + ONNX graphs, no participant Python) lives in
``mdeval.common.submission`` so that ``mdeval validate`` gives a participant the same verdict.
"""
from __future__ import annotations

import io
import shutil
import tarfile
import zipfile
from pathlib import Path

from .submission import PolicyManifest, SubmissionErrorCode, SubmissionRejected, load_manifest

MAX_SUBMISSION_BYTES = 500 * 1024 * 1024
MAX_FILES = 2000


FETCH_ATTEMPTS = 4


def fetch(source: str, dest_file: Path) -> Path:
    """``source`` is a local path (file or directory) or an http(s) URL (presigned S3 etc.).
    Transient network/5xx failures are retried; only a size-cap breach blames the participant."""
    if source.startswith(("http://", "https://")):
        import time
        import httpx
        last: Exception | None = None
        for attempt in range(FETCH_ATTEMPTS):
            size = 0
            try:
                with httpx.stream("GET", source, timeout=60.0, follow_redirects=True) as r:
                    if r.status_code in (429, 500, 502, 503, 504):
                        raise httpx.HTTPStatusError(f"status {r.status_code}", request=r.request, response=r)
                    r.raise_for_status()
                    with open(dest_file, "wb") as f:
                        for chunk in r.iter_bytes(64 * 1024):
                            size += len(chunk)
                            if size > MAX_SUBMISSION_BYTES:
                                raise SubmissionRejected(SubmissionErrorCode.SUBMISSION_TOO_LARGE, f"> {MAX_SUBMISSION_BYTES} bytes")
                            f.write(chunk)
                return dest_file
            except SubmissionRejected:
                raise
            except httpx.HTTPStatusError as e:
                last = e
                if e.response is not None and 400 <= e.response.status_code < 500 and e.response.status_code != 429:
                    break                                        # 403 (expired URL) / 404: retrying cannot help
            except Exception as e:  # noqa: BLE001
                last = e
            time.sleep(0.5 * 2 ** attempt)
        raise SubmissionRejected(SubmissionErrorCode.SUBMISSION_FETCH_FAILED,
                                 f"fetch failed after {FETCH_ATTEMPTS} attempts: {type(last).__name__}") from last
    p = Path(source)
    if not p.exists():
        raise SubmissionRejected(SubmissionErrorCode.SUBMISSION_FETCH_FAILED, f"{source} does not exist")
    return p


def _contained(dest: Path, member_path: str) -> Path:
    if member_path.startswith("/") or ".." in Path(member_path).parts:
        raise SubmissionRejected(SubmissionErrorCode.UNSAFE_ARCHIVE_MEMBER, f"unsafe member path {member_path!r}")
    target = (dest / member_path).resolve()
    if dest.resolve() not in target.parents and target != dest.resolve():
        raise SubmissionRejected(SubmissionErrorCode.UNSAFE_ARCHIVE_MEMBER, f"member escapes destination: {member_path!r}")
    return target


def extract(src: Path, dest: Path) -> Path:
    """Unpack ``src`` into ``dest``; returns the directory that holds the submission files."""
    dest.mkdir(parents=True, exist_ok=True)
    if src.is_dir():
        n = 0
        for p in src.rglob("*"):
            if p.is_symlink():
                raise SubmissionRejected(SubmissionErrorCode.UNSAFE_ARCHIVE_MEMBER, f"symlink {p.name}")
            n += 1
            if n > MAX_FILES:
                raise SubmissionRejected(SubmissionErrorCode.FILE_COUNT_EXCEEDED, f"> {MAX_FILES} files")
        shutil.copytree(src, dest, dirs_exist_ok=True, symlinks=False)
        return _strip_single_root(dest)
    if src.stat().st_size > MAX_SUBMISSION_BYTES:
        raise SubmissionRejected(SubmissionErrorCode.SUBMISSION_TOO_LARGE, f"archive > {MAX_SUBMISSION_BYTES} bytes")
    with src.open("rb") as fh:
        head = fh.read(4)
    if head[:2] == b"PK":
        _extract_zip(src, dest)
    else:
        _extract_tar(src, dest)
    return _strip_single_root(dest)


def _strip_single_root(dest: Path) -> Path:
    entries = [p for p in dest.iterdir() if not p.name.startswith(".")]
    # an archive that wraps everything in one directory: descend into it, unless the manifest is
    # already here (in which case that directory is part of the submission, not a wrapper)
    if len(entries) == 1 and entries[0].is_dir() and not (dest / "manifest.json").exists():
        return entries[0]
    return dest


def _extract_tar(src: Path, dest: Path) -> None:
    try:
        tf = tarfile.open(src, mode="r:*")
    except tarfile.TarError as e:
        raise SubmissionRejected(SubmissionErrorCode.MANIFEST_INVALID, f"not a tar archive: {e}") from e
    with tf:
        members = []
        total = 0
        for i, m in enumerate(tf):                   # incremental: never materialise a giant member list
            if i >= MAX_FILES:
                raise SubmissionRejected(SubmissionErrorCode.FILE_COUNT_EXCEEDED, f"> {MAX_FILES} files")
            if m.issym() or m.islnk() or not (m.isfile() or m.isdir()):
                raise SubmissionRejected(SubmissionErrorCode.UNSAFE_ARCHIVE_MEMBER, f"member {m.name!r} is not a regular file/dir")
            total += m.size
            if total > MAX_SUBMISSION_BYTES:
                raise SubmissionRejected(SubmissionErrorCode.SUBMISSION_TOO_LARGE, "uncompressed size exceeds cap")
            _contained(dest, m.name)
            members.append(m)
        for m in members:
            target = _contained(dest, m.name)
            if m.isdir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                f = tf.extractfile(m)
                assert f is not None
                with open(target, "wb") as out:
                    shutil.copyfileobj(f, out)
                target.chmod((m.mode & 0o755) | 0o600)          # keep the executable bit, never setuid/world-writable


def _extract_zip(src: Path, dest: Path) -> None:
    try:
        zf = zipfile.ZipFile(src)
    except zipfile.BadZipFile as e:
        raise SubmissionRejected(SubmissionErrorCode.MANIFEST_INVALID, f"not a zip archive: {e}") from e
    with zf:
        infos = zf.infolist()
        if len(infos) > MAX_FILES:
            raise SubmissionRejected(SubmissionErrorCode.FILE_COUNT_EXCEEDED, f"> {MAX_FILES} files")
        if sum(i.file_size for i in infos) > MAX_SUBMISSION_BYTES:
            raise SubmissionRejected(SubmissionErrorCode.SUBMISSION_TOO_LARGE, "uncompressed size exceeds cap")
        for i in infos:
            mode = (i.external_attr >> 16) & 0o170000
            if mode == 0o120000:
                raise SubmissionRejected(SubmissionErrorCode.UNSAFE_ARCHIVE_MEMBER, f"symlink {i.filename!r}")
            target = _contained(dest, i.filename)
            if i.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(i) as f, open(target, "wb") as out:
                shutil.copyfileobj(f, out)


def prepare(source: str, work_dir: Path) -> tuple[Path, PolicyManifest]:
    """Fetch, unpack and read the manifest. The graphs themselves are validated once the worker
    knows the competition's tasks (``validate_submission`` in ``worker/main.py``), so a missing
    task or a wrong action scale is reported with the same submission-error machinery."""
    work_dir.mkdir(parents=True, exist_ok=True)
    fetched = fetch(source, work_dir / "submission.bin")
    sub_dir = extract(fetched, work_dir / "submission")
    return sub_dir, load_manifest(sub_dir)
