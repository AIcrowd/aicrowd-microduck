"""The submission format: a manifest plus ONNX graphs. No participant Python runs anywhere.

VENDORED from the evaluator (microduck-evaluator, src/mdeval/common/submission.py) so that
`aicrowd-microduck validate` gives the grader's own verdict without installing the evaluator.
Only the import of SubmissionErrorCode changed; regenerate with scripts/sync_from_evaluator.py
rather than editing by hand.

This mirrors what the robot's daemon will load, so a submission that passes here is a submission
that a real Microduck can run (see ``docs/SIM_VS_ROBOT.md`` for the gaps that remain). The rules
are deliberately the daemon's, not ours:

- one graph per task, named by the manifest (by task id, else by the task's family, else ``default``);
  the input is ``obs`` float32 with trailing dim 61 and the first output is float32 with trailing dim 14;
- ``model_api`` 1 is feed-forward, 2 adds the explicit LSTM state tensors ``h_in``/``c_in`` ->
  ``h_out``/``c_out`` with positive static layer and hidden dimensions, identical across all four;
- tensors are matched by NAME, not by position, and every tensor is float32;
- the action scale is a property of the evaluation, not of the submission: the manifest must declare
  the same value the task uses, and a mismatch is rejected rather than silently overridden (the
  robot's own clamp absorbs a wrong scale without an error, so nothing downstream would catch it).

Validation runs in the trusted worker before anything is loaded for real, and the same code backs
``mdeval validate`` so a participant sees the identical verdict locally.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from enum import Enum


class SubmissionErrorCode(str, Enum):
    SUBMISSION_FETCH_FAILED = "SUBMISSION_FETCH_FAILED"
    SUBMISSION_TOO_LARGE = "SUBMISSION_TOO_LARGE"
    FILE_COUNT_EXCEEDED = "FILE_COUNT_EXCEEDED"
    UNSAFE_ARCHIVE_MEMBER = "UNSAFE_ARCHIVE_MEMBER"
    MANIFEST_INVALID = "MANIFEST_INVALID"
    POLICY_FILE_MISSING = "POLICY_FILE_MISSING"          # manifest names a graph the submission does not contain
    ONNX_INVALID = "ONNX_INVALID"                        # onnxruntime cannot load or run the graph
    ONNX_SIGNATURE_MISMATCH = "ONNX_SIGNATURE_MISMATCH"  # wrong tensor names / shapes / dtype
    ONNX_OUTPUT_INVALID = "ONNX_OUTPUT_INVALID"          # non-finite or observation-independent output
    ACTION_SCALE_MISMATCH = "ACTION_SCALE_MISMATCH"      # manifest disagrees with the task's fixed scale
    SETUP_FAILED = "SETUP_FAILED"

SCHEMA_VERSION = 1
MODEL_API_FEEDFORWARD = 1
MODEL_API_RECURRENT = 2
OBS_LEN = 61
ACT_LEN = 14

OBS_INPUT = "obs"
STATE_IN = ("h_in", "c_in")
STATE_OUT = ("h_out", "c_out")

MAX_GRAPH_BYTES = 200 * 1024 * 1024


class SubmissionRejected(Exception):
    """A participant-facing rejection: the code is reported verbatim, the message is shown to them."""

    def __init__(self, code: SubmissionErrorCode, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


# --------------------------------------------------------------------------------- manifest
@dataclass
class PolicyManifest:
    model_api: int = MODEL_API_FEEDFORWARD
    action_scale: float = 1.0
    policies: dict[str, str] = field(default_factory=dict)   # task_id or family -> graph file, relative to the submission dir
    default: str | None = None                                # graph for any task not routed above
    schema_version: int = SCHEMA_VERSION
    obs_len: int = OBS_LEN
    action_len: int = ACT_LEN
    name: str = ""
    description: str = ""

    def graph_for(self, task_id: str, family: str | None = None) -> str:
        """The graph for a task: its task id, else its family (v2 levels share one), else ``default``."""
        g = self.policies.get(task_id) or (self.policies.get(family) if family else None) or self.default
        if g is None:
            where = f"task {task_id!r}" + (f" (family {family!r})" if family else "")
            raise SubmissionRejected(SubmissionErrorCode.POLICY_FILE_MISSING,
                                     f"no policy for {where}: name it in manifest 'policies' or set 'default'")
        return g

    def graph_files(self) -> list[str]:
        seen = list(self.policies.values()) + ([self.default] if self.default else [])
        return sorted(dict.fromkeys(seen))

    def to_dict(self) -> dict[str, Any]:
        return {"schema_version": self.schema_version, "model_api": self.model_api, "obs_len": self.obs_len,
                "action_len": self.action_len, "action_scale": self.action_scale, "policies": dict(self.policies),
                "default": self.default, "name": self.name, "description": self.description}


def _reject(msg: str, code: SubmissionErrorCode = SubmissionErrorCode.MANIFEST_INVALID) -> "SubmissionRejected":
    return SubmissionRejected(code, msg)


def _safe_relpath(value: str, directory: Path) -> Path:
    """A manifest may only name a regular file inside the submission directory."""
    if not isinstance(value, str) or not value:
        raise _reject(f"policy path must be a non-empty string, got {value!r}")
    p = Path(value)
    if p.is_absolute() or ".." in p.parts:
        raise _reject(f"policy path {value!r} must be relative and stay inside the submission")
    full = (directory / p).resolve()
    if not str(full).startswith(str(directory.resolve())):
        raise _reject(f"policy path {value!r} escapes the submission directory")
    return full


def load_manifest(directory: str | Path) -> PolicyManifest:
    """Parse and structurally validate ``manifest.json``. Does not open any graph."""
    d = Path(directory)
    p = d / "manifest.json"
    if not p.exists():
        raise _reject("manifest.json is required: it names the ONNX graph for each task")
    try:
        data = json.loads(p.read_text())
    except Exception as e:  # noqa: BLE001
        raise _reject(f"manifest.json is not valid JSON: {e}") from e
    if not isinstance(data, dict):
        raise _reject("manifest.json must be a JSON object")

    unknown = set(data) - set(PolicyManifest().to_dict())
    if unknown:
        raise _reject(f"unknown manifest keys {sorted(unknown)}; allowed: {sorted(PolicyManifest().to_dict())}")

    m = PolicyManifest()
    m.schema_version = int(data.get("schema_version", SCHEMA_VERSION))
    if m.schema_version != SCHEMA_VERSION:
        raise _reject(f"schema_version {m.schema_version} is not supported (this evaluator speaks {SCHEMA_VERSION})")
    m.model_api = int(data.get("model_api", MODEL_API_FEEDFORWARD))
    if m.model_api not in (MODEL_API_FEEDFORWARD, MODEL_API_RECURRENT):
        raise _reject(f"model_api {m.model_api} is not supported (1 = feed-forward, 2 = LSTM)")
    m.obs_len = int(data.get("obs_len", OBS_LEN))
    m.action_len = int(data.get("action_len", ACT_LEN))
    if m.obs_len != OBS_LEN or m.action_len != ACT_LEN:
        raise _reject(f"obs_len/action_len must be {OBS_LEN}/{ACT_LEN}, got {m.obs_len}/{m.action_len}")
    try:
        m.action_scale = float(data.get("action_scale", 1.0))
    except (TypeError, ValueError) as e:
        raise _reject(f"action_scale must be a number: {data.get('action_scale')!r}") from e

    policies = data.get("policies", {})
    if not isinstance(policies, dict) or not all(isinstance(k, str) for k in policies):
        raise _reject("'policies' must be an object mapping task_id or family -> onnx file")
    m.policies = {k: str(v) for k, v in policies.items()}
    m.default = str(data["default"]) if data.get("default") else None
    if not m.policies and not m.default:
        raise _reject("manifest names no policy: set 'policies' (task_id or family -> file) and/or 'default'")
    for value in m.graph_files():
        full = _safe_relpath(value, d)
        if not full.is_file():
            raise SubmissionRejected(SubmissionErrorCode.POLICY_FILE_MISSING,
                                     f"manifest names {value!r} but the submission has no such file")
        if full.stat().st_size > MAX_GRAPH_BYTES:
            raise SubmissionRejected(SubmissionErrorCode.ONNX_INVALID, f"{value}: graph larger than {MAX_GRAPH_BYTES} bytes")

    m.name = str(data.get("name", ""))[:200]
    m.description = str(data.get("description", ""))[:1000]
    return m


# ----------------------------------------------------------------------------------- graphs
@dataclass
class GraphSignature:
    file: str
    dynamic_batch: bool
    recurrent: bool
    layers: int = 0
    hidden: int = 0
    action_output: str = "actions"

    def to_dict(self) -> dict[str, Any]:
        return {"file": self.file, "dynamic_batch": self.dynamic_batch, "recurrent": self.recurrent,
                "layers": self.layers, "hidden": self.hidden, "action_output": self.action_output}


def _dim(shape: list, i: int):
    return shape[i] if i < len(shape) else None


def _is_dynamic(d) -> bool:
    return d is None or isinstance(d, str)


def _static(d) -> int | None:
    return d if isinstance(d, int) and d > 0 else None


def open_session(path: str | Path):
    """One-thread CPU session. Raises SubmissionRejected on anything onnxruntime refuses to load."""
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    so.inter_op_num_threads = 1
    so.log_severity_level = 3
    try:
        return ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
    except Exception as e:  # noqa: BLE001
        raise SubmissionRejected(SubmissionErrorCode.ONNX_INVALID, f"{Path(path).name}: cannot load: {e}"[:400]) from e


def inspect_graph(path: str | Path, model_api: int, session=None) -> GraphSignature:
    """Check the tensor contract and return what the runner needs to know."""
    name = Path(path).name
    sess = session or open_session(path)

    def bad(msg: str):
        return SubmissionRejected(SubmissionErrorCode.ONNX_SIGNATURE_MISMATCH, f"{name}: {msg}")

    inputs = {i.name: i for i in sess.get_inputs()}
    outputs = list(sess.get_outputs())
    if OBS_INPUT not in inputs:
        raise bad(f"no input named {OBS_INPUT!r} (inputs: {sorted(inputs)}); tensors are matched by name")
    for tensor in (*inputs.values(), *outputs):
        if tensor.type != "tensor(float)":
            raise bad(f"tensor {tensor.name!r} is {tensor.type}, every tensor must be float32")

    obs_shape = inputs[OBS_INPUT].shape
    if _static(_dim(obs_shape, len(obs_shape) - 1)) != OBS_LEN:
        raise bad(f"input 'obs' has shape {obs_shape}, the trailing dimension must be {OBS_LEN}")
    act = outputs[0]
    if _static(_dim(act.shape, len(act.shape) - 1)) != ACT_LEN:
        raise bad(f"first output {act.name!r} has shape {act.shape}, the trailing dimension must be {ACT_LEN}")
    dynamic_batch = _is_dynamic(_dim(obs_shape, 0)) if len(obs_shape) > 1 else False

    recurrent = any(n in inputs for n in STATE_IN)
    layers = hidden = 0
    if recurrent:
        out_names = {o.name for o in outputs}
        missing = [n for n in (*STATE_IN, *STATE_OUT) if n not in (set(inputs) | out_names)]
        if missing:
            raise bad(f"recurrent graph is missing {missing}; the contract is obs, h_in, c_in -> actions, h_out, c_out")
        if model_api != MODEL_API_RECURRENT:
            raise bad(f"graph carries LSTM state but the manifest declares model_api {model_api}; recurrent policies need 2")
        shapes = [inputs[n].shape for n in STATE_IN] + [o.shape for o in outputs if o.name in STATE_OUT]
        for s in shapes:
            if len(s) != 3:
                raise bad(f"state tensors must be rank 3 [layers, batch, hidden], got {s}")
        layer_dims = {_static(_dim(s, 0)) for s in shapes}
        hidden_dims = {_static(_dim(s, 2)) for s in shapes}
        if None in layer_dims or len(layer_dims) != 1:
            raise bad(f"layer dimension must be one positive static value across all state tensors, got {layer_dims}")
        if None in hidden_dims or len(hidden_dims) != 1:
            raise bad(f"hidden width must be one positive static value across all state tensors, got {hidden_dims}")
        layers, hidden = layer_dims.pop(), hidden_dims.pop()
    elif model_api == MODEL_API_RECURRENT and any(n in {o.name for o in outputs} for n in STATE_OUT):
        raise bad("graph returns LSTM state but does not accept it as input")

    return GraphSignature(file=name, dynamic_batch=dynamic_batch, recurrent=recurrent, layers=layers,
                          hidden=hidden, action_output=act.name)


def plausible_observations(n: int = 3, seed: int = 0) -> np.ndarray:
    """Observations a standing duck could actually produce, for a smoke run of the graph.

    Slot layout is ``mdeval_sdk.OBS_LAYOUT``: gyro, projected gravity, joint pos/vel relative to HOME,
    last action, then the command block. Row 0 is a still, upright robot; the rest perturb it, which is
    what makes the constant-output check meaningful.
    """
    rng = np.random.default_rng(seed)
    obs = np.zeros((n, OBS_LEN), dtype=np.float32)
    obs[:, 3:6] = np.array([0.0, 0.0, -1.0], dtype=np.float32)          # upright
    if n > 1:
        obs[1:, 0:3] = rng.normal(0.0, 0.2, (n - 1, 3))                 # gyro
        obs[1:, 6:20] = rng.normal(0.0, 0.1, (n - 1, 14))               # joint pos around HOME
        obs[1:, 20:34] = rng.normal(0.0, 0.5, (n - 1, 14))              # joint vel
        obs[1:, 48:51] = rng.uniform(-0.3, 0.3, (n - 1, 3))             # twist command
    return obs


def smoke_run(path: str | Path, sig: GraphSignature, session=None) -> dict[str, Any]:
    """Run the graph on plausible inputs; reject NaN, infinity, and an output that never changes.

    A constant graph is the classic broken export (a checkpoint converted without its normalizer, or
    a dead network) and it would otherwise show up as a mysteriously terrible score.
    """
    name = Path(path).name
    sess = session or open_session(path)
    obs = plausible_observations()
    feeds_extra: dict[str, np.ndarray] = {}
    if sig.recurrent:
        zeros = np.zeros((sig.layers, 1, sig.hidden), dtype=np.float32)
        feeds_extra = {STATE_IN[0]: zeros, STATE_IN[1]: zeros.copy()}
    out_names = [sig.action_output]
    rows = []
    for i in range(obs.shape[0]):
        try:
            res = sess.run(out_names, {OBS_INPUT: obs[i : i + 1], **feeds_extra})
        except Exception as e:  # noqa: BLE001
            raise SubmissionRejected(SubmissionErrorCode.ONNX_INVALID,
                                     f"{name}: inference failed on a plausible observation: {e}"[:400]) from e
        a = np.asarray(res[0], dtype=np.float32).reshape(-1)
        if a.size != ACT_LEN:
            raise SubmissionRejected(SubmissionErrorCode.ONNX_SIGNATURE_MISMATCH,
                                     f"{name}: returned {a.size} actions, expected {ACT_LEN}")
        if not np.all(np.isfinite(a)):
            raise SubmissionRejected(SubmissionErrorCode.ONNX_OUTPUT_INVALID,
                                     f"{name}: produced a non-finite action on a plausible observation")
        rows.append(a)
    stacked = np.stack(rows)
    if np.allclose(stacked, stacked[0], atol=0.0, rtol=0.0):
        raise SubmissionRejected(SubmissionErrorCode.ONNX_OUTPUT_INVALID,
                                 f"{name}: output does not depend on the observation (identical actions for "
                                 "different inputs) — usually an export that lost its observation normalizer")
    return {"action_abs_max": float(np.abs(stacked).max()), "action_std": float(stacked.std())}


# ------------------------------------------------------------------------------- full check
def validate_submission(directory: str | Path, tasks: list | None = None) -> dict[str, Any]:
    """Manifest + every graph + a smoke run, and if ``tasks`` is given, the per-task cross-checks.

    ``tasks`` is a list of objects with ``task_id``, ``action_scale`` and optionally ``family`` (the
    protocol's ``TaskDescriptor``), so the worker can reject a submission that does not cover every task
    or that disagrees with the evaluation's action scale. A task is covered by its task id, its family or
    the manifest's ``default``.
    """
    d = Path(directory)
    manifest = load_manifest(d)
    graphs: dict[str, dict[str, Any]] = {}
    for file in manifest.graph_files():
        sess = open_session(d / file)
        sig = inspect_graph(d / file, manifest.model_api, session=sess)
        stats = smoke_run(d / file, sig, session=sess)
        graphs[file] = {**sig.to_dict(), **stats}
        del sess
    if manifest.model_api == MODEL_API_RECURRENT and not any(g["recurrent"] for g in graphs.values()):
        raise _reject("manifest declares model_api 2 but no graph carries LSTM state; use 1 for feed-forward")

    covered = None
    if tasks is not None:
        missing = [t.task_id for t in tasks if t.task_id not in manifest.policies
                   and getattr(t, "family", None) not in manifest.policies and not manifest.default]
        if missing:
            raise SubmissionRejected(SubmissionErrorCode.POLICY_FILE_MISSING,
                                     f"no policy for task(s) {missing}: name them (or their family) in 'policies' "
                                     "or set 'default'")
        for t in tasks:
            if abs(float(t.action_scale) - manifest.action_scale) > 1e-9:
                raise SubmissionRejected(
                    SubmissionErrorCode.ACTION_SCALE_MISMATCH,
                    f"task {t.task_id!r} is evaluated at action_scale {t.action_scale}, the manifest declares "
                    f"{manifest.action_scale}. The scale is fixed by the task; fold the difference into your "
                    "graph's output instead of declaring a different one.")
        covered = {t.task_id: manifest.graph_for(t.task_id, getattr(t, "family", None)) for t in tasks}
    # A task id (or family) the manifest names but this competition does not run is reported, not rejected:
    # the same submission is meant to run against the smoke config and the full competition alike.
    unused = (sorted(set(manifest.policies) - {t.task_id for t in tasks} - {getattr(t, "family", None) for t in tasks})
              if tasks is not None else [])
    return {"manifest": manifest.to_dict(), "graphs": graphs, "tasks": covered, "unused_task_ids": unused}
