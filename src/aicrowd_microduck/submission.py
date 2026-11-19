"""The submission format: a manifest plus ONNX graphs. No participant Python runs anywhere.

VENDORED from the evaluator (microduck-evaluator, src/mdeval/common/submission.py) so that
`aicrowd-microduck validate` gives the grader's own verdict without installing the evaluator.
Only the import of SubmissionErrorCode changed; regenerate with scripts/sync_from_evaluator.py
rather than editing by hand.

This is **schema 2 of the Microduck policy manifest** — upstream's format, not a format of our own.
The contract lives in ``pollen-robotics/microduck``, ``docs/policy-manifest.md``, is written by
``uv run publish`` in ``pollen-robotics/microduck_rl``, and is read by the robot's ``updaterd`` and
``robotd``. We are a *superset*: everything upstream writes, we accept, plus the task routing a
competition needs (``policies`` / ``default``). So a submission that scores here is a policy a real
Microduck can install, and a published Hub policy repo is a valid submission as it stands. See
``docs/SIM_VS_ROBOT.md`` for the gaps that remain.

The rules are deliberately the daemon's, not ours:

- one graph per task, named by the manifest (by task id, else by the task's family, else ``default``);
  a repo that carries a single ``.onnx`` and routes nothing is that graph for every task, which is
  how the daemon reads a single-policy repo;
- the input is ``obs`` float32 with trailing dim 61 and the action output float32 with trailing dim 14;
- ``model_api`` 1 is feed-forward, 2 adds the explicit LSTM state tensors ``h_in``/``c_in`` ->
  ``h_out``/``c_out`` with positive static layer and hidden dimensions, identical across all four.
  The numbers are upstream's: a recurrent policy published for the robot declares 2 here too;
- tensors are matched by NAME, not by position; every tensor is float32; a batch dimension is
  symbolic or exactly 1; extra tensors are refused, as the daemon refuses them at load;
- **absent fields are not evidence.** Following upstream, a submission is refused for a claim that
  is present and wrong, never for a field we do not recognise — otherwise every field upstream adds
  to its own manifest would break submissions here;
- the action scale is the one place we are stricter, and deliberately: upstream treats
  ``action_scale`` as a property of the policy, but here it is a property of the *evaluation*. The
  manifest must declare the value the task uses, and a mismatch is rejected rather than silently
  overridden (the robot's own clamp absorbs a wrong scale without an error, so nothing downstream
  would catch it).

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

# Upstream's schema version. 2 is a superset of 1 and upstream's own rule is that nothing gates on
# it, so we accept either and write 2.
SCHEMA_VERSION = 2
SCHEMA_VERSIONS_ACCEPTED = (1, 2)
MODEL_API_FEEDFORWARD = 1
MODEL_API_RECURRENT = 2
OBS_LEN = 61
ACT_LEN = 14

OBS_INPUT = "obs"
STATE_IN = ("h_in", "c_in")
STATE_OUT = ("h_out", "c_out")

# What the robot this challenge is for actually is. Upstream refuses a policy whose manifest claims
# another robot; so do we, because such a submission is a mistake rather than a bad policy.
ROBOT_MODEL = "microduck"
CONTROL_HZ = 50

MAX_GRAPH_BYTES = 200 * 1024 * 1024
# Per upstream's recurrent-policy contract: each state tensor is capped at 2**20 elements. A graph
# over it loads here and is refused on the robot, which is the one outcome this format exists to
# prevent.
MAX_STATE_ELEMENTS = 1_048_576

# Fields of upstream schema 2 that describe a policy rather than route one. We keep them verbatim so
# a submission can be republished as a Hub policy repo without losing anything, and so an organizer
# can see what a participant declared. None of them change how a submission is scored.
#
# `policy-manifest.md` calls these display-or-skill fields; the daemon reads some of them when it
# installs a skill, and we are not the daemon. Carried, not interpreted.
PASSTHROUGH_FIELDS = (
    "kind", "slot", "robot", "training", "eval", "command", "duration_s", "chain",
    "unwind_s", "ramp_s", "entry_pose", "mode",
)


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
    #: Upstream schema-2 fields the participant declared, carried verbatim and not interpreted here
    #: (see ``PASSTHROUGH_FIELDS``). Kept so a scored submission can be republished as a Hub policy
    #: repo unchanged, and so an organizer can see what was claimed.
    declared: dict[str, Any] = field(default_factory=dict)
    #: Keys we did not recognise at all. Not a rejection — upstream's rule is that an unknown field
    #: is not evidence — but worth naming back to a participant whose routing key was a typo.
    unknown_keys: tuple[str, ...] = ()

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
        """The round-trip form: every key is a constructor argument, because the worker rebuilds the
        manifest from this across a process boundary. For a publishable manifest use
        :meth:`to_manifest_json`."""
        return {"schema_version": self.schema_version, "model_api": self.model_api, "obs_len": self.obs_len,
                "action_len": self.action_len, "action_scale": self.action_scale, "policies": dict(self.policies),
                "default": self.default, "name": self.name, "description": self.description,
                "declared": dict(self.declared), "unknown_keys": tuple(self.unknown_keys)}

    def to_manifest_json(self) -> dict[str, Any]:
        """A ``manifest.json`` conforming to upstream schema 2, so what we write is directly
        publishable as a Hub policy repo (``robotctl policy load <slot> <repo>``).

        ``policies``/``default`` ride along as the routing extension: upstream ignores fields it does
        not read, which is exactly the property that lets one file serve both.
        """
        out: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "model_api": self.model_api,
            "obs_len": self.obs_len,
            "action_len": self.action_len,
            "robot": {"model": ROBOT_MODEL, "hw_rev": 1, "servos": "xl330", "control_hz": CONTROL_HZ},
            "action_scale": self.action_scale,
        }
        out.update({k: v for k, v in self.declared.items() if k != "robot"})
        if "robot" in self.declared and isinstance(self.declared["robot"], dict):
            out["robot"] = {**out["robot"], **self.declared["robot"]}
        if self.name:
            out["name"] = self.name
        if self.description:
            out["description"] = self.description
        if self.policies:
            out["policies"] = dict(self.policies)
        if self.default:
            out["default"] = self.default
        return out


def _reject(msg: str, code: SubmissionErrorCode = SubmissionErrorCode.MANIFEST_INVALID) -> "SubmissionRejected":
    return SubmissionRejected(code, msg)


def _as_int(data: dict[str, Any], key: str, default: int) -> int:
    """An integer claim, or a participant-facing rejection. ``int("2.0")`` raises, and an unhandled
    ValueError here would surface as an evaluator crash rather than as a bad manifest."""
    if key not in data or data[key] is None:
        return default
    try:
        return int(data[key])
    except (TypeError, ValueError) as e:
        raise _reject(f"{key} must be an integer: {data[key]!r}") from e


def _routing_table(policies: Any) -> dict[str, str]:
    """``policies`` in either upstream shape.

    Ours is an object, ``task_id or family -> file``. Upstream's "official set" is an *array* of
    entries, each a ``file`` plus per-policy fields; there the entry's ``name`` is what a client asks
    for, so that is the routing key (falling back to the file's stem, as upstream does). Accepting
    both means a participant can hand us either shape and neither needs converting.
    """
    if isinstance(policies, dict):
        if not all(isinstance(k, str) for k in policies):
            raise _reject("'policies' keys must be strings (a task id or a task family)")
        return {k: str(v) for k, v in policies.items()}
    if isinstance(policies, list):
        table: dict[str, str] = {}
        for i, entry in enumerate(policies):
            if not isinstance(entry, dict) or not entry.get("file"):
                raise _reject(f"'policies'[{i}] must be an object with a 'file' (upstream's policy-set shape)")
            file = str(entry["file"])
            # Upstream's seeder *skips* a malformed entry; we reject it. Silently dropping a policy
            # would change what a participant is scored on without telling them.
            table[str(entry.get("name") or Path(file).stem)] = file
        return table
    raise _reject("'policies' must be an object (task_id or family -> onnx file) or an array of "
                  "{file, name} entries")


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

    # Upstream's rule, adopted verbatim: "absent fields are not evidence: a robot refuses a policy
    # only on a claim that is present and wrong". So an unrecognised key is carried, not rejected --
    # otherwise every field upstream adds to schema 2 would break submissions here, and a policy
    # published with `uv run publish` would be refused for describing itself.
    known = set(PolicyManifest().to_dict()) | set(PASSTHROUGH_FIELDS)
    m = PolicyManifest()
    m.unknown_keys = tuple(sorted(set(data) - known))
    m.declared = {k: data[k] for k in PASSTHROUGH_FIELDS if k in data}

    m.schema_version = _as_int(data, "schema_version", SCHEMA_VERSION)
    if m.schema_version not in SCHEMA_VERSIONS_ACCEPTED:
        raise _reject(f"schema_version {m.schema_version} is not supported (this evaluator speaks "
                      f"{' and '.join(str(v) for v in SCHEMA_VERSIONS_ACCEPTED)} of the Microduck policy manifest)")
    m.model_api = _as_int(data, "model_api", MODEL_API_FEEDFORWARD)
    if m.model_api not in (MODEL_API_FEEDFORWARD, MODEL_API_RECURRENT):
        raise _reject(f"model_api {m.model_api} is not supported (1 = feed-forward, 2 = LSTM)")
    m.obs_len = _as_int(data, "obs_len", OBS_LEN)
    m.action_len = _as_int(data, "action_len", ACT_LEN)
    if m.obs_len != OBS_LEN or m.action_len != ACT_LEN:
        raise _reject(f"obs_len/action_len must be {OBS_LEN}/{ACT_LEN}, got {m.obs_len}/{m.action_len}")
    try:
        m.action_scale = float(data.get("action_scale", 1.0))
    except (TypeError, ValueError) as e:
        raise _reject(f"action_scale must be a number: {data.get('action_scale')!r}") from e

    # `robot.model` is a claim, so a wrong one is refused -- exactly where the daemon refuses it.
    # The rest of the block (hw_rev, servos, control_hz) is upstream's display data; we carry it.
    robot = data.get("robot")
    if isinstance(robot, dict) and robot.get("model") not in (None, ROBOT_MODEL):
        raise _reject(f"robot.model is {robot.get('model')!r}; this challenge evaluates {ROBOT_MODEL!r}")

    # Upstream lets a set's entry carry its own `action_scale`, because there a scale belongs to the
    # policy. Here it belongs to the task, so an entry declaring a different one cannot be honoured --
    # and quietly ignoring it would score the policy at a scale its author did not choose, which is
    # the exact silent failure the top-level check below exists to prevent.
    raw_policies = data.get("policies", {})
    if isinstance(raw_policies, list):
        for i, entry in enumerate(raw_policies):
            if isinstance(entry, dict) and entry.get("action_scale") is not None \
                    and abs(float(entry["action_scale"]) - m.action_scale) > 1e-9:
                raise SubmissionRejected(
                    SubmissionErrorCode.ACTION_SCALE_MISMATCH,
                    f"'policies'[{i}] declares action_scale {entry['action_scale']} but the manifest declares "
                    f"{m.action_scale}. The scale is fixed by the task, so it cannot differ per policy; fold "
                    "the difference into that graph's output layer instead.")
    m.policies = _routing_table(raw_policies)
    m.default = str(data["default"]) if data.get("default") else None
    if not m.policies and not m.default:
        # A single-policy repo routes nothing and carries one `policy.onnx`. The daemon "takes the
        # sole `.onnx` in a repo and refuses several", so that graph is the policy for every task
        # and this is a complete submission rather than an incomplete one.
        lone = sorted(p.name for p in d.glob("*.onnx"))
        if len(lone) == 1:
            m.default = lone[0]
        else:
            where = (f"the submission has {len(lone)} .onnx files ({', '.join(lone)}), so one of them has "
                     "to be named" if lone else "the submission has no .onnx file")
            hint = f" (unrecognised key(s) {list(m.unknown_keys)} -- a typo?)" if m.unknown_keys else ""
            raise _reject(f"manifest names no policy and {where}: set 'policies' (task_id or family -> file) "
                          f"and/or 'default'{hint}")
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
    out_names = {o.name for o in outputs}
    if OBS_INPUT not in inputs:
        raise bad(f"no input named {OBS_INPUT!r} (inputs: {sorted(inputs)}); tensors are matched by name")
    for tensor in (*inputs.values(), *outputs):
        if tensor.type != "tensor(float)":
            raise bad(f"tensor {tensor.name!r} is {tensor.type}, every tensor must be float32")

    # A graph that carries state either way is a recurrent graph, including one that returns state
    # without accepting it -- that one is caught below as a missing input, which is what it is.
    recurrent = any(n in inputs for n in STATE_IN) or any(n in out_names for n in STATE_OUT)

    # "Unsupported names, ranks, types, widths and extra tensors fail at load." A graph with a
    # tensor the daemon does not know is refused there, so it is refused here: accepting it would
    # score a policy that cannot be installed.
    extra_in = sorted(set(inputs) - {OBS_INPUT} - (set(STATE_IN) if recurrent else set()))
    if extra_in:
        raise bad(f"unexpected input(s) {extra_in}; the contract is "
                  f"{'obs, h_in, c_in' if recurrent else 'obs'} and nothing else")
    action_outs = [o for o in outputs if o.name not in STATE_OUT]
    if len(action_outs) != 1:
        raise bad(f"expected exactly one action output, got {sorted(o.name for o in action_outs)}; a graph with "
                  "extra outputs is refused at load on the robot (export the actor alone, not the critic)")
    act = action_outs[0]

    obs_shape = inputs[OBS_INPUT].shape
    if len(obs_shape) != 2 or _static(_dim(obs_shape, 1)) != OBS_LEN:
        raise bad(f"input 'obs' has shape {obs_shape}, it must be [batch, {OBS_LEN}]")
    if len(act.shape) != 2 or _static(_dim(act.shape, 1)) != ACT_LEN:
        raise bad(f"output {act.name!r} has shape {act.shape}, it must be [batch, {ACT_LEN}]")
    obs_batch = _dim(obs_shape, 0)
    if not _is_dynamic(obs_batch) and _static(obs_batch) != 1:
        raise bad(f"input 'obs' has a fixed batch of {obs_batch}; inference always uses one robot, so the batch "
                  "dimension must be symbolic or exactly 1")
    dynamic_batch = _is_dynamic(obs_batch)

    layers = hidden = 0
    if recurrent:
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
        if layers * hidden > MAX_STATE_ELEMENTS:
            raise bad(f"each state tensor holds {layers}x{hidden} = {layers * hidden} elements, over the "
                      f"{MAX_STATE_ELEMENTS} the robot's runtime allows")
        # The batched path feeds one state tensor for every parallel episode at once, so a graph
        # whose state batch is pinned to 1 has to be driven one episode at a time even when `obs`
        # would take them all. Getting this wrong is a runtime shape error mid-rollout, not a
        # rejection, so it is settled here instead.
        if dynamic_batch and not all(_is_dynamic(_dim(s, 1)) for s in shapes):
            dynamic_batch = False

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
    # Ask for the state back as well, and feed it forward: this mirrors the robot's warm-up, which
    # runs the actor from zeroed memory and rejects invalid or non-finite outputs before committing
    # either state. A graph whose memory diverges is refused there, so it is refused here.
    out_names = [sig.action_output] + (list(STATE_OUT) if sig.recurrent else [])
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
        if sig.recurrent:
            h, c = (np.asarray(res[1], dtype=np.float32), np.asarray(res[2], dtype=np.float32))
            for tensor, label in ((h, STATE_OUT[0]), (c, STATE_OUT[1])):
                if not np.all(np.isfinite(tensor)):
                    raise SubmissionRejected(
                        SubmissionErrorCode.ONNX_OUTPUT_INVALID,
                        f"{name}: returned a non-finite {label} on a plausible observation. The robot's "
                        "runtime clears memory and refuses the step when this happens, so the policy would "
                        "stop controlling the robot")
            feeds_extra = {STATE_IN[0]: h, STATE_IN[1]: c}
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
