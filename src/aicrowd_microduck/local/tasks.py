# VENDORED from microduck-evaluator/src/mdeval/common/tasks.py by scripts/sync_from_evaluator.py.
# Do not edit by hand: change the evaluator and re-run the sync.
"""Task and competition specifications (YAML) — the "drop in any env" surface.

A *task* names one mjlab env id plus how to evaluate it: how many rollout batches, how many
parallel envs per batch (each env is one episode), the seeds, the horizon, the eval-time
config overrides, which metrics to compute and which one is the task score, and how to
render. A *competition* is a weighted list of tasks.

The mjlab env cfg is only touched here (``apply_eval_overrides``) so every knob that changes
physics or randomization is visible in one place and gets recorded in the manifest.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

_FAMILY_RE = re.compile(r"^[a-z0-9_]+$")
# TaskSpec fields added for v2, with the default at which ``to_dict`` omits them (see ``TaskSpec.to_dict``).
_V2_FIELD_DEFAULTS: dict[str, Any] = {"family": None, "record_after_done_s": 0.0}


@dataclass
class RenderSpec:
    enabled: bool = True
    env_ids: list[int] = field(default_factory=lambda: [0])
    fps: int = 20
    width: int = 480
    height: int = 270
    shadows: bool = False          # llvmpipe: shadows cost ~5x
    camera_distance: float = 0.8
    camera_azimuth: float = 135.0
    camera_elevation: float = -15.0
    track_body: str = "trunk_base"
    # Render-only floor dressing (server/decor.py): "auto" picks trampoline / foam from the env id,
    # "plain" is the bare compiled model, "terrain" / "roller" repaint the ground in one muted palette (roller
    # adds the lane lines; server/terrain_decor.py). Pixels and replay poses only; never touches simulation or metrics.
    scene: str = "auto"
    # Also write one self-contained replay_standalone.html (viewer + packed replay data, opens from file://)
    # next to report.json when the run finishes. Needs the built viewer in the checkout; off for cloud runs.
    standalone: bool = False


@dataclass
class EvalOverrides:
    disable_curriculum: bool = True
    terrain_rows: int | None = 2          # rough terrain grid, None = keep cfg
    terrain_cols: int | None = 3
    terrain_curriculum: bool | None = False   # False: random difficulty per tile instead of row-graded; None = keep cfg
    disable_pushes: bool = False
    disable_obs_noise: bool = False
    disable_domain_randomization: bool = False
    # per-term parameter edits, applied last: {event_name: {param: value}}, {command_name: {attr: value}}
    event_params: dict[str, dict[str, Any]] = field(default_factory=dict)
    command_params: dict[str, dict[str, Any]] = field(default_factory=dict)


@dataclass
class TaskSpec:
    task_id: str
    env_id: str
    # v2: the family this task is a level of (e.g. rough_terrain_v2). Manifests route by task id, then family,
    # then default; the report scores each family as the weighted mean of its levels.
    family: str | None = None
    env_module: str | None = None         # importable module that registers env_id (drop-in envs outside the plugin entry point)
    num_rollouts: int = 1
    num_envs: int = 8
    episode_length_s: float = 10.0
    seeds: list[int] | None = None        # one per rollout; default seed_base + i
    seed_base: int = 1000
    seed_mode: str = "fixed"              # fixed: public seeds above | secret: HMAC(seed_secret, task_id:i)
    action_scale: float = 1.0
    step_deadline_s: float = 1.0          # worker-enforced wall budget per act() call
    max_step_timeouts: int = 10
    metrics: list[str] = field(default_factory=lambda: ["episode_length", "survival", "termination_reasons", "reward_terms"])
    primary_metric: str = "survival_fraction"
    higher_is_better: bool = True
    eval_overrides: EvalOverrides = field(default_factory=EvalOverrides)
    render: RenderSpec = field(default_factory=RenderSpec)
    description: str = ""
    # Keep recording each env this long past its done step; the rollout ends once every env has had it, or at the
    # horizon. 0 (v1) ends the rollout at the last done step; v2 tasks use 0.5. Rows of an env after its done
    # step belong to its auto-reset episode, so metrics mask them (``alive_mask`` / ``metrics.alive_horizon``).
    record_after_done_s: float = 0.0

    def __post_init__(self) -> None:
        if self.family is not None and not _FAMILY_RE.match(str(self.family)):
            raise ValueError(f"{self.task_id}: family {self.family!r} must match [a-z0-9_]+")
        if not self.record_after_done_s >= 0.0:
            raise ValueError(f"{self.task_id}: record_after_done_s must be >= 0, got {self.record_after_done_s!r}")

    def rollout_seeds(self, seed_secret: str | None = None) -> list[int]:
        """Seeds per rollout. ``seed_mode: secret`` derives them from a secret the participant never sees,
        so the evaluation set cannot be regenerated offline (the sim is deterministic); ``fixed`` uses the
        public ``seeds``/``seed_base`` (fine for smoke tests, not for a competition)."""
        if self.seed_mode == "secret":
            if not seed_secret:
                raise ValueError(f"{self.task_id}: seed_mode=secret but no seed secret was provided (MDEVAL_SEED_SECRET)")
            import hmac
            return [int.from_bytes(hmac.new(seed_secret.encode(), f"{self.task_id}:{i}".encode(), "sha256").digest()[:4], "big") & 0x7FFFFFFF
                    for i in range(self.num_rollouts)]
        if self.seed_mode != "fixed":
            raise ValueError(f"{self.task_id}: unknown seed_mode {self.seed_mode!r}")
        if self.seeds is not None:
            if len(self.seeds) != self.num_rollouts:
                raise ValueError(f"{self.task_id}: {len(self.seeds)} seeds for {self.num_rollouts} rollouts")
            return list(self.seeds)
        return [self.seed_base + i for i in range(self.num_rollouts)]

    @property
    def seeds_are_public(self) -> bool:
        return self.seed_mode == "fixed"

    @property
    def control_dt(self) -> float:
        return 0.02  # 50 Hz for every microduck env (sim dt 0.005 x decimation 4); verified at build time

    @property
    def max_steps(self) -> int:
        return int(round(self.episode_length_s / self.control_dt))

    @property
    def post_done_steps(self) -> int:
        """``record_after_done_s`` in control steps (0.5 s -> 25)."""
        return int(round(self.record_after_done_s / self.control_dt))

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        # Fields added for v2 are left out at their defaults, so a v1 spec keeps the dict, and therefore the
        # task_spec_digest and the public competition_digest, it had before they existed.
        for k, default in _V2_FIELD_DEFAULTS.items():
            if d.get(k) == default:
                d.pop(k, None)
        return d

    def digest(self) -> str:
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True).encode()).hexdigest()[:16]


@dataclass
class CompetitionSpec:
    name: str
    tasks: list[TaskSpec]
    weights: dict[str, float] = field(default_factory=dict)
    description: str = ""
    source_path: str | None = None

    def __post_init__(self) -> None:
        ids = [t.task_id for t in self.tasks]
        dup = sorted({i for i in ids if ids.count(i) > 1})
        if dup:
            raise ValueError(f"{self.name}: task id(s) {dup} listed twice")
        # a family key and a task id share the manifest's 'policies' namespace and the public score_<id> keys
        clash = sorted(set(self.families()) & set(ids))
        if clash:
            raise ValueError(f"{self.name}: family name(s) {clash} are also task ids")
        if self.weights:
            stray = sorted(set(self.weights) - set(ids))
            if stray:
                raise ValueError(f"{self.name}: weights names no task {stray}")
            # with weights given, a task left out would silently weigh 1.0 against weights like 0.05
            missing = [i for i in ids if i not in self.weights]
            if missing:
                raise ValueError(f"{self.name}: task(s) {missing} have no weight")

    def weight(self, task_id: str) -> float:
        return float(self.weights.get(task_id, 1.0))

    def families(self) -> dict[str, list[str]]:
        """family -> its task ids, in competition order. Tasks without a family are not listed."""
        out: dict[str, list[str]] = {}
        for t in self.tasks:
            if t.family:
                out.setdefault(t.family, []).append(t.task_id)
        return out

    def task(self, task_id: str) -> TaskSpec:
        for t in self.tasks:
            if t.task_id == task_id:
                return t
        raise KeyError(task_id)

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "description": self.description, "weights": self.weights,
                "tasks": [t.to_dict() for t in self.tasks]}

    def digest(self) -> str:
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True).encode()).hexdigest()[:16]


def _dataclass_from(cls, d: dict[str, Any]):
    if d is None:
        return cls()
    kwargs = dict(d)
    if cls is TaskSpec:
        kwargs["eval_overrides"] = _dataclass_from(EvalOverrides, kwargs.get("eval_overrides"))
        kwargs["render"] = _dataclass_from(RenderSpec, kwargs.get("render"))
    unknown = set(kwargs) - set(cls.__dataclass_fields__)
    if unknown:
        raise ValueError(f"{cls.__name__}: unknown keys {sorted(unknown)}")
    return cls(**kwargs)


def load_task(path: str | Path) -> TaskSpec:
    with open(path) as f:
        return _dataclass_from(TaskSpec, yaml.safe_load(f))


def load_competition(path: str | Path) -> CompetitionSpec:
    path = Path(path)
    with open(path) as f:
        d = yaml.safe_load(f)
    tasks: list[TaskSpec] = []
    for entry in d["tasks"]:
        if isinstance(entry, str):
            tasks.append(load_task(path.parent / entry))
        else:
            tasks.append(_dataclass_from(TaskSpec, entry))
    return CompetitionSpec(name=d["name"], tasks=tasks, weights=d.get("weights", {}),
                           description=d.get("description", ""), source_path=str(path))


# --------------------------------------------------------------------- mjlab cfg edits
def apply_eval_overrides(cfg, spec: TaskSpec, num_envs: int, seed: int):
    """Mutate a freshly loaded mjlab ``ManagerBasedRlEnvCfg`` for evaluation.

    Every physics/randomization-relevant edit lives here so it is recorded via
    ``describe_cfg`` in the rollout manifest.
    """
    ov = spec.eval_overrides
    cfg.scene.num_envs = num_envs
    cfg.seed = seed
    cfg.episode_length_s = spec.episode_length_s
    if ov.disable_curriculum:
        cfg.curriculum = {}
    tg = getattr(cfg.scene.terrain, "terrain_generator", None)
    if tg is not None:
        if ov.terrain_rows is not None:
            tg.num_rows = ov.terrain_rows
        if ov.terrain_cols is not None:
            tg.num_cols = ov.terrain_cols
        if ov.terrain_curriculum is not None:
            tg.curriculum = ov.terrain_curriculum
    if ov.disable_pushes:
        cfg.events.pop("push_robot", None)
    if ov.disable_obs_noise:
        for grp in cfg.observations.values():
            for term in grp.terms.values():
                term.noise = None
    if ov.disable_domain_randomization:
        keep = {"expand_bam_friction_fields", "reset_base", "reset_robot_joints", "reset_action_history"}
        cfg.events = {k: v for k, v in cfg.events.items() if k in keep}
        # the encoder-bias startup event is a randomization; the biased obs flag must go with it
        for grp in cfg.observations.values():
            t = grp.terms.get("joint_pos")
            if t is not None and isinstance(t.params, dict) and "biased" in t.params:
                t.params["biased"] = False
    for name, params in ov.event_params.items():
        if name not in cfg.events:
            raise KeyError(f"{spec.task_id}: event_params names unknown event {name!r}; have {sorted(cfg.events)}")
        cfg.events[name].params = {**(cfg.events[name].params or {}), **params}
    for name, attrs in ov.command_params.items():
        if name not in cfg.commands:
            raise KeyError(f"{spec.task_id}: command_params names unknown command {name!r}; have {sorted(cfg.commands)}")
        for k, v in attrs.items():
            if not hasattr(cfg.commands[name], k):
                raise KeyError(f"{spec.task_id}: command {name!r} has no attribute {k!r}")
            setattr(cfg.commands[name], k, v)
    # the action scale is applied by mjlab's JointPositionAction
    if "joint_pos" in cfg.actions:
        cfg.actions["joint_pos"].scale = spec.action_scale
    return cfg


def describe_cfg(cfg) -> dict[str, Any]:
    """A JSON-able, digestable description of the resolved env cfg (best effort)."""
    def conv(o, depth=0):
        if depth > 6:
            return repr(o)
        if isinstance(o, (int, float, str, bool)) or o is None:
            return o
        if isinstance(o, (list, tuple)):
            return [conv(x, depth + 1) for x in o]
        if isinstance(o, dict):
            return {str(k): conv(v, depth + 1) for k, v in o.items()}
        if hasattr(o, "__dataclass_fields__"):
            return {k: conv(getattr(o, k), depth + 1) for k in o.__dataclass_fields__}
        if callable(o):
            return f"{getattr(o, '__module__', '')}.{getattr(o, '__qualname__', repr(o))}"
        return repr(o)
    d = conv(cfg)
    d["_digest"] = hashlib.sha256(json.dumps(d, sort_keys=True, default=repr).encode()).hexdigest()[:16]
    return d
