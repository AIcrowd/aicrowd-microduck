# VENDORED from microduck-evaluator/src/mdeval/server/metrics_v2/__init__.py by scripts/sync_from_evaluator.py.
# Do not edit by hand: change the evaluator and re-run the sync.
"""v2 metric plugins: one module per task family, discovered automatically.

``aicrowd_microduck.local.metrics`` imports every module of this package whose name does not start with an underscore
(sorted by name; subpackages are skipped) when it is itself imported. A family therefore adds its metric by adding
one file, ``metrics_v2/<family>.py``, and touches no shared code::

    import numpy as np

    from aicrowd_microduck.local.metrics import _f, alive_horizon, metric, qpos_horizon
    from aicrowd_microduck.local.recorder import RolloutTrace

    @metric("my_family")
    def my_family(tr: RolloutTrace) -> dict:
        H = tr.max_steps                      # the horizon; never tr.T
        alive = alive_horizon(tr)             # [H, K]
        ...
        return {"my_family_score": _f(score.mean()), "my_family_per_env": score.tolist()}

``_example.py`` is a complete, tested example (underscore: never imported in production).

Rules for v2 metrics (docs/tasks/README.md, "Commands, falls and recording"):

- **Metric names are global.** Registering a name that another module already registered raises at import, so
  two families cannot silently replace each other's metric.
- **Normalise by the horizon** ``tr.max_steps``, never by the trace length ``tr.T``. Recording stops 0.5 s after
  the last fall of a rollout (``record_after_done_s``), so ``tr.T`` depends on when the batch fell.
- **Read commands from the schedule saved at reset**, ``tr.schedule(name)`` -> ``[max_steps, K, dims]``, not from
  ``tr.commands``: after an env's done step the recorded commands belong to its auto-reset episode.
- **Rows of an env after its done step are its auto-reset episode.** ``alive_horizon`` is False there and past the
  trace end; ``qpos_horizon`` / ``qvel_horizon`` / ``pad_horizon`` put NaN there, so use ``np.where(alive, x, 0)``
  or nan-aware reductions, not ``x * alive`` (``NaN * 0`` is NaN).
- A module that fails to import fails the server at start: a broken metric must not score zero quietly.

``command_progress.py`` is shared: ``rough_terrain_v2`` and ``jelly_walk_v2`` both score with it
(docs/tasks/rough_terrain.md §5.2 defines it once).
"""
