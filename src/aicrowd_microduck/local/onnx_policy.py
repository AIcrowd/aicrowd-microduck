# VENDORED from microduck-evaluator/src/mdeval/common/onnx_policy.py by scripts/sync_from_evaluator.py.
# Do not edit by hand: change the evaluator and re-run the sync.
"""Runs a submission's ONNX graphs with the robot daemon's LSTM state lifecycle.

Shared by the grader's sandboxed inference child (``worker/child_entry.py``) and the single-process
local scorer (``server/local_eval.py``), so both drive a graph identically: same graph per task, same
batching, same hidden-state resets. This module must stay free of anything sandbox- or
transport-specific; it is vendored into the public ``aicrowd-microduck`` CLI.
"""
from __future__ import annotations

import numpy as np


class OnnxPolicy:
    """Runs one graph per task, with the LSTM state lifecycle the robot's daemon implements.

    Hidden state is owned here, never by the graph: it starts zeroed at every rollout and an
    episode's column is re-zeroed when that episode ends. The simulator auto-resets finished
    episodes independently, so without the per-episode reset one episode's memory would leak into
    the next — the daemon's equivalent is resetting on fall recovery and policy switch.
    """

    def __init__(self, submission_dir: str, manifest_dict: dict):
        from aicrowd_microduck.submission import PolicyManifest

        self.dir = submission_dir
        self.manifest = PolicyManifest(**manifest_dict)
        self.sessions: dict[str, tuple] = {}
        self.sess = None
        self.sig = None
        self.h: np.ndarray | None = None
        self.c: np.ndarray | None = None
        self.out_names: list[str] = []

    def setup(self) -> dict:
        """Load and re-check every graph. Validation already ran in the trusted worker; doing it
        again here means the process that actually executes a graph is the one that verified it."""
        import os.path

        from aicrowd_microduck.submission import inspect_graph, open_session

        for file in self.manifest.graph_files():
            path = os.path.join(self.dir, file)
            sess = open_session(path)
            sig = inspect_graph(path, self.manifest.model_api, session=sess)
            self.sessions[file] = (sess, sig)
        return {f: s.to_dict() for f, (_, s) in self.sessions.items()}

    def reset(self, task_id: str, num_envs: int, family: str | None = None) -> None:
        file = self.manifest.graph_for(task_id, family)
        self.sess, self.sig = self.sessions[file]
        self.out_names = [self.sig.action_output]
        if self.sig.recurrent:
            from aicrowd_microduck.submission import STATE_OUT

            self.out_names += list(STATE_OUT)
            shape = (self.sig.layers, num_envs, self.sig.hidden)
            self.h = np.zeros(shape, dtype=np.float32)
            self.c = np.zeros(shape, dtype=np.float32)
        else:
            self.h = self.c = None

    def act(self, obs: np.ndarray, done: np.ndarray | None = None) -> np.ndarray:
        from aicrowd_microduck.submission import OBS_INPUT, STATE_IN

        assert self.sess is not None and self.sig is not None, "reset not done"
        if self.sig.recurrent and done is not None and self.h is not None and self.c is not None:
            idx = np.flatnonzero(done)
            if idx.size:
                self.h[:, idx, :] = 0.0
                self.c[:, idx, :] = 0.0
        K = obs.shape[0]
        if self.sig.dynamic_batch:
            feeds = {OBS_INPUT: obs}
            if self.sig.recurrent:
                feeds[STATE_IN[0]], feeds[STATE_IN[1]] = self.h, self.c
            out = self.sess.run(self.out_names, feeds)
            actions = np.asarray(out[0], dtype=np.float32).reshape(K, -1)
            if self.sig.recurrent:
                self.h = np.asarray(out[1], dtype=np.float32)
                self.c = np.asarray(out[2], dtype=np.float32)
            return actions
        # fixed batch of one, which is also exactly how the robot calls it
        rows = np.empty((K, self.manifest.action_len), dtype=np.float32)
        for k in range(K):
            feeds = {OBS_INPUT: obs[k : k + 1]}
            if self.sig.recurrent:
                assert self.h is not None and self.c is not None
                feeds[STATE_IN[0]] = self.h[:, k : k + 1, :]
                feeds[STATE_IN[1]] = self.c[:, k : k + 1, :]
            out = self.sess.run(self.out_names, feeds)
            rows[k] = np.asarray(out[0], dtype=np.float32).reshape(-1)
            if self.sig.recurrent:
                assert self.h is not None and self.c is not None
                self.h[:, k : k + 1, :] = np.asarray(out[1], dtype=np.float32)
                self.c[:, k : k + 1, :] = np.asarray(out[2], dtype=np.float32)
        return rows
