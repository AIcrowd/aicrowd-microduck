# aicrowd-microduck

Command-line tool for the [AIcrowd Microduck Sim2Real Challenge](https://www.aicrowd.com/challenges/microduck-sim2real-challenge-2026).
It checks a submission with the evaluator's own rules, scores it locally with the evaluator's own
simulation, then submits it to AIcrowd.

```bash
pip install aicrowd-microduck            # or: uv tool install aicrowd-microduck

aicrowd-microduck login                  # paste your API key from your AIcrowd profile page
aicrowd-microduck validate my_policy/    # the evaluator's acceptance checks, in seconds
aicrowd-microduck evaluate my_policy/ --only rough_terrain_v2 -n 1   # a quick local score (needs the simulator)
aicrowd-microduck submit my_policy/ --watch
```

## What a submission is

A directory holding `manifest.json` plus the ONNX graphs it names. No Python code is run. See the
challenge page for the manifest format and the 61-D observation / 14-D action contract.

```json
{
  "schema_version": 2,
  "model_api": 1,
  "action_scale": 1.0,
  "policies": {
    "rough_terrain_v2": "walking.onnx",
    "roller_descent_v2": "roller.onnx",
    "jelly_walk_v2": "walking.onnx",
    "flamingo_switch_v2": "flamingo.onnx",
    "trampoline_height_v2": "walking.onnx"
  }
}
```

A policy is chosen per task by its task id, then by its family (as above), then by `default`.

## Commands

| command | what it does |
|---|---|
| `login [--api-key KEY]` | Checks the key with AIcrowd and saves it. aicrowd-cli reads the same file, so logging in with either tool works for both. |
| `validate PATH` | Unpacks and checks a directory, `.zip` or `.tar.gz` the way the evaluator does: manifest, tensor names/shapes/dtypes, a smoke inference that rejects NaN and observation-independent output, and coverage of every task in the competition. A pass means the evaluator will accept it. |
| `evaluate PATH [--only FAMILY,...] [-n N] [-j JOBS]` | Simulates and scores the submission on your machine with the evaluator's own environments, policy runner, metrics and scoring, and writes `runs/local/report.json`. `--only` picks task ids or families, `-n` caps rollouts per task for a quick, partial score. Needs the simulator: install the starter kit, or `pip install "aicrowd-microduck[sim]"` plus the Microduck environments. |
| `pack DIR [-o OUT.zip]` | Zips a submission with its files at the archive root. `submit` does this for you. |
| `submit PATH [--watch] [--dry-run]` | Validates, then uploads to AIcrowd. `--watch` waits for the score. `--dry-run` checks everything, including that AIcrowd would accept a submission from you right now, without uploading anything. |
| `status ID [--watch]` | Shows a submission's grading status and score. |

All commands accept `--json` for scripting. The API key can also come from `AICROWD_API_KEY`.

Rejections carry the evaluator's error code (`MANIFEST_INVALID`, `POLICY_FILE_MISSING`, `ONNX_INVALID`,
`ONNX_SIGNATURE_MISMATCH`, `ONNX_OUTPUT_INVALID`, `ACTION_SCALE_MISMATCH`, `SUBMISSION_TOO_LARGE`,
`FILE_COUNT_EXCEEDED`, `UNSAFE_ARCHIVE_MEMBER`). Limits: archive up to 256 MiB, at most 500 MB
unpacked and 2000 files, no symlinks, a single graph up to 200 MB.

## Local scores and the grader's

`evaluate` runs the grader's code, so a policy scored with the same seeds gets the same score
(the evaluator's parity test checks this rollout by rollout). The grader draws its episodes from a
secret seed, though, and `evaluate` uses a public default (or `--seed-secret` / `MDEVAL_SEED_SECRET`),
so your leaderboard score is a nearby number, not the identical one. Two more differences: the
grader fails a rollout whose policy misses the 1 s per-step deadline more than 10 times (`evaluate`
only warns), and it retries its own infrastructure failures.

## Why not `aicrowd submission create`?

aicrowd-cli looks the challenge up on a separate service before it uploads, and that lookup does not
find this challenge, so it stops with "Challenge Not Found". This tool sends the slug straight to the
AIcrowd web API, the same way `whest submit` does for WhestBench, and never needs that lookup.

## For maintainers

`submission.py`, `ingestion.py` and everything under `local/` (the scorer, its metrics and the
competition's task YAMLs) are copied from the evaluator, and `tasks.json` is a snapshot of the
competition it runs. After changing any of them in the evaluator:

```bash
uv run python scripts/sync_from_evaluator.py      # --check reports drift without writing
uv run pytest
```

The tests diff the copies against `../microduck-evaluator` when that checkout exists. Set
`MDEVAL_REPO` to point elsewhere.
