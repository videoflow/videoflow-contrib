# CLAUDE.md — videoflow-contrib

The extension repository for [videoflow](../videoflow). It holds two kinds of thing:

- **Components** (one directory each, ~18 of them) — reusable producer/processor/consumer nodes
  that depend on heavy third-party ML stacks the core library deliberately doesn't.
- **Solutions** (`solutions/`) — end-to-end deployable applications wiring components into a graph.

The organizing principle: **each sub-package is independently installable, versioned, and
containerized.** They are not one library. A component that needs TensorFlow must not force
TensorFlow on someone using the PyTorch tracker.

## Layout

```
videoflow-contrib/
├── <component>/                    x18: yolo, detector_tf, tracker_sort, offside_engine, …
│   ├── pyproject.toml              hatchling; deps declared here
│   ├── component.yaml              machine-readable descriptor
│   ├── Dockerfile / gpu.Dockerfile
│   ├── videoflow_contrib/<name>/   native namespace package
│   │   └── __init__.py             re-exports the node classes
│   └── tests/                      optional
└── solutions/{offside, human_tracking, face_obfuscation}/
```

Every solution here depends on ML components, so all of them need models and
footage to run. The dependency-free framework exercisers that used to live here
(`toy_calculator`, `toy_fusion`, `toy_router`) now live in the **core** repo at
[../videoflow/solutions/](../videoflow/solutions/), where they double as its
end-to-end integration tests. They import no contrib code, so they remain the
cheapest first target for any deploy verification — reach for them (from the core
checkout) before spending minutes on an ML solution.

**`videoflow_contrib/` is a native namespace package — there is no
`videoflow_contrib/__init__.py`, in any sub-package.** Adding one shadows the namespace and breaks
importing any *other* contrib component installed alongside it.

The naming triple for a directory `foo_bar`:

| | |
|---|---|
| Distribution | `videoflow_contrib_foo_bar` |
| Import path | `videoflow_contrib.foo_bar` |
| Descriptor name | `videoflow/foo-bar` |

Everything is currently at version `1.0.0`.

## Commands

```bash
cd <component> && uv build          # build the wheel
cd <component> && pytest            # run that component's tests
./validate-components.sh            # validate every component.yaml
uv tool install pre-commit && pre-commit install
```

**pre-commit must be >= 3.2.0** — the pinned `pre-commit-hooks` v5 uses the renamed stages, and an
older binary fails with `InvalidManifestError: ... but got: 'pre-commit'`. Ubuntu's packaged
2.17.0 at `/usr/bin/pre-commit` is a common culprit; install into `~/.local/bin` and keep that
ahead of `/usr/bin` on `PATH`. See
[../videoflow/CLAUDE.md](../videoflow/CLAUDE.md#commands) for the full explanation.

There is no pre-push hook here (unlike core) because there is no repo-wide test run — see below.

**There is no repo-wide test run, and that is deliberate.** Components have mutually incompatible
dependencies (tensorflow vs torch), so a single local environment isn't meaningful — each is
exercised in its own image. This is why `.pre-commit-config.yaml` has no pytest hook and why CI
(`.github/workflows/components.yml`) only validates descriptors and builds wheels. Don't "fix"
this by adding a root test runner.

Only a handful of packages have tests today (`offside_engine`, `detector_tf`, `soccer_detector`,
`team_classifier`, `pitch_calib`). New components should ship tests for whatever logic doesn't
require model weights.

## Python conventions

Python 3.12 everywhere. **The core repo's [CLAUDE.md](../videoflow/CLAUDE.md#python-conventions)
is the authority** — the prescriptive/descriptive split there applies here unchanged. In short:
type-hint every parameter and return in new code; match the surrounding file's existing style
rather than reformatting it.

Two generations of style coexist here. Older files use
`from __future__ import absolute_import, ...`, `super(Class, self).__init__`, and spaced kwargs
(`nb_tasks = 1`); newer files use `from __future__ import annotations` and bare `super()`. **New
code follows the newer style.**

### Imports

Module scope by default, as in core. This repo has exactly **one** sanctioned exception: the heavy
ML framework import (`torch`, `tensorflow`, `ultralytics`, `rfdetr`) goes inside `open()`, so that
`component.yaml` validation, wheel builds, and `--help` all work without the framework installed.
Comment it where you do it. The exception does not extend to stdlib, numpy, or other light
dependencies — those go at module scope.

### Model weights

Always via `videoflow.utils.downloader.get_file(fname, origin)`, which caches under
`~/.videoflow/models`. Call it from `open()` (or from a solution's `prepare.py`), never from
`__init__`. `origin` may be a **list** of URLs tried in order — the upstream checkpoint first, then
a GitHub-release mirror under
`https://github.com/videoflow/videoflow-contrib/releases/download/<tag>/`. See
[`soccer_detector/videoflow_contrib/soccer_detector/detector.py`](soccer_detector/videoflow_contrib/soccer_detector/detector.py).

### Docker

`ARG BASE_IMAGE=videoflow-base:py3.12` (`-cuda` for GPU). **Never set `ENTRYPOINT`** — the base
image's `python -m videoflow.worker` must be inherited. The GPU variant must be named exactly
`gpu.Dockerfile`; `videoflow deploy` looks for that filename.

## Hard rules

**Store every constructor argument verbatim as `self._<name>`.** Nodes are rebuilt inside their
worker container via `type(node)(**get_params())`, so a param not stored under a matching
attribute raises `AttributeError` at graph-build time. Full contract, including when to override
`get_params()`: [../videoflow/.claude/docs/NODE_CONTRACT.md](../videoflow/.claude/docs/NODE_CONTRACT.md).

**If a subclass fixes a parent's parameter, pop it first** — `kwargs.pop('nb_tasks', None)` before
passing a literal, or reconstruction collides with the captured value. `TfliteObjectDetector` is
the reference.

**`process()` returning `None` does not drop a message.** End-of-stream travels on a separate
`_eos` subject.

**Load models in `open()`, release in `close()`.** `__init__` runs on the machine building the
graph, which may have no GPU and no weights.

**Multi-GPU models (RFC 0003).** A component whose model spans GPUs relies on one contract:
inside the worker, the visible devices are exactly the granted devices, `cuda:0..N-1`, with
`N == self.gpu_count`. Shard in `open()` (`device_map = 'auto'` for HF models, explicit
`.to('cuda:1')` for multi-model nodes) and put **zero device arithmetic** anywhere else. Declare
the default need in `component.yaml` — `spec: {resources: {gpu: {count: 2}}}` — so graph authors
don't pass `gpu_count=` by hand; treat the descriptor count as a default, not a floor, and
enforce any hard minimum in `open()` via `videoflow.utils.system.granted_gpus()`. Multi-GPU
requires whole exclusive devices: MIG slices and time-sliced units can't be combined.

**Prefer subclassing a core domain base** (`videoflow.processors.vision.detectors.ObjectDetector`,
`BoundingBoxTracker`, `videoflow.producers.video.VideoFileReader`) over the raw node classes, so
your component is a drop-in for others of its kind.

## Reference packages

- **`offside_engine/`** — the cleanest component overall: pure Python, no ML deps, good tests.
  Read this first.
- **`detector_tf/`** — the reference `ProcessorNode`, including the fixed-parameter idiom.
- **`synced_video_reader/`** — the canonical hand-written `get_params()` override.
- **`solutions/offside/`** — the fullest solution.
- **`../videoflow/solutions/toy_calculator/`** — the smallest complete solution (in the core
  repo); read it before writing a new one.

## Keep docs in sync with code

A change isn't done until the docs describing it are updated **in the same commit**:

- The component's `component.yaml` — the params schema and image refs must track the constructor.
  A new or renamed constructor argument is a descriptor change. Re-run `./validate-components.sh`.
- The sub-package's own `README.md`, if it has one.
- The root [README.md](README.md) — the components and solutions tables.
- For solutions: the solution's `README.md` **and** `config.template.yaml`.
- [.claude/agents/videoflow-author.md](.claude/agents/videoflow-author.md) — when the node contract
  or sub-package layout changes.
- [.claude/docs/DEPLOY_VERIFY.md](.claude/docs/DEPLOY_VERIFY.md) — when a solution's config keys,
  prep artifacts, or required deploy flags change. Its recipes are executable, so they go stale
  silently and only fail on the next cluster run.
- `CLAUDE.md` and `.claude/docs/*.md` — when conventions, layout, or commands change.
- The sibling [../videoflow](../videoflow) repo, if the change was driven by a core-API change.

## Where to look next

- [.claude/agents/videoflow-author.md](.claude/agents/videoflow-author.md) — the deep node-contract
  and authoring reference. Not duplicated here; read it before writing a component.
- [.claude/agents/solution-verifier.md](.claude/agents/solution-verifier.md) — the other side of
  that: deploys every solution to a real cluster, and triages failures across this repo, the core
  repo, and the infrastructure.
- [.claude/docs/ADDING_A_COMPONENT.md](.claude/docs/ADDING_A_COMPONENT.md) — checklist for a new
  sub-package.
- [.claude/docs/SOLUTIONS.md](.claude/docs/SOLUTIONS.md) — anatomy of a solution and how it deploys.
- [.claude/docs/DEPLOY_VERIFY.md](.claude/docs/DEPLOY_VERIFY.md) — cluster preconditions, the
  mandatory deploy flags, and the per-solution recipes.
- [README.md](README.md) — the user-facing guide.
- [../videoflow/CLAUDE.md](../videoflow/CLAUDE.md) — the core framework.
