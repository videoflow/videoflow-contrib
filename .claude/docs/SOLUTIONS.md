# Solutions

A solution is an end-to-end, deployable videoflow application: a graph module plus the sibling
files `videoflow deploy` expects. `solutions/offside/` is the fullest example — read it alongside
this document.

> Keep this file in sync with the solution conventions in
> [`../../../videoflow/videoflow/solution.py`](../../../videoflow/videoflow/solution.py) (its
> module docstring is the normative spec) and with each solution's own `README.md`.

## Anatomy

```
solutions/offside/
├── offside.py                # build_flow() — the graph module
├── offside_nodes.py          # glue node classes (see "path rules" below)
├── common.py                 # shared helpers (config loading)
├── config.template.yaml      # x-questions + x-mounts
├── config.example.yaml       # fully documented reference config
├── prepare.py                # idempotent prep hook
├── calibrate.py, fit_teams.py, sync_offsets.py, download_weights.py
├── requirements.txt / requirements-gpu.txt
├── Dockerfile / gpu.Dockerfile
└── README.md
```

## The graph module

Exposes **`build_flow() -> Flow`** and must **not** call `.run()` — the engine does that.

```python
def build_flow(cfg=None):
    if cfg is None:
        # Module-dir-relative so `videoflow deploy` works from any cwd.
        cfg = load_config(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.yaml'))
    ...
```

### Path rules

These are the rules that make a flow work in a pod rather than only on a laptop:

- **Resolve config paths relative to the module directory**, never the cwd. Deploy runs the graph
  from arbitrary working directories.
- **Define node classes in a sibling `*_nodes.py`**, not in the graph module. Workers reconstruct
  nodes by fully-qualified class path; a class defined in the graph module may not be importable
  under the same path inside the worker.
- **Any path baked into node params must exist at the same absolute path inside the pod.** That's
  what the same-path hostPath mounts from `x-mounts` are for.

## `config.template.yaml`

A valid config plus two extension blocks, both stripped when `config.yaml` is generated.

**`x-questions`** — what deploy prompts for when no config exists:

```yaml
x-questions:
  - key: cameras                    # dotted path into the config
    prompt: 'Video file per camera, comma-separated'
    type: paths                     # str | int | float | choice | path | paths
    item_key: 'cam{i}'
    item_value: {video: '{path}'}
  - key: pitch.length
    prompt: 'Pitch length in metres'
    type: float
```

**`x-mounts`** — paths from the resolved config that must be hostPath-mounted into the prep
container and the worker pods:

```yaml
x-mounts:
  - '{cameras.*.video}:ro'          # dotted lookup; * fans out
  - '{work_dir}'
  - '~/.videoflow:/root/.videoflow' # explicit host:container mapping
```

A bare path becomes a **same-path** mount (identical absolute path on host and in container),
because paths baked into node params at compile time must resolve identically in the pods. A
`host:container` pair maps them explicitly — used for caches like the model directory.

## `prepare.py`

An idempotent prep hook: model weight downloads, calibration, any one-shot artifact the graph
needs. Deploy runs it **inside the solution image, before compiling**, so its outputs are baked
into the compiled specs.

Contract: accepts `--config <path>`, runs with the solution directory as cwd, and **skips steps
whose outputs already exist** (a `--force` flag to redo them is the convention). It will be run
repeatedly; make that cheap.

## Images

`Dockerfile` and `gpu.Dockerfile` (exact filename — deploy looks for it) build on
`videoflow-base:py3.12[-cuda]`, install `requirements.txt`, and **never set `ENTRYPOINT`**.

Deploy auto-selects the GPU variant when GPUs are available.

## Device placement

Put GPU where it's genuinely needed, and say why in the config. From `solutions/offside/`:

```yaml
# Only the detector is genuinely GPU-bound; tracker and pose default to CPU so a
# 3-camera run claims 3 GPUs (one per detector), not 9.
device:
  detector: gpu
  tracker: cpu
  pose: cpu
```

Each `gpu` stage claims one whole exclusive GPU per camera on Kubernetes, so per-stage device
placement is a real cost decision, not a detail.

## Deploying

```bash
cd solutions/offside
videoflow run-local offside.py        # local subprocesses, dev NATS in Docker
videoflow deploy offside.py           # Kubernetes
```

Both paths run the same config Q&A and the same `prepare.py`. Full pipeline:
[../../../videoflow/.claude/docs/DEPLOYMENT.md](../../../videoflow/.claude/docs/DEPLOYMENT.md).

## When changing a solution

Update the solution's `README.md` and `config.template.yaml` alongside the code — a new config key
that isn't in the template is invisible to anyone deploying fresh, and a changed `x-question` that
isn't in the README leaves the docs describing a prompt that no longer appears.
