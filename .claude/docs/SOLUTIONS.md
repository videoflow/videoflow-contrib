# Solutions

A solution is an end-to-end, deployable videoflow application: a graph module plus the sibling
files `videoflow deploy` expects. `solutions/offside/` is the fullest example — read it alongside
this document.

The smallest complete examples live in the **core** repo, under
[`../../../videoflow/solutions/`](../../../videoflow/solutions/), where they also serve as its
end-to-end test suite: `toy_calculator` (the minimal reference for the full file convention, prep
hook included), `toy_fusion` (a REALTIME solution, and the example of legitimately shipping *no*
`prepare.py` and no GPU Dockerfile — both files are optional), and `toy_router` (partitioned
routing). The file convention is identical wherever a solution lives.

> Keep this file in sync with the solution conventions in
> [`../../../videoflow/videoflow/deploy/solution.py`](../../../videoflow/videoflow/deploy/solution.py) (its
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

**It is also where the bundled sample input is fetched.** All four solutions here default to a
sample clip on the `example_videos` release when their input is left empty, so that
`videoflow deploy <graph>.py` needs no footage of your own. The split that makes this work is
always the same pair of methods on the config object:

```python
def resolve_input(self) -> str:          # pure — the graph calls this
    return self.input_video or os.path.join(self.work_dir, SAMPLE_VIDEO_NAME)

def fetch_input(self) -> str:            # downloads — only prepare.py calls this
    if self.input_video:
        return self.input_video
    return get_file(SAMPLE_VIDEO_NAME, SAMPLE_VIDEO_URL,
                    cache_dir=self.work_dir, cache_subdir='')
```

Two things are load-bearing:

- **`build_flow` must never download.** Compiling has to stay side-effect free — `--dry-run` calls
  it, and progress output would corrupt the manifest stream. The graph resolves the path the fetch
  *will* produce; prep produces it.
- **The sample goes in `work_dir`, not the shared model cache.** That path is baked into the
  reader's params at compile time and must resolve identically inside the pod, and `work_dir` is a
  same-path mount whereas the caches are remapped onto `/root`.

A solution whose input is a *set* (offside's `cameras`) does the same thing with a dict:
empty means sample, and `fetch_inputs()` pulls each file. Drop the input's `x-question` when you
add a default, and drop its `x-mounts` entry too — with the input empty, `'{input_video}:ro'`
would expand to the graph directory and shadow it inside the pod. Users with their own footage set
the path and pass `--mount /path/to/video.mp4:ro`.

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
