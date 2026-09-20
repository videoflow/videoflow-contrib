# Solutions

A solution is an end-to-end, deployable videoflow application: a graph module plus the sibling
files `videoflow deploy` and `videoflow run-local` expect. `solutions/human_tracking/` is the
fullest example here — read it alongside this document; `solutions/face_obfuscation/` is the
smaller one.

The smallest complete examples live in the **core** repo, under
[`../../../videoflow/solutions/`](../../../videoflow/solutions/), where they also serve as its
end-to-end test suite: `toy_calculator` (the minimal reference for the full file convention, prep
hook included), `toy_fusion` (a REALTIME solution, and the example of legitimately shipping *no*
`prepare.py` and no GPU Dockerfile — both files are optional), `toy_router` (partitioned
routing) and `toy_recovery` (the error taxonomy). The file convention is identical wherever a
solution lives.

> Keep this file in sync with the solution conventions in
> [`../../../videoflow/videoflow/deploy/solution.py`](../../../videoflow/videoflow/deploy/solution.py) (its
> module docstring is the normative spec) and with each solution's own `README.md`.

## Anatomy

```
solutions/human_tracking/
├── human_tracking.py         # build_flow() — the graph module
├── human_tracking_nodes.py   # glue node classes (see "path rules" below)
├── common.py                 # shared helpers (config loading, the sample-clip fallback)
├── config.template.yaml      # x-questions + x-mounts + x-gpu
├── config.example.yaml       # fully documented reference config
├── prepare.py                # idempotent prep hook (sample clip, model weights)
├── requirements.txt / requirements-gpu.txt
├── Dockerfile / gpu.Dockerfile
└── README.md
```

## The graph module

Exposes **`build_flow() -> Flow`** and must **not** call `.run()` — the engine does that.

```python
def build_flow(cfg=None):
    if cfg is None:
        # deploy/run-local export the config they resolved as VF_SOLUTION_CONFIG;
        # otherwise the config.yaml beside this module, so it builds from any cwd.
        here = os.path.dirname(os.path.abspath(__file__))
        cfg = load_config(os.environ.get('VF_SOLUTION_CONFIG') or os.path.join(here, 'config.yaml'))
    ...
```

### Path rules

These are the rules that make a flow work in a pod (or a worker container) rather than only on
a laptop:

- **Resolve config paths relative to the config file's directory**, never the cwd. Deploy runs
  the graph from arbitrary working directories.
- **Define node classes in a sibling `*_nodes.py`**, not in the graph module. Workers reconstruct
  nodes by fully-qualified class path; a class defined in the graph module may not be importable
  under the same path inside the worker.
- **Any path baked into node params must exist at the same absolute path inside the container.**
  That's what the same-path mounts from `x-mounts` are for. A bundled sample therefore lives in
  `work_dir` (`common.resolve_input`/`fetch_input`), never in the `~` cache that is remapped.

## `config.template.yaml`

A valid config plus three extension blocks, all stripped when `config.yaml` is generated.

**`x-questions`** — what deploy and run-local prompt for when no config exists:

```yaml
x-questions:
  - key: work_dir                   # dotted path into the config
    prompt: 'Directory for the output'
    type: str                       # str | int | float | choice | path | paths
    default: ./out
  - key: device
    prompt: 'Run the models on'
    type: choice
    choices: [cpu, gpu]
    default: cpu
```

An input that has a bundled fallback (a sample clip the prep hook downloads) is a `str`
question with default `''`, never a `path` one — `path` validates existence.

**`x-mounts`** — paths from the resolved config that must be mounted into the prep container and
the workers (bind mounts locally, hostPath or claim volumes in the cluster):

```yaml
x-mounts:
  - '{input_video}:ro'              # dotted lookup; an empty value mounts nothing
  - '{work_dir}'
  - '~/.videoflow:/root/.videoflow' # explicit host:container mapping
```

A bare path becomes a **same-path** mount (identical absolute path on host and in container),
because paths baked into node params at compile time must resolve identically in the pods. A
`host:container` pair maps them explicitly — used for caches like the model directory; on a
multi-node cluster `--mount-home` (usually from the cluster profile) puts those inside the shared
claim's directory, where the claim serves them as `subPath` mounts.

**`x-gpu`** — the config values that decide whether `gpu.Dockerfile` is built:

```yaml
x-gpu:
  - '{device}'                      # or '{device.*}' for per-stage placement
```

The image is the flow's decision, never the docker daemon's: without `x-gpu`, deploy reads the
compiled graph's device placement when it imports on the host, and otherwise builds the CPU image
with a note.

## `prepare.py`

An idempotent prep hook: model weight downloads, the sample clip, any one-shot artifact the
graph needs. Deploy and run-local run it **inside the solution image, before compiling**, so its
outputs are baked into the compiled specs — and so a worker never has to download anything (a
pod without internet access runs all the same). Pre-fetch with the **same** `get_file` key and
URL the component uses in `open()`, so the node finds the weights already warmed.

Contract: accepts `--config <path>`, runs with the solution directory as cwd, and **skips steps
whose outputs already exist** (a `--force` flag to redo them is the convention). It will be run
repeatedly; make that cheap.

## Images

`Dockerfile` and `gpu.Dockerfile` (exact filename — deploy looks for it) build on
`videoflow-base:py3.12[-cuda]`, install `requirements.txt`, and **never set `ENTRYPOINT`**.

Deploy and run-local build the variant `x-gpu` selects, deploy it under a content-addressed tag,
and push it when the cluster profile names a registry. Never build by hand unless you are
debugging the Dockerfile itself.

## Device placement

Put GPU where it's genuinely needed, and say why in the config. Each `gpu` stage claims one whole
exclusive GPU per replica on Kubernetes, so per-stage device placement is a real cost decision,
not a detail. Both solutions here expose one `device` knob that the GPU-capable stages share.

## Deploying

```bash
cd solutions/human_tracking
videoflow run-local human_tracking.py     # workers as containers of the solution image, dev NATS in docker
videoflow deploy human_tracking.py        # Kubernetes
```

Both paths run the same config Q&A, the same in-image `prepare.py` and the same in-image
compile. Full pipeline and the multi-node story: the core README's *Deploying to Kubernetes*
section and [DEPLOY_VERIFY.md](DEPLOY_VERIFY.md).

## When changing a solution

Update the solution's `README.md` and `config.template.yaml` alongside the code — a new config key
that isn't in the template is invisible to anyone deploying fresh, and a changed `x-question` that
isn't in the README leaves the docs describing a prompt that no longer appears.
