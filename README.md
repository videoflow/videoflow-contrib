# videoflow-contrib: Videoflow community contributions

[![Build Status](https://github.com/videoflow/videoflow-contrib/actions/workflows/components.yml/badge.svg)](https://github.com/videoflow/videoflow-contrib/actions/workflows/components.yml)
[![license](https://img.shields.io/github/license/mashape/apistatus.svg?maxAge=2592000)](https://github.com/videoflow/videoflow-contrib/blob/master/LICENSE)

This library is the official extension repository for the Videoflow library.
It contains additional consumers, producers, processors, subflows, etc. which are not yet available within Videoflow itself.
All of these additional modules can be used in conjunction with core Videoflow flows.
This is done in the interest of keeping Videoflow succinct, clean, and simple, with as minimal dependencies to third-party
libraries as necessaries.

This contribution repository is both the proving ground for new functionality, and the archive for functionality that (while useful) may not fit well into the Videoflow paradigm.

## Quick start

Clone both repositories side by side, install videoflow from its checkout (or
`pip install 'videoflow[all]'` from PyPI — the editable checkout is for working
against unreleased core), and run a solution — locally or on the cluster
`kubectl` points at. Nothing else goes on your machine: the ML stacks live in
the solution images, which both commands build for you.

```bash
git clone https://github.com/videoflow/videoflow
git clone https://github.com/videoflow/videoflow-contrib
uv tool install --editable './videoflow[all]'          # `videoflow` on your PATH
#   or: python3 -m venv .venv && .venv/bin/pip install -e './videoflow[all]'

cd videoflow-contrib/solutions/human_tracking
videoflow run-local human_tracking.py        # local: the workers run in the solution image
videoflow deploy human_tracking.py           # cluster: the same image, as pods
```

The first run asks a few questions (Enter takes the defaults: the bundled
sample clip, CPU, batch) and writes `config.yaml`; the first build takes minutes
(`videoflow-base`, then the solution image); prep runs inside the image and
fetches the sample and the weights; the annotated video lands in `out/`. Docker
must be running; for `deploy`, `kubectl` must point at your cluster. That is
the whole story on a laptop cluster (kind, minikube, k3s, Docker Desktop).

**A multi-node or shared cluster** needs a handful of per-cluster values — the
registry the nodes pull from, an RWX claim and the directory it is served at,
the namespace, a PriorityClass. They go in a cluster profile once, keyed by the
kubectl context, and the commands above stay the same:

```yaml
# ~/.config/videoflow/clusters.yaml
clusters:
  lab:
    context: default                         # kubectl config current-context
    namespace: videoflow
    registry: 10.0.0.1:5000
    push_tool: crane                         # for a plain-HTTP registry the docker daemon does not trust
    mount_pvc: ['vf-share:/shared/videoflow']
    mount_home: /shared/videoflow/home
    priority_class: cluster-batch
```

Answer the `work_dir` question with a directory under the shared claim. The
full recipe, including how to find the claim's directory and how to verify a
run, is in [.claude/docs/DEPLOY_VERIFY.md](.claude/docs/DEPLOY_VERIFY.md); the
mechanics are in the core README's *Deploying to Kubernetes* section.

**GPU:** answer `gpu` when asked for the device. The GPU image is built instead,
`deploy` requests GPUs for the pods and puts the cluster's `nvidia` RuntimeClass
on them; `run-local` hands the workers their devices when the docker daemon has
the NVIDIA runtime.

## Independent sub-packages
Each folder in the repository corresponds to an individual sub-package that follows the [native namespace package](https://packaging.python.org/guides/packaging-namespace-packages/#native-namespace-packages) Python 3 standard.  The project follows that structure to facilitate per subpackage independent licensing and installation.

See the [Tensorflow Object detection](detector_tf) sub-package for an example of how to structure ``videoflow_contrib`` sub-packages.  Each sub-package is a [uv](https://docs.astral.sh/uv/) project with a ``pyproject.toml`` (hatchling build backend; its runtime dependencies are declared there, with a ``[project.optional-dependencies] gpu`` extra where a CUDA build differs) and a ``Dockerfile`` (plus a ``gpu.Dockerfile`` where relevant) that describes the environment needed to use it. Build a component's wheel with ``uv build`` from its directory.

## Component descriptors

Each component also ships a `component.yaml` **descriptor** — the machine-readable
manifest that makes it a first-class, distributable videoflow component: its name and
version, the container images (cpu/gpu), the Python class it provides, a JSON Schema
for its constructor params, its input/output payload types, its device support, and
constraints (e.g. `singleton` for stateful trackers). This is the contract the
videoflow marketplace uses to discover, validate, and (eventually) resolve a component
by reference.

Validate every descriptor against the videoflow schema before pushing:

```bash
./validate-components.sh            # or: videoflow component validate detector_tf/
```

CI (`.github/workflows/components.yml`) runs this on every push, plus a check that each
sub-package builds a wheel from its `pyproject.toml`. When you add a component, add a
`component.yaml` next to its `pyproject.toml` (copy the closest existing one and adjust
`pythonClass`, `params.schema`, and `io`).

## Writing a component

A component is one reusable node (or a small family of them) in its own
sub-package. The checklist:

1. **Package layout** — a top-level folder with a `pyproject.toml` (hatchling,
   `dependencies = ["videoflow>=<current>", ...]` — the version every sibling
   declares, `./set-version.py --current`) and the code under
   `videoflow_contrib/<name>/` as a native namespace package (no
   `videoflow_contrib/__init__.py`). Runtime deps go in `pyproject.toml`; a
   CUDA-differing build gets a `[project.optional-dependencies] gpu` extra.
2. **Node class rules** — subclass `ProducerNode` / `ProcessorNode` /
   `ConsumerNode`; accept and forward `**kwargs` to `super().__init__` and keep
   every constructor argument JSON-serializable, because distributed workers
   reconstruct your node from its class path + params
   (`videoflow_contrib.<name>.<Class>`). Load models in `open()` (not
   `__init__`) so readiness probes and startup work; declare
   `device_type=GPU` support where relevant. Stateful nodes that cannot be
   replicated declare `singleton`.
3. **Dockerfile** — `FROM videoflow-base:py3.12` (`ARG BASE_IMAGE=` so deploy
   and CI can override), `uv pip install .`; add a `gpu.Dockerfile` (`FROM
   videoflow-base:py3.12-cuda`) when the component can use a GPU. Don't set an
   `ENTRYPOINT` — `python -m videoflow.worker` is inherited from the base.
4. **`component.yaml`** — the descriptor described above; validate with
   `videoflow component validate <dir>`.
5. **Tests** — a `tests/` folder runnable with `pytest` from the sub-package
   directory.

## Solutions

End-to-end flows that wire components together. Each one runs with a single
command from its own directory, locally or on a cluster:

| Solution | What it does | Run |
|---|---|---|
| [face_obfuscation](solutions/face_obfuscation) | Detects, tracks and Gaussian-blurs every face in a video. | `videoflow run-local face_obfuscation.py` / `videoflow deploy face_obfuscation.py` |
| [human_tracking](solutions/human_tracking) | Pose estimation + appearance re-identification: tracks people through occlusion. | `videoflow run-local human_tracking.py` / `videoflow deploy human_tracking.py` |

Each asks for its inputs the first time (a bundled sample clip is the default),
writes a `config.yaml`, builds its image, fetches its weights in the prep hook,
provisions a dev broker, runs, and tears down. See each solution's README for
its configuration reference.

> **Looking for the toy solutions?** `toy_calculator`, `toy_fusion`,
> `toy_router` and `toy_recovery` live in the core repo:
> [videoflow/solutions](https://github.com/videoflow/videoflow/tree/master/solutions),
> where they also serve as its end-to-end test suite. They need no models, no
> footage and no dependencies beyond the videoflow base image, and together they
> exercise most of the framework (the node types, both flow types, both join
> modes, replication and partitioned routing, lifecycle hooks, metadata and
> idempotent consumers, the prep hook). Deploy one in seconds to prove your
> cluster works before spending minutes on an ML solution.

## Writing a solution (a deployable graph)

A *solution* is an end-to-end flow that wires components together — see the table
above, with [solutions/human_tracking](solutions/human_tracking) as the fullest example. The
smallest complete one is
[toy_calculator](https://github.com/videoflow/videoflow/tree/master/solutions/toy_calculator)
in the core repo; it is the place to start reading.
Solutions live under `solutions/<name>/` and follow a file convention that
`videoflow deploy` understands; every file beyond the graph module is optional,
and each one you add removes a manual step from deployment.

```
solutions/<name>/
├── <name>.py              # graph module: exposes build_flow(cfg=None) -> Flow
├── <name>_nodes.py        # glue nodes in a real importable module (not the graph module)
├── common.py              # config loading shared by the graph and prep scripts
├── Dockerfile             # CPU image;  ARG BASE_IMAGE=videoflow-base:py3.12
├── gpu.Dockerfile         # GPU image;  ARG BASE_IMAGE=videoflow-base:py3.12-cuda
├── requirements[-gpu].txt # the solution's ML stack
├── config.example.yaml    # fully documented config (for humans)
├── config.template.yaml   # config template + x-questions/x-mounts/x-gpu (for deploy and run-local)
└── prepare.py             # idempotent one-shot prep hook (run in the image, before compile)
```

**The graph module.** Expose a factory `build_flow(cfg=None) -> Flow` that
builds the graph *without* running it. When called with no argument (which is
what `videoflow deploy` and `run-local` do) it must load its own config: the
one they resolved, exported as `VF_SOLUTION_CONFIG`, else the `config.yaml`
next to the module — never a path relative to the cwd:

```python
def build_flow(cfg=None):
    if cfg is None:
        here = os.path.dirname(os.path.abspath(__file__))
        cfg = load_config(os.environ.get('VF_SOLUTION_CONFIG') or os.path.join(here, 'config.yaml'))
    ...
    return Flow(sinks, flow_type=cfg.flow_type)
```

Give every node a stable `name=`. Put custom glue nodes in their own module
(`<name>_nodes.py`), never in the graph module: workers reconstruct them by
class path, and the graph module itself is not importable inside workers.

**Paths.** Everything the flow reads or writes at runtime must be an absolute
path that is valid *both* on the machine you deploy from *and* inside the pods
— deploy hostPath-mounts each one at the same location. The simplest way to
guarantee this: resolve every relative config path against the config file's
directory (see `solutions/human_tracking/common.py`), and list the path-bearing
config keys in `x-mounts` (below).

Two rules follow from *when* each path is resolved, and getting them wrong
produces a flow that compiles fine and then fails in the pod:

- **A path baked into node params at compile time must be under a same-path
  mount.** `build_flow` runs on the operator machine, so whatever it passes to a
  node (a video path, an output directory) is frozen into that node's
  parameters. It must therefore live under an `x-mounts` entry of the
  single-path form (`'{work_dir}'`), never under a remapped
  `host:container` entry (`'~/.videoflow:/root/.videoflow'`) — the pod would
  look for the host path, which doesn't exist there. Remapped entries are only
  for caches that components resolve *at runtime* inside the pod via `~` (which
  is `/root`, since worker containers run as root).
- **`build_flow` must not download or otherwise touch the network.** Compiling a
  graph has to stay side-effect free — `videoflow deploy --dry-run` calls it, and
  any progress output would also corrupt the manifest stream. Put fetches in
  `prepare.py` and have `build_flow` only compute the path the fetch will
  produce. `solutions/human_tracking/common.py` shows the pattern:
  `resolve_input()` (pure, used by the graph) alongside `fetch_input()`
  (downloads into `work_dir`, used by prep).

**Dockerfiles.** Solution images are built from the **repo root** (they COPY
the sibling component sub-packages), which is exactly what `videoflow deploy`
and `run-local` do: they use the git root enclosing the graph as the build
context, pick `gpu.Dockerfile` when the config values named in the template's
`x-gpu` say `gpu` (never because of the docker daemon on your machine), and
auto-build the `videoflow-base` image first when it's missing. Declare the
base as `ARG BASE_IMAGE=videoflow-base:py3.12[-cuda]` so deploy can find it.
Install the ML stack first, then the component sub-packages with `--no-deps`
(videoflow is in the base and the stack is already resolved), then COPY the
solution modules. The built image is deployed under a content-addressed tag and
pushed to the cluster profile's registry, so you never tag or push by hand.

**`config.template.yaml`.** A valid config body plus three blocks that deploy
strips from the generated `config.yaml`:

- `x-questions` — what deploy asks interactively when no config exists. Each
  entry: `key` (dotted path into the config; digit segments index int-keyed
  maps), `prompt`, `type` (`str` | `float` | `choice` + `choices` | `paths`),
  and optional `default`. The `paths` type takes comma-separated file paths,
  validates each exists, and expands them into a mapping using `item_key`
  (e.g. `'cam{i}'`) and `item_value` (e.g. `{video: '{path}'}`).
- `x-mounts` — path templates resolved against the final config, each becoming
  a mount on the prep/compile containers and every worker (a bind mount
  locally, a hostPath or claim volume in the cluster): `'{input_video}:ro'`
  (dotted lookup, `*` fans out, read-only; an empty value mounts nothing),
  `'{work_dir}'` (same path on host and container), and
  `'~/.videoflow:/root/.videoflow'` (host:container pair — maps the operator's
  weight cache onto the container root's; `--mount-home` re-roots the `~`).
- `x-gpu` — the config values that decide whether `gpu.Dockerfile` is built:
  `['{device}']`, or `['{device.*}']` for per-stage placement.

**`prepare.py`.** A hook deploy and run-local run *inside the solution image*
before compiling (its outputs get baked into the compiled node params). Contract:
accepts `--config PATH`; is **idempotent** — check each step's output and skip
it (print why) unless `--force`; fetches every model's weights with the same
`get_file` key and URL the component uses, so no worker downloads anything;
exits non-zero with the exact manual command to run when a step needs human
interaction, so the user can do it once and re-run.

With all of the above in place, running a solution is:

```bash
cd solutions/<name>
videoflow run-local <name>.py     # locally, the workers in the solution image
videoflow deploy <name>.py        # on the cluster
```

and every automated step still has a manual override (`--config`, `--image`,
`--nats`, `--mount`, `--no-prepare`, `--no-build`, ...). See the
[videoflow deployment guide](https://videoflow.github.io/videoflow/distributed/deploying-to-kubernetes.html)
(source: `../videoflow/docs/source/distributed/deploying-to-kubernetes.rst`)
for the full pipeline.

## Releasing

Every sub-package is released at **one version, in lockstep with the core
`videoflow` release**: contrib vX.Y.Z ships with core vX.Y.Z and pins
`videoflow>=X.Y.Z`. A contrib-only fix is released by bumping core's patch
version. There is nothing to run here by hand — the core release triggers it:

1. In `videoflow`, bump `videoflow/version.py`, merge to master, and run its
   **Publish to PyPI** workflow. Before uploading, that workflow installs the
   new core wheel and checks that this repo's descriptors still validate and
   every sub-package still builds against it.
2. Once the core tag exists, it dispatches this repo's **Release** workflow
   (`.github/workflows/release.yml`) with the same version. That run waits until
   `videoflow==X.Y.Z` is installable from PyPI, rewrites every `pyproject.toml`
   and `component.yaml` with `./set-version.py X.Y.Z`, validates and builds all
   the wheels, pushes a `Release vX.Y.Z` commit to master, and creates the
   `vX.Y.Z` tag and GitHub Release with the wheels attached.

If step 2 fails after core is already on PyPI, fix master and re-run it — the
workflow is idempotent:

```bash
gh workflow run release.yml -R videoflow/videoflow-contrib --ref master -f version=X.Y.Z
```

Never edit a version string by hand: `./set-version.py --check` runs in CI and
fails if any of the 38 files disagree. The other tags in this repository
(`detector_tf`, `tracktor`, `models`, `offside_models`, `example_videos`, ...)
are not versions; they are GitHub Releases that host model weights and sample
clips, referenced by URL from the code. Leave them alone.

## Example Usage
Consumers, producers and processors from the Videoflow-contrib library are used
in the same way as the components within Videoflow itself.

Videoflow now runs every node as its own worker process wired together through a
message broker (locally via `LocalProcessEngine`, in production on Kubernetes). A
flow is described by a `build_flow()` function that returns a `Flow` built from its
consumer (sink) nodes — producers are discovered automatically by walking the graph
backwards. Give each node a stable `name=` so it can be addressed across processes,
and store/forward `**kwargs` on any custom node so it can be reconstructed inside a
worker.

```python
import videoflow
from videoflow.core import Flow
from videoflow.core.constants import BATCH
from videoflow.consumers import VideofileWriter
from videoflow.producers import VideofileReader
from videoflow.processors.vision.annotators import BoundingBoxAnnotator
from videoflow.utils.downloader import get_file

BASE_URL_EXAMPLES = "https://github.com/videoflow/videoflow/releases/download/examples/"
VIDEO_NAME = 'intersection.mp4'
URL_VIDEO = BASE_URL_EXAMPLES + VIDEO_NAME

class FrameIndexSplitter(videoflow.core.node.ProcessorNode):
    def __init__(self, **kwargs):
        super(FrameIndexSplitter, self).__init__(**kwargs)

    def process(self, data):
        index, frame = data
        return frame

def build_flow():
    from videoflow_contrib.detector_tf import TensorflowObjectDetector
    input_file = get_file(VIDEO_NAME, URL_VIDEO)
    output_file = "output.avi"
    reader = VideofileReader(input_file, name = 'reader')
    frame = FrameIndexSplitter(name = 'frame')(reader)
    detector = TensorflowObjectDetector(name = 'detector')(frame)
    annotator = BoundingBoxAnnotator(name = 'annotator')(frame, detector)
    writer = VideofileWriter(output_file, fps = 30, name = 'writer')(annotator)
    return Flow([writer], flow_type = BATCH)

if __name__ == "__main__":
    # Direct run against a broker you started (docker compose up -d in the core
    # repo); `videoflow run-local my_flow.py` starts one for you instead.
    from videoflow.engines.local import LocalProcessEngine
    flow = build_flow()
    flow.run(LocalProcessEngine())
    flow.join()

# Deploy to Kubernetes (one workload per node) with the videoflow CLI:
#   videoflow deploy my_flow.py:build_flow --nats nats://nats:4222 --image <your-image>
```
