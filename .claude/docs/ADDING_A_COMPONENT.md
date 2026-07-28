# Adding a component

Checklist for creating a new contrib sub-package. Read
[../agents/videoflow-author.md](../agents/videoflow-author.md) and
[../../../videoflow/.claude/docs/NODE_CONTRACT.md](../../../videoflow/.claude/docs/NODE_CONTRACT.md)
first — this document covers the packaging, not the node semantics.

> Keep this file in sync with the actual conventions. If the descriptor schema, base image, or
> build tooling changes, update this document and the root `README.md` together.

## Files

```
my_component/
├── pyproject.toml
├── component.yaml
├── Dockerfile
├── gpu.Dockerfile              # only if it has a GPU variant; exact filename
├── videoflow_contrib/
│   └── my_component/           # NO videoflow_contrib/__init__.py
│       ├── __init__.py         # re-export the node classes
│       └── nodes.py
└── tests/
```

Copy `offside_engine/` as the starting point — it is the cleanest example and has no ML
dependencies to strip out.

## `pyproject.toml`

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "videoflow_contrib_my_component"      # underscores
version = "1.0.0"
description = "..."
license = { text = "MIT" }
requires-python = ">=3.12"
dependencies = [
    "videoflow>=1.0.0",
    "numpy>=1.24",
]

[project.optional-dependencies]
gpu = ["torch>=2.0"]                          # CUDA variants go here, not in dependencies

[tool.hatch.build.targets.wheel]
only-include = ["videoflow_contrib/my_component"]   # required — namespace package
```

`only-include` is not optional. Without it the wheel claims the whole `videoflow_contrib`
namespace and collides with every other contrib package.

## `component.yaml`

The machine-readable descriptor. It is what makes the component usable from a non-Python graph and
what `videoflow deploy` validates against at graph-build time.

```yaml
apiVersion: videoflow.io/v1
kind: Component
metadata:
  name: videoflow/my-component        # dashes
  version: "1.0.0"
  license: MIT
  description: One clear sentence.
spec:
  role: processor                     # producer | processor | consumer
  protocol: 1
  runtime:
    pythonClass: videoflow_contrib.my_component.MyNode
    images:
      cpu: ghcr.io/videoflow/contrib-my-component:1.0.0
      gpu: ghcr.io/videoflow/contrib-my-component:1.0.0-cuda
  device: [cpu, gpu]
  params:
    schema:                           # JSON Schema — must match the constructor
      type: object
      additionalProperties: false
      properties:
        threshold: {type: number, default: 0.5, minimum: 0, maximum: 1}
  io:
    inputs:
      - {name: frame, type: videoflow.v1.Tensor}
    output: {type: videoflow.v1.Value}
  constraints: {partitionable: true}  # or {singleton: true}
```

- **`params.schema` must track the constructor.** Adding, renaming, or re-defaulting a constructor
  argument is a descriptor change. This is the single most commonly missed step.
- **`constraints`**: `singleton: true` for anything stateful across messages — trackers,
  aggregators, engines, and producers. `partitionable: true` only if replicas can safely each own
  a subset of keys.
- **`io` types** are `videoflow.v1.Tensor` (arrays/frames) or `videoflow.v1.Value` (structured
  data).

## Errors

Two things to get right before the component is done, both from `videoflow.core.errors`:

1. **Validate parameters with `ConfigError`** (or `CapabilityError` for something the component
   declines to do), in `__init__` for pure-value checks and in `open()` for anything needing the
   framework. Always pass `remedy = ...` naming the fix and `node = self.name`. The CLI renders
   both and exits 2; a bare `ValueError` gets a traceback and exit 1.

2. **Choose a disposition for every per-message failure.** A malformed payload is `SchemaError`
   (poison — dead-lettered on the first failure). A wedged accelerator is `DeviceError`
   (worker_fatal — the message is handed back, the worker stops). A blipped upstream service is
   `UpstreamUnavailable` (transient — retried). Anything unclassified defaults to transient.

Then register the framework exceptions you cannot subclass, **in the module that actually imports
the framework**:

```python
register_error_classifier(tf.errors.ResourceExhaustedError, WORKER_FATAL)   # tf at module scope
register_classifier_for('torch.cuda.OutOfMemoryError', WORKER_FATAL, _torch_oom_type)  # torch lazy
```

`soccer_detector/detector.py` is the reference for the lazy (torch, from `open()`) form,
`detector_tf/tensorflow_utils.py` for the eager one. Full rationale and the disposition table:
[../agents/videoflow-author.md](../agents/videoflow-author.md#failing-correctly--the-disposition-decides-what-a-failure-costs).

## Dockerfiles

```dockerfile
ARG BASE_IMAGE=videoflow-base:py3.12
FROM ${BASE_IMAGE}
WORKDIR /app
COPY . ./
RUN uv pip install --system --no-cache .
# ENTRYPOINT (python -m videoflow.worker) is inherited from the base image.
```

The GPU variant uses `videoflow-base:py3.12-cuda`, installs `'.[gpu]'`, and adds
`--break-system-packages`. **Never set `ENTRYPOINT`.** Build the base images first from the
videoflow repo root: `./docker/build-images.sh`.

## Verify

```bash
cd my_component
uv build                                  # wheel builds without the ML stack installed
pytest                                    # if the component has tests
cd .. && ./validate-components.sh         # descriptor validates
```

Then confirm the node actually reconstructs the way a worker would:

```python
node = MyNode(threshold = 0.7)
assert type(node)(**node.get_params()).get_params() == node.get_params()
```

That two-line check catches the `self._<name>` mistake before it becomes a pod crash loop.

And confirm the classifier registration actually took effect — a registration in a module nothing
imports silently never runs:

```python
from videoflow.core.errors import classify
import videoflow_contrib.my_component        # noqa: the import is the point
assert classify(TheFrameworkOomError()) == 'worker_fatal'
```

## Finally

Update the components table in the root [README.md](../../README.md), and add a `README.md` to the
sub-package if it needs more explanation than its descriptor gives.
