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

## Finally

Update the components table in the root [README.md](../../README.md), and add a `README.md` to the
sub-package if it needs more explanation than its descriptor gives.
