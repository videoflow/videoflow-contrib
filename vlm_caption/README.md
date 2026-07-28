# VLM caption

`VlmCaptioner` captions video frames with a Hugging Face vision-language model
(default: [Qwen/Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct)).
It takes an `(h, w, 3)` BGR frame and emits a one-line caption string. It is
also the **reference multi-GPU component** for videoflow RFC 0003.

## Multi-GPU usage

A 7B VLM in bf16 wants more memory than one small GPU has, so the descriptor
declares a default grant of two whole devices
(`spec: {resources: {gpu: {count: 2}}}`). Wiring the component into a graph by
descriptor picks that up automatically — no `gpu_count=` by hand:

```python
from videoflow.core import component

captioner = component('vlm_caption', device_type = 'gpu',
                      name = 'captioner')(frame)   # pod requests nvidia.com/gpu: 2
```

The descriptor count is a default, not a floor. For a smaller or quantized
model that fits one device, override it (either on the Python node or via
`component(..., gpu_count = 1)`):

```python
from videoflow_contrib.vlm_caption import VlmCaptioner

captioner = VlmCaptioner(model_id = 'Qwen/Qwen2.5-VL-3B-Instruct',
                         gpu_count = 1, name = 'captioner')(frame)
```

Why the node needs no device code at all: videoflow guarantees that inside a
worker the visible GPUs are exactly the granted GPUs, numbered `cuda:0..N-1`
with `N == gpu_count` — on Kubernetes the device plugin mounts only the granted
devices into the pod, and under `run-local` the engine partitions
`CUDA_VISIBLE_DEVICES` across workers. `open()` therefore just loads the model
with `device_map = 'auto'` and Hugging Face shards it across everything visible,
which is by construction everything granted. Multi-GPU grants need whole
exclusive devices — MIG slices and time-sliced units cannot be combined.

## CPU

`device_type = CPU` loads the model on CPU in float32. It works (slowly) and
exists mainly so the CPU image stays testable.

## Parameters

| Param | Default | Meaning |
|---|---|---|
| `model_id` | `Qwen/Qwen2.5-VL-7B-Instruct` | HF id of an image-text-to-text model |
| `prompt` | `Describe this image in one sentence.` | Instruction sent with every frame |
| `max_new_tokens` | `64` | Generation budget per caption |

The captioner is stateless: `nb_tasks > 1` and `partition_by` are both safe.

## Tests

The pure helpers (chat building, caption cleanup, BGR→RGB conversion) and the
worker-reconstruction contract are covered by weights-free tests that need
neither torch nor transformers:

```bash
cd vlm_caption && pytest
```
