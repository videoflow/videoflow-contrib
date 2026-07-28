# Video Captioning

Describes a video with a vision-language model and writes the descriptions out as
a **subtitle track** — `captions.srt` and `captions.vtt`, ready to drop next to
the video in VLC or into a browser `<track>` element.

It is also the reference deployment of videoflow's multi-GPU support (RFC 0003):
the captioner declares whole GPUs per replica and Hugging Face shards the model
across them, with no device code anywhere in this solution.

The graph, per sampled frame:

```
SampledVideoFileReader ─┬→ FrameSelector ─→ VlmCaptioner (N replicas × M GPUs) ─┐
                        │                                                       │
                        └→ CueSelector ─────────────────────────────────────────┴→ CaptionCueAssembler ─┬→ SubtitleWriter
                                                                                                        ├→ JsonLinesConsumer
                                                                                                        └→ CommandlineConsumer
```

Three parts of that shape are load-bearing, and each is a thing a first attempt
usually gets wrong:

- **The reader samples, not a filter node.** A 7B VLM costs seconds per frame, so
  the producer publishes one frame in `every_n_frames`. A downstream filter can't
  do this — in videoflow, a `process()` that returns `None` still publishes a
  message, so it would move exactly as much data.
- **The frame goes down one branch only.** `CueSelector` keeps the two numbers a
  subtitle cue needs and drops the pixels, so the branch that bypasses the model
  doesn't push megabytes per sample through the broker.
- **The join needs a large pending budget.** The cue branch runs the whole video
  ahead of the model, so nearly every sampled frame is an open join group at
  once. The policy default caps that at 256 groups and evicts past it — silently
  losing captions. `join.max_pending` (default 100000) is why the track comes out
  complete.

`SubtitleWriter` buffers and writes from `close()` because a subtitle format is
ordered and a cue ends where the next one begins: no cue is final until the one
after it has arrived, and captions arrive out of order as soon as the captioner
is replicated. The other two sinks exist so a long run is observable while it
happens.

## Deploying to Kubernetes (one command)

With a local cluster (k3s / kind / minikube / Docker Desktop) plus docker and
kubectl, the host needs only the videoflow CLI — none of the ML dependencies:

```bash
pip install -e "/path/to/videoflow[deploy]"
cd solutions/video_captioning
videoflow deploy video_captioning.py --gpu-runtime-class nvidia
```

That asks for the video path and a few knobs, writes `config.yaml`, builds and
loads the image, pre-fetches the model weights, provisions a dev NATS+Redis, runs
the flow to completion, and tears it all down. The subtitle files land in `out/`
on this machine.

`--gpu-runtime-class nvidia` is required on clusters where the NVIDIA runtime is
opt-in (k3s among them): without it the pod schedules happily and starts
device-less, then fails deep inside model loading. See the
[deployment guide](../../../videoflow/docs/source/distributed/deploying-to-kubernetes.rst)
for the full pipeline and every override flag.

## GPUs

Total demand is `captioner.workers × captioner.gpu_count` whole devices.

| Config | Devices claimed | When |
|---|---|---|
| `workers: 1, gpu_count: 2` (default) | 2 | Qwen2.5-VL-7B in bf16 — doesn't fit one small card |
| `workers: 1, gpu_count: 1` | 1 | a 3B or quantized model |
| `workers: 3, gpu_count: 1` | 3 | throughput: three replicas competing for frames |
| `device: cpu` (`gpu_count` must be 1) | 0 | testing the pipeline; minutes per frame |

Multi-GPU grants need **whole exclusive devices**. A model cannot span MIG slices
or time-sliced units, so `gpu_count > 1` on a time-sliced cluster will not do what
the number says.

Nothing in this solution places a tensor. The captioner loads with
`device_map='auto'`, and the framework guarantees that inside a worker the
visible GPUs are exactly the granted GPUs (`cuda:0..N-1`) — on Kubernetes via the
device plugin, under `run-local` via `CUDA_VISIBLE_DEVICES` partitioning.

## Install (local runs & manual prep)

```bash
# from the videoflow-contrib repo root
pip install -e ./vlm_caption
pip install -r solutions/video_captioning/requirements.txt      # or requirements-gpu.txt
```

## Run locally (no cluster)

One command — it asks for the config if there isn't one, warms the model cache,
starts a dev broker in Docker if none is running, runs every node as a local
subprocess, and cleans up after itself:

```bash
cd solutions/video_captioning
videoflow run-local video_captioning.py
```

Or drive it manually, which needs a broker of your own:

```bash
cd /path/to/videoflow && docker compose up -d nats redis   # NATS :4222, Redis :6379
export VIDEOFLOW_BLOB_REDIS_URL=redis://localhost:6379/0   # frames >512KB use the blob store

cd /path/to/videoflow-contrib/solutions/video_captioning
cp config.example.yaml config.yaml     # then set input_video
python prepare.py --config config.yaml # probe the video, warm the model cache
python video_captioning.py --config config.yaml
```

Start with `max_captions: 5` the first time: it bounds the run to a few minutes
and still exercises every edge of the graph.

## Output

Everything lands in `work_dir` (default `out/`):

| File | What it is |
|---|---|
| `captions.srt` | SubRip subtitle track — the primary artifact. Play with `vlc video.mp4 --sub-file out/captions.srt` |
| `captions.vtt` | The same cues as WebVTT, for a browser `<track src=...>` |
| `captions.jsonl` | One JSON record per caption (`index`, `start`, `caption`), flushed as it is produced — `tail -f` it to watch a long run |

`captions.srt` and `captions.vtt` are rewritten atomically at the end of each
run. `captions.jsonl` is **appended**, so it accumulates across runs against the
same `work_dir` — it is a log, not an artifact.

Worker stdout carries one line per caption as it is generated, so
`kubectl logs -l videoflow.io/run-id=<run>` shows progress too.

## Configuration reference

`config.example.yaml` documents every key. The ones that decide what a run costs:

| Key | Default | Meaning |
|---|---|---|
| `input_video` | — | required; the video to caption |
| `every_n_frames` | 60 | caption one frame in N (60 ≈ one caption per 2s of 30fps footage) |
| `max_captions` | -1 | stop after this many captions; -1 for the whole video |
| `device` | `gpu` | `gpu` or `cpu` |
| `captioner.model_id` | `Qwen/Qwen2.5-VL-7B-Instruct` | any HF image-text-to-text model |
| `captioner.prompt` | `Describe this image in one sentence.` | the whole "what kind of caption" control |
| `captioner.workers` | 1 | competing replicas — the throughput knob |
| `captioner.gpu_count` | 2 | whole GPUs per replica the model is sharded across |
| `cues.min_seconds` / `max_seconds` | 1.0 / 5.0 | how long a cue stays on screen |
| `join.max_pending` | 100000 | open join groups allowed; must exceed the caption count |

Changing the prompt is the cheapest way to change what the solution *does* —
`'List the objects visible in this image.'` turns it into an index,
`'Describe what the people are doing, in one sentence.'` into an activity log.

`prepare.py` prints the caption count and a rough runtime before you deploy; a
config that implies an overnight job says so there rather than in a pod.
