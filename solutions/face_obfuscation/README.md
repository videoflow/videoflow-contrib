# Face Obfuscation

Detects every face in a video, tracks them across frames, and Gaussian-blurs each
one — producing a privacy-safe copy of the footage.

The graph, per frame:

```
VideofileReader → FrameIndexSplitter ─┬→ TensorflowObjectDetector (faces) ─┬→ KalmanFilterBoundingBoxTracker ─┐
                                      │                                    │                                  │
                                      └────────────────────────────────────┴──→ BoundingboxObfuscator ────────┴→ VideofileWriter
```

The tracker matters for privacy: a face the detector misses in a single frame is
still blurred, because the Kalman tracker keeps predicting its box for
`tracker.max_age` frames.

## Run it

With videoflow installed (`pip install 'videoflow[all]'`, see the
[quick start](../../README.md#quick-start)) and docker running:

```bash
videoflow run-local videoflow-contrib://face_obfuscation      # on this machine
videoflow deploy videoflow-contrib://face_obfuscation         # on the cluster kubectl points at
```

That fetches this repository at your videoflow version into
`~/.videoflow/solutions/videoflow-contrib@v<version>/` once and works from
there — `config.yaml` and `out/` land next to the graph in that directory,
which is printed on every run. From a checkout, the path form is the same
thing and keeps them here:

```bash
cd solutions/face_obfuscation
videoflow run-local face_obfuscation.py
```

Both ask the same questions the first time (answer with Enter to take every
default: the bundled sample clip, CPU, batch), write `config.yaml`, build the
solution image from `Dockerfile` (and `videoflow-base` before it, pulled from `ghcr.io/videoflow` or built from a core checkout, once), run
`prepare.py` inside it to fetch the clip and the detector weights, and run the
flow to completion. None of the ML stack is installed on your machine: the
image is the environment. `run-local` runs every worker as a container of that
image against a dev NATS + Redis it starts in docker; `deploy` runs them as pods
against a broker it provisions in the namespace, and tears everything down at
the end. The blurred video lands in `out/blurred_video.avi`.

**Your own footage:** answer the `input_video` question with an absolute path
(or set it in `config.yaml`). It is mounted read-only into the workers.

**GPU:** answer `gpu` when asked for the device (or set `device: gpu`) — both
commands then build `gpu.Dockerfile` instead, and `deploy` requests a GPU for
the detector's pods (`run-local` hands the worker its device when the docker
daemon has the NVIDIA runtime).

**Multi-node or shared cluster:** put the per-cluster values (registry, RWX
claim, namespace, priority class) in a cluster profile once — see
[deploying and verifying](../../.claude/docs/DEPLOY_VERIFY.md) and the core
README's *Multi-node and shared clusters* — and answer the `work_dir` question
with a directory under the shared claim. The command stays the same.

## Configuration reference (`config.yaml`)

The starred (★) values are asked for; everything else has a sensible default.
Relative paths resolve against the config file's directory.

| Key | Default | Meaning |
|---|---|---|
| `work_dir` ★ | `./out` | Where the output (and the sample clip) is written. Mounted read-write into the workers, so results appear on your machine — on a multi-node cluster, a directory under the shared claim. |
| `input_video` ★ | `''` (sample) | The video whose faces are blurred. Empty uses the bundled `people_walking.mp4` sample, which `prepare.py` downloads into `work_dir` (so the workers see it at the path it was compiled with). An absolute path for your own footage, mounted read-only. |
| `output_video` | `blurred_video.avi` | Output filename, written inside `work_dir`. **Must end in `.avi`** — videoflow's `VideofileWriter` only supports that container (the loader rejects anything else with a clear error). |
| `fps` | `30` | Frames per second of the written video. Match your source footage, or the output plays at the wrong speed. |
| `device` ★ | `cpu` | `cpu` or `gpu` — sets `device_type` on the detector node and selects the image (`x-gpu`). `gpu` needs a GPU node in the cluster (deploy warns if the cluster isn't ready). |
| `flow_type` ★ | `batch` | `batch` for recorded files: loss-free, backpressured, runs to completion then exits. `realtime` only for a genuine live source — it drops frames to stay current. |
| `detector.architecture` | `ssd-mobilenetv2` | Detector architecture; with `dataset: faces` this selects the face-trained SSD checkpoint (auto-downloaded). |
| `detector.dataset` | `faces` | Weight variant. Keep `faces` for face blurring. |
| `detector.num_classes` | `1` | Classes in the checkpoint (1 = face). |
| `detector.min_score_threshold` ★ | `0.2` | Confidence floor. **Lower it to blur more aggressively** — a missed face is a privacy leak, a false positive is just a blurred patch. |
| `tracker.max_age` | `12` | Frames a face keeps being blurred after the detector loses it. Raise for flickery detections. |
| `tracker.min_hits` | `0` | Detections before a track is emitted. `0` blurs from the first frame — the right default for privacy. |
| `blur.expand` | `0.20` | Grows each box by this fraction per side before blurring, so hair and chin are covered rather than just the tight detector box. |
| `blur.kernel` | `23` | Gaussian kernel size in pixels (forced odd). Larger = blurrier. |
| `blur.sigma` ★ | `30` | Gaussian standard deviation. Larger = blurrier and less reversible. |

## Developer mode (the ML stack on your machine)

Only worth it when you are editing the components themselves. Build the same
environment the `Dockerfile` describes (TensorFlow from `requirements.txt`, the
`detector_tf` and `tracker_sort` sub-packages, and `protobuf>=5.27` restored
after TensorFlow downgrades it) into a virtualenv that also has videoflow, and
`videoflow run-local face_obfuscation.py` then runs the workers as host
processes instead of containers. Do not install these into the environment that
holds the `videoflow` CLI itself: the contrib components pin stacks that do not
coexist.

## Files

| File | Role |
|---|---|
| `face_obfuscation.py` | Graph module: `build_flow(cfg=None)` plus a local `main()`. |
| `face_obfuscation_nodes.py` | Glue nodes (`FrameIndexSplitter`, `BoundingboxObfuscator`) in their own importable module, so workers can reconstruct them by class path. |
| `common.py` | `load_config()` — resolves every path against the config file's directory; `resolve_input()` falls back to the sample clip. |
| `config.example.yaml` / `config.template.yaml` | Documented example / the template the questions, mounts and image choice (`x-gpu`) come from. |
| `prepare.py` | Idempotent input + weight pre-fetch, run inside the image before compiling. |
| `Dockerfile` / `gpu.Dockerfile` | CPU / CUDA images, built from the repo root. |
