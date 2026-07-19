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

## Deploying to Kubernetes (one command)

With a local cluster (k3s / kind / minikube / Docker Desktop) plus docker and
kubectl, the host needs only the videoflow CLI — none of the ML dependencies:

```bash
pip install -e "/path/to/videoflow[deploy]"
cd solutions/face_obfuscation
videoflow deploy face_obfuscation.py
```

That asks for the video path and a couple of knobs, writes `config.yaml`, builds
and loads the image, pre-fetches the detector weights, provisions a dev
NATS+Redis, runs the flow to completion, and tears it all down. The blurred video
lands in `out/` on this machine. See the
[deployment guide](../../../videoflow/docs/source/distributed/deploying-to-kubernetes.rst)
for the full pipeline and every override flag.

For a GPU run, pick `gpu` when asked for the device (or set `device: gpu` in the
config) — deploy then builds `gpu.Dockerfile` automatically.

## Install (local runs & manual prep)

```bash
# from the videoflow-contrib repo root
pip install -e ./detector_tf -e ./tracker_sort
pip install -r solutions/face_obfuscation/requirements.txt      # or requirements-gpu.txt
```

## Run locally (no cluster)

One command — it asks for the config if there isn't one, warms the weight cache,
starts a dev broker in Docker if none is running, runs every node as a local
subprocess, and cleans up after itself:

```bash
cd solutions/face_obfuscation
videoflow run-local face_obfuscation.py
```

Or drive it manually, which needs a broker of your own:

```bash
cd /path/to/videoflow && docker compose up -d nats redis   # NATS :4222, Redis :6379
export VIDEOFLOW_BLOB_REDIS_URL=redis://localhost:6379/0   # frames >512KB use the blob store

cd /path/to/videoflow-contrib/solutions/face_obfuscation
cp config.example.yaml config.yaml     # then set input_video
python prepare.py --config config.yaml # warm the weight cache
python face_obfuscation.py --config config.yaml [--flow-type batch|realtime]
```

Output: `<work_dir>/<output_video>` (default `out/blurred_video.avi`).

## Configuration reference (`config.yaml`)

`videoflow deploy` asks for the starred (★) values; everything else has a
sensible default. Relative paths resolve against the config file's directory.

| Key | Default | Meaning |
|---|---|---|
| `input_video` ★ | — | **Required.** The video whose faces are blurred. Mounted read-only into the pods. |
| `work_dir` | `./out` | Where the output is written. Mounted read-write, so results appear on your machine. |
| `output_video` | `blurred_video.avi` | Output filename, written inside `work_dir`. **Must end in `.avi`** — videoflow's `VideofileWriter` only supports that container (the loader rejects anything else with a clear error). |
| `fps` | `30` | Frames per second of the written video. Match your source footage, or the output plays at the wrong speed. |
| `device` ★ | `cpu` | `cpu` or `gpu` — sets `device_type` on the detector node. `gpu` needs the GPU image and a GPU node (deploy warns if the cluster isn't ready). |
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

## Files

| File | Role |
|---|---|
| `face_obfuscation.py` | Graph module: `build_flow(cfg=None)` plus a local `main()`. |
| `face_obfuscation_nodes.py` | Glue nodes (`FrameIndexSplitter`, `BoundingboxObfuscator`) in their own importable module, so distributed workers can reconstruct them by class path. |
| `common.py` | `load_config()` — resolves every path against the config file's directory. |
| `config.example.yaml` / `config.template.yaml` | Documented example / the template deploy asks questions from. |
| `prepare.py` | Idempotent weight pre-fetch, run by deploy before compiling. |
| `Dockerfile` / `gpu.Dockerfile` | CPU / CUDA images, built from the repo root. |
