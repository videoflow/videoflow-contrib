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

That asks for a couple of knobs, writes `config.yaml`, builds and loads the image,
downloads the sample clip and the detector weights, provisions a dev NATS+Redis,
runs the flow to completion, and tears it all down. The blurred video lands in
`out/` on this machine. See the
[deployment guide](../../../videoflow/docs/source/distributed/deploying-to-kubernetes.rst)
for the full pipeline and every override flag.

For a GPU run, pick `gpu` when asked for the device (or set `device: gpu` in the
config) — deploy then builds `gpu.Dockerfile` automatically.

**Your own footage:** set `input_video` in `config.yaml` to an absolute path and
add a mount so the pods can see it:

```bash
videoflow deploy face_obfuscation.py --mount /data/my_video.mp4:ro
```

Leaving `input_video` empty uses the bundled sample clip
(`street_crossing.mp4`), which `prepare.py` downloads into `work_dir`.

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
cp config.example.yaml config.yaml          # optionally set input_video
python prepare.py --config config.yaml      # fetch the sample clip + weights
python face_obfuscation.py --config config.yaml [--flow-type batch|realtime]
```

Output: `<work_dir>/<output_video>` (default `out/blurred_video.avi`).

## The sample clip

`street_crossing.mp4` is a 13-second, 1280×720 excerpt of
[*DiagonalCrosswalkYongeDundas*](https://commons.wikimedia.org/wiki/File:DiagonalCrosswalkYongeDundas.webm)
by Raysonho @ Open Grid Scheduler / Grid Engine, **CC0** (public domain
dedication), from Wikimedia Commons. It is a busy pedestrian crossing at
Yonge–Dundas Square in Toronto: a dense crowd walking towards the camera, which
is what makes it a useful privacy demo — the detector finds roughly five faces
per frame at the default threshold, ranging from 20 px in the crowd to 100 px in
the foreground. It is hosted as a release asset alongside the other solutions'
samples:
`https://github.com/videoflow/videoflow-contrib/releases/download/example_videos/street_crossing.mp4`.

## Verified run

Run end to end on a local **k3s** cluster on 2026-07-28, one worker pod per graph
node, all stages on CPU, from the bundled sample with an otherwise stock config
(only `work_dir` was pointed at a gitignored directory):

```bash
cd /home/jadiel/workspace/videoflow-contrib
docker build -f solutions/face_obfuscation/Dockerfile -t videoflow-face-obfuscation:r5 .

cd solutions/face_obfuscation
videoflow deploy face_obfuscation.py \
    --no-build --image videoflow-face-obfuscation:r5 \
    --config config.yaml --non-interactive \
    --namespace videoflow \
    --flow-id face-obfuscation --run-id face-obfuscation-d2 \
    --gpu-runtime-class nvidia --keep-infra
```

`--no-build --image <ref>:rN` rather than letting deploy autobuild: autobuild
picks `gpu.Dockerfile` whenever the *docker daemon* exposes an NVIDIA runtime —
regardless of `device: cpu` — and tags `:latest`, which Kubernetes defaults to
`imagePullPolicy: Always` and therefore re-pulls from a registry that hasn't got
it. Immutable `:rN` tags get `IfNotPresent`, which is what makes a locally
imported image usable at all.

It ends with `Flow face-obfuscation completed.` and writes 326 frames of
1280×720 at 25 fps to `<work_dir>/blurred_video.avi` (MJPEG, ~60 MB), every
detected and tracked face Gaussian-blurred.

**If your output comes out with blue skin**, the reader and the writer disagree
about channel order: `VideoFileReader` defaults to `swap_channels=False` (frames
stay BGR) while `VideofileWriter` defaults to `swap_channels=True`. This graph
passes `swap_channels=False` to the writer for that reason — keep it if you edit
`build_flow`.

## Configuration reference (`config.yaml`)

`videoflow deploy` asks for the starred (★) values; everything else has a
sensible default. Relative paths resolve against the config file's directory.

| Key | Default | Meaning |
|---|---|---|
| `input_video` | `''` (sample) | The video whose faces are blurred. Empty uses the bundled `street_crossing.mp4` sample, which `prepare.py` downloads into `work_dir` (so the pods see it at the same path they were compiled with). Set an absolute path for your own footage and mount it with `--mount`. |
| `work_dir` | `./out` | Where the output is written. Mounted read-write, so results appear on your machine. |
| `output_video` | `blurred_video.avi` | Output filename, written inside `work_dir`. **Must end in `.avi`** — videoflow's `VideofileWriter` only supports that container (the loader rejects anything else with a clear error). |
| `fps` | `25` | Frames per second of the written video. Match your source footage, or the output plays at the wrong speed. The sample clip is 25 fps. |
| `device` ★ | `cpu` | `cpu` or `gpu` — sets `device_type` on the detector node. `gpu` needs the GPU image and a GPU node (deploy warns if the cluster isn't ready). |
| `flow_type` ★ | `batch` | `batch` for recorded files: loss-free, backpressured, runs to completion then exits. `realtime` only for a genuine live source — it drops frames to stay current. |
| `detector.architecture` | `ssd-mobilenetv2` | Detector architecture; with `dataset: faces` this selects the face-trained SSD checkpoint (auto-downloaded). |
| `detector.dataset` | `faces` | Weight variant. Keep `faces` for face blurring. |
| `detector.num_classes` | `1` | Classes in the checkpoint (1 = face). |
| `detector.min_score_threshold` ★ | `0.1` | Confidence floor. **Lower it to blur more aggressively** — a missed face is a privacy leak, a false positive is just a blurred patch. `0.1` is what the wide street-scene sample needs (its faces are 20–100 px); raise it towards `0.3` for footage shot close to the subjects. |
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
