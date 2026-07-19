# Human Tracking (pose + appearance re-identification)

Detects every person's skeleton with Detectron2, encodes their appearance, and
tracks them with DeepSort so identities survive occlusion and people leaving and
re-entering the frame. Writes an annotated video with skeletons and track ids.

The graph:

```
VideofileReader → FrameIndexSplitter ─┬→ Detectron2HumanPose ─┬→ KeypointsExtractor → HumanPoseAnnotator ─┐
                                      │                       │                                            │
                                      │                       └→ BoundingBoxesExtractor ─┬────────────────┐│
                                      │                                                  │                ││
                                      └→ CropBoundingBoxes → HumanEncoder → AppendFeaturesToBoundingBoxes ─┘│
                                                                                    │                       │
                                                                     DeepSort → ConvertTracksForAnotation → TrackerAnnotator → VideofileWriter
```

Appearance encoding is what separates this from a pure motion tracker: two people
who cross paths keep their ids because the encoder distinguishes them visually.

## Deploying to Kubernetes (one command)

With a local cluster (k3s / kind / minikube / Docker Desktop) plus docker and
kubectl, the host needs only the videoflow CLI — none of the ML dependencies:

```bash
pip install -e "/path/to/videoflow[deploy]"
cd solutions/human_tracking
videoflow deploy human_tracking.py
```

That asks which device and models to use, writes `config.yaml`, builds and loads
the image, downloads the sample clip and encoder weights, provisions a dev
NATS+Redis, runs the flow to completion, and tears it all down. The annotated
video lands in `out/` on this machine. See the
[deployment guide](../../../videoflow/docs/source/distributed/deploying-to-kubernetes.rst)
for the full pipeline and every override flag.

**GPU:** answer `gpu` when asked for the device (or set `device: gpu`) — deploy
then builds `gpu.Dockerfile` and requests a GPU for the pose and encoder pods.
The pose model is by far the heaviest stage; CPU works but is slow.

**Your own footage:** set `input_video` in `config.yaml` to an absolute path and
add a mount so the pods can see it:

```bash
videoflow deploy human_tracking.py --mount /data/my_video.mp4:ro
```

Leaving `input_video` empty uses the bundled sample clip (`people_walking.mp4`),
which `prepare.py` downloads into `work_dir`.

## Install (local runs & manual prep)

```bash
# from the videoflow-contrib repo root
pip install -e ./detectron2 -e ./tracker_deepsort -e ./humanencoder
pip install -r solutions/human_tracking/requirements.txt   # or requirements-gpu.txt
# plus torch/torchvision and detectron2 itself — see the Dockerfile for the exact
# index URLs and the source build.
```

## Run locally (no cluster)

One command — it asks for the config if there isn't one, fetches the sample clip
and encoder weights, starts a dev broker in Docker if none is running, runs every
node as a local subprocess, and cleans up after itself:

```bash
cd solutions/human_tracking
videoflow run-local human_tracking.py
```

Or drive it manually, which needs a broker of your own:

```bash
cd /path/to/videoflow && docker compose up -d nats redis   # NATS :4222, Redis :6379
export VIDEOFLOW_BLOB_REDIS_URL=redis://localhost:6379/0   # frames >512KB use the blob store

cd /path/to/videoflow-contrib/solutions/human_tracking
cp config.example.yaml config.yaml
python prepare.py --config config.yaml    # fetch the clip + encoder weights
python human_tracking.py --config config.yaml [--flow-type batch|realtime]
```

Output: `<work_dir>/<output_video>` (default `out/annotated_video.avi`).

## Configuration reference (`config.yaml`)

`videoflow deploy` asks for the starred (★) values; everything else has a
sensible default. Relative paths resolve against the config file's directory.

| Key | Default | Meaning |
|---|---|---|
| `input_video` | `''` (sample) | The video to process. Empty uses the bundled `people_walking.mp4` sample, which `prepare.py` downloads into `work_dir` (so the pods see it at the same path they were compiled with). Set an absolute path for your own footage and mount it with `--mount`. |
| `work_dir` | `./out` | Where the annotated video is written. Mounted read-write, so results appear on your machine. |
| `output_video` | `annotated_video.avi` | Output filename, written inside `work_dir`. The default writer codec pairs with `.avi`. |
| `device` ★ | `cpu` | `cpu` or `gpu` — sets `device_type` on the pose **and** encoder nodes. `gpu` needs the GPU image and a GPU node (deploy warns if the cluster isn't ready). |
| `flow_type` ★ | `batch` | `batch` for recorded files: loss-free, backpressured, runs to completion then exits. `realtime` only for a genuine live source — it drops frames to stay current. |
| `pose.architecture` ★ | `R50_FPN_3x` | Detectron2 keypoint model. `R50_FPN_3x` is the balanced default; `R101_FPN_3x` / `X101_FPN_3x` are more accurate and slower; `R50_FPN_1x` is faster and weaker. |
| `encoder.batch_size` | `32` | Person crops encoded per forward pass. Lower it if VRAM is tight; raise it for throughput on a big GPU. |
| `tracker.min_height` | `0` | Ignore detections shorter than this many pixels — useful to drop distant, unreliable figures. `0` keeps everything. |
| `tracker.max_cosine_distance` ★ | `0.2` | Appearance-match threshold. **Lower = stricter**: fewer id swaps between similar-looking people, but more identity fragmentation when someone's appearance changes. |
| `tracker.nn_budget` | `null` | Max appearance samples retained per identity. `null` is unbounded (best accuracy, grows with time); set e.g. `100` to cap memory on long videos. |

## Files

| File | Role |
|---|---|
| `human_tracking.py` | Graph module: `build_flow(cfg=None)` plus a local `main()`. |
| `human_tracking_nodes.py` | The six glue nodes in their own importable module, so distributed workers can reconstruct them by class path. |
| `common.py` | `load_config()` — resolves paths against the config file's directory; `resolve_input()` falls back to the sample clip. |
| `config.example.yaml` / `config.template.yaml` | Documented example / the template deploy asks questions from. |
| `prepare.py` | Idempotent input + weight pre-fetch, run by deploy before compiling. |
| `Dockerfile` / `gpu.Dockerfile` | CPU / CUDA images (detectron2 built from source), built from the repo root. |
