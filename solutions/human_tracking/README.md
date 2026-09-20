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

## Run it

With videoflow installed (`pip install 'videoflow[all]'`, see the
[quick start](../../README.md#quick-start)) and docker running:

```bash
videoflow run-local videoflow-contrib://human_tracking      # on this machine
videoflow deploy videoflow-contrib://human_tracking         # on the cluster kubectl points at
```

That fetches this repository at your videoflow version into
`~/.videoflow/solutions/videoflow-contrib@v<version>/` once and works from
there — `config.yaml` and `out/` land next to the graph in that directory,
which is printed on every run. From a checkout, the path form is the same
thing and keeps them here:

```bash
cd solutions/human_tracking
videoflow run-local human_tracking.py
```

Both ask the same questions the first time (Enter takes every default: CPU,
batch), write `config.yaml`, build the solution image from `Dockerfile` (torch,
TensorFlow and a detectron2 source build: ten minutes or so, once; and
`videoflow-base` before it, pulled from `ghcr.io/videoflow` or built from a core checkout), run `prepare.py` inside it to fetch the sample clip
and every model's weights, and run the flow to completion. None of that stack is
installed on your machine. `run-local` runs every worker as a container of the
image against a dev NATS + Redis it starts in docker; `deploy` runs them as pods
against a broker it provisions in the namespace, and tears everything down at
the end. The annotated video lands in `out/annotated_video.avi`.

**Your own footage:** set `input_video` in `config.yaml` to an absolute path. It
is mounted read-only into the workers. Leaving it empty uses the bundled sample
clip (`people_walking.mp4`), which `prepare.py` downloads into `work_dir`.

**GPU:** answer `gpu` when asked for the device (or set `device: gpu`) — both
commands then build `gpu.Dockerfile` instead, and `deploy` requests a GPU for
the pose and encoder pods (`run-local` hands the workers their devices when the
docker daemon has the NVIDIA runtime). The pose model is by far the heaviest
stage; CPU works but is slow.

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
| `work_dir` ★ | `./out` | Where the annotated video (and the sample clip) is written. Mounted read-write into the workers, so results appear on your machine — on a multi-node cluster, a directory under the shared claim. |
| `input_video` | `''` (sample) | The video to process. Empty uses the bundled `people_walking.mp4` sample, which `prepare.py` downloads into `work_dir` (so the workers see it at the path it was compiled with). Set an absolute path for your own footage; it is mounted read-only. |
| `output_video` | `annotated_video.avi` | Output filename, written inside `work_dir`. The default writer codec pairs with `.avi`. |
| `device` ★ | `cpu` | `cpu` or `gpu` — sets `device_type` on the pose **and** encoder nodes and selects the image (`x-gpu`). `gpu` needs a GPU node in the cluster (deploy warns if the cluster isn't ready). |
| `flow_type` ★ | `batch` | `batch` for recorded files: loss-free, backpressured, runs to completion then exits. `realtime` only for a genuine live source — it drops frames to stay current. |
| `pose.architecture` | `R50_FPN_3x` | Detectron2 keypoint model. The component ships the config for this one; `prepare.py` pre-fetches its weights. |
| `encoder.batch_size` | `32` | Person crops encoded per forward pass. Lower it if VRAM is tight; raise it for throughput on a big GPU. |
| `tracker.min_height` | `0` | Ignore detections shorter than this many pixels — useful to drop distant, unreliable figures. `0` keeps everything. |
| `tracker.max_cosine_distance` ★ | `0.2` | Appearance-match threshold. **Lower = stricter**: fewer id swaps between similar-looking people, but more identity fragmentation when someone's appearance changes. |
| `tracker.nn_budget` | `null` | Max appearance samples retained per identity. `null` is unbounded (best accuracy, grows with time); set e.g. `100` to cap memory on long videos. |

## Developer mode (the ML stack on your machine)

Only worth it when you are editing the components themselves. Build the same
environment the `Dockerfile` describes (CPU torch from the PyTorch index,
TensorFlow and scipy from `requirements.txt`, `protobuf>=5.27` restored after
TensorFlow downgrades it, detectron2 built from source, then the `detectron2`,
`tracker_deepsort` and `humanencoder` sub-packages) into a virtualenv that also
has videoflow, and `videoflow run-local human_tracking.py` then runs the workers
as host processes instead of containers. Do not install these into the
environment that holds the `videoflow` CLI itself: the contrib components pin
stacks that do not coexist.

## Files

| File | Role |
|---|---|
| `human_tracking.py` | Graph module: `build_flow(cfg=None)` plus a local `main()`. |
| `human_tracking_nodes.py` | The six glue nodes in their own importable module, so workers can reconstruct them by class path. |
| `common.py` | `load_config()` — resolves paths against the config file's directory; `resolve_input()` falls back to the sample clip. |
| `config.example.yaml` / `config.template.yaml` | Documented example / the template the questions, mounts and image choice (`x-gpu`) come from. |
| `prepare.py` | Idempotent input + weight pre-fetch (sample clip, encoder and pose weights), run inside the image before compiling. |
| `Dockerfile` / `gpu.Dockerfile` | CPU / CUDA images (detectron2 built from source), built from the repo root. |
