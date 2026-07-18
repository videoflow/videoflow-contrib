# FIFA-style Semi-Automated Offside Detection

An end-to-end offside (VAR-style) system on [videoflow](https://github.com/videoflow/videoflow).
It ingests 2–5 time-synced camera angles of a soccer play, detects players and the
ball, finds who kicks the ball and who touches it next, and — when the next toucher
is a **teammate** of the kicker — reconstructs the situation in 3D at the kick
instant, computes the offside line, and produces a verdict with visualizations.

This is a **video-only approximation** of FIFA's Semi-Automated Offside Technology.
FIFA uses 12 dedicated roof cameras (29 skeletal points/player @ 50 Hz) plus a
500 Hz IMU inside the ball for kick timing. Here, kick/touch events are inferred
from change-points in the reconstructed 3D ball trajectory combined with player-limb
proximity, and 3D geometry comes from multi-view triangulation of pose keypoints
against per-camera pitch calibration.

## Verdict semantics

Per resolved kick→teammate-touch pair the engine emits one of:

| Verdict | Meaning |
|---|---|
| `OFFSIDE` | Receiver's forward-most legal body part is beyond both the ball and the second-last defender by more than the uncertainty band. |
| `ONSIDE` | Receiver is level with or behind the line (or in their own half). |
| `TOO_CLOSE` | Within the uncertainty band — the honest "we can't call it" for a video-only system. |
| `INCONCLUSIVE` | Missing inputs (no ball fix at the kick, <2 defenders visible, receiver track lost). |

**Law approximations.** Legal body parts = head/body/feet (arms/hands excluded,
boundary at the shoulder). "Becomes involved" is approximated by *the next touch by
a teammate*. Timing is frame-rate limited (≈ ±1 frame at 30 fps); record at 60 fps
for tighter kick localization. Treat outputs as decision support, not officiating.

## Models & licenses

| Stage | Model / method | Component | License |
|---|---|---|---|
| Detection (player/GK/ref/ball) | RF-DETR-Large SoccerNet (`backend='rfdetr'`, needs torch≥2.4) or YOLOv8 soccer (`backend='yolo'`, runs anywhere) | `soccer_detector` | Apache-2.0 |
| Tracking | BoxMOT — BoT-SORT-ReID (default) | `tracker_botsort` | AGPL-3.0 |
| Pose | rtmlib top-down — Halpe26 body+feet (`rtmpose`, default) or WholeBody-133 (`rtmw`) | `pose_topdown` | Apache-2.0 |
| Pitch calibration | YOLOv8x-pose 32 pitch keypoints (roboflow layout) + PnP/DLT solve | `pitch_calib` | AGPL-3.0 |
| Team ID | HSV histogram (default) / SigLIP-2 | `team_classifier` | MIT |
| Time sync | Audio cross-correlation | `sync_offsets.py` | (scipy) |
| 3D reconstruction | Ground-plane association + DLT triangulation | `multiview_fuser` | MIT |
| Offside logic | Change-point touch detection + Law-11 geometry | `offside_engine` | MIT |
| Visualization | Projected line + top-down diagram + clips | `offside_visualizer` | MIT |

Sapiens pose was evaluated and **rejected** as a default: its weights are CC-BY-NC
(non-commercial), incompatible with the component marketplace. It remains available
behind `pose_topdown(allow_noncommercial=True)`. The AGPL components (tracker, pitch
calibration) are isolated — the rest of the stack is permissive.

## Recording guide

- **2–5 fixed cameras** on tripods/poles around the pitch. Opposite sidelines with
  **≥ 40° baselines** between views gives the best triangulation; more cameras add
  robustness to occlusion.
- Each camera must see **pitch line markings** (for calibration). Lock focus and
  exposure; don't pan/zoom.
- **1080p60** or **4K30** (higher resolution helps the small, distant ball).
- Capture a **loud shared sound at the start** — a clap or the kickoff whistle — so
  audio sync has a sharp common event.
- **Measure the pitch** length and width with a tape (amateur pitches vary; the
  offside line is computed in metres).

## Deploying to Kubernetes (one command)

With a local cluster (k3s / kind / minikube / Docker Desktop) and docker + kubectl
on this machine, the host needs only the videoflow CLI — none of the ML deps:

```bash
pip install -e "/path/to/videoflow[deploy]"
cd solutions/offside
videoflow deploy offside.py
```

That single command:
1. asks for the config inputs (video paths, pitch dims, team names, flow type)
   and writes `config.yaml` — driven by `config.template.yaml`; skipped when a
   `config.yaml` already exists or `--config` is passed;
2. builds the solution image from `gpu.Dockerfile` (auto-building the
   `videoflow-base` image first if missing) and loads it into the detected
   cluster flavor — skipped with `--image <ref>` / `--no-build`;
3. runs `prepare.py` inside that image (weights → offsets → calibration → team
   fit, each skipped when its output exists) — `--no-prepare` to skip;
4. compiles the graph inside the image (no local ML deps needed) and hostPath-
   mounts the template's `x-mounts` (videos read-only, `out/`, weight caches)
   into every worker pod — plus any extra `--mount /path[:ro]`;
5. provisions a dev NATS + Redis in the namespace (skipped when you pass
   `--nats` / `--blob-redis-url`), applies the flow, warns with exact fix
   commands if the GPU node label or NVIDIA device plugin is missing;
6. for a BATCH flow: waits for completion, then tears down the run *and* the
   infra it created (`--keep-infra` to keep NATS/Redis for faster redeploys).

Results land in `out/results/` on this machine (the work dir is mounted).
If automatic pitch calibration fails, deploy stops and prints the one-time
`calibrate.py --manual` command; re-running deploy resumes where prep left off.

## Install (local runs & manual prep)

```bash
# from the videoflow-contrib repo root
pip install -e ./synced_video_reader -e ./soccer_detector -e ./tracker_botsort \
            -e ./pose_topdown -e ./team_classifier -e ./pitch_calib \
            -e ./multiview_fuser -e ./offside_engine -e ./offside_visualizer
pip install -r solutions/offside/requirements.txt      # or requirements-gpu.txt
# ffmpeg must be on PATH (audio extraction for sync)
```

## Prep (run once per recording — `videoflow deploy` runs this for you)

One shot (idempotent; skips finished steps, `--force` to redo):

```bash
cd solutions/offside
python prepare.py --config config.yaml
```

Or the three steps by hand:

```bash
cd solutions/offside
cp config.example.yaml config.yaml     # then edit: video paths, measured pitch dims

python sync_offsets.py --config config.yaml   # → out/offsets.json
python calibrate.py    --config config.yaml   # → out/calib/*.json + calib_overlay_*.png
python fit_teams.py    --config config.yaml   # → out/teams.json + teams_montage.png
```

Sanity-check each:
- **offsets.json** — confidence ≥ 3 per camera; if low, ensure a shared clap/whistle.
- **calib_overlay_\<cam\>.png** — the yellow pitch wireframe should sit on the real
  lines (RMS < 5 px reported). If markings are faint, rerun `calibrate.py --cam <cam> --manual`.
- **teams_montage.png** — the two team rows should be visibly different colors; set
  `team_names` in the config accordingly.

## Run

**On Kubernetes** — see [Deploying to Kubernetes](#deploying-to-kubernetes-one-command)
above: `videoflow deploy offside.py` does config, prep, images, broker, run, and
teardown in one command.

**Locally** (all workers as subprocesses on this machine; needs the
[Install](#install-local-runs--manual-prep) step):

```bash
# from the videoflow repo (NATS + Redis for the distributed workers)
cd /path/to/videoflow && docker compose up -d nats redis
export VIDEOFLOW_BLOB_REDIS_URL=redis://localhost:6379/0   # frames >512KB use the blob store

cd /path/to/videoflow-contrib/solutions/offside
python offside.py --config config.yaml [--flow-type batch|realtime]
```

Either way, outputs land in `<work_dir>/results/` (default `out/results/`):
- `verdicts.json` — every verdict + the touch log.
- `verdict_<n>.json` — one record per verdict (margin, uncertainty, gids, timestamps).
- `still_<n>_<cam>.png` — each camera at the kick instant with the projected offside
  line, receiver/second-defender/ball markers, and a verdict banner.
- `topdown_<n>.png` — top-down pitch diagram with the line and all players.
- `clip_<n>_<cam>.avi` — optional annotated clip around the play.

GPU is optional (CPU works but is slow — RF-DETR + RTMW per camera). Set
`debug_overlays: true` to also dump `world_states.jsonl`.

## Configuration reference (`config.yaml`)

All inputs to the solution live in one YAML file. `videoflow deploy` generates it
by asking for the starred (★) values; everything else has a sensible default.
Relative paths resolve against the config file's directory.

| Key | Default | Meaning |
|---|---|---|
| `work_dir` | `./out` | Where every artifact goes: prep outputs (`offsets.json`, `calib/`, `teams.json`) and `results/`. Mounted into the pods, so results appear on this machine. |
| `flow_type` ★ | `batch` | `batch` for recorded clips: loss-free, backpressured, runs to completion then exits. `realtime` for a genuine live source: freshest-frame-wins, a straggler can't stall verdicts, but frames drop if the source outruns the fuser. Only use `realtime` with a live/RTSP-cadence producer, never fast file replay. |
| `cameras.<cam>.video` ★ | — | One entry per camera (2–5), e.g. `cam0: {video: /data/cam0.mp4}`. **Order matters**: it is the fusion input order, and the first camera is the audio-sync reference (offset 0). Fixed tripod cameras with audio, ideally covering both penalty areas from different angles. |
| `pitch.length` ★ | `105.0` | Pitch length in metres. **Measure it** — amateur pitches vary a lot and the offside line is computed in metres (FIFA standard 105). |
| `pitch.width` ★ | `68.0` | Pitch width in metres (FIFA standard 68). |
| `attack_direction` | `auto` | Which way each team attacks. `auto` infers it per team from the goalkeepers' positions; override with `{0: '+x', 1: '-x'}` (team id → direction along the pitch length axis) when auto is unreliable (e.g. short clips without keepers in view). |
| `team_names` ★ | `{0: Reds, 1: Blues}` | Cosmetic labels for the two jersey-color clusters found by `fit_teams.py`. Check `out/teams_montage.png` to see which cluster is team 0 and name them accordingly. |
| `trim.start_s` / `trim.end_s` | `null` | Optional trim to the play of interest, in event-seconds on the common (synchronized) time axis. `null` processes the full recording. |
| `detector.checkpoint` | `null` | Path to an RF-DETR checkpoint; `null` auto-downloads the soccer-finetuned one. |
| `detector.resolution` | `1288` | Detector input resolution (must be divisible by 56). Raise it when the ball is small/distant; costs GPU time quadratically. |
| `detector.conf_ball` | `0.15` | Ball detection confidence threshold. Lower it if the ball is missed (more false positives for the tracker to filter). |
| `detector.tile_inference` | `false` | Split each frame into tiles and detect per tile — recovers very distant balls at a large speed cost. |
| `fusion.ref_fps` | `30.0` | Rate of the fused 3D world-state stream. Match your cameras' true fps. |
| `fusion.quorum` | `2` | Minimum number of cameras that must see a moment before it is fused. `2` is the minimum for 3D triangulation; raise with more cameras for robustness. |
| `debug_overlays` | `false` | Also write per-camera annotated `.avi` (tracks/teams/skeletons) and `world_states.jsonl` — slow, for debugging the pipeline. |

Prep artifacts consumed at compile time (produced by `prepare.py`, all under
`work_dir`): `offsets.json` (per-camera audio-sync offset + drift), `calib/<cam>.json`
(camera→pitch homography + intrinsics), `teams.json` (jersey color centroids).

### Deploy-time parameters

`videoflow deploy offside.py` additionally accepts (all optional): `--config PATH`
(use an existing config), `--image REF` (skip the image build), `--nats URL` /
`--blob-redis-url URL` (bring your own broker instead of the auto-provisioned dev
NATS/Redis), `--mount /path[:ro]` (extra hostPath mounts beyond the automatic
videos/work-dir/cache ones), `--no-prepare` / `--no-build` (skip those stages),
`--keep-infra` (leave NATS/Redis up for faster redeploys), `--flow-id NAME`
(stable resource naming across redeploys), and `--dry-run` (print the manifests
without touching the cluster).

## Tuning

| Symptom | Knob |
|---|---|
| Ball missed at range | detector `resolution` ↑, `conf_ball` ↓, `tile_inference: true` |
| Too many false touches | engine `vel_jump_ms` ↑, `dir_change_deg` ↑ |
| Missed touches | engine `touch_radius_m` ↑, `vel_jump_ms` ↓ |
| ID switches | tracker `method: deepocsort`/`boosttrack` |
| Calibration drift on faint lines | `calibrate.py --manual` |
| Verdicts all `TOO_CLOSE` | improve camera baselines/resolution (widens confidence at range) |

## Scaling & near-live notes

Per-camera stages (detector, pose, team) are `partitionable` — raise `nb_tasks`
(with `partition_by='trace_id'`) to scale across GPUs. The fuser and engine are
single-task by design (event-time join + sequential state).

### BATCH vs REALTIME (`flow_type`)

The flow runs in one of two modes, selectable in the config (`flow_type: batch|realtime`)
or on the CLI (`python offside.py --flow-type realtime`, which overrides the config):

- **`batch`** (default) — for **recorded clips**. Lossless: streams use interest
  retention with blocking backpressure, so no frame is dropped, and the flow runs to
  completion (each worker exits when its upstream drains). This is what the prep-script
  workflow below assumes.
- **`realtime`** — for a **genuine live source**. Streams keep only the freshest frame
  (`max_msgs=1`, drop-oldest, non-blocking publish), so a straggler frame can never stall
  a live verdict — but frames are **dropped whenever the source outruns the fuser**. The
  fuser/engine/visualizer code is identical in both modes (the event-time join policy is
  set explicitly, independent of flow type); only the broker retention and termination
  characteristics change.

  Because REALTIME is freshest-wins, it is only appropriate when the producer emits at
  true real-time cadence and dropping frames to keep up is acceptable. **Do not** point it
  at fast file replay through `SyncedVideoReader` — the reader emits frames as fast as it
  can decode, so most would be evicted before the fuser sees them and the kick→touch
  detection would fall apart. For a real near-live deployment, swap `SyncedVideoReader`
  for an RTSP reader with `timestamp_source='clock'` on PTP/NTP-disciplined hosts (the
  time-join keys on `event_ts`, so live sources must be clock-synchronized) and set
  `flow_type: realtime`; the rest of the pipeline is unchanged.

  The REALTIME broker path (event-time join → engine verdicts → visualizer) is verified
  end-to-end on a synthetic 2-camera fixture driven by real-time-paced producers, yielding
  the same OFFSIDE verdict as BATCH.


## Model weights (auto-downloaded on first use)

Each model backend fetches its weights on first `open()` (cached under `~/.videoflow`
or the library's own cache) — no manual download step is required to run:

- **Detector (yolo):** `martinjolif/yolo-football-player-detection` (HF) — classes
  `{0:ball, 1:goalkeeper, 2:player, 3:referee}`.
- **Detector (rfdetr):** `julianzu9612/RFDETR-Soccernet` (HF, `weights/checkpoint_best_regular.pth`,
  1.57 GB) — classes `{0:ball, 1:player, 2:referee, 3:goalkeeper}`.
- **Pose:** OpenMMLab RTMPose-x Halpe26 ONNX (rtmlib cache).
- **Pitch:** `martinjolif/yolo-football-pitch-detection` (HF) — 32 keypoints in
  roboflow `SoccerPitchConfiguration` order.
- **Tracker ReID:** pass `reid_weights=` to enable appearance ReID; otherwise BoT-SORT
  runs motion + camera-motion-compensation only.

### Source, mirror & durability

The detector and pitch weights live on **community HuggingFace repos** (the primary
source). To stay resilient if an upstream repo disappears or rate-limits, each of these
also has a **durable fallback** on our own release:
`https://github.com/videoflow/videoflow-contrib/releases/download/offside_models/`.
`videoflow.utils.downloader.get_file` receives an ordered `[upstream, mirror]` URL list
and tries them in order, so a run keeps working even if the community repo goes away.
Pose (OpenMMLab), tracker ReID (boxmot zoo) and the optional SigLIP team backend (Google)
stay on their official project infrastructure.

To (re)build the mirror release (maintainers only — needs the `gh` CLI and release
access to the `videoflow` org): pre-fetch the files locally with the script below, then
`gh release create offside_models --title 'Offside model weights' \
~/.videoflow/models/rfdetr_soccernet.pth \
~/.videoflow/models/yolo-football-player-detection.pt \
~/.videoflow/models/yolo-football-pitch-detection.pt`.

### Pre-fetching (containers / offline / cache volume)

The Docker images stay lean — weights are **not** baked in; they download on first run.
To avoid re-downloading (especially the 1.57 GB RF-DETR checkpoint) on every fresh
container, warm the caches once and mount them as volumes:

```bash
# Warm ~/.videoflow/models and ~/.cache/rtmlib on the host (uses the same URL lists the
# components use; --skip-rfdetr / --skip-pose to trim). Run in an env with the deps.
python download_weights.py                 # full default pipeline
python download_weights.py --skip-rfdetr   # yolo detector backend only (skip 1.57 GB)

# Then reuse those caches inside the container instead of re-downloading:
docker run --rm \
  -v "$HOME/.videoflow:/root/.videoflow" \
  -v "$HOME/.cache:/root/.cache" \
  <offside-image> python offside.py --config config.yaml
```

Set `VIDEOFLOW_HOME` to relocate the `~/.videoflow` cache if `$HOME` isn't writable.

## Verification status

The model inference paths have been **run with real weights on a real soccer frame**
(verified: YOLO detector → BoT-SORT tracker → RTMPose pose chain producing 26/26
confident keypoints; HSV team classifier splitting two teams; YOLO pitch keypoints →
PnP/DLT calibration, which solves exactly on full-pitch geometry). The **rfdetr
backend** requires `torch>=2.4` and so cannot run on x86-64 macOS (no PyTorch wheel
> 2.2.2 there) — its output-format logic is unit-tested via a mock, and it runs on
Linux/GPU or Apple Silicon.

Weights download at runtime from their upstream sources with a durable GitHub-release
fallback (see above); pre-fetch with `download_weights.py` for offline/container runs.
The maintainer step of uploading the mirror release itself is a one-time ops task.
Not yet done: the full **distributed flow** end-to-end (needs NATS + Redis + Docker +
multi-camera footage). Two inherent limits,
documented rather than hidden: sub-frame kick timing is frame-rate-limited (~±1 frame
at 30 fps — why FIFA uses a 500 Hz ball IMU), and the ground-point back-projection is
association-grade (~0.2–0.3 m at range; the precise offside line uses the triangulated
3D keypoints, not the ground point).
