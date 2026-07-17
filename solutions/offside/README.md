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

## Install

```bash
# from the videoflow-contrib repo root
pip install -e ./synced_video_reader -e ./soccer_detector -e ./tracker_botsort \
            -e ./pose_topdown -e ./team_classifier -e ./pitch_calib \
            -e ./multiview_fuser -e ./offside_engine -e ./offside_visualizer
pip install -r solutions/offside/requirements.txt      # or requirements-gpu.txt
# ffmpeg must be on PATH (audio extraction for sync)
```

## Prep (three scripts, run once per recording)

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

```bash
# from the videoflow repo (NATS + Redis for the distributed workers)
cd /path/to/videoflow && docker compose up -d nats redis
export VIDEOFLOW_BLOB_REDIS_URL=redis://localhost:6379/0   # frames >512KB use the blob store

cd /path/to/videoflow-contrib/solutions/offside
python offside.py --config config.yaml
```

Outputs land in `out/results/`:
- `verdicts.json` — every verdict + the touch log.
- `verdict_<n>.json` — one record per verdict (margin, uncertainty, gids, timestamps).
- `still_<n>_<cam>.png` — each camera at the kick instant with the projected offside
  line, receiver/second-defender/ball markers, and a verdict banner.
- `topdown_<n>.png` — top-down pitch diagram with the line and all players.
- `clip_<n>_<cam>.avi` — optional annotated clip around the play.

GPU is optional (CPU works but is slow — RF-DETR + RTMW per camera). Set
`debug_overlays: true` to also dump `world_states.jsonl`.

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
single-task by design (event-time join + sequential state). For a near-live mode,
swap `SyncedVideoReader` for an RTSP reader with `timestamp_source='clock'` on
PTP-disciplined hosts and switch the flow to `REALTIME`; the rest is unchanged.


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
