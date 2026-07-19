'''
FIFA-style offside detection — end-to-end videoflow flow.

Per camera: synced read → soccer detect → track (BoT-SORT-ReID) → pose + team →
pack compact features. All cameras fuse by event time into a 3D world-state
stream; the offside engine finds kick→teammate-touch pairs and emits verdicts; the
visualizer renders them.

Run (after the three prep scripts), from this directory:
    python offside.py --config config.yaml

The glue nodes live in ``offside_nodes.py`` (a real importable module) so the
distributed workers can reconstruct them by class path (the local engine puts
this directory on each worker's PYTHONPATH automatically).
'''
from __future__ import annotations

import argparse
import os

from common import load_config
from offside_nodes import BallPick, FeaturePacker, FrameIndexSplitter, PersonBoxes, WorldStateJsonlWriter


def build_flow(cfg=None):
    if cfg is None:
        # Module-dir-relative so `videoflow deploy` works from any cwd.
        cfg = load_config(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.yaml'))
    from videoflow.core import Flow
    from videoflow.core.constants import BATCH, REALTIME
    from videoflow.core.policies import JoinPolicy
    from videoflow_contrib.multiview_fuser import MultiviewFuser
    from videoflow_contrib.offside_engine import OffsideEngine
    from videoflow_contrib.offside_visualizer import OffsideVisualizer
    from videoflow_contrib.pose_topdown import PoseTopDown
    from videoflow_contrib.soccer_detector import SoccerDetector
    from videoflow_contrib.synced_video_reader import SyncedVideoReader
    from videoflow_contrib.team_classifier import TeamClassifier
    from videoflow_contrib.tracker_botsort import BoxmotTracker

    offsets = cfg.load_offsets()
    calib = cfg.load_calibration()
    teams = cfg.load_teams()
    ref_fps = float(cfg.fusion.get('ref_fps', 30.0))
    det = cfg.detector

    packers = []
    for cam in cfg.cameras:
        reader = SyncedVideoReader(cfg.videos[cam], offset_s=offsets[cam]['offset_s'],
                                   drift_ppm=offsets[cam].get('drift_ppm', 0.0),
                                   start_s=cfg.start_s, end_s=cfg.end_s, name=f'reader-{cam}')
        frame = FrameIndexSplitter(name=f'frame-{cam}')(reader)
        dets = SoccerDetector(checkpoint=det.get('checkpoint'),
                              resolution=int(det.get('resolution', 1288)),
                              conf_ball=float(det.get('conf_ball', 0.15)),
                              tile_inference=bool(det.get('tile_inference', False)),
                              device_type=cfg.device_for('detector'), name=f'detector-{cam}')(frame)
        pboxes = PersonBoxes(name=f'personboxes-{cam}')(dets)
        ball = BallPick(name=f'ball-{cam}')(dets)
        tracks = BoxmotTracker(method='botsort', device_type=cfg.device_for('tracker'), name=f'tracker-{cam}')(frame, pboxes)
        pose = PoseTopDown(backend='rtmw', device_type=cfg.device_for('pose'), name=f'pose-{cam}')(frame, tracks)
        team = TeamClassifier(centroids=teams, name=f'team-{cam}')(frame, tracks)
        packers.append(FeaturePacker(cam=cam, name=cam)(tracks, pose, team, ball))

    period_ms = 1000.0 / ref_fps
    n = len(cfg.cameras)
    quorum = int(cfg.fusion.get('quorum', 2))
    fuser = MultiviewFuser(
        cameras=list(cfg.cameras), calibration=calib, ref_fps=ref_fps,
        min_views=2, name='fuser',
        # Quorum lives on the join policy (emit a moment with >= `quorum` views).
        join_policy=JoinPolicy(mode='time', tolerance_ms=period_ms / 2 - 1,
                               timeout_seconds=60, quorum=min(quorum, n),
                               missing='drop', max_pending=4096),
    )(*packers)

    verdicts = OffsideEngine(pitch_length=cfg.pitch_length, pitch_width=cfg.pitch_width,
                             attack_direction=cfg.attack_direction, ref_fps=ref_fps,
                             name='engine')(fuser)
    results_dir = os.path.join(cfg.work_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)
    viz = OffsideVisualizer(output_dir=results_dir,
                            videos=cfg.videos,
                            offsets={c: offsets[c]['offset_s'] for c in cfg.cameras},
                            calibration=calib, pitch_length=cfg.pitch_length,
                            pitch_width=cfg.pitch_width, name='visualizer')(verdicts)
    sinks = [viz]
    if cfg.debug_overlays:
        sinks.append(WorldStateJsonlWriter(cfg.work_path('world_states.jsonl'),
                                           name='worldstate-writer')(fuser))
    # BATCH (default) for recorded clips — lossless, blocking backpressure. REALTIME
    # for a genuine live source: freshest-wins retention (a straggler frame can't stall
    # live verdicts), but frames are dropped if the source outruns the fuser.
    flow_type = REALTIME if cfg.flow_type == 'realtime' else BATCH
    return Flow(sinks, flow_type=flow_type)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='config.yaml')
    ap.add_argument('--flow-type', choices=('batch', 'realtime'), default=None,
                    help='override the config flow_type (default: from config, else batch)')
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.flow_type is not None:
        cfg.flow_type = args.flow_type
    from videoflow.engines.local import LocalProcessEngine
    flow = build_flow(cfg)
    engine = LocalProcessEngine(blob_redis_url=os.environ.get('VIDEOFLOW_BLOB_REDIS_URL'))
    flow.run(engine)
    flow.join()
    if engine.failures():
        engine.report_failures()
        raise SystemExit(1)


if __name__ == '__main__':
    main()
