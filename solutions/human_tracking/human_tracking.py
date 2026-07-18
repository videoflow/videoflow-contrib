'''
Human tracking — pose estimation, appearance encoding and re-identification.

Per frame: read → Detectron2 human pose → split keypoints/boxes → annotate the
skeletons; crop each person, encode their appearance, and feed boxes+features to
DeepSort so identities survive occlusion → annotate track ids → write the video.

Deploy to Kubernetes (config Q&A, image build, broker, run and teardown in one
command — see README.md):

    videoflow deploy human_tracking.py

Local run, all workers as subprocesses on this machine:

    python human_tracking.py --config config.yaml

The glue nodes live in ``human_tracking_nodes.py`` (a real importable module) so
distributed workers can reconstruct them by class path; ``main`` puts this
directory on PYTHONPATH so the worker subprocesses can import that module.
'''
from __future__ import annotations

import argparse
import os

from common import load_config
from human_tracking_nodes import (
    AppendFeaturesToBoundingBoxes,
    BoundingBoxesExtractor,
    ConvertTracksForAnotation,
    CropBoundingBoxes,
    FrameIndexSplitter,
    KeypointsExtractor,
)
from videoflow.consumers import VideofileWriter
from videoflow.core import Flow
from videoflow.processors.vision.annotators import TrackerAnnotator
from videoflow.producers import VideofileReader


def build_flow(cfg=None):
    if cfg is None:
        # Module-dir-relative so `videoflow deploy` works from any cwd.
        cfg = load_config(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.yaml'))
    from videoflow_contrib.detectron2 import Detectron2HumanPose, HumanPoseAnnotator
    from videoflow_contrib.humanencoder import HumanEncoder
    from videoflow_contrib.tracker_deepsort import DeepSort

    pose_cfg = cfg.pose
    enc = cfg.encoder
    trk = cfg.tracker

    reader = VideofileReader(cfg.resolve_input(), name='reader')
    frame = FrameIndexSplitter(name='frame')(reader)
    results = Detectron2HumanPose(
        architecture=pose_cfg.get('architecture', 'R50_FPN_3x'),
        device_type=cfg.device,
        name='pose')(frame)
    keypoints = KeypointsExtractor(name='keypoints')(results)
    bounding_boxes = BoundingBoxesExtractor(name='bounding-boxes')(results)
    anotated_keypoints = HumanPoseAnnotator(name='pose-annotator')(frame, keypoints)
    cropped_humans = CropBoundingBoxes(name='crop')(frame, bounding_boxes)
    human_features = HumanEncoder(
        batch_size=int(enc.get('batch_size', 32)),
        device_type=cfg.device,
        name='encoder')(cropped_humans)
    tracker_input = AppendFeaturesToBoundingBoxes(name='append-features')(bounding_boxes, human_features)
    tracks = DeepSort(
        min_height=int(trk.get('min_height', 0)),
        max_cosine_distance=float(trk.get('max_cosine_distance', 0.2)),
        nn_budget=trk.get('nn_budget'),
        name='tracker')(tracker_input)
    tracks_anotator_input = ConvertTracksForAnotation(name='convert-tracks')(tracks)
    anotated_tracks = TrackerAnnotator(name='track-annotator')(anotated_keypoints, tracks_anotator_input)
    writer = VideofileWriter(cfg.output_path(), name='writer')(anotated_tracks)
    return Flow([writer], flow_type=cfg.flow_type)


def main():
    ap = argparse.ArgumentParser(description='Track people across a video with pose + appearance re-id.')
    ap.add_argument('--config', default='config.yaml')
    ap.add_argument('--flow-type', choices=('batch', 'realtime'), default=None,
                    help='override the config flow_type (default: from config, else batch)')
    args = ap.parse_args()
    # The local engine spawns worker subprocesses that must import
    # `human_tracking_nodes` (and `common`) to reconstruct the glue nodes. Put
    # this directory on PYTHONPATH so those imports resolve in the workers.
    here = os.path.dirname(os.path.abspath(__file__))
    os.environ['PYTHONPATH'] = here + os.pathsep + os.environ.get('PYTHONPATH', '')

    cfg = load_config(args.config)
    if args.flow_type is not None:
        cfg.flow_type = args.flow_type
    from videoflow.engines.local import LocalProcessEngine
    flow = build_flow(cfg)
    engine = LocalProcessEngine(blob_redis_url=os.environ.get('VIDEOFLOW_BLOB_REDIS_URL'))
    flow.run(engine)
    flow.join()
    print(f'Wrote {cfg.output_path()}')


if __name__ == '__main__':
    main()
