'''
Face obfuscation — detect, track and blur every face in a video.

Per frame: read → detect faces (TensorFlow SSD) → track (Kalman/SORT, so faces
missed in one frame stay blurred) → Gaussian-blur every box → write the video.

Deploy to Kubernetes (config Q&A, image build, broker, run and teardown in one
command — see README.md):

    videoflow deploy face_obfuscation.py

Local run, all workers as subprocesses on this machine:

    python face_obfuscation.py --config config.yaml

The glue nodes live in ``face_obfuscation_nodes.py`` (a real importable module)
so distributed workers can reconstruct them by class path; ``main`` puts this
directory on PYTHONPATH so the worker subprocesses can import that module.
'''
from __future__ import annotations

import argparse
import os

from common import load_config
from face_obfuscation_nodes import BoundingboxObfuscator, FrameIndexSplitter
from videoflow.consumers import VideofileWriter
from videoflow.core import Flow
from videoflow.producers import VideofileReader


def build_flow(cfg=None):
    if cfg is None:
        # Module-dir-relative so `videoflow deploy` works from any cwd.
        cfg = load_config(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.yaml'))
    from videoflow_contrib.detector_tf import TensorflowObjectDetector
    from videoflow_contrib.tracker_sort import KalmanFilterBoundingBoxTracker

    det = cfg.detector
    trk = cfg.tracker
    blur = cfg.blur

    reader = VideofileReader(cfg.input_video, name='reader')
    frame = FrameIndexSplitter(name='frame')(reader)
    faces = TensorflowObjectDetector(
        num_classes=int(det.get('num_classes', 1)),
        architecture=det.get('architecture', 'ssd-mobilenetv2'),
        dataset=det.get('dataset', 'faces'),
        min_score_threshold=float(det.get('min_score_threshold', 0.2)),
        device_type=cfg.device,
        name='detector')(frame)
    tracked_faces = KalmanFilterBoundingBoxTracker(
        max_age=int(trk.get('max_age', 12)),
        min_hits=int(trk.get('min_hits', 0)),
        name='tracker')(faces)
    blurred_faces = BoundingboxObfuscator(
        expand=float(blur.get('expand', 0.20)),
        kernel=int(blur.get('kernel', 23)),
        sigma=float(blur.get('sigma', 30)),
        name='obfuscator')(frame, faces, tracked_faces)
    writer = VideofileWriter(cfg.output_path(), fps=cfg.fps, name='writer')(blurred_faces)
    return Flow([writer], flow_type=cfg.flow_type)


def main():
    ap = argparse.ArgumentParser(description='Detect, track and blur faces in a video.')
    ap.add_argument('--config', default='config.yaml')
    ap.add_argument('--flow-type', choices=('batch', 'realtime'), default=None,
                    help='override the config flow_type (default: from config, else batch)')
    args = ap.parse_args()
    # The local engine spawns worker subprocesses that must import
    # `face_obfuscation_nodes` (and `common`) to reconstruct the glue nodes. Put
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
