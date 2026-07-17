'''
Pre-fetch all model weights the offside pipeline needs, so the first real run (or a
freshly built container) starts without downloading anything.

Weights are cached exactly where the components load them from at runtime:
  - detector + pitch YOLO/RF-DETR weights → ``~/.videoflow/models/`` (via ``get_file``;
    honors ``$VIDEOFLOW_HOME``). Each source tries the upstream URL first, then our
    durable GitHub-release mirror.
  - pose ONNX (rtmlib)                     → ``~/.cache/rtmlib``.

The URL constants are imported from the component modules, so this stays a single
source of truth — no duplicated URLs here.

Usage:
    python download_weights.py                 # full default pipeline (rfdetr + yolo + pitch + rtmw pose)
    python download_weights.py --skip-rfdetr   # skip the 1.57 GB RF-DETR checkpoint (yolo backend only)
    python download_weights.py --skip-pose     # skip the pose ONNX (needs rtmlib/onnxruntime)
    python download_weights.py --pose rtmpose  # warm the Halpe26 pose model instead of rtmw
'''
from __future__ import annotations

import argparse
import sys

from videoflow.utils.downloader import get_file


def _fetch(label: str, fname: str, origin) -> None:
    print(f'\n==> {label}')
    path = get_file(fname, origin)
    print(f'    cached at {path}')


def main() -> int:
    ap = argparse.ArgumentParser(description='Pre-fetch offside model weights into the runtime caches.')
    ap.add_argument('--skip-rfdetr', action='store_true',
                    help='skip the 1.57 GB RF-DETR SoccerNet checkpoint')
    ap.add_argument('--skip-yolo', action='store_true',
                    help='skip the YOLO player-detection weights')
    ap.add_argument('--skip-pitch', action='store_true',
                    help='skip the pitch-landmark weights')
    ap.add_argument('--skip-pose', action='store_true',
                    help='skip warming the pose ONNX (rtmlib)')
    ap.add_argument('--pose', default='rtmw', choices=['rtmw', 'rtmpose'],
                    help='pose backend to warm (default: rtmw)')
    args = ap.parse_args()

    # Detector (soccer_detector): yolo (small) + rfdetr (large). Both route through
    # get_file → [upstream HF, GitHub-release mirror].
    from videoflow_contrib.soccer_detector.detector import _RFDETR_WEIGHTS, _YOLO_WEIGHTS
    if not args.skip_yolo:
        _fetch('detector: YOLO player-detection (~41 MB)', *_YOLO_WEIGHTS)
    if not args.skip_rfdetr:
        _fetch('detector: RF-DETR SoccerNet checkpoint (~1.57 GB)', *_RFDETR_WEIGHTS)

    # Pitch calibration (pitch_calib): yolo32 landmark model.
    if not args.skip_pitch:
        from videoflow_contrib.pitch_calib.landmarks import _DEFAULT_WEIGHTS as PITCH
        _fetch('pitch: YOLO landmark model (~130 MB)', *PITCH['yolo32'])

    # Pose (pose_topdown / rtmlib): instantiating RTMPose downloads + extracts the ONNX
    # into ~/.cache/rtmlib. Best-effort — needs rtmlib + onnxruntime installed.
    if not args.skip_pose:
        print(f'\n==> pose: rtmlib {args.pose} ONNX (into ~/.cache/rtmlib)')
        try:
            from rtmlib import RTMPose
            from videoflow_contrib.pose_topdown.pose import _POSE_MODELS
            url, size, _ = _POSE_MODELS[args.pose]
            RTMPose(onnx_model=url, model_input_size=size, backend='onnxruntime', device='cpu')
            print('    rtmlib cache warmed')
        except ImportError as e:
            print(f'    skipped (rtmlib/onnxruntime not installed: {e}); '
                  f're-run with those deps, or pass --skip-pose')

    print('\nDone.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
