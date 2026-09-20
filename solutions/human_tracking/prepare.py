'''
Prep for the human-tracking solution: fetch the input video (the bundled sample
clip when the config leaves ``input_video`` empty) and warm the model weights —
the appearance encoder's and the Detectron2 pose model's — so no worker pod
downloads anything (a pod without internet access runs all the same).

`videoflow deploy human_tracking.py` and `videoflow run-local` run this
automatically inside the solution image before compiling; it can also be run by
hand:

    python prepare.py --config config.yaml

Idempotent: `get_file` skips anything already in ~/.videoflow/models.
'''
from __future__ import annotations

import argparse
import os

from common import load_config

# Must match videoflow_contrib.humanencoder.encoder.open() exactly — same cache
# key ('human_encoder.pb') and same source, so the node finds it already warmed.
ENCODER_URL = 'https://github.com/videoflow/videoflow-contrib/releases/download/models/humanencoder_mars_128.pb'
# Likewise videoflow_contrib.detectron2.humanpose.Detectron2HumanPose.open(): the
# cache key is 'detectron2_model.pkl' and the file is <architecture>.pkl there.
POSE_URL_BASE = 'https://github.com/videoflow/videoflow-contrib/releases/download/detectron2/'


def main():
    ap = argparse.ArgumentParser(description='Fetch the input video and warm the encoder weights.')
    ap.add_argument('--config', default='config.yaml')
    ap.add_argument('--force', action='store_true', help='re-download even if cached')
    ap.add_argument('--skip-encoder', action='store_true', help='do not pre-fetch the encoder weights')
    ap.add_argument('--skip-pose', action='store_true', help='do not pre-fetch the pose weights')
    args = ap.parse_args()
    cfg = load_config(args.config)

    from videoflow.utils.downloader import get_file

    # fetch_input downloads the sample clip when input_video is empty (build_flow
    # only ever resolves the path, never downloads).
    print('==> input video', flush=True)
    video = cfg.fetch_input()
    if not os.path.exists(video):
        raise SystemExit(f'input_video does not exist: {video}\n'
                         f'Fix the path in {cfg.path} (and make sure it is mounted into the pods).')
    print(f'    {video}')

    if not args.skip_encoder:
        print('==> appearance encoder weights', flush=True)
        path = get_file('human_encoder.pb', ENCODER_URL)
        print(f'    cached at {path}')

    if not args.skip_pose:
        architecture = cfg.pose.get('architecture', 'R50_FPN_3x')
        print(f'==> pose weights ({architecture})', flush=True)
        path = get_file('detectron2_model.pkl', POSE_URL_BASE + f'{architecture}.pkl')
        print(f'    cached at {path}')

    print('Prep complete.')


if __name__ == '__main__':
    main()
