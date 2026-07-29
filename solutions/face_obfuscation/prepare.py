'''
Prep for the face-obfuscation solution: fetch the input video (the bundled
sample clip when the config leaves ``input_video`` empty) and pre-fetch the face
detector weights into the videoflow cache so the first worker pod doesn't stall
(and so an offline or air-gapped run works at all).

`videoflow deploy face_obfuscation.py` runs this automatically inside the
solution image before compiling; it can also be run by hand:

    python prepare.py --config config.yaml [--force]

Idempotent: `get_file` skips anything already in ~/.videoflow/models.
'''
from __future__ import annotations

import argparse
import os

from common import load_config

BASE_URL_DETECTION = 'https://github.com/videoflow/videoflow-contrib/releases/download/detector_tf/'


def main():
    ap = argparse.ArgumentParser(
        description='Fetch the input video and pre-fetch face-detector weights into the videoflow cache.')
    ap.add_argument('--config', default='config.yaml')
    ap.add_argument('--force', action='store_true', help='re-download even if cached')
    args = ap.parse_args()
    cfg = load_config(args.config)

    from videoflow.utils.downloader import get_file

    det = cfg.detector
    model_file = f"{det.get('architecture', 'ssd-mobilenetv2')}_{det.get('dataset', 'faces')}.pb"
    print(f'==> detector weights: {model_file}', flush=True)
    path = get_file(model_file, BASE_URL_DETECTION + model_file)
    print(f'    cached at {path}')

    # fetch_input downloads the sample clip when input_video is empty (build_flow
    # only ever resolves the path, never downloads).
    print('==> input video', flush=True)
    video = cfg.fetch_input()
    if not os.path.exists(video):
        raise SystemExit(f'input_video does not exist: {video}\n'
                         f'Fix the path in {cfg.path} (and make sure it is mounted into the pods).')
    print(f'    {video}')
    print('Prep complete.')


if __name__ == '__main__':
    main()
