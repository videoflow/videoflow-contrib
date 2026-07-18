'''
Prep for the human-tracking solution: fetch the input video (the bundled sample
clip when the config leaves ``input_video`` empty) and warm the appearance
encoder weights, so the first worker pod doesn't stall on a download.

`videoflow deploy human_tracking.py` runs this automatically inside the solution
image before compiling; it can also be run by hand:

    python prepare.py --config config.yaml

Idempotent: `get_file` skips anything already in ~/.videoflow/models. The
Detectron2 pose weights are fetched by detectron2's own model zoo on first use
(they need the torch stack, so they are left to the worker).
'''
from __future__ import annotations

import argparse
import os

from common import load_config

# Must match videoflow_contrib.humanencoder.encoder.open() exactly — same cache
# key ('human_encoder.pb') and same source, so the node finds it already warmed.
ENCODER_URL = 'https://github.com/videoflow/videoflow-contrib/releases/download/models/humanencoder_mars_128.pb'


def main():
    ap = argparse.ArgumentParser(description='Fetch the input video and warm the encoder weights.')
    ap.add_argument('--config', default='config.yaml')
    ap.add_argument('--force', action='store_true', help='re-download even if cached')
    ap.add_argument('--skip-encoder', action='store_true', help='do not pre-fetch the encoder weights')
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

    print('==> pose weights are fetched by the detectron2 model zoo on first use (needs torch).')
    print('Prep complete.')


if __name__ == '__main__':
    main()
