'''
Prep for the video-captioning solution: fetch and probe the input video (the
bundled sample clip when the config leaves ``input_video`` empty) and pre-fetch
the vision-language model into the Hugging Face cache.

The model fetch is the reason this hook exists. A 7B VLM is ~16GB of weights;
without a warm cache the first worker pod stalls for the length of that download
before it captions anything, every replica downloads it again, and an offline or
air-gapped run does not work at all. Prep runs once, in the solution image,
against a cache directory mounted from the host (``x-mounts`` maps
``~/.cache/huggingface``), so every later pod finds the weights already there.

Note this is *not* ``videoflow.utils.downloader.get_file``, which is the right
tool for a checkpoint at a URL. A Hugging Face repo is a set of files behind a
revision, and ``snapshot_download`` is the API that resolves it into the same
cache ``from_pretrained`` reads in ``open()`` — using anything else would warm a
cache the component never looks at.

`videoflow deploy video_captioning.py` runs this automatically inside the
solution image before compiling; it can also be run by hand:

    python prepare.py --config config.yaml [--force]

Idempotent: an already-cached model is left alone (``--force`` re-downloads it).
'''
from __future__ import annotations

import argparse

import cv2
from common import Config, load_config

#: Weight formats to skip when the repo also publishes ``.safetensors``. Many
#: repos ship every frame of weights twice; for a 7B model the duplicate is
#: ~16GB of download that ``from_pretrained`` will never open.
DUPLICATE_WEIGHT_PATTERNS = ['*.bin', '*.pth', '*.msgpack', '*.h5']

#: Rough per-caption cost used only for the runtime estimate printed below.
#: Deliberately coarse — the point is to warn that a 10,000-caption config is an
#: overnight job, not to predict a number.
SECONDS_PER_CAPTION_GUESS = 3.0


def probe_video(cfg: Config) -> dict:
    '''
    Fetches the input video (the bundled sample clip when ``input_video`` is
    empty), then reads frame count and frame rate off it and works out how many
    captions the configured sampling stride will produce.

    - Raises:
        - SystemExit: if the video cannot be opened — a bad path or an \
            unreadable file, which is worth catching here rather than in a pod.
    '''
    # fetch_input downloads the sample clip when input_video is empty (build_flow
    # only ever resolves the path, never downloads).
    path = cfg.fetch_input()
    capture = cv2.VideoCapture(path)
    if not capture.isOpened():
        raise SystemExit(f'cannot open input_video: {path}\n'
                         f'Fix the path in {cfg.path} (and make sure it is mounted into '
                         f'the pods — see x-mounts in config.template.yaml).')
    try:
        frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    finally:
        capture.release()

    # A container that reports no frame count is legal (some streams and some
    # VFR files); the run still works, only the estimate is unavailable.
    sampled = frames // cfg.every_n_frames if frames > 0 else 0
    if cfg.max_captions > 0:
        sampled = min(sampled, cfg.max_captions) if sampled else cfg.max_captions
    return {'path': path, 'frames': frames, 'fps': fps, 'captions': sampled}


def fetch_model(model_id: str, force: bool = False) -> str:
    '''
    Materializes the model in the Hugging Face cache and returns the snapshot
    directory.

    - Arguments:
        - model_id: Hugging Face repo id, as passed to the captioner.
        - force: re-download even when the model is already cached.
    '''
    # huggingface_hub ships with transformers, which is a solution dependency
    # rather than a videoflow one — imported here so `--help` and config
    # validation work in an environment that only has the core CLI.
    from huggingface_hub import list_repo_files, snapshot_download
    from huggingface_hub.errors import LocalEntryNotFoundError

    if not force:
        try:
            path = snapshot_download(model_id, local_files_only=True)
            print(f'==> model {model_id}: already cached at {path}; skipping (--force to redo)')
            return path
        except LocalEntryNotFoundError:
            pass

    print(f'==> model {model_id}: downloading (this is tens of GB the first time)', flush=True)
    ignore_patterns = None
    if any(name.endswith('.safetensors') for name in list_repo_files(model_id)):
        ignore_patterns = DUPLICATE_WEIGHT_PATTERNS
    path = snapshot_download(model_id, ignore_patterns=ignore_patterns, force_download=force)
    print(f'    cached at {path}')
    return path


def main():
    ap = argparse.ArgumentParser(
        description='Probe the input video and warm the VLM weight cache.')
    ap.add_argument('--config', default='config.yaml')
    ap.add_argument('--force', action='store_true', help='re-download even if cached')
    ap.add_argument('--skip-model', action='store_true',
                    help='probe the video only (useful when the cache is known warm)')
    args = ap.parse_args()
    cfg = load_config(args.config)

    info = probe_video(cfg)
    print(f'==> input video: {info["path"]}')
    print(f'    {info["frames"]} frames @ {info["fps"]:.2f} fps')
    if info['captions']:
        seconds = info['captions'] * SECONDS_PER_CAPTION_GUESS
        estimate = f'{seconds:.0f} s' if seconds < 90 else f'{seconds / 60.0:.0f} min'
        print(f'    every_n_frames={cfg.every_n_frames} -> ~{info["captions"]} captions '
              f'(order of {estimate} on one GPU replica)')
        if seconds > 3600:
            print('    NOTE: that is over an hour. Raise every_n_frames, cap the run with '
                  'max_captions, or add captioner.workers replicas before deploying.')
    else:
        print(f'    frame count unavailable; every_n_frames={cfg.every_n_frames} '
              f'will caption one frame in {cfg.every_n_frames}')

    if args.skip_model:
        print('==> model: skipped (--skip-model)')
    else:
        fetch_model(cfg.model_id, force=args.force)

    print('Prep complete.')


if __name__ == '__main__':
    main()
