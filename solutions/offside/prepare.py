'''
One-shot, idempotent prep for the offside solution — fetches the input videos and
runs the four prep steps in order, skipping any whose outputs already exist:

    videos -> weights -> sync_offsets -> calibrate -> fit_teams

`videoflow deploy offside.py` runs this automatically (inside the solution
image) before compiling; it can also be run by hand:

    python prepare.py --config config.yaml [--force]

If automatic calibration fails for a camera, run the manual click UI once on a
machine with a display, then re-run (the finished steps are skipped):

    python calibrate.py --config config.yaml --cam <cam> --manual
'''
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

from common import Config, load_config

HERE = os.path.dirname(os.path.abspath(__file__))


def _run(script, *extra):
    print(f'==> {script} {" ".join(extra)}', flush=True)
    subprocess.run([sys.executable, os.path.join(HERE, script), *extra], check=True, cwd=HERE)


def _skip(script, output):
    print(f'==> {script}: {output} exists, skipping (--force to redo)', flush=True)


def _write_sample_offsets(cfg: Config) -> None:
    '''
    Writes a zero-offset ``offsets.json`` for the bundled sample recording, in
    place of running ``sync_offsets.py`` on it.

    The sample's cameras were genlocked in hardware and the published clips carry
    no audio track, so audio cross-correlation has nothing to correlate — it is
    not that sync is being skipped for speed, it is that the answer is known
    exactly and the measurement is impossible. Your own recordings go through
    ``sync_offsets.py`` as normal.
    '''
    offsets = {cam: {'offset_s': 0.0, 'drift_ppm': 0.0, 'confidence': 9.0}
               for cam in cfg.cameras}
    with open(cfg.offsets_path(), 'w') as f:
        json.dump(offsets, f, indent=2)
    print(f'==> sync_offsets.py: sample recording is hardware-synchronized and has no '
          f'audio; wrote zero offsets to {cfg.offsets_path()}', flush=True)


def main():
    ap = argparse.ArgumentParser(description='One-shot prep: weights -> offsets -> calibration -> team fit.')
    ap.add_argument('--config', default='config.yaml')
    ap.add_argument('--force', action='store_true', help='re-run steps whose outputs already exist')
    ap.add_argument('--skip-rfdetr', action='store_true', help='forwarded to download_weights.py')
    ap.add_argument('--skip-pose', action='store_true', help='forwarded to download_weights.py')
    args = ap.parse_args()
    cfg = load_config(args.config)

    # Downloads the bundled sample recording when `cameras` is empty; a no-op for
    # your own footage. get_file skips anything already in work_dir.
    print('==> input videos', flush=True)
    for cam, video in cfg.fetch_inputs().items():
        if not os.path.exists(video):
            raise SystemExit(f'{cam}: video does not exist: {video}\n'
                             f'Fix cameras.{cam}.video in {cfg.path} (and make sure it is '
                             f'mounted into the pods).')
        print(f'    {cam}: {video}')

    # get_file skips already-cached weights, so this is cheap when warm.
    weight_args = [flag for flag, on in [('--skip-rfdetr', args.skip_rfdetr),
                                         ('--skip-pose', args.skip_pose)] if on]
    _run('download_weights.py', *weight_args)

    if args.force or not os.path.exists(cfg.offsets_path()):
        if cfg.uses_sample:
            _write_sample_offsets(cfg)
        else:
            _run('sync_offsets.py', '--config', args.config)
    else:
        _skip('sync_offsets.py', cfg.offsets_path())

    calib_done = all(os.path.exists(os.path.join(cfg.calib_dir(), f'{cam}.json'))
                     for cam in cfg.cameras)
    if args.force or not calib_done:
        try:
            _run('calibrate.py', '--config', args.config)
        except subprocess.CalledProcessError:
            missing = [cam for cam in cfg.cameras
                       if not os.path.exists(os.path.join(cfg.calib_dir(), f'{cam}.json'))]
            # from None: calibrate.py's own output already went to the terminal, so
            # the traceback would only bury the instructions below.
            raise SystemExit(
                f'automatic calibration failed for: {", ".join(missing) or "?"}. Run the '
                f'manual click UI once on a machine with a display, then re-run:\n'
                f'  python calibrate.py --config {args.config} --cam <cam> --manual') from None
    else:
        _skip('calibrate.py', os.path.join(cfg.calib_dir(), '<cam>.json'))

    if args.force or not os.path.exists(cfg.teams_path()):
        _run('fit_teams.py', '--config', args.config)
    else:
        _skip('fit_teams.py', cfg.teams_path())

    print('Prep complete. Inspect out/teams_montage.png and the calib overlays before deploying.')


if __name__ == '__main__':
    main()
