'''
Shared config loading for the offside solution scripts.

The config is a small YAML file (see config.example.yaml). ``load_config`` returns
a Config with resolved paths and camera ordering; the prep scripts and the main
flow all read the same object.

``cameras`` may be left empty to use the bundled two-camera sample recording,
which ``prepare.py`` downloads into ``work_dir`` and references from there.
'''
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

import yaml
from videoflow.core.errors import ConfigError

BASE_URL_EXAMPLES = 'https://github.com/videoflow/videoflow-contrib/releases/download/example_videos/'

#: The bundled sample recording: two hardware-synchronized views of the same
#: penalty area, from opposite sidelines. Ordered — the first camera is the
#: audio-sync reference, and the order is the fusion input order.
SAMPLE_CAMERAS = {'cam0': 'offside_cam0.mp4', 'cam1': 'offside_cam1.mp4'}


@dataclass
class Config:
    path: str
    work_dir: str
    cameras: list                      # ordered camera names
    videos: dict                       # {cam: video_path}
    pitch_length: float
    pitch_width: float
    attack_direction: object           # 'auto' or {team: '+x'|'-x'}
    team_names: dict
    start_s: object
    end_s: object
    detector: dict = field(default_factory=dict)
    fusion: dict = field(default_factory=dict)
    debug_overlays: bool = False
    flow_type: str = 'batch'           # 'batch' (recorded clips) or 'realtime' (live)
    device: dict = field(default_factory=dict)  # per-stage 'cpu'/'gpu' (see device_for)
    uses_sample: bool = False          # True when `cameras` was empty and the sample is in use

    # Per-stage device defaults. Only the detector is genuinely GPU-bound: the
    # shipped tracker runs without ReID weights (its GPU pod would hold an idle
    # CUDA context), and top-down pose is usable on CPU. On Kubernetes every GPU
    # stage claims a whole exclusive device per camera, so these defaults cut a
    # 3-camera run from 9 GPU claims to 3 — see the videoflow GPU-sharing docs.
    DEVICE_DEFAULTS = {'detector': 'gpu', 'tracker': 'cpu', 'pose': 'cpu'}

    def device_for(self, stage: str) -> str:
        '''``videoflow.core.constants`` device for a pipeline stage ('detector',
        'tracker', 'pose'), from ``device.<stage>`` in the config.'''
        return self.device.get(stage, self.DEVICE_DEFAULTS[stage])

    @property
    def work(self) -> str:
        return self.work_dir

    def work_path(self, *parts) -> str:
        p = os.path.join(self.work_dir, *parts)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        return p

    def offsets_path(self) -> str:
        return os.path.join(self.work_dir, 'offsets.json')

    def calib_dir(self) -> str:
        return os.path.join(self.work_dir, 'calib')

    def teams_path(self) -> str:
        return os.path.join(self.work_dir, 'teams.json')

    def load_offsets(self) -> dict:
        with open(self.offsets_path()) as f:
            return json.load(f)

    def load_calibration(self) -> dict:
        calib = {}
        for cam in self.cameras:
            p = os.path.join(self.calib_dir(), f'{cam}.json')
            if os.path.exists(p):
                with open(p) as f:
                    calib[cam] = json.load(f)
        return calib

    def load_teams(self) -> dict:
        with open(self.teams_path()) as f:
            return json.load(f)

    def fetch_inputs(self) -> dict:
        '''
        Downloads the bundled sample recording into ``work_dir`` when the config
        left ``cameras`` empty; returns ``{cam: video_path}`` either way.

        Only ``prepare.py`` calls this. ``load_config`` deliberately resolves the
        paths without downloading, because compiling a graph must stay
        side-effect free — ``videoflow deploy --dry-run`` would otherwise pull
        20 MB per camera and write progress onto stdout.
        '''
        if not self.uses_sample:
            return self.videos
        from videoflow.utils.downloader import get_file
        for name in SAMPLE_CAMERAS.values():
            # cache_subdir='' puts it directly in work_dir, matching load_config().
            get_file(name, BASE_URL_EXAMPLES + name, cache_dir=self.work_dir, cache_subdir='')
        return self.videos


def load_config(path: str) -> Config:
    with open(path) as f:
        raw = yaml.safe_load(f)
    # Relative paths resolve against the config file's directory (not the process
    # cwd), so every consumer — prep scripts, local runs, `videoflow deploy`
    # compiling from any cwd — bakes the same absolute paths.
    cfg_dir = os.path.dirname(os.path.abspath(path))
    pitch = raw.get('pitch', {})
    trim = raw.get('trim', {})
    work_dir = os.path.abspath(os.path.join(cfg_dir, raw.get('work_dir', './out')))
    os.makedirs(work_dir, exist_ok=True)

    # An empty `cameras` means "use the bundled sample recording". It lives in
    # work_dir rather than a shared cache because these paths are baked into each
    # reader's parameters at compile time and must resolve to the same absolute
    # location inside the worker pods — work_dir is mounted at an identical path.
    cams_raw = raw.get('cameras') or {}
    uses_sample = not cams_raw
    if uses_sample:
        cameras = list(SAMPLE_CAMERAS)
        videos = {c: os.path.join(work_dir, SAMPLE_CAMERAS[c]) for c in cameras}
    else:
        cameras = list(cams_raw.keys())
        videos = {c: os.path.join(cfg_dir, cams_raw[c]['video']) for c in cameras}
    # ConfigError rather than ValueError: `videoflow deploy` and `run-local` render
    # a VideoflowError as message + remedy and exit 2, so a typo in the config
    # reads as "here is the fix" instead of a traceback with exit 1.
    cfg_path = os.path.abspath(path)
    flow_type = str(raw.get('flow_type', 'batch')).lower()
    if flow_type not in ('batch', 'realtime'):
        raise ConfigError(f'flow_type is {flow_type!r}.',
                        remedy = f"Set flow_type to 'batch' or 'realtime' in {cfg_path}.",
                        config = cfg_path, flow_type = flow_type)
    device = {k: str(v).lower() for k, v in (raw.get('device') or {}).items()}
    for stage, dev in device.items():
        if stage not in Config.DEVICE_DEFAULTS:
            raise ConfigError(
                f'device.{stage} is not a pipeline stage.',
                remedy = f'Use one of: {", ".join(sorted(Config.DEVICE_DEFAULTS))} '
                        f'in {cfg_path}.',
                config = cfg_path, stage = stage)
        if dev not in ('cpu', 'gpu'):
            raise ConfigError(f'device.{stage} is {dev!r}.',
                            remedy = f"Set device.{stage} to 'cpu' or 'gpu' in {cfg_path}.",
                            config = cfg_path, stage = stage, device = dev)
    return Config(
        path=cfg_path,
        work_dir=work_dir,
        cameras=cameras,
        videos={c: os.path.abspath(v) for c, v in videos.items()},
        pitch_length=float(pitch.get('length', 105.0)),
        pitch_width=float(pitch.get('width', 68.0)),
        attack_direction=raw.get('attack_direction', 'auto'),
        team_names=raw.get('team_names', {}),
        start_s=trim.get('start_s'),
        end_s=trim.get('end_s'),
        detector=raw.get('detector', {}),
        fusion=raw.get('fusion', {}),
        debug_overlays=bool(raw.get('debug_overlays', False)),
        flow_type=flow_type,
        device=device,
        uses_sample=uses_sample,
    )
