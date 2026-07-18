'''
Shared config loading for the offside solution scripts.

The config is a small YAML file (see config.example.yaml). ``load_config`` returns
a Config with resolved paths and camera ordering; the prep scripts and the main
flow all read the same object.
'''
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

import yaml


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


def load_config(path: str) -> Config:
    with open(path) as f:
        raw = yaml.safe_load(f)
    # Relative paths resolve against the config file's directory (not the process
    # cwd), so every consumer — prep scripts, local runs, `videoflow deploy`
    # compiling from any cwd — bakes the same absolute paths.
    cfg_dir = os.path.dirname(os.path.abspath(path))
    cams_raw = raw['cameras']
    cameras = list(cams_raw.keys())
    videos = {c: os.path.join(cfg_dir, cams_raw[c]['video']) for c in cameras}
    pitch = raw.get('pitch', {})
    trim = raw.get('trim', {})
    work_dir = os.path.abspath(os.path.join(cfg_dir, raw.get('work_dir', './out')))
    os.makedirs(work_dir, exist_ok=True)
    flow_type = str(raw.get('flow_type', 'batch')).lower()
    if flow_type not in ('batch', 'realtime'):
        raise ValueError(f"flow_type must be 'batch' or 'realtime', got {flow_type!r}")
    return Config(
        path=os.path.abspath(path),
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
    )
