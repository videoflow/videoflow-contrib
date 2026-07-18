'''
Shared config loading for the face-obfuscation solution.

The config is a small YAML file (see config.example.yaml). ``load_config``
returns a Config with every path resolved to an absolute location **relative to
the config file's directory**, not the process cwd — so the prep script, a local
run, and ``videoflow deploy`` (which compiles the graph from any cwd) all bake
the same paths into the node parameters.
'''
from __future__ import annotations

import os
from dataclasses import dataclass, field

import yaml


@dataclass
class Config:
    path: str
    work_dir: str
    input_video: str
    output_video: str
    fps: int
    device: str                        # 'cpu' or 'gpu'
    flow_type: str                     # 'batch' or 'realtime'
    detector: dict = field(default_factory=dict)
    tracker: dict = field(default_factory=dict)
    blur: dict = field(default_factory=dict)

    def work_path(self, *parts) -> str:
        p = os.path.join(self.work_dir, *parts)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        return p

    def output_path(self) -> str:
        return self.work_path(self.output_video)


def load_config(path: str) -> Config:
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    cfg_dir = os.path.dirname(os.path.abspath(path))

    input_video = raw.get('input_video')
    if not input_video:
        raise ValueError("config must set 'input_video' (path to the video to obfuscate)")

    work_dir = os.path.abspath(os.path.join(cfg_dir, raw.get('work_dir', './out')))
    os.makedirs(work_dir, exist_ok=True)

    device = str(raw.get('device', 'cpu')).lower()
    if device not in ('cpu', 'gpu'):
        raise ValueError(f"device must be 'cpu' or 'gpu', got {device!r}")

    flow_type = str(raw.get('flow_type', 'batch')).lower()
    if flow_type not in ('batch', 'realtime'):
        raise ValueError(f"flow_type must be 'batch' or 'realtime', got {flow_type!r}")

    # VideofileWriter only supports .avi; catch it here with a clear message
    # rather than deep inside graph construction.
    output_video = raw.get('output_video', 'blurred_video.avi')
    if not output_video.endswith('.avi'):
        raise ValueError(f"output_video must end in .avi (videoflow's VideofileWriter "
                         f'only supports that container), got {output_video!r}')

    return Config(
        path=os.path.abspath(path),
        work_dir=work_dir,
        input_video=os.path.abspath(os.path.join(cfg_dir, input_video)),
        # Relative to work_dir (which is already absolute), so results land next
        # to the other artifacts and the whole directory is one mount.
        output_video=output_video,
        fps=int(raw.get('fps', 30)),
        device=device,
        flow_type=flow_type,
        detector=raw.get('detector', {}),
        tracker=raw.get('tracker', {}),
        blur=raw.get('blur', {}),
    )
