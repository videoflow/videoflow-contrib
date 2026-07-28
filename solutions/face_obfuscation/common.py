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
from videoflow.core.errors import ConfigError


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

    # ConfigError rather than ValueError: `videoflow deploy` and `run-local` render
    # a VideoflowError as message + remedy and exit 2, so a typo in the config
    # reads as "here is the fix" instead of a traceback with exit 1.
    cfg_path = os.path.abspath(path)
    input_video = raw.get('input_video')
    if not input_video:
        raise ConfigError('input_video is not set.',
                        remedy = f'Set input_video to the path of the video to '
                                f'obfuscate in {cfg_path}.',
                        config = cfg_path)

    work_dir = os.path.abspath(os.path.join(cfg_dir, raw.get('work_dir', './out')))
    os.makedirs(work_dir, exist_ok=True)

    device = str(raw.get('device', 'cpu')).lower()
    if device not in ('cpu', 'gpu'):
        raise ConfigError(f'device is {device!r}.',
                        remedy = f"Set device to 'cpu' or 'gpu' in {cfg_path}.",
                        config = cfg_path, device = device)

    flow_type = str(raw.get('flow_type', 'batch')).lower()
    if flow_type not in ('batch', 'realtime'):
        raise ConfigError(f'flow_type is {flow_type!r}.',
                        remedy = f"Set flow_type to 'batch' or 'realtime' in {cfg_path}.",
                        config = cfg_path, flow_type = flow_type)

    # VideofileWriter only supports .avi; catch it here with a clear message
    # rather than deep inside graph construction.
    output_video = raw.get('output_video', 'blurred_video.avi')
    if not output_video.endswith('.avi'):
        raise ConfigError(
            f'output_video is {output_video!r}.',
            remedy = f"Give output_video an .avi extension in {cfg_path} — "
                    f"videoflow's VideofileWriter supports only that container.",
            config = cfg_path, output_video = output_video)

    return Config(
        path=cfg_path,
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
