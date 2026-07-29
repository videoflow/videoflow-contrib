'''
Shared config loading for the human-tracking solution.

The config is a small YAML file (see config.example.yaml). ``load_config``
returns a Config with every path resolved to an absolute location **relative to
the config file's directory**, not the process cwd — so the prep script, a local
run, and ``videoflow deploy`` (which compiles the graph from any cwd) all bake
the same paths into the node parameters.

``input_video`` may be left empty to use the bundled sample clip, which is
downloaded into the videoflow cache by ``prepare.py`` and referenced from there.
'''
from __future__ import annotations

import os
from dataclasses import dataclass, field

import yaml
from videoflow.core.errors import ConfigError

BASE_URL_EXAMPLES = 'https://github.com/videoflow/videoflow-contrib/releases/download/example_videos/'
SAMPLE_VIDEO_NAME = 'people_walking.mp4'
SAMPLE_VIDEO_URL = BASE_URL_EXAMPLES + SAMPLE_VIDEO_NAME


@dataclass
class Config:
    path: str
    work_dir: str
    input_video: str                   # '' means "use the downloaded sample"
    output_video: str
    fps: int                           # frame rate of the written video
    device: str                        # 'cpu' or 'gpu'
    flow_type: str                     # 'batch' or 'realtime'
    pose: dict = field(default_factory=dict)
    encoder: dict = field(default_factory=dict)
    tracker: dict = field(default_factory=dict)

    def work_path(self, *parts) -> str:
        p = os.path.join(self.work_dir, *parts)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        return p

    def output_path(self) -> str:
        return self.work_path(self.output_video)

    def resolve_input(self) -> str:
        '''
        The path the reader opens: the configured video, or the sample clip inside
        ``work_dir``.

        The sample deliberately lives in work_dir rather than the shared videoflow
        model cache, because this path is baked into the reader's parameters at
        compile time and must resolve to the *same* absolute location inside the
        worker pods — work_dir is mounted at an identical path, whereas the model
        cache is remapped onto the container's ``/root``.

        Deliberately does NOT download either: compiling a graph must stay
        side-effect free (``--dry-run`` would otherwise pull ~29MB and write
        progress onto stdout). ``prepare.py`` calls ``fetch_input`` to populate it
        before the workers run.
        '''
        if self.input_video:
            return self.input_video
        return os.path.join(self.work_dir, SAMPLE_VIDEO_NAME)

    def fetch_input(self) -> str:
        '''Downloads the sample clip into work_dir if needed; returns the path (used by prepare.py).'''
        if self.input_video:
            return self.input_video
        from videoflow.utils.downloader import get_file
        # cache_subdir='' puts it directly in work_dir, matching resolve_input().
        return get_file(SAMPLE_VIDEO_NAME, SAMPLE_VIDEO_URL,
                        cache_dir=self.work_dir, cache_subdir='')


def load_config(path: str) -> Config:
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    cfg_dir = os.path.dirname(os.path.abspath(path))

    work_dir = os.path.abspath(os.path.join(cfg_dir, raw.get('work_dir', './out')))
    os.makedirs(work_dir, exist_ok=True)

    input_video = raw.get('input_video') or ''
    if input_video:
        input_video = os.path.abspath(os.path.join(cfg_dir, input_video))

    # ConfigError rather than ValueError: `videoflow deploy` and `run-local` render
    # a VideoflowError as message + remedy and exit 2, so a typo in the config
    # reads as "here is the fix" instead of a traceback with exit 1.
    cfg_path = os.path.abspath(path)
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

    return Config(
        path=cfg_path,
        work_dir=work_dir,
        input_video=input_video,
        # Relative to work_dir (already absolute), so results land next to the
        # other artifacts and the whole directory is one mount.
        output_video=raw.get('output_video', 'annotated_video.avi'),
        fps=int(raw.get('fps', 25)),
        device=device,
        flow_type=flow_type,
        pose=raw.get('pose', {}),
        encoder=raw.get('encoder', {}),
        tracker=raw.get('tracker', {}),
    )
