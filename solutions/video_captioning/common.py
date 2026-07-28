'''
Shared config loading for the video-captioning solution.

The config is a small YAML file (see config.example.yaml). ``load_config``
returns a Config with every path resolved to an absolute location **relative to
the config file's directory**, not the process cwd — so the prep script, a local
run, and ``videoflow deploy`` (which compiles the graph from any cwd) all bake
the same paths into the node parameters.

Every knob is validated here rather than in ``build_flow``, and the failures are
``ConfigError``: ``videoflow deploy`` and ``run-local`` render a VideoflowError
as message + remedy and exit 2, so a typo reads as "here is the fix" instead of
a traceback. The GPU knobs get the same treatment even though ``ProcessorNode``
would eventually reject them, because the message an operator gets should name
the config key they typed, not the node argument it became.
'''
from __future__ import annotations

import os
from dataclasses import dataclass

import yaml
from videoflow.core.errors import ConfigError

DEVICES = ('cpu', 'gpu')
FLOW_TYPES = ('batch', 'realtime')
MISSING_POLICIES = ('wait', 'drop', 'error')


@dataclass
class Config:
    path: str
    work_dir: str
    input_video: str
    output_basename: str
    every_n_frames: int
    max_captions: int
    min_cue_seconds: float
    max_cue_seconds: float
    device: str                        # 'cpu' or 'gpu'
    flow_type: str                     # 'batch' or 'realtime'
    model_id: str
    prompt: str
    max_new_tokens: int
    workers: int                       # captioner replicas (nb_tasks)
    gpu_count: int                     # whole GPUs per captioner replica
    join_timeout_s: float | None
    join_missing: str
    join_max_pending: int

    def work_path(self, *parts: str) -> str:
        p = os.path.join(self.work_dir, *parts)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        return p

    def srt_path(self) -> str:
        '''The SubRip subtitle track — the solution's primary artifact.'''
        return self.work_path(f'{self.output_basename}.srt')

    def vtt_path(self) -> str:
        '''The WebVTT track: what a browser ``<track>`` element accepts.'''
        return self.work_path(f'{self.output_basename}.vtt')

    def jsonl_path(self) -> str:
        '''The live, append-as-you-go caption log.'''
        return self.work_path(f'{self.output_basename}.jsonl')

    def reader_nb_frames(self) -> int:
        '''
        The producer's ``nb_frames`` (frames *read*) that bounds the run at
        ``max_captions`` frames *published*. -1 (the default) reads the file to
        the end. Bounding the caption count is the useful knob — a VLM costs
        seconds per frame, so "how long will this take" is a function of
        captions, not of the source's length.
        '''
        if self.max_captions < 0:
            return -1
        return self.max_captions * self.every_n_frames


def _positive_int(raw: dict, key: str, default: int, cfg_path: str, prefix: str = '') -> int:
    '''
    Reads a count that must be >= 1. ``prefix`` is the dotted path of the
    enclosing block ('captioner.', 'join.') so the error names the key as it
    appears in the file rather than as it appears in this function.
    '''
    label = f'{prefix}{key}'
    value = int(raw.get(key, default))
    if value < 1:
        raise ConfigError(f'{label} is {value}.',
                        remedy = f'Set {label} to a positive integer in {cfg_path}.',
                        config = cfg_path, key = label, value = value)
    return value


def load_config(path: str) -> Config:
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    cfg_dir = os.path.dirname(os.path.abspath(path))
    cfg_path = os.path.abspath(path)

    input_video = raw.get('input_video')
    if not input_video:
        raise ConfigError('input_video is not set.',
                        remedy = f'Set input_video to the path of the video to caption '
                                f'in {cfg_path}.',
                        config = cfg_path)

    work_dir = os.path.abspath(os.path.join(cfg_dir, raw.get('work_dir', './out')))
    os.makedirs(work_dir, exist_ok=True)

    output_basename = str(raw.get('output_basename', 'captions'))
    if os.sep in output_basename or output_basename in ('', '.', '..'):
        raise ConfigError(f'output_basename is {output_basename!r}.',
                        remedy = f'Set output_basename in {cfg_path} to a bare filename stem '
                                f'(no directory, no extension) — the .srt/.vtt/.jsonl '
                                f'suffixes are added, and everything lands in work_dir.',
                        config = cfg_path, output_basename = output_basename)

    device = str(raw.get('device', 'gpu')).lower()
    if device not in DEVICES:
        raise ConfigError(f'device is {device!r}.',
                        remedy = f"Set device to 'cpu' or 'gpu' in {cfg_path}.",
                        config = cfg_path, device = device)

    flow_type = str(raw.get('flow_type', 'batch')).lower()
    if flow_type not in FLOW_TYPES:
        raise ConfigError(f'flow_type is {flow_type!r}.',
                        remedy = f"Set flow_type to 'batch' or 'realtime' in {cfg_path}.",
                        config = cfg_path, flow_type = flow_type)

    every_n_frames = _positive_int(raw, 'every_n_frames', 60, cfg_path)

    max_captions = int(raw.get('max_captions', -1))
    if max_captions == 0:
        raise ConfigError('max_captions is 0, so the flow would caption nothing.',
                        remedy = f'Set max_captions to a positive count, or to -1 to caption '
                                f'the whole video, in {cfg_path}.',
                        config = cfg_path, max_captions = max_captions)

    cues = raw.get('cues') or {}
    min_cue_seconds = float(cues.get('min_seconds', 1.0))
    max_cue_seconds = float(cues.get('max_seconds', 5.0))
    if min_cue_seconds <= 0 or max_cue_seconds <= 0:
        raise ConfigError(f'cues.min_seconds/cues.max_seconds are '
                        f'{min_cue_seconds}/{max_cue_seconds}.',
                        remedy = f'Set both to positive numbers of seconds in {cfg_path}.',
                        config = cfg_path)
    if min_cue_seconds > max_cue_seconds:
        raise ConfigError(f'cues.min_seconds ({min_cue_seconds}) is greater than '
                        f'cues.max_seconds ({max_cue_seconds}).',
                        remedy = f'Lower cues.min_seconds or raise cues.max_seconds in '
                                f'{cfg_path} — the floor cannot exceed the cap.',
                        config = cfg_path)

    captioner = raw.get('captioner') or {}
    model_id = str(captioner.get('model_id', 'Qwen/Qwen2.5-VL-7B-Instruct'))
    prompt = str(captioner.get('prompt', 'Describe this image in one sentence.'))
    max_new_tokens = _positive_int(captioner, 'max_new_tokens', 64, cfg_path, 'captioner.')
    workers = _positive_int(captioner, 'workers', 1, cfg_path, 'captioner.')
    gpu_count = _positive_int(captioner, 'gpu_count', 2, cfg_path, 'captioner.')

    # gpu_count is whole devices *per replica*, so a CPU run has nothing to hold
    # a grant. Caught here rather than in ProcessorNode so the message names the
    # config keys the operator set, and both readings of the mistake are offered.
    if device == 'cpu' and gpu_count > 1:
        raise ConfigError(f'captioner.gpu_count is {gpu_count} but device is cpu.',
                        remedy = f"Set device to 'gpu' in {cfg_path}, or set "
                                f'captioner.gpu_count to 1 — a CPU node cannot hold a '
                                f'GPU grant.',
                        config = cfg_path, device = device, gpu_count = gpu_count)

    join = raw.get('join') or {}
    join_missing = str(join.get('missing', 'wait')).lower()
    if join_missing not in MISSING_POLICIES:
        raise ConfigError(f'join.missing is {join_missing!r}.',
                        remedy = f'Set join.missing to one of {", ".join(MISSING_POLICIES)} '
                                f'in {cfg_path}.',
                        config = cfg_path, missing = join_missing)
    raw_timeout = join.get('timeout_s')
    join_timeout_s = float(raw_timeout) if raw_timeout is not None else None
    # The cue branch is instant and the captioner branch costs seconds per frame,
    # so essentially every sampled frame sits in the join as an incomplete group
    # at once. The policy default (256) would evict — i.e. silently lose captions
    # — past that, so the default here is well beyond any plausible caption count.
    join_max_pending = _positive_int(join, 'max_pending', 100000, cfg_path, 'join.')

    return Config(
        path=cfg_path,
        work_dir=work_dir,
        input_video=os.path.abspath(os.path.join(cfg_dir, input_video)),
        output_basename=output_basename,
        every_n_frames=every_n_frames,
        max_captions=max_captions,
        min_cue_seconds=min_cue_seconds,
        max_cue_seconds=max_cue_seconds,
        device=device,
        flow_type=flow_type,
        model_id=model_id,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        workers=workers,
        gpu_count=gpu_count,
        join_timeout_s=join_timeout_s,
        join_missing=join_missing,
        join_max_pending=join_max_pending,
    )
