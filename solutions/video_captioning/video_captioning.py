'''
Video captioning — describe a video with a vision-language model and write the
descriptions out as a subtitle track.

Per sampled frame: read → caption with ``videoflow_contrib.vlm_caption``
(Qwen2.5-VL by default) → pair the caption back with the frame's position on the
video timeline → write ``captions.srt`` / ``captions.vtt``, plus a live
``captions.jsonl`` and one line per caption on stdout.

    reader ─┬─> frames ──> captioner (VLM, N replicas x M GPUs) ─┐
            │                                                    │
            └─> cues ────────────────────────────────────────────┴─> caption ─┬─> subtitles
                                                                              ├─> stream
                                                                              └─> live

Three things about that shape are load-bearing:

- **The reader samples.** A 7B VLM costs seconds per frame, so the producer
  publishes one frame in ``every_n_frames``. A downstream filter could not do
  this: ``process()`` returning ``None`` still publishes a message.
- **The frame goes down one branch only.** ``cues`` strips the frame and keeps
  the two numbers a subtitle needs, so the branch that bypasses the model does
  not carry megabytes per sample through the broker.
- **``caption`` is a trace join with a large pending budget.** The cue branch
  runs the whole video ahead of the model, so nearly every sampled frame is an
  open join group at once; the default 256-group cap would silently evict
  captions. ``join.max_pending`` in the config sets it.

This is also the reference deployment of RFC 0003 multi-GPU: the captioner
declares ``gpu_count`` whole devices per replica and shards the model across
them with ``device_map='auto'``, with no device arithmetic anywhere in the graph.

Deploy to Kubernetes (config Q&A, image build, broker, run and teardown in one
command — see README.md):

    videoflow deploy video_captioning.py

Local run, all workers as subprocesses on this machine:

    python video_captioning.py --config config.yaml

The glue nodes live in ``video_captioning_nodes.py`` (a real importable module)
so distributed workers can reconstruct them by class path (the local engine puts
this directory on each worker's PYTHONPATH automatically).
'''
from __future__ import annotations

import argparse
import os

from common import load_config
from video_captioning_nodes import (
    CaptionCueAssembler,
    CueSelector,
    FrameSelector,
    JsonLinesConsumer,
    SampledVideoFileReader,
    SubtitleWriter,
)
from videoflow.consumers import CommandlineConsumer
from videoflow.core import Flow
from videoflow.core.policies import JoinPolicy
from videoflow.engines.local import LocalProcessEngine


def build_flow(cfg=None):
    if cfg is None:
        # Module-dir-relative so `videoflow deploy` works from any cwd.
        cfg = load_config(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.yaml'))
    from videoflow_contrib.vlm_caption import VlmCaptioner

    reader = SampledVideoFileReader(cfg.input_video,
                                    every_n_frames=cfg.every_n_frames,
                                    nb_frames=cfg.reader_nb_frames(),
                                    name='reader')
    frames = FrameSelector(name='frames')(reader)
    cues = CueSelector(name='cues')(reader)
    # gpu_count is whole devices per replica: the model is sharded across them by
    # Hugging Face, so `workers x gpu_count` GPUs are claimed in total.
    captions = VlmCaptioner(model_id=cfg.model_id,
                            prompt=cfg.prompt,
                            max_new_tokens=cfg.max_new_tokens,
                            nb_tasks=cfg.workers,
                            device_type=cfg.device,
                            gpu_count=cfg.gpu_count,
                            name='captioner')(frames)
    caption = CaptionCueAssembler(
        join_policy=JoinPolicy(timeout_seconds=cfg.join_timeout_s, missing=cfg.join_missing,
                               max_pending=cfg.join_max_pending),
        name='caption')(cues, captions)
    # The subtitle files can only be written once every cue is in (a cue ends
    # where the next one starts), so the other two sinks exist to make a long run
    # observable while it happens.
    subtitles = SubtitleWriter(cfg.srt_path(), cfg.vtt_path(),
                               min_cue_seconds=cfg.min_cue_seconds,
                               max_cue_seconds=cfg.max_cue_seconds,
                               name='subtitles')(caption)
    stream = JsonLinesConsumer(cfg.jsonl_path(), name='stream')(caption)
    live = CommandlineConsumer(name='live')(caption)
    return Flow([subtitles, stream, live], flow_type=cfg.flow_type)


def main():
    ap = argparse.ArgumentParser(description='Caption a video with a VLM and write a subtitle track.')
    ap.add_argument('--config', default='config.yaml')
    ap.add_argument('--flow-type', choices=('batch', 'realtime'), default=None,
                    help='override the config flow_type (default: from config, else batch)')
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.flow_type is not None:
        cfg.flow_type = args.flow_type

    flow = build_flow(cfg)
    engine = LocalProcessEngine(blob_redis_url=os.environ.get('VIDEOFLOW_BLOB_REDIS_URL'))
    flow.run(engine)
    flow.join()
    if engine.failures():
        engine.report_failures()
        raise SystemExit(1)
    print(f'Wrote {cfg.srt_path()}')


if __name__ == '__main__':
    main()
