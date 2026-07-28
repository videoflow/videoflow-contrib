'''
Glue nodes for the video-captioning solution.

These live in their own importable module (not in the graph module) because
distributed workers reconstruct each node from its class path — recorded as
``video_captioning_nodes.<Class>`` — and importing the graph module in a worker
would re-run graph-level code. Every constructor argument is stored as
``self._<name>`` so the node round-trips through ``get_params()`` into the
worker unchanged.

The shape of the solution is set by one fact: a 7B VLM needs seconds per frame,
so the graph must not carry every frame. The sampling therefore happens in the
**producer** (``SampledVideoFileReader``), not downstream — a ``ProcessorNode``
cannot drop a message by returning ``None``, so a "filter" node would publish
just as many messages as it received and buy nothing.

Everything after the captioner is about turning an unordered stream of
``(frame index, caption)`` into a subtitle file, which is an ordered format:
``SubtitleWriter`` buffers and writes once from ``close()``, because the last
cue's end time is only knowable after the whole run.
'''
from __future__ import annotations

import json
import os
from typing import Any, Iterable, TextIO

import cv2
from videoflow.core.context import RuntimeContext
from videoflow.core.errors import ConfigError
from videoflow.core.node import ConsumerNode, ProcessorNode
from videoflow.producers.video import VideoFileReader

#: Subtitle formats the writer renders. SRT separates the milliseconds with a
#: comma, WebVTT with a dot and needs a magic first line — the only differences.
SRT_MS_SEPARATOR = ','
VTT_MS_SEPARATOR = '.'


def format_timestamp(seconds: float, ms_separator: str = SRT_MS_SEPARATOR) -> str:
    '''
    Renders a time offset as the ``HH:MM:SS,mmm`` stamp both subtitle formats
    use. Pure so it is unit-testable without a video.

    - Arguments:
        - seconds: offset from the start of the video; negatives clamp to zero.
        - ms_separator: ``','`` for SRT, ``'.'`` for WebVTT.
    '''
    total_ms = int(round(max(seconds, 0.0) * 1000))
    hours, total_ms = divmod(total_ms, 3_600_000)
    minutes, total_ms = divmod(total_ms, 60_000)
    secs, millis = divmod(total_ms, 1000)
    return f'{hours:02d}:{minutes:02d}:{secs:02d}{ms_separator}{millis:03d}'


def timed_cues(entries: Iterable[dict], min_cue_seconds: float,
               max_cue_seconds: float) -> list[dict]:
    '''
    Turns the caption entries collected during the run into ordered subtitle
    cues with start *and* end times. Pure so it is unit-testable without a
    video.

    Each cue runs until the next one starts, so the track has no gaps — capped
    at ``max_cue_seconds`` so the final cue (and any cue before a long sampling
    gap) doesn't hang on screen for the rest of the film, and floored at
    ``min_cue_seconds`` so a dense sampling stride still leaves each line long
    enough to read. The floor can push a cue past the next one's start; every
    player tolerates overlapping cues, whereas an unreadable 200ms flash is
    just noise.

    Captions that are blank after whitespace collapsing are dropped rather than
    emitted as empty cues, which some players render as a black bar.

    - Arguments:
        - entries: dicts with ``index`` (frame number), ``start`` (seconds) and \
            ``caption``, in any order.
        - min_cue_seconds: shortest a cue may be displayed.
        - max_cue_seconds: longest a cue may be displayed.
    - Returns: cues ordered by start time, each with ``index``, ``start``, \
        ``end`` and ``caption``.
    '''
    # Ordered by time, tie-broken by frame index: replicas of the captioner
    # compete for frames and finish out of order, so arrival order says nothing.
    ordered = sorted(entries, key=lambda e: (float(e['start']), int(e['index'])))
    cues: list[dict] = []
    for position, entry in enumerate(ordered):
        caption = ' '.join(str(entry['caption']).split())
        if not caption:
            continue
        start = float(entry['start'])
        if position + 1 < len(ordered):
            next_start = float(ordered[position + 1]['start'])
        else:
            next_start = start + max_cue_seconds
        end = max(min(next_start, start + max_cue_seconds), start + min_cue_seconds)
        cues.append({'index': int(entry['index']), 'start': start, 'end': end,
                     'caption': caption})
    return cues


def render_srt(cues: Iterable[dict]) -> str:
    '''Renders cues as a SubRip (``.srt``) document. Cues are renumbered from 1 —
    SRT counters must be sequential, and the frame index is not.'''
    blocks = [
        f"{number}\n"
        f"{format_timestamp(cue['start'])} --> {format_timestamp(cue['end'])}\n"
        f"{cue['caption']}\n"
        for number, cue in enumerate(cues, start=1)
    ]
    return '\n'.join(blocks)


def render_vtt(cues: Iterable[dict]) -> str:
    '''Renders cues as a WebVTT (``.vtt``) document — the format browsers accept
    in a ``<track>`` element, which ``.srt`` is not.'''
    blocks = ['WEBVTT\n']
    blocks.extend(
        f"{format_timestamp(cue['start'], VTT_MS_SEPARATOR)} --> "
        f"{format_timestamp(cue['end'], VTT_MS_SEPARATOR)}\n"
        f"{cue['caption']}\n"
        for cue in cues
    )
    return '\n'.join(blocks)


def _write_atomic(path: str, text: str) -> None:
    '''Writes via a temp file and ``os.replace`` so a reader never sees a
    half-written subtitle file (and a crashed run leaves the previous one).'''
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
        f.write(text)
    os.replace(tmp, path)


class SampledVideoFileReader(VideoFileReader):
    '''
    Reads a video file but publishes only every ``every_n_frames``-th frame,
    together with the timing information a subtitle cue needs.

    Sampling belongs here rather than in a downstream filter: a captioning VLM
    costs seconds per frame, and a ``ProcessorNode`` that returns ``None`` still
    publishes a message, so dropping frames anywhere but the producer would put
    the whole video on the wire to no purpose.

    - Emits: ``(frame_index, seconds, frame)`` — the 1-based frame number, its \
        offset on the video's own timeline, and the ``(h, w, 3)`` BGR frame.

    - Arguments:
        - video_file: path to the video. It must resolve inside the worker \
            container too, which is what the ``x-mounts`` same-path mount is for.
        - every_n_frames: publish one frame out of this many. Frames in between \
            are still decoded (sequential decoding is the only reliable way to \
            walk an arbitrary codec) but never leave the producer.
        - nb_frames: stop after this many frames have been *read* — so it bounds \
            the run at ``nb_frames / every_n_frames`` captions. -1 reads to the \
            end of the file.
    '''

    def __init__(self, video_file: str, every_n_frames: int = 60, nb_frames: int = -1,
                 **kwargs: Any) -> None:
        every_n_frames = int(every_n_frames)
        if every_n_frames < 1:
            raise ConfigError(f'every_n_frames is {every_n_frames}.',
                            remedy = 'Set every_n_frames to a positive integer — it is a '
                                    'stride, and 1 means caption every frame.',
                            node = kwargs.get('name'), every_n_frames = every_n_frames)
        self._every_n_frames = every_n_frames
        self._source_fps = 0.0        # read from the capture in open()
        # BGR is the frame convention VlmCaptioner assumes (it flips to RGB
        # itself), so the channel order is fixed here rather than left to the
        # caller. Popped first so reconstruction can't pass it twice.
        kwargs.pop('swap_channels', None)
        super().__init__(video_file, swap_channels = False, nb_frames = nb_frames, **kwargs)

    def open(self) -> None:
        super().open()
        # Only a fallback for cue timing; a container that reports no fps is
        # handled in _cue_seconds rather than here, because 0 is a legal answer.
        self._source_fps = (float(self._video.get(cv2.CAP_PROP_FPS) or 0.0)
                            if self._video is not None else 0.0)

    def next(self, ctx: RuntimeContext | None = None) -> tuple[int, float, Any]:  # type: ignore[override]
        '''
        - Returns: ``(frame_index, seconds, frame)`` for the next sampled frame.
        - Raises:
            - StopIteration: at end of file or once ``nb_frames`` have been read.
        '''
        while True:
            # super().next() raises StopIteration at end of stream, which ends
            # this loop, and RuntimeError if open() never ran.
            index, frame = super().next(ctx)
            if (index - 1) % self._every_n_frames:
                continue
            # Read *after* the decode: with the FFMPEG backend CAP_PROP_POS_MSEC
            # then holds the presentation time of the frame just returned (before
            # the read it still holds the previous frame's — one frame early).
            # This is also exactly what the base class stamps as the message's
            # event time under timestamp_source='position', so the cue in the
            # payload and the event time on the envelope agree.
            position_ms = self._video.get(cv2.CAP_PROP_POS_MSEC) if self._video is not None else 0.0
            return (index, self._cue_seconds(index, position_ms), frame)

    def _cue_seconds(self, index: int, position_ms: float) -> float:
        '''
        The frame's offset on the video's timeline, preferring the container's
        own answer because it stays correct on variable-frame-rate footage.
        Some codecs and backends report 0 for every frame; the frame-rate
        fallback covers those. Only both being unavailable collapses the
        timeline to zero — which still writes a valid subtitle file, just one
        where every cue sits at 00:00.
        '''
        if position_ms > 0:
            return position_ms / 1000.0
        if self._source_fps > 0:
            return (index - 1) / self._source_fps
        return 0.0

    def get_params(self) -> dict:
        # VideoFileReader hand-writes get_params() (its own __init__ renames the
        # base's url_or_deviceid), so the MRO walk that would have picked up
        # every_n_frames never runs — extend the hand-written dict instead of
        # inheriting it. swap_channels is fixed in __init__ and deliberately absent.
        return {
            'video_file': self._url_or_deviceid,
            'every_n_frames': self._every_n_frames,
            'nb_frames': self._nb_frames,
            'timestamp_source': self._timestamp_source,
            'name': self._name,
        }


class FrameSelector(ProcessorNode):
    '''
    Keeps only the frame from the reader's ``(index, seconds, frame)`` tuple.
    The captioner's contract is a bare ``(h, w, 3)`` BGR array — handed the
    tuple it would raise ``SchemaError`` and dead-letter the frame.
    '''

    def process(self, sample: tuple) -> Any:
        index, seconds, frame = sample
        return frame


class CueSelector(ProcessorNode):
    '''
    Keeps only the timing half of the reader's tuple, so the frame never travels
    down the branch that bypasses the captioner. On a 1080p stream that is ~6MB
    of payload per sample which would otherwise go through the broker (or the
    blob store) for the sake of two numbers.
    '''

    def process(self, sample: tuple) -> dict:
        index, seconds, frame = sample
        return {'index': int(index), 'start': float(seconds)}


class CaptionCueAssembler(ProcessorNode):
    '''
    Two-parent join: pairs each frame's timing with the caption generated from
    that same frame. The branches are joined by lineage (trace id), not by
    arrival order, which is what makes it safe for the captioner to run several
    competing replicas that finish out of order — and for the cue branch to run
    thousands of frames ahead of the model.

    Emits one flat record per captioned frame, which all three sinks share.
    '''

    # One positional argument per parent, so the signature can't match the
    # single-input base method.
    def process(self, cue: dict, caption: Any) -> dict:  # type: ignore[override]
        return {'index': int(cue['index']), 'start': float(cue['start']),
                'caption': str(caption)}


class JsonLinesConsumer(ConsumerNode):
    '''
    Appends each caption record to a file as one JSON object per line, flushing
    every line so a ``tail -f`` (or ``kubectl logs``' file-less equivalent, a
    mounted work dir) shows a long captioning run progressing.

    This is the live view; ``SubtitleWriter`` is the finished artifact and can
    only be written at the end. The file handle follows the node lifecycle:
    opened in ``open()``, released in ``close()``.

    - Arguments:
        - out_path: file to append to. It is appended, not truncated, so \
            re-running against the same ``work_dir`` accumulates runs — the \
            subtitle files are rewritten in place, this one is a log.
    '''

    def __init__(self, out_path: str, **kwargs: Any) -> None:
        self._out_path = out_path
        self._fh: TextIO | None = None
        super().__init__(**kwargs)

    def open(self) -> None:
        self._fh = open(self._out_path, 'a')

    def consume(self, entry: dict) -> None:
        assert self._fh is not None, 'consume() called before open()'
        self._fh.write(json.dumps(entry) + '\n')
        self._fh.flush()

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None


class SubtitleWriter(ConsumerNode):
    '''
    Collects every caption record and writes the subtitle files from
    ``close()`` — the lifecycle hook that runs in the worker after
    end-of-stream.

    It has to buffer: a cue's end time is the next cue's start time, so no cue
    is final until the one after it has arrived, and captions arrive out of
    order whenever the captioner is replicated. Records are kept in a dict keyed
    by frame index, so an at-least-once redelivery overwrites its own cue
    instead of duplicating a line into the file.

    - Arguments:
        - srt_path: where to write the SubRip file (inside ``work_dir``, which \
            is mounted at the same absolute path in the worker pod).
        - vtt_path: where to write the WebVTT file, or None to skip it.
        - min_cue_seconds: shortest a cue may be displayed.
        - max_cue_seconds: longest a cue may be displayed.
    '''

    def __init__(self, srt_path: str, vtt_path: str | None = None,
                 min_cue_seconds: float = 1.0, max_cue_seconds: float = 5.0,
                 **kwargs: Any) -> None:
        self._srt_path = srt_path
        self._vtt_path = vtt_path
        self._min_cue_seconds = float(min_cue_seconds)
        self._max_cue_seconds = float(max_cue_seconds)
        self._entries: dict[int, dict] = {}
        super().__init__(**kwargs)

    def consume(self, entry: dict) -> None:
        self._entries[int(entry['index'])] = entry

    def close(self) -> None:
        cues = timed_cues(self._entries.values(), self._min_cue_seconds,
                          self._max_cue_seconds)
        _write_atomic(self._srt_path, render_srt(cues))
        if self._vtt_path:
            _write_atomic(self._vtt_path, render_vtt(cues))
        # Printed rather than logged: it is the one line that tells an operator
        # watching `kubectl logs` that the run produced something.
        print(f'{self.name}: wrote {len(cues)} cues to {self._srt_path}', flush=True)
