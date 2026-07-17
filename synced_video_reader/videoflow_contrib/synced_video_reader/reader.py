'''
Videofile reader that shifts each frame's event timestamp by a per-camera offset
(and optional clock drift), so several independently-recorded cameras align on a
common event-time axis for the downstream time-synchronized join.

The offset comes from the audio-cross-correlation prep step (``sync_offsets.py``):
``t_common = t_position · (1 + drift_ppm·1e-6) + offset_s``. Reference camera has
``offset_s = 0``.
'''
from __future__ import annotations

import cv2
from videoflow.producers.video import VideoFileReader


class _OffsetCtx:
    '''Wraps the runtime ctx so the producer's ``set_event_timestamp`` is offset/scaled.'''
    def __init__(self, ctx, offset_s: float, drift_ppm: float):
        self._ctx = ctx
        self._b = offset_s
        self._k = 1.0 + drift_ppm * 1e-6

    def set_event_timestamp(self, ts: float) -> None:
        self._ctx.set_event_timestamp(ts * self._k + self._b)

    def __getattr__(self, name):
        return getattr(self._ctx, name)


class SyncedVideoReader(VideoFileReader):
    '''
    - Arguments:
        - video_file: path to the camera's recording.
        - offset_s: seconds to add to this camera's timeline to reach the common axis.
        - drift_ppm: clock-drift correction in parts-per-million (optional).
        - start_s / end_s: trim the clip to this event-time window (optional).
        - swap_channels: BGR→RGB (default False; the offside pipeline works in BGR).
        - nb_frames: max frames to read (-1 = all).
    '''
    def __init__(self, video_file: str, offset_s: float = 0.0, drift_ppm: float = 0.0,
                 start_s=None, end_s=None, swap_channels: bool = False, nb_frames: int = -1,
                 **kwargs) -> None:
        self._offset_s = float(offset_s)
        self._drift_ppm = float(drift_ppm)
        self._start_s = start_s
        self._end_s = end_s
        # timestamp_source is forced to 'position' — synchronized recordings align on
        # the file timeline, then offset_s shifts to the shared axis.
        super().__init__(video_file, swap_channels=swap_channels, nb_frames=nb_frames,
                         timestamp_source='position', **kwargs)

    def open(self) -> None:
        super().open()
        if self._start_s is not None and self._video is not None:
            self._video.set(cv2.CAP_PROP_POS_MSEC, float(self._start_s) * 1000.0)

    def next(self, ctx=None) -> tuple:
        if self._end_s is not None and self._video is not None and self._video.isOpened():
            pos_s = self._video.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            if pos_s > self._end_s:
                raise StopIteration()
        shim = _OffsetCtx(ctx, self._offset_s, self._drift_ppm) if ctx is not None else None
        return super().next(ctx=shim)

    def get_params(self) -> dict:
        # Own __init__ renames the base's first arg (stored as _url_or_deviceid) and
        # adds sync params, so the default MRO get_params() can't round-trip it.
        return {
            'video_file': self._url_or_deviceid,
            'offset_s': self._offset_s,
            'drift_ppm': self._drift_ppm,
            'start_s': self._start_s,
            'end_s': self._end_s,
            'swap_channels': self._swap_channels,
            'nb_frames': self._nb_frames,
            'name': self._name,
        }
