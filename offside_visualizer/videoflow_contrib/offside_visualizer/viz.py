'''
Offside visualizer (consumer).

Consumes the engine's verdict/touch-event stream. For each verdict it writes a
JSON record, re-opens each camera's video from disk (so full frames never cross
the broker), seeks to the kick instant, and draws the projected offside line +
player/ball markers + a verdict banner, plus a top-down pitch diagram and an
optional annotated clip. ``None`` and touch events are logged, not drawn.
'''
from __future__ import annotations

import json
import os

import cv2
from videoflow.core.node import ConsumerNode

from . import drawing


class OffsideVisualizer(ConsumerNode):
    '''
    - Arguments:
        - output_dir: where results are written.
        - videos: ``{cam: video_path}``.
        - offsets: ``{cam: offset_s}`` (event-time → each camera's own timeline).
        - calibration / calibration_path: per-camera calib.
        - pitch_length / pitch_width: for the projected line + top-down diagram.
        - clip_padding_s / write_clips: annotated-clip options.
    '''
    def __init__(self, output_dir: str, videos: dict, offsets: dict, calibration=None,
                 calibration_path=None, pitch_length: float = 105.0, pitch_width: float = 68.0,
                 clip_padding_s: float = 3.0, write_clips: bool = True, **kwargs):
        self._output_dir = output_dir
        self._videos = dict(videos)
        self._offsets = dict(offsets)
        self._calibration = calibration
        self._calibration_path = calibration_path
        self._pitch_length = float(pitch_length)
        self._pitch_width = float(pitch_width)
        self._clip_padding_s = float(clip_padding_s)
        self._write_clips = bool(write_clips)
        self._calib: dict = {}
        self._verdicts: list = []
        self._touch_log: list = []
        super().__init__(**kwargs)

    def open(self):
        os.makedirs(self._output_dir, exist_ok=True)
        self._calib = self._load_calib()
        self._verdicts, self._touch_log = [], []

    def _load_calib(self) -> dict:
        if self._calibration:
            return dict(self._calibration)
        if self._calibration_path and os.path.isdir(self._calibration_path):
            out = {}
            for cam in self._videos:
                p = os.path.join(self._calibration_path, f'{cam}.json')
                if os.path.exists(p):
                    with open(p) as f:
                        out[cam] = json.load(f)
            return out
        return {}

    def consume(self, item):
        if item is None or not isinstance(item, dict):
            return
        kind = item.get('type')
        if kind == 'touch_event':
            self._touch_log.append(item)
        elif kind == 'offside_verdict':
            self._verdicts.append(item)
            self._render_verdict(item)

    def _render_verdict(self, v: dict):
        n = v.get('index', len(self._verdicts))
        with open(os.path.join(self._output_dir, f'verdict_{n}.json'), 'w') as f:
            json.dump(v, f, indent=2)
        # per-camera stills
        for cam, path in self._videos.items():
            if cam not in self._calib:
                continue
            frame = self._grab(path, v['t_kick'] - self._offsets.get(cam, 0.0))
            if frame is None:
                continue
            self._annotate(frame, v, cam)
            cv2.imwrite(os.path.join(self._output_dir, f'still_{n}_{cam}.png'), frame)
            if self._write_clips:
                self._write_clip(cam, path, v, n)
        # top-down diagram
        top = drawing.draw_pitch_topdown(self._pitch_length, self._pitch_width,
                                         v.get('players_t0', []), v.get('offside_line_x'),
                                         v.get('attack_sign', 1))
        cv2.imwrite(os.path.join(self._output_dir, f'topdown_{n}.png'), top)

    def _annotate(self, frame, v, cam):
        calib = self._calib[cam]
        if v.get('offside_line_x') is not None:
            drawing.draw_offside_line(frame, v['offside_line_x'], self._pitch_width, calib)
        for p in v.get('players_t0', []):
            color = drawing.TEAM_COLORS.get(p.get('team', -1), (150, 150, 150))
            lbl = None
            if p['gid'] == v.get('receiver_gid'):
                lbl, color = 'R', (0, 0, 255)
            elif p['gid'] == v.get('second_defender_gid'):
                lbl = 'D2'
            drawing.draw_player_marker(frame, p['pos'], calib, color, lbl)
        drawing.draw_banner(frame, v['verdict'], v.get('margin_m'), v.get('uncertainty_m', 0.0))
        return frame

    def _grab(self, video_path, t_local):
        if not os.path.exists(video_path):
            return None
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, t_local) * 1000.0)
        ok, frame = cap.read()
        cap.release()
        return frame if ok else None

    def _write_clip(self, cam, path, v, n):
        t0 = max(0.0, v['t_kick'] - self._offsets.get(cam, 0.0) - self._clip_padding_s)
        t1 = v['t_touch'] - self._offsets.get(cam, 0.0) + self._clip_padding_s
        cap = cv2.VideoCapture(path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        cap.set(cv2.CAP_PROP_POS_MSEC, t0 * 1000.0)
        ok, frame = cap.read()
        if not ok:
            cap.release()
            return
        h, w = frame.shape[:2]
        writer = cv2.VideoWriter(os.path.join(self._output_dir, f'clip_{n}_{cam}.avi'),
                                 cv2.VideoWriter_fourcc(*'MJPG'), fps, (w, h))
        while ok and cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0 <= t1:
            annotated = self._annotate(frame.copy(), v, cam)
            writer.write(annotated)
            ok, frame = cap.read()
        writer.release()
        cap.release()

    def close(self):
        with open(os.path.join(self._output_dir, 'verdicts.json'), 'w') as f:
            json.dump({'verdicts': self._verdicts, 'touches': self._touch_log}, f, indent=2)

    def get_params(self) -> dict:
        return {
            'output_dir': self._output_dir, 'videos': self._videos, 'offsets': self._offsets,
            'calibration': self._calibration, 'calibration_path': self._calibration_path,
            'pitch_length': self._pitch_length, 'pitch_width': self._pitch_width,
            'clip_padding_s': self._clip_padding_s, 'write_clips': self._write_clips,
            'name': self._name,
        }
