'''
Per-camera calibration from pitch markings (fixed cameras → calibrate once).

For each camera we sample frames across the clip, detect pitch landmarks, robustly
average each landmark (fixed camera ⇒ averaging kills jitter/occlusion), then solve
full intrinsics + extrinsics. Writes ``calib/<cam>.json`` and a wireframe-reprojection
overlay PNG to eyeball the fit. ``--manual`` opens a click UI to place >=6 named
points when automatic markings are too faint.

    python calibrate.py --config config.yaml [--cam cam0] [--manual]
'''
from __future__ import annotations

import argparse
import json
import os

import cv2
import numpy as np
from common import load_config
from videoflow_contrib.offside_visualizer.drawing import project_world_points
from videoflow_contrib.pitch_calib import PitchLandmarkDetector, PitchModel, solve_camera

N_SAMPLES = 40


def sample_frames(video_path: str, n: int):
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    idxs = np.linspace(0, max(0, total - 1), n).astype(int)
    frames = []
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, frame = cap.read()
        if ok:
            frames.append(frame)
    cap.release()
    return frames


def robust_average(observations: dict, n_frames: int, min_frac: float = 0.3) -> dict:
    '''observations: {name: [(x,y,conf), ...]} → {name: (x,y)} robustly averaged.'''
    out = {}
    for name, obs in observations.items():
        if len(obs) < max(2, min_frac * n_frames):
            continue
        arr = np.array(obs)
        xy, conf = arr[:, :2], arr[:, 2]
        med = np.median(xy, axis=0)
        mad = np.median(np.abs(xy - med), axis=0) + 1e-6
        keep = np.all(np.abs(xy - med) <= 2.0 * mad * 1.4826 + 3.0, axis=1)
        if keep.sum() == 0:
            keep = np.ones(len(xy), dtype=bool)
        w = conf[keep]
        out[name] = tuple(np.average(xy[keep], axis=0, weights=w))
    return out


def draw_overlay(frame, pitch: PitchModel, calib: dict):
    img = frame.copy()
    for seg in pitch.line_segments():
        px = project_world_points(seg, calib)
        pts = px[np.all(np.isfinite(px), axis=1)].astype(np.int32)
        if len(pts) >= 2:
            cv2.polylines(img, [pts.reshape(-1, 1, 2)], False, (0, 255, 255), 2, cv2.LINE_AA)
    return img


def manual_points(frame, pitch: PitchModel):
    '''Click UI: user places named pitch points; returns {name: (x,y)}.'''
    names, _ = pitch.keypoint_array()
    picked, idx = {}, [0]
    disp = frame.copy()

    def on_click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and idx[0] < len(names):
            picked[names[idx[0]]] = (float(x), float(y))
            cv2.circle(disp, (x, y), 4, (0, 0, 255), -1)
            idx[0] += 1

    cv2.namedWindow('calib', cv2.WINDOW_NORMAL)
    cv2.setMouseCallback('calib', on_click)
    print('Click each named landmark (ESC to finish, s to skip one):')
    while True:
        d = disp.copy()
        if idx[0] < len(names):
            cv2.putText(d, f'Place: {names[idx[0]]}', (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        cv2.imshow('calib', d)
        k = cv2.waitKey(20) & 0xFF
        if k == 27:
            break
        if k == ord('s'):
            idx[0] += 1
    cv2.destroyAllWindows()
    return picked


def calibrate_camera(cfg, cam, manual=False):
    pitch = PitchModel(cfg.pitch_length, cfg.pitch_width)
    frames = sample_frames(cfg.videos[cam], N_SAMPLES)
    if not frames:
        raise RuntimeError(f'no frames read from {cfg.videos[cam]}')
    h, w = frames[0].shape[:2]

    if manual:
        obs = manual_points(frames[len(frames) // 2], pitch)
        weights = {n: 3.0 for n in obs}
    else:
        detector = PitchLandmarkDetector(backend='yolo32')
        acc: dict = {}
        for fr in frames:
            for name, (x, y) in detector.detect_named(fr).items():
                acc.setdefault(name, []).append((x, y, 1.0))
        obs = robust_average(acc, len(frames))
        weights = None

    calib = solve_camera(obs, pitch, (h, w), refine=True, weights=weights)
    os.makedirs(cfg.calib_dir(), exist_ok=True)
    with open(os.path.join(cfg.calib_dir(), f'{cam}.json'), 'w') as f:
        json.dump(calib, f, indent=2)
    overlay = draw_overlay(frames[len(frames) // 2], pitch, calib)
    cv2.imwrite(os.path.join(cfg.work_dir, f'calib_overlay_{cam}.png'), overlay)
    status = 'OK' if calib['rms_px'] < 5.0 else 'HIGH RMS — check overlay / try --manual'
    print(f'{cam}: RMS {calib["rms_px"]:.2f} px  ({calib["n_landmarks"]} landmarks)  [{status}]')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--cam', default=None, help='calibrate a single camera')
    ap.add_argument('--manual', action='store_true', help='manual click calibration')
    args = ap.parse_args()
    cfg = load_config(args.config)
    cams = [args.cam] if args.cam else cfg.cameras
    for cam in cams:
        calibrate_camera(cfg, cam, manual=args.manual)
    print(f'\nWrote calibration to {cfg.calib_dir()}')


if __name__ == '__main__':
    main()
