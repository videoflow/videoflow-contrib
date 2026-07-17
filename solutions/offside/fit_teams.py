'''
Fit team / goalkeeper / referee color centroids once, across all cameras, for a
globally-consistent stateless team classifier at runtime.

Samples frames from every camera, runs the soccer detector, embeds torso crops,
fits centroids, and writes ``teams.json`` plus ``teams_montage.png`` (sample crops
per class) for you to eyeball and to fill in ``team_names`` in the config.

    python fit_teams.py --config config.yaml
'''
from __future__ import annotations

import argparse
import json
import os

import cv2
import numpy as np
from common import load_config
from videoflow_contrib.soccer_detector import BALL, SoccerDetector
from videoflow_contrib.team_classifier.fitting import CLASSES, embed_crops, fit_teams, torso_crop

N_PER_CAM = 60


def sample_frames(video_path, n):
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    idxs = np.linspace(0, max(0, total - 1), n).astype(int)
    frames = []
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, fr = cap.read()
        if ok:
            frames.append(fr)
    cap.release()
    return frames


def montage(crops_by_class, out_path, per_class=16, cell=(48, 96)):
    rows = []
    for cls_name in CLASSES:
        crops = crops_by_class.get(cls_name, [])[:per_class]
        cells = []
        for c in crops:
            cells.append(cv2.resize(c, cell))
        while len(cells) < per_class:
            cells.append(np.zeros((cell[1], cell[0], 3), dtype=np.uint8))
        rows.append(np.hstack(cells))
    grid = np.vstack(rows)
    cv2.imwrite(out_path, grid)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)

    detector = SoccerDetector(checkpoint=cfg.detector.get('checkpoint'),
                              resolution=int(cfg.detector.get('resolution', 1288)),
                              device_type='cpu')
    detector.open()

    crops, det_classes = [], []
    montage_by_class = {c: [] for c in CLASSES}
    for cam in cfg.cameras:
        for fr in sample_frames(cfg.videos[cam], N_PER_CAM):
            dets = detector._detect(fr)
            for d in dets:
                cls = int(d[4])
                if cls == BALL or d[5] < 0.5:
                    continue
                crop = torso_crop(fr, d[:4])
                if crop is None:
                    continue
                crops.append(crop)
                det_classes.append(cls)
                name = CLASSES[cls] if cls < len(CLASSES) else 'team0'
                if len(montage_by_class[name]) < 40:
                    montage_by_class[name].append(crop)

    if len(crops) < 4:
        raise RuntimeError('too few player crops detected; check the detector/checkpoint')

    embs = embed_crops(crops, method='hsv')
    centroids = fit_teams(embs, np.array(det_classes), method='hsv')
    with open(cfg.teams_path(), 'w') as f:
        json.dump(centroids, f, indent=2)
    montage(montage_by_class, os.path.join(cfg.work_dir, 'teams_montage.png'))
    print(f'Wrote {cfg.teams_path()} and teams_montage.png '
          f'({len(crops)} crops). Inspect the montage and set team_names in the config.')


if __name__ == '__main__':
    main()
