'''
Soccer object detector (players / goalkeepers / referees / ball) built on RF-DETR
(Apache-2.0), fine-tuned on soccer footage.

Output follows the repo's y-first detection convention:
``(N, 6) = [ymin, xmin, ymax, xmax, class_id, score]`` with class ids
``{0: player, 1: goalkeeper, 2: referee, 3: ball}``. Per-class thresholds keep
the small, low-confidence ball while holding players to a stricter bar. Optional
stateless tiled inference improves recall on distant balls.

The rfdetr import is lazy (open()) so this module imports and the descriptor
validates without torch/rfdetr present.
'''
from __future__ import annotations

from typing import Any

import numpy as np
from videoflow.core.constants import GPU
from videoflow.processors.vision.detectors import ObjectDetector
from videoflow.utils.downloader import get_file

BASE_URL = 'https://github.com/videoflow/videoflow-contrib/releases/download/offside_models/'
_DEFAULT_WEIGHTS = 'rfdetr_soccernet.pth'
# Real HF-hosted YOLOv8 soccer detector (runs anywhere via ultralytics; the RF-DETR
# backend needs torch >= 2.4, which some platforms — e.g. x86 macOS — can't provide).
_YOLO_WEIGHTS = ('yolo-football-player-detection.pt',
                 'https://huggingface.co/martinjolif/yolo-football-player-detection/'
                 'resolve/main/yolo-football-player-detection.pt')
# Canonical class ids the rest of the pipeline expects.
PLAYER, GOALKEEPER, REFEREE, BALL = 0, 1, 2, 3


class SoccerDetector(ObjectDetector):
    '''
    - Arguments:
        - checkpoint: local RF-DETR weights path (else fetched).
        - resolution: model input size (must be divisible by 56; 1288 = 56·23).
        - conf_person / conf_ball: per-class score thresholds.
        - tile_inference: run overlapping tiles and merge (better distant-ball recall).
        - class_map: raw model class id → canonical id. Default matches the HF
          `julianzu9612/RFDETR-Soccernet` checkpoint, whose config.json declares
          {0:ball, 1:player, 2:referee, 3:goalkeeper} — remapped here to the
          pipeline's canonical {player:0, goalkeeper:1, referee:2, ball:3}.
    '''
    # Raw model class ids → canonical ids. Order differs per checkpoint:
    #   RF-DETR SoccerNet (config.json):  {0:ball, 1:player, 2:referee, 3:goalkeeper}
    #   YOLO football-player (model.names): {0:ball, 1:goalkeeper, 2:player, 3:referee}
    SOCCERNET_CLASS_MAP = {0: BALL, 1: PLAYER, 2: REFEREE, 3: GOALKEEPER}
    YOLO_CLASS_MAP = {0: BALL, 1: GOALKEEPER, 2: PLAYER, 3: REFEREE}

    def __init__(self, checkpoint=None, backend: str = 'rfdetr', resolution: int = 1280,
                 conf_person: float = 0.35, conf_ball: float = 0.15, tile_inference: bool = False,
                 tile_rows: int = 2, tile_cols: int = 3, tile_overlap: float = 0.1, class_map=None,
                 model_size: str = 'large', nb_tasks: int = 1, device_type=GPU, **kwargs):
        self._backend = backend
        self._checkpoint = checkpoint
        self._resolution = int(resolution)
        self._conf_person = float(conf_person)
        self._conf_ball = float(conf_ball)
        self._tile_inference = bool(tile_inference)
        self._tile_rows = int(tile_rows)
        self._tile_cols = int(tile_cols)
        self._tile_overlap = float(tile_overlap)
        self._class_map = class_map        # None → resolved per-backend in open()
        self._model_size = model_size
        self._model: Any = None    # backend model, set in open()
        super().__init__(nb_tasks=nb_tasks, device_type=device_type, **kwargs)

    def open(self):
        if self._backend == 'yolo':
            from ultralytics import YOLO  # lazy: runs anywhere (torch >= 2.0)
            path = self._checkpoint or get_file(*_YOLO_WEIGHTS)
            self._model = YOLO(path)
            if self._class_map is None:
                self._class_map = dict(self.YOLO_CLASS_MAP)
        else:  # rfdetr — RF-DETR-Large SoccerNet checkpoint (128M, DINOv2, 1280 res)
            import rfdetr  # lazy: heavy (torch >= 2.4)
            path = self._checkpoint or get_file(_DEFAULT_WEIGHTS, BASE_URL + _DEFAULT_WEIGHTS)
            Model = rfdetr.RFDETRLarge if self._model_size == 'large' else rfdetr.RFDETRBase
            self._model = Model(pretrain_weights=path, resolution=self._resolution)
            try:
                self._model.optimize_for_inference()
            except Exception:
                pass
            if self._class_map is None:
                self._class_map = dict(self.SOCCERNET_CLASS_MAP)

    def _min_threshold(self) -> float:
        return min(self._conf_person, self._conf_ball)

    def _detect(self, im: np.ndarray) -> np.ndarray:
        # YOLO (ultralytics) expects BGR; RF-DETR expects RGB.
        img = im if self._backend == 'yolo' else (im[..., ::-1] if im.shape[-1] == 3 else im)
        if self._tile_inference:
            dets = self._detect_tiled(img)
        else:
            dets = self._predict(img, 0, 0)
        return self._filter(dets)

    def _predict(self, img: np.ndarray, off_x: float, off_y: float) -> np.ndarray:
        if self._backend == 'yolo':
            res = self._model.predict(img, conf=self._min_threshold(), verbose=False)[0]
            b = res.boxes
            xyxy = b.xyxy.cpu().numpy().astype(np.float64).reshape(-1, 4)
            conf = b.conf.cpu().numpy().astype(np.float64).reshape(-1)
            cls = b.cls.cpu().numpy().reshape(-1)
        else:
            det = self._model.predict(img, threshold=self._min_threshold())
            xyxy = np.asarray(det.xyxy, dtype=np.float64).reshape(-1, 4)
            conf = np.asarray(det.confidence, dtype=np.float64).reshape(-1)
            cls = np.asarray(det.class_id).reshape(-1)
        out = np.zeros((len(xyxy), 6))
        for i in range(len(xyxy)):
            x1, y1, x2, y2 = xyxy[i]
            canon = self._class_map.get(int(cls[i]), PLAYER)
            out[i] = [y1 + off_y, x1 + off_x, y2 + off_y, x2 + off_x, canon, conf[i]]
        return out

    def _detect_tiled(self, rgb: np.ndarray) -> np.ndarray:
        h, w = rgb.shape[:2]
        th, tw = h // self._tile_rows, w // self._tile_cols
        oh, ow = int(th * self._tile_overlap), int(tw * self._tile_overlap)
        all_dets = [self._predict(rgb, 0, 0)]     # full-frame pass anchors large objects
        for r in range(self._tile_rows):
            for c in range(self._tile_cols):
                y0, x0 = max(0, r * th - oh), max(0, c * tw - ow)
                y1, x1 = min(h, (r + 1) * th + oh), min(w, (c + 1) * tw + ow)
                tile = rgb[y0:y1, x0:x1]
                if tile.size:
                    all_dets.append(self._predict(tile, x0, y0))
        merged = np.concatenate(all_dets, axis=0) if all_dets else np.zeros((0, 6))
        return _class_wise_nms(merged, iou_thr=0.5)

    def _filter(self, dets: np.ndarray) -> np.ndarray:
        if len(dets) == 0:
            return np.zeros((0, 6))
        keep = []
        for d in dets:
            thr = self._conf_ball if int(d[4]) == BALL else self._conf_person
            if d[5] >= thr:
                keep.append(d)
        return np.array(keep) if keep else np.zeros((0, 6))


def _class_wise_nms(dets: np.ndarray, iou_thr: float = 0.5) -> np.ndarray:
    '''NMS within each class. dets rows: [ymin,xmin,ymax,xmax,class,score].'''
    if len(dets) == 0:
        return dets
    out = []
    for cls in np.unique(dets[:, 4]):
        rows = dets[dets[:, 4] == cls]
        idx = np.argsort(rows[:, 5])[::-1]
        while idx.size:
            i = idx[0]
            out.append(rows[i])
            if idx.size == 1:
                break
            rest = idx[1:]
            ious = _iou_yfirst(rows[i, :4], rows[rest, :4])
            idx = rest[ious < iou_thr]
    return np.array(out)


def _iou_yfirst(box, boxes):
    y1 = np.maximum(box[0], boxes[:, 0]); x1 = np.maximum(box[1], boxes[:, 1])
    y2 = np.minimum(box[2], boxes[:, 2]); x2 = np.minimum(box[3], boxes[:, 3])
    inter = np.clip(y2 - y1, 0, None) * np.clip(x2 - x1, 0, None)
    a = (box[2] - box[0]) * (box[3] - box[1])
    b = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    return inter / (a + b - inter + 1e-9)
