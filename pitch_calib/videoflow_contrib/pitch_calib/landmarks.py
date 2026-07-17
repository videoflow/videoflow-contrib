'''
Pitch-landmark detector — locates named pitch keypoints in a frame.

Backends (selected at construction):
- ``yolo32``: an Ultralytics YOLO-pose model trained to regress ~32 pitch keypoints
  (roboflow/sports style). AGPL — acceptable per the project's licensing decision.
- ``pnlcalib``: reserved for a PnLCalib-style HRNet backend (pending its license check).

Output is a ``(K, 3)`` array of ``[x, y, conf]`` in the canonical ``PitchModel``
keypoint order (missing/low-confidence landmarks have conf 0). The heavy model is
imported lazily so this module imports (and the descriptor validates / the wheel
builds) without ultralytics installed.
'''
from __future__ import annotations

from typing import Any

import numpy as np
from videoflow.utils.downloader import get_file

from .model import PitchModel

# HF-hosted YOLOv8x-pose pitch model (32 keypoints, roboflow SoccerPitchConfiguration
# order): martinjolif/yolo-football-pitch-detection. Primary = the community HF repo;
# fallback = our durable GitHub-release mirror. get_file tries them in order.
_MIRROR_URL = ('https://github.com/videoflow/videoflow-contrib/releases/download/'
               'offside_models/')
_DEFAULT_WEIGHTS = {
    'yolo32': ('yolo-football-pitch-detection.pt',
               ['https://huggingface.co/martinjolif/yolo-football-pitch-detection/'
                'resolve/main/yolo-football-pitch-detection.pt',
                _MIRROR_URL + 'yolo-football-pitch-detection.pt']),
}


class PitchLandmarkDetector:
    '''
    - Arguments:
        - backend: 'yolo32' (default) or 'pnlcalib'.
        - weights: explicit local weights path; if None, fetched by backend name.
        - conf: minimum keypoint confidence to report.
    '''
    def __init__(self, backend: str = 'yolo32', weights: str | None = None, conf: float = 0.3):
        self._backend = backend
        self._weights = weights
        self._conf = float(conf)
        self._model: Any = None    # backend model, set in load()
        # yolo32 outputs the 32 roboflow vertices in order → rf0..rf31 (dims-independent).
        self._names, _ = PitchModel().roboflow32_array()

    @property
    def landmark_names(self) -> list[str]:
        return list(self._names)

    def load(self) -> None:
        if self._model is not None:
            return
        if self._backend == 'yolo32':
            from ultralytics import YOLO  # lazy: heavy, AGPL
            name, url = _DEFAULT_WEIGHTS['yolo32']
            path = self._weights or get_file(name, url)
            self._model = YOLO(path)
        elif self._backend == 'pnlcalib':
            raise NotImplementedError('pnlcalib backend pending license clearance; use yolo32')
        else:
            raise ValueError(f'unknown backend {self._backend!r}')

    def detect(self, frame_bgr: np.ndarray) -> np.ndarray:
        '''Return (K, 3) [x, y, conf] in canonical landmark order.'''
        self.load()
        K = len(self._names)
        out = np.zeros((K, 3), dtype=np.float64)
        res = self._model.predict(frame_bgr, verbose=False)[0]
        kpts = getattr(res, 'keypoints', None)
        if kpts is None or kpts.data is None or len(kpts.data) == 0:
            return out
        data = kpts.data[0].cpu().numpy()      # (K_model, 3) [x, y, conf]
        m = min(K, data.shape[0])
        out[:m] = data[:m]
        out[out[:, 2] < self._conf] = 0.0
        return out

    def detect_named(self, frame_bgr: np.ndarray) -> dict:
        '''Convenience: ``{landmark_name: (x, y)}`` for confident detections.'''
        arr = self.detect(frame_bgr)
        return {n: (float(arr[i, 0]), float(arr[i, 1]))
                for i, n in enumerate(self._names) if arr[i, 2] > 0}
