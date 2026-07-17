'''
Videoflow node wrapper for the pitch-landmark detector.

Emits per-frame landmark observations for streaming (re)calibration. Offline
fixed-camera calibration normally uses the library directly (``calibrate.py`` calls
``PitchLandmarkDetector`` + ``solve_camera`` over sampled frames), but this node
lets calibration run inside a flow when cameras move.
'''
from __future__ import annotations

import numpy as np
from videoflow.core.constants import CPU
from videoflow.core.node import ProcessorNode

from .landmarks import PitchLandmarkDetector


class PitchLandmarks(ProcessorNode):
    '''
    - Arguments:
        - backend: 'yolo32' (default) or 'pnlcalib'.
        - weights: explicit local weights path (else fetched by backend name).
        - conf: minimum keypoint confidence.
    - process(frame) → (K, 3) [x, y, conf] in canonical PitchModel landmark order.
    '''
    def __init__(self, backend: str = 'yolo32', weights=None, conf: float = 0.3,
                 nb_tasks: int = 1, device_type=CPU, **kwargs):
        self._backend = backend
        self._weights = weights
        self._conf = float(conf)
        self._detector: PitchLandmarkDetector | None = None
        super().__init__(nb_tasks=nb_tasks, device_type=device_type, **kwargs)

    def open(self):
        self._detector = PitchLandmarkDetector(self._backend, self._weights, self._conf)
        self._detector.load()

    def process(self, frame) -> np.ndarray:
        if isinstance(frame, tuple):        # (frame_idx, frame) from a reader
            frame = frame[1]
        assert self._detector is not None   # set in open()
        return self._detector.detect(np.asarray(frame))
