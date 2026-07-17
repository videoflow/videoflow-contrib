'''
Appearance-aware multi-object tracker built on BoxMOT (AGPL-3.0).

Defaults to BoT-SORT with ReID + camera-motion compensation — the right fit for
soccer (near-identical jerseys, erratic motion, tripod jitter). ``method`` swaps
in OC-SORT / DeepOCSORT / BoostTrack / ByteTrack / HybridSORT with no interface
change. Consumes the frame + detections (ReID needs the image) and preserves the
repo's y-first box convention.

Stateful across frames ⇒ OneTaskProcessorNode (single task). The boxmot import is
lazy (open()) so this module imports without torch/boxmot present.
'''
from __future__ import annotations

from typing import Any

import numpy as np
from videoflow.core.constants import CPU, GPU
from videoflow.core.node import OneTaskProcessorNode


class BoxmotTracker(OneTaskProcessorNode):
    '''
    - Arguments:
        - method: 'botsort' (default) | 'ocsort' | 'deepocsort' | 'boosttrack' |
          'bytetrack' | 'hybridsort'.
        - reid_weights: ReID model (path or a BoxMOT model-zoo name).
        - half: fp16 inference (GPU).
        - min_conf: drop detections below this score before tracking.
        - per_class: track each class independently.
    - process(frame, detections) where detections is (N,6) [ymin,xmin,ymax,xmax,class,score]
      → (M,5) [ymin,xmin,ymax,xmax,track_id].
    '''
    def __init__(self, method: str = 'botsort', reid_weights=None, half: bool = True,
                 min_conf: float = 0.3, per_class: bool = False, cmc_method: str = 'ecc',
                 with_reid=None, frame_rate: int = 30, device_type=GPU, **kwargs):
        self._method = method
        self._reid_weights = reid_weights
        self._half = bool(half)
        self._min_conf = float(min_conf)
        self._per_class = bool(per_class)
        self._cmc_method = cmc_method
        # Appearance ReID on by default when a reid model is given (else motion + CMC).
        self._with_reid = with_reid
        self._frame_rate = int(frame_rate)
        self._tracker: Any = None    # BoxMOT tracker, set in open()
        super().__init__(device_type=device_type, **kwargs)

    def open(self):
        import inspect
        from pathlib import Path

        from boxmot.trackers import (  # lazy: heavy (torch)
            BoostTrack,
            BotSort,
            ByteTrack,
            DeepOcSort,
            HybridSort,
            OcSort,
            StrongSort,
        )
        classes = {'botsort': BotSort, 'ocsort': OcSort, 'deepocsort': DeepOcSort,
                   'boosttrack': BoostTrack, 'bytetrack': ByteTrack,
                   'hybridsort': HybridSort, 'strongsort': StrongSort}
        Cls = classes.get(self._method)
        if Cls is None:
            raise ValueError(f'unknown tracker method {self._method!r} (known: {sorted(classes)})')
        with_reid = self._with_reid if self._with_reid is not None else (self._reid_weights is not None)
        # Pass only the kwargs this tracker class actually accepts — robust across the
        # differing per-tracker signatures (and across boxmot versions).
        params = inspect.signature(Cls.__init__).parameters
        kw: dict = {}
        if 'reid_model' in params and self._reid_weights is not None:
            kw['reid_model'] = Path(self._reid_weights)
        if 'with_reid' in params:
            kw['with_reid'] = with_reid
        if 'cmc_method' in params:
            kw['cmc_method'] = self._cmc_method
        if 'frame_rate' in params:
            kw['frame_rate'] = self._frame_rate
        if 'per_class' in params:
            kw['per_class'] = self._per_class
        if 'half' in params:
            kw['half'] = self._half and self.device_type != CPU
        if 'device' in params:
            kw['device'] = 'cpu' if self.device_type == CPU else 'cuda:0'
        self._tracker = Cls(**kw)

    def process(self, frame, detections) -> np.ndarray:
        if isinstance(frame, tuple):
            frame = frame[1]
        frame = np.asarray(frame)
        dets = np.asarray(detections, dtype=np.float64).reshape(-1, 6)
        if len(dets):
            dets = dets[dets[:, 5] >= self._min_conf]
        # y-first [ymin,xmin,ymax,xmax,class,score] → BoxMOT [x1,y1,x2,y2,conf,cls]
        if len(dets):
            bm = np.stack([dets[:, 1], dets[:, 0], dets[:, 3], dets[:, 2],
                           dets[:, 5], dets[:, 4]], axis=1)
        else:
            bm = np.zeros((0, 6))
        tracks = self._tracker.update(bm, frame)      # TrackResults (N,8): [x1,y1,x2,y2,id,conf,cls,det_ind]
        tracks = np.asarray(tracks, dtype=np.float64)
        if tracks.size == 0:
            return np.empty((0, 5))
        tracks = tracks.reshape(-1, tracks.shape[-1])
        # BoxMOT xyxy+id → repo y-first [ymin,xmin,ymax,xmax,track_id]
        return np.stack([tracks[:, 1], tracks[:, 0], tracks[:, 3], tracks[:, 2],
                         tracks[:, 4]], axis=1)
