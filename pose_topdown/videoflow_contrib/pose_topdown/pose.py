'''
Top-down multi-person pose on tracked boxes, via rtmlib (Apache-2.0 ONNX models).

Backends: ``rtmw`` (whole-body 133, incl. feet — default, best accuracy), ``rtmpose``
(halpe26, faster), ``vitpose`` (max accuracy). Whatever the backend, output is
remapped to ONE canonical 26-keypoint layout (halpe26-style: COCO-17 body + head/
neck/hip-centre + 6 feet) so the fuser/engine are backend-independent.

Runs on the tracker's boxes and returns keypoints ROW-ALIGNED with the input
tracks — this row-alignment is the identity-preserving contract. Sapiens is
gated behind ``allow_noncommercial`` (CC-BY-NC weights).

Output: ``(N, 26, 3) = [x, y, score]`` pixel coordinates.
'''
from __future__ import annotations

from typing import Any

import numpy as np
from videoflow.core.constants import CPU, GPU
from videoflow.core.node import ProcessorNode

_NUM_CANON = 26
# Real rtmlib/OpenMMLab ONNX models (rtmlib downloads + extracts the .zip and caches
# under ~/.cache/rtmlib). Each entry: (onnx_url, (w, h) input size, source keypoint count).
#   'rtmpose' → Halpe26 (body + 6 feet) — IS the canonical 26 layout, so no remap.
#   'rtmw'    → COCO-WholeBody 133 (adds face/hands, dropped in the canonical remap).
_POSE_MODELS = {
    'rtmpose': ('https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/'
                'rtmpose-x_simcc-body7_pt-body7-halpe26_700e-384x288-7fb6e239_20230606.zip',
                (288, 384), 26),
    'rtmw': ('https://download.openmmlab.com/mmpose/v1/projects/rtmw/onnx_sdk/'
             'rtmw-dw-x-l_simcc-cocktail14_270e-384x288_20231122.zip',
             (288, 384), 133),
}


class PoseTopDown(ProcessorNode):
    '''
    - Arguments:
        - backend: 'rtmw' | 'rtmpose' | 'vitpose' | 'sapiens'.
        - kpt_format: nominal source format (informational; remap is by backend).
        - onnx_url: explicit model URL/path (else default per backend).
        - model_input_size: (w, h) network input.
        - kpt_min_conf: floor applied to reported scores.
        - allow_noncommercial: required to enable the Sapiens backend (CC-BY-NC).
    '''
    def __init__(self, backend: str = 'rtmw', kpt_format: str = 'wholebody133', onnx_url=None,
                 model_input_size=None, kpt_min_conf: float = 0.0,
                 allow_noncommercial: bool = False, nb_tasks: int = 1, device_type=GPU, **kwargs):
        self._backend = backend
        self._kpt_format = kpt_format
        self._onnx_url = onnx_url
        self._model_input_size = tuple(model_input_size) if model_input_size else None
        self._kpt_min_conf = float(kpt_min_conf)
        self._allow_noncommercial = bool(allow_noncommercial)
        self._pose: Any = None    # RTMPose, set in open()
        super().__init__(nb_tasks=nb_tasks, device_type=device_type, **kwargs)

    def open(self):
        if self._backend == 'sapiens' and not self._allow_noncommercial:
            raise RuntimeError("Sapiens weights are CC-BY-NC (non-commercial); set "
                               "allow_noncommercial=True to enable, or use 'rtmw'/'rtmpose'.")
        from rtmlib import RTMPose  # lazy: onnxruntime
        if self._backend not in _POSE_MODELS and self._onnx_url is None:
            raise ValueError(f"backend {self._backend!r} needs an explicit onnx_url "
                             f"(known: {sorted(_POSE_MODELS)})")
        default_url, default_size, _ = _POSE_MODELS.get(self._backend, (None, (288, 384), 26))
        url = self._onnx_url or default_url        # rtmlib downloads/extracts the .zip and caches
        size = self._model_input_size or default_size
        device = 'cpu' if self.device_type == CPU else 'cuda'
        self._pose = RTMPose(onnx_model=url, model_input_size=size,
                             backend='onnxruntime', device=device)

    def process(self, frame, tracks) -> np.ndarray:
        if isinstance(frame, tuple):
            frame = frame[1]
        frame = np.asarray(frame)
        tr = np.asarray(tracks, dtype=np.float64).reshape(-1, 5)
        if len(tr) == 0:
            return np.zeros((0, _NUM_CANON, 3))
        # y-first [ymin,xmin,ymax,xmax] → xyxy for rtmlib
        bboxes = np.stack([tr[:, 1], tr[:, 0], tr[:, 3], tr[:, 2]], axis=1)
        keypoints, scores = self._pose(frame, bboxes=bboxes)
        keypoints = np.asarray(keypoints, dtype=np.float64)   # (N, K, 2)
        scores = np.asarray(scores, dtype=np.float64)         # (N, K)
        out = np.zeros((len(tr), _NUM_CANON, 3))
        for i in range(len(tr)):
            out[i] = _to_canonical(keypoints[i], scores[i], self._backend, self._kpt_min_conf)
        return out


def _to_canonical(kpts: np.ndarray, scores: np.ndarray, backend: str, cmin: float) -> np.ndarray:
    '''Map a backend's keypoints to the canonical 26-point layout (x, y, score).'''
    K = kpts.shape[0]
    out = np.zeros((_NUM_CANON, 3))
    n_body = min(17, K)
    out[:n_body, :2] = kpts[:n_body]
    out[:n_body, 2] = scores[:n_body]

    if backend == 'rtmpose' or K == 26:               # halpe26: identity
        m = min(_NUM_CANON, K)
        out[:m, :2] = kpts[:m]
        out[:m, 2] = scores[:m]
    elif K >= 23:                                     # wholebody133 (rtmw / vitpose): feet 17-22
        out[20:26, :2] = kpts[17:23]
        out[20:26, 2] = scores[17:23]

    # Derived canonical points (head/neck/hip-centre) from body joints, when not present.
    def mid(a, b):
        if scores[a] > 0 and scores[b] > 0:
            return (kpts[a] + kpts[b]) / 2.0, min(scores[a], scores[b])
        return None, 0.0

    if out[17, 2] == 0 and K > 0:                     # head ← nose
        out[17] = [kpts[0, 0], kpts[0, 1], scores[0]]
    if out[18, 2] == 0 and K > 6:                     # neck ← mid-shoulder
        p, s = mid(5, 6)
        if p is not None:
            out[18] = [p[0], p[1], s]
    if out[19, 2] == 0 and K > 12:                    # hip centre ← mid-hip
        p, s = mid(11, 12)
        if p is not None:
            out[19] = [p[0], p[1], s]

    out[out[:, 2] < cmin, 2] = 0.0
    return out
