'''
Stateless team classifier — assigns each tracked player to team0/team1/GK/referee
by nearest centroid (centroids fit offline by ``fit_teams.py``).

Row-aligned with the input tracks. HSV embedding by default (no heavy deps);
SigLIP-2 optional. Partitionable (stateless).

Output: ``(N, 2) = [class_id, confidence]`` where class_id ∈ {0,1,2,3, -1(unknown)}.
'''
from __future__ import annotations

import json

import numpy as np
from videoflow.core.constants import CPU
from videoflow.core.errors import (
    WORKER_FATAL,
    ConfigError,
    SchemaError,
    register_classifier_for,
)
from videoflow.core.node import ProcessorNode

from . import fitting

_torch_classifiers_registered = False


def _torch_oom_type() -> type | None:
    import torch  # lazy: only the siglip path pulls torch in at all
    return getattr(torch.cuda, 'OutOfMemoryError', None)


def _register_torch_classifiers() -> None:
    '''
    Maps torch's CUDA out-of-memory error onto ``worker_fatal``, for the siglip
    embedding path (the default HSV path is pure numpy and never sees torch).

    A component cannot subclass ``torch.cuda.OutOfMemoryError``, so videoflow is
    told about it instead — from ``open()``, since torch must not be a
    module-scope import here. An OOM belongs to this worker, not to the frame it
    was holding, and the mapping is what keeps the frame out of the dead-letter
    queue.
    '''
    global _torch_classifiers_registered
    if _torch_classifiers_registered:
        return
    _torch_classifiers_registered = register_classifier_for(
        'torch.cuda.OutOfMemoryError', WORKER_FATAL, _torch_oom_type)


class TeamClassifier(ProcessorNode):
    '''
    - Arguments:
        - centroids: inline ``{method, classes, vectors}`` dict (preferred), or
        - centroids_path: path to a teams.json with the same schema.
        - method: 'hsv' (default) or 'siglip'.
        - min_crop_h: skip boxes shorter than this (px).
    '''
    def __init__(self, centroids=None, centroids_path=None, method: str = 'hsv',
                 min_crop_h: int = 24, siglip_model: str = 'google/siglip2-base-patch16-224',
                 nb_tasks: int = 1, device_type=CPU, **kwargs):
        self._centroids = centroids
        self._centroids_path = centroids_path
        self._method = method
        self._min_crop_h = int(min_crop_h)
        self._siglip_model = siglip_model
        self._model = None
        super().__init__(nb_tasks=nb_tasks, device_type=device_type, **kwargs)

    def open(self):
        if self._centroids is None:
            if not self._centroids_path:
                raise ConfigError(
                    'TeamClassifier was given neither centroids nor centroids_path.',
                    remedy = 'Pass the inline centroids dict from fit_teams, or a '
                            'centroids_path pointing at the teams.json it wrote.',
                    node = self.name)
            with open(self._centroids_path) as f:
                self._centroids = json.load(f)
        self._method = self._centroids.get('method', self._method)
        # Checked here rather than on the first frame: a method the embedder does
        # not know is a config mistake, and finding it at worker start is one
        # failure instead of one per message.
        if self._method not in fitting.METHODS:
            raise ConfigError(
                f'TeamClassifier got method {self._method!r}.',
                remedy = f'Use one of: {", ".join(fitting.METHODS)}.',
                node = self.name, method = self._method)
        if self._method == 'siglip':
            _register_torch_classifiers()
            from transformers import AutoModel, AutoProcessor
            proc = AutoProcessor.from_pretrained(self._siglip_model)
            net = AutoModel.from_pretrained(self._siglip_model)
            self._model = (proc, net)

    def process(self, frame, tracks) -> np.ndarray:
        if isinstance(frame, tuple):
            frame = frame[1]
        frame = np.asarray(frame)
        # A track array of the wrong shape fails identically on every redelivery,
        # so it is dead-lettered on the first failure rather than retried to the
        # end of the budget under a bare numpy reshape error.
        try:
            tr = np.asarray(tracks, dtype=np.float64).reshape(-1, 5)
        except (TypeError, ValueError) as e:
            raise SchemaError(
                f'expected an (N, 5) [ymin,xmin,ymax,xmax,track_id] array: {e}',
                remedy = 'Check the tracker feeding this classifier emits the '
                        'y-first 5-column format.',
                node = self.name) from e
        out = np.full((len(tr), 2), [-1.0, 0.0])
        if len(tr) == 0:
            return out
        crops, rows = [], []
        for i in range(len(tr)):
            crop = fitting.torso_crop(frame, tr[i, :4], self._min_crop_h)
            if crop is not None:
                crops.append(crop)
                rows.append(i)
        if not crops:
            return out
        embs = fitting.embed_crops(crops, self._method, self._model)
        for k, i in enumerate(rows):
            cid, conf = fitting.assign(embs[k], self._centroids)
            out[i] = [float(cid), float(conf)]
        return out
