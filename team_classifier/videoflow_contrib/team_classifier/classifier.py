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
from videoflow.core.node import ProcessorNode

from . import fitting


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
                raise ValueError('TeamClassifier needs centroids or centroids_path')
            with open(self._centroids_path) as f:
                self._centroids = json.load(f)
        self._method = self._centroids.get('method', self._method)
        if self._method == 'siglip':
            from transformers import AutoModel, AutoProcessor
            proc = AutoProcessor.from_pretrained(self._siglip_model)
            net = AutoModel.from_pretrained(self._siglip_model)
            self._model = (proc, net)

    def process(self, frame, tracks) -> np.ndarray:
        if isinstance(frame, tuple):
            frame = frame[1]
        frame = np.asarray(frame)
        tr = np.asarray(tracks, dtype=np.float64).reshape(-1, 5)
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
