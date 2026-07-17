'''
Glue nodes for the offside flow, in their OWN importable module.

These small ProcessorNode/ConsumerNode subclasses must be reconstructable by the
distributed workers, which import a node by ``<module>.<ClassName>`` (see
videoflow/compiler.py — ``node_class = type(node).__module__ + '.' + type(node).__name__``).
If they were defined inside ``offside.py`` and that file were run as ``__main__``
(``python offside.py``) or loaded by the CLI (which uses a synthetic module name),
their class path would be ``__main__.X`` / ``_videoflow_user_graph.X`` — unimportable
by a worker subprocess. Defining them here makes the path ``offside_nodes.X``, which
any worker can import as long as this directory is on PYTHONPATH (``offside.main``
sets it; the Dockerfile ships this file).
'''
from __future__ import annotations

import json

import numpy as np
from videoflow.core.node import ConsumerNode, ProcessorNode


class FrameIndexSplitter(ProcessorNode):
    '''(frame_idx, frame) → frame.'''
    def process(self, item):
        return item[1] if isinstance(item, tuple) else item


class PersonBoxes(ProcessorNode):
    '''Keep player/GK/referee detections (classes 0,1,2) as (M,6)
    [ymin,xmin,ymax,xmax,class,score] — the tracker needs the class for BoxMOT.'''
    def process(self, dets):
        dets = np.asarray(dets, dtype=np.float64).reshape(-1, 6)
        if len(dets) == 0:
            return np.empty((0, 6))
        keep = np.isin(dets[:, 4].astype(int), [0, 1, 2])
        return dets[keep] if keep.any() else np.empty((0, 6))


class BallPick(ProcessorNode):
    '''Highest-score ball detection (class 3) → {'yx':[y,x],'score'} or None.'''
    def process(self, dets):
        dets = np.asarray(dets, dtype=np.float64).reshape(-1, 6)
        balls = dets[dets[:, 4].astype(int) == 3]
        if len(balls) == 0:
            return None
        b = balls[np.argmax(balls[:, 5])]
        cy = (b[0] + b[2]) / 2.0
        cx = (b[1] + b[3]) / 2.0
        return {'yx': [float(cy), float(cx)], 'score': float(b[5])}


class FeaturePacker(ProcessorNode):
    '''Row-aligned zip of (tracks, pose, team, ball) → compact per-camera Value dict.'''
    def __init__(self, cam, **kwargs):
        self._cam = cam
        super().__init__(**kwargs)

    def process(self, tracks, pose, team, ball):
        tracks = np.asarray(tracks, dtype=np.float64).reshape(-1, 5)
        pose = np.asarray(pose, dtype=np.float64)
        team = np.asarray(team, dtype=np.float64).reshape(-1, 2)
        out_tracks = []
        for i in range(len(tracks)):
            kp = pose[i].tolist() if i < len(pose) else np.zeros((26, 3)).tolist()
            tm = int(team[i, 0]) if i < len(team) else -1
            tc = float(team[i, 1]) if i < len(team) else 0.0
            out_tracks.append({'tid': int(tracks[i, 4]), 'box': tracks[i, :4].tolist(),
                               'team': tm, 'team_conf': tc, 'kpts': kp})
        return {'cam': self._cam, 'tracks': out_tracks, 'ball': ball}


class WorldStateJsonlWriter(ConsumerNode):
    '''Debug sink: append each world state to a JSONL file.'''
    def __init__(self, path, **kwargs):
        self._path = path
        super().__init__(**kwargs)

    def open(self):
        self._f = open(self._path, 'w')

    def consume(self, item):
        if item is not None:
            self._f.write(json.dumps(item) + '\n')

    def close(self):
        self._f.close()

    def get_params(self):
        return {'path': self._path, 'name': self._name}
