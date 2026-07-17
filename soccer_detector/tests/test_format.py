'''
Format-logic tests for SoccerDetector using a mock model — verifies the class
remapping, xyxy→y-first conversion, per-class thresholds, and tiled-NMS merge
without needing the heavy backends (rfdetr needs torch>=2.4; ultralytics needs
torch). The live backends are exercised separately against real weights.
'''
import numpy as np
from videoflow_contrib.soccer_detector.detector import (
    BALL,
    GOALKEEPER,
    PLAYER,
    REFEREE,
    SoccerDetector,
    _class_wise_nms,
)


class _RFDETRDetections:
    '''Mimics supervision.Detections (what rfdetr.predict returns).'''
    def __init__(self, xyxy, confidence, class_id):
        self.xyxy = np.asarray(xyxy)
        self.confidence = np.asarray(confidence)
        self.class_id = np.asarray(class_id)


class _MockRFDETR:
    def __init__(self, dets):
        self._dets = dets

    def predict(self, img, threshold=0.0):
        return self._dets


def _mk(backend='rfdetr'):
    d = SoccerDetector(backend=backend, conf_person=0.35, conf_ball=0.15, name='d')
    d._class_map = dict(d.SOCCERNET_CLASS_MAP)   # what open() would set for rfdetr
    return d


def test_rfdetr_class_remap_and_yfirst():
    # SoccerNet raw order {0:ball,1:player,2:referee,3:goalkeeper}
    dets = _RFDETRDetections(
        xyxy=[[100, 50, 140, 200],     # raw class 1 = player
              [300, 80, 305, 85],      # raw class 0 = ball
              [500, 60, 540, 210],     # raw class 3 = goalkeeper
              [700, 70, 735, 205]],    # raw class 2 = referee
        confidence=[0.9, 0.5, 0.8, 0.7],
        class_id=[1, 0, 3, 2])
    d = _mk('rfdetr')
    d._model = _MockRFDETR(dets)
    out = d._detect(np.zeros((1080, 1920, 3), np.uint8))
    # canonical classes recovered
    canon = sorted(int(r[4]) for r in out)
    assert canon == [PLAYER, GOALKEEPER, REFEREE, BALL], canon
    # y-first: row for the player det = [y1,x1,y2,x2,class,score] = [50,100,200,140,...]
    player = out[[int(r[4]) == PLAYER for r in out].index(True)]
    assert list(player[:4]) == [50, 100, 200, 140]


def test_per_class_thresholds():
    # a ball at 0.20 passes (conf_ball=0.15); a player at 0.20 is dropped (conf_person=0.35)
    dets = _RFDETRDetections(
        xyxy=[[10, 10, 14, 14], [50, 50, 90, 200]],
        confidence=[0.20, 0.20],
        class_id=[0, 1])     # ball, player
    d = _mk('rfdetr')
    d._model = _MockRFDETR(dets)
    out = d._detect(np.zeros((100, 100, 3), np.uint8))
    classes = [int(r[4]) for r in out]
    assert BALL in classes and PLAYER not in classes


def test_class_wise_nms_dedups_overlaps():
    # two near-identical player boxes → NMS keeps the higher-score one
    dets = np.array([[100, 100, 200, 140, PLAYER, 0.9],
                     [102, 101, 201, 141, PLAYER, 0.6],
                     [500, 500, 560, 540, BALL, 0.8]])
    kept = _class_wise_nms(dets, iou_thr=0.5)
    players = kept[kept[:, 4] == PLAYER]
    assert len(players) == 1 and players[0, 5] == 0.9
    assert (kept[:, 4] == BALL).sum() == 1
