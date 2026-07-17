'''
Parametric soccer-pitch model — the world coordinate frame + canonical landmarks.

Pure geometry (numpy only). The pitch outer dimensions (length, width) are
parametric because amateur pitches vary; all *internal* markings are fixed by the
Laws of the Game (penalty area 16.5 m, goal area 5.5 m, penalty spot 11 m, centre
circle r 9.15 m, goal 7.32 m wide, etc.).

World frame: origin at the pitch centre, +X along the length (goal to goal), +Y
along the width, +Z up. Left goal line at ``X = -length/2``, right at ``+length/2``.
All landmarks lie on the ground plane ``Z = 0``.
'''
from __future__ import annotations

from collections import OrderedDict

import numpy as np

# Law-fixed markings (metres).
PENALTY_AREA_DEPTH = 16.5
PENALTY_AREA_HALF_WIDTH = 20.16      # 7.32/2 + 16.5
GOAL_AREA_DEPTH = 5.5
GOAL_AREA_HALF_WIDTH = 9.16          # 7.32/2 + 5.5
PENALTY_SPOT_DIST = 11.0
GOAL_HALF_WIDTH = 3.66               # 7.32 / 2
CENTER_CIRCLE_R = 9.15
# Penalty-arc / penalty-area-line junction: arc (r=9.15 about the spot) meets the
# 16.5 m line, which is 5.5 m from the spot → y = sqrt(9.15² − 5.5²).
_PEN_ARC_Y = float(np.sqrt(CENTER_CIRCLE_R ** 2 - (PENALTY_AREA_DEPTH - PENALTY_SPOT_DIST) ** 2))


class PitchModel:
    '''
    - Arguments:
        - length: goal-line to goal-line distance in metres (default 105).
        - width: touchline to touchline distance in metres (default 68).
    '''
    def __init__(self, length: float = 105.0, width: float = 68.0):
        self.length = float(length)
        self.width = float(width)
        self.half_l = self.length / 2.0
        self.half_w = self.width / 2.0

    def keypoints(self) -> "OrderedDict[str, np.ndarray]":
        '''
        Canonical named landmarks → world (X, Y, 0). ~35 points: pitch corners,
        halfway/centre features, and per-end penalty/goal-area/spot/post junctions.
        The landmark detector must emit points in *this* name order.
        '''
        kp: "OrderedDict[str, np.ndarray]" = OrderedDict()
        hl, hw = self.half_l, self.half_w

        # Centre features.
        kp['center_spot'] = (0.0, 0.0)
        kp['halfway_top'] = (0.0, hw)
        kp['halfway_bottom'] = (0.0, -hw)
        kp['center_circle_top'] = (0.0, CENTER_CIRCLE_R)
        kp['center_circle_bottom'] = (0.0, -CENTER_CIRCLE_R)

        for tag, sx in (('left', -1.0), ('right', 1.0)):
            gx = sx * hl                                   # goal-line X
            kp[f'{tag}_corner_top'] = (gx, hw)
            kp[f'{tag}_corner_bottom'] = (gx, -hw)
            # Penalty area.
            kp[f'{tag}_pen_goalline_top'] = (gx, PENALTY_AREA_HALF_WIDTH)
            kp[f'{tag}_pen_goalline_bottom'] = (gx, -PENALTY_AREA_HALF_WIDTH)
            kp[f'{tag}_pen_corner_top'] = (gx - sx * PENALTY_AREA_DEPTH, PENALTY_AREA_HALF_WIDTH)
            kp[f'{tag}_pen_corner_bottom'] = (gx - sx * PENALTY_AREA_DEPTH, -PENALTY_AREA_HALF_WIDTH)
            # Penalty arc "D" junctions on the penalty-area line.
            kp[f'{tag}_pen_arc_top'] = (gx - sx * PENALTY_AREA_DEPTH, _PEN_ARC_Y)
            kp[f'{tag}_pen_arc_bottom'] = (gx - sx * PENALTY_AREA_DEPTH, -_PEN_ARC_Y)
            # Goal area.
            kp[f'{tag}_goalarea_goalline_top'] = (gx, GOAL_AREA_HALF_WIDTH)
            kp[f'{tag}_goalarea_goalline_bottom'] = (gx, -GOAL_AREA_HALF_WIDTH)
            kp[f'{tag}_goalarea_corner_top'] = (gx - sx * GOAL_AREA_DEPTH, GOAL_AREA_HALF_WIDTH)
            kp[f'{tag}_goalarea_corner_bottom'] = (gx - sx * GOAL_AREA_DEPTH, -GOAL_AREA_HALF_WIDTH)
            # Penalty spot + goal posts.
            kp[f'{tag}_penalty_spot'] = (gx - sx * PENALTY_SPOT_DIST, 0.0)
            kp[f'{tag}_goalpost_top'] = (gx, GOAL_HALF_WIDTH)
            kp[f'{tag}_goalpost_bottom'] = (gx, -GOAL_HALF_WIDTH)

        return OrderedDict((k, np.array([x, y, 0.0], dtype=np.float64)) for k, (x, y) in kp.items())

    def keypoint_array(self) -> tuple[list[str], np.ndarray]:
        '''(names, (K,3) world array) in canonical order.'''
        kp = self.keypoints()
        return list(kp.keys()), np.stack(list(kp.values()), axis=0)

    def roboflow32_keypoints(self) -> "OrderedDict[str, np.ndarray]":
        '''
        The 32 pitch vertices in the exact order of roboflow/sports'
        ``SoccerPitchConfiguration`` — the layout the ``yolo32`` landmark model
        outputs. Names ``rf0``..``rf31``. Built with regulation markings (the box
        sizes are law-fixed even on amateur pitches; only length/width vary) in this
        model's centred world frame, so calibration is metrically correct for the
        user's measured pitch.
        '''
        L, W = self.length, self.width
        pa_len, pa_hw = PENALTY_AREA_DEPTH, PENALTY_AREA_HALF_WIDTH
        ga_len, ga_hw = GOAL_AREA_DEPTH, GOAL_AREA_HALF_WIDTH
        cr, spot = CENTER_CIRCLE_R, PENALTY_SPOT_DIST
        h = W / 2.0
        # roboflow corner-origin frame (x along length 0..L, y along width 0..W).
        verts = [
            (0, 0), (0, h - pa_hw), (0, h - ga_hw), (0, h + ga_hw), (0, h + pa_hw), (0, W),
            (ga_len, h - ga_hw), (ga_len, h + ga_hw), (spot, h),
            (pa_len, h - pa_hw), (pa_len, h - ga_hw), (pa_len, h + ga_hw), (pa_len, h + pa_hw),
            (L / 2, 0), (L / 2, h - cr), (L / 2, h + cr), (L / 2, W),
            (L - pa_len, h - pa_hw), (L - pa_len, h - ga_hw), (L - pa_len, h + ga_hw),
            (L - pa_len, h + pa_hw), (L - spot, h),
            (L - ga_len, h - ga_hw), (L - ga_len, h + ga_hw),
            (L, 0), (L, h - pa_hw), (L, h - ga_hw), (L, h + ga_hw), (L, h + pa_hw), (L, W),
            (L / 2 - cr, h), (L / 2 + cr, h),
        ]
        # → centred world frame (origin at pitch centre).
        return OrderedDict(
            (f'rf{i}', np.array([x - L / 2.0, y - W / 2.0, 0.0], dtype=np.float64))
            for i, (x, y) in enumerate(verts))

    def roboflow32_array(self) -> tuple[list[str], np.ndarray]:
        kp = self.roboflow32_keypoints()
        return list(kp.keys()), np.stack(list(kp.values()), axis=0)

    def line_segments(self, arc_steps: int = 24) -> list[np.ndarray]:
        '''
        Pitch outline + markings as a list of world polylines ((M,3) each), for
        wireframe reprojection overlays. Circles/arcs are sampled to polylines.
        '''
        hl, hw = self.half_l, self.half_w
        segs: list[np.ndarray] = []

        def seg(p0, p1):
            segs.append(np.array([[*p0, 0.0], [*p1, 0.0]], dtype=np.float64))

        # Touchlines, goal lines, halfway line.
        seg((-hl, hw), (hl, hw))
        seg((-hl, -hw), (hl, -hw))
        seg((-hl, -hw), (-hl, hw))
        seg((hl, -hw), (hl, hw))
        seg((0.0, -hw), (0.0, hw))

        # Centre circle.
        th = np.linspace(0, 2 * np.pi, arc_steps * 2)
        circle = np.stack([CENTER_CIRCLE_R * np.cos(th), CENTER_CIRCLE_R * np.sin(th),
                           np.zeros_like(th)], axis=1)
        segs.append(circle)

        for sx in (-1.0, 1.0):
            gx = sx * hl
            paw, pad = PENALTY_AREA_HALF_WIDTH, PENALTY_AREA_DEPTH
            gaw, gad = GOAL_AREA_HALF_WIDTH, GOAL_AREA_DEPTH
            # Penalty area (3 sides; the 4th is the goal line).
            seg((gx, paw), (gx - sx * pad, paw))
            seg((gx, -paw), (gx - sx * pad, -paw))
            seg((gx - sx * pad, -paw), (gx - sx * pad, paw))
            # Goal area.
            seg((gx, gaw), (gx - sx * gad, gaw))
            seg((gx, -gaw), (gx - sx * gad, -gaw))
            seg((gx - sx * gad, -gaw), (gx - sx * gad, gaw))
            # Penalty arc (only the part outside the penalty area).
            spot_x = gx - sx * PENALTY_SPOT_DIST
            a0 = np.arctan2(_PEN_ARC_Y, (gx - sx * pad) - spot_x)
            phi = np.linspace(-a0, a0, arc_steps)
            arc = np.stack([spot_x + CENTER_CIRCLE_R * np.cos(phi) * sx,
                            CENTER_CIRCLE_R * np.sin(phi),
                            np.zeros_like(phi)], axis=1)
            segs.append(arc)

        return segs
