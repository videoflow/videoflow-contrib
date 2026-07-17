'''
Drawing helpers for offside visualizations (pure cv2 + numpy).

Projects the world-space offside line and player markers into a camera view using
that camera's calibration, and renders a top-down pitch diagram. Self-contained
(no videoflow / no other component) so the visualizer packages independently.
'''
from __future__ import annotations

import cv2
import numpy as np

TEAM_COLORS = {0: (60, 60, 220), 1: (220, 140, 40), 2: (40, 200, 40), 3: (30, 220, 220), -1: (150, 150, 150)}


def project_world_points(pts_world: np.ndarray, calib: dict) -> np.ndarray:
    '''World (N,3) → pixel (N,2), applying radial-tangential distortion.'''
    K = np.asarray(calib['K'], dtype=np.float64)
    R = np.asarray(calib['R'], dtype=np.float64)
    t = np.asarray(calib['t'], dtype=np.float64)
    dist = np.asarray(calib.get('dist', [0, 0, 0, 0, 0]), dtype=np.float64)
    X = np.asarray(pts_world, dtype=np.float64).reshape(-1, 3)
    cam = X @ R.T + t
    z = cam[:, 2]
    with np.errstate(divide='ignore', invalid='ignore'):
        xy = cam[:, :2] / z[:, None]
    k1, k2, p1, p2, k3 = (list(dist) + [0] * 5)[:5]
    r2 = (xy ** 2).sum(axis=1)
    radial = 1 + k1 * r2 + k2 * r2 ** 2 + k3 * r2 ** 3
    xd = xy[:, 0] * radial + 2 * p1 * xy[:, 0] * xy[:, 1] + p2 * (r2 + 2 * xy[:, 0] ** 2)
    yd = xy[:, 1] * radial + p1 * (r2 + 2 * xy[:, 1] ** 2) + 2 * p2 * xy[:, 0] * xy[:, 1]
    px = np.stack([xd, yd, np.ones_like(xd)], axis=1) @ K.T
    out = px[:, :2]
    out[z <= 0] = np.nan
    return out


def draw_offside_line(img: np.ndarray, line_x: float, pitch_width: float, calib: dict,
                      color=(0, 0, 255), thickness=3) -> np.ndarray:
    '''Project the world segment x=line_x, y∈[-W/2,W/2], z=0 and draw it as a polyline.'''
    ys = np.linspace(-pitch_width / 2.0, pitch_width / 2.0, 40)
    seg = np.stack([np.full_like(ys, line_x), ys, np.zeros_like(ys)], axis=1)
    px = project_world_points(seg, calib)
    pts = px[np.all(np.isfinite(px), axis=1)].astype(np.int32)
    if len(pts) >= 2:
        cv2.polylines(img, [pts.reshape(-1, 1, 2)], False, color, thickness, cv2.LINE_AA)
    return img


def draw_player_marker(img: np.ndarray, ground_xy, calib: dict, color, label=None, radius=6):
    p = project_world_points(np.array([[ground_xy[0], ground_xy[1], 0.0]]), calib)[0]
    if not np.all(np.isfinite(p)):
        return img
    c = (int(p[0]), int(p[1]))
    cv2.circle(img, c, radius, color, -1, cv2.LINE_AA)
    cv2.circle(img, c, radius + 2, (255, 255, 255), 1, cv2.LINE_AA)
    if label:
        cv2.putText(img, label, (c[0] + 8, c[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1, cv2.LINE_AA)
    return img


def draw_banner(img: np.ndarray, verdict: str, margin_m, uncertainty_m) -> np.ndarray:
    color = {'OFFSIDE': (0, 0, 220), 'ONSIDE': (0, 180, 0),
             'TOO_CLOSE': (0, 180, 220), 'INCONCLUSIVE': (150, 150, 150)}.get(verdict, (150, 150, 150))
    h, w = img.shape[:2]
    cv2.rectangle(img, (0, 0), (w, 46), (30, 30, 30), -1)
    txt = verdict
    if margin_m is not None:
        # cv2's Hershey font is ASCII-only — use +/- rather than the ± glyph.
        txt += f'   margin {margin_m * 100:+.0f} cm  (+/-{uncertainty_m * 100:.0f} cm)'
    cv2.putText(img, txt, (14, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)
    return img


def draw_pitch_topdown(pitch_length: float, pitch_width: float, players: list, line_x=None,
                       attack_sign=1, scale=8, margin=40) -> np.ndarray:
    '''Top-down pitch diagram: green field, players as team-colored dots, offside line.'''
    W = int(pitch_length * scale) + 2 * margin
    H = int(pitch_width * scale) + 2 * margin
    img = np.full((H, W, 3), (40, 120, 40), dtype=np.uint8)

    def to_px(x, y):
        return (int(margin + (x + pitch_length / 2) * scale),
                int(margin + (y + pitch_width / 2) * scale))

    white = (235, 235, 235)
    cv2.rectangle(img, to_px(-pitch_length / 2, -pitch_width / 2),
                  to_px(pitch_length / 2, pitch_width / 2), white, 2)
    cv2.line(img, to_px(0, -pitch_width / 2), to_px(0, pitch_width / 2), white, 1)
    cv2.circle(img, to_px(0, 0), int(9.15 * scale), white, 1)
    for p in players:
        gx, gy = p['pos'] if 'pos' in p else p['ground']
        color = TEAM_COLORS.get(p.get('team', -1), (150, 150, 150))
        c = to_px(gx, gy)
        cv2.circle(img, c, 6, color, -1, cv2.LINE_AA)
        if p.get('is_gk'):
            cv2.circle(img, c, 8, (0, 220, 220), 1, cv2.LINE_AA)
        cv2.putText(img, str(p.get('gid', '')), (c[0] + 6, c[1] - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, white, 1, cv2.LINE_AA)
    if line_x is not None:
        cv2.line(img, to_px(line_x, -pitch_width / 2), to_px(line_x, pitch_width / 2),
                 (0, 0, 235), 2)
        arrow_y = -pitch_width / 2 + 3
        cv2.arrowedLine(img, to_px(line_x, arrow_y), to_px(line_x + attack_sign * 6, arrow_y),
                        (0, 0, 235), 2, tipLength=0.4)
    return img
