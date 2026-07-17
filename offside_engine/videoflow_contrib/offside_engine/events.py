'''
Ball-event detection — the video-only stand-in for FIFA's in-ball IMU.

Touches/kicks are change-points in the smoothed 3D ball trajectory: a sudden jump
in speed or a sharp change of velocity direction, coincident with an acceleration
peak. Each candidate is attributed to the nearest player limb in 3D. Offline lets
us smooth non-causally (centred Savitzky–Golay), which localizes change-points
better than a causal filter.

All pure numpy; operates on time-ordered arrays the engine extracts from its
world-state buffer.
'''
from __future__ import annotations

import math

import numpy as np

# Canonical halpe26 limbs that can "play" the ball (feet/legs/head/body — not hands).
_TOUCH_KPTS_HALPE26 = [0, 5, 6, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]
_TOUCH_KPTS_COCO17 = [0, 5, 6, 11, 12, 13, 14, 15, 16]


def _savgol_1d(y: np.ndarray, window: int, poly: int, deriv: int = 0, dt: float = 1.0) -> np.ndarray:
    '''Small self-contained Savitzky–Golay (avoids a hard scipy dep in this pure module).'''
    n = len(y)
    window = min(window, n if n % 2 == 1 else n - 1)
    if window < poly + 2 or window < 3:
        # too short to smooth; fall back to finite differences for derivatives
        if deriv == 0:
            return y.astype(np.float64).copy()
        g = np.gradient(y.astype(np.float64), dt)
        return g if deriv == 1 else np.gradient(g, dt)
    if window % 2 == 0:
        window -= 1
    half = window // 2
    # Design matrix of powers, least-squares fit in each sliding window.
    A = np.vander(np.arange(-half, half + 1), poly + 1, increasing=True)
    ATA_inv = np.linalg.pinv(A.T @ A)
    coeffs = ATA_inv @ A.T                      # (poly+1, window)
    c = coeffs[deriv] * (math.factorial(deriv) / (dt ** deriv))
    ypad = np.pad(y.astype(np.float64), half, mode='edge')
    out = np.convolve(ypad, c[::-1], mode='valid')
    return out[:n]


def smooth_ball(times: np.ndarray, pos: np.ndarray, valid: np.ndarray,
                window: int = 9, poly: int = 2, max_gap: int = 3):
    '''
    Smooth the 3D ball trajectory and produce velocity/speed/acceleration.

    - Arguments:
        - times: (N,) event times (s).
        - pos: (N,3) ball positions (NaN where invalid).
        - valid: (N,) bool, True where a real (non-predicted) triangulated ball exists.
        - max_gap: linearly bridge invalid runs up to this many frames.
    - Returns: (pos_s (N,3), vel (N,3), speed (N,), accel (N,), usable (N,) bool).
    '''
    times = np.asarray(times, dtype=np.float64)
    pos = np.asarray(pos, dtype=np.float64).copy()
    valid = np.asarray(valid, dtype=bool).copy()
    n = len(times)
    if n == 0:
        z = np.zeros((0, 3))
        return z, z, np.zeros(0), np.zeros(0), np.zeros(0, dtype=bool)

    # Bridge short gaps by interpolation; usable stays True unless a gap is too long.
    idx = np.arange(n)
    usable = np.ones(n, dtype=bool)
    for axis in range(3):
        good = valid & np.isfinite(pos[:, axis])
        if good.sum() >= 2:
            pos[:, axis] = np.interp(idx, idx[good], pos[good, axis])
    # Mark long gaps as unusable (keep interpolation for continuity but flag).
    run_start = None
    for k in range(n):
        if not valid[k]:
            if run_start is None:
                run_start = k
        else:
            if run_start is not None and (k - run_start) > max_gap:
                usable[run_start:k] = False
            run_start = None
    if run_start is not None and (n - run_start) > max_gap:
        usable[run_start:n] = False

    dt = float(np.median(np.diff(times))) if n > 1 else 1.0
    if dt <= 0:
        dt = 1.0
    pos_s = np.stack([_savgol_1d(pos[:, a], window, poly, 0, dt) for a in range(3)], axis=1)
    vel = np.stack([_savgol_1d(pos[:, a], window, poly, 1, dt) for a in range(3)], axis=1)
    speed = np.linalg.norm(vel, axis=1)
    acc = np.stack([_savgol_1d(pos[:, a], window, poly, 2, dt) for a in range(3)], axis=1)
    accel = np.linalg.norm(acc, axis=1)
    return pos_s, vel, speed, accel, usable


def touch_candidates(times: np.ndarray, vel: np.ndarray, speed: np.ndarray, accel: np.ndarray,
                     usable: np.ndarray, vel_jump_ms: float = 3.0, dir_change_deg: float = 25.0,
                     min_gap_frames: int = 5, min_speed: float = 1.5) -> list[int]:
    '''
    Frame indices where the ball trajectory changes abruptly (a touch/kick).

    Criterion at k: (speed jump over ±2 frames > ``vel_jump_ms``) OR (velocity
    direction change > ``dir_change_deg``), with an acceleration local maximum and
    non-maximum suppression over ``min_gap_frames``.
    '''
    n = len(times)
    scores = np.full(n, -np.inf)
    dir_change_rad = np.deg2rad(dir_change_deg)
    for k in range(2, n - 2):
        if not (usable[k - 2] and usable[k + 2]):
            continue
        v0, v1 = vel[k - 2], vel[k + 2]
        s0, s1 = speed[k - 2], speed[k + 2]
        speed_jump = abs(s1 - s0)
        angle = 0.0
        if s0 > min_speed and s1 > min_speed:
            cosang = float(np.clip(np.dot(v0, v1) / (s0 * s1), -1.0, 1.0))
            angle = np.arccos(cosang)
        if speed_jump > vel_jump_ms or angle > dir_change_rad:
            # require accel to be a local max around k
            if accel[k] >= accel[k - 1] and accel[k] >= accel[k + 1]:
                scores[k] = speed_jump + angle * 3.0    # combined salience
    # Non-maximum suppression.
    order = np.argsort(scores)[::-1]
    chosen: list[int] = []
    for k in order:
        if not np.isfinite(scores[k]):
            break
        if all(abs(k - c) >= min_gap_frames for c in chosen):
            chosen.append(int(k))
    return sorted(chosen)


def refine_touch_time(times: np.ndarray, accel: np.ndarray, k: int) -> float:
    '''
    Sub-frame touch time by parabolic interpolation of the acceleration-magnitude
    peak around frame ``k`` (robust for both speed jumps and direction changes,
    where a velocity-line intersection would be ill-conditioned).

    Note: video-only kick timing is fundamentally frame-rate limited (≈ ±1 frame on
    sharp change-points) — this is exactly why FIFA's real system uses a 500 Hz
    in-ball IMU. At 30 fps expect ~½-frame precision here.
    '''
    n = len(times)
    if k <= 0 or k >= n - 1:
        return float(times[k])
    a0, a1, a2 = float(accel[k - 1]), float(accel[k]), float(accel[k + 1])
    denom = a0 - 2.0 * a1 + a2
    dt = float(np.median(np.diff(times))) if n > 1 else 1.0
    if abs(denom) < 1e-12:
        return float(times[k])
    delta = 0.5 * (a0 - a2) / denom              # sub-sample offset in [-1, 1] frames
    delta = float(np.clip(delta, -1.0, 1.0))
    return float(times[k] + delta * dt)


def attribute_touch(players: list[dict], ball_xyz, touch_radius_m: float = 1.1,
                    kpt_min_conf: float = 0.3, kpt_format: str = 'halpe26',
                    contest_margin_m: float = 0.3) -> dict:
    '''
    Attribute a ball contact to the nearest player limb in 3D.

    - Returns: ``{gid, dist_m, contested, second_gid, team}``; ``gid`` is None when
      no player is within ``touch_radius_m``.
    '''
    ball = np.asarray(ball_xyz, dtype=np.float64)
    limbs = _TOUCH_KPTS_COCO17 if kpt_format == 'coco17' else _TOUCH_KPTS_HALPE26
    dists = []
    for p in players:
        if p.get('is_ref'):
            continue
        d = _player_ball_distance(p, ball, limbs, kpt_min_conf)
        if np.isfinite(d):
            dists.append((d, p['gid'], p.get('team')))
    if not dists:
        return {'gid': None, 'dist_m': None, 'contested': False, 'second_gid': None, 'team': None}
    dists.sort(key=lambda e: e[0])
    d0, gid0, team0 = dists[0]
    if d0 > touch_radius_m:
        return {'gid': None, 'dist_m': float(d0), 'contested': False, 'second_gid': None, 'team': None}
    contested = False
    second_gid = None
    if len(dists) > 1:
        d1, gid1, _ = dists[1]
        if d1 <= touch_radius_m and (d1 - d0) <= contest_margin_m:
            contested = True
            second_gid = int(gid1)
    return {'gid': int(gid0), 'dist_m': float(d0), 'contested': contested,
            'second_gid': second_gid, 'team': team0}


def _player_ball_distance(player: dict, ball: np.ndarray, limbs: list[int], cmin: float) -> float:
    kpts = player.get('kpts3d')
    best = np.inf
    if kpts is not None:
        arr = np.asarray(kpts, dtype=np.float64)
        for j in limbs:
            if j >= arr.shape[0] or arr[j, 3] < cmin or not np.isfinite(arr[j, 0]):
                continue
            best = min(best, float(np.linalg.norm(arr[j, :3] - ball)))
    if not np.isfinite(best):                       # fall back to ground point (2D + assume ~1 m torso)
        g = np.asarray(player['ground'], dtype=np.float64)
        best = float(np.hypot(np.linalg.norm(g - ball[:2]), max(0.0, ball[2] - 0.9)))
    return best


def is_ground_bounce(ball_z: float, vz_before: float, vz_after: float,
                     nearest_player_dist: float, touch_radius_m: float) -> bool:
    '''Veto: a low ball whose vertical velocity flips with no player nearby is a bounce.'''
    return (ball_z < 0.25 and vz_before < 0 and vz_after > 0
            and (nearest_player_dist is None or nearest_player_dist > touch_radius_m))
