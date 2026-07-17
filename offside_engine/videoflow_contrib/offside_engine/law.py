'''
Offside-law geometry (Law 11) — pure functions over a reconstructed world state.

A player is in an offside *position* if any part of the head, body or feet (NOT
the arms/hands — the boundary is the bottom of the armpit ≈ the shoulder) is
nearer to the opponents' goal line than BOTH the ball AND the second-last
opponent, while in the opponents' half.

Coordinates: world X runs goal-to-goal (see PitchModel). ``attack_sign`` s ∈ {+1,-1}
is the direction the attacking team attacks, so "forward" (toward the opponents'
goal) is increasing ``s·X``. All inputs are metres; keypoints are the canonical
26-point (halpe26) layout unless ``kpt_format='coco17'``.
'''
from __future__ import annotations

import numpy as np

# Canonical halpe26 indices excluded from offside (elbows + wrists).
_HALPE26_ARMS = {7, 8, 9, 10}
_HALPE26_ALL = set(range(26))
# coco17: same arm exclusion; no feet keypoints exist (ankles are the lowest).
_COCO17_ARMS = {7, 8, 9, 10}
_COCO17_ALL = set(range(17))

LEGAL_KEYPOINTS = {
    'halpe26': sorted(_HALPE26_ALL - _HALPE26_ARMS),
    'wholebody133': sorted(_HALPE26_ALL - _HALPE26_ARMS),   # remapped to 26 upstream
    'coco17': sorted(_COCO17_ALL - _COCO17_ARMS),
}

VERDICT_OFFSIDE = 'OFFSIDE'
VERDICT_ONSIDE = 'ONSIDE'
VERDICT_TOO_CLOSE = 'TOO_CLOSE'
VERDICT_INCONCLUSIVE = 'INCONCLUSIVE'


def player_extreme(player: dict, attack_sign: float, legal_idx: list[int],
                   kpt_min_conf: float = 0.3) -> tuple[float, float, float]:
    '''
    Forward-most legal point of a player.

    - Returns: ``(s·X_max, world_X_at_max, conf_at_max)``. Falls back to the ground
      point (with a low nominal confidence) when no legal keypoint is confident.
    '''
    kpts = player.get('kpts3d')
    s = attack_sign
    best_sx, best_x, best_conf = -np.inf, None, 0.0
    if kpts is not None:
        arr = np.asarray(kpts, dtype=np.float64)
        for j in legal_idx:
            if j >= arr.shape[0]:
                continue
            X, conf = arr[j, 0], arr[j, 3]
            if conf < kpt_min_conf or not np.isfinite(X):
                continue
            sx = s * X
            if sx > best_sx:
                best_sx, best_x, best_conf = sx, X, float(conf)
    if best_x is None:                              # no confident legal keypoint
        gx = float(player['ground'][0])
        return s * gx, gx, 0.0
    return best_sx, best_x, best_conf


def _uncertainty(conf: float) -> float:
    '''Per-player longitudinal uncertainty (m): 5 cm floor, grows as confidence drops.'''
    return 0.05 + 0.10 * (1.0 - float(np.clip(conf, 0.0, 1.0)))


def compute_offside(players: list[dict], receiver_gid: int, attack_team: int,
                    attack_sign: float, ball_xyz, kpt_format: str = 'halpe26',
                    kpt_min_conf: float = 0.3, margin_tolerance_m: float = 0.15) -> dict:
    '''
    Evaluate offside for ``receiver_gid`` at the (already reconstructed) moment.

    - Arguments:
        - players: world-state players (dicts with gid, team, is_gk, is_ref, ground, kpts3d).
        - receiver_gid: the attacker being judged (the teammate who receives the ball).
        - attack_team: team id of the attacking side.
        - attack_sign: +1/-1, direction the attacking team attacks.
        - ball_xyz: (X,Y,Z) or None (ball position at the kick — used as the 2nd line).
    - Returns: a verdict dict (see the plan's schema). Never raises on bad geometry;
      returns INCONCLUSIVE with a ``reason`` instead.
    '''
    legal = LEGAL_KEYPOINTS.get(kpt_format, LEGAL_KEYPOINTS['halpe26'])
    by_gid = {p['gid']: p for p in players}
    if receiver_gid not in by_gid:
        return _inconclusive('receiver_missing', receiver_gid, attack_team, attack_sign)
    receiver = by_gid[receiver_gid]

    # Defenders = opponents, excluding referees and provisional/unknown-team players.
    defenders = [p for p in players
                 if p.get('team') is not None and p['team'] != attack_team
                 and not p.get('is_ref', False) and not p.get('provisional', False)]
    if len(defenders) < 2:
        return _inconclusive('defenders_missing', receiver_gid, attack_team, attack_sign)

    recv_sx, recv_x, recv_conf = player_extreme(receiver, attack_sign, legal, kpt_min_conf)
    sigma_recv = _uncertainty(recv_conf)

    # Receiver must be in the opponents' half (s·X_ground > 0) to be offside.
    recv_ground_sx = attack_sign * float(receiver['ground'][0])
    if recv_ground_sx <= 0:
        return _verdict(VERDICT_ONSIDE, margin=-abs(recv_ground_sx), uncertainty=sigma_recv,
                        receiver_gid=receiver_gid, attack_team=attack_team, attack_sign=attack_sign,
                        receiver_extreme_x=recv_x, reason='own_half')

    # Second-last opponent line: the 2nd largest defender extreme (s·X); last is usually the GK.
    dext = []
    for d in defenders:
        sx, x, conf = player_extreme(d, attack_sign, legal, kpt_min_conf)
        dext.append((sx, x, conf, d['gid']))
    dext.sort(key=lambda e: e[0], reverse=True)     # deepest (largest s·X) first
    second = dext[1]
    def_sx, def_x, def_conf, def_gid = second
    sigma_def = _uncertainty(def_conf)

    # Ball line (in s·X). Missing ball → use only the defender line (flagged).
    flags: list[str] = []
    if ball_xyz is not None and np.all(np.isfinite(ball_xyz)):
        ball_sx = attack_sign * float(ball_xyz[0])
        ball_x = float(ball_xyz[0])
    else:
        ball_sx = -np.inf
        ball_x = None
        flags.append('no_ball_reference')

    # Offside iff nearer to goal than BOTH ball and 2nd-last opponent → beyond the max.
    line_sx = max(def_sx, ball_sx)
    margin = recv_sx - line_sx                       # >0 means beyond both → offside
    tol = max(margin_tolerance_m, float(np.hypot(sigma_recv, sigma_def)))

    if margin > tol:
        verdict = VERDICT_OFFSIDE
    elif margin < -tol:
        verdict = VERDICT_ONSIDE
    else:
        verdict = VERDICT_TOO_CLOSE

    line_x = def_x if def_sx >= ball_sx else ball_x  # world X of the binding line
    return {
        'verdict': verdict,
        'reason': None,
        'margin_m': float(margin),
        'uncertainty_m': float(tol),
        'receiver_gid': receiver_gid,
        'team': attack_team,
        'attack_sign': int(attack_sign),
        'offside_line_x': float(line_x) if line_x is not None else None,
        'ball_x': ball_x,
        'receiver_extreme_x': float(recv_x),
        'second_defender_gid': int(def_gid),
        'flags': flags,
    }


def _verdict(verdict, margin, uncertainty, receiver_gid, attack_team, attack_sign,
            receiver_extreme_x=None, reason=None):
    return {
        'verdict': verdict, 'reason': reason,
        'margin_m': float(margin), 'uncertainty_m': float(uncertainty),
        'receiver_gid': receiver_gid, 'team': attack_team, 'attack_sign': int(attack_sign),
        'offside_line_x': None, 'ball_x': None,
        'receiver_extreme_x': None if receiver_extreme_x is None else float(receiver_extreme_x),
        'second_defender_gid': None, 'flags': [],
    }


def _inconclusive(reason, receiver_gid, attack_team, attack_sign):
    return {
        'verdict': VERDICT_INCONCLUSIVE, 'reason': reason,
        'margin_m': None, 'uncertainty_m': 0.0,
        'receiver_gid': receiver_gid, 'team': attack_team, 'attack_sign': int(attack_sign),
        'offside_line_x': None, 'ball_x': None, 'receiver_extreme_x': None,
        'second_defender_gid': None, 'flags': [],
    }
