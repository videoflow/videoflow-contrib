'''
Backward re-association — follow the receiver from the touch instant back to the
kick instant across tracker ID switches.

The receiver is identified by its global id at the moment of the touch. To judge
offside we need that same person's identity at the (earlier) kick instant. If the
global id is continuous, this is trivial; when it breaks (occlusion, a cluster of
players), we predict the position one step back and adopt the nearest same-team
candidate within a gate, tolerating a bounded run of pure predictions.

Pure numpy; operates on the engine's ordered world-state buffer slice.
'''
from __future__ import annotations

import numpy as np


def _get(state: dict, gid: int):
    for p in state['players']:
        if p['gid'] == gid:
            return p
    return None


def backtrack_receiver(states: list[dict], receiver_gid: int, team=None,
                       gate_m: float = 0.6, max_pred_steps: int = 8,
                       ambiguous_m: float = 0.4) -> dict:
    '''
    - Arguments:
        - states: world-state buffer slice ordered by ascending time, from the kick
          instant (index 0) to the touch instant (last index).
        - receiver_gid: the receiver's global id at the touch instant (last state).
        - team: receiver's team id, used to gate candidate adoption on an id break.
    - Returns: ``{found, gid_at_kick, pos_at_kick, positions{t:pos}, flags, reason}``.
    '''
    n = len(states)
    if n == 0:
        return _lost('empty_buffer', receiver_gid)
    last = _get(states[-1], receiver_gid)
    if last is None:
        return _lost('receiver_absent_at_touch', receiver_gid)

    cur_gid = receiver_gid
    last_pos = np.asarray(last['ground'], dtype=np.float64)
    prev_pos = last_pos.copy()                        # for velocity estimation
    positions = {states[-1]['t']: last_pos.tolist()}
    flags: list[str] = []
    pred_streak = 0

    for i in range(n - 2, -1, -1):
        state = states[i]
        p = _get(state, cur_gid)
        if p is not None:
            pos = np.asarray(p['ground'], dtype=np.float64)
            positions[state['t']] = pos.tolist()
            prev_pos, last_pos = last_pos, pos
            pred_streak = 0
            continue

        # ID break — predict one step back using the recent forward velocity.
        fwd_vel = last_pos - prev_pos
        predicted = last_pos - fwd_vel
        cands = []
        for q in state['players']:
            if q['gid'] == cur_gid:
                continue
            if team is not None and q.get('team') is not None and q['team'] != team:
                continue
            if q.get('is_ref'):
                continue
            d = float(np.linalg.norm(np.asarray(q['ground'], dtype=np.float64) - predicted))
            if d <= gate_m:
                cands.append((d, q['gid'], np.asarray(q['ground'], dtype=np.float64)))
        cands.sort(key=lambda e: e[0])
        if cands:
            if len(cands) > 1 and (cands[1][0] - cands[0][0]) <= ambiguous_m:
                flags.append('ambiguous_track')
            _, new_gid, new_pos = cands[0]
            if new_gid != cur_gid:
                flags.append('id_switch')
            cur_gid = new_gid
            positions[state['t']] = new_pos.tolist()
            prev_pos, last_pos = last_pos, new_pos
            pred_streak = 0
        else:
            pred_streak += 1
            if pred_streak > max_pred_steps:
                return _lost('track_lost', receiver_gid, flags=flags)
            positions[state['t']] = predicted.tolist()
            prev_pos, last_pos = last_pos, predicted
            if 'predicted_gap' not in flags:
                flags.append('predicted_gap')

    return {
        'found': True,
        'gid_at_kick': int(cur_gid),
        'pos_at_kick': positions[states[0]['t']],
        'positions': positions,
        'flags': sorted(set(flags)),
        'reason': None,
    }


def _lost(reason: str, receiver_gid: int, flags=None) -> dict:
    return {
        'found': False,
        'gid_at_kick': None,
        'pos_at_kick': None,
        'positions': {},
        'flags': sorted(set(flags or [])),
        'reason': reason,
    }
