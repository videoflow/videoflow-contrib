'''
Offside engine — the stateful node that turns a stream of reconstructed world
states into offside verdicts.

It keeps a rolling buffer of world states, detects ball touches (change-points in
the 3D trajectory), runs a kick→next-touch phase machine, and when a teammate
receives the ball it backward-associates the receiver to the kick instant and
evaluates Law 11. Emits ``None`` on ordinary instants, a ``touch_event`` dict on
each committed touch, and an ``offside_verdict`` dict when a kick→teammate-touch
pair resolves.

Single-task (OneTaskProcessorNode) because the buffer/phase state is sequential.
'''
from __future__ import annotations

from collections import deque

import numpy as np
from videoflow.core.node import OneTaskProcessorNode

from . import events as ev
from . import law
from .backtrack import backtrack_receiver


class OffsideEngine(OneTaskProcessorNode):
    '''
    - Arguments (see the plan for the full list): pitch dims, attack_direction
      ('auto' or a ``{team_id: +1|-1}`` map), touch thresholds, buffer length,
      margin tolerance, keypoint format, and the detection lag (frames of future
      context the change-point detector needs before committing a touch).
    '''
    def __init__(self, pitch_length: float = 105.0, pitch_width: float = 68.0,
                 attack_direction='auto', touch_radius_m: float = 1.1,
                 vel_jump_ms: float = 3.0, dir_change_deg: float = 25.0,
                 min_gap_frames: int = 5, buffer_s: float = 15.0,
                 margin_tolerance_m: float = 0.15, kpt_format: str = 'halpe26',
                 detection_lag_frames: int = 8, ref_fps: float = 30.0, **kwargs) -> None:
        self._pitch_length = float(pitch_length)
        self._pitch_width = float(pitch_width)
        self._attack_direction = attack_direction
        self._touch_radius_m = float(touch_radius_m)
        self._vel_jump_ms = float(vel_jump_ms)
        self._dir_change_deg = float(dir_change_deg)
        self._min_gap_frames = int(min_gap_frames)
        self._buffer_s = float(buffer_s)
        self._margin_tolerance_m = float(margin_tolerance_m)
        self._kpt_format = kpt_format
        self._detection_lag_frames = int(detection_lag_frames)
        self._ref_fps = float(ref_fps)
        self._reset_state()
        super().__init__(**kwargs)

    def _reset_state(self):
        self._buffer: deque = deque()
        self._committed_ts: list[float] = []   # touch times already processed
        self._phase = None                     # {'kicker','team','t0'}
        self._verdict_count = 0

    def open(self):
        self._reset_state()

    # -- main entry -----------------------------------------------------------
    def process(self, world_state, ctx=None):
        '''world_state: a fused world-state dict (or None at a quorum gap).'''
        if world_state is None:
            return None
        self._buffer.append(world_state)
        self._trim_buffer()

        touch = self._detect_next_committable_touch()
        if touch is None:
            return None
        return self._advance_phase(touch)

    def _trim_buffer(self):
        if not self._buffer:
            return
        t_end = self._buffer[-1]['t']
        while self._buffer and (t_end - self._buffer[0]['t']) > self._buffer_s:
            self._buffer.popleft()

    # -- touch detection ------------------------------------------------------
    def _ball_series(self):
        times, pos, valid = [], [], []
        for st in self._buffer:
            times.append(st['t'])
            b = st.get('ball')
            if b is not None and b.get('p') is not None and not b.get('predicted', False):
                times[-1] = st['t']
                pos.append(b['p'])
                valid.append(True)
            else:
                pos.append([np.nan, np.nan, np.nan])
                valid.append(False)
        return np.array(times), np.array(pos, dtype=np.float64), np.array(valid, dtype=bool)

    def _detect_next_committable_touch(self):
        '''Return the earliest not-yet-committed, stable (past the lag) touch, or None.'''
        n = len(self._buffer)
        if n < 2 * self._detection_lag_frames:
            return None
        times, pos, valid = self._ball_series()
        _, vel, speed, accel, usable = ev.smooth_ball(times, pos, valid)
        cands = ev.touch_candidates(times, vel, speed, accel, usable,
                                    vel_jump_ms=self._vel_jump_ms,
                                    dir_change_deg=self._dir_change_deg,
                                    min_gap_frames=self._min_gap_frames)
        last_idx = n - 1
        frame_period = 1.0 / self._ref_fps
        for c in cands:
            if (last_idx - c) < self._detection_lag_frames:
                continue                       # not enough future context yet
            t0 = ev.refine_touch_time(times, accel, c)
            if any(abs(t0 - tc) < 0.5 * frame_period for tc in self._committed_ts):
                continue                       # already handled
            state = self._buffer[c]
            attr = ev.attribute_touch(state['players'], state.get('ball', {}).get('p')
                                      if state.get('ball') else None,
                                      touch_radius_m=self._touch_radius_m,
                                      kpt_format=self._kpt_format)
            self._committed_ts.append(t0)
            return {'t0': t0, 'frame_idx': c, 'attr': attr, 'state': state}
        return None

    # -- phase machine --------------------------------------------------------
    def _advance_phase(self, touch):
        attr = touch['attr']
        gid = attr.get('gid')
        team = attr.get('team')
        t0 = touch['t0']
        base_event = {'type': 'touch_event', 't': t0, 'toucher_gid': gid, 'team': team,
                      'contested': attr.get('contested', False)}
        if gid is None or team is None:
            return base_event                  # unattributed — logged, no phase change

        if self._phase is None:
            self._phase = {'kicker': gid, 'team': team, 't0': t0}
            return base_event

        kicker, kteam, kt0 = self._phase['kicker'], self._phase['team'], self._phase['t0']
        if gid == kicker:                      # same player again → dribble; re-anchor
            self._phase['t0'] = t0
            return base_event
        if team == kteam:                      # teammate received → evaluate offside
            verdict = self._evaluate(kicker, gid, kteam, kt0, t0, attr)
            self._phase = {'kicker': gid, 'team': team, 't0': t0}
            return verdict
        # opponent touch → possession change
        self._phase = {'kicker': gid, 'team': team, 't0': t0}
        return base_event

    # -- offside evaluation ---------------------------------------------------
    def _evaluate(self, kicker_gid, receiver_gid, team, t_kick, t_touch, attr):
        kick_state, kick_idx = self._state_nearest(t_kick)
        touch_idx = self._index_nearest(t_touch)
        if kick_state is None or abs(kick_state['t'] - t_kick) > 0.1:
            return self._verdict_dict(law._inconclusive('kick_state_missing', receiver_gid, team,
                                                        self._attack_sign(team, None)),
                                      t_kick, t_touch, kicker_gid, [])

        # Backward-associate the receiver from the touch instant to the kick instant.
        states_slice = [self._buffer[i] for i in range(kick_idx, touch_idx + 1)]
        bt = backtrack_receiver(states_slice, receiver_gid, team=team)
        flags = list(bt.get('flags', []))
        if not bt['found']:
            return self._verdict_dict(law._inconclusive('track_lost', receiver_gid, team,
                                                        self._attack_sign(team, kick_state)),
                                      t_kick, t_touch, kicker_gid, flags)
        gid_at_kick = bt['gid_at_kick']

        sign = self._attack_sign(team, kick_state)
        ball_at_kick = kick_state.get('ball')
        ball_xyz = ball_at_kick['p'] if ball_at_kick and ball_at_kick.get('p') is not None else None
        res = law.compute_offside(kick_state['players'], gid_at_kick, team, sign, ball_xyz,
                                  kpt_format=self._kpt_format,
                                  margin_tolerance_m=self._margin_tolerance_m)
        if attr.get('contested'):
            flags.append('contested_touch')
        return self._verdict_dict(res, t_kick, t_touch, kicker_gid, flags + res.get('flags', []))

    def _verdict_dict(self, res, t_kick, t_touch, kicker_gid, flags):
        self._verdict_count += 1
        players_t0 = []
        kick_state, _ = self._state_nearest(t_kick)
        if kick_state is not None:
            sign = res.get('attack_sign', 1)
            legal = law.LEGAL_KEYPOINTS.get(self._kpt_format, law.LEGAL_KEYPOINTS['halpe26'])
            for p in kick_state['players']:
                sx, x, _ = law.player_extreme(p, sign, legal)
                players_t0.append({'gid': p['gid'], 'team': p.get('team'),
                                   'is_gk': p.get('is_gk', False), 'pos': list(p['ground']),
                                   'extreme_x': float(x)})
        out = {'type': 'offside_verdict', 'index': self._verdict_count,
               't_kick': float(t_kick), 't_touch': float(t_touch),
               'kicker_gid': int(kicker_gid), 'players_t0': players_t0,
               'flags': sorted(set(flags))}
        out.update(res)
        return out

    def _attack_sign(self, team, state):
        '''+1/-1 direction ``team`` attacks. Config override wins; else infer from geometry.'''
        if isinstance(self._attack_direction, dict):
            v = self._attack_direction.get(team, self._attack_direction.get(str(team)))
            if v is not None:
                if isinstance(v, str):
                    return 1.0 if v.strip().lstrip('+').lower().startswith('x') or v.strip() == '+x' else -1.0
                return 1.0 if v >= 0 else -1.0
        if state is None:
            state = self._buffer[-1] if self._buffer else None
        if state is None:
            return 1.0
        # Opponents' goalkeeper defends the goal ``team`` attacks.
        opp_gk_x = [p['ground'][0] for p in state['players']
                    if p.get('team') is not None and p['team'] != team and p.get('is_gk')]
        if opp_gk_x:
            return 1.0 if float(np.mean(opp_gk_x)) >= 0 else -1.0
        # Fallback: opponents' deepest players.
        opp_x = [p['ground'][0] for p in state['players']
                 if p.get('team') is not None and p['team'] != team and not p.get('is_ref')]
        if opp_x:
            return 1.0 if float(np.mean(opp_x)) >= 0 else -1.0
        return 1.0

    def _state_nearest(self, t):
        best, best_i, best_d = None, None, np.inf
        for i, st in enumerate(self._buffer):
            d = abs(st['t'] - t)
            if d < best_d:
                best, best_i, best_d = st, i, d
        return best, best_i

    def _index_nearest(self, t):
        _, i = self._state_nearest(t)
        return i if i is not None else len(self._buffer) - 1
