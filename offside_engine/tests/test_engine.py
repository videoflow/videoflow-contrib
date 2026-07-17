'''
End-to-end engine test on synthetic world states (no models, no flow).

Scenario: attacker A (team 0) kicks at t≈1.0 s to teammate B, who is standing
downfield ahead of the second-last defender; B receives at t≈2.0 s. The engine
should detect both touches, pair them, and return an OFFSIDE verdict naming A as
kicker and B as receiver.
'''
import numpy as np
from videoflow_contrib.offside_engine.engine import OffsideEngine


def make_player(gid, team, x, y, is_gk=False, is_ref=False, conf=0.9):
    kpts = np.zeros((26, 5))
    kpts[:, 0] = x
    kpts[:, 1] = y
    kpts[:, 2] = 1.0
    kpts[:, 3] = conf
    kpts[:, 4] = 2
    for j in (15, 16, 20, 21, 22, 23, 24, 25):     # feet near ground
        kpts[j, 2] = 0.05
    return {'gid': gid, 'team': team, 'is_gk': is_gk, 'is_ref': is_ref,
            'provisional': False, 'ground': [float(x), float(y)], 'vel': [0.0, 0.0],
            'kpts3d': kpts.tolist()}


def ball_at(t):
    # At A's feet, then driven to B (18,3) over [1,2], then rests at B.
    if t < 1.0:
        return [0.0, 0.0, 0.11]
    if t < 2.0:
        f = (t - 1.0)
        return [18.0 * f, 3.0 * f, 0.11]
    return [18.0, 3.0, 0.11]


def build_states(fps=30.0, duration=3.0, seed=0):
    rng = np.random.default_rng(seed)
    n = int(duration * fps)
    states = []
    for k in range(n):
        t = k / fps
        players = [
            make_player(1, 0, 0.0, 0.0),        # A (kicker)
            make_player(2, 0, 18.0, 3.0),       # B (receiver, offside)
            make_player(3, 1, 40.0, 0.0, is_gk=True),  # opponent GK (deepest)
            make_player(4, 1, 15.0, 0.0),       # opponent defender → 2nd-last line at x=15
        ]
        p = np.array(ball_at(t)) + rng.normal(0, 0.01, 3)
        states.append({'t': t, 'players': players,
                       'ball': {'p': p.tolist(), 'predicted': False, 'conf': 0.9}})
    return states


def run(states, **kw):
    eng = OffsideEngine(detection_lag_frames=6, min_gap_frames=5, ref_fps=30.0, **kw)
    eng.open()
    outputs = [eng.process(s) for s in states]
    return [o for o in outputs if o is not None]


def test_engine_emits_offside_verdict():
    outputs = run(build_states())
    verdicts = [o for o in outputs if o.get('type') == 'offside_verdict']
    assert len(verdicts) >= 1, f'no verdict; outputs={[o.get("type") for o in outputs]}'
    v = verdicts[0]
    assert v['verdict'] == 'OFFSIDE'
    assert v['kicker_gid'] == 1
    assert v['receiver_gid'] == 2
    assert v['second_defender_gid'] == 4
    assert v['margin_m'] > 0
    assert abs(v['t_kick'] - 1.0) < 0.15
    assert abs(v['t_touch'] - 2.0) < 0.15


def test_engine_detects_both_touches():
    outputs = run(build_states())
    touches = [o for o in outputs if o.get('type') == 'touch_event']
    verdicts = [o for o in outputs if o.get('type') == 'offside_verdict']
    # first touch opens the phase (touch_event); second resolves as a verdict
    assert len(touches) >= 1
    assert len(verdicts) >= 1


def test_engine_onside_when_receiver_behind_line():
    # Move B behind the defender line (x=12 < 15) → ONSIDE.
    states = build_states()
    for st in states:
        for p in st['players']:
            if p['gid'] == 2:
                p['ground'][0] = 12.0
                arr = np.array(p['kpts3d']); arr[:, 0] = 12.0; p['kpts3d'] = arr.tolist()
    # ball must still arrive at B's new location for the receive touch
    for st in states:
        t = st['t']
        if t >= 2.0:
            st['ball']['p'] = [12.0, 3.0, 0.11]
        elif t >= 1.0:
            f = t - 1.0
            st['ball']['p'] = [12.0 * f, 3.0 * f, 0.11]
    outputs = run(states)
    verdicts = [o for o in outputs if o.get('type') == 'offside_verdict']
    assert len(verdicts) >= 1
    assert verdicts[0]['verdict'] == 'ONSIDE'


def test_engine_returns_none_on_quiet_and_none_input():
    eng = OffsideEngine(detection_lag_frames=6, ref_fps=30.0)
    eng.open()
    assert eng.process(None) is None
    # a few quiet states with a stationary ball produce no events
    for k in range(5):
        s = {'t': k / 30.0, 'players': [make_player(1, 0, 0, 0)],
             'ball': {'p': [0.0, 0.0, 0.11], 'predicted': False, 'conf': 0.9}}
        assert eng.process(s) is None
