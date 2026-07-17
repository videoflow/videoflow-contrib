'''
Ball-event detection on synthetic trajectories: two scripted kicks should be
found within a frame, sub-frame timing recovered, and touches attributed to the
nearest player. Also checks the ground-bounce veto.
'''
import numpy as np
from videoflow_contrib.offside_engine.events import (
    attribute_touch,
    is_ground_bounce,
    refine_touch_time,
    smooth_ball,
    touch_candidates,
)


def scripted_ball(fps=30.0, seed=0):
    '''Ball: rolls +X, kicked at t=1.0 s (turns +Y), kicked again at t=2.0 s (turns -X).'''
    rng = np.random.default_rng(seed)
    dt = 1.0 / fps
    n = 90
    times = np.arange(n) * dt
    pos = np.zeros((n, 3))
    vel_segments = [(0.0, np.array([6.0, 0.0, 0.0])),
                    (1.0, np.array([0.0, 7.0, 0.0])),
                    (2.0, np.array([-5.0, 2.0, 0.0]))]
    p = np.array([-10.0, 0.0, 0.2])
    for k in range(n):
        t = times[k]
        v = vel_segments[0][1]
        for ts, vv in vel_segments:
            if t >= ts:
                v = vv
        if k > 0:
            p = p + v * dt
        pos[k] = p
    # true kick frames
    true_kicks = [int(round(1.0 * fps)), int(round(2.0 * fps))]
    pos += rng.normal(0, 0.02, pos.shape)      # 2 cm measurement noise
    valid = np.ones(n, dtype=bool)
    return times, pos, valid, true_kicks


def test_detects_two_kicks():
    times, pos, valid, true_kicks = scripted_ball()
    _, vel, speed, accel, usable = smooth_ball(times, pos, valid)
    cands = touch_candidates(times, vel, speed, accel, usable,
                             vel_jump_ms=3.0, dir_change_deg=25.0, min_gap_frames=5)
    # each true kick has a candidate within 1 frame
    for tk in true_kicks:
        assert any(abs(c - tk) <= 1 for c in cands), f'missed kick at {tk}, got {cands}'


def test_subframe_time_accuracy():
    # Video-only kick timing is frame-rate limited; assert ≈ 1 frame precision at 30 fps.
    fps = 30.0
    times, pos, valid, true_kicks = scripted_ball(fps=fps)
    _, vel, speed, accel, usable = smooth_ball(times, pos, valid)
    cands = touch_candidates(times, vel, speed, accel, usable)
    for tk in true_kicks:
        c = min(cands, key=lambda x: abs(x - tk))
        t0 = refine_touch_time(times, accel, c)
        assert abs(t0 - tk / fps) < 1.2 / fps, f'sub-frame t0 {t0} vs {tk/fps}'


def test_gap_bridging_marks_long_gaps_unusable():
    times, pos, valid, _ = scripted_ball()
    valid[40:50] = False           # 10-frame gap > max_gap(3)
    pos[40:50] = np.nan
    _, _, _, _, usable = smooth_ball(times, pos, valid, max_gap=3)
    assert not usable[45]
    # a short gap is bridged and stays usable
    valid2 = np.ones(len(times), dtype=bool); valid2[60:62] = False
    pos2 = pos.copy(); pos2[60:62] = np.nan
    _, _, _, _, usable2 = smooth_ball(times, pos2, valid2, max_gap=3)
    assert usable2[60] and usable2[61]


def make_player(gid, team, X, Y, Z_offsets=None):
    kpts = np.zeros((26, 5))
    kpts[:, 0] = X; kpts[:, 1] = Y; kpts[:, 2] = 1.0; kpts[:, 3] = 0.9; kpts[:, 4] = 2
    # feet near the ground
    for j in (15, 16, 20, 21, 22, 23, 24, 25):
        kpts[j, 2] = 0.05
    return {'gid': gid, 'team': team, 'is_ref': False, 'ground': [X, Y], 'kpts3d': kpts.tolist()}


def test_attribute_touch_nearest_player():
    ball = [10.0, 5.0, 0.1]
    players = [make_player(1, 0, 10.2, 5.1), make_player(2, 1, 14.0, 2.0)]
    res = attribute_touch(players, ball, touch_radius_m=1.1)
    assert res['gid'] == 1
    assert not res['contested']


def test_attribute_touch_out_of_range_returns_none():
    ball = [10.0, 5.0, 2.0]
    players = [make_player(1, 0, 20.0, 20.0)]
    res = attribute_touch(players, ball, touch_radius_m=1.1)
    assert res['gid'] is None


def test_attribute_touch_contested():
    ball = [10.0, 5.0, 0.1]
    players = [make_player(1, 0, 10.2, 5.0), make_player(2, 1, 10.35, 5.05)]
    res = attribute_touch(players, ball, touch_radius_m=1.5, contest_margin_m=0.5)
    assert res['gid'] == 1 and res['contested'] and res['second_gid'] == 2


def test_ground_bounce_veto():
    assert is_ground_bounce(0.1, vz_before=-3.0, vz_after=2.5, nearest_player_dist=5.0,
                            touch_radius_m=1.1)
    # a player is right there → NOT a bounce (it's a touch)
    assert not is_ground_bounce(0.1, vz_before=-3.0, vz_after=2.5, nearest_player_dist=0.5,
                                touch_radius_m=1.1)
    # ball in the air → not a bounce
    assert not is_ground_bounce(1.5, vz_before=-3.0, vz_after=2.5, nearest_player_dist=5.0,
                                touch_radius_m=1.1)
