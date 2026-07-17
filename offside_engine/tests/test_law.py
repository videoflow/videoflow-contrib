'''
Hand-built world-state scenarios for the offside law. No models, no fusion — we
place players/ball at known world positions and check the verdict, including the
subtle cases: arm-ahead-of-body is ONSIDE, level is TOO_CLOSE, GK counts as a
defender, and sign handling for either attack direction.
'''
import numpy as np
import pytest
from videoflow_contrib.offside_engine.law import (
    VERDICT_INCONCLUSIVE,
    VERDICT_OFFSIDE,
    VERDICT_ONSIDE,
    VERDICT_TOO_CLOSE,
    compute_offside,
    player_extreme,
)

LEGAL = list(range(26))


def make_player(gid, team, X, Y=0.0, is_gk=False, is_ref=False, conf=0.9, arm_x=None, provisional=False):
    '''26-keypoint (halpe26) player with every legal joint at X; arms optionally ahead.'''
    kpts = np.zeros((26, 5), dtype=np.float64)
    kpts[:, 0] = X
    kpts[:, 1] = Y
    kpts[:, 2] = 1.0
    kpts[:, 3] = conf
    kpts[:, 4] = 2
    if arm_x is not None:
        for j in (7, 8, 9, 10):        # elbows + wrists
            kpts[j, 0] = arm_x
    return {'gid': gid, 'team': team, 'is_gk': is_gk, 'is_ref': is_ref,
            'provisional': provisional, 'ground': [X, Y], 'vel': [0.0, 0.0], 'kpts3d': kpts.tolist()}


def base_defense(attack_sign, gk_x, line_x):
    '''GK (deepest) + one outfield defender defining the 2nd-last line, for team 1.'''
    return [make_player(100, 1, gk_x, is_gk=True), make_player(101, 1, line_x)]


def test_clear_offside():
    s = 1.0
    players = [make_player(1, 0, 30.0)] + base_defense(s, gk_x=45.0, line_x=28.0)
    res = compute_offside(players, receiver_gid=1, attack_team=0, attack_sign=s,
                          ball_xyz=[25.0, 0, 0])
    assert res['verdict'] == VERDICT_OFFSIDE
    assert res['margin_m'] == pytest.approx(2.0, abs=1e-6)
    assert res['second_defender_gid'] == 101


def test_clear_onside():
    s = 1.0
    players = [make_player(1, 0, 26.0)] + base_defense(s, gk_x=45.0, line_x=28.0)
    res = compute_offside(players, receiver_gid=1, attack_team=0, attack_sign=s,
                          ball_xyz=[24.0, 0, 0])
    assert res['verdict'] == VERDICT_ONSIDE
    assert res['margin_m'] == pytest.approx(-2.0, abs=1e-6)


def test_arm_ahead_is_onside():
    # Body behind the line (27) but an arm reaches ahead (32). Arms are NOT legal.
    s = 1.0
    receiver = make_player(1, 0, 27.0, arm_x=32.0)
    players = [receiver] + base_defense(s, gk_x=45.0, line_x=28.0)
    res = compute_offside(players, receiver_gid=1, attack_team=0, attack_sign=s,
                          ball_xyz=[24.0, 0, 0])
    assert res['verdict'] == VERDICT_ONSIDE
    # sanity: with arms included the extreme would be 32 (offside) — confirm exclusion
    sx_legal, x_legal, _ = player_extreme(receiver, s, [j for j in range(26) if j not in (7, 8, 9, 10)])
    assert x_legal == pytest.approx(27.0)


def test_level_is_too_close():
    s = 1.0
    players = [make_player(1, 0, 28.1)] + base_defense(s, gk_x=45.0, line_x=28.0)
    res = compute_offside(players, receiver_gid=1, attack_team=0, attack_sign=s,
                          ball_xyz=[20.0, 0, 0])
    assert res['verdict'] == VERDICT_TOO_CLOSE


def test_ball_can_be_the_binding_line():
    # Receiver ahead of the defender line (28) but behind the ball (32) → still onside,
    # because you must beat BOTH ball and 2nd-last defender.
    s = 1.0
    players = [make_player(1, 0, 30.0)] + base_defense(s, gk_x=45.0, line_x=28.0)
    res = compute_offside(players, receiver_gid=1, attack_team=0, attack_sign=s,
                          ball_xyz=[32.0, 0, 0])
    assert res['verdict'] == VERDICT_ONSIDE
    assert res['margin_m'] == pytest.approx(-2.0, abs=1e-6)


def test_own_half_is_onside():
    s = 1.0
    players = [make_player(1, 0, -10.0)] + base_defense(s, gk_x=45.0, line_x=-12.0)
    res = compute_offside(players, receiver_gid=1, attack_team=0, attack_sign=s,
                          ball_xyz=[-15.0, 0, 0])
    assert res['verdict'] == VERDICT_ONSIDE
    assert res['reason'] == 'own_half'


def test_fewer_than_two_defenders_inconclusive():
    s = 1.0
    players = [make_player(1, 0, 30.0), make_player(100, 1, 45.0, is_gk=True)]
    res = compute_offside(players, receiver_gid=1, attack_team=0, attack_sign=s,
                          ball_xyz=[25.0, 0, 0])
    assert res['verdict'] == VERDICT_INCONCLUSIVE
    assert res['reason'] == 'defenders_missing'


def test_referees_excluded_from_defender_line():
    # A referee standing very deep must NOT be treated as the 2nd-last defender.
    s = 1.0
    players = [make_player(1, 0, 30.0),
               make_player(100, 1, 45.0, is_gk=True),
               make_player(101, 1, 25.0),                 # real 2nd-last line = 25
               make_player(200, 1, 33.0, is_ref=True)]    # ref ahead — must be ignored
    res = compute_offside(players, receiver_gid=1, attack_team=0, attack_sign=s,
                          ball_xyz=[20.0, 0, 0])
    assert res['verdict'] == VERDICT_OFFSIDE
    assert res['second_defender_gid'] == 101
    assert res['margin_m'] == pytest.approx(5.0, abs=1e-6)


def test_negative_attack_sign_mirrors():
    # Attacking toward -X: forward = decreasing X. Mirror of the clear-offside case.
    s = -1.0
    players = [make_player(1, 0, -30.0)] + base_defense(s, gk_x=-45.0, line_x=-28.0)
    res = compute_offside(players, receiver_gid=1, attack_team=0, attack_sign=s,
                          ball_xyz=[-25.0, 0, 0])
    assert res['verdict'] == VERDICT_OFFSIDE
    assert res['margin_m'] == pytest.approx(2.0, abs=1e-6)


def test_low_confidence_widens_tolerance():
    # Same 0.2 m margin, but low-confidence keypoints widen the band to TOO_CLOSE.
    s = 1.0
    hi = [make_player(1, 0, 28.2, conf=0.95)] + base_defense(s, 45.0, 28.0)
    lo = [make_player(1, 0, 28.2, conf=0.05)] + [make_player(100, 1, 45.0, is_gk=True),
                                                 make_player(101, 1, 28.0, conf=0.05)]
    r_hi = compute_offside(hi, 1, 0, s, ball_xyz=[20, 0, 0])
    r_lo = compute_offside(lo, 1, 0, s, ball_xyz=[20, 0, 0])
    assert r_hi['verdict'] == VERDICT_OFFSIDE      # 0.2 > tol(0.15) at high conf
    assert r_lo['verdict'] == VERDICT_TOO_CLOSE    # widened tol swallows 0.2 at low conf
