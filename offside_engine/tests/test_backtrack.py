'''
Backward re-association tests: a continuous track is followed trivially; an ID
switch mid-play is bridged by position+team gating; a long disappearance is
reported as track_lost.
'''

from videoflow_contrib.offside_engine.backtrack import backtrack_receiver


def player(gid, team, x, y):
    return {'gid': gid, 'team': team, 'is_ref': False, 'ground': [x, y]}


def moving_states(n=10, dt=1 / 30, fps_speed=6.0, start_x=0.0, gid=1, team=0, others=None):
    '''Receiver ``gid`` moves +X at fps_speed; ``others`` is a list of static distractors.'''
    states = []
    for k in range(n):
        x = start_x + fps_speed * (k * dt)
        players = [player(gid, team, x, 2.0)]
        for o in (others or []):
            players.append(player(*o))
        states.append({'t': k * dt, 'players': players, 'ball': None})
    return states


def test_continuous_track_followed():
    states = moving_states()
    res = backtrack_receiver(states, receiver_gid=1, team=0)
    assert res['found']
    assert res['gid_at_kick'] == 1
    # position at kick (state 0) is near start_x
    assert abs(res['pos_at_kick'][0] - 0.0) < 1e-6


def test_id_switch_bridged():
    # Receiver is gid=1 for the last 4 frames, but was gid=7 for the first 6 (an ID
    # switch at frame 6). Position is continuous, same team → should be bridged.
    dt = 1 / 30
    speed = 6.0
    states = []
    for k in range(10):
        x = speed * (k * dt)
        gid = 7 if k < 6 else 1
        players = [player(gid, 0, x, 2.0), player(50, 1, 30.0, 10.0)]  # + a far distractor
        states.append({'t': k * dt, 'players': players, 'ball': None})
    res = backtrack_receiver(states, receiver_gid=1, team=0)
    assert res['found']
    assert res['gid_at_kick'] == 7            # correctly re-associated to the earlier id
    assert 'id_switch' in res['flags']
    assert abs(res['pos_at_kick'][0] - 0.0) < 0.2


def test_wrong_team_not_adopted():
    # After the receiver (gid 1, team 0) vanishes, only an opponent is nearby → not adopted.
    dt = 1 / 30
    states = []
    for k in range(10):
        x = 6.0 * (k * dt)
        if k >= 4:
            players = [player(1, 0, x, 2.0)]
        else:
            # receiver gone; an opponent occupies a nearby spot
            players = [player(99, 1, x, 2.0)]
        states.append({'t': k * dt, 'players': players, 'ball': None})
    res = backtrack_receiver(states, receiver_gid=1, team=0, max_pred_steps=8)
    # It should predict through the gap (opponent rejected) and still resolve via prediction.
    assert res['found']
    assert 'predicted_gap' in res['flags']
    assert 99 not in [res['gid_at_kick']]


def test_track_lost_when_gap_too_long():
    dt = 1 / 30
    states = []
    for k in range(20):
        if k >= 15:
            players = [player(1, 0, 6.0 * k * dt, 2.0)]
        else:
            players = [player(500, 1, 40.0, 30.0)]   # nothing plausible nearby
        states.append({'t': k * dt, 'players': players, 'ball': None})
    res = backtrack_receiver(states, receiver_gid=1, team=0, max_pred_steps=8)
    assert not res['found']
    assert res['reason'] == 'track_lost'


def test_receiver_absent_at_touch():
    states = moving_states()
    res = backtrack_receiver(states, receiver_gid=999, team=0)
    assert not res['found']
    assert res['reason'] == 'receiver_absent_at_touch'
