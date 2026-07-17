'''
Team-fitting tests (HSV path, pure numpy + cv2): synthetic solid-color torso crops
cluster into two teams, and nearest-centroid assignment recovers the right team.
'''
import numpy as np
from videoflow_contrib.team_classifier.fitting import (
    assign,
    embed_crops,
    fit_teams,
    hsv_embedding,
    torso_crop,
)


def solid(color_bgr, h=80, w=40):
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:] = color_bgr
    return img


def test_hsv_embedding_normalized_and_discriminative():
    red = hsv_embedding(solid((40, 40, 200)))
    blue = hsv_embedding(solid((200, 60, 40)))
    assert abs(np.linalg.norm(red) - 1.0) < 1e-6
    assert red @ blue < 0.5           # different colors → low similarity


def test_torso_crop_region():
    frame = np.zeros((200, 200, 3), dtype=np.uint8)
    frame[100:160, 90:110] = (0, 0, 255)      # a red torso patch
    crop = torso_crop(frame, [100, 80, 180, 120])   # y-first box around it
    assert crop is not None and crop.size > 0


def test_fit_and_assign_two_teams():
    rng = np.random.default_rng(0)
    crops, classes = [], []
    for _ in range(12):
        crops.append(solid((40 + rng.integers(-15, 15), 40, 200)))   # reddish → team A
        classes.append(0)
    for _ in range(12):
        crops.append(solid((200, 60, 40 + rng.integers(-15, 15))))   # bluish → team B
        classes.append(0)
    # a goalkeeper (green) and a referee (yellow)
    crops.append(solid((40, 200, 40))); classes.append(2)
    crops.append(solid((40, 220, 220))); classes.append(3)

    embs = embed_crops(crops, method='hsv')
    centroids = fit_teams(embs, np.array(classes), method='hsv')
    assert centroids['classes'] == ['team0', 'team1', 'gk', 'referee']

    # a fresh red crop and blue crop should land on different teams
    red_id, red_conf = assign(hsv_embedding(solid((40, 40, 200))), centroids)
    blue_id, blue_conf = assign(hsv_embedding(solid((200, 60, 40))), centroids)
    assert red_id in (0, 1) and blue_id in (0, 1)
    assert red_id != blue_id
    # gk (green) and referee (yellow) recovered
    gk_id, _ = assign(hsv_embedding(solid((40, 200, 40))), centroids)
    ref_id, _ = assign(hsv_embedding(solid((40, 220, 220))), centroids)
    assert gk_id == 2
    assert ref_id == 3


def test_assign_empty_embedding_is_unknown():
    centroids = {'method': 'hsv', 'classes': ['team0', 'team1', 'gk', 'referee'],
                 'vectors': (np.eye(4, 128)).tolist()}
    cid, conf = assign(np.zeros(128), centroids)
    assert cid == -1 and conf == 0.0
