'''
Team-membership fitting + assignment.

Centroids for {team0, team1, goalkeeper, referee} are fit ONCE offline across all
cameras (``fit_teams.py``) so cluster→team labels are globally consistent; the node
is then a stateless nearest-centroid assigner. Default embedding is a normalized
HSV torso histogram (robust, no heavy deps); SigLIP-2 embeddings are optional.

The HSV path is pure (numpy + cv2) and unit-tested; SigLIP is lazy-imported.
'''
from __future__ import annotations

import numpy as np

CLASSES = ['team0', 'team1', 'gk', 'referee']
_HSV_DIM = 128   # 16 hue × 8 sat bins


def torso_crop(frame_bgr: np.ndarray, box, min_crop_h: int = 24):
    '''Central torso sub-crop from a y-first box [ymin,xmin,ymax,xmax].'''
    ymin, xmin, ymax, xmax = [int(round(v)) for v in box]
    h, w = ymax - ymin, xmax - xmin
    if h < min_crop_h or w <= 0:
        return None
    y0, y1 = ymin + int(0.20 * h), ymin + int(0.60 * h)     # skip head, take torso
    x0, x1 = xmin + int(0.20 * w), xmax - int(0.20 * w)      # central 60% width
    y0, x0 = max(0, y0), max(0, x0)
    crop = frame_bgr[y0:y1, x0:x1]
    return crop if crop.size else None


def hsv_embedding(crop_bgr: np.ndarray) -> np.ndarray:
    '''128-d L2-normalized hue-saturation histogram over saturated/bright pixels.'''
    import cv2
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    mask = (S > 40) & (V > 40)
    if int(mask.sum()) < 10:
        return np.zeros(_HSV_DIM)
    hist, _, _ = np.histogram2d(H[mask].ravel(), S[mask].ravel(),
                                bins=[16, 8], range=[[0, 180], [0, 256]])
    v = hist.flatten().astype(np.float64)
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def embed_crops(crops: list, method: str = 'hsv', model=None) -> np.ndarray:
    if method == 'hsv':
        return np.array([hsv_embedding(c) for c in crops]) if crops else np.zeros((0, _HSV_DIM))
    if method == 'siglip':
        return _siglip_embed(crops, model)
    raise ValueError(f'unknown method {method!r}')


def fit_teams(embeddings: np.ndarray, det_classes: np.ndarray, method: str = 'hsv') -> dict:
    '''
    Fit the four class centroids. Players (class 0) are split into two teams by
    KMeans(2), ordered deterministically; GK/referee centroids are the means of
    their detector-class crops.

    - Returns: ``{method, classes, vectors}`` (vectors row-aligned with ``classes``).
    '''
    embeddings = np.asarray(embeddings, dtype=np.float64)
    det_classes = np.asarray(det_classes)
    dim = embeddings.shape[1] if embeddings.ndim == 2 and embeddings.shape[0] else _HSV_DIM
    vectors = np.zeros((4, dim))

    players = embeddings[np.isin(det_classes, [0, 1])]
    if len(players) >= 2:
        from sklearn.cluster import KMeans
        km = KMeans(n_clusters=2, n_init=10, random_state=0).fit(players)
        c0, c1 = km.cluster_centers_
        # deterministic order: cluster with the lower argmax-bin index becomes team0
        if int(np.argmax(c0)) > int(np.argmax(c1)):
            c0, c1 = c1, c0
        vectors[0], vectors[1] = c0, c1
    elif len(players) == 1:
        vectors[0] = players[0]

    gk = embeddings[det_classes == 2]
    if len(gk):
        vectors[2] = gk.mean(axis=0)
    ref = embeddings[det_classes == 3]
    if len(ref):
        vectors[3] = ref.mean(axis=0)

    return {'method': method, 'classes': list(CLASSES), 'vectors': vectors.tolist()}


def assign(embedding: np.ndarray, centroids: dict) -> tuple[int, float]:
    '''
    Nearest centroid by cosine similarity. Returns ``(class_id, confidence)`` where
    class_id ∈ {0,1,2,3} and confidence is the top-2 similarity margin; ``(-1, 0)``
    when the embedding is empty or no centroid is populated.
    '''
    e = np.asarray(embedding, dtype=np.float64)
    if np.linalg.norm(e) == 0:
        return -1, 0.0
    vecs = np.asarray(centroids['vectors'], dtype=np.float64)
    sim_list = []
    for i in range(vecs.shape[0]):
        v = vecs[i]
        nv = np.linalg.norm(v)
        sim_list.append((e @ v) / (np.linalg.norm(e) * nv) if nv > 0 else -1.0)
    sims = np.array(sim_list)
    if np.all(sims <= -1.0):
        return -1, 0.0
    order = np.argsort(sims)[::-1]
    best = int(order[0])
    margin = float(sims[order[0]] - sims[order[1]]) if len(order) > 1 else float(sims[best])
    return best, max(0.0, margin)


def _siglip_embed(crops, model):
    if model is None:
        raise RuntimeError('SigLIP embedding requested but no model provided')
    import torch
    proc, net = model
    import cv2
    imgs = [cv2.cvtColor(c, cv2.COLOR_BGR2RGB) for c in crops]
    inputs = proc(images=imgs, return_tensors='pt')
    with torch.no_grad():
        feats = net.get_image_features(**inputs)
    feats = feats.cpu().numpy()
    return feats / (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-9)
