'''
Fixed-camera calibration from pitch-landmark correspondences.

Given 2D observations of named pitch landmarks (from the landmark detector) and
their known 3D world positions (from ``PitchModel``), recover the full pinhole
camera: intrinsics ``K``, distortion, and world→camera extrinsics ``R, t``.

Pipeline (single planar view, so a homography suffices for initialization):
  1. normalized-DLT homography  world(X,Y) ↔ image(u,v)
  2. Zhang closed-form intrinsics (principal point = image centre, square pixels)
  3. extrinsics from ``H, K`` (SVD orthonormalization, in-front sign disambiguation)
  4. optional nonlinear refinement over [f, cx, cy, rvec, t, k1, k2] (Huber loss)

Core (steps 1–3) is numpy-only and unit-tested by project→recover; refinement
uses scipy (and scipy's Rotation for the rotation-vector parametrization).
'''
from __future__ import annotations

import numpy as np

__all__ = ['homography_dlt', 'intrinsics_from_homography', 'extrinsics_from_homography',
           'solve_camera']


def _normalization_matrix(pts: np.ndarray) -> np.ndarray:
    '''Hartley isotropic normalization: translate to centroid, scale to mean dist √2.'''
    c = pts.mean(axis=0)
    d = np.sqrt(((pts - c) ** 2).sum(axis=1)).mean()
    s = np.sqrt(2.0) / d if d > 1e-12 else 1.0
    return np.array([[s, 0, -s * c[0]], [0, s, -s * c[1]], [0, 0, 1.0]])


def homography_dlt(world_xy: np.ndarray, img_xy: np.ndarray) -> np.ndarray:
    '''
    Normalized-DLT homography mapping world plane (X, Y) → image (u, v).

    - Arguments: world_xy (N,2), img_xy (N,2), N ≥ 4.
    - Returns: 3x3 homography H such that ``[u,v,1]ᵀ ∝ H [X,Y,1]ᵀ``.
    '''
    world_xy = np.asarray(world_xy, dtype=np.float64).reshape(-1, 2)
    img_xy = np.asarray(img_xy, dtype=np.float64).reshape(-1, 2)
    n = world_xy.shape[0]
    if n < 4:
        raise ValueError(f'homography needs >= 4 correspondences, got {n}')
    Tw = _normalization_matrix(world_xy)
    Ti = _normalization_matrix(img_xy)
    wn = (np.c_[world_xy, np.ones(n)] @ Tw.T)[:, :2]
    inp = (np.c_[img_xy, np.ones(n)] @ Ti.T)[:, :2]
    A = np.zeros((2 * n, 9))
    for i in range(n):
        X, Y = wn[i]
        u, v = inp[i]
        A[2 * i] = [-X, -Y, -1, 0, 0, 0, u * X, u * Y, u]
        A[2 * i + 1] = [0, 0, 0, -X, -Y, -1, v * X, v * Y, v]
    _, _, Vt = np.linalg.svd(A)
    Hn = Vt[-1].reshape(3, 3)
    H = np.linalg.inv(Ti) @ Hn @ Tw           # denormalize
    return H / H[2, 2]


def intrinsics_from_homography(H: np.ndarray, image_size: tuple[int, int]) -> tuple[float, bool]:
    '''
    Closed-form focal length from one homography (Zhang), principal point fixed at
    the image centre and square pixels. Returns (focal_px, ok). ``ok=False`` when
    the geometry is near-degenerate (caller should fall back to a focal prior).
    '''
    h, w = image_size
    cx, cy = w / 2.0, h / 2.0
    h1 = H[:, 0]
    h2 = H[:, 1]
    # Shift columns to principal-point-centred coordinates.
    a = np.array([h1[0] - cx * h1[2], h1[1] - cy * h1[2], h1[2]])
    b = np.array([h2[0] - cx * h2[2], h2[1] - cy * h2[2], h2[2]])
    inv_f2_estimates = []
    denom_orth = a[0] * b[0] + a[1] * b[1]
    if abs(denom_orth) > 1e-12:
        inv_f2_estimates.append(-a[2] * b[2] / denom_orth)
    denom_norm = (a[0] ** 2 + a[1] ** 2) - (b[0] ** 2 + b[1] ** 2)
    if abs(denom_norm) > 1e-12:
        inv_f2_estimates.append((b[2] ** 2 - a[2] ** 2) / denom_norm)
    valid = [e for e in inv_f2_estimates if e > 1e-12]
    if not valid:
        return 1.2 * max(w, h), False
    if len(valid) == 2 and (max(valid) / min(valid) > 1.44):   # estimates disagree > 20% in f
        return 1.2 * max(w, h), False
    f = 1.0 / np.sqrt(np.mean(valid))
    if not np.isfinite(f) or f <= 0:
        return 1.2 * max(w, h), False
    return float(f), True


def extrinsics_from_homography(H: np.ndarray, K: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    '''World→camera (R, t) from a planar homography and known intrinsics.'''
    Kinv = np.linalg.inv(K)
    h1, h2, h3 = H[:, 0], H[:, 1], H[:, 2]
    l1 = np.linalg.norm(Kinv @ h1)
    l2 = np.linalg.norm(Kinv @ h2)
    lam_abs = 2.0 / (l1 + l2)
    # Choose sign so the world origin sits in front of the camera (t_z > 0).
    sign = 1.0 if (Kinv @ h3)[2] >= 0 else -1.0
    lam = sign * lam_abs
    r1 = lam * (Kinv @ h1)
    r2 = lam * (Kinv @ h2)
    r3 = np.cross(r1, r2)
    t = lam * (Kinv @ h3)
    R0 = np.stack([r1, r2, r3], axis=1)
    U, _, Vt = np.linalg.svd(R0)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        R = U @ np.diag([1.0, 1.0, -1.0]) @ Vt
    return R, t


def _K_of(f: float, cx: float, cy: float) -> np.ndarray:
    return np.array([[f, 0, cx], [0, f, cy], [0, 0, 1.0]])


def _project(world_pts, K, R, t, dist=None):
    cam = world_pts @ R.T + t
    z = cam[:, 2]
    xy = cam[:, :2] / z[:, None]
    if dist is not None and np.any(dist):
        k1, k2 = dist[0], dist[1]
        r2 = (xy ** 2).sum(axis=1)
        radial = 1.0 + k1 * r2 + k2 * r2 * r2
        xy = xy * radial[:, None]
    px = np.c_[xy, np.ones(len(xy))] @ K.T
    return px[:, :2]


def solve_camera(observations: dict, pitch_model, image_size: tuple[int, int],
                 refine: bool = True, weights: dict | None = None,
                 min_landmarks: int = 6) -> dict:
    '''
    Solve one camera from named landmark observations.

    - Arguments:
        - observations: ``{landmark_name: (u, v)}`` image points.
        - pitch_model: a ``PitchModel`` (provides world positions per name).
        - image_size: (h, w).
        - refine: run nonlinear refinement (needs scipy).
        - weights: optional ``{name: w}`` confidence weights for refinement.
    - Returns: calib dict ``{image_size, K, dist, R, rvec, t, C, rms_px,
      n_landmarks, per_landmark_residuals}``.
    - Raises: ValueError if fewer than ``min_landmarks`` usable correspondences.
    '''
    # Look up world coordinates from either the descriptive landmarks or the
    # roboflow-32 vertices (what the yolo32 detector emits).
    world_all = {**pitch_model.keypoints(), **pitch_model.roboflow32_keypoints()}
    names = [n for n in observations if n in world_all and np.all(np.isfinite(observations[n]))]
    if len(names) < min_landmarks:
        raise ValueError(f'need >= {min_landmarks} landmarks, got {len(names)}')
    world_xyz = np.stack([world_all[n] for n in names], axis=0)
    world_xy = world_xyz[:, :2]
    img_xy = np.stack([np.asarray(observations[n], dtype=np.float64) for n in names], axis=0)
    h, w = image_size

    H = homography_dlt(world_xy, img_xy)
    f, ok = intrinsics_from_homography(H, image_size)
    K = _K_of(f, w / 2.0, h / 2.0)
    R, t = extrinsics_from_homography(H, K)
    dist = np.zeros(2)

    if refine:
        K, dist, R, t = _refine(world_xyz, img_xy, K, R, t,
                                None if weights is None else np.array([weights.get(n, 1.0) for n in names]))

    proj = _project(world_xyz, K, R, t, dist)
    resid = np.linalg.norm(proj - img_xy, axis=1)
    rms = float(np.sqrt(np.mean(resid ** 2)))
    C = (-R.T @ t)
    return {
        'image_size': [int(h), int(w)],
        'K': K.tolist(),
        'dist': [float(dist[0]), float(dist[1]), 0.0, 0.0, 0.0],
        'R': R.tolist(),
        'rvec': _rotvec(R).tolist(),
        't': t.tolist(),
        'C': C.tolist(),
        'rms_px': rms,
        'n_landmarks': len(names),
        'per_landmark_residuals': {n: float(r) for n, r in zip(names, resid)},
    }


def _rotvec(R: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation
    return Rotation.from_matrix(R).as_rotvec()


def _refine(world_xyz, img_xy, K, R, t, weights):
    from scipy.optimize import least_squares
    from scipy.spatial.transform import Rotation
    rvec0 = Rotation.from_matrix(R).as_rotvec()
    p0 = np.concatenate([[K[0, 0], K[0, 2], K[1, 2]], rvec0, t, [0.0, 0.0]])
    w = np.ones(len(world_xyz)) if weights is None else np.asarray(weights, dtype=np.float64)
    sw = np.sqrt(np.repeat(w, 2))

    def residuals(p):
        f, cx, cy = p[0], p[1], p[2]
        Rm = Rotation.from_rotvec(p[3:6]).as_matrix()
        tt = p[6:9]
        dist = p[9:11]
        proj = _project(world_xyz, _K_of(f, cx, cy), Rm, tt, dist)
        return (sw * (proj - img_xy).reshape(-1))

    sol = least_squares(residuals, p0, method='trf', loss='huber', f_scale=2.0, max_nfev=200)
    p = sol.x
    K_r = _K_of(p[0], p[1], p[2])
    R_r = Rotation.from_rotvec(p[3:6]).as_matrix()
    return K_r, np.array([p[9], p[10]]), R_r, p[6:9]
