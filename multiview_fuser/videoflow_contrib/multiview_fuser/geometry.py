'''
Pure multi-view geometry primitives for the offside fuser.

Everything here is plain numpy (no OpenCV, no videoflow) so it can be unit-tested
in isolation by projecting known 3D points through known cameras and recovering
them. The camera model is the standard pinhole with radial-tangential distortion:

    x_pixel = K · distort( (R · X_world + t) )

where ``R`` is the world→camera rotation (3x3, orthonormal, det +1), ``t`` the
world→camera translation, and ``K`` the intrinsic matrix. The camera centre in
world coordinates is ``C = -Rᵀ t``.

Conventions:
- Image points are ``(x, y)`` = ``(column, row)`` in pixels (x-first), matching the
  pose keypoint convention used across the pipeline.
- World points are ``(X, Y, Z)`` metres, pitch frame: origin at the pitch centre,
  X along the length (goal-to-goal), Y along the width, Z up.
'''
from __future__ import annotations

import numpy as np

__all__ = [
    'build_projection',
    'camera_center',
    'project_points',
    'distort_normalized',
    'undistort_pixels',
    'backproject_to_ground',
    'triangulate_dlt',
    'reprojection_errors',
    'triangulate_ransac',
]


def build_projection(K: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    '''3x4 projection matrix ``P = K [R | t]`` (maps world homogeneous → image homogeneous).'''
    K = np.asarray(K, dtype=np.float64).reshape(3, 3)
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    t = np.asarray(t, dtype=np.float64).reshape(3)
    Rt = np.empty((3, 4), dtype=np.float64)
    Rt[:, :3] = R
    Rt[:, 3] = t
    return K @ Rt


def camera_center(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    '''World-space camera centre ``C = -Rᵀ t``.'''
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    t = np.asarray(t, dtype=np.float64).reshape(3)
    return -R.T @ t


def distort_normalized(xy_n: np.ndarray, dist: np.ndarray) -> np.ndarray:
    '''
    Apply radial-tangential distortion to normalized (camera-plane) coordinates.

    - Arguments:
        - xy_n: (N, 2) normalized coords (camera plane, before K).
        - dist: (k1, k2, p1, p2, k3) — any trailing terms may be omitted / zero.
    - Returns: (N, 2) distorted normalized coords.
    '''
    xy_n = np.asarray(xy_n, dtype=np.float64).reshape(-1, 2)
    d = np.zeros(5, dtype=np.float64)
    dd = np.asarray(dist, dtype=np.float64).reshape(-1)
    d[:min(5, dd.size)] = dd[:5]
    k1, k2, p1, p2, k3 = d
    x, y = xy_n[:, 0], xy_n[:, 1]
    r2 = x * x + y * y
    radial = 1.0 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2
    x_d = x * radial + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
    y_d = y * radial + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y
    return np.stack([x_d, y_d], axis=1)


def project_points(X_world: np.ndarray, K: np.ndarray, R: np.ndarray, t: np.ndarray,
                   dist: np.ndarray | None = None) -> np.ndarray:
    '''
    Project world points to pixels through the full pinhole+distortion model.

    - Arguments:
        - X_world: (N, 3) world points.
        - K, R, t: intrinsics / world→camera extrinsics.
        - dist: optional distortion coefficients.
    - Returns: (N, 2) pixel coords (x, y). Points behind the camera get NaN.
    '''
    X = np.asarray(X_world, dtype=np.float64).reshape(-1, 3)
    K = np.asarray(K, dtype=np.float64).reshape(3, 3)
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    t = np.asarray(t, dtype=np.float64).reshape(3)
    cam = X @ R.T + t                      # (N, 3) in camera frame
    z = cam[:, 2]
    with np.errstate(divide='ignore', invalid='ignore'):
        xy_n = cam[:, :2] / z[:, None]
    if dist is not None:
        xy_n = distort_normalized(xy_n, dist)
    ones = np.ones((xy_n.shape[0], 1), dtype=np.float64)
    px = np.concatenate([xy_n, ones], axis=1) @ K.T
    out = px[:, :2].copy()
    out[z <= 0] = np.nan                    # behind the camera → undefined
    return out


def undistort_pixels(pts_px: np.ndarray, K: np.ndarray,
                     dist: np.ndarray | None = None, iters: int = 6) -> np.ndarray:
    '''
    Map distorted pixels to *undistorted normalized* coordinates (camera plane).

    Iterative inverse of the radial-tangential model (same math as
    ``cv2.undistortPoints`` with ``P=None``). With zero distortion this is just
    ``K⁻¹ · [u, v, 1]``.

    - Returns: (N, 2) normalized coords suitable for triangulation / back-projection.
    '''
    pts = np.asarray(pts_px, dtype=np.float64).reshape(-1, 2)
    K = np.asarray(K, dtype=np.float64).reshape(3, 3)
    Kinv = np.linalg.inv(K)
    ones = np.ones((pts.shape[0], 1), dtype=np.float64)
    xy_n = (np.concatenate([pts, ones], axis=1) @ Kinv.T)[:, :2]
    if dist is None or not np.any(dist):
        return xy_n
    d = np.zeros(5, dtype=np.float64)
    dd = np.asarray(dist, dtype=np.float64).reshape(-1)
    d[:min(5, dd.size)] = dd[:5]
    k1, k2, p1, p2, k3 = d
    x_d, y_d = xy_n[:, 0].copy(), xy_n[:, 1].copy()
    x, y = x_d.copy(), y_d.copy()
    for _ in range(iters):
        r2 = x * x + y * y
        radial = 1.0 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2
        dx = 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
        dy = p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y
        x = (x_d - dx) / radial
        y = (y_d - dy) / radial
    return np.stack([x, y], axis=1)


def backproject_to_ground(pts_px: np.ndarray, K: np.ndarray, R: np.ndarray, t: np.ndarray,
                          dist: np.ndarray | None = None, plane_z: float = 0.0) -> np.ndarray:
    '''
    Back-project pixels onto the world plane ``Z = plane_z`` (the pitch).

    Ray from the camera centre ``C`` through the pixel: ``X(λ) = C + λ · Rᵀ · d_cam``,
    intersected with the horizontal plane.

    - Returns: (N, 3) world points on the plane. Rays parallel to the plane give NaN.
    '''
    xy_n = undistort_pixels(pts_px, K, dist)                    # (N, 2)
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    t = np.asarray(t, dtype=np.float64).reshape(3)
    C = camera_center(R, t)
    ones = np.ones((xy_n.shape[0], 1), dtype=np.float64)
    d_cam = np.concatenate([xy_n, ones], axis=1)                # (N, 3) ray dirs in camera frame
    d_world = d_cam @ R                                         # Rᵀ · d_cam  (row-vector form)
    with np.errstate(divide='ignore', invalid='ignore'):
        lam = (plane_z - C[2]) / d_world[:, 2]
    X = C[None, :] + lam[:, None] * d_world
    X[~np.isfinite(lam)] = np.nan
    return X


def triangulate_dlt(pts_norm: np.ndarray, projections: list[np.ndarray],
                    weights: np.ndarray | None = None) -> np.ndarray:
    '''
    Linear (DLT) triangulation of one 3D point from ≥2 views.

    - Arguments:
        - pts_norm: (V, 2) *undistorted normalized* observations, one row per view.
        - projections: list of V normalized projection matrices ``[R | t]`` (3x4).
          Because observations are normalized (K already removed), pass the extrinsic
          projection, NOT ``K[R|t]``. (Build with ``build_projection(I, R, t)``.)
        - weights: optional (V,) per-view confidence weights.
    - Returns: (3,) world point. Returns NaN if fewer than 2 finite views.
    '''
    pts = np.asarray(pts_norm, dtype=np.float64).reshape(-1, 2)
    V = pts.shape[0]
    if weights is None:
        weights = np.ones(V, dtype=np.float64)
    else:
        weights = np.asarray(weights, dtype=np.float64).reshape(-1)
    rows = []
    for i in range(V):
        if not np.all(np.isfinite(pts[i])) or weights[i] <= 0:
            continue
        P = np.asarray(projections[i], dtype=np.float64).reshape(3, 4)
        w = weights[i]
        x, y = pts[i]
        rows.append(w * (x * P[2, :] - P[0, :]))
        rows.append(w * (y * P[2, :] - P[1, :]))
    if len(rows) < 4:                                          # < 2 usable views
        return np.full(3, np.nan)
    A = np.stack(rows, axis=0)
    _, _, Vt = np.linalg.svd(A)
    Xh = Vt[-1]
    if abs(Xh[3]) < 1e-12:
        return np.full(3, np.nan)
    return Xh[:3] / Xh[3]


def reprojection_errors(X_world: np.ndarray, pts_px: np.ndarray,
                        Ks: list[np.ndarray], Rs: list[np.ndarray], ts: list[np.ndarray],
                        dists: list[np.ndarray] | None = None) -> np.ndarray:
    '''Per-view reprojection error (pixels) of a world point. NaN where a view is missing.'''
    X = np.asarray(X_world, dtype=np.float64).reshape(1, 3)
    pts = np.asarray(pts_px, dtype=np.float64).reshape(-1, 2)
    errs = np.full(pts.shape[0], np.nan)
    for i in range(pts.shape[0]):
        if not np.all(np.isfinite(pts[i])):
            continue
        d = None if dists is None else dists[i]
        proj = project_points(X, Ks[i], Rs[i], ts[i], d)[0]
        if np.all(np.isfinite(proj)):
            errs[i] = float(np.hypot(*(proj - pts[i])))
    return errs


def triangulate_ransac(pts_norm: np.ndarray, projections: list[np.ndarray],
                       pts_px: np.ndarray, Ks: list[np.ndarray], Rs: list[np.ndarray],
                       ts: list[np.ndarray], dists: list[np.ndarray] | None = None,
                       weights: np.ndarray | None = None,
                       inlier_px: float = 8.0) -> tuple[np.ndarray, np.ndarray]:
    '''
    Robust triangulation over camera pairs (RANSAC-style).

    With ≥3 views, tries every camera pair, triangulates, counts views whose
    reprojection error is below ``inlier_px``, keeps the largest inlier set and
    refits DLT on it. With exactly 2 finite views it falls back to plain DLT.

    - Returns: (X (3,), inlier_mask (V,) bool).
    '''
    pts_norm = np.asarray(pts_norm, dtype=np.float64).reshape(-1, 2)
    V = pts_norm.shape[0]
    finite = np.array([np.all(np.isfinite(pts_norm[i])) for i in range(V)])
    if weights is not None:
        weights = np.asarray(weights, dtype=np.float64).reshape(-1)
        finite &= weights > 0
    idx = np.where(finite)[0]
    if idx.size < 2:
        return np.full(3, np.nan), np.zeros(V, dtype=bool)
    if idx.size == 2:
        X = triangulate_dlt(pts_norm[idx], [projections[i] for i in idx],
                            None if weights is None else weights[idx])
        return X, finite

    best_inliers: np.ndarray = np.zeros(V, dtype=bool)
    best_count = -1
    for a_pos in range(idx.size):
        for b_pos in range(a_pos + 1, idx.size):
            i, j = idx[a_pos], idx[b_pos]
            X = triangulate_dlt(pts_norm[[i, j]], [projections[i], projections[j]])
            if not np.all(np.isfinite(X)):
                continue
            errs = reprojection_errors(X, pts_px, Ks, Rs, ts, dists)
            inliers = np.isfinite(errs) & (errs <= inlier_px)
            if inliers.sum() > best_count:
                best_count, best_inliers = int(inliers.sum()), inliers
    if best_count < 2:                                        # no consistent pair
        X = triangulate_dlt(pts_norm[idx], [projections[i] for i in idx],
                            None if weights is None else weights[idx])
        return X, finite
    in_idx = np.where(best_inliers)[0]
    w = None if weights is None else weights[in_idx]
    X = triangulate_dlt(pts_norm[in_idx], [projections[i] for i in in_idx], w)
    return X, best_inliers
