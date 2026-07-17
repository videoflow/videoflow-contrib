'''
Synthetic project→recover tests for the pure multi-view geometry.

No models, no OpenCV, no videoflow — just numpy. We build cameras with known
intrinsics/extrinsics around a soccer pitch, project known 3D points, and check
that triangulation and ground back-projection recover them.
'''
import numpy as np
from videoflow_contrib.multiview_fuser.geometry import (
    backproject_to_ground,
    build_projection,
    camera_center,
    project_points,
    triangulate_dlt,
    triangulate_ransac,
    undistort_pixels,
)


def look_at_camera(C, target, f=1200.0, w=1920, h=1080, up_world=(0.0, 0.0, 1.0)):
    '''Build (K, R, t) for a pinhole camera at ``C`` looking at ``target``.'''
    C = np.asarray(C, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    up_world = np.asarray(up_world, dtype=np.float64)
    z_axis = target - C
    z_axis /= np.linalg.norm(z_axis)                 # forward (camera +z)
    x_axis = np.cross(z_axis, up_world)
    x_axis /= np.linalg.norm(x_axis)                 # right (camera +x)
    y_axis = np.cross(z_axis, x_axis)                # down (camera +y)
    R = np.stack([x_axis, y_axis, z_axis], axis=0)   # rows = camera axes in world
    t = -R @ C
    K = np.array([[f, 0, w / 2.0],
                  [0, f, h / 2.0],
                  [0, 0, 1.0]], dtype=np.float64)
    return K, R, t


def make_rig():
    '''Three cameras around a ~100x64 m pitch, looking at the centre circle.'''
    target = np.array([0.0, 0.0, 0.0])
    rig = [
        look_at_camera([-20.0, -55.0, 9.0], target),   # near-left stand
        look_at_camera([25.0, -58.0, 11.0], target),    # near-right stand
        look_at_camera([0.0, 60.0, 12.0], target),      # far side
    ]
    return rig


def test_look_at_is_valid_rotation():
    for _K, R, _t in make_rig():
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-9)
        assert np.isclose(np.linalg.det(R), 1.0, atol=1e-9)


def test_project_center_spot_lands_near_principal_point():
    # A camera looking straight at the origin should image the origin at its centre.
    K, R, t = look_at_camera([0.0, -50.0, 10.0], [0.0, 0.0, 0.0])
    px = project_points(np.array([[0.0, 0.0, 0.0]]), K, R, t)[0]
    assert np.allclose(px, [K[0, 2], K[1, 2]], atol=1e-6)


def test_camera_center_recovered():
    C_true = np.array([12.0, -40.0, 8.0])
    K, R, t = look_at_camera(C_true, [0.0, 0.0, 0.0])
    assert np.allclose(camera_center(R, t), C_true, atol=1e-9)


def test_triangulate_recovers_known_points():
    rig = make_rig()
    projections = [build_projection(np.eye(3), R, t) for _, R, t in rig]
    rng = np.random.default_rng(0)
    pts3d = rng.uniform([-45, -30, 0.0], [45, 30, 2.0], size=(40, 3))

    max_err = 0.0
    for X in pts3d:
        obs_norm, obs_px = [], []
        for K, R, t in rig:
            px = project_points(X[None], K, R, t)[0]
            obs_px.append(px)
            obs_norm.append(undistort_pixels(px[None], K)[0])
        Xr = triangulate_dlt(np.array(obs_norm), projections)
        max_err = max(max_err, np.linalg.norm(Xr - X))
    assert max_err < 1e-6, f'triangulation error too large: {max_err}'


def test_triangulate_with_pixel_noise_is_cm_accurate():
    rig = make_rig()
    projections = [build_projection(np.eye(3), R, t) for _, R, t in rig]
    rng = np.random.default_rng(1)
    pts3d = rng.uniform([-45, -30, 0.0], [45, 30, 2.0], size=(60, 3))

    errs = []
    for X in pts3d:
        obs_norm = []
        for K, R, t in rig:
            px = project_points(X[None], K, R, t)[0]
            px = px + rng.normal(0.0, 1.0, size=2)          # 1 px gaussian noise
            obs_norm.append(undistort_pixels(px[None], K)[0])
        Xr = triangulate_dlt(np.array(obs_norm), projections)
        errs.append(np.linalg.norm(Xr - X))
    # Physical expectation: at ~55 m stand distance with f=1200 px, 1 px of keypoint
    # noise ≈ 55/1200 ≈ 4.6 cm of angular resolution per camera, so ~6 cm median 3D
    # error across 3 views is the real floor here — this IS the per-joint uncertainty
    # the offside engine's TOO_CLOSE band must absorb. (Noise-free recovers to 1e-6.)
    assert np.median(errs) < 0.09, f'median 3D error {np.median(errs):.3f} m too large'


def test_ransac_discards_one_corrupted_view():
    rig = make_rig()
    projections = [build_projection(np.eye(3), R, t) for _, R, t in rig]
    Ks = [c[0] for c in rig]
    Rs = [c[1] for c in rig]
    ts = [c[2] for c in rig]
    X = np.array([15.0, -8.0, 1.4])
    obs_px, obs_norm = [], []
    for K, R, t in rig:
        px = project_points(X[None], K, R, t)[0]
        obs_px.append(px)
        obs_norm.append(undistort_pixels(px[None], K)[0])
    # Corrupt camera 1 with a large outlier (e.g. a bad keypoint / wrong detection).
    obs_px[1] = obs_px[1] + np.array([120.0, -90.0])
    obs_norm[1] = undistort_pixels(obs_px[1][None], Ks[1])[0]

    Xr, inliers = triangulate_ransac(np.array(obs_norm), projections, np.array(obs_px),
                                     Ks, Rs, ts, inlier_px=8.0)
    assert not inliers[1], 'corrupted view should be rejected'
    assert inliers[0] and inliers[2]
    assert np.linalg.norm(Xr - X) < 0.05


def test_backproject_ground_roundtrip():
    rig = make_rig()
    rng = np.random.default_rng(2)
    ground = np.concatenate([rng.uniform([-45, -30], [45, 30], size=(30, 2)),
                             np.zeros((30, 1))], axis=1)
    for K, R, t in rig:
        px = project_points(ground, K, R, t)
        back = backproject_to_ground(px, K, R, t, plane_z=0.0)
        assert np.allclose(back, ground, atol=1e-6)


def test_distortion_undistort_roundtrip():
    K, R, t = look_at_camera([0.0, -50.0, 10.0], [0.0, 0.0, 0.0])
    dist = np.array([-0.28, 0.09, 0.0, 0.0, 0.0])            # typical barrel distortion
    rng = np.random.default_rng(3)
    pts3d = rng.uniform([-40, -25, 0.0], [40, 25, 2.0], size=(50, 3))
    px = project_points(pts3d, K, R, t, dist=dist)
    good = np.all(np.isfinite(px), axis=1)
    # Undistort the distorted pixels, re-apply K, and compare to the *undistorted* pixels.
    xy_n = undistort_pixels(px[good], K, dist)
    px_undist_expected = project_points(pts3d[good], K, R, t, dist=None)
    ones = np.ones((xy_n.shape[0], 1))
    px_recovered = (np.concatenate([xy_n, ones], axis=1) @ K.T)[:, :2]
    assert np.allclose(px_recovered, px_undist_expected, atol=0.2)  # sub-pixel


def test_triangulate_needs_two_views():
    rig = make_rig()
    projections = [build_projection(np.eye(3), R, t) for _, R, t in rig]
    obs = np.array([[0.1, 0.05], [np.nan, np.nan], [np.nan, np.nan]])
    X = triangulate_dlt(obs, projections)
    assert np.all(np.isnan(X))
