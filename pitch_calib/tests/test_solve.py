'''
Synthetic calibration tests: build a known camera, project the pitch landmarks it
can see, add pixel noise, and check ``solve_camera`` recovers focal length, pose,
and a low reprojection RMS.
'''
import numpy as np
import pytest
from videoflow_contrib.pitch_calib.model import PitchModel
from videoflow_contrib.pitch_calib.solve import (
    homography_dlt,
    intrinsics_from_homography,
    solve_camera,
)


def look_at_camera(C, target, f=1400.0, w=1920, h=1080, up_world=(0.0, 0.0, 1.0)):
    C = np.asarray(C, float); target = np.asarray(target, float); up_world = np.asarray(up_world, float)
    z = target - C; z /= np.linalg.norm(z)
    x = np.cross(z, up_world); x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.stack([x, y, z], axis=0)
    t = -R @ C
    K = np.array([[f, 0, w / 2.0], [0, f, h / 2.0], [0, 0, 1.0]])
    return K, R, t


def project(world_xyz, K, R, t):
    cam = world_xyz @ R.T + t
    z = cam[:, 2]
    xy = cam[:, :2] / z[:, None]
    px = np.c_[xy, np.ones(len(xy))] @ K.T
    out = px[:, :2].copy()
    out[z <= 0] = np.nan
    return out


def visible_observations(pitch, K, R, t, w=1920, h=1080, rng=None, noise=0.0):
    names, world = pitch.keypoint_array()
    px = project(world, K, R, t)
    obs = {}
    for i, n in enumerate(names):
        p = px[i]
        if not np.all(np.isfinite(p)):
            continue
        if -50 <= p[0] <= w + 50 and -50 <= p[1] <= h + 50:   # roughly in frame
            if rng is not None and noise > 0:
                p = p + rng.normal(0, noise, 2)
            obs[n] = (float(p[0]), float(p[1]))
    return obs


def test_homography_roundtrip():
    pitch = PitchModel(105, 68)
    K, R, t = look_at_camera([0, -55, 12], [0, 0, 0])
    _, world = pitch.keypoint_array()
    px = project(world, K, R, t)
    good = np.all(np.isfinite(px), axis=1)
    H = homography_dlt(world[good, :2], px[good])
    # A homography should map world XY to the same pixels (planar scene).
    reproj = np.c_[world[good, :2], np.ones(good.sum())] @ H.T
    reproj = reproj[:, :2] / reproj[:, 2:3]
    assert np.allclose(reproj, px[good], atol=1e-6)


def test_intrinsics_recovered_noise_free():
    pitch = PitchModel(105, 68)
    for f_true in (1200.0, 1500.0, 1800.0):
        K, R, t = look_at_camera([-15, -52, 10], [5, 0, 0], f=f_true)
        _, world = pitch.keypoint_array()
        px = project(world, K, R, t)
        good = np.all(np.isfinite(px), axis=1)
        H = homography_dlt(world[good, :2], px[good])
        f_est, ok = intrinsics_from_homography(H, (1080, 1920))
        assert ok
        assert abs(f_est - f_true) / f_true < 0.03, f'f {f_est} vs {f_true}'


def test_solve_camera_noise_free_recovers_pose():
    pitch = PitchModel(105, 68)
    C_true = np.array([-18.0, -50.0, 11.0])
    K, R, t = look_at_camera(C_true, [8, 3, 0], f=1500.0)
    obs = visible_observations(pitch, K, R, t)
    calib = solve_camera(obs, pitch, (1080, 1920), refine=True)
    assert calib['rms_px'] < 0.5
    assert abs(calib['K'][0][0] - 1500.0) / 1500.0 < 0.02
    assert np.linalg.norm(np.array(calib['C']) - C_true) < 0.5   # camera centre within 0.5 m


def test_solve_camera_with_noise_and_missing_landmarks():
    pitch = PitchModel(100, 64)                       # amateur non-standard dims
    rng = np.random.default_rng(7)
    C_true = np.array([22.0, -46.0, 9.0])
    K, R, t = look_at_camera(C_true, [-5, 0, 0], f=1300.0)
    obs = visible_observations(pitch, K, R, t, rng=rng, noise=1.5)
    # Drop a few landmarks to simulate occlusion/missed detections.
    drop = list(obs)[::5]
    for d in drop:
        obs.pop(d)
    calib = solve_camera(obs, pitch, (1080, 1920), refine=True)
    assert calib['rms_px'] < 3.0
    assert np.linalg.norm(np.array(calib['C']) - C_true) < 1.5


def test_cross_camera_center_spot_agreement():
    # Two cameras: back-projecting the centre spot should agree near (0,0).
    # Integration check — needs the multiview_fuser package too; skip if absent.
    geom = pytest.importorskip('videoflow_contrib.multiview_fuser.geometry')
    backproject_to_ground = geom.backproject_to_ground
    pitch = PitchModel(105, 68)
    cams = [look_at_camera([-20, -52, 10], [0, 0, 0], f=1500.0),
            look_at_camera([25, -55, 12], [0, 0, 0], f=1450.0)]
    grounds = []
    for K, R, t in cams:
        obs = visible_observations(pitch, K, R, t)
        calib = solve_camera(obs, pitch, (1080, 1920), refine=True)
        Kc = np.array(calib['K']); Rc = np.array(calib['R']); tc = np.array(calib['t'])
        # image of the true centre spot
        spot_px = project(np.array([[0.0, 0.0, 0.0]]), K, R, t)[0]
        g = backproject_to_ground(spot_px[None], Kc, Rc, tc)[0]
        grounds.append(g[:2])
    assert np.linalg.norm(grounds[0] - grounds[1]) < 0.2   # agree within 0.2 m


def test_too_few_landmarks_raises():
    pitch = PitchModel(105, 68)
    with pytest.raises(ValueError):
        solve_camera({'center_spot': (100, 100), 'halfway_top': (200, 50)}, pitch, (1080, 1920))
