'''
Multi-view fuser — turns per-camera 2D features into one 3D world-state stream.

For each time-synchronized group (from the event-time join) it: resamples each
camera to a common reference time, back-projects foot points to the ground,
assigns global ids (registry), triangulates each player's skeleton and the ball,
and emits a compact world-state dict. Single-task: the registry/ball state is
sequential and the event-time join requires ``nb_tasks == 1``.
'''
from __future__ import annotations

import json

import numpy as np
from videoflow.core.node import OneTaskProcessorNode

from .ballkf import BallKalman
from .geometry import (
    backproject_to_ground,
    build_projection,
    triangulate_ransac,
    undistort_pixels,
)
from .registry import GlobalTrackRegistry

# Canonical halpe26 foot keypoints (for the ground point) and the joint count.
_FOOT_KPTS = [15, 16, 20, 21, 22, 23, 24, 25]
_NUM_KPTS = 26


class MultiviewFuser(OneTaskProcessorNode):
    '''
    - Arguments:
        - cameras: ordered list of parent (camera) names — must match wiring order.
        - calibration: ``{cam: calib_dict}`` inline, or calibration_path to a dir/json.
        - ref_fps, assoc_gate_m, min_views, kpt_min_conf, team_vote_window,
          ball_gap_bridge_s: see the plan.
    '''
    def __init__(self, cameras, calibration=None, calibration_path=None, ref_fps: float = 30.0,
                 assoc_gate_m: float = 1.2, min_views: int = 2, kpt_min_conf: float = 0.3,
                 team_vote_window: int = 90, ball_gap_bridge_s: float = 0.3, **kwargs) -> None:
        self._cameras = list(cameras)
        self._calibration = calibration
        self._calibration_path = calibration_path
        self._ref_fps = float(ref_fps)
        self._assoc_gate_m = float(assoc_gate_m)
        self._min_views = int(min_views)
        self._kpt_min_conf = float(kpt_min_conf)
        self._team_vote_window = int(team_vote_window)
        self._ball_gap_bridge_s = float(ball_gap_bridge_s)
        self._calib: dict = {}
        self._P: dict = {}
        self._prev: dict = {}
        self._registry: GlobalTrackRegistry | None = None
        self._ball_kf: BallKalman | None = None
        self._ball_miss_run = 0
        self._last_t = None
        super().__init__(**kwargs)

    def open(self):
        self._calib = self._load_calibration()
        for cam, c in self._calib.items():
            R = np.asarray(c['R'], dtype=np.float64)
            t = np.asarray(c['t'], dtype=np.float64)
            self._P[cam] = build_projection(np.eye(3), R, t)
        self._registry = GlobalTrackRegistry(assoc_gate_m=self._assoc_gate_m,
                                             min_views=self._min_views)
        self._ball_kf = BallKalman()
        self._prev = {}
        self._ball_miss_run = 0
        self._last_t = None

    def _load_calibration(self) -> dict:
        if self._calibration:
            return dict(self._calibration)
        if self._calibration_path:
            import os
            calib = {}
            if os.path.isdir(self._calibration_path):
                for cam in self._cameras:
                    p = os.path.join(self._calibration_path, f'{cam}.json')
                    if os.path.exists(p):
                        with open(p) as f:
                            calib[cam] = json.load(f)
            else:
                with open(self._calibration_path) as f:
                    calib = json.load(f)
            return calib
        raise ValueError('MultiviewFuser needs calibration or calibration_path')

    # -- main -----------------------------------------------------------------
    def process(self, *inputs, ctx=None):
        info = getattr(ctx, 'input_info', {}) if ctx is not None else {}
        present = {}
        for cam, packed in zip(self._cameras, inputs):
            if packed is None:
                continue
            ev = None
            if info and info.get(cam):
                ev = info[cam].get('event_ts')
            present[cam] = (ev, packed)
        if not present:
            return None

        times = [ev for ev, _ in present.values() if ev is not None]
        if times:
            t_ref = float(np.mean(times))
        else:
            t_ref = (self._last_t + 1.0 / self._ref_fps) if self._last_t is not None else 0.0

        resampled = {cam: self._resample(cam, ev, packed, t_ref)
                     for cam, (ev, packed) in present.items()}

        observations = []
        for cam, packed in resampled.items():
            for trk in packed.get('tracks', []):
                g = self._ground_point(cam, trk)
                if g is None:
                    continue
                observations.append({'cam': cam, 'tid': trk['tid'], 'ground': g.tolist(),
                                     'cls': int(trk.get('team', -1)),
                                     'conf': float(trk.get('team_conf', 1.0))})
        self._registry.update(t_ref, observations)
        players = self._registry.players()
        self._resolve_gk_teams(players)

        resid_acc = []
        for p in players:
            kpts3d, resid = self._triangulate_player(p, resampled)
            p['kpts3d'] = kpts3d
            if resid is not None:
                resid_acc.append(resid)

        ball = self._fuse_ball(resampled, t_ref)
        self._last_t = t_ref
        return {
            't': t_ref, 'players': players, 'ball': ball,
            'nviews': len(present),
            'resid_px': float(np.mean(resid_acc)) if resid_acc else None,
        }

    # -- resampling -----------------------------------------------------------
    def _resample(self, cam, ev, packed, t_ref):
        prev = self._prev.get(cam)
        self._prev[cam] = (ev, packed)
        if prev is None or ev is None or prev[0] is None:
            return packed
        ev0, packed0 = prev
        if abs(ev - ev0) < 1e-9 or not (min(ev0, ev) <= t_ref <= max(ev0, ev)):
            return packed
        a = (t_ref - ev0) / (ev - ev0)
        return self._interp_packed(packed0, packed, a)

    def _interp_packed(self, p0, p1, a):
        by_tid0 = {t['tid']: t for t in p0.get('tracks', [])}
        out_tracks = []
        for t1 in p1.get('tracks', []):
            t0 = by_tid0.get(t1['tid'])
            if t0 is None:
                out_tracks.append(t1)
                continue
            box = ((1 - a) * np.asarray(t0['box']) + a * np.asarray(t1['box'])).tolist()
            k0 = np.asarray(t0['kpts'], dtype=np.float64)
            k1 = np.asarray(t1['kpts'], dtype=np.float64)
            kpts = ((1 - a) * k0 + a * k1)
            # keep the min confidence of the two endpoints
            kpts[:, 2] = np.minimum(k0[:, 2], k1[:, 2])
            out_tracks.append({**t1, 'box': box, 'kpts': kpts.tolist()})
        ball = p1.get('ball')
        b0, b1 = p0.get('ball'), p1.get('ball')
        if b0 is not None and b1 is not None:
            yx = ((1 - a) * np.asarray(b0['yx']) + a * np.asarray(b1['yx'])).tolist()
            ball = {'yx': yx, 'score': min(b0['score'], b1['score'])}
        return {**p1, 'tracks': out_tracks, 'ball': ball}

    # -- geometry helpers -----------------------------------------------------
    def _ground_point(self, cam, trk):
        if cam not in self._calib:
            return None
        c = self._calib[cam]
        K = np.asarray(c['K']); R = np.asarray(c['R']); t = np.asarray(c['t'])
        dist = np.asarray(c.get('dist', [0, 0, 0, 0, 0]))
        kpts = np.asarray(trk['kpts'], dtype=np.float64)
        foot = [j for j in _FOOT_KPTS if j < len(kpts) and kpts[j, 2] >= self._kpt_min_conf]
        if foot:
            px = kpts[foot, :2].mean(axis=0)
        else:
            ymin, xmin, ymax, xmax = trk['box']
            px = np.array([(xmin + xmax) / 2.0, ymax])   # bottom-centre (x, y)
        g = backproject_to_ground(px[None], K, R, t, dist)[0]
        if not np.all(np.isfinite(g)):
            return None
        return g[:2]

    def _triangulate_player(self, player, resampled):
        per_cam = player.get('per_cam', {})
        cams = [c for c in per_cam if c in self._calib and c in resampled]
        if len(cams) < 2:
            return None, None
        # gather each camera's keypoint array for this player's tid
        cam_kpts = {}
        for cam in cams:
            tid = per_cam[cam]
            for trk in resampled[cam].get('tracks', []):
                if trk['tid'] == tid:
                    cam_kpts[cam] = np.asarray(trk['kpts'], dtype=np.float64)
                    break
        cams = [c for c in cams if c in cam_kpts]
        if len(cams) < 2:
            return None, None
        Ks = [np.asarray(self._calib[c]['K']) for c in cams]
        Rs = [np.asarray(self._calib[c]['R']) for c in cams]
        ts = [np.asarray(self._calib[c]['t']) for c in cams]
        dists = [np.asarray(self._calib[c].get('dist', [0, 0, 0, 0, 0])) for c in cams]
        Ps = [self._P[c] for c in cams]

        out = np.zeros((_NUM_KPTS, 5))
        resids = []
        for j in range(_NUM_KPTS):
            pts_px, pts_norm, weights = [], [], []
            for ci, cam in enumerate(cams):
                kp = cam_kpts[cam]
                if j < len(kp) and kp[j, 2] >= self._kpt_min_conf:
                    px = kp[j, :2]
                    pts_px.append(px)
                    pts_norm.append(undistort_pixels(px[None], Ks[ci], dists[ci])[0])
                    weights.append(kp[j, 2])
                else:
                    pts_px.append([np.nan, np.nan])
                    pts_norm.append([np.nan, np.nan])
                    weights.append(0.0)
            nvalid = int(np.sum(np.array(weights) > 0))
            if nvalid < 2:
                out[j] = [np.nan, np.nan, np.nan, 0.0, nvalid]
                continue
            X, inliers = triangulate_ransac(np.array(pts_norm), Ps, np.array(pts_px),
                                            Ks, Rs, ts, dists, np.array(weights))
            conf = float(np.mean([weights[i] for i in range(len(cams)) if inliers[i]])) if inliers.any() else 0.0
            out[j] = [X[0], X[1], X[2], conf, int(inliers.sum())]
            if np.all(np.isfinite(X)):
                from .geometry import reprojection_errors
                e = reprojection_errors(X, np.array(pts_px), Ks, Rs, ts, dists)
                e = e[np.isfinite(e)]
                if e.size:
                    resids.append(float(np.mean(e)))
        return out.tolist(), (float(np.mean(resids)) if resids else None)

    def _fuse_ball(self, resampled, t_ref):
        pts_px, pts_norm, Ks, Rs, ts, dists, Ps, weights = [], [], [], [], [], [], [], []
        for cam, packed in resampled.items():
            b = packed.get('ball')
            if b is None or cam not in self._calib:
                continue
            c = self._calib[cam]
            K = np.asarray(c['K']); dist = np.asarray(c.get('dist', [0, 0, 0, 0, 0]))
            y, x = b['yx']
            px = np.array([x, y])
            pts_px.append(px)
            pts_norm.append(undistort_pixels(px[None], K, dist)[0])
            Ks.append(K); Rs.append(np.asarray(c['R'])); ts.append(np.asarray(c['t']))
            dists.append(dist); Ps.append(self._P[cam]); weights.append(float(b.get('score', 1.0)))

        dt = (t_ref - self._last_t) if self._last_t is not None else 1.0 / self._ref_fps
        dt = max(1e-3, dt)
        measured = None
        if len(pts_px) >= 2:
            X, inliers = triangulate_ransac(np.array(pts_norm), Ps, np.array(pts_px),
                                            Ks, Rs, ts, dists, np.array(weights))
            if np.all(np.isfinite(X)):
                measured = X

        if self._ball_kf.initialized:
            self._ball_kf.predict(dt)
        accepted = False
        if measured is not None:
            accepted = self._ball_kf.update(measured)
        if accepted:
            self._ball_miss_run = 0
        else:
            self._ball_miss_run += 1

        pos = self._ball_kf.position()
        if pos is None:
            return None
        if self._ball_miss_run * dt > self._ball_gap_bridge_s:
            return None
        conf = 1.0 if accepted else max(0.0, 1.0 - self._ball_miss_run * 0.2)
        return {'p': pos.tolist(), 'predicted': (not accepted), 'conf': float(conf)}

    # -- gk team resolution ---------------------------------------------------
    def _resolve_gk_teams(self, players):
        '''
        Assign goalkeepers to teams by position, appearance-independent. The two GKs
        sit at opposite goals; pairing the GK at each end to the team whose outfield
        centroid leans toward that end is robust for both the defending GK (near its
        own outfield) and the attacking GK (at the empty end). Sort GKs by X and teams
        by outfield-mean X, then pair in order (optimal for the 2-GK / 2-team case).
        '''
        team_x = {0: [], 1: []}
        for p in players:
            if p.get('is_gk') or p.get('is_ref') or p.get('provisional'):
                continue
            if p.get('team') in (0, 1):
                team_x[p['team']].append(p['ground'][0])
        means = {tm: float(np.mean(xs)) for tm, xs in team_x.items() if xs}
        gks = [p for p in players if p.get('is_gk')]
        if not gks or not means:
            return
        if len(means) < 2:
            only = next(iter(means))
            for p in gks:
                p['team'] = only
            return
        teams_sorted = sorted(means, key=lambda tm: means[tm])
        for i, p in enumerate(sorted(gks, key=lambda q: q['ground'][0])):
            p['team'] = teams_sorted[min(i, len(teams_sorted) - 1)]
