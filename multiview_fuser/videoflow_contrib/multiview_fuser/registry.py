'''
Global track registry — assigns persistent global ids to players seen across
cameras and frames.

Each per-camera track contributes a ground-plane observation (back-projected foot
point). A sticky per-(camera, tid) map handles the common case cheaply; new or
changed tids are matched to existing global tracks by predicted ground position
(Hungarian assignment within a gate). Team / goalkeeper / referee membership is a
rolling weighted vote from the per-camera team classifier.
'''
from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment


class _Track:
    __slots__ = ('gid', 'pos', 'vel', 'last_t', 'votes', 'miss', 'per_cam',
                 'seen_frames', 'nviews_last', 'ever_multiview')

    def __init__(self, gid, pos, t):
        self.gid = gid
        self.pos = np.asarray(pos, dtype=np.float64)
        self.vel = np.zeros(2)
        self.last_t = t
        self.votes = np.zeros(4)          # class votes: [team0, team1, gk, ref]
        self.miss = 0
        self.per_cam: dict = {}
        self.seen_frames = 0
        self.nviews_last = 0
        self.ever_multiview = False


class GlobalTrackRegistry:
    def __init__(self, assoc_gate_m: float = 1.2, min_views: int = 2,
                 retire_after_s: float = 3.0, alpha: float = 0.5, beta: float = 0.2,
                 provisional_frames: int = 5):
        self._gate = float(assoc_gate_m)
        self._min_views = int(min_views)
        self._retire = float(retire_after_s)
        self._alpha = float(alpha)
        self._beta = float(beta)
        self._prov_frames = int(provisional_frames)
        self._tracks: dict[int, _Track] = {}
        self._cam_tid_to_gid: dict[tuple, int] = {}
        self._next_gid = 1

    def update(self, t: float, observations: list[dict]) -> None:
        '''
        - observations: list of ``{cam, tid, ground:(x,y), cls:int(0..3), conf:float}``.
        '''
        # Predicted positions for matching.
        pred = {}
        for gid, tr in self._tracks.items():
            dt = max(1e-3, t - tr.last_t)
            pred[gid] = tr.pos + tr.vel * dt

        assigned: dict[int, list[dict]] = {}
        leftovers: list[dict] = []
        used_gids_this_cam: dict[str, set] = {}

        # 1) sticky (cam,tid) → gid
        for obs in observations:
            key = (obs['cam'], obs['tid'])
            sticky_gid = self._cam_tid_to_gid.get(key)
            if sticky_gid is not None and sticky_gid in self._tracks:
                assigned.setdefault(sticky_gid, []).append(obs)
                used_gids_this_cam.setdefault(obs['cam'], set()).add(sticky_gid)
            else:
                leftovers.append(obs)

        # 2) Hungarian match leftovers to predicted tracks (gate; per-camera uniqueness).
        free_gids = [g for g in self._tracks if g not in
                     {gg for s in used_gids_this_cam.values() for gg in s}]
        if leftovers and free_gids:
            cost = np.full((len(leftovers), len(free_gids)), 1e6)
            for i, obs in enumerate(leftovers):
                g = np.asarray(obs['ground'], dtype=np.float64)
                for j, gid in enumerate(free_gids):
                    used = used_gids_this_cam.get(obs['cam'], set())
                    if gid in used:
                        continue
                    d = float(np.linalg.norm(g - pred[gid]))
                    if d <= self._gate:
                        cost[i, j] = d
            rows, cols = linear_sum_assignment(cost)
            matched_obs = set()
            for r, c in zip(rows, cols):
                if cost[r, c] >= 1e6:
                    continue
                gid = free_gids[c]
                obs = leftovers[r]
                assigned.setdefault(gid, []).append(obs)
                used_gids_this_cam.setdefault(obs['cam'], set()).add(gid)
                self._cam_tid_to_gid[(obs['cam'], obs['tid'])] = gid
                matched_obs.add(r)
            leftovers = [o for i, o in enumerate(leftovers) if i not in matched_obs]

        # 3) cross-view cluster the remaining leftovers (co-located observations from
        #    different cameras are the same physical player), one new track per cluster.
        clusters: list[dict] = []
        for obs in leftovers:
            g = np.asarray(obs['ground'], dtype=np.float64)
            best, best_d = None, self._gate
            for cl in clusters:
                if obs['cam'] in cl['cams']:
                    continue
                d = float(np.linalg.norm(g - cl['pos']))
                if d < best_d:
                    best, best_d = cl, d
            if best is None:
                clusters.append({'pos': g.copy(), 'cams': {obs['cam']}, 'obs': [obs]})
            else:
                best['obs'].append(obs)
                best['cams'].add(obs['cam'])
                best['pos'] = np.mean([np.asarray(o['ground'], dtype=np.float64)
                                       for o in best['obs']], axis=0)
        for cl in clusters:
            gid = self._next_gid
            self._next_gid += 1
            self._tracks[gid] = _Track(gid, cl['pos'], t)
            assigned[gid] = list(cl['obs'])
            for o in cl['obs']:
                self._cam_tid_to_gid[(o['cam'], o['tid'])] = gid

        # 4) aggregate per gid
        for gid, obs_list in assigned.items():
            tr = self._tracks[gid]
            cams = {o['cam'] for o in obs_list}
            mean_ground = np.mean([np.asarray(o['ground'], dtype=np.float64) for o in obs_list], axis=0)
            dt = max(1e-3, t - tr.last_t)
            resid = mean_ground - (tr.pos + tr.vel * dt)
            tr.pos = tr.pos + tr.vel * dt + self._alpha * resid
            tr.vel = tr.vel + (self._beta / dt) * resid
            tr.last_t = t
            tr.miss = 0
            tr.seen_frames += 1
            tr.nviews_last = len(cams)
            if len(cams) >= self._min_views:
                tr.ever_multiview = True
            for o in obs_list:
                c = int(o.get('cls', -1))
                if 0 <= c < 4:
                    tr.votes[c] += float(o.get('conf', 1.0))
            tr.per_cam = {o['cam']: o['tid'] for o in obs_list}

        # 5) age + retire unmatched tracks
        for gid in list(self._tracks):
            if gid not in assigned:
                tr = self._tracks[gid]
                tr.miss += 1
                if (t - tr.last_t) > self._retire:
                    self._drop(gid)

    def _drop(self, gid: int) -> None:
        self._tracks.pop(gid, None)
        for key in [k for k, v in self._cam_tid_to_gid.items() if v == gid]:
            self._cam_tid_to_gid.pop(key, None)

    def players(self) -> list[dict]:
        '''Confirmed + provisional tracks as world-state player stubs (no 3D kpts yet).'''
        out = []
        for tr in self._tracks.values():
            if tr.miss > 0:
                continue                    # only tracks observed this frame
            votes = tr.votes
            is_ref = votes[3] > max(votes[0], votes[1], votes[2]) and votes[3] > 0
            is_gk = (not is_ref) and votes[2] > max(votes[0], votes[1]) and votes[2] > 0
            team = int(np.argmax(votes[:2])) if max(votes[:2]) > 0 else None
            provisional = (not tr.ever_multiview) and tr.seen_frames < self._prov_frames
            out.append({
                'gid': tr.gid, 'team': team, 'is_gk': bool(is_gk), 'is_ref': bool(is_ref),
                'provisional': bool(provisional), 'ground': tr.pos.tolist(),
                'vel': tr.vel.tolist(), 'per_cam': dict(tr.per_cam),
                'nviews': tr.nviews_last,
            })
        return out
