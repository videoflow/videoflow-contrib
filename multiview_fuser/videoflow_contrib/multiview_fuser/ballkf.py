'''
Constant-acceleration 3D Kalman filter for the ball, with Mahalanobis gating.

Hand-rolled (numpy only) so gating is explicit and there's no filterpy dependency.
State is [x,y,z, vx,vy,vz, ax,ay,az]; measurements are 3D positions. Large
acceleration process noise keeps kicks (sudden velocity changes) from being
rejected, while gating rejects spurious triangulated detections and lets the
filter coast (predict-only) through short gaps.
'''
from __future__ import annotations

import numpy as np


class BallKalman:
    '''
    - Arguments:
        - process_accel_std: std of the acceleration random walk (m/s²); large so
          kicks aren't gated out.
        - meas_std: measurement (triangulated position) std (m).
        - gate_mahalanobis: chi-square gate on the 3-DOF innovation (9.0 ≈ p<0.03).
    '''
    def __init__(self, process_accel_std: float = 30.0, meas_std: float = 0.15,
                 gate_mahalanobis: float = 9.0, max_reject: int = 3):
        self._qa = float(process_accel_std)
        self._r = float(meas_std)
        self._gate = float(gate_mahalanobis)
        self._max_reject = int(max_reject)     # re-init after this many gated measurements
        self._reject_run = 0
        self.x: np.ndarray | None = None       # (9,) state
        self.P: np.ndarray | None = None       # (9,9) covariance
        self._H = np.zeros((3, 9))
        self._H[0, 0] = self._H[1, 1] = self._H[2, 2] = 1.0

    @property
    def initialized(self) -> bool:
        return self.x is not None

    def _F(self, dt: float) -> np.ndarray:
        F = np.eye(9)
        for i in range(3):
            F[i, 3 + i] = dt
            F[i, 6 + i] = 0.5 * dt * dt
            F[3 + i, 6 + i] = dt
        return F

    def _Q(self, dt: float) -> np.ndarray:
        # White-noise-jerk-ish: acceleration random walk mapped through the CA model.
        q = self._qa ** 2
        Q = np.zeros((9, 9))
        for i in range(3):
            Q[6 + i, 6 + i] = q * dt
            Q[3 + i, 3 + i] = q * dt ** 3 / 3.0
            Q[i, i] = q * dt ** 5 / 20.0
            Q[3 + i, 6 + i] = Q[6 + i, 3 + i] = q * dt ** 2 / 2.0
            Q[i, 3 + i] = Q[3 + i, i] = q * dt ** 4 / 8.0
            Q[i, 6 + i] = Q[6 + i, i] = q * dt ** 3 / 6.0
        return Q

    def init(self, pos: np.ndarray) -> None:
        self.x = np.zeros(9)
        self.x[:3] = np.asarray(pos, dtype=np.float64)
        self.P = np.eye(9)
        self.P[:3, :3] *= self._r ** 2
        self.P[3:6, 3:6] *= 25.0               # velocity prior (up to ~5 m/s std)
        self.P[6:, 6:] *= (self._qa) ** 2

    def predict(self, dt: float) -> np.ndarray:
        if self.x is None:
            raise RuntimeError('BallKalman not initialized')
        F = self._F(dt)
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + self._Q(dt)
        return self.x[:3].copy()

    def update(self, pos: np.ndarray) -> bool:
        '''
        Gated measurement update. Returns True if the measurement was accepted
        (within the Mahalanobis gate), False if rejected (state left as predicted).

        On ``max_reject`` consecutive rejections the filter has diverged from reality
        (e.g. the CA model overshot an abrupt stop) — it re-initializes to the current
        measurement and accepts it, so the ball recovers instead of dropping out.
        '''
        if self.x is None:
            self.init(pos)
            self._reject_run = 0
            return True
        z = np.asarray(pos, dtype=np.float64)
        y = z - self._H @ self.x
        S = self._H @ self.P @ self._H.T + np.eye(3) * self._r ** 2
        try:
            Sinv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            return False
        d2 = float(y @ Sinv @ y)
        if d2 > self._gate:
            self._reject_run += 1
            if self._reject_run >= self._max_reject:
                self.init(pos)                 # diverged — reset to the measurement
                self._reject_run = 0
                return True
            return False
        self._reject_run = 0
        K = self.P @ self._H.T @ Sinv
        self.x = self.x + K @ y
        self.P = (np.eye(9) - K @ self._H) @ self.P
        return True

    def position(self) -> np.ndarray | None:
        return None if self.x is None else self.x[:3].copy()

    def velocity(self) -> np.ndarray | None:
        return None if self.x is None else self.x[3:6].copy()
