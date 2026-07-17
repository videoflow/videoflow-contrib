'''
Estimate per-camera time offsets from shared ambient audio (cross-correlation).

For each camera we extract a mono audio track, build an onset-novelty envelope
(robust to per-camera gain/distance), cross-correlate against the reference
camera for a coarse lag, then refine on the band-passed waveform for sub-ms
accuracy. Writes ``offsets.json`` = ``{cam: {offset_s, drift_ppm, confidence}}``
where ``t_reference = t_cam + offset_s``.

    python sync_offsets.py --config config.yaml
'''
from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile

import numpy as np
from common import load_config
from scipy.io import wavfile
from scipy.signal import butter, correlate, sosfiltfilt

SR = 8000
HOP = 80          # 10 ms at 8 kHz


def extract_audio(video_path: str, out_wav: str) -> None:
    subprocess.run(['ffmpeg', '-y', '-i', video_path, '-vn', '-ac', '1', '-ar', str(SR),
                    '-f', 'wav', out_wav], check=True, capture_output=True)


def bandpass(x: np.ndarray) -> np.ndarray:
    sos = butter(4, [300, 3000], btype='band', fs=SR, output='sos')
    return sosfiltfilt(sos, x)


def onset_envelope(x: np.ndarray) -> np.ndarray:
    xb = bandpass(x)
    n = len(xb) // HOP
    energy = np.array([np.log(np.sum(xb[i * HOP:(i + 1) * HOP] ** 2) + 1e-9) for i in range(n)])
    onset = np.maximum(0.0, np.diff(energy, prepend=energy[0]))
    return (onset - onset.mean()) / (onset.std() + 1e-9)


def coarse_lag(ref_env: np.ndarray, env: np.ndarray) -> tuple[int, float]:
    '''Lag (in hops) maximizing correlation, plus a peak/second-peak confidence.'''
    corr = correlate(ref_env, env, mode='full')
    lags = np.arange(-len(env) + 1, len(ref_env))
    peak = int(np.argmax(corr))
    lag = int(lags[peak])
    hi = corr[peak]
    mask = np.abs(np.arange(len(corr)) - peak) > 5
    second = np.max(corr[mask]) if mask.any() else 0.0
    conf = float(hi / (second + 1e-9))
    return lag, conf


def refine_waveform(ref: np.ndarray, x: np.ndarray, coarse_lag_samples: int, win_ms: int = 50) -> int:
    '''Refine the coarse lag by correlating raw band-passed waveforms in a window.'''
    rb, xb = bandpass(ref), bandpass(x)
    w = int(win_ms * SR / 1000)
    c0 = len(xb)
    lo, hi = max(0, c0 + coarse_lag_samples - w), c0 + coarse_lag_samples + w
    corr = correlate(rb, xb, mode='full')
    seg = corr[lo:hi]
    if seg.size == 0:
        return coarse_lag_samples
    lags = np.arange(-len(xb) + 1, len(rb))
    return int(lags[lo + int(np.argmax(seg))])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)

    with tempfile.TemporaryDirectory() as tmp:
        audio, envs = {}, {}
        for cam in cfg.cameras:
            wav = os.path.join(tmp, f'{cam}.wav')
            extract_audio(cfg.videos[cam], wav)
            sr, data = wavfile.read(wav)
            x = data.astype(np.float64)
            if x.ndim > 1:
                x = x.mean(axis=1)
            x /= (np.max(np.abs(x)) + 1e-9)
            audio[cam] = x
            envs[cam] = onset_envelope(x)

        ref = cfg.cameras[0]
        offsets = {ref: {'offset_s': 0.0, 'drift_ppm': 0.0, 'confidence': 999.0}}
        for cam in cfg.cameras[1:]:
            lag_hops, conf = coarse_lag(envs[ref], envs[cam])
            lag_samples = refine_waveform(audio[ref], audio[cam], lag_hops * HOP)
            offset_s = lag_samples / SR
            offsets[cam] = {'offset_s': float(offset_s), 'drift_ppm': 0.0,
                            'confidence': round(conf, 2)}
            flag = '' if conf >= 3.0 else '   [LOW CONFIDENCE — check for a shared clap/whistle]'
            print(f'{cam}: offset {offset_s:+.4f} s  (confidence {conf:.1f}){flag}')

    with open(cfg.offsets_path(), 'w') as f:
        json.dump(offsets, f, indent=2)
    print(f'\nWrote {cfg.offsets_path()}')


if __name__ == '__main__':
    main()
