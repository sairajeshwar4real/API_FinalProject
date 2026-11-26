"""Quick test night from ESC-50 so Person B can work before A's generator is finished.

    python detection/make_test_night.py --esc50 ~/data/ESC-50-master --minutes 20 --events 40 \
        --out data/test_nights --seed 1

Writes <out>/night_test_01.wav (16 kHz mono) and <out>/ground_truth_test.csv with
    night_id, start_s, end_s, class, snr_db, level_db, source_file
start_s/end_s are the *audible* part of the clip (leading/trailing silence trimmed),
which is what timing error should be measured against. Worth copying into A's generator.
"""
import argparse
import csv
import os
import sys
from math import gcd

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from classes import DATASET_TO_OURS  # noqa: E402

SR = 16000


def rms_db(x):
    return 10 * np.log10(np.mean(x ** 2) + 1e-12)


def trim(x, rel_db=-20.0, above_floor_db=10.0, block=320, smooth=5):
    """Cut leading/trailing parts that are only the clip's own recording noise.

    Active = smoothed 20 ms level above BOTH (peak + rel_db) and (clip noise floor + above_floor_db).
    ESC-50 clips contain recording noise, so a peak-relative rule alone keeps the whole 5 s.
    """
    n = len(x) // block
    if n == 0:
        return x, 0
    e = np.mean(x[: n * block].reshape(n, block) ** 2, axis=1)
    e = 10 * np.log10(np.convolve(e, np.ones(smooth) / smooth, mode="same") + 1e-12)
    thr = max(e.max() + rel_db, np.percentile(e, 10) + above_floor_db)
    on = np.where(e > thr)[0]
    if len(on) == 0:
        return x, 0
    a, b = on[0] * block, (on[-1] + 1) * block
    return x[a:b], a


def pink_noise(n, rng):
    """1/f noise: flatter than brown, closer to a quiet bedroom (fan/ventilation hum)."""
    X = np.fft.rfft(rng.standard_normal(n))
    f = np.arange(len(X), dtype=np.float64)
    f[0] = 1.0
    x = np.fft.irfft(X / np.sqrt(f), n)
    return (x / (np.std(x) + 1e-9)).astype(np.float32)


def background(kind, n, rng):
    if kind == "pink":
        return pink_noise(n, rng)
    if kind == "brown":
        return brown_noise(n, rng)
    x, sr = sf.read(kind, dtype="float32", always_2d=True)       # a real room recording
    x = x.mean(axis=1)
    if sr != SR:
        g = gcd(SR, sr)
        x = resample_poly(x, SR // g, sr // g).astype(np.float32)
    x = np.tile(x, int(np.ceil(n / len(x))))[:n]
    return (x / (np.sqrt(np.mean(x ** 2)) + 1e-9)).astype(np.float32)


def brown_noise(n, rng):
    w = rng.standard_normal(n).astype(np.float64)
    b = np.cumsum(w)
    b -= np.convolve(b, np.ones(SR) / SR, mode="same")   # remove drift
    return (b / (np.std(b) + 1e-9)).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--esc50", required=True, help="ESC-50-master folder (has audio/ and meta/esc50.csv)")
    ap.add_argument("--out", default="data/test_nights")
    ap.add_argument("--night-id", default="night_test_01")
    ap.add_argument("--minutes", type=float, default=20)
    ap.add_argument("--events", type=int, default=40)
    ap.add_argument("--floor-db", type=float, default=-55.0, help="background RMS in dBFS")
    ap.add_argument("--background", default="pink", help="pink | brown | path/to/room_recording.wav")
    ap.add_argument("--snrs", type=float, nargs="+", default=[-5, 0, 5, 10, 15, 20])
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    rng = np.random.default_rng(a.seed)
    mapping = DATASET_TO_OURS["esc50"]
    with open(os.path.join(a.esc50, "meta", "esc50.csv"), newline="") as f:
        clips = [(r["filename"], mapping[r["category"]]) for r in csv.DictReader(f) if r["category"] in mapping]

    n = int(a.minutes * 60 * SR)
    night = background(a.background, n, rng) * (10 ** (a.floor_db / 20))
    slots = np.sort(rng.choice(np.arange(5, a.minutes * 60 - 10, 6.0), size=a.events, replace=False))
    rows = []
    for t in slots:
        fname, cls = clips[rng.integers(len(clips))]
        x, sr = sf.read(os.path.join(a.esc50, "audio", fname), dtype="float32", always_2d=True)
        x = x.mean(axis=1)
        if sr != SR:
            g = gcd(SR, sr)
            x = resample_poly(x, SR // g, sr // g).astype(np.float32)
        x, _ = trim(x)
        if len(x) < SR // 10:
            continue
        snr = float(rng.choice(a.snrs))
        x = x * 10 ** ((a.floor_db + snr - rms_db(x)) / 20)
        t = float(t + rng.uniform(0, 1))
        i = int(t * SR)
        x = x[: n - i]
        night[i:i + len(x)] += x
        rows.append(dict(night_id=a.night_id, start_s=round(t, 3), end_s=round(t + len(x) / SR, 3),
                         **{"class": cls}, snr_db=snr, level_db=round(a.floor_db + snr, 2), source_file=fname))

    peak = np.abs(night).max()
    if peak > 0.99:
        night *= 0.99 / peak
        print(f"warning: clipped, scaled by {0.99 / peak:.3f} (levels in GT now off by that factor)")
    os.makedirs(a.out, exist_ok=True)
    sf.write(os.path.join(a.out, f"{a.night_id}.wav"), night, SR, subtype="PCM_16")
    gt_path = os.path.join(a.out, "ground_truth_test.csv")
    with open(gt_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {a.night_id}.wav ({a.minutes} min, {len(rows)} events) and {gt_path}")


if __name__ == "__main__":
    main()
