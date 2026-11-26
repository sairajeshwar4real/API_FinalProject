"""End-to-end test with a fake YAMNet (no TensorFlow needed).

    python detection/tests/test_pipeline.py [path/to/yamnet_class_map.csv]

The fake model uses YAMNet's exact framing (0.975 s window, 0.48 s hop, end padding)
and "detects" pure tones: 500 Hz -> Bark, 2000 Hz -> Siren, 1000 Hz -> Knock, 120 Hz -> Snoring.
Checks: chunked inference == whole-file inference, events found at the right times,
CSV follows the contract, embeddings load, resampling path (44.1 kHz stereo) works.
"""
import csv
import os
import shutil
import sys
import tempfile

import numpy as np
import soundfile as sf

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import detect  # noqa: E402
import quick_check  # noqa: E402
from yamnet_runner import HOP, SR, WIN, read_class_names, run_night  # noqa: E402

TONES = {500: ("Bark", "dog_bark"), 2000: ("Siren", "alarm"), 1000: ("Knock", "door"), 120: ("Snoring", "snoring")}


def make_fake(names):
    idx = {n: i for i, n in enumerate(names)}
    proj = np.random.default_rng(0).standard_normal((WIN // 2 + 1, 1024)).astype(np.float32) / 100
    freqs = np.fft.rfftfreq(WIN, 1 / SR)

    def run(x):
        n = len(x)
        frames = 1 + int(np.ceil(max(0, n - WIN) / HOP))
        x = np.pad(x, (0, (frames - 1) * HOP + WIN - n))
        S = np.zeros((frames, len(names)), np.float32)
        E = np.zeros((frames, 1024), np.float32)
        for i in range(frames):
            spec = np.abs(np.fft.rfft(x[i * HOP:i * HOP + WIN] * np.hanning(WIN))) ** 2
            tot = spec.sum() + 1e-12
            for f, (label, _) in TONES.items():
                band = (freqs > f - 15) & (freqs < f + 15)
                S[i, idx[label]] = min(1.0, 1.2 * spec[band].sum() / tot)
            E[i] = np.log1p(spec) @ proj
        return S, E

    return run


def make_night(path, minutes, sr, channels, rng, floor_db=-50):
    n = int(minutes * 60 * sr)
    x = rng.standard_normal((n, channels)).astype(np.float32) * 10 ** (floor_db / 20)
    gt, t = [], 4.3
    freqs = list(TONES)
    while t < minutes * 60 - 8:
        f = freqs[len(gt) % len(freqs)]
        dur = float(rng.uniform(1.0, 5.0))
        snr = float(rng.choice([0, 5, 10, 20]))
        i0, i1 = int(t * sr), int((t + dur) * sr)
        tt = np.arange(i1 - i0) / sr
        tone = np.sin(2 * np.pi * f * tt) * np.sqrt(2) * 10 ** ((floor_db + snr) / 20)
        x[i0:i1] += tone[:, None].astype(np.float32)
        gt.append(dict(night_id=os.path.splitext(os.path.basename(path))[0], start_s=t, end_s=t + dur,
                       **{"class": TONES[f][1]}, snr_db=snr))
        t += dur + float(rng.uniform(6, 30))
    sf.write(path, x if channels > 1 else x[:, 0], sr, subtype="FLOAT")
    return gt


def main():
    cmap_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "yamnet_class_map.csv")
    names = read_class_names(cmap_path)
    fake = make_fake(names)
    rng = np.random.default_rng(42)
    tmp = tempfile.mkdtemp()
    try:
        nights = os.path.join(tmp, "nights")
        os.makedirs(nights)
        gt = make_night(os.path.join(nights, "night_01.wav"), 30, 16000, 1, rng)
        gt += make_night(os.path.join(nights, "night_02.wav"), 12, 44100, 2, rng)
        gt_csv = os.path.join(tmp, "gt.csv")
        with open(gt_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(gt[0].keys()))
            w.writeheader()
            w.writerows(gt)

        # 1. chunked == whole-file
        x, _ = sf.read(os.path.join(nights, "night_01.wav"), dtype="float32")
        s_full, e_full = fake(x)
        fr = run_night(os.path.join(nights, "night_01.wav"), fake, chunk_frames=64, verbose=False)
        assert fr["scores"].shape == s_full.shape, (fr["scores"].shape, s_full.shape)
        assert np.allclose(fr["scores"], s_full, atol=1e-5)
        assert np.allclose(fr["embeddings"], e_full, atol=1e-3)
        print(f"OK chunked inference matches whole-file ({s_full.shape[0]} frames)")

        # 2. full CLI, plain and refined
        out = os.path.join(tmp, "out")
        for extra, tag in [([], "center"), (["--refine"], "refined")]:
            csv_path = detect.main(["--nights", nights, "--out", out, "--chunk-frames", "64"] + extra,
                                   run_fn=fake, class_names=names)
            rows = list(csv.DictReader(open(csv_path)))
            assert list(rows[0].keys()) == detect.CONTRACT, rows[0].keys()
            emb = np.load(os.path.join(out, rows[0]["embedding_file"]))
            assert emb.shape == (1024,)
            print(f"\n--- {tag}: {len(rows)} events vs {len(gt)} GT ---")
            stats, pairs = quick_check.match(quick_check.read_rows(gt_csv), quick_check.read_rows(csv_path), 1.0)
            quick_check.report(stats, pairs)
            tp = sum(s["tp"] for s in stats.values())
            assert tp >= 0.9 * len(gt), "too many misses"

        # 3. extended columns + threshold sweep on cached frames
        detect.main(["--nights", nights, "--out", out, "--extended"], run_fn=fake, class_names=names)
        from events import Config
        best, _ = quick_check.sweep(quick_check.read_rows(gt_csv), os.path.join(out, "frames"),
                                    {"night_01"}, 1.0, Config(), np.arange(0.05, 0.7, 0.05))
        print("sweep OK:", best)
        print("\nALL TESTS PASSED")
    finally:
        shutil.rmtree(tmp)


if __name__ == "__main__":
    main()
