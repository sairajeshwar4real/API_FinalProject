#!/usr/bin/env python3
"""
make_nights.py - Synthetic "night" generator (Person A).

Mixes short public sound clips (ESC-50, optionally UrbanSound8K) into long,
quiet background recordings at KNOWN times and loudness, and writes a
ground-truth CSV so detection accuracy can be measured exactly.

Usage:
    pip install numpy scipy soundfile pandas matplotlib
    python make_nights.py --download                       # fetch ESC-50 (~600 MB)
    python make_nights.py --esc50 data/ESC-50-master       # 6 nights x 1 hour
    python make_nights.py --esc50 data/ESC-50-master --urbansound data/UrbanSound8K \
                          --nights 8 --hours 2 --out out

Outputs (in --out):
    nights/night_01.wav ...      16 kHz mono 16-bit
    ground_truth.csv             all events of all nights
    ground_truth/night_01.csv    per-night events
    plots/night_01_timeline.png  quick visual for slides
    manifest.json                settings used (for the report)
"""
import argparse
import json
import sys
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import lfilter, resample_poly

SR = 16000
ESC50_URL = "https://github.com/karolpiczak/ESC-50/archive/master.zip"

# source class name -> (unified class, group). Unified names let both datasets share a label.
ESC50_MAP = {
    "dog": ("dog", "pets"), "cat": ("cat", "pets"), "rooster": ("rooster", "pets"),
    "door_wood_knock": ("door_knock", "door"), "door_wood_creaks": ("door_creak", "door"),
    "car_horn": ("car_horn", "traffic"), "siren": ("siren", "traffic"),
    "engine": ("engine", "traffic"), "train": ("train", "traffic"),
    "airplane": ("airplane", "traffic"), "helicopter": ("helicopter", "traffic"),
    "clock_alarm": ("alarm", "alarm"),
    "snoring": ("snoring", "body"), "coughing": ("cough", "body"), "sneezing": ("sneeze", "body"),
    "crying_baby": ("crying_baby", "voices"), "laughing": ("laughing", "voices"),
    "footsteps": ("footsteps", "household"), "toilet_flush": ("toilet_flush", "household"),
    "vacuum_cleaner": ("vacuum_cleaner", "household"),
    "rain": ("rain", "weather"), "wind": ("wind", "weather"), "thunderstorm": ("thunderstorm", "weather"),
}
US8K_MAP = {
    "car_horn": ("car_horn", "traffic"), "dog_bark": ("dog", "pets"),
    "engine_idling": ("engine", "traffic"), "siren": ("siren", "traffic"),
    "children_playing": ("children", "voices"), "street_music": ("street_music", "music"),
}


# ----------------------------------------------------------------- download
def download_esc50(dest="data"):
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    zpath = dest / "ESC-50.zip"
    if not zpath.exists():
        print(f"Downloading ESC-50 to {zpath} ...")

        def hook(b, bs, total):
            if total > 0:
                print(f"\r  {min(100, b * bs * 100 // total)}%", end="", flush=True)

        urllib.request.urlretrieve(ESC50_URL, zpath, hook)
        print()
    print("Extracting audio + metadata ...")
    with zipfile.ZipFile(zpath) as z:
        members = [m for m in z.namelist() if "/audio/" in m or "/meta/" in m]
        z.extractall(dest, members=members)
    root = dest / "ESC-50-master"
    print(f"Done. Use: --esc50 {root}")
    return root


# ----------------------------------------------------------------- clips
def load_clip(path):
    x, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if x.ndim > 1:
        x = x.mean(axis=1)
    if sr != SR:
        g = int(np.gcd(sr, SR))
        x = resample_poly(x, SR // g, sr // g).astype(np.float32)
    return x


def trim_silence(x, rel_db=-40.0, frame=320):
    """ESC-50 pads short sounds with silence; trim it so start/end times are honest."""
    n = len(x) // frame
    if n == 0:
        return x
    rms = np.sqrt(np.mean(x[: n * frame].reshape(n, frame) ** 2, axis=1))
    if rms.max() < 1e-6:
        return x[:0]
    active = np.where(rms > rms.max() * 10 ** (rel_db / 20))[0]
    return x[active[0] * frame: (active[-1] + 1) * frame]


def build_catalog(esc50, urbansound):
    """Return {unified_class: {"group": str, "files": [(dataset, path), ...]}}"""
    cat = {}

    def add(unified, group, ds, path):
        cat.setdefault(unified, {"group": group, "files": []})["files"].append((ds, str(path)))

    if esc50:
        root = Path(esc50)
        meta = pd.read_csv(root / "meta" / "esc50.csv")
        for _, r in meta.iterrows():
            if r["category"] in ESC50_MAP:
                u, g = ESC50_MAP[r["category"]]
                add(u, g, "ESC-50", root / "audio" / r["filename"])
    if urbansound:
        root = Path(urbansound)
        meta = pd.read_csv(root / "metadata" / "UrbanSound8K.csv")
        for _, r in meta.iterrows():
            if r["class"] in US8K_MAP:
                u, g = US8K_MAP[r["class"]]
                add(u, g, "UrbanSound8K", root / "audio" / f"fold{r['fold']}" / r["slice_file_name"])
    if not cat:
        sys.exit("No clips found. Check --esc50 / --urbansound paths.")
    return cat


# ----------------------------------------------------------------- background
def pink_noise(n, rng, target_rms_db):
    """Quiet 'room tone': pink-ish noise, generated in chunks (streaming filter)."""
    b = [0.049922035, -0.095993537, 0.050612699, -0.004408786]
    a = [1, -2.494956002, 2.017265875, -0.522189400]
    out = np.empty(n, dtype=np.float32)
    zi = np.zeros(max(len(a), len(b)) - 1)
    chunk = SR * 300
    for s in range(0, n, chunk):
        e = min(n, s + chunk)
        y, zi = lfilter(b, a, rng.standard_normal(e - s), zi=zi)
        out[s:e] = y
    out *= 10 ** (target_rms_db / 20) / np.sqrt(np.mean(out.astype(np.float64) ** 2))
    return out


# ----------------------------------------------------------------- placement
def place_events(lengths, dur, overlap_prob, rng, guard=1.0, margin=5.0):
    """Return one (start, end) per clip, or None if it could not be placed.
    Non-overlapping by default; with overlap_prob an event is dropped on top of
    an existing one (hard case for the detector)."""
    placed, spans = [], []
    for L in lengths:
        span = None
        for _ in range(300):
            overlap_mode = bool(placed) and rng.random() < overlap_prob
            if overlap_mode:
                s0, e0 = placed[rng.integers(len(placed))]
                s = s0 + rng.uniform(0, max(0.1, (e0 - s0) * 0.7))
            else:
                s = rng.uniform(margin, dur - L - margin)
            e = s + L
            if e > dur - margin:
                continue
            if overlap_mode or all(e + guard < ps or s - guard > pe for ps, pe in placed):
                span = (s, e)
                placed.append(span)
                break
        spans.append(span)
    return spans


def fade(x, ms=10):
    k = min(len(x) // 2, int(SR * ms / 1000))
    if k > 0:
        ramp = np.linspace(0, 1, k, dtype=np.float32)
        x = x.copy(); x[:k] *= ramp; x[-k:] *= ramp[::-1]
    return x


# ----------------------------------------------------------------- one night
def make_night(idx, cat, args, rng, rate, overlap_prob):
    dur = args.hours * 3600.0
    n = int(dur * SR)
    night = pink_noise(n, rng, args.bg_dbfs)
    bg_rms = 10 ** (args.bg_dbfs / 20)

    n_events = max(5, int(rate * args.hours))
    classes = list(cat.keys())
    clips, meta = [], []
    for _ in range(n_events):
        c = classes[rng.integers(len(classes))]
        ds, path = cat[c]["files"][rng.integers(len(cat[c]["files"]))]
        x = trim_silence(load_clip(path))
        if len(x) < 0.3 * SR:
            continue
        x = x[: int(args.max_event_s * SR)]
        clips.append(fade(x)); meta.append((c, cat[c]["group"], ds, Path(path).name))

    spans = place_events([len(x) / SR for x in clips], dur, overlap_prob, rng)
    rows = []
    for span, x, (c, g, ds, fn) in zip(spans, clips, meta):
        if span is None:
            continue
        s, e = span
        snr = rng.uniform(args.snr_min, args.snr_max)
        rms = np.sqrt(np.mean(x ** 2)) + 1e-12
        gain = bg_rms * 10 ** (snr / 20) / rms
        peak = np.abs(x).max() * gain
        if peak > 0.95:
            gain *= 0.95 / peak
        y = x * gain
        i0 = int(s * SR); i1 = min(n, i0 + len(y))
        night[i0:i1] += y[: i1 - i0]
        lvl = 20 * np.log10(np.sqrt(np.mean(y ** 2)) + 1e-12)
        rows.append(dict(night_id=f"night_{idx:02d}", start_s=round(i0 / SR, 3),
                         end_s=round(i1 / SR, 3), **{"class": c}, group=g, dataset=ds,
                         source_file=fn, snr_db=round(lvl - args.bg_dbfs, 1),
                         level_dbfs=round(lvl, 1)))

    gt = pd.DataFrame(rows).sort_values("start_s").reset_index(drop=True)
    gt.insert(1, "event_id", range(1, len(gt) + 1))
    # flag overlapping events
    ov = np.zeros(len(gt), dtype=bool)
    for i in range(len(gt)):
        for j in range(i + 1, len(gt)):
            if gt.start_s[j] >= gt.end_s[i]:
                break
            ov[i] = ov[j] = True
    gt["overlaps_other"] = ov
    if np.abs(night).max() > 1.0:
        print("  warning: clipping, lower --snr-max")
    night = np.clip(night, -1.0, 1.0)
    return night, gt


def plot_night(gt, dur_s, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    classes = sorted(gt["class"].unique())
    fig, ax = plt.subplots(figsize=(11, 0.35 * len(classes) + 1.6))
    sc = None
    for _, r in gt.iterrows():
        y = classes.index(r["class"])
        sc = ax.scatter([(r.start_s + r.end_s) / 120], [y], c=[r.snr_db], vmin=0, vmax=40,
                        cmap="viridis", s=40 + 4 * (r.end_s - r.start_s), edgecolor="k", linewidth=0.3)
    ax.set_yticks(range(len(classes))); ax.set_yticklabels(classes)
    ax.set_xlim(0, dur_s / 60); ax.set_xlabel("time (min)")
    ax.set_title(f"{gt.night_id.iloc[0]}: ground-truth events (colour = SNR dB over background)")
    if sc is not None:
        fig.colorbar(sc, ax=ax, label="SNR (dB)")
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


# ----------------------------------------------------------------- main
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--download", action="store_true", help="download + extract ESC-50 into ./data and exit")
    p.add_argument("--esc50", help="path to ESC-50-master")
    p.add_argument("--urbansound", help="path to UrbanSound8K (optional)")
    p.add_argument("--out", default="out")
    p.add_argument("--nights", type=int, default=6)
    p.add_argument("--hours", type=float, default=1.0, help="length of each night")
    p.add_argument("--events-per-hour", type=float, default=40)
    p.add_argument("--bg-dbfs", type=float, default=-50, help="background RMS level")
    p.add_argument("--snr-min", type=float, default=0, help="event level above background, dB")
    p.add_argument("--snr-max", type=float, default=30)
    p.add_argument("--max-event-s", type=float, default=5.0)
    p.add_argument("--hard-overlap-prob", type=float, default=0.3,
                   help="overlap probability on the last 2 'hard' nights")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    if args.download:
        download_esc50()
        return
    if not (args.esc50 or args.urbansound):
        p.error("give --esc50 and/or --urbansound (or run --download first)")

    rng = np.random.default_rng(args.seed)
    cat = build_catalog(args.esc50, args.urbansound)
    print("Classes:", {c: len(v["files"]) for c, v in cat.items()})

    out = Path(args.out)
    for d in ("nights", "ground_truth", "plots"):
        (out / d).mkdir(parents=True, exist_ok=True)

    all_gt = []
    for i in range(1, args.nights + 1):
        hard = args.nights >= 4 and i > args.nights - 2
        rate = args.events_per_hour * rng.uniform(0.6, 1.4)
        ov = args.hard_overlap_prob if hard else 0.0
        print(f"Night {i}/{args.nights}: ~{rate * args.hours:.0f} events, overlap_prob={ov}"
              f"{' (hard night)' if hard else ''}")
        night, gt = make_night(i, cat, args, rng, rate, ov)
        name = f"night_{i:02d}"
        sf.write(out / "nights" / f"{name}.wav", night, SR, subtype="PCM_16")
        gt.to_csv(out / "ground_truth" / f"{name}.csv", index=False)
        plot_night(gt, args.hours * 3600, out / "plots" / f"{name}_timeline.png")
        all_gt.append(gt)
        print(f"  wrote {len(gt)} events, {int(gt.overlaps_other.sum())} overlapping")

    pd.concat(all_gt).to_csv(out / "ground_truth.csv", index=False)
    (out / "manifest.json").write_text(json.dumps(
        {**vars(args), "sample_rate": SR, "classes": sorted(cat.keys())}, indent=2))
    print(f"\nDone. See {out}/ (nights, ground_truth.csv, plots)")


if __name__ == "__main__":
    main()
