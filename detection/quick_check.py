"""Sanity check + threshold tuning for the detector against ground truth.

(Person D owns the real evaluation; this is for B to tune quickly.)

Score an events CSV:
    python detection/quick_check.py --gt data/ground_truth.csv --events detection/output/events.csv

Sweep thresholds on cached YAMNet frames (no re-inference) and save the best ones:
    python detection/quick_check.py --gt data/ground_truth.csv --frames detection/output/frames \
        --sweep --nights night_01 --write-thresholds detection/thresholds.json
Tune on one or two nights only, then report results on the others.

Matching: same night, same class, |onset difference| <= collar seconds, one-to-one,
highest-confidence detections matched first.
GT CSV needs night_id, class, start_s/start, end_s/end (other columns are ignored).
"""
import argparse
import csv
import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from classes import OUR_CLASSES  # noqa: E402


def read_rows(path):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    out = []
    for r in rows:
        out.append({
            "night_id": r["night_id"], "class": r["class"],
            "start_s": float(r.get("start_s", r.get("start"))),
            "end_s": float(r.get("end_s", r.get("end"))),
            "confidence": float(r.get("confidence", 1.0) or 1.0),
        })
    return out


def match(gt, pred, collar=1.0, nights=None, criterion="onset"):
    """Returns per-class stats dict and a list of (gt, pred) matched pairs.

    criterion="onset":   |onset difference| <= collar   (strict, DCASE-style)
    criterion="overlap": any time overlap with a same-class GT event (did we find it at all?)
    """
    by_g, by_p = defaultdict(list), defaultdict(list)
    for g in gt:
        if nights is None or g["night_id"] in nights:
            by_g[(g["night_id"], g["class"])].append(g)
    for p in pred:
        if nights is None or p["night_id"] in nights:
            by_p[(p["night_id"], p["class"])].append(p)
    stats = {c: dict(tp=0, fp=0, fn=0) for c in OUR_CLASSES}
    pairs = []
    for key in set(by_g) | set(by_p):
        c = key[1]
        if c not in stats:
            continue
        G, P = by_g.get(key, []), sorted(by_p.get(key, []), key=lambda e: -e["confidence"])
        used = np.zeros(len(G), bool)
        onsets = np.array([g["start_s"] for g in G])
        gs = np.array([g["start_s"] for g in G])
        ge = np.array([g["end_s"] for g in G])
        for p in P:
            if len(G):
                d = np.abs(onsets - p["start_s"])
                if criterion == "overlap":
                    ok = (gs < p["end_s"]) & (p["start_s"] < ge)
                    d = np.where(ok, d, np.inf)
                    lim = np.inf
                else:
                    lim = collar
                d[used] = np.inf
                j = int(d.argmin())
                if np.isfinite(d[j]) and d[j] <= lim:
                    used[j] = True
                    pairs.append((G[j], p))
                    stats[c]["tp"] += 1
                    continue
            stats[c]["fp"] += 1
        stats[c]["fn"] += int((~used).sum())
    return stats, pairs


def prf(s):
    p = s["tp"] / (s["tp"] + s["fp"]) if s["tp"] + s["fp"] else 0.0
    r = s["tp"] / (s["tp"] + s["fn"]) if s["tp"] + s["fn"] else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def report(stats, pairs):
    print(f"{'class':10s} {'TP':>4s} {'FP':>4s} {'FN':>4s} {'prec':>6s} {'rec':>6s} {'F1':>6s}")
    tot = dict(tp=0, fp=0, fn=0)
    for c in OUR_CLASSES:
        s = stats[c]
        for k in tot:
            tot[k] += s[k]
        p, r, f = prf(s)
        print(f"{c:10s} {s['tp']:4d} {s['fp']:4d} {s['fn']:4d} {p:6.3f} {r:6.3f} {f:6.3f}")
    p, r, f = prf(tot)
    print(f"{'micro':10s} {tot['tp']:4d} {tot['fp']:4d} {tot['fn']:4d} {p:6.3f} {r:6.3f} {f:6.3f}")
    if pairs:
        on = np.array([p["start_s"] - g["start_s"] for g, p in pairs])
        off = np.array([p["end_s"] - g["end_s"] for g, p in pairs])
        print(f"onset error: mean |e| {np.abs(on).mean():.3f} s (bias {on.mean():+.3f}); "
              f"offset error: mean |e| {np.abs(off).mean():.3f} s (bias {off.mean():+.3f})")


def sweep(gt, frames_dir, nights, collar, base_cfg, grid, criterion="onset"):
    from detect import load_frames
    from events import class_index_map, detect_events

    files = sorted(glob.glob(os.path.join(frames_dir, "*.npz")))
    loaded = []
    for fpath in files:
        nid = os.path.splitext(os.path.basename(fpath))[0]
        if nights is None or nid in nights:
            fr, names = load_frames(fpath)
            fr.pop("embeddings", None)
            loaded.append((nid, fr, names, class_index_map(names, warn=False)))
    if not loaded:
        sys.exit("no cached frames matched; run detect.py first")
    use = {n for n, *_ in loaded}
    table = {c: [] for c in OUR_CLASSES}
    for t in grid:
        base_cfg.thresholds = {c: float(t) for c in OUR_CLASSES}
        pred = []
        for nid, fr, names, cmap in loaded:
            evs, _ = detect_events(fr, names, base_cfg, cmap)
            pred += [dict(night_id=nid, **{k: e[k] for k in ("class", "start_s", "end_s", "confidence")}) for e in evs]
        stats, _ = match(gt, pred, collar, use, criterion)
        for c in OUR_CLASSES:
            table[c].append((t, *prf(stats[c])))
    from classes import DEFAULT_THRESHOLDS
    n_gt = defaultdict(int)
    for g in gt:
        if g["night_id"] in use:
            n_gt[g["class"]] += 1
    best = {}
    print(f"tuned on: {sorted(use)}")
    for c in OUR_CLASSES:
        if n_gt[c] == 0:
            best[c] = DEFAULT_THRESHOLDS[c]
            print(f"{c:10s} no GT events in tuning nights -> keeping default {best[c]:.3f}")
            continue
        top = max(x[3] for x in table[c])
        tied = [x for x in table[c] if x[3] >= top - 1e-9]
        t, p, r, f = tied[len(tied) // 2]          # middle of the best plateau = more robust
        best[c] = round(float(t), 3)
        print(f"{c:10s} best thr {t:.3f}  P {p:.3f} R {r:.3f} F1 {f:.3f}  (n_gt={n_gt[c]})")
    return best, table


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--events")
    ap.add_argument("--frames", help="detection/output/frames (for --sweep)")
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--nights", nargs="*", help="restrict to these night_ids")
    ap.add_argument("--collar", type=float, default=1.0)
    ap.add_argument("--criterion", choices=["onset", "overlap"], default="onset",
                    help="onset: |onset diff|<=collar; overlap: any same-class overlap counts as found")
    ap.add_argument("--write-thresholds")
    ap.add_argument("--refine", action="store_true")
    ap.add_argument("--exclusive", action="store_true")
    ap.add_argument("--nms", action="store_true")
    ap.add_argument("--class-gap", nargs="*", metavar="CLASS=SEC", help="same as detect.py (for --sweep)")
    a = ap.parse_args()

    gt = read_rows(a.gt)
    nights = set(a.nights) if a.nights else None
    if a.sweep:
        from events import Config, parse_class_values
        cfg = Config(refine=a.refine, exclusive=a.exclusive, nms=a.nms, class_gaps=parse_class_values(a.class_gap))
        best, _ = sweep(gt, a.frames, nights, a.collar, cfg,
                        np.round(np.arange(0.05, 0.701, 0.025), 3), a.criterion)
        if a.write_thresholds:
            with open(a.write_thresholds, "w") as f:
                json.dump(best, f, indent=2)
            print(f"wrote {a.write_thresholds}")
    else:
        stats, pairs = match(gt, read_rows(a.events), a.collar, nights, a.criterion)
        report(stats, pairs)


if __name__ == "__main__":
    main()
