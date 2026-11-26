"""Explain the errors behind quick_check's numbers.

    python diagnose.py --gt data/test_nights/ground_truth_test.csv --events output_test/events.csv

Prints:
  - every false positive: is it a class confusion (overlaps a GT event of another class) or a "ghost"
    (no GT event at all -> background noise triggered it)? plus YAMNet's top label (needs --extended)
  - every miss: its SNR and whether something of another class was detected there instead
  - confusion counts (true class -> detected class) and recall per SNR level
"""
import argparse
import csv
import os
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from quick_check import match  # noqa: E402


def load(path):
    rows = list(csv.DictReader(open(path, newline="")))
    for r in rows:
        r["start_s"] = float(r.get("start_s", r.get("start")))
        r["end_s"] = float(r.get("end_s", r.get("end")))
        r["confidence"] = float(r.get("confidence") or 1.0)
    return rows


def overlaps(a, b):
    return a["night_id"] == b["night_id"] and a["start_s"] < b["end_s"] and b["start_s"] < a["end_s"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True)
    ap.add_argument("--events", required=True)
    ap.add_argument("--collar", type=float, default=1.0)
    ap.add_argument("--criterion", choices=["onset", "overlap"], default="onset")
    ap.add_argument("--summary", action="store_true", help="only counts, no per-event lines")
    a = ap.parse_args()

    gt, ev = load(a.gt), load(a.events)
    _, pairs = match(gt, ev, a.collar, criterion=a.criterion)
    hit_g = {id(g) for g, _ in pairs}
    hit_p = {id(p) for _, p in pairs}

    conf = Counter()
    print("=== FALSE POSITIVES ===")
    kinds, kinds_cls = Counter(), Counter()
    for p in sorted(ev, key=lambda e: e["start_s"]):
        if id(p) in hit_p:
            continue
        over = [g for g in gt if overlaps(p, g)]
        same = [g for g in over if g["class"] == p["class"]]
        if same and a.criterion == "overlap":
            kind, why = "fragment", "extra piece of an already-matched event (split detection)"
        elif same:
            kind, why = "timing", f"same class but onset off by {p['start_s'] - same[0]['start_s']:+.2f} s (> collar)"
        elif over:
            kind, why = "confusion", f"overlaps GT {over[0]['class']} ({over[0].get('source_file', '')})"
            conf[(over[0]["class"], p["class"])] += 1
        else:
            kind, why = "ghost", "no GT event here (background)"
        kinds[kind] += 1
        kinds_cls[(p["class"], kind)] += 1
        if not a.summary:
            print(f"{p['start_s']:8.2f}-{p['end_s']:8.2f}  {p['class']:9s} conf {p['confidence']:.2f}  "
                  f"top: {p.get('top_label', '?'):28s} -> {kind}: {why}")
    print("FP breakdown:", dict(kinds))
    for c in sorted({c for c, _ in kinds_cls}):
        print(f"  {c:9s} " + "  ".join(f"{k}={n}" for (cc, k), n in sorted(kinds_cls.items()) if cc == c))

    print("\n=== MISSES ===")
    by_snr = defaultdict(lambda: [0, 0])
    miss_kinds = Counter()
    for g in gt:
        snr = g.get("snr_db", "?")
        by_snr[snr][1] += 1
        if id(g) in hit_g:
            by_snr[snr][0] += 1
            continue
        other = [p for p in ev if overlaps(p, g) and p["class"] != g["class"]]
        same = [p for p in ev if overlaps(p, g) and p["class"] == g["class"]]
        mk = "late/early onset" if same else (f"as {other[0]['class']}" if other else "nothing")
        miss_kinds[(g["class"], mk)] += 1
        if same:
            note = f"detected but onset off by {same[0]['start_s'] - g['start_s']:+.2f} s"
        elif other:
            note = f"detected as {other[0]['class']} instead"
        else:
            note = "nothing detected"
        if not a.summary:
            print(f"{g['start_s']:8.2f}-{g['end_s']:8.2f}  {g['class']:9s} snr {snr:>5s} dB  "
                  f"{g.get('source_file', ''):22s} -> {note}")

    print("Miss breakdown:")
    for c in sorted({c for c, _ in miss_kinds}):
        print(f"  {c:9s} " + "  ".join(f"{k}={n}" for (cc, k), n in sorted(miss_kinds.items()) if cc == c))

    print("\n=== CONFUSIONS (true -> detected) ===")
    for (t, d), n in conf.most_common():
        print(f"{t:9s} -> {d:9s} {n}")

    print("\n=== RECALL BY SNR ===")
    def key(s):
        try:
            return float(s)
        except ValueError:
            return 1e9
    for snr in sorted(by_snr, key=key):
        h, n = by_snr[snr]
        print(f"snr {snr:>5s} dB: {h}/{n} = {h / n:.2f}")


if __name__ == "__main__":
    main()
