"""Turn frame-level YAMNet output into events (pure numpy, no TensorFlow needed).

Steps per night:
  1. collapse 521 AudioSet scores -> our 6 classes (max over the mapped AudioSet classes)
  2. optional "exclusive" mode: each frame keeps only its strongest class
  3. hysteresis thresholding per class: a frame is active if score >= hyst * thr,
     a run of active frames becomes an event only if its peak >= thr
  4. merge runs separated by gaps <= max_gap_s
  5. frame indices -> seconds; optional refinement of start/end on the 20 ms RMS track
  6. confidence = peak score, level_db = mean energy (dBFS) over the event,
     snr_db = level_db - night noise floor (median of the RMS track),
     embedding = mean YAMNet embedding over the event frames
"""
import sys
from dataclasses import dataclass

import numpy as np

from classes import DEFAULT_THRESHOLDS, OUR_CLASSES, YAMNET_TO_OURS
from yamnet_runner import HOP_S, RMS_HOP_S, WIN_S


@dataclass
class Config:
    thresholds: dict = None
    hysteresis: float = 0.75     # offset threshold = hysteresis * onset threshold
    max_gap_s: float = 1.0       # merge same-class events closer than this
    class_gaps: dict = None      # per-class override, e.g. {"snoring": 3.0}
    min_frames: int = 1          # drop events shorter than this many YAMNet frames
    boundary: str = "center"     # "center" (frame-centre based) or "window" (full 0.975 s windows)
    refine: bool = False         # refine start/end with the 20 ms RMS track
    refine_margin_db: float = 6.0
    refine_search_s: float = 0.5
    exclusive: bool = False
    nms: bool = False            # drop an event mostly covered by a stronger event of another class
    nms_overlap: float = 0.5     # fraction of the weaker event's duration that must be covered

    def gap_s(self, c):
        return float((self.class_gaps or {}).get(c, self.max_gap_s))

    def thr(self, c):
        t = self.thresholds or DEFAULT_THRESHOLDS
        return float(t.get(c, DEFAULT_THRESHOLDS[c]))


def parse_class_values(items):
    """["snoring=3", "voices=2"] -> {"snoring": 3.0, "voices": 2.0}"""
    out = {}
    for it in items or []:
        c, v = it.split("=")
        if c not in OUR_CLASSES:
            raise ValueError(f"unknown class {c!r}; use one of {OUR_CLASSES}")
        out[c] = float(v)
    return out


def class_index_map(class_names, mapping=YAMNET_TO_OURS, warn=True):
    idx = {n: i for i, n in enumerate(class_names)}
    out = {}
    for c in OUR_CLASSES:
        missing = [n for n in mapping[c] if n not in idx]
        if missing and warn:
            print(f"warning: YAMNet has no class named {missing} (class '{c}')", file=sys.stderr)
        out[c] = np.array([idx[n] for n in mapping[c] if n in idx], dtype=int)
    return out


def our_scores(scores, cmap):
    """[N,521] -> [N,K] by max over each class's AudioSet columns."""
    return np.stack([scores[:, cmap[c]].max(axis=1) for c in OUR_CLASSES], axis=1)


def _runs(mask):
    """Start/end (inclusive) frame indices of True runs."""
    if not mask.any():
        return []
    d = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    starts = np.where(d == 1)[0]
    ends = np.where(d == -1)[0] - 1
    return list(zip(starts, ends))


def _frame_times(i0, i1, boundary):
    if boundary == "window":
        return i0 * HOP_S, i1 * HOP_S + WIN_S
    # centre of frame i is i*HOP + WIN/2; extend half a hop each side
    c0 = i0 * HOP_S + WIN_S / 2
    c1 = i1 * HOP_S + WIN_S / 2
    return max(0.0, c0 - HOP_S / 2), c1 + HOP_S / 2


def _refine(start, end, rms_db, floor_db, cfg):
    a = max(0, int((start - cfg.refine_search_s) / RMS_HOP_S))
    b = min(len(rms_db), int(np.ceil((end + cfg.refine_search_s) / RMS_HOP_S)))
    seg = rms_db[a:b]
    if len(seg) == 0:
        return start, end
    # smooth over 100 ms so random background fluctuations don't cross the margin
    k = min(5, len(seg))
    seg = 10 * np.log10(np.convolve(10 ** (seg / 10), np.ones(k) / k, mode="same") + 1e-12)
    above = np.where(seg > floor_db + cfg.refine_margin_db)[0]
    if len(above) == 0:
        return start, end
    s, e = (a + above[0]) * RMS_HOP_S, (a + above[-1] + 1) * RMS_HOP_S
    if e <= start or s >= end:      # refined interval must overlap the detection
        return start, end
    return s, e


def _level_db(start, end, rms_db):
    a, b = int(start / RMS_HOP_S), int(np.ceil(end / RMS_HOP_S))
    seg = rms_db[a:max(b, a + 1)]
    if len(seg) == 0:
        return float("nan")
    return float(10 * np.log10(np.mean(10 ** (seg / 10))))


def detect_events(frames, class_names, cfg=Config(), cmap=None):
    """frames: dict with scores [N,521], rms_db [M], optional embeddings [N,1024].

    Returns list of event dicts sorted by start time (embedding included if available).
    """
    cmap = cmap if cmap is not None else class_index_map(class_names)
    S = our_scores(frames["scores"], cmap)                  # [N,K]
    thr = np.array([cfg.thr(c) for c in OUR_CLASSES])
    if cfg.exclusive:
        ratio = S / thr
        best = ratio.argmax(axis=1)
        keep = np.zeros_like(S, dtype=bool)
        keep[np.arange(len(S)), best] = True
        S = np.where(keep, S, 0.0)

    rms_db = frames["rms_db"]
    floor_db = float(np.median(rms_db)) if len(rms_db) else 0.0
    raw = frames["scores"]
    emb = frames.get("embeddings")
    names = np.asarray(class_names)

    events = []
    for k, c in enumerate(OUR_CLASSES):
        s = S[:, k]
        gap = int(round(cfg.gap_s(c) / HOP_S))
        runs = _runs(s >= cfg.hysteresis * thr[k])
        merged = []
        for r in runs:                                   # merge small gaps
            if merged and r[0] - merged[-1][1] - 1 <= gap:
                merged[-1] = (merged[-1][0], r[1])
            else:
                merged.append(r)
        for i0, i1 in merged:
            peak = float(s[i0:i1 + 1].max())
            if peak < thr[k] or (i1 - i0 + 1) < cfg.min_frames:
                continue
            start, end = _frame_times(i0, i1, cfg.boundary)
            if cfg.refine:
                start, end = _refine(start, end, rms_db, floor_db, cfg)
            sub = raw[i0:i1 + 1][:, cmap[c]]
            top = names[cmap[c][np.unravel_index(sub.argmax(), sub.shape)[1]]]
            lvl = _level_db(start, end, rms_db)
            ev = {
                "class": c, "start_s": start, "end_s": end, "confidence": peak,
                "level_db": lvl, "snr_db": lvl - floor_db,
                "frame_start": int(i0), "frame_end": int(i1), "top_label": str(top),
            }
            if emb is not None:
                ev["embedding"] = emb[i0:i1 + 1].astype(np.float32).mean(axis=0)
            events.append(ev)
    if cfg.nms:
        events = _cross_class_nms(events, cfg)
    events.sort(key=lambda e: (e["start_s"], e["class"]))
    return events, floor_db


def _cross_class_nms(events, cfg):
    """Keep the strongest class where events of different classes overlap.

    Strength = confidence / class threshold, so classes with low YAMNet scores (snoring, door)
    are not always beaten by classes that score high.
    """
    def strength(e):
        return e["confidence"] / cfg.thr(e["class"])
    kept = []
    for e in sorted(events, key=strength, reverse=True):
        dur = max(e["end_s"] - e["start_s"], 1e-6)
        covered = sum(max(0.0, min(e["end_s"], k["end_s"]) - max(e["start_s"], k["start_s"]))
                      for k in kept if k["class"] != e["class"])
        if covered / dur < cfg.nms_overlap:
            kept.append(e)
    return kept
