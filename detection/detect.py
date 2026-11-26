"""Person B entry point: nights (wav) -> events CSV + embeddings.

    python detection/detect.py --nights data/nights --out detection/output

Stage 1 (slow, once per night): YAMNet over the night -> cached in OUT/frames/<night_id>.npz
Stage 2 (fast): frames -> events -> OUT/events.csv and OUT/embeddings/<night_id>/*.npy

Re-running with different thresholds/options only redoes stage 2 (seconds).
Use --recompute to force stage 1 again.

events.csv columns (team contract):
    night_id, start_s, end_s, class, confidence, level_db, embedding_file
With --extended, these are appended: event_id, snr_db, top_label, frame_start, frame_end
embedding_file is relative to the folder that contains events.csv.
"""
import argparse
import csv
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from events import Config, class_index_map, detect_events, parse_class_values  # noqa: E402

CONTRACT = ["night_id", "start_s", "end_s", "class", "confidence", "level_db", "embedding_file"]
EXTENDED = ["event_id", "snr_db", "top_label", "frame_start", "frame_end"]


def find_nights(path):
    if os.path.isfile(path):
        return [path]
    files = []
    for ext in ("wav", "flac", "WAV", "FLAC"):
        files += glob.glob(os.path.join(path, f"*.{ext}"))
    return sorted(set(files))


def save_frames(path, fr, class_names, keep_embeddings=True):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data = dict(
        scores=fr["scores"].astype(np.float16),
        rms_db=fr["rms_db"].astype(np.float32),
        duration_s=np.float64(fr["duration_s"]),
        class_names=np.array(class_names),
    )
    if keep_embeddings:
        data["embeddings"] = fr["embeddings"].astype(np.float16)
    np.savez(path, **data)


def load_frames(path):
    z = np.load(path)
    fr = {k: z[k] for k in z.files}
    fr["scores"] = fr["scores"].astype(np.float32)
    if "embeddings" in fr:
        fr["embeddings"] = fr["embeddings"].astype(np.float32)
    return fr, [str(n) for n in fr.pop("class_names")]


def config_from_args(a):
    thresholds = None
    if a.thresholds:
        with open(a.thresholds) as f:
            thresholds = json.load(f)
    return Config(thresholds=thresholds, hysteresis=a.hysteresis, max_gap_s=a.max_gap,
                  class_gaps=parse_class_values(a.class_gap),
                  min_frames=a.min_frames, boundary=a.boundary, refine=a.refine,
                  refine_margin_db=a.refine_margin, exclusive=a.exclusive, nms=a.nms)


def main(argv=None, run_fn=None, class_names=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--nights", required=True, help="folder of night_XX.wav files (or a single file)")
    p.add_argument("--out", default="detection/output")
    p.add_argument("--events-name", default="events.csv")
    p.add_argument("--model", default=None, help="TF-Hub handle or local SavedModel dir (default: YAMNet on TF-Hub)")
    p.add_argument("--chunk-frames", type=int, default=256, help="YAMNet frames per inference call (~2 min)")
    p.add_argument("--recompute", action="store_true", help="ignore cached frames and rerun YAMNet")
    p.add_argument("--no-embeddings", action="store_true", help="don't write embedding .npy files")
    p.add_argument("--thresholds", help="JSON {class: threshold}; default from classes.py")
    p.add_argument("--hysteresis", type=float, default=0.75)
    p.add_argument("--max-gap", type=float, default=1.0)
    p.add_argument("--class-gap", nargs="*", metavar="CLASS=SEC",
                   help="per-class merge gap, e.g. --class-gap snoring=3 voices=2 dog_bark=2")
    p.add_argument("--min-frames", type=int, default=1)
    p.add_argument("--boundary", choices=["center", "window"], default="center")
    p.add_argument("--refine", action="store_true", help="refine start/end on 20 ms RMS track")
    p.add_argument("--refine-margin", type=float, default=6.0)
    p.add_argument("--exclusive", action="store_true", help="one class per frame (strongest)")
    p.add_argument("--nms", action="store_true", help="cross-class suppression of overlapping events")
    p.add_argument("--extended", action="store_true", help="append extra columns after the contract ones")
    a = p.parse_args(argv)

    nights = find_nights(a.nights)
    if not nights:
        sys.exit(f"no audio files found in {a.nights}")
    cfg = config_from_args(a)
    frames_dir = os.path.join(a.out, "frames")
    os.makedirs(a.out, exist_ok=True)

    rows, eid = [], 0
    for path in nights:
        night_id = os.path.splitext(os.path.basename(path))[0]
        cache = os.path.join(frames_dir, f"{night_id}.npz")
        if a.recompute or not os.path.exists(cache):
            if run_fn is None:
                from yamnet_runner import DEFAULT_HANDLE, load_yamnet
                print("loading YAMNet ...", file=sys.stderr)
                run_fn, class_names = load_yamnet(a.model or DEFAULT_HANDLE)
            from yamnet_runner import run_night
            print(f"[{night_id}] running YAMNet", file=sys.stderr)
            fr = run_night(path, run_fn, chunk_frames=a.chunk_frames)
            save_frames(cache, fr, class_names)
        fr, names = load_frames(cache)

        events, floor_db = detect_events(fr, names, cfg, class_index_map(names))
        print(f"[{night_id}] {fr['duration_s'] / 3600:.2f} h, floor {floor_db:.1f} dBFS, "
              f"{len(events)} events", file=sys.stderr)

        emb_dir = os.path.join(a.out, "embeddings", night_id)
        if not a.no_embeddings:
            os.makedirs(emb_dir, exist_ok=True)
            for old in glob.glob(os.path.join(emb_dir, "*.npy")):
                os.remove(old)
        for n, ev in enumerate(events):
            eid += 1
            emb_rel = ""
            if not a.no_embeddings and "embedding" in ev:
                emb_rel = os.path.join("embeddings", night_id, f"{night_id}_e{n:04d}.npy")
                np.save(os.path.join(a.out, emb_rel), ev["embedding"])
            row = {
                "night_id": night_id, "start_s": f"{ev['start_s']:.3f}", "end_s": f"{ev['end_s']:.3f}",
                "class": ev["class"], "confidence": f"{ev['confidence']:.4f}",
                "level_db": f"{ev['level_db']:.2f}", "embedding_file": emb_rel.replace(os.sep, "/"),
            }
            if a.extended:
                row.update(event_id=eid, snr_db=f"{ev['snr_db']:.2f}", top_label=ev["top_label"],
                           frame_start=ev["frame_start"], frame_end=ev["frame_end"])
            rows.append(row)

    out_csv = os.path.join(a.out, a.events_name)
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CONTRACT + (EXTENDED if a.extended else []))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} events -> {out_csv}", file=sys.stderr)
    return out_csv


if __name__ == "__main__":
    main()
