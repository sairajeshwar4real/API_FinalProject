"""Chunked YAMNet inference over long recordings.

YAMNet framing (fixed by the model):
  input 16 kHz mono float32 in [-1, 1]
  frame i covers [i*0.48 s, i*0.48 + 0.975 s]   (window 15600 samples, hop 7680)

A whole 8 h night does not fit in one call, so we feed chunks of `chunk_frames`
frames. Every chunk starts on a multiple of the hop, so frame indices stay global
and nothing is duplicated or skipped at chunk borders.

We also compute a fine RMS track (20 ms blocks, dBFS) for loudness and boundary refinement.
"""
import csv
import sys
from math import gcd

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

SR = 16000
WIN = 15600            # 0.975 s
HOP = 7680             # 0.48 s
WIN_S = WIN / SR
HOP_S = HOP / SR
RMS_BLOCK = 320        # 20 ms; divides HOP exactly, so blocks stay aligned across chunks
RMS_HOP_S = RMS_BLOCK / SR

DEFAULT_HANDLE = "https://tfhub.dev/google/yamnet/1"


def read_class_names(path):
    with open(path, newline="") as f:
        return [row["display_name"] for row in csv.DictReader(f)]


def load_yamnet(handle=DEFAULT_HANDLE):
    """Returns (run_fn, class_names). run_fn(wave_16k_float32) -> (scores[N,521], embeddings[N,1024])."""
    import tensorflow as tf
    import tensorflow_hub as hub

    model = hub.load(handle)
    names = read_class_names(model.class_map_path().numpy().decode("utf-8"))

    def run(wave):
        scores, embeddings, _ = model(tf.constant(wave, dtype=tf.float32))
        return scores.numpy(), embeddings.numpy()

    return run, names


def _read_16k(f, start16, n16):
    """Read n16 samples (16 kHz timeline) starting at start16 from an open SoundFile, mono, any input sr."""
    sr = f.samplerate
    if sr == SR:
        f.seek(min(start16, f.frames))
        x = f.read(n16, dtype="float32", always_2d=True).mean(axis=1)
    else:
        a = int(np.floor(start16 * sr / SR))
        b = int(np.ceil((start16 + n16) * sr / SR)) + 1
        f.seek(min(a, f.frames))
        x = f.read(b - a, dtype="float32", always_2d=True).mean(axis=1)
        g = gcd(SR, sr)
        x = resample_poly(x, SR // g, sr // g).astype(np.float32)
    if len(x) < n16:
        x = np.pad(x, (0, n16 - len(x)))
    return x[:n16].astype(np.float32)


def _rms_db(x):
    n = len(x) // RMS_BLOCK
    if n == 0:
        return np.zeros(0, np.float32)
    blocks = x[: n * RMS_BLOCK].reshape(n, RMS_BLOCK)
    return (10 * np.log10(np.mean(blocks ** 2, axis=1) + 1e-12)).astype(np.float32)


def run_night(path, run, chunk_frames=256, verbose=True):
    """Run YAMNet over one (possibly hours-long) file.

    Returns dict with scores [N,521], embeddings [N,1024], rms_db [M], duration_s.
    """
    with sf.SoundFile(path) as f:
        n16 = int(round(f.frames * SR / f.samplerate))
        if f.samplerate != SR and verbose:
            print(f"  note: {path} is {f.samplerate} Hz, resampling to 16 kHz on the fly", file=sys.stderr)
        need = WIN + (chunk_frames - 1) * HOP
        scores, embs, rms = [], [], []
        start, n_chunks = 0, int(np.ceil(n16 / (chunk_frames * HOP)))
        k = 0
        while start < n16:
            remaining = n16 - start
            full = remaining >= need
            x = _read_16k(f, start, need if full else remaining)
            s, e = run(x)
            if full:
                if len(s) < chunk_frames:
                    raise RuntimeError(f"model returned {len(s)} frames for a full chunk, expected {chunk_frames}")
                s, e = s[:chunk_frames], e[:chunk_frames]
                rms.append(_rms_db(x[: chunk_frames * HOP]))
                start += chunk_frames * HOP
            else:
                rms.append(_rms_db(x))
                start = n16
            scores.append(s.astype(np.float32))
            embs.append(e.astype(np.float32))
            k += 1
            if verbose and (k % 20 == 0 or start >= n16):
                print(f"  chunk {k}/{n_chunks}  ({start / SR / 60:.1f} / {n16 / SR / 60:.1f} min)", file=sys.stderr)

    return {
        "scores": np.concatenate(scores),
        "embeddings": np.concatenate(embs),
        "rms_db": np.concatenate(rms),
        "duration_s": n16 / SR,
    }
