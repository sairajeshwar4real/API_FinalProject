# detection/ — Sound event detection with YAMNet (Person B)

**In one sentence:** this module takes long "night" recordings (WAV), runs the pretrained YAMNet audio
classifier over them, and turns its output into a table of sound events (when, what class, how confident,
how loud) plus one embedding vector per event, for the dashboard (C) and the evaluation (D).

```
 Person A                     Person B (this folder)                        Person C / Person D
 ─────────                    ─────────────────────                          ──────────────────
 night_XX.wav  ──────────►  detect.py ──► events.csv + embeddings/*.npy ──►  index + dashboard (C)
 ground_truth.csv ───────────────────────────────────────────────────────►  evaluation (D)
                             quick_check.py / diagnose.py  (B's own tuning tools, D may reuse)
```

The 6 classes everywhere in the project: `traffic, voices, alarm, dog_bark, snoring, door`.

---

## 1. Quick start

```bash
conda create -n api python=3.11 -y && conda activate api
pip install -r detection/requirements.txt
pip install "setuptools<81"                 # tensorflow-hub still needs pkg_resources

python detection/tests/test_pipeline.py     # self-test, no TensorFlow needed -> "ALL TESTS PASSED"

bash detection/run_detection.sh data/nights detection/output     # the real run (final settings)
```

The first run downloads YAMNet from TF-Hub (about 15 MB) and runs it over every night. That takes about 1–2 min
per hour of audio on a MacBook (CPU). Results are cached, so later runs take seconds.

All commands assume you are in the repo root. If the scripts are in the current folder instead of `detection/`,
drop the `detection/` prefix.

---

## 2. Files

| File | What it is |
|---|---|
| `classes.py` | **Shared team definitions**: the 6 classes, which ESC-50/UrbanSound8K labels belong to each class (for A), and which YAMNet labels count as each class |
| `yamnet_runner.py` | Loads YAMNet and runs it over long files in chunks; also computes a 20 ms loudness track |
| `events.py` | Turns YAMNet frame scores into events (thresholds, merging, suppression, boundaries, loudness, embeddings). Pure numpy |
| `detect.py` | **Main entry point** (CLI): nights → `events.csv` + embeddings |
| `run_detection.sh` | `detect.py` with the final, tuned settings. Use this |
| `thresholds.json` | Tuned per-class thresholds (from the threshold sweep on `night_test_01`) |
| `quick_check.py` | Scores detections against ground truth (precision/recall/F1, timing error); `--sweep` tunes thresholds |
| `diagnose.py` | Explains every error: confusion between classes, ghost (background), fragment, or miss; recall per loudness level |
| `make_test_night.py` | Builds a test night from ESC-50 with ground truth (B's stand-in until A's generator was ready; **reference for A**) |
| `tests/test_pipeline.py` | End-to-end test with a fake YAMNet (checks chunking, timing, CSV format, embeddings) |
| `requirements.txt` | numpy, scipy, soundfile, tensorflow, tensorflow-hub |

---

## 3. Interfaces: what goes in, what comes out

### 3a. Input from Person A (night generator)

**Audio:** one file per night in one folder, e.g. `data/nights/night_01.wav … night_08.wav`
- **16 kHz, mono WAV** preferred (other rates and stereo also work but get resampled on the fly).
- The file name without extension is the `night_id` (`night_03.wav` → `night_03`).
- No clipping (peak below 1.0).

**Ground truth:** one CSV for all nights, e.g. `data/ground_truth.csv`:

| column | required | meaning |
|---|---|---|
| `night_id` | yes | matches the file name |
| `start_s`, `end_s` | yes | seconds from the start of the night; **the audible part of the clip only** (see below) |
| `class` | yes | one of the 6 classes, mapped with `classes.DATASET_TO_OURS` |
| `snr_db` | recommended | event level minus background level, in dB (for "accuracy vs. loudness") |
| `level_db` | optional | event RMS level in dBFS |
| `source_file` | optional | original clip name (makes `diagnose.py` output readable) |

**Rules for A, learned from testing:**
1. **Class mapping:** use `classes.DATASET_TO_OURS`. Team decision: `car_horn` → **alarm** (YAMNet hears horns as
   beeps/car alarms), and door = **`door_wood_knock` only** (YAMNet hears creaks as buzzers/snoring).
   "voices" = laughing, crying baby, children playing (neither dataset has normal speech).
2. **Trim clip silence before writing start/end.** ESC-50 clips contain recording noise, so a 5 s clip often holds a
   1 s sound. If the ground truth says 5 s, correct detections look like timing errors. Copy `trim()` from
   `make_test_night.py`: a 20 ms block counts as active when it is above both (peak − 20 dB) and (clip noise floor + 10 dB).
3. **Background:** use **pink noise** or a real quiet-room recording at about −55 dBFS, not brown noise. Brown noise
   is low rumble and hides engine sounds. `make_test_night.py` has `pink_noise()` and a `--background file.wav` option.
4. **Spacing:** events must not overlap, and leave **at least 3 s between events** (otherwise two neighbouring
   events of the same class get merged into one detection; the snoring merge gap is 3 s).
5. **Loudness levels:** spread events over SNR −5, 0, 5, 10, 15, 20 dB so D gets an accuracy-vs-loudness curve.
6. **Amount:** at least 50 events per class in total, and keep one night separate for tuning
   (we tune on one night and report on the others).

### 3b. Output to Person C (index + dashboard) and Person D (evaluation)

`detection/output/events.csv`, one row per detected event, sorted by night and time. The **first 7 columns are
the team contract**; the remaining ones are extras (written by `run_detection.sh`, since it uses `--extended`):

| column | type | meaning |
|---|---|---|
| `night_id` | str | which night |
| `start_s`, `end_s` | float | event start/end in seconds from the start of the night |
| `class` | str | one of the 6 classes |
| `confidence` | float 0–1 | highest YAMNet score for this class during the event |
| `level_db` | float | average loudness of the event in dBFS (0 = maximum, −60 = very quiet) |
| `embedding_file` | str | path to a `.npy` file holding a 1024-dim float32 vector (**relative to the folder containing events.csv**) |
| `event_id` | int | unique id across all nights (extra) |
| `snr_db` | float | `level_db` minus that night's background level (extra) |
| `top_label` | str | the YAMNet label that triggered it, e.g. "Bark", "Siren" (extra, nice for the dashboard) |
| `frame_start`, `frame_end` | int | YAMNet frame indices (extra, debugging) |

Example:
```
night_id,start_s,end_s,class,confidence,level_db,embedding_file,event_id,snr_db,top_label,frame_start,frame_end
night_01,41.300,45.860,dog_bark,0.8312,-44.10,embeddings/night_01/night_01_e0000.npy,1,12.20,Bark,85,93
```

Also in the output folder:
- `embeddings/<night_id>/<night_id>_eXXXX.npy`: one vector per event, the average YAMNet embedding over the event.
  For similarity search, L2-normalise the vectors and use cosine similarity / FAISS `IndexFlatIP`.
- `frames/<night_id>.npz`: cache of the raw YAMNet output (`scores` [N×521], `embeddings` [N×1024], `rms_db`,
  `duration_s`, `class_names`). D can use it for precision-recall curves without rerunning YAMNet.

`sample_output/` contains an example in the exact format (made with a fake model, so the values are meaningless).

---

## 4. How it works

**Step 1: YAMNet over the night (`yamnet_runner.py`).**
YAMNet is a CNN from Google, pretrained on AudioSet. It recognises 521 sound classes. It takes 16 kHz mono audio
and outputs one score vector (521 scores, each 0–1) and one 1024-dim embedding per **frame**: a 0.975 s window
every 0.48 s. A whole night is too long for one call, so we run it in chunks of 256 frames (~2 min). Every chunk
starts exactly on the 0.48 s grid, so the result is identical to one call on the whole file (verified in the test).
At the same time we compute the loudness (RMS, in dBFS) for every 20 ms block.

**Step 2: 521 AudioSet classes → our 6 classes (`classes.YAMNET_TO_OURS`, `events.our_scores`).**
Each of our classes is a list of AudioSet labels. Its score for a frame is the maximum score over those labels.
Example: `dog_bark` = max(Dog, Bark, Yip, Howl, Bow-wow, Growling, Whimper (dog), Canidae…).

**Step 3: frames → events (`events.detect_events`).**
1. `--exclusive`: in each frame only the strongest class survives (strength = score / class threshold).
2. **Hysteresis thresholds:** a frame is "active" if score ≥ 0.75 × threshold. A run of active frames becomes
   an event only if its peak reaches the full threshold. This avoids flickering on/off around the threshold.
3. **Merging:** events of the same class closer than the merge gap are joined (default 1 s; snoring 3 s,
   voices 2 s, dog_bark 2 s, because snores, laughs and barks come in bursts).
4. **Boundaries:** frame indices → seconds using the frame centres. `--refine` then moves start/end to where the
   20 ms loudness (smoothed over 100 ms) rises 6 dB above the night's background, searching within ±0.5 s.
5. `--nms` (cross-class suppression): if an event is at least 50% covered by a stronger event of a
   *different* class, drop it. This handles one sound producing two labels (a bark detected as dog and as voices).
6. Per event: `confidence` = peak score, `level_db` = mean energy over the event, `snr_db` = `level_db` − background
   (median of the loudness track), embedding = mean of the YAMNet embeddings of its frames.

**Step 4: write `events.csv` + embedding files (`detect.py`).**

---

## 5. Settings and results so far

Final settings (`run_detection.sh`): `--refine --exclusive --nms --class-gap snoring=3 voices=2 dog_bark=2
--thresholds thresholds.json`.

Tested on 3 synthetic nights (1 h each, ~150 ESC-50 events each, pink-noise background, SNR −5…20 dB).
"Found" = the detection overlaps a ground-truth event of the same class (`--criterion overlap`).
Thresholds were tuned on night 1 and scored on nights 2 and 3.

| Setup (micro-averaged) | Precision | Recall | F1 |
|---|---|---|---|
| Baseline (refine + exclusive, default thresholds) | 0.66 | 0.74 | 0.70 |
| + per-class merge gaps | 0.72 | 0.74 | 0.73 |
| + cross-class suppression (nms) | 0.80 | 0.72 | 0.75 |
| + thresholds tuned on night 1 (scored on nights 2–3) | 0.77 | 0.75 | **0.76** |

Per class (nights 2–3, final): alarm 0.87, snoring 0.77, traffic 0.74, voices 0.73, dog_bark 0.66, door 0.56 (F1).

Recall vs. loudness (all 3 nights, before tuning): −5 dB 0.22 · 0 dB 0.56 · 5 dB 0.83 · 10 dB 0.84 · 15 dB 0.94 · 20 dB 0.96.

Timing: mean onset error ≈ 0.47 s (detections start ~0.3 s late), mean offset error ≈ 0.6 s.

**Known limitations (for the report):**
- **Door knocks** are short, and YAMNet averages over 0.975 s windows, so knocks score low. Even at threshold 0.05,
  recall stays around 0.46.
- **Voices** needs a low threshold (0.10), which brings false positives. Barks and shouts are confused with each other.
- **Quiet events** (≤ 0 dB SNR) are mostly missed. This is the main accuracy-vs-loudness result.
- Thresholds were tuned on synthetic ESC-50 nights, so real recordings may need retuning.

---

## 6. Common tasks

```bash
# full run with the final settings
bash detection/run_detection.sh data/nights detection/output

# audio files changed (new nights with the same names)? force YAMNet to run again
bash detection/run_detection.sh data/nights detection/output --recompute

# score against ground truth
python detection/quick_check.py --gt data/ground_truth.csv --events detection/output/events.csv --criterion overlap
python detection/quick_check.py ... --nights night_02 night_03          # only some nights

# explain the errors (counts only)
python detection/diagnose.py --gt data/ground_truth.csv --events detection/output/events.csv --criterion overlap --summary

# re-tune thresholds on one night (uses cached frames, ~1 min)
python detection/quick_check.py --gt data/ground_truth.csv --frames detection/output/frames --sweep \
  --nights night_01 --criterion overlap --refine --exclusive --nms \
  --class-gap snoring=3 voices=2 dog_bark=2 --write-thresholds detection/thresholds.json

# make a test night yourself from ESC-50
python detection/make_test_night.py --esc50 data/ESC-50-master --out data/test_nights --minutes 60 --events 150 --seed 1
```

All `detect.py` options: `--refine`, `--exclusive`, `--nms`, `--class-gap CLASS=SEC …`, `--thresholds file.json`,
`--hysteresis 0.75`, `--max-gap 1.0`, `--min-frames 1`, `--boundary center|window`, `--extended`,
`--no-embeddings`, `--recompute`, `--model <TF-Hub handle or local SavedModel folder>`, `--chunk-frames 256`.

Matching rules in `quick_check.py`: `--criterion onset` = onset within `--collar` seconds (default 1 s, strict);
`--criterion overlap` = any time overlap with a same-class event. Matching is one-to-one, highest confidence first.

---

## 7. Troubleshooting

| Problem | Fix |
|---|---|
| `ModuleNotFoundError: No module named 'pkg_resources'` | `pip install "setuptools<81"` |
| `UserWarning: pkg_resources is deprecated` | harmless, ignore |
| YAMNet download fails | download from kaggle.com/models/google/yamnet, unzip, pass `--model path/to/folder` |
| zsh shows `quote>` after pasting commands | zsh doesn't treat `#` as a comment by default: `setopt interactivecomments`, or paste without comment lines |
| results don't change after regenerating the nights | the frame cache is still the old one: add `--recompute` |
| "warning: YAMNet has no class named …" | a label in `classes.YAMNET_TO_OURS` is misspelt; check `tests/yamnet_class_map.csv` |

---

## 8. Notes for an LLM helping with this code

- Entry point is `detect.py:main()`. It accepts `run_fn` and `class_names` arguments so tests can inject a fake model.
- Frame timing constants live in `yamnet_runner.py`: `HOP_S = 0.48`, `WIN_S = 0.975`, `RMS_HOP_S = 0.02`.
  Frame *i* covers `[i·0.48, i·0.48 + 0.975]` seconds.
- `events.Config` holds every post-processing option; `detect_events(frames, class_names, cfg)` is a pure function.
  It returns `(events, floor_db)`, where each event is a dict with
  `class, start_s, end_s, confidence, level_db, snr_db, frame_start, frame_end, top_label, embedding`.
- The output CSV contract (first 7 columns) must not change without agreeing with C and D.
- Class names and mappings live only in `classes.py`; change them there, never hard-code them elsewhere.
