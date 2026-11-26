# Night-time sound event detection and indexing (API final project)

Pipeline: night generator -> detection (YAMNet) -> event table -> index + dashboard -> evaluation

| Folder | Owner | What |
|---|---|---|
| `generator/` | A | builds synthetic nights from ESC-50 / UrbanSound8K + ground truth |
| `detection/` | B | YAMNet detection -> `events.csv` + embeddings (see `detection/README.md`) |
| `index/` | C | SQLite + embedding search + Streamlit dashboard |
| `eval/` | D | evaluation, figures, report |
| `data/` | all | local only (not in git): datasets, nights, ground truth |

Event CSV contract: `night_id, start_s, end_s, class, confidence, level_db, embedding_file`
Classes: `traffic, voices, alarm, dog_bark, snoring, door` (mapping in `detection/classes.py`)

Expected data layout (shared outside git):
data/ESC-50-master/  data/UrbanSound8K/  data/nights/night_XX.wav  data/ground_truth.csv
