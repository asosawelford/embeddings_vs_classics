"""
ejemplo para correr con 4 workers, usando BASE_DIR de config: 
python compare_functionals_extraction.py "" "" 2
:)
"""

import os
import sys
import traceback
from pathlib import Path
import opensmile
import pandas as pd
from multiprocessing import Pool, cpu_count


BASE_DIR = ""

exclude_text_files = ["readme","error","config"] # tareas a excluir del analisis de texto

def extract_fileid_task(file_path): # Modificar de acuerdo a el proyecto
    file_id = file_path.split("_")[1]
    task = file_path.split("_")[2]
    task = task.split(".")[0]
    return file_id, task

def find_wavs(root_dir, exclude_substrings=None):
    exclude_substrings = exclude_substrings or []
    for dirpath, _, filenames in os.walk(root_dir):
        for fn in filenames:
            if not fn.lower().endswith(".wav"):
                continue
            full = os.path.join(dirpath, fn)
            low = full.lower()
            if any(substr.lower() in low for substr in exclude_substrings):
                continue
            yield full


# Worker function: initialize a global Smile per worker for efficiency
_smile = None
def init_worker():
    global _smile
    _smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.ComParE_2016,
        feature_level=opensmile.FeatureLevel.Functionals,
    )

def process_one(wav_path):
    global _smile
    try:
        id_, task = extract_fileid_task(wav_path)
        feat_df = _smile.process_file(wav_path)
        if feat_df.shape[0] == 0:
            raise RuntimeError(f"No features returned for {wav_path}")
        # ensure single-row
        if feat_df.shape[0] > 1:
            feat_row = feat_df.mean(axis=0)
        else:
            feat_row = feat_df.iloc[0]
        # build dict with task-prefixed keys
        prefixed = {f"{task}__{name}": float(feat_row[name]) for name in feat_row.index}
        return (id_, prefixed, None)
    except Exception as e:
        return (None, None, (wav_path, traceback.format_exc()))

def main(input_dir=None, out_csv="compare2016_features.csv", workers=None):
    input_dir = str(Path(input_dir or BASE_DIR))
    if not out_csv:
        out_csv = "compare2016_features.csv"
    if workers is None:
        workers = max(1, cpu_count() - 1)

    all_wavs = list(find_wavs(input_dir, exclude_substrings=exclude_audio_files))
    # For visibility, also count excluded files (optional)
    total_found = 0
    print(input_dir)
    for dirpath, _, filenames in os.walk(input_dir):
        for fn in filenames:
            if fn.lower().endswith(".wav"):
                total_found += 1
    skipped = total_found - len(all_wavs)

    if not all_wavs:
        print("No WAV files found (after applying excludes).")
        return

    print(f"Found {len(all_wavs)} WAVs (skipped {skipped} matching exclude list).")

    # parallel processing
    with Pool(processes=workers, initializer=init_worker) as pool:
        results = pool.map(process_one, all_wavs)

    errors = [e for (_, _, e) in results if e is not None]
    success = [(id_, d) for (id_, d, e) in results if e is None]

    # aggregate by id: produce one dict per id with all prefixed features
    data_by_id = {}
    for id_, feat_dict in success:
        if id_ not in data_by_id:
            data_by_id[id_] = {}
        # if same prefixed key already exists, overwrite (last wins)
        data_by_id[id_].update(feat_dict)

    if not data_by_id:
        print("No successful feature extractions.")
        if errors:
            print("Errors:")
            for p, tb in errors:
                print(f"- {p}:\n{tb}")
        return

    # Determine full column list (union of all keys) sorted for stability
    all_feat_keys = sorted({k for d in data_by_id.values() for k in d.keys()})
    header = ["id"] + all_feat_keys

    rows = []
    for id_, feats in sorted(data_by_id.items()):
        row = [id_] + [feats.get(k, "") for k in all_feat_keys]
        rows.append(row)

    out_df = pd.DataFrame(rows, columns=header)
    out_path = Path(input_dir) / out_csv
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    print(f"Wrote {len(out_df)} rows to {out_path}")

    if errors:
        print(f"Encountered {len(errors)} errors; first 5 shown:")
        for p, tb in errors[:5]:
            print(f"- {p}:\n{tb}")

if __name__ == "__main__":
    # Usage: python extract_compare_by_task.py [input_dir] [out.csv] [num_workers]
    input_dir = sys.argv[1] if len(sys.argv) >= 2 else None
    out_csv = sys.argv[2] if len(sys.argv) >= 3 else "compare2016_features.csv"
    workers = int(sys.argv[3]) if len(sys.argv) >= 4 else None
    main(input_dir, out_csv, workers)

