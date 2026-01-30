import os
import glob
import argparse
import numpy as np
from tqdm import tqdm

from disvoice.glottal import Glottal
from disvoice.prosody import Prosody
from disvoice.phonation import Phonation
from disvoice.articulation import Articulation
from disvoice.phonological import Phonological

import warnings
warnings.filterwarnings("ignore")

def safe_extract(module_name, extractor, wav_files, output_dir, log_file, static=True):
    """Extract features safely; skip existing outputs and log module + file on failure."""
    os.makedirs(output_dir, exist_ok=True)

    for wav_file in tqdm(wav_files, desc=f"Extracting {module_name}"):
        output_path = os.path.join(output_dir, os.path.basename(wav_file)).replace(".wav", ".npz")

        # Skip already processed files
        if os.path.exists(output_path):
            continue

        try:
            features = extractor.extract_features_file(wav_file, static=static, plots=False, fmt='npy')
            if features.ndim == 1:
                features = features[None, :]

            features[np.isnan(features)] = 0
            np.savez_compressed(output_path, data=features)

        except Exception:

            with open(log_file, "a") as logf:
                logf.write(f"{module_name}: {os.path.basename(wav_file)}\n")

def extract_all(wav_dir, output_dir, static=True):
    wav_files = [f for f in glob.glob(os.path.join(os.path.abspath(wav_dir), "*.wav"))]
    if not wav_files:
        print(f"No WAV files found in {wav_dir}")
        return

    log_file = os.path.join(output_dir, "failed_files.log")
    if os.path.exists(log_file):
        os.remove(log_file)

    safe_extract("Prosody", Prosody(), wav_files, os.path.join(output_dir, "prosody"), log_file, static)
    safe_extract("Articulation", Articulation(), wav_files, os.path.join(output_dir, "articulation"), log_file, static)
    safe_extract("Phonation", Phonation(), wav_files, os.path.join(output_dir, "phonation"), log_file, static)
    safe_extract("Phonological", Phonological(), wav_files, os.path.join(output_dir, "phonological"), log_file, static)
    safe_extract("Glottal", Glottal(), wav_files, os.path.join(output_dir, "glottal"), log_file, static)

    # -- Summary
    if os.path.exists(log_file):
        with open(log_file) as f: 
            fails = f.readlines()
        print(f"\n{len(fails)} failures. Check {log_file} for details.")
    else:
        print("\nAll files processed successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DisVoice-based Feature Extraction", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--wav-dir", type=str, default="audios/")
    parser.add_argument("--output-dir", type=str, default="features/")
    parser.add_argument("--static", action="store_true", help="Expand features to static form")
    args = parser.parse_args()

    extract_all(args.wav_dir, os.path.abspath(args.output_dir), static=args.static)