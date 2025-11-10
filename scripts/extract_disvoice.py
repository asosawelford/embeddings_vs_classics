import os
import glob
import argparse
import numpy as np
from tqdm import tqdm
import warnings
from disvoice.prosody import Prosody
from disvoice.articulation import Articulation
from disvoice.phonological import Phonological

warnings.filterwarnings("ignore")

def safe_extract(module_name, extractor, wav_files, wav_dir, output_dir, log_file, static=True):
    """Extract features safely; replicate folder structure and skip existing outputs."""
    for wav_file in tqdm(wav_files, desc=f"Extracting {module_name}"):
        # replicate relative path
        rel_path = os.path.relpath(wav_file, start=wav_dir)
        rel_folder = os.path.dirname(rel_path)

        # create same subfolder structure under output_dir/module_name
        out_folder = os.path.join(output_dir, module_name.lower(), rel_folder)
        os.makedirs(out_folder, exist_ok=True)

        # save output with same name but .npz
        output_path = os.path.join(out_folder, os.path.splitext(os.path.basename(wav_file))[0] + ".npz")

        if os.path.exists(output_path):
            continue

        try:
            features = extractor.extract_features_file(wav_file, static=static, plots=False, fmt='npy')
            features[np.isnan(features)] = 0
            np.savez_compressed(output_path, data=features)
        except Exception as e:
            with open(log_file, "a") as logf:
                logf.write(f"{module_name}: {wav_file} | {e}\n")

def extract_all(wav_dir, output_dir, static=True):
    # recursive search for all wav files
    wav_files = glob.glob(os.path.join(os.path.abspath(wav_dir), "**", "*.wav"), recursive=True)
    if not wav_files:
        print(f"No WAV files found in {wav_dir}")
        return

    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, "failed_files.log")
    if os.path.exists(log_file):
        os.remove(log_file)

    safe_extract("Articulation", Articulation(), wav_files, wav_dir, output_dir, log_file, static)
    safe_extract("Phonological", Phonological(), wav_files, wav_dir, output_dir, log_file, static)
    safe_extract("Prosody", Prosody(), wav_files, wav_dir, output_dir, log_file, static)

    if os.path.exists(log_file):
        with open(log_file) as f:
            fails = f.readlines()
        print(f"\n{len(fails)} failures. Check {log_file} for details.")
    else:
        print("\nAll files processed successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Recursive DisVoice-based Feature Extraction",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--wav-dir", type=str, required=True, help="Directory containing WAV files (recursively)")
    parser.add_argument("--output-dir", type=str, default="features/", help="Output directory for features")
    parser.add_argument("--static", action="store_true", help="Expand features to static form")
    args = parser.parse_args()

    extract_all(args.wav_dir, os.path.abspath(args.output_dir), static=args.static)
