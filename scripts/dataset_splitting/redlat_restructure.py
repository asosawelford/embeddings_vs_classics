import os
import re
import glob
import shutil
import argparse
import pandas as pd
from tqdm import tqdm
from distutils.util import strtobool

if __name__ == "__main__":

    # -- command line arguments
    parser = argparse.ArgumentParser(description='Restructure the original RedLat audio samples', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data-dir', required=True, type=str)
    parser.add_argument('--metadata-path', required=True, type=str)
    parser.add_argument('--new-data-dir', default='./data/redlat/audios/', type=str)
    args = parser.parse_args()

    os.makedirs(args.new_data_dir, exist_ok=True)

    # -- loading and processing metadata
    # -- original column IDs: RECODING ORIGINAL NAME,UPDRS,UPDRS-speech,H/Y,SEX,AGE,time after diagnosis
    metadata = pd.read_csv(args.metadata_path)

    # -- processing audio samples
    ignored_samples = 0
    wavs = glob.glob(os.path.join(args.data_dir, '**/*.wav'), recursive=True)
    for wav_path in tqdm(wavs):
        record_id = wav_path.split(os.path.sep)[-1].split('.')[0]
        sample_id = record_id.split('_')[1]
        
        if sample_id in metadata['record_id'].tolist():

            sample = metadata[metadata['record_id'] == sample_id]
            group_id = sample['clinical_diagnosis'].values[0].upper()
            task_id = record_id.split('_')[2]

            new_wav_path = os.path.join(args.new_data_dir, f'{group_id}_{task_id}_{sample_id}.wav')
            shutil.copy(wav_path, new_wav_path)

