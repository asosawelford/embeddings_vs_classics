import os
import glob
import argparse
import pandas as pd
from tqdm import tqdm

if __name__ == "__main__":

    # -- command line arguments
    parser = argparse.ArgumentParser(description='Prepare RedLat dataset into a CSV', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--samples-dir', required=True, type=str)
    parser.add_argument('--metadata-path', required=True, type=str)
    parser.add_argument('--output-dir', default='./splits/redlat/', type=str)
    args = parser.parse_args()

    # -- loading and processing metadata
    # -- original column IDs: RECODING ORIGINAL NAME,UPDRS,UPDRS-speech,H/Y,SEX,AGE,time after diagnosis,G,I,D,MEDIAN
    metadata = pd.read_csv(args.metadata_path)
    #metadata['label'] = metadata['RECODING ORIGINAL NAME'].map(lambda x: 1 if 'AVPEPUDEAC' not in x else 0)
    metadata['group_id'] = metadata['clinical_diagnosis']
    metadata['sex'] = metadata['demo_sex']

    # -- processing audio samples
    dataset = []
    wavs = glob.glob(os.path.join(args.samples_dir, '*.wav'), recursive=True)
    for wav_path in tqdm(wavs):

        sample_id = wav_path.split(os.path.sep)[-1].split('.')[0]
        subject_id = sample_id.split('_')[2]
        task_id = sample_id.split('_')[1]

        # -- retriveing metadata per sample
        sample = metadata[metadata['record_id'] == subject_id]

        sex = sample['sex'].values[0]
        age = sample['demo_age'].values[0]
        group_id = sample['group_id'].values[0].upper()
        site = sample['site'].values[0]

        # -- preparing the dataset
        dataset.append( (subject_id, sample_id, task_id, site, group_id, sex, age) )


    # -- building the dataset dataframe
    dataset_df = pd.DataFrame(dataset, columns=['subject_id', 'sample_id', 'task_id', 'site', 'group_id', 'sex', 'age'])

    # -- adding information about speech features paths
    dataset_dir = os.path.sep.join(args.samples_dir.split(os.path.sep)[:-2])
    for feature_type in ['disvoice/articulation', 'disvoice/glottal', 'disvoice/phonation', 'disvoice/phonological', 'disvoice/prosody', 'wav2vec/layer07']:
        feature_samples = []

        for i, sample in dataset_df.iterrows():
            sample_path = os.path.join(dataset_dir, 'speech_features', feature_type, f'{sample["sample_id"]}.npz')
            feature_samples.append(sample_path)

        dataset_df[feature_type.replace('/', '').replace('disvoice', '').replace('layer07', '')] = feature_samples

    # -- saving dataset split
    os.makedirs(args.output_dir, exist_ok=True)
    dataset_df.sort_values(by='subject_id', inplace=True)
    dataset_df.to_csv(os.path.join(args.output_dir, 'dataset.csv'), index=False)