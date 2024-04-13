import os
import pandas as pd
import argparse
from pathlib import Path

parser = argparse.ArgumentParser(description='Process HRV data.')
parser.add_argument('--data_dir', type=str, default='data/processed', help='Path to the data dir with *_mat_features_csv files')
parser.add_argument('--output_file', type=str, default='data/summary/features.csv', help='Path to the output file with combined features')
args = parser.parse_args()

data_dir = Path(args.data_dir)
if not data_dir.is_dir():
    raise ValueError(f"{args.data_dir} is not a valid directory path.")

codes = [Path(file).stem for file in sorted(os.listdir(data_dir)) if file.endswith('.mat')]
features_csv_files = [(data_dir / f'{code}_mat_features.csv', data_dir / f'{code}_nk2_features.csv') for code in codes]
features_csv_files = [t for t in features_csv_files if all([f.is_file() for f in t])]

num_csv_files = len(features_csv_files)
print(f"Number of feature files: {num_csv_files}")

if num_csv_files == 0:
    raise ValueError("No *_mat_features.csv files found in the specified directory.")

dfs = []
for mat_file, nk2_file in features_csv_files:
    nk_df = pd.read_csv(nk2_file)
    nk_df.columns = ['NK2_' + col[4:] for col in nk_df.columns]
    mat_df = pd.read_csv(mat_file)
    code = mat_df['code'].values[0]
    session_id, segment_id, participant_id = code.split('_')
    mat_df['session_id'] = session_id
    mat_df['segment_id'] = segment_id
    mat_df['participant_id'] = participant_id
    mat_df['mat_features_csv'] = mat_file
    mat_df = mat_df[['session_id', 'segment_id', 'participant_id', 'mat_features_csv'] + list(mat_df.columns[:-4])]
    mat_df.columns = ['session_id', 'segment_id', 'participant_id', 'mat_features_csv', 'code'] + ['MAT_' + col for col in mat_df.columns[5:]]
    df = pd.concat([mat_df, nk_df], axis=1)
    dfs.append(df)

Path(args.output_file).resolve().parent.mkdir(parents=True, exist_ok=True)
combined_df = pd.concat(dfs, ignore_index=True)
combined_df.to_csv(args.output_file, index=False)
xlsx_path = args.output_file.replace('csv', 'xlsx')
combined_df.to_excel(xlsx_path, index=False)

print(f"Combined features saved to {args.output_file}")
print(f"Combined features saved to {xlsx_path}")
print("Done!")