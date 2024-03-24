import os
import pandas as pd
import argparse
from pathlib import Path

parser = argparse.ArgumentParser(description='Process HRV data.')
parser.add_argument('--data_dir', type=str, default='data', help='Path to the data dir with *_hrv_features_csv files')
parser.add_argument('--output_file', type=str, default='features.csv', help='Path to the output file with combined features')
args = parser.parse_args()

data_dir = Path(args.data_dir)
if not data_dir.is_dir():
    raise ValueError(f"{args.data_dir} is not a valid directory path.")

csv_files = [file for file in os.listdir(data_dir) if file.endswith('.csv')]

num_csv_files = len(csv_files)
print(f"Number of *_hrv_features.csv files found: {num_csv_files}")

if num_csv_files == 0:
    raise ValueError("No *_hrv_features.csv files found in the specified directory.")

dfs = []
for file in csv_files:
    file_path = os.path.join(data_dir, file)
    df = pd.read_csv(file_path)
    code = df['code'].values[0]
    session_id, segment_id, participant_id = code.split('_')
    df['session_id'] = session_id
    df['segment_id'] = segment_id
    df['participant_id'] = participant_id
    df['hrv_features_csv'] = file_path
    df = df[['session_id', 'segment_id', 'participant_id', 'hrv_features_csv'] + list(df.columns[:-4])]
    dfs.append(df)

combined_df = pd.concat(dfs, ignore_index=True)
combined_df.to_csv(args.output_file, index=False)
xlsx_path = args.output_file.replace('csv', 'xlsx')
combined_df.to_excel(xlsx_path, index=False)

print(f"Combined features saved to {args.output_file}")
print(f"Combined features saved to {xlsx_path}")
print("Done!")