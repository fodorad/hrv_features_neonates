import argparse
import scipy.io
import neurokit2
from tqdm import tqdm
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from datetime import datetime, timedelta


def test_time_format(time_string: str) -> None:
    try:
        time_parts = time_string.split(':')
        hours = int(time_parts[0])
        minutes = int(time_parts[1])
        seconds = int(time_parts[2])
    except:
        print(f'Error: Invalid time format {time_string}. Please provide time in HH:MM:SS format.')
        exit()


def plot_ecg_with_peaks_and_segments(ecg_data, r_peaks, ecg_mean, ecg_std, output_path: Path | str):
    COLORS = {
        -1: 'gray',
        0: 'black',
        1: 'blue',
        2: 'green',
        3: 'magenta'
    }
    plt.figure(figsize=(15, 5))
    for i in range(4):
        segment = ecg_data[:, 3] == i
        plt.plot(ecg_data[segment,1], ecg_data[segment, 2], c=COLORS[i])
    plt.scatter(r_peaks[:,1], r_peaks[:,2], s=10, color='red')
    plt.xlabel('Time')
    plt.ylabel('Voltage')
    plt.xlim(ecg_data[0,1], ecg_data[-1,1])
    plt.ylim(ecg_mean-10*ecg_std, ecg_mean+10*ecg_std)
    plt.title('Raw ECG Data')
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()


def r_peak_detection(ecg_data, sampling_rate: int = 130):
    # R-peaks detection with neurokit2
    signal = ecg_data[:, 2]
    _, results = neurokit2.ecg_peaks(signal, sampling_rate=sampling_rate)
    r_peaks = ecg_data[results["ECG_R_Peaks"], :]
    return r_peaks

def test_segment_occurrence(segment_id: int) -> None:
    if sum(segment_id == 0) == 0:
        print('Error: Baseline segment not found.')
        exit()
    if sum(segment_id == 1) == 0:
        print('Error: SFP1 segment not found.')
        exit()
    if sum(segment_id == 2) == 0:
        print('Error: SFP2 segment not found.')
        exit()
    if sum(segment_id == 3) == 0:
        print('Error: SFP3 segment not found.')
        exit()  

def preprocess(id: int,
               polar_csv_anya: str,
               polar_csv_baba: str,
               baseline_start: str,
               baseline_end: str,
               sfp1_start: str,
               sfp1_end: str,
               sfp2_start: str,
               sfp2_end: str,
               sfp3_start: str,
               sfp3_end: str,
               save_ecg: bool) -> None:

    # Load the data from the csv file to a pandas dataframe
    polar_csv_anya_data = pd.read_csv(polar_csv_anya)
    polar_csv_baba_data = pd.read_csv(polar_csv_baba)

    # Convert timestamp strings to datetime objects
    timestamp_anya = datetime.strptime(polar_csv_anya_data.columns.tolist()[0].split('_')[0], '%Y-%m-%dT%H:%M:%S.%fZ')
    timestamp_anya += timedelta(hours=1)
    timestamp_baba = datetime.strptime(polar_csv_baba_data.columns.tolist()[0].split('_')[0], '%Y-%m-%dT%H:%M:%S.%fZ')
    timestamp_baba += timedelta(hours=1)

    # Extract the date part from the timestamp
    timestamp_YYYY_mm_dd = timestamp_anya.strftime('%Y-%m-%d')

    # Convert other time strings to datetime objects
    baseline_start = datetime.strptime(timestamp_YYYY_mm_dd + ' ' + baseline_start, '%Y-%m-%d %H:%M:%S')
    baseline_end = datetime.strptime(timestamp_YYYY_mm_dd + ' ' + baseline_end, '%Y-%m-%d %H:%M:%S')
    sfp1_start = datetime.strptime(timestamp_YYYY_mm_dd + ' ' + sfp1_start, '%Y-%m-%d %H:%M:%S')
    sfp1_end = datetime.strptime(timestamp_YYYY_mm_dd + ' ' + sfp1_end, '%Y-%m-%d %H:%M:%S')
    sfp2_start = datetime.strptime(timestamp_YYYY_mm_dd + ' ' + sfp2_start, '%Y-%m-%d %H:%M:%S')
    sfp2_end = datetime.strptime(timestamp_YYYY_mm_dd + ' ' + sfp2_end, '%Y-%m-%d %H:%M:%S')
    sfp3_start = datetime.strptime(timestamp_YYYY_mm_dd + ' ' + sfp3_start, '%Y-%m-%d %H:%M:%S')
    sfp3_end = datetime.strptime(timestamp_YYYY_mm_dd + ' ' + sfp3_end, '%Y-%m-%d %H:%M:%S')

    # Load time deltas and ecg data
    ecg_data_anya = polar_csv_anya_data.iloc[:, :2].dropna().to_numpy()
    ecg_data_baba = polar_csv_baba_data.iloc[:, :2].dropna().to_numpy()


    ''' QRS from Polar H10. This is not used anymore. 
    # Load the QRS time column
    qrs_time = polar_csv_anya_data['QRS_time'].dropna()

    # Create a numpy array from the QRS time column
    ecg_r_peaks_timestamp = qrs_time.to_numpy()

    # Find the index of the closest value in ecg_data[:,0] to each value in ecg_r_peaks_timestamp
    closest_indices = np.abs(ecg_r_peaks_timestamp[:, np.newaxis] - ecg_data[:,0]).argmin(axis=1)

    # Get the R-peaks from polar H10
    r_peaks = ecg_data[closest_indices, :]
    '''

    segment_id_anya = np.zeros_like(ecg_data_anya[:, 0]) - 1
    segment_id_baba = np.zeros_like(ecg_data_baba[:, 0]) - 1

    # Calculate the timestamps for each data point
    full_timestamp_anya = np.array([timestamp_anya + timedelta(seconds=data_point) for data_point in ecg_data_anya[:,0]])
    full_timestamp_baba = np.array([timestamp_baba + timedelta(seconds=data_point) for data_point in ecg_data_baba[:,0]])

    # Assign segment IDs based on timestamps
    segment_id_anya[(full_timestamp_anya >= baseline_start) & (full_timestamp_anya <= baseline_end)] = 0
    segment_id_anya[(full_timestamp_anya >= sfp1_start) & (full_timestamp_anya <= sfp1_end)] = 1
    segment_id_anya[(full_timestamp_anya >= sfp2_start) & (full_timestamp_anya <= sfp2_end)] = 2
    segment_id_anya[(full_timestamp_anya >= sfp3_start) & (full_timestamp_anya <= sfp3_end)] = 3
    
    segment_id_baba[(full_timestamp_baba >= baseline_start) & (full_timestamp_baba <= baseline_end)] = 0
    segment_id_baba[(full_timestamp_baba >= sfp1_start) & (full_timestamp_baba <= sfp1_end)] = 1
    segment_id_baba[(full_timestamp_baba >= sfp2_start) & (full_timestamp_baba <= sfp2_end)] = 2
    segment_id_baba[(full_timestamp_baba >= sfp3_start) & (full_timestamp_baba <= sfp3_end)] = 3

    # Add the segment ID as the third column in ecg_data_anya
    ecg_data_anya = np.column_stack((full_timestamp_anya, ecg_data_anya))
    ecg_data_anya = np.column_stack((ecg_data_anya, segment_id_anya))
    
    ecg_data_baba = np.column_stack((full_timestamp_baba, ecg_data_baba))
    ecg_data_baba = np.column_stack((ecg_data_baba, segment_id_baba))

    r_peaks_anya = r_peak_detection(ecg_data_anya, sampling_rate=130)
    r_peaks_baba = r_peak_detection(ecg_data_baba, sampling_rate=130)

    test_segment_occurrence(segment_id_anya)
    test_segment_occurrence(segment_id_baba)

    if save_ecg:
        # Filter ecg_data_anya to include only valid segments
        ecg_data_anya[ecg_data_anya[:, 3] == -1, 2] = np.nan
        ecg_data_baba[ecg_data_baba[:, 3] == -1, 2] = np.nan

        '''
        # Calculate the mean and standard deviation of the ECG data using the segment intervals
        ecg_anya_mean = np.nanmean(ecg_data_anya[:, 2])
        ecg_anya_std = np.nanstd(ecg_data_anya[:, 2])
        threshold = 3
        # Calculate z-scores for each data point in the signal
        z_scores = np.abs((ecg_data_anya[:, 2] - ecg_anya_mean) / ecg_anya_std)

        # Identify outlier indices where z-score exceeds the threshold
        outlier_indices = np.where(z_scores > threshold)[0]
        ecg_data_anya[outlier_indices, 2] = np.nan
        '''
        ecg_anya_mean = np.nanmean(ecg_data_anya[:, 2])
        ecg_anya_std = np.nanstd(ecg_data_anya[:, 2])
        # sampling_rate = int(1 / np.nanmean(np.diff(ecg_data_anya[:, 1]))) # ~130 Hz
        
        ecg_baba_mean = np.nanmean(ecg_data_baba[:, 2])
        ecg_baba_std = np.nanstd(ecg_data_baba[:, 2])

        for i in tqdm(range(0, len(ecg_data_anya), 2000), total=len(ecg_data_anya)//2000+1, desc=f'Saving ECG plots for {str(id)} anya'):
            tmp_ecg_data = ecg_data_anya[i:i+2000,:]
            tmp_r_peaks = r_peaks_anya[(r_peaks_anya[:, 1] >= tmp_ecg_data[0, 1]) & (r_peaks_anya[:, 1] <= tmp_ecg_data[-1, 1])]
            plot_ecg_with_peaks_and_segments(tmp_ecg_data, tmp_r_peaks, ecg_anya_mean, ecg_anya_std, output_path=f'visualization/{str(id)}_anya/{i}-{i+2000}.png')
        
        for i in tqdm(range(0, len(ecg_data_baba), 2000), total=len(ecg_data_baba)//2000+1, desc=f'Saving ECG plots for {str(id)} baba'):
            tmp_ecg_data = ecg_data_baba[i:i+2000,:]
            tmp_r_peaks = r_peaks_baba[(r_peaks_baba[:, 1] >= tmp_ecg_data[0, 1]) & (r_peaks_baba[:, 1] <= tmp_ecg_data[-1, 1])]
            plot_ecg_with_peaks_and_segments(tmp_ecg_data, tmp_r_peaks, ecg_baba_mean, ecg_baba_std, output_path=f'visualization/{str(id)}_baba/{i}-{i+2000}.png')


    for segment_id in range(4):
        # Filter the R-peaks to include only the current segment
        r_peaks_segment_anya = r_peaks_anya[(r_peaks_anya[:,3] == segment_id),:]
        save_struct(f'data/{str(id)}_{segment_id}_anya', r_peaks_segment_anya[:,1])
        
        r_peaks_segment_baba = r_peaks_baba[(r_peaks_baba[:,3] == segment_id),:]
        save_struct(f'data/{str(id)}_{segment_id}_baba', r_peaks_segment_baba[:,1])


def save_struct(ID, rr_peaks):
    # Create a struct with the following fields:
    # - code: The session ID and segment ID and the participant ID
    # - RR_peaks: A numpy array containing the RR peaks
    # - RR_intervals: A numpy array containing the RR intervals
    struct = {
        'code': ID,
        'rr_interval': np.diff(rr_peaks),
        'rr_peaks': rr_peaks,
    }

    # Save the struct to a mat file
    mat_file_path = f'{ID}.mat'
    scipy.io.savemat(mat_file_path, struct)

if __name__ == '__main__':

    # Create the command-line interface
    parser = argparse.ArgumentParser(description='Preprocessing script')
    parser.add_argument('--id', type=int, required=True, help='Session ID')
    parser.add_argument('--polar_csv_anya', type=str, required=True, help='Path to the mother\'s polar csv file')
    parser.add_argument('--polar_csv_baba', type=str, required=True, help='Path to the baby\'s input csv file')
    parser.add_argument('--baseline_start', type=str, required=True, help='Baseline start time in HH:MM:SS format')
    parser.add_argument('--baseline_end', type=str, required=True, help='Baseline end time in HH:MM:SS format')
    parser.add_argument('--sfp1_start', type=str, required=True, help='Play 1 start time in HH:MM:SS format')
    parser.add_argument('--sfp1_end', type=str, required=True, help='Play 1 end time in HH:MM:SS format')
    parser.add_argument('--sfp2_start', type=str, required=True, help='Still Face start time in HH:MM:SS format')
    parser.add_argument('--sfp2_end', type=str, required=True, help='Still Face end time in HH:MM:SS format')
    parser.add_argument('--sfp3_start', type=str, required=True, help='Play 2 start time in HH:MM:SS format')
    parser.add_argument('--sfp3_end', type=str, required=True, help='Play 2 end time in HH:MM:SS format')
    parser.add_argument('--save_ecg_plots', action='store_true', help='Flag to save ECG plots')

    # Parse the command-line arguments
    args = parser.parse_args()

    test_time_format(args.baseline_start)
    test_time_format(args.baseline_end)
    test_time_format(args.sfp1_start)
    test_time_format(args.sfp1_end)
    test_time_format(args.sfp2_start)
    test_time_format(args.sfp2_end)
    test_time_format(args.sfp3_start)
    test_time_format(args.sfp3_end)

    preprocess(id=args.id,
               polar_csv_anya=args.polar_csv_anya,
               polar_csv_baba=args.polar_csv_baba,
               baseline_start=args.baseline_start,
               baseline_end=args.baseline_end,
               sfp1_start=args.sfp1_start,
               sfp1_end=args.sfp1_end,
               sfp2_start=args.sfp2_start,
               sfp2_end=args.sfp2_end,
               sfp3_start=args.sfp3_start,
               sfp3_end=args.sfp3_end,
               save_ecg=args.save_ecg_plots)