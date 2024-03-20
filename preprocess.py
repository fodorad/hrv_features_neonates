import argparse
from enum import Enum
from pathlib import Path
from datetime import datetime, timedelta
import scipy.io
import neurokit2
from tqdm import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


COLORS = {
    -1: 'gray',
    0: 'black',
    1: 'blue',
    2: 'green',
    3: 'magenta'
}


class ExperimentSegment(Enum):
    OUTSIDE = -1
    BASELINE = 0
    SFP1 = 1
    SFP2 = 2
    SFP3 = 3


class PolarECG:

    def __init__(self, session_id: int, participant_id: str, csv_file: str):
        self.session_id = session_id
        self.participant_id = participant_id
        self.csv_file = csv_file
        self.csv_data = pd.read_csv(self.csv_file) # pandas dataframe
        self.ecg_data = self.csv_data.iloc[:, :2].dropna().to_numpy() # (N, 2)
        self.sampling_frequency = self._determine_sampling_frequency()
        session_start = datetime.strptime(self.csv_data.columns.tolist()[0].split('_')[0], '%Y-%m-%dT%H:%M:%S.%fZ') + timedelta(hours=1) # timezone correction
        self.session_timestamps = (session_start, session_start + timedelta(seconds=self.ecg_data[-1, 0]))
        print(f"CSV file is loaded: {csv_file}")
        print(f"Participant ID: {self.participant_id}")
        print(f"Session start: {self.session_timestamps[0].strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Session end: {self.session_timestamps[1].strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Number of data points: {len(self.ecg_data)}")
        print(f"Mean sampling frequency: {self.sampling_frequency} Hz")

    def _determine_sampling_frequency(self) -> int:
        delta_time = np.diff(self.ecg_data[:, 0])
        average_delta_time = np.mean(delta_time)
        sampling_frequency = 1 / average_delta_time
        return np.rint(sampling_frequency).astype(int)


class SegmentedPolarECG(PolarECG):

    def __init__(self, session_id: int,
                 participant_id: str,
                 csv_file: str,
                 baseline_start: str,
                 baseline_end: str,
                 sfp1_start: str,
                 sfp1_end: str,
                 sfp2_start: str,
                 sfp2_end: str,
                 sfp3_start: str,
                 sfp3_end: str):

        super().__init__(session_id, participant_id, csv_file)
        self.segment_timestamps = {}
        self.segment_timestamps['baseline'] = (self._parse_segment_timestamp(baseline_start), self._parse_segment_timestamp(baseline_end))
        self.segment_timestamps['sfp1'] = (self._parse_segment_timestamp(sfp1_start), self._parse_segment_timestamp(sfp1_end))
        self.segment_timestamps['sfp2'] = (self._parse_segment_timestamp(sfp2_start), self._parse_segment_timestamp(sfp2_end))
        self.segment_timestamps['sfp3'] = (self._parse_segment_timestamp(sfp3_start), self._parse_segment_timestamp(sfp3_end))
        self._check_segment_timestamps()

        full_timestamp = np.array([self.session_timestamps[0] + timedelta(seconds=data_point) for data_point in self.ecg_data[:, 0]])
        segment_ids = np.zeros_like(self.ecg_data[:, 0]) - 1
        segment_ids[(full_timestamp >= self.segment_timestamps['baseline'][0]) & (full_timestamp <= self.segment_timestamps['baseline'][1])] = ExperimentSegment.BASELINE.value
        segment_ids[(full_timestamp >= self.segment_timestamps['sfp1'][0]) & (full_timestamp <= self.segment_timestamps['sfp1'][1])] = ExperimentSegment.SFP1.value
        segment_ids[(full_timestamp >= self.segment_timestamps['sfp2'][0]) & (full_timestamp <= self.segment_timestamps['sfp2'][1])] = ExperimentSegment.SFP2.value
        segment_ids[(full_timestamp >= self.segment_timestamps['sfp3'][0]) & (full_timestamp <= self.segment_timestamps['sfp3'][1])] = ExperimentSegment.SFP3.value
        self._check_segment_occurrence(segment_ids)

        self.ecg_data = np.column_stack((full_timestamp, self.ecg_data, segment_ids))
        self.r_peaks = self.r_peak_detection(self.sampling_frequency)

    def _check_segment_timestamps(self) -> None:
        session_start = self.session_timestamps[0]
        session_end = self.session_timestamps[1]

        for segment, timestamps in self.segment_timestamps.items():
            start = timestamps[0]
            end = timestamps[1]

            if start < session_start or end > session_end:
                raise ValueError(f"Segment '{segment}' timestamps are not within the session interval." +
                                 f"Session start: {session_start} - Session end: {session_end}, " +
                                 f"Segment start: {start} - Segment end: {end}")

    def _parse_segment_timestamp(self, timestamp: str) -> datetime:
        return datetime.strptime(self.session_timestamps[0].strftime('%Y-%m-%d') + ' ' + timestamp, '%Y-%m-%d %H:%M:%S')

    def r_peak_detection(self, sampling_rate: int = 130) -> np.ndarray:
        signal = self.ecg_data[:, 2]
        _, results = neurokit2.ecg_peaks(signal, sampling_rate=sampling_rate)
        r_peaks = self.ecg_data[results["ECG_R_Peaks"], :]
        return r_peaks

    def _check_segment_occurrence(self, segment_ids) -> None:
        segment_names = {segment.value: segment.name for segment in ExperimentSegment}

        for segment_id in segment_names.keys():
            if sum(segment_ids == segment_id) == 0:
                raise ValueError(f"Error: {segment_names[segment_id]} segment not found." + 
                                 "Probably there is no overlap between the session and segment timestamps.")

    def save_ecg_plots(self, n_points: int = 2000, output_dir: str = 'visualization') -> None:
        ecg_mean = np.nanmean(self.ecg_data[self.ecg_data[:, 3] != ExperimentSegment.OUTSIDE.value, 2])
        ecg_std = np.nanstd(self.ecg_data[self.ecg_data[:, 3] != ExperimentSegment.OUTSIDE.value, 2])

        for ind in tqdm(range(0, len(self.ecg_data), n_points), total=len(self.ecg_data) // n_points + 1, desc=f'Saving ECG plots for {str(self.session_id) + " " + self.participant_id}'):
            part_ecg_data = self.ecg_data[ind:ind + n_points, :]
            part_r_peaks = self.r_peaks[(self.r_peaks[:, 1] >= part_ecg_data[0, 1]) & (self.r_peaks[:, 1] <= part_ecg_data[-1, 1])]
            SegmentedPolarECG.plot_ecg_with_peaks_and_segments(part_ecg_data, part_r_peaks, ecg_mean, ecg_std, 
                                                               output_path=str(Path(output_dir) / f'{str(self.session_id)}_{self.participant_id}' / f'{ind}-{ind + n_points}.png'))

    @classmethod
    def plot_ecg_with_peaks_and_segments(cls, ecg_data: np.ndarray, r_peaks: np.ndarray, ecg_mean: float, ecg_std: float, output_path: str):
        plt.figure(figsize=(15, 5))
        for i in range(-1, 4, 1):
            mask = ecg_data[:, 3] == i
            plt.plot(ecg_data[mask, 1], ecg_data[mask, 2], c=COLORS[i])
        plt.scatter(r_peaks[:, 1], r_peaks[:, 2], s=10, color='red')
        plt.xlabel('Time')
        plt.ylabel('Voltage')
        plt.xlim(ecg_data[0, 1], ecg_data[-1, 1])
        plt.ylim(ecg_mean - 10 * ecg_std, ecg_mean + 10 * ecg_std)
        plt.title('Raw ECG Data')
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path)
        plt.close()

    def save_structs(self, output_dir: str = 'data') -> 'SegmentedPolarECG':
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        for segment in ExperimentSegment:

            if segment == ExperimentSegment.OUTSIDE:
                continue

            code = f'{self.session_id}_{segment.name}_{self.participant_id}'
            rr_peaks = self.r_peaks[self.r_peaks[:, 3] == segment.value, 1]

            struct = {
                'code': code,
                'rr_interval': np.diff(rr_peaks),
                'rr_peaks': rr_peaks,
            }

            scipy.io.savemat(str(Path(output_dir) / f'{code}.mat'), struct)
            print('Struct is saved:', str(Path(output_dir) / f'{code}.mat'))

        return self


if __name__ == '__main__':

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
    args = parser.parse_args()

    hrv_anya = SegmentedPolarECG(args.id, "A", args.polar_csv_anya,
                                 args.baseline_start, args.baseline_end,
                                 args.sfp1_start, args.sfp1_end,
                                 args.sfp2_start, args.sfp2_end,
                                 args.sfp3_start, args.sfp3_end).save_structs()

    if bool(args.save_ecg_plots):
        hrv_anya.save_ecg_plots()

    hrv_baba = SegmentedPolarECG(args.id, "B", args.polar_csv_baba,
                                 args.baseline_start, args.baseline_end,
                                 args.sfp1_start, args.sfp1_end,
                                 args.sfp2_start, args.sfp2_end,
                                 args.sfp3_start, args.sfp3_end).save_structs()

    if bool(args.save_ecg_plots):
        hrv_baba.save_ecg_plots()