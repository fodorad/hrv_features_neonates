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

        self.ecg_data = np.column_stack((full_timestamp, self.ecg_data, segment_ids))
        self.r_peaks = self.r_peak_detection(self.sampling_frequency)

    def _check_segment_timestamps(self, raise_exception: bool = False) -> None:
        session_start = self.session_timestamps[0]
        session_end = self.session_timestamps[1]

        self.msg = ""
        self.skip_segments = [ExperimentSegment.OUTSIDE.value]
        for segment, timestamps in self.segment_timestamps.items():
            start = timestamps[0]
            end = timestamps[1]

            if start < session_start or end > session_end:
                self.msg += f"[Error] Session {self.session_id}. Participant: {self.participant_id}. " + \
                      f"\n\tSegment '{segment}' timestamps are not within the session interval. " + \
                      f"\n\tSession start: {session_start} - Session end: {session_end}, " + \
                      f"\n\tSegment start: {start} - Segment end: {end}\n"

                if raise_exception:
                    raise ValueError(self.msg)

                self.skip_segments.append(ExperimentSegment[segment.upper()].value)

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
                raise ValueError(f"Error in session {self.session_id}. {segment_names[segment_id]} segment not found." + 
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

            if segment.value in self.skip_segments:
                continue

            code = f'{self.session_id}_{segment.name}_{self.participant_id}'
            rr_peaks = self.r_peaks[self.r_peaks[:, 3] == segment.value, 1]
            
            r_peaks_ts = self.r_peaks[self.r_peaks[:, 3] == segment.value, :]
            np.savetxt(str(Path(output_dir) / f'{code}_ts.csv'), r_peaks_ts, delimiter=",", header="r_peak_timestamp,r_peak_delta_sec,r_peak_value,segment", comments='')
            df = pd.DataFrame(r_peaks_ts, columns=["r_peak_timestamp","r_peak_delta_sec","r_peak_value","segment"])
            df.to_excel(str(Path(output_dir) / f'{code}_ts.xlsx'), index=False)

            struct = {
                'code': code,
                'rr_interval': np.diff(rr_peaks),
                'rr_peaks': rr_peaks,
            }

            scipy.io.savemat(str(Path(output_dir) / f'{code}.mat'), struct)
            print('Struct is saved:', str(Path(output_dir) / f'{code}.mat'))

        return self


def read_excel(file_path: str) -> pd.DataFrame:
    df = pd.read_excel(file_path, dtype=str)
    n_rows_before = len(df)
    df.columns = [col.lower().replace(' ', '_') for col in df.columns] # normalize column names
    df['participant_code'] = df['participant_code'].astype(int)
    df = df.dropna() # drop rows with NaN values
    n_rows_after = len(df)
    print(f"Excel file is loaded: {file_path}")
    print(f"Number of rows without missing timestamps: {len(df)}")
    print(f"Number of dropped rows: {n_rows_before - n_rows_after}")
    return df


def check_args_sample(args):
    if args.id is None:
        raise ValueError(f"Missing session ID. Given value: {args.id}")
    if args.polar_csv_anya is None or not Path(args.polar_csv_anya).is_file():
        raise ValueError(f"Missing path to the mother's polar csv file. Given value: {args.polar_csv_anya}")
    if args.polar_csv_baba is None or not Path(args.polar_csv_anya).is_file():
        raise ValueError(f"Missing path to the baby's input csv file. Given value: {args.polar_csv_baba}")
    if args.baseline_start is None:
        raise ValueError(f"Session id: {args.id}; Missing baseline start time. Given value: {args.baseline_start}")
    if args.baseline_end is None:
        raise ValueError(f"Session id: {args.id}; Missing baseline end time. Given value: {args.baseline_end}")
    if args.sfp1_start is None:
        raise ValueError(f"Session id: {args.id}; Missing Play 1 start time. Given value: {args.sfp1_start}")
    if args.sfp1_end is None:
        raise ValueError(f"Session id: {args.id}; Missing Play 1 end time. Given value: {args.sfp1_end}")
    if args.sfp2_start is None:
        raise ValueError(f"Session id: {args.id}; Missing Still Face start time. Given value: {args.sfp2_start}")
    if args.sfp2_end is None:
        raise ValueError(f"Session id: {args.id}; Missing Still Face end time. Given value: {args.sfp2_end}")
    if args.sfp3_start is None:
        raise ValueError(f"Session id: {args.id}; Missing Play 2 start time. Given value: {args.sfp3_start}")
    if args.sfp3_end is None:
        raise ValueError(f"Session id: {args.id}; Missing Play 2 end time. Given value: {args.sfp3_end}")


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Preprocessing script')
    parser.add_argument('--xlsx', type=str, help='Path to the .xlsx file')
    parser.add_argument('--id', type=int, help='Session ID')
    parser.add_argument('--polar_csv_anya', type=str, help='Path to the mother\'s polar csv file')
    parser.add_argument('--polar_csv_baba', type=str, help='Path to the baby\'s input csv file')
    parser.add_argument('--baseline_start', type=str, help='Baseline start time in HH:MM:SS format')
    parser.add_argument('--baseline_end', type=str, help='Baseline end time in HH:MM:SS format')
    parser.add_argument('--sfp1_start', type=str, help='Play 1 start time in HH:MM:SS format')
    parser.add_argument('--sfp1_end', type=str, help='Play 1 end time in HH:MM:SS format')
    parser.add_argument('--sfp2_start', type=str, help='Still Face start time in HH:MM:SS format')
    parser.add_argument('--sfp2_end', type=str, help='Still Face end time in HH:MM:SS format')
    parser.add_argument('--sfp3_start', type=str, help='Play 2 start time in HH:MM:SS format')
    parser.add_argument('--sfp3_end', type=str, help='Play 2 end time in HH:MM:SS format')
    parser.add_argument('--save_ecg_plots', action='store_true', help='Flag to save ECG plots')
    args = parser.parse_args()


    if args.xlsx is not None and Path(args.xlsx).is_file():
        print("Running on multiple samples...")
        df = read_excel(args.xlsx)
        
        with open('skipped_segments.txt', 'w') as f:

            for index, row in df.iterrows():
                print("=" * 50)
                hrv_anya = SegmentedPolarECG(row['participant_code'], "A", row['polar_csv_anya'],
                                            row['baseline_start'], row['baseline_end'],
                                            row['sfp_1_start'], row['sfp_1_end'],
                                            row['sfp_2_start'], row['sfp_2_end'],
                                            row['sfp_3_start'], row['sfp_3_end']).save_structs()

                if bool(args.save_ecg_plots):
                    hrv_anya.save_ecg_plots()

                hrv_baba = SegmentedPolarECG(row['participant_code'], "B", row['polar_csv_baba'],
                                            row['baseline_start'], row['baseline_end'],
                                            row['sfp_1_start'], row['sfp_1_end'],
                                            row['sfp_2_start'], row['sfp_2_end'],
                                            row['sfp_3_start'], row['sfp_3_end']).save_structs()

                if bool(args.save_ecg_plots):
                    hrv_baba.save_ecg_plots()

                for line in hrv_anya.msg.split('\n'):
                    if line != "":
                        f.write(line + '\n')
                for line in hrv_baba.msg.split('\n'):
                    if line != "":
                        f.write(line + '\n')

    else:
        print("Running on a single sample...")
        check_args_sample(args)

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
        
    print("=" * 50, '\nDone!')