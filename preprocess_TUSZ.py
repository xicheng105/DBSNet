import os

import mne
import numpy as np

from pathlib import Path
from scipy.signal import resample, butter, sosfilt

from utilities import delete_exist_file


# %% get_all_session_paths
def get_all_session_paths(data_path):
    session_paths = []
    reference_type_count = {}
    all_patients = os.listdir(data_path)
    for patient in all_patients:
        patient_sessions = os.listdir(os.path.join(data_path, patient))
        for patient_session in patient_sessions:
            reference_types = os.listdir(os.path.join(data_path, patient, patient_session))
            for reference_type in reference_types:
                if reference_type not in reference_type_count:
                    reference_type_count[reference_type] = 1
                else:
                    reference_type_count[reference_type] += 1
                files = os.listdir(os.path.join(data_path, patient, patient_session, reference_type))
                sessions = []
                for file in files:
                    if file.endswith('.edf'):
                        sessions.append(file.split('.')[0])
                for session in sessions:
                    session_paths.append(
                        os.path.join(data_path, patient, patient_session, reference_type, session + '.edf')
                    )
    return session_paths, all_patients, reference_type_count


# %% get_channels_from_raw
def get_channels_from_raw(raw, reference_type="01_tcp_ar"):
    if reference_type == "01_tcp_ar":
        montage_1 = [
            'EEG FP1-REF', 'EEG F7-REF', 'EEG T3-REF', 'EEG T5-REF', 'EEG FP2-REF', 'EEG F8-REF', 'EEG T4-REF',
            'EEG T6-REF', 'EEG A1-REF', 'EEG T3-REF', 'EEG C3-REF', 'EEG CZ-REF', 'EEG C4-REF', 'EEG T4-REF',
            'EEG FP1-REF', 'EEG F3-REF', 'EEG C3-REF', 'EEG P3-REF', 'EEG FP2-REF', 'EEG F4-REF', 'EEG C4-REF',
            'EEG P4-REF'
        ]
        montage_2 = [
            'EEG F7-REF', 'EEG T3-REF', 'EEG T5-REF', 'EEG O1-REF', 'EEG F8-REF', 'EEG T4-REF', 'EEG T6-REF',
            'EEG O2-REF', 'EEG T3-REF', 'EEG C3-REF', 'EEG CZ-REF', 'EEG C4-REF', 'EEG T4-REF', 'EEG A2-REF',
            'EEG F3-REF', 'EEG C3-REF', 'EEG P3-REF', 'EEG O1-REF', 'EEG F4-REF', 'EEG C4-REF', 'EEG P4-REF',
            'EEG O2-REF'
        ]
    elif reference_type == '03_tcp_ar_a':
        montage_1 = [
            'EEG FP1-REF', 'EEG F7-REF', 'EEG T3-REF', 'EEG T5-REF', 'EEG FP2-REF', 'EEG F8-REF', 'EEG T4-REF',
            'EEG T6-REF', 'EEG T3-REF', 'EEG C3-REF', 'EEG CZ-REF', 'EEG C4-REF', 'EEG FP1-REF', 'EEG F3-REF',
            'EEG C3-REF', 'EEG P3-REF', 'EEG FP2-REF', 'EEG F4-REF', 'EEG C4-REF', 'EEG P4-REF'
        ]
        montage_2 = [
            'EEG F7-REF', 'EEG T3-REF', 'EEG T5-REF', 'EEG O1-REF', 'EEG F8-REF', 'EEG T4-REF', 'EEG T6-REF',
            'EEG O2-REF', 'EEG C3-REF', 'EEG CZ-REF', 'EEG C4-REF', 'EEG T4-REF', 'EEG F3-REF', 'EEG C3-REF',
            'EEG P3-REF', 'EEG O1-REF', 'EEG F4-REF', 'EEG C4-REF', 'EEG P4-REF', 'EEG O2-REF'
        ]
    else:
        raise ValueError("Invalid ReferenceType. Expected '01_tcp_ar' or '03_tcp_ar_a'.")
    montage_1_indices = [raw.ch_names.index(ch) for ch in montage_1]
    montage_2_indices = [raw.ch_names.index(ch) for ch in montage_2]
    try:
        signals_1 = raw.get_data(picks=montage_1_indices)
        signals_2 = raw.get_data(picks=montage_2_indices)
    except ValueError:
        print('Something is wrong when reading channels of the raw EEG signal')
        flag_wrong = True
        return flag_wrong, 0
    else:
        flag_wrong = False
    return flag_wrong, signals_1 - signals_2


# %% butter_bandpass_filter
def butter_bandpass_filter(data, low_cut, high_cut, sampling_frequency, order=3):
    nyq = 0.5 * sampling_frequency
    low = low_cut / nyq
    high = high_cut / nyq
    sos = butter(order, [low, high], btype='band', output='sos')
    y = sosfilt(sos, data)
    return y


# %% slice_signal_to_segments
def slice_signal_to_segments(data, sampling_frequency, annotation, slice_length, overlapping_ratio):
    segments = []
    if annotation[3] == "bckg":
        label = 0
    else:
        label = 1

    for i in range(
        int(float(annotation[1])) * sampling_frequency,
        int(float(annotation[2])) * sampling_frequency,
        int(slice_length * (1 - overlapping_ratio[label])) * sampling_frequency
    ):
        if i + slice_length * sampling_frequency > int(float(annotation[2])) * sampling_frequency:
            break
        one_window = []
        noise_flag = False
        incomplete_flag = False
        for j in range(data.shape[0]):
            this_channel = data[j, :][i: i + slice_length * sampling_frequency]
            if len(this_channel) != slice_length * sampling_frequency:
                incomplete_flag = True
                break
            if max(abs(this_channel)) > 500 / 10 ** 6:
                noise_flag = True
                break
            one_window.append(this_channel)

        if incomplete_flag is False and noise_flag is False and one_window:
            segments.append((np.array(one_window), label))
    return segments


# %% preprocess_TUSZ
def preprocess_TUSZ(
        slice_length=4,
        reference_type="01_tcp_ar",
        resampling_frequency=250,
        low_cut=1.5,
        high_cut=30,
        overlapping_ratio=None
):
    if overlapping_ratio is None:
        overlapping_ratio = [0, 0.75]

    # Create file directory.
    original_data_path = "/data4/louxicheng/EEG_data/seizure/TUSZ/v2.0.3/edf/"
    output_path = "/data4/louxicheng/EEG_data/seizure/TUSZ/v2.0.3/processed/"
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    processed_data_path = (
        Path(output_path)
        / f"{reference_type}_slice_length_{slice_length}_seconds"
    )
    delete_exist_file(processed_data_path)

    # Load data.
    train_data_path = Path(original_data_path) / "train"
    validation_data_path = Path(original_data_path) / "dev"
    test_data_path = Path(original_data_path) / "eval"
    folders = {
        "train": train_data_path,
        "validation": validation_data_path,
        "test": test_data_path
    }
    for folder_name, folder_path in folders.items():
        session_paths, all_patients, reference_type_count = get_all_session_paths(folder_path)
        print(f'\n{folder_name} Set:')
        print('Number of sessions:', len(session_paths))
        print('Number of patients:', len(all_patients))
        print('Reference type count:', reference_type_count)
        count_session = 0
        samples = []
        for data_path in session_paths:
            if folder_name == "train":
                reference = data_path.split("train/")[1].split("/")[2]
            elif folder_name == "validation":
                reference = data_path.split("dev/")[1].split("/")[2]
            else:
                reference = data_path.split("eval/")[1].split("/")[2]
            if reference != reference_type:
                continue
            count_session += 1
            raw = mne.io.read_raw_edf(data_path, preload=True, verbose='warning')
            flag_wrong, signals = get_channels_from_raw(raw, reference_type=reference_type)
            if flag_wrong:
                continue

            filtered_signal = []
            sampling_frequency = int(raw.info['sfreq'])
            if sampling_frequency == resampling_frequency:
                resampled_signal = signals
            else:
                resampled_signal = []
                for i in range(signals.shape[0]):
                    resampled_signal_raw = resample(
                        signals[i, :], int(len(signals[i, :]) * resampling_frequency / sampling_frequency)
                    )
                    resampled_signal.append(resampled_signal_raw)
                resampled_signal = np.array(resampled_signal)
            for i in range(resampled_signal.shape[0]):
                bandpass_filtered_signal = butter_bandpass_filter(
                    resampled_signal[i, :], low_cut, high_cut, sampling_frequency, order=3
                )
                filtered_signal.append(bandpass_filtered_signal)
            filtered_signal = np.array(filtered_signal)

            # Extract annotation.
            annotation_file_root = data_path[:-4] + '.csv_bi'
            with open(annotation_file_root, 'r') as annotation_file:
                annotation = annotation_file.readlines()
                annotation = annotation[-1]
                annotations = annotation.split(',')

            # Slicing signals.
            segments = slice_signal_to_segments(
                filtered_signal, resampling_frequency, annotations, slice_length, overlapping_ratio
            )
            for segment in segments:
                samples.append(segment)

        # save samples
        samples_path = Path(processed_data_path) / folder_name
        if not os.path.exists(samples_path):
            os.makedirs(samples_path)
        for idx, segment in enumerate(samples):
            data, label = segment
            print(f"Sample {idx} data shape: {data.shape}, Label: {label}")
            segment_path = Path(samples_path) / f'sample_{idx}.npz'
            # sys.exit(0)
            np.savez(segment_path, X=data, y=label)
