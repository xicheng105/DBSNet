import mne
import pandas as pd
import numpy as np
import json
import glob
import os
from collections import Counter

def find_segments(label_array, target_label):
    segments = []
    in_segment = False
    onset = 0
    for i, label in enumerate(label_array):
        if label == target_label and not in_segment:
            in_segment = True
            onset = i
        elif label != target_label and in_segment:
            in_segment = False
            offset = i
            segments.append((onset, offset))
    if in_segment:
        segments.append((onset, len(label_array)))
    return segments

def preprocess_Siena(window_size_sec=4, seizure_overlap=0.75, background_overlap=0.0):
    base_path = "/data4/louxicheng/EEG_data/seizure/Siena_BIDS/"
    output_root = "/data4/louxicheng/EEG_data/seizure/Siena/processed"
    os.makedirs(output_root, exist_ok=True)

    # Slice parameters.
    window_size_sec = window_size_sec  # The duration of each slice, in seconds.
    seizure_overlap = seizure_overlap  # Overlap ratio of epileptic seizure slices.
    background_overlap = background_overlap  # Overlap ratio of background slices.

    edf_files = glob.glob(os.path.join(base_path, "sub-*/ses-*/eeg/*.edf"))
    print(f"Find {len(edf_files)} EDF files.")

    sample_index = 0

    for edf_path in edf_files:
        filepath = os.path.dirname(edf_path)  # Extract the file folder name.
        basename = os.path.basename(edf_path).replace("_eeg.edf", "")
        subject_id = basename.split("_")[0]

        json_path = os.path.join(filepath, basename + "_eeg.json")
        event_path = os.path.join(filepath, basename + "_events.tsv")

        if not (os.path.exists(json_path) and os.path.exists(event_path)):
            print(f"Warning: Missing supporting JSON or TSV file, skipping {edf_path}")
            continue

        with open(json_path, 'r') as f:
            eeg_metadata = json.load(f)
        sampling_rate = eeg_metadata['SamplingFrequency']

        raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
        data = raw.get_data()
        print(f"\nData shape：{data.shape} (channels，sampling point)")

        events = pd.read_csv(event_path, sep='\t')
        print(events.loc[0])
        n_time_points = data.shape[1]
        labels = np.zeros(n_time_points)

        for _, row in events.iterrows():
            onset_sample = int(row['onset'] * sampling_rate)
            offset_sample = int((row['onset'] + row['duration']) * sampling_rate)
            labels[onset_sample:offset_sample] = 1

        print(f"Unique labels after marking: {np.unique(labels)}")

        seizure_segments = find_segments(labels, target_label=1)
        background_segments = find_segments(labels, target_label=0)

        window_size = int(window_size_sec * sampling_rate)
        seizure_stride = int(window_size * (1 - seizure_overlap))
        background_stride = int(window_size * (1 - background_overlap))

        subject_output_dir = os.path.join(output_root, subject_id)
        os.makedirs(subject_output_dir, exist_ok=True)

        for segments, label in [(seizure_segments, 1), (background_segments, 0)]:
            for start, end in segments:
                i = start
                while i + window_size <= end:
                    segment = data[:, i:i + window_size]

                    save_path = os.path.join(subject_output_dir, f"sample_{sample_index}.npz")
                    np.savez(save_path, X=segment, y=label)
                    sample_index += 1

                    i += seizure_stride if label == 1 else background_stride

        print(f"Done! sub-{subject_id[-2:]}：Saved {sample_index} samples to {subject_output_dir}")