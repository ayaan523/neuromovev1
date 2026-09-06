#!/usr/bin/env python3
"""
EEG Preprocessing & Feature Extraction
======================================

This script:
1. Loads downloaded PhysioNet EDF files
2. Filters EEG signals (bandpass 8-30 Hz for motor imagery)
3. Extracts epochs around motor imagery events
4. Computes Common Spatial Pattern (CSP) features
5. Saves processed features and labels for SVM training

Usage:
    python eeg_preprocessing.py

Output:
    features.npz  - CSP features (X) and labels (y)
    scaler.pkl     - StandardScaler for normalization

Author: College Demo Project
"""

import os
import sys
import glob
import pickle
import numpy as np
import mne
from sklearn.preprocessing import StandardScaler
from mne.decoding import CSP
from mne import Epochs, pick_types
from mne.io import read_raw_edf

# Suppress MNE warnings for cleaner output
mne.set_log_level("WARNING")

# ===================== CONFIGURATION =====================
# Path to downloaded EDF files
DATASET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eeg_dataset")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# EEG processing parameters
SAMPLING_FREQ = 160          # PhysioNet dataset is 160 Hz
LOW_FREQ = 8.0               # Lower bound of mu/beta band (motor imagery)
HIGH_FREQ = 30.0             # Upper bound of beta band
TMIN = -0.5                  # Epoch start: 0.5s before cue
TMAX = 4.0                   # Epoch end: 4s after cue
CROP_TMIN = 0.5              # Use only 0.5s to 2.5s after cue for classification
CROP_TMAX = 2.5

# Channel selection - use motor cortex channels (C3, Cz, C4 are key for motor imagery)
# PhysioNet uses 10-20 system with these channel names
MOTOR_CHANNELS = ["C3", "Cz", "C4", "CP1", "CP2", "FC1", "FC2"]

# Event mapping for PhysioNet EEGMMIDB
# T1 = left fist, T2 = right fist (for runs 3, 7, 11)
# T1 = both fists, T2 = both feet (for runs 4, 8, 12)
# We'll map these to our 4 wheelchair directions:
#   left fist     -> left
#   right fist    -> right
#   both fists    -> forward
#   both feet     -> backward
#   rest/baseline -> stop

# ===================== LOAD AND PROCESS FILES =====================
def load_and_process_file(filepath: str) -> tuple:
    """
    Load a single EDF file, filter, and extract epochs.

    Returns:
        epochs_data: np.ndarray of shape (n_trials, n_channels, n_times)
        labels: np.ndarray of shape (n_trials,)
        run_type: str indicating which motor task this run contains
    """
    print(f"  Loading {os.path.basename(filepath)}...")

    # Load EDF file
    raw = read_raw_edf(filepath, preload=True)

    # Set standard 10-20 montage for channel names
    raw.rename_channels(lambda x: x.replace("EEG ", "").replace("-REF", "").strip())

    # Pick only EEG channels
    raw.pick_types(meg=False, eeg=True, stim=False, eog=False)

    # Filter: bandpass 8-30 Hz (mu and beta rhythms - key for motor imagery)
    raw.filter(LOW_FREQ, HIGH_FREQ, fir_design="firwin", verbose=False)

    # Get events from annotations
    events, event_id = mne.events_from_annotations(raw)

    # Determine run type from filename
    # Run numbers: 3,4,7,8,11,12
    # R03, R07, R11 = left vs right fist
    # R04, R08, R12 = both fists vs both feet
    basename = os.path.basename(filepath)
    run_num = int(basename.split("R")[1][:2])

    if run_num in [3, 7, 11]:
        run_type = "left_right"
        event_map = {"T1": "left", "T2": "right"}
    else:
        run_type = "fists_feet"
        event_map = {"T1": "forward", "T2": "backward"}

    # Find available events in this file
    available_events = {}
    for old_name, new_name in event_map.items():
        if old_name in event_id:
            available_events[new_name] = event_id[old_name]

    if not available_events:
        print(f"    Warning: No valid events found in {basename}, skipping.")
        return None, None, run_type

    # Extract epochs
    picks = pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False)

    epochs = Epochs(
        raw,
        events=events,
        event_id=available_events,
        tmin=TMIN,
        tmax=TMAX,
        proj=True,
        picks=picks,
        baseline=None,
        preload=True,
        verbose=False,
    )

    # Crop to the motor imagery period (0.5s to 2.5s after cue)
    epochs.crop(tmin=CROP_TMIN, tmax=CROP_TMAX)

    # Get data and labels
    data = epochs.get_data()  # Shape: (n_trials, n_channels, n_times)
    labels = epochs.events[:, -1]  # Event codes

    # Map event codes to class names
    label_map = {v: k for k, v in available_events.items()}
    label_names = np.array([label_map[l] for l in labels])

    print(f"    -> Extracted {len(data)} trials: {dict(zip(*np.unique(label_names, return_counts=True)))}")

    return data, label_names, run_type


# ===================== MAIN PIPELINE =====================
def main():
    print("=" * 60)
    print("  EEG Preprocessing & Feature Extraction")
    print("=" * 60)
    print()

    # Find all EDF files
    edf_files = sorted(glob.glob(os.path.join(DATASET_DIR, "*.edf")))

    if not edf_files:
        print(f"ERROR: No EDF files found in {DATASET_DIR}")
        print("Please run: python download_dataset.py")
        sys.exit(1)

    print(f"Found {len(edf_files)} EDF files to process\n")

    # Collect all data
    all_data = []
    all_labels = []

    for filepath in edf_files:
        data, labels, run_type = load_and_process_file(filepath)
        if data is not None and len(data) > 0:
            all_data.append(data)
            all_labels.append(labels)

    if not all_data:
        print("ERROR: No valid data extracted from any file!")
        sys.exit(1)

    # Concatenate all data
    X = np.concatenate(all_data, axis=0)  # (n_total_trials, n_channels, n_times)
    y = np.concatenate(all_labels, axis=0)  # (n_total_trials,)

    print(f"\n{'='*60}")
    print(f"  Total trials collected: {len(X)}")
    print(f"  Data shape: {X.shape}")
    print(f"  Classes: {np.unique(y)}")

    # Print class distribution
    unique, counts = np.unique(y, return_counts=True)
    print(f"  Class distribution:")
    for cls, count in zip(unique, counts):
        print(f"    {cls}: {count} trials")

    # ===================== CSP FEATURE EXTRACTION =====================
    print(f"\n{'='*60}")
    print("  Computing Common Spatial Pattern (CSP) features...")
    print(f"{'='*60}")

    # CSP reduces dimensionality and maximizes class discrimination
    # We'll use 6 CSP components (good balance for 4-class problem)
    n_csp_components = 6

    # CSP expects integer labels, so we need to encode
    class_names = sorted(np.unique(y))
    class_to_int = {name: i for i, name in enumerate(class_names)}
    y_encoded = np.array([class_to_int[label] for label in y])

    print(f"  Class encoding: {class_to_int}")

    # Apply CSP
    csp = CSP(n_components=n_csp_components, reg=None, log=True, norm_trace=False)

    # CSP expects (n_trials, n_channels, n_times)
    X_csp = csp.fit_transform(X, y_encoded)

    print(f"  CSP features shape: {X_csp.shape}")
    print(f"  CSP components: {n_csp_components}")

    # ===================== NORMALIZATION =====================
    print("\n  Normalizing features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_csp)

    # ===================== SAVE EVERYTHING =====================
    print(f"\n{'='*60}")
    print("  Saving processed data...")
    print(f"{'='*60}")

    # Save features and labels
    features_path = os.path.join(OUTPUT_DIR, "features.npz")
    np.savez(features_path,
             X=X_scaled,
             y=y_encoded,
             class_names=class_names,
             class_to_int=class_to_int)
    print(f"  ✅ Features saved: {features_path}")

    # Save scaler
    scaler_path = os.path.join(OUTPUT_DIR, "scaler.pkl")
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    print(f"  ✅ Scaler saved: {scaler_path}")

    # Save CSP transformer
    csp_path = os.path.join(OUTPUT_DIR, "csp_transformer.pkl")
    with open(csp_path, "wb") as f:
        pickle.dump(csp, f)
    print(f"  ✅ CSP transformer saved: {csp_path}")

    print(f"\n{'='*60}")
    print("  ✅ PREPROCESSING COMPLETE!")
    print(f"{'='*60}")
    print(f"\n  Next step: python train_svm.py")
    print()


if __name__ == "__main__":
    main()
