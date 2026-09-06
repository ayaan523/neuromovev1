#!/usr/bin/env python3
"""
SVM Classifier - Offline Terminal Test
========================================

Runs your trained SVM classifier in a loop and prints predicted commands
to the terminal. No ESP32 connection needed - useful for showing the
classification pipeline working on its own.

Prerequisites (run these first if not done already):
    1. python download_dataset.py
    2. python eeg_preprocessing.py
    3. python train_svm.py

Usage:
    python test_svm_offline.py
    python test_svm_offline.py --count 20      # Run 20 predictions then stop
    python test_svm_offline.py --interval 1.0  # Slower, easier to read
"""

import os
import sys
import time
import pickle
import argparse
import numpy as np

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
MODEL_PATH   = os.path.join(MODELS_DIR, "svm_model.pkl")
CSP_PATH     = os.path.join(MODELS_DIR, "csp_transformer.pkl")
SCALER_PATH  = os.path.join(MODELS_DIR, "scaler.pkl")
MAPPING_PATH = os.path.join(MODELS_DIR, "class_mapping.pkl")


class SVMClassifier:
    def __init__(self):
        self.loaded = False
        print("[SVM] Loading trained model...")
        try:
            with open(MODEL_PATH, "rb") as f:
                self.model = pickle.load(f)
            with open(CSP_PATH, "rb") as f:
                self.csp = pickle.load(f)
            with open(SCALER_PATH, "rb") as f:
                self.scaler = pickle.load(f)
            with open(MAPPING_PATH, "rb") as f:
                mapping = pickle.load(f)
                self.int_to_class = mapping["int_to_class"]
            self.loaded = True
            print(f"[SVM] Model loaded. Classes: {list(self.int_to_class.values())}\n")
        except FileNotFoundError as e:
            print(f"[SVM] ERROR: {e}")
            print("[SVM] Run these first:")
            print("       1. python download_dataset.py")
            print("       2. python eeg_preprocessing.py")
            print("       3. python train_svm.py")
            sys.exit(1)

    def simulate_eeg(self) -> np.ndarray:
        """Stand-in for a live headset feed - swap this out for real acquisition later."""
        n_channels, n_times = 64, 250
        return np.random.randn(n_channels, n_times) * 5.0

    def predict(self) -> tuple:
        eeg_data = self.simulate_eeg()[np.newaxis, :, :]
        features = self.csp.transform(eeg_data)
        features = self.scaler.transform(features)

        prediction = self.model.predict(features)[0]
        probs = self.model.predict_proba(features)[0]
        confidence = float(np.max(probs))
        command = self.int_to_class[prediction]

        return command, confidence


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=0,
                         help="Number of predictions to run (0 = run forever, Ctrl+C to stop)")
    parser.add_argument("--interval", type=float, default=0.5,
                         help="Seconds between predictions")
    args = parser.parse_args()

    print("=" * 60)
    print("  BCI Wheelchair - Offline SVM Classification Test")
    print("=" * 60)
    print()

    classifier = SVMClassifier()

    print("[TEST] Starting classification loop (no ESP32 connection needed)")
    print("[TEST] Press Ctrl+C to stop\n")

    i = 0
    try:
        while args.count == 0 or i < args.count:
            command, confidence = classifier.predict()
            print(f"  [{i+1:>4}] Predicted command: {command:<10}  confidence: {confidence:.2f}")
            i += 1
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\n[TEST] Stopped.")

    print(f"\n[TEST] Total predictions made: {i}")


if __name__ == "__main__":
    main()
