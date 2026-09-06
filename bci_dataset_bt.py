#!/usr/bin/env python3
"""
BCI Wheelchair - Offline Dataset Bluetooth Controller
=====================================================
Loads the preprocessed PhysioNet dataset features, predicts the
wheelchair commands using the trained SVM, and sends them to the
ESP32 via direct Bluetooth RFCOMM.
"""

import os
import sys
import time
import socket
import pickle
import argparse
import numpy as np

# ===================== CONFIGURATION =====================
ESP32_BT_MAC = "58:2A:BD:7D:E8:EE"
PORT = 1

# Paths based on your project structure
MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
MODEL_PATH = os.path.join(MODELS_DIR, "svm_model.pkl")
FEATURES_PATH = os.path.join(MODELS_DIR, "features.npz")
MAPPING_PATH = os.path.join(MODELS_DIR, "class_mapping.pkl")


class DatasetBluetoothDriver:
    def __init__(self, mac):
        self.mac = mac
        self.sock = None
        self.model = None
        self.int_to_class = None
        self.X_test = None
        self.y_true = None
        
        self.load_models_and_data()
        self.connect()

    def load_models_and_data(self):
        """Loads the SVM, class mappings, and preprocessed dataset."""
        try:
            print("[INIT] Loading trained SVM and class mapping...")
            # Load SVM model and mappings[cite: 7]
            with open(MODEL_PATH, "rb") as f:
                self.model = pickle.load(f)
                
            with open(MAPPING_PATH, "rb") as f:
                mapping = pickle.load(f)
                self.int_to_class = mapping["int_to_class"]
            
            print("[INIT] Loading preprocessed EEG dataset...")
            # Load the scaled CSP features[cite: 2]
            data = np.load(FEATURES_PATH, allow_pickle=True)
            self.X_test = data["X"]
            self.y_true = data["y"]
            
            print(f"[INIT] Successfully loaded {len(self.X_test)} EEG trials.")
            print(f"[INIT] Available commands: {list(self.int_to_class.values())}")
            
        except FileNotFoundError as e:
            print(f"\n[ERROR] Missing file: {e}")
            print("Please ensure you have run the following scripts first:")
            print(" 1. python download_dataset.py")
            print(" 2. python eeg_preprocessing.py[cite: 4]")
            print(" 3. python train_svm.py[cite: 4]")
            sys.exit(1)

    def connect(self):
        if self.sock:
            try:
                self.sock.close()
            except Exception:
                pass
            
        try:
            self.sock = socket.socket(socket.AF_BLUETOOTH, socket.SOCK_STREAM, socket.BTPROTO_RFCOMM)
            self.sock.settimeout(2.0)
            self.sock.connect((self.mac, PORT))
            print(f"[BLUETOOTH] Connected to {self.mac}")
            return True
        except Exception as e:
            print(f"[BLUETOOTH ERROR] Failed to connect: {e}")
            self.sock = None
            return False

    def send_command(self, command):
        if self.sock is None:
            if not self.connect():
                return
                
        try:
            self.sock.sendall((command + "\n").encode('utf-8'))
        except (socket.error, OSError) as e:
            print(f"[BLUETOOTH ERROR] Connection lost ({e}). Attempting reconnect...")
            if self.connect():
                try:
                    self.sock.sendall((command + "\n").encode('utf-8'))
                except Exception:
                    pass

    def run(self):
        print("\n" + "=" * 60)
        print("  Starting EEG Dataset Drive Loop")
        print("  Press Ctrl+C to stop the wheelchair and exit.")
        print("=" * 60 + "\n")
        
        try:
            # Iterate through the actual EEG dataset[cite: 2]
            for i in range(len(self.X_test)):
                # Get the features for a single trial
                features = self.X_test[i].reshape(1, -1)
                
                # Predict the movement intent
                prediction = self.model.predict(features)[0]
                command = self.int_to_class[prediction]
                
                # Get the ground truth (what the volunteer was actually thinking)
                actual_label = self.int_to_class[self.y_true[i]]
                
                print(f"[TRIAL {i+1:03d}] Expected: {actual_label:<10} | Predicted: {command:<10}")
                
                # Transmit to ESP32
                self.send_command(command)
                
                # Wait before the next prediction to simulate real-time processing
                time.sleep(1.5)
                
        except KeyboardInterrupt:
            print("\n[INFO] Manual interrupt received. Stopping wheelchair...")
        finally:
            self.send_command("stop")
            if self.sock:
                self.sock.close()
            print("[INFO] Exited gracefully.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mac", default=ESP32_BT_MAC, help="ESP32 Bluetooth MAC address")
    args = parser.parse_args()

    driver = DatasetBluetoothDriver(args.mac)
    driver.run()

if __name__ == "__main__":
    main()