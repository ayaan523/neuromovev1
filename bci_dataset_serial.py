#!/usr/bin/env python3
"""
BCI Wheelchair - Offline Dataset USB Serial Controller
=====================================================
Loads the preprocessed PhysioNet dataset features, predicts the
wheelchair commands using the trained SVM, and sends them to the
ESP32 via USB serial to the ESP32.
"""

import os
import sys
import time
import serial
import serial.tools.list_ports
import pickle
import argparse
import numpy as np

# ===================== CONFIGURATION =====================
SERIAL_PORT = "/dev/ttyUSB0"
SERIAL_BAUD = 115200
SERIAL_TIMEOUT = 0.2

# Paths based on your project structure
MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
MODEL_PATH = os.path.join(MODELS_DIR, "svm_model.pkl")
FEATURES_PATH = os.path.join(MODELS_DIR, "features.npz")
MAPPING_PATH = os.path.join(MODELS_DIR, "class_mapping.pkl")


class DatasetSerialDriver:
    def __init__(self, port):
        self.requested_port = port
        self.serial = None
        self.port = None
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

    def find_serial_port(self):
        if self.requested_port and self.requested_port != "auto" and os.path.exists(self.requested_port):
            return self.requested_port

        ports = list(serial.tools.list_ports.comports())
        for preferred in ("/dev/ttyUSB0", "/dev/ttyUSB1", "/dev/ttyACM0", "/dev/ttyACM1"):
            for p in ports:
                if p.device == preferred:
                    return p.device

        for p in ports:
            if p.device.startswith("/dev/ttyUSB") or p.device.startswith("/dev/ttyACM"):
                return p.device

        return None

    def connect(self):
        self.close_serial()

        port = self.find_serial_port()
        if port is None:
            print("[SERIAL ERROR] No ESP32 USB serial port found.")
            return False

        try:
            self.serial = serial.Serial(
                port=port,
                baudrate=SERIAL_BAUD,
                timeout=SERIAL_TIMEOUT,
                write_timeout=1.0,
            )
            self.port = port
            time.sleep(2.0)

            while self.serial.in_waiting:
                self.serial.readline()

            print(f"[SERIAL] Connected to ESP32 on {port}")
            return True
        except (serial.SerialException, OSError) as e:
            print(f"[SERIAL ERROR] Failed to connect: {e}")
            self.serial = None
            self.port = None
            return False

    def send_command(self, command):
        if command not in {"forward", "backward", "left", "right", "stop"}:
            print(f"[ERROR] Invalid command: {command}")
            return False

        if self.serial is None or not self.serial.is_open:
            if not self.connect():
                return False

        try:
            self.serial.write((command + "\n").encode("utf-8"))
            self.serial.flush()
            return True
        except (serial.SerialException, OSError) as e:
            print(f"[SERIAL ERROR] Connection lost: {e}")
            if self.connect():
                try:
                    self.serial.write((command + "\n").encode("utf-8"))
                    self.serial.flush()
                    return True
                except Exception:
                    pass
            return False

    def close_serial(self):
        if getattr(self, "serial", None) is not None:
            try:
                self.serial.close()
            except Exception:
                pass
        self.serial = None
        self.port = None

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
            if self.serial:
                self.serial.close()
            print("[INFO] Exited gracefully.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default=SERIAL_PORT, help="ESP32 USB serial port, e.g. /dev/ttyUSB0")
    args = parser.parse_args()

    driver = DatasetSerialDriver(args.port)
    driver.run()

if __name__ == "__main__":
    main()