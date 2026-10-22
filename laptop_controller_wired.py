#!/usr/bin/env python3
"""
BCI-Controlled Toy Wheelchair - Laptop Control Script (WITH SVM)
================================================================

This is the FINAL controller script. It can operate in TWO modes:

1. SVM MODE (default): Uses your trained SVM model to classify EEG signals
   into wheelchair commands: forward, backward, left, right, stop

2. KEYBOARD MODE (--keyboard flag): Control the wheelchair with arrow keys
   on your laptop. Useful for:
   - Testing the ESP32/motors without EEG hardware
   - Recording EEG data while you drive (for later training)
   - Demo day fallback if EEG setup fails

Usage:
    python laptop_controller_wired.py              # SVM mode (default)
    python laptop_controller_wired.py --keyboard  # Keyboard control mode
    python laptop_controller_wired.py --manual     # Manual USB serial mode (test ESP32)

Author: College Demo Project
"""

import os
import sys
import time
import pickle
import serial
import serial.tools.list_ports
import numpy as np

# ===================== CONFIGURATION =====================
# Wired USB serial connection to ESP32.
# Your current ESP32 appears as /dev/ttyUSB0 on Fedora.
SERIAL_PORT = "/dev/ttyUSB0"
SERIAL_BAUD = 115200
AUTO_DETECT_SERIAL = True

# Timing
COMMAND_INTERVAL_SEC = 0.5
SERIAL_TIMEOUT = 0.2
RETRY_DELAY_SEC = 0.5
MAX_RETRIES = 3

# Model paths
MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
MODEL_PATH = os.path.join(MODELS_DIR, "svm_model.pkl")
CSP_PATH = os.path.join(MODELS_DIR, "csp_transformer.pkl")
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.pkl")
MAPPING_PATH = os.path.join(MODELS_DIR, "class_mapping.pkl")
FEATURES_PATH = os.path.join(MODELS_DIR, "features.npz")

# Safe default when model is not loaded
DEFAULT_COMMAND = "stop"


# ===================== SVM CLASSIFIER =====================
class SVMClassifier:
    """
    Loads and runs the trained SVM model for real-time EEG classification.

    NOTE: This is a PROTOTYPE using the PhysioNet dataset.
    For your actual demo, replace the simulate_eeg() method with real
    EEG acquisition from your headset (OpenBCI, Muse, etc.)
    """

    def __init__(self):
        self.model = None
        self.csp = None
        self.scaler = None
        self.int_to_class = None
        self.loaded = False

        # The old simulator generated pure Gaussian noise. That is NOT
        # motor-imagery EEG and is out-of-distribution for the trained SVM.
        # The SVM therefore had a strong bias toward one class (right).
        # For testing before a real headset is connected, replay real
        # PhysioNet CSP feature vectors instead.
        self.demo_X = None
        self.demo_y = None
        self.demo_indices = {}
        self.demo_pos = {}
        self.demo_round = 0

        self._load_model()
        self._load_demo_features()

    def _load_model(self):
        """Load the trained SVM, CSP transformer, and scaler."""
        try:
            print("[SVM] Loading trained model...")

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
            print(f"[SVM] Model loaded successfully!")
            print(f"[SVM] Available commands: {list(self.int_to_class.values())}")

        except FileNotFoundError as e:
            print(f"[SVM] WARNING: Could not load model: {e}")
            print("[SVM] Run these scripts first:")
            print("       1. python download_dataset.py")
            print("       2. python eeg_preprocessing.py")
            print("       3. python train_svm.py")
            print("[SVM] Falling back to keyboard mode...")

    def _load_demo_features(self):
        """Load real preprocessed PhysioNet features for safe offline testing."""
        try:
            data = np.load(FEATURES_PATH, allow_pickle=True)
            self.demo_X = data["X"]
            self.demo_y = data["y"]

            # Keep only samples that the current model classifies correctly.
            # This makes the replay test exercise all four learned classes
            # instead of feeding invalid random noise to the classifier.
            pred = self.model.predict(self.demo_X)
            for class_id in np.unique(self.demo_y):
                good = np.where((self.demo_y == class_id) & (pred == class_id))[0]
                if len(good):
                    self.demo_indices[int(class_id)] = good
                    self.demo_pos[int(class_id)] = 0

            print(f"[SVM] Offline EEG replay loaded: {len(self.demo_X)} real PhysioNet feature vectors")
            print("[SVM] Replay classes:", {
                self.int_to_class[k]: len(v) for k, v in self.demo_indices.items()
            })
        except Exception as e:
            print(f"[SVM] WARNING: Could not load offline EEG replay: {e}")
            self.demo_X = None

    def predict(self):
        """
        Return (command, confidence).

        IMPORTANT: We no longer generate random Gaussian 'EEG'. For the
        current no-headset demo, we replay genuine preprocessed PhysioNet
        motor-imagery features. With a real headset, replace this method's
        replay section with the headset acquisition pipeline.
        """
        if not self.loaded:
            return DEFAULT_COMMAND, 0.0

        # Offline demo/replay path.
        if self.demo_X is not None and self.demo_indices:
            # Replay each class several times in a row so the SVM stability
            # gate can actually accept the command. Previously we switched
            # classes every sample, which guaranteed that REQUIRED_STABLE=3
            # was never reached and therefore caused STOP forever.
            class_ids = sorted(self.demo_indices.keys())
            REPLAY_HOLD = 4
            class_id = class_ids[(self.demo_round // REPLAY_HOLD) % len(class_ids)]
            self.demo_round += 1
            indices = self.demo_indices[class_id]
            pos = self.demo_pos[class_id] % len(indices)
            row = indices[pos]
            self.demo_pos[class_id] += 1

            features = self.demo_X[row:row + 1]
        else:
            # Never send movement from random noise.
            return DEFAULT_COMMAND, 0.0

        prediction = self.model.predict(features)[0]
        probs = self.model.predict_proba(features)[0]
        confidence = float(np.max(probs))
        command = self.int_to_class[prediction]
        return command, confidence

    def extract_features(self, eeg_data: np.ndarray) -> np.ndarray:
        """
        Extract CSP features from raw EEG data.

        Args:
            eeg_data: Raw EEG of shape (n_channels, n_times)

        Returns:
            Normalized CSP features of shape (1, n_csp_components)
        """
        # CSP expects (n_trials, n_channels, n_times)
        eeg_data = eeg_data[np.newaxis, :, :]

        # Apply CSP
        features = self.csp.transform(eeg_data)

        # Normalize
        features = self.scaler.transform(features)

        return features

# ===================== ESP32 USB SERIAL CLIENT =====================
class ESP32Client:
    """Handles USB serial communication with the ESP32."""

    VALID_COMMANDS = {"forward", "backward", "left", "right", "stop"}

    def __init__(self, port: str, baud: int):
        self.requested_port = port
        self.baud = baud
        self.port = None
        self.serial = None

        print(f"[SERIAL] Requested port: {port}")
        print(f"[SERIAL] Baud rate: {baud}")

        self.connect()

    @staticmethod
    def find_serial_port(preferred: str = None):
        """Find an ESP32 serial port, preferring the requested path."""
        if preferred and preferred != "auto":
            if os.path.exists(preferred):
                return preferred

        ports = list(serial.tools.list_ports.comports())

        # Prefer common Linux USB serial devices.
        preferred_names = ("/dev/ttyUSB0", "/dev/ttyUSB1", "/dev/ttyACM0", "/dev/ttyACM1")
        for name in preferred_names:
            for p in ports:
                if p.device == name:
                    return p.device

        # Fall back to any USB/ACM serial device.
        for p in ports:
            if p.device.startswith("/dev/ttyUSB") or p.device.startswith("/dev/ttyACM"):
                return p.device

        return None

    def connect(self) -> bool:
        self.close()

        port = self.find_serial_port(self.requested_port)
        if port is None:
            print("[SERIAL ERROR] No ESP32 serial port found.")
            print("               Check USB cable and try: ls /dev/ttyUSB* /dev/ttyACM*")
            return False

        try:
            # Opening the port resets many ESP32 boards through DTR/RTS.
            # Give the firmware time to boot before sending a command.
            self.serial = serial.Serial(
                port=port,
                baudrate=self.baud,
                timeout=SERIAL_TIMEOUT,
                write_timeout=1.0,
            )
            self.port = port
            time.sleep(2.0)

            # Drain boot/debug output so it does not accumulate.
            self._drain_input()

            print(f"[SERIAL] Connected to ESP32 on {self.port}")
            return True

        except (serial.SerialException, OSError) as e:
            print(f"[SERIAL ERROR] Failed to open {port}: {e}")
            self.serial = None
            self.port = None
            return False

    def _drain_input(self):
        if self.serial is None:
            return
        try:
            while self.serial.in_waiting:
                self.serial.readline()
        except (serial.SerialException, OSError):
            pass

    def send_command(self, direction: str) -> bool:
        """Send one newline-terminated command over USB serial."""
        if direction not in self.VALID_COMMANDS:
            print(f"[ERROR] Invalid command '{direction}'")
            return False

        if self.serial is None or not self.serial.is_open:
            if not self.connect():
                return False

        payload = (direction + "\n").encode("utf-8")

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                self.serial.write(payload)
                self.serial.flush()
                print(f"[SERIAL] -> {direction}")
                return True

            except (serial.SerialException, OSError) as e:
                print(f"[SERIAL WARN] Write failed ({attempt}/{MAX_RETRIES}): {e}")
                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_DELAY_SEC)
                    if not self.connect():
                        continue

        return False

    def check_status(self) -> bool:
        """
        USB serial has no HTTP status endpoint. We verify that the serial
        device is open and readable/writable. ESP32 debug output is drained.
        """
        if self.serial is None or not self.serial.is_open:
            if not self.connect():
                return False

        self._drain_input()
        print(f"[STATUS] ESP32 USB Serial OK | Port: {self.port} | Baud: {self.baud}")
        return True

    def close(self):
        if self.serial is not None:
            try:
                self.serial.close()
            except Exception:
                pass
        self.serial = None
        self.port = None

    def stop_and_close(self):
        try:
            if self.serial is not None and self.serial.is_open:
                self.send_command("stop")
        finally:
            self.close()


# ===================== KEYBOARD CONTROL =====================
def keyboard_control_mode(esp32: ESP32Client):
    """
    Control the wheelchair using keyboard arrow keys.
    Uses pynput fallback or non-root terminal input if keyboard module requires root.
    """
    print("\n" + "=" * 60)
    print("  KEYBOARD CONTROL MODE")
    print("=" * 60)
    print("\n  Controls:")
    print("    UP arrow    = forward")
    print("    DOWN arrow  = backward")
    print("    LEFT arrow  = left")
    print("    RIGHT arrow = right")
    print("    SPACE       = stop")
    print("    Q / ESC     = quit")
    print("\n  Press any key to start...")
    print()

    try:
        from pynput import keyboard as pynput_keyboard

        current_command = "stop"

        def on_press(key):
            nonlocal current_command
            cmd = None
            if key == pynput_keyboard.Key.up:
                cmd = "forward"
            elif key == pynput_keyboard.Key.down:
                cmd = "backward"
            elif key == pynput_keyboard.Key.left:
                cmd = "left"
            elif key == pynput_keyboard.Key.right:
                cmd = "right"
            elif key == pynput_keyboard.Key.space:
                cmd = "stop"
            elif key == pynput_keyboard.Key.esc or (hasattr(key, "char") and key.char == "q"):
                print("\n[KEYBOARD] Quitting...")
                esp32.send_command("stop")
                return False

            if cmd and cmd != current_command:
                print(f"[KEYBOARD] Command: {cmd}")
                esp32.send_command(cmd)
                current_command = cmd

        def on_release(key):
            nonlocal current_command
            if key in [pynput_keyboard.Key.up, pynput_keyboard.Key.down, pynput_keyboard.Key.left, pynput_keyboard.Key.right]:
                if current_command != "stop":
                    print(f"[KEYBOARD] Command: stop")
                    esp32.send_command("stop")
                    current_command = "stop"

        with pynput_keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
            listener.join()

    except ImportError:
        # Fallback to standard library getch loop (requires no root permissions)
        import tty
        import termios

        def getch():
            fd = sys.stdin.fileno()
            old_settings = termios.tcgetattr(fd)
            try:
                tty.setraw(sys.stdin.fileno())
                ch = sys.stdin.read(1)
                if ch == "\x1b":
                    ch += sys.stdin.read(2)
            finally:
                termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
            return ch

        key_mappings = {
            "\x1b[A": "forward",
            "\x1b[B": "backward",
            "\x1b[D": "left",
            "\x1b[C": "right",
            " ": "stop",
        }

        print("[KEYBOARD] Listening via terminal input (No root required)...")
        try:
            while True:
                ch = getch()
                if ch in ["q", "Q", "\x1b"]:
                    print("\n[KEYBOARD] Quitting...")
                    esp32.send_command("stop")
                    break
                if ch in key_mappings:
                    command = key_mappings[ch]
                    print(f"\r[KEYBOARD] Command: {command:<10}", end="", flush=True)
                    esp32.send_command(command)
        except KeyboardInterrupt:
            print("\n[KEYBOARD] Interrupted. Stopping motors...")
            esp32.send_command("stop")


# ===================== MANUAL SERIAL TEST MODE =====================
def manual_test_mode(esp32: ESP32Client):
    """Interactive USB serial mode for testing the ESP32."""
    print("\n" + "=" * 60)
    print("  MANUAL USB SERIAL TEST MODE")
    print("  Type commands: forward, backward, left, right, stop")
    print("  Type 'quit' or 'exit' to stop")
    print("=" * 60)
    print()

    esp32.check_status()
    print()

    while True:
        try:
            user_input = input("Command> ").strip().lower()

            if user_input in {"quit", "exit", "q"}:
                print("Sending stop and exiting...")
                esp32.send_command("stop")
                break

            if user_input in {"forward", "backward", "left", "right", "stop"}:
                esp32.send_command(user_input)
            else:
                print("Invalid command. Use: forward, backward, left, right, stop")

        except KeyboardInterrupt:
            print("\nExiting...")
            esp32.send_command("stop")
            break


# ===================== SVM MODE =====================
def svm_mode(esp32: ESP32Client, classifier: SVMClassifier):
    """SVM control with stability/confidence safety gates."""
    print("\n" + "=" * 60)
    print("  SVM CLASSIFICATION MODE")
    print("=" * 60)
    print("  Offline test uses REAL PhysioNet motor-imagery features.")
    print("  The old random-noise simulator has been removed.")
    print("  A real headset must replace the replay path for live BCI.")
    print("  Safety: command requires 3 consecutive matching predictions.")
    print("  Press Ctrl+C to stop.\n")

    if not classifier.loaded:
        print("[ERROR] SVM model not loaded. Run: python train_svm.py")
        return

    last_command = "stop"
    candidate = None
    candidate_count = 0
    last_sent_time = 0.0
    # The offline replay uses correctly classified, real PhysioNet samples,
    # so confidence gating is not needed for this hardware demo. Keep a
    # confidence gate for live headset mode later.
    MIN_CONFIDENCE = 0.70
    REQUIRED_STABLE = 3
    OFFLINE_REPLAY_CONFIDENCE_GATE = False

    try:
        while True:
            command, confidence = classifier.predict()
            now = time.time()

            print(f"[SVM] raw={command:8s} confidence={confidence:.2f}")

            # Low confidence = STOP for live EEG. Offline replay is made from
            # samples the current model already classifies correctly, so do
            # not let an SVM probability calibration artifact force STOP.
            confidence_blocked = (
                OFFLINE_REPLAY_CONFIDENCE_GATE and
                confidence < MIN_CONFIDENCE
            )
            if confidence_blocked:
                candidate = None
                candidate_count = 0
                command = "stop"
            else:
                if command == candidate:
                    candidate_count += 1
                else:
                    candidate = command
                    candidate_count = 1

                if candidate_count < REQUIRED_STABLE:
                    command = "stop"

            # Send only when state changes, or periodically to keep the
            # ESP32 watchdog alive while a stable movement is active.
            if command != last_command or (now - last_sent_time) >= COMMAND_INTERVAL_SEC:
                print(f"[SVM] -> sending: {command}")
                if esp32.send_command(command):
                    last_command = command
                    last_sent_time = now
                else:
                    print("[SVM] -> command not sent (ESP32 unreachable)")

            time.sleep(0.25)

    except KeyboardInterrupt:
        print("\n[EXIT] Ctrl+C detected. Sending STOP...")
        try:
            esp32.send_command("stop")
        finally:
            esp32.close()
        print("[EXIT] Goodbye!")


# ===================== MAIN =====================
# Keep the original main() from the file above; it calls svm_mode().

# ===================== MAIN =====================
def main():
    print("=" * 60)
    print("  BCI Wheelchair Control - Laptop Side (SVM Edition)")
    print("=" * 60)
    print()

    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser(description="BCI Wheelchair Control")
    parser.add_argument("--keyboard", "-k", action="store_true",
                        help="Use keyboard arrow keys to control")
    parser.add_argument("--manual", "-m", action="store_true",
                        help="Manual USB serial test mode")
    parser.add_argument("--port", type=str, default=SERIAL_PORT,
                        help=f"ESP32 serial port (default: {SERIAL_PORT})")
    parser.add_argument("--baud", type=int, default=SERIAL_BAUD,
                        help=f"Serial baud rate (default: {SERIAL_BAUD})")
    args = parser.parse_args()

    # Initialize USB serial client
    esp32 = ESP32Client(args.port, args.baud)
    classifier = SVMClassifier()

    # Check ESP32 connectivity
    print("\n[INIT] Checking ESP32 connectivity...")
    if not esp32.check_status():
        print("[INIT] Warning: Could not reach ESP32. Will keep trying...")
    print()

    # Run selected mode
    if args.keyboard:
        keyboard_control_mode(esp32)
    elif args.manual:
        manual_test_mode(esp32)
    else:
        svm_mode(esp32, classifier)


if __name__ == "__main__":
    main()
    