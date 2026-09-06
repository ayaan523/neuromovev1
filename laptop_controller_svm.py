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
    python laptop_controller_svm.py              # SVM mode (default)
    python laptop_controller_svm.py --keyboard  # Keyboard control mode
    python laptop_controller_svm.py --manual     # Manual HTTP mode (test ESP32)

Author: College Demo Project
"""

import os
import sys
import time
import pickle
import requests
import numpy as np

# ===================== CONFIGURATION =====================
ESP32_IP = "192.168.1.100"   # <-- UPDATE THIS after checking Serial Monitor
ESP32_PORT = 80

# Timing
COMMAND_INTERVAL_SEC = 0.5       # Minimum time between commands
CONNECTION_TIMEOUT = 3
RETRY_DELAY_SEC = 2
MAX_RETRIES = 3

# Model paths
MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
MODEL_PATH = os.path.join(MODELS_DIR, "svm_model.pkl")
CSP_PATH = os.path.join(MODELS_DIR, "csp_transformer.pkl")
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.pkl")
MAPPING_PATH = os.path.join(MODELS_DIR, "class_mapping.pkl")

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

        self._load_model()

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

    def simulate_eeg(self) -> np.ndarray:
        """
        PROTOTYPE: Simulates EEG data for demonstration.

        Fixed to 64 channels to match the PhysioNet dataset model.
        """
        n_channels = 64
        n_times = 250
        return np.random.randn(n_channels, n_times) * 5.0

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

    def predict(self) -> str:
        """
        Get the next command from the SVM classifier.

        Returns:
            str: One of "forward", "backward", "left", "right", "stop"
        """
        if not self.loaded:
            return DEFAULT_COMMAND

        # Acquire EEG data (replace with real headset in production)
        eeg_data = self.simulate_eeg()

        # Extract features
        features = self.extract_features(eeg_data)

        # Predict
        prediction = self.model.predict(features)[0]
        command = self.int_to_class[prediction]

        return command


# ===================== ESP32 CLIENT =====================
class ESP32Client:
    """Handles HTTP communication with the ESP32 wheelchair controller."""

    def __init__(self, ip: str, port: int):
        self.base_url = f"http://{ip}:{port}"
        self.session = requests.Session()
        self.session.timeout = CONNECTION_TIMEOUT
        self.consecutive_failures = 0
        self.max_failures = 5

        print(f"[CLIENT] Initialized for ESP32 at {self.base_url}")

    def send_command(self, direction: str) -> bool:
        """Send a movement command to the ESP32."""
        valid_commands = {"forward", "backward", "left", "right", "stop"}
        if direction not in valid_commands:
            print(f"[ERROR] Invalid command '{direction}'")
            return False

        url = f"{self.base_url}/move"
        params = {"dir": direction}

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                response = self.session.get(url, params=params)

                if response.status_code == 200:
                    self.consecutive_failures = 0
                    data = response.json()

                    if data.get("status") == "blocked":
                        print(f"  -> ESP32 BLOCKED: {data.get('reason')} "
                              f"(distance: {data.get('distance_cm', '?')} cm)")

                    return True

            except requests.exceptions.ConnectionError:
                print(f"[WARN] Connection refused (attempt {attempt}/{MAX_RETRIES})")
            except requests.exceptions.Timeout:
                print(f"[WARN] Request timed out (attempt {attempt}/{MAX_RETRIES})")
            except requests.exceptions.RequestException as e:
                print(f"[WARN] Request error: {e} (attempt {attempt}/{MAX_RETRIES})")

            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY_SEC)

        self.consecutive_failures += 1

        if self.consecutive_failures >= self.max_failures:
            print("\n[ERROR] ESP32 appears to be unreachable!")
            print("        Check that:")
            print("        1. The ESP32 is powered on and connected to WiFi")
            print("        2. The IP address in this script is correct")
            print("        3. Both devices are on the same WiFi network")
            print(f"        4. The ESP32 web server is running on port {ESP32_PORT}")
            self.consecutive_failures = 0

        return False

    def check_status(self) -> bool:
        """Quick health check to verify ESP32 is reachable."""
        try:
            response = self.session.get(f"{self.base_url}/", timeout=2)
            if response.status_code == 200:
                data = response.json()
                print(f"[STATUS] ESP32 OK | Current: {data.get('current_command', '?')} | "
                      f"Distance: {data.get('distance_cm', '?')} cm | "
                      f"Obstacle: {data.get('obstacle', '?')}")
                return True
        except Exception as e:
            print(f"[STATUS] ESP32 unreachable: {e}")
        return False


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


# ===================== MANUAL HTTP MODE =====================
def manual_test_mode(esp32: ESP32Client):
    """Interactive mode for testing ESP32 via typed commands."""
    print("\n" + "=" * 60)
    print("  MANUAL TEST MODE")
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
    """
    Main control loop using SVM classifier.
    """
    print("\n" + "=" * 60)
    print("  SVM CLASSIFICATION MODE")
    print("=" * 60)
    print("\n  This mode uses your trained SVM model to classify")
    print("  EEG signals into wheelchair commands.")
    print("\n  NOTE: Currently using simulated EEG data.")
    print("  Replace simulate_eeg() with real headset code for demo.")
    print("\n  Press Ctrl+C to stop.\n")

    if not classifier.loaded:
        print("[ERROR] SVM model not loaded. Cannot run SVM mode.")
        print("        Run: python train_svm.py")
        print("        Or use: python laptop_controller_svm.py --keyboard")
        return

    last_command = None
    last_sent_time = 0

    try:
        while True:
            # Get command from SVM classifier
            command = classifier.predict()

            current_time = time.time()

            # Only send if command changed or enough time passed
            if command != last_command or (current_time - last_sent_time) >= COMMAND_INTERVAL_SEC:

                print(f"[SVM] Predicted command: {command}")

                success = esp32.send_command(command)

                if success:
                    last_command = command
                    last_sent_time = current_time
                else:
                    print("  -> Command not sent (ESP32 unreachable)")

                print()

            time.sleep(0.1)

    except KeyboardInterrupt:
        print("\n[EXIT] Ctrl+C detected. Stopping...")
        esp32.send_command("stop")
        print("[EXIT] Goodbye!")


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
                        help="Manual HTTP test mode")
    parser.add_argument("--ip", type=str, default=ESP32_IP,
                        help=f"ESP32 IP (default: {ESP32_IP})")
    parser.add_argument("--port", type=int, default=ESP32_PORT,
                        help=f"ESP32 port (default: {ESP32_PORT})")
    args = parser.parse_args()

    # Override config (module-level constants)
    esp32_ip = args.ip
    esp32_port = args.port

    # Initialize components
    esp32 = ESP32Client(esp32_ip, esp32_port)
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