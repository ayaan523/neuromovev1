#!/usr/bin/env python3
"""
BCI Wheelchair - Advanced Hybrid GUI
====================================
Combines Manual Control (Arrow Keys/Buttons) with an Automated BCI Dataset Mode.
Features live EEG waveform visualization and real-time accuracy/confidence tracking.
"""

import os
import sys
import time
import serial
import serial.tools.list_ports
import pickle
import argparse
import numpy as np
import tkinter as tk
from tkinter import messagebox

# Matplotlib for embedded graphs
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

# ===================== CONFIGURATION =====================
SERIAL_PORT = "/dev/ttyUSB0"
SERIAL_BAUD = 115200
AUTO_DETECT_SERIAL = True

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
MODEL_PATH = os.path.join(MODELS_DIR, "svm_model.pkl")
FEATURES_PATH = os.path.join(MODELS_DIR, "features.npz")
MAPPING_PATH = os.path.join(MODELS_DIR, "class_mapping.pkl")


class HybridBCIGUI:
    def __init__(self, root, port):
        self.root = root
        self.root.title("NeuroMove Advanced Controller")
        self.requested_port = port
        self.serial_port = None
        self.serial = None
        
        # State variables
        self.serial = None
        self.auto_mode = False
        self.current_trial = 0
        self.correct_predictions = 0
        self.total_predictions = 0
        
        # Models and Data
        self.model = None
        self.int_to_class = None
        self.X_test = None
        self.y_true = None
        
        # Layout setup
        self.setup_ui()
        self.load_models()
        self.connect_serial()
        self.bind_keys()
        
        # Start GUI update loops
        self.root.after(100, self.bci_auto_loop)

    def load_models(self):
        """Loads SVM, mappings, and features[cite: 2, 7]."""
        try:
            with open(MODEL_PATH, "rb") as f:
                self.model = pickle.load(f)
            with open(MAPPING_PATH, "rb") as f:
                self.int_to_class = pickle.load(f)["int_to_class"]
            
            data = np.load(FEATURES_PATH, allow_pickle=True)
            self.X_test = data["X"]
            self.y_true = data["y"]
            self.log_msg(f"Loaded {len(self.X_test)} EEG trials.")
        except Exception as e:
            messagebox.showerror("Model Error", f"Missing models/data: {e}\nRun training scripts first[cite: 4].")
            self.root.destroy()

    def find_serial_port(self):
        """Find the ESP32 USB serial port."""
        if self.requested_port and self.requested_port != "auto":
            if os.path.exists(self.requested_port):
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

    def connect_serial(self):
        """Open the ESP32 USB serial connection."""
        self.close_serial()

        port = self.find_serial_port()
        if port is None:
            self.status_lbl.config(text="USB: ESP32 not found", fg="red")
            self.log_msg("No /dev/ttyUSB* or /dev/ttyACM* device found.")
            return False

        try:
            self.serial = serial.Serial(
                port=port,
                baudrate=SERIAL_BAUD,
                timeout=0.2,
                write_timeout=1.0,
            )
            self.serial_port = port

            # ESP32 may reset when the serial port is opened.
            time.sleep(2.0)

            self._drain_serial()

            self.status_lbl.config(text=f"USB Serial: Connected ({port})", fg="green")
            self.log_msg(f"Connected to ESP32 on {port} @ {SERIAL_BAUD}")
            return True

        except (serial.SerialException, OSError) as e:
            self.serial = None
            self.serial_port = None
            self.status_lbl.config(text="USB: Connection failed", fg="red")
            self.log_msg(f"Serial connection failed: {e}")
            return False

    def _drain_serial(self):
        if self.serial is None:
            return
        try:
            while self.serial.in_waiting:
                self.serial.readline()
        except (serial.SerialException, OSError):
            pass

    def send_cmd(self, command, source="Manual"):
        """Send a command to the ESP32 over USB serial."""
        self.cmd_lbl.config(text=f"Sent: {command.upper()} ({source})")

        if command not in {"forward", "backward", "left", "right", "stop"}:
            self.log_msg(f"Invalid command: {command}")
            return False

        if self.serial is None or not self.serial.is_open:
            if not self.connect_serial():
                return False

        try:
            self.serial.write((command + "\\n").encode("utf-8"))
            self.serial.flush()
            return True
        except (serial.SerialException, OSError) as e:
            self.status_lbl.config(text="USB: Error reconnecting...", fg="orange")
            self.log_msg(f"Serial write failed: {e}")

            if self.connect_serial():
                try:
                    self.serial.write((command + "\\n").encode("utf-8"))
                    self.serial.flush()
                    return True
                except (serial.SerialException, OSError):
                    pass

            return False

    def close_serial(self):
        if self.serial is not None:
            try:
                self.serial.close()
            except Exception:
                pass
        self.serial = None
        self.serial_port = None

    def toggle_auto(self):
        self.auto_mode = not self.auto_mode
        if self.auto_mode:
            self.auto_btn.config(text="Stop Auto BCI", bg="#F44336")
            self.log_msg("Auto BCI Mode: STARTED")
        else:
            self.auto_btn.config(text="Start Auto BCI", bg="#4CAF50")
            self.log_msg("Auto BCI Mode: PAUSED")
            self.send_cmd("stop")

    def draw_waveform(self, csp_features):
        """
        Reconstructs a visual EEG waveform using the 6 CSP features.
        Maps features to mu/beta band frequencies (8-30 Hz)[cite: 2].
        """
        self.ax.clear()
        t = np.linspace(0, 1, 200)
        wave = np.zeros_like(t)
        
        # Frequencies to represent the 6 CSP components
        freqs = [10, 14, 18, 22, 26, 30] 
        
        for i, val in enumerate(csp_features):
            wave += float(val) * np.sin(2 * np.pi * freqs[i] * t)
            
        # Add slight noise
        wave += np.random.normal(0, 0.5, size=wave.shape)
        
        self.ax.plot(t, wave, color="#2196F3", linewidth=1.5)
        self.ax.set_ylim(-10, 10)
        self.ax.axis('off')
        self.ax.set_title("Reconstructed EEG Sensorimotor Rhythm", fontsize=9, color="gray")
        self.canvas.draw()

    def bci_auto_loop(self):
        """Processes the next EEG trial if Auto Mode is ON[cite: 2, 7]."""
        if self.auto_mode and self.model is not None:
            if self.current_trial < len(self.X_test):
                features = self.X_test[self.current_trial]
                features_reshaped = features.reshape(1, -1)
                
                # Predict and calculate confidence
                probs = self.model.predict_proba(features_reshaped)[0]
                prediction_idx = np.argmax(probs)
                confidence = probs[prediction_idx] * 100
                command = self.int_to_class[prediction_idx]
                
                # Ground truth accuracy
                actual_idx = self.y_true[self.current_trial]
                self.total_predictions += 1
                if prediction_idx == actual_idx:
                    self.correct_predictions += 1
                    
                accuracy = (self.correct_predictions / self.total_predictions) * 100
                
                # Update UI
                self.stats_lbl.config(
                    text=f"Accuracy: {accuracy:.1f}% ({self.correct_predictions}/{self.total_predictions}) | Confidence: {confidence:.1f}%"
                )
                self.draw_waveform(features)
                self.send_cmd(command, source="Auto BCI")
                
                self.current_trial += 1
            else:
                self.toggle_auto()
                self.log_msg("Dataset completed. Loop finished.")
        
        # Schedule next iteration (1.5 seconds between BCI commands)
        self.root.after(1500, self.bci_auto_loop)

    def log_msg(self, msg):
        print(f"[GUI] {msg}")

    # ================= UI SETUP =================
    def setup_ui(self):
        self.root.geometry("750x450")
        self.root.configure(bg="#f4f4f9")
        
        # Top Frame
        top_frame = tk.Frame(self.root, bg="#f4f4f9")
        top_frame.pack(fill=tk.X, pady=5)
        tk.Label(top_frame, text="NeuroMove BCI Dashboard", font=("Helvetica", 16, "bold"), bg="#f4f4f9").pack()
        
        self.status_lbl = tk.Label(top_frame, text="USB Serial: Connecting...", fg="orange", bg="#f4f4f9", font=("Helvetica", 10))
        self.status_lbl.pack()

        # Middle Content Frame
        content_frame = tk.Frame(self.root, bg="#f4f4f9")
        content_frame.pack(fill=tk.BOTH, expand=True, padx=10)

        # Left: Controls
        control_frame = tk.Frame(content_frame, bg="#f4f4f9")
        control_frame.pack(side=tk.LEFT, padx=20)
        
        btn_style = {"width": 8, "height": 2, "font": ("Helvetica", 11, "bold"), "fg": "white"}
        
        tk.Button(control_frame, text="▲\nFWD", bg="#4CAF50", command=lambda: self.send_cmd("forward"), **btn_style).grid(row=0, column=1, pady=3)
        tk.Button(control_frame, text="◀\nLEFT", bg="#2196F3", command=lambda: self.send_cmd("left"), **btn_style).grid(row=1, column=0, padx=3)
        tk.Button(control_frame, text="■\nSTOP", bg="#9E9E9E", command=lambda: self.send_cmd("stop"), **btn_style).grid(row=1, column=1)
        tk.Button(control_frame, text="▶\nRIGHT", bg="#2196F3", command=lambda: self.send_cmd("right"), **btn_style).grid(row=1, column=2, padx=3)
        tk.Button(control_frame, text="▼\nBACK", bg="#FF9800", command=lambda: self.send_cmd("backward"), **btn_style).grid(row=2, column=1, pady=3)
        
        self.cmd_lbl = tk.Label(control_frame, text="Sent: STOP (Manual)", font=("Helvetica", 10), bg="#f4f4f9", fg="#333")
        self.cmd_lbl.grid(row=3, column=0, columnspan=3, pady=10)

        # Right: Waveform
        graph_frame = tk.Frame(content_frame, bg="white", bd=2, relief=tk.GROOVE)
        graph_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        self.fig, self.ax = plt.subplots(figsize=(4, 2.5), dpi=100)
        self.fig.patch.set_facecolor('#ffffff')
        self.ax.axis('off')
        self.ax.set_title("Waiting for BCI Data...", fontsize=9, color="gray")
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=graph_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Bottom Frame
        bottom_frame = tk.Frame(self.root, bg="#f4f4f9")
        bottom_frame.pack(fill=tk.X, pady=15, padx=20)
        
        self.stats_lbl = tk.Label(bottom_frame, text="Accuracy: 0.0% (0/0) | Confidence: 0.0%", font=("Helvetica", 12, "bold"), bg="#f4f4f9", fg="#3F51B5")
        self.stats_lbl.pack(side=tk.LEFT)
        
        self.auto_btn = tk.Button(bottom_frame, text="Start Auto BCI", bg="#4CAF50", fg="white", font=("Helvetica", 12, "bold"), width=15, command=self.toggle_auto)
        self.auto_btn.pack(side=tk.RIGHT)

    def bind_keys(self):
        self.root.bind("<Up>",    lambda e: self.send_cmd("forward"))
        self.root.bind("<Down>",  lambda e: self.send_cmd("backward"))
        self.root.bind("<Left>",  lambda e: self.send_cmd("left"))
        self.root.bind("<Right>", lambda e: self.send_cmd("right"))
        self.root.bind("<space>", lambda e: self.send_cmd("stop"))
        self.root.focus_set()

    def quit_app(self):
        self.send_cmd("stop")
        self.close_serial()
        self.root.quit()
        self.root.destroy()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default=SERIAL_PORT, help="ESP32 USB serial port, e.g. /dev/ttyUSB0")
    args = parser.parse_args()

    root = tk.Tk()
    app = HybridBCIGUI(root, args.port)
    root.protocol("WM_DELETE_WINDOW", app.quit_app)
    root.mainloop()

if __name__ == "__main__":
    main()