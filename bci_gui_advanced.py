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
import socket
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
ESP32_BT_MAC = "58:2A:BD:7D:E8:EE"
PORT = 1

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
MODEL_PATH = os.path.join(MODELS_DIR, "svm_model.pkl")
FEATURES_PATH = os.path.join(MODELS_DIR, "features.npz")
MAPPING_PATH = os.path.join(MODELS_DIR, "class_mapping.pkl")


class HybridBCIGUI:
    def __init__(self, root, mac):
        self.root = root
        self.root.title("NeuroMove Advanced Controller")
        self.mac = mac
        
        # State variables
        self.sock = None
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
        self.connect_bt()
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

    def connect_bt(self):
        if self.sock:
            try:
                self.sock.close()
            except:
                pass
        try:
            self.sock = socket.socket(socket.AF_BLUETOOTH, socket.SOCK_STREAM, socket.BTPROTO_RFCOMM)
            self.sock.settimeout(1.0)
            self.sock.connect((self.mac, PORT))
            self.status_lbl.config(text="Bluetooth: Connected", fg="green")
            return True
        except Exception as e:
            self.status_lbl.config(text="Bluetooth: Disconnected", fg="red")
            self.sock = None
            return False

    def send_cmd(self, command, source="Manual"):
        """Sends command to ESP32."""
        self.cmd_lbl.config(text=f"Sent: {command.upper()} ({source})")
        if self.sock is None:
            if not self.connect_bt():
                return
        try:
            self.sock.sendall((command + "\n").encode('utf-8'))
        except Exception:
            self.status_lbl.config(text="Bluetooth: Error Reconnecting...", fg="orange")
            if self.connect_bt():
                try:
                    self.sock.sendall((command + "\n").encode('utf-8'))
                except:
                    pass

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
        
        self.status_lbl = tk.Label(top_frame, text="Bluetooth: Connecting...", fg="orange", bg="#f4f4f9", font=("Helvetica", 10))
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
        if self.sock:
            try: self.sock.close()
            except: pass
        self.root.quit()
        self.root.destroy()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mac", default=ESP32_BT_MAC, help="ESP32 Bluetooth MAC address")
    args = parser.parse_args()

    root = tk.Tk()
    app = HybridBCIGUI(root, args.mac)
    root.protocol("WM_DELETE_WINDOW", app.quit_app)
    root.mainloop()

if __name__ == "__main__":
    main()