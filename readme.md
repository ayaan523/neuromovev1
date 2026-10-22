# 1. Install Python dependencies
pip install -r requirements.txt
pip install matplotlib   # not in requirements.txt but needed by the GUI

# 2. Make sure Bluetooth service is running
    sudo systemctl status bluetooth
    # if not active:
    sudo systemctl start bluetooth

    # 3. Pair and trust the ESP32 (one-time setup)
    bluetoothctl


    scan on 




    scan off
    pair 58:2A:BD:7D:E8:EE
    trust 58:2A:BD:7D:E8:EE
    exit




# 4. Download the EEG dataset (this can take a while)
python download_dataset.py

# 5. Preprocess EEG and extract CSP features
python eeg_preprocessing.py

# 6. Train the SVM classifier
python train_svm.py

# 7. Launch the controller GUI — this is what actually moves the car
python bci_gui_advanced.py