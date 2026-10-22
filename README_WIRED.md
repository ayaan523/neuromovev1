BCI Wheelchair — USB/Wired Version
This version uses the ESP32's USB serial connection instead of Bluetooth or Wi-Fi for laptop-to-ESP32 control.
Architecture
EEG dataset / SVM → Python controller → /dev/ttyUSB0 → USB → ESP32 → motor driver → 4 wheels
The ESP32 firmware accepts newline-terminated commands:
forward
backward
left
right
stop
The current ESP32 firmware keeps Wi-Fi and Bluetooth enabled, but the laptop-side software does not use either one.
1. Hardware
Keep the ESP32 connected to the laptop with USB.
Important:
- Keep the HC-SR04 ECHO line reduced from 5 V to 3.3 V before GPIO 18.
- Power the motors from the separate battery supply.
- Keep ESP32 and motor-driver grounds common.
- Test with wheels lifted before putting the robot on the floor.
2. Install Python dependencies
python3 -m pip install -r requirements.txt
python3 -m pip install matplotlib pyserial
3. Confirm the ESP32 port
On Fedora:
ls -l /dev/ttyUSB* /dev/ttyACM* 2>/dev/null
Your current board has been detected as:
/dev/ttyUSB0
Do not leave Arduino IDE Serial Monitor open while a Python program is using /dev/ttyUSB0; only one program should own the serial port at a time.
4. Prepare the ML pipeline
If models are not already generated:
python3 download_dataset.py
python3 eeg_preprocessing.py
python3 train_svm.py
The preprocessing pipeline uses 8–30 Hz filtering and CSP features, and writes the CSP/scaler artifacts used by the controller.
5. Test the wired controller manually
python3 laptop_controller_wired.py --manual --port /dev/ttyUSB0
Then type:
stop
forward
stop
backward
stop
left
stop
right
stop
Keep the wheels lifted for this test.
6. Keyboard control
python3 laptop_controller_wired.py --keyboard --port /dev/ttyUSB0
Controls:
- Up → forward
- Down → backward
- Left → left
- Right → right
- Space → stop
- Q/Esc → quit and stop
7. GUI
python3 bci_gui_wired.py --port /dev/ttyUSB0
The GUI supports:
- Manual buttons
- Arrow keys
- Auto BCI dataset mode
- Accuracy/confidence display
- CSP-feature visualization
8. SVM mode
python3 laptop_controller_wired.py --port /dev/ttyUSB0
This runs the existing SVM pipeline. Important: the supplied controller currently uses simulate_eeg(), so this is a software/dataset demonstration, not live EEG acquisition.
For actual EEG hardware, replace the simulated EEG acquisition with your headset's data stream while keeping the same CSP → scaler → SVM pipeline.
9. Offline dataset drive
python3 bci_dataset_serial.py --port /dev/ttyUSB0
This walks through the preprocessed dataset trials and sends the predicted command over USB.
Safety behavior
The ESP32 firmware provides:
- 3-second command watchdog → STOP
- Forward blocked when HC-SR04 reports less than 15 cm
- No potentiometer autonomous fallback
- Motors initialized OFF
If Python exits or the USB cable is disconnected, the ESP32's watchdog will stop the motors after the command timeout.
