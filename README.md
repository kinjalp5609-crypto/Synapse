# Synapse - BCI Robot Project
The project is about building a Brain-Computer Interface robot controlled entirely by eye movements and brainwaves. No hands. No voice. No touch.

## What exactly synapse is and the motivation behind it:

So basically we are building a robot that a paralyzed person can control using only their eyes and brain signals. No keyboard. No touchscreen. Nothing physical at all.The idea is simple your brain still works even if your body doesn't. Your eyes still move. So why can't a machine just read that and do what you want?That's what synapse does.

## What it does?

You look left → robot goes left
You look right → robot goes right
You look up → robot goes forward
You blink → robot stops
You focus hard → robot goes fast
You relax → robot slows down

No physical input at all..

## How it works?

So there are three parts working together.
### 1. Eye Tracking
A normal webcam watches your face. Our code finds your iris and tracks where it's moving 26 times every second. That tells the robot which direction to go.
### 2. Brain Signals
A small sensor called BioAmp EXG Pill sits on your forehead. It reads tiny electrical signals your brain produces. When you concentrate hard your brain produces stronger Beta waves. Our code measures that and decides the robot speed.
### 3. Robot Control
An Arduino sits inside the robot. Our laptop sends it one command wirelessly via Bluetooth F for forward, L for left, R for right, S for stop. Arduino reads it and spins the motors.Everything runs together in real time. You wear the sensor, sit in front of the camera, and just think and look. The robot moves.

# Software

## File descriptions:

1. Main.py: It combines all the other files in this repo. Just run this file to make the robot move.

2. Eye_tracking.py: It is an eye tracking module. Detects our gaze direction LEFT / RIGHT / UP / DOWN / CENTER using a webcam. Includes calibration, blink detection, head yaw correction, and smoothing buffer. And it finally outputs the processed gaze direction to the main pipeline.

3. eeg.py: It reads brain signals from electrodes attached to our brain through sensor via arduino uno. It processes the signals to determine user's focus level and outputs a speed level (FAST/ MEDIUM/ SLOW).    

4. robot.py:  Handles Bluetooth communication between the laptop and the robot. Receives direction and speed from the main pipeline and sends movement commands (F / B / L / R / S) to the Arduino via the HC-05 Bluetooth module.

5. Synapse_Robot.ino: Upload to Arduino Uno on the robot. Reads commands from HC-05 and drives motors via L298N motor driver.

6. Requirements.txt: Python dependencies — install with pip install -r Requirements.txt

 ## Tech stack

Python 3.11
MediaPipe: Face landmark detection and iris tracking |
OpenCV: Webcam access and frame display | 
NumPy: Signal processing and FFT computation | 
PySerial: Serial communication with Arduino | 
scikit-learn: Random Forest classifier for EEG command prediction

## How the whole code works?

run main.py to start everything. Eye_tracking.py detects where you're looking, eeg.py reads and processes brain signals, Robot.py sends commands to robot wirelesslys and synapse_robot.ino upload this to Arduino on the robot.


## How to run?

Clone this repo | 
Install requirements |
Run main.py |
Look straight at camera and hold SPACE to calibrate (60 frames) |
Look in any direction robot will follow.
Press R to recalibrate and Q to quit.

# Team

Members and their roles:
Kinjal - Team Lead — software development and hardware testing
Joshika - software development and documentation
Pema - circuit diagram and circuit wiring
Prajna -  Bill of Materials and CAD design

All team members will work together on hardware integration and debugging once we reach campus.
