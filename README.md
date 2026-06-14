# Synapse
Brain-Computer Interface robot controlled by eye movements and brainwaves. No hands. No voice. No touch.

What exactly synapse is?

So basically we are building a robot that a paralyzed person can control using only their eyes and brain signals. No keyboard. No touchscreen. Nothing physical at all.The idea is simple your brain still works even if your body doesn't. Your eyes still move. So why can't a machine just read that and do what you want?That's what synapse does.

You look left → robot goes left
You look right → robot goes right
You look up → robot goes forward
You blink → robot stops
You focus hard → robot goes fast
You relax → robot slows down

How it works?

So there are three parts working together.
1. Eye Tracking
A normal webcam watches your face. Our code finds your iris and tracks where it's moving 26 times every second. That tells the robot which direction to go.
2. Brain Signals
A small sensor called BioAmp EXG Pill sits on your forehead. It reads tiny electrical signals your brain produces. When you concentrate hard your brain produces stronger Beta waves. Our code measures that and decides the robot speed.
3. Robot Control
An Arduino sits inside the robot. Our laptop sends it one command wirelessly via Bluetooth F for forward, L for left, R for right, S for stop. Arduino reads it and spins the motors.Everything runs together in real time. You wear the sensor, sit in front of the camera, and just think and look. The robot moves.

Software

Python 3.11
MediaPipe (eye tracking)
OpenCV (camera)
NumPy + FFT (brain signal processing)
PySerial (Bluetooth communication)

Exactly what's happening here?

main.py run this is to start everything.Eye_tracking.py detects where you're looking.eeg.py reads and processes brain signals.Robot.py sends commands to robot wirelessly.synapse_robot.ino upload this to Arduino on the robot.


How to run?

Clone this repo
Install requirements
Run main.py
Look straight at camera and hold SPACE to calibrate (60 frames)
Look in any direction robot will follow.
