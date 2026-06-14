# SYNAPSE(Main Pipeline)
# Integrates Eye Tracking, EEG, and Robot Control
# Run this file to start the full system

import cv2
import mediapipe as mp
import numpy as np
import os
import urllib.request
from collections import deque
import threading
import time

from eeg import SimulatedEEG          # Change to SynapseEEG
from robot import SynapseRobot

BaseOptions        = mp.tasks.BaseOptions
FaceLandmarker     = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode  = mp.tasks.vision.RunningMode

model_path = "face_landmarker.task"
if not os.path.exists(model_path):
    print("Downloading face model...")
    urllib.request.urlretrieve(
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
        model_path
    )

# Landmark indices for iris, eye corners, and head pose reference points
LEFT_IRIS    = [468, 469, 470, 471, 472]
RIGHT_IRIS   = [473, 474, 475, 476, 477]
LEFT_EYE     = [33, 133, 160, 159, 158, 144, 145, 153]
RIGHT_EYE    = [362, 263, 387, 386, 385, 373, 374, 380]
LEFT_INNER   = 133
LEFT_OUTER   = 33
RIGHT_INNER  = 362
RIGHT_OUTER  = 263
LEFT_TOP     = 159
LEFT_BOT     = 145
RIGHT_TOP    = 386
RIGHT_BOT    = 374
LEFT_TEMPLE  = 234
RIGHT_TEMPLE = 454
NOSE_TIP     = 4
BLINK_THRESH = 0.18

# Asymmetric thresholds tuned to eyes from calibration
H_THRESH_LEFT  = 0.040
H_THRESH_RIGHT = 0.018
V_THRESH_UP    = 0.035
V_THRESH_DOWN  = 0.050

COLORS = {
    "LEFT":        (255, 80,  0),
    "RIGHT":       (0,   80,  255),
    "UP":          (0,   220, 255),
    "DOWN":        (0,   200, 100),
    "CENTER":      (0,   255, 0),
    "BLINK":       (180, 180, 180),
    "NO FACE":     (0,   0,   255),
    "CALIBRATING": (255, 255, 0),
}

def iris_center(lm, indices, w, h):
    xs = [lm[i].x * w for i in indices]
    ys = [lm[i].y * h for i in indices]
    return np.array([np.mean(xs), np.mean(ys)])

def eye_center(lm, indices, w, h):
    xs = [lm[i].x * w for i in indices]
    ys = [lm[i].y * h for i in indices]
    return np.array([np.mean(xs), np.mean(ys)])

def eye_width(lm, inner, outer, w, h):
    p1 = np.array([lm[inner].x * w, lm[inner].y * h])
    p2 = np.array([lm[outer].x * w, lm[outer].y * h])
    return np.linalg.norm(p1 - p2) + 1e-6

def ear(lm, top, bot, inner, outer, w, h):
    # Eye Aspect Ratio: vertical opening / horizontal width
    v  = abs(lm[top].y - lm[bot].y)
    hz = abs(lm[inner].x - lm[outer].x)
    return v / (hz + 1e-6)

def head_yaw(lm, w, h):
    # Estimate horizontal head rotation using nose position relative to face center
    lt = lm[LEFT_TEMPLE].x  * w
    rt = lm[RIGHT_TEMPLE].x * w
    nx = lm[NOSE_TIP].x     * w
    fc = (lt + rt) / 2
    fw = abs(rt - lt)
    return (nx - fc) / (fw + 1e-6)

def normalized_gaze(lm, w, h):
    # Compute iris offset relative to eye center, normalized by eye width
    # Subtract head yaw contribution to isolate pure eye movement
    l_iris_c = iris_center(lm, LEFT_IRIS,  w, h)
    r_iris_c = iris_center(lm, RIGHT_IRIS, w, h)
    l_eye_c  = eye_center(lm,  LEFT_EYE,   w, h)
    r_eye_c  = eye_center(lm,  RIGHT_EYE,  w, h)
    l_w      = eye_width(lm, LEFT_INNER, LEFT_OUTER,   w, h)
    r_w      = eye_width(lm, RIGHT_INNER, RIGHT_OUTER, w, h)
    l_ox = (l_iris_c[0] - l_eye_c[0]) / l_w
    r_ox = (r_iris_c[0] - r_eye_c[0]) / r_w
    l_oy = (l_iris_c[1] - l_eye_c[1]) / l_w
    r_oy = (r_iris_c[1] - r_eye_c[1]) / r_w
    gx   = (l_ox + r_ox) / 2
    gy   = (l_oy + r_oy) / 2
    yaw  = head_yaw(lm, w, h)
    return gx - yaw * 0.5, gy

def classify_gaze(ox, oy):
    # Horizontal uses asymmetric thresholds; vertical uses symmetric
    if   ox < -H_THRESH_LEFT:  return "LEFT"
    elif ox >  H_THRESH_RIGHT: return "RIGHT"
    elif oy < -V_THRESH_UP:    return "UP"
    elif oy >  V_THRESH_DOWN:  return "DOWN"
    else:                       return "CENTER"


class SYNAPSE:

    def __init__(self, use_robot=False):
        # use_robot=False runs in simulation mode (no hardware needed)
        # use_robot=True connects to the real robot via Bluetooth
        self.use_robot = use_robot

        self.eeg = SimulatedEEG()
        self.eeg.connect()

        self.robot = SynapseRobot()
        if use_robot:
            connected = self.robot.connect()
            if not connected:
                print("Robot not found. Running without robot.")
                self.use_robot = False

        # Eye tracking state
        self.gaze_x_buf    = deque(maxlen=7)
        self.gaze_y_buf    = deque(maxlen=7)
        self.direction_buf = deque(maxlen=5)
        self.last_stable   = "CENTER"

        # Calibration state
        self.calibrated   = False
        self.calib_x      = []
        self.calib_y      = []
        self.calib_count  = 0
        self.CALIB_NEEDED = 60
        self.center_x     = 0.0
        self.center_y     = 0.0

        # Stats
        self.frame_count     = 0
        self.start_time      = time.time()
        self.command_history = deque(maxlen=5)

    def run(self):
        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=VisionRunningMode.IMAGE,
            num_faces=1
        )

        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        print("SYNAPSE starting...")
        print("Look STRAIGHT at camera then hold SPACE to calibrate")
        print("Q = quit | R = recalibrate")

        with FaceLandmarker.create_from_options(options) as landmarker:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                self.frame_count += 1
                h, w     = frame.shape[:2]
                rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB, data=rgb
                )
                results = landmarker.detect(mp_image)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('r'):
                    self._reset_calibration()

                # Get EEG state: brain focus level determines robot speed
                eeg_state = self.eeg.get_state()
                speed     = eeg_state['speed_level']
                brain     = eeg_state['brain_state']
                attention = eeg_state['attention']

                direction = "NO FACE"
                ox = oy = 0.0

                if results.face_landmarks:
                    lm = results.face_landmarks[0]

                    # Blink detection using Eye Aspect Ratio on both eyes
                    l_ear    = ear(lm, LEFT_TOP,  LEFT_BOT,
                                   LEFT_INNER,  LEFT_OUTER,  w, h)
                    r_ear    = ear(lm, RIGHT_TOP, RIGHT_BOT,
                                   RIGHT_INNER, RIGHT_OUTER, w, h)
                    blinking = (l_ear < BLINK_THRESH) or \
                               (r_ear < BLINK_THRESH)

                    # Draw iris markers on frame
                    for iris_idx in [LEFT_IRIS, RIGHT_IRIS]:
                        cx, cy = iris_center(lm, iris_idx, w, h)
                        cv2.circle(frame, (int(cx),int(cy)), 7,
                                   (0,0,255), -1)
                        cv2.circle(frame, (int(cx),int(cy)), 10,
                                   (0,200,255), 2)

                    # Only update gaze buffer when eyes are open
                    if not blinking:
                        gx, gy = normalized_gaze(lm, w, h)
                        self.gaze_x_buf.append(gx)
                        self.gaze_y_buf.append(gy)

                    gaze_x_s = sum(self.gaze_x_buf) / \
                               len(self.gaze_x_buf) \
                               if self.gaze_x_buf else 0
                    gaze_y_s = sum(self.gaze_y_buf) / \
                               len(self.gaze_y_buf) \
                               if self.gaze_y_buf else 0

                    # Calibration: collect gaze samples while user looks straight ahead
                    if not self.calibrated:
                        if key == ord(' ') and not blinking:
                            self.calib_x.append(gaze_x_s)
                            self.calib_y.append(gaze_y_s)
                            self.calib_count += 1
                            if self.calib_count >= self.CALIB_NEEDED:
                                self.center_x = sum(self.calib_x) / \
                                                len(self.calib_x)
                                self.center_y = sum(self.calib_y) / \
                                                len(self.calib_y)
                                self.calibrated = True
                                print("Calibrated!")

                        prog = int((self.calib_count /
                                   self.CALIB_NEEDED) * (w-60))
                        cv2.rectangle(frame,
                                      (30, h-60), (30+prog, h-35),
                                      (0,255,0), -1)
                        cv2.rectangle(frame,
                                      (30, h-60), (w-30, h-35),
                                      (80,80,80), 2)
                        cv2.putText(frame,
                                    "LOOK STRAIGHT + hold SPACE",
                                    (30, 55),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    1.0, (0,255,255), 2)
                        direction = "CALIBRATING"

                    else:
                        if blinking:
                            direction = "BLINK"
                        else:
                            ox  = gaze_x_s - self.center_x
                            oy  = gaze_y_s - self.center_y
                            raw = classify_gaze(ox, oy)
                            self.direction_buf.append(raw)
                            # Direction is stable when 4 of last 5 frames agree
                            if self.direction_buf.count(raw) >= 4:
                                self.last_stable = raw
                            direction = self.last_stable

                        if self.use_robot:
                            self.robot.move(direction, speed)
                        else:
                            # Log simulated command for display
                            self.command_history.append(
                                f"{direction} (speed={speed})"
                            )

                self._draw_ui(frame, w, h, direction,
                              brain, attention, speed,
                              ox, oy)

                cv2.imshow("SYNAPSE", frame)

        cap.release()
        cv2.destroyAllWindows()
        self.eeg.disconnect()
        if self.use_robot:
            self.robot.disconnect()
        print("SYNAPSE stopped.")

    def _draw_ui(self, frame, w, h, direction,
                 brain, attention, speed, ox, oy):

        color = COLORS.get(direction, (255,255,255))

        # Gaze direction — top left
        cv2.putText(frame, f"GAZE: {direction}",
                    (30, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)

        # EEG brain state — top right
        brain_color = (0,255,0) if brain == "FOCUSED" else \
                      (0,165,255) if brain == "NEUTRAL" else \
                      (0,0,255)
        cv2.putText(frame, f"BRAIN: {brain}",
                    (w-350, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, brain_color, 2)

        # Attention bar
        bar_w = int((attention / 100) * 300)
        cv2.rectangle(frame, (w-350, 75), (w-50, 100),
                      (50,50,50), -1)
        cv2.rectangle(frame, (w-350, 75), (w-350+bar_w, 100),
                      brain_color, -1)
        cv2.putText(frame, f"Focus: {attention}%",
                    (w-350, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    brain_color, 2)

        # Speed indicator (driven by EEG attention level)
        speed_text = ["", "SLOW", "MEDIUM", "FAST"][speed]
        speed_color = [(0,0,0),
                       (0,200,255),
                       (0,165,255),
                       (0,255,0)][speed]
        cv2.putText(frame, f"SPEED: {speed_text}",
                    (w-350, 155),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    speed_color, 2)

        # Directional arrow overlay
        cx = w // 2
        cy = h // 2
        if direction == "LEFT":
            cv2.arrowedLine(frame, (cx+60,cy),
                           (cx-60,cy), color, 4, tipLength=0.4)
        elif direction == "RIGHT":
            cv2.arrowedLine(frame, (cx-60,cy),
                           (cx+60,cy), color, 4, tipLength=0.4)
        elif direction == "UP":
            cv2.arrowedLine(frame, (cx,cy+60),
                           (cx,cy-60), color, 4, tipLength=0.4)
        elif direction == "DOWN":
            cv2.arrowedLine(frame, (cx,cy-60),
                           (cx,cy+60), color, 4, tipLength=0.4)
        elif direction == "CENTER":
            cv2.circle(frame, (cx,cy), 20, color, 3)

        # Recent robot command log — bottom left
        cv2.putText(frame, "ROBOT COMMANDS:",
                    (30, h-120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (150,150,150), 1)
        for i, cmd in enumerate(
                list(self.command_history)[-4:]):
            cv2.putText(frame, cmd,
                        (30, h-100 + i*22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (150,150,150), 1)

        # FPS counter
        elapsed = time.time() - self.start_time
        fps = self.frame_count / elapsed if elapsed > 0 else 0
        cv2.putText(frame, f"FPS: {fps:.1f}",
                    (30, h-20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (100,100,100), 1)

        # Debug: raw gaze offsets
        if self.calibrated:
            cv2.putText(frame, f"ox:{ox:.3f} oy:{oy:.3f}",
                        (30, h-40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (100,100,100), 1)

    def _reset_calibration(self):
        self.calibrated  = False
        self.calib_x     = []
        self.calib_y     = []
        self.calib_count = 0
        self.gaze_x_buf.clear()
        self.gaze_y_buf.clear()
        self.direction_buf.clear()
        self.last_stable = "CENTER"
        print("Recalibrating...")


if __name__ == "__main__":
    print("=" * 50)
    print("        SYNAPSE — Brain Computer Interface")
    print("   No hands. No voice. No touch. Just your mind.")
    print("=" * 50)
    print()
    print("Starting without robot (simulation mode)")
    print("To use real robot: change use_robot=True")
    print()

    synapse = SYNAPSE(use_robot=False)
    synapse.run()
