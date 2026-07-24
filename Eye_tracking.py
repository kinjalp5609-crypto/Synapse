import cv2
import mediapipe as mp
import numpy as np
import os
import urllib.request
from collections import deque

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Download MediaPipe face landmark model if not present
model_path = "face_landmarker.task"
if not os.path.exists(model_path):
    print("Downloading model...")
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

# Asymmetric thresholds derived from calibration:
# RIGHT offset from center is smaller (-0.025) than LEFT (-0.065),
# so H_THRESH_RIGHT must be set lower to reliably detect rightward gaze.
H_THRESH_LEFT  = 0.040
H_THRESH_RIGHT = 0.018
V_THRESH_UP    = 0.035
V_THRESH_DOWN  = 0.030

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
    l_w      = eye_width(lm, LEFT_INNER,  LEFT_OUTER,  w, h)
    r_w      = eye_width(lm, RIGHT_INNER, RIGHT_OUTER, w, h)

    l_ox = (l_iris_c[0] - l_eye_c[0]) / l_w
    r_ox = (r_iris_c[0] - r_eye_c[0]) / r_w
    l_oy = (l_iris_c[1] - l_eye_c[1]) / l_w
    r_oy = (r_iris_c[1] - r_eye_c[1]) / r_w

    gx  = (l_ox + r_ox) / 2
    gy  = (l_oy + r_oy) / 2
    yaw = head_yaw(lm, w, h)
    return gx - yaw * 0.5, gy

def classify_gaze(ox, oy):
    # Horizontal uses asymmetric thresholds; vertical uses symmetric
    if   ox < -H_THRESH_LEFT:   return "LEFT"
    elif ox >  H_THRESH_RIGHT:  return "RIGHT"
    elif oy < -V_THRESH_UP:     return "UP"
    elif oy >  V_THRESH_DOWN:   return "DOWN"
    else:                        return "CENTER"

# Smoothing buffers for gaze and direction stability
gaze_x_buf    = deque(maxlen=7)
gaze_y_buf    = deque(maxlen=7)
direction_buf = deque(maxlen=5)
last_stable   = "CENTER"

calibrated    = False
calib_x       = []
calib_y       = []
calib_count   = 0
CALIB_NEEDED  = 60
center_x      = 0.0
center_y      = 0.0

options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.IMAGE,
    num_faces=1
)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

print("Look STRAIGHT at camera then hold SPACE to calibrate (60 frames)")
print("R = recalibrate | Q = quit")

with FaceLandmarker.create_from_options(options) as landmarker:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w     = frame.shape[:2]
        rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        results  = landmarker.detect(mp_image)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            calibrated  = False
            calib_x     = []; calib_y = []
            calib_count = 0
            gaze_x_buf.clear(); gaze_y_buf.clear()
            direction_buf.clear()
            last_stable = "CENTER"
            print("Recalibrating...")

        direction = "NO FACE"
        ox = 0.0
        oy = 0.0

        if results.face_landmarks:
            lm = results.face_landmarks[0]

            # Blink detection using Eye Aspect Ratio on both eyes
            l_ear    = ear(lm, LEFT_TOP,  LEFT_BOT,  LEFT_INNER,  LEFT_OUTER,  w, h)
            r_ear    = ear(lm, RIGHT_TOP, RIGHT_BOT, RIGHT_INNER, RIGHT_OUTER, w, h)
            blinking = (l_ear < BLINK_THRESH) or (r_ear < BLINK_THRESH)

            # Draw iris markers on frame
            for iris_idx, color in [(LEFT_IRIS, (0,200,255)), (RIGHT_IRIS, (0,200,255))]:
                cx, cy = iris_center(lm, iris_idx, w, h)
                cv2.circle(frame, (int(cx), int(cy)), 7,  (0, 0, 255), -1)
                cv2.circle(frame, (int(cx), int(cy)), 10, (0, 200, 255), 2)

            # Only update gaze buffer when eyes are open
            if not blinking:
                gx, gy = normalized_gaze(lm, w, h)
                gaze_x_buf.append(gx)
                gaze_y_buf.append(gy)

            gaze_x_s = sum(gaze_x_buf)/len(gaze_x_buf) if gaze_x_buf else 0
            gaze_y_s = sum(gaze_y_buf)/len(gaze_y_buf) if gaze_y_buf else 0

            # Calibration: collect gaze samples while user looks straight ahead
            if not calibrated:
                if key == ord(' ') and not blinking:
                    calib_x.append(gaze_x_s)
                    calib_y.append(gaze_y_s)
                    calib_count += 1
                    if calib_count >= CALIB_NEEDED:
                        center_x   = sum(calib_x) / len(calib_x)
                        center_y   = sum(calib_y) / len(calib_y)
                        calibrated = True
                        print(f"Calibrated! cx={center_x:.4f}  cy={center_y:.4f}")

                prog = int((calib_count / CALIB_NEEDED) * (w - 60))
                cv2.rectangle(frame, (30, h-60), (30+prog, h-35), (0,255,0), -1)
                cv2.rectangle(frame, (30, h-60), (w-30,    h-35), (80,80,80), 2)
                cv2.putText(frame, "LOOK STRAIGHT + hold SPACE",
                            (30, 55), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,255), 2)
                cv2.putText(frame, f"Progress: {calib_count}/{CALIB_NEEDED}",
                            (30, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,200,200), 2)
                direction = "CALIBRATING"

            else:
                if blinking:
                    direction = "BLINK"
                else:
                    ox = gaze_x_s - center_x
                    oy = gaze_y_s - center_y
                    raw = classify_gaze(ox, oy)
                    direction_buf.append(raw)
                    # Direction is stable when 4 of last 5 frames agree
                    if direction_buf.count(raw) >= 4:
                        last_stable = raw
                    direction = last_stable

                cv2.putText(frame, f"ox:{ox:.3f}  oy:{oy:.3f}",
                            (30, h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (150,150,150), 1)

        color = COLORS.get(direction, (255,255,255))

        cv2.putText(frame, f"LOOKING: {direction}",
                    (30, 65), cv2.FONT_HERSHEY_SIMPLEX, 1.8, color, 4)

        # Draw directional arrow or center indicator
        cx_screen = w // 2
        cy_screen = h // 2
        if direction == "LEFT":
            cv2.arrowedLine(frame, (cx_screen+80, cy_screen),
                           (cx_screen-80, cy_screen), color, 4, tipLength=0.4)
        elif direction == "RIGHT":
            cv2.arrowedLine(frame, (cx_screen-80, cy_screen),
                           (cx_screen+80, cy_screen), color, 4, tipLength=0.4)
        elif direction == "UP":
            cv2.arrowedLine(frame, (cx_screen, cy_screen+80),
                           (cx_screen, cy_screen-80), color, 4, tipLength=0.4)
        elif direction == "DOWN":
            cv2.arrowedLine(frame, (cx_screen, cy_screen-80),
                           (cx_screen, cy_screen+80), color, 4, tipLength=0.4)
        elif direction == "CENTER":
            cv2.circle(frame, (cx_screen, cy_screen), 20, color, 3)

        cv2.imshow("SYNAPSE - Eye Tracking", frame)

cap.release()
cv2.destroyAllWindows()
