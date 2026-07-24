import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import time

pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0

MODELO = "C:\\Users\\reyes\\OneDrive\\Documents\\viejos\\open code\\IDENTIFICADOR DE MANO Y ROSTROS\\1 IDEA AL AZAR\\hand_landmarker.task"

Base = mp.tasks.BaseOptions
HL = mp.tasks.vision.HandLandmarker
HLO = mp.tasks.vision.HandLandmarkerOptions
VRM = mp.tasks.vision.RunningMode
dib = mp.tasks.vision.drawing_utils
conn = mp.tasks.vision.HandLandmarksConnections

opts = HLO(
    base_options=Base(model_asset_path=MODELO),
    running_mode=VRM.VIDEO, num_hands=2,
    min_hand_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)
detector = HL.create_from_options(opts)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

sw, sh = pyautogui.size()
GESTO_FRAMES = 2

GESTOS = {
    "cursor": (0, 200, 255),
    "click": (0, 255, 100),
    "click_der": (100, 150, 255),
    "scroll": (100, 200, 255),
    "nada": (80, 80, 80),
}

TIPS = [4, 8, 12, 16, 20]
PIPS = [3, 6, 10, 14, 18]


def dist(i, j, lm):
    return np.sqrt((lm[i].x - lm[j].x) ** 2 + (lm[i].y - lm[j].y) ** 2)


def finger_extended(i, lm):
    if i == 0:
        return abs(lm[4].x - lm[17].x) > abs(lm[3].x - lm[17].x)
    return dist(TIPS[i], 0, lm) > dist(PIPS[i], 0, lm) * 1.08


def count_extended(lm):
    return sum(finger_extended(i, lm) for i in range(5))


def pinch_ratio(lm, tip_b=8):
    hs = dist(0, 9, lm)
    pd = dist(4, tip_b, lm)
    return pd / hs if hs > 0 else 1.0


class HandTracker:
    def __init__(self, label, color):
        self.label = label
        self.color = color
        self.gesto = "nada"
        self.frames = 0
        self.smooth_x = 0.0
        self.smooth_y = 0.0
        self.init = False
        self.alpha = 0.4
        self.cooldown = 0.0
        self.dragging = False
        self.scroll_ref_y = None
        self.scroll_dir = 0

    def detectar(self, lm):
        idx = finger_extended(1, lm)
        mid = finger_extended(2, lm)
        ring = finger_extended(3, lm)
        pr = pinch_ratio(lm, 8)
        n = count_extended(lm)

        # Drag = pinza pulgar+indice (mantener y mover)
        if pr < 0.2:
            return "drag"

        # Click derecho = 4 dedos (indice+medio+anular+menique)
        if idx and mid and ring and finger_extended(4, lm):
            return "click_der"

        # Cursor = solo indice
        if idx and not mid:
            return "cursor"

        # Scroll = punio cerrado
        if n <= 1:
            return "scroll"

        return "nada"

    def estabilizar(self, nuevo):
        if nuevo == self.gesto:
            self.frames += 1
        else:
            self.gesto = nuevo
            self.frames = 1

    def mover_cursor(self, lm):
        tx = lm[8].x * sw
        ty = lm[8].y * sh
        if not self.init:
            self.smooth_x = tx
            self.smooth_y = ty
            self.init = True
        else:
            self.smooth_x = self.smooth_x * (1 - self.alpha) + tx * self.alpha
            self.smooth_y = self.smooth_y * (1 - self.alpha) + ty * self.alpha
        cx = int(max(0, min(sw, self.smooth_x)))
        cy = int(max(0, min(sh, self.smooth_y)))
        pyautogui.moveTo(cx, cy, _pause=False)

    def ejecutar(self, lm, ahora):
        if self.gesto == "cursor":
            if self.dragging:
                pyautogui.mouseUp(_pause=False)
                self.dragging = False
            self.mover_cursor(lm)
            self.scroll_ref_y = None

        elif self.gesto == "drag":
            if not self.dragging and self.frames >= GESTO_FRAMES:
                pyautogui.mouseDown(_pause=False)
                self.dragging = True
            if self.dragging:
                self.mover_cursor(lm)
            self.scroll_ref_y = None

        elif self.gesto == "click_der" and self.frames >= GESTO_FRAMES:
            if self.dragging:
                pyautogui.mouseUp(_pause=False)
                self.dragging = False
            if ahora - self.cooldown > 0.6:
                pyautogui.rightClick(_pause=False)
                self.cooldown = ahora

        elif self.gesto == "scroll" and self.frames >= GESTO_FRAMES:
            # Scroll bidireccional: comparar Y del puño contra ref anterior
            y_actual = lm[0].y  # muñeca
            if self.scroll_ref_y is None:
                self.scroll_ref_y = y_actual
            else:
                delta = y_actual - self.scroll_ref_y
                if abs(delta) > 0.02:
                    clicks = int(abs(delta) / 0.02 * 3) + 1
                    if delta > 0:
                        pyautogui.scroll(-clicks, _pause=False)
                    else:
                        pyautogui.scroll(clicks, _pause=False)
                    self.scroll_ref_y = y_actual
            if ahora - self.cooldown > 0.1:
                self.cooldown = ahora


mano_der = HandTracker("DER", (0, 255, 0))
mano_izq = HandTracker("IZQ", (255, 200, 0))

print("Hand-App v5.0")
print("Indice = cursor | 3 dedos = click izq | pinza = click der | punio = scroll bidireccional")
print("Q = salir")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    hf, wf = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    timestamp = int(time.time() * 1000)
    results = detector.detect_for_video(img, timestamp)
    ahora = time.time()

    # ─── Asignar manos por handedness ──
    manos = {}
    if results.hand_landmarks and results.handedness:
        for i, lm in enumerate(results.hand_landmarks):
            nombre = results.handedness[i][0].category_name
            manos[nombre] = lm

    for nombre, tracker in (("Right", mano_der), ("Left", mano_izq)):
        hand = manos.get(nombre)
        if hand is None:
            if tracker.dragging:
                pyautogui.mouseUp(_pause=False)
                tracker.dragging = False
            tracker.gesto = "nada"
            tracker.frames = 0
            tracker.scroll_ref_y = None
            continue

        lm = hand
        dib.draw_landmarks(frame, lm, conn.HAND_CONNECTIONS,
                           dib.DrawingSpec(color=tracker.color, thickness=2, circle_radius=3),
                           dib.DrawingSpec(color=(80, 80, 80), thickness=1))

        nuevo = tracker.detectar(lm)
        tracker.estabilizar(nuevo)
        tracker.ejecutar(lm, ahora)

        # Debug
        pr = pinch_ratio(lm, 8)
        n = count_extended(lm)
        dbg = f"{tracker.label}: {tracker.gesto} f={tracker.frames} n={n} pr={pr:.3f}"
        y_pos = 60 if tracker is mano_der else 85
        cv2.putText(frame, dbg, (10, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, tracker.color, 1)

    cv2.putText(frame, "v5.0", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    if not manos:
        cv2.putText(frame, "Muestra las manos", (wf // 2 - 80, hf // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

    cv2.imshow("Hand-App", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

detector.close()
cap.release()
cv2.destroyAllWindows()
