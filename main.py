import cv2
import mediapipe as mp
import numpy as np
from pathlib import Path

MODEL = "hand_landmarker.task"
WBOARD_W, WBOARD_H = 1280, 720
CAM_W, CAM_H = 240, 180

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

tools = [
    {"name": "lapiz",  "color": (30, 30, 30),   "size": 3,   "icon": "\u270E"},
    {"name": "marcador", "color": (30, 30, 30),  "size": 10,  "icon": "\u2B1B"},
    {"name": "resalte", "color": (255, 255, 0),  "size": 25,  "icon": "\u2B1C"},
    {"name": "borrador", "color": (255, 255, 255), "size": 30, "icon": "\u2B55"},
]
palette = [
    (30, 30, 30), (220, 50, 50), (50, 150, 50), (50, 50, 220),
    (220, 180, 50), (150, 50, 150), (50, 180, 180), (255, 255, 255),
]

hand_data = {"lms": None, "fingers": []}
pen_down = False
current_tool = 0
current_color = 0
canvas = np.ones((WBOARD_H, WBOARD_W, 3), dtype=np.uint8) * 255
prev_pos = None
smooth_pos = None

def get_fingers(lms):
    h = []
    h.append(1 if lms[4].x < lms[3].x else 0)
    for tip, pip in [(8, 6), (12, 10), (16, 14), (20, 18)]:
        h.append(1 if lms[tip].y < lms[pip].y else 0)
    return h

def dist(a, b):
    return np.sqrt((a.x - b.x)**2 + (a.y - b.y)**2)

def result_cb(result, image, ts):
    global hand_data
    hand_data["lms"] = None
    hand_data["fingers"] = []
    if result.hand_landmarks:
        lms = result.hand_landmarks[0]
        hand_data["lms"] = lms
        hand_data["fingers"] = get_fingers(lms)

print("Iniciando...")
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=result_cb, num_hands=1,
    min_hand_detection_confidence=0.6
)
landmarker = HandLandmarker.create_from_options(options)

def draw_toolbar(img):
    h, w = img.shape[:2]
    cv2.rectangle(img, (0, 0), (w, 40), (240, 240, 240), -1)
    cv2.rectangle(img, (0, 40), (w, 42), (200, 200, 200), 1)
    for i, t in enumerate(tools):
        x = 10 + i * 70
        color = (100, 100, 255) if i == current_tool else (180, 180, 180)
        cv2.rectangle(img, (x, 5), (x + 60, 35), color, 2 if i == current_tool else 1)
        cv2.putText(img, t["icon"], (x + 18, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (60, 60, 60), 2)
    for i, c in enumerate(palette):
        x = w - 280 + i * 30
        cv2.rectangle(img, (x, 8), (x + 24, 32), c, -1)
        if i == current_color:
            cv2.rectangle(img, (x - 2, 6), (x + 26, 34), (0, 200, 0), 2)

def draw_status(img, text, color=(60, 60, 60)):
    h, w = img.shape[:2]
    cv2.rectangle(img, (0, h - 28), (w, h), (240, 240, 240), -1)
    cv2.putText(img, text, (10, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

ts = 0
print("Pizarra lista! ESC para salir")
print("  Pinza (pulgar+indice) = dibujar/levantar lapiz")
print("  Index finger = cursor")
print("  Senal de paz (2 dedos) = siguiente herramienta")
print("  Mano abierta (5 dedos) = limpiar todo")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    fh, fw = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mpimg = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    ts += 1
    landmarker.detect_async(mpimg, ts)

    display = canvas.copy()
    draw_toolbar(display)

    lms = hand_data["lms"]
    fingers = hand_data["fingers"]
    status = "Sin mano detectada"
    status_color = (150, 150, 150)

    if lms and fingers:
        x = int(lms[8].x * fw)
        y = int(lms[8].y * fh)
        cv2.circle(frame, (x, y), 8, (0, 255, 0), 2)
        cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)

        for i, lm in enumerate(lms):
            cx, cy = int(lm.x * fw), int(lm.y * fh)
            if i in [4, 8, 12, 16, 20]:
                cv2.circle(frame, (cx, cy), 4, (255, 255, 0), -1)

        cx = int(lms[8].x * WBOARD_W)
        cy = int(lms[8].y * WBOARD_H)
        if smooth_pos is None:
            smooth_pos = (cx, cy)
        smooth_pos = (int(smooth_pos[0] * 0.7 + cx * 0.3),
                      int(smooth_pos[1] * 0.7 + cy * 0.3))
        sx, sy = smooth_pos

        pinch = dist(lms[4], lms[8]) < 0.08
        fcount = sum(fingers)

        if pinch and sy > 45 and sy < WBOARD_H - 35:
            if not pen_down:
                pen_down = True
                prev_pos = None
            t = tools[current_tool]
            if t["name"] == "borrador":
                cv2.circle(display, (sx, sy), t["size"], (255, 255, 255), -1)
            else:
                color = palette[current_color] if t["name"] != "resalte" else t["color"]
                if prev_pos:
                    cv2.line(display, prev_pos, (sx, sy), color, t["size"] * 2, cv2.LINE_AA)
                cv2.circle(display, (sx, sy), max(1, t["size"] // 2), color, -1)
            prev_pos = (sx, sy)
            status = f"Dibujando con {t['name']} - {palette[current_color]}"
            status_color = palette[current_color]
        else:
            if pen_down:
                pen_down = False
                prev_pos = None
            t = tools[current_tool]
            if sy > 45 and sy < WBOARD_H - 35:
                clr = palette[current_color]
                cv2.circle(display, (sx, sy), max(4, t["size"] // 2), clr, 2)
                cv2.line(display, (sx - 8, sy), (sx + 8, sy), clr, 2)
                cv2.line(display, (sx, sy - 8), (sx, sy + 8), clr, 2)
            status = f"{fcount} dedos | Pinza para dibujar"
            status_color = (60, 60, 60)

        if sy < 42:
            tool_idx = max(0, min(len(tools) - 1, (sx - 10) // 70))
            if pinch:
                current_tool = tool_idx
        if sy > 5 and sy < 40:
            ci = max(0, min(len(palette) - 1, (sx - (WBOARD_W - 280)) // 30))
            if 0 <= ci < len(palette) and sx >= WBOARD_W - 280 and pinch:
                current_color = ci

        if fcount >= 4 and pinch:
            canvas[:] = 255
            status = "Pizarra limpiada!"
            status_color = (50, 180, 50)

        if fingers == [0, 1, 1, 0, 0] and not pinch:
            current_tool = (current_tool + 1) % len(tools)

    pipe_x = 15
    pipe_y = WBOARD_H - CAM_H - 15
    frame_resized = cv2.resize(frame, (CAM_W, CAM_H))
    display[pipe_y:pipe_y + CAM_H, pipe_x:pipe_x + CAM_W] = frame_resized
    cv2.rectangle(display, (pipe_x, pipe_y), (pipe_x + CAM_W, pipe_y + CAM_H), (100, 100, 100), 2)
    display[pipe_y - 22:pipe_y, pipe_x:pipe_x + CAM_W] = (50, 50, 50)
    cv2.putText(display, "CAMARA", (pipe_x + 5, pipe_y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

    draw_status(display, f"{status} | [ESC] salir", status_color)

    cv2.imshow("Pizarra Inteligente - Hand Camera", display)
    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break
    elif key == ord('c'):
        canvas[:] = 255
    elif key == ord('s'):
        cv2.imwrite("pizarra.png", canvas)
        print("Captura guardada: pizarra.png")

landmarker.close()
cap.release()
cv2.destroyAllWindows()
