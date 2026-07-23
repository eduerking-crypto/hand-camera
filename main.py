import cv2, mediapipe as mp, numpy as np

MODEL = "hand_landmarker.task"
W, H = 1280, 720
CW, CH = 200, 150
MIRROR = True

Base = mp.tasks.BaseOptions
HL = mp.tasks.vision.HandLandmarker
HLO = mp.tasks.vision.HandLandmarkerOptions
VRM = mp.tasks.vision.RunningMode

colors = [(30,30,30),(200,40,40),(40,140,40),(40,40,200),(200,160,40),(160,40,160)]
palette_w = 40

canvas = np.ones((H, W, 3), dtype=np.uint8) * 255
prev = None
sp = None
hand = {"lms": None}
size = 4
ci = 0

def d(a, b):
    return ((a.x-b.x)**2 + (a.y-b.y)**2)**0.5

def cb(result, img, ts):
    global hand
    hand["lms"] = None
    if result.hand_landmarks:
        hand["lms"] = result.hand_landmarks[0]

print("Cargando...")
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

opts = HLO(base_options=Base(model_asset_path=MODEL),
           running_mode=VRM.LIVE_STREAM, result_callback=cb,
           num_hands=1, min_hand_detection_confidence=0.6)
det = HL.create_from_options(opts)

print("Pinza (pulgar+indice) = dibujar | ESC=salir | C=limpiar | 1-6=color | +/-=tamano")

ts = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    if MIRROR: frame = cv2.flip(frame, 1)
    mpimg = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    ts += 1
    det.detect_async(mpimg, ts)

    disp = canvas.copy()
    lms = hand["lms"]

    for i, c in enumerate(colors):
        x = 10 + i * (palette_w + 6)
        cv2.rectangle(disp, (x,8), (x+palette_w,38), c, -1)
        if i == ci:
            cv2.rectangle(disp, (x-2,6), (x+palette_w+2,40), (0,200,0), 2)

    if lms:
        cx = int(lms[8].x * W)
        cy = int(lms[8].y * H)
        if sp is None: sp = (cx, cy)
        sp = (int(sp[0]*0.6 + cx*0.4), int(sp[1]*0.6 + cy*0.4))
        sx, sy = sp
        pinch = d(lms[4], lms[8]) < 0.07
        color = colors[ci]

        px, py = int(lms[8].x * frame.shape[1]), int(lms[8].y * frame.shape[0])
        cv2.circle(frame, (px,py), 6, (0,255,0), 2)

        if pinch and sy > 45:
            if prev:
                cv2.line(disp, prev, (sx,sy), color, size, cv2.LINE_AA)
            prev = (sx, sy)
            cv2.circle(disp, (sx,sy), size//2, color, -1)
        else:
            prev = None
            cv2.circle(disp, (sx,sy), 5, color, 2)
            cv2.line(disp, (sx-8,sy), (sx+8,sy), color, 2)
            cv2.line(disp, (sx,sy-8), (sx,sy+8), color, 2)

    px, py = 10, H - CH - 10
    fr = cv2.resize(frame, (CW, CH))
    disp[py:py+CH, px:px+CW] = fr
    cv2.rectangle(disp, (px,py), (px+CW,py+CH), (80,80,80), 2)

    cv2.imshow("Pizarra", disp)
    key = cv2.waitKey(1) & 0xFF
    if key == 27: break
    elif key == ord('c'): canvas[:] = 255; prev = None
    elif key == ord('1'): ci = 0
    elif key == ord('2'): ci = 1
    elif key == ord('3'): ci = 2
    elif key == ord('4'): ci = 3
    elif key == ord('5'): ci = 4
    elif key == ord('6'): ci = 5
    elif key == ord('=') or key == ord('+'): size = min(30, size + 2)
    elif key == ord('-') or key == ord('_'): size = max(2, size - 2)
    elif key == ord('m'): MIRROR = not MIRROR

det.close()
cap.release()
cv2.destroyAllWindows()
