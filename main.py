import cv2, mediapipe as mp, numpy as np

MODEL = "hand_landmarker.task"
W, H = 1280, 720
CW, CH = 200, 150
MIRROR = True

Base = mp.tasks.BaseOptions
HL = mp.tasks.vision.HandLandmarker
HLO = mp.tasks.vision.HandLandmarkerOptions

colors = [(30,30,30),(200,40,40),(40,140,40),(40,40,200),(200,160,40),(160,40,160)]
canvas = np.ones((H, W, 3), dtype=np.uint8) * 255
prev = None
sp = None
size = 4
ci = 0

opts = HLO(base_options=Base(model_asset_path=MODEL),
           running_mode=mp.tasks.vision.RunningMode.IMAGE,
           num_hands=1, min_hand_detection_confidence=0.5)
det = HL.create_from_options(opts)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("PIZARRA - Pinza para dibujar | ESC=salir | C=limpiar | 1-6=color | +/-=tamano")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    if MIRROR: frame = cv2.flip(frame, 1)
    hf, wf = frame.shape[:2]

    mpimg = mp.Image(image_format=mp.ImageFormat.SRGB,
                     data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    result = det.detect(mpimg)

    disp = canvas.copy()
    col = colors[ci]
    drawing = False
    sx, sy = -100, -100

    for i, c in enumerate(colors):
        x = 10 + i * 46
        cv2.rectangle(disp, (x,8), (x+40,36), c, -1)
        if i == ci:
            cv2.rectangle(disp, (x-2,6), (x+42,38), (0,200,0), 2)

    if result.hand_landmarks:
        lms = result.hand_landmarks[0]
        cx = int(lms[8].x * W)
        cy = int(lms[8].y * H)
        if sp is None: sp = (cx, cy)
        sp = (int(sp[0]*0.55 + cx*0.45), int(sp[1]*0.55 + cy*0.45))
        sx, sy = sp

        t4, t8 = lms[4], lms[8]
        drawing = ((t4.x-t8.x)**2 + (t4.y-t8.y)**2)**0.5 < 0.10

        px, py = int(t8.x*wf), int(t8.y*hf)
        cv2.circle(frame, (px,py), 8, (0,255,0), 2)
        for i in [4,8,12,16,20]:
            lx, ly = int(lms[i].x*wf), int(lms[i].y*hf)
            cv2.circle(frame, (lx,ly), 4, (255,255,0), -1)

        if drawing and sy > 45:
            if prev:
                cv2.line(disp, prev, (sx,sy), col, size, cv2.LINE_AA)
            prev = (sx, sy)
            cv2.circle(disp, (sx,sy), max(2,size//2), col, -1)
        else:
            prev = None

    if not drawing and sy > 45:
        cv2.circle(disp, (sx,sy), 6, col, 2)
        cv2.line(disp, (sx-10,sy), (sx+10,sy), col, 2)
        cv2.line(disp, (sx,sy-10), (sx,sy+10), col, 2)
        if result.hand_landmarks:
            p = ((lms[4].x-lms[8].x)**2 + (lms[4].y-lms[8].y)**2)**0.5
            cv2.putText(frame, f"{p:.2f}", (10,20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)

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
    elif key == ord('=') or key == ord('+'): size = min(40, size + 2)
    elif key == ord('-') or key == ord('_'): size = max(2, size - 2)
    elif key == ord('m'): MIRROR = not MIRROR

det.close()
cap.release()
cv2.destroyAllWindows()
