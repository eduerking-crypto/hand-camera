import cv2, mediapipe as mp, numpy as np

MODEL = "hand_landmarker.task"
W, H = 1280, 720
CW, CH = 220, 165
MIRROR = True

Base = mp.tasks.BaseOptions
HL = mp.tasks.vision.HandLandmarker
HLO = mp.tasks.vision.HandLandmarkerOptions
VRM = mp.tasks.vision.RunningMode

tools = [
    {"n":"lapiz","c":(30,30,30),"s":2,"ic":"\u270E"},
    {"n":"pluma","c":(30,30,30),"s":4,"ic":"\u2712"},
    {"n":"marcador","c":(30,30,30),"s":10,"ic":"\u2B1B"},
    {"n":"resalte","c":(255,255,0),"s":25,"ic":"\u2B1C"},
    {"n":"borrador","c":(255,255,255),"s":35,"ic":"\u2B55"},
]
palette = [(30,30,30),(200,40,40),(40,140,40),(40,40,200),
           (200,160,40),(160,40,160),(40,160,160),(255,255,255),
           (255,100,50),(100,200,255)]

MODE_NAV, MODE_DRAW, MODE_ERASE = 0, 1, 2
mode = MODE_NAV
tool_idx = 0
color_idx = 0
canvas = np.ones((H, W, 3), dtype=np.uint8) * 255
prev = None
sp = None
hand = {"lms": None, "f": []}
last_gesture = ""
gesture_timer = 0

def gf(lms):
    f = [1 if lms[4].x < lms[3].x else 0]
    for t, p in [(8,6),(12,10),(16,14),(20,18)]:
        f.append(1 if lms[t].y < lms[p].y else 0)
    return f

def d(a, b):
    return ((a.x-b.x)**2 + (a.y-b.y)**2) ** 0.5

def cb(result, img, ts):
    global hand
    hand["lms"] = None
    hand["f"] = []
    if result.hand_landmarks:
        l = result.hand_landmarks[0]
        hand["lms"] = l
        hand["f"] = gf(l)

print("Cargando...")
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

opts = HLO(base_options=Base(model_asset_path=MODEL),
           running_mode=VRM.LIVE_STREAM, result_callback=cb,
           num_hands=1, min_hand_detection_confidence=0.6)
detector = HL.create_from_options(opts)

ts = 0
print("PIZARRA INTELIGENTE")
print("  Pinza (tap)  = dibujar / soltar")
print("  2 dedos      = borrador")
print("  3 dedos      = siguiente color")
print("  4 dedos      = siguiente herramienta")
print("  5 dedos      = limpiar todo")
print("  M            = alternar espejo")
print("  S            = guardar")
print("  U            = deshacer")
print("  ESC          = salir")

undos = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    if MIRROR: frame = cv2.flip(frame, 1)
    fh, fw = frame.shape[:2]
    mpimg = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    ts += 1
    detector.detect_async(mpimg, ts)

    disp = canvas.copy()
    lms = hand["lms"]
    f = hand["f"]

    cv2.rectangle(disp, (0,0), (W,42), (235,235,235), -1)
    cv2.rectangle(disp, (0,42), (W,44), (200,200,200), 1)

    for i, t in enumerate(tools):
        x = 8 + i * 70
        sel = i == tool_idx
        cl = (80,80,200) if sel else (180,180,180)
        cv2.rectangle(disp, (x,4), (x+62,38), cl, 2 if sel else 1)
        cv2.putText(disp, t["ic"], (x+18, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50,50,50), 2)

    for i, c in enumerate(palette):
        x = W - 320 + i * 30
        cv2.rectangle(disp, (x,7), (x+24,35), c, -1)
        if i == color_idx:
            cv2.rectangle(disp, (x-2,5), (x+26,37), (0,200,0), 2)

    mode_names = ["NAVEGAR", "DIBUJAR", "BORRAR"]
    mode_colors = [(100,100,100),(50,150,50),(200,60,60)]
    mx = W // 2 - 60
    cv2.rectangle(disp, (mx,5), (mx+120,36), mode_colors[mode], -1)
    cv2.putText(disp, mode_names[mode], (mx+10,27), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

    if lms and f:
        cx = int(lms[8].x * W)
        cy = int(lms[8].y * H)
        if sp is None: sp = (cx, cy)
        sp = (int(sp[0]*0.65 + cx*0.35), int(sp[1]*0.65 + cy*0.35))
        sx, sy = sp
        pinch = d(lms[4], lms[8]) < 0.065
        fc = sum(f)
        cur_tool = tools[tool_idx]
        cur_color = palette[color_idx]

        px, py = int(lms[8].x*fw), int(lms[8].y*fh)
        for i, lm in enumerate(lms):
            lx, ly = int(lm.x*fw), int(lm.y*fh)
            if i in [4,8,12,16,20]:
                cv2.circle(frame, (lx,ly), 4, (255,255,0), -1)
        cv2.circle(frame, (px,py), 7, (0,255,0), 2)
        cv2.circle(frame, (px,py), 3, (0,255,0), -1)

        if fc == 5:
            if last_gesture != "clear":
                undos.append(canvas.copy())
                if len(undos) > 20: undos.pop(0)
            canvas[:] = 255
            last_gesture = "clear"
            mode = MODE_NAV
            gest_text = "LIMPIAR"
            cv2.putText(disp, gest_text, (W//2-40, H//2), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (50,180,50), 3)
        elif fc == 4:
            if last_gesture != "tool_next":
                tool_idx = (tool_idx + 1) % len(tools)
            last_gesture = "tool_next"
            gest_text = f"HERRAMIENTA: {tools[tool_idx]['n'].upper()}"
            cv2.putText(disp, gest_text, (W//2-130, H//2), cv2.FONT_HERSHEY_SIMPLEX, 1, (100,100,200), 3)
        elif fc == 3:
            if last_gesture != "color_next":
                color_idx = (color_idx + 1) % len(palette)
            last_gesture = "color_next"
            cc = palette[color_idx]
            gest_text = f"COLOR {color_idx+1}"
            cv2.rectangle(disp, (W//2-50, H//2-15), (W//2+50, H//2+15), cc, -1)
            cv2.putText(disp, gest_text, (W//2-40, H//2+5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        elif fc == 2 and not pinch:
            if last_gesture != "erase_mode":
                mode = MODE_ERASE if mode != MODE_ERASE else MODE_NAV
            last_gesture = "erase_mode"
            if mode == MODE_ERASE:
                cv2.putText(disp, "BORRADOR", (W//2-60, H//2), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (200,60,60), 3)
        elif pinch:
            if last_gesture != "draw_toggle":
                mode = MODE_DRAW if mode != MODE_DRAW else MODE_NAV
                prev = None
            last_gesture = "draw_toggle"
            if mode == MODE_DRAW:
                cv2.putText(disp, "DIBUJAR", (W//2-50, H//2), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (50,150,50), 3)
        elif fc == 1:
            last_gesture = "navegar"
            mode = MODE_NAV

        if mode == MODE_DRAW and sy > 46 and sy < H - 30:
            if cur_tool["n"] == "borrador":
                cv2.circle(disp, (sx,sy), cur_tool["s"], (255,255,255), -1)
            else:
                col = cur_color if cur_tool["n"] != "resalte" else (255,255,0)
                if prev:
                    cv2.line(disp, prev, (sx,sy), col, cur_tool["s"]*2, cv2.LINE_AA)
                prev = (sx,sy)
            cv2.circle(disp, (sx,sy), 5, (0,200,0), 2)
        elif mode == MODE_ERASE and sy > 46 and sy < H - 30:
            cv2.circle(disp, (sx,sy), cur_tool["s"], (255,255,255), -1)
            cv2.circle(disp, (sx,sy), 5, (0,0,200), 2)
        else:
            prev = None
            if fc == 1 and sy > 46 and sy < H - 30:
                cv2.circle(disp, (sx,sy), 6, cur_color, 2)
                cv2.line(disp, (sx-10,sy), (sx+10,sy), cur_color, 2)
                cv2.line(disp, (sx,sy-10), (sx,sy+10), cur_color, 2)

        if fc < 5 and last_gesture not in ["clear"]:
            pass

        if f == [0,1,1,0,0] and pinch:
            if last_gesture != "tool_select":
                tool_idx = (tool_idx + 1) % len(tools)
            last_gesture = "tool_select"
    else:
        last_gesture = ""
        mode = MODE_NAV

    px, py = 12, H - CH - 12
    fr = cv2.resize(frame, (CW, CH))
    disp[py:py+CH, px:px+CW] = fr
    cv2.rectangle(disp, (px,py), (px+CW, py+CH), (80,80,80), 2)
    cv2.rectangle(disp, (px,py-20), (px+CW, py), (40,40,40), -1)
    m = "ESPEJO" if MIRROR else "NATURAL"
    cv2.putText(disp, f"CAM {m}", (px+4, py-6), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180,180,180), 1)

    st = f"{mode_names[mode]} | {tools[tool_idx]['n']} | color #{color_idx+1}"
    cv2.rectangle(disp, (0,H-26), (W,H), (235,235,235), -1)
    cv2.putText(disp, st, (8, H-8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (60,60,60), 1)
    cv2.putText(disp, "ESC=salir M=espejo S=guardar U=undo", (W-280, H-8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (120,120,120), 1)

    cv2.imshow("Pizarra Inteligente - Hand Camera", disp)
    key = cv2.waitKey(1) & 0xFF
    if key == 27: break
    elif key == ord('c'): canvas[:] = 255
    elif key == ord('s'):
        cv2.imwrite("pizarra.png", canvas)
        print("Guardado: pizarra.png")
    elif key == ord('u'):
        if undos:
            canvas = undos.pop()
    elif key == ord('m'):
        MIRROR = not MIRROR

detector.close()
cap.release()
cv2.destroyAllWindows()
