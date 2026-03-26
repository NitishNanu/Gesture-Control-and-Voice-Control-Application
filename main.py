import cv2
import time
import threading
import pyautogui
from collections import Counter, deque

from gesture_module.hand_detector      import HandDetector
from gesture_module.gesture_classifier import GestureClassifier
from gesture_module.gesture_actions    import GestureActions


# ── CONFIG ────────────────────────────────────────────────────────────────────
CONF_THRESH    = 60
HISTORY_SIZE   = 5          # majority-vote window
SWIPE_THRESHOLD = 0.18      # normalized x units
SWIPE_FRAMES   = 8
CURSOR_SMOOTH  = 0.4        # lowered: 0.5 felt sluggish
FLICK_DIP      = 0.04       # normalized y units
FLICK_FRAMES   = 8
CLICK_COOLDOWN = 0.6        # seconds
WARMUP_FRAMES  = 20

# Active zone for cursor (crop inner 60 % of frame so edges are reachable)
ZONE_X1, ZONE_X2 = 0.20, 0.80
ZONE_Y1, ZONE_Y2 = 0.15, 0.85

# Gestures that should NOT trigger a swipe (hand moves laterally by design)
SWIPE_BLACKLIST = {"ONE_FINGER", "TWO_FINGERS", "FIST"}
# ─────────────────────────────────────────────────────────────────────────────


# ── MODULES ───────────────────────────────────────────────────────────────────
detector   = HandDetector(detectionCon=0.85, trackCon=0.85)
classifier = GestureClassifier()
actions    = GestureActions()

screen_w, screen_h = pyautogui.size()
pyautogui.FAILSAFE = False

# ── THREADED CAPTURE ──────────────────────────────────────────────────────────
# Runs camera read in a background thread so the main loop is never blocked
# waiting for the next frame from the OS camera driver.
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  960)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)

_frame_lock  = threading.Lock()
_latest_frame = None
_cam_running  = True

def _capture_thread():
    global _latest_frame, _cam_running
    while _cam_running:
        ok, frame = cap.read()
        if ok:
            with _frame_lock:
                _latest_frame = frame

_t = threading.Thread(target=_capture_thread, daemon=True)
_t.start()
# ─────────────────────────────────────────────────────────────────────────────


# ── STATE ─────────────────────────────────────────────────────────────────────
# deque(maxlen) gives O(1) append and auto-drops old items — replaces pop(0)
gesture_history = deque(maxlen=HISTORY_SIZE)
swipe_x_history = deque(maxlen=SWIPE_FRAMES)
index_y_buf     = deque(maxlen=FLICK_FRAMES)

prev_sx, prev_sy    = screen_w // 2, screen_h // 2
last_click_time     = 0
flick_triggered     = False
frame_count         = 0
pTime               = 0

current_gesture    = "NONE"
current_action     = "NONE"
current_confidence = 0
# ─────────────────────────────────────────────────────────────────────────────


# ── HELPERS ───────────────────────────────────────────────────────────────────

def stable_gesture(gesture):
    gesture_history.append(gesture)
    return Counter(gesture_history).most_common(1)[0][0]


def detect_swipe(normList, current_gest):
    if current_gest in SWIPE_BLACKLIST:
        swipe_x_history.clear()
        return None
    wx = normList[0][1]
    swipe_x_history.append(wx)
    if len(swipe_x_history) < SWIPE_FRAMES:
        return None
    movement = swipe_x_history[-1] - swipe_x_history[0]
    if movement > SWIPE_THRESHOLD:
        swipe_x_history.clear()
        return "RIGHT"
    if movement < -SWIPE_THRESHOLD:
        swipe_x_history.clear()
        return "LEFT"
    return None


def move_cursor(normList):
    global prev_sx, prev_sy
    nx = normList[8][1]
    ny = normList[8][2]
    # Remap from active zone to 0-1
    nx = (nx - ZONE_X1) / (ZONE_X2 - ZONE_X1)
    ny = (ny - ZONE_Y1) / (ZONE_Y2 - ZONE_Y1)
    nx = max(0.0, min(1.0, nx))
    ny = max(0.0, min(1.0, ny))
    raw_x = int(nx * screen_w)
    raw_y = int(ny * screen_h)
    alpha = 1 - CURSOR_SMOOTH
    sx = int(prev_sx + (raw_x - prev_sx) * alpha)
    sy = int(prev_sy + (raw_y - prev_sy) * alpha)
    prev_sx, prev_sy = sx, sy
    pyautogui.moveTo(sx, sy)


def detect_flick_click(normList, fingers):
    global last_click_time, flick_triggered
    if not fingers or fingers[1] != 1:
        index_y_buf.clear()
        flick_triggered = False
        return
    tip_y = normList[8][2]
    index_y_buf.append(tip_y)
    if len(index_y_buf) < FLICK_FRAMES:
        return
    dip = max(index_y_buf) - min(index_y_buf)
    now = time.time()
    if dip > FLICK_DIP and not flick_triggered:
        if now - last_click_time > CLICK_COOLDOWN:
            pyautogui.click()
            last_click_time = now
            flick_triggered = True
            print("Action: Flick Click")
    elif dip < FLICK_DIP * 0.4:
        flick_triggered = False


def draw_ui(img, fingers, swipe_delta, fps):
    cv2.putText(img, f"FPS: {int(fps)}",
                (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    cv2.rectangle(img, (10, 50), (440, 200), (0, 0, 0), -1)
    cv2.putText(img, f"Gesture : {current_gesture}",
                (18, 85),  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(img, f"Action  : {current_action}",
                (18, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0),   2)
    cv2.putText(img, f"Fingers : {fingers}",
                (18, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 2)
    cv2.putText(img, f"Swipe   : {swipe_delta}",
                (18, 172), cv2.FONT_HERSHEY_SIMPLEX, 0.55,(180, 180, 180), 1)
    lines = [
        "INDEX UP    : Cursor    FLICK : Click",
        "TWO_FINGERS : Scroll    FIST  : Stop",
        "OPEN_PALM   : Pause     THUMB : Vol+",
        "GUN         : YouTube   ROCK  : Back",
        "SWIPE L/R   : Browser navigation",
    ]
    y = 215
    for l in lines:
        cv2.putText(img, l, (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1)
        y += 18


# ── WINDOW SETUP (once, outside loop) ────────────────────────────────────────
cv2.namedWindow("Gesture Control", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
cv2.resizeWindow("Gesture Control", 960, 720)
cv2.setWindowProperty("Gesture Control", cv2.WND_PROP_TOPMOST, 1)  # set once only


# ── MAIN LOOP ─────────────────────────────────────────────────────────────────
while True:
    with _frame_lock:
        img = _latest_frame
    if img is None:
        continue  # thread hasn't delivered a frame yet

    img = img.copy()  # don't mutate the shared buffer
    frame_count += 1
    img = cv2.flip(img, 1)
    img = detector.findHands(img, draw=True)

    hand    = detector.findHand(img)
    fingers = []
    swipe_delta = "—"

    if hand:
        fingers  = detector.fingerUp(hand["lmList"], hand["side"])
        normList = hand["normList"]

        # ── Classify first so swipe/flick can use the stable gesture ─────
        gesture, action, confidence = classifier.get_action(fingers)
        if confidence >= CONF_THRESH:
            gesture = stable_gesture(gesture)
            action  = classifier.action_map.get(gesture, "NONE")
        else:
            gesture = stable_gesture("UNKNOWN")

        # ── Cursor move ───────────────────────────────────────────────────
        if fingers == [0, 1, 0, 0, 0]:
            move_cursor(normList)

        # ── Flick click — gated on index finger being up ──────────────────
        detect_flick_click(normList, fingers)

        # ── Swipe — gated on gesture blacklist ────────────────────────────
        if frame_count > WARMUP_FRAMES:
            swipe = detect_swipe(normList, gesture)
            if swipe == "RIGHT":
                pyautogui.hotkey("alt", "right")
                print("Action: Swipe Right → Forward")
            elif swipe == "LEFT":
                pyautogui.hotkey("alt", "left")
                print("Action: Swipe Left → Back")

        if len(swipe_x_history) == SWIPE_FRAMES:
            delta = swipe_x_history[-1] - swipe_x_history[0]
            swipe_delta = f"{delta:+.3f} (need ±{SWIPE_THRESHOLD})"

        # ── Execute action ────────────────────────────────────────────────
        if (confidence >= CONF_THRESH
                and gesture not in ("ONE_FINGER", "UNKNOWN", "NONE")
                and frame_count > WARMUP_FRAMES):
            actions.execute(gesture, action, side=hand["side"].lower())

        current_gesture    = gesture
        current_action     = action
        current_confidence = confidence

    else:
        index_y_buf.clear()
        swipe_x_history.clear()
        gesture_history.clear()
        flick_triggered = False

    cTime = time.time()
    fps   = 1 / (cTime - pTime) if pTime else 0
    pTime = cTime

    draw_ui(img, fingers, swipe_delta, fps)
    cv2.imshow("Gesture Control", img)

    if cv2.waitKey(1) & 0xFF == 27:
        break

_cam_running = False
_t.join(timeout=1)
cap.release()
cv2.destroyAllWindows()