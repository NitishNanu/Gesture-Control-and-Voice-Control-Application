import cv2
import mediapipe as mp
import pyautogui
from collections import defaultdict
from gesture_module.gesture_classifier import GestureClassifier

# ── CONFIG ────────────────────────────────────────────────────────────────────
GESTURES = [
    "FIST", "OPEN_PALM", "ONE_FINGER", "TWO_FINGERS",
    "THREE_FINGERS", "FOUR_FINGERS", "THUMB_UP", "GUN", "ROCK"
]
CONF_THRESH  = 60
CAM_W, CAM_H = 640, 480
# ─────────────────────────────────────────────────────────────────────────────

conf_matrix   = defaultdict(lambda: defaultdict(int))
current_label = None

mp_hands = mp.solutions.hands
hands    = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.85,
    min_tracking_confidence=0.85
)
mp_draw      = mp.solutions.drawing_utils
clf          = GestureClassifier()
screen_w, screen_h = pyautogui.size()

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAM_W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)
pyautogui.FAILSAFE = False


# ── HELPERS ───────────────────────────────────────────────────────────────────

def get_fingers(hand_landmarks, side="Right"):
    lm      = hand_landmarks.landmark
    fingers = []

    # Thumb
    if side == "Right":
        fingers.append(1 if lm[4].x > lm[2].x else 0)
    else:
        fingers.append(1 if lm[4].x < lm[2].x else 0)

    # Index → Pinky
    for tip, pip in [(8, 6), (12, 10), (16, 14), (20, 18)]:
        fingers.append(1 if lm[tip].y < lm[pip].y else 0)

    return fingers


def get_landmark_vector(hand_landmarks):
    wrist = hand_landmarks.landmark[0]
    vec   = []
    for lm in hand_landmarks.landmark:
        vec.append(round(lm.x - wrist.x, 3))
        vec.append(round(lm.y - wrist.y, 3))
    return vec


def move_cursor(hand_landmarks):
    x = int(hand_landmarks.landmark[8].x * screen_w)
    y = int(hand_landmarks.landmark[8].y * screen_h)
    pyautogui.moveTo(x, y)


def print_confusion_matrix():
    labels = [g for g in GESTURES
              if any(conf_matrix[g].values()) or
                 any(conf_matrix[t][g] for t in GESTURES)]
    if not labels:
        print("No data yet.")
        return

    col_w        = 14
    header_label = 'True \\ Pred'
    header       = f"{header_label:<14}" + "".join(f"{l[:12]:>{col_w}}" for l in labels)
    sep    = "=" * len(header)
    print(f"\n{sep}\nCONFUSION MATRIX\n{sep}")
    print(header)
    print("-" * len(header))
    for true in labels:
        row = f"{true[:13]:<14}"
        for pred in labels:
            row += f"{conf_matrix[true][pred]:>{col_w}}"
        print(row)
    print(sep + "\n")


# ── STARTUP ───────────────────────────────────────────────────────────────────
print("\n=== GESTURE DEBUG MODE ===")
for i, g in enumerate(GESTURES):
    print(f"  {i+1}  →  {g}")
print("  P  →  Print confusion matrix")
print("  R  →  Clear label (keep matrix)")
print("  X  →  Full reset")
print("  ESC→  Quit")
print("\nLabel locks once set. Press R to unlock.\n")


# ── MAIN LOOP ─────────────────────────────────────────────────────────────────
while True:
    ok, frame = cap.read()
    if not ok:
        break

    frame = cv2.flip(frame, 1)
    rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res   = hands.process(rgb)

    fingers = []
    lm_vec  = []

    if res.multi_hand_landmarks:
        hl   = res.multi_hand_landmarks[0]
        hand = res.multi_handedness[0].classification[0].label
        side = "Left" if hand == "Right" else "Right"   # mirror-corrected

        mp_draw.draw_landmarks(frame, hl, mp_hands.HAND_CONNECTIONS)

        fingers = get_fingers(hl, side)
        lm_vec  = get_landmark_vector(hl)

        gesture, confidence = clf.classify(fingers)

        if fingers == [0, 1, 0, 0, 0]:
            move_cursor(hl)

        if current_label and confidence >= CONF_THRESH:
            conf_matrix[current_label][gesture] += 1

        # HUD
        label_text  = f"Label: {current_label} [LOCKED]" if current_label \
                      else "Label: NONE  (press 1-9)"
        label_color = (0, 255, 0) if current_label else (0, 100, 255)

        cv2.putText(frame, f"Fingers: {fingers}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(frame, f"Gesture: {gesture} ({confidence}%)",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0),   2)
        cv2.putText(frame, label_text,
                    (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, label_color,   2)

        print(f"\r[{gesture:<14}] fingers={fingers}  vec[0:6]={lm_vec[:6]}", end="")

    else:
        cv2.putText(frame, "No hand detected",
                    (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow("Gesture Debug", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == 27:
        break
    elif key in (ord('p'), ord('P')):
        print_confusion_matrix()
    elif key in (ord('r'), ord('R')):
        current_label = None
        print("\n[Label cleared]\n")
    elif key in (ord('x'), ord('X')):
        conf_matrix.clear()
        current_label = None
        print("\n[Full reset]\n")
    elif chr(key) in [str(i) for i in range(1, len(GESTURES) + 1)]:
        current_label = GESTURES[int(chr(key)) - 1]
        print(f"\n[LOCKED → {current_label}]\n")

cap.release()
cv2.destroyAllWindows()
print_confusion_matrix()