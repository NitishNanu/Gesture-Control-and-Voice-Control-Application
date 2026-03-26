# Gesture Detection (Gesture-only Mode)

This repository contains a gesture recognition pipeline built using MediaPipe and OpenCV, along with action mapping via `pyautogui`. This README describes the gesture-only portion that can be used independently from the voice module.

## ✅ What it provides

- Real-time hand detection using `gesture_module/hand_detector.py`
- Gesture classification (`FIST`, `OPEN_PALM`, `ONE_FINGER`, etc.) using `gesture_module/gesture_classifier.py`
- Action mapping via gesture-to-keypress and system control in `gesture_module/gesture_actions.py`
- Interactive debug tool in `gesture_debug.py` for live testing and labeling

---

## 📦 Requirements

- Python 3.8+
- OpenCV: `opencv-python`
- MediaPipe: `mediapipe`
- PyAutoGUI: `pyautogui`

Optionally:
- `wheel` (for pyautogui installation smoothness), etc.

### Install dependencies

```bash
python -m pip install -r requirements.txt
```

If there is no `requirements.txt`, run:

```bash
python -m pip install opencv-python mediapipe pyautogui
```

---

## ▶️ Gesture-only Quick Start

1. Connect a camera (built-in webcam is OK).
2. Run the debug script:

```bash
python gesture_debug.py
```

3. In the window, make gestures and observe console output:
   - `G` prints confusion matrix
   - `R` clears label lock
   - `X` resets matrix and label
   - `ESC` exits

4. Optional: lock a label by pressing keys `1`..`9` to evaluate classifier behavior.

---

## 🧠 Gesture Classifier

`gesture_module/gesture_classifier.py`:

- `GestureClassifier.classify(fingers)`
  - `fingers`: list of five {0,1} (thumb..pinky)
  - returns `(gesture, confidence)`
- `GestureClassifier.get_action(fingers)`
  - returns `(gesture, action, confidence)`
  - action map:
    - `TWO_FINGERS` -> `SCROLL`
    - `FOUR_FINGERS` -> `DRAG`
    - `FIST` -> `STOP`
    - `OPEN_PALM` -> `PAUSE`
    - `THUMB_UP` -> `CONFIRM`
    - `GUN` -> `SELECT`
    - `ROCK` -> `BACK`

---

## 🖐 Hand Detection Utils

`gesture_module/hand_detector.py`:

- `HandDetector.findHands(img, draw=True)`
- `HandDetector.findHand(img)` returns landmarks, bounding box, side
- `HandDetector.fingerUp(lmList, side)` returns 5-finger state

This uses MediaPipe in camera pipeline and normalizes coordinates.

---

## 🎮 Gesture Actions

`gesture_module/gesture_actions.py` maps actions to OS inputs:

- `SCROLL` (wheel scroll)
- `DRAG` (click/drag)
- `STOP` (pause/stop media)
- `PAUSE` (space)
- `CONFIRM` (volume up)
- `BACK` (browser back)
- `SELECT` (open YouTube)

Use it as:

```python
from gesture_module.gesture_actions import GestureActions
act = GestureActions(cooldown=1.0)
act.execute('FOUR_FINGERS', 'DRAG', side='right')
```

---

## 🔧 To integrate into your own app

1. Capture frame with OpenCV.
2. Detect with `HandDetector.findHands()`.
3. Extract finger flags via `HandDetector.fingerUp()`.
4. Classify with `GestureClassifier.get_action()`.
5. Execute mapped action with `GestureActions.execute()`.

---

## 📝 Notes

- `gesture_debug.py` has a custom in-loop `get_fingers()` plus state tracking and confusion matrix output.
- Code assumes right/left “mirror-corrected” side mapping (flipped camera frame).

---

## 📌 License

Use as needed and adapt to your requirements.
>>>>>>> d7e694a (Initial commit: gesture detection project + gitignore)
