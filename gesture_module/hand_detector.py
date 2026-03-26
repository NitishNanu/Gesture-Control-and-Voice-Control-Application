import cv2
import mediapipe as mp


class HandDetector:

    def __init__(self, detectionCon=0.85, trackCon=0.85):
        self.mpHands = mp.solutions.hands
        self.hands   = self.mpHands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=detectionCon,
            min_tracking_confidence=trackCon,
            model_complexity=0,   
        )
        self.mpDraw  = mp.solutions.drawing_utils
        self.results = None

        # Cache draw spec objects — creating them every frame allocates memory
        self._lm_spec   = self.mpDraw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3)
        self._conn_spec = self.mpDraw.DrawingSpec(color=(255, 255, 255), thickness=1)


    def findHands(self, img, draw=True):
        rgb          = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False          
        self.results = self.hands.process(rgb)
        rgb.flags.writeable = True
        if self.results.multi_hand_landmarks and draw:
            for hlm in self.results.multi_hand_landmarks:
                self.mpDraw.draw_landmarks(
                    img, hlm, self.mpHands.HAND_CONNECTIONS,
                    self._lm_spec, self._conn_spec
                )
        return img


    def findHand(self, img):
        if not self.results or not self.results.multi_hand_landmarks:
            return None

        h, w     = img.shape[:2]
        hlm      = self.results.multi_hand_landmarks[0]
        lmList   = []
        normList = []
        xs, ys   = [], []

        for idx, lm in enumerate(hlm.landmark):
            cx, cy = int(lm.x * w), int(lm.y * h)
            lmList.append([idx, cx, cy])
            normList.append([idx, lm.x, lm.y])
            xs.append(cx)
            ys.append(cy)

        bbox      = (min(xs), min(ys), max(xs), max(ys))
        raw_label = self.results.multi_handedness[0].classification[0].label
        side      = "Left" if raw_label == "Right" else "Right"

        return {"lmList": lmList, "normList": normList, "bbox": bbox, "side": side}


    def fingerUp(self, lmList, side="Right"):
        if not lmList:
            return []

        fingers     = []
        palm_height = abs(lmList[0][2] - lmList[9][2])

        # Guard against degenerate palm_height (hand nearly horizontal)
        if palm_height < 10:
            return [0, 0, 0, 0, 0]

        # Thumb — horizontal check with adaptive gap
        thumb_gap = int(palm_height * 0.2)
        if side == "Right":
            fingers.append(1 if (lmList[4][1] - lmList[2][1]) > thumb_gap else 0)
        else:
            fingers.append(1 if (lmList[2][1] - lmList[4][1]) > thumb_gap else 0)

        # Index → Pinky — vertical check with per-finger margin
        margins = {
            8:  int(palm_height * 0.06),
            12: int(palm_height * 0.08),
            16: int(palm_height * 0.12),
            20: int(palm_height * 0.14),
        }
        for tip, pip in [(8, 6), (12, 10), (16, 14), (20, 18)]:
            fingers.append(1 if (lmList[pip][2] - lmList[tip][2]) > margins[tip] else 0)

        return fingers