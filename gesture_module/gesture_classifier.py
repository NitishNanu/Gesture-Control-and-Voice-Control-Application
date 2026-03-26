class GestureClassifier:

    def __init__(self):

        self.gesture_map = {
            (0, 0, 0, 0, 0): "FIST",
            (1, 1, 1, 1, 1): "OPEN_PALM",
            (0, 1, 0, 0, 0): "ONE_FINGER",
            (0, 1, 1, 0, 0): "TWO_FINGERS",
            (0, 1, 1, 1, 0): "THREE_FINGERS",
            (0, 1, 1, 1, 1): "FOUR_FINGERS",
            (1, 0, 0, 0, 0): "THUMB_UP",
            (1, 1, 0, 0, 0): "GUN",
            (0, 1, 0, 0, 1): "ROCK",
        }

        self.action_map = {
            "ONE_FINGER":    "NONE",
            "TWO_FINGERS":   "SCROLL",
            "THREE_FINGERS": "NONE",
            "FOUR_FINGERS":  "DRAG",
            "FIST":          "STOP",
            "OPEN_PALM":     "PAUSE",
            "THUMB_UP":      "CONFIRM",
            "GUN":           "SELECT",
            "ROCK":          "BACK",
        }

    def classify(self, fingers):
        if not fingers or len(fingers) != 5:
            return "UNKNOWN", 0
        gesture    = self.gesture_map.get(tuple(fingers), "UNKNOWN")
        confidence = 100 if gesture != "UNKNOWN" else 0
        return gesture, confidence

    def get_action(self, fingers):
        gesture, confidence = self.classify(fingers)
        action = self.action_map.get(gesture, "NONE")
        return gesture, action, confidence