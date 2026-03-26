import pyautogui
import webbrowser
import time


class GestureActions:

    def __init__(self, cooldown=1.0):
        self.cooldown         = cooldown
        self.last_action_time = 0

        pyautogui.FAILSAFE = False
        pyautogui.PAUSE    = 0        # was 0.05 — added 50 ms to every call for no reason;
                                      # gesture cooldown already prevents over-firing

        self._dispatch = {
            "SCROLL":  self._scroll,
            "DRAG":    self._drag,
            "STOP":    lambda s: self.stop_media(),
            "PAUSE":   lambda s: self.pause_media(),
            "CONFIRM": lambda s: self.volume_up(),
            "BACK":    lambda s: self.browser_back(),
            "SELECT":  lambda s: self.open_youtube(),
        }

    def execute(self, gesture, action, side="right"):
        if gesture in ("UNKNOWN", "NONE") or action == "NONE":
            return
        now = time.time()
        if now - self.last_action_time < self.cooldown:
            return
        fn = self._dispatch.get(action)
        if fn:
            fn(side)
            self.last_action_time = now

    def _scroll(self, side):
        amt = -300 if side == "right" else 300
        pyautogui.scroll(amt)
        print(f"Action: Scroll {'Down' if side == 'right' else 'Up'}")

    def _drag(self, side):
        pyautogui.mouseDown()
        time.sleep(0.2)
        pyautogui.mouseUp()
        print("Action: Drag")

    def pause_media(self):
        pyautogui.press("space")
        print("Action: Pause / Play")

    def stop_media(self):
        pyautogui.press("k")
        print("Action: Stop Media")

    def volume_up(self):
        pyautogui.press("volumeup")
        print("Action: Volume Up")

    def browser_back(self):
        pyautogui.hotkey("alt", "left")
        print("Action: Browser Back")

    def open_youtube(self):
        webbrowser.open("https://www.youtube.com")
        print("Action: Open YouTube")