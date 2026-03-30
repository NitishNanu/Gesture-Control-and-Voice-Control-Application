import json
import os
import platform
import queue
import re
import shutil
import subprocess
import threading
import audioop
import webbrowser
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional
import getpass

import sounddevice as sd
from vosk import KaldiRecognizer, Model

# English-only configuration for faster performance
MODEL_PATH = "vosk-model-small-en-us-0.15"
PROFILE_PATH = Path.home() / ".gvcsda_user_profile.json"

SAMPLE_RATE = 16000
SILENCE_THRESHOLD = 500
SILENCE_CHUNKS = 15
MAX_BUFFER_CHUNKS = 50

WAKE_WORDS = ["hey buddy"]


class UserProfile:
    """Manage persistent user information and preferences."""

    def __init__(self):
        self.system_username = getpass.getuser()
        self.custom_name: Optional[str] = None
        self.preferences: dict = {}
        self.load_profile()

    def load_profile(self) -> None:
        """Load the profile from a stable user location."""
        if not PROFILE_PATH.exists():
            return

        try:
            with PROFILE_PATH.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
                self.custom_name = data.get("name")
                self.preferences = data.get("preferences", {})
        except (OSError, json.JSONDecodeError) as error:
            print(f"Warning: failed to load user profile: {error}")

    def save_profile(self) -> None:
        """Save the profile to the user home directory."""
        PROFILE_PATH.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "name": self.custom_name,
            "preferences": self.preferences,
        }

        try:
            with PROFILE_PATH.open("w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2)
        except OSError as error:
            print(f"Warning: failed to save user profile: {error}")

    def set_name(self, name: str) -> None:
        self.custom_name = name
        self.save_profile()

    def get_name(self) -> str:
        return self.custom_name or self.system_username

    def get_greeting(self) -> str:
        hour = datetime.now().hour
        name = self.get_name()
        if 5 <= hour < 12:
            return f"Good morning, {name}!"
        if 12 <= hour < 17:
            return f"Good afternoon, {name}!"
        if 17 <= hour < 22:
            return f"Good evening, {name}!"
        return f"Hello, {name}!"


class CommandExecutor:
    """Parse commands and execute user intents."""

    APP_MAP = {
        # Web applications
        "youtube": "https://www.youtube.com",
        "gmail": "https://mail.google.com",
        "mail": "https://mail.google.com",
        "email": "https://mail.google.com",
        "calendar": "https://calendar.google.com",
        "maps": "https://maps.google.com",
        "drive": "https://drive.google.com",
        "facebook": "https://www.facebook.com",
        "twitter": "https://www.twitter.com",
        "instagram": "https://www.instagram.com",
        "reddit": "https://www.reddit.com",
        "linkedin": "https://www.linkedin.com",
        "github": "https://www.github.com",
        "whatsapp": "https://web.whatsapp.com",
        # Desktop applications
        "chrome": "chrome",
        "firefox": "firefox",
        "edge": "msedge",
        "notepad": "notepad",
        "calculator": "calc",
        "terminal": "cmd",
        "command prompt": "cmd",
        "powershell": "powershell",
        "paint": "mspaint",
        "word": "winword",
        "excel": "excel",
        "powerpoint": "powerpnt",
        "outlook": "outlook",
        "spotify": "spotify",
        "code": "code",
        "visual studio": "code",
        "vs code": "code",
    }

    def __init__(self, user_profile: UserProfile):
        self.system = platform.system()
        self.user = user_profile

    def parse_command(self, text: str) -> Dict[str, str]:
        text = text.lower().strip()
        text = re.sub(r"[?.!]+$", "", text)
        text = self._normalize_speech_errors(text)

        if any(word in text for word in ["hello", "hi", "hey there", "greetings"]):
            return {"intent": "greeting", "query": ""}

        if any(phrase in text for phrase in ["who am i", "what's my name", "what is my name", "my name", "who is the user"]):
            return {"intent": "username", "query": ""}

        if re.search(r"\b(?:call me|my name is)\b", text):
            return {"intent": "set_name", "name": self._extract_name(text)}

        if any(word in text for word in ["search", "find", "look up", "google"]):
            return {"intent": "search", "query": self._extract_search_query(text)}

        if any(word in text for word in ["news", "headlines", "latest news"]):
            return {"intent": "news", "query": ""}

        if re.match(r"\b(?:open|launch|start|show)\b", text):
            return {"intent": "open_app", "app": self._extract_app_name(text)}

        if "time" in text or "what time" in text:
            return {"intent": "time", "query": ""}

        if "date" in text or "what date" in text or "what day" in text:
            return {"intent": "date", "query": ""}

        if re.search(r"\bplay\b", text):
            return {"intent": "play", "query": self._extract_play_query(text)}

        if "weather" in text:
            return {"intent": "weather", "location": self._extract_location(text)}

        if "help" in text or "what can you do" in text or "commands" in text:
            return {"intent": "help", "query": ""}

        return {"intent": "unknown", "query": text}

    def _normalize_speech_errors(self, text: str) -> str:
        replacements = {
            "you tube": "youtube",
            "you two": "youtube",
            "g mail": "gmail",
            "face book": "facebook",
        }
        for search, replace in replacements.items():
            text = text.replace(search, replace)
        return text

    def _extract_name(self, text: str) -> str:
        match = re.search(r"(?:call me|my name is)\s+(.+)", text)
        return match.group(1).strip() if match else ""

    def _extract_search_query(self, text: str) -> str:
        patterns = [
            r"search (?:for |about )?(.+)",
            r"find (?:me )?(?:about )?(.+)",
            r"look up (.+)",
            r"google (.+)",
        ]
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(1).strip()
        return text

    def _extract_app_name(self, text: str) -> str:
        match = re.search(r"(?:open|launch|start|show)\s+(?:the\s+)?(.+)", text)
        return match.group(1).strip() if match else text.strip()

    def _extract_play_query(self, text: str) -> str:
        match = re.search(r"play\s+(.+)", text)
        return match.group(1).strip() if match else ""

    def _extract_location(self, text: str) -> str:
        match = re.search(r"weather (?:in |for )?(.+)", text)
        return match.group(1).strip() if match else ""

    def execute(self, command: Dict[str, str]) -> str:
        intent = command["intent"]
        if intent == "greeting":
            return self._greet_user()
        if intent == "username":
            return self._tell_username()
        if intent == "set_name":
            return self._set_username(command["name"])
        if intent == "search":
            return self._search_web(command["query"])
        if intent == "news":
            return self._get_news()
        if intent == "open_app":
            return self._open_application(command["app"])
        if intent == "time":
            return self._get_time()
        if intent == "date":
            return self._get_date()
        if intent == "play":
            return self._play_media(command["query"])
        if intent == "weather":
            return self._get_weather(command["location"])
        if intent == "help":
            return self._show_help()
        return f"Sorry {self.user.get_name()}, I didn't understand that command."

    def _greet_user(self) -> str:
        return self.user.get_greeting()

    def _tell_username(self) -> str:
        name = self.user.get_name()
        system_user = self.user.system_username
        if self.user.custom_name:
            return f"You are {name}. Your system username is {system_user}."
        return f"Your username is {system_user}. You can ask me to call you something else by saying 'call me [name]'."

    def _set_username(self, name: str) -> str:
        if not name:
            return "I didn't catch your name. Please say 'call me [your name]'."
        self.user.set_name(name)
        return f"Okay, I'll call you {name} from now on!"

    def _search_web(self, query: str) -> str:
        query = query.strip()
        url = f"https://www.google.com/search?q={query.replace(' ', '+')}"
        webbrowser.open(url)
        return f"Searching for: {query}"

    def _get_news(self) -> str:
        webbrowser.open("https://news.google.com")
        return "Opening latest news headlines"

    def _open_application(self, app_name: str) -> str:
        app_name = app_name.lower().strip()
        if not app_name:
            return "Please tell me what to open."

        target = self._resolve_app_target(app_name)
        if not target:
            return f"Application '{app_name}' not recognized. Try: YouTube, Gmail, Chrome, Calculator, etc."

        if target.startswith("http"):
            webbrowser.open(target)
            return f"Opening {app_name.title()}"

        if self._launch_desktop_target(target):
            return f"Opening {app_name.title()}"
        return f"Couldn't open {app_name.title()}."

    def _resolve_app_target(self, app_name: str) -> Optional[str]:
        if app_name in self.APP_MAP:
            return self.APP_MAP[app_name]
        for alias, target in self.APP_MAP.items():
            if alias in app_name or app_name in alias:
                return target
        return None

    def _launch_desktop_target(self, target: str) -> bool:
        try:
            if self.system == "Windows":
                os.startfile(target)
                return True
            if self.system == "Darwin":
                subprocess.Popen(["open", "-a", target], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                return True
            if shutil.which(target):
                subprocess.Popen([target], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                return True
            if Path(target).exists():
                subprocess.Popen([str(target)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                return True
        except OSError as error:
            print(f"Error launching {target}: {error}")
        return False

    def _get_time(self) -> str:
        return f"The time is {datetime.now().strftime('%I:%M %p')}, {self.user.get_name()}"

    def _get_date(self) -> str:
        return f"Today is {datetime.now().strftime('%A, %B %d, %Y')}"

    def _play_media(self, query: str) -> str:
        query = query.strip() or "music"
        url = f"https://www.youtube.com/results?search_query={query.replace(' ', '+')}"
        webbrowser.open(url)
        return f"Playing {query} on YouTube"

    def _get_weather(self, location: str) -> str:
        location = location.strip() or "current location"
        url = f"https://www.google.com/search?q=weather+{location.replace(' ', '+')}"
        webbrowser.open(url)
        return f"Getting weather for {location}"

    def _show_help(self) -> str:
        return (
            "Available commands:\n"
            "- Search: 'search for [query]'\n"
            "- News: 'get me the news'\n"
            "- Open apps: 'open [YouTube/Gmail/Chrome/etc]'\n"
            "- Time: 'what's the time?'\n"
            "- Date: 'what's the date?'\n"
            "- Play: 'play [song/video]'\n"
            "- Weather: 'weather in [location]'\n"
            "- Username: 'who am I?'\n"
            "- Set name: 'call me [name]'\n"
            "- Greeting: 'hello'"
        )


def load_model(model_path: str) -> Model:
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found at: {model_path}\n"
            "Download from: https://alphacephei.com/vosk/models"
        )
    print(f"Loading English model from: {model_path}")
    model = Model(model_path)
    print("[OK] Model loaded successfully")
    return model


def detect_wake_word(text: str) -> bool:
    text = text.lower().strip()
    return any(wake_word in text for wake_word in WAKE_WORDS)


def remove_wake_word(text: str) -> str:
    text_lower = text.lower()
    for wake_word in WAKE_WORDS:
        if wake_word in text_lower:
            idx = text_lower.index(wake_word)
            return text[idx + len(wake_word):].strip()
    return text


class VoiceAssistant:
    """Voice assistant with wake word detection and command execution."""

    def __init__(self, model: Model):
        self.model = model
        self.audio_queue: queue.Queue = queue.Queue()
        self._stop_event = threading.Event()
        self.user_profile = UserProfile()
        self.executor = CommandExecutor(self.user_profile)
        self.listening_for_command = False

    def _mic_callback(self, indata, frames, time, status):
        self.audio_queue.put(bytes(indata))

    def _process_segment(self, audio_chunks: list) -> None:
        recognizer = KaldiRecognizer(self.model, SAMPLE_RATE)
        for chunk in audio_chunks:
            recognizer.AcceptWaveform(chunk)

        result = json.loads(recognizer.FinalResult())
        text = result.get("text", "").strip()
        if not text:
            return

        print(f"[HEARD] {text}")
        if detect_wake_word(text):
            print("[ASSISTANT] Yes? I'm listening...")
            self.listening_for_command = True
            command_text = remove_wake_word(text)
            if command_text:
                self._execute_command(command_text)
        elif self.listening_for_command:
            self._execute_command(text)
            self.listening_for_command = False

    def _execute_command(self, text: str) -> None:
        if not text:
            return
        print(f"[COMMAND] {text}")
        command = self.executor.parse_command(text)
        print(f"[INTENT] {command['intent']}")
        response = self.executor.execute(command)
        print(f"[ASSISTANT] {response}")

    def _listener(self) -> None:
        buffer: list = []
        silent_count = 0
        speaking = False

        while not self._stop_event.is_set():
            try:
                chunk = self.audio_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            if audioop.rms(chunk, 2) < SILENCE_THRESHOLD:
                silent_count += 1
                buffer.append(chunk)
                if speaking and silent_count >= SILENCE_CHUNKS:
                    if buffer:
                        self._process_segment(list(buffer))
                        buffer = []
                    speaking = False
                    silent_count = 0
            else:
                speaking = True
                silent_count = 0
                buffer.append(chunk)
                if len(buffer) >= MAX_BUFFER_CHUNKS:
                    self._process_segment(list(buffer))
                    buffer = []

    def start(self) -> None:
        print("\n" + "=" * 60)
        print("🎤 VOICE ASSISTANT ACTIVE (English Only - Optimized)")
        print("=" * 60)
        print(f"👤 User: {self.user_profile.get_name()}")
        print(f"💻 System: {self.user_profile.system_username}")
        print(f"🔊 Wake word: {WAKE_WORDS[0]}")
        print("\n✨ Example commands:")
        print("  • 'Hey buddy, who am I?'")
        print("  • 'Hey buddy, open YouTube'")
        print("  • 'Hey buddy, search for AI news'")
        print("  • 'Hey buddy, what's the time?'")
        print("  • 'Hey buddy, get me the news'")
        print("  • 'Hey buddy, play music'")
        print("\nPress Ctrl+C to stop.\n")
        print("=" * 60 + "\n")

        print(f"[ASSISTANT] {self.user_profile.get_greeting()} I'm ready to help!\n")

        listener_thread = threading.Thread(target=self._listener, daemon=True)
        listener_thread.start()

        try:
            with sd.RawInputStream(
                samplerate=SAMPLE_RATE,
                blocksize=8000,
                dtype="int16",
                channels=1,
                callback=self._mic_callback,
            ):
                while not self._stop_event.is_set():
                    sd.sleep(100)
        except KeyboardInterrupt:
            print("\n[INFO] Voice assistant stopped. Goodbye!")
            self._stop_event.set()
            listener_thread.join()


if __name__ == "__main__":
    try:
        model = load_model(MODEL_PATH)
    except FileNotFoundError as error:
        print(f"\n❌ Error: {error}")
        print("\nDownload the English model:")
        print("1. Visit: https://alphacephei.com/vosk/models")
        print("2. Download: vosk-model-small-en-us-0.15")
        print(f"3. Extract to: {MODEL_PATH}")
        exit(1)

    assistant = VoiceAssistant(model)
    assistant.start()
