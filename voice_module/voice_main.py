import sounddevice as sd
import queue
import json
from vosk import Model, KaldiRecognizer
import os
import threading
import audioop

MODEL_CONFIG = {
    "en": "vosk-model-small-en-us-0.15",
    "hi": "vosk-model-small-hi-0.22",
    "fr": "vosk-model-small-fr-0.22",
    "de": "vosk-model-small-de-0.15",
    "cn": "vosk-model-small-cn-0.22",
}

SAMPLE_RATE = 16000
SILENCE_THRESHOLD = 500
SILENCE_CHUNKS = 20
MAX_BUFFER_CHUNKS = 60


def load_models(config: dict) -> dict:
    loaded = {}
    for lang, path in config.items():
        if not os.path.exists(path):
            print(f"[SKIP] Model for '{lang}' not found at: {path}")
            continue
        print(f"Loading model for '{lang}' ...")
        loaded[lang] = Model(path)
        print(f"[OK]   Model '{lang}' loaded.")
    return loaded


def create_recognizers(models: dict, sample_rate: int = SAMPLE_RATE) -> dict:
    return {
        lang: KaldiRecognizer(model, sample_rate)
        for lang, model in models.items()
    }


def get_avg_confidence(rec: KaldiRecognizer, audio_chunks: list) -> tuple[str, float]:
    for chunk in audio_chunks:
        rec.AcceptWaveform(chunk)
    result = json.loads(rec.FinalResult())
    text = result.get("text", "")
    words = result.get("result", [])
    if not words:
        return text, 0.0
    avg_conf = sum(w.get("conf", 0.0) for w in words) / len(words)
    return text, avg_conf


def detect_language_by_confidence(models: dict, audio_chunks: list) -> tuple[str, str]:
    best_lang = None
    best_text = ""
    best_conf = -1.0
    results = {}

    def evaluate(lang, model):
        rec = KaldiRecognizer(model, SAMPLE_RATE)
        rec.SetWords(True)
        text, conf = get_avg_confidence(rec, audio_chunks)
        results[lang] = (text, conf)

    threads = [
        threading.Thread(target=evaluate, args=(lang, model))
        for lang, model in models.items()
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    for lang, (text, conf) in results.items():
        if conf > best_conf:
            best_conf = conf
            best_lang = lang
            best_text = text

    return best_lang, best_text


class SmartMultiLanguageStream:
    def __init__(self, models: dict):
        self.models = models
        self.audio_queue = queue.Queue()
        self._stop_event = threading.Event()

    def _mic_callback(self, indata, frames, time, status):
        self.audio_queue.put(bytes(indata))

    def _process_segment(self, audio_chunks: list):
        lang, text = detect_language_by_confidence(self.models, audio_chunks)
        if text.strip():
            print(f"[{lang.upper()}] {text}")

    def _listener(self):
        buffer = []
        silent_count = 0
        speaking = False

        while not self._stop_event.is_set():
            try:
                chunk = self.audio_queue.get(timeout=1)
            except queue.Empty:
                continue

            if audioop.rms(chunk, 2) < SILENCE_THRESHOLD:
                silent_count += 1
                buffer.append(chunk)
                if speaking and silent_count >= SILENCE_CHUNKS:
                    if buffer:
                        threading.Thread(
                            target=self._process_segment,
                            args=(list(buffer),),
                            daemon=True
                        ).start()
                        buffer = []
                    speaking = False
                    silent_count = 0
            else:
                speaking = True
                silent_count = 0
                buffer.append(chunk)
                if len(buffer) >= MAX_BUFFER_CHUNKS:
                    threading.Thread(
                        target=self._process_segment,
                        args=(list(buffer),),
                        daemon=True
                    ).start()
                    buffer = []

    def start(self):
        print("\n[INFO] Listening ... Press Ctrl+C to stop.\n")
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
                while True:
                    sd.sleep(100)
        except KeyboardInterrupt:
            print("\n[INFO] Stopped.")
            self._stop_event.set()
            listener_thread.join()


class ManualLanguageStream:
    def __init__(self, models: dict, lang: str):
        if lang not in models:
            raise ValueError(f"Language '{lang}' not loaded. Available: {list(models)}")
        self.lang = lang
        self.model = models[lang]
        self.audio_queue = queue.Queue()

    def _mic_callback(self, indata, frames, time, status):
        self.audio_queue.put(bytes(indata))

    def _worker(self):
        rec = KaldiRecognizer(self.model, SAMPLE_RATE)
        print(f"[{self.lang.upper()}] Listening ... Press Ctrl+C to stop.\n")
        while True:
            data = self.audio_queue.get()
            if data is None:
                break
            if rec.AcceptWaveform(data):
                text = json.loads(rec.Result()).get("text", "")
                if text:
                    print(f"[{self.lang.upper()}] {text}")
            else:
                partial = json.loads(rec.PartialResult()).get("partial", "")
                if partial:
                    print(f"  ... {partial}", end="\r")

    def start(self):
        t = threading.Thread(target=self._worker, daemon=True)
        t.start()
        try:
            with sd.RawInputStream(
                samplerate=SAMPLE_RATE,
                blocksize=8000,
                dtype="int16",
                channels=1,
                callback=self._mic_callback,
            ):
                while True:
                    sd.sleep(100)
        except KeyboardInterrupt:
            print("\n[INFO] Stopped.")
            self.audio_queue.put(None)
            t.join()


if __name__ == "__main__":
    models = load_models(MODEL_CONFIG)
    if not models:
        print("No models loaded. Exiting.")
        exit(1)

    print(f"\nLoaded: {list(models.keys())}\n")
    print("Choose mode:")
    print("  1 — Auto language detection")
    print("  2 — Manual language selection")
    choice = input("Enter 1 or 2: ").strip()

    if choice == "1":
        stream = SmartMultiLanguageStream(models)
        stream.start()
    elif choice == "2":
        print(f"Available languages: {list(models.keys())}")
        lang = input("Enter language code (e.g. en / hi / fr): ").strip()
        stream = ManualLanguageStream(models, lang)
        stream.start()