import sounddevice as sd
import queue
import json
from vosk import Model, KaldiRecognizer

# ------ Config -------
MODEL_PATH = "vosk-model-small-en-us-0.15"
SAMPLE_RATE = 16000

# ------ LOAD MODEL -------
print("Loading model...")
model = Model(MODEL_PATH)
recognizer = KaldiRecognizer(model, SAMPLE_RATE)

# ------ Queue for audio chunks -------
audio_queue = queue.Queue()

# ------ Audio callback function -------
def callback(indata, frames, time, status):
    if status:
        print(f"Audio callback error: {status}")
    audio_queue.put(bytes(indata))


# ------ Start audio stream -------
print("Starting audio stream...")
try:
    with sd.RawInputStream(
        samplerate = SAMPLE_RATE,
        blocksize = 8000,
        dtype = "int16",
        channels = 1,
        callback = callback
    ):
        while True:
            data = audio_queue.get()

            if recognizer.AcceptWaveform(data):
                result = json.loads(recognizer.Result())
                text = result.get("text", "")
                if text:
                    print(f"Recognized: {text}")
            else:
                partial = json.loads(recognizer.PartialResult())
                partial_text = partial.get("partial", "")
except KeyboardInterrupt:
    print("Stopping audio stream...")