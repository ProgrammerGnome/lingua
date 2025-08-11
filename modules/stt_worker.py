import time
import queue
import threading
import numpy as np
import sounddevice as sd
import torch
from PySide6.QtCore import QObject, Signal

try:
    import whisper

    WHISPER_AVAILABLE = True
except Exception as e:
    WHISPER_AVAILABLE = False
    print("Whisper import error:", e)

from translator import TranslatorWrapper

SAMPLE_RATE = 16000
CHANNELS = 1


class STTWorker(QObject):

    new_result = Signal(str, str, str)  # original, src_lang, translated
    status = Signal(str)
    error = Signal(str)

    def __init__(self, device_index, model_lang, whisper_model_name="small",
                 buffer_seconds=6.0, transcribe_interval=1.0, parent=None):
        super().__init__(parent)
        self.device_index = device_index
        self.model_lang = model_lang
        self.model_name = whisper_model_name
        self.buffer_seconds = float(buffer_seconds)
        self.transcribe_interval = float(transcribe_interval)

        self._running = False
        self._audio_q = queue.Queue()
        self._stream = None
        self._worker_thread = None

        if not WHISPER_AVAILABLE:
            raise RuntimeError("Whisper library not installed (pip install openai-whisper).")

        try:
            self.status.emit("Loading Whisper model...")
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model = whisper.load_model(self.model_name, device=device)
            self.status.emit(f"Whisper '{self.model_name}' loaded on {device}.")
        except Exception as e:
            raise RuntimeError(f"Error during the load Whisper model: {e}")

        self.translator = TranslatorWrapper()

    def _audio_callback(self, indata, frames, time_info, status):
        if status:
            print("Audio callback status:", status, flush=True)
        try:
            self._audio_q.put_nowait(indata.copy())
        except queue.Full:
            pass

    def start(self):
        if self._running:
            return
        self._running = True
        try:
            self._stream = sd.InputStream(samplerate=SAMPLE_RATE,
                                          channels=CHANNELS,
                                          dtype='int16',
                                          device=self.device_index,
                                          callback=self._audio_callback,
                                          blocksize=1024)
            self._stream.start()
        except Exception as e:
            self.error.emit(f"Audio input stream error: {e}")
            self._running = False
            return

        self._worker_thread = threading.Thread(target=self._process_loop, daemon=True)
        self._worker_thread.start()
        self.status.emit("Listening (streaming)...")

    def stop(self):
        self._running = False
        try:
            if self._stream is not None:
                self._stream.stop()
                self._stream.close()
                self._stream = None
        except Exception:
            pass
        self.status.emit("Stopped.")

    def _process_loop(self):
        buffer = np.zeros((0,), dtype=np.float32)
        max_samples = int(self.buffer_seconds * SAMPLE_RATE)
        last_transcribe = 0.0

        try:
            while self._running:
                pulled = False
                while True:
                    try:
                        block = self._audio_q.get_nowait()
                        pulled = True
                        block_f = block.astype(np.float32).reshape(-1) / 32768.0
                        buffer = np.concatenate((buffer, block_f))
                        if buffer.shape[0] > max_samples:
                            buffer = buffer[-max_samples:]
                    except queue.Empty:
                        break

                now = time.monotonic()
                if (now - last_transcribe) >= self.transcribe_interval and buffer.shape[0] > SAMPLE_RATE * 0.5:
                    audio_for_model = buffer.copy()
                    last_transcribe = now
                    try:
                        if self.model_lang == "hu":
                            res = self.model.transcribe(audio_for_model, language="hu", task="transcribe")
                            original = (res.get("text") or "").strip()
                            translated = self.translator.translate(original, "hu")
                            if not translated:
                                res2 = self.model.transcribe(audio_for_model, language="hu", task="translate")
                                translated = (res2.get("text") or "").strip()
                            src = "hu"
                        else:
                            res = self.model.transcribe(audio_for_model, language="en", task="transcribe")
                            original = (res.get("text") or "").strip()
                            translated = self.translator.translate(original, "en")
                            src = "en"

                        if original:
                            self.new_result.emit(original, src, translated or "")

                    except Exception as e:
                        print("Processing/transcribe error:", e, flush=True)
                        self.error.emit(f"Transcribe error: {e}")

                    try:
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    except Exception:
                        pass

                if not pulled:
                    time.sleep(0.01)
                else:
                    time.sleep(0.001)

        except Exception as e:
            self.error.emit(str(e))
        finally:
            try:
                if self._stream is not None:
                    self._stream.stop()
                    self._stream.close()
            except Exception:
                pass
            self._running = False
