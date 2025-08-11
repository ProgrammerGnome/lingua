#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Streaming STT + Translate GUI using Whisper (no Vosk).
- continuous audio via sounddevice.InputStream (callback)
- buffer (moving window) and periodic transcribe => near-realtime
- GPU usage if available, and torch.cuda.empty_cache() after transcription
- PySide6 GUI
"""

import sys
import time
import queue
import threading

from PySide6 import QtWidgets, QtCore
from PySide6.QtCore import Signal, Slot

import numpy as np
import sounddevice as sd
import torch

try:
    import whisper
    WHISPER_AVAILABLE = True
except Exception as e:
    WHISPER_AVAILABLE = False
    print("Whisper import error:", e)

try:
    from googletrans import Translator as GoogleTranslator
    GOOGLETRANS_AVAILABLE = True
except Exception:
    GOOGLETRANS_AVAILABLE = False

SAMPLE_RATE = 16000
CHANNELS = 1

def list_input_devices():
    devs = sd.query_devices()
    input_devs = []
    for i, d in enumerate(devs):
        if d['max_input_channels'] > 0:
            input_devs.append({'index': i, 'name': d['name'], 'max_input_channels': d['max_input_channels']})
    return input_devs

class TranslatorWrapper:
    def __init__(self):
        self.google = False
        if GOOGLETRANS_AVAILABLE:
            self.google = True
            self.gtr = GoogleTranslator()
            print("Googletrans available (online fallback).")

    def translate(self, text, src):
        if not text:
            return None
        try:
            if self.google:
                dest = 'hu' if src.startswith('en') else 'en'
                res = self.gtr.translate(text, src=src, dest=dest)
                return res.text
            return None
        except Exception as e:
            print("Translate error:", e)
            return None

class STTWorker(QtCore.QObject):
    new_result = Signal(str, str, str)  # original, src_lang, translated
    status = Signal(str)
    error = Signal(str)

    def __init__(self, device_index, model_lang, whisper_model_name="small",
                 buffer_seconds=6.0, transcribe_interval=1.0, parent=None):
        super().__init__(parent)
        self.device_index = device_index
        self.model_lang = model_lang  # 'en' or 'hu' (what we're expecting to listen for)
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

class MainWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Whisper Streaming STT + Translate (EN <-> HU)")
        self.resize(720, 480)

        self.layout = QtWidgets.QVBoxLayout(self)

        self.layout.addWidget(QtWidgets.QLabel("Please select the microphone:"))
        self.dev_combo = QtWidgets.QComboBox()
        for d in list_input_devices():
            self.dev_combo.addItem(f"{d['name']} (#{d['index']})", d['index'])
        self.layout.addWidget(self.dev_combo)

        self.layout.addWidget(QtWidgets.QLabel("Language of recognized speech (Whisper):"))
        self.lang_combo = QtWidgets.QComboBox()
        self.lang_combo.addItem("English (listen in EN => translate to HU)", "en")
        self.lang_combo.addItem("Magyar (listen in HU => translate to EN)", "hu")
        self.layout.addWidget(self.lang_combo)

        self.layout.addWidget(QtWidgets.QLabel("Please select the Whisper model (performance vs accuracy):"))
        self.model_combo = QtWidgets.QComboBox()
        for m in ["tiny", "base", "small", "medium", "large"]:
            self.model_combo.addItem(m)
        self.model_combo.setCurrentText("small")
        self.layout.addWidget(self.model_combo)

        row = QtWidgets.QHBoxLayout()
        self.start_btn = QtWidgets.QPushButton("Start")
        self.stop_btn = QtWidgets.QPushButton("Stop")
        self.stop_btn.setEnabled(False)
        row.addWidget(self.start_btn)
        row.addWidget(self.stop_btn)
        self.layout.addLayout(row)

        self.status_label = QtWidgets.QLabel("Ready")
        self.layout.addWidget(self.status_label)

        self.layout.addWidget(QtWidgets.QLabel("Recognized word/text:"))
        self.orig_text = QtWidgets.QTextEdit()
        self.orig_text.setReadOnly(True)
        self.layout.addWidget(self.orig_text, 3)

        self.layout.addWidget(QtWidgets.QLabel("Translated recognized word/text:"))
        self.trans_text = QtWidgets.QTextEdit()
        self.trans_text.setReadOnly(True)
        self.layout.addWidget(self.trans_text, 2)

        self.start_btn.clicked.connect(self.start_listening)
        self.stop_btn.clicked.connect(self.stop_listening)

        self.worker = None
        self._worker_thread = None

    @Slot()
    def start_listening(self):
        if self.dev_combo.count() == 0:
            QtWidgets.QMessageBox.warning(self, "Error", "Available microphone device not found.")
            return

        device_index = self.dev_combo.currentData()
        model_lang = self.lang_combo.currentData()
        model_size = self.model_combo.currentText()

        try:
            self.worker = STTWorker(device_index=device_index,
                                    model_lang=model_lang,
                                    whisper_model_name=model_size,
                                    buffer_seconds=6.0,
                                    transcribe_interval=1.0)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"Worker init error: {e}")
            return

        self.worker.new_result.connect(self.on_new_result)
        self.worker.status.connect(lambda s: self.status_label.setText(s))
        self.worker.error.connect(lambda e: QtWidgets.QMessageBox.critical(self, "Worker error", e))

        self._worker_thread = threading.Thread(target=self.worker.start, daemon=True)
        self._worker_thread.start()

        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.status_label.setText("Started...")

    @Slot()
    def stop_listening(self):
        if self.worker:
            self.worker.stop()
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.status_label.setText("Stopped.")

    @Slot(str, str, str)
    def on_new_result(self, original, src, translated):
        timestamp = time.strftime("%H:%M:%S")
        self.orig_text.append(f"[{timestamp}] [{src}] {original}")
        if translated:
            self.trans_text.append(f"[{timestamp}] → {translated}")
        else:
            self.trans_text.append(f"[{timestamp}] → (Translation not available)")

    def closeEvent(self, event):
        try:
            if self.worker:
                self.worker.stop()
        except Exception:
            pass
        super().closeEvent(event)

def main():
    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
