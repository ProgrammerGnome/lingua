#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
STT + Translate GUI using Whisper (no Vosk).
- list microphones
- select device
- record short chunks, transcribe with whisper (local)
- translate EN<->HU using Argos if available, otherwise googletrans (online)
- Qt GUI with PySide6

Requirements: see installation instructions in accompanying message.
"""

import sys
import threading
import queue
import tempfile
import time
import os
from pathlib import Path

from PySide6 import QtWidgets, QtCore
from PySide6.QtCore import Signal, Slot

import sounddevice as sd
import soundfile as sf

# whisper
try:
    import whisper
    WHISPER_AVAILABLE = True
except Exception as e:
    WHISPER_AVAILABLE = False
    print("Whisper import error:", e)

# argos
try:
    import argostranslate.package
    import argostranslate.translate
    ARGOS_AVAILABLE = True
except Exception:
    ARGOS_AVAILABLE = False

# googletrans fallback
try:
    from googletrans import Translator as GoogleTranslator
    GOOGLETRANS_AVAILABLE = True
except Exception:
    GOOGLETRANS_AVAILABLE = False

SAMPLE_RATE = 16000
CHANNELS = 1
CHUNK_SECONDS = 4.0  # how long each recorded chunk is (seconds)

def list_input_devices():
    devs = sd.query_devices()
    input_devs = []
    for i, d in enumerate(devs):
        if d['max_input_channels'] > 0:
            input_devs.append({'index': i, 'name': d['name'], 'max_input_channels': d['max_input_channels']})
    return input_devs

class TranslatorWrapper:
    """Try Argos offline first; if not available, try googletrans online."""
    def __init__(self):
        self.argos = False
        self.google = False
        if ARGOS_AVAILABLE:
            try:
                self.langs = argostranslate.translate.get_installed_languages()
                # find en and hu
                self.en = next((l for l in self.langs if l.code.startswith("en")), None)
                self.hu = next((l for l in self.langs if l.code.startswith("hu")), None)
                if self.en and self.hu:
                    self.en_to_hu = self.en.get_translation(self.hu)
                    self.hu_to_en = self.hu.get_translation(self.en)
                    self.argos = True
                    print("Argos Translate ready (offline).")
            except Exception as e:
                print("Argos init error:", e)
                self.argos = False
        if not self.argos and GOOGLETRANS_AVAILABLE:
            self.google = True
            self.gtr = GoogleTranslator()
            print("Googletrans available (online fallback).")

    def translate(self, text, src):
        """
        src: 'en' or 'hu' (what whisper produced)
        returns: translated string or None
        """
        if not text:
            return None
        try:
            if self.argos:
                if src.startswith("en") and self.en_to_hu:
                    return self.en_to_hu.translate(text)
                if src.startswith("hu") and self.hu_to_en:
                    return self.hu_to_en.translate(text)
                return None
            elif self.google:
                # googletrans auto-detects target
                dest = 'hu' if src.startswith('en') else 'en'
                res = self.gtr.translate(text, src=src, dest=dest)
                return res.text
            else:
                return None
        except Exception as e:
            print("Translate error:", e)
            return None

class STTWorker(QtCore.QObject):
    new_result = Signal(str, str, str)  # original, src_lang, translated
    status = Signal(str)
    error = Signal(str)

    def __init__(self, device_index, model_lang, whisper_model_name="small", parent=None):
        super().__init__(parent)
        self.device_index = device_index
        self.model_lang = model_lang  # "en" or "hu"
        self.model_name = whisper_model_name
        self._running = False
        self.audio_q = queue.Queue()
        self._rec_thread = None
        self._proc_thread = None

        if not WHISPER_AVAILABLE:
            raise RuntimeError("Whisper könyvtár nincs telepítve (pip install openai-whisper).")

        # load whisper model (may take time)
        try:
            self.status.emit("Loading Whisper model...")
            self.model = whisper.load_model(self.model_name)
            self.status.emit(f"Whisper model '{self.model_name}' loaded.")
        except Exception as e:
            raise RuntimeError(f"Whisper model betöltési hiba: {e}")

        self.translator = TranslatorWrapper()

    @Slot()
    def start(self):
        if self._running:
            return
        self._running = True
        # start recorder thread
        self._rec_thread = threading.Thread(target=self._rec_loop, daemon=True)
        self._rec_thread.start()
        # start processor thread
        self._proc_thread = threading.Thread(target=self._proc_loop, daemon=True)
        self._proc_thread.start()
        self.status.emit("Listening...")

    @Slot()
    def stop(self):
        self._running = False
        self.status.emit("Stopping...")

    def _rec_loop(self):
        """Record CHUNK_SECONDS segments and put file path into queue."""
        try:
            while self._running:
                # record CHUNK_SECONDS seconds (blocking)
                data = sd.rec(int(CHUNK_SECONDS * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=CHANNELS, dtype='int16', device=self.device_index)
                sd.wait()
                # save to a temp wav file
                fd, tmpname = tempfile.mkstemp(suffix=".wav")
                os.close(fd)
                sf.write(tmpname, data, SAMPLE_RATE, subtype='PCM_16')
                self.audio_q.put(tmpname)
                # small sleep to avoid tight loop
                time.sleep(0.05)
        except Exception as e:
            self.error.emit(str(e))
            self._running = False

    def _proc_loop(self):
        """Take audio files from queue, transcribe with Whisper, translate, emit results."""
        try:
            while self._running or not self.audio_q.empty():
                try:
                    path = self.audio_q.get(timeout=0.5)
                except queue.Empty:
                    continue
                try:
                    # whisper transcribe (for HU->EN we can use task='translate' to get English)
                    # choose task depending on desired behavior:
                    if self.model_lang == "hu":
                        # Transcribe Hungarian; for HU->EN, we can ask whisper to translate to English directly:
                        # But to keep original text + translation, do both:
                        # 1) transcribe in Hungarian
                        res = self.model.transcribe(path, language="hu", task="transcribe")
                        hu_text = (res.get("text") or "").strip()
                        translated = None
                        # try Argos/Google translate HU->EN
                        translated = self.translator.translate(hu_text, "hu")
                        # If translator is None, as fallback we can ask whisper to translate to English using task='translate'
                        if not translated:
                            res2 = self.model.transcribe(path, language="hu", task="translate")
                            translated = (res2.get("text") or "").strip()
                        original = hu_text
                        src = "hu"
                    else:
                        # model_lang == "en"
                        # transcribe English
                        res = self.model.transcribe(path, language="en", task="transcribe")
                        en_text = (res.get("text") or "").strip()
                        # translate to Hungarian (prefer Argos)
                        translated = self.translator.translate(en_text, "en")
                        original = en_text
                        src = "en"

                    # emit result
                    if original:
                        self.new_result.emit(original, src, translated or "")
                except Exception as e:
                    print("Processing error:", e)
                finally:
                    try:
                        os.remove(path)
                    except Exception:
                        pass
        except Exception as e:
            self.error.emit(str(e))
            self._running = False

class MainWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Whisper STT + Translate (EN <-> HU)")
        self.resize(700, 420)

        self.layout = QtWidgets.QVBoxLayout(self)

        # devices
        self.layout.addWidget(QtWidgets.QLabel("Válassz mikrofont:"))
        self.dev_combo = QtWidgets.QComboBox()
        for d in list_input_devices():
            self.dev_combo.addItem(f"{d['name']} (#{d['index']})", d['index'])
        self.layout.addWidget(self.dev_combo)

        # model selection
        self.layout.addWidget(QtWidgets.QLabel("Beszédfelismerés modell nyelve (Whisper):"))
        self.lang_combo = QtWidgets.QComboBox()
        self.lang_combo.addItem("English (listen in EN => translate to HU)", "en")
        self.lang_combo.addItem("Magyar (listen in HU => translate to EN)", "hu")
        self.layout.addWidget(self.lang_combo)

        # whisper model size selection
        self.layout.addWidget(QtWidgets.QLabel("Whisper modell (teljesítmény vs pontosság):"))
        self.model_combo = QtWidgets.QComboBox()
        for m in ["tiny", "base", "small", "medium", "large"]:
            self.model_combo.addItem(m)
        self.model_combo.setCurrentText("small")
        self.layout.addWidget(self.model_combo)

        # start/stop
        row = QtWidgets.QHBoxLayout()
        self.start_btn = QtWidgets.QPushButton("Start")
        self.stop_btn = QtWidgets.QPushButton("Stop")
        self.stop_btn.setEnabled(False)
        row.addWidget(self.start_btn)
        row.addWidget(self.stop_btn)
        self.layout.addLayout(row)

        # status
        self.status_label = QtWidgets.QLabel("Ready")
        self.layout.addWidget(self.status_label)

        # results
        self.layout.addWidget(QtWidgets.QLabel("Felismert szöveg:"))
        self.orig_text = QtWidgets.QTextEdit()
        self.orig_text.setReadOnly(True)
        self.layout.addWidget(self.orig_text)

        self.layout.addWidget(QtWidgets.QLabel("Fordítás:"))
        self.trans_text = QtWidgets.QTextEdit()
        self.trans_text.setReadOnly(True)
        self.layout.addWidget(self.trans_text)

        # connections
        self.start_btn.clicked.connect(self.start_listening)
        self.stop_btn.clicked.connect(self.stop_listening)

        self.worker = None
        self.worker_thread = None

    @Slot()
    def start_listening(self):
        if self.dev_combo.count() == 0:
            QtWidgets.QMessageBox.warning(self, "Hiba", "Nincs elérhető bemeneti eszköz.")
            return
        device_index = self.dev_combo.currentData()
        model_lang = self.lang_combo.currentData()
        model_size = self.model_combo.currentText()
        try:
            self.worker = STTWorker(device_index=device_index, model_lang=model_lang, whisper_model_name=model_size)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Hiba a munkás létrehozásakor", str(e))
            return

        # connect signals
        self.worker.new_result.connect(self.on_new_result)
        self.worker.status.connect(lambda s: self.status_label.setText(s))
        self.worker.error.connect(lambda e: QtWidgets.QMessageBox.critical(self, "Worker hiba", e))

        # start worker (runs threads internally)
        self.worker.start()
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.status_label.setText("Indítva...")

    @Slot()
    def stop_listening(self):
        if self.worker:
            self.worker.stop()
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.status_label.setText("Leállítva.")

    @Slot(str, str, str)
    def on_new_result(self, original, src, translated):
        timestamp = time.strftime("%H:%M:%S")
        self.orig_text.append(f"[{timestamp}] [{src}] {original}")
        if translated:
            self.trans_text.append(f"[{timestamp}] → {translated}")
        else:
            self.trans_text.append(f"[{timestamp}] → (fordítás nem elérhető)")

    def closeEvent(self, event):
        try:
            self.stop_listening()
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
