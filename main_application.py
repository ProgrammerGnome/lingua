import sys
import time
import threading
from PySide6 import QtWidgets
from PySide6.QtCore import Slot

from modules.devices import list_input_devices
from modules.stt_worker import STTWorker


class MainWindow(QtWidgets.QWidget):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Lingua (EN <-> HU)")
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

        self.status_label = QtWidgets.QLabel(None)
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
    from PySide6.QtWidgets import QApplication
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
