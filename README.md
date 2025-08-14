# Lingua

This application is a real-time speech-to-text and translation tool that uses the **OpenAI Whisper** model for continuous audio stream processing.  
It can transcribe and translate **English ⇄ Hungarian** conversations in near real time.

## Key Features
- **Real-time (streaming) speech recognition** with GPU acceleration (CUDA) when available.
- **Bidirectional translation** (EN ⇄ HU) using Whisper or Google Translate as fallback.
- **PySide6 graphical interface** with microphone selection, language setting, and model size options.
- **Minimal latency** thanks to continuous buffering and timed transcription.

## Installation

1.) Clone the repository:
```bash
https://github.com/ProgrammerGnome/lingua.git
cd whisper-stt-translate
```
2.) Install the required dependencies via the provided install.sh script:
```
chmod +x install.sh
./install.sh
```
3.) Run the appliaction:
```
python main.py
```
