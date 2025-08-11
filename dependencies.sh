#!/bin/sh

echo "Frissítem a pip-et..."
pip install --upgrade pip

echo "Telepítem a PySide6, sounddevice, soundfile, cmake csomagokat..."
pip install PySide6 sounddevice soundfile cmake

echo "Telepítem a sentencepiece csomagot..."
pip install sentencepiece

echo "Telepítem az openai-whisper csomagot..."
pip install -U openai-whisper

echo "Telepítem a googletrans 4.0.0-rc1 verzióját..."
pip install googletrans==4.0.0-rc1

echo "Telepítem a PyTorch CUDA 11.7 verzióját..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117

echo "Telepítem a numpy < 2 verziót..."
pip install "numpy<2"

echo "Telepítés kész."
