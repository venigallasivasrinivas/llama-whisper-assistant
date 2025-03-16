import pyttsx3
import whisper
import sounddevice as sd
import numpy as np
import os
import requests
import time
import json
import re

SAMPLE_RATE = 16000
DURATION = 5
OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL_NAME = "llama3.2"

# Load Whisper ONCE
print("🔊 Loading Whisper model...")
model = whisper.load_model("base")

# Initialize TTS engine ONCE
engine = pyttsx3.init(driverName='nsss')
engine.setProperty('volume', 1.0)
voices = engine.getProperty('voices')
for v in voices:
    if "Samantha" in v.name or "Alex" in v.name:
        engine.setProperty('voice', v.id)
        break

def record_audio(duration=DURATION):
    print("🎤 Recording...")
    audio = sd.rec(int(duration * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='float32')
    sd.wait()
    return np.squeeze(audio)

def transcribe_audio(audio_data):
    print("🔍 Transcribing...")
    audio_np = whisper.pad_or_trim(np.array(audio_data))
    mel = whisper.log_mel_spectrogram(audio_np).to(model.device)
    options = whisper.DecodingOptions(language="en", fp16=False)
    result = whisper.decode(model, mel, options)
    print(f"📝 You said: {result.text}")
    return result.text

def query_ollama(prompt):
    print("🧠 Asking LLaMA...")
    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False
    }

    try:
        response = requests.post(OLLAMA_URL, json=payload)
        response.raise_for_status()
        lines = response.text.strip().splitlines()
        full_response = ""
        for line in lines:
            try:
                parsed = json.loads(line)
                msg = parsed.get("message", {}).get("content", "")
                full_response += msg
            except json.JSONDecodeError:
                continue
        return full_response or "Sorry, no valid response found."
    except Exception as e:
        print(f"❌ Ollama error: {e}")
        return "Sorry, I couldn't reach the model."

def speak(text):
    try:
        clean_text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
        length = len(clean_text)

        # Dynamically adjust rate
        if length < 100:
            rate = 190
        elif length < 300:
            rate = 170
        else:
            rate = 150

        engine.setProperty('rate', rate)
        print(f"🔈 Speaking (rate={rate})...")
        engine.say(clean_text)
        engine.runAndWait()
        print("✅ Done speaking.")
    except Exception as e:
        print(f"❌ TTS error: {e}")

def main():
    print("🎙️ Voice Assistant Ready!\n")
    while True:
        audio = record_audio()
        transcription = transcribe_audio(audio)
        if transcription.strip() == "":
            print("⚠️ Nothing heard. Try again.")
            continue
        response = query_ollama(transcription)
        print("=" * 50)
        print(response)
        print("=" * 50)
        speak(response)

if __name__ == "__main__":
    main()