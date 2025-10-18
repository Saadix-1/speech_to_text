# app.py
import streamlit as st
import sounddevice as sd
import numpy as np
import librosa
from recognize import recognize  

SAMPLE_RATE = 16000  # same as your project
DURATION = 2

st.title("🎤 Voice Command Interface")

st.write("Click the button and speak your command!")

if st.button("Start Listening"):
    st.info("Recording...")
    audio = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='float32')
    sd.wait()
    st.success("Recording complete!")

    # Flatten audio
    audio = audio.flatten()

    # You can either modify your recognize() to accept audio array
    # or save audio to temp file and call recognize() on it
    result = recognize(audio)  # if recognize() supports passing audio directly
    st.write(f"✅ Recognized command: **{result}**")
