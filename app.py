import streamlit as st
import sounddevice as sd
from recognize import recognize
from gtts import gTTS
import tempfile
import os
import numpy as np

SAMPLE_RATE = 16000
DURATION = 2

st.title("🎤 Voice Command Interface")
st.markdown("**Known commands:** yes, no, fuck, coming")
st.write("Click the button and speak your command!")

if st.button("Start Listening"):
    st.info("Recording...")
    audio = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='float32')
    sd.wait()
    audio = audio.flatten()
    st.success("Recording complete!")

    result = recognize(audio)
    if result:
        st.write(f"✅ Recognized command: **{result}**")

        # Text-to-speech via gTTS
        tts = gTTS(result)
        tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        tts.save(tmp_file.name)

        st.audio(tmp_file.name, format="audio/mp3")
        tmp_file.close()
        os.unlink(tmp_file.name)
    else:
        st.write("❌ Command not recognized.")
