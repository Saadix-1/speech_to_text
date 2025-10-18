import sounddevice as sd
import numpy as np
import librosa
import joblib
import pyttsx3
import matplotlib.pyplot as plt

sr_rate = 16000
clf = joblib.load("command_model.joblib")
engine = pyttsx3.init()
MIN_CONFIDENCE = 0.5

def record_until_enter():
    print("Press Enter to start speaking...")
    input()
    print("Recording... Press Enter to stop.")
    recording = []
    def callback(indata, frames, time, status):
        recording.append(indata.copy())
    with sd.InputStream(samplerate=sr_rate, channels=1, callback=callback):
        input()
    audio = np.concatenate(recording).flatten()
    plot_waveform(audio)
    return audio

def plot_waveform(audio):
    times = np.arange(len(audio)) / sr_rate
    plt.figure(figsize=(10, 3))
    plt.plot(times, audio)
    plt.title("Recorded Audio Waveform")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.show()

def extract_features(audio):
    mfccs = librosa.feature.mfcc(y=audio, sr=sr_rate, n_mfcc=13)
    mfccs_delta = librosa.feature.delta(mfccs)
    features = np.concatenate((np.mean(mfccs.T, axis=0), np.mean(mfccs_delta.T, axis=0)))
    return features

def speak(text):
    engine.say(text)
    engine.runAndWait()

# -----------------------------
# Real-time loop with confidence
# -----------------------------
while True:
    try:
        audio = record_until_enter()
        features = extract_features(audio)
        prediction = clf.predict([features])[0]

        # Get confidence score
        proba = clf.predict_proba([features])[0]
        class_index = list(clf.classes_).index(prediction)
        confidence = proba[class_index]
        if confidence >= MIN_CONFIDENCE:
            print(f"Recognized command: {prediction} (confidence: {confidence:.2f})")
            speak(prediction)
        else:
            print(f"No valid command detected. Confidence {confidence:.2f} too low.")

        print(f"Recognized command: {prediction} (confidence: {confidence:.2f})")
        speak(prediction)
    except KeyboardInterrupt:
        print("Exiting...")
        break
