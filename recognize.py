# recognize_command.py
import json
import librosa
import numpy as np
import sounddevice as sd
from scipy.spatial.distance import cdist

SAMPLE_RATE = 16000 # speech recognition, 16,000 Hz
# 16,000 samples per second
THRESHOLD = 0.7  # ↑ Increase for stricter comparison (0.8–0.9 is very strict)
DURATION = 2
N_MFCC = 13

def record_audio():
    print("\n🎙️ Press ENTER to start recording...")
    input()
    print(f"🎤 Recording for {DURATION} seconds... Speak now!")
    audio = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='float32')
    sd.wait()
    print("✅ Recording complete.")
    return audio.flatten()

def extract_features(audio):
    mfcc = librosa.feature.mfcc(y=audio, sr=SAMPLE_RATE, n_mfcc=N_MFCC)
    return mfcc.T.astype(float)  # Transpose → (n_frames, n_mfcc)

def dtw_similarity(f1, f2):
    from librosa.sequence import dtw
    import numpy as np

    f1 = np.atleast_2d(f1)
    f2 = np.atleast_2d(f2)

    # Ensure same number of MFCC coefficients
    if f1.shape[1] != f2.shape[1]:
        raise ValueError(f"Feature dimension mismatch: f1={f1.shape}, f2={f2.shape}")

    D, wp = dtw(f1, f2, metric='cosine')
    sim = 1 - (D[-1, -1] / max(D.shape))
    return sim


def recognize(audio):
    with open("commands.json", "r") as f:
        commands = json.load(f)

    # audio = record_audio()
    features = extract_features(audio)

    best_match = None
    best_score = -1

    for cmd, data in commands.items():
        for sample in data["samples"]:
            sample_mfcc = np.array(sample, dtype=np.float64)  # shape: (13, n_frames_sample)
            sim = dtw_similarity(features, sample_mfcc)
        if sim > best_score and sim < 1.0:
            best_score = sim
            best_match = cmd


    if (best_score >= THRESHOLD and best_score < 1.00):
       print(f"✅ Recognized: {best_match} (Confidence: {best_score:.2f}).\nFeatures: {features.tolist()}\nSample: {sample}")
    else:
        print(f"❌ Unknown command (Confidence: {best_score:.2f})")
    return {best_match}

if __name__ == "__main__":
    print("🔊 Voice command recognizer initialized.")
    print("Press Ctrl+C to exit.\n")

    while True:
        try:
            recognize()
            print("---")
        except KeyboardInterrupt:
            print("\n👋 Exiting. Goodbye!")
            break
