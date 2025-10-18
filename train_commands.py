import sounddevice as sd
import librosa
import numpy as np
import json
import os
import pronouncing

COMMANDS = ["go", "stop", "no", "know"]
SAMPLE_RATE = 16000
DURATION = 2 
COMMANDS_FILE = "commands.json"
SAMPLES = 4

def record_audio():
    print(f"🎤 Speak now ({DURATION}s)...")
    audio = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='float32')
    sd.wait()
    return audio.flatten() 

def extract_features(audio):
    mfcc = librosa.feature.mfcc(y=audio, sr=SAMPLE_RATE, n_mfcc=13)
    return mfcc.astype(float).tolist() 

def get_phonemes(word): 
    phones = pronouncing.phones_for_word(word) 
    return phones[0] if phones else ""

def save_command(word, features):
    if os.path.exists(COMMANDS_FILE):
        with open(COMMANDS_FILE, "r") as f:
            commands = json.load(f)
    else:
        commands = {}

    if word not in commands:
        commands[word] = {"samples": []}
        commands[word]["samples"].append(features)

    with open(COMMANDS_FILE, "w") as f:
        json.dump(commands, f, indent=2)

def main():
    data = {}
    for cmd in COMMANDS:
        print(f"\n--- Recording command: {cmd} ---")
        phonemes = get_phonemes(cmd)
        data[cmd] = {"phonemes": phonemes, "samples": []}
        for i in range(SAMPLES):  
            filename = os.path.join(COMMANDS_FILE, f"{cmd}_{i}.wav")
            audio = record_audio()
            features = extract_features(audio)
            data[cmd]["samples"].append(features)
    with open("commands.json", "w") as f:
        json.dump(data, f, indent=2)
    print("\n✅ Training data saved to commands.json")

if __name__ == "__main__":
    main()


# import sounddevice as sd
# import numpy as np
# import librosa
# import matplotlib.pyplot as plt
# from sklearn.ensemble import RandomForestClassifier
# import joblib
# import os

# sr_rate = 16000
# COMMANDS = ["yes", "no", "stop", "go", "help", "left", "right", "up", "down"]
# TRAINING_DIR = "training_data"

# if not os.path.exists(TRAINING_DIR):
#     os.makedirs(TRAINING_DIR)

# def record(duration=2.0):
#     print(f"Recording for {duration} seconds...")
#     audio = sd.rec(int(duration*sr_rate), samplerate=sr_rate, channels=1, dtype='float32')
#     sd.wait()
#     audio = audio.flatten()
#     plot_waveform(audio)
#     return audio

# def plot_waveform(audio):
#     times = np.arange(len(audio)) / sr_rate
#     plt.figure(figsize=(10, 3))
#     plt.plot(times, audio)
#     plt.title("Recorded Audio Waveform")
#     plt.xlabel("Time (s)")
#     plt.ylabel("Amplitude")
#     plt.show()

# def extract_features(audio):
#     mfccs = librosa.feature.mfcc(y=audio, sr=sr_rate, n_mfcc=13)
#     mfccs_delta = librosa.feature.delta(mfccs)
#     features = np.concatenate((np.mean(mfccs.T, axis=0), np.mean(mfccs_delta.T, axis=0)))
#     return features

# X = []
# y = []

# # -----------------------------
# # Record multiple examples per command
# # -----------------------------
# for cmd in COMMANDS:
#     n_samples = int(input(f"How many recordings for '{cmd}'? "))
#     cmd_dir = os.path.join(TRAINING_DIR, cmd)
#     if not os.path.exists(cmd_dir):
#         os.makedirs(cmd_dir)
#     for i in range(n_samples):
#         input(f"Press Enter and say '{cmd}' ({i+1}/{n_samples})...")
#         audio = record(duration=2.0)
#         features = extract_features(audio)
#         X.append(features)
#         y.append(cmd)
#         # Save raw audio for future reference
#         np.save(os.path.join(cmd_dir, f"{i+1}.npy"), audio)

# # -----------------------------
# # Train a classifier
# # -----------------------------
# clf = RandomForestClassifier(n_estimators=100)
# clf.fit(X, y)

# # Save the trained model
# joblib.dump(clf, "command_model.joblib")
# print("Training finished and model saved!")
