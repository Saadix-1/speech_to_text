

# 🗣️ InnovaSpeech – Speech-to-Text & Voice Command System

**Accessible AI Voice Recognition for People With Speech Impairments**

> Part of the GNG2501 Design Project – Prototype 1
> Team: **InnovaSpeech** (
---

## 📌 Overview

This project implements an **offline, lightweight, real-time speech-to-text and voice-command system** designed for accessibility.
It supports two prototypes:

### **Prototype 1 – Voice Transcription & Speech Feedback (Whisper-based)**

* Records audio from microphone
* Transcribes speech in real time using **Whisper (OpenAI)**
* Displays text in a clean Tkinter UI
* Can optionally repeat the recognized sentence using Text-to-Speech
* Fully offline after installation
* Optimized for devices without GPU

---

### **Prototype 2 – Voice Command Recognition (MFCC + DTW)**

* Recognizes custom commands (e.g., *yes*, *no*, *help*, *coming*)
* Stores user audio samples as MFCC features
* Compares new audio using **Dynamic Time Warping (DTW)**
* Runs in a **Streamlit web interface**
* Works even with atypical voices (important for Jody)
* Only stores features → **no raw audio stored** for privacy

---

## 🧠 How It Works

### 🎤 Audio Collection

When the user records a word:

* The waveform is captured as a list of floating-point samples
* MFCC features are extracted (summarizing vocal characteristics)
* Features are saved in JSON (e.g., `commands.json`)

### 🔍 Command Matching (DTW)

When a new audio command is spoken:

1. MFCC features are extracted
2. DTW compares them with saved samples
3. Similarity score ∈ [0,1]
4. The system returns the closest command if score > threshold

This allows recognition even when speech is slow, irregular, or impaired.

---

## 📁 Repository Structure

```
speech_to_text/
│
├── app.py                 # Streamlit UI (Prototype 2)
├── speech_to_text.py      # Prototype 1 - Whisper transcription app
├── recognize.py           # MFCC + DTW recognition logic
├── train_commands.py      # Training script for voice commands
├── commands.json          # Stored MFCC feature sets
├── command_model.joblib   # Optional model for classification
│
├── recordings/            # Training audio files
├── training_data/         # Extracted MFCC features
│
├── requirements.txt       # Python dependencies
└── README.md              # Project documentation
```

---

## 🚀 Installation

### 1. Clone the repository

```bash
git clone https://github.com/Saadix-1/speech_to_text.git
cd speech_to_text
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Run Prototype 1 – Whisper STT UI

```bash
python speech_to_text.py
```

---

## ▶️ Run Prototype 2 – Voice Command Recognition

```bash
streamlit run app.py
```

---

## 🧪 Training New Commands

You can train new words by running:

```bash
python train_commands.py
```

This will:

* Record your voice
* Extract MFCC
* Update `commands.json`

---

## 🎨 Current UI (Prototype 2)

![UI Example](./example_ui_screenshot.jpg)
*(Replace with your screenshot if needed)*

---

## 🔧 Technologies Used

* **Python 3.10+**
* **Whisper (OpenAI)**
* **Streamlit**
* **NumPy**
* **Librosa** (MFCC extraction)
* **DTW** (Dynamic Time Warping)
* **Tkinter** (Prototype 1 UI)
* **Joblib**

---

## 🎯 Project Goal

To support **accessible communication** for users with speech impairments by creating a system that:

* Learns *their* specific voice
* Works offline
* Is simple, fast, and reliable
* Integrates with assistive technologies

---

## 👥 Team – InnovaSpeech



---

## 📜 License

MIT License – use freely for educational purposes.

---

## 📞 Contact

For questions or collaboration:
📧 [saad.mehmdi@gmail.com](mailto:saad.mehmdi@gmail.com)
📍 University of Ottawa – Faculty of Engineering

---

If you want, I can also generate:
✅ A contributor guide
✅ A logo banner for your GitHub repo
✅ A nicer badges section (Python version, license, etc.)

Just tell me!
