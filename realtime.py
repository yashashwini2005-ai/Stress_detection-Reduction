import pickle
import numpy as np
import time
import pygame

# -------------------------------
# LOAD MODEL
# -------------------------------
with open("stress_model.pkl", "rb") as f:
    model = pickle.load(f)

print("Model loaded successfully")

# -------------------------------
# LOAD SAMPLE DATA
# -------------------------------
with open("S10/S10.pkl", "rb") as f:
    data = pickle.load(f, encoding="latin1")

eda = data['signal']['wrist']['EDA'].reshape(-1)

# Normalize (same as training)
eda = (eda - np.mean(eda)) / np.std(eda)

window_size = 20

# -------------------------------
# INIT AUDIO
# -------------------------------
pygame.mixer.init()

music_playing = False

def play_music():
    global music_playing
    if not music_playing:
        pygame.mixer.music.load("calm.mp3")
        pygame.mixer.music.play(-1)   # loop
        music_playing = True

def stop_music():
    global music_playing
    if music_playing:
        pygame.mixer.music.stop()
        music_playing = False

# -------------------------------
# REAL-TIME LOOP
# -------------------------------
print("\nStarting real-time stress detection...\n")

for i in range(0, len(eda) - window_size, window_size):

    sample = eda[i:i + window_size].reshape(1, -1)

    prediction = model.predict(sample)[0]

    if prediction == 2:
        print("⚠️ Stress detected → Playing music")
        play_music()

    elif prediction == 1:
        print("😌 Relaxed → Stop music")
        stop_music()

    else:
        print("🙂 Neutral")

    time.sleep(1)