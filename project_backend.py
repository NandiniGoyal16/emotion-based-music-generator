import numpy as np
import random
import pandas as pd
import os
import cv2
import time

# Attempt Keras Import
try:
    import tensorflow as tf
    from tensorflow.keras.models import model_from_json, Sequential
    from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input, InputLayer
    KERAS_AVAILABLE = True
except ImportError:
    KERAS_AVAILABLE = False
    print("⚠️ TensorFlow/Keras not found. CNN will run in 'Simulated' mode.")

# Constants - FROM EXTERNAL REPO
EMOTIONS = ["angry", "disgusted", "fearful", "happy", "neutral", "sad", "surprised", "romantic", "calm"]
# Repo Dict matches: 0:Angry, 1:Disgusted, 2:Fearful, 3:Happy, 4:Neutral, 5:Sad, 6:Surprised

# Instruments
INSTRUMENTS = ["sarod", "santoor", "tabla", "harmonium", "veena", "flute", "sitar"]

SWARAS = {
    "Sa": 240, "Re": 270, "Ga": 300,
    "Ma": 320, "Pa": 360, "Dha": 400, "Ni": 450
}

RAGAS = {
    "sad": ["Sa", "Ga", "Ma", "Pa"],
    "happy": ["Sa", "Re", "Ga", "Ma", "Pa", "Ni"],
    "calm": ["Sa", "Re", "Ma", "Pa"],
    "neutral": ["Sa", "Re", "Ma", "Pa"],
    "angry": ["Sa", "Ga", "Dha", "Ni"],
    "disgusted": ["Sa", "Ga", "Dha", "Ni"],
    "fearful": ["Sa", "Re", "Ga", "Pa"],
    "surprised": ["Sa", "Re", "Ga", "Ma", "Pa", "Ni"],
    "romantic": ["Sa", "Re", "Ga", "Pa", "Ni"]
}

# ============================================================
# 📊 DATA HANDLER
# ============================================================
class DataHandler:
    def __init__(self, csv_path):
        self.csv_path = csv_path
        self.data = None
        self.instrument_stats = {}
    
    def load_data(self):
        if not os.path.exists(self.csv_path):
            print(f"❌ CSV not found at {self.csv_path}")
            return False
        
        try:
            self.data = pd.read_csv(self.csv_path)
            print(f"✅ Loaded Dataset: {len(self.data)} rows")
            
            # Normalize instrument labels to lowercase for matching
            if 'instrument_label' in self.data.columns:
                self.data['instrument_label'] = self.data['instrument_label'].str.lower()
            
            # Use 'tempo' column if available
            if 'tempo' in self.data.columns:
                grouped = self.data.groupby('instrument_label')['tempo'].mean()
                self.instrument_stats = grouped.to_dict()
            
            return True
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return False

    def get_instrument_features(self, instrument):
        """Returns avg tempo and other features for the instrument."""
        inst_key = instrument.lower()
        stats = {'tempo': 90.0, 'spectral_centroid_mean': 1500.0}
        
        if inst_key in self.instrument_stats:
            stats['tempo'] = self.instrument_stats[inst_key]
            
        if self.data is not None and inst_key in self.data['instrument_label'].values:
            subset = self.data[self.data['instrument_label'] == inst_key]
            if 'spectral_centroid_mean' in subset.columns:
                stats['spectral_centroid_mean'] = subset['spectral_centroid_mean'].mean()
        
        return stats

# ============================================================
# 🧠 CNN EMOTION MODEL (External Repo Integration)
# ============================================================
class EmotionCNN:
    def __init__(self):
        self.model = None
        self.is_trained = False
        # External Repo uses JSON + Weights
        self.json_path = "Emotion-Detection-CNN/emotion_model.json"
        self.weights_path = "Emotion-Detection-CNN/emotion_model.h5"
        
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        self._load_from_json()
    
    def _load_from_json(self):
        if not KERAS_AVAILABLE: return

        if os.path.exists(self.json_path) and os.path.exists(self.weights_path):
            try:
                print(f"📥 Loading Model JSON: {self.json_path}")
                with open(self.json_path, 'r') as json_file:
                    loaded_model_json = json_file.read()
                
                custom_objects = {
                    "Sequential": Sequential,
                    "InputLayer": InputLayer,
                    "Conv2D": Conv2D,
                    "MaxPooling2D": MaxPooling2D,
                    "Flatten": Flatten,
                    "Dense": Dense,
                    "Dropout": Dropout
                }
                self.model = model_from_json(loaded_model_json, custom_objects=custom_objects)
                
                print(f"📥 Loading Model Weights: {self.weights_path}")
                self.model.load_weights(self.weights_path)
                
                self.is_trained = True
                self.load_error = None
                print("✅ Model loaded successfully (JSON + Weights)!")
                return
            except Exception as e:
                print(f"⚠️ Failed to load model: {e}")
                self.load_error = str(e)
        else:
            self.load_error = "Missing emotion_model.json or emotion_model.h5"
            print(self.load_error)

        # Fallback Mock
        self.model = Sequential() 

    def predict(self, image_input):
        if not KERAS_AVAILABLE: return "Error: Keras Not Found"
        if not self.is_trained: return f"Error: {getattr(self, 'load_error', 'Model Not Loaded')}"
        
        try:
            # 1. Load/Convert to Grayscale
            img = image_input
            if isinstance(image_input, str): img = cv2.imread(image_input)
            
            if len(img.shape) == 3:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                gray = img

            # 2. Preprocessing (Exact Match to Repo Logic)
            # The repo finds faces in video stream. For static image, we must find the face first.
            if gray.shape[0] > 100: # Only detect if image is large
                faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
                if len(faces) > 0:
                    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
                    gray = gray[y:y+h, x:x+w]
            
            roi_gray = cv2.resize(gray, (48, 48))
            
            # Repo: cropped_img = np.expand_dims(np.expand_dims(..., -1), 0)
            # CRITICAL: NO division by 255.0 here because the Repo's logic (website.py) doesn't do it!
            # It passes raw 0-255 values to predict
            cropped_img = np.expand_dims(np.expand_dims(roi_gray, -1), 0)
            
            # Predict
            preds = self.model.predict(cropped_img, verbose=0)
            idx = np.argmax(preds[0])
            conf = np.max(preds[0])
            
            print(f"🧠 Prediction: {EMOTIONS[idx]} ({conf:.2f})")
            return EMOTIONS[idx]
            
        except Exception as e:
            print(f"Error in prediction: {e}")
            return "neutral"

# ============================================================
# 🤖 TRPO AGENT
# ============================================================
class TRPOAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.policy = np.ones((state_size, action_size)) / action_size
    
    def train_from_data(self, instrument_features, emotion, episodes=50):
        target_tempo = instrument_features.get('tempo', 90)
        target_brightness = instrument_features.get('spectral_centroid_mean', 1500)
        prefer_high = target_brightness > 2000
        
        raga_notes = RAGAS.get(emotion, RAGAS.get('neutral')) # Default to Neutral
        
        notes_list = list(SWARAS.keys())
        
        for _ in range(episodes):
            for s in range(self.state_size):
                for a in range(self.action_size):
                    note_name = notes_list[a]
                    r_harmonic = 1.0 if note_name in raga_notes else -1.0
                    
                    note_idx = a
                    r_timbre = 0
                    if prefer_high and note_idx > 3: r_timbre = 0.5
                    if not prefer_high and note_idx < 4: r_timbre = 0.5
                    
                    reward = r_harmonic + r_timbre
                    self.policy[s, a] += 0.01 * reward
                
                self.policy[s] = np.maximum(self.policy[s], 0)
                if np.sum(self.policy[s]) > 0:
                    self.policy[s] /= np.sum(self.policy[s])

    def select_action(self, state_idx):
        probs = self.policy[state_idx]
        return np.random.choice(self.action_size, p=probs)

# ============================================================
# 🎹 GENERATION SESSION
# ============================================================
def synthesize_note_v2(freq, duration, instrument, sr=22050):
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    audio = np.zeros_like(t)
    instrument = instrument.lower()

    if instrument in ["flute", "bansuri"]:
        audio += 1.0 * np.sin(2 * np.pi * freq * t)
        audio += 0.3 * np.sin(2 * np.pi * 2 * freq * t)
        envelope = np.minimum(t * 10, 1) * np.exp(-2 * t)
    elif instrument in ["sitar", "veena", "guitar"]:
        for n in range(1, 8):
            audio += (0.6 / n) * np.sin(2 * np.pi * n * freq * t)
        envelope = np.exp(-5 * t)
    elif instrument in ["tabla", "drums"]:
        f_sweep = np.linspace(freq, freq*0.8, len(t))
        audio = np.sin(2 * np.pi * f_sweep * t)
        envelope = np.exp(-8 * t)
    elif instrument == "harmonium":
        for n in range(1, 8):
            if n%2!=0: audio += (0.5/n) * np.sin(2 * np.pi * n * freq * t)
        envelope = np.ones_like(t) * 0.9
    else:
        audio = np.sin(2 * np.pi * freq * t)
        envelope = np.exp(-3 * t)

    return audio * envelope

def generate_session(emotion, instrument, data_handler=None, duration=10):
    features = {'tempo': 90}
    if data_handler:
        features = data_handler.get_instrument_features(instrument)
        print(f"📊 Features: {features}")

    agent = TRPOAgent(state_size=7, action_size=7)
    agent.train_from_data(features, emotion)
    
    target_tempo = features['tempo']
    # Adjust for emotion
    if emotion in ['sad', 'fear']: target_tempo *= 0.8
    if emotion in ['happy', 'surprise', 'angry']: target_tempo *= 1.2
    if emotion in ['romantic']: target_tempo *= 0.9
    if emotion in ['calm']: target_tempo *= 0.7
    
    audio_full = []
    sr = 22050
    beat_step = 60.0 / target_tempo
    
    notes_list = list(SWARAS.keys())
    current_idx = 0
    total_time = 0
    
    while total_time < duration:
        next_idx = agent.select_action(current_idx)
        note_name = notes_list[next_idx]
        freq = SWARAS[note_name]
        
        dur = beat_step * random.choice([0.5, 1.0, 2.0])
        wave = synthesize_note_v2(freq, dur, instrument, sr)
        audio_full.append(wave)
        total_time += dur
        current_idx = next_idx

    final_audio = np.concatenate(audio_full)
    final_audio = final_audio / (np.max(np.abs(final_audio)) + 1e-9)
    
    return final_audio, sr, agent.policy

print("✅ project_backend loaded successfully")
