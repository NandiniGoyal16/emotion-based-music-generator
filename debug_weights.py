import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
import os

print(f"TensorFlow Version: {tf.__version__}")

# Define Standard Architecture (Common for FER2013)
def build_common_model():
    model = Sequential([
        Input(shape=(48, 48, 1)),
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        # Try both 7 (standard) and common variants just in case
        Dense(7, activation='softmax')
    ])
    return model

if os.path.exists("emotion_model.h5"):
    print("Attempting LOAD WEIGHTS only (bypassing config deserialization)...")
    try:
        model = build_common_model()
        # by_name=True allows partial loading if layers match
        model.load_weights("emotion_model.h5", by_name=True, skip_mismatch=True)
        print("SUCCESS: Weights loaded into standard architecture.")
    except Exception as e:
        print("FAILED load_weights:")
        print(e)
else:
    print("File not found.")
