import os
import traceback

print("CWD:", os.getcwd())
if os.path.exists("emotion_model.h5"):
    print("File exists. Size:", os.path.getsize("emotion_model.h5"))
else:
    print("File DOES NOT EXIST")

try:
    from tensorflow.keras.models import load_model
    print("Attempting to load model...")
    model = load_model("emotion_model.h5")
    print("SUCCESS: Model loaded.")
except Exception:
    traceback.print_exc()
