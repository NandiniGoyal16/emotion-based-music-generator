from flask import Flask, render_template, request
import numpy as np
import cv2
from tensorflow.keras.models import load_model
import traceback

app = Flask(__name__)

# Load model
model = load_model("emotion_model.h5")

# Correct labels in alphabetical order (matching Keras flow_from_directory)
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

def preprocess_image(file):
    # Read image bytes → OpenCV
    file_bytes = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if img is None:
        raise ValueError("Invalid image")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5
    )

    if len(faces) == 0:
        raise ValueError("No face detected")

    # Largest face
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    face = gray[y:y+h, x:x+w]

    face = cv2.resize(face, (48, 48))
    face = cv2.equalizeHist(face)
    face = face / 255.0

    face = face.reshape(1, 48, 48, 1)
    return face

@app.route("/", methods=["GET", "POST"])
def index():
    emotion = None
    confidence = None
    error = None

    try:
        if request.method == "POST":
            if "image" not in request.files:
                raise ValueError("No file uploaded")

            file = request.files["image"]
            processed_image = preprocess_image(file)

            preds = model.predict(processed_image)
            # No filtering, use all classes
            idx = np.argmax(preds[0])
            emotion = emotion_labels[idx]
            confidence = round(float(np.max(preds[0])) * 100, 2)

    except Exception as e:
        print("ERROR:", e)
        traceback.print_exc()
        error = str(e)

    return render_template(
        "index.html",
        emotion=emotion,
        confidence=confidence,
        error=error
    )

if __name__ == "__main__":
    app.run(debug=True)