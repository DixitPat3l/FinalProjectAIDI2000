from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import base64

app = Flask(__name__)
CORS(app)

# Load the pre-trained model
model = load_model('emotion_model_trained.h5')
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def detect_emotion(image):
    gray_frame = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=7, minSize=(100, 100),
                                          flags=cv2.CASCADE_SCALE_IMAGE)

    emotions_detected = []

    for (x, y, w, h) in faces:
        roi_gray = gray_frame[y:y + h, x:x + w]
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

        if np.sum([roi_gray]) != 0:
            roi = roi_gray.astype('float') / 255.0
            roi = np.expand_dims(roi, axis=0)
            roi = np.expand_dims(roi, axis=-1)

            prediction = model.predict(roi)[0]
            max_index = np.argmax(prediction)
            confidence = prediction[max_index] * 100
            emotion = emotion_labels[max_index]
            emotions_detected.append({
                "emotion": emotion,
                "confidence": f"{confidence:.2f}%",
                "box": {"x": int(x), "y": int(y), "w": int(w), "h": int(h)}
            })

    return emotions_detected

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/realtime')
def realtime():
    return render_template('realtime.html')

@app.route('/upload', methods=['POST'])
def upload():
    data = request.json
    if 'image' not in data:
        return jsonify({"error": "No image data provided"})

    image_data = data['image'].split(',')[1]
    image = np.fromstring(base64.b64decode(image_data), np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)

    emotions = detect_emotion(image)
    return jsonify(emotions)

if __name__ == '__main__':
    app.run(debug=True)
