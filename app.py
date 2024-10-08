from flask import Flask, render_template, request, jsonify, Response
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
    confidence_threshold = 30.0  # Lower the confidence threshold

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

            if confidence > confidence_threshold:
                emotions_detected.append({
                    "emotion": emotion,
                    "confidence": f"{confidence:.2f}%",
                    "box": {"x": int(x), "y": int(y), "w": int(w), "h": int(h)}
                })

    return emotions_detected

def gen():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Mirror the frame for the webcam feed
        frame = cv2.flip(frame, 1)

        emotions = detect_emotion(frame)
        for emotion in emotions:
            x, y, w, h = emotion['box']['x'], emotion['box']['y'], emotion['box']['w'], emotion['box']['h']
            label = f"{emotion['emotion']} ({emotion['confidence']})"
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

        ret, jpeg = cv2.imencode('.jpg', frame)
        frame = jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/realtime')
def realtime():
    return render_template('realtime.html')

@app.route('/upload', methods=['POST'])
def upload():
    try:
        data = request.json
        if 'image' not in data:
            return jsonify({"error": "No image data provided"})

        image_data = data['image'].split(',')[1]
        image = np.frombuffer(base64.b64decode(image_data), np.uint8)
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)

        if image is None:
            return jsonify({"error": "Could not decode image"})

        emotions = detect_emotion(image)
        return jsonify({"emotions": emotions})
    except Exception as e:
        print(f"Error during image upload: {e}")
        return jsonify({"error": "An error occurred during processing"})

@app.route('/video_feed')
def video_feed():
    return Response(gen(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
