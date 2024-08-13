# Emotion Detection Application

## Team Members-

- Bhowmik Doshi (100891425)
- Dixit Patel (100893847)
- Meharan Shaikh (100896426)
- Sanchit Kalra (100901585)

## Overview

This is a web-based application that detects emotions from images using a pre-trained deep learning model. The application provides a simple user interface where users can upload an image, and the model will analyze the image to determine the emotion displayed by the person in the image. Additionally, there is a real-time emotion detection feature that uses the user's webcam to detect emotions in real time.

## Features

- **Image Upload**: Upload an image to detect emotions.
- **Real-Time Detection**: Detect emotions in real time using your webcam.
- **Responsive UI**: Modern and responsive user interface designed with a focus on simplicity and usability.
- **Emotion Analysis**: Displays the detected emotion along with the confidence percentage.

## Technologies Used

- **Flask**: A lightweight WSGI web application framework used to develop the backend of the application.
- **TensorFlow/Keras**: Used for loading the pre-trained deep learning model.
- **HTML/CSS**: Used for the frontend structure and styling.
- **JavaScript**: Used for handling file uploads and displaying the results.

## Installation

### Prerequisites

- Python 3.x
- Flask
- TensorFlow
- OpenCV
- A modern web browser

### Setup

1. **Clone the repository**:
    ```bash
    git clone https://github.com/your-username/emotion-detection-app.git
    cd emotion-detection-app
    ```

2. **Install the required Python packages**:
    ```bash
    pip install -r requirements.txt
    ```

3. **Download the pre-trained model**:
    - Place your pre-trained emotion detection model (e.g., `emotion_model.h5`) in the `models` directory.

4. **Run the Flask application**:
    ```bash
    python app.py
    ```

5. **Open the application**:
    - Open your web browser and navigate to `http://127.0.0.1:5000/`.

## Usage

### Image Upload

1. On the homepage, click on "Choose an Image" to upload an image from your device.
2. The application will analyze the image and display the detected emotion on the right side of the image.

### Real-Time Detection

1. Navigate to the "Real-Time Detection" page.
2. The application will use your webcam to detect emotions in real time.
3. The detected emotion will be displayed next to the webcam feed.
