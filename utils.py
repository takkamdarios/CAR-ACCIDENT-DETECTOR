import cv2
import numpy as np
from tensorflow.keras.models import load_model as keras_load_model  # Renamed with alias

def load_accident_model():
    """Load and return the pre-trained model."""
    model_path = 'outputs/models/car_accident_detector.h5'  # Make sure the path is correct
    model = keras_load_model(model_path)  # Using the aliased function name
    return model

def load_image(uploaded_file):
    """Convert uploaded file to an image."""
    image = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(image, 1)
    return image

def preprocess_frame(frame, size=(224, 224)):
    """
    Preprocess a single frame for model prediction.
    
    Args:
    - frame (np.array): The frame to preprocess.
    - size (tuple): Target size of the frame.
    
    Returns:
    - preprocessed_frame (np.array): Preprocessed frame.
    """
    frame = cv2.resize(frame, size)
    frame = frame / 255.0
    frame = np.expand_dims(frame, axis=0)
    return frame

def predict_image(model, image):
    """Process and predict the class of the image."""
    image = cv2.resize(image, (224, 224))
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    prediction = model.predict(image)
    return prediction[0, 0]

def extract_frames(video_path, frame_rate=1):
    """
    Extracts frames from a video at a given frame rate.
    
    Args:
    - video_path (str): Path to the video file.
    - frame_rate (int): Number of frames to skip between each frame capture.
    
    Returns:
    - frames (list of np.array): List of frames extracted from the video.
    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % frame_rate == 0:
            frames.append(frame)
        frame_count += 1
    
    cap.release()
    return frames
