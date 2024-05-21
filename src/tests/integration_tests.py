import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Path to your model and images for testing
model_path = '/usr/src/app/outputs/models/accident_detector.h5'  # Updated to new model format
test_image_path = '/usr/src/app/data/test_images/'  # Adjust as needed

# Load the trained model
model = load_model(model_path)
logging.info("Model loaded successfully.")

# Define a function for preprocessing the images (this should match training)
def preprocess_image(image_path):
    try:
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Image not found at {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (224, 224))  # Assuming model uses 224x224 inputs
        image = image / 255.0
        image = np.expand_dims(image, axis=0)  # Add batch dimension
        return image
    except Exception as e:
        logging.error(f"Error processing image {image_path}: {str(e)}")
        return None

# Define a function for prediction
def predict(image):
    if image is not None:
        pred = model.predict(image)
        return pred
    else:
        return None

# Integration test
def test_pipeline():
    passed = 0
    failed = 0
    for image_file in os.listdir(test_image_path):
        image_path = os.path.join(test_image_path, image_file)
        image = preprocess_image(image_path)
        if image is None:
            logging.error(f"Skipping prediction for {image_file} due to preprocessing failure.")
            failed += 1
            continue
        pred = predict(image)
        if pred is None:
            logging.error(f"Prediction failed for {image_file}.")
            failed += 1
        else:
            # Here, you should include your logic for what constitutes a pass
            # For example, if you're expecting a binary classification:
            if pred[0][0] in [0, 1]:
                logging.info(f"Test passed for {image_file}: Prediction - {pred[0][0]}")
                passed += 1
            else:
                logging.error(f"Test failed for {image_file}: Prediction - {pred[0][0]}")
                failed += 1
    total = passed + failed
    logging.info(f"Integration Test Completed. Passed: {passed}/{total}, Failed: {failed}/{total}")

# Run the integration test
if __name__ == "__main__":
    test_pipeline()
