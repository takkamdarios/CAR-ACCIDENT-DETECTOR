import cv2
import numpy as np
import os
from tensorflow.keras.models import load_model

# Replace with the path to your trained model
model_path = 'outputs/trained_models/accident_detection_model.h5'

# Load the trained model
model = load_model(model_path)


# Function to preprocess the image (this should match the preprocessing used during training)
def preprocess_image(image_path, target_size=(224, 224)):
    # Read the image
    image = cv2.imread(image_path)
    # Convert the image to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Resize the image
    image = cv2.resize(image, target_size)
    # Normalize the image
    image = image / 255.0
    # Expand dimensions to match model's expected input
    image = np.expand_dims(image, axis=0)
    return image


# Function to predict if an image contains an accident
def predict_accident(image_path):
    processed_image = preprocess_image(image_path)
    prediction = model.predict(processed_image)
    # Assuming your model's final layer is a sigmoid activation, the output will be between 0 and 1
    # You can adjust the threshold as needed.
    accident_probability = prediction[0][0]
    return accident_probability > 0.5, accident_probability


# Example usage
if __name__ == '__main__':
    # Path to the directory containing images
    image_dir = 'D:\\ECOLEIT\\4IA\\Car Accident Detection Systeme\\dataset\\images'

    # Loop through the images in the directory
    for image_name in os.listdir(image_dir):
        image_path = os.path.join(image_dir, image_name)
        # Make a prediction
        is_accident, probability = predict_accident(image_path)

        print(f"Image: {image_name} - Is there an accident? {'Yes' if is_accident else 'No'}")
        print(f"Accident Probability: {probability:.2f}")
