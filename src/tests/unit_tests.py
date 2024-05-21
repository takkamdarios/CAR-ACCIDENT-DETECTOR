import unittest
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from some_module import preprocess_image, predict  # Replace with your actual imports

# Path to a sample image and your model (for testing purposes)
model_path = 'outputs/models/accident_detector.h5'
sample_image_path = 'dataset/train/accidents/ezgif-frame-086_jpg.rf.15736ae6f748ca1a3bdaacdfe2f3224e.jpg'


class PreprocessingTestCase(unittest.TestCase):
    def test_preprocess_image(self):
        # Load an image using OpenCV
        image = cv2.imread(sample_image_path)
        
        # Test the preprocessing function
        processed_image = preprocess_image(image)
        
        # Check that preprocessing outputs the correct shape
        self.assertEqual(processed_image.shape, (224, 224, 3))  # Adjust expected shape as necessary

        # Check that pixel values are in the expected range (0-1 if normalized)
        self.assertTrue((processed_image >= 0).all() and (processed_image <= 1).all(),
                        "Preprocessed image pixels are not normalized.")


class ModelLoadingTestCase(unittest.TestCase):
    def test_load_model(self):
        # Test the model loading function
        model = load_model(model_path)
        
        # Check that model is loaded (model is not None)
        self.assertIsNotNone(model, "Failed to load model.")

        # Check that model has the expected architecture by checking the output shape
        self.assertEqual(model.output_shape, (None, 1))  # Adjust based on your model's output shape


class PredictionTestCase(unittest.TestCase):
    def test_predict(self):
        # Load an image and preprocess it
        image = cv2.imread(sample_image_path)
        processed_image = preprocess_image(image)
        
        # Load the model
        model = load_model(model_path)
        
        # Test the prediction function
        prediction = predict(processed_image, model)
        
        # Check that prediction returns a result
        self.assertIsNotNone(prediction, "Prediction function did not return a result.")
        
        # Check that prediction is in the expected format (e.g., list, np.array)
        self.assertIsInstance(prediction, np.ndarray, "Prediction is not an numpy array.")

        # Check the prediction shape is expected
        self.assertEqual(prediction.shape, (1,), "Prediction shape is not as expected.")


if __name__ == '__main__':
    unittest.main()
