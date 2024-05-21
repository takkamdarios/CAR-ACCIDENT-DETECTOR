from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import json
import numpy as np
import os

# Path to your trained model and test data
model_path = 'outputs/models/accident_detector.h5'
test_data_dir = 'dataset/test'  # Adjust this if your test data is located elsewhere

# Load the model
model = load_model(model_path)

# Prepare the data
test_datagen = ImageDataGenerator(rescale=1./255)  # Rescale the images as per model training

test_generator = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=(224, 224),  # Use the same image size as you used during training
    batch_size=32,  # Adjust this according to your setup
    class_mode='binary',  # or 'categorical' if you have more than two classes
    shuffle=False
)

# Evaluate the model on the test data
eval_results = model.evaluate(test_generator, verbose=1)

# Get predictions
predictions = model.predict(test_generator)
y_pred = np.where(predictions > 0.5, 1, 0)  # Convert probabilities to binary predictions
y_true = test_generator.classes

# Generate a classification report
class_report = classification_report(y_true, y_pred, target_names=test_generator.class_indices)

# Calculate confusion matrix
conf_matrix = confusion_matrix(y_true, y_pred)

# Calculate ROC curve and AUC
fpr, tpr, thresholds = roc_curve(y_true, predictions)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")
plt.savefig('results/roc_curve.png')  # Save the ROC curve
plt.show()

# Print out the evaluation results
print(f"Evaluation results: {eval_results}")
print(f"Classification report: \n{class_report}")
print(f"Confusion matrix: \n{conf_matrix}")

# Optionally, save the results to a file
results_path = 'results/evaluation_metrics.json'
os.makedirs(os.path.dirname(results_path), exist_ok=True)

with open(results_path, 'w') as f:
    results = {
        'evaluation_results': eval_results,
        'classification_report': class_report,
        'confusion_matrix': conf_matrix.tolist(),  # Convert numpy array to list for JSON serialization
        'roc_curve': {'fpr': fpr.tolist(), 'tpr': tpr.tolist(), 'thresholds': thresholds.tolist()},
        'auc': roc_auc
    }
    json.dump(results, f, indent=4)

print(f"Saved evaluation results to {results_path}")
