import cv2
import os
import numpy as np
import pandas as pd
from tqdm import tqdm

# Define the directory paths
input_image_dir = 'data/processed/images/'  # Adjust with your actual directory for processed images
input_video_dir = 'data/processed/videos/'  # Directory for processed videos
output_dir = 'data/features/'

# Create the output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Function to extract color histograms
def extract_color_histogram(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Compute the histogram for each color channel
    hist = [cv2.calcHist([image], [i], None, [256], [0, 256]).flatten() for i in range(3)]
    return np.concatenate(hist)

# Function to extract frames from a video
def extract_frames(video_path, frame_rate=1):
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

# Prepare a DataFrame to hold features
features = []

# Extract features from each image
print("Extracting features from images...")
for image_file in tqdm(os.listdir(input_image_dir)):
    if image_file.lower().endswith(('.png', '.jpg', '.jpeg')):  # Check for image files
        image_path = os.path.join(input_image_dir, image_file)
        image = cv2.imread(image_path)
        feature = extract_color_histogram(image)
        features.append(feature)

# Extract features from each video
print("Extracting features from videos...")
for video_file in tqdm(os.listdir(input_video_dir)):
    if video_file.lower().endswith(('.mp4', '.avi', '.mov')):  # Check for video files
        video_path = os.path.join(input_video_dir, video_file)
        frames = extract_frames(video_path, frame_rate=30)  # Adjust frame rate as needed
        
        for frame in frames:
            feature = extract_color_histogram(frame)
            features.append(feature)

# Convert to a DataFrame
features_df = pd.DataFrame(features)

# Save the features to a CSV file for later use
features_df.to_csv(os.path.join(output_dir, 'image_video_features.csv'), index=False)
