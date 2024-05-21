import streamlit as st
import cv2
import numpy as np
import utils  # Import the utility functions
from utils import load_accident_model, preprocess_frame, extract_frames
import smtplib  # For sending emails
import os  # For accessing environment variables
import logging
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders

# Set up logging
logging.basicConfig(level=logging.INFO)

# Load the trained model
model = utils.load_accident_model()

st.set_page_config(page_title="Car Accident Detection", layout="wide")

# Custom CSS to style the webpage
st.markdown("""
    <style>
    .big-font {
        font-size:30px !important;
        color: #5A641F;  # Red color for predictions
        font-weight: bold;
    }
    .streamlit-header {
        font-size:40px !important;
        color: #008CBA;  # Light blue color for title
        font-weight: bold;
        text-align: center;
        background-color: #C59B5D;  # Background color for the header
    }
    .image-gallery {
        display: flex;
        justify-content: space-between;
    }
    .gallery-img {
        width: 150px;  # Smaller images for the gallery
        border-radius: 10px;
        transition: transform .2s;  # Animation effect for hover
        margin-right: 5px;
    }
    .gallery-img:hover {
        transform: scale(1.5);  # Enlarge image on hover
    }
    .file-uploader {
        background-color: #f0f0f0;  # Light grey background
        border: 2px dashed #ccc;  # Dashed border
        border-radius: 1px;
        transition: background-color .3s, border-color .3s;  # Smooth transition for color
    }
    .file-uploader:hover {
        background-color: #c8e6c9;  # Light green background on hover
        border-color: #4CAF50;  # Green border on hover
    }
    </style>
    """, unsafe_allow_html=True)

st.markdown('<p class="streamlit-header">Car Accident Detector</p>', unsafe_allow_html=True)

# Input fields for recipient email address
email_address = st.text_input("Enter email address to receive notifications")

# Get sender email credentials from environment variables
email_user = os.getenv('EMAIL_USER')
email_password = os.getenv('EMAIL_PASSWORD')

# Log email credentials retrieval
logging.info(f"Email user: {email_user}")
logging.info(f"Email password: {'****' if email_password else 'Not Set'}")

def send_email(sender, recipient, subject, body, attachment_path=None):
    try:
        msg = MIMEMultipart()
        msg['From'] = sender
        msg['To'] = recipient
        msg['Subject'] = subject

        msg.attach(MIMEText(body, 'plain'))

        if attachment_path:
            attachment = open(attachment_path, "rb")
            part = MIMEBase('application', 'octet-stream')
            part.set_payload((attachment).read())
            encoders.encode_base64(part)
            part.add_header('Content-Disposition', f"attachment; filename= {os.path.basename(attachment_path)}")
            msg.attach(part)

        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        logging.info("Starting email login...")
        logging.info(f"Email user: {sender}")
        logging.info(f"Email password: {email_password}")
        server.login(sender, email_password)
        logging.info("Email login successful.")
        text = msg.as_string()
        server.sendmail(sender, recipient, text)
        server.quit()
        logging.info("Email sent successfully.")
    except Exception as e:
        logging.error(f"Failed to send email: {e}")

col1, col2 = st.columns(2)

with col1:
    uploaded_file = st.file_uploader("Choose an image or video...", type=["jpg", "png", "mp4", "avi", "mov"], help="Drag and drop or click to browse files")
    
    if uploaded_file is not None:
        file_type = uploaded_file.type.split('/')[0]
        
        if file_type == 'image':
            image = utils.load_image(uploaded_file)
            st.image(image, caption='Uploaded Image', width=200)
            st.write("Classifying...")
            
            prediction = utils.predict_image(model, image)
            if prediction > 0.5:
                st.markdown('<p class="big-font">The image is predicted to be safe, no accident.</p>', unsafe_allow_html=True)
            else:
                st.markdown('<p class="big-font">The image is predicted to involve an accident.</p>', unsafe_allow_html=True)
                if email_address and email_user and email_password:
                    logging.info(f"Attempting to send email to {email_address}")
                    cv2.imwrite('accident_image.jpg', image)
                    send_email(email_user, email_address, "Accident Detection Alert", "Accident detected in the uploaded image.", 'accident_image.jpg')
                
        elif file_type == 'video':
            video_bytes = uploaded_file.read()
            video_path = "temp_video.mp4"
            
            with open(video_path, "wb") as f:
                f.write(video_bytes)
                
            st.video(video_bytes)
            st.write("Analyzing video...")
            
            frames = extract_frames(video_path, frame_rate=30)  # Adjust frame_rate as needed
            
            accident_detected = False
            for frame in frames:
                preprocessed_frame = preprocess_frame(frame)
                prediction = model.predict(preprocessed_frame)
                if prediction > 0.5:  # Assuming 0.5 is the threshold for accident detection
                    accident_detected = True
                    st.markdown('<p class="big-font">Accident detected in the video.</p>', unsafe_allow_html=True)
                    st.image(frame, caption='Accident Frame')
                    if email_address and email_user and email_password:
                        logging.info(f"Attempting to send email to {email_address} with attachment")
                        cv2.imwrite('accident_frame.jpg', frame)
                        send_email(email_user, email_address, "Accident Detection Alert", "Accident detected in the uploaded video.", 'accident_frame.jpg')
                    break
            
            if not accident_detected:
                st.markdown('<p class="big-font">No accident detected in the video.</p>', unsafe_allow_html=True)
                
with col2:
    st.markdown('## Example Images')
    st.markdown('<div class="image-gallery">', unsafe_allow_html=True)
    st.image('dataset/train/accidents/acc1 (3).jpg', caption='Accident Example', width=150)
    st.image('dataset/train/no_accidents/5_5.jpg', caption='No Accident Example', width=150)
    st.markdown('</div>', unsafe_allow_html=True)
