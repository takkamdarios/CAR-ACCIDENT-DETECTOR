# AI-Car-Accident-Detection

![image](https://github.com/takkamdarios/CAR-ACCIDENT-DETECTOR/assets/53516925/495e7e92-e06b-4f31-a195-64dde7abd296)

## Table of Contents

- [Car Accident Detection](#car-accident-detection)
  - [Overview](#overview)
  - [Table of Contents](#table-of-contents)
  - [Features](#features)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Screenshots](#screenshots)
  - [Contact](#contact)

 
## Features

- Detect accidents in images and videos
- Send email notifications with images of detected accidents
- Easy-to-use web interface built with Streamlit


## Prerequisites

- Docker installed on your system
- Gmail account for sending email notifications

## Installation AND EXECUTION

1. **Clone the Repository**

   ```bash
   git clone https://github.com/takkamdarios/CAR-ACCIDENT-DETECTOR.git
   cd CAR-ACCIDENT-DETECTOR


2. **Build the Docker Image**

   ```bash
   docker build -t car-accident-detector .



3. **Run the Docker Container**

   ```bash
   docker run -p 8501:8501 -e EMAIL_USER='your_email@gmail.com' -e EMAIL_PASSWORD='your_app_password' car-accident-detector



4. **Access the Streamlit App**
  Open your web browser and go to:
   ```bash
   http://localhost:8501



5. **Upload Images or Videos**
   
  - Drag and drop an image or video file into the uploader.
  - Enter the email address to receive notifications.


 
6. **Receive Email Notifications**

   - You will receive an email notification with the details and images of detected accidents.


# ScreenShots
![image](https://github.com/takkamdarios/CAR-ACCIDENT-DETECTOR/assets/53516925/2d73e36f-bc3b-474a-ae2f-50806ff6343c)

![image](https://github.com/takkamdarios/CAR-ACCIDENT-DETECTOR/assets/53516925/59216905-9768-4bbb-aedd-234260c846c6)


**ACCIDENT SCENARIO**
![image](https://github.com/takkamdarios/CAR-ACCIDENT-DETECTOR/assets/53516925/1d0ad8a1-fd27-4484-a536-552ba7532141)

![image](https://github.com/takkamdarios/CAR-ACCIDENT-DETECTOR/assets/53516925/eefc5abf-f307-48c9-9950-45afbf999ca0)

![image](https://github.com/takkamdarios/CAR-ACCIDENT-DETECTOR/assets/53516925/ce81beb3-2ef3-4c7d-aac1-6c242ea2323e)

![image](https://github.com/takkamdarios/CAR-ACCIDENT-DETECTOR/assets/53516925/9f891b3f-ad65-4e16-9395-f0eb59a1814b)


**NO ACCIDENT SCENARIO**
![image](https://github.com/takkamdarios/CAR-ACCIDENT-DETECTOR/assets/53516925/e49674ab-beff-4006-b064-c277e09f455e)


# Contact
**By**
- LOIQUE TAKAM
- HAROLD RAJA
  
   
