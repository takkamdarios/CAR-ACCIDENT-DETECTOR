# Dockerfile

FROM python:3.9

# Install system libraries required by OpenCV
RUN apt-get update

# Set the working directory in the container
WORKDIR /usr/src/app

# Copy only the requirements.txt first to leverage Docker cache
COPY requirements.txt ./

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Install tensorflow separately if needed
RUN pip install tensorflow

# After ensuring dependencies are installed, copy the rest of the application
COPY . .

# Make port 8501 available to the world outside this container
EXPOSE 8501

# Define environment variables for email credentials
ENV EMAIL_USER takkamdarios@gmail.com
ENV EMAIL_PASSWORD hycd jnhw pwji xazp

# Run app.py using Streamlit when the container launches
CMD ["streamlit", "run", "app.py"]
