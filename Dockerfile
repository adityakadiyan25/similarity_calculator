# Use an official Python runtime as a parent image
FROM python:3.9.7-slim

# Set the working directory in the container
WORKDIR /usr/src/app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    python3-dev \
    openjdk-11-jre \
    && rm -rf /var/lib/apt/lists/*

# Copy only the requirements.txt first to leverage Docker layer caching
COPY requirements.txt ./

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Download necessary NLTK and Spacy packages
RUN python -m spacy download en_core_web_sm 

# Download NLTK data
RUN python -m nltk.downloader -d /usr/local/share/nltk_data stopwords
RUN python -m nltk.downloader -d /usr/local/share/nltk_data wordnet

# Copy your application code and data
COPY . .

# Expose the port the app runs on
EXPOSE 4001

# Create a non-root user and change ownership of the app files
RUN useradd -m appuser
RUN chown -R appuser:appuser /usr/src/app
USER appuser

# Run the application
CMD ["python", "./main.py"]
