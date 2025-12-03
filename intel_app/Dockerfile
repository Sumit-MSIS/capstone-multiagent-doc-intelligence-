# Use the official Python 3.11 slim image as the base
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /app

ARG ENVIRONMENT=DEV
ENV ENVIRONMENT=${ENVIRONMENT}

# Install required system dependencies for OpenCV, PDF, OCR, etc.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    default-jre \
    procps \
    poppler-utils \
    tesseract-ocr \
    sqlite3 \
    && rm -rf /var/lib/apt/lists/*

# RUN apt-get install -y unoconv

# Copy requirements.txt first for better caching
COPY requirements.txt ./


# Install dependencies from requirements.txt in one step
RUN pip install -r requirements.txt
RUN pip install fuzzywuzzy
RUN pip install agno==2.2.10

# Copy the rest of the application code into the container
COPY . .

# Copy the .env file
COPY .env ./

# Copy the downloaded NLTK data
# COPY nltk_data/ /usr/local/nltk_data

RUN python -m nltk.downloader \
    punkt punkt_tab stopwords averaged_perceptron_tagger averaged_perceptron_tagger_eng wordnet \
    -d /usr/local/nltk_data

# # Set the NLTK_DATA environment variable
ENV NLTK_DATA=/usr/local/nltk_data

# Expose the port that FastAPI will run on
EXPOSE 9000 9500

# Copy the run.sh script to the container and mark it as executable.

COPY start.sh /app/start.sh
RUN chmod +x /app/start.sh

# Use the run.sh script as the entrypoint.
# CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "9000", "--workers", "4", "--log-level", "debug"]
# Run both apps
CMD ["/app/start.sh"]