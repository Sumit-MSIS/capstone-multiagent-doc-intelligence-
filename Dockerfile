# Use the official Python 3.11 slim image as the base
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /app

# Install required system dependencies for OpenCV, PDF, OCR, etc.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    default-jre \
    procps \
    sqlite3 \
    && rm -rf /var/lib/apt/lists/*


# Copy requirements.txt first for better caching
COPY requirements.txt ./
COPY .env ./

# Install dependencies from requirements.txt in one step
RUN pip install -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

# Expose the port that FastAPI will run on
EXPOSE 8501

CMD ["streamlit", "run", "app.py"]
