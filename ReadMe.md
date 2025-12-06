sudo docker stop streamlit_app
sudo docker rm streamlit_app

sudo docker build -t streamlit_app:latest .

sudo docker run -d --name streamlit_app \
  --restart=always \
  -p 8501:8501 \
  -v /home/ubuntu/streamlit_app/data:/app/data \
  streamlit_app:latest 

sudo docker logs streamlit_app --tail=100 -f


📘 Project: Deep Thinker — Intelligent Document Analysis & Conversational AI

Deep Thinker is a Streamlit-based AI chatbot application designed to perform document ingestion, file storage, insights generation, and conversational querying.
The application interacts with:

Backend APIs (FastAPI-based)

MLFlow Tracking (for backend + insights logging)

AWS S3 (for file upload and retrieval)

MySQL Database (file metadata, user mapping, tags)

The Streamlit app itself resides on the main branch, while backend services run from separate branches using Docker.

🚀 Features
1. Document Upload System

Upload PDF or DOCX files

Files are stored in AWS S3

Metadata is also stored in a MySQL DB

A presigned URL is generated for backend processing

2. File Insights Pipeline

Each uploaded file triggers a backend Get Insights API

Backend extracts vectors, metadata, and stores embeddings

3. Chat Interface

Uses backend Chat API for answering queries

Supports chat with or without selected documents

Maintains per-session:

session_id

chat_id counter

history persistence

4. File Management

Delete files

Remove DB entries, vector store entries (via backend API)

Auto-refresh UI components

5. Session Management

New session creation

Persistent chat history

Local file-based JSON storage for offline history

📂 Folder Structure
project-root/
│
├── app.py                     # Streamlit UI application
├── requirements.txt           # Python dependencies
├── Dockerfile                 # Docker image for Streamlit app
│
├── data/                      # Local storage for chat + file metadata
│   ├── uploaded_files.json
│   ├── chat_history.json
│   └── session_data.json
│
├── README.md                  # YOU ARE HERE
│
└── .env                       # Environment variables (ignored in git)


Note: The backend and MLFlow code live on separate GitHub branches, explained below.

🌿 Branching Strategy (GitHub)

The project follows a multi-branch architecture:

1. main branch

Contains only:

app.py (Streamlit UI)

Dockerfile

requirements.txt

Documentation

This branch is exclusively for frontend/UI.

2. backend-fastapi branch

Contains:

FastAPI application

APIs:

/get-insights

/chat

/delete-files

Vector DB integration

DB ORM models

Dockerfile for backend

Logging + monitoring hooks connecting to MLflow

Runs via:

docker compose up --build

3. mlflow-tracking branch

Contains:

MLflow tracking configurations

Feature extraction pipelines

Model embeddings storage

Dockerfile for MLflow

Artifact store configuration (S3-compatible)

Runs via:

docker run mlflow:latest

🔗 How Streamlit App Connects to Backend & MLflow
                    ┌──────────────────┐
                    │   Streamlit App   │   (main branch)
                    └─────────┬────────┘
                              │ REST API CALLS
                              ▼
         ┌────────────────────────────────────────┐
         │           FastAPI Backend              │   (backend-fastapi branch)
         │ - Chat API                              │
         │ - Insights API                          │
         │ - Delete Files API                      │
         └──────────────┬─────────────────────────┘
                         │ MLflow logging
                         ▼
              ┌────────────────────────┐
              │       MLflow Server    │  (mlflow-tracking branch)
              └────────────────────────┘


📦 Requirements

Your requirements.txt:


requirements

streamlit
pandas
python-dotenv
pymysql
boto3


Additional recommended packages (optional):

requests
uuid

⚙ Environment Variables

Create a .env file in project root:

# Backend API URLs
GET_INSIGHTS_URL=http://backend:8000/get-insights
GET_ANSWER_URL=http://backend:8000/chat
DELETE_FILE_URL=http://backend:8000/delete-files

# AWS
AWS_REGION=us-east-1
AWS_ACCESS_KEY_ID=YOUR_KEY
AWS_SECRET_ACCESS_KEY=YOUR_SECRET

# Database
DB_HOST=host
DB_USER=user
DB_PASSWORD=password
DB_NAME=dbname

▶️ Running Locally (without Docker)
1. Create virtual environment
python3 -m venv venv
source venv/bin/activate

2. Install dependencies
pip install -r requirements.txt

3. Run Streamlit
streamlit run app.py

🐳 Running the Streamlit App with Docker

Your Dockerfile:
(loaded in your workspace but not shown—still supported here)

Build image
docker build -t deepthinker-streamlit .

Run container
docker run -p 8501:8501 --env-file .env deepthinker-streamlit


Open browser:

http://localhost:8501

🐳 Running Backend (other branch) Using Docker

Switch to backend branch:

git checkout backend-fastapi


Run backend:

docker compose up --build


This exposes your APIs at:

http://localhost:8000/get-insights
http://localhost:8000/chat
http://localhost:8000/delete-files

🐳 Running MLflow Server (other branch)

Switch branch:

git checkout mlflow-tracking


Run MLflow container:

docker run -p 5000:5000 mlflow-server


MLflow UI:

http://localhost:5000

🧪 Testing Workflow
1. Upload a PDF/DOCX → Goes to S3

Metadata stored in MySQL

Insights triggered via backend

2. Select file → Chat with context

Streamlit sends file IDs to backend

Backend retrieves vectors → generates answer

3. Delete file

Removes from DB

Removes from S3

Removes vectors via Delete API

Updates local JSON store

⚠ Troubleshooting
Streamlit says "Failed to initialize S3 client"

Check .env AWS credentials.

Chat API returns timeout

Backend may not be running or long-running query:

Increase backend timeout

Check Docker logs

File not uploaded

Check:

S3 bucket permission

IAM user role

File size limits

Database errors

Verify DB credentials and schema.

📄 License

Internal/Private (customize as needed)

🎯 Final Notes

Streamlit (UI) stays isolated on main branch

Backend + MLflow live independently

Communication happens only via API calls

Everything is containerized for easy deployment
