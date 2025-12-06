


# 🚀 **Deep Thinker – Intelligent Document Analysis & Conversational AI**

<p align="center">
  <img src="https://via.placeholder.com/180x60?text=Deep+Thinker+Logo" alt="Deep Thinker Logo"/>
</p>

<p align="center">
  <b>A Streamlit-powered AI chatbot for document understanding, conversational intelligence, and automated insights.</b>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9+-blue.svg"/>
  <img src="https://img.shields.io/badge/Framework-Streamlit-red.svg"/>
  <img src="https://img.shields.io/badge/Backend-FastAPI-green.svg"/>
  <img src="https://img.shields.io/badge/MLFlow-Tracking-orange.svg"/>
  <img src="https://img.shields.io/badge/Cloud-AWS S3-yellow.svg"/>
  <img src="https://img.shields.io/badge/Status-Active-brightgreen.svg"/>
</p>

---

# 📚 **Table of Contents**

* [📘 Overview](#-overview)
* [✨ Features](#-features)
* [🏗 Architecture](#-architecture)
* [🌿 Branching Strategy](#-branching-strategy)
* [📂 Folder Structure](#-folder-structure)
* [⚙ Environment Variables](#-environment-variables)
* [🚀 Running Locally](#-running-locally)
* [🐳 Running with Docker](#-running-with-docker)
* [🔗 Backend & MLflow Setup](#-backend--mlflow-setup)
* [🧪 Workflow](#-workflow)
* [⚠ Troubleshooting](#-troubleshooting)
* [📄 License](#-license)

---

# 📘 **Overview**

**Deep Thinker** is a production-grade Streamlit application designed to:

* Upload PDF/DOCX documents
* Store files in AWS S3
* Trigger backend insight extraction
* Enable LLM-powered conversational querying
* Provide multi-session chat support
* Track model behavior using MLFlow
* Manage file metadata in MySQL

This application is optimized for enterprise document workflows, conversational search, and RAG-based AI systems.

---

# ✨ **Features**

### 🔹 **Document Upload**

* Upload PDFs and DOCX files
* Stored automatically in AWS S3
* Metadata saved in MySQL

### 🔹 **Insights Triggering**

Automatically triggers backend API to:

* Parse documents
* Generate embeddings
* Extract metadata
* Store vectors in vector DB

### 🔹 **Chat Interface**

* Rich two-way conversation
* Answers questions with or without selected documents
* Per-session chat history
* Beautiful UI with custom CSS

### 🔹 **File Manager**

* Select multiple documents
* Delete file (DB + S3 + vector DB removal)
* Smart autosync

### 🔹 **Logging & Tracing**

* Session tracking
* API call history
* MLflow pipeline logging (backend)

---

# 🏗 **Architecture**

```
                             ┌─────────────────────────┐
                             │     Streamlit UI        │
                             │   (main branch)         │
                             └───────────┬─────────────┘
                                         │ REST Calls
                                         ▼
                     ┌──────────────────────────────────────┐
                     │           FastAPI Backend             │
                     │      (backend-fastapi branch)         │
                     │ - Chat API                            │
                     │ - File insights API                   │
                     │ - Vector delete API                   │
                     └───────────┬──────────────────────────┘
                                 │
                                 ▼ Logging/Tracking
                    ┌────────────────────────────────────┐
                    │          MLflow Server              │
                    │     (mlflow-tracking branch)        │
                    └────────────────────────────────────┘

                ┌──────────┐     ┌─────────────┐     ┌──────────────┐
                │   MySQL   │     │   S3 Bucket │     │ Vector Store │
                └──────────┘     └─────────────┘     └──────────────┘
```

---

# 🌿 **Branching Strategy**

### **1️⃣ main branch**

Contains only:

* Streamlit UI (`app.py`)
* Dockerfile
* requirements.txt
* README

➡ Clean, isolated UI layer

---

### **2️⃣ backend-fastapi branch**

Contains:

* FastAPI application
* Vector DB integration
* MySQL repository logic
* MLflow logging hooks
* Docker setup

➡ All backend APIs used by your Streamlit app.

---

### **3️⃣ mlflow-tracking branch**

Contains:

* MLflow server configuration
* Pipelines & artifact storage
* Dockerfile for MLflow

➡ Tracks models, embeddings, pipeline performance.

---

# 📂 **Folder Structure**

```
project-root/
│
├── app.py
├── requirements.txt
├── Dockerfile
├── README.md
├── .env
│
└── data/
     ├── uploaded_files.json
     ├── chat_history.json
     └── session_data.json
```

---

# ⚙ **Environment Variables**

Create `.env` in project root:

```
# Backend API URLs
GET_INSIGHTS_URL=http://localhost:8000/get-insights
GET_ANSWER_URL=http://localhost:8000/chat
DELETE_FILE_URL=http://localhost:8000/delete-files

# AWS
AWS_REGION=us-east-1
AWS_ACCESS_KEY_ID=XXXX
AWS_SECRET_ACCESS_KEY=XXXX

# Database
DB_HOST=localhost
DB_USER=user
DB_PASSWORD=password
DB_NAME=mydb
```

---

# 🚀 **Running Locally (Due to dependancy it will not run correctly, i would recommend go with Docker route)**

### 1️⃣ Create virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### 2️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Run the Streamlit App

```bash
streamlit run app.py
```

---

# 🐳 **Running with Docker**

### Build the image

```bash
sudo docker stop streamlit_app
sudo docker rm streamlit_app
sudo docker build -t streamlit_app:latest .

```

### Run the container

```bash
docker run -p 8501:8501 --env-file .env deepthinker-ui
```

Access:

```
http://localhost:8501
```

---

# 🔗 **Backend & MLflow Setup**

### ▶ Backend (FastAPI)

Switch branch:

```bash
git checkout backend-fastapi
```

Run docker:

```bash
docker compose up --build
```

APIs exposed:

```
/get-insights
/chat
/delete-files
```

---

### ▶ MLflow Server

```bash
git checkout mlflow-tracking
docker run -p 5000:5000 mlflow-server
```

Open:

```
http://localhost:5000
```

---

# 🧪 **Workflow**

### ✔ Upload Document → stored in S3

### ✔ Backend triggered → insights generated

### ✔ Chat API → uses selected files

### ✔ Vector delete API → cleans embeddings

### ✔ Chat session maintained locally

---

# ⚠ **Troubleshooting**

| Issue           | Cause             | Fix                 |
| --------------- | ----------------- | ------------------- |
| S3 upload error | Wrong keys        | Update `.env`       |
| Chat timeout    | Backend down      | Restart FastAPI     |
| DB failure      | Wrong credentials | Verify MySQL env    |
| No response     | MLflow offline    | Start MLflow server |

---

# 📄 **License**

Private/Internal Use Only
(Replace with MIT/Apache if open-sourcing)

---


