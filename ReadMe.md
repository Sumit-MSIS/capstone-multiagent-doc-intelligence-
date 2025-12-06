

# 🚀 MLflow Tracing Application – README

## 📌 Overview

This repository hosts a **multi-branch, multi-service architecture** consisting of:

1. **Streamlit Frontend (Main Branch)**
   A lightweight UI allowing users to interact with the chatbot application.

2. **Backend API (Backend Branch)**
   A FastAPI-based microservice that powers the chatbot logic and connects with MLflow for experiment tracking.

3. **MLflow Tracking Server (MLflow Branch)**
   A fully containerized MLflow server with MySQL backend and S3 artifact storage used for tracing model runs, experiments, logs, and metrics.

The application is designed to provide **end-to-end observability**, allowing users to track every inference, input payload, and associated metadata through MLflow while interacting via Streamlit.

---

# 🏗️ Architecture Diagram

```
                       ┌─────────────────────────┐
                       │      Streamlit UI       │
                       │   (Main Branch Docker)  │
                       └─────────────┬───────────┘
                                     │ API Calls
                                     ▼
                     ┌────────────────────────────────┐
                     │          FastAPI Backend        │
                     │      (Backend Branch Docker)    │
                     └───────────────────┬─────────────┘
                                         │ MLflow Logging
                                         ▼
                      ┌─────────────────────────────────────┐
                      │         MLflow Tracking Server       │
                      │       (MLflow Branch Docker)         │
                      │  MySQL Backend + S3 Artifact Store   │
                      └──────────────────────────────────────┘
```

---

# 🌐 Hosted URLs

| Service                    | URL                                                                  |
| -------------------------- | -------------------------------------------------------------------- |
| **Streamlit App**          | [http://52.70.125.119:8501/](http://52.70.125.119:8501/)             |
| **FastAPI Backend**        | [http://52.70.125.119:9000/docs#/](http://52.70.125.119:9000/docs#/) |
| **MLflow Tracking Server** | [http://52.70.125.119:8000/](http://52.70.125.119:8000/)             |

---

# 📁 Repository Structure

This repository has **multiple branches**, each dedicated to a system component:

```
.
├── main                          # Streamlit App (Production UI)
│   ├── app.py
│   ├── pages/
│   ├── Dockerfile
│   ├── README.md
│   └── requirements.txt
│
├── backend                       # FastAPI Backend
│   ├── src/
│   ├── models/
│   ├── routers/
│   ├── Dockerfile
│   ├── requirements.txt
│   └── README.md
│
└── mlflow                        # MLflow Tracking Server
    ├── Dockerfile
    ├── start.sh
    ├── requirements.txt
    ├── env/.env
    └── README.md
```

### **How Branches Connect**

* **Streamlit → Backend**: REST API calls to FastAPI deployed on port `9000`.
* **Backend → MLflow**: Uses MLflow Python API to log metrics, params, and outputs.
* **MLflow Server**: Backed by

  * **MySQL** DB (tracking metadata)
  * **S3 bucket** (storing model artifacts)

---

# 🔧 MLflow Tracking Server (This Branch)

## 📌 Description

This MLflow deployment uses:

* **MySQL backend store** for experiments & run metadata
* **S3 bucket** for artifacts
* **Dockerized MLflow server** running on port **8000**

It enables:

* Full experiment tracking
* Model versioning
* Parameter + metrics logging
* Input/output traceability

---

# 📦 Folder Structure (MLflow Branch)

```
mlflow/
├── Dockerfile              # Builds MLflow server image
├── start.sh                # Entry script to run MLflow server
├── requirements.txt        # Python dependencies
└── README.md               # This documentation
```

---

# 📜 File Breakdown

### **requirements.txt**

(Loaded from your uploaded file)

Contains essential packages:

* `mlflow==3.1.4`
* `pymysql`
* `boto3`
* `gunicorn`
* `python-dotenv`

### **start.sh**

```bash
mlflow server \
  --backend-store-uri "mysql+pymysql://admin:Intel%4021@52.70.125.119:3306/mlflow_db_latest" \
  --artifacts-destination "s3://intel-mlflow" \
  --default-artifact-root "s3://intel-mlflow" \
  --host 0.0.0.0 \
  --port 8000
```

Ensures MLflow uses:

* **MySQL** at `52.70.125.119`
* **S3 bucket:** `intel-mlflow`
* **MLflow UI:** port `8000`

---

# 🐳 How to Run MLflow via Docker

### 1️⃣ Build the Docker Image

```sh
sudo docker stop mlflow-server 
sudo docker rm mlflow-server 
sudo docker build -t mlflow-server . 
```

### 2️⃣ Run the Container

```sh
sudo docker run -d
--restart=always
-p 8000:8000
-e AWS_ACCESS_KEY_ID=AKIA2PX4YZCUGBPNG
-e AWS_SECRET_ACCESS_KEY=RW4XRuoOi2Q5S4/j8yMMUBBoAPj6q0pj//U1
-e AWS_DEFAULT_REGION=us-east-1
--name mlflow-server
mlflow-server sudo docker logs mlflow-server --tail=100 -f


```

MLflow UI will be accessible at:
👉 [[http://52.70.125.119:8000](http://52.70.125.119:8000/ )


---

# 🔧 Running MLflow Without Docker (Local Mode)

### Step 1: Install dependencies

```sh
pip install -r requirements.txt
```

### Step 2: Run the MLflow server

```sh
bash start.sh
```

---

# 🔗 Integration With Other Branches

### ✔ Streamlit App Logs User Interactions

The UI collects:

* User input
* Model responses
* Execution timestamps

These are forwarded to the backend.

### ✔ Backend Logs Everything to MLflow

FastAPI backend logs:

* Parameters
* Metrics
* Artifacts
* Payload traces

This provides **full traceability for debugging, analytics, and performance evaluation**.

### ✔ MLflow Server Stores & Displays All Traces

Accessible at:
👉 [http://52.70.125.119:8000/](http://52.70.125.119:8000/)

---

# 🧪 Testing the Workflow End-to-End

1. **Open Streamlit:**
   [http://52.70.125.119:8501/](http://52.70.125.119:8501/)

2. Ask a question → request goes to FastAPI backend.

3. Backend triggers:

   * Inference
   * Logging → MLflow

4. Visit MLflow UI:
   [http://52.70.125.119:8000/](http://52.70.125.119:8000/)
   Check:

   * Run traces
   * Input/output artifacts
   * Metrics
   * Params

---

# 🚀 Production Deployment Notes

### ✔ Each branch has its **own Dockerfile**

You must:

* Switch to the respective branch
* Build the Docker container for that branch
* Deploy each container individually on your EC2 machine

### ✔ Recommended Naming Convention for Images

```
mlflow-tracker:latest
fastapi-backend:latest
streamlit-ui:latest
```

### ✔ Recommended Deployment Order

1. MLflow Server
2. Backend
3. Streamlit App

---

# 📌 Environment Variables (Optional)

Place these inside `.env`:

```
MLFLOW_S3_ENDPOINT_URL=https://s3.amazonaws.com
AWS_ACCESS_KEY_ID=<your_key>
AWS_SECRET_ACCESS_KEY=<your_secret>
MYSQL_USER=admin
MYSQL_PASSWORD=password
```

---

# 🛠️ Troubleshooting

### MLflow UI not loading?

* Ensure MySQL is reachable from container.
* Ensure S3 bucket exists & IAM permissions are correct.

### Backend not logging?

* Check MLflow tracking URI inside backend code:

  ```python
  mlflow.set_tracking_uri("http://52.70.125.119:8000")
  ```

### Streamlit errors?

* Verify API endpoint:

  ```python
  API_URL = "http://52.70.125.119:9000"
  ```

---

# 🎯 Final Notes

This README provides:

* Fully detailed **Pro-Tier documentation**
* Everything needed for your GitHub repository
* Complete explanation of **multi-branch architecture**
* Instructions for **running**, **deploying**, and **scaling**



