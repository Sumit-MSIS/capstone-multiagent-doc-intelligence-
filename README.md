

# Data Science Dev Environment



## Folder Structure

- **`CompareContract/`**: Analyzes and generates reports comparing contracts.

- **`DeleteFiles/`**: Manages the safe deletion of unnecessary files.

- **`Documentation/`**: Official guides, API references, and user manuals.

- **`GenerateContract/`**: Automates contract creation based on user inputs and templates.

- **`GetAnswer/`**: Retrieves context-aware answers using NLP techniques.

- **`GetInsights/`**: Extracts insights from data for analysis and reporting.

- **`GetRiskMitigation/`**: Identifies and suggests risk mitigation strategies in contracts.

- **`SearchContracts/`**: Provides search capabilities across stored contracts.


# 🐳 How to Run Backend via Docker

### 1️⃣ Build the Docker Image

```sh
sudo docker stop dev_intel_app
sudo docker rm dev_intel_app
sudo docker build -t dev_intel_app:latest .
```

### 2️⃣ Run the Container

```sh
sudo docker run -d --name dev_intel_app \
  --restart=always \
  -p 9000:9000 \
  -p 9500:9500 \
  -e AWS_ACCESS_KEY_ID=VVVVVVVVV \
  -e AWS_SECRET_ACCESS_KEY=XXXXXXXXXXXxx \
  -e AWS_DEFAULT_REGION=us-east-1 \
  -e ENVIRONMENT=DEV \
  dev_intel_app:latest 
sudo docker logs dev_intel_app --tail=100 -f


```

MLflow UI will be accessible at:
👉 [[http://52.70.125.119:9000](http://52.70.125.119:9000/ )


---

## Usage

1. **Navigate to a Folder**: Choose the folder relevant to your task (e.g., `GenerateContract/`).
2. **Consult Documentation**: Follow instructions in the `Documentation/` folder.
3. **Run the Scripts**: Execute the necessary scripts with the appropriate setup.

## Contribution

Contributions are welcome! Follow the guidelines in the `Documentation/` folder.

## Contact

For support, reach out to the development team.


Commands to RUN intel_app

1) Open terminal in app directory
    pip install .

2) Start the app with below command in the terminal:
   app
