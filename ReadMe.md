MLFLOW URL - http://52.70.125.119:8000/
sudo docker stop mlflow-server
sudo docker rm mlflow-server
sudo docker build -t mlflow-server .
sudo docker run -d \
    --restart=always \
    -p 8000:8000 \
    -e AWS_ACCESS_KEY_ID=AKIA2PX4YZCUGBPNGIUF \
    -e AWS_SECRET_ACCESS_KEY=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 \
    -e AWS_DEFAULT_REGION=us-east-1 \
    --name mlflow-server \
    mlflow-server
sudo docker logs mlflow-server --tail=100 -f
