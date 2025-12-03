sudo docker stop streamlit_app
sudo docker rm streamlit_app

sudo docker build -t streamlit_app:latest .

sudo docker run -d --name streamlit_app \
  --restart=always \
  -p 8501:8501 \
  -v /home/ubuntu/streamlit_app/data:/app/data \
  streamlit_app:latest 

sudo docker logs streamlit_app --tail=100 -f