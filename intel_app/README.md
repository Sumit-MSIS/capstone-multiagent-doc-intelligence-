.
├── app/

│   ├── main.py     

│   ├── routers/                 
│   │   ├── compare_contract.py  
│   │   ├── get_answer.py        
│   │   ├── get_cc_answer.py     
│   │   ├── get_insights.py    

│   ├── services/                
│   │   ├── compare_contract/
│   │   ├── get_answer/        
│   │   ├── get_cc_answer/     
│   │   ├── insights/   

│   ├── utils/                
│   │   ├── retriever/

│   ├── models/                  
│   │   ├── schemas.py  

│   └── config.py   

├── Dockerfile    

├── requirements.txt    

├── README.md    
















/project
├── /app
│   ├── /api
│   │   ├── /v1
│   │   │   ├── __init__.py
│   │   │   ├── endpoints.py         # API routes and endpoints
│   │   │   ├── schemas.py          # Pydantic models for request/response
│   ├── /config
│   │   ├── __init__.py
│   │   ├── settings.py             # Configuration management
│   ├── /services
│   │   ├── /insights
│   │   │   ├── __init__.py
│   │   │   ├── insights_generator.py
│   │   │   ├── medical_insights_generator.py
│   │   ├── /intel_chat
│   │   │   ├── __init__.py
│   │   │   ├── streaming.py
│   │   ├── /contract
│   │   │   ├── __init__.py
│   │   │   ├── compare.py
│   │   │   ├── ai_contract.py
│   │   │   ├── search.py
│   │   │   ├── recurring_payment.py
│   │   ├── /file_handler
│   │   │   ├── __init__.py
│   │   │   ├── file_downloader.py
│   │   │   ├── page_counter.py
│   │   │   ├── doc_converter.py
│   │   ├── /mlflow
│   │   │   ├── __init__.py
│   │   │   ├── trace_manager.py
│   │   ├── __init__.py
│   ├── /utils
│   │   ├── __init__.py
│   │   ├── logger.py               # Centralized logging
│   │   ├── error_handler.py        # Centralized error handling
│   │   ├── mlflow_utils.py         # MLflow utilities
│   │   ├── file_utils.py          # File handling utilities
│   ├── /middleware
│   │   ├── __init__.py
│   │   ├── auth.py                # Authentication middleware
│   │   ├── rate_limiter.py        # Rate limiting middleware
│   ├── main.py                    # Main FastAPI app entry point
│   ├── dependencies.py            # Dependency injection
├── /tests
│   ├── __init__.py
│   ├── test_endpoints.py          # Unit tests for endpoints
│   ├── test_services.py           # Unit tests for services
├── /logs
│   ├── app.log                    # Application logs
├── Dockerfile                     # Docker configuration
├── requirements.txt               # Dependencies
├── .env                           # Environment variables
├── README.md                      # Project documentation


sudo docker stop dev_intel_app
sudo docker rm dev_intel_app

sudo docker build -t dev_intel_app:latest .
 
 
sudo docker run -d --name dev_intel_app \
  --restart=always \
  -p 9000:9000 \
  -p 9500:9500 \
  -e ENVIRONMENT=DEV \
  dev_intel_app:latest 

sudo docker logs dev_intel_app --tail=100 -f