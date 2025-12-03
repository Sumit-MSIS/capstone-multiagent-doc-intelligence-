from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.config.base_config import config
from src.routers.common_routes import common_router
from src.routers.intel_routes import intel_router
import nltk
import os

os.environ["NLTK_DATA"] = "/usr/local/nltk_data"
nltk.data.path.append("/usr/local/nltk_data")

app = FastAPI(
    title=config.APP_NAME,
    description=config.APP_DESCRIPTION,
    version=config.APP_VERSION,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(common_router, prefix="/intel", tags=["Insights"])
app.include_router(intel_router, prefix="/intel", tags=["Insights"])
