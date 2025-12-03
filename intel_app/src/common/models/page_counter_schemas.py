from pydantic import BaseModel, Field, HttpUrl, validator
from typing import List, Optional


class GetPageCountRequest(BaseModel):
    """Request model for retrieving the page count of a file."""
    file_type: str = Field(..., description="Type of the file")
    url: str = Field(..., description="URL for the file")
