from pydantic import BaseModel, Field
from typing import List, Optional, Dict

class BaseResponse(BaseModel):
    """Base response model for API responses."""
    success: bool = Field(..., description="Indicates whether the operation was successful")
    error: Optional[str] = Field(None, description="Error message if the operation failed")


class GetTableOfContentsRequest(BaseModel):
    """Request model for retrieving table of contents for files."""
    user_id: int = Field(..., description="User ID")
    org_id: int = Field(..., description="Organization ID")
    file_ids: List[str] = Field(..., description="List of file IDs to get table of contents")


class GetTableOfContentsResponse(BaseResponse):
    """Response model for table of contents retrieval."""
    data: List[dict] = Field(..., description="List of table of contents data for the files")
