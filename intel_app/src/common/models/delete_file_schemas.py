from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Union

class BaseResponse(BaseModel):
    """Base response model for API responses."""
    success: bool = Field(..., description="Indicates whether the operation was successful")
    error: Optional[str] = Field(None, description="Error message if the operation failed")



class DeleteFileRequest(BaseModel):
    """Request model for deleting files."""
    user_id: int = Field(..., description="User ID")
    org_id: int = Field(..., description="Organization ID")
    file_ids: List[str] = Field(..., description="List of file IDs to delete")


class DeleteFileResponse(BaseResponse):
    """Response model for file deletion."""
    data: Dict[str, Union[List[str], int, Dict[str, Dict[str, Union[int, str]]]]] = Field(
        ..., description="Details of deleted files or related metadata"
    )
