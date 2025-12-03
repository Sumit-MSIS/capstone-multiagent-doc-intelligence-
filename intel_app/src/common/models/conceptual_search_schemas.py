from pydantic import BaseModel, Field
from typing import List, Optional, Dict

class ConceptualSearchFileRequest(BaseModel):
    """Request model for conceptual file search."""
    org_id: int = Field(..., ge=1, description="Organization ID")
    user_id: int = Field(..., ge=1, description="User ID")
    search_text: str = Field(..., description="Text to search for")
    search_id: int = Field(..., ge=1, description="Unique identifier for the search")
    # tag_ids: List[int] = Field(
    #     default_factory=list, description="List of tag IDs to filter the search"
    # )
    tag_ids: List[int] = Field(
        ..., description="List of tag IDs associated with the file"
    )

class BaseResponse(BaseModel):
    """Base response model for API responses."""
    success: bool = Field(..., description="Indicates whether the operation was successful")
    error: Optional[str] = Field(None, description="Error message if the operation failed")


class ConceptualSearchFileResponse(BaseResponse):
    """Response model for conceptual file search."""
    data: Dict[str, List[Dict[str, str]]] = Field(
        ..., description="Search results with matching file details"
    )