from pydantic import BaseModel, Field, HttpUrl, validator
from typing import List, Optional
from enum import Enum


# Shared Enums and Base Classes
class FileType(str, Enum):
    """Enum defining allowed file types for processing."""
    html = "html"
    pptx = "pptx"
    eml = "eml"
    md = "md"
    msg = "msg"
    rst = "rst"
    rtf = "rtf"
    txt = "txt"
    xml = "xml"
    png = "png"
    jpg = "jpg"
    jpeg = "jpeg"
    tiff = "tiff"
    bmp = "bmp"
    heic = "heic"
    csv = "csv"
    doc = "doc"
    docx = "docx"
    epub = "epub"
    odt = "odt"
    tsv = "tsv"
    xlsx = "xlsx"
    pdf = "pdf"


class BaseResponse(BaseModel):
    """Base response model for API responses."""
    success: bool = Field(..., description="Indicates whether the operation was successful")
    error: Optional[str] = Field(None, description="Error message if the operation failed")


class InsightRequest(BaseModel):
    """Request model for generating insights from a file."""
    file_id: str = Field(..., description="Unique identifier for the file")
    file_name: str = Field(..., description="Name of the file")
    file_type: FileType = Field(..., description="Type of the file")
    user_id: int = Field(..., ge=1, description="User ID")
    org_id: int = Field(..., ge=1, description="Organization ID")
    url: str = Field(..., description="URL for the file")
    retry_no: int = Field(0, ge=0, description="Retry count")
    retry_process_id: List[int] = Field(
        default_factory=list, description="List of retry processes to be executed"
    )
    target_metadata_fields: List[str] = Field(
        default_factory=list,
        description='List of metadata fields to refresh, or ["*"] for all fields'
    )
    # tag_ids: List[int] = Field(
    #     default_factory=list, description="List of tag IDs associated with the file"
    # )
    tag_ids: List[int] = Field(
        ..., description="List of tag IDs associated with the file"
    )

    @validator('target_metadata_fields')
    def validate_metadata_fields(cls, value):
        if "*" in value and len(value) > 1:
            raise ValueError('If "*" is used, it must be the only item in target_metadata_fields')
        return value


class InsightResponse(BaseResponse):
    """Response model for file insights."""
    data: str = Field(..., description="Generated insights from the file")