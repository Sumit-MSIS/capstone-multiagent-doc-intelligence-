from pydantic import BaseModel, Field
from typing import List, Optional, Dict

# Request models
class FileTag(BaseModel):
    file_id: str
    tag_ids: List[int]


class UpdateTagsRequest(BaseModel):
    user_id: int
    org_id: int
    file_tag_list: List[FileTag]


# Response models
class FileTagUpdateResult(BaseModel):
    file_id: str
    status: str
    chunks: Optional[int] = None  # Present if updated


class UpdateTagsResponse(BaseModel):
    success: bool
    error: Optional[str] = None
    data: Optional[List[FileTagUpdateResult]] = None