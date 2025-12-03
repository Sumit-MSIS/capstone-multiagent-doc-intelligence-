from pydantic import BaseModel, Field, HttpUrl, validator
from typing import List, Optional

class RecurringPaymentDueRequest(BaseModel):
    """Request model for checking recurring payment due dates."""
    file_id: List[str] = Field(..., description="Unique identifier for the file")
    user_id: int = Field(..., ge=1, description="User ID")
    org_id: int = Field(..., ge=1, description="Organization ID")

class PaymentData(BaseModel):
    """Model for payment-related data."""
    is_recursive: bool = Field(..., description="Indicates if the payment is recurring")
    updated_payment_due_date: str = Field(
        ..., description="Updated payment due date as a string"
    )

class RecurringPaymentDueItem(BaseModel):
    """Model for an individual recurring payment due item."""
    file_id: str = Field(..., description="Unique identifier for the file")
    success: bool = Field(..., description="Indicates if the operation was successful")
    error: Optional[str] = Field(None, description="Error message if the operation failed")
    data: PaymentData = Field(..., description="Payment-related data")

class RecurringPaymentDueResponse(BaseModel):
    """Response model for recurring payment due checks."""
    root: List[RecurringPaymentDueItem] = Field(
        ..., description="List of recurring payment due items"
    )