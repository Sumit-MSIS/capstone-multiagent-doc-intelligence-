from pydantic import BaseModel, Field
from typing import List, Optional, Dict
from enum import Enum

class IntelChatRequestStream(BaseModel):
    """Request model for streaming intelligent chat interactions."""
    session_id: int = Field(..., description="The session ID for the chat")
    client_id: str = Field(..., description="Client ID for the request")
    parent_session_id: Optional[int] = Field(None, description="The parent session ID")
    question: str = Field(..., description="The question to answer")
    action: str = Field(..., description="The action to take")
    file_ids: List[str] = Field(..., description="List of file IDs")
    user_id: int = Field(..., ge=1, description="User ID")
    org_id: int = Field(..., ge=1, description="Organization ID")
    chat_id: int = Field(..., ge=1, description="Chat ID")
    connection_id: Optional[str] = Field(None, description="Connection ID for streaming")
    request_id: Optional[str] = Field(None, description="Request ID for tracking")
    enable_agent: Optional[bool] = Field(True, description="Enable agent processing")
    tag_ids: List[int] = Field(
        ..., description="List of tag IDs associated with the file"
    )


class IntelChatResponseStream(BaseModel):
    """Response model for streaming intelligent chat interactions."""

    success: bool = Field(..., description="Indicates whether the operation was successful")
    error: Optional[str] = Field("", description="Error message if the operation failed")
    data: str = Field(..., description="Streaming chat response data")

#***************************************************** INTEL CHAT Q&A STREAMING (End) ******************************************************

#***************************************************** Intel Q&A Tool Support (Start) ******************************************************
class Summary(BaseModel):
    """
    Summarize one or more contract documents based on the user's query.  
    Use when the user explicitly requests a **summary, overview, or key points** of contracts.

    Examples:
    - "Summarize the contract"
    - "Give me the key points in 5 bullets"
    - "Provide a short version of the agreement"
    """
    question: str = Field(..., description="The user's query specifying how the contract should be summarized")
    file_ids: List[str] = Field(..., description="List of file IDs for contract documents to summarize")


class Normal(BaseModel):
    """
    Provide accurate answers and explanations for general contract questions.  
    Use when the query involves **clauses, obligations, rights, warranties, terms, or comparisons** with contract files.  

    Examples:
    - "Explain the payment terms in Infosys contract"
    - "Compare confidentiality clauses between File A and File B"
    - "What warranties are included?"
    """
    question: str = Field(..., description="The user's query about the contract")
    file_ids: List[str] = Field(..., description="List of file IDs for contract documents to analyze")


class GeneralLLM(BaseModel):
    """
    Handle tasks outside strict contract lookup, including:  
    - Contract generation or drafting  
    - Rewriting clauses in plain English or legal language  
    - Data conversion (e.g., JSON → text, bullets → paragraph)  
    - Formatting for reports, templates, or emails  

    Examples:
    - "Draft an NDA between two companies"
    - "Convert this summary into legal terms"
    - "Format the output as a contract clause"
    """
    question: str = Field(..., description="The user's query for contract generation or data transformation")
    file_ids: List[str] = Field(default=[], description="List of file IDs, optional if not applicable")


class AnalyticalIntent(str, Enum):
    """Intent options for the Analytical tool"""
    SEARCH = "SEARCH"   # Retrieve metadata/details
    COUNT = "COUNT"     # Count matching contracts
    QA = "Q&A"          # Direct answers/explanations


class Analytical(BaseModel):
    """
    Extract structured metadata or provide direct Q&A about contract attributes.  
    Use when the user explicitly refers to **dates, values, parties, jurisdiction, type, or scope**.  

    Supported intents:
    - SEARCH → Retrieve files or filter/list by metadata
    - COUNT → Return number of contracts matching criteria
    - Q&A   → Direct explanations/answers from contract text

    Examples:
    - "What is the effective date of this agreement?" → Q&A  
    - "List all NDAs signed in 2023" → SEARCH  
    - "How many contracts expire this year?" → COUNT  
    - "Can you show me the details of the California purchase agreement?" → Q&A
    """

    intent: AnalyticalIntent = Field(
        default=AnalyticalIntent.SEARCH,
        description="Task intent type: SEARCH (metadata/details), COUNT (contract counts), Q&A (direct answers)"
    )
    question: str = Field(
        ...,
        description="The user's query for contract metadata or details"
    )
    file_ids: List[str] = Field(
        default_factory=list,
        description="List of file IDs for contract documents to analyze (may be empty if intent=COUNT)"
    )


class Websearch(BaseModel):
    """
    Retrieve real-time or external information beyond the contract vault.  
    Use when the query involves **URLs, news, companies, industries, or general facts**.  

    Examples:
    - "Latest amendments to Contract Act"
    - "Check news about Infosys and TCS agreements"
    - "Search for warranty rules in EU law"
    """
    question: str = Field(..., description="The user's query requiring real-time or external information")
    file_ids: List[str] = Field(default=[], description="List of file IDs, optional if not applicable")
