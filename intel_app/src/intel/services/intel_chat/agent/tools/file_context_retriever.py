import os
import asyncio
import mlflow
from agno.tools import tool
from agno.agent import Agent
from agno.models.openai import OpenAIChat
from textwrap import dedent
from typing import List, Dict, Any
from src.intel.services.intel_chat.agent.utils import get_metadata
from src.intel.services.intel_chat.agent.pinecone_retriever.hybrid_search_retrieval import get_embeddings
from src.intel.services.intel_chat.agent.query_rephraser import rephrase_search_text
from src.intel.services.intel_chat.agent.pinecone_retriever.hybrid_search_retrieval import get_context_from_pinecone
from openai import AsyncOpenAI
import httpx
mlflow.openai.autolog()

# -------------------------------------------------------------
# Helper Functions
# -------------------------------------------------------------

# async_client = AsyncOpenAI(
#     api_key=os.getenv("OPENAI_API_KEY"),
#     http_client=httpx.AsyncClient(
#         limits=httpx.Limits(
#             max_keepalive_connections=100,  # Increase from default 10
#             max_connections=200,            # Increase from default 100
#             keepalive_expiry=30
#         )
#     )
# )
async_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
def escape_markdown_filename(file_name: str) -> str:
    """Escapes square brackets for Markdown-safe filenames."""
    return file_name.replace("[", r"\[").replace("]", r"\]")


async def call_llm(original_user_query:str, user_query: str, context: str) -> str:
    """Calls an LLM to identify and extract the most relevant content from the provided context."""


    system_prompt = dedent("""
    You are a senior legal domain expert and legal information extraction assistant.
                           You are a senior legal expert. Your responsibility is to examine the material inside the <document_context> 
    and identify only those portions that are relevant to the <user_query>. 
    Relevance may come from two types of content:  
    (1) metadata fields in the document, and  
    (2) the actual legal clauses, sections, paragraphs, or defined terms within the document body.

    ------------------------------------------------------------
    PRIORITY RULES
    ------------------------------------------------------------

    1. If the user's query refers to information that normally appears in metadata 
    (for example, parties, dates, title of the agreement, jurisdiction, contract type, signatures, or document category), 
    you must extract the corresponding metadata before anything else.

    2. If the query asks about legal obligations, rights, terms, conditions, restrictions, definitions, penalties, 
    responsibilities, confidentiality, employment rules, termination, or any substantive provision, 
    then you must extract the full clause or section that contains that information.  
    The extracted text must be complete and presented exactly as written in the document.

    3. If the question requires both metadata and substantive clauses, 
    you must return the metadata first, followed by the relevant clauses 
    in the same sequence in which they appear within the document.

    ------------------------------------------------------------
    EXTRACTION RULES
    ------------------------------------------------------------

    • You must extract the entire clause, section, or paragraph.  
    Do not remove sentences or edit the language in any way.

    • You must preserve all numbering, headings, formatting, and original language from the document.

    • Do not summarize, paraphrase, explain, interpret, or comment on the text.

    • Each extracted clause must be self-contained and understandable on its own.  
    If a clause spans multiple paragraphs, include all related paragraphs.

    • If multiple clauses are relevant, present each one as a separate block in the same order 
    in which they appear in the original document.

    • Include nothing except the extracted text.

    ------------------------------------------------------------
    NO MATCH RULE
    ------------------------------------------------------------

    If no portion of the metadata or the document text is relevant to the user's query, 
    you must return exactly the following sentence:

    "No relevant context found."

    ------------------------------------------------------------
    OUTPUT REQUIREMENT
    ------------------------------------------------------------

    Your final output must contain only the extracted metadata fields 
    and/or the extracted clauses.  
    Do not include any commentary or explanation.
    """)

    prompt = dedent(f"""
    {context}

    <user_query>
    {original_user_query} 
    </user_query>

    Please extract only the metadata and the complete legal clauses that directly relate to the user's query.
    Return only the extracted text and nothing else.
    """)

    response = await async_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": system_prompt},
                   {"role": "user", "content": prompt}],
        temperature=0,
        top_p=1,
    )

    return response.choices[0].message.content


@mlflow.trace(name="Get Context From Single File")
async def get_context_from_single_file(
    file_id: str,
    user_query:str, # temporary for context extractor llm call
    keyword_search_query: str,
    semantic_search_query: str,
    org_id: int,
    tag_ids: List[str],
    metadata_dict,
    dense_vectors
) -> str:
    """Retrieve structured XML context for a single file from Pinecone (or other vector store)."""
    try:
        filters = {"file_id": {"$eq": file_id}}
        top_k = 10


        data = await get_context_from_pinecone(
            os.getenv("DOCUMENT_SUMMARY_AND_CHUNK_INDEX"),
            os.getenv("PINECONE_API_KEY"),
            filters,
            top_k,
            semantic_search_query,
            [file_id],
            org_id,
            tag_ids,
            keyword_search_query,
            dense_vectors,
            0.30
        )

        # Handle empty results
        if not data or "matches" not in data:
            return dedent(f"""
                <file>
                    <file_name>{file_id}</file_name>
                    <file_content>No relevant context found.</file_content>
                </file>
            """).strip()

        # Extract matched texts
        matched_texts = [
            match.get("metadata", {}).get("text", "")
            for match in data.get("matches", [])
            if match.get("metadata", {}).get("text")
        ]
        context = "\n\n---\n\n".join(matched_texts) if matched_texts else "No relevant context found."

        # Metadata
        file_metadata_entry = metadata_dict.get(file_id, {})
        file_metadata = file_metadata_entry.get("metadata", {})
        file_name = escape_markdown_filename(file_metadata_entry.get("file_name") or file_id)

        llm_call_context = dedent(f"""
        <document_context>
            <document_name>{file_name}</document_name>

            <document_metadata>
            {file_metadata}
            </document_metadata>

            <document_content>
            {context}
            </document_content>

        </document_context>
        """)

        return {
            "file_id": file_id,
            "file_name": file_name,
            "file_metadata": file_metadata,
            "llm_context": llm_call_context,
            "user_query":user_query,
            "semantic_search_query": semantic_search_query
        }
        # relevant_content = await call_llm(semantic_search_query, llm_call_context)

        # structured_context = dedent(f"""
        # <document file_id="{file_id}">
        #     <document_name>{file_name}</document_name>
        #     <document_link>[{file_name}](/preview?fileId={file_id})</document_link>

        #     <document_metadata>
        #         {file_metadata}
        #     </document_metadata>

        #     <document_content>
        #         {relevant_content}
        #     </document_content>
        # </document>
        # """)
        # return structured_context

    except Exception as e:
        return dedent(f"""
            <file_error>Failed to retrieve context for file {file_id}: {str(e)}</file_error>
        """).strip()
    


@tool(name="get_relevant_context_from_files")
@mlflow.trace(name="Get Relevant Context From Files Tool")
async def get_relevant_context_from_files(user_query: str, dependencies) -> str:
    """
    Retrieve relevant chunks context from given files based on user query.

    Args:
        user_query (str): The query from the leader.
        session_state (Dict): session state.
        agent (Agent): The agent instance containing dependencies.

    Returns:
        str: XML-formatted context string.
    """
    file_ids = dependencies.get("file_ids")

    if not file_ids:
        return "File GUIDs list is empty."
    
    org_id = dependencies.get("org_id")
    tag_ids = dependencies.get("tag_ids")

    metadata_dict = await get_metadata(file_ids)

    keyword_search_query, semantic_search_query, reranking_query = await rephrase_search_text(user_query)

    dense_vectors = await get_embeddings(org_id, file_ids, semantic_search_query)

    if not dense_vectors:
        dense_vectors = {}

    tasks = [
        get_context_from_single_file(fid, user_query, keyword_search_query, semantic_search_query, org_id, tag_ids, metadata_dict, dense_vectors)
        for fid in file_ids
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Collect contexts
    context_strings = []
    for res in results:
        if isinstance(res, Exception):
            context_strings.append(dedent(f"<file_error>{str(res)}</file_error>"))
        else:
            context_strings.append(res)

    # Final wrapped XML context for multi-file LLM input

    full_context = dedent(f"""
    <document_collection>
        {chr(10).join(context_strings)}
    </document_collection>
    """)
    return full_context
