
import os,sys
import json
from src.intel.services.intel_chat.agent.pinecone_retriever.db_connection_manager import DBConnectionManager
# from src.intel.services.intel_chat.agent.async_db_connection_manager import DBConnectionManager

from src.intel.services.intel_chat.agent.pinecone_retriever.bm25_encoder import CustomBM25Encoder
from typing import List, Dict, Union, Optional,Any
import time
from dotenv import load_dotenv
load_dotenv()
import mlflow
mlflow.openai.autolog()
import logging
logger = logging.getLogger("sql_agent")
CONTEXTUAL_RAG_DB = os.getenv("CONTEXTUAL_RAG_DB", "contextual_rag_db")
BM25_CORPUS_DB_TABLE = os.getenv("BM25_CORPUS_DB_TABLE", "bm25_corpus")
BM25_METADATA_DB_TABLE = os.getenv("BM25_METADATA_DB_TABLE", "bm25_metadata")
BM25_SUMMARY_DB_TABLE = os.getenv("BM25_SUMMARY_DB_TABLE", "bm25_summary")
CONTRACT_INTEL_DB = os.getenv("CONTRACT_INTEL_DB", "contract_intel_db")

@mlflow.trace(name="Calculate BM25 IDF")
async def calculate_idf(query: str, org_id: int, file_ids: Union[None, List[str]] = None, tag_ids=[]):
    try:

        result = []
        async with DBConnectionManager(CONTEXTUAL_RAG_DB) as conn:
            async with conn.cursor() as cursor:
                if file_ids:
                    await cursor.execute(
                        f"""
                        SELECT term_frequency 
                        FROM {BM25_CORPUS_DB_TABLE} 
                        WHERE is_archived IS NULL 
                        AND file_id IN %s;
                        """,
                        (tuple(file_ids),)
                    )
                    result = await cursor.fetchall()
                else:
                    return {}


                # Extract list of term indices from each document
                term_lists = []
                for row in result:
                    tf_json = row.get("term_frequency") if isinstance(row, dict) else row[0]
                    terms = list(json.loads(tf_json).keys())
                    term_lists.append(terms)


                # Calculate document frequency
                bm25 = CustomBM25Encoder()
                doc_freq = bm25.count_values_across_lists(term_lists)
                total_docs = len(term_lists)

                # Compute IDF values for the query
                idf_vals = bm25.custom_doc_query(query, doc_freq=doc_freq, N=total_docs)

                return idf_vals

    except Exception as e:
        return {}
