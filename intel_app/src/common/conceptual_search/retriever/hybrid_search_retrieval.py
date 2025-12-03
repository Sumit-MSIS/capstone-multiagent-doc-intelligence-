from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone
import time
from src.common.hybrid_retriever.reranking import rerank_documents,pinecone_rerank_documents
from src.config.base_config import config
from src.common.logger import _log_message
import json
import orjson
import mlflow
import requests
from src.common.file_insights.hybrid_vector_handler.bm25_handler import calculate_idf
from src.common.file_insights.hybrid_vector_handler.bm25_reindexing import BM25ReIndexing
from concurrent.futures import ThreadPoolExecutor, as_completed
from opentelemetry import context as ot_context

mlflow.openai.autolog()
MODULE_NAME = "hybrid_search_retrieval.py"

# this mapping is required to map dense index with its associated sparse index
# after the implementation of separate indexing for hybrid retrieval there will be two individual indexes, one for sparse and one for dense
# to not affect the get_context_from_pinecone() method's parameters , we are using this mapping approach to get the corresponding sparse index.
INDEX_MAPPING = {
    config.DOCUMENT_SUMMARY_AND_CHUNK_INDEX: config.DOCUMENT_SUMMARY_AND_CHUNK_INDEX_SPARSE,
    config.DOCUMENT_SUMMARY_INDEX: config.DOCUMENT_SUMMARY_INDEX_SPARSE,
    config.DOCUMENT_METADATA_INDEX: config.DOCUMENT_METADATA_INDEX_SPARSE
}


def submit_with_context(executor, fn, *args, **kwargs):
    """Submit a function to executor with OpenTelemetry context propagation."""
    parent_ctx = ot_context.get_current()
    def wrapped():
        token = ot_context.attach(parent_ctx)
        try:
            return fn(*args, **kwargs)
        finally:
            ot_context.detach(token)
    return executor.submit(wrapped)


@mlflow.trace(name="Get Hybrid Pinecone Index")
def get_pinecone_index(index_name, api_key, logger):
    """Fetch the Pinecone index."""
    try:
        start_time = time.perf_counter()
        pinecone_client = Pinecone(api_key=api_key)
        index = pinecone_client.Index(index_name)

        logger.debug(_log_message(f"Fetching Pinecone index '{index_name}'.", "get_pinecone_index", MODULE_NAME))
        logger.info(_log_message(f"Pinecone index '{index_name}' connected successfully.", "get_pinecone_index", MODULE_NAME))

        execution_time = round(time.perf_counter() - start_time, 6)
        logger.debug(_log_message(f"Index fetch time: {execution_time} seconds.", "get_pinecone_index", MODULE_NAME))

        return index
    except Exception as e:
        logger.error(_log_message(f"Error connecting to Pinecone index '{index_name}': {e}", "get_pinecone_index", MODULE_NAME))
        raise

@mlflow.trace(name="Get Hybrid Vectors")
def get_embeddings(org_id, file_id, query, logger) -> dict:
    """Generate dense and sparse embeddings for the query."""
    try:
        start_time = time.perf_counter()

        embeddings = OpenAIEmbeddings(
            api_key=config.OPENAI_API_KEY,
            model=config.OPENAI_EMBEDING_MODEL_NAME
        )
        dense_embedding = embeddings.embed_query(query)
        org_str = f"org_id_{org_id}"


        execution_time = round(time.perf_counter() - start_time, 6)

        logger.debug(_log_message(f"Embeddings generated for query: '{query}'.", "get_embeddings", MODULE_NAME))
        logger.info(_log_message(f"Embeddings computed in {execution_time} seconds.", "get_embeddings", MODULE_NAME))

        return dense_embedding
    except Exception as e:
        logger.error(_log_message(f"Error generating embeddings for query '{query}': {e}", "get_embeddings", MODULE_NAME))
        return None


@mlflow.trace(name="Hybrid Score Normalization")
def hybrid_score_norm(dense, sparse, alpha: float):
    """Hybrid score using a convex combination

    alpha * dense + (1 - alpha) * sparse

    Args:
        dense: Array of floats representing
        sparse: a dict of `indices` and `values`
        alpha: scale between 0 and 1
    """
    if alpha < 0 or alpha > 1:
        raise ValueError("Alpha must be between 0 and 1")
    
    hs = {
        'indices': sparse['indices'],
        'values':  [v * (1 - alpha) for v in sparse['values']]
    }
    
    return [v * alpha for v in dense], hs

########-------------------------------------------------------------------------------
# using RRF on both retrievals
###------------------------------------------------------------------------------------

def extract_result(res):
    """Normalize Pinecone results into dicts with id, score, and metadata."""
    if isinstance(res, dict):  # already dict
        return {
            "id": res.get("id"),
            "score": res.get("score", 0),
            "metadata": res.get("metadata", {})
        }
    else:  # Pinecone ScoredVector object
        return {
            "id": getattr(res, "id", None),
            "score": getattr(res, "score", 0),
            "metadata": getattr(res, "metadata", {})
        }


@mlflow.trace(name="Reciprocal Rank Fusion")
def reciprocal_rank_fusion(sparse_results: dict, dense_results: dict, k: int, logger) -> list[dict]:
    """
    Combines sparse and dense search results from Pinecone using Reciprocal Rank Fusion (RRF).

    Args:
        sparse_results (dict): Pinecone query response with "matches".
        dense_results (dict): Pinecone query response with "matches".
        k (int): RRF constant, typically 60.
        logger: logger instance.

    Returns:
        list[dict]: Fused results sorted by RRF score.
    """
    try:
        logger.info(_log_message("Initializing Reciprocal Rank Fusion", "reciprocal_rank_fusion", MODULE_NAME))

        sparse_matches = sparse_results.get("matches", [])
        if not sparse_matches:
            logger.warning(_log_message("No matches found in sparse results", "reciprocal_rank_fusion", MODULE_NAME))

        dense_matches = dense_results.get("matches", [])
        if not dense_matches:
            logger.warning(_log_message("No matches found in dense results", "reciprocal_rank_fusion", MODULE_NAME))

        @mlflow.trace(name="RRF - Rank Results")
        def rank_results(results: list[dict]) -> dict:
            """Convert Pinecone matches into dict: id -> (rank, rrf_score, raw_score)."""
            scores = {}
            for rank, item in enumerate(sorted(results, key=lambda x: x["score"], reverse=True), start=1):
                rrf_score = 1 / (k + rank)
                scores[item["id"]] = (rank, rrf_score, item.get("score", 0))
            return scores

        # Rank each list
        sparse_scores = rank_results(sparse_matches)
        dense_scores = rank_results(dense_matches)

        # Merge results
        combined_scores = {}
        all_ids = set(sparse_scores) | set(dense_scores)
        logger.info(_log_message(
            f"Total unique ids from both sparse and dense results: {len(all_ids)}",
            "reciprocal_rank_fusion", MODULE_NAME
        ))

        for chunk_id in all_ids:
            sparse_rank, sparse_rrf,sparse_score = sparse_scores.get(chunk_id, (0, 0, 0)) #default is set to zero to reduce the weightage for the chunk that is not present wither in dense or sparse
            dense_rank, dense_rrf,dense_score = dense_scores.get(chunk_id, (0, 0, 0))
            combined_scores[chunk_id] = {
                "rrf_score": sparse_rrf + dense_rrf,
                "sparse_rank": sparse_rank,
                "dense_rank": dense_rank,
                "sparse_score": sparse_score,
                "dense_score": dense_score,
            }

        # Build metadata map
        id_to_metadata = {}
        for res in sparse_matches + dense_matches:
            normalized = extract_result(res)
            if normalized.get("metadata"):
                id_to_metadata[normalized["id"]] = normalized

        # Final fused results
        fused_results = [
            {
                "id": chunk_id,
                "rrf_score": data["rrf_score"],
                "sparse_rank": data["sparse_rank"],
                "dense_rank": data["dense_rank"],
                "sparse_score": data["sparse_score"],
                "dense_score": data["dense_score"],
                "metadata": id_to_metadata[chunk_id]["metadata"],
            }
            for chunk_id, data in sorted(combined_scores.items(), key=lambda x: x[1]["rrf_score"], reverse=True)
            if chunk_id in id_to_metadata
        ]

        logger.info(_log_message("Reciprocal Rank Fusion complete", "reciprocal_rank_fusion", MODULE_NAME))
        return fused_results

    except Exception as e:
        logger.error(_log_message(f"Error in RRF: {e}", "reciprocal_rank_fusion", MODULE_NAME))
        raise RuntimeError(f"Reciprocal Rank Fusion failed: {e}")



@mlflow.trace(name="Retrieve Context from Pinecone")
def get_context_from_pinecone(
    pinecone_dense_index_name,
    pinecone_api_key,
    custom_filter,
    top_k,
    keyword_search_query,
    semantic_search_query,
    reranking_query,
    actual_user_query,
    file_id,
    user_id,
    org_id,
    tag_ids,
    logger
):
    """
    Hybrid search retrieval using Pinecone with dual indexing (sparse + dense).
    Each query function handles both vector preparation and Pinecone querying.
    OpenTelemetry context is propagated across threads.
    """
    try:
        start_time = time.perf_counter()
        query = actual_user_query
        logger.debug(_log_message(f"[{org_id}] Starting hybrid search for '{query}'", 
                                  "get_context_from_pinecone", MODULE_NAME))

        # Prepare Pinecone indices
        sparse_index_name = INDEX_MAPPING.get(pinecone_dense_index_name)
        if not sparse_index_name:
            raise ValueError(f"Unsupported Pinecone sparse index for {pinecone_dense_index_name}")
        dense_index = get_pinecone_index(pinecone_dense_index_name, pinecone_api_key, logger)
        sparse_index = get_pinecone_index(sparse_index_name, pinecone_api_key, logger)

        # --- Define unified query functions ---
        @mlflow.trace(name="Run Sparse Query")
        def run_sparse_query():
            """Compute sparse vectors + run sparse Pinecone query"""
            try:
                if pinecone_dense_index_name != config.DOCUMENT_SUMMARY_AND_CHUNK_INDEX:
                    return "sparse_results", ({}, 0.0)
                vectors = calculate_idf(keyword_search_query, org_id, file_id if isinstance(file_id, list) else [file_id], tag_ids, logger)
                if not vectors:
                    return "sparse_results", ({}, 0.0)

                start = time.perf_counter()
                result = sparse_index.query(
                    sparse_vector=vectors,
                    namespace=f"org_id_{org_id}#",
                    top_k=top_k,
                    filter=custom_filter,
                    include_metadata=True,
                    include_values=False
                )
                return "sparse_results", (result, round(time.perf_counter() - start, 6))
            except Exception as e:
                logger.error(_log_message(f"[{org_id}] Sparse query failed: {e}", 
                                          "get_context_from_pinecone", MODULE_NAME))
                return "sparse_results", ({}, 0.0)
            
        @mlflow.trace(name="Run Dense Query")
        def run_dense_query():
            """Compute dense embedding + run dense Pinecone query"""
            try:
                embedding = get_embeddings(
                    org_id, file_id if isinstance(file_id, list) else [file_id],
                    semantic_search_query, logger
                )
                if not embedding:
                    return "dense_results", ({}, 0.0)

                start = time.perf_counter()
                result = dense_index.query(
                    vector=embedding,
                    namespace=f"org_id_{org_id}#",
                    top_k=top_k,
                    filter=custom_filter,
                    include_metadata=True,
                    include_values=False
                )
                return "dense_results", (result, round(time.perf_counter() - start, 6))
            except Exception as e:
                logger.error(_log_message(f"[{org_id}] Dense query failed: {e}", 
                                          "get_context_from_pinecone", MODULE_NAME))
                return "dense_results", ({}, 0.0)

        # --- Run sparse + dense in parallel with context ---
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [
                submit_with_context(executor, run_sparse_query),
                submit_with_context(executor, run_dense_query)
            ]
            results = {name: val for name, val in (f.result() for f in as_completed(futures))}

        sparse_results, sparse_time = results.get("sparse_results", ({}, 0.0))
        dense_results, dense_time = results.get("dense_results", ({}, 0.0))

        if not sparse_results or not sparse_results.get("matches"):
            logger.error(_log_message(f"[{org_id}] No sparse results found for query '{query}'.", "get_context_from_pinecone", MODULE_NAME))
            
        if not dense_results or not dense_results.get("matches"):
            logger.error(_log_message(f"[{org_id}] No dense results found for query '{query}'.", "get_context_from_pinecone", MODULE_NAME))

        no_dense = not dense_results or not dense_results.get("matches")
        no_sparse = not sparse_results or not sparse_results.get("matches")

        if no_dense and no_sparse:
            logger.error(_log_message(
                f"[{org_id}] No results found for query '{query}'.",
                "get_context_from_pinecone",
                MODULE_NAME
            ))
            return {
                "matches": [],
                "namespace": f"org_id_{org_id}#",
                "usage": {"read_units": 0}
            }


        # --- Fusion + rerank ---
        fused = reciprocal_rank_fusion(sparse_results, dense_results, 60, logger)
        rerank_start = time.perf_counter()
        reranked = rerank_documents(reranking_query, fused, logger)
        rerank_time = round(time.perf_counter() - rerank_start, 6)

        total_time = round(time.perf_counter() - start_time, 6)
        logger.info(_log_message(
            f"[{org_id}] Hybrid search completed in {total_time}s "
            f"(sparse={sparse_time}s, dense={dense_time}s, rerank={rerank_time}s)",
            "get_context_from_pinecone", MODULE_NAME
        ))

        return reranked

    except Exception as e:
        logger.error(_log_message(f"[{org_id}] Hybrid search failed: {e}", 
                                  "get_context_from_pinecone", MODULE_NAME))
        raise RuntimeError(f"Hybrid search failed for Org ID {org_id}: {e}")

@mlflow.trace(name="Get Chunks Text")
def get_chunks_text(context_chunks):
    matches = context_chunks.get('matches', [])
    if not matches:
        return []
    context = [match['metadata']['text'] for match in matches]
    return context