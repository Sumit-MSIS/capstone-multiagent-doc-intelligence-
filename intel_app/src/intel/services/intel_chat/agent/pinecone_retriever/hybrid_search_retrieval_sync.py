from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone
import time
from src.intel.services.intel_chat.agent.pinecone_retriever.reranking import rerank_documents
from src.intel.services.intel_chat.agent.pinecone_retriever.bm25_handler import calculate_idf
from concurrent.futures import ThreadPoolExecutor, as_completed
from opentelemetry import context as ot_context
import os
from dotenv import load_dotenv
load_dotenv()
import mlflow

MODULE_NAME = "hybrid_search_retrieval.py"


INDEX_MAPPING = {
    os.getenv("DOCUMENT_SUMMARY_AND_CHUNK_INDEX", "document_summary_and_chunk_index"): os.getenv("DOCUMENT_SUMMARY_AND_CHUNK_INDEX_SPARSE", "document_summary_and_chunk_index_sparse"),
    os.getenv("DOCUMENT_SUMMARY_INDEX", "document_summary_index"): os.getenv("DOCUMENT_SUMMARY_INDEX_SPARSE", "document_summary_index_sparse"),
    os.getenv("DOCUMENT_METADATA_INDEX", "document_metadata_index"): os.getenv("DOCUMENT_METADATA_INDEX_SPARSE", "document_metadata_index_sparse")
}

@mlflow.trace(name="Get Pinecone Index")
def get_pinecone_index(index_name, api_key):
    """Fetch the Pinecone index."""
    try:
        start_time = time.perf_counter()
        pinecone_client = Pinecone(api_key=api_key)
        index = pinecone_client.Index(index_name)

        return index
    except Exception as e:
        raise e

@mlflow.trace(name="Get Embeddings")
def get_embeddings(org_id, file_id, query) -> dict:
    """Generate dense and sparse embeddings for the query."""
    try:
        embeddings = OpenAIEmbeddings(
            api_key=os.getenv("OPENAI_API_KEY"),
            model=os.getenv("OPENAI_EMBEDING_MODEL_NAME")
        )
        dense_embedding = embeddings.embed_query(query)

        return dense_embedding
    except Exception as e:
        raise RuntimeError(f"Failed to get embeddings for Org ID {org_id}, File ID {file_id}: {str(e)}")


########-------------------------------------------------------------------------------
# using RRF on both retrievals
###------------------------------------------------------------------------------------
@mlflow.trace(name="Reciprocal Rank Fusion")
def reciprocal_rank_fusion(sparse_results, dense_results, k:int) -> dict:
    """
    Combines sparse and dense search results using Reciprocal Rank Fusion (RRF).

    sparse_results, dense_results = dict of matches from Pinecone
    """
    try:
        if "matches" in sparse_results:
            sparse_matches = sparse_results.get("matches",[])  
        else:
            sparse_matches = []
        
        if "matches" in dense_results :
            dense_matches = dense_results.get("matches",[]) 
        else:
            dense_matches = []

        @mlflow.trace(name="Rank Results")
        def rank_results(results):
            """Return a dict: id -> RRF score."""
            scores = {}
            for rank, item in enumerate(results, start=1):
                rrf_score = 1 / (k + rank)
                scores[item["id"]] = (rank,rrf_score,item['score'])
            return scores

        # Rank each list
        sparse_scores = rank_results(sparse_matches)
        dense_scores = rank_results(dense_matches)

        # Merge scores
        combined_scores = {}
        all_ids = set(sparse_scores.keys()) | set(dense_scores.keys())
        # print(all_ids)
        # print(len(all_ids))

        for chunk_id in all_ids:
            # combined_scores[chunk_id] = sparse_scores.get(chunk_id, 0) + dense_scores.get(chunk_id, 0)
            sparse_rank, sparse_rrf,sparse_score = sparse_scores.get(chunk_id, (0, 0, 0))
            dense_rank, dense_rrf,dense_score = dense_scores.get(chunk_id, (0, 0, 0))
            combined_scores[chunk_id] = {
                "rrf_score": sparse_rrf + dense_rrf,
                "sparse_rank": sparse_rank,
                "dense_rank": dense_rank,
                "sparse_score":sparse_score,
                "dense_score":dense_score
            }

        # Build combined result list
        id_to_metadata = {}
        for res in sparse_matches+ dense_matches:
            id_to_metadata[res["id"]] = res

        fused_results = [
            {
                "id": chunk_id,
                "rrf_score": data["rrf_score"],
                "sparse_rank": data["sparse_rank"],
                "dense_rank": data["dense_rank"],
                "metadata": id_to_metadata[chunk_id]["metadata"]
            }
            for chunk_id, data in sorted(combined_scores.items(), key=lambda x: x[1]["rrf_score"], reverse=True)
        ]
        return fused_results
    except Exception as e:
        raise RuntimeError(f"Reciprocal Rank Fusion failed: {e}")



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

@mlflow.trace(name="Hybrid Search Retrieval")
def get_context_from_pinecone(
    pinecone_dense_index_name,
    pinecone_api_key,
    custom_filter,
    top_k,
    query,
    file_id,
    org_id,
    tag_ids,
    keyword_search_query, 
    dense_vectors,
    reranker_threshold=None):
    """
    Hybrid search retrieval using Pinecone with dual indexing (sparse + dense).
    Runs sparse & dense retrievals in parallel and merges them via RRF.
    """
    try:
        start_time = time.perf_counter()

        # Map to sparse index internally
        sparse_index_name = INDEX_MAPPING.get(pinecone_dense_index_name)
        if not sparse_index_name:
            raise ValueError(f"Unsupported Pinecone sparse index name for {pinecone_dense_index_name}")

        dense_index = get_pinecone_index(pinecone_dense_index_name, pinecone_api_key)
        sparse_index = get_pinecone_index(sparse_index_name, pinecone_api_key)

        # ----------------------
        # Define parallel tasks
        # ----------------------
        @mlflow.trace(name="Run Sparse Query")
        def run_sparse_query():
            try:
                if pinecone_dense_index_name != os.getenv("DOCUMENT_SUMMARY_AND_CHUNK_INDEX"):
                    return "sparse", ({}, 0.0)

                if keyword_search_query:
                    sparse_vectors = calculate_idf(keyword_search_query, org_id, file_id if isinstance(file_id, list) else [file_id], tag_ids)
                else:
                    sparse_vectors = calculate_idf(query, org_id, file_id if isinstance(file_id, list) else [file_id], tag_ids)

                if not sparse_vectors:
                    return "sparse", ({}, 0.0)

                start = time.perf_counter()
                result = sparse_index.query(
                    sparse_vector=sparse_vectors,
                    namespace=f"org_id_{org_id}#",
                    top_k=top_k,
                    filter=custom_filter,
                    include_metadata=True,
                    include_values=False
                )
                exec_time = round(time.perf_counter() - start, 6)
                return "sparse", (result, exec_time)
            except Exception as e:
                return "sparse", ({}, 0.0)
            
        @mlflow.trace(name="Run Dense Query")
        def run_dense_query():
            try:
                start = time.perf_counter()
                result = dense_index.query(
                    vector=dense_vectors,
                    namespace=f"org_id_{org_id}#",
                    top_k=top_k,
                    filter=custom_filter,
                    include_metadata=True,
                    include_values=False
                )
                exec_time = round(time.perf_counter() - start, 6)
                return "dense", (result, exec_time)
            except Exception as e:
                return "dense", ({}, 0.0)

        # ----------------------
        # Run both in parallel
        # ----------------------
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [
                submit_with_context(executor, run_sparse_query),
                submit_with_context(executor, run_dense_query)
            ]
            results = {name: val for name, val in (f.result() for f in as_completed(futures))}

        sparse_results, sparse_time = results.get("sparse", ({}, 0.0))
        dense_results, dense_time = results.get("dense", ({}, 0.0))

        no_dense = not dense_results or not dense_results.get("matches")
        no_sparse = not sparse_results or not sparse_results.get("matches")

        if no_dense and no_sparse:
            return {
                "matches": [],
                "namespace": f"org_id_{org_id}#",
                "usage": {"read_units": 0}
            }

        # ----------------------
        # Merge & rerank
        # ----------------------
        results = reciprocal_rank_fusion(sparse_results, dense_results, 60)

        # Rerank
        rerank_start = time.perf_counter()
        reranked_context = rerank_documents(query, results)
        rerank_time = round(time.perf_counter() - rerank_start, 6)

        # Handle both dict and list return types from rerank_documents
        if isinstance(reranked_context, dict) and "matches" in reranked_context:
            matches = reranked_context["matches"]
        elif isinstance(reranked_context, list):
            matches = reranked_context
        else:
            matches = []


        threshold = reranker_threshold if reranker_threshold is not None else 0.30

        # Apply filtering threshold
        filtered_matches = [
            r for r in matches
            if r.get("score", r.get("rrf_score", 0)) >= threshold
        ]

        # Wrap filtered list back in dict form
        filtered_reranked_context = {"matches": filtered_matches}
        

        return filtered_reranked_context

    except Exception as e:
        raise RuntimeError(f"Hybrid search failed for Org ID {org_id}: {e}")

@mlflow.trace(name="Get Chunks Text")
def get_chunks_text(context_chunks):
    matches = context_chunks.get('matches', [])
    if not matches:
        return []
    context = [match['metadata']['text'] for match in matches]
    return context