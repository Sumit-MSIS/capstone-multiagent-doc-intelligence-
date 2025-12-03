# async_hybrid_search_retrieval.py

import os
import time
import asyncio
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone
from openai import AsyncOpenAI
from src.intel.services.intel_chat.agent.pinecone_retriever.bm25_handler import calculate_idf
from src.intel.services.intel_chat.agent.pinecone_retriever.reranking import rerank_documents_async
from opentelemetry import context as ot_context
import mlflow

MODULE_NAME = "async_hybrid_search_retrieval.py"

client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
INDEX_MAPPING = {
    os.getenv("DOCUMENT_SUMMARY_AND_CHUNK_INDEX"): os.getenv("DOCUMENT_SUMMARY_AND_CHUNK_INDEX_SPARSE"),
    os.getenv("DOCUMENT_SUMMARY_INDEX"): os.getenv("DOCUMENT_SUMMARY_INDEX_SPARSE"),
    os.getenv("DOCUMENT_METADATA_INDEX"): os.getenv("DOCUMENT_METADATA_INDEX_SPARSE")
}

# -----------------------------
# Async Pinecone Index fetch
# -----------------------------
@mlflow.trace(name="Get Async Pinecone Index")
async def get_pinecone_index_async(index_name, api_key):
    client = Pinecone(api_key=api_key)
    return client.Index(index_name)


# -----------------------------
# Async Embeddings
# -----------------------------
# @mlflow.trace(name="Get Async Embeddings")
async def get_embeddings(org_id, file_id, query):
    try:
        response = await client.embeddings.create(
            model=os.getenv("OPENAI_EMBEDING_MODEL_NAME"),
            input=query
        )

        # response.data[0].embedding → vector
        return response.data[0].embedding
    
    except Exception as e:
        raise RuntimeError(
            f"Embedding failed for Org {org_id}, File {file_id}: {str(e)}"
        )
    
# -----------------------------
# RRF (unchanged, synchronous-safe)
# -----------------------------
async def reciprocal_rank_fusion(sparse_results, dense_results, k: int):
    try:
        sparse_matches = sparse_results.get("matches", [])
        dense_matches = dense_results.get("matches", [])

        def rank_results(results):
            scores = {}
            for rank, item in enumerate(results, start=1):
                rrf_score = 1 / (k + rank)
                scores[item["id"]] = (rank, rrf_score, item.get("score", 0))
            return scores

        sparse_scores = rank_results(sparse_matches)
        dense_scores = rank_results(dense_matches)

        combined_scores = {}
        all_ids = set(sparse_scores.keys()) | set(dense_scores.keys())

        for cid in all_ids:
            s_rank, s_rrf, s_score = sparse_scores.get(cid, (0, 0, 0))
            d_rank, d_rrf, d_score = dense_scores.get(cid, (0, 0, 0))
            combined_scores[cid] = {
                "rrf_score": s_rrf + d_rrf,
                "sparse_rank": s_rank,
                "dense_rank": d_rank,
                "sparse_score": s_score,
                "dense_score": d_score
            }

        id_to_metadata = {res["id"]: res for res in (sparse_matches + dense_matches)}

        fused_results = [
            {
                "id": cid,
                "metadata": id_to_metadata[cid]["metadata"],
                **scores
            }
            for cid, scores in sorted(combined_scores.items(),
                                      key=lambda x: x[1]["rrf_score"],
                                      reverse=True)
        ]

        return fused_results

    except Exception as e:
        raise RuntimeError(f"RRF failed: {e}")


# -----------------------------
# Async Sparse Search
# -----------------------------
async def run_sparse_query_async(index, org_id, custom_filter, top_k, query, keyword_query, file_id, tag_ids):
    try:
        if keyword_query:
            sparse_vec = await calculate_idf(keyword_query, org_id, [file_id], tag_ids)
        else:
            sparse_vec = await calculate_idf(query, org_id, [file_id], tag_ids)

        if not sparse_vec:
            return {"matches": []}

        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            None,
            lambda: index.query(
                sparse_vector=sparse_vec,
                namespace=f"org_id_{org_id}#",
                top_k=top_k,
                filter=custom_filter,
                include_metadata=True
            )
        )
        return result
    except:
        return {"matches": []}


# -----------------------------
# Async Dense Search
# -----------------------------
async def run_dense_query_async(index, org_id, custom_filter, top_k, dense_vectors):
    try:
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            None,
            lambda: index.query(
                vector=dense_vectors,
                namespace=f"org_id_{org_id}#",
                top_k=top_k,
                filter=custom_filter,
                include_metadata=True
            )
        )
        return result
    except:
        return {"matches": []}


# -----------------------------
# FULL ASYNC HYBRID RETRIEVAL
# -----------------------------
@mlflow.trace(name="Async Hybrid Search Retrieval")
async def get_context_from_pinecone(
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
        reranker_threshold=None
):
    try:
        sparse_index_name = INDEX_MAPPING.get(pinecone_dense_index_name)
        if not sparse_index_name:
            raise ValueError(f"No sparse index mapped for {pinecone_dense_index_name}")

        # Fetch indexes asynchronously
        dense_index, sparse_index = await asyncio.gather(
            get_pinecone_index_async(pinecone_dense_index_name, pinecone_api_key),
            get_pinecone_index_async(sparse_index_name, pinecone_api_key)
        )

        # Run sparse + dense search in parallel (correct async usage)
        sparse_res, dense_res = await asyncio.gather(
            run_sparse_query_async(
                sparse_index, org_id, custom_filter, top_k,
                query, keyword_search_query, file_id, tag_ids
            ),
            run_dense_query_async(
                dense_index, org_id, custom_filter, top_k, dense_vectors
            )
        )

        if not sparse_res.get("matches") and not dense_res.get("matches"):
            return {"matches": []}

        # RRF Merge
        rrf_results = await reciprocal_rank_fusion(sparse_res, dense_res, k=60)

        # Async Reranking
        reranked = await rerank_documents_async(query, rrf_results)

        matches = reranked.get("matches", [])
        threshold = reranker_threshold or 0.30

        filtered = [
            m for m in matches
            if m.get("score", m.get("rrf_score", 0)) >= threshold
        ]

        return {"matches": filtered}

    except Exception as e:
        raise RuntimeError(f"Async Hybrid Search failed for org {org_id}: {e}")


# -----------------------------
# Get Chunks Text (unchanged)
# -----------------------------
def get_chunks_text(context_chunks):
    matches = context_chunks.get("matches", [])
    return [m["metadata"]["text"] for m in matches]
