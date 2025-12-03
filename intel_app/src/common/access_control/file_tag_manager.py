import asyncio
from pinecone import Pinecone
from src.config.base_config import config


INDEX_MAPPING = {
    config.DOCUMENT_SUMMARY_AND_CHUNK_INDEX: config.DOCUMENT_SUMMARY_AND_CHUNK_INDEX_SPARSE,
    config.DOCUMENT_SUMMARY_INDEX: config.DOCUMENT_SUMMARY_INDEX_SPARSE,
    config.DOCUMENT_METADATA_INDEX: config.DOCUMENT_METADATA_INDEX_SPARSE
}

pc = Pinecone(api_key=config.PINECONE_API_KEY)
index = pc.Index(config.DOCUMENT_SUMMARY_AND_CHUNK_INDEX)

sparse_index_name = INDEX_MAPPING.get(config.DOCUMENT_SUMMARY_AND_CHUNK_INDEX)
sparse_index = pc.Index(sparse_index_name)


async def update_chunk_metadata(namespace: str, chunk_id: str, tag_ids: list[str]):
    """Update a single chunk asynchronously."""
    await asyncio.to_thread(
        index.update,
        namespace=namespace,
        id=chunk_id,
        set_metadata={"tag_ids": tag_ids}
    )

    # Update sparse index
    await asyncio.to_thread(
        sparse_index.update,
        namespace=namespace,
        id=chunk_id,
        set_metadata={"tag_ids": tag_ids}
    )


async def update_file_tags(org_id: int, file_tag_list: list[dict], logger):
    """
    Bulk update metadata tag_ids for multiple files concurrently.
    """
    logical_partition = f"org_id_{org_id}#"

    async def process_file(file_entry: dict):
        file_id = file_entry["file_id"]
        new_tags = [str(t) for t in file_entry["tag_ids"]]

        # Fetch initial chunk to get chunks count
        initial_id = [f"{org_id}#{file_id}#1"]
        fetch_resp = await asyncio.to_thread(index.fetch, ids=initial_id, namespace=logical_partition)

        if not fetch_resp.vectors:
            return {"file_id": file_id, "status": "not_found"}

        chunks_count = int(next(iter(fetch_resp['vectors'].values()))['metadata']['chunks_count'])
        chunk_ids = [f"{org_id}#{file_id}#{i+1}" for i in range(chunks_count)]

        # Update all chunks concurrently
        await asyncio.gather(*(update_chunk_metadata(logical_partition, cid, new_tags) for cid in chunk_ids))

        return {"file_id": file_id, "status": "updated", "chunks": chunks_count}

    # Run all files concurrently
    results = await asyncio.gather(*(process_file(f) for f in file_tag_list))
    return results


async def query_by_tags(org_id: int, tag_ids: list[str], top_k: int = 10):
    """
    Query vectors by filtering on tag_ids asynchronously.
    """
    logical_partition = f"org_id_{org_id}#"
    query_vector = [0.0] * 3072  # Replace with actual embedding
    filter_condition = {"tag_ids": {"$in": tag_ids}}

    results = await asyncio.to_thread(
        index.query,
        namespace=logical_partition,
        vector=query_vector,
        top_k=top_k,
        include_metadata=True,
        filter=filter_condition
    )
    return results
