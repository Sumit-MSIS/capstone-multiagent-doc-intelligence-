from src.common.file_insights.hybrid_vector_handler.master_bm25_retrieval import retrieve_master_bm25
from src.common.file_insights.hybrid_vector_handler.upload_master_bm25 import upload_bm25_master
import pickle
from src.common.file_insights.hybrid_vector_handler.master_bm25_retrieval import retrieve_master_metadata_bm25
from src.common.file_insights.hybrid_vector_handler.upload_master_bm25 import upload_bm25_metadata_master
from src.common.logger import _log_message

MODULE_NAME ="update_bm25_corpus.py"

def update_corpus(file_id, user_id, org_id, logger):

    try:
        logger.info(_log_message("Updating BM25 Master corpus...", "update_corpus", MODULE_NAME))
        master_bm25_encoder = retrieve_master_bm25(org_id, logger)
        logger.info(_log_message("BM25 corpus retrieved successfully.", "update_corpus", MODULE_NAME))
        if master_bm25_encoder is None:
            logger.warning(_log_message("No BM25 corpus found for the org_id", "update_corpus", MODULE_NAME))
            return {"success": False, "error": "No BM25 corpus found for the org_id", "data": {}}
        elif len(file_id) == 0:
            logger.warning(_log_message("No file IDs provided", "update_corpus", MODULE_NAME))
            return {"success": False, "error": "No file_ids provided", "data": {}}


        logger.info(_log_message(f"No of Docs before deletion: {master_bm25_encoder.n_docs}", "update_corpus", MODULE_NAME))

        updated_master_bm25 = master_bm25_encoder.delete(f"org_id_{org_id}", file_id)

        logger.info(_log_message(f"No of Docs after deletion: {updated_master_bm25.n_docs}", "update_corpus", MODULE_NAME))

        serialized_bm25_index = pickle.dumps(updated_master_bm25)

        upload_bm25_master(serialized_bm25_index, org_id, logger)
        logger.info(_log_message("BM25 corpus updated successfully.", "update_corpus", MODULE_NAME))

        logger.info(_log_message("Updating MetaData BM25 corpus...", "update_corpus", MODULE_NAME))
        master_metadata_bm25_encoder = retrieve_master_metadata_bm25(org_id, logger)
        logger.info(_log_message("Metadata BM25 corpus retrieved successfully.", "update_corpus", MODULE_NAME))

        if master_metadata_bm25_encoder is None:
            logger.warning(_log_message("No master_metadata_bm25_encoder corpus found for the org_id", "update_corpus", MODULE_NAME))
            return {"success": False, "error": "No master_metadata_bm25_encoder corpus found for the org_id", "data": {}}
        elif len(file_id) == 0:
            logger.warning(_log_message("No file IDs provided", "update_corpus", MODULE_NAME))
            return {"success": False, "error": "No file_ids provided", "data": {}}


        logger.info(_log_message(f"No of metadata Docs before deletion: {master_metadata_bm25_encoder.n_docs}", "update_corpus", MODULE_NAME))

        updated_metadata_master_bm25 = master_metadata_bm25_encoder.delete(f"org_id_{org_id}", file_id)

        logger.info(_log_message(f"No of metadata Docs after deletion: {updated_metadata_master_bm25.n_docs}", "update_corpus", MODULE_NAME))

        metadata_serialized_bm25_index = pickle.dumps(updated_metadata_master_bm25)

        upload_bm25_metadata_master(metadata_serialized_bm25_index, org_id, logger)
        logger.info(_log_message("Metadata BM25 corpus updated successfully.", "update_corpus", MODULE_NAME))

    except Exception as e:
        logger.error(_log_message(f"Error updating BM25 corpus: {e}", "update_corpus", MODULE_NAME))
        raise e



    
