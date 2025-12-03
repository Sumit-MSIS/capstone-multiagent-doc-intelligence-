from langchain_openai import OpenAIEmbeddings
from pinecone.grpc import PineconeGRPC as Pinecone
import time
import json
import re
import pickle
from pinecone_text.sparse import BM25Encoder
from src.config.base_config import config
from src.common.logger import _log_message
from src.common.file_insights.hybrid_vector_handler.master_bm25_retrieval import retrieve_master_metadata_bm25
from src.common.file_insights.hybrid_vector_handler.bm25_encoder import CustomBM25Encoder
from src.common.file_insights.hybrid_vector_handler.upload_master_bm25 import upload_bm25_metadata_master
from src.common.file_insights.hybrid_vector_handler.bm25_handler import get_avgdl_from_db_metadata, calculate_tf_update_db_metadata
import mlflow
from opentelemetry import context as ot_context
# mlflow.config.enable_async_logging()
# mlflow.langchain.autolog()
mlflow.openai.autolog()

MODULE_NAME = "contract_template_generator.py"

PINECONE_API_KEY = config.PINECONE_API_KEY
DOCUMENT_METADATA_INDEX = config.DOCUMENT_METADATA_INDEX
OPENAI_API_KEY = config.OPENAI_API_KEY
OPENAI_EMBEDDING_MODEL_NAME = config.OPENAI_EMBEDING_MODEL_NAME
embedding_generator = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY, model=OPENAI_EMBEDDING_MODEL_NAME)


class ContractMetadataVectorUpserter:
    def __init__(self, logger=None):
        self.pinecone = Pinecone(api_key=PINECONE_API_KEY)
        self.dev_contract_metadata = DOCUMENT_METADATA_INDEX
        self.logger = logger

    def _log_message(self, message, function_name):
        """
        Internal method to create a standardized log message with the module name.
        """
        return _log_message(message, function_name, MODULE_NAME)
    
    @mlflow.trace(name="Contract Metadata - Get Pinecone Index")
    def get_pinecone_index_for_contract_template(self):
        """
        Fetch the Pinecone index for storing or querying document embeddings.
        """
        try:
            self.logger.info(self._log_message(f"Fetching Pinecone index: {self.dev_contract_metadata}", "get_pinecone_index_for_contract_template"))
            return self.pinecone.Index(self.dev_contract_metadata)
        except Exception as e:
            error_message = f"Error fetching Pinecone index '{self.dev_contract_metadata}': {e}"
            self.logger.error(self._log_message(error_message, "get_pinecone_index_for_contract_template"))
            return None

    @mlflow.trace(name="Contract Metadata - Get Hybrid Vectors")
    def get_embeddings(self, avgdl_value, chunk):
        """
        Generate dense and sparse embeddings for a given chunk of text.
        """
        try:
            self.logger.info(self._log_message("Generating embeddings for the chunk...", "get_embeddings"))
            dense_embedding = embedding_generator.embed_query(chunk)
            # --------------------------------------------------------------------------------------------
            bm25 = CustomBM25Encoder()
            doc_sparse_vector = bm25.custom_encode_documents(avgdl_value,[chunk])
            # doc_sparse_vector = bm25_encoder.encode_documents(chunk)
            # --------------------------------------------------------------------------------------------


            return {
                "values": dense_embedding,
                "sparse_values": doc_sparse_vector[0]
            }
        except Exception as e:
            error_message = f"Error generating embeddings: {e}"
            self.logger.error(self._log_message(error_message, "get_embeddings"))
            return None

    @mlflow.trace(name="Contract Metadata - Normalize JSON to Text")
    def normalize_json_to_text(self, json_data):
        """
        Convert JSON metadata into a normalized text format for embeddings.
        """
        self.logger.info(self._log_message(f"Normalizing JSON metadata: {json_data}", "normalize_json_to_text"))
        text_parts = []

        metadata = json_data.get("metadata", {})
        for category, items in metadata.items():
            if isinstance(items, list):
                for item in items:
                    title = item.get("title", "")
                    value = item.get("value", "N/A")
                    if value and value not in ("null", ""):
                        text_parts.append(f"{title}: {value}")

        return " | ".join(text_parts)

    @mlflow.trace(name="Contract Metadata - Create Metadata")
    def create_metadata(self, json_data):
        """
        Extract and structure metadata from JSON for Pinecone.
        """
        self.logger.info(self._log_message(f"Creating structured metadata from JSON: {json_data}", "create_metadata"))
        metadata = {}

        for category, items in json_data.get("metadata", {}).items():
            for item in items:
                title = item.get("title", "").lower().replace(" ", "_")
                value = item.get("value", "N/A")
                if value and value not in ("null", ""):
                    metadata[title] = value

        return metadata

    @mlflow.trace(name="Contract Metadata - Create Contract Template Vector")
    def create_contract_template_vector(self, text, file_id, file_name, file_type, user_id, org_id, chunks_no):
        """
        Create and upsert a contract template vector in Pinecone.
        """
        try:
            self.logger.info(self._log_message(f"Starting vector creation for file_id: {file_id}", "create_contract_template_vector"))

            # Parse and normalize input text
            if isinstance(text, str):
                try:
                    text = re.sub(r"(?<!\w)'(.*?)'(?!\w)", r'"\1"', text)
                    text = text.replace("None", "null")
                    text_data = json.loads(text)
                except json.JSONDecodeError:
                    self.logger.error(self._log_message(f"Invalid JSON string: {text}", "create_contract_template_vector"))
                    raise ValueError("Provided text is not valid JSON.")
            elif isinstance(text, dict):
                text_data = text
            else:
                raise ValueError("Provided text must be a JSON string or dictionary.")

            normalized_text = self.normalize_json_to_text(text_data)
            # ---------------------------------------------------------------------------------------------------
            # master_metadata_bm25_encoder = retrieve_master_metadata_bm25(org_id, self.logger)

            # if not master_metadata_bm25_encoder:
            #     self.logger.info(self._log_message("No existing BM25 encoder found. Creating a new one.", "create_contract_template_vector"))
            #     master_metadata_bm25_encoder = CustomBM25Encoder()

            # updated_bm25_object = master_metadata_bm25_encoder.insert([normalized_text], f"org_id_{org_id}", [file_id])
            # if updated_bm25_object:
            #     upload_bm25_metadata_master(pickle.dumps(updated_bm25_object), org_id, self.logger)
            #     self.logger.info(self._log_message("BM25 metadata updated successfully.", "create_contract_template_vector"))

            calculate_tf_update_db_metadata(org_id,user_id,file_id,[normalized_text],self.logger)

            # ---------------------------------------------------------------------------------------------------

            # --------------------------------------------------------------------------------------
            avgdl_value = get_avgdl_from_db_metadata(file_id,self.logger)
            # --------------------------------------------------------------------------------------

            # response = self.get_embeddings(updated_bm25_object, normalized_text)

            response = self.get_embeddings(avgdl_value, normalized_text)
            if not response:
                raise ValueError("Failed to generate embeddings.")

            contract_template_index = self.get_pinecone_index_for_contract_template()
            contract_template_namespace = f"org_id_{org_id}#"

            metadata = {
                'org_id': org_id,
                'user_id': user_id,
                'file_id': file_id,
                'chunks': chunks_no,
                'page_no': 1,
                'text': normalized_text,
                'file_name': file_name,
                'file_type': file_type,
                **self.create_metadata(text_data)
            }

            vector = {
                "id": f"{org_id}#{file_id}",
                "values": response["values"],
                "sparse_values": response["sparse_values"],
                "metadata": metadata
            }

            self.logger.info(self._log_message(f"Generated vector ID: {vector['id']}", "create_contract_template_vector"))

            # Upsert the vector into Pinecone
            try:
                contract_template_index.upsert(vectors=[vector], namespace=contract_template_namespace)
                self.logger.info(self._log_message("Vector upserted successfully to Pinecone.", "create_contract_template_vector"))
            except Exception as upsert_error:
                error_message = f"Error during Pinecone upsertion: {upsert_error}"
                self.logger.error(self._log_message(error_message, "create_contract_template_vector"))
                raise upsert_error

        except Exception as e:
            error_message = f"Error creating contract template vector: {e}"
            self.logger.error(self._log_message(error_message, "create_contract_template_vector"))
            raise e

    @mlflow.trace(name="Contract Metadata - Process Contract Template")
    def process_contract_template(self, text, file_id, file_name, file_type, user_id, org_id, total_chunks):
        """
        Process and upsert a contract template vector asynchronously.
        """
        try:
            start_time = time.perf_counter()
            self.logger.info(self._log_message("Processing contract template...", "process_contract_template"))

            self.create_contract_template_vector(text, file_id, file_name, file_type, user_id, org_id, total_chunks)

            elapsed_time = time.perf_counter() - start_time
            self.logger.info(self._log_message(f"Contract template processed and upserted in {elapsed_time:.2f} seconds.", "process_contract_template"))

        except Exception as e:
            error_message = f"Error processing contract template for file_id: {file_id}, org_id: {org_id}. Error: {e}"
            self.logger.error(self._log_message(error_message, "process_contract_template"))
            raise
