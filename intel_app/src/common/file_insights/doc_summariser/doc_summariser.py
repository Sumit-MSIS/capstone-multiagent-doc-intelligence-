import time
import concurrent.futures
from src.common.logger import _log_message
from src.common.file_insights.doc_summariser.llm_call import open_ai_llm_call
from src.config.base_config import config
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from langchain_openai import OpenAIEmbeddings
from src.common.file_insights.doc_summariser.utils import get_pinecone_client, get_pinecone_index
from src.common.file_insights.hybrid_vector_handler.bm25_handler import get_avgdl_from_db_summary,calculate_tf_update_db_summary
from src.common.file_insights.hybrid_vector_handler.bm25_encoder import CustomBM25Encoder
import mlflow
# mlflow.config.enable_async_logging()
# mlflow.langchain.autolog()
mlflow.openai.autolog()
from opentelemetry import context as ot_context

OPENAI_MODEL_NAME = config.OPENAI_MODEL_NAME
OPENAI_EMBEDING_MODEL_NAME = config.OPENAI_EMBEDING_MODEL_NAME
OPENAI_API_KEY = config.OPENAI_API_KEY

PINECONE_API_KEY = config.PINECONE_API_KEY
DOCUMENT_SUMMARY_INDEX = config.DOCUMENT_SUMMARY_INDEX

embeddings_generator = OpenAIEmbeddings(api_key=OPENAI_API_KEY, model=OPENAI_EMBEDING_MODEL_NAME)

class DocSummariser:

    def __init__(self, logger):
        self.logger = logger

    def _log_message(self, message: str, function_name: str) -> str:
        return _log_message(message, function_name, "doc_summariser.py")

    @mlflow.trace(name="Doc Summarizer - Summarize Text")
    def summarize_text(self, text: str, chunk_size: int = 8000) -> str:
        """
        Summarizes the given text into a structured format.
        """
        function_name = "summarize_text"
        try:
            start_time = time.perf_counter()

            final_summary_prompt = final_summary_prompt = """

                You are a legal assistant tasked with understanding type of agreement and then combining various summaries and important information pieces from a legal agreement **(also includes templates of such, onclude ______, [ ] then follow that in summary as well, don't give empty results)** into a final, structured summary. The final summary should cover all the important aspects of the document, including the key dates, title of agreemnent, type of agreement,values, clauses, parties, laws, costs, taxes, bills, property related, values, goeverning laws,and any other relevant details that are critical to understanding the agreement. 
                ******Strictly extract all below points  ******
                - Title of Agreement: Title (E.g: Lease Agreement,Franchise Agreement,Purchase Agreement,etc.)
                - All dates mention in Agreement (E.g: Effective Date: date, Commencement Date: date, ,Start Date: date,Termination date: date, End date: , Signing date: date, Closing Date: date, Expire Date: date, Renewal Date: date, etc.) 
                - Type of Agreement (E.g: Lease, Purchase, Rental, Franchise, etc.)
                - Termination Clause (E.g: Termination due to breach,termination after term, ALL other termination clauses mentioned, etc.)
                - Term of Agreement(E.g: 5 years, 10 years 5 months,etc.)
                - Renewal Clause (E.g: Renewal after term, Renewal after 5 years, etc.)
                - Key Clauses (Give details of  Confidentiality, Indemnification, Governing Law, Dispute Resolution, Notice related (E.g: notice period),etc.)
                - Parties Involved (Agreement made between with their roles in it)
                - Any Third Party Involved and their roles.
                - Costs and Taxes (Give details of Deposit, Maintenance, Taxes, etc.)
                - Contarct Value (Give details of price,rent,annual rent, pruchase price,deposit,security deposit,fixed deposit,insurance related, Disclousure ,etc.) in detail.
                - Any other important information or any other clause mentioned. 
                
                *** Final Format of Summary should be in Paragraph format ONLY  ***

            """ 
            def submit_with_context(executor, fn, *args, **kwargs):
                # Capture the current OpenTelemetry context (which includes the parent span)
                parent_ctx = ot_context.get_current()
                def wrapped():
                    token = ot_context.attach(parent_ctx)
                    try:
                        return fn(*args, **kwargs)
                    finally:
                        ot_context.detach(token)
                return executor.submit(wrapped)


            
            # Split text into chunks of specified size
            chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

            # Summarize each chunk separately
            chunk_summaries = []
            with concurrent.futures.ThreadPoolExecutor() as executor:
                # future_to_chunk = {executor.submit(self.summarize_chunk, chunk): chunk for chunk in chunks}
                future_to_chunk = {submit_with_context(executor,self.summarize_chunk, chunk): chunk for chunk in chunks}
                
                for future in concurrent.futures.as_completed(future_to_chunk):
                    try:
                        chunk_summary = future.result()
                        chunk_summaries.append(chunk_summary)
                    except Exception as e:
                        self.logger.warning(self._log_message(f"Error summarizing chunk: {e}", "summarize_text"))

            chunking_time = time.perf_counter()
            self.logger.info(self._log_message(f"Chunking and summarizing each chunk took {chunking_time - start_time:.4f} seconds", "summarize_text"))

            # Combine summaries into a final summary
            combined_text = " ".join(chunk_summaries)
            
            final_summary_response = open_ai_llm_call(final_summary_prompt, combined_text, OPENAI_MODEL_NAME, 0.5, function_name, self.logger)
            final_summary_time = time.perf_counter()
            self.logger.info(self._log_message(f"Combining and final summary response time: {final_summary_time - chunking_time:.4f} seconds", "summarize_text"))
            
            return final_summary_response.strip()

        except Exception as e:
            self.logger.error(self._log_message(f"Error during summarization: {e}", "summarize_text"))
            return None


    @mlflow.trace(name="Doc Summarizer - Summarize Chunk")
    def summarize_chunk(self, chunk: str) -> str:
        function_name = "summarize_chunk"
        try:
            chunk_prompt = """
                You are a legal assistant specializing in the detailed summarization of legal agreements and contracts **(also includes templates of such, onclude ______, [ ] then follow that in summary as well, don't give empty results)** . You will understand the type of contract and give a detailed summary which will include all important aspects of a legal agreement. You can make use of belo points as well.
                - For Title, you will get it like (E.g: Lease Agreement, Purchase Agreement,Room Rental Agreement,etc.) 
                - Important dates and their meanings.
                - Give details of parties involved (names and their roles) and any third party involved.
                - Key clauses and their details.
                - Termination-related clauses, information all.
                - Contarct Type (Lease,Purchase,Rental, Franchise,etc.).
                - Contract value, deposit, taxes, maintenance, insurance, notice ,and other costs.
                - Any additional important information, such as parties involved, confidentiality terms, dispute resolution clauses, idemnification ,Disclousure,etc.
                - Include the detail and entire summary of sections mentioned.
                - Also, summarize the given context in a detailed manner.

                Ensure that you include a comprehensive summary of all major points from the document.Give summary in paragraph format ONLY.
                
            """
            user_prompt = f"###Chunk:\n\n\n{chunk}"
            start_time = time.perf_counter()
            response = open_ai_llm_call(chunk_prompt, user_prompt, OPENAI_MODEL_NAME, 0.2, function_name, self.logger)
            summary_time = time.perf_counter()
            self.logger.info(self._log_message(f"Summary of chunk of length {len(chunk)} chars took {summary_time - start_time:.4f} seconds", "summarize_chunk"))
            return response.strip()
        except Exception as e:
            self.logger.error(self._log_message(f"Error summarizing chunk: {e}", "summarize_chunk"))
            raise e
        
    @mlflow.trace(name="Doc Summarizer - Upsert to Pinecone")
    def upsert_to_pinecone(self, file_id, file_name, summary, org_id, user_id):
        start_time = time.perf_counter()
        function_name = "upsert_to_pinecone"
        try:
            self.logger.info(self._log_message(f"Upserting summary to Pinecone for {file_id}...", function_name))
            
            pinecone_index = get_pinecone_index(get_pinecone_client(PINECONE_API_KEY), DOCUMENT_SUMMARY_INDEX) 
            # Generate embedding from summary
            embedding = embeddings_generator.embed_query(summary)
            # -----------------------------------------------------------------------------
            calculate_tf_update_db_summary(org_id,user_id,file_id,[summary],self.logger)

            avgdl_value = get_avgdl_from_db_summary(file_id,self.logger)

            bm25 = CustomBM25Encoder()

            sparse_vector = bm25.custom_encode_documents(avgdl_value, [summary])
    
            # -----------------------------------------------------------------------------
            if embedding:
                metadata = {
                    'org_id': org_id,
                    'user_id': user_id,
                    'file_id': file_id,
                    'text': summary,
                    'file_name': file_name,
                    'char_count': len(summary)
                }
                upsert_result = pinecone_index.upsert(
                    [{"id": f"{org_id}#{file_id}", "values": embedding, "sparse_values": sparse_vector[0], "metadata": metadata}],
                    namespace="org_id_" + str(org_id) + "#"
                )
                self.logger.info(self._log_message(f"Upserted result for {file_id}: {upsert_result}", function_name))
            else:
                self.logger.error(self._log_message(f"Failed to generate embedding for {file_id}", function_name))
                
            end_time = time.perf_counter()
            self.logger.info(self._log_message(f"Time taken to upsert summary to Pinecone: {end_time - start_time:.4f} seconds", function_name))

        except Exception as e:
            self.logger.error(self._log_message(f"Error in upsert_to_pinecone: {e}", function_name))

# Example usage
# logger = YourLoggerImplementation()
# doc_summariser = DocSummariser(logger)
# summary = doc_summariser.summarize_text(file_id, user_id, org_id, document_text)
