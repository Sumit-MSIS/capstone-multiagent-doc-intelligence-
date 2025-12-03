import re
import time
from src.common.logger import _log_message
from src.config.base_config import config
from src.common.file_insights.doc_summariser.utils import save_markdown_to_db
import concurrent.futures
import mlflow

# mlflow.config.enable_async_logging()
mlflow.openai.autolog()
from opentelemetry import context as ot_context

OPENAI_API_KEY = config.OPENAI_API_KEY
OPENAI_MODEL_NAME = config.OPENAI_MODEL_NAME

MODULE_NAME = "file_parser.py"
class DocParser:

    def __init__(self, logger):
        self.logger = logger
        self.logger.info(self._log_message(f"Initializing DocumentConverter", "DocParser class"))

    def _log_message(self, message: str, function_name: str) -> str:
        return _log_message(message, function_name, MODULE_NAME)


    @mlflow.trace(name="Doc Parser - Split Markdown into Chunks")
    def split_markdown_into_chunks(self, markdown: str, chunk_size: int = 5196) -> list:
        """
        Chunks markdown without splitting headers and paragraphs.
        """
        function_name = "split_markdown_into_chunks"
        start_time = time.perf_counter()
        try:
            # Check if there are any headers starting with ##
            if '## ' not in markdown:
                chunks = [markdown[i:i+chunk_size] for i in range(0, len(markdown), chunk_size)]
                self.logger.info(self._log_message(f"Markdown split into {len(chunks)} chunks directly", function_name))
            else:
                # Split the markdown by headings
                sections = re.split(r'(## .+)', markdown)
                chunks = []
                current_chunk = ""
                
                for i in range(1, len(sections), 2):
                    heading = sections[i]
                    content = sections[i + 1]
                    
                    if len(current_chunk) + len(heading) + len(content) > chunk_size:
                        chunks.append(current_chunk)
                        current_chunk = heading + content
                    else:
                        current_chunk += heading + content
                
                if current_chunk:
                    chunks.append(current_chunk)
                
                self.logger.info(self._log_message(f"Markdown split into {len(chunks)} chunks by headings", function_name))
            
            end_time = time.perf_counter()
            self.logger.info(self._log_message(f"Time taken to split markdown: {end_time - start_time:.4f} seconds", function_name))
            
            return chunks
        except Exception as e:
            self.logger.error(self._log_message(f"Error splitting markdown into chunks: {e}", function_name))
            raise Exception(f"Error splitting markdown into chunks: {str(e)}")
    

    @mlflow.trace(name="Doc Parser - Generate Markdown Template")
    def generate_markdown_template(self, user_id: str, org_id: str, file_id: str, file_name: str, file_type: str, markdown_chunk: str, chunk_count: int) -> None:
        """
        Processes each markdown chunk to extract section name and store it.
        """
        function_name = "generate_markdown_template"
        start_time = time.perf_counter()
        try:
            print(f"Processing chunk {chunk_count} for file {file_name}...")
            self.logger.info(self._log_message(f"Total chunks - {chunk_count}", function_name))

            # Extract section name directly from markdown - look for lines starting with ##
            section_name = "Unknown"
            lines = markdown_chunk.split('\n')
            for line in lines:
                line = line.strip()
                if line.startswith('##'):
                    # Extract the section name without the ## prefix
                    section_name = line.lstrip('#').strip()
                    break

            # Set tags as empty string as requested
            tags_list = []

            extract_time = time.perf_counter()
            self.logger.info(self._log_message(f"Metadata extraction time: {extract_time - start_time:.4f} seconds", function_name))

            # Save results to the database
            save_markdown_to_db({
                'user_id': user_id,
                'org_id': org_id,
                'file_id': file_id,
                'file_name': file_name,
                'file_type': file_type
            }, markdown_chunk, "", section_name, chunk_count, None, self.logger)

            end_time = time.perf_counter()
            self.logger.info(self._log_message(f"Time taken to process chunk: {end_time - start_time:.4f} seconds", function_name))

        except Exception as e:
            self.logger.error(self._log_message(f"Unexpected error occurred during file parsing: {e}", function_name))
            raise Exception(f"Unexpected error occurred during file parsing: {str(e)}")
        
    
    @mlflow.trace(name="Doc Parser - Process File")
    def process_file(self, markdown_text, file_name, file_type, user_id, org_id, file_id):
        try:            
            
            t2 = time.perf_counter()
            # Convert the document to Markdown
            self.logger.info(self._log_message(f"Length of Markdown text: {len(markdown_text)}", "process_file"))
            
            # Split the Markdown text into chunks
            chunks = self.split_markdown_into_chunks(markdown_text)
            self.logger.info(self._log_message(f"Number of chunks: {len(chunks)}", "process_file"))
            
            self.logger.info(self._log_message(f"Text to Markdown conversion time: {t2 - time.perf_counter():.4f} seconds", "process_file"))
            
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

            # Process each chunk
            @mlflow.trace(name="Process File - Process Chunk")
            def process_chunk(chunk_count, chunk):
                try:
                    self.generate_markdown_template(user_id, org_id, file_id, file_name, file_type, chunk, chunk_count)
                except Exception as ex:
                    self.logger.error(self._log_message(f"Error processing chunk: {ex}", "process_chunk"))


            # Use ThreadPoolExecutor to process chunks in parallel
            with concurrent.futures.ThreadPoolExecutor() as executor:
                # futures = [executor.submit(process_chunk, chunk_count, chunk) for chunk_count, chunk in enumerate(chunks, start=1)]
                futures = [submit_with_context(executor,process_chunk, chunk_count, chunk) for chunk_count, chunk in enumerate(chunks, start=1)]

                # Wait for all threads to complete
                concurrent.futures.wait(futures)

            return markdown_text
            
        except Exception as e:
            self.logger.error(self._log_message(f"Unexpected error during processing file: {e}", "process_file"))
            raise
