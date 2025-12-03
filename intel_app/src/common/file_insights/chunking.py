from typing import List
import tiktoken
from langchain_openai import OpenAIEmbeddings
from src.config.base_config import config
from src.common.logger import _log_message
import re
import time
MODULE_NAME = "chunking.py"
OPENAI_API_KEY = config.OPENAI_API_KEY
OPENAI_EMBEDING_MODEL_NAME = config.OPENAI_EMBEDING_MODEL_NAME

embeddings_generator = OpenAIEmbeddings(
    openai_api_key=OPENAI_API_KEY,
    model=OPENAI_EMBEDING_MODEL_NAME
)

class ClusterSemanticChunker:
    def __init__(self, logger=None, chunk_size=800, overlap=400):
        """
        Initializes the ClusterSemanticChunker with token-based chunking.
        """
        self.logger = logger
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.encoding = tiktoken.encoding_for_model(OPENAI_EMBEDING_MODEL_NAME)

    def tokenize_text(self, text: str) -> List[int]:
        """
        Converts text into a list of token ids.
        """
        return self.encoding.encode(text)

    def detokenize_text(self, tokens: List[int]) -> str:
        """
        Converts a list of token ids back to text.
        """
        return self.encoding.decode(tokens)

    def create_chunks(self, text: str) -> List[str]:
        """
        Splits text into overlapping chunks based on token count.
        """
        try:
            start_time = time.time()
              
            tokens = self.tokenize_text(text)
            self.logger.info(f"Total tokens in text: {len(tokens)}")
            chunks = []

            start = 0
            while start < len(tokens):
                end = min(start + self.chunk_size, len(tokens))
                chunk_tokens = tokens[start:end]
                chunk_text = self.detokenize_text(chunk_tokens)
                chunks.append(chunk_text)
                start += self.chunk_size - self.overlap  # Slide with overlap

            self.logger.info(f"Created {len(chunks)} chunks.")
            self.logger.info(f"Chunking completed in {time.time() - start_time:.2f} seconds.")
            return chunks

        except Exception as e:
            self.logger.error(f"Error during chunking: {e}")
            return []
            
    
# chunker = ClusterSemanticChunker()