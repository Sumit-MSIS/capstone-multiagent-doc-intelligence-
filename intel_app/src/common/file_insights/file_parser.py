import logging
import os
import time
from datetime import datetime
from src.config.base_config import config
from src.common.file_insights.chunking import ClusterSemanticChunker
from src.common.llm_status_handler.status_handler import set_llm_file_status
from src.common.logger import _log_message
from typing import List, Dict, Any
import re
from functools import partial
from unstructured.partition.csv import partition_csv
from unstructured.partition.docx import partition_docx
from unstructured.partition.epub import partition_epub
from unstructured.partition.html import partition_html
from unstructured.partition.image import partition_image
from unstructured.partition.odt import partition_odt
from unstructured.partition.pdf import partition_pdf
from unstructured.partition.pptx import partition_pptx
from unstructured.partition.text import partition_text
from unstructured.partition.tsv import partition_tsv
from unstructured.partition.xlsx import partition_xlsx
import mlflow
# from unstructured.documents.elements import Element  # for isinstance checks
import orjson
import mlflow
from opentelemetry import context as ot_context
import psutil

from unstructured_client import UnstructuredClient
import fitz  # PyMuPDF
import nltk
from functools import lru_cache

# nltk.download('punkt_tab')

nltk.data.path.append("/usr/local/nltk_data")
# mlflow.config.enable_async_logging()
# mlflow.langchain.autolog()
mlflow.openai.autolog()

#Get Module Name
MODULE_NAME = "file_parser.py"


UNSTRUCTURED_API_KEY = config.UNSTRUCTURED_API_KEY

# Initialize client with API key
unstructured_client = UnstructuredClient(
        api_key_auth=UNSTRUCTURED_API_KEY
)

PROCESS_MAP = {
    '.html': partition_html,
    '.pptx': partition_pptx,
    '.eml': partition_text, '.md': partition_text, '.msg': partition_text,
    '.rst': partition_text, '.rtf': partition_text, '.txt': partition_text, '.xml': partition_text,
    '.png': partition_image, '.jpg': partition_image, '.jpeg': partition_image,
    '.tiff': partition_image, '.bmp': partition_image, '.heic': partition_image,
    '.csv': partition_csv,
    '.doc': partition_docx, '.docx': partition_docx,
    '.epub': partition_epub,
    '.odt': partition_odt,
    '.tsv': partition_tsv,
    '.xlsx': partition_xlsx,
    '.pdf': lambda f: partition_pdf(f, strategy="hi_res", hi_res_model_name="yolox"),
    'pdf': lambda f: partition_pdf(f, strategy="hi_res",  hi_res_model_name="yolox")


}
class FileParser:
    
    def __init__(self, logger: logging.Logger) -> None:
        """
        Initializes the FileProcessor instance.

        :param logger: Logger instance for logging purposes.
        """
        self.logger = logger
        self.OPENAI_API_KEY = config.OPENAI_API_KEY
        self.EMBEDING_MODEL_NAME = config.OPENAI_EMBEDING_MODEL_NAME
        self.TEMP_DIR = config.TEMP_DIR
        if not os.path.exists(self.TEMP_DIR):
            os.makedirs(self.TEMP_DIR, exist_ok=True)

        self.in_queue = False
        # self.process_map = {
        #     '.html': partition_html,
        #     '.pptx': partition_pptx,
        #     '.eml': partition_text, '.md': partition_text, '.msg': partition_text,
        #     '.rst': partition_text, '.rtf': partition_text, '.txt': partition_text, '.xml': partition_text,
        #     '.png': partition_image, '.jpg': partition_image, '.jpeg': partition_image,
        #     '.tiff': partition_image, '.bmp': partition_image, '.heic': partition_image,
        #     '.csv': partition_csv,
        #     '.doc': partition_docx, '.docx': partition_docx,
        #     '.epub': partition_epub,
        #     '.odt': partition_odt,
        #     '.tsv': partition_tsv,
        #     '.xlsx': partition_xlsx,
        #     '.pdf': lambda f: partition_pdf(f, strategy="hi_res", infer_table_structure=True, model_name="yolox"),
        #     'pdf': lambda f: partition_pdf(f, strategy="hi_res", infer_table_structure=True, model_name="yolox")
        # }


    @mlflow.trace(name="File Parser - Unstructured File Parser")
    def _unstructured_file_parser_paid(self, file_path, file_type):
        """Parse a document using UnstructuredClient and return elements."""
        
        try:
            if file_path.lower().endswith(".pdf") or file_type in ["pdf", ".pdf"]:
                with open(file_path, "rb") as file:
                    file_content = file.read()
                    # Prepare request parameters
                    request = {
                        "partition_parameters": {
                            "files": {
                                "content": file_content,
                                "file_name": os.path.basename(file_path)
                            },
                            "strategy": "hi_res",
                            "split_pdf_page": True,
                            "split_pdf_allow_failed": False,
                            "split_pdf_concurrency_level": 15
            
                        }
                    }
                    
                    # Make API call
                    response = unstructured_client.general.partition(request=request)
                    
                    # Return elements 
                    return response.elements
                
                
                
            # Handle plain text files (.txt)
            elif file_path.lower().endswith(".txt") or file_type in ["txt", ".txt"]:
                with open(file_path, "r", encoding="utf-8") as file:
                    text = file.read()

                class TextElement:
                    def __init__(self, text, type_="NarrativeText", metadata=None):
                        self.text = text
                        self.type = type_
                        self.metadata = metadata or {}

                # Return in a format consistent with other file types (e.g., list of dicts or strings)
                return [TextElement(text=text, metadata={"filename": os.path.basename(file_path)})]

            else:
                # Handle other formats as a single file
                with open(file_path, "rb") as file:
                    file_content = file.read()
                    request = {
                        "partition_parameters": {
                            "files": {
                                "content": file_content,
                                "file_name": os.path.basename(file_path)
                            },
                            "strategy": "hi_res",
                        }
                    }
                    response = unstructured_client.general.partition(request=request)
                    return response.elements
        except Exception as e:
            self.logger.error(_log_message(f"Error during unstructured text extraction {file_path}: {e}", "unstructured_file_parser_paid", MODULE_NAME))
            raise e
    
    @mlflow.trace(name="File Parser - Text Extraction from Unstructured Elements Paid")
    def _extract_text_from_paid_unstructured_elements(self, unstructured_elements):
        """
        Extract text from unstructured elements represented as dictionaries.
        
        Parameters:
        - unstructured_elements: List of dictionaries returned from unstructured parser
        
        Returns:
        - String containing the extracted text
        """
        texts = []
        
        for element in unstructured_elements:
            # For dictionaries, we directly access the 'text' key
            if 'text' in element:
                texts.append(element['text'])
            # Fallback if 'text' is not a direct key but nested in metadata
            elif 'metadata' in element and 'text' in element['metadata']:
                texts.append(element['metadata']['text'])
        
        # Join all texts with double newline separator
        return "\n\n".join(texts)
    
    @mlflow.trace(name="File Parser - Text Extraction from Unstructured Elements")
    def _extract_text_from_elements(self, unstructured_elements):
        """
        Extract text from unstructured element objects.
        
        Parameters:
        - unstructured_elements: List of element objects from unstructured library
        
        Returns:
        - String containing the extracted text
        """
        texts = [result.text for result in unstructured_elements if result.text.strip() != ""]


        # [result.text for result in results if result.text.strip() != ""]


        
        # for element in unstructured_elements:
        #     # For dictionaries, we directly access the 'text' key
        #     if 'text' in element:
        #         texts.append(element['text'])
        #     # Fallback if 'text' is not a direct key but nested in metadata
        #     elif 'metadata' in element and 'text' in element['metadata']:
        #         texts.append(element['metadata']['text'])
        
        # Join all texts with double newline separator
        return "\n\n".join(texts)
    
    @mlflow.trace(name="File Parser - TOC from Unstructured Elements")
    def extract_toc_from_elements(self, elements):
        """
        Extract a table of contents directly from unstructured element objects.
        Identifies section headers based on formatting and patterns.
        
        Parameters:
        - elements: List of element objects from unstructured library
        
        Returns:
        - Structured TOC in the required format
        """
        toc = []
        current_section = None
        
        # Expanded patterns to identify main sections
        main_section_patterns = [
            # Basic section formats
            r'^([A-Z]\.\s+.+)$',               # Matches "A. Section Name"
            r'^([0-9]+\.\s+.+)$',              # Matches "1. Section Name"
            r'^(Policy\s+Intent.+)$',          # Matches "Policy Intent" heading
            r'^([A-Z][A-Z\s]+):?\s*$',         # Matches "SECTION TITLE:" format
            
            # Roman numerals
            r'^(?:ARTICLE|Section)?\s*((?:IX|IV|V?I{0,3})\.\s+.+)$',  # Matches "I. Section", "IV. Section"
            r'^(?:IX|IV|V?I{0,3})\.\s+(.+)$',  # Plain Roman numerals
            r'^(?:ix|iv|v?i{0,3})\.\s+(.+)$',  # Lowercase Roman numerals
            
            # Common section identifiers
            r'^(?:Section|SECTION|Article|ARTICLE|Chapter|CHAPTER)\s+([0-9A-Z]+[\.\:].+)$',  # Matches "Section 1: Title"
            r'^(?:Section|SECTION|Article|ARTICLE|Chapter|CHAPTER)\s+([0-9A-Z]+)\s+(.+)$',  # Matches "Section 1 Title"
            r'^(?:Section|SECTION|Article|ARTICLE|Chapter|CHAPTER)\s+([0-9A-Z]+)$',  # Just "Section 1" alone
            
            # Word-based sections
            r'^(?:PART|Part)\s+(?:ONE|one|Two|TWO|Three|THREE|Four|FOUR|Five|FIVE)\s*[\.\:]\s*(.+)$',  # "Part One: Title"
            r'^(?:PART|Part)\s+([A-Z])\s*[\.\:]\s*(.+)$',  # "Part A: Title"
            
            # Common standalone headings
            r'^(?:(?:INTRODUCTION|Introduction|EXECUTIVE SUMMARY|Executive Summary|BACKGROUND|Background|CONCLUSION|Conclusion|DEFINITIONS|Definitions|OVERVIEW|Overview|SCOPE|Scope|PURPOSE|Purpose|ELIGIBILITY|Eligibility|REQUIREMENTS|Requirements|METHODOLOGY|Methodology|RECOMMENDATIONS|Recommendations|APPENDIX|Appendix|TERMS|Terms|CONDITIONS|Conditions|ASSUMPTIONS|Assumptions|OBJECTIVES|Objectives|SUMMARY|Summary|ABSTRACT|Abstract)(?:\s*\:|\.)?\s*.*)$',
            
            # Parenthetical numbering
            r'^\(([0-9]+)\)\s+(.+)$',          # Matches "(1) Section Name"
            r'^\(([A-Z])\)\s+(.+)$',           # Matches "(A) Section Name"
            
            # Hash/pound headings (markdown style)
            r'^#+\s+(.+)$',                    # Matches "# Section Title", "## Section Title"
            
            # Title case standalone sections (likely to be headings)
            r'^([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,7})$',  # Title Case Words (2-8 words)
            
            # Numbered with dash
            r'^([0-9]+)\s*-\s+(.+)$',          # Matches "1 - Section Title"
            r'^([A-Z])\s*-\s+(.+)$',           # Matches "A - Section Title"
            
            # Alphanumeric codes
            r'^([A-Z][0-9]{1,3})\s+(.+)$',     # Matches "A1 Title", "B22 Title"
            r'^([0-9]{1,2}[A-Z])\s+(.+)$',     # Matches "1A Title", "22B Title"
            
            # Numbered paragraphs
            r'^(?:Para|Paragraph|PARAGRAPH)\s+([0-9]+)[\.\:]?\s*(.+)?$',  # "Paragraph 1: Title"
            
            # Law/regulation specific
            r'^(?:Clause|CLAUSE)\s+([0-9]+)[\.\:]?\s*(.+)?$',  # "Clause 1: Title"
            r'^(?:Rule|RULE)\s+([0-9]+)[\.\:]?\s*(.+)?$',      # "Rule 1: Title"
            r'^(?:Regulation|REG)\s+([0-9]+)[\.\:]?\s*(.+)?$',  # "Regulation 1: Title"
            
            # Academic document sections
            r'^(?:Chapter|CHAPTER)\s+([0-9]+)[\.\:]?\s*(.+)?$',  # "Chapter 1: Title"
            r'^(?:Module|MODULE)\s+([0-9]+)[\.\:]?\s*(.+)?$',    # "Module 1: Title"
            r'^(?:Unit|UNIT)\s+([0-9]+)[\.\:]?\s*(.+)?$',        # "Unit 1: Title"
            
            # Contract sections
            r'^(?:Schedule|SCHEDULE)\s+([0-9A-Z]+)[\.\:]?\s*(.+)?$',  # "Schedule A: Title"
            r'^(?:Exhibit|EXHIBIT)\s+([0-9A-Z]+)[\.\:]?\s*(.+)?$',    # "Exhibit A: Title"
            r'^(?:Annex|ANNEX)\s+([0-9A-Z]+)[\.\:]?\s*(.+)?$',        # "Annex A: Title"
            r'^(?:Appendix|APPENDIX)\s+([0-9A-Z]+)[\.\:]?\s*(.+)?$',  # "Appendix A: Title"
        ]

        # Expanded patterns to identify subsections with modifications to prevent single-character matches
        subsection_patterns = [
            # Basic subsection formats
            r'^([A-Z][0-9]+\.\s+.+)$',         # Matches "A1. Subsection"
            r'^([0-9]+\.[0-9]+\s+.+)$',        # Matches "1.1 Subsection"
            r'^\s*•\s+(.+)$',                  # Matches bullet points
            r'^\s*\[\*\]\s*(.*)$',             # Matches [*] markers
            r'^\s*-\s+(.+)$',                  # Matches dash bullet points
            
            # Complex hierarchical numbering
            r'^([0-9]+\.[0-9]+\.[0-9]+(?:\.[0-9]+)*)\.?\s+(.+)$',  # Matches "1.1.1 Title", "1.2.3.4 Title"
            r'^([0-9]+[a-z])\.\s+(.+)$',       # Matches "1a. Title"
            r'^([0-9]+\.[a-z])\.\s+(.+)$',     # Matches "1.a. Title"
            r'^([0-9]+\.[0-9]+[a-z])\.\s+(.+)$',  # Matches "1.1a. Title"
            
            # Parenthetical lettering/numbering - now requiring content
            r'^\(([a-z])\)\s+(.+)$',           # Matches "(a) Subsection with content"
            r'^\(([0-9]+)\)\s+(.+)$',          # Matches "(1) Subsection with content"
            r'^\(([a-z]+)\)\s+(.+)$',          # Matches "(iv) Subsection with content"
            r'^\(([0-9]+[a-z])\)\s+(.+)$',     # Matches "(1a) Subsection with content"
            r'^\(([a-z][0-9]+)\)\s+(.+)$',     # Matches "(a1) Subsection with content"
            
            # More bullet types
            r'^\s*\*\s+(.+)$',                 # Matches asterisk bullets
            r'^\s*\+\s+(.+)$',                 # Matches plus sign bullets
            r'^\s*[→>➢➤▶︎◆◇■□●○]\s*(.+)$',     # Matches various symbol bullets
            r'^\s*(?:✓|✔|☑|☒)\s*(.+)$',        # Matches checkmark bullets
            
            # Letter-based subsections - modified to require content after the letter
            r'^\s*([a-z])\.\s+(.{3,})$',       # Matches "a. Subsection" with at least 3 chars of content
            r'^\s*([a-z])\)\s+(.{3,})$',       # Matches "a) Subsection" with at least 3 chars of content
            r'^\s*([A-Z])\.\s+(.{3,})$',       # Matches "A. Subsection" with at least 3 chars of content
            r'^\s*([A-Z])\)\s+(.{3,})$',       # Matches "A) Subsection" with at least 3 chars of content
            
            # Roman numeral subsections
            r'^\s*(i{1,3}|iv|v|vi{1,3}|ix|x|xi{1,3}|xi{0,2}v|xv|xvi{1,3})\.\s+(.+)$',  # More Roman numerals
            r'^\s*\((i{1,3}|iv|v|vi{1,3}|ix|x)\)\s+(.+)$',  # Matches "(i) Subsection"
            
            # Indented subsections (based on whitespace)
            r'^\s{2,}([A-Z][a-z].+)$',         # Matches indented text that starts with capital letter
            r'^\t+([A-Z][a-z].+)$',            # Matches tab-indented text
            
            # Specialized formats
            r'^Sub-section\s+(.+)$',           # Matches "Sub-section Title"
            r'^Subsection\s+([0-9.]+)[\.\:]\s*(.+)$',  # Matches "Subsection 1.1: Title"
            r'^Item\s+([0-9]+)[\.\:]\s*(.+)$', # Matches "Item 1: Title"
            r'^Note\s*[\:\-]?\s*(.+)$',        # Matches "Note: Content"
            r'^Example\s*[\:\-]?\s*(.+)$',     # Matches "Example: Content"
            r'^Case Study\s*[\:\-]?\s*(.+)$',  # Matches "Case Study: Content"
            
            # Square and other brackets
            r'^\s*\[([^*].*?)\]',              # Matches "[Title]" but not "[*]"
            r'^\s*\{(.+?)\}',                  # Matches "{Title}"
            r'^\s*\<(.+?)\>',                  # Matches "<Title>"
            
            # Common content dividers
            r'^Details\s*[\:\-]?\s*(.+)$',     # Matches "Details: content"
            r'^Summary\s*[\:\-]?\s*(.+)$',     # Matches "Summary: content"
            r'^Description\s*[\:\-]?\s*(.+)$', # Matches "Description: content"
            r'^Specification\s*[\:\-]?\s*(.+)$', # Matches "Specification: content"
        ]
        
        # List to track single letters that should be combined with the next element
        pending_letter_prefix = None
        
        # Function to check if a subsection title should be excluded
        def should_exclude_subsection(title):
            """
            Check if a subsection title should be excluded from the TOC.
            
            Args:
                title: The title of the subsection
            
            Returns:
                True if the title should be excluded, False otherwise
            """
            # Check for "PAGE" entries
            if title.strip() == "PAGE":
                return True
            
            # Check for number with dash format (e.g., "2-", "3-", etc.)
            if re.match(r'^\d+-$', title.strip()):
                return True
            
            # Check for single characters or very short entries
            if len(title.strip()) <= 2:
                return True
            
            # Check for horizontal lines or separators
            if re.match(r'^-+$', title.strip()):
                return True
            
            # Check for entries that start with "= omitted"
            if title.strip().startswith("= omitted"):
                return True
            
            return False
            
        # Function to extract just the heading portion from section text
        def extract_heading_from_text(text):
            """
            Extract just the heading portion from a text that may include a full paragraph.
            Uses structural patterns and length-based heuristics without domain-specific terms.
            
            Args:
                text: The full text that may include heading and paragraph content
            
            Returns:
                The extracted heading
            """
            # If text is shorter than 60 characters, just return it as is
            if len(text) <= 60:
                return text
                
            # CASE 1: Numbered sections with a clear heading part 
            # Match patterns like "1. Title" or "1.2. Title" and extract until the first sentence boundary
            numbered_section_match = re.match(r'^(\d+\.(?:\d+\.)*)(?:\s+)(.+?)(?:\.|\:|\,|\;|\()', text)
            if numbered_section_match:
                prefix = numbered_section_match.group(1)  # The number with dot
                heading_text = numbered_section_match.group(2)  # The text until first punctuation
                
                # Extract just the section title, not the full content
                return f"{prefix} {heading_text}"
            
            # CASE 2: Match just number and first 1-3 words for numbered sections (generic approach)
            if re.match(r'^\d+\.', text):
                words = text.split()
                # Calculate the number of words to include (at least the number and 2-3 more words)
                word_count = min(4, len(words))
                return ' '.join(words[:word_count]) + ('...' if len(words) > word_count else '')
            
            # CASE 3: For any text, extract until first major punctuation within a reasonable distance
            first_sentence = re.match(r'^(.{10,100}?)(?:\.|\:|\;|\?)', text)
            if first_sentence:
                heading_text = first_sentence.group(1).strip()
                # If heading is still long, truncate it
                if len(heading_text) > 60:
                    words = heading_text.split()
                    return ' '.join(words[:5]) + '...'
                return heading_text
                
            # CASE 4: For all other text, just take first few words (aggressive truncation)
            words = text.split()
            if len(words) > 5:
                return ' '.join(words[:5]) + '...'
                
            # Default: return the text as is if it's reasonably short
            return text
        
        for element in elements:
            # Get element type from class name instead of dictionary access
            element_type = type(element).__name__
            
            # Use getattr to safely access the text attribute
            text = getattr(element, 'text', '').strip() if hasattr(element, 'text') else ''
            
            # Skip empty text
            if not text:
                continue
                
            # Check for emphasis or formatting that indicates headings
            is_emphasized = False
            
            # Get metadata if available
            metadata = getattr(element, 'metadata', None)
            
            # Check if text is emphasized (bold, etc.)
            if metadata and hasattr(metadata, 'emphasized_text_contents'):
                emphasized_text_contents = getattr(metadata, 'emphasized_text_contents', [])
                if emphasized_text_contents:
                    emphasized_text = ''.join(emphasized_text_contents)
                    if emphasized_text and (emphasized_text in text or text.startswith(emphasized_text)):
                        is_emphasized = True
            
            # Check if this element is a main section heading
            is_main_section = False
            
            # First, check explicit patterns
            for pattern in main_section_patterns:
                match = re.match(pattern, text)
                if match:
                    is_main_section = True
                    break
                    
            # Then consider emphasized text that might be section headers
            if (is_emphasized and 
                (text.startswith('Policy') or 
                any(text.startswith(f"{letter}.") for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ") or
                element_type in ['Title', 'UncategorizedText'])):
                is_main_section = True
            
            # Create new section if this is a main section
            if is_main_section:
                # Clean up the section name (remove excessive whitespace, etc.)
                text_clean = ' '.join(text.split())
                
                # Extract just the heading portion, not the full paragraph
                section_name = extract_heading_from_text(text_clean)
                
                current_section = {
                    "section_name": section_name,
                    "sub_sections": []
                }
                toc.append(current_section)
                pending_letter_prefix = None  # Reset any pending letter
                continue
            
            # Check if this is a single letter that might be part of a list (a, b, c, etc.)
            is_single_letter = re.match(r'^[a-zA-Z]$', text)
            if is_single_letter:
                pending_letter_prefix = text
                continue
                
            # Check for letter followed by a period or parenthesis
            is_letter_marker = re.match(r'^([a-zA-Z])[\.|\)]$', text)
            if is_letter_marker:
                pending_letter_prefix = is_letter_marker.group(1)
                continue
            
            # Check for subsections
            is_subsection = False
            subsection_text = text
            
            # If we have a pending letter and this line could be its content
            if pending_letter_prefix and len(text.strip()) > 0:
                subsection_text = f"{pending_letter_prefix}. {text}"
                is_subsection = True
                pending_letter_prefix = None  # Reset the pending letter
            else:
                # Try matching subsection patterns
                for pattern in subsection_patterns:
                    match = re.match(pattern, text)
                    if match and match.groups():
                        # For letter-based patterns with 2 groups, use the second group (the content)
                        if len(match.groups()) > 1 and re.match(r'^[a-zA-Z]$', match.group(1)):
                            subsection_text = match.group(2).strip()
                        else:
                            subsection_text = match.group(1).strip()
                        is_subsection = True
                        break
                        
                # Special cases for bullet points and [*] markers
                if text.startswith('•') or text.startswith('[*]') or text.startswith('-'):
                    is_subsection = True
                    # Clean up the subsection text
                    subsection_text = text.lstrip('•').lstrip('[*]').lstrip('-').strip()
                    if not subsection_text:  # If it was just a marker with no content, use the whole text
                        subsection_text = text
                
                # Special case for tables that might contain subsection-like content
                if element_type == 'Table' and '•' in text:
                    parts = text.split('•', 1)
                    if len(parts) > 1:
                        subsection_text = parts[1].strip()
                        is_subsection = True
            
            # Add subsection to current section if applicable, but filter out unwanted subsections
            if is_subsection and current_section is not None and subsection_text:
                # Clean up the subsection title
                clean_title = ' '.join(subsection_text.split())
                
                # Check if subsection should be excluded
                if should_exclude_subsection(clean_title):
                    continue
                
                # Avoid duplicates and meaningless subsections
                if not any(sub['title'] == clean_title for sub in current_section['sub_sections']):
                    current_section["sub_sections"].append({"title": clean_title})
        
        # Return just the TOC list without metadata wrapper
        
        return toc
    
    @mlflow.trace(name="File Parser - TOC from Unstructured Elements Paid")
    def _extract_toc_from_paid_unstructured_elements(self, unstructured_elements):
        """
        Extract a table of contents directly from unstructured elements.
        Identifies section headers based on formatting and patterns.
        
        Parameters:
        - elements: List of dictionaries from unstructured parser
        
        Returns:
        - Structured TOC in the required format
        """
        try:
            self.logger.info(self._log_message("Extracting TOC from unstructured elements", MODULE_NAME))
            toc = []
            current_section = None
            
            # Expanded patterns to identify main sections
            main_section_patterns = [
                # Basic section formats
                r'^([A-Z]\.\s+.+)$',               # Matches "A. Section Name"
                r'^([0-9]+\.\s+.+)$',              # Matches "1. Section Name"
                r'^(Policy\s+Intent.+)$',          # Matches "Policy Intent" heading
                r'^([A-Z][A-Z\s]+):?\s*$',         # Matches "SECTION TITLE:" format
                
                # Roman numerals
                r'^(?:ARTICLE|Section)?\s*((?:IX|IV|V?I{0,3})\.\s+.+)$',  # Matches "I. Section", "IV. Section"
                
                # Common section identifiers
                r'^(?:Section|SECTION|Article|ARTICLE|Chapter|CHAPTER)\s+([0-9A-Z]+[\.\:].+)$',  # Matches "Section 1: Title"
                r'^(?:Section|SECTION|Article|ARTICLE|Chapter|CHAPTER)\s+([0-9A-Z]+)\s+(.+)$',  # Matches "Section 1 Title"
                
                # Word-based sections
                r'^(?:PART|Part)\s+(?:ONE|one|Two|TWO|Three|THREE|Four|FOUR|Five|FIVE)\s*[\.\:]\s*(.+)$',  # "Part One: Title"
                
                # Common standalone headings
                r'^(?:(?:INTRODUCTION|Introduction|EXECUTIVE SUMMARY|Executive Summary|BACKGROUND|Background|CONCLUSION|Conclusion|DEFINITIONS|Definitions|OVERVIEW|Overview|SCOPE|Scope|PURPOSE|Purpose)(?:\s*\:|\.)?\s*.*)$',
                
                # Parenthetical numbering
                r'^\(([0-9]+)\)\s+(.+)$',          # Matches "(1) Section Name"
                
                # Hash/pound headings (markdown style)
                r'^#+\s+(.+)$',                    # Matches "# Section Title", "## Section Title"
                
                # Title case standalone sections (likely to be headings)
                r'^([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,5})$',  # Title Case Words (2-6 words)
                
                # Numbered with dash
                r'^([0-9]+)\s*-\s+(.+)$',          # Matches "1 - Section Title"
                
                # Alphanumeric codes
                r'^([A-Z][0-9]{1,3})\s+(.+)$',     # Matches "A1 Title", "B22 Title"
            ]

            # Expanded patterns to identify subsections
            subsection_patterns = [
                # Basic subsection formats
                r'^([A-Z][0-9]+\.\s+.+)$',         # Matches "A1. Subsection"
                r'^([0-9]+\.[0-9]+\s+.+)$',        # Matches "1.1 Subsection"
                r'^\s*•\s+(.+)$',                  # Matches bullet points
                r'^\s*\[\*\]\s*(.*)$',             # Matches [*] markers
                r'^\s*-\s+(.+)$',                  # Matches dash bullet points
                
                # Complex hierarchical numbering
                r'^([0-9]+\.[0-9]+\.[0-9]+(?:\.[0-9]+)*)\.?\s+(.+)$',  # Matches "1.1.1 Title", "1.2.3.4 Title"
                r'^([0-9]+[a-z])\.\s+(.+)$',       # Matches "1a. Title"
                r'^([0-9]+\.[a-z])\.\s+(.+)$',     # Matches "1.a. Title"
                
                # Parenthetical lettering/numbering
                r'^\(([a-z])\)\s+(.+)$',           # Matches "(a) Subsection"
                r'^\(([0-9]+)\)\s+(.+)$',          # Matches "(1) Subsection"
                r'^\(([a-z]+)\)\s+(.+)$',          # Matches "(iv) Subsection"
                
                # More bullet types
                r'^\s*\*\s+(.+)$',                 # Matches asterisk bullets
                r'^\s*\+\s+(.+)$',                 # Matches plus sign bullets
                r'^\s*[→>➢➤▶︎]\s*(.+)$',           # Matches arrow bullets
                
                # Letter-based subsections
                r'^\s*([a-z])\.\s+(.+)$',          # Matches "a. Subsection"
                r'^\s*([a-z])\)\s+(.+)$',          # Matches "a) Subsection"
                
                # Roman numeral subsections
                r'^\s*(?:i{1,3}|iv|v|vi{1,3}|ix|x)\.\s+(.+)$',  # Matches "i. Subsection", "iv. Subsection"
                
                # Indented subsections (based on whitespace)
                r'^\s{2,}([A-Z][a-z].+)$',         # Matches indented text that starts with capital letter
                
                # Specialized formats
                r'^Sub-section\s+(.+)$',           # Matches "Sub-section Title"
                r'^Item\s+([0-9]+)[\.\:]\s*(.+)$', # Matches "Item 1: Title"
                r'^Note\s*[\:\-]?\s*(.+)$',        # Matches "Note: Content"
                
                # Square brackets
                r'^\s*\[([^*].*?)\]',              # Matches "[Title]" but not "[*]"
                
                # Common content dividers
                r'^Details\s*[\:\-]?\s*(.+)$',     # Matches "Details: content"
                r'^Summary\s*[\:\-]?\s*(.+)$',     # Matches "Summary: content"
            ]
            

            # Function to check if a text represents a paragraph or sentence rather than a heading
            # @mlflow.trace(name="TOC - Validate Paragraph or Sentence")
            def is_paragraph_or_sentence(text):
                """
                Check if the text appears to be a paragraph or complete sentence rather than a heading.
                
                Args:
                    text: The text to check
                    
                Returns:
                    True if the text appears to be a paragraph or complete sentence, False otherwise
                """
                text = text.strip()
                
                # Check if text is too long to be a heading (more than 100 characters)
                if len(text) > 100:
                    return True
                    
                # Count the number of words - headings are typically shorter
                word_count = len(text.split())
                if word_count > 15:  # Headings rarely have more than 15 words
                    return True
                    
                # Check for multiple sentences (periods followed by space and capital letter)
                if re.search(r'\.\s+[A-Z]', text):
                    return True
                    
                # Check for common sentence-ending punctuation patterns
                if re.search(r'[.!?]\s*$', text) and not re.match(r'^[0-9]+\.[0-9]*\s', text):  # Avoid matching decimal numbers
                    # Make sure it's not just a numbered heading that ends with a period
                    if not re.match(r'^[0-9A-Z]+\.\s+.+\.$', text):
                        return True
                        
                # Check for conjunctions and other indicators of full sentences
                sentence_indicators = [' and ', ' or ', ' but ', ' because ', ' however ', ' therefore ', 
                                    ' thus ', ' in addition ', ' furthermore ', ' moreover ', 
                                    ' consequently ', ' as a result ', ' for example ']
                for indicator in sentence_indicators:
                    if indicator in text.lower():
                        return True
                        
                # Check for verbs common in full sentences but rare in headers
                sentence_verbs = [' is ', ' are ', ' was ', ' were ', ' has ', ' have ', ' had ', 
                                ' can ', ' could ', ' will ', ' would ', ' should ', ' may ', 
                                ' might ', ' must ', ' shall ', ' should ', ' being ']
                for verb in sentence_verbs:
                    if verb in ' ' + text.lower() + ' ':
                        # Only count if not part of a common heading pattern
                        if not re.match(r'^(How|What|When|Where|Why|Who)\s', text):
                            return True
                
                # Check for personal pronouns common in sentences
                if re.search(r'\b(I|we|you|he|she|they|them|us|our|your|their)\b', text.lower()):
                    return True
                    
                # Check for commas, which often indicate lists or complex sentences
                if text.count(',') > 1:
                    return True
                    
                # Check for parenthetical expressions within the text (not at beginning)
                if re.search(r'.+\(.+\).+', text):
                    return True
                    
                return False
            

            # Function to check if a subsection title should be excluded
            # @mlflow.trace(name="TOC - Validate Exclude Subsection")
            def should_exclude_subsection(title):
                """
                Check if a subsection title should be excluded from the TOC.
                
                Args:
                    title: The title of the subsection
                
                Returns:
                    True if the title should be excluded, False otherwise
                """
                # Check for empty or very short titles
                if not title or len(title.strip()) <= 2:
                    return True
                    
                # Check for "PAGE" entries
                if title.strip() == "PAGE":
                    return True
                
                # Check for number with dash format (e.g., "2-", "3-", etc.)
                if re.match(r'^\d+-$', title.strip()):
                    return True
                
                # Check for horizontal lines or separators
                if re.match(r'^-+$', title.strip()):
                    return True
                
                # Check for entries that start with "= omitted"
                if title.strip().startswith("= omitted"):
                    return True
                    
                # Check if the title appears to be a paragraph or sentence
                if is_paragraph_or_sentence(title):
                    return True
                    
                # Check for very common words that are likely not headings
                common_words = ["the", "a", "an", "and", "or", "but", "if", "then", "so"]
                if title.strip().lower() in common_words:
                    return True
                    
                # Check for titles that are just one word and not capitalized (unless it's a number)
                words = title.strip().split()
                if len(words) == 1 and not words[0][0].isupper() and not words[0][0].isdigit():
                    return True
                    
                # Check for titles with excessive punctuation (likely not a heading)
                punctuation_count = sum(1 for char in title if char in ',.;:!?()[]{}')
                if punctuation_count > 3:
                    return True
                    
                # Check for titles that contain full URLs or email addresses
                if re.search(r'https?://|www\.|@[a-zA-Z0-9]+\.[a-zA-Z]', title):
                    return True
                
                return False
                

            # Function to extract just the heading portion from section text
            # @mlflow.trace(name="TOC - Extract Heading from Text")
            def extract_heading_from_text(text):
                """
                Extract just the heading portion from a text that may include a full paragraph.
                Uses structural patterns and length-based heuristics.
                
                Args:
                    text: The full text that may include heading and paragraph content
                
                Returns:
                    The extracted heading
                """
                # If text is already short (likely a heading), just return it
                if len(text) <= 60 and not is_paragraph_or_sentence(text):
                    return text
                    
                # CASE 1: For numbered sections, extract the number and first part
                numbered_match = re.match(r'^(\d+(?:\.\d+)*|\([a-z0-9]+\)|[A-Z]\.)\s+(.+?)(?:\.|\:|,|;|\n|\r|$)', text)
                if numbered_match:
                    prefix = numbered_match.group(1)
                    heading_part = numbered_match.group(2)
                    
                    # Further limit the heading part to a reasonable number of words
                    words = heading_part.split()
                    if len(words) > 8:  # Typical headings are 3-8 words
                        heading_part = ' '.join(words[:8]) + '...'
                        
                    # Check if the extracted part still looks like a sentence
                    if is_paragraph_or_sentence(heading_part):
                        # Take just the first 5-6 words
                        heading_part = ' '.join(heading_part.split()[:6]) + '...'
                        
                    return f"{prefix} {heading_part}"
                    
                # CASE 2: For capitalized headings, extract until first sentence break
                if text[0].isupper():
                    # Try to find the first sentence break or other natural delimiter
                    for delimiter in ['. ', '? ', '! ', ': ', '; ', '\n', '\r']:
                        parts = text.split(delimiter, 1)
                        if len(parts) > 1 and len(parts[0]) > 3:  # Make sure first part is substantial
                            first_part = parts[0].strip()
                            
                            # Check if first part looks reasonable as a heading
                            if len(first_part) <= 80 and not is_paragraph_or_sentence(first_part):
                                return first_part
                
                # CASE 3: Last resort - take first N words based on total length
                words = text.split()
                if len(words) <= 8:  # If already short, use as is
                    return text
                    
                # For longer text, be increasingly aggressive in truncation
                if len(text) > 200:
                    return ' '.join(words[:4]) + '...'
                elif len(text) > 100:
                    return ' '.join(words[:6]) + '...'
                else:
                    return ' '.join(words[:8]) + '...'
            
            for element in unstructured_elements:
                element_type = element.get('type', '')
                text = element.get('text', '').strip()
                
                # Skip empty text
                if not text:
                    continue
                    
                # Check for emphasis or formatting that indicates headings
                metadata = element.get('metadata', {})
                is_emphasized = False
                
                # Check if text is emphasized (bold, etc.)
                if 'emphasized_text_contents' in metadata and metadata.get('emphasized_text_contents'):
                    emphasized_text = ''.join(metadata.get('emphasized_text_contents', []))
                    if emphasized_text and (emphasized_text in text or text.startswith(emphasized_text)):
                        is_emphasized = True
                
                # Check if this element is a main section heading
                is_main_section = False
                
                # First, check explicit patterns
                for pattern in main_section_patterns:
                    match = re.match(pattern, text)
                    if match:
                        is_main_section = True
                        break
                        
                # Then consider emphasized text that might be section headers
                if (is_emphasized and 
                    (text.startswith('Policy') or 
                    any(text.startswith(f"{letter}.") for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ") or
                    element_type in ['Title', 'UncategorizedText'])):
                    is_main_section = True
                
                # Create new section if this is a main section
                if is_main_section:
                    # Clean up the section name (remove excessive whitespace, etc.)
                    text_clean = ' '.join(text.split())
                    
                    # Skip if it looks like a paragraph rather than a heading
                    if is_paragraph_or_sentence(text_clean):
                        continue
                        
                    # Extract just the heading portion, not the full paragraph
                    section_name = extract_heading_from_text(text_clean)
                    
                    # Final validation - if after extraction it still looks like a paragraph, skip it
                    if is_paragraph_or_sentence(section_name) or should_exclude_subsection(section_name):
                        continue
                        
                    current_section = {
                        "section_name": section_name,
                        "sub_sections": []
                    }
                    toc.append(current_section)
                    continue
                
                # Check for subsections
                is_subsection = False
                subsection_text = text
                
                # Skip if this looks like a paragraph rather than a potential subsection heading
                if is_paragraph_or_sentence(text) and not text.startswith('•') and not text.startswith('-'):
                    continue
                
                # Try matching subsection patterns
                for pattern in subsection_patterns:
                    match = re.match(pattern, text)
                    if match and match.groups():
                        subsection_text = match.group(1).strip()
                        is_subsection = True
                        break
                        
                # Special cases for bullet points and [*] markers
                if text.startswith('•') or text.startswith('[*]') or text.startswith('-'):
                    is_subsection = True
                    # Clean up the subsection text
                    subsection_text = text.lstrip('•').lstrip('[*]').lstrip('-').strip()
                    if not subsection_text:  # If it was just a marker with no content, use the whole text
                        subsection_text = text
                
                # Special case for tables that might contain subsection-like content
                if element_type == 'Table' and '•' in text:
                    parts = text.split('•', 1)
                    if len(parts) > 1:
                        subsection_text = parts[1].strip()
                        is_subsection = True
                
                # Add subsection to current section if applicable
                if is_subsection and current_section is not None and subsection_text:
                    # Clean up the subsection title and extract just the heading part
                    clean_title = ' '.join(subsection_text.split())
                    clean_title = extract_heading_from_text(clean_title)
                    
                    # Apply strict validation to exclude paragraphs and sentences
                    if should_exclude_subsection(clean_title) or is_paragraph_or_sentence(clean_title):
                        continue
                        
                    # Check character length (typical heading length)
                    if len(clean_title) > 80:
                        # Truncate if too long
                        words = clean_title.split()
                        clean_title = ' '.join(words[:8]) + '...'
                    
                    # Avoid duplicates
                    if not any(sub['title'] == clean_title for sub in current_section['sub_sections']):
                        current_section["sub_sections"].append({"title": clean_title})
            
            
            return toc
        except Exception as e:
            self.logger.error(self._log_message(f"Error extracting TOC from unstructured elements: {str(e)}", MODULE_NAME))
            raise e
    
    

    @mlflow.trace(name="File Parser - Unstructured Elements to Markdown Paid")
    def _convert_paid_unstructured_elements_to_markdown(self, unstructured_elements):
        """
        Convert unstructured elements (as dicts or Element objects) to markdown format, preserving structure.

        Parameters:
        - unstructured_elements: List of dicts OR Element instances returned from unstructured parser

        Returns:
        - String containing formatted markdown text
        """
        try:
            # Helpers to unify dict vs. object access
            def safe_type(el):
                if isinstance(el, dict):
                    return el.get("type", "")
                return type(el).__name__

            def safe_text(el):
                if isinstance(el, dict):
                    return el.get("text", "") or ""
                return getattr(el, "text", "") or ""

            def safe_meta(el):
                if isinstance(el, dict):
                    return el.get("metadata", {}) or {}
                return getattr(el, "metadata", {}) or {}

            markdown_parts = []
            in_list = False
            list_type = None

            self.logger.info(self._log_message(
                "Converting unstructured elements to markdown",
                "convert_paid_unstructured_elements_to_markdown"
            ))
            self.logger.debug(self._log_message(
                f"Unstructured elements type: {type(unstructured_elements)}",
                "convert_paid_unstructured_elements_to_markdown"
            ))
            self.logger.debug(self._log_message(
                f"Unstructured elements content: {unstructured_elements}",
                "convert_paid_unstructured_elements_to_markdown"
            ))

            for element in unstructured_elements:
                element_type = safe_type(element)
                text        = safe_text(element)
                metadata    = safe_meta(element)

                # --- Titles / Headings ---
                if element_type == "Title":
                    if in_list:
                        markdown_parts.append("")
                        in_list = False
                    level = metadata.get("heading_level", 2)
                    if not isinstance(level, int) or level < 1:
                        level = 2
                    markdown_parts.append(f"{'#' * level} {text}")

                # --- Plain paragraphs ---
                elif element_type in ["Text", "NarrativeText", "UncategorizedText"]:
                    if in_list:
                        markdown_parts.append("")
                        in_list = False
                    markdown_parts.append(text)

                # --- List items ---
                elif element_type == "ListItem":
                    marker = metadata.get("list_marker", "")
                    current_list_type = "numbered" if marker and marker[0].isdigit() else "bullet"
                    if in_list and list_type != current_list_type:
                        markdown_parts.append("")
                    in_list = True
                    list_type = current_list_type

                    if list_type == "numbered":
                        number = marker.rstrip(".") if marker and marker[0].isdigit() else "1"
                        markdown_parts.append(f"{number}. {text}")
                    else:
                        markdown_parts.append(f"- {text}")

                # --- Tables ---
                elif element_type == "Table":
                    if in_list:
                        markdown_parts.append("")
                        in_list = False

                    cells = metadata.get("cells")
                    if cells:
                        # Build cell map
                        table_rows = {}
                        for cell in cells:
                            row = cell.get("metadata", {}).get("row_index", 0) if isinstance(cell, dict) else getattr(cell.metadata, "row_index", 0)
                            col = cell.get("metadata", {}).get("column_index", 0) if isinstance(cell, dict) else getattr(cell.metadata, "column_index", 0)
                            cell_text = cell.get("text", "") if isinstance(cell, dict) else getattr(cell, "text", "")
                            table_rows.setdefault(row, {})[col] = cell_text

                        # Determine column count
                        max_cols = max((max(cols.keys()) for cols in table_rows.values()), default=-1) + 1
                        # Header
                        headers = [table_rows.get(0, {}).get(c, f"Column {c+1}") for c in range(max_cols)]
                        md_table = ["| " + " | ".join(headers) + " |",
                                    "| " + " | ".join(["---"] * max_cols) + " |"]
                        # Rows
                        for r in sorted(table_rows.keys()):
                            if r == 0: continue
                            row_cells = [table_rows[r].get(c, "") for c in range(max_cols)]
                            md_table.append("| " + " | ".join(row_cells) + " |")

                        markdown_parts.append("\n".join(md_table))
                    else:
                        markdown_parts.append(f"```\n{text}\n```")

                # --- Images ---
                elif element_type == "Image":
                    alt = metadata.get("alt_text", "Image")
                    src = metadata.get("source", "")
                    if src:
                        markdown_parts.append(f"![{alt}]({src})")
                    else:
                        markdown_parts.append(f"[Image: {alt}]")

                # --- Formulas ---
                elif element_type == "Formula":
                    if "\n" in text:
                        markdown_parts.append(f"$$\n{text}\n$$")
                    else:
                        markdown_parts.append(f"${text}$")

                # --- Headers/Footers ---
                elif element_type in ["Header", "Footer"]:
                    markdown_parts.append("---")
                    markdown_parts.append(f"> {text}")
                    markdown_parts.append("---")

                # --- Fallback for anything with text ---
                elif text:
                    markdown_parts.append(text)

            return "\n\n".join(markdown_parts)

        except Exception as e:
            self.logger.error(self._log_message(
                f"Error converting unstructured elements to markdown: {e}",
                "convert_paid_unstructured_elements_to_markdown"
            ))
            raise e


    @mlflow.trace(name="File Parser - Unstructured Elements to Markdown")
    def _convert_elements_to_markdown(self, unstructured_elements):
        """
        Convert unstructured element objects to markdown format, preserving structure.
        
        Parameters:
        - unstructured_elements: List of element objects from unstructured library
        
        Returns:
        - String containing formatted markdown text
        """

        try:
            self.logger.info(self._log_message("Converting unstructured elements to markdown", MODULE_NAME))
            self.logger.info(f"Type of unstructured_elements: {type(unstructured_elements)}")
            markdown_parts = []
            in_list = False
            list_type = None
            
            for element in unstructured_elements:
                # Get element type from the object's class name
                element_type = type(element).__name__
                
                # Handle different element types
                if element_type == 'Title':
                    # Close any open list before adding a title
                    if in_list:
                        markdown_parts.append('')  # Empty line to end list
                        in_list = False
                    
                    # Safely access metadata and heading level
                    level = 2  # Default heading level
                    if hasattr(element, 'metadata'):
                        metadata = element.metadata
                        if hasattr(metadata, 'heading_level'):
                            level_value = metadata.heading_level
                            if isinstance(level_value, int) and level_value >= 1:
                                level = level_value
                    
                    # Safely access text
                    element_text = element.text if hasattr(element, 'text') else str(element)
                    markdown_parts.append(f"{'#' * level} {element_text}")
                
                elif element_type in ['Text', 'NarrativeText', 'UncategorizedText']:
                    # Close any open list before adding regular text
                    if in_list:
                        markdown_parts.append('')  # Empty line to end list
                        in_list = False
                        
                    markdown_parts.append(element.text if hasattr(element, 'text') else str(element))
                
                elif element_type == 'ListItem':
                    marker = ""
                    # Safely access the list marker from metadata
                    if hasattr(element, 'metadata'):
                        metadata = element.metadata
                        if hasattr(metadata, 'list_marker'):
                            marker = metadata.list_marker
                    
                    # Determine list type based on marker
                    current_list_type = 'numbered' if marker and marker[0].isdigit() else 'bullet'
                    
                    # If switching list types, end previous list
                    if in_list and list_type != current_list_type:
                        markdown_parts.append('')
                        
                    in_list = True
                    list_type = current_list_type
                    
                    element_text = element.text if hasattr(element, 'text') else ''
                    
                    if list_type == 'numbered':
                        # Try to keep original numbering if available
                        if marker and marker[0].isdigit():
                            number = marker.rstrip('.')
                            markdown_parts.append(f"{number}. {element_text}")
                        else:
                            markdown_parts.append(f"1. {element_text}")
                    else:
                        markdown_parts.append(f"- {element_text}")
                
                elif element_type == 'Table':
                    # Close any open list before adding a table
                    if in_list:
                        markdown_parts.append('')
                        in_list = False
                    
                    has_html = False
                    # Safely check for HTML representation
                    if hasattr(element, 'metadata'):
                        metadata = element.metadata
                        if hasattr(metadata, 'text_as_html'):
                            has_html = True
                            markdown_parts.append("<!-- Table converted from HTML -->")
                    
                    # Process table cells if available
                    has_cells = hasattr(element, 'cells') and element.cells
                    if has_cells:
                        # Build table from cells
                        table_rows = {}
                        for cell in element.cells:
                            row_idx = 0
                            col_idx = 0
                            
                            # Safely get row and column indices
                            if hasattr(cell, 'row_index'):
                                row_idx = cell.row_index
                            if hasattr(cell, 'column_index'):
                                col_idx = cell.column_index
                            
                            if row_idx not in table_rows:
                                table_rows[row_idx] = {}
                            
                            cell_text = cell.text if hasattr(cell, 'text') else ''
                            table_rows[row_idx][col_idx] = cell_text
                        
                        # Only proceed if we have rows
                        if table_rows:
                            # Convert to markdown table
                            markdown_table = []
                            max_columns = 0
                            
                            # Find the maximum number of columns
                            for row in table_rows.values():
                                if row:  # Only process non-empty rows
                                    max_columns = max(max_columns, max(row.keys()) + 1 if row else 0)
                            
                            # Only create table if we have columns
                            if max_columns > 0:
                                # Create headers
                                headers = []
                                if 0 in table_rows and table_rows[0]:
                                    # Use first row as headers
                                    for i in range(max_columns):
                                        # Use dictionary-style access since this is our own dictionary
                                        headers.append(table_rows[0].get(i, ''))
                                else:
                                    # Create default headers
                                    headers = [f"Column {i+1}" for i in range(max_columns)]
                                
                                # Generate table markdown
                                markdown_table.append("| " + " | ".join(headers) + " |")
                                markdown_table.append("| " + " | ".join(["---" for _ in range(max_columns)]) + " |")
                                
                                # Add data rows
                                for row_idx in sorted(table_rows.keys()):
                                    if row_idx == 0 and 0 in table_rows:  # Skip header which we've already added
                                        continue
                                        
                                    row = table_rows[row_idx]
                                    row_texts = []
                                    for i in range(max_columns):
                                        # Use dictionary-style access since this is our own dictionary
                                        row_texts.append(row.get(i, ''))
                                        
                                    markdown_table.append("| " + " | ".join(row_texts) + " |")
                                
                                markdown_parts.append("\n".join(markdown_table))
                            else:
                                # Fallback for tables without proper columns
                                element_text = element.text if hasattr(element, 'text') else ''
                                markdown_parts.append(f"```\n{element_text}\n```")
                        else:
                            # Fallback for tables without proper rows
                            element_text = element.text if hasattr(element, 'text') else ''
                            markdown_parts.append(f"```\n{element_text}\n```")
                    else:
                        # Fallback for tables without cell structure
                        element_text = element.text if hasattr(element, 'text') else ''
                        markdown_parts.append(f"```\n{element_text}\n```")
                
                elif element_type == 'Image':
                    alt_text = "Image"
                    source = ""
                    
                    # Safely access metadata attributes
                    if hasattr(element, 'metadata'):
                        metadata = element.metadata
                        if hasattr(metadata, 'alt_text'):
                            alt_text = metadata.alt_text
                        if hasattr(metadata, 'source'):
                            source = metadata.source
                    
                    if source:
                        markdown_parts.append(f"![{alt_text}]({source})")
                    else:
                        markdown_parts.append(f"[Image: {alt_text}]")
                        
                elif element_type == 'Formula':
                    # Safely access text for math formulas
                    element_text = ""
                    if hasattr(element, 'text'):
                        element_text = element.text.strip()
                        
                    if element_text:
                        if '\n' in element_text:  # Display math
                            markdown_parts.append(f"$$\n{element_text}\n$$")
                        else:  # Inline math
                            markdown_parts.append(f"${element_text}$")
                        
                elif element_type in ['Header', 'Footer']:
                    # Add a divider and format headers/footers as quotes
                    element_text = element.text if hasattr(element, 'text') else ''
                    markdown_parts.append("---")
                    markdown_parts.append(f"> {element_text}")
                    markdown_parts.append("---")
                
                elif element_type == 'PageBreak':
                    # Add a horizontal rule for page breaks
                    markdown_parts.append("\n---\n")
                        
                elif hasattr(element, 'text'):
                    # Default handling for other element types with text
                    markdown_parts.append(element.text)
                else:
                    # Last resort fallback for any other types
                    try:
                        markdown_parts.append(str(element))
                    except:
                        markdown_parts.append("[Unprocessable element]")
            
            return "\n\n".join(markdown_parts)
        except Exception as e:
            self.logger.error(self._log_message(f"Error converting unstructured elements to markdown: {str(e)}", MODULE_NAME))
            raise e
        
    def _log_message(self, message, function_name):
        """
        Internal method to create a standardized log message with the module name.
        """
        return _log_message(message, function_name, MODULE_NAME)


    @mlflow.trace(name="File Parser - Estimate Legal Pages")
    def estimate_legal_pages(self, word_count: int, words_per_page: int = 2500) -> int:
        num_pages=(word_count // words_per_page) + (1 if word_count / words_per_page < 0 else 0)
        self.logger.info(self._log_message(f"Estimated number of pages: {num_pages}", "estimate_legal_pages"))
        return num_pages


    @mlflow.trace(name="File Parser - Preprocess")
    def preprocess(self, complete_file_text, file_id, user_id, org_id, retry_count, start_datetime):
       
        if self.estimate_legal_pages(len(complete_file_text)) == 1:
            self.logger.info(self._log_message("The document is too short. Single chunk is created.", "preprocess"))
            return [complete_file_text]
        else:
            document_chunker = ClusterSemanticChunker(self.logger)
            chunks = document_chunker.create_chunks(complete_file_text)
            return chunks

    
    @mlflow.trace(name="File Parser - Get Process Function")
    def _get_process_function(self, file_extension: str):
        """
        Retrieves the corresponding processing function for a given file extension.

        :param file_extension: The file extension (e.g., '.pdf', '.txt').
        :return: Corresponding processing function.
        :raises: ValueError if the file type is unsupported.
        """
        process_function = PROCESS_MAP.get(file_extension.lower())
        if not process_function:
            raise ValueError(f"Unsupported file type: {file_extension}")
        return process_function
    
    @mlflow.trace(name="File Parser - Process File")
    def process_file(self, filepath: str) -> List[str]:
        """
        Processes a file based on its extension and returns the results.
        :param filepath: The path to the file to be processed.
        :param file_id: Unique identifier for the file.
        :param user_id: Unique identifier for the user.
        :param org_id: Unique identifier for the organization.
        :return: List of processed results (e.g., text chunks).
        :raises: ValueError if the file type is unsupported.
        """
        try:
            file_extension = os.path.splitext(filepath)[1].lower()
            process_function = self._get_process_function(file_extension)
            self.logger.info(self._log_message(f"Processing file: {filepath}", "process_file"))
            return process_function(filepath)
        except ValueError as e:
            self.logger.error(self._log_message(str(e), "process_file"))
            raise ValueError(e)
        
    

    
    @mlflow.trace(name="File Parser - Extract and Chunk Text")
    def extract_and_chunk_text(self, complete_file_text, file_name, file_type, file_id, user_id, org_id, retry_count, start_datetime):
        """
        Downloads a file, processes it, and divides content into manageable chunks.
        """
        function_name = "extract_and_chunk_text"
        overall_start_time = time.perf_counter()
        process = psutil.Process()
        # start_datetime = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        log_data = {
            "file_id": file_id,
            "file_name": file_name,
            "user_id": user_id,
            "org_id": org_id,
            "time_taken": {}
        }

        try:
            # Log initialization
            init_start_time = time.perf_counter()
            
            set_llm_file_status(file_id, user_id, org_id, 1, True, retry_count, 1, "", start_datetime, "", True, False, self.in_queue, 75, self.logger)
            log_data["time_taken"]["llm_status_initiate_1_time"] = round(time.perf_counter() - init_start_time, 2)

            # Process chunking
            chunking_start_time = time.perf_counter()
            
            
            chunks = self.preprocess(complete_file_text, file_id, user_id, org_id, retry_count, start_datetime)
            log_data["time_taken"]["chunking_time"] = round(time.perf_counter() - chunking_start_time, 2)
            self.logger.info(self._log_message(f"Chunking completed. Total chunks: {len(chunks)}", function_name))

            if not chunks:
                self.logger.error(self._log_message("No text extracted from the file.", function_name))
                set_llm_file_status(file_id, user_id, org_id, 1, True, retry_count, 2, "Text Extraction Failed!", start_datetime, datetime.now().strftime('%Y-%m-%d %H:%M:%S'), True, False, self.in_queue, 100, self.logger)
            else:
                self.logger.info(self._log_message(f"Extracted {len(chunks)} chunks from the file.", function_name))
                set_llm_file_status(file_id, user_id, org_id, 1, True, retry_count, 3, "", start_datetime, datetime.now().strftime('%Y-%m-%d %H:%M:%S'), True, False, self.in_queue, 100, self.logger)

            init_end_start_time = time.perf_counter()
            log_data["time_taken"]["llm_status_completion_1_time"] = round(time.perf_counter() - init_end_start_time, 2)

            # Total time
            log_data["time_taken"]["total_chunking_execution_time"] = round(time.perf_counter() - overall_start_time, 2)
            self.logger.debug(self._log_message(f"CHUNKING SUMMARY: {orjson.dumps(log_data).decode()}", function_name))

            memory_usage = process.memory_info().rss / (1024 * 1024)  # Memory usage in MB
            cpu_usage = process.cpu_percent(interval=0.1)
            log_data["memory_usage"] = round(memory_usage, 2)
            log_data["cpu_usage"] = round(cpu_usage, 2)
            log_data["chunks_count"] = len(chunks)



            # Update BM25 encoder
            # self.update_encoder_and_upsert_vectors(chunks, file_name, file_type, file_id, user_id, org_id, retry_count)
            

            return chunks

        except Exception as e:
            error_msg = f"Error during {function_name}: {e}"
            self.logger.error(self._log_message(error_msg, function_name))
            
            set_llm_file_status(file_id, user_id, org_id, 1, True, retry_count, 2, error_msg, start_datetime, datetime.now().strftime('%Y-%m-%d %H:%M:%S'), True, False, self.in_queue, 100, self.logger)
            raise e
