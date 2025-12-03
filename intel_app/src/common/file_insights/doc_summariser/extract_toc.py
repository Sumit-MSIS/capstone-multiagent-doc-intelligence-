import re
import logging
from src.config.base_config import config
from typing import List, Tuple, Dict, Any
from src.common.logger import _log_message
import mlflow
from opentelemetry import context as ot_context

TRACKING_URI = config.MLFLOW_TRACKING_URI
ENV = config.MLFLOW_ENV

# mlflow.config.enable_async_logging()
mlflow.openai.autolog()


# Set up logging
MODULE_NAME = "toc_extraction"

class ExtractTOC:

    def __init__(self, logger):
        self.logger = logger

    @mlflow.trace(name="Extract TOC - Parse TOC")
    def parse_toc(self, document_content: str) -> List[Dict[str, Any]]:
        try:
            sections = []
            current_section = None
            page_number_pattern = re.compile(r'\s*\(PAGE \d+\)\s*$')
            section_numbering_pattern = re.compile(r'^\s*\d+\.?\d*\s*')  # Updated regex
            subsection_numbering_pattern = re.compile(r'^\s*\d+\.?\d*\s*')  # Updated regex

            lines = document_content.split('\n')
            for line in lines:
                stripped_line = line.strip()
                
                # Skip lines that are just page numbers
                if re.fullmatch(r'\(PAGE \d+\)', stripped_line):
                    continue
                
                # Check for section headers (##)
                if stripped_line.startswith('##'):
                    # Extract section name, removing '##' and any page numbers
                    section_name = stripped_line[2:].strip()
                    section_name = page_number_pattern.sub('', section_name)
                    section_name = section_numbering_pattern.sub('', section_name).strip() 
                    # Only create a section if the name is non-empty
                    if section_name:
                        current_section = {
                            "section_name": section_name,
                            "sub_sections": []
                        }
                        sections.append(current_section)
                    else:
                        current_section = None  # Skip empty sections
                # Check for subsections starting with '-'
                elif stripped_line.startswith('-'):
                    # Only process if there is an active section
                    if current_section is not None:
                        # Extract subsection title, removing '-', numbering, and page numbers
                        subsection_title = stripped_line[1:].strip()
                        subsection_title = page_number_pattern.sub('', subsection_title)
                        # Remove subsection numbering using regex (e.g., "1.1 ", "10. ", etc.)
                        subsection_title = subsection_numbering_pattern.sub('', subsection_title).strip()
                        current_section["sub_sections"].append({"title": subsection_title})
            
            return sections
        except Exception as e:
            self.logger.error(_log_message(f"Error parsing TOC: {e}", "parse_toc", MODULE_NAME))
            return []
    
    @mlflow.trace(name="Extract TOC - Extract TOC from Elements")
    def extract_toc_from_elements(self, elements: List[Any]) -> List[Dict]:
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
                section_name = ' '.join(text.split())
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
            
            # Add subsection to current section if applicable, but filter out single-character subsections
            if is_subsection and current_section is not None and subsection_text:
                # Check if the subsection text is meaningful (not just a single letter or character)
                if len(subsection_text.strip()) <= 1:
                    continue  # Skip single character subsections
                    
                # Clean up the subsection title
                clean_title = ' '.join(subsection_text.split())
                
                # Avoid duplicates and meaningless subsections
                if (not any(sub['title'] == clean_title for sub in current_section['sub_sections']) and
                    len(clean_title) > 1):  # Ensure it's not just a single character
                    current_section["sub_sections"].append({"title": clean_title})
        
        # Return just the TOC list without metadata wrapper
        return toc

    @mlflow.trace(name="Extract TOC - Extract TOC")
    def extract_toc(self, unstructured_elements: str, file_id:str):
        try:
            self.logger.info(_log_message("Extracting Table of Contents", "extract_toc", MODULE_NAME))
            toc = self.extract_toc_from_elements(unstructured_elements)
            self.logger.info(_log_message(f"Results for TOC: {toc}", "extract_toc", MODULE_NAME))
            result = {
                "file_id": file_id,
                "table_of_contents": toc
            }
            return result
        except Exception as e:
            self.logger.error(_log_message(f"Error extracting TOC: {e}", "extract_toc", MODULE_NAME))
            return None
