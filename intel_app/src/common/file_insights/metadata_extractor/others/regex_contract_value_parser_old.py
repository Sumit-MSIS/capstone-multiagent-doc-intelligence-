from src.common.logger import _log_message
import re
from src.common.file_insights.llm_call import dynamic_llm_call
from nltk.tokenize import word_tokenize

MODULE_NAME = "regex_contract_value_parser"

contract_value_regex = r"""
(  # Removed the enclosing \b(...)\b
    # Symbol-Based Currency Amounts
    (?:[\$\€\£\¥\₹]|AED|USD|INR|GBP|EUR|CNY|RMB|JPY|SAR)\s?
    (?:
        \d{1,3}(?:,\d{3})*(?:\.\d+)? |
        \d+(?:\.\d+)?\s*(?:million|billion|crore|lakh|thousand)?
    )
|
    # Name-Based Currency Amounts
    (?:
        (?:(?:[Oo]ne|[Tt]wo|[Tt]hree|[Ff]our|[Ff]ive|[Ss]ix|[Ss]even|[Ee]ight|[Nn]ine|[Tt]en|
        [Ee]leven|[Tt]welve|[Tt]hirteen|[Ff]ourteen|[Ff]ifteen|[Ss]ixteen|[Ss]eventeen|
        [Ee]ighteen|[Nn]ineteen|[Tt]wenty|[Tt]hirty|[Ff]orty|[Ff]ifty|[Ss]ixty|[Ss]eventy|
        [Ee]ighty|[Nn]inety|[Oo]ne\s+hundred|[Oo]ne\s+thousand|
        [Oo]ne\s+million|[Oo]ne\s+billion|[Oo]ne\s+crore|[Oo]ne\s+lakh)(\s+and\s+)?)*
        (?:[Mm]illion|[Bb]illion|[Tt]housand|[Cc]rore|[Ll]akh)?
    )
    \s+
    (?:dollars?|pounds?|euros?|rupees?|dirhams?|yuan|yen)
)
"""

contract_value_instructions = """
**Task**: Calculate and extract the **total contract value** from the text.  
- Contract value refers to the financial worth of the agreement, including totals, installments, or budgets.  
- **Return only the numerical value** as an integer (remove commas/currency symbols).  
- **Handle complex cases**:  
  - If multiple amounts are stated (e.g., "$500,000 upfront and $300,000 annually"), **sum them** and return the total.  
  - If a **total value** is explicitly mentioned, use it instead of partial amounts.  
- Return "null" if:  
  - No value is mentioned.  
  - Values are ambiguous (e.g., "subject to funding", "confidential").  
  - Non-numeric descriptions (e.g., "to be negotiated").  

**Output Format**:
-**Strictly do not include "```" or "json" or any markers in your response.**  
{"Contract Value": "<number_total_or_null>"}  
"""

def clean_and_split_sentences(text):
    """
    Cleans markdown/special characters (preserving hyphens, slashes, and date characters)
    and splits text into sentences.
    """
    # Remove markdown images and links
    text = re.sub(r'!\[.*?\]\(.*?\)', '', text)  # Remove markdown images
    text = re.sub(r'\[.*?\]\(.*?\)', '', text)   # Remove markdown links
   
    # Strip markdown symbols while preserving hyphens and slashes
    # Kept: - / (date characters) | Removed: *_~`#+=|>[](){}!\\<>
    text = re.sub(r'[*_~`#+=|>\[\](){}!\\<>]', '', text)
   
    # Collapse multiple spaces
    text = re.sub(r'\s+', ' ', text).strip()
   
    # Split on . followed by optional markdown or spacing chars
    sentences = re.split(r'\.(?:\s|\||\*|,)*', text)
   
    # Remove any empty strings from result
    return [s.strip() for s in sentences if s.strip()]

def get_regex_contract_value(chunks, tag_ids, logger):
    try:
        combined_text = []
        seen_sentences = set()  # Track sentences to avoid repetition
        chunks = "\n".join([chunk for chunk in chunks])
        sentences = clean_and_split_sentences(chunks)
        regex_pattern = contract_value_regex
        
        # Create a set to track processed indices
        processed_indices = set()
        
        for i in range(len(sentences)):
            # Skip if this index has already been processed
            if i in processed_indices:
                continue
                
            if re.search(regex_pattern, sentences[i], re.VERBOSE):
                # Extract 1 sentence before, the matched sentence, and 1 sentence after
                start = max(0, i - 1)
                end = min(len(sentences), i + 2)  # Include the matched sentence and 1 after
                selected_sentences = sentences[start:end]
                
                # Mark these indices as processed
                for idx in range(start, end):
                    processed_indices.add(idx)
                
                # Limit the length of sentences before and after to 25 words
                trimmed_sentences = []
                for j, sent in enumerate(selected_sentences):
                    if sent not in seen_sentences:  # Avoid duplicate sentences
                        words = word_tokenize(sent)
                        if j == 0 and i > 0:  # Sentence before
                            trimmed_sentences.append(" ".join(words[-30:]))
                        elif j == len(selected_sentences) - 1 and i + 1 < len(sentences):  # Sentence after
                            trimmed_sentences.append(" ".join(words[:30]))
                        else:  # Matched sentence
                            trimmed_sentences.append(sent)
                        seen_sentences.add(sent)  # Mark sentence as seen
                
                combined_text.extend(trimmed_sentences)
            # No need for an else clause as we just continue to the next index
        
        contract_value_context = ""
        # Combine all matched text into a single chunk
        if combined_text:
            filtered = [{"chunk_number": "combined", "text": " ".join(combined_text)}]
            contract_value_context = filtered[0]['text'] if filtered else ""

        if not contract_value_context:
            logger.debug(_log_message("No contract value context found using regex", "get_regex_contract_value", MODULE_NAME))
            return {"Contract Value": "null"}

        logger.info(_log_message(f"Context filtered for contract value extraction:{len(contract_value_context)}", "get_regex_contract_value", MODULE_NAME))
        system_prompt = """You are an assistant that understands and extracts contract value from the legal contract context"""

        user_prompt = f"""
        {contract_value_instructions}
        Below is the text paragraph consisting of all the required context filtered from the legal agreement's, contract's context:
        ---
        {contract_value_context}. 
        ---
        Output should be a minimal json, DO NOT provide any extra words or markers '```' OR '```json'."""

        json_schema = {
            "type": "json_schema",
            "json_schema": {
                "name": "contract_value_extraction",
                "schema": {
                "type": "object",
                "properties": {
                    "Contract Value": {
                    "type": ["number", "null"],
                    "description": "The total contract value as a float (rounded if needed), or null if it is not found or not calculable."
                    }
                },
                "required": ["Contract Value"],
                "additionalProperties": False
                },
                "strict": True
            }
        }



        llm_response = dynamic_llm_call(user_prompt, system_prompt, json_schema, logger)  
        logger.info(_log_message(f"Contract value extraction response: {llm_response}", "get_regex_contract_value", MODULE_NAME))
        return llm_response
    except Exception as e:
        logger.error(_log_message(f"Error in contract value extraction: {str(e)}", "get_regex_contract_value", MODULE_NAME))
        return {"Jurisdiction": "null"}