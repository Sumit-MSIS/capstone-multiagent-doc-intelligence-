import json
import re
from src.common.logger import _log_message
import mlflow
from opentelemetry import context as ot_context
# mlflow.config.enable_async_logging()
# mlflow.langchain.autolog()
mlflow.openai.autolog()

MODULE_NAME = "json_parser.py"

class LLMOutputParser:
    def __init__(self, logger=None):
        self.logger = logger
    
    def _log_message(self, message, function_name):
        """
        Internal method to create a standardized log message with the module name.
        """
        return _log_message(message, function_name, MODULE_NAME)

    @mlflow.trace(name="JSON Parser - Parse")
    def parse(self, llm_response: str, single_call: bool = True):
        try:
            # Try to parse as JSON directly
            parsed = self._try_parse_json(llm_response, single_call)
            if parsed:
                return parsed

            # Attempt to extract JSON using regex and parse
            json_matches = re.findall(r"({[\s\S]*?})", llm_response)
            for json_str in json_matches:
                parsed = self._try_parse_json(json_str, single_call)
                if parsed:
                    return parsed

            # Attempt to extract JSON from markdown-like code blocks
            code_blocks = re.findall(r"`{3}(?:json)?([\s\S]*?)`{3}", llm_response)
            for block in code_blocks:
                parsed = self._try_parse_json(block.strip(), single_call)
                if parsed:
                    return parsed

            # Log and raise error if no valid JSON found
            self.logger.error(self._log_message(f"Could not parse valid JSON from response: {llm_response}", "parse"))
            raise ValueError("No valid JSON found in response")

        except Exception as e:
            self.logger.exception(self._log_message(f"Error parsing response: {str(e)}", "parse"))
            return None

    @mlflow.trace(name="JSON Parser - Try Parse JSON")
    def _try_parse_json(self, json_str: str, single_call: bool) -> dict:
        """
        Attempts to parse a string as JSON and validates its structure.
        """
        try:
            parsed = json.loads(json_str)
            parsed.pop("Reasoning", None)
            if single_call and self._validate_job_description_structure(parsed):
                return parsed
        except json.JSONDecodeError:
            self.logger.debug(self._log_message(f"Failed to decode JSON: {json_str}", "_try_parse_json"))
        return None

    @mlflow.trace(name="JSON Parser - Validate Job Description Structure")
    def _validate_job_description_structure(self, parsed_json: dict) -> bool:
        """
        Validates the structure of the parsed job description JSON.
        """
        self.logger.info(self._log_message(f"Validating JSON structure: {parsed_json}", "_validate_job_description_structure"))

        valid_keys = {
            "Effective Date": "date",
            "Term Date": "date",
            "Payment Due Date": "date",
            "Delivery Date": "date",
            "Termination Date": "date",
            "Renewal Date": "date",
            "Expiration Date": "date",
            "Title of the Contract": "string",
            "Scope of Work": "string",
            "Parties Involved": "string",
            "Contract Type": "string",
            "File Type": "string",
            "Jurisdiction": "string",
            "Version Control": "string",
            "Contract Duration": "string",
            "Contract Value": "float",
            "Risk Mitigation Score": "string",
            "flag": "bool",
            "Justification": "list"
        }

        



        for key, value in parsed_json.items():
            expected_type = valid_keys.get(key)
            if not expected_type:
                self.logger.error(self._log_message(f"Invalid key: {key}", "_validate_job_description_structure"))
                return False
            if expected_type == "date":
                if value and value != "null" and not re.match(r"^\d{4}-\d{2}-\d{2}$", str(value)):
                    self.logger.error(self._log_message(f"Invalid date format for key: {key}, value: {value}", "_validate_job_description_structure"))
                    return False
            # if expected_type == "date":
            #     if value and not re.match(r"^\d{4}-\d{2}-\d{2}$", str(value)):
            #         self.logger.error(self._log_message(f"Invalid date format for key: {key}, value: {value}", "_validate_job_description_structure"))
            #         return False
            elif expected_type == "string":
                if value and value != "null" and not isinstance(value, str):
                    self.logger.error(self._log_message(f"Invalid string value for key: {key}, value: {value}", "_validate_job_description_structure"))
                    return False
            elif expected_type == "bool":
                if value and value != "null" and not isinstance(value, bool):
                    self.logger.error(self._log_message(f"Invalid boolean value for key: {key}, value: {value}", "_validate_job_description_structure"))
                    return False
            elif expected_type == "float":
                if value is not None and value != "null" and not isinstance(float(value), float):
                    return False   
            elif expected_type == "list":
                if value and value != "null" and not isinstance(value, list):
                    self.logger.error(self._log_message(f"Invalid list value for key: {key}, value: {value}", "_validate_job_description_structure"))
                    return False
                        

        self.logger.info(self._log_message("All keys in the JSON are valid.", "_validate_job_description_structure"))
        return True

    
# parser = LLMOutputParser()