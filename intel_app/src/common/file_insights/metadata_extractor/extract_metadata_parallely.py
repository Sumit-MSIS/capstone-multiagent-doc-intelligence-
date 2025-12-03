import time
import psutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import orjson
import mlflow
from opentelemetry import context as ot_context

# Import services and utilities (keeping same import structure)
from src.common.file_insights.metadata_extractor.contract_metadata_vector_handler import ContractMetadataVectorUpserter
from src.common.llm_status_handler.status_handler import set_llm_file_status, set_meta_data
from src.common.file_insights.metadata_extractor.dates.extract_delivery_date import get_delivery_date
from src.common.file_insights.metadata_extractor.dates.extract_effective_date import get_effective_date
from src.common.file_insights.metadata_extractor.dates.extract_expiration_date import get_expiration_date
from src.common.file_insights.metadata_extractor.dates.extract_termination_date import get_termination_date
from src.common.file_insights.metadata_extractor.dates.extract_renewal_date import get_renewal_date
from src.common.file_insights.metadata_extractor.dates.extract_payment_due_date import get_payment_due_date
from src.common.file_insights.metadata_extractor.dates.extract_term_date import get_term_date
from src.common.file_insights.metadata_extractor.dates.regex_date_parser import get_regex_dates
from src.common.file_insights.metadata_extractor.others.extract_contract_value import get_contract_value
from src.common.file_insights.metadata_extractor.others.extract_jurisdiction import get_jurisdiction
from src.common.file_insights.metadata_extractor.others.extract_scope_of_work import get_scope_of_work
from src.common.file_insights.metadata_extractor.others.extract_title_of_contract import get_title_of_contract
from src.common.file_insights.metadata_extractor.others.extract_parties_involved import get_parties_involved
from src.common.file_insights.metadata_extractor.others.extract_contract_type import get_contract_type
from src.common.file_insights.metadata_extractor.others.extract_version_control import get_version_control
from src.common.file_insights.metadata_extractor.others.extract_risk_mitigation_score import get_risk_mitigation_score
from src.common.file_insights.metadata_extractor.others.extract_contract_duration import get_contract_duration
from src.common.file_insights.metadata_extractor.others.extract_file_type import get_file_type
from src.common.file_insights.metadata_extractor.others.tcv_module.regex_contract_value_parser import get_regex_contract_value
from src.common.file_insights.metadata_extractor.others.regex_jurisdiction_parser import get_regex_jurisdiction
from src.common.file_insights.metadata_extractor.contract_metadata_vector_handler import ContractMetadataVectorUpserter
from src.common.file_insights.llm_call import payment_due_date_validatior
from src.common.logger import _log_message
from src.common.hybrid_retriever.hybrid_search_retrieval import get_context_from_pinecone

# Enable MLflow async logging
# mlflow.config.enable_async_logging()
mlflow.openai.autolog()

MODULE_NAME = "metadata_extractor"
REQUIRED_FIELDS = ['Effective Date', 'Termination Date', 'Renewal Date', 'Expiration Date', 
                   'Delivery Date', 'Term Date', 'Jurisdiction']

class RegexDateExtractor:
    def __init__(self, chunks, complete_file_text, tag_ids, logger):
        """
        Initialize the RegexDateExtractor with chunks and logger.
        
        Args:
            chunks (list): List of text chunks to extract dates from
            logger: Logger instance for logging information
        """
        self.chunks = chunks
        self.complete_file_text = complete_file_text
        self.tag_ids = tag_ids
        self.logger = logger

    @mlflow.trace(name="RegexDateExtractor - Extract")
    def extract(self):
        """
        Extract dates using regex patterns.
        
        Returns:
            dict: Dictionary of extracted dates
        """
        try:
            self.logger.info(_log_message("Starting regex date extraction", "RegexDateExtractor.extract", MODULE_NAME))
            result = get_regex_dates([self.complete_file_text], self.tag_ids, self.logger)
            self.logger.info(_log_message(f"Completed regex date extraction: {result}", "RegexDateExtractor.extract", MODULE_NAME))
            return result
        except Exception as e:
            self.logger.error(_log_message(f"Error in regex date extraction: {str(e)}", "RegexDateExtractor.extract", MODULE_NAME))
            # Return empty default values for date fields
            return {key: None for key in REQUIRED_FIELDS if key not in ["Jurisdiction", "Contract Value"]}

class RegexJurisdictionExtractor:
    def __init__(self, chunks, tag_ids, logger):
        """
        Initialize the RegexJurisdictionExtractor with chunks and logger.
        
        Args:
            chunks (list): List of text chunks to extract jurisdiction from
            logger: Logger instance for logging information
        """
        self.chunks = chunks
        self.tag_ids = tag_ids
        self.logger = logger
    
    @mlflow.trace(name="RegexJurisdictionExtractor - Extract")
    def extract(self):
        """
        Extract jurisdiction using regex patterns.
        
        Returns:
            dict: Dictionary with jurisdiction
        """
        try:
            self.logger.info(_log_message("Starting regex jurisdiction extraction", "RegexJurisdictionExtractor.extract", MODULE_NAME))
            result = get_regex_jurisdiction(self.chunks, self.tag_ids, self.logger)
            self.logger.info(_log_message(f"Completed regex jurisdiction extraction: {result}", "RegexJurisdictionExtractor.extract", MODULE_NAME))
            return result
        except Exception as e:
            self.logger.error(_log_message(f"Error in regex jurisdiction extraction: {str(e)}", "RegexJurisdictionExtractor.extract", MODULE_NAME))
            return {"Jurisdiction": None}

class RegexContractValueExtractor:
    def __init__(self, chunks, complete_file_text, contract_type, file_id, user_id, org_id, logger):
        """
        Initialize the RegexContractValueExtractor with chunks and logger.
        
        Args:
            chunks (list): List of text chunks to extract contract value from
            logger: Logger instance for logging information
        """
        self.chunks = chunks
        self.complete_file_text = complete_file_text
        self.contract_type = contract_type
        self.file_id = file_id
        self.user_id = user_id
        self.org_id = org_id
        self.logger = logger

    @mlflow.trace(name="RegexContractValueExtractor - Extract")
    def extract(self):
        """
        Extract contract value using regex patterns.
        
        Returns:
            dict: Dictionary with contract value
        """
        try:
            self.logger.info(_log_message("Starting regex contract value extraction", "RegexContractValueExtractor.extract", MODULE_NAME))
            result = get_regex_contract_value([self.complete_file_text], self.contract_type, self.file_id, self.user_id, self.org_id, self.logger)
            self.logger.info(_log_message(f"Completed regex contract value extraction: {result}", "RegexContractValueExtractor.extract", MODULE_NAME))
            return result
        except Exception as e:
            self.logger.error(_log_message(f"Error in regex contract value extraction: {str(e)}", "RegexContractValueExtractor.extract", MODULE_NAME))
            return {"Contract Value": None}

from concurrent.futures import ThreadPoolExecutor

class DependentMetadataFields:
    def __init__(self, chunks, complete_file_text, contract_type, file_id, user_id, org_id, tag_ids, logger,effective_date, expiry_date):
        self.chunks = chunks
        self.contract_type = contract_type
        self.complete_file_text = complete_file_text
        self.file_id = file_id
        self.user_id = user_id
        self.org_id = org_id
        self.tag_ids = tag_ids
        self.logger = logger
        self.expiry_date = expiry_date
        self.effective_date = effective_date
    
    def submit_with_context(self, executor, fn, *args, **kwargs):
        """Helper method to submit functions with OpenTelemetry context preservation"""
        parent_ctx = ot_context.get_current()
        
        def wrapped():
            token = ot_context.attach(parent_ctx)
            try:
                return fn(*args, **kwargs)
            finally:
                ot_context.detach(token)
        
        return executor.submit(wrapped)

    @mlflow.trace(name="DependentMetadataFields - Extract Contract Value")
    def extract_contract_value(self):
        extractor = RegexContractValueExtractor(self.chunks, self.complete_file_text, self.contract_type, self.file_id, self.user_id, self.org_id, self.logger)
        return extractor.extract()

    @mlflow.trace(name="DependentMetadataFields - Extract Payment Due Date")
    def extract_payment_due_date(self):
        flag_value, payment_due_date = get_payment_due_date(
            self.user_id, self.org_id, self.file_id, self.tag_ids, self.logger,self.effective_date ,self.expiry_date
        )
        return {'Payment Due Date': payment_due_date, 'Has Recurring Payment': flag_value}

    @mlflow.trace(name="DependentMetadataFields - Run Parallel Extraction")
    def run_parallel(self):

        
        answers = []
        with ThreadPoolExecutor(max_workers=2) as executor:
            future_contract_value = self.submit_with_context(executor, self.extract_contract_value)
            future_payment_due_date = self.submit_with_context(executor, self.extract_payment_due_date)
            # future_contract_value = executor.submit(self.extract_contract_value)
            # future_payment_due_date = executor.submit(self.extract_payment_due_date)
            contract_value_result = future_contract_value.result()
            payment_due_date_result = future_payment_due_date.result()
            answers.append(contract_value_result)
            answers.append(payment_due_date_result)
        self.logger.info(_log_message(f"Final extraction results: {answers}", "run_parallel", MODULE_NAME))
        return answers


class MetadataExtractor:
    def __init__(self, names):
        """
        Initialize the MetadataExtractor with required field names.
        
        Args:
            names (list): List of metadata field names to extract
        """
        self.names_dict = {}
        for name in names:
            self.names_dict[name] = None
            
        # Store context for later use
        self.file_id = None
        self.file_name = None
        self.file_type = None
        self.user_id = None
        self.org_id = None
        self.retry_count = None
        self.chunks = None
        self.complete_file_text = None
        self.logger = None
        self.start_datetime = None
        self.in_queue = False
        self.null_keys = []
        self.regex_date_extracted = {}
        self.regex_jurisdiction_extracted = {"Jurisdiction": None}
        self.regex_contract_value_extracted = {"Contract Value": None}
        self.expiry_date = None
        self.effective_date = None
        
    def submit_with_context(self, executor, fn, *args, **kwargs):
        """Helper method to submit functions with OpenTelemetry context preservation"""
        parent_ctx = ot_context.get_current()
        
        def wrapped():
            token = ot_context.attach(parent_ctx)
            try:
                return fn(*args, **kwargs)
            finally:
                ot_context.detach(token)
        
        return executor.submit(wrapped)
    
    @mlflow.trace(name="MetadataExtractor - Setup Context")
    def setup_context(self, file_id, file_name, file_type, user_id, org_id, retry_count, chunks, complete_file_text, tag_ids,  logger):
        """Set up the context for metadata extraction"""
        self.file_id = file_id
        self.file_name = file_name
        self.file_type = file_type
        self.user_id = user_id
        self.org_id = org_id
        self.retry_count = retry_count
        self.chunks = chunks
        self.complete_file_text = complete_file_text
        self.tag_ids = tag_ids
        self.logger = logger
        self.start_datetime = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Update status to indicate processing has started
        set_llm_file_status(
            self.file_id, self.user_id, self.org_id, 3, True, 
            self.retry_count, 1, "", self.start_datetime, "", 
            False, False, self.in_queue, 25, self.logger
        )
    
    @mlflow.trace(name="MetadataExtractor - Get Regex")
    def get_regex(self):
        """
        Extract metadata using regex methods first and then hybrid search for null values.
        Returns a dictionary of extracted values.
        """
        self.logger.info(_log_message("Starting regex extraction in parallel", "get_regex", MODULE_NAME))
        self.logger.info(_log_message(f"fields provided in target_metadatafields {self.names_dict}", "get_regex", MODULE_NAME))

        # Create the extractor instances
        date_extractor = RegexDateExtractor(self.chunks, self.complete_file_text, self.tag_ids, self.logger)
        jurisdiction_extractor = RegexJurisdictionExtractor(self.chunks, self.tag_ids, self.logger)
        # contract_value_extractor = RegexContractValueExtractor(self.chunks, self.logger)

        # Run regex extractions in parallel
        regex_results = {}
        with ThreadPoolExecutor(max_workers=3) as executor:
            # Submit all regex extraction tasks
            futures = {
                "dates": self.submit_with_context(executor, date_extractor.extract),
                "jurisdiction": self.submit_with_context(executor, jurisdiction_extractor.extract)
                # "contract_value": self.submit_with_context(executor, contract_value_extractor.extract)
            }
            
            # Collect results as they complete
            for name, future in futures.items():
                try:
                    result = future.result()
                    regex_results[name] = result
                    self.logger.info(_log_message(f"Completed regex extraction for {name}", "get_regex", MODULE_NAME))
                except Exception as e:
                    self.logger.error(_log_message(f"Error in regex extraction for {name}: {str(e)}", "get_regex", MODULE_NAME))
                    # Provide empty default values based on extraction type
                    if name == "dates":
                        regex_results[name] = {key: None for key in REQUIRED_FIELDS if key not in ["Jurisdiction", "Contract Value"]}
                    elif name == "jurisdiction":
                        regex_results[name] = {"Jurisdiction": None}
                    # elif name == "contract_value":
                    #     regex_results[name] = {"Contract Value": None}

        # Store the regex results in instance variables
        self.regex_date_extracted = regex_results.get("dates", {})
        self.regex_jurisdiction_extracted = regex_results.get("jurisdiction", {"Jurisdiction": None})
        # self.regex_contract_value_extracted = regex_results.get("contract_value", {"Contract Value": None})

        self.logger.info(_log_message(f"Retrieved Dates using regex: {self.regex_date_extracted}", "get_regex", MODULE_NAME))
        self.logger.info(_log_message(f"Retrieved Jurisdiction: {self.regex_jurisdiction_extracted}", "get_regex", MODULE_NAME))
        # self.logger.info(_log_message(f"Retrieved Contract Value: {self.regex_contract_value_extracted}", "get_regex", MODULE_NAME))
        
        # Step 2: Get keys with null or None values for hybrid search
        self.null_keys = [key for key, value in self.regex_date_extracted.items() if value in ("null", None)]
        if self.regex_jurisdiction_extracted.get("Jurisdiction") in ("null", None):
            self.null_keys.append("Jurisdiction")
        # if self.regex_contract_value_extracted.get("Contract Value") in ("null", None):
        #     self.null_keys.append("Contract Value")
            
        self.logger.info(_log_message(f"Null keys requiring hybrid search: {self.null_keys}", "get_regex", MODULE_NAME))
        
        # If there are null values, run hybrid search for those specific fields
        if self.null_keys:
            retrieved_regex_date_extracted = f"\n\nPrevious extraction results: {self.regex_date_extracted}"
            hybrid_results = []
            
            # Map fields to their extraction functions
            function_map = {
                "Effective Date": get_effective_date,
                "Termination Date": get_termination_date,
                "Renewal Date": get_renewal_date,
                "Expiration Date": get_expiration_date,
                "Delivery Date": get_delivery_date,
                "Term Date": get_term_date,
                "Jurisdiction": get_jurisdiction
                # "Contract Value": get_contract_value
            }
            
            with ThreadPoolExecutor(max_workers=len(self.null_keys)) as executor:
                # Create a dictionary to track futures
                futures = {}
                
                # Submit hybrid search tasks for null values
                for key in self.null_keys:
                    if key in function_map:
                        # For date fields that need the previous extraction results
                        if key in self.regex_date_extracted.keys():
                            futures[key] = self.submit_with_context(
                                executor, 
                                function_map[key], 
                                self.user_id, self.org_id, self.file_id, self.tag_ids, self.logger, retrieved_regex_date_extracted
                            )
                        # For Jurisdiction and Contract Value which don't need previous results
                        else:
                            futures[key] = self.submit_with_context(
                                executor, 
                                function_map[key],
                                self.user_id, self.org_id, self.file_id, self.tag_ids, self.logger
                            )
                
                # Collect results as they complete
                for key, future in futures.items():
                    try:
                        result = future.result()
                        hybrid_results.append(result)
                        
                        # Update the appropriate dictionary based on the key
                        for result_key, result_value in result.items():
                            if result_key in self.regex_date_extracted:
                                self.regex_date_extracted[result_key] = result_value
                            elif result_key == "Jurisdiction":
                                self.regex_jurisdiction_extracted["Jurisdiction"] = result_value
                            # elif result_key == "Contract Value":   
                            #     self.regex_contract_value_extracted["Contract Value"] = result_value
                        
                        self.logger.info(_log_message(f"Completed hybrid extraction for {key}: {result}", "get_regex", MODULE_NAME))
                    except Exception as e:
                        self.logger.error(_log_message(f"Error extracting {key}: {str(e)}", "get_regex", MODULE_NAME))
            
            self.logger.info(_log_message(f"Hybrid search results: {hybrid_results}", "get_regex", MODULE_NAME))
        
        # Combine all results into a list of dictionaries
        results = []
        # Add date fields
        for key, value in self.regex_date_extracted.items():
            results.append({key: value})
            if key == "Expiration Date":
                self.expiry_date = value
            if key == "Effective Date": # required in payment_due_date
                self.effective_date = value 
                
        # Add jurisdiction and contract value
        results.append(self.regex_jurisdiction_extracted)
        # results.append(self.regex_contract_value_extracted)
        
        # Update status to indicate regex extraction is complete
        set_llm_file_status(
            self.file_id, self.user_id, self.org_id, 3, True, 
            self.retry_count, 1, "", self.start_datetime, "", 
            False, False, self.in_queue, 50, self.logger
        )
        
        # Update the names_dict with extracted values
        for result in results:
            if isinstance(result, dict):
                for field_name, field_value in result.items():
                    if field_name in self.names_dict:
                        self.names_dict[field_name] = field_value
        
        return results
    
    @mlflow.trace(name="MetadataExtractor - Get Semantic")
    def get_semantic(self, null_keys=None):
        """
        Extract metadata using semantic search methods.
        If null_keys is provided, only extract those keys.
        """
        self.logger.info(_log_message("Starting semantic extraction", "get_semantic", MODULE_NAME))
        
        # If null_keys is provided, check if we need to return early for certain fields
        if null_keys:
            for k in null_keys:
                if k == "first":
                    return None
        
        # Run semantic extraction functions in parallel
        results = []
        with ThreadPoolExecutor(max_workers=7) as executor:
            # Define the functions and their arguments
            futures = {
                "contract_duration": self.submit_with_context(executor, get_contract_duration, self.user_id, self.org_id, self.file_id, self.tag_ids, self.logger),
                "title_of_contract": self.submit_with_context(executor, get_title_of_contract, self.user_id, self.org_id, self.file_id, self.tag_ids, self.logger),
                "scope_of_work": self.submit_with_context(executor, get_scope_of_work, self.user_id, self.org_id, self.file_id, self.tag_ids, self.logger),
                "parties_involved": self.submit_with_context(executor, get_parties_involved, self.user_id, self.org_id, self.file_id, self.tag_ids, self.logger),
                "contract_type": self.submit_with_context(executor, get_contract_type, self.user_id, self.org_id, self.file_id, self.file_name, self.tag_ids, self.logger),
                "version_control": self.submit_with_context(executor, get_version_control, self.user_id, self.org_id, self.file_id, self.tag_ids, self.logger),
                "risk_mitigation_score": self.submit_with_context(executor, get_risk_mitigation_score, self.user_id, self.org_id, self.file_id, self.tag_ids, self.logger),
            }
            
            # Collect results as they complete
            for func_name, future in futures.items():
                try:
                    # Format the key as a proper title (e.g., "Contract Duration")
                    title = " ".join(func_name.split('_')).title()
                    
                    # Wait for the future to complete and get the result
                    result = future.result()
                    
                    # Add to results list
                    if isinstance(result, dict) and len(result) == 1:
                        # If result is already a dict with one key-value pair, append it
                        results.append(result)
                    else:
                        # Otherwise create a new dict with the formatted title
                        results.append({title: result})
                    
                    self.logger.info(_log_message(f"Completed {func_name} extraction", "get_semantic", MODULE_NAME))
                except Exception as e:
                    self.logger.error(_log_message(f"Error in {func_name} extraction: {str(e)}", "get_semantic", MODULE_NAME))
                    # Add a null value for this field
                    title = " ".join(func_name.split('_')).title()
                    results.append({title: None})
        
        self.logger.info(_log_message(f"Semantic extraction results before dependent fields: {results}", "get_semantic", MODULE_NAME))
        # Retrieve Contract Type.
        for result in results:
            if "Contract Type" in result:
                contract_type = result["Contract Type"]
                break
        self.logger.info(_log_message(f"Determined Contract Type: {contract_type}", "get_semantic", MODULE_NAME))

        # try:
        #     contract_value_extractor = RegexContractValueExtractor(self.chunks, contract_type, file_id=self.file_id, user_id=self.user_id, org_id=self.org_id, logger=self.logger)

        # Process payment due date separately since it depends on expiration date
        # try:
        #     flag_value, payment_due_date = get_payment_due_date(
        #         self.user_id, self.org_id, self.file_id, self.logger, self.expiry_date
        #     )
        #     self.logger.info(_log_message(f"Payment Due Date: {payment_due_date}", "get_semantic", MODULE_NAME))
        #     results.extend([{'Payment Due Date': payment_due_date}, {'Has Recurring Payment': flag_value}])
        # except Exception as e:
        #     self.logger.error(_log_message(f"Error processing payment due date: {str(e)}", "get_semantic", MODULE_NAME))
        #     results.extend([{'Payment Due Date': None}, {'Has Recurring Payment': None}])


        # Initialize the class
        answers = []
        try:
            dependent_fields_extractor = DependentMetadataFields(
                chunks=self.chunks,
                complete_file_text=self.complete_file_text,
                contract_type=contract_type,
                file_id=self.file_id,
                user_id=self.user_id,
                org_id=self.org_id,
                tag_ids=self.tag_ids,
                logger=self.logger,
                effective_date=self.effective_date,
                expiry_date=self.expiry_date
            )

            # Run the parallel extraction
            answers = dependent_fields_extractor.run_parallel()

            # results will be a list: [contract_value_result, payment_due_date_result]
            self.logger.info(_log_message(f"Dependent fields extraction results: {answers}", "get_semantic", MODULE_NAME))
            results.extend(answers)
            self.logger.info(_log_message(f"Final semantic extraction results: {results}", "get_semantic", MODULE_NAME))
        except Exception as e:
            self.logger.error(_log_message(f"Error processing payment due date and contract value: {str(e)}", "get_semantic", MODULE_NAME))
            results.extend([{'Payment Due Date': None}, {'Has Recurring Payment': None}, {'Contract Value': None}])

        # Update the names_dict with extracted values
        for result in results:
            if isinstance(result, dict):
                for field_name, field_value in result.items():
                    if field_name in self.names_dict:
                        self.names_dict[field_name] = field_value

        self.logger.info(_log_message(f"Final extracted values: {results}", "get_semantic", MODULE_NAME))

        return results
    
    @mlflow.trace(name="MetadataExtractor - Get Value")
    def get_value(self):
        """
        Main extraction method that calls get_regex first, then get_semantic if needed.
        Returns the names and values as a tuple.
        """
        self.logger.info(_log_message("Starting metadata extraction", "get_value", MODULE_NAME))
        
        regex_results = self.get_regex()
        self.logger.info(_log_message(f"Regex extraction results: {regex_results}", "get_value", MODULE_NAME))

        # Then do semantic extraction
        semantic_results = self.get_semantic(self.null_keys)
        self.logger.info(_log_message(f"Semantic extraction results: {semantic_results}", "get_value", MODULE_NAME))

        # Combine results into a final metadata structure
        data = regex_results + semantic_results if semantic_results else regex_results
        self.logger.info(_log_message(f"Combining metadata results: {data}, len: {len(data)}", "get_value", MODULE_NAME))
        dates_metadata, others_metadata = self.map_metadata(
            data
        )
        
        # Update contract duration if possible
        dates_metadata, others_metadata = self.update_contract_duration(dates_metadata, others_metadata)
        self.logger.info(f"other_metadata {others_metadata}")
        # Create the final metadata structure
        metadata = {"metadata": {"dates": dates_metadata, "others": others_metadata}}
        
        # Update status to indicate extraction is complete
        set_llm_file_status(
            self.file_id, self.user_id, self.org_id, 3, True, 
            self.retry_count, 1, "", self.start_datetime, "", 
            False, False, self.in_queue, 75, self.logger
        )
        
        # Save metadata and update final status
        set_meta_data(
            self.file_id, self.user_id, self.org_id, 3, 
            self.start_datetime, datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 
            3, False, "", self.retry_count, metadata, 
            self.in_queue, 100, self.tag_ids, self.logger
        )
        
        # Process contract template
        metadata_vector_handler = ContractMetadataVectorUpserter(self.logger)
        metadata_vector_handler.process_contract_template(
            metadata, self.file_id, self.file_name, self.file_type, 
            self.user_id, self.org_id, len(self.chunks)
        )
        
        # Final status update
        set_llm_file_status(
            self.file_id, self.user_id, self.org_id, 3, True, 
            self.retry_count, 3, "", self.start_datetime, 
            datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 
            False, False, self.in_queue, 100, self.logger
        )
        
        return (self.names_dict.keys(), self.names_dict)

    @mlflow.trace(name="MetadataExtractor - Map Metadata")    
    def map_metadata(self, data):
        """Map the extracted data to the standardized metadata format"""
        self.logger.info(_log_message(f"Mapping metadata with {len(data)} entries", "map_metadata", MODULE_NAME))
        
        # Default structures for dates and other metadata
        dates_metadata = [
            {"title": "Effective Date", "value": None},
            {"title": "Term Date", "value": None},
            {"title": "Payment Due Date", "value": None},
            {"title": "Delivery Date", "value": None},
            {"title": "Termination Date", "value": None},
            {"title": "Renewal Date", "value": None},
            {"title": "Expiration Date", "value": None}
        ]

        others_metadata = [
            {"title": "Title of the Contract", "value": None},
            {"title": "Scope of Work", "value": None},
            {"title": "Parties Involved", "value": None},
            {"title": "Contract Type", "value": None},
            {"title": "File Type", "value": "Contract"},
            {"title": "Jurisdiction", "value": None},
            {"title": "Version Control", "value": None},
            {"title": "Contract Duration", "value": None},
            {"title": "Contract Value", "value": None},
            {"title": "Risk Mitigation Score", "value": None},
            {"title": "Has Recurring Payment", "value": None}
        ]
        
        # Process each entry in the data
        for entry in data:
            if isinstance(entry, dict):
                # Extract the key-value pairs from the dictionary
                for field_name, field_value in entry.items():
                    # Normalize value
                    field_value = None if field_value in ('null', {}, '', [], None) else field_value

                    
                    # Map to dates metadata
                    date_match = next((item for item in dates_metadata if item["title"].lower() == field_name.lower()), None)
                    if date_match:
                        date_match["value"] = field_value
                        continue
                    
                    # Map to others metadata
                    other_match = next((item for item in others_metadata if item["title"].lower() == field_name.lower()), None)
                    if other_match:
                        other_match["value"] = field_value
        
        return dates_metadata, others_metadata
    
    @mlflow.trace(name="MetadataExtractor - Update Contract Duration")
    def update_contract_duration(self, dates_metadata, others_metadata):
        """Update contract duration if it's null but can be derived from dates"""
        # Find the contract duration metadata item
        contract_duration_item = next((item for item in others_metadata if item["title"] == "Contract Duration"), None)
                    
        # If contract duration is empty/None
        if contract_duration_item:
            # Find effective date
            effective_date_item = next((item for item in dates_metadata if item["title"] == "Effective Date"), None)
            # Find expiration date
            expiration_date_item = next((item for item in dates_metadata if item["title"] == "Expiration Date"), None)
            # Find termination date
            termination_date_item = next((item for item in dates_metadata if item["title"] == "Termination Date"), None)
            
            # Check if effective date exists and either expiration or termination date exists
            if effective_date_item and effective_date_item["value"]:
                end_date = None
                
                if expiration_date_item and expiration_date_item["value"]:
                    end_date = expiration_date_item["value"]
                elif termination_date_item and termination_date_item["value"]:
                    end_date = termination_date_item["value"]
                
                # if end_date:
                #     contract_duration_item["value"] = f"from {effective_date_item['value']} to {end_date}"
                
                if end_date and effective_date_item["value"] != end_date:
                    start = datetime.strptime(effective_date_item["value"], "%Y-%m-%d")
                    end = datetime.strptime(end_date, "%Y-%m-%d")
                    ## 2026-01-10 - 2026-30-10 -> 30 
                    # Calculate days difference
                    days_diff = (end - start).days 
                    # Convert to months (month length = 30.436 days)
                    months = round(days_diff / 30.436, 1)  # 2 decimal place
                    # mutating the dict in place.
                    self.logger.info(_log_message(f"Updating contract duration: {months} months", "update_contract_duration", MODULE_NAME))
                    contract_duration_item["value"] = f"{months} months"                
                    
        return dates_metadata, others_metadata


@mlflow.trace(name="Filter Metadata Fields")
def filter_metadata_fields(target_metadata_fields, names):
    if "*" in target_metadata_fields:
        return names
    return [name for name in names if name in target_metadata_fields]

@mlflow.trace(name="Extract Metadata Parallely")
def extract_metadata(file_id, file_name, file_type, user_id, org_id, retry_count, chunks, complete_file_text, target_metadata_fields, tag_ids, logger):
    """Main function to extract metadata from a document"""
    overall_start = time.perf_counter()
    
    try:
        logger.info(_log_message(f"Starting metadata extraction. Received target_metadata_fields - {target_metadata_fields}", "extract_metadata", MODULE_NAME))
        
        # Create MetadataExtractor with required field names
        names = [
            "Effective Date", "Termination Date", "Renewal Date", "Expiration Date", 
            "Delivery Date", "Term Date", "Jurisdiction", "Contract Value",
            "Title of the Contract", "Scope of Work", "Parties Involved", "Contract Type",
            "File Type", "Version Control", "Contract Duration", "Risk Mitigation Score",
            "Payment Due Date", "Has Recurring Payment"
        ]

        
        filtered_names = filter_metadata_fields(target_metadata_fields, names)
        
        extractor = MetadataExtractor(filtered_names)
        extractor.setup_context(file_id, file_name, file_type, user_id, org_id, retry_count, chunks, complete_file_text, tag_ids, logger)
        
        # Extract metadata and get the results
        _, metadata = extractor.get_value()
        logger.info(_log_message(f"Metadata extraction successful: {metadata}", "extract_metadata", MODULE_NAME))
        
        return metadata
        
    except Exception as e:
        error_message = f"Error during metadata extraction: {e}"
        logger.error(_log_message(error_message, "extract_metadata", MODULE_NAME))
        
        # Update status to indicate error
        set_llm_file_status(
            file_id, user_id, org_id, 3, True, retry_count, 2, 
            error_message, datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 
            datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 
            False, False, False, 100, logger
        )
        raise
    
    finally:
        elapsed_time = time.perf_counter() - overall_start
        logger.info(_log_message(f"Metadata extraction completed in {elapsed_time:.2f} seconds.", "extract_metadata", MODULE_NAME))