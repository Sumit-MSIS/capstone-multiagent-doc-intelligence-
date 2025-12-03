from src.common.logger import _log_message
import re
from src.common.file_insights.llm_call import dynamic_llm_call
from nltk.tokenize import word_tokenize
import mlflow
mlflow.openai.autolog()
MODULE_NAME = "regex_jurisdiction_parser"

jurisdiction_regex = r"""(?xi)\b(
    # Governing law clauses
    governed\s+by\s+the\s+laws?\s+of |
    governing\s+law(?:\s+shall\s+be)?(?:\s+of)? |
    shall\s+be\s+governed\s+by(?:\s+the\s+laws?\s+of)? |
    construed\s+(?:and\s+enforced\s+)?(?:in\s+accordance\s+)?with\s+the\s+laws?\s+of |
    interpreted\s+(?:in\s+accordance\s+)?(?:with|under)\s+the\s+laws?\s+of |
    enforced\s+(?:in\s+accordance\s+)?with\s+the\s+laws?\s+of |
    applicable\s+laws?\s+(?:shall\s+be\s+that|are\s+those)\s+of |
    laws?\s+applicable\s+(?:to|in) |

    # Subject‐to/jurisdiction clauses
    subject\s+to\s+(?:the\s+)?jurisdiction\s+of |
    (?:exclusive|non[-\s]?exclusive|sole)\s+jurisdiction\s+of |
    (?:proper|applicable)\s+venue(?:\s+shall\s+be\s+in)? |
    venue\s+shall\s+be\s+in |
    jurisdiction\s+and\s+venue\s+(?:shall|will)\s+(?:be|lie|exist|vest)\s+(?:exclusively\s+)?in |
    courts?\s+of\s+competent\s+jurisdiction\s+(?:in|of|located\s+in) |
    competent\s+jurisdiction\s+(?:in|of|located\s+in) |

    # Legal/forum clauses
    (?:legal|judicial)\s+forum(?:\s+of)? |
    forum\s+for\s+(?:any|all)\s+disputes? |
    forum\s+(?:selection|clause) |
    disputes?\s+(?:shall|will|may)\s+be\s+(?:resolved|determined|heard|litigated|adjudicated|decided|settled)\s+
        (?:in\s+accordance\s+with\s+)?(?:the\s+)?laws?\s+of |
    choice\s+of\s+law |
    conflict\s+of\s+laws? |
    without\s+(?:regard\s+to|giving\s+effect\s+to)\s+(?:its\s+)?conflict\s+of\s+laws?\s+(?:principles|provisions|rules) |
    irrespective\s+of\s+conflict\s+of\s+laws |
    excluding\s+(?:the\s+application\s+of\s+)?(?:any\s+)?conflict\s+of\s+laws?\s+(?:principles|provisions|rules) |

    # Submission clauses
    (?:submitted|subject)\s+to\s+(?:the\s+)?(?:courts?|jurisdiction|tribunals?)(?:\s+of)? |
    submits?\s+(?:themselves|itself|himself|herself)\s+to\s+the\s+jurisdiction\s+of |
    consents?\s+to\s+the\s+jurisdiction\s+of |
    
    # Legal proceedings clauses
    (?:any|all)\s+(?:legal\s+)?proceedings?\s+(?:shall|must|will|may)\s+be\s+brought\s+(?:exclusively\s+)?in |
    (?:any|all)\s+(?:actions?|suits?|claims?|proceedings?)\s+(?:arising|resulting)\s+
        (?:out\s+of|from|under|in\s+connection\s+with)\s+this |
    (?:any|all)\s+disputes?\s+(?:arising|resulting)\s+
        (?:out\s+of|from|under|in\s+connection\s+with)\s+this |
    (?:any|all)\s+(?:legal\s+)?(?:actions?|proceedings?)\s+to\s+enforce |
    
    # Seat/place of arbitration/jurisdiction
    place\s+of\s+(?:arbitration|jurisdiction|performance) |
    seat\s+of\s+(?:arbitration|jurisdiction) |
    arbitration\s+(?:shall|will)\s+be\s+conducted\s+in |
    arbitration\s+proceedings\s+shall\s+take\s+place\s+in |
    arbitration\s+venue\s+shall\s+be |
    
    # Court types and specific references
    (?:federal|state|district|circuit|supreme|high|appellate|superior|county|provincial|municipal)\s+courts?\s+
        (?:sitting|located)\s+in |
    courts?\s+of\s+the\s+(?:state|province|district|county|city|country|nation)\s+of |
    
    # Country-specific jurisdictional terms
    (?:chancery|common\s+law|crown|administrative|commercial|civil|criminal|family|probate|labor|tax|bankruptcy)\s+
        courts?\s+of |
    
    # "Courts of X" or "laws of X" with multi‐word or acronym regions
    (?:courts?|laws?|tribunals?)\s+of\s+
        (?:[A-Z][a-zA-Z\-'.]*(?:\s+(?:and|&|or)\s+[A-Z][a-zA-Z\-'.]*)*
            (?:\s+(?:County|State|Province|Country|Nation|Republic|Kingdom|Government|Commonwealth|Federation|Union|Emirate|Territory))?|
        (?:USA|US|U\.S\.|U\.S\.A\.|UK|U\.K\.|UAE|U\.A\.E\.|EU|E\.U\.|PRC|P\.R\.C\.|ROK|R\.O\.K\.|APAC|EMEA|LATAM))
    |
    
    # Specific country and region references
    # Major countries and regions
    (?:in|of|under\s+the\s+laws?\s+of|subject\s+to\s+the\s+jurisdiction\s+of)\s+
    (?:
        # Countries (major ones and common legal jurisdictions)
        Afghanistan|Albania|Algeria|Andorra|Angola|Antigua\s+and\s+Barbuda|Argentina|Armenia|Australia|Austria|
        Azerbaijan|Bahamas|Bahrain|Bangladesh|Barbados|Belarus|Belgium|Belize|Benin|Bhutan|Bolivia|
        Bosnia\s+and\s+Herzegovina|Botswana|Brazil|Brunei|Bulgaria|Burkina\s+Faso|Burundi|Cabo\s+Verde|Cambodia|
        Cameroon|Canada|Central\s+African\s+Republic|Chad|Chile|China|Colombia|Comoros|Congo|Costa\s+Rica|
        Croatia|Cuba|Cyprus|Czech\s+Republic|Denmark|Djibouti|Dominica|Dominican\s+Republic|
        East\s+Timor|Ecuador|Egypt|El\s+Salvador|Equatorial\s+Guinea|Eritrea|Estonia|Eswatini|Ethiopia|
        Fiji|Finland|France|Gabon|Gambia|Georgia|Germany|Ghana|Greece|Grenada|Guatemala|Guinea|Guinea-Bissau|
        Guyana|Haiti|Honduras|Hungary|Iceland|India|Indonesia|Iran|Iraq|Ireland|Israel|Italy|
        Jamaica|Japan|Jordan|Kazakhstan|Kenya|Kiribati|Korea|Kosovo|Kuwait|Kyrgyzstan|Laos|Latvia|
        Lebanon|Lesotho|Liberia|Libya|Liechtenstein|Lithuania|Luxembourg|Madagascar|Malawi|Malaysia|
        Maldives|Mali|Malta|Marshall\s+Islands|Mauritania|Mauritius|Mexico|Micronesia|Moldova|Monaco|
        Mongolia|Montenegro|Morocco|Mozambique|Myanmar|Namibia|Nauru|Nepal|Netherlands|New\s+Zealand|
        Nicaragua|Niger|Nigeria|North\s+Korea|North\s+Macedonia|Norway|Oman|Pakistan|Palau|Palestine|
        Panama|Papua\s+New\s+Guinea|Paraguay|Peru|Philippines|Poland|Portugal|Qatar|Romania|Russia|
        Rwanda|Saint\s+Kitts\s+and\s+Nevis|Saint\s+Lucia|Saint\s+Vincent\s+and\s+the\s+Grenadines|Samoa|
        San\s+Marino|Sao\s+Tome\s+and\s+Principe|Saudi\s+Arabia|Senegal|Serbia|Seychelles|Sierra\s+Leone|
        Singapore|Slovakia|Slovenia|Solomon\s+Islands|Somalia|South\s+Africa|South\s+Korea|South\s+Sudan|
        Spain|Sri\s+Lanka|Sudan|Suriname|Sweden|Switzerland|Syria|Taiwan|Tajikistan|Tanzania|Thailand|
        Togo|Tonga|Trinidad\s+and\s+Tobago|Tunisia|Turkey|Turkmenistan|Tuvalu|Uganda|Ukraine|
        United\s+Arab\s+Emirates|United\s+Kingdom|United\s+States(?:\s+of\s+America)?|Uruguay|Uzbekistan|
        Vanuatu|Vatican\s+City|Venezuela|Vietnam|Yemen|Zambia|Zimbabwe|
        Hong\s+Kong|Macau|Puerto\s+Rico|Scotland|Northern\s+Ireland|Wales|England|
        
        # US States
        Alabama|Alaska|Arizona|Arkansas|California|Colorado|Connecticut|Delaware|Florida|Georgia|
        Hawaii|Idaho|Illinois|Indiana|Iowa|Kansas|Kentucky|Louisiana|Maine|Maryland|
        Massachusetts|Michigan|Minnesota|Mississippi|Missouri|Montana|Nebraska|Nevada|
        New\s+Hampshire|New\s+Jersey|New\s+Mexico|New\s+York|North\s+Carolina|North\s+Dakota|
        Ohio|Oklahoma|Oregon|Pennsylvania|Rhode\s+Island|South\s+Carolina|South\s+Dakota|
        Tennessee|Texas|Utah|Vermont|Virginia|Washington|West\s+Virginia|Wisconsin|Wyoming|
        District\s+of\s+Columbia|D\.C\.|Washington\s+D\.C\.|
        
        # Canadian Provinces/Territories
        Alberta|British\s+Columbia|Manitoba|New\s+Brunswick|Newfoundland\s+and\s+Labrador|
        Northwest\s+Territories|Nova\s+Scotia|Nunavut|Ontario|Prince\s+Edward\s+Island|Quebec|
        Saskatchewan|Yukon|
        
        # Australian States/Territories
        New\s+South\s+Wales|Queensland|South\s+Australia|Tasmania|Victoria|Western\s+Australia|
        Australian\s+Capital\s+Territory|Northern\s+Territory|
        
        # Indian States/Territories
        Andhra\s+Pradesh|Arunachal\s+Pradesh|Assam|Bihar|Chhattisgarh|Goa|Gujarat|Haryana|
        Himachal\s+Pradesh|Jharkhand|Karnataka|Kerala|Madhya\s+Pradesh|Maharashtra|Manipur|
        Meghalaya|Mizoram|Nagaland|Odisha|Punjab|Rajasthan|Sikkim|Tamil\s+Nadu|Telangana|
        Tripura|Uttar\s+Pradesh|Uttarakhand|West\s+Bengal|Andaman\s+and\s+Nicobar\s+Islands|
        Chandigarh|Dadra\s+and\s+Nagar\s+Haveli\s+and\s+Daman\s+and\s+Diu|Delhi|Jammu\s+and\s+Kashmir|
        Ladakh|Lakshadweep|Puducherry|
        
        # UK Countries/Regions
        England|Scotland|Wales|Northern\s+Ireland|
        
        # Chinese Provinces/Regions
        Anhui|Beijing|Chongqing|Fujian|Gansu|Guangdong|Guangxi|Guizhou|Hainan|Hebei|
        Heilongjiang|Henan|Hubei|Hunan|Inner\s+Mongolia|Jiangsu|Jiangxi|Jilin|Liaoning|
        Ningxia|Qinghai|Shaanxi|Shandong|Shanghai|Shanxi|Sichuan|Tianjin|Tibet|Xinjiang|
        Yunnan|Zhejiang|
        
        # Brazilian States
        Acre|Alagoas|Amapá|Amazonas|Bahia|Ceará|Espírito\s+Santo|Goiás|Maranhão|
        Mato\s+Grosso|Mato\s+Grosso\s+do\s+Sul|Minas\s+Gerais|Pará|Paraíba|Paraná|
        Pernambuco|Piauí|Rio\s+de\s+Janeiro|Rio\s+Grande\s+do\s+Norte|Rio\s+Grande\s+do\s+Sul|
        Rondônia|Roraima|Santa\s+Catarina|São\s+Paulo|Sergipe|Tocantins|
        
        # Mexican States
        Aguascalientes|Baja\s+California|Baja\s+California\s+Sur|Campeche|Chiapas|
        Chihuahua|Coahuila|Colima|Durango|Guanajuato|Guerrero|Hidalgo|Jalisco|
        México|Mexico\s+City|Michoacán|Morelos|Nayarit|Nuevo\s+León|Oaxaca|Puebla|
        Querétaro|Quintana\s+Roo|San\s+Luis\s+Potosí|Sinaloa|Sonora|Tabasco|
        Tamaulipas|Tlaxcala|Veracruz|Yucatán|Zacatecas|
        
        # German States
        Baden-Württemberg|Bavaria|Berlin|Brandenburg|Bremen|Hamburg|Hesse|
        Lower\s+Saxony|Mecklenburg-Vorpommern|North\s+Rhine-Westphalia|Rhineland-Palatinate|
        Saarland|Saxony|Saxony-Anhalt|Schleswig-Holstein|Thuringia|
        
        # Russian Regions (Federal Subjects)
        Moscow|Saint\s+Petersburg|Adygea|Altai|Bashkortostan|Buryatia|Chechnya|
        Chuvashia|Dagestan|Ingushetia|Kabardino-Balkaria|Kalmykia|Karachay-Cherkessia|
        Karelia|Khakassia|Komi|Mari\s+El|Mordovia|North\s+Ossetia-Alania|Tatarstan|
        Tuva|Udmurtia|Sakha|Yakutia
    )
)\b"""

jurisdiction_instruction ="""  
**Task**: Extract **jurisdiction(s)** from the contract text.  
- A contract may specify **multiple jurisdictions** for different purposes (e.g., "governed by California laws" and "disputes resolved in Delaware courts").  
- Return **all valid jurisdictions** as a **single string**, joined by "and" (e.g., `"California and Delaware"`).  
- Include jurisdictions **only if explicitly stated** (e.g., "laws of [Location]", "courts of [Location]").  
- Return "null" if:  
  - No jurisdiction is mentioned.  
  - Jurisdiction is vague (e.g., "applicable laws", "competent courts" without specifics).  
  - Ambiguous or inferred jurisdictions.  

**Output Format**:  
Return a JSON object with the key "Jurisdiction". 
-**Strictly do not include "```" or "json" or any markers in your response.**
Examples:  
{"Jurisdiction": "Texas"}  # Single jurisdiction  
{"Jurisdiction": "Singapore and UK"}  # Multiple jurisdictions  
{"Jurisdiction": "null"}  # No valid jurisdiction"""

@mlflow.trace(name="Clean and Split Sentences")
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

@mlflow.trace(name="Get Regex Jurisdiction")
def get_regex_jurisdiction(chunks, tag_ids, logger):
    try:
        combined_text = []
        seen_sentences = set()  # Track sentences to avoid repetition
        chunks = "\n".join([chunk for chunk in chunks])
        sentences = clean_and_split_sentences(chunks)
        regex_pattern = jurisdiction_regex
        i = 0
        logger.info(_log_message(f"INSIDE JURISDICTION", "get_regex_jurisdiction", MODULE_NAME))
        logger.info(_log_message(f"length of Sentences: {len(sentences)}", "get_regex_jurisdiction", MODULE_NAME))

        print("inside jurisdiction regex")
        # Create a list to keep track of processed indices
        processed_indices = set()

        for i in range(len(sentences)):
            # Skip if this index was already processed
            if i in processed_indices:
                continue
                
            if re.search(regex_pattern, sentences[i], re.VERBOSE):
                # Extract 1 sentence before, the matched sentence, and 1 sentence after
                start = max(0, i - 1)
                end = min(len(sentences), i + 2)  # Include the matched sentence and 1 after
                selected_sentences = sentences[start:end]
                
                # Mark these indices as processed to avoid revisiting them
                for idx in range(start, end):
                    processed_indices.add(idx)
                    
                # Limit the length of sentences before and after to 30 words
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
        
        # Combine all matched text into a single chunk
        jurisdiction_context = ""
        # Combine all matched text into a single chunk
        if combined_text:
            filtered = [{"chunk_number": "combined", "text": " ".join(combined_text)}]
            jurisdiction_context  = filtered[0]['text'] if filtered else ""

        if not jurisdiction_context:
            logger.debug(_log_message("No jurisdiction context found using regex", "get_regex_jurisdiction", MODULE_NAME))
            return {"Jurisdiction": "null"}
        
        logger.info(_log_message(f"Context filtered for jurisdiction extraction:{len(jurisdiction_context)}", "get_regex_jurisdiction", MODULE_NAME))
        system_prompt = """You are an assistant that understands and extracts jurisdiction from the legal contract context"""

        user_prompt = f"""
        {jurisdiction_instruction}
        Below is the text paragraph consisting of all the required context filtered from the legal agreement's, contract's context:
        ---
        {jurisdiction_context}
        ---
        Output should be a minimal json, DO NOT provide any extra words or markers '```' OR '```json'."""

        json_schema = {
            "type": "json_schema",
            "json_schema": {
                "name": "jurisdiction_extraction",
                "schema": {
                "type": "object",
                "properties": {
                    "Jurisdiction": {
                    "type": ["string", "null"],
                    "description": "The name of the jurisdiction (such as a country, state, or region) that governs the contract, or null if not found."
                    }
                },
                "required": ["Jurisdiction"],
                "additionalProperties": False
                },
                "strict": True
            }
        }

        llm_response = dynamic_llm_call(user_prompt, system_prompt, json_schema, logger)  
        logger.info(_log_message(f"Jurisdiction extraction response: {llm_response}", "get_regex_jurisdiction", MODULE_NAME))
        return llm_response
    except Exception as e:
        logger.error(_log_message(f"Error in jurisdiction extraction: {str(e)}", "get_regex_jurisdiction", MODULE_NAME))
        return {"Jurisdiction": "null"}