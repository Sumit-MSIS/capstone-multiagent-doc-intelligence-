from src.common.logger import _log_message
import re
from src.common.file_insights.llm_call import dynamic_llm_call
from nltk.tokenize import word_tokenize
import mlflow
mlflow.openai.autolog()


MODULE_NAME = "regex_date_extraction"

date_pattern = r"""
\b(
    # === DOCUMENT-SPECIFIC DATE FIELD PATTERNS ===
    
    # Effective / Commencement / Execution / Signing Date patterns
    (?:                                     # any of these leading phrases
        effective\s+date |
        commencement\s+date |
        execution\s+date |
        date\s+of\s+signing |
        signed\s+on |
        made\s+as\s+of
    )\s*[:;]?\s*
    (?:
        \d{1,2}[-/\.]\d{1,2}[-/\.]\d{2,4} |
        \d{4}[-/\.]\d{1,2}[-/\.]\d{1,2} |
        (?:\d{1,2}(?:st|nd|rd|th)?[\s,]*)?
        (?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|
           Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|
           Dec(?:ember)?)
        [,\s]*\d{1,2}(?:st|nd|rd|th)?[,\s]*\d{2,4}
    )
    |
        # "Dated this X day of Month, YYYY"
        dated\s+this\s+\d{1,2}(?:st|nd|rd|th)?\s+day\s+of\s+
        (?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|
        Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|
        Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)
        ,?\s*\d{4}
    |
        # "This agreement is made as of ..."
        (?:(?:this\s+agreement\s+is|this\s+agreement)\s+made\s+as\s+of)\s*
        .{0,50}?
        (?:
            \d{1,2}[-/\.]\d{1,2}[-/\.]\d{2,4} |
            (?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|
            May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|
            Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)
            [,\s]*\d{1,2}(?:st|nd|rd|th)?[,\s]*\d{2,4}
        )
    |
        # Termination Date patterns
        (?:termination\s+date|terminates)\s*
        (?:[:;]|\s+(?:on|at|after|before|by))\s*
        (?:
            \d{1,2}[-/\.]\d{1,2}[-/\.]\d{2,4} |
            \d{4}[-/\.]\d{1,2}[-/\.]\d{1,2} |
            (?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|
            Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|
            Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)
            [,\s]*\d{1,2}(?:st|nd|rd|th)?[,\s]*\d{2,4}
        )
    |
        # Expiration / Expiry Date patterns
        (?:expiration\s+date|expiry\s+date|expires)\s*
        (?:[:;]|\s+(?:on|at|after|before|by))\s*
        (?:
            \d{1,2}[-/\.]\d{1,2}[-/\.]\d{2,4} |
            \d{4}[-/\.]\d{1,2}[-/\.]\d{1,2} |
            (?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|
            Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|
            Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)
            [,\s]*\d{1,2}(?:st|nd|rd|th)?[,\s]*\d{2,4}
        )
    |
        # Term Date / Commencement of Term patterns
        (?:term\s+date|term\s+commence(?:ment)?)\s*
        (?:[:;]|\s+(?:on|at|from))\s*
        (?:
            \d{1,2}[-/\.]\d{1,2}[-/\.]\d{2,4} |
            \d{4}[-/\.]\d{1,2}[-/\.]\d{1,2} |
            (?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|
            Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|
            Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)
            [,\s]*\d{1,2}(?:st|nd|rd|th)?[,\s]*\d{2,4}
        )
    |
        # Payment / Delivery / Renewal Date patterns
        (?:payment\s+due\s+date|delivery\s+date|renewal\s+date)\s*
        (?:[:;]|\s+(?:on|at|after|before|by))\s*
        (?:
            \d{1,2}[-/\.]\d{1,2}[-/\.]\d{2,4} |
            \d{4}[-/\.]\d{1,2}[-/\.]\d{1,2} |
            (?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|
            Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|
            Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)
            [,\s]*\d{1,2}(?:st|nd|rd|th)?[,\s]*\d{2,4}
        )
    |
        # Section headings with dates
        (?:^|\n|\r)\s*
        (?:effective|term|termination|expiration|renewal|payment\s+due|delivery)\s+date
        [\s*:-]*\s*
        (?:
            \d{1,2}[-/\.]\d{1,2}[-/\.]\d{2,4} |
            \d{4}[-/\.]\d{1,2}[-/\.]\d{1,2} |
            (?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|
            Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|
            Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)
            [,\s]*\d{1,2}(?:st|nd|rd|th)?[,\s]*\d{2,4}
        )
    # === GENERAL DATE PATTERNS ===
    
     # ISO 8601 with time components
    \d{4}[-/]\d{1,2}[-/]\d{1,2}[T ]\d{1,2}:\d{2}(?::\d{2})?(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})? |
    
    # YYYY-MM-DD or YYYY/MM/DD or YYYY.MM.DD
    \d{4}[-/\.]\d{1,2}[-/\.]\d{1,2} |
    
    # DD-MM-YYYY or DD/MM/YYYY or DD.MM.YYYY
    \d{1,2}[-/\.]\d{1,2}[-/\.]\d{2,4} |
    
    # DD MM YYYY or YYYY MM DD (space separated)
    \d{1,2}\s+\d{1,2}\s+\d{4} |
    \d{4}\s+\d{1,2}\s+\d{1,2} |
    
    # Month DD, YYYY or DD Month YYYY
    (?:\d{1,2}(?:st|nd|rd|th)?[ -/]?)?(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)[ -/]?(?:\d{1,2}(?:st|nd|rd|th)?[,]?[ -/])?\d{2,4} |
    
    # Month YYYY or Month DD
    (?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)[ -/\.]\d{4} |
    (?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)[ -/\.]\d{1,2}(?:st|nd|rd|th)? |

    # Day of week with date
    (?:Mon(?:day)?|Tue(?:sday)?|Wed(?:nesday)?|Thu(?:rsday)?|Fri(?:day)?|Sat(?:urday)?|Sun(?:day)?)[,]?\s+(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{1,2}(?:st|nd|rd|th)?[,]?\s+\d{4} |
    
    # Quarter notation
    Q[1-4]\s+\d{4} |
    \d{4}\s+Q[1-4] |
    [1-4]Q\s+\d{4} |
    \d{4}\s+[1-4]Q |
    (?:First|Second|Third|Fourth|1st|2nd|3rd|4th)\s+quarter\s+\d{4} |
    \d{4}\s+(?:First|Second|Third|Fourth|1st|2nd|3rd|4th)\s+quarter |
    
    # Week notation
    Week\s+\d{1,2}\s+\d{4} |
    \d{4}\s+Week\s+\d{1,2} |
    W\d{1,2}\s+\d{4} |
    \d{4}\s+W\d{1,2} |
    W\d{1,2}-\d{4} |
    
    # Month/Year formats
    \d{1,2}/\d{4} |
    \d{1,2}-\d{4} |
    
    # Date ranges
    \d{4}[-/\.]\d{1,2}[-/\.]\d{1,2}\s+(?:to|through|thru|-)\s+\d{4}[-/\.]\d{1,2}[-/\.]\d{1,2} |
    \d{1,2}[-/\.]\d{1,2}[-/\.]\d{4}\s+(?:to|through|thru|-)\s+\d{1,2}[-/\.]\d{1,2}[-/\.]\d{4} |
    (?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{1,2}(?:st|nd|rd|th)?\s+-\s+(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{1,2}(?:st|nd|rd|th)?[,]?\s+\d{4} |
    
    # Fiscal year notation
    FY\s*\d{2,4} |
    Fiscal\s+Year\s+\d{4} |
    Fiscal\s+\d{4} |
    
    # Chinese/Japanese year format (e.g., 令和5年10月1日)
    [令和|平成|昭和|大正|明治]\d{1,2}年\d{1,2}月\d{1,2}日 |
    
    # Timestamps with dates
    \d{1,2}[-/\.]\d{1,2}[-/\.]\d{2,4}\s+\d{1,2}:\d{2}(?::\d{2})?\s*(?:AM|PM|am|pm)? |
    \d{4}[-/\.]\d{1,2}[-/\.]\d{1,2}\s+\d{1,2}:\d{2}(?::\d{2})?\s*(?:AM|PM|am|pm)? |
    
    # Unix timestamp (10-13 digits)
    \b\d{10,13}\b |
    
    # RFC 2822 format
    (?:Mon|Tue|Wed|Thu|Fri|Sat|Sun),\s+\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{4}\s+\d{2}:\d{2}(?::\d{2})?\s+(?:[+-]\d{4}|UTC|GMT|[A-Z]{3}) |
    
    # French date format (e.g., 1er janvier 2023)
    \d{1,2}(?:er|ème|e|ère)?\s+(?:janvier|février|mars|avril|mai|juin|juillet|août|septembre|octobre|novembre|décembre)\s+\d{4} |
    
    # Spanish date format
    \d{1,2}\s+de\s+(?:enero|febrero|marzo|abril|mayo|junio|julio|agosto|septiembre|octubre|noviembre|diciembre)(?:\s+de)?\s+\d{4} |
    
    # German date format
    \d{1,2}\.\s+(?:Januar|Februar|März|April|Mai|Juni|Juli|August|September|Oktober|November|Dezember)\s+\d{4} |
    
    # Year-specific patterns
    
    # Year preceded by words
    (?:year|in|during|of|for|by|before|after|since|until|around|circa|ca\.?|c\.?)\s+\d{4} |
    
    # Years with era designations
    \d{1,4}\s*(?:AD|BC|BCE|CE|B\.C\.(?:E\.)?|C\.E\.|A\.D\.) |
    (?:AD|BC|BCE|CE|B\.C\.(?:E\.)?|C\.E\.|A\.D\.)\s*\d{1,4} |
    
    # Decade references
    \d{3}0s |
    \d{4}s |
    \d{4}-\d{2} |
    \d{2}s |
    (?:the\s+)?(?:twenty|nineteen|eighteen|seventeen|sixteen|fifteen|fourteen|thirteen|twelve|twenty-first|twentieth|nineteenth|eighteenth|seventeenth|sixteenth|fifteenth|fourteenth|thirteenth|twelfth)\s+(?:century|hundreds) |
    (?:the\s+)?(?:twenties|thirties|forties|fifties|sixties|seventies|eighties|nineties) |
    
    # Year ranges
    \d{4}\s*(?:-|to|through|until|and|&|–|—)\s*\d{4} |
    \d{4}\s*(?:-|to|through|until|and|&|–|—)\s*\d{2} |
    
    # Academic/school years
    (?:academic\s+year\s+)?\d{4}[-/]\d{2,4} |
    (?:academic\s+year\s+)?\d{4}[-/]\d{2} |
    (?:AY|SY)\s*\d{4}[-/]\d{2,4} |
    
    # Centuries
    (?:(?:\d{1,2}(?:st|nd|rd|th)?|(?:twenty|nineteen|eighteen|seventeen|sixteen|fifteen|fourteen|thirteen|twelve|twenty-first|twentieth|nineteenth|eighteenth|seventeenth|sixteenth|fifteenth|fourteenth|thirteenth|twelfth))\s+century) |
    
    # Year only (must be a complete 4-digit number)
    \b\d{4}\b |
    
    # ----- NEW PATTERNS BELOW -----
    
    # Relative time expressions
    (?:(?:last|next|previous|coming|this)\s+(?:week|month|year|decade|century|spring|summer|fall|autumn|winter)) |
    (?:(?:\d{1,3}|a|an|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|thousand|few|several|many|couple)\s+(?:second|minute|hour|day|week|fortnight|month|quarter|year|decade|century)s?\s+(?:ago|from\s+now|hence|later|earlier|before|after)) |
    
    # Specific day references
    (?:today|yesterday|tomorrow|day\s+before\s+yesterday|day\s+after\s+tomorrow) |
    
    # Days of week references
    (?:on\s+)?(?:Mon(?:day)?|Tue(?:sday)?|Wed(?:nesday)?|Thu(?:rsday)?|Fri(?:day)?|Sat(?:urday)?|Sun(?:day)?)(?:\s+(?:morning|afternoon|evening|night))? |
    
    # Month/season + year patterns
    (?:(?:early|mid|late|beginning\s+of|end\s+of)\s+)?(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+(?:of\s+)?\d{4} |
    
    # Season + year patterns
    (?:(?:early|mid|late|beginning\s+of|end\s+of)\s+)?(?:Spring|Summer|Fall|Autumn|Winter)\s+(?:of\s+)?\d{4} |
    \d{4}\s+(?:Spring|Summer|Fall|Autumn|Winter) |
    
    # Holiday references with year
    (?:Christmas|Easter|Thanksgiving|Halloween|New\s+Year(?:'s)?|Valentine's\s+Day|St\.\s+Patrick's\s+Day|Independence\s+Day|Labor\s+Day|Memorial\s+Day|Veterans\s+Day|MLK\s+Day|Martin\s+Luther\s+King\s+Day|Presidents'\s+Day|Columbus\s+Day|Hanukkah|Passover|Rosh\s+Hashanah|Yom\s+Kippur|Diwali|Eid(?:\s+al-Fitr|\s+al-Adha)?|Ramadan|Chinese\s+New\s+Year|Lunar\s+New\s+Year)\s+(?:of\s+)?\d{4} |
    \d{4}\s+(?:Christmas|Easter|Thanksgiving|Halloween|New\s+Year(?:'s)?|Valentine's\s+Day|St\.\s+Patrick's\s+Day|Independence\s+Day|Labor\s+Day|Memorial\s+Day|Veterans\s+Day|MLK\s+Day|Martin\s+Luther\s+King\s+Day|Presidents'\s+Day|Columbus\s+Day|Hanukkah|Passover|Rosh\s+Hashanah|Yom\s+Kippur|Diwali|Eid(?:\s+al-Fitr|\s+al-Adha)?|Ramadan|Chinese\s+New\s+Year|Lunar\s+New\s+Year) |
    
    # Duration expressions
    (?:for|during|over|within|in|throughout|across|spanning)\s+(?:\d{1,3}|a|an|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|thousand|few|several|many|couple)\s+(?:second|minute|hour|day|week|month|year|decade|century)s? |
    
    # Time periods like "first half of 2023"
    (?:first|second|third|fourth|1st|2nd|3rd|4th|H1|H2|initial|early|mid|middle|late|latter|final)\s+(?:half|part|portion|quarter)\s+of\s+(?:\d{4}|the\s+year) |
    (?:Q[1-4]|[1-4]Q)\s+of\s+\d{4} |
    
    # Date and day ordinals
    (?:(?:the\s+)?\d{1,2}(?:st|nd|rd|th)\s+(?:of\s+)?(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)) |
    
    # Approximate dates
    (?:circa|ca\.|c\.|around|approximately|about|roughly|in\s+or\s+around|somewhere\s+around)\s+\d{4} |
    
    # Frequency expressions
    (?:(?:every|each)\s+(?:other\s+)?(?:second|minute|hour|day|week|month|quarter|year|decade|century|Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday|Mon|Tue|Wed|Thu|Fri|Sat|Sun|January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Oct|Nov|Dec)) |
    (?:daily|weekly|monthly|quarterly|yearly|annually|bi-weekly|bi-monthly|bi-annually|semi-annually|fortnightly) |
    
    # Time periods with prepositions
    (?:from|since|between|after|before|prior\s+to|following|as\s+of|as\s+at|starting|ending|beginning|until|till|up\s+to)\s+(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+(?:\d{1,2}(?:st|nd|rd|th)?\s+)?(?:\d{4})? |
    
    # Age references
    (?:aged|age)\s+\d{1,3} |45 j
    \d{1,3}\s+(?:years|months|weeks|days)\s+old |
    
    # Century parts
    (?:early|mid|late|beginning\s+of|end\s+of)\s+(?:the\s+)?\d{1,2}(?:st|nd|rd|th)\s+century |
    (?:early|mid|late|beginning\s+of|end\s+of)\s+(?:the\s+)?(?:twentieth|nineteenth|eighteenth|seventeenth|sixteenth|fifteenth|fourteenth|thirteenth|twelfth|twenty-first)\s+century |
    
    # Time spans with "from...to" format
    from\s+\d{4}\s+to\s+\d{4} |
    from\s+(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+(?:\d{1,2})?\s+to\s+(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+(?:\d{1,2})?\s+\d{4} |
    
    # Recurring dates
    (?:every|each)\s+(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday|Mon|Tue|Wed|Thu|Fri|Sat|Sun|January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2}(?:st|nd|rd|th)? |
    
    # X days/weeks/months/years ago/from now
    \d{1,3}\s+(?:second|minute|hour|day|week|month|year|decade)s?\s+(?:ago|from\s+now|later|before|after|hence) |
    
    # Morning/afternoon/evening references
    (?:this|tomorrow|yesterday|next|last|previous|coming)\s+(?:morning|afternoon|evening|night) |
    
    # Weekend references
    (?:this|next|last|previous|coming)\s+weekend |
    
    # Named periods
    (?:the\s+)?(?:Great\s+Depression|Renaissance|Medieval\s+period|Middle\s+Ages|Stone\s+Age|Bronze\s+Age|Iron\s+Age|Industrial\s+Revolution|Digital\s+Age|Information\s+Age|Cold\s+War|World\s+War\s+(?:I|II|One|Two|1|2)|Victorian\s+Era|Georgian\s+Era|Edwardian\s+Era|Elizabethan\s+Era|Tudor\s+period|Roman\s+Empire|Byzantine\s+Empire|Ming\s+Dynasty|Qing\s+Dynasty|Han\s+Dynasty|Ottoman\s+Empire|Paleolithic|Mesolithic|Neolithic|post-war\s+era) |
    
    # Specific time period references
    (?:the\s+)?(?:Stone|Bronze|Iron|Dark|Middle|Golden|Modern|Post-modern|Contemporary|Colonial|Post-colonial|Post-war|Pre-war|Antebellum|Post-apocalyptic)\s+(?:Age|Era|Period) |
    
    # Business quarters and fiscal periods
    (?:Q[1-4]|[1-4]Q|first\s+quarter|second\s+quarter|third\s+quarter|fourth\s+quarter|H1|H2|first\s+half|second\s+half)\s+(?:FY)?\s*\d{2,4} |
    
    # Anniversary references
    \d{1,3}(?:st|nd|rd|th)?\s+anniversary |
    
    # Vague time references
    (?:a\s+(?:short|long)\s+(?:time|while)\s+(?:ago|from\s+now)|some\s+time\s+(?:ago|from\s+now)|in\s+recent\s+(?:days|weeks|months|years)|in\s+the\s+(?:near|distant)\s+(?:future|past))
)\b
"""

#######################################################################################################################################################################################################

# New prompt - 3rd sep 2025

#######################################################################################################################################################################################################
#######################################################################################################################################################################################################


date_extraction_instructions = """
You are tasked with extracting multiple types of dates from a contract or agreement document.  
All extracted dates must be returned strictly in the format: "YYYY-MM-DD".  
If a date is not present or cannot be confidently extracted, return the string "null".  

---

### Field-Specific Instructions

**1. Effective Date**  
- Definition: Date the agreement begins, becomes active, or is deemed effective.  
- Keywords: "effective date", "made effective as of", "commencement date", "start date", "executed on", "entered into on", "dated as of".  
- Location: Typically in the opening recital or first paragraph.  

**2. Expiration Date**  
- Definition: Final date when the agreement naturally ends at the conclusion of its term.  
- Keywords: "expiration date", "this agreement expires on", "contract ends on", "end of term", "valid until", "shall continue until",  "termination date", "shall terminate on", "may be terminated on", "early termination", "termination effective on".  
- Rule: Only extract if explicitly mentioned. Do not assume dates.

**4. Renewal Date**  
- Definition: Date the agreement automatically renews or is extended.  
- Keywords: "renewal date", "renewed on", "automatically renew on", "extension begins", "renewal term commences", "option to renew on".  
- Location: Typically in term/renewal provisions.  
 
**5. Delivery Date**  
- Definition: Date goods/services are scheduled to be or were delivered, or when obligations/performance are due.  
- Keywords: "delivery date", "shall be delivered on", "completion date", "scheduled delivery", "due date", "shipment date", "to be provided by", "performance completed by".  
- Location: Typically in performance, deliverables sections 

---

### General Rules for All Dates
1. **Formatting**  
   - Use "YYYY-MM-DD" format for all valid dates.  
   - Do not fabricate missing components.  

2. **Incomplete or Placeholder Dates**  
   - If only a year is provided (e.g., "2019") → return "null".  
   - If only a month and year are provided (e.g., "August 2024", "082024", "08/2024") → return "null".  
   - If placeholders appear (e.g., "as of __________, 2019") → return "null".  
    - For example:
            - "August 2024" → return "null"
            - "082024" → return "null"
            - "2024" → return "null"
   - Do not assume or construct default days like "01" or "20".  

3. **Reference-Based Dates**  
   - If a date is relative (e.g., "15 years after Effective Date", "30 days after delivery"):  
     a. Identify the reference date.  
     b. Identify the period of time.  
     c. Calculate the actual date **only if both values are explicitly provided**.  
     d. If calculation is not possible, return "null".  
   - Example: Effective Date = "2020-01-01", Expiration = "15 years after Effective Date" → return "2035-01-01".  

4. **Context Awareness**  
   - Only extract dates explicitly tied to valid keywords for each field.  
   - Give priority to amended/superseding dates if multiple are present.  
   - If multiple candidates exist, return the one most specific and contextually appropriate.  
   - Do not confuse expiration date with termination dates, identify them seprately

5. **No Assumptions**

    - Do not infer or approximate dates.
    - Do not confuse fields (e.g., termination vs expiration).
    - If multiple dates match, select the most specific and contextually correct one. If ambiguity remains, return "null".
---

### Output Format
Return a single JSON object with all five fields:  

{
  "Effective Date": "<YYYY-MM-DD or 'null'>",
  "Renewal Date": "<YYYY-MM-DD or 'null'>",
  "Expiration Date": "<YYYY-MM-DD or 'null'>",
  "Delivery Date": "<YYYY-MM-DD or 'null'>"
}
"""

@mlflow.trace(name="Clean and Split Sentences")
def clean_and_split_sentences(text):
    """
    Cleans markdown/special characters while preserving:
    - Bullet points (*, -, •) and numbering
    - Clause numbers (1.1, 2.3.4) and numerical values (11.5, 2.5%)
    - Numbered lists (e.g., "8. Text")
    - Doesn't remove dots in non-sentence-ending contexts
    Splits text into sentences only at valid sentence-ending periods.
    """
    # Remove markdown images and links
    text = re.sub(r'!\[.*?\]\(.*?\)', '', text)
    text = re.sub(r'\[.*?\]\(.*?\)', '', text)
    
    # Preserve bullet points (*, -, •) and numbering patterns
    # Only remove problematic markdown symbols (preserve asterisks for bullet points)
    text = re.sub(r'[_~`#+=|>\[\](){}!\\<>]', '', text)
    
    # Collapse multiple spaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Protect special patterns using temporary placeholders
    protected_patterns = [
        (r'\d+\.\d+', '<NUMDOT>'),        # Number patterns (1.23, 45.67)
        (r'\b\d+\.(\s+|$)', '<LISTDOT>'),  # Numbered lists (8. , 1. )
        (r'\b[A-Za-z]\.', '<LETTERDOT>'),  # Single-letter references (A., B.)
        (r'\d+\.\d+%', '<PERCENTDOT>'),    # Percentages (2.5%)
        (r'\d+,\d+\.\d+', '<COMMADOT>')    # Comma numbers (15,005.3)
    ]
    
    # Apply protection in reverse order to avoid overlap
    for pattern, placeholder in reversed(protected_patterns):
        text = re.sub(pattern, lambda m: m.group().replace('.', placeholder), text)
    
    # Split sentences at periods followed by space or end-of-string
    # print(text)
    sentences = re.split(r'\.(?:\s+|$)', text)
    
    # Process and restore protected patterns
    result = []
    for s in sentences:
        s = s.strip()
        if not s:
            continue
            
        # Restore all protected patterns
        s = s.replace('<NUMDOT>', '.')
        s = s.replace('<LISTDOT>', '.')
        s = s.replace('<LETTERDOT>', '.')
        s = s.replace('<PERCENTDOT>', '.')
        s = s.replace('<COMMADOT>', '.')
        
        # Handle numbered list fragments
        if re.fullmatch(r'\d+', s) and result and re.fullmatch(r'\d+', result[-1]):
            result[-1] += '.'  # Add missing dot between numbers
        result.append(s)
    
    # Merge numbered list fragments with their text
    final_result = []
    i = 0
    while i < len(result):
        # Merge if current item is number and next exists
        if i < len(result) - 1 and re.fullmatch(r'\d+', result[i]):
            merged = result[i] + '. ' + result[i+1]
            final_result.append(merged)
            i += 2  # Skip next item since merged
        else:
            final_result.append(result[i])
            i += 1
            
    return final_result

# def get_regex_dates(chunks, logger):
#     try: 
#         combined_text = []
#         seen_sentences = set()  # Track sentences to avoid repetition
#         chunks = "\n".join([chunk for chunk in chunks])
#         sentences = clean_and_split_sentences(chunks)
#         logger.info(_log_message(f"Length of sentences:{len(sentences)} ", "get_regex_dates", MODULE_NAME))
        
#         regex_pattern = date_pattern
#         # Create a set to track processed indices
#         processed_indices = set()
        
#         for i in range(len(sentences)):
#             # Skip if this index has already been processed
#             if i in processed_indices:
#                 continue
                
#             if re.search(regex_pattern, sentences[i], re.VERBOSE):
#                 # Extract 2 sentences before, the matched sentence, and 2 sentences after
#                 start = max(0, i - 2)
#                 end = min(len(sentences), i + 3)  # Include the matched sentence and 2 after
#                 selected_sentences = sentences[start:end]
                
#                 # Mark these indices as processed
#                 for idx in range(start, end):
#                     processed_indices.add(idx)
                
#                 # Limit the length of sentences before and after to 45 words
#                 trimmed_sentences = []
#                 for j, sent in enumerate(selected_sentences):
#                     if sent not in seen_sentences:  # Avoid duplicate sentences
#                         words = word_tokenize(sent)
#                         if j == 0 and i > 0:  # Sentence before
#                             trimmed_sentences.append(" ".join(words[-45:]))
#                         elif j == len(selected_sentences) - 1 and i + 1 < len(sentences):  # Sentence after
#                             trimmed_sentences.append(" ".join(words[:45]))
#                         else:  # Matched sentence
#                             trimmed_sentences.append(sent)
#                             seen_sentences.add(sent)  # Mark sentence as seen
                
#                 combined_text.extend(trimmed_sentences)


#         # Combine all matched text into a single chunk
#         if combined_text:
#             filtered = [{"chunk_number": "combined", "text": " ".join(combined_text)}]
#             # with open(f'sentences_formed_{len(sentences)}.json','w') as f:
#             #     import json
#             #     json.dump(filtered,f,indent=2)
#         else:
#             logger.debug(_log_message("No dates found in the provided text using regex", "get_regex_dates", MODULE_NAME))
#             return {"Effective Date": "null", "Termination Date": "null", "Renewal Date": "null", "Expiration Date": "null", "Delivery Date": "null", "Term Date": "null"}



@mlflow.trace(name="Get Regex Dates")
def get_regex_dates(chunks, tag_ids, logger):
    try: 
        combined_text = []
        seen_sentences = set()  # Track sentences to avoid repetition
        chunks = "\n".join([chunk for chunk in chunks])
        sentences = clean_and_split_sentences(chunks)
        logger.info(_log_message(f"Length of sentences:{len(sentences)} ", "get_regex_dates", MODULE_NAME))
        
        regex_pattern = date_pattern
        processed_indices = set()  # Track processed indices
        
        for i in range(len(sentences)):
            if i in processed_indices:
                continue
                
            if re.search(regex_pattern, sentences[i], re.VERBOSE):
                start = max(0, i - 2)
                end = min(len(sentences), i + 3)  # Include matched + 2 after
                selected_sentences = sentences[start:end]
                
                for idx in range(start, end):
                    processed_indices.add(idx)
                
                trimmed_sentences = []
                for j, sent in enumerate(selected_sentences):
                    if sent not in seen_sentences:  # Avoid duplicate sentences
                        words = word_tokenize(sent)
                        if j == 0 and i > 0:  # Before sentence
                            trimmed_sentences.append(" ".join(words[-45:]))
                        elif j == len(selected_sentences) - 1 and i + 1 < len(sentences):  # After sentence
                            trimmed_sentences.append(" ".join(words[:45]))
                        else:  # Matched sentence
                            trimmed_sentences.append(sent)
                        seen_sentences.add(sent)  # Track as seen
                
                # Join the block with spaces, then add separator
                block_text = " ".join(trimmed_sentences)
                combined_text.append(block_text)
                combined_text.append("---")  # Separator after each match block

        if combined_text:
            # Remove last trailing separator if present
            if combined_text[-1] == "---":
                combined_text = combined_text[:-1]
            
            filtered = [{
                "chunk_number": "combined",
                "text": "\n\n".join(combined_text)  # double line-break for readability
            }]
        else:
            logger.debug(_log_message("No dates found in the provided text using regex", "get_regex_dates", MODULE_NAME))
            filtered = {
                "Effective Date": "null", 
                "Termination Date": "null", 
                "Renewal Date": "null", 
                "Expiration Date": "null", 
                "Delivery Date": "null", 
                "Term Date": "null"
            }




# def get_regex_dates(chunks, logger):
#     try: 
#         text = "\n".join(chunks)
#         sentences = clean_and_split_sentences(text)
#         logger.info(_log_message(f"Length of sentences:{len(sentences)}", "get_regex_dates", MODULE_NAME))

#         combined_text = []
#         seen_sentences = set()
#         processed_indices = set()
        
#         for i, sentence in enumerate(sentences):
#             if i in processed_indices:
#                 continue

#             if re.search(date_pattern, sentence, re.VERBOSE):
#                 start, end = max(0, i-2), min(len(sentences), i+3)
                
#                 for idx in range(start, end):
#                     if idx not in processed_indices:
#                         sent = sentences[idx].strip()
#                         # Trim only non-matched sentences
#                         if idx != i:
#                             words = word_tokenize(sent)
#                             sent = " ".join(words[:45])
                        
#                         # Deduplicate (case-insensitive)
#                         normalized = sent.lower()
#                         if normalized not in seen_sentences:
#                             combined_text.append(sent)
#                             seen_sentences.add(normalized)
                        
#                         processed_indices.add(idx)

#         if combined_text:
#             return {"matched_text": " ".join(combined_text)}
#         else:
#             logger.debug(_log_message("No dates found in text", "get_regex_dates", MODULE_NAME))
#             return {
#                 "Effective Date": "null",
#                 "Termination Date": "null",
#                 "Renewal Date": "null",
#                 "Expiration Date": "null",
#                 "Delivery Date": "null",
#                 "Term Date": "null"
#             }


        # system_prompt = """
        # You are ChatGPT, a large language model trained by OpenAI.  
        # Knowledge cutoff: 2024-06  

        # ## Personality & Style
        # - Act as a precise, detail-oriented **legal document specialist**.  
        # - Be professional, concise, and conservative: never guess or infer unstated dates.  
        # - Always prioritize explicit legal wording over assumptions.  

        # ## Role & Objective
        # Your role is to analyze legal documents of any type (e.g., contracts, agreements, addendums, amendments, memoranda of understanding, leases, treaties, policies, or related instruments) to identify **multiple critical dates**:  
        # - Effective Date  
        # - Expiration Date  
        # - Renewal Date  
        # - Delivery Date  

        # Each date must be extracted only when explicitly tied to legal language. If not explicitly present, incomplete, conditional, or placeholder → return `null`.

        # ---

        # ## Scope & Terminology

        # ### 1. Effective Date
        # - **Definition**: Date the agreement begins, becomes active, or comes into force.  
        # - **Keywords**: “effective date”, “effective as of”, “commencement date”, “start date”, “executed on”, “entered into on”, “date first written above”.  
        # - **Rules**: Prefer explicit effective/commencement wording. Execution/signature only if explicitly tied to effectiveness.

        # ### 2. Expiration Date
        # - **Definition**: The natural end of the contract term (not early termination).  
        # - **Keywords**: “expiration date”, “expires on”, “end of term”, “valid until”, “shall continue until”.  
        # - **Rules**: only extract if clearly stated.

        # ### 3. Renewal Date
        # - **Definition**: Date the contract automatically renews or extends.  
        # - **Keywords**: “renewal date”, “automatically renew”, “renewal term commences”, “option to renew on”, “extension begins”.  
        # - **Rules**: Only extract if clearly linked to renewal or extension.

        # ### 4. Delivery Date
        # - **Definition**: Date goods/services must be delivered, performed, or completed.  
        # - **Keywords**: “delivery date”, “shall be delivered on”, “completion date”, “scheduled delivery”, “due date”, “performance completed by”.  
        # - **Rules**: Typically found in performance/deliverables section.

        # ---

        # ## General Rules for All Dates
        # 1. **Formatting**: Always return dates as ISO `YYYY-MM-DD`.  
        # 2. **Incomplete or Placeholder Dates**: If only year/month or placeholders → return `"null"`.  
        # 3. **Reference-Based Dates**: Compute relative dates (e.g., “15 years after Effective Date”) only if both reference and base date are explicitly present. Otherwise → `"null"`.  
        # 4. **Context Priority**: Prefer amended/superseding dates if multiple candidates exist.  
        # 5. **No Assumptions**: Do not infer from context, metadata, or today’s date.  
        # 6. **Regex Dates**: Use regex-extracted dates only if the surrounding clause explicitly ties them to the relevant date type.  

        # ---

        # ## Output Formatting
        # - Return **strict JSON** with all five fields:  
        # ```json
        # {
        #     "Effective Date": "<YYYY-MM-DD or null>",
        #     "Expiration Date": "<YYYY-MM-DD or null>",
        #     "Renewal Date": "<YYYY-MM-DD or null>",
        #     "Delivery Date": "<YYYY-MM-DD or null>",
        # }
        # """
        system_prompt = """
        You are ChatGPT, a large language model trained by OpenAI.  
        Knowledge cutoff: 2024-06  

        ## Personality & Style
        - Act as a precise, detail-oriented **legal document specialist**.  
        - Be professional, concise, and conservative: never guess or infer unstated dates.  
        - Always prioritize explicit contractual language and legal context over assumptions.  

        ## Role & Objective
        Your role is to analyze **any type of legal document** (contracts, agreements, addendums, amendments, memoranda of understanding, leases, treaties, policies, promissory notes, debt instruments, real estate documents, etc.) to extract **critical dates** in the following strict sequence:  
        1. Effective Date  
        2. Expiration Date  
        3. Renewal Date (only if Expiration is valid and in the future)  
        4. Delivery Date    

        If a date is not explicit, incomplete, conditional, or only a placeholder → return `"null"`.  

        ---

        ## Date Categories & Rules

        ### 1. Effective Date
        - **Definition**: Date the agreement begins, becomes active, or comes into force.  
        **Step 1: Explicit term start date (highest precedence)**
        - Look for explicit dates defining when the agreement/contract/document term starts:
            Keywords: “Effective date”, “commencement date”, “starting on [date]”, “shall run from [date] to [date]”, “lease term will be from [date]”.
        - If found, this overrides any signature/execution clause, even if the contract also states “effective since being signed/sealed.”
        - Example:
            Clause 1: “The lease term will be from 1 January 2006 to 31 December 2008.”
            Clause 2: “This contract is effective since being signed.”
            Effective Date → 2006-01-01 (lease start date, not signature date).

        **Step 2: Signature/execution clause (FALLBACK)**
        - Only if no explicit term-start date is found, check whether effectiveness tied to signature/execution:
            look for "effective upon signature", "effective when signed", "executed on", "executed as of", "signed on", or signature block lines such as "Signature Date:".

        **Step 3: No dates found**
        - If neither an explicit term-start nor a signature/execution-linked date can be found, return `"null"`.

        ### 2. Expiration Date
        - **Definition**: Date the agreement naturally ends or fully terminates.  
        - **Step 1: Explicit check** → Use direct clauses: “expiration date,” “valid until,” “shall continue until.”, "will conclude on"
        - **Step 2: Duration-based expiration**:
            - If the agreement defines expiration as a fixed duration from the Effective Date
            1. When duration is expressed in days:
            - Determine inclusivity based on the wording:
                **Exclusive wording** (e.g., “expires 365 days after the Effective Date”, “shall terminate one year after commencement”):
                    Exclude the start date.
                    Formula: Expiration = Effective Date + Duration.
                **Inclusive wording** (e.g., “expires within 365 days of the Effective Date”, “shall continue for 365 days beginning on the Effective Date”):
                    Include the start date.
                    Formula: Expiration = Effective Date + (Duration − 1).
            2. When duration is expressed in months or years:
            - Do not apply inclusive/exclusive adjustment.
            - Always use direct calendar addition.
            - Formula: Expiration = Effective Date + Duration (in months/years).
        - **Step 3: Duration plus maximum end date (cap)** → 
            - If the clause says the agreement ends after a duration but not later than a stated date:
            - The duration-based date is the actual expiration date.
            - The “no later than” date is not the expiration date. It only applies if the duration-based date would extend beyond it.
            - In other words:
                If (Effective Date + duration) ≤ cap → use the duration-based date.
                If (Effective Date + duration) > cap → use the cap date.
            ** Rule of precedence ** : The duration-based date governs unless it exceeds the cap.
            
        - **Step 4: Context rule** → If none found, check → Apply contract-type defaults:
        - **Sale & Purchase / Real Estate / M&A** → use **Closing Date** (final transfer/consummation).  
            - Never use loan repayment schedules, promissory note maturity dates, or financing deadlines as the Expiration Date.  
            - If only a maturity/repayment date is provided and no closing date → return `"null"`.  

        - **Debt / Loan / Promissory Note** → use **Maturity Date** (final repayment due).  

        - **Lease / Rental** → use lease end date.  

        - **Employment / Service Agreements** → use fixed termination date if defined.  

        - **Exemption**: Inspection/review/option expiry periods are milestones, not Expiration Dates, unless explicitly stated as contract end.  

        ---

        ### 3. Renewal Date (Dependent on Expiration Date)
        - **Definition**: Date the contract automatically renews or can be extended.  
        - **Dependency Rule**:
        - If Expiration Date = `"null"` → Renewal Date = `"null"`.  
        - If Expiration Date < today’s date → Renewal Date = `"null"` (expired contracts cannot renew).  
        - If Expiration Date ≥ today’s date → extract Renewal Date:  
            - **Explicit check**: “automatically renews,” “renewal term begins,” “option to renew,” “extension commences.”  
            - Otherwise → `"null"`.  

        ---

        ### 4. Delivery Date
        - **Definition**: Date goods, services, property, or obligations must be delivered, performed, or completed.  
        - **Rules**:
        - Explicit indicators: “delivery date,” “completion date,” “due date,” “shall be delivered on.”  
        - Contract-type defaults (fallback only):  
            - Sales/Procurement/Supply/Service/Lease → closing date, handover date, possession date.  
            - Construction/Project → completion date, substantial completion date, project handover.  
            - Licensing/Software → implementation completion date, go-live date, delivery of licensed materials.  
            - Grants/Donations/Research → only if explicit deliverables specified.  
        - If no explicit or contract-type rule applies → `"null"`.  
        
        ---

        ## General Rules for All Dates
        1. **Order of evaluation**: Always check for **explicitly stated dates first**. Only if none are found, apply context-based rules or exemptions.  
        2. **Strict sequence**: Extract Effective → Expiration → Renewal → Delivery.  
        3. **Format**: Always return ISO `YYYY-MM-DD`.  
        4. **Incomplete Dates**: If only year/month or placeholder → `"null"`.  
        5. **Relative Dates**: Compute only if both base date & offset are explicit.  
        6. **Amendments**: If amended/superseding dates exist, prefer those.  
        7. **No Assumptions**: Never infer from metadata, today’s date, or unstated terms.  
        8. **Consistency**: Distinguish contract Expiration from milestone dates (e.g., inspection periods).  

        ---

        ## Output Formatting
        Return **strict JSON only**, no extra commentary:  

        {
            "Effective Date": "<YYYY-MM-DD or null>",
            "Expiration Date": "<YYYY-MM-DD or null>",
            "Renewal Date": "<YYYY-MM-DD or null>",
            "Delivery Date": "<YYYY-MM-DD or null>",
            "Reasoning": {
                "Effective Date": "<reason>",
                "Expiration Date": "<reason>",
                "Renewal Date": "<reason>",
                "Delivery Date": "<reason>"
            }
        }
        """     


        user_prompt = f"""
        ### Legal Document Context
        ---
        {filtered[0]['text']}
        ---

        ### Task
        Extract the following dates from the document in `YYYY-MM-DD` format, following this strict sequence:  
        1. **Effective Date** → first determine when the contract starts.  
        2. **Expiration Date** → then identify when the contract ends.  
        3. **Renewal Date** → extract only if the Expiration Date is not `"null"` **and** is in the future (≥ today’s date).  
        - If Expiration Date < today → Renewal Date = `"null"`.  
        4. **Delivery Date** → extract independently of the above dates, only based on explicit contract language or contract-type defaults.

        ---

        ### Instructions
        1. Use only explicit language tied to each date type.  
        2. Return `"null"` if the date is missing, incomplete, conditional, or only a placeholder.  
        3. Compute relative dates only if both base date and period are explicitly present. Otherwise → `"null"`.  
        4. Renewal Date must depend on Expiration Date as described above.  
        5. Validate regex-extracted dates only if explicitly supported by surrounding legal wording.  
        6. Provide concise reasoning for each extracted date, citing the exact clause or phrase.  

        ### Output
        Return JSON only, no extra words or formatting.  
        Schema:  
        {{
            "Effective Date": <YYYY-MM-DD or null>,
            "Expiration Date": <YYYY-MM-DD or null>,
            "Renewal Date": <YYYY-MM-DD or null>,
            "Delivery Date": <YYYY-MM-DD or null>,
            "Reasoning": {{
                "Effective Date": "<reason>",
                "Expiration Date": "<reason>",
                "Renewal Date": "<reason>",
                "Delivery Date": "<reason>",
            }}
    
        }}
        """

        json_schema = {"type": "json_object"}
        llm_response = dynamic_llm_call(user_prompt, system_prompt, json_schema, logger)
        logger.info(_log_message(f"LLM Response: {llm_response}", "get_regex_dates", MODULE_NAME))      
        return llm_response 
    except Exception as e:
        logger.error(_log_message(f"Error in get_regex_dates: {str(e)}", "get_regex_dates", MODULE_NAME))
        return {"Effective Date": "null", "Termination Date": "null", "Renewal Date": "null", "Expiration Date": "null", "Delivery Date": "null", "Term Date": "null"}