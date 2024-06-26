GENERATE_PROMPT_TEMPLATE = \
"""
### Task Desscription:
Based on the Format Instructions and in-context examples, generate 5 new examples. The example is a JSON string encoding relations among Question, Data and Chart. Ensure that Data-Info in new examples are in diverse domains.

### Format Instructions:
{{
    "Question": <The question that includes the user's intent>,
    "Data-Info": {{
        "Data-Fields": <The list of all source data field names>
    }},
    "Chart-Info": {{
        "Chart-Type": <One specific type in ["Bar", "Line", "Pie", "Scatter", "Area"]>,
        "X-Field": <One field of chart x-axis>,
        "Y-Field": <One field or the list of multiple fields of chart y-axis>
    }},
    "Overlay": {{
        "Operation": <One specific operation in [{CONTEXT_OPERATIONS}]>,
        "X-Value": <Options include [null, "all", Specific Value, Specific Value-Range List With Start-Value and End-Value]>,
        "Y-Value": <Options include [null, "all", Specific Data Field, Specific Data Fields List]>
    }}
}}

### Operation Descriptions:
{CONTEXT_OPERATIONS_DESC}

### In-context Examples:
{EXAMPLES}

### Generate Examples:
1. """

INSTRUCTION_PROMPT_TEMPLATE = \
"""
### Task:
Based on the input Question, Data-Info and Chart-Info details, output the encoded mapping in JSON format.
### Output Format:
{{
    "Operation": <One specific operation in ["Reference", "Highlight", "Trendline", "Statistic-min", "Statistic-max", "Statistic-mean", "Label", "Extension", "Creation"]>,
    "X-Value": <Options include [null, "all", Specific Value, Specific Value-Range List With Start-Value and End-Value]>,
    "Y-Value": <Options include [null, "all", Specific Data Field, Specific Data Fields List]>
}}
"""

INPUT_PROMPT_TEMPLATE = \
"""
### INPUT:
{{
    "Question": {QUESTION_SLOT},
    "Data-Info": {{
        "Data-Fields": {DATA_FIELDS_SLOT}
    }},
    "Chart-Info": {{
        "Chart-Type": {CHART_TYPE_SLOT},
        "X-Field": {X_FIELD_SLOT},
        "Y-Field": {Y_FIELD_SLOT}
    }}
}}
### OUTPUT:
"""

OUTPUT_PROMPT_TEMPLATE = \
"""
{{
    "Operation": {OPERATION_SLOT},
    "X-Value": {X_VALUE_SLOT},
    "Y-Value": {Y_VALUE_SLOT}
}}
"""
