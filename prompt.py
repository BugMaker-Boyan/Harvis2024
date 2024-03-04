EXAMPLE_TEMPLATE = """
INPUT: {INPUT}
OUTPUT: {OUTPUT}
"""


PROMPT_TEMPLATE = """

Generate 5 new examples following specific rules and the output must be in JSON format. Try to make the INPUT in new examples more varied and colorful. 

"operations": "The operation to do, must be one of ({VALID_OPERATIONS})"
                                                   
"file": "The concrete file name when mentioned, or 'null' if it is not needed."

"pointer": "A list of integers to data row or column index (starting from 0)  to operate. Note that each index should be included in list, do not use slice. Negative integers indicate counting from the last. 'null' is set when not needed or by default meaning all data."
    
"group": "The concrete group name when needed, or 'null' if not mentioned."

{EXAMPLES}

New Examples:
1. INPUT: 
"""