EXAMPLE_TEMPLATE = """
INPUT: {INPUT}
OUTPUT: {OUTPUT}
"""


PROMPT_TEMPLATE = """

Generate 10 new examples following specific rules and the OUTPUT must be in valid JSON format that can be parsed. Try to make the INPUT in new examples more varied and colorful.

"operations": The operation to do, must be one of ({VALID_OPERATIONS}). It is a string.
                                                   
"file": The concrete file name when mentioned, or null if it is not needed. It is a string or null.

"pointer": A list of integers to data row index (starting from 1 from begin, or starting from -1 from last)  to operate. Note that each index should be included in list, do not use slice. Negative integers indicate counting from the last. null is set when not needed or by default meaning all data. It is a list of integers or null.
    
"group": The concrete group name when needed, or null if not mentioned. It is a string or null.

Note that, the size of pointer list (if exists) in new examples should be no more than 15.

{EXAMPLES}

New Examples:
1. INPUT: 
"""