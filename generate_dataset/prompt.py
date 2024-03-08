EXAMPLE_TEMPLATE = """
INPUT: {INPUT}
GROUP: {GROUP}
OUTPUT: {OUTPUT}
"""


PROMPT_TEMPLATE = """Generate 5 new examples following specific rules and the OUTPUT must be in valid JSON format that can be parsed.

The keys in OUTPUT:
"operations": The operation to do in INPUT, must be one of ({VALID_OPERATIONS}). It is a string.                                        
"file": The concrete file name mentioned in INPUT, or null if it is not. It is a string or null.
"pointer": A list of integers to data row index (starting from 1 from begin, or starting from -1 from last) to operate, which is mentioned in INPUT. Note that each index should be included in list, do not use slice. Negative integers indicate counting from the last. null is set when not needed or by default meaning all data rows. It is a list of integers or null.
"group": The concrete group name when needed, which must be one of GROUP list. It is null if not needed even if GROUP list is not empty. It is a string or null.

Examples:
{EXAMPLES}

New Examples:
1. INPUT: """