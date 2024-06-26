import json
import argparse
from prompt import *


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        type=str
    )
    parser.add_argument(
        "--save_path",
        type=str
    )
    args = parser.parse_args()
    data = json.load(open(args.data_path, "r", encoding="utf-8"))
    
    alpaca_dataset = []
    
    for item in data:
        instruction = INSTRUCTION_PROMPT_TEMPLATE.strip()
        input = INPUT_PROMPT_TEMPLATE.format(
            QUESTION_SLOT=item["Question"],
            DATA_FIELDS_SLOT=item["Data-Fields"],
            CHART_TYPE_SLOT=item["Chart-Info"]["Chart-Type"],
            X_FIELD_SLOT=item["Chart-Info"]["X-Field"],
            Y_FIELD_SLOT=item["Chart-Info"]["Y-Field"]
        )
        output = OUTPUT_PROMPT_TEMPLATE.format(
            OPERATION_SLOT=item["Overlay"]["Operation"],
            X_VALUE_SLOT=item["Overlay"]["X-Value"],
            Y_VALUE_SLOT=item["Overlay"]["Y-Value"]
        )
        
        alpaca_dataset.append({
            "instruction": instruction,
            "input": input,
            "output": output
        })

    json.dump(alpaca_dataset, open(args.save_path, "w", encoding="utf-8"), indent=4)
