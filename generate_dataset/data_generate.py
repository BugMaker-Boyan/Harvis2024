import openai
from prompt import *
import json
import re
from util import SimilarityUtil
import argparse
import random
from retrying import retry
import os


COST_PER_THOUSAND = {
    "gpt-4": [0.03, 0.06],
    "gpt-3.5-turbo": [0.0010, 0.0020]
}

OPERATIONS_DESC = {
    "Reference": "Add a group of reference line for chart",
    "Highlight": "Highlight user nominated parts of chart",
    "Trendline": "Show the accurate trendline of specific data groups",
    "Statistic-min": "Statistic minumum (min) value of specific data groups",
    "Statistic-max": "Statistic maximum (max) value of specific data groups",
    "Statistic-mean": "Statistic average (mean) value of specific data groups",
    "Label": "Add data labels of specific data groups",
    "Extension": "Extend new data records based on x-field into chart",
    "Creation": "Create new data group based on y-field into chart"
}


def format_prompt(seed_examples, return_messages=False):
    context_operations = set([example["Overlay"]["Operation"] for example in seed_examples])
    examples_str = ""
    for idx, example in enumerate(seed_examples):
        _example_repr = json.dumps(example).replace('\n', '')
        examples_str += f"{idx+1}. {_example_repr}\n"
        
    operations_desc_str = ""
    for idx, operation in enumerate(context_operations):
        operations_desc_str += f"{idx+1}. {operation}: {OPERATIONS_DESC[operation]}\n"
    
    prompt = GENERATE_PROMPT_TEMPLATE.format(
        CONTEXT_OPERATIONS=", ".join(context_operations).strip(),
        CONTEXT_OPERATIONS_DESC=operations_desc_str.strip(),
        EXAMPLES=examples_str.strip()
    ).strip()
    
    if return_messages:
        return [{"role": "user", "content": prompt}]
    else:
        return prompt
    
    
def _ask_chat_retry_condition(exception):
    return isinstance(
        exception,
        (Exception,)
    )


@retry(
    wait_fixed=2000,
    retry_on_exception=_ask_chat_retry_condition
)
def ask_chat(client, model, messages: list, temperature=1.0, max_tokens=1024):
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    response_clean = response.choices[0].message.content
    return dict(
        response=response_clean,
        completion_tokens=response.usage.completion_tokens,
        prompt_tokens=response.usage.prompt_tokens,
        total_tokens=response.usage.total_tokens
    )


def is_valid(example_json):
    if set(example_json.keys()) != set(["Question", "Data-Info", "Chart-Info", "Overlay"]):
        return False
    if set(example_json["Data-Info"].keys()) != set(["Data-Fields"]):
        return False
    if not isinstance(example_json["Data-Info"]["Data-Fields"], list):
        return False
    if set(example_json["Chart-Info"].keys()) != set(["Chart-Type", "X-Field", "Y-Field"]):
        return False
    if not isinstance(example_json["Chart-Info"]["Chart-Type"], str):
        return False
    if example_json["Chart-Info"]["Chart-Type"] not in ["Bar", "Line", "Pie", "Scatter", "Area"]:
        return False
    if example_json["Chart-Info"]["X-Field"] not in example_json["Data-Info"]["Data-Fields"]:
        return False
    if not isinstance(example_json["Chart-Info"]["Y-Field"], str) and not isinstance(example_json["Chart-Info"]["Y-Field"], list):
        return False
    if isinstance(example_json["Chart-Info"]["Y-Field"], str) and example_json["Chart-Info"]["Y-Field"] not in example_json["Data-Info"]["Data-Fields"]:
        return False
    if isinstance(example_json["Chart-Info"]["Y-Field"], list) and not set(example_json["Chart-Info"]["Y-Field"]).issubset(set(example_json["Data-Info"]["Data-Fields"])):
        return False
    if set(example_json["Overlay"].keys()) != set(["Operation", "X-Value", "Y-Value"]):
        return False
    if not isinstance(example_json["Overlay"]["Operation"], str):
        return False
    if example_json["Overlay"]["Operation"] not in ["Reference", "Highlight", "Trendline", "Statistic-min", "Statistic-max", "Statistic-mean", "Label", "Extension", "Creation"]:
        return False
    
    # It's hard to check "X-Value", skip it
    
    if example_json["Overlay"]["Y-Value"] is not None:
        if isinstance(example_json["Overlay"]["Y-Value"], str):
            if example_json["Overlay"]["Y-Value"] != "all" and example_json["Overlay"]["Y-Value"] not in example_json["Data-Info"]["Data-Fields"]:
                return False
            return True
        if isinstance(example_json["Overlay"]["Y-Value"], list):
            if not set(example_json["Overlay"]["Y-Value"]).issubset(set(example_json["Data-Info"]["Data-Fields"])):
                return False
            return True
        return False

    return True


def extract_examples(response):
    pattern = r"\d+\.\s*(\{.*?\})(?=\n*\d+\.|$)"
    response = "1. " + response
    all_matches = re.findall(pattern, response)
    examples = []
    for match in all_matches:
        try:
            example_json = json.loads(match)
            if not is_valid(example_json):
                continue
        except Exception as e:
            continue
        examples.append(example_json)
    return examples


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="gpt-3.5-turbo",
        type=str,
        choices=["gpt-3.5-turbo", "gpt-4"]
    )
    parser.add_argument(
        "--generate_size",
        default=10000,
        type=int
    )
    parser.add_argument(
        "--seed_example_num",
        default=5,
        type=int
    )
    parser.add_argument(
        "--threshold",
        default=0.7,
        type=float
    )
    parser.add_argument(
        "--generate_dataset_path",
        default="./data/generate_dataset.json",
        type=str
    )
    parser.add_argument(
        "--seed_dataset_path",
        default="./data/seed_dataset.json",
        type=str
    )
    args = parser.parse_args()
    return args


def main(args):
    with open(args.seed_dataset_path, "r", encoding="utf-8") as f:
        all_seed_examples = json.load(f)
    
    random.shuffle(all_seed_examples)
    
    generate_dataset = []
    
    if os.path.exists(args.generate_dataset_path):
        generate_dataset = json.load(open(args.generate_dataset_path, "r", encoding="utf-8"))
    
    client = openai.OpenAI()
    
    similarity_util = SimilarityUtil()
    
    total_cost = 0
    
    while len(generate_dataset) < args.generate_size:
        seed_examples = random.choices(all_seed_examples, k=args.seed_example_num)
        gpt_res = ask_chat(
            client,
            args.model,
            format_prompt(seed_examples, return_messages=True),
        )
        total_cost += gpt_res["prompt_tokens"] / 1000 * COST_PER_THOUSAND[args.model][0] + gpt_res["completion_tokens"] / 1000 * COST_PER_THOUSAND[args.model][1]
        
        new_examples = extract_examples(gpt_res["response"])
        if new_examples:
            filter_examples = similarity_util.filter_based_on_similarity_threshold(
                old_examples=seed_examples,
                new_examples=new_examples,
                threshold=args.threshold
            )
            generate_dataset.extend(filter_examples)
            all_seed_examples.extend(filter_examples)
        else:
            filter_examples = []
        with open(args.generate_dataset_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(generate_dataset, indent=4))
        
        print(f"{len(generate_dataset)} / {args.generate_size}, select {len(filter_examples)} from {len(new_examples)} examples, GPT cost: {total_cost}")

if __name__ == "__main__":
    args = parse_args()
    main(args)
