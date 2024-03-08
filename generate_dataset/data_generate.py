import openai
import os
from prompt import PROMPT_TEMPLATE, EXAMPLE_TEMPLATE
import json
import re
from util import SimilarityUtil
import argparse
import random
from retrying import retry


COST_PER_THOUSAND = {
    "gpt-4": [0.03, 0.06],
    "gpt-3.5-turbo": [0.0010, 0.0020]
}


def setup_openai(openai_api_key=None, openai_base_url=None):
    if openai_api_key:
        os.environ["OPENAI_API_KEY"] = openai_api_key
    if openai_base_url:
        os.environ["OPENAI_BASE_URL"] = openai_base_url


def format_prompt(seed_examples, return_messages=False):
    example_list = []
    valid_operations = set([json.loads(example["output"])["operation"] for example in seed_examples])
    for example in seed_examples:
        format_example = EXAMPLE_TEMPLATE.format(
            INPUT=example["input"].strip(),
            GROUP=example["group"].strip(),
            OUTPUT=example["output"].strip()
        ).strip()
        example_list.append(format_example)
    
    prompt = PROMPT_TEMPLATE.format(
        VALID_OPERATIONS=", ".join(valid_operations),
        EXAMPLES="\n\n".join(example_list)
    )
    
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
def ask_chat(client, model, messages: list, temperature=1.0, max_tokens=512):
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


def is_output_valid(output_json):        
    
    def is_operation_valid(operation):
        return isinstance(operation, str) and operation in ("create", "encodings", "extend", "highlight", "trendline", "reference", "max", "mean", "min")
    
    def is_file_valid(file):
        return file is None or isinstance(file, str)

    def is_pointer_valid(pointer):
        if pointer is None:
            return True
        if not isinstance(pointer, list):
            return False
        all_integers_flag = True
        for idx in pointer:
            if not isinstance(idx, int):
                all_integers_flag = False
                break
        return all_integers_flag
    
    def is_group_valid(group):
        return group is None or isinstance(group, str)

    if set(output_json.keys()) != set(["operation", "file", "pointer", "group"]):
        return False
    
    return is_operation_valid(output_json["operation"]) and is_file_valid(output_json["file"]) and is_pointer_valid(output_json["pointer"]) and is_group_valid(output_json["group"])


def is_group_valid(group_json, output_json):
    all_strings_flag = True
    for g in group_json:
        if not isinstance(g, str):
            all_strings_flag = False
            break
    if not all_strings_flag:
        return False
    else:
        if output_json["group"] is None:
            return True
        else:
            return output_json["group"] in group_json

def extract_examples(response):
    pattern = r"\d+\.\s*INPUT:\s*(.*?)\nGROUP:\s*(\[.*?\])\nOUTPUT:\s*(\{.*?\})(?=\n*)"
    response = "1. INPUT: " + response
    all_matches = re.findall(pattern, response)
    examples = []
    for match_input, match_group, match_output in all_matches:
        try:
            output_json = json.loads(match_output)
            group_json = json.loads(match_group)
            sample = {
                "input": match_input,
                "group": json.dumps(group_json).replace("\n", "").strip(),
                "output": json.dumps(output_json).replace("\n", "").strip()
            }
            if not is_output_valid(output_json):
                continue
            if not is_group_valid(group_json, output_json):
                continue
        except Exception as e:
            continue
        examples.append(sample)
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
        default=3,
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
        default="./data/seed_data.json",
        type=str
    )
    parser.add_argument(
        "--openai_api_key",
        default=None,
        type=str
    )
    parser.add_argument(
        "--openai_base_url",
        default=None,
        type=str
    )
    args = parser.parse_args()
    return args


def main(args):
    with open(args.seed_dataset_path, "r", encoding="utf-8") as f:
        all_seed_examples = json.load(f)
    
    random.shuffle(all_seed_examples)
    
    generate_dataset = []
    
    setup_openai(args.openai_api_key, args.openai_base_url)
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
