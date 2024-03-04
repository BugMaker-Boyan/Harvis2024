import openai
import os
from prompt import PROMPT_TEMPLATE, EXAMPLE_TEMPLATE
import json
import re
from util import SimilarityUtil
import argparse
import random
import retry


COST_PER_THOUSAND = {
    "gpt-4": [0.03, 0.06],
    "gpt-3.5-turbo": [0.0010, 0.0020]
}


def setup_openai(openai_api_key=None, openai_base_url=None):
    openai.api_key = openai_api_key if openai_api_key else os.getenv("OPENAI_API_KEY")
    openai.base_url = openai_base_url if openai_base_url else os.getenv("OPENAI_BASE_URL")


def format_prompt(seed_examples, return_messages=False):
    example_list = []
    valid_operations = set([example["output"]["operation"] for example in seed_examples])
    for example in seed_examples:
        format_example = EXAMPLE_TEMPLATE.format(
            INPUT=example["input"].strip(),
            OUTPUT=json.dumps(example["output"]).replace("\n", "").strip()
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

def ask_chat(client, model, messages: list, temperature=0.7, max_tokens=512):
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


def extract_examples(response):
    pattern = r"\d+\.\s*INPUT:\s*(.*?)\nOUTPUT:\s*(\{.*?\})(?=\n*)"
    response = "1. INPUT: " + response
    all_pairs = re.findall(pattern, response)
    examples = []
    for pair_input, pair_output in all_pairs:
        try:
            output = json.loads(pair_output)
        except:
            continue
        examples.append({
            "input": pair_input,
            "output": output
        })
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
        default=2000,
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
        default="./generate_dataset.json",
        type=str
    )
    parser.add_argument(
        "--seed_dataset_path",
        default="./seed_dataset.json",
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
        
        total_cost += gpt_res["prompt_tokens"] * COST_PER_THOUSAND[args.model][0] + gpt_res["completion_tokens"] * COST_PER_THOUSAND[args.model][1]
        
        new_examples = extract_examples(gpt_res["response"])
        if new_examples:
            filter_examples = similarity_util.filter_based_on_similarity_threshold(
                old_examples=seed_examples,
                new_examples=new_examples,
                threshold=args.threshold
            )
            all_seed_examples.extend(filter_examples)
        with open(args.generate_dataset_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(all_seed_examples, indent=4))
        
        print(f"{len(all_seed_examples)} / {args.generate_size}, select {len(filter_examples)} from {len(new_examples)} examples, GPT cost: {total_cost}")

if __name__ == "__main__":
    args = parse_args()
    main(args)
