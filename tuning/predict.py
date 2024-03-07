import json
from finetune import load_datasets
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def compute_accuracy(predictions, references):
    assert len(predictions) == len(references)
    count = 0
    for pred, ref in zip(predictions, references):
        try:
            pred = json.loads(pred)
            ref = json.loads(ref)
            if pred == ref:
                count += 1
        except:
            continue
    return count / len(predictions)


def predict(checkpoint_path, save_path):
    tokenizer = AutoTokenizer.from_pretrained("/home/liboyan/docker_workspace/hf_models/llama-2-7b")
    model = AutoModelForCausalLM.from_pretrained(checkpoint_path, torch_dtype=torch.float16, device_map="auto")
    print(tokenizer.special_tokens_map)
    tokenizer.padding_side = "right"
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenized_ds = load_datasets("havis", tokenizer)
    predictions = []
    for sample in tqdm(tokenized_ds["test"]):
        input_len = 0
        while sample["labels"][input_len] == -100:
            input_len += 1
        input_ids = sample["input_ids"][:input_len]
        attention_mask = sample["attention_mask"][:input_len]
        output_ids = model.generate(
            input_ids=torch.tensor(input_ids, device=model.device).unsqueeze(0),
            attention_mask=torch.tensor(attention_mask, device=model.device).unsqueeze(0),
            do_sample=False,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            max_new_tokens=256
        )[0]
        output_ids = output_ids[input_len:]
        generate_text = tokenizer.decode(output_ids, skip_special_tokens=True)
        output = generate_text.split("RESPONSE:", maxsplit=1)[-1].split("INSTRUCTION:", maxsplit=1)[0].replace("\n", "").strip()
        predictions.append(output)
        print(f"{output}")
        with open(save_path, "w", encoding="utf-8") as f:
            for pred in predictions:
                f.write(pred)
                f.write("\n")


if __name__ == "__main__":
    MODEL_CHECKPOINT = "/home/liboyan/Harvis2024/checkpoint/llama2-7b_havis/checkpoint-759"
    SAVE_PATH = "./predictions.txt"
    predict(MODEL_CHECKPOINT, SAVE_PATH)
