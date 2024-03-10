import datasets
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForSeq2Seq, TrainingArguments, Trainer
import torch
import argparse

DATASETS_DATA_MAPPING = {
    "havis": {
        "train": "data/train_dataset.json",
        "validation": "data/dev_dataset.json",
        "test": "data/seed_dataset.json"
    }
}

MAX_LENGTH = 2048


def load_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_REPO_MAPPING[model_name])
    model = AutoModelForCausalLM.from_pretrained(MODEL_REPO_MAPPING[model_name], torch_dtype=torch.float16, device_map="auto")
    print(tokenizer.special_tokens_map)
    tokenizer.padding_side = "right"
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer, model


def process_func(example, tokenizer):
    input_ids, attention_mask, labels = [], [], []
    INPUT_TEMPLATE = "INSTRUCTION: {}\n"
    OUTPUT_TEMPLATE = "RESPONSE: {}"
    instruction = tokenizer(INPUT_TEMPLATE.format(example["input"].strip()), add_special_tokens=False)
    example_output = str(example["output"]).replace("\n", "").strip()
    response = tokenizer(OUTPUT_TEMPLATE.format(example_output), add_special_tokens=False)
    input_ids = instruction["input_ids"] + response["input_ids"] + [tokenizer.eos_token_id]
    attention_mask = instruction["attention_mask"] + response["attention_mask"] + [1]
    labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [tokenizer.eos_token_id]
    if len(input_ids) > MAX_LENGTH:
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }


def load_datasets(datasets_name, tokenizer):
    ds = datasets.load_dataset('json', data_files=DATASETS_DATA_MAPPING[datasets_name])
    tokenized_ds = ds.map(lambda x: process_func(x, tokenizer), remove_columns=ds["train"].column_names)
    return tokenized_ds


def train(model_name, datasets_name):
    tokenizer, model = load_model(model_name)
    tokenized_ds = load_datasets(datasets_name, tokenizer)
    args = TrainingArguments(
        output_dir=f"./checkpoint/{model_name}_{datasets_name}",
        per_device_train_batch_size=16,
        gradient_accumulation_steps=2,
        logging_steps=1,
        num_train_epochs=3,
        max_grad_norm=1.0,
        learning_rate=1e-5,
        save_strategy="epoch",
        save_total_limit=1,
        metric_for_best_model="eval_loss",
        bf16=True,
        fp16=False,
        lr_scheduler_type="cosine",
        adam_epsilon=1e-5,
        seed=42,
        report_to=["tensorboard"]
    )
    trainer = Trainer(
        args=args,
        model=model,
        train_dataset=tokenized_ds["train"],
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True)
    )
    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Havis-Llama script')

    parser.add_argument('--model_name', type=str, choices=["llama2-7b"])
    parser.add_argument('--datasets_name', type=str, choices=["havis"])

    args = parser.parse_args()

    train(args.model_name, args.datasets_name)
