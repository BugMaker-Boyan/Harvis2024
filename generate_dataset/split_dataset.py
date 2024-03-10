import json
import random


if __name__ == "__main__":
    DATASET_JSON_PATH = "data/generate_dataset.json"
    
    TRAIN_DATASET_JSON_PATH = "data/train_dataset.json"
    DEV_DATASET_JSON_PATH = "data/dev_dataset.json"
    
    DEV_DATASET_SAMPLES = 1000
    
    with open(DATASET_JSON_PATH, "r", encoding="utf-8") as f:
        dataset_json = json.load(f)
    
    random.seed(2024)
    random.shuffle(dataset_json)
    
    dev_dataset = dataset_json[:DEV_DATASET_SAMPLES]
    train_dataset = dataset_json[DEV_DATASET_SAMPLES:]
        
    with open(TRAIN_DATASET_JSON_PATH, "w", encoding="utf-8") as f:
        f.write(json.dumps(train_dataset, indent=4))

    with open(DEV_DATASET_JSON_PATH, "w", encoding="utf-8") as f:
        f.write(json.dumps(dev_dataset, indent=4))

    print(f"Train Dataset Size: {len(train_dataset)}, Dev Dataset Size: {len(dev_dataset)}")