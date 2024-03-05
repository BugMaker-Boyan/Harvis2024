import json
import random


if __name__ == "__main__":
    DATASET_JSON_PATH = "data/clean_generate_dataset.json"
    
    TRAIN_DATASET_JSON_PATH = "data/train_dataset.json"
    TEST_DATASET_JSON_PATH = "data/test_dataset.json"
    
    TEST_DATASET_SAMPLES = 1000
    
    with open(DATASET_JSON_PATH, "r", encoding="utf-8") as f:
        dataset_json = json.load(f)
    
    random.seed(2024)
    random.shuffle(dataset_json)
    
    test_dataset = dataset_json[:TEST_DATASET_SAMPLES]
    train_dataset = dataset_json[TEST_DATASET_SAMPLES:]
        
    with open(TRAIN_DATASET_JSON_PATH, "w", encoding="utf-8") as f:
        f.write(json.dumps(train_dataset, indent=4))

    with open(TEST_DATASET_JSON_PATH, "w", encoding="utf-8") as f:
        f.write(json.dumps(test_dataset, indent=4))

    print(f"Train Dataset Size: {len(train_dataset)}, Test Dataset Size: {len(test_dataset)}")