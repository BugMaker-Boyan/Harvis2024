import json


def is_valid(data_json):        
    
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
    
    OUTPUT = data_json["output"]

    if set(OUTPUT.keys()) != set(["operation", "file", "pointer", "group"]):
        return False
    
    return is_operation_valid(OUTPUT["operation"]) and is_file_valid(OUTPUT["file"]) and is_pointer_valid(OUTPUT["pointer"]) and is_group_valid(OUTPUT["group"])


def clean_dataset(dataset_json):
    dataset_clean = []
    for data_json in dataset_json:
        if is_valid(data_json):
            dataset_clean.append(data_json)
        else:
            print(f"Invalid: {data_json}")
    return dataset_clean


if __name__ == "__main__":
    DATASET_JSON_PATH = "./generate_dataset.json"
    CLEAN_DATASET_JSON_PATH = "./clean_generate_dataset.json"
    
    with open(DATASET_JSON_PATH, "r", encoding="utf-8") as f:
        dataset_json = json.load(f)
    
    dataset_clean = clean_dataset(dataset_json)
    
    with open(CLEAN_DATASET_JSON_PATH, "w", encoding="utf-8") as f:
        f.write(json.dumps(dataset_clean, indent=4))

    print(f"Before Clean: {len(dataset_json)}, After Clean: {len(dataset_clean)}")