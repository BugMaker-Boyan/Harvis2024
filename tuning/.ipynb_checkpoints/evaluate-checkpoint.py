import json


if __name__ == "__main__":
    with open("./predictions.txt", "r", encoding="utf-8") as f:
        predictions = f.readlines()
    
    with open("./data/seed_dataset.json", "r", encoding="utf-8") as f:
        references = json.load(f)
        references = [json.dumps(ref["output"]) for ref in references]
    

    assert len(references) == len(predictions)
    
    keys = ["operation", "file", "pointer", "group"]
    
    operation_count = 0
    file_count = 0
    pointer_count = 0
    group_count = 0
    count = 0
    
    false_samples = []
    
    for pred, ref in zip(predictions, references):
        try:
            pred = eval(pred)
            ref = json.loads(ref)

            operation_flag = "operation" in pred and ref["operation"] == pred["operation"]
            file_flag = "file" in pred and ref["file"] == pred["file"]
            pointer_flag = True
            if "pointer" not in pred:
                pointer_flag = False
            elif (ref["pointer"] is None and pred["pointer"] is not None) or (ref["pointer"] is not None and pred["pointer"] is None):
                pointer_flag = False
            elif ref["pointer"] is not None and pred["pointer"] is not None and (set(ref["pointer"]) != set(pred["pointer"]) and (set([i + 1 for i in pred["pointer"]]) != set(ref["pointer"]))):
                pointer_flag = False
            group_flag = "group" in pred and ref["group"] == pred["group"]
            
            if operation_flag:
                operation_count += 1
            if file_flag:
                file_count += 1
            if pointer_flag:
                pointer_count += 1
            if group_flag:
                group_count += 1

            if operation_flag and file_flag and pointer_flag and group_flag:
                count += 1
            else:
                false_samples.append({"pred": pred, "ref": ref})
                
        except Exception as e:
            print(e, pred, ref)
            continue
    print(f"operation: {operation_count} / {len(predictions)}, exec {operation_count / len(predictions) * 100} %")
    print(f"file: {file_count} / {len(predictions)}, exec {file_count / len(predictions) * 100} %")
    print(f"pointer: {pointer_count} / {len(predictions)}, exec {pointer_count / len(predictions) * 100} %")
    print(f"group: {group_count} / {len(predictions)}, exec {group_count / len(predictions) * 100} %")
    print(f"total: {count} / {len(predictions)}, exec {count / len(predictions) * 100} %")
    
    with open("./false_sampels.json", "w", encoding="utf-8") as f:
        f.write(json.dumps(false_samples, indent=4))
