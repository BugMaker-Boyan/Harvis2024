
with open("./temp.txt", "r", encoding="utf-8") as f:
    data = f.readlines()
    
newline_idx = [idx for idx, line in enumerate(data) if line == "\n"]
newline_idx.append(len(data))

seed_examples = []

import json

previous_idx = 0
for idx in newline_idx:
    group_data = data[previous_idx: idx]
    previous_idx = idx + 1
    
    input_str = group_data[0].replace(";", ",").replace("\"null\"", "null").replace("\n", "")
    if input_str.endswith(","):
        input_str = input_str[:-1]
    input_str = "{" + input_str + "}"
    print(input_str)
    output = json.loads(input_str)
    

    for input in group_data[1:]:
        seed_examples.append({
            "input": input.replace("\"", "").strip(),
            "output": output
        })

with open("./seed_dataset.json", "w", encoding="utf-8") as f:
    f.write(json.dumps(seed_examples, indent=4))
    