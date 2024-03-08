import json

with open("./data/raw_seed_data.txt", "r", encoding="utf-8") as f:
    raw_data = f.readlines()

newline_index = [idx for idx, line in enumerate(raw_data) if line == "\n"]

data_json = []

previous_idx = 0
for newline_idx in newline_index:
    data_group = raw_data[previous_idx: newline_idx]
    print(data_group[0])
    output = json.loads(data_group[0])
    nl_input = data_group[1:]
    for nl in nl_input:
        nl, group = nl.split("| All data groups include")
        nl = nl.strip()
        group = eval(group.replace(".", "").strip())
        group = group if group != ["DEFAULT_GROUP"] else []
        data_json.append({
            "input": nl,
            "group": json.dumps(group).replace("\n", "").strip(),
            "output": json.dumps(output).replace("\n", "").strip()
        })
    previous_idx = newline_idx + 1

print(previous_idx, len(raw_data))

with open("./data/seed_data.json", "w", encoding="utf-8") as f:
    f.write(json.dumps(data_json, indent=4))
    