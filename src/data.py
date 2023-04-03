from config import *

corpora_path = corpora_path[-1:]

with open("data.txt", "w", encoding="utf-8") as data_file:
    for path, format, labels in corpora_path:  # should be jsonl
        with open(path, "r", encoding=format) as file:
            for line in file.readlines():
                content_dict = json.loads(line)
                for label in labels:
                    data_file.write(content_dict[label] + '\n')
