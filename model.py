import json

from typing import Dict

from config import *

def load_vocab():
    # load vocab
    vocabs = {}
    for path, format in vocab_path:
        with open(path, "r", encoding=format) as file:
            for vocab in file.read():
                vocabs[vocab] = ""

    # build pinyin dict
    pinyin_dict = {}
    for path, format in pinyin_path:
        with open(path, "r", encoding=format) as file:
            for line in file.readlines():
                alphabet = line.rstrip("\n").split(" ")
                if alphabet and alphabet[0] not in pinyin_dict:
                    pinyin_dict[alphabet[0]] = alphabet[1:]
                    for vocab in alphabet[1:]:
                        vocabs[vocab] = alphabet[0]
                else:
                    print("Conflict in pinyin dict.")
    
    return vocabs, pinyin_dict


def build_probabilistic_model(vocabs):
    # build frequency dict
    single_token_frequency_dict = {}
    conditional_frequency_dict = {}

    for prefix in vocabs.keys():
        conditional_frequency_dict[prefix] = {}
        single_token_frequency_dict[prefix] = 0

    for path, format, labels in corpora_path:  # should be jsonl
        with open(path, "r", encoding=format) as file:
            for line in file.readlines():
                content_dict = json.loads(line)
                for label in labels:
                    content = content_dict[label]
                    for token in content:
                        if token not in vocabs:
                            continue
                        single_token_frequency_dict[token] += 1
                    if len(content) < 2:
                        continue
                    for i in range(len(content) - 1):
                        prefix = content[i]
                        token = content[i + 1]
                        if prefix not in vocabs or token not in vocabs:
                            continue
                        if token in conditional_frequency_dict[prefix]:
                            conditional_frequency_dict[prefix][token] += 1
                        else:
                            conditional_frequency_dict[prefix][token] = 1

    with open(frequency_dict_path, "w", encoding="utf-8") as file:
        json.dump(
            {
                "single_token_frequency_dict": single_token_frequency_dict,
                "conditional_frequency_dict": conditional_frequency_dict,
            },
            file,
            ensure_ascii=False
        )


    # build probabilistic model
    conditional_probabilistic_dict = {}
    single_token_probabilistic_dict = {}

    with open(frequency_dict_path, "r", encoding="utf-8") as file:
        frequency_dict = json.load(file)
        single_token_frequency_dict: Dict = frequency_dict["single_token_frequency_dict"]
        conditional_frequency_dict: Dict = frequency_dict["conditional_frequency_dict"]

        total_frequency = sum(single_token_frequency_dict.values())

        for prefix, token_frequency_dict in conditional_frequency_dict.items():
            single_token_probabilistic_dict[prefix] = single_token_frequency_dict[prefix] / total_frequency
            conditional_probabilistic_dict[prefix] = {}
            total_conditional_frequency = sum(conditional_frequency_dict[prefix].values())
            for token, frequency in token_frequency_dict.items():
                conditional_probabilistic_dict[prefix][token] = frequency / total_conditional_frequency

    with open(probabilistic_model_path, "w", encoding="utf-8") as file:
        json.dump(
            {
                "single_token_probabilistic_dict": single_token_probabilistic_dict,
                "conditional_probabilistic_dict": conditional_probabilistic_dict,
            },
            file,
            ensure_ascii=False
        )
    
    return single_token_probabilistic_dict, conditional_probabilistic_dict


def load_probabilistic_model():
    with open(probabilistic_model_path, "r", encoding="utf-8") as file:
        probabilistic_model = json.load(file)
        single_token_probabilistic_dict: Dict = probabilistic_model["single_token_probabilistic_dict"]
        conditional_probabilistic_dict: Dict = probabilistic_model["conditional_probabilistic_dict"]
    return single_token_probabilistic_dict, conditional_probabilistic_dict

