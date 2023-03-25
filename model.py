import json
import os

from typing import Dict, List

from config import *

def load_vocab():
    # load vocab
    vocabs = {}
    for path, format in vocab_path:
        with open(path, "r", encoding=format) as file:
            for vocab in file.read():
                vocabs[vocab] = []

    # build pinyin dict
    pinyin_dict = {}
    for path, format in pinyin_path:
        with open(path, "r", encoding=format) as file:
            for line in file.readlines():
                alphabet = line.rstrip("\n").split(" ")
                if alphabet and alphabet[0] not in pinyin_dict:
                    pinyin_dict[alphabet[0]] = alphabet[1:]
                    for vocab in alphabet[1:]:
                        vocabs[vocab].append(alphabet[0])
                else:
                    print("Conflict in pinyin dict.")
    
    return vocabs, pinyin_dict


def split(content:str, vocabs:Dict) -> List[str]:
    """ split content by none vocabs """
    i = 0
    text_list = []
    for j in range(len(content)):
        if i != j and content[j] not in vocabs:
            text_list.append(content[i:j])
            i = j + 1
    return text_list


def add_conditional_frequency(conditional_frequency_dict, prefix, token, frequency=1):
    if prefix not in conditional_frequency_dict:
        conditional_frequency_dict[prefix] = {}
    if token not in conditional_frequency_dict[prefix]:
        conditional_frequency_dict[prefix][token] = 0
    conditional_frequency_dict[prefix][token] += frequency


def get_conditional_frequency(conditional_frequency_dict, prefix, token):
    if prefix not in conditional_frequency_dict:
        return 0
    if token not in conditional_frequency_dict[prefix]:
        return 0
    return conditional_frequency_dict[prefix][token]


def get_conditional_probability(conditional_probabilistic_dict, prefix, token):
    if prefix not in conditional_probabilistic_dict:
        return 0
    if token not in conditional_probabilistic_dict[prefix]:
        return 0
    return conditional_probabilistic_dict[prefix][token]


def build_frequency_dict(vocabs):
    # build frequency dict
    conditional_frequency_dict = {}

    for path, format, labels in corpora_path:  # should be jsonl
        with open(path, "r", encoding=format) as file:
            for line in file.readlines():
                content_dict = json.loads(line)
                for label in labels:
                    content = content_dict[label]
                    # split content by none vocabs
                    text_list = split(content, vocabs=vocabs)
                    for text in text_list:
                        for suffix_start_pos in range(len(text)):
                            suffix = text[suffix_start_pos : ]
                            prefix = ""
                            for token in suffix:
                                add_conditional_frequency(conditional_frequency_dict, prefix, token, frequency=1)
                                prefix += token
                                if len(prefix) > max_prefix_length:
                                    break

    with open(frequency_dict_path, "w", encoding="utf-8") as file:
        json.dump(
            {
                "config": config,
                "conditional_frequency_dict": conditional_frequency_dict
            },
            file,
            ensure_ascii=False
        )
    
    return conditional_frequency_dict


def build_probabilistic_model(conditional_frequency_dict, vocabs):
    # build probabilistic model
    conditional_probabilistic_dict = {}
    conditional_probabilistic_dict[""] = {}

    for prefix, token_frequency_dict in conditional_frequency_dict.items():
        conditional_probabilistic_dict[prefix] = {}
        total_conditional_frequency = sum(conditional_frequency_dict[prefix].values())
        for token, frequency in token_frequency_dict.items():
            conditional_probabilistic_dict[prefix][token] = frequency / total_conditional_frequency

    with open(probabilistic_model_path, "w", encoding="utf-8") as file:
        json.dump(
            {
                "config": config,
                "conditional_probabilistic_dict": conditional_probabilistic_dict
            },
            file,
            ensure_ascii=False
        )
    
    return conditional_probabilistic_dict


def load_frequency_dict(vocabs):
    if not os.path.exists(frequency_dict_path):
        conditional_frequency_dict: Dict = build_frequency_dict(vocabs=vocabs)
    else:
        with open(frequency_dict_path, "r", encoding="utf-8") as file:
            frequency_dict = json.load(file)
            if "config" in frequency_dict and config == frequency_dict["config"]:
                conditional_frequency_dict: Dict = frequency_dict["conditional_frequency_dict"]
            else:
                conditional_frequency_dict: Dict = build_frequency_dict(vocabs=vocabs)
    return conditional_frequency_dict


def load_probabilistic_model(vocabs):
    if not os.path.exists(probabilistic_model_path):
        conditional_probabilistic_dict: Dict = build_probabilistic_model(load_frequency_dict(vocabs), vocabs)
    else:
        with open(probabilistic_model_path, "r", encoding="utf-8") as file:
            probabilistic_model = json.load(file)
            if "config" in probabilistic_model and config == probabilistic_model["config"]:
                conditional_probabilistic_dict: Dict = probabilistic_model["conditional_probabilistic_dict"]
            else:
                conditional_probabilistic_dict: Dict = build_probabilistic_model(load_frequency_dict(vocabs), vocabs)
    return conditional_probabilistic_dict
