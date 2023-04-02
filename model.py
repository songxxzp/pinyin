import json
import pickle
import os
import gzip

from typing import Dict, List

from utils import extract_data_path


def load_vocab():
    from config import vocab_path, pinyin_path
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
        if content[j] not in vocabs:
            if i != j:
                text_list.append(content[i:j])
            i = j + 1

    if i != len(content):
        text_list.append(content[i:])

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


def add_frequency(frequency_dict, string, frequency=1):
    if string not in frequency_dict:
        frequency_dict[string] = 0
    frequency_dict[string] += frequency


def get_frequency(frequency_dict, string):
    if string not in frequency_dict:
        return 0
    return frequency_dict[string]


def get_conditional_probability(conditional_probabilistic_dict, prefix, token):
    if prefix not in conditional_probabilistic_dict:
        return 0
    if token not in conditional_probabilistic_dict[prefix]:
        return 0
    return conditional_probabilistic_dict[prefix][token]


def process_content(content, conditional_frequency_dict, vocabs, max_prefix_length):
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


def build_frequency_dict(vocabs):
    from config import frequency_dict_path, config, corpora_path, max_prefix_length
    # build frequency dict
    print("Building conditional frequency dict")
    conditional_frequency_dict = {}
    frequency_dict = None  # frequency dict cache

    if os.path.exists(frequency_dict_path + ".cache"):  # use cache
        print("Loading cache")
        with gzip.open(frequency_dict_path + ".cache", "rb") as file:
            frequency_dict = pickle.load(file)
            try:
                if "config" in frequency_dict and config["max_prefix_length"] == frequency_dict["config"]["max_prefix_length"]:
                    print("Loaded cache, trained on:")
                    cache_check = True
                    corpus_basenames = [os.path.basename(data_path) for data_path, _, _ in corpora_path]
                    frequency_dict["config"]["corpora_path"] = extract_data_path(frequency_dict["config"]["corpora_path"])
                    for path, format, labels in frequency_dict["config"]["corpora_path"]:
                        print((path, format, labels))
                        if os.path.basename(path) not in corpus_basenames:
                            print("Unknown :", path)
                            cache_check = False
                    if cache_check:
                        conditional_frequency_dict: Dict = frequency_dict["conditional_frequency_dict"]
                    else:
                        frequency_dict = None
                        print("Found unknown corpus, cache cannot be used.")
                else:
                    frequency_dict = None
                    conditional_frequency_dict = {}
            except Exception as exception:
                print(repr(exception))
                frequency_dict = None
                conditional_frequency_dict = {}

    for path, format, labels in corpora_path:  # should be jsonl or txt
        print("Processing :", (path, format, labels))

        if frequency_dict is not None:
            try:
                if os.path.basename(path) in [os.path.basename(data_path) for data_path, _, _ in frequency_dict["config"]["corpora_path"]]:
                    print("Use cache :", (path, format, labels))
                    continue
            except Exception as exception:
                print(repr(exception))

        with open(path, "r", encoding=format) as file:
            for line in file.readlines():
                line = line.rstrip('\n')
                try:
                    if len(labels) and line:  # jsonl
                        content_dict = json.loads(line)
                        for label in labels:
                            content = content_dict[label]
                            process_content(content, conditional_frequency_dict, vocabs, max_prefix_length)
                    else:
                        content = line
                        process_content(content, conditional_frequency_dict, vocabs, max_prefix_length)
                except Exception as exception:
                    print(repr(exception))
                    print("Cannot load line, check your format:", line)

    with gzip.open(frequency_dict_path, "wb") as file:
        print("saving :", frequency_dict_path)
        pickle.dump(
            {
                "config": config,
                "conditional_frequency_dict": conditional_frequency_dict
            },
            file
        )
    
    return conditional_frequency_dict


def build_probabilistic_model(conditional_frequency_dict, vocabs):
    from config import probabilistic_model_path, config
    # build probabilistic model
    print("Building probabilistic model")
    conditional_probabilistic_dict = {}
    conditional_probabilistic_dict[""] = {}

    for prefix, token_frequency_dict in conditional_frequency_dict.items():
        conditional_frequency_dict[prefix] = {}  # release memory
        conditional_probabilistic_dict[prefix] = {}
        total_conditional_frequency = sum(token_frequency_dict.values())
        for token, frequency in token_frequency_dict.items():
            conditional_probabilistic_dict[prefix][token] = frequency / total_conditional_frequency

    with gzip.open(probabilistic_model_path, "wb") as file:
        print("saving :", probabilistic_model_path)
        pickle.dump(
            {
                "config": config,
                "conditional_probabilistic_dict": conditional_probabilistic_dict
            },
            file
        )
    
    return conditional_probabilistic_dict


def load_frequency_dict(vocabs):
    from config import frequency_dict_path, config
    if not os.path.exists(frequency_dict_path):
        conditional_frequency_dict: Dict = build_frequency_dict(vocabs=vocabs)
    else:
        with gzip.open(frequency_dict_path, "rb") as file:
            frequency_dict = pickle.load(file)
            try:
                if "config" in frequency_dict and config["max_prefix_length"] == frequency_dict["config"]["max_prefix_length"]:
                    conditional_frequency_dict: Dict = frequency_dict["conditional_frequency_dict"]
                else:
                    conditional_frequency_dict: Dict = build_frequency_dict(vocabs=vocabs)
            except Exception as exception:
                print(repr(exception))
                conditional_frequency_dict: Dict = build_frequency_dict(vocabs=vocabs)

    return conditional_frequency_dict


def load_prefix_frequency_dict(conditional_frequency_dict):
    prefix_frequency_dict = {}
    for prefix, token_frequency_dict in conditional_frequency_dict.items():
        prefix_frequency_dict[prefix] = sum(token_frequency_dict.values())
    return prefix_frequency_dict


def load_probabilistic_model(vocabs, conditional_frequency_dict=None):
    from config import probabilistic_model_path, config
    if not os.path.exists(probabilistic_model_path):
        if conditional_frequency_dict is None:
            conditional_frequency_dict = load_frequency_dict(vocabs)
        conditional_probabilistic_dict: Dict = build_probabilistic_model(conditional_frequency_dict, vocabs)
    else:
        with gzip.open(probabilistic_model_path, "rb") as file:
            print("Loading probabilistic model")
            probabilistic_model = pickle.load(file)
            try:
                if "config" in probabilistic_model and config["max_prefix_length"] == probabilistic_model["config"]["max_prefix_length"]:
                    conditional_probabilistic_dict: Dict = probabilistic_model["conditional_probabilistic_dict"]
                else:
                    if conditional_frequency_dict is None:
                        conditional_frequency_dict = load_frequency_dict(vocabs)
                    conditional_probabilistic_dict: Dict = build_probabilistic_model(conditional_frequency_dict, vocabs)
            except Exception as exception:
                print(repr(exception))
                conditional_probabilistic_dict: Dict = build_probabilistic_model(conditional_frequency_dict, vocabs)

    return conditional_probabilistic_dict
