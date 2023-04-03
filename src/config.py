import sys
import os
import json

from typing import List, Union
from utils import extract_data_path

model_config_path = ""
frequency_dict_path = ""
probabilistic_model_path = ""

config = {}

max_prefix_length = 0
corpora_path = []
vocab_path = [[os.path.join(os.path.dirname(os.path.realpath(__file__)), "./task_data/vocab.txt"), "gbk"]]
pinyin_path = [[os.path.join(os.path.dirname(os.path.realpath(__file__)), "./task_data/pinyin.txt"), "gbk"]]

input_path = []
std_output_path = []
output_path = []

def load_config(model_path, output_file_path: Union[List, None]=None, input_file_path: Union[List, None]=None, std_file_path: Union[List, None]=None, vocab_file_path: Union[List, None]=None, pinyin_file_path: Union[List, None]=None):
    global model_config_path, frequency_dict_path, probabilistic_model_path, config
    global max_prefix_length, corpora_path, vocab_path, pinyin_path
    global input_path, std_output_path, output_path

    if not os.path.isabs(model_path):
        if not os.path.exists(os.path.join(os.getcwd(), model_path)):
            if os.path.exists(os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", model_path)):
                model_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", model_path)
            else:
                print("Model not find :", model_path)
                exit()
        else:
            model_path = os.path.join(os.getcwd(), model_path)

    model_config_path = os.path.join(model_path, "config.json")
    frequency_dict_path = os.path.join(model_path, "frequency_dict.gz")
    probabilistic_model_path = os.path.join(model_path, "probabilistic_model.gz")

    with open(model_config_path, "r", encoding="utf-8") as file:
        config = json.load(file)

    max_prefix_length = config["max_prefix_length"]
    corpora_path = extract_data_path(config["corpora_path"])

    if output_file_path is None:
        output_file_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "./output.txt")

    if input_file_path is None:
        input_file_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "./task_data/input.txt")

    if std_file_path is None:
        std_file_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "./task_data/std_output.txt")

    if "vocab_path" in config and config["vocab_path"]:
        vocab_path = config["vocab_path"]

    if "pinyin_path" in config and config["pinyin_path"]:
        pinyin_path = config["pinyin_path"]

    if vocab_file_path is not None:
        vocab_path = vocab_file_path

    if pinyin_file_path is not None:
        pinyin_path = pinyin_file_path

    if isinstance(input_file_path, str):
        input_file_path = [(input_file_path, "utf-8")]
    if isinstance(std_file_path, str):
        std_file_path = [(std_file_path, "utf-8")]
    if isinstance(output_file_path, str):
        output_file_path = [(output_file_path, "utf-8")]

    input_path = input_file_path
    std_output_path = std_file_path
    output_path = output_file_path
