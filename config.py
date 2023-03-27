import os
import json

model_config_path = ""
frequency_dict_path = ""
probabilistic_model_path = ""

config = {}

max_prefix_length = 0
corpora_path = []
vocab_path = []
pinyin_path = []

input_path = []
std_output_path = []
output_path = []

def load_config(model_path, input_file_path, std_file_path, output_file_path):
    global model_config_path, frequency_dict_path, probabilistic_model_path, config
    global max_prefix_length, corpora_path, vocab_path, pinyin_path
    global input_path, std_output_path, output_path
    # model_path = "/home/song/workspace/pinyin/trigram_news_2017_to_2022_with_sina"

    model_config_path = os.path.join(model_path, "config.json")
    frequency_dict_path = os.path.join(model_path, "frequency_dict.gz")
    probabilistic_model_path = os.path.join(model_path, "probabilistic_model.gz")

    with open(model_config_path, "r", encoding="utf-8") as file:
        config = json.load(file)

    max_prefix_length = config["max_prefix_length"]
    corpora_path = config["corpora_path"]
    vocab_path = config["vocab_path"]
    pinyin_path = config["pinyin_path"]

    if isinstance(input_file_path, str):
        input_file_path = [(input_file_path, "utf-8")]
    if isinstance(std_file_path, str):
        std_file_path = [(std_file_path, "utf-8")]
    if isinstance(output_file_path, str):
        output_file_path = [(output_file_path, "utf-8")]

    input_path = input_file_path
    std_output_path = std_file_path
    output_path = output_file_path
    # input_path = [("/home/song/下载/拼音输入法作业-2023春/测试语料/input.txt", "utf-8")]
    # std_output_path = [("/home/song/下载/拼音输入法作业-2023春/测试语料/std_output.txt", "utf-8")]
    # output_path = [("/home/song/workspace/pinyin/output.txt", "utf-8")]
