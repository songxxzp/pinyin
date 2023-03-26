import os
import json

model_path = "/home/song/workspace/pinyin/trigram_news_2017_to_2022_with_sina"
model_config_path = os.path.join(model_path, "config.json")
frequency_dict_path = os.path.join(model_path, "frequency_dict.gz")
probabilistic_model_path = os.path.join(model_path, "probabilistic_model.gz")

with open(model_config_path, "r", encoding="utf-8") as file:
    config = json.load(file)

max_prefix_length = config["max_prefix_length"]
corpora_path = config["corpora_path"]
vocab_path = config["vocab_path"]
pinyin_path = config["pinyin_path"]

input_path = [("/home/song/下载/拼音输入法作业-2023春/测试语料/input.txt", "utf-8")]
std_output_path = [("/home/song/下载/拼音输入法作业-2023春/测试语料/std_output.txt", "utf-8")]
output_path = [("/home/song/workspace/pinyin/output.txt", "utf-8")]
