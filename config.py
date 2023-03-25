import json

max_prefix_length = 3

# data_path = "/home/song/下载/拼音输入法作业-2023春/"
config_path = "/home/song/workspace/pinyin/config.json"

with open(config_path, "r", encoding="utf-8") as file:
    config = json.load(file)

corpora_path = config["corpora_path"]
vocab_path = config["vocab_path"]
pinyin_path = config["pinyin_path"]

input_path = [("/home/song/下载/拼音输入法作业-2023春/测试语料/input.txt", "utf-8")]
std_output_path = [("/home/song/下载/拼音输入法作业-2023春/测试语料/std_output.txt", "utf-8")]
output_path = [("/home/song/workspace/pinyin/output.txt", "utf-8")]

frequency_dict_path = "/home/song/workspace/pinyin/frequency_dict.json"
probabilistic_model_path = "/home/song/workspace/pinyin/probabilistic_model.json"
