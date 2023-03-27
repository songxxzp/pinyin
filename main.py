import os
import json
import math

from typing import List
from functools import partial

from config import load_config

load_config(
    model_path="/home/song/workspace/pinyin/trigram_weibo",
    input_file_path="/home/song/下载/拼音输入法作业-2023春/测试语料/input.txt",
    std_file_path="/home/song/下载/拼音输入法作业-2023春/测试语料/std_output.txt",
    output_file_path="/home/song/workspace/pinyin/output.txt"
)

from config import *

from metric import eval
from selector import default_top_k_selector, get_top_k_selector
from probability import get_probability_function
from model import load_vocab


para_lambda = 0.03  # 0.03
top_k_storage = 40
top_k_calculate = 40
max_conditional_prefix_length = 2
normalized = False
top_k_selector = "std"  # default, std, gpt
probability_function = "interpolation"  # interpolation, laplace
device = "cuda"

def pinyin_to_character(pinyin_str: str, probability_fn, pinyin_dict, max_conditional_prefix_length=1, top_k_storage=1, top_k_calculate=1, top_k_selector=default_top_k_selector):
    pinyin_list = pinyin_str.rstrip("\n").split(" ")
    if len(pinyin_list) < 1:
        return ""
    search_state = []
    for i in range(len(pinyin_list)):
        search_state.append({})
        pinyin = pinyin_list[i]
        for token in pinyin_dict[pinyin]:
            if i == 0:
                search_state[i][token] = [(probability_fn(token=token, prefix="", pinyin=pinyin), "")]
            else:
                for pre_token, last_token_prob_list in search_state[i - 1].items():
                    for token_prob_tuple in last_token_prob_list[:top_k_calculate]:
                        max_prefix_prob = token_prob_tuple[0]
                        prefix = token_prob_tuple[1] + pre_token
                        seq_prob = max_prefix_prob * probability_fn(token=token, prefix=prefix[-max_conditional_prefix_length:], pinyin=pinyin)  # TODO: use log
                        if token not in search_state[i]:
                            search_state[i][token] = [(seq_prob, prefix)]
                        else:
                            search_state[i][token].append((seq_prob, prefix))

            search_state[i][token].sort(key=lambda x: x[0], reverse=True)
            search_state[i][token] = search_state[i][token][: top_k_storage]

    # get last token with top k probability
    final_search_state = []
    for token, last_token_prob_list in search_state[-1].items():
        for token_prob_tuple in last_token_prob_list[:top_k_calculate]:
            final_search_state.append((token_prob_tuple[0], token_prob_tuple[1] + token))

    final_search_state.sort(key=lambda x: x[0], reverse=True)
    final_search_state = final_search_state[:top_k_storage]
    sentences = [sentence for _, sentence in final_search_state]  # sort from high prob to low prob

    return top_k_selector(seq_list=sentences)


def main():
    _, pinyin_dict = load_vocab()
    selector = get_top_k_selector(top_k_selector, device=device) # default, std, gpt
    probability_fn = get_probability_function(probability_function, para_lambda=para_lambda, normalized=normalized)  # interpolation, laplace

    process_fn = partial(
        pinyin_to_character,
        probability_fn=probability_fn,
        pinyin_dict=pinyin_dict,
        max_conditional_prefix_length=min(max_conditional_prefix_length, max_prefix_length),
        top_k_storage=top_k_storage,
        top_k_calculate=top_k_calculate,
        top_k_selector=selector
    )

    for (input_file_path, input_format), (output_file_path, output_format) in zip(input_path, output_path):
        with open(input_file_path, "r", encoding=input_format) as input_file:
            answers = list(map(process_fn, input_file.readlines()))
        with open(output_file_path, "w", encoding=output_format) as output_file:
            answers = [answer + '\n' for answer in answers]
            output_file.writelines(answers)


if __name__ == "__main__":
    main()
    print(para_lambda, top_k_storage, top_k_calculate, max_conditional_prefix_length, top_k_selector, probability_function)
    eval()
