import os
import sys
import json
import math
import time
import copy

from typing import List
from functools import partial

from config import load_config

current_path = os.path.dirname(os.path.realpath(__file__))

load_config(
    model_path="/home/junli/workspace/pinyin/trigram_weibo_newscrawl_baike"  # trigram_weibo_newscrawl_baike, trigram_newscrawl_baike, bigram_weibo
)

from config import *

from metric import eval
from selector import default_top_k_selector, get_top_k_selector
from probability import get_probability_function
from model import load_vocab


interpolation_lambda = 0.01  # 0.03
top_k_storage = 4
top_k_calculate = 4
max_conditional_prefix_length = 2
normalized = False
top_k_selector = "default"  # default, std, gpt
probability_function = "interpolation"  # interpolation, laplace
device = "cuda"


def print_parameter():
    print("max conditional prefix length :", max_conditional_prefix_length)
    print("top k storage :", top_k_storage)
    print("top k calculate :", top_k_calculate)
    print("top k selector :", top_k_selector)
    if top_k_selector == "gpt":
        print("    device", device)
    print("probability function :", probability_function)
    if probability_function == "interpolation":
        print("    lambda :", interpolation_lambda)
    print("normalized :", normalized)


def pinyin_to_character(pinyin_str: str, probability_fn, pinyin_dict, max_conditional_prefix_length=1, top_k_storage=1, top_k_calculate=1, top_k_selector=default_top_k_selector):
    pinyin_list = pinyin_str.rstrip("\n").split(" ")
    if len(pinyin_list) < 1:
        return ""
    last_search_state = {}
    for i in range(len(pinyin_list)):
        search_state = {}
        pinyin = pinyin_list[i]
        for token in pinyin_dict[pinyin]:
            if i == 0:
                search_state[token] = [(probability_fn(token=token, prefix="", pinyin=pinyin), token)]
            else:
                search_state[token] = []
                for _, last_token_prob_list in last_search_state.items():
                    for max_prefix_prob, prefix in last_token_prob_list[:top_k_calculate]:  # (prob, prefix)
                        seq_prob = max_prefix_prob * probability_fn(token=token, prefix=prefix[-max_conditional_prefix_length:], pinyin=pinyin)
                        search_state[token].append((seq_prob, prefix + token))

            search_state[token].sort(key=lambda x: x[0], reverse=True)
            search_state[token] = search_state[token][: top_k_storage]

        last_search_state = search_state

    # get sentences with top k probability
    final_search_state = []
    for _, last_token_prob_list in last_search_state.items():
        final_search_state += last_token_prob_list[:top_k_calculate]

    final_search_state.sort(key=lambda x: x[0], reverse=True)
    final_search_state = final_search_state[:top_k_storage]

    sentences = [sentence for _, sentence in final_search_state]  # sort from high prob to low prob

    return top_k_selector(seq_list=sentences)


def inference():
    _, pinyin_dict = load_vocab()
    selector = get_top_k_selector(top_k_selector, device=device) # default, std, gpt
    probability_fn = get_probability_function(probability_function, para_lambda=interpolation_lambda, normalized=normalized)  # interpolation, laplace

    process_fn = partial(
        pinyin_to_character,
        probability_fn=probability_fn,
        pinyin_dict=pinyin_dict,
        max_conditional_prefix_length=min(max_conditional_prefix_length, max_prefix_length),
        top_k_storage=top_k_storage,
        top_k_calculate=top_k_calculate,
        top_k_selector=selector
    )

    time_usage = 0

    for (input_file_path, input_format), (output_file_path, output_format) in zip(input_path, output_path):
        with open(input_file_path, "r", encoding=input_format) as input_file:
            start_time = time.time()
            answers = list(map(process_fn, input_file.readlines()))
            stop_time = time.time()
            time_usage += stop_time - start_time
        with open(output_file_path, "w", encoding=output_format) as output_file:
            answers = [answer + '\n' for answer in answers]
            output_file.writelines(answers)
    
    return time_usage


if __name__ == "__main__":
    print("Inferencing")

    print_parameter()
    time_usage = inference()
    print("Time usage : {}s".format(str(round(time_usage, 2))))
    
    sentence_acc, word_acc = eval()
    print("sentence accuracy :", sentence_acc)
    print("word accuracy :", word_acc)
