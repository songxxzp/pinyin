import os
import json
import math

from typing import List
from functools import partial

from config import *
from model import *
from metric import *


para_lambda = 0.03
top_k = 10
max_conditional_prefix_length = 2


def laplace_smoothing(conditional_frequency_dict, prefix_frequency_dict, token, prefix="", vocab_num=6763):
    probability = (get_conditional_frequency(conditional_frequency_dict, prefix, token) + 1) / (get_frequency(prefix_frequency_dict, prefix) + vocab_num)
    return probability


def interpolation_smoothing(conditional_probabilistic_dict, token, prefix=""):
    """ probability smoothing """
    # if len(prefix) > 1:
    #     return get_conditional_probability(probabilistic_dict, prefix, token) * (1 - para_lambda) + smoothing_probability(probabilistic_dict, prefix[-1:], token) * para_lambda
    probability = get_conditional_probability(conditional_probabilistic_dict, prefix, token) * (1 - para_lambda) + get_conditional_probability(conditional_probabilistic_dict, "", token) * para_lambda
    return probability


def pinyin_to_character(pinyin_str: str, smoothing, pinyin_dict, conditional_probabilistic_dict, max_conditional_prefix_length=1, top_k=1):
    pinyin_list = pinyin_str.rstrip("\n").split(" ")
    if len(pinyin_list) < 1:
        return ""
    search_state = []
    for i in range(len(pinyin_list)):
        search_state.append({})
        pinyin = pinyin_list[i]
        for token in pinyin_dict[pinyin]:
            if i == 0:
                search_state[i][token] = [(smoothing(token=token), "")]
            else:
                for pre_token, max_prefix_prob_list in search_state[i - 1].items():
                    for j in range(len(max_prefix_prob_list)):
                        max_prefix_prob = max_prefix_prob_list[-j][0]
                        prefix = (max_prefix_prob_list[-j][1] + pre_token)[-max_conditional_prefix_length:]
                        seq_prob = max_prefix_prob * smoothing(token=token, prefix=prefix)  # TODO: use log
                        if token not in search_state[i]:
                            search_state[i][token] = [(seq_prob, prefix)]
                        else:
                            search_state[i][token].append((seq_prob, prefix))

            search_state[i][token].sort(key=lambda x: x[0])
            search_state[i][token] = search_state[i][token][- top_k : ]

    # get last token with highst probability
    last_token = ""
    max_prob = -1e-8
    for token, max_token_prob_list in search_state[-1].items():
        max_token_prob = max_token_prob_list[-1][0]
        if max_token_prob > max_prob:
            max_prob = max_token_prob
            last_token = token
    
    # track last token
    sentence = ""
    for i in reversed(range(len(pinyin_list))):
        sentence = last_token + sentence
        max_token_prob_list = search_state[i][last_token]
        last_token = max_token_prob_list[-1][1]
        if len(last_token) > 1:
            last_token = last_token[-1]

    return sentence


def main():
    vocabs, pinyin_dict = load_vocab()
    conditional_frequency_dict = load_frequency_dict(vocabs=vocabs)
    prefix_frequency_dict = load_prefix_frequency_dict(conditional_frequency_dict)
    conditional_probabilistic_dict = load_probabilistic_model(vocabs=vocabs)

    print("model loaded")

    smoothing = partial(
        interpolation_smoothing,
        conditional_probabilistic_dict=conditional_probabilistic_dict
    )

    # smoothing = partial(
    #     laplace_smoothing,
    #     conditional_frequency_dict=conditional_frequency_dict,
    #     prefix_frequency_dict=prefix_frequency_dict,
    #     vocab_num=len(vocabs)
    # )

    process_fn = partial(
        pinyin_to_character,
        smoothing=smoothing,
        pinyin_dict=pinyin_dict,
        conditional_probabilistic_dict=conditional_probabilistic_dict,
        max_conditional_prefix_length=min(max_conditional_prefix_length, max_prefix_length),
        top_k=top_k
    )

    for (input_file_path, input_format), (output_file_path, output_format) in zip(input_path, output_path):
        with open(input_file_path, "r", encoding=input_format) as input_file:
            answers = map(process_fn, input_file.readlines())
        with open(output_file_path, "w", encoding=output_format) as output_file:
            answers = [answer + '\n' for answer in answers]
            output_file.writelines(answers)


if __name__ == "__main__":
    main()
    eval()
