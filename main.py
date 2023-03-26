import os
import json
import math

from typing import List
from functools import partial

from config import *
from model import *
from metric import *
from gpt import *


para_lambda = 0.03
top_k_storage = 10
top_k_calculate = 1
max_conditional_prefix_length = 2


def laplace_smoothing(conditional_frequency_dict, prefix_frequency_dict, token, pinyin=None, prefix="", vocab_num=6763):
    """
        laplace smoothing
        return p(token|prefix)
    """
    probability = (get_conditional_frequency(conditional_frequency_dict, prefix, token) + 1) / (get_frequency(prefix_frequency_dict, prefix) + vocab_num)
    return probability


def interpolation_smoothing(conditional_probabilistic_dict, token, pinyin=None, prefix=""):
    """
        interpolation smoothing
        return p(token|prefix)
    """
    # if len(prefix) > 1:
    #     return get_conditional_probability(probabilistic_dict, prefix, token) * (1 - para_lambda) + smoothing_probability(probabilistic_dict, prefix[-1:], token) * para_lambda
    probability = get_conditional_probability(conditional_probabilistic_dict, prefix, token) * (1 - para_lambda) + get_conditional_probability(conditional_probabilistic_dict, "", token) * para_lambda
    return probability


def normalized_probability(pinyin_dict, probability_fn, token, pinyin, prefix=""):
    """
        normalized probability
        return p(token|prefix, pinyin)
    """
    probability = probability_fn(token=token, prefix=prefix)
    totall_probability = sum([probability_fn(token=token, prefix=prefix) for token in pinyin_dict[pinyin]])
    return probability / totall_probability


def default_top_k_selector(seq_list: List[str]) -> str:
    """
        choose one answer from a list
        return seq_list[0] as default
    """
    return seq_list[0]


def gpt_top_k_selector(seq_list: List[str], model, tokenizer) -> str:
    """
        choose one answer from a list
        return seq with minima PPL
    """
    ppl = perplexity(seq_list, model, tokenizer)
    seq = seq_list[ppl.index(min(ppl))]
    return seq


def pinyin_to_character(pinyin_str: str, get_probability, pinyin_dict, max_conditional_prefix_length=1, top_k_storage=1, top_k_calculate=1, top_k_selector=default_top_k_selector):
    pinyin_list = pinyin_str.rstrip("\n").split(" ")
    if len(pinyin_list) < 1:
        return ""
    search_state = []
    for i in range(len(pinyin_list)):
        search_state.append({})
        pinyin = pinyin_list[i]
        for token in pinyin_dict[pinyin]:
            if i == 0:
                search_state[i][token] = [(get_probability(token=token, prefix="", pinyin=pinyin), "")]
            else:
                for pre_token, max_prefix_prob_list in search_state[i - 1].items():
                    for j in range(len(max_prefix_prob_list)):
                        max_prefix_prob = max_prefix_prob_list[-j][0]
                        prefix = (max_prefix_prob_list[-j][1] + pre_token)
                        seq_prob = max_prefix_prob * get_probability(token=token, prefix=prefix[-max_conditional_prefix_length:], pinyin=pinyin)  # TODO: use log
                        if token not in search_state[i]:
                            search_state[i][token] = [(seq_prob, prefix)]
                        else:
                            search_state[i][token].append((seq_prob, prefix))

            search_state[i][token].sort(key=lambda x: x[0])
            search_state[i][token] = search_state[i][token][- top_k_storage : ]

    # get last token with highst probability

    final_search_state = []
    for token, last_token_prob_list in search_state[-1].items():
        for token_prob_tuple in last_token_prob_list:
            final_search_state.append((token_prob_tuple[0], token_prob_tuple[1] + token))

    final_search_state.sort(key=lambda x: x[0], reverse=True)
    final_search_state = final_search_state[:top_k_storage]

    # track last token
    sentences = []
    for _, sentence in final_search_state:
        sentences.append(sentence)

    # print(sentences)

    return top_k_selector(seq_list=sentences)


def main():
    vocabs, pinyin_dict = load_vocab()
    conditional_probabilistic_dict = load_probabilistic_model(vocabs=vocabs)

    probability_fn = partial(
        interpolation_smoothing,
        conditional_probabilistic_dict=conditional_probabilistic_dict
    )

    # model, tokenizer = load_gpt_model(device="cuda")
    # selector = partial(
    #     gpt_top_k_selector,
    #     model=model,
    #     tokenizer=tokenizer
    # )

    # conditional_frequency_dict = load_frequency_dict(vocabs=vocabs)
    # prefix_frequency_dict = load_prefix_frequency_dict(conditional_frequency_dict)
    # probability_fn = partial(
    #     laplace_smoothing,
    #     conditional_frequency_dict=conditional_frequency_dict,
    #     prefix_frequency_dict=prefix_frequency_dict,
    #     vocab_num=len(vocabs)
    # )

    get_probability = probability_fn

    # get_probability = partial(
    #     normalized_probability,
    #     pinyin_dict=pinyin_dict,
    #     probability_fn=probability_fn
    # )

    process_fn = partial(
        pinyin_to_character,
        get_probability=get_probability,
        pinyin_dict=pinyin_dict,
        max_conditional_prefix_length=min(max_conditional_prefix_length, max_prefix_length),
        top_k_storage=top_k_storage,
        top_k_calculate=top_k_calculate,
        # top_k_selector=selector
    )

    for (input_file_path, input_format), (output_file_path, output_format) in zip(input_path, output_path):
        with open(input_file_path, "r", encoding=input_format) as input_file:
            answers = list(map(process_fn, input_file.readlines()))
        with open(output_file_path, "w", encoding=output_format) as output_file:
            answers = [answer + '\n' for answer in answers]
            output_file.writelines(answers)


if __name__ == "__main__":
    main()
    eval()
