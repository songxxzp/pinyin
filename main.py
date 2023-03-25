import os
import json

from typing import List

from config import *
from model import *


vocabs, pinyin_dict = load_vocab()
# single_token_probabilistic_dict, conditional_probabilistic_dict = build_probabilistic_model(vocabs=vocabs)
single_token_probabilistic_dict, conditional_probabilistic_dict = load_probabilistic_model()

para_lambda = 0.01
top_k = 0


def laplace_smoothing_probability(token, prefix): #  laplace smoothing
    if not token in conditional_probabilistic_dict[prefix]:
        return single_token_probabilistic_dict[token] * para_lambda
    return conditional_probabilistic_dict[prefix][token] * (1 - para_lambda) + single_token_probabilistic_dict[token] * para_lambda


def get_conditional_probability(token, prefix=None):
    if token not in vocabs:
        return 0
    if prefix is None:
        return single_token_probabilistic_dict[token]
    if prefix not in vocabs:
        return 0
    return laplace_smoothing_probability(token=token, prefix=prefix)


def pinyin_to_character(pinyin_str: str):
    pinyin_list = pinyin_str.rstrip("\n").split(" ")
    if len(pinyin_list) < 1:
        return ""
    search_state = []
    for i in range(len(pinyin_list)):
        search_state.append({})
        pinyin = pinyin_list[i]
        for token in pinyin_dict[pinyin]:
            if i == 0:
                search_state[i][token] = [(get_conditional_probability(token), None)]  # TODO: Use ""
                # print(token, search_state[i][token])
                # input()
            else:
                for prefix, max_prefix_prob_list in search_state[i - 1].items():
                    max_prefix_prob = max_prefix_prob_list[-1][0]
                    seq_prob = max_prefix_prob * get_conditional_probability(token=token, prefix=prefix)
                    if token not in search_state[i]:
                        search_state[i][token] = [(seq_prob, prefix)]
                    elif seq_prob > search_state[i][token][-1][0]:
                        search_state[i][token].append((seq_prob, prefix))
            search_state[i][token] = search_state[i][token][- top_k : ]
            # print(len(search_state[i][token]))

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
    
    # print(pinyin_str, sentence)

    return sentence


def main():
    for (input_file_path, input_format), (output_file_path, output_format) in zip(input_path, output_path):
        with open(input_file_path, "r", encoding=input_format) as input_file:
            answers = map(pinyin_to_character, input_file.readlines())
        with open(output_file_path, "w", encoding=output_format) as output_file:
            answers = [answer + '\n' for answer in answers]
            output_file.writelines(answers)


if __name__ == "__main__":
    main()


