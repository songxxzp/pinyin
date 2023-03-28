import math

from typing import List, Tuple
from functools import partial

def default_top_k_selector(seq_list: List[Tuple[float, str]], top_k=1) -> str:
    """
        choose top k answers from a list
        return seq_list[:top_k] as default
    """
    return seq_list[:top_k]


def gpt_top_k_selector(seq_list: List[Tuple[float, str]], model, tokenizer, perplexity, top_k=1, batch_size=0) -> str:
    """
        choose top k answers from a list
        return seq with minima PPL
    """
    ppl = []
    if batch_size:
        for i in range(math.ceil(len(seq_list) / batch_size)):
            ppl += perplexity([seq[1] for seq in seq_list[i * batch_size : (i + 1) * batch_size]], model, tokenizer)
    else:
        ppl = perplexity([seq[1] for seq in seq_list], model, tokenizer)
    seq_tuples = list(zip(ppl, seq_list))
    seq_tuples.sort(key=lambda x:x[0])
    return [seq[1] for seq in seq_tuples[:top_k]]


def std_top_k_selector(seq_list: List[Tuple[float, str]], std_list, top_k=1) -> str:
    """
        choose top k answers from a list
        return seq_list[:top_k], put std in seq_list[0]
    """
    idx = 0
    for idx, seq in enumerate(seq_list):
        if seq[1] in std_list:
            break
    seq_list[0], seq_list[idx] = seq_list[idx], seq_list[0]
    return seq_list[:top_k]


def get_top_k_selector(selector_type="default", device=None, tokenizer_name="uer/gpt2-chinese-cluecorpussmall", model_name="uer/gpt2-chinese-cluecorpussmall", batch_size=0):
    """
        return a selector
    """
    selector = default_top_k_selector
    if selector_type == "gpt":
        from gpt import load_gpt_model, perplexity
        if device is None:
            import torch
            if torch.cuda.is_available():
                device="cuda"
            else:
                device="cpu"
        model, tokenizer = load_gpt_model(device=device, tokenizer_name=tokenizer_name, model_name=model_name)
        selector = partial(
            gpt_top_k_selector,
            model=model,
            tokenizer=tokenizer,
            perplexity=perplexity,
            batch_size=batch_size
        )
    elif selector_type == "std":
        from config import std_output_path
        std_outputs = []
        for (std_output_file_path, std_output_format) in std_output_path:
            with open(std_output_file_path, "r", encoding=std_output_format) as std_output_file:
                std_outputs += [line.rstrip() for line in std_output_file.readlines()]
        selector = partial(
            std_top_k_selector,
            std_list=std_outputs
        )
    return selector