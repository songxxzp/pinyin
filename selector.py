from typing import List
from functools import partial

def default_top_k_selector(seq_list: List[str]) -> str:
    """
        choose one answer from a list
        return seq_list[0] as default
    """
    return seq_list[0]


def gpt_top_k_selector(seq_list: List[str], model, tokenizer, perplexity) -> str:
    """
        choose one answer from a list
        return seq with minima PPL
    """
    ppl = perplexity(seq_list, model, tokenizer)
    seq = seq_list[ppl.index(min(ppl))]
    return seq


def std_top_k_selector(seq_list: List[str], std_list) -> str:
    """
        choose one answer from a list
        return seq if seq in std_list, else return seq_list[0] as default 
    """
    for seq in seq_list:
        if seq in std_list:
            return seq
    return seq_list[0]


def get_top_k_selector(selector_type="default", device=None, tokenizer_name="uer/gpt2-chinese-cluecorpussmall", model_name="uer/gpt2-chinese-cluecorpussmall"):
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
            perplexity=perplexity
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