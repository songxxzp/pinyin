import torch

from torch.nn import CrossEntropyLoss
from typing import List
from torch import tensor
from torch.nn.functional import softmax
from transformers import BertTokenizerFast, AutoModel, AutoTokenizer, GPT2Config, BertTokenizer, GPT2LMHeadModel, BertTokenizer, BartForConditionalGeneration


def load_gpt_model(device="cpu", tokenizer_name="uer/gpt2-chinese-cluecorpussmall", model_name="uer/gpt2-chinese-cluecorpussmall"):
    tokenizer = BertTokenizer.from_pretrained(tokenizer_name)
    model = GPT2LMHeadModel.from_pretrained(model_name).to(device)

    return model, tokenizer


def build_tokenized_pinyin_dict(pinyin_dict, tokenizer, device="cpu"):
    tokenized_pinyin_dict = {}
    for pinyin, token_list in pinyin_dict.items():
        token_id_list = tensor([tokenizer.convert_tokens_to_ids(c) for c in token_list], device=device)
        tokenized_pinyin_dict[pinyin] = token_id_list


def gpt_probability(prefix, token, pinyin, model, tokenizer, tokenized_pinyin_dict):
    input_ids = tokenizer(prefix, return_tensors='pt').to(model.device)
    logits = model(**input_ids).logits
    prob_list = softmax(logits[0][-2], dim=-1)

    total_prob = prob_list[tokenized_pinyin_dict[pinyin]].sum()

    token_id = tokenizer.convert_tokens_to_ids(token)

    return float(prob_list[token_id] / total_prob)


def perplexity(seq: List[str], model, tokenizer):
    input_ids = tokenizer(seq, return_tensors='pt', padding=True).to(model.device)
    logits = model(**input_ids).logits

    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids['input_ids'][:, 1:].contiguous()
    shift_attentions = input_ids['attention_mask'][:, 1:].contiguous()

    loss_fn = CrossEntropyLoss(ignore_index=0, reduction="none")
    loss = loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)).detach().reshape(len(seq), -1)
    meanloss = loss.sum(1) / shift_attentions.sum(1)
    ppl = torch.exp(meanloss).tolist()
    return ppl
