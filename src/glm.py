import torch

from torch.nn import CrossEntropyLoss
from typing import List
from torch import tensor
from torch.nn.functional import softmax
from transformers import AutoModel, AutoTokenizer, AutoModelForSeq2SeqLM


def load_glm_model(device="cpu", tokenizer_path="/home/song/workspace/glm-large-chinese/", model_path="/home/song/workspace/glm-large-chinese/"):  # THUDM/glm-large-chinese
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path, trust_remote_code=True).to(device)
    return model, tokenizer


def perplexity(seq: List[str], model, tokenizer):
    input_ids = tokenizer(seq, return_tensors='pt', padding=True).to(model.device)
    labels = input_ids['input_ids']
    lm_logits = model(**input_ids).logits
    shift_logits = lm_logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    shift_attentions = input_ids['attention_mask'][:, 1:].contiguous()

    loss_fct = CrossEntropyLoss(ignore_index=-100, reduction="none")
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)).reshape(shift_labels.shape)
    meanloss = loss.sum(dim=1) / shift_attentions.sum(dim=1)
    ppl = torch.exp(meanloss).tolist()
    # loss_fct = CrossEntropyLoss(ignore_index=-100, reduction="none")
    # loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1)).reshape(labels.shape)
    # meanloss = loss.sum(dim=1) / input_ids['attention_mask'].sum(dim=1)
    # ppl = torch.exp(meanloss).tolist()
    return ppl


if __name__ == "__main__":
    # TRANSFORMERS_OFFLINE=1

    model, tokenizer = load_glm_model(device="cuda")

    seq = ["北京市首个举办过夏奥会与冬奥会的城市", "北京是首个举办过夏奥会与冬奥会的城市", "用干毛毛不怕困难", "勇敢猫猫不怕困难", "晶状体有痛心的纤维细胞层组成", "晶状体由同心的纤维细胞层组成", "毕竟老佛爷不是什么恶魔", "毕竟老夫也不是什么恶魔"]

    input_ids = tokenizer(seq, return_tensors='pt', padding=True).to(model.device)

    # print(input_ids)

    labels = input_ids['input_ids']
    lm_logits = model(**input_ids, labels=labels).logits

    print(model(**input_ids, labels=labels).loss)

    shift_logits = lm_logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    shift_attentions = input_ids['attention_mask'][:, 1:].contiguous()

    loss_fct = CrossEntropyLoss(ignore_index=-100, reduction="none")
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)).reshape(shift_labels.shape)
    meanloss = loss.sum(dim=1) / shift_attentions.sum(dim=1)
    ppl = torch.exp(meanloss).tolist()

    print(meanloss)
    print(ppl)

    loss_fct = CrossEntropyLoss(ignore_index=-100, reduction="none")
    loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1)).reshape(labels.shape)
    meanloss = loss.sum(dim=1) / input_ids['attention_mask'].sum(dim=1)
    ppl = torch.exp(meanloss).tolist()

    print(meanloss)
    print(ppl)
