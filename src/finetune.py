import torch
import json
import random

from utils import extract_data_path

from torch.nn import CrossEntropyLoss
from typing import List, Dict
from torch import tensor
from torch.nn.functional import softmax
from transformers import BertTokenizerFast, AutoModel, AutoTokenizer, GPT2Config, BertTokenizer, GPT2LMHeadModel, BertTokenizer, BartForConditionalGeneration, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from datasets import load_dataset

device = "cuda"
model_config_path = "/home/junli/workspace/pinyin/trigram_sina_newscrawl_baike_webtext_wiki/config.json"
vocab_path = [["/data/TST/pinyin/拼音输入法作业-2023春/拼音汉字表/一二级汉字表.txt", "gbk"]]
pinyin_path = [["/data/TST/pinyin/拼音输入法作业-2023春/拼音汉字表/拼音汉字表.txt", "gbk"]]

def load_vocab():
    # load vocab
    vocabs = {}
    for path, format in vocab_path:
        with open(path, "r", encoding=format) as file:
            for vocab in file.read():
                vocabs[vocab] = []

    # build pinyin dict
    pinyin_dict = {}
    for path, format in pinyin_path:
        with open(path, "r", encoding=format) as file:
            for line in file.readlines():
                alphabet = line.rstrip("\n").split(" ")
                if alphabet and alphabet[0] not in pinyin_dict:
                    pinyin_dict[alphabet[0]] = alphabet[1:]
                    for vocab in alphabet[1:]:
                        vocabs[vocab].append(alphabet[0])
                else:
                    print("Conflict in pinyin dict.")
    
    return vocabs, pinyin_dict

def process_data():
    from model import split
    vocabs, pinyin_dict = load_vocab()
    full_text_list = []
    with open(model_config_path, "r", encoding="utf-8") as file:
        config = json.load(file)
        for path, format, labels in extract_data_path(config["corpora_path"]):  # should be jsonl or txt
            with open(path, "r", encoding=format) as file:
                for line in file.readlines():
                    line = line.rstrip('\n')
                    if len(labels) and line:  # jsonl
                        try:
                            content_dict = json.loads(line)
                            for label in labels:
                                content = content_dict[label]
                                full_text_list.append(content)
                        except Exception as exception:
                            print(repr(exception))
                            print("Cannot load line, check your format:", line)
                    else:
                        full_text_list.append(line)

    full_text_list = [{"text": text} for text in full_text_list]

    with open("trigram_sina_newscrawl_baike_webtext_wiki.json", "w", encoding="utf-8") as file:
        json.dump(
            {
                "model_config_path": model_config_path,
                "data": full_text_list
            },
            file,
            ensure_ascii=False
        )


def load_data():
    with open("trigram_sina_newscrawl_baike_webtext_wiki.json", "r", encoding="utf-8") as file:
        return json.load(file)


# process_data()


# data_dict = load_data()
# data_list = data_dict["data"]
# random.seed(42)
# random.shuffle(data_list)
# part_text_list = data_list  # [:10000000]

# print(len(data_list))

# with open("train.json", "w", encoding="utf-8") as file:
#     json.dump(
#         {
#             "model_config_path": model_config_path,
#             "data": part_text_list[:int(0.95 * len(part_text_list))]
#         },
#         file,
#         ensure_ascii=False
#     )


# with open("test.json", "w", encoding="utf-8") as file:
#     json.dump(
#         {
#             "model_config_path": model_config_path,
#             "data": part_text_list[int(0.95 * len(part_text_list)):]
#         },
#         file,
#         ensure_ascii=False
#     )


tokenizer = BertTokenizer.from_pretrained("uer/gpt2-chinese-cluecorpussmall")

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False,
)


def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, return_tensors='pt').to(device)


train_dataset = load_dataset("json", data_files="train.json", field="data", split="train").shuffle()
test_dataset = load_dataset("json", data_files="test.json", field="data", split="train").shuffle()

train_tokenized_datasets = train_dataset.map(tokenize_function)
test_tokenized_datasets = test_dataset.map(tokenize_function)

print(train_tokenized_datasets)
print(test_tokenized_datasets)

model = GPT2LMHeadModel.from_pretrained("uer/gpt2-chinese-cluecorpussmall").to(device)

training_args = TrainingArguments(
    output_dir="/home/junli/workspace/pinyin/gpt2", #The output directory
    learning_rate=2e-5,
    overwrite_output_dir=True, #overwrite the content of the output directory
    num_train_epochs=1, # number of training epochs
    per_device_train_batch_size=2, # batch size for training
    per_device_eval_batch_size=8,  # batch size for evaluation
    eval_steps = 5000, # Number of update steps between two evaluations.
    save_steps=5000, # after # steps model is saved
    warmup_steps=500,# number of warmup steps for learning rate scheduler
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_tokenized_datasets,
    eval_dataset=test_tokenized_datasets,
    # prediction_loss_only=True,
)

trainer.train()

trainer.save_model()
