import  os


from torch import tensor
from torch.nn.functional import softmax
from transformers import BertTokenizerFast, GPT2LMHeadModel, AutoModel, AutoTokenizer, GPT2Config, BartForConditionalGeneration
from transformers import Trainer, TrainingArguments
from transformers import TextDataset,DataCollatorForLanguageModeling
from datasets import load_dataset

tokenizer = BertTokenizerFast.from_pretrained('bert-base-chinese')


def load_dataset(train_path,test_path,tokenizer):
    train_dataset = TextDataset(
          tokenizer=tokenizer,
          file_path=train_path,
          block_size=128)

    test_dataset = TextDataset(
          tokenizer=tokenizer,
          file_path=test_path,
          block_size=128)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False,
    )
    return train_dataset,test_dataset,data_collator

train_dataset,test_dataset,data_collator = load_dataset("/home/song/下载/拼音输入法作业-2023春/测试语料/std_output.txt", "/home/song/下载/拼音输入法作业-2023春/测试语料/std_output.txt", tokenizer)


training_ars = TrainingArguments(
    output_dir="results",
    num_train_epochs=1000,
    overwrite_output_dir=True,
    per_device_train_batch_size=128,
    per_device_eval_batch_size=256,
    eval_steps=500,
    save_steps=2000,
    warmup_steps=500,
)

model = GPT2LMHeadModel.from_pretrained("ckiplab/gpt2-tiny-chinese").to("cuda")

trainer = Trainer(
    model=model,
    args=training_ars,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    data_collator=data_collator
)

trainer.train()
