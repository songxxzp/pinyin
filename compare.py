import os
from config import load_config

current_path = os.path.dirname(os.path.realpath(__file__))

load_config(
    model_path="/home/song/workspace/pinyin/trigram_weibo_newscrawl_baike_webtext_wiki",  # trigram_weibo_newscrawl_baike_webtext, trigram_weibo_newscrawl_baike, trigram_newscrawl_baike, bigram_weibo
    output_file_path=os.path.join(current_path, "./output_gpt_selector.txt")
)

from config import *
from gpt import perplexity, load_gpt_model


if __name__ == "__main__":
    model, tokenizer = load_gpt_model(device="cuda")

    for (std_output_file_path, std_output_format), (output_file_path, output_format) in zip(std_output_path, output_path):
        with open(output_file_path, "r", encoding=output_format) as output_file:
            outputs = [line.rstrip() for line in output_file.readlines()]

        with open(std_output_file_path, "r", encoding=std_output_format) as std_output_file:
            std_outputs = [line.rstrip() for line in std_output_file.readlines()]

    unmatch = 0
    uncorrect_ppl = 0

    for output, std_output in zip(outputs, std_outputs):
        if not output == std_output:
            unmatch += 1
            ppl = perplexity([output, std_output], model, tokenizer)
            if ppl[0] < ppl[1]:
                uncorrect_ppl += 1
                print(output, std_output, ppl)

    print("unmatch :", unmatch)
    print("uncorrect ppl :", uncorrect_ppl)
