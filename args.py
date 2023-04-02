import os
import argparse

def get_parser():
    current_path = os.path.dirname(os.path.realpath(__file__))

    parser = argparse.ArgumentParser(description='Pinyin config')

    parser.add_argument('--model-path', type=str, default="models/bigram_std", help='model folder path')

    parser.add_argument('--output-file-path', type=str, default=os.path.join(current_path, "./output.txt"), help='output file path')

    parser.add_argument('--input-file-path', type=str, default=None, help='input file path or interactive')
    parser.add_argument('--std-file-path', type=str, default=None, help='std output file path')
    parser.add_argument('--vocab-file-path', type=str, default=None, help='vocab file path')
    parser.add_argument('--pinyin-file-path', type=str, default=None, help='pinyin file path')

    parser.add_argument('--probability-function', type=str, default="interpolation", help='probability function type(interpolation / laplace)')
    parser.add_argument('--interpolation-lambda', type=float, default=0.03, help='lambda for interpolation probability function')
    parser.add_argument('--top-k-storage', type=int, default=4, help='top k used when storage state')
    parser.add_argument('--top-k-calculate', type=int, default=4, help='top k used when calculate state')
    parser.add_argument('--max-conditional-prefix-length', type=int, default=2, help='max conditional prefix length')
    parser.add_argument('--normalized', type=bool, default=False, help='normalize probability function')
    parser.add_argument('--storage-top-k-selector', type=str, default="default", help='top k selector used when storage state')
    parser.add_argument('--calculate-top-k-selector', type=str, default="default", help='top k selector used when calculate state')
    parser.add_argument('--final-top-k-selector', type=str, default="default", help='top k selector used when final state')
    parser.add_argument('--batch-size', type=int, default=0, help='batch_size for lm(0 for infinite)')
    parser.add_argument('--device', type=str, default="cuda", help='device for lm')
    parser.add_argument('--lm-tokenizer-path', type=str, default="uer/gpt2-chinese-cluecorpussmall", help='tokenizer path for lm')
    parser.add_argument('--lm-model-path', type=str, default="uer/gpt2-chinese-cluecorpussmall", help='model path for lm')
    
    return parser
    
