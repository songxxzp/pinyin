import json
import pickle
import os

from main import main
from args import get_parser


if __name__ == "__main__":
    current_path = os.path.dirname(os.path.realpath(__file__))
    results_path = os.path.join(current_path, "results")

    parser = get_parser()
    args = parser.parse_args()

    model_path_list = ["models/bigram", "models/bigram_std", "models/bigram_weibo_baike_webtext_wiki", "models/bigram_weibo_newscrawl_baike_webtext_wiki", "models/trigram", "models/trigram_std", "models/trigram_weibo_baike_webtext_wiki", "models/trigram_weibo_newscrawl_baike_webtext_wiki"]

    top_k_list = [1, 3, 5, 10]

    final_selector_list = ["std", "default", "gpt"]

    result_dict = {}
    result_list = []

    for model_path in model_path_list:
        for top_k in top_k_list:
            for final_selector in final_selector_list:
                output_file_path = os.path.join(results_path, os.path.basename(model_path) + ".txt")
                args.model_path = model_path
                args.calculate_top_k_selector = top_k
                args.storage_top_k_selector = top_k
                args.final_top_k_selector = final_selector
                args.output_file_path = output_file_path

                time_usage, sentence_acc, word_acc = main(args)
                
                result_dict[(model_path, top_k, final_selector)] = (time_usage, sentence_acc, word_acc)
                result_list.append([model_path, top_k, final_selector, time_usage, sentence_acc, word_acc])

    with open(os.path.join(results_path, "results.json"), "w", encoding="utf-8") as file:
        json.dump(result_list, file)

    with open(os.path.join(results_path, "results.pkl"), "wb") as file:
        pickle.dump(result_dict, file)

    # parser.add_argument('--probability-function', type=str, default="interpolation", help='probability function type(interpolation / laplace)')
    # parser.add_argument('--interpolation-lambda', type=float, default=0.03, help='lambda for interpolation probability function')
    # parser.add_argument('--normalized', type=bool, default=False, help='normalize probability function')

    # parser.add_argument('--lm-tokenizer-path', type=str, default=None, help='tokenizer path for lm')
    # parser.add_argument('--lm-model-path', type=str, default=None, help='model path for lm')

