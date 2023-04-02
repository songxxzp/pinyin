import json
import pickle
import os
import datetime

from args import get_parser


if __name__ == "__main__":
    current_path = os.path.dirname(os.path.realpath(__file__))
    results_path = os.path.join(current_path, "results")

    parser = get_parser()
    args = parser.parse_args()

    model_path_list = ["models/bigram", "models/bigram_std", "models/bigram_weibo_baike_webtext_wiki", "models/bigram_weibo_newscrawl_baike_webtext_wiki", "models/trigram", "models/trigram_std", "models/trigram_weibo_baike_webtext_wiki", "models/trigram_weibo_newscrawl_baike_webtext_wiki"]

    top_k_list = [1, 3, 5, 10]

    final_selector_list = ["std", "default", "gpt"]

    # probability_function
    # lambda
    # normalized
    # language modeling

    result_dict = {}
    result_list = []

    start_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    for model_path in model_path_list:
        for top_k in top_k_list:
            for final_selector in final_selector_list:
                output_file_path = os.path.join(results_path, os.path.basename(model_path) + ".txt")
                args.model_path = model_path
                args.top_k_storage = top_k
                args.top_k_calculate = top_k
                args.final_top_k_selector = final_selector
                args.output_file_path = output_file_path
                args.max_conditional_prefix_length = 2

                from main import main
                time_usage, sentence_acc, word_acc = main(args)
                
                result_dict[(model_path, top_k, final_selector)] = (time_usage, sentence_acc, word_acc)
                result_list.append([model_path, top_k, final_selector, time_usage, sentence_acc, word_acc])

        with open(os.path.join(results_path, "results {}.json".format(start_time)), "w", encoding="utf-8") as file:
            json.dump(result_list, file)

        with open(os.path.join(results_path, "results {}.pkl".format(start_time)), "wb") as file:
            pickle.dump(result_dict, file)
