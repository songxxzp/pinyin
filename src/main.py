import time

from functools import partial

from args import get_parser
from config import load_config
from selector import default_top_k_selector, get_top_k_selector
from metric import eval


def print_parameter(args):
    print("model path :", args.model_path)
    print("max conditional prefix length :", args.max_conditional_prefix_length)
    print("top k storage :", args.top_k_storage)
    print("top k calculate :", args.top_k_calculate)
    print("calculate top k selector :", args.calculate_top_k_selector)
    print("storage top k selector :", args.storage_top_k_selector)
    print("final top k selector :", args.final_top_k_selector)
    if args.final_top_k_selector == "gpt" or args.storage_top_k_selector == "gpt" or args.calculate_top_k_selector == "gpt":
        if args.batch_size:
            print("    batchsize", args.batch_size)
        print("    lm tokenizer :", args.lm_tokenizer_path)
        print("    lm model :", args.lm_model_path)
        print("    device", args.device)
    print("probability function :", args.probability_function)
    if args.probability_function == "interpolation":
        print("    lambda :", args.interpolation_lambda)
    print("normalized :", args.normalized)


def pinyin_to_character(pinyin_str: str, probability_fn, pinyin_dict, max_conditional_prefix_length=1, top_k_storage=1, top_k_calculate=1, final_top_k_selector=default_top_k_selector, storage_top_k_selector=default_top_k_selector, calculate_top_k_selector=default_top_k_selector):
    pinyin_list = pinyin_str.rstrip("\n").split(" ")
    if len(pinyin_list) < 1:
        return ""
    last_search_state = {}
    for i in range(len(pinyin_list)):
        search_state = {}
        pinyin = pinyin_list[i]
        for token in pinyin_dict[pinyin]:
            if i == 0:
                search_state[token] = [(probability_fn(token=token, prefix="", pinyin=pinyin), token)]
            else:
                search_state[token] = []
                for _, last_token_prob_list in last_search_state.items():
                    for max_prefix_prob, prefix in calculate_top_k_selector(seq_list=last_token_prob_list, top_k=top_k_calculate):  # (prob, prefix)
                        seq_prob = max_prefix_prob * probability_fn(token=token, prefix=prefix[-max_conditional_prefix_length:], pinyin=pinyin)
                        search_state[token].append((seq_prob, prefix + token))

            search_state[token].sort(key=lambda x: x[0], reverse=True)
            search_state[token] = storage_top_k_selector(seq_list=search_state[token], top_k=top_k_storage)

        last_search_state = search_state

    # get sentences with top k probability
    final_search_state = []
    for _, last_token_prob_list in last_search_state.items():
        final_search_state += calculate_top_k_selector(seq_list=last_token_prob_list, top_k=top_k_calculate)

    final_search_state.sort(key=lambda x: x[0], reverse=True)
    final_search_state = storage_top_k_selector(seq_list=final_search_state, top_k=top_k_storage)

    return final_top_k_selector(seq_list=final_search_state)[0][1]


def inference(args):
    from probability import get_probability_function
    from model import load_vocab
    from config import max_prefix_length, input_path, output_path

    _, pinyin_dict = load_vocab()
    final_selector = get_top_k_selector(args.final_top_k_selector, device=args.device, batch_size=args.batch_size, tokenizer_path=args.lm_tokenizer_path, model_path=args.lm_model_path) # default, std, gpt, glm
    storage_selector = get_top_k_selector(args.storage_top_k_selector, device=args.device, batch_size=args.batch_size, tokenizer_path=args.lm_tokenizer_path, model_path=args.lm_model_path)
    calculate_selector = get_top_k_selector(args.calculate_top_k_selector, device=args.device, batch_size=args.batch_size, tokenizer_path=args.lm_tokenizer_path, model_path=args.lm_model_path)
    probability_fn = get_probability_function(args.probability_function, para_lambda=args.interpolation_lambda, normalized=args.normalized)  # interpolation, laplace

    args.max_conditional_prefix_length = min(args.max_conditional_prefix_length, max_prefix_length)

    process_fn = partial(
        pinyin_to_character,
        probability_fn=probability_fn,
        pinyin_dict=pinyin_dict,
        max_conditional_prefix_length=args.max_conditional_prefix_length,
        top_k_storage=args.top_k_storage,
        top_k_calculate=args.top_k_calculate,
        final_top_k_selector=final_selector,
        storage_top_k_selector=storage_selector,
        calculate_top_k_selector=calculate_selector
    )

    time_usage = 0

    print("Inferencing")

    if args.input_file_path is not None and "interactive" in args.input_file_path:
        print_parameter(args)
        while True:
            try:
                input_pinyin = input("pinyin (stop to exit):")
                if "stop" in input_pinyin:
                    exit()
                start_time = time.time()
                answer = process_fn(input_pinyin)
                stop_time = time.time()
                print("answer :", answer)
                print("Time usage : {}s".format(str(round(stop_time - start_time, 2))))
                time_usage += stop_time - start_time
            except Exception as exception:
                print(repr(exception))

    for (input_file_path, input_format), (output_file_path, output_format) in zip(input_path, output_path):
        with open(input_file_path, "r", encoding=input_format) as input_file:
            start_time = time.time()
            answers = list(map(process_fn, input_file.readlines()))
            stop_time = time.time()
            time_usage += stop_time - start_time
        with open(output_file_path, "w", encoding=output_format) as output_file:
            answers = [answer + '\n' for answer in answers]
            output_file.writelines(answers)
    
    return time_usage


def main(args):
    load_config(
        model_path=args.model_path,
        output_file_path=args.output_file_path,
        input_file_path=args.input_file_path,
        std_file_path=args.std_file_path,
        vocab_file_path=args.vocab_file_path,
        pinyin_file_path=args.pinyin_file_path
    )

    time_usage = inference(args)
    print_parameter(args)
    print("Total time usage : {}s".format(str(round(time_usage, 2))))

    sentence_acc, word_acc = eval()
    print("sentence accuracy :", sentence_acc)
    print("word accuracy :", word_acc)
    return (time_usage, sentence_acc, word_acc)


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)
